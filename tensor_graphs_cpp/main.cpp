#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <tuple>
#include <float.h>

#include "core/types.hpp"
#include "core/memory.hpp"
#include "core/graph.hpp"
#include "core/session.hpp"
#include "core/kernels.hpp"

#include "kernels/fused/tanh/F32_1D.hpp"
#include "kernels/reference/add/F32_ND.hpp"
#include "kernels/reference/cast/BF16_F32_ND.hpp"
#include "kernels/reference/div/F32_ND.hpp"
#include "kernels/reference/dot/F32_3D.hpp"
#include "kernels/reference/gather/F32_I32_ND.hpp"
#include "kernels/reference/mul/F32_ND.hpp"
#include "kernels/reference/neg/F32_ND.hpp"
#include "kernels/reference/permute/F32_ND.hpp"
#include "kernels/reference/pow/F32_ND.hpp"
#include "kernels/reference/repeat/F32_ND.hpp"
#include "kernels/reference/reshape/ND.hpp"
#include "kernels/reference/sum/F32_ND.hpp"

struct ModelConfig
{
    uint32_t vocab_size = 262144;
    uint32_t n_layers = 18;
    uint32_t emb_dim = 640;
    uint32_t hidden_dim = 2048;
    uint32_t n_heads = 4;
    uint32_t head_dim = 256;
    uint32_t n_kv_groups = 1;
    uint32_t query_pre_attn_scalar = 256;
};

class Gemma3Model
{
private:
    ModelConfig cfg;
    Graph &g;
    MemoryManager &mem;
    const std::string w_path;
    float eps;
    uint32_t seq_len;

    uint32_t one_fp32;
    uint32_t eps_fp32;
    uint32_t half_fp32;

public:
    Gemma3Model(ModelConfig config, uint32_t sequence_length, Graph &graph, MemoryManager &memory, const std::string &weight_path)
        : cfg(config), g(graph), mem(memory), w_path(weight_path), eps(1e-6f), seq_len(sequence_length)
    {
        float one_val = 1.0f;
        one_fp32 = g.constant({1}, &one_val, DType::FLOAT32);
        eps_fp32 = g.constant({1}, &eps, DType::FLOAT32);

        float half_val = 0.5f;
        half_fp32 = g.constant({1}, &half_val, DType::FLOAT32);
    }

    uint32_t weight(const std::string &path, const std::string &name)
    {
        uint32_t raw_weight = g.weight(path, name);
        return g.cast(raw_weight, DType::FLOAT32);
    }

    uint32_t expand_scalar_to_3d(uint32_t scalar_id, uint32_t dim0, uint32_t dim1, uint32_t dim2)
    {
        int32_t shape_3d[] = {1, 1, 1};
        uint32_t shape_3d_node = g.constant({3}, shape_3d, DType::INT32);
        uint32_t out = g.reshape(scalar_id, shape_3d_node);

        if (dim0 > 1)
        {
            int32_t rep[] = {(int32_t)dim0};
            uint32_t rep_node = g.constant({1}, rep, DType::INT32);
            int32_t ax[] = {0};
            uint32_t ax_node = g.constant({1}, ax, DType::INT32);
            out = g.repeat(out, rep_node, ax_node);
        }
        if (dim1 > 1)
        {
            int32_t rep[] = {(int32_t)dim1};
            uint32_t rep_node = g.constant({1}, rep, DType::INT32);
            int32_t ax[] = {1};
            uint32_t ax_node = g.constant({1}, ax, DType::INT32);
            out = g.repeat(out, rep_node, ax_node);
        }
        if (dim2 > 1)
        {
            int32_t rep[] = {(int32_t)dim2};
            uint32_t rep_node = g.constant({1}, rep, DType::INT32);
            int32_t ax[] = {2};
            uint32_t ax_node = g.constant({1}, ax, DType::INT32);
            out = g.repeat(out, rep_node, ax_node);
        }
        return out;
    }

    uint32_t expand_1d_to_3d(uint32_t vec_id, uint32_t vec_len, uint32_t dim0, uint32_t dim1)
    {
        int32_t shape_3d[] = {1, 1, (int32_t)vec_len};
        uint32_t shape_3d_node = g.constant({3}, shape_3d, DType::INT32);
        uint32_t out = g.reshape(vec_id, shape_3d_node);

        if (dim0 > 1)
        {
            int32_t rep[] = {(int32_t)dim0};
            uint32_t rep_node = g.constant({1}, rep, DType::INT32);
            int32_t ax[] = {0};
            uint32_t ax_node = g.constant({1}, ax, DType::INT32);
            out = g.repeat(out, rep_node, ax_node);
        }
        if (dim1 > 1)
        {
            int32_t rep[] = {(int32_t)dim1};
            uint32_t rep_node = g.constant({1}, rep, DType::INT32);
            int32_t ax[] = {1};
            uint32_t ax_node = g.constant({1}, ax, DType::INT32);
            out = g.repeat(out, rep_node, ax_node);
        }
        return out;
    }

    uint32_t repeat_3d_axis(uint32_t tensor_id, uint32_t repeats, uint32_t axis)
    {
        if (repeats <= 1)
            return tensor_id;
        int32_t rep[] = {(int32_t)repeats};
        uint32_t rep_node = g.constant({1}, rep, DType::INT32);
        int32_t ax[] = {(int32_t)axis};
        uint32_t ax_node = g.constant({1}, ax, DType::INT32);
        return g.repeat(tensor_id, rep_node, ax_node);
    }

    uint32_t rms_norm_gemma_atomic(uint32_t x_id, uint32_t weight_id)
    {
        uint32_t x_sq = g.mul(x_id, x_id);

        int32_t axis_val = -1;
        uint32_t axis_node = g.constant({1}, &axis_val, DType::INT32);

        uint32_t sum_sq = g.sum(x_sq, axis_node);

        float n_val = (float)cfg.emb_dim;
        uint32_t n_node = g.constant({1}, &n_val, DType::FLOAT32);
        n_node = expand_scalar_to_3d(n_node, 1, seq_len, 1);

        uint32_t mean_sq = g.div(sum_sq, n_node);

        uint32_t eps_expanded = expand_scalar_to_3d(eps_fp32, 1, seq_len, 1);
        uint32_t mean_sq_plus_eps = g.add(mean_sq, eps_expanded);

        float half_val = 0.5f;
        uint32_t sqrt_node = g.constant({1}, &half_val, DType::FLOAT32);
        sqrt_node = expand_scalar_to_3d(sqrt_node, 1, seq_len, 1);

        uint32_t std = g.pow(mean_sq_plus_eps, sqrt_node);

        uint32_t one_node = expand_scalar_to_3d(one_fp32, 1, seq_len, 1);
        uint32_t inv_std = g.div(one_node, std);

        uint32_t inv_std_expanded = repeat_3d_axis(inv_std, cfg.emb_dim, 2);
        uint32_t x_norm = g.mul(x_id, inv_std_expanded);

        uint32_t weight_expanded = expand_1d_to_3d(weight_id, cfg.emb_dim, 1, seq_len);

        uint32_t one_node_640 = expand_scalar_to_3d(one_fp32, 1, seq_len, cfg.emb_dim);
        uint32_t scale = g.add(weight_expanded, one_node_640);

        return g.mul(x_norm, scale);
    }

    uint32_t gelu_atomic(uint32_t x_id, uint32_t last_dim)
    {
        float c1_val = 0.044715f;
        uint32_t c1_node = expand_scalar_to_3d(g.constant({1}, &c1_val, DType::FLOAT32), 1, seq_len, last_dim);

        float c2_val = 0.79788456f;
        uint32_t c2_node = expand_scalar_to_3d(g.constant({1}, &c2_val, DType::FLOAT32), 1, seq_len, last_dim);

        uint32_t x_sq = g.mul(x_id, x_id);
        uint32_t x_cube = g.mul(x_sq, x_id);

        uint32_t term1 = g.mul(x_cube, c1_node);
        uint32_t term2 = g.add(x_id, term1);
        uint32_t term3 = g.mul(term2, c2_node);

        uint32_t tanh_result = tanh_atomic(term3, last_dim);

        uint32_t one_node = expand_scalar_to_3d(one_fp32, 1, seq_len, last_dim);
        uint32_t term4 = g.add(one_node, tanh_result);

        uint32_t half_node = expand_scalar_to_3d(half_fp32, 1, seq_len, last_dim);
        uint32_t term5 = g.mul(x_id, half_node);

        return g.mul(term5, term4);
    }

    uint32_t tanh_atomic(uint32_t x_id, uint32_t last_dim)
    {
        float neg_two_val = -2.0f;
        uint32_t neg_two = expand_scalar_to_3d(g.constant({1}, &neg_two_val, DType::FLOAT32), 1, seq_len, last_dim);

        float two_val = 2.0f;
        uint32_t two = expand_scalar_to_3d(g.constant({1}, &two_val, DType::FLOAT32), 1, seq_len, last_dim);

        float e_val = 2.718281828459045f;
        uint32_t e_node = expand_scalar_to_3d(g.constant({1}, &e_val, DType::FLOAT32), 1, seq_len, last_dim);

        uint32_t one_node = expand_scalar_to_3d(one_fp32, 1, seq_len, last_dim);

        // 1. exp(-2x)
        uint32_t neg_2x = g.mul(x_id, neg_two);
        uint32_t exp_neg_2x = g.pow(e_node, neg_2x);

        // 2. 1 + exp(-2x)
        uint32_t den = g.add(one_node, exp_neg_2x);

        // 3. 2 / (1 + exp(-2x))
        uint32_t quotient = g.div(two, den);

        // 4. quotient - 1
        uint32_t neg_one = g.neg(one_node);
        return g.add(quotient, neg_one);
    }

    std::tuple<uint32_t, uint32_t, uint32_t> attention_qkv_atomic(uint32_t x, const std::string &prefix)
    {
        // Transpose logic remains 2D, then we reshape to 3D
        int32_t perm_dims[] = {1, 0};
        uint32_t dims_node = g.constant({2}, perm_dims, DType::INT32);

        auto project = [&](const std::string &suffix, uint32_t in_d, uint32_t out_d)
        {
            uint32_t w = weight(w_path, prefix + suffix);
            uint32_t w_t = g.permute(w, dims_node);
            int32_t s3[] = {1, (int32_t)in_d, (int32_t)out_d};
            return g.dot(x, g.reshape(w_t, g.constant({3}, s3, DType::INT32)));
        };

        uint32_t q = project(".self_attn.q_proj.weight", cfg.emb_dim, cfg.n_heads * cfg.head_dim);
        uint32_t k = project(".self_attn.k_proj.weight", cfg.emb_dim, cfg.n_kv_groups * cfg.head_dim);
        uint32_t v = project(".self_attn.v_proj.weight", cfg.emb_dim, cfg.n_kv_groups * cfg.head_dim);

        return std::make_tuple(q, k, v);
    }

    uint32_t attention_output_atomic(std::tuple<uint32_t, uint32_t, uint32_t> qkv, const std::string &prefix)
    {
        uint32_t q = std::get<0>(qkv);
        float scale_val = 1.0f / std::sqrt((float)cfg.query_pre_attn_scalar);
        uint32_t scale_node = expand_scalar_to_3d(g.constant({1}, &scale_val, DType::FLOAT32), 1, seq_len, cfg.n_heads * cfg.head_dim);
        uint32_t scaled_q = g.mul(q, scale_node);

        uint32_t w_o = weight(w_path, prefix + ".self_attn.o_proj.weight");
        int32_t perm_dims[] = {1, 0};
        uint32_t w_o_t = g.permute(w_o, g.constant({2}, perm_dims, DType::INT32));

        int32_t s3[] = {1, (int32_t)(cfg.n_heads * cfg.head_dim), (int32_t)cfg.emb_dim};
        uint32_t w_o_3d = g.reshape(w_o_t, g.constant({3}, s3, DType::INT32));

        return g.dot(scaled_q, w_o_3d);
    }

    uint32_t mlp_atomic(uint32_t x, const std::string &prefix)
    {
        int32_t perm_dims[] = {1, 0};
        uint32_t p_node = g.constant({2}, perm_dims, DType::INT32);

        auto project = [&](const std::string &suffix, uint32_t in_d, uint32_t out_d)
        {
            uint32_t w = weight(w_path, prefix + suffix);
            uint32_t w_t = g.permute(w, p_node);
            int32_t s3[] = {1, (int32_t)in_d, (int32_t)out_d};
            return g.dot(x, g.reshape(w_t, g.constant({3}, s3, DType::INT32)));
        };

        uint32_t gate = project(".mlp.gate_proj.weight", cfg.emb_dim, cfg.hidden_dim);
        gate = gelu_atomic(gate, cfg.hidden_dim);

        uint32_t up = project(".mlp.up_proj.weight", cfg.emb_dim, cfg.hidden_dim);
        uint32_t gate_up = g.mul(gate, up);

        uint32_t w_down = weight(w_path, prefix + ".mlp.down_proj.weight");
        uint32_t w_down_t = g.permute(w_down, p_node);
        int32_t s3[] = {1, (int32_t)cfg.hidden_dim, (int32_t)cfg.emb_dim};

        return g.dot(gate_up, g.reshape(w_down_t, g.constant({3}, s3, DType::INT32)));
    }

    uint32_t build_graph(uint32_t input_ids_id)
    {
        uint32_t w_emb = weight(w_path, "model.embed_tokens.weight");
        uint32_t x = g.gather(w_emb, input_ids_id);

        float scale_val = std::sqrt((float)cfg.emb_dim);
        uint32_t scale_node = expand_scalar_to_3d(g.constant({1}, &scale_val, DType::FLOAT32), 1, seq_len, cfg.emb_dim);
        x = g.mul(x, scale_node);

        for (uint32_t i = 0; i < cfg.n_layers; ++i)
        {
            std::string prefix = "model.layers." + std::to_string(i);

            uint32_t residual = x;

            uint32_t w_ln1 = weight(w_path, prefix + ".input_layernorm.weight");
            x = rms_norm_gemma_atomic(x, w_ln1);

            auto qkv = attention_qkv_atomic(x, prefix);
            x = attention_output_atomic(qkv, prefix);
            x = g.add(residual, x);

            residual = x;

            uint32_t w_ln2 = weight(w_path, prefix + ".pre_feedforward_layernorm.weight");
            x = rms_norm_gemma_atomic(x, w_ln2);
            x = mlp_atomic(x, prefix);
            x = g.add(residual, x);
        }

        uint32_t w_final_ln = weight(w_path, "model.norm.weight");
        x = rms_norm_gemma_atomic(x, w_final_ln);

        int32_t perm_dims[] = {1, 0};
        uint32_t dims_node = g.constant({2}, perm_dims, DType::INT32);
        uint32_t w_emb_t = g.permute(w_emb, dims_node);

        int32_t s3[] = {1, (int32_t)cfg.emb_dim, (int32_t)cfg.vocab_size};
        uint32_t w_emb_3d = g.reshape(w_emb_t, g.constant({3}, s3, DType::INT32));

        uint32_t logits = g.dot(x, w_emb_3d);

        return logits;
    }
};

int main()
{
    // Enable floating-point exceptions for invalid operations (NaN)
    _controlfp_s(nullptr, 0, 0);                                                 // optional: reset control word
    _controlfp_s(nullptr, _EM_INVALID | _EM_ZERODIVIDE | _EM_OVERFLOW, _MCW_EM); // enable invalid op exception
    std::vector<uint32_t> tokens = {2, 105, 2364, 107, 155122, 27825, 49087, 531, 496, 236743, 236810, 1051, 2255, 236761, 106, 107, 105, 4368, 107};
    // should output 70895 as argmax of output logits
    uint32_t maxSeqLen = 128;
    std::string modelPath = "C:/Users/aaron/CODING/tensor_graphs/resources/model.safetensors";

    ModelConfig cfg;
    MemoryManager mem;
    mem.buffers.emplace(Backend::CPU, DeviceBuffer(2ULL * 1024 * 1024 * 1024));

    Graph g;

    uint32_t inputIdsId = g.allocateId();
    uint64_t sizeBytes = maxSeqLen * getDTypeSize(DType::INT32);
    mem.allocate(Backend::CPU, inputIdsId, sizeBytes, StorageType::PERSISTENT);

    TensorView inputView;
    inputView.shape = {1, maxSeqLen};
    inputView.strides = TensorView::calcContiguousStrides(inputView.shape);
    inputView.dtype = DType::INT32;

    g.inputWithId(inputIdsId, {1, maxSeqLen}, DType::INT32, inputView, StorageType::PERSISTENT);

    std::cout << "Building Graph..." << std::endl;

    Gemma3Model model(cfg, maxSeqLen, g, mem, modelPath);
    uint32_t logits_id = model.build_graph(inputIdsId);

    std::cout << "Initializing Session..." << std::endl;
    Session session(g, mem, logits_id, "dirty_region_caches/gemma-3-270m-cpp.jsonl");

    mem.buffers.at(Backend::CPU).init();

    std::cout << "Running Inference..." << std::endl;

    std::vector<int32_t> input_data(maxSeqLen, 0);
    for (size_t i = 0; i < tokens.size(); ++i)
    {
        input_data[i] = (int32_t)tokens[i];
    }

    std::unordered_map<uint32_t, const void *> inputs;
    inputs[inputIdsId] = input_data.data();

    // The early "return 0;" from the old script before evaluation was stripped
    // to actually allow session execution to happen.
    session.run(inputs);

    const float *output_ptr = static_cast<const float *>(session.getOutput(logits_id));

    if (output_ptr)
    {
        // 1. Identify the offset for the next token
        uint32_t next_token_pos = (uint32_t)tokens.size();
        uint64_t offset = (uint64_t)next_token_pos * cfg.vocab_size;
        const float* next_token_logits = output_ptr + offset;

        // 2. Find Argmax
        float max_val = -FLT_MAX;
        int32_t argmax_idx = -1;

        for (uint32_t i = 0; i < cfg.vocab_size; ++i)
        {
            if (next_token_logits[i] > max_val)
            {
                max_val = next_token_logits[i];
                argmax_idx = i;
            }
        }

        std::cout << "\nInference Results:" << std::endl;
        std::cout << "------------------" << std::endl;
        std::cout << "Next Token Position: " << next_token_pos << std::endl;
        std::cout << "Predicted Token ID (Argmax): " << argmax_idx << std::endl;
        std::cout << "Logit Value: " << max_val << std::endl;

        std::cout << "\nFirst 5 logits for this token: " << std::endl;
        for (int i = 0; i < 5; ++i) {
            std::cout << "  [" << i << "]: " << next_token_logits[i] << std::endl;
        }
    }
    else
    {
        std::cerr << "Failed to retrieve output." << std::endl;
    }

    return 0;
}