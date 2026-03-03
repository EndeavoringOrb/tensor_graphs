#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <tuple>

#include "core/types.hpp"
#include "core/memory.hpp"
#include "core/graph.hpp"
#include "core/session.hpp"
#include "core/kernels.hpp"

#include "kernels/reference/cast/BF16_F32_ND.hpp"
#include "kernels/fused/tanh/F32_1D.hpp"
#include "kernels/reference/gather/F32_I32_ND.hpp"

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

        bool keepdims_val = true;
        uint32_t keepdims_node = g.constant({1}, &keepdims_val, DType::BOOL);

        uint32_t sum_sq = g.sum(x_sq, axis_node, keepdims_node);

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

        uint32_t weight_fp32 = g.cast(weight_id, DType::FLOAT32);
        uint32_t weight_expanded = expand_1d_to_3d(weight_fp32, cfg.emb_dim, 1, seq_len);

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
        uint32_t neg_x = g.neg(x_id);

        float e_val = 2.718281828459045f;
        uint32_t e_node = expand_scalar_to_3d(g.constant({1}, &e_val, DType::FLOAT32), 1, seq_len, last_dim);

        uint32_t exp_x = g.pow(e_node, x_id);
        uint32_t exp_neg_x = g.pow(e_node, neg_x);

        uint32_t neg_exp_neg = g.neg(exp_neg_x);
        uint32_t num = g.add(exp_x, neg_exp_neg);
        uint32_t den = g.add(exp_x, exp_neg_x);

        return g.div(num, den);
    }

    std::tuple<uint32_t, uint32_t, uint32_t> attention_qkv_atomic(uint32_t x, const std::string &prefix)
    {
        int32_t perm_dims[] = {1, 0};
        uint32_t dims_node = g.constant({2}, perm_dims, DType::INT32);

        uint32_t w_q = weight(w_path, prefix + ".self_attn.q_proj.weight");
        uint32_t w_q_t = g.permute(w_q, dims_node);
        uint32_t q = g.dot(x, w_q_t);

        uint32_t w_k = weight(w_path, prefix + ".self_attn.k_proj.weight");
        uint32_t w_k_t = g.permute(w_k, dims_node);
        uint32_t k = g.dot(x, w_k_t);

        uint32_t w_v = weight(w_path, prefix + ".self_attn.v_proj.weight");
        uint32_t w_v_t = g.permute(w_v, dims_node);
        uint32_t v = g.dot(x, w_v_t);

        return std::make_tuple(q, k, v);
    }

    uint32_t attention_output_atomic(std::tuple<uint32_t, uint32_t, uint32_t> qkv, const std::string &prefix)
    {
        uint32_t q = std::get<0>(qkv); // Shape [1, 128, 1024]

        float scale_val = 1.0f / std::sqrt((float)cfg.query_pre_attn_scalar);
        uint32_t scale_node = expand_scalar_to_3d(g.constant({1}, &scale_val, DType::FLOAT32), 1, seq_len, cfg.n_heads * cfg.head_dim);
        uint32_t scaled_q = g.mul(q, scale_node);

        uint32_t w_o = weight(w_path, prefix + ".self_attn.o_proj.weight"); // Shape [640, 1024]

        int32_t perm_dims[] = {1, 0};
        uint32_t dims_node = g.constant({2}, perm_dims, DType::INT32);
        uint32_t w_o_t = g.permute(w_o, dims_node); // Shape[1024, 640]

        return g.dot(scaled_q, w_o_t);
    }

    uint32_t mlp_atomic(uint32_t x, const std::string &prefix)
    {
        int32_t perm_dims[] = {1, 0};
        uint32_t dims_node = g.constant({2}, perm_dims, DType::INT32);

        uint32_t w_gate = weight(w_path, prefix + ".mlp.gate_proj.weight");
        uint32_t w_gate_t = g.permute(w_gate, dims_node);
        uint32_t gate = g.dot(x, w_gate_t);
        gate = gelu_atomic(gate, cfg.hidden_dim);

        uint32_t w_up = weight(w_path, prefix + ".mlp.up_proj.weight");
        uint32_t w_up_t = g.permute(w_up, dims_node);
        uint32_t up = g.dot(x, w_up_t);

        uint32_t gate_up = g.mul(gate, up);

        uint32_t w_down = weight(w_path, prefix + ".mlp.down_proj.weight");
        uint32_t w_down_t = g.permute(w_down, dims_node);

        return g.dot(gate_up, w_down_t);
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
        uint32_t logits = g.dot(x, w_emb_t);

        return logits;
    }
};

int main()
{
    std::vector<uint32_t> tokens = {2, 818, 6789, 531, 1972, 563};
    uint32_t maxSeqLen = 128;
    std::string modelPath = "C:/Users/aaron/CODING/tensor_graphs/resources/model.safetensors";

    ModelConfig cfg;
    MemoryManager mem;
    // Boosted slightly from 2GB to 3GB since 2GB is right on the boundary after expanding weights
    // down the sequence dim + retaining intermediate activation nodes
    mem.buffers.emplace(Backend::CPU, DeviceBuffer(3ULL * 1024 * 1024 * 1024));

    Graph g;

    TensorView inputView;
    uint32_t inputIdsId = g.input({1, maxSeqLen}, DType::INT32, inputView, StorageType::TRANSIENT);

    std::cout << "Building Graph..." << std::endl;

    Gemma3Model model(cfg, maxSeqLen, g, mem, modelPath);
    uint32_t logits_id = model.build_graph(inputIdsId);

    std::cout << "Initializing Session..." << std::endl;
    Session session(g, mem, logits_id, "gemma_cache.jsonl");

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
        std::cout << "Inference successful. First 5 logits at last token: " << std::endl;
        uint64_t last_token_offset = (tokens.size() - 1) * cfg.vocab_size;
        for (int i = 0; i < 5; ++i)
        {
            std::cout << output_ptr[last_token_offset + i] << " ";
        }
        std::cout << std::endl;
    }
    else
    {
        std::cerr << "Failed to retrieve output." << std::endl;
    }

    return 0;
}