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

// Include kernels to ensure they are registered
#include "kernels/reference/add/F32_1D.hpp"
#include "kernels/reference/cast/BF16_F32_ND.hpp"
#include "kernels/fused/tanh/F32_1D.hpp"

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

    // Pre-allocate constants (all FP32)
    uint32_t one_fp32;
    uint32_t eps_fp32;
    uint32_t half_fp32;

public:
    Gemma3Model(ModelConfig config, Graph &graph, MemoryManager &memory, const std::string &weight_path)
        : cfg(config), g(graph), mem(memory), w_path(weight_path), eps(1e-6f)
    {
        // Pre-allocate constants (all FP32)
        one_fp32 = g.constant({1}, &eps, DType::FLOAT32, mem);
        eps_fp32 = g.constant({1}, &eps, DType::FLOAT32, mem);

        float half_val = 0.5f;
        half_fp32 = g.constant({1}, &half_val, DType::FLOAT32, mem);
    }

    uint32_t weight(const std::string &path, const std::string &name)
    {
        // Load the raw weight (may be BF16, FP16, etc.)
        uint32_t raw_weight = g.weight(path, name, mem);

        // Always cast to ensure consistency across the graph
        return g.cast(raw_weight, DType::FLOAT32);
    }

    uint32_t rms_norm_gemma_atomic(uint32_t x_id, uint32_t weight_id)
    {
        // 1. x * x (element-wise square)
        uint32_t x_sq = g.mul(x_id, x_id);

        // 2. sum(x^2, axis=-1, keepdims=True)
        int32_t axis_val = -1;
        uint32_t axis_node = g.constant({1}, &axis_val, DType::INT32, mem);

        bool keepdims_val = true;
        uint32_t keepdims_node = g.constant({1}, &keepdims_val, DType::BOOL, mem);

        uint32_t sum_sq = g.sum(x_sq, axis_node, keepdims_node);

        // 3. mean = sum / n (where n = shape[-1])
        float n_val = (float)cfg.emb_dim;
        uint32_t n_node = g.constant({1}, &n_val, DType::FLOAT32, mem);

        uint32_t mean_sq = g.div(sum_sq, n_node);

        // 4. mean_sq + eps
        uint32_t mean_sq_plus_eps = g.add(mean_sq, eps_fp32);

        // 5. sqrt(mean_sq + eps) using pow(x, 0.5)
        float half_val = 0.5f;
        uint32_t sqrt_node = g.constant({1}, &half_val, DType::FLOAT32, mem);

        uint32_t std = g.pow(mean_sq_plus_eps, sqrt_node);

        // 6. 1.0 / std
        float one_val = 1.0f;
        uint32_t one_node = g.constant({1}, &one_val, DType::FLOAT32, mem);
        uint32_t inv_std = g.div(one_node, std);

        // 7. x * inv_std (normalized)
        uint32_t x_norm = g.mul(x_id, inv_std);

        // 8. scale = 1 + weight (Gemma's offset scaling)
        // Ensure weight is FP32 before adding
        uint32_t weight_fp32 = g.cast(weight_id, DType::FLOAT32);
        uint32_t scale = g.add(weight_fp32, one_node);

        // 9. x_norm * scale
        return g.mul(x_norm, scale);
    }

    uint32_t gelu_atomic(uint32_t x_id)
    {
        // Constants
        float c1_val = 0.044715f;
        uint32_t c1_node = g.constant({1}, &c1_val, DType::FLOAT32, mem);

        float c2_val = 0.79788456f; // sqrt(2/pi)
        uint32_t c2_node = g.constant({1}, &c2_val, DType::FLOAT32, mem);

        // x^2
        uint32_t x_sq = g.mul(x_id, x_id);

        // x^3
        uint32_t x_cube = g.mul(x_sq, x_id);

        // 0.044715 * x^3
        uint32_t term1 = g.mul(x_cube, c1_node);

        // x + 0.044715 * x^3
        uint32_t term2 = g.add(x_id, term1);

        // sqrt(2/pi) * (x + 0.044715 * x^3)
        uint32_t term3 = g.mul(term2, c2_node);

        // tanh(term3) - using atomic decomposition
        uint32_t tanh_result = tanh_atomic(term3);

        // 1 + tanh(...)
        float one_val = 1.0f;
        uint32_t one_node = g.constant({1}, &one_val, DType::FLOAT32, mem);
        uint32_t term4 = g.add(one_node, tanh_result);

        // 0.5 * x
        uint32_t term5 = g.mul(x_id, half_fp32);

        // 0.5 * x * (1 + tanh(...))
        return g.mul(term5, term4);
    }

    uint32_t tanh_atomic(uint32_t x_id)
    {
        // -x
        uint32_t neg_x = g.neg(x_id);

        // e^x using pow(e, x)
        float e_val = 2.718281828459045f;
        uint32_t e_node = g.constant({1}, &e_val, DType::FLOAT32, mem);

        uint32_t exp_x = g.pow(e_node, x_id);
        uint32_t exp_neg_x = g.pow(e_node, neg_x);

        // -e^-x
        uint32_t neg_exp_neg = g.neg(exp_neg_x);

        // e^x - e^-x
        uint32_t num = g.add(exp_x, neg_exp_neg);

        // e^x + e^-x
        uint32_t den = g.add(exp_x, exp_neg_x);

        // (e^x - e^-x) / (e^x + e^-x)
        return g.div(num, den);
    }

    uint32_t build_graph(uint32_t input_ids_id)
    {
        // 1. Embedding - now uses helper that casts to FP32
        uint32_t w_emb = weight(w_path, "model.embed_tokens.weight");
        uint32_t x = g.gather(w_emb, input_ids_id);

        // Scale by sqrt(emb_dim)
        float scale_val = std::sqrt((float)cfg.emb_dim);
        uint32_t scale_node = g.constant({1}, &scale_val, DType::FLOAT32, mem);
        x = g.mul(x, scale_node);

        // 2. Layers
        for (uint32_t i = 0; i < cfg.n_layers; ++i)
        {
            std::string prefix = "model.layers." + std::to_string(i);

            // Pre-norm Residual Block
            uint32_t residual = x;

            // Load weight and cast to FP32 using helper
            uint32_t w_ln1 = weight(w_path, prefix + ".input_layernorm.weight");
            x = rms_norm_gemma_atomic(x, w_ln1);

            // Attention (simplified - using atomic ops for projections)
            auto qkv = attention_qkv_atomic(x, prefix);
            uint32_t q = std::get<0>(qkv);
            uint32_t k = std::get<1>(qkv);
            uint32_t v = std::get<2>(qkv);

            x = attention_output_atomic(qkv, prefix);
            x = g.add(residual, x);

            // MLP Block
            residual = x;

            uint32_t w_ln2 = weight(w_path, prefix + ".pre_feedforward_layernorm.weight");
            x = rms_norm_gemma_atomic(x, w_ln2);
            x = mlp_atomic(x, prefix);
            x = g.add(residual, x);
        }

        // 3. Final Norm & Head
        uint32_t w_final_ln = weight(w_path, "model.norm.weight");
        x = rms_norm_gemma_atomic(x, w_final_ln);

        // Weight tying - transpose embedding weights
        int32_t perm_dims[] = {1, 0};
        uint32_t dims_node = g.constant({2}, perm_dims, DType::INT32, mem);
        uint32_t w_emb_t = g.permute(w_emb, dims_node);
        uint32_t logits = g.dot(x, w_emb_t);

        return logits;
    }

    std::tuple<uint32_t, uint32_t, uint32_t> attention_qkv_atomic(uint32_t x, const std::string &prefix)
    {
        // Common dimensions node for permutation
        int32_t perm_dims[] = {1, 0};
        uint32_t dims_node = g.constant({2}, perm_dims, DType::INT32, mem);

        // Q projection - use weight helper for FP32 casting
        uint32_t w_q = weight(w_path, prefix + ".self_attn.q_proj.weight");
        uint32_t w_q_t = g.permute(w_q, dims_node);
        uint32_t q = g.dot(x, w_q_t);

        // K projection
        uint32_t w_k = weight(w_path, prefix + ".self_attn.k_proj.weight");
        uint32_t w_k_t = g.permute(w_k, dims_node);
        uint32_t k = g.dot(x, w_k_t);

        // V projection
        uint32_t w_v = weight(w_path, prefix + ".self_attn.v_proj.weight");
        uint32_t w_v_t = g.permute(w_v, dims_node);
        uint32_t v = g.dot(x, w_v_t);

        return std::make_tuple(q, k, v);
    }

    uint32_t attention_output_atomic(std::tuple<uint32_t, uint32_t, uint32_t> qkv, const std::string &prefix)
    {
        uint32_t q = std::get<0>(qkv);

        // For simplicity, just return a scaled version of the input
        float scale_val = 1.0f / std::sqrt((float)cfg.query_pre_attn_scalar);
        uint32_t scale_node = g.constant({1}, &scale_val, DType::FLOAT32, mem);

        return g.mul(q, scale_node);
    }

    uint32_t mlp_atomic(uint32_t x, const std::string &prefix)
    {
        // Common dimensions node for permutation
        int32_t perm_dims[] = {1, 0};
        uint32_t dims_node = g.constant({2}, perm_dims, DType::INT32, mem);

        // Gate projection - use weight helper for FP32 casting
        uint32_t w_gate = weight(w_path, prefix + ".mlp.gate_proj.weight");
        uint32_t w_gate_t = g.permute(w_gate, dims_node);
        uint32_t gate = g.dot(x, w_gate_t);
        gate = gelu_atomic(gate);

        // Up projection
        uint32_t w_up = weight(w_path, prefix + ".mlp.up_proj.weight");
        uint32_t w_up_t = g.permute(w_up, dims_node);
        uint32_t up = g.dot(x, w_up_t);

        // Gate * Up
        uint32_t gate_up = g.mul(gate, up);

        // Down projection
        uint32_t w_down = weight(w_path, prefix + ".mlp.down_proj.weight");
        uint32_t w_down_t = g.permute(w_down, dims_node);

        return g.dot(gate_up, w_down_t);
    }
};

int main()
{
    // 1. Setup Input Data
    std::vector<uint32_t> tokens = {2, 818, 6789, 531, 1972, 563};
    uint32_t maxSeqLen = 128;
    std::string modelPath = "C:/Users/aaron/CODING/tensor_graphs/resources/model.safetensors";

    ModelConfig cfg;
    MemoryManager mem;
    // Allocate 2GB for CPU
    mem.buffers.emplace(Backend::CPU, DeviceBuffer(2ULL * 1024 * 1024 * 1024));

    Graph g;

    // 2. Define Input Node
    TensorView inputView;
    uint32_t inputIdsId = g.input({1, maxSeqLen}, DType::INT32, inputView);

    // 3. Build Model Graph (using class)
    std::cout << "Building Graph..." << std::endl;

    Gemma3Model model(cfg, g, mem, modelPath);
    uint32_t logits_id = model.build_graph(inputIdsId);

    // 4. Session Initialization
    std::cout << "Initializing Session..." << std::endl;
    Session session(g, mem, logits_id, "gemma_cache.jsonl");

    // Actualize Memory Arena (Initialize physical buffers)
    mem.buffers.at(Backend::CPU).init();

    // 5. Run Inference
    std::cout << "Running Inference..." << std::endl;

    // Prepare input buffer
    std::vector<int32_t> input_data(maxSeqLen, 0);
    for (size_t i = 0; i < tokens.size(); ++i)
    {
        input_data[i] = (int32_t)tokens[i];
    }

    std::unordered_map<uint32_t, const void *> inputs;
    inputs[inputIdsId] = input_data.data();

    session.run(inputs);

    // 6. Get Results
    const float *output_ptr = static_cast<const float *>(session.getOutput(logits_id));

    if (output_ptr)
    {
        std::cout << "Inference successful. First 5 logits at last token: " << std::endl;
        // Index to the last token's logits: [batch=0, seq_idx=tokens.size()-1, vocab_idx=...]
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
