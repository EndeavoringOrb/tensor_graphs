// File: tensor_graphs_cpp/kernels/cpu/general/rmsnorm/F32_3D.hpp
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cmath>

inline bool matchRMSNormF32_3D(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].dtype != DType::FLOAT32 || inputs[1].dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32)
        return false;
    if (inputs[0].getShape().size() != 3 || inputs[1].getShape().size() != 1)
        return false;
    if (inputs[0].getShape()[2] != inputs[1].getShape()[0])
        return false;
    if (output.getShape() != inputs[0].getShape())
        return false;
    if (!isContiguous(output))
        return false;
    return true;
}

inline void runRMSNormF32_3D(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                             const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *x = static_cast<const float *>(inputs[0]);
    const float *w = static_cast<const float *>(inputs[1]);
    float *out = static_cast<float *>(outputs[0]);

    uint32_t B = inViews[0].getShape()[0];
    uint32_t S = inViews[0].getShape()[1];
    uint32_t D = inViews[0].getShape()[2];

    float eps = 1e-6f;
    for (uint32_t b = 0; b < B; ++b)
    {
        for (uint32_t s = 0; s < S; ++s)
        {
            const float *x_row = x + b * S * D + s * D;
            float *out_row = out + b * S * D + s * D;

            float sum_sq = 0.0f;
            for (uint32_t d = 0; d < D; ++d)
            {
                sum_sq += x_row[d] * x_row[d];
            }
            float mean_sq = sum_sq / D;
            float inv_std = 1.0f / std::sqrt(mean_sq + eps);

            for (uint32_t d = 0; d < D; ++d)
            {
                out_row[d] = x_row[d] * inv_std * (w[d] + 1.0f);
            }
        }
    }
}

// Helper: Broadcast a scalar to [B, S, 1] or [B, S, D]
inline uint32_t ref_rms_broadcast_scalar(Graph &g, uint32_t scalar_id, uint32_t B, uint32_t S, uint32_t D, bool full_d = false)
{
    int32_t shape_3d[] = {1, 1, 1};
    uint32_t out = g.reshape(scalar_id, g.constant({3}, shape_3d, DType::INT32));

    int32_t b_rep = (int32_t)B;
    int32_t s_rep = (int32_t)S;
    int32_t b_ax = 0, s_ax = 1;

    out = g.repeat(out, g.constant({1}, &b_rep, DType::INT32), g.constant({1}, &b_ax, DType::INT32));
    out = g.repeat(out, g.constant({1}, &s_rep, DType::INT32), g.constant({1}, &s_ax, DType::INT32));

    if (full_d)
    {
        int32_t d_rep = (int32_t)D;
        int32_t d_ax = 2;
        out = g.repeat(out, g.constant({1}, &d_rep, DType::INT32), g.constant({1}, &d_ax, DType::INT32));
    }
    return out;
}

inline uint32_t refFactoryRMSNorm(const std::vector<uint32_t> &inputs, Graph &graph)
{
    uint32_t x_id = inputs[0];
    uint32_t weight_id = inputs[1];

    auto shapeX = graph.getNode(x_id).getShape();
    uint32_t B = shapeX[0];
    uint32_t S = shapeX[1];
    uint32_t D = shapeX[2];

    // 1. Compute Mean Square
    uint32_t x_sq = graph.mul(x_id, x_id);
    int32_t axis_val = -1;
    uint32_t axis_node = graph.constant({1}, &axis_val, DType::INT32);
    uint32_t sum_sq = graph.sum(x_sq, axis_node);

    float d_float = (float)D;
    uint32_t n_node = ref_rms_broadcast_scalar(graph, graph.constant({1}, &d_float, DType::FLOAT32), B, S, 1, false);
    uint32_t mean_sq = graph.div(sum_sq, n_node);

    // 2. Compute inv_std: 1.0 / sqrt(mean_sq + eps)
    float eps = 1e-6f;
    uint32_t eps_expanded = ref_rms_broadcast_scalar(graph, graph.constant({1}, &eps, DType::FLOAT32), B, S, 1, false);
    uint32_t mean_sq_plus_eps = graph.add(mean_sq, eps_expanded);

    float half_val = 0.5f;
    uint32_t sqrt_exp = ref_rms_broadcast_scalar(graph, graph.constant({1}, &half_val, DType::FLOAT32), B, S, 1, false);
    uint32_t std_dev = graph.pow(mean_sq_plus_eps, sqrt_exp);

    float one_val = 1.0f;
    uint32_t one_node = ref_rms_broadcast_scalar(graph, graph.constant({1}, &one_val, DType::FLOAT32), B, S, 1, false);
    uint32_t inv_std = graph.div(one_node, std_dev);

    // 3. Normalize: x * inv_std
    int32_t d_rep = (int32_t)D;
    int32_t d_ax = 2;
    uint32_t inv_std_expanded = graph.repeat(inv_std, graph.constant({1}, &d_rep, DType::INT32), graph.constant({1}, &d_ax, DType::INT32));
    uint32_t x_norm = graph.mul(x_id, inv_std_expanded);

    // 4. Scale: x_norm * (weight + 1.0)
    // Expand weight [D] -> [1, 1, D] -> [B, S, D]
    int32_t reshape_dims[] = {1, 1, (int32_t)D};
    uint32_t w_reshaped = graph.reshape(weight_id, graph.constant({3}, reshape_dims, DType::INT32));
    int32_t b_rep = (int32_t)B;
    int32_t s_rep = (int32_t)S;
    int32_t b_ax = 0, s_ax = 1;
    uint32_t w_expanded = graph.repeat(w_reshaped, graph.constant({1}, &b_rep, DType::INT32), graph.constant({1}, &b_ax, DType::INT32));
    w_expanded = graph.repeat(w_expanded, graph.constant({1}, &s_rep, DType::INT32), graph.constant({1}, &s_ax, DType::INT32));

    uint32_t one_full = ref_rms_broadcast_scalar(graph, graph.constant({1}, &one_val, DType::FLOAT32), B, S, D, true);
    uint32_t scale = graph.add(w_expanded, one_full);

    return graph.mul(x_norm, scale);
}

REGISTER_KERNEL("RMSNorm", 2, matchRMSNormF32_3D, runRMSNormF32_3D, refFactoryRMSNorm, {Backend::CPU}, {DType::FLOAT32, DType::FLOAT32}, {{1, 1, 640}, {640}}, {true, true}, {{Backend::CPU}, {Backend::CPU}});
