// File: tensor_graphs_cpp/kernels/cpu/general/rmsnorm/F32_3D.hpp
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cmath>

inline bool matchRMSNormF32_3D(const std::vector<TensorNode> &inputs, const TensorNode &output, const std::unordered_map<uint32_t, uint32_t> &refCounts)
{
    if (inputs.size() != 2)
        return false;
    if (inputs[0].dtype != DType::FLOAT32 || inputs[1].dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32)
        return false;
    if (inputs[0].getShape().size() != 3 || inputs[1].getShape().size() != 1)
        return false;
    if (inputs[0].getShape()[2] != inputs[1].getShape()[0])
        return false;
    if (output.getShape() != inputs[0].getShape())
        return false;
    if (!isContiguous(inputs[0]) || !isContiguous(inputs[1]) || !isContiguous(output))
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

inline uint32_t ref_rms_expand_scalar(Graph &g, uint32_t scalar_id, bool rep2 = false)
{
    int32_t shape_3d[] = {1, 1, 1};
    uint32_t out = g.reshape(scalar_id, g.constant({3}, shape_3d, DType::INT32));
    int32_t rep = 1;
    int32_t a0 = 0, a1 = 1, a2 = 2;
    uint32_t rN = g.constant({1}, &rep, DType::INT32);
    out = g.repeat(out, rN, g.constant({1}, &a0, DType::INT32));
    out = g.repeat(out, rN, g.constant({1}, &a1, DType::INT32));
    if (rep2)
    {
        out = g.repeat(out, rN, g.constant({1}, &a2, DType::INT32));
    }
    return out;
}

inline uint32_t ref_rms_expand_1d(Graph &g, uint32_t vec_id)
{
    int32_t shape_3d[] = {1, 1, 1}; // dummy
    uint32_t out = g.reshape(vec_id, g.constant({3}, shape_3d, DType::INT32));
    int32_t rep = 1;
    int32_t a0 = 0, a1 = 1;
    uint32_t rN = g.constant({1}, &rep, DType::INT32);
    out = g.repeat(out, rN, g.constant({1}, &a0, DType::INT32));
    out = g.repeat(out, rN, g.constant({1}, &a1, DType::INT32));
    return out;
}

inline uint32_t refFactoryRMSNorm(const std::vector<uint32_t> &inputs, Graph &graph)
{
    uint32_t x_id = inputs[0];
    uint32_t weight_id = inputs[1];

    uint32_t x_sq = graph.mul(x_id, x_id);
    int32_t axis_val = -1;
    uint32_t axis_node = graph.constant({1}, &axis_val, DType::INT32);

    uint32_t sum_sq = graph.sum(x_sq, axis_node);

    float n_val = 1.0f; // dummy
    uint32_t n_node = ref_rms_expand_scalar(graph, graph.constant({1}, &n_val, DType::FLOAT32), false);

    uint32_t mean_sq = graph.div(sum_sq, n_node);

    float eps = 1e-6f;
    uint32_t eps_expanded = ref_rms_expand_scalar(graph, graph.constant({1}, &eps, DType::FLOAT32), false);
    uint32_t mean_sq_plus_eps = graph.add(mean_sq, eps_expanded);

    float half_val = 0.5f;
    uint32_t sqrt_node = ref_rms_expand_scalar(graph, graph.constant({1}, &half_val, DType::FLOAT32), false);
    uint32_t std = graph.pow(mean_sq_plus_eps, sqrt_node);

    float one_val = 1.0f;
    uint32_t one_fp32 = graph.constant({1}, &one_val, DType::FLOAT32);
    uint32_t one_node = ref_rms_expand_scalar(graph, one_fp32, false);
    uint32_t inv_std = graph.div(one_node, std);

    int32_t rep = 1;
    int32_t a2 = 2;
    uint32_t inv_std_expanded = graph.repeat(inv_std, graph.constant({1}, &rep, DType::INT32), graph.constant({1}, &a2, DType::INT32));

    uint32_t x_norm = graph.mul(x_id, inv_std_expanded);

    uint32_t weight_expanded = ref_rms_expand_1d(graph, weight_id);
    uint32_t one_node_full = ref_rms_expand_scalar(graph, one_fp32, true);
    uint32_t scale = graph.add(weight_expanded, one_node_full);

    return graph.mul(x_norm, scale);
}

REGISTER_KERNEL("RMSNorm", 2, matchRMSNormF32_3D, runRMSNormF32_3D, refFactoryRMSNorm, {Backend::CPU}, {DType::FLOAT32, DType::FLOAT32}, {{1, 1, 640}, {640}}, {true, true});