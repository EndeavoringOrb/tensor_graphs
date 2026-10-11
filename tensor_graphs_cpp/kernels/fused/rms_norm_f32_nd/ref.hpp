#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryRMSNorm_F32_CUDA_ND(const std::vector<LogicalId>& inputs,
                                               Graph& graph)
{
    LogicalId x_id = inputs[0];
    LogicalId w_id = inputs[1];

    const auto& xShape = graph.getNode(x_id).getShape();
    uint32_t dim0     = xShape[0];
    uint32_t seq_len  = xShape[1];
    uint32_t dim_size = xShape[2];

    // Helper: expand a scalar constant to a 3D tensor with shape [dim0, seq_len, 1]
    auto expand_scalar_to_3d_1 = [&](float val) -> LogicalId {
        LogicalId node = graph.constant({1}, &val, DType::FLOAT32);
        int32_t sh3[] = {1, 1, 1};
        LogicalId out = graph.reshape(node, graph.constant({3}, sh3, DType::INT32));
        // repeat along axis 0 and 1 if needed
        if (dim0 > 1) {
            int32_t rep = (int32_t)dim0;
            int32_t ax = 0;
            out = graph.repeat(out, graph.constant({1}, &rep, DType::INT32),
                               graph.constant({1}, &ax, DType::INT32));
        }
        if (seq_len > 1) {
            int32_t rep = (int32_t)seq_len;
            int32_t ax = 1;
            out = graph.repeat(out, graph.constant({1}, &rep, DType::INT32),
                               graph.constant({1}, &ax, DType::INT32));
        }
        return out;
    };

    // 1. x_sq = x * x
    LogicalId x_sq = graph.mul(x_id, x_id);

    // 2. sum_sq = sum(x_sq, axis=-1)  => shape [dim0, seq_len, 1]
    int32_t axis_val = -1;
    LogicalId axis_node = graph.constant({1}, &axis_val, DType::INT32);
    LogicalId sum_sq = graph.sum(x_sq, axis_node);

    // 3. mean_sq = sum_sq / dim_size
    //    expand scalar dim_size to [dim0, seq_len, 1]
    float dim_size_f = (float)dim_size;
    LogicalId n_node = expand_scalar_to_3d_1(dim_size_f);
    LogicalId mean_sq = graph.div(sum_sq, n_node);

    // 4. var = mean_sq + eps
    float eps = 1e-6f;   // matches norm_eps in DeepSeekV4FlashConfig
    LogicalId eps_node = expand_scalar_to_3d_1(eps);
    LogicalId var = graph.add(mean_sq, eps_node);

    // 5. std = sqrt(var) = pow(var, 0.5)
    float half = 0.5f;
    LogicalId half_node = expand_scalar_to_3d_1(half);
    LogicalId std = graph.pow(var, half_node);

    // 6. inv_std = 1 / std   (shape [dim0, seq_len, 1])
    float one = 1.0f;
    LogicalId one_node = expand_scalar_to_3d_1(one);
    LogicalId inv_std = graph.div(one_node, std);

    // 7. inv_std_expanded = repeat(inv_std, dim_size, axis=2)
    //    => shape [dim0, seq_len, dim_size]
    int32_t rep_dim = (int32_t)dim_size;
    int32_t ax2 = 2;
    LogicalId inv_std_expanded = graph.repeat(inv_std,
                                              graph.constant({1}, &rep_dim, DType::INT32),
                                              graph.constant({1}, &ax2, DType::INT32));

    // 8. x_norm = x * inv_std_expanded
    LogicalId x_norm = graph.mul(x_id, inv_std_expanded);

    // 9. weight_expanded: reshape weight to [1, 1, dim_size], repeat to [dim0, seq_len, dim_size]
    int32_t w_shape[] = {1, 1, (int32_t)dim_size};
    LogicalId w_reshaped = graph.reshape(w_id, graph.constant({3}, w_shape, DType::INT32));
    LogicalId w_exp = w_reshaped;
    if (dim0 > 1) {
        int32_t rep0 = (int32_t)dim0;
        int32_t ax0 = 0;
        w_exp = graph.repeat(w_exp, graph.constant({1}, &rep0, DType::INT32),
                             graph.constant({1}, &ax0, DType::INT32));
    }
    if (seq_len > 1) {
        int32_t rep1 = (int32_t)seq_len;
        int32_t ax1 = 1;
        w_exp = graph.repeat(w_exp, graph.constant({1}, &rep1, DType::INT32),
                             graph.constant({1}, &ax1, DType::INT32));
    }

    // 10. out = x_norm * w_exp
    return graph.mul(x_norm, w_exp);
}
