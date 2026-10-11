#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryJinaSwiGLU_F32_3D(const std::vector<LogicalId> &inputs, Graph &g)
{
    LogicalId gate_id = inputs[0];
    LogicalId up_id = inputs[1];
    const auto &shape = g.getNode(gate_id).getShape();
    uint32_t S = shape[1];
    uint32_t D = shape[2];

    auto expand_scalar_SD = [&](float val) -> LogicalId {
        LogicalId node = g.constant({1}, &val, DType::FLOAT32);
        int32_t sh[] = {1, 1, 1};
        LogicalId out = g.reshape(node, g.constant({3}, sh, DType::INT32));
        if (S > 1)
        {
            int32_t rep = (int32_t)S;
            int32_t ax = 1;
            out = g.repeat(out, g.constant({1}, &rep, DType::INT32), g.constant({1}, &ax, DType::INT32));
        }
        if (D > 1)
        {
            int32_t rep = (int32_t)D;
            int32_t ax = 2;
            out = g.repeat(out, g.constant({1}, &rep, DType::INT32), g.constant({1}, &ax, DType::INT32));
        }
        return out;
    };

    // --- silu_atomic(gate) ---
    LogicalId neg_one = expand_scalar_SD(-1.0f);
    LogicalId neg_gate = g.mul(gate_id, neg_one);
    LogicalId e_node = expand_scalar_SD(2.718281828459045f);
    LogicalId exp_neg = g.pow(e_node, neg_gate);
    LogicalId one_node = expand_scalar_SD(1.0f);
    LogicalId den = g.add(one_node, exp_neg);
    LogicalId sig = g.div(one_node, den);
    LogicalId gate_silu = g.mul(gate_id, sig);

    // --- mul(gate_silu, up) ---
    return g.mul(gate_silu, up_id);
}
