#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryKreaSwiglu(const std::vector<LogicalId> &inputs, Graph &g)
{
    LogicalId x = inputs[0];
    LogicalId up = inputs[1];
    auto shape = g.getNode(x).getShape();

    LogicalId neg_x = g.mul(x, g.fill(-1.0f, shape));
    LogicalId exp_neg_x = g.pow(g.fill(TGConstants::E, shape), neg_x);
    LogicalId one = g.fill(1.0f, shape);
    LogicalId sig = g.div(one, g.add(one, exp_neg_x));
    LogicalId gate_silu = g.mul(x, sig);
    return g.mul(gate_silu, up);
}
