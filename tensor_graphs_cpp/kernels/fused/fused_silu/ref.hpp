#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryFusedSilu_CUDA(const std::vector<LogicalId> &inputs,
                                          Graph &g) {
    if (inputs.size() != 1) {
        Error::throw_err("FusedSilu_CUDA requires exactly 1 input.");
    }

    LogicalId x_id = inputs[0];
    const TensorNode &x_node = g.getNode(x_id);
    const std::vector<uint32_t> &shape = x_node.getShape();
    DType dtype = x_node.dtype;  // should be FLOAT32

    // Helper to create a fill node: constant scalar -> fill with shape
    auto fill_scalar = [&](float val) -> LogicalId {
        LogicalId scalar = g.constant({1}, &val, dtype);
        // Build a shape tensor constant (INT32) from the actual shape
        std::vector<int32_t> shape_int(shape.begin(), shape.end());
        LogicalId shape_node = g.constant({(uint32_t)shape.size()},
                                          shape_int.data(), DType::INT32);
        return g.fill(scalar, shape_node);
    };

    // Build the exact decomposition
    LogicalId neg_one = fill_scalar(-1.0f);
    LogicalId neg_x = g.mul(x_id, neg_one);

    float e_val = 2.718281828459045f;
    LogicalId e_node = fill_scalar(e_val);
    LogicalId exp_neg = g.pow(e_node, neg_x);

    LogicalId one_node = fill_scalar(1.0f);
    LogicalId den = g.add(one_node, exp_neg);
    LogicalId sig = g.div(one_node, den);

    return g.mul(x_id, sig);
}
