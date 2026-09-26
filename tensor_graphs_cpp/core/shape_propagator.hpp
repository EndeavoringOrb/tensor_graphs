#pragma once
#include "core/misc.hpp"
#include "core/ops/ops.hpp"
#include "core/reference_graph_registry.hpp"
#include "core/shapes.hpp"

struct ShapePropagator
{
    void inferShapeRecursive(LogicalId nodeId, Graph &graph)
    {
        if (!graph.hasNode(nodeId))
            return;

        if (!graph.getNode(nodeId).getShape().empty())
            return;

        if (graph.getNode(nodeId).opType == OpType::INPUT)
            return;

        for (LogicalId pid : graph.getNode(nodeId).child_ids)
        {
            inferShapeRecursive(pid, graph);
        }

        inferShape(nodeId, graph);
    }

    void inferShape(LogicalId nodeId, Graph &graph)
    {
        if (!graph.hasNode(nodeId) || !graph.getNode(nodeId).getShape().empty())
            return;
        if (graph.getNode(nodeId).opType == OpType::INPUT)
            return;

        const auto &traits = getOpTraits(graph.getNode(nodeId).opType);
        if (traits.inferShape)
            traits.inferShape(nodeId, graph);
        else
            Error::throw_err("[ShapePropagator.inferShape] Unsupported OpType: " +
                             toString(graph.getNode(nodeId).opType));

        for (auto d : graph.getNode(nodeId).getShape())
        {
            if (d == 0)
                Error::throw_err("Zero-sized dimension in tensor shape!" + toString(graph.getNode(nodeId), graph, ""));
        }
    }

    std::vector<Region> forward(const TensorNode &node, const Graph &graph,
                                const std::vector<std::vector<Region>> &parentRegions)
    {
        const auto &traits = getOpTraits(node.opType);
        if (traits.forwardRegion)
            return traits.forwardRegion(node, graph, parentRegions);
        Error::throw_err("[ShapePropagator.forward] Unsupported OpType: " + toString(node.opType));
    }

    std::vector<std::vector<Region>> backward(const TensorNode &node, const Graph &graph,
                                              const std::vector<Region> &outputRegions)
    {
        const auto &traits = getOpTraits(node.opType);
        if (traits.backwardRegion)
            return traits.backwardRegion(node, graph, outputRegions);
        Error::throw_err("[ShapePropagator.backward] Unsupported OpType: " + toString(node.opType));
    }
};

inline WorkloadMetrics computeWorkloadFromRefFactory(
    ReferenceFactory factory,
    const std::vector<std::vector<uint32_t>> &in_shapes,
    const std::vector<DType> &in_dtypes,
    const std::vector<uint32_t> &out_shape,
    DType out_dtype,
    const std::vector<std::vector<uint8_t>> &in_constants,
    OpType reference_op)
{
    if (!factory || in_shapes.empty())
    {
        return op_common::defaultWorkload(in_shapes, in_dtypes, out_shape, out_dtype, 0.0);
    }

    try
    {
        Graph ref_graph;
        std::vector<LogicalId> ref_inputs;
        ref_inputs.reserve(in_shapes.size());

        auto default_control_values = [&](size_t input_idx) {
            std::vector<int32_t> values(countElements(in_shapes[input_idx]), 0);
            const auto &source_shape = in_shapes.empty() ? out_shape : in_shapes[0];
            const auto &target_shape = out_shape.empty() ? source_shape : out_shape;

            auto copy_shape = [&](const std::vector<uint32_t> &shape) {
                for (size_t i = 0; i < values.size() && i < shape.size(); ++i)
                    values[i] = static_cast<int32_t>(shape[i]);
            };

            switch (reference_op)
            {
            case OpType::PERMUTE:
                for (size_t i = 0; i < values.size(); ++i)
                    values[i] = source_shape.empty() ? 0 : static_cast<int32_t>(i % source_shape.size());
                break;
            case OpType::SLICE:
                if (input_idx == 2)
                    copy_shape(source_shape); // ends
                else if (input_idx == 3)
                    std::fill(values.begin(), values.end(), 1); // steps
                break;
            case OpType::SCATTER:
                if (input_idx == 2 || input_idx == 4)
                    copy_shape(target_shape); // ends / output shape
                else if (input_idx == 3)
                    std::fill(values.begin(), values.end(), 1); // steps
                break;
            case OpType::RESHAPE:
            case OpType::FILL:
                copy_shape(target_shape);
                break;
            case OpType::CONCAT:
            case OpType::SUM:
            case OpType::MAX:
            case OpType::ARGMAX:
            case OpType::TRIU:
                // Zero is a valid axis/diagonal.  ARGMAX's k input is
                // corrected below so it remains positive.
                break;
            case OpType::REPEAT:
                if (input_idx == 1)
                    std::fill(values.begin(), values.end(), 1);
                break;
            case OpType::ARANGE:
                if (input_idx == 1)
                    std::fill(values.begin(), values.end(), 1);
                else if (input_idx == 2)
                    std::fill(values.begin(), values.end(), 1);
                break;
            case OpType::IM2COL:
                if (input_idx == 1 || input_idx == 2)
                    std::fill(values.begin(), values.end(), 1);
                break;
            default:
                break;
            }

            if (reference_op == OpType::ARGMAX && input_idx == 2)
                std::fill(values.begin(), values.end(), 1); // k
            return values;
        };

        for (size_t i = 0; i < in_shapes.size(); ++i)
        {
            DType dt = (i < in_dtypes.size()) ? in_dtypes[i] : DType::FLOAT32;
            const std::vector<uint8_t> *provided =
                (i < in_constants.size() && !in_constants[i].empty()) ? &in_constants[i] : nullptr;
            uint64_t expected_bytes = getSizeBytes(in_shapes[i], dt);

            // Reference factories perform shape inference on control inputs.
            // Represent known constant inputs as Graph::constant nodes so the
            // normal constant staging path is used instead of attaching data
            // to a runtime Graph::input node.
            if (provided != nullptr && provided->size() == expected_bytes)
            {
                LogicalId in_id = ref_graph.constant(in_shapes[i], provided->data(), dt);
                ref_inputs.push_back(in_id);
                continue;
            }

            if (dt == DType::INT32 && isConstant(reference_op, i, in_shapes.size()))
            {
                std::vector<int32_t> values = default_control_values(i);
                if (values.size() * sizeof(int32_t) == expected_bytes)
                {
                    LogicalId in_id = ref_graph.constant(in_shapes[i], values.data(), dt);
                    ref_inputs.push_back(in_id);
                    continue;
                }
            }

            LogicalId in_id = ref_graph.input(in_shapes[i], dt);
            if (provided != nullptr)
            {
                ref_graph.constantStaging[in_id] = std::make_shared<std::vector<uint8_t>>(*provided);
                ref_graph.input_data_types[in_id] = InputDataType::CONSTANT;
            }
            ref_inputs.push_back(in_id);
        }

        LogicalId root_id = factory(ref_inputs, ref_graph);
        if (!ref_graph.hasNode(root_id))
        {
            return op_common::defaultWorkload(in_shapes, in_dtypes, out_shape, out_dtype, 0.0);
        }

        ShapePropagator prop;
        prop.inferShapeRecursive(root_id, ref_graph);

        std::vector<LogicalId> topo = topologicalSort({root_id}, ref_graph);

        double total_flops = 0.0;
        for (LogicalId node_id : topo)
        {
            const TensorNode &node = ref_graph.getNode(node_id);
            if (node.opType == OpType::INPUT)
            {
                continue;
            }

            std::vector<std::vector<uint32_t>> child_shapes;
            std::vector<DType> child_dtypes;
            child_shapes.reserve(node.child_ids.size());
            child_dtypes.reserve(node.child_ids.size());

            for (LogicalId child_id : node.child_ids)
            {
                if (ref_graph.hasNode(child_id))
                {
                    const TensorNode &child_node = ref_graph.getNode(child_id);
                    child_shapes.push_back(child_node.getShape());
                    child_dtypes.push_back(child_node.dtype);
                }
            }

            WorkloadMetrics node_w = computeWorkload(node.opType, child_shapes, child_dtypes,
                                                     node.getShape(), node.dtype, node.opName, {}, nullptr);
            total_flops += node_w.flops;
        }

        const auto &actual_out_shape = (!out_shape.empty()) ? out_shape : ref_graph.getNode(root_id).getShape();
        return op_common::defaultWorkload(in_shapes, in_dtypes, actual_out_shape, out_dtype, total_flops);
    }
    catch (...)
    {
        return op_common::defaultWorkload(in_shapes, in_dtypes, out_shape, out_dtype, 0.0);
    }
}

inline WorkloadMetrics FusedOp::computeWorkload(const std::vector<std::vector<uint32_t>> &in_shapes,
                                                const std::vector<DType> &in_dtypes,
                                                const std::vector<uint32_t> &out_shape,
                                                DType out_dtype, const std::string &op_name)
{
    const auto *entry = ReferenceGraphRegistry::get().getFactory(op_name);
    if (entry && entry->factory)
    {
        return computeWorkloadFromRefFactory(entry->factory, in_shapes, in_dtypes, out_shape, out_dtype, {}, OpType::FUSED);
    }
    return op_common::defaultWorkload(in_shapes, in_dtypes, out_shape, out_dtype, 0.0);
}
