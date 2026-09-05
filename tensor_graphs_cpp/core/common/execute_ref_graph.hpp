// tensor_graphs_cpp/core/common/execute_ref_graph.hpp
#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/graph.hpp"
#include "core/kernels.hpp"
#include "core/loaders/loader.hpp"
#include "core/memory.hpp"
#include "core/misc.hpp"
#include "core/repo.hpp"
#include "core/shape_propagator.hpp"
#include "core/types.hpp"

struct RefGraphOptions
{
    const std::unordered_map<LogicalId, std::vector<uint8_t>> *raw_input_data = nullptr;
    const std::vector<LogicalId> *dynamic_inputs = nullptr;
    bool only_clean_nodes = false;
    bool fold_weights = false;
    bool force_non_contiguous = false;
};

inline std::vector<float> extractFloatTensor(LogicalId node_id, const Graph &graph, const TensorView &view,
                                             const void *data)
{
    const TensorNode &node = graph.getNode(node_id);
    uint64_t num_elems = countElements(node);
    std::vector<float> final_out(num_elems, 0.0f);
    const uint8_t *src_bytes = static_cast<const uint8_t *>(data);

    for (uint64_t i = 0; i < num_elems; ++i)
    {
        uint64_t idx = getStridedIndex(i, view.getShape(), view.strides);
        if (node.dtype == DType::FLOAT32)
        {
            std::memcpy(&final_out[i], src_bytes + idx * 4, 4);
        }
        else if (node.dtype == DType::INT32)
        {
            int32_t val;
            std::memcpy(&val, src_bytes + idx * 4, 4);
            final_out[i] = static_cast<float>(val);
        }
        else if (node.dtype == DType::BF16)
        {
            uint16_t val;
            std::memcpy(&val, src_bytes + idx * 2, 2);
            uint32_t f32_bits = static_cast<uint32_t>(val) << 16;
            std::memcpy(&final_out[i], &f32_bits, 4);
        }
        else if (node.dtype == DType::BOOL)
        {
            uint8_t val;
            std::memcpy(&val, src_bytes + idx, 1);
            final_out[i] = static_cast<float>(val);
        }
        else
        {
            Error::throw_err("[extractFloatTensor] Unsupported dtype: " + toString(node.dtype));
        }
    }
    return final_out;
}

inline std::vector<float> executeReferenceGraph(Graph &graph, const std::vector<LogicalId> &root_ids,
                                                ITensorStore &store, const RefGraphOptions &options = {})
{
    std::vector<LogicalId> topo = topologicalSort(root_ids, graph);

    std::unordered_set<LogicalId> dynamic_input_set;
    if (options.dynamic_inputs)
    {
        dynamic_input_set.insert(options.dynamic_inputs->begin(), options.dynamic_inputs->end());
    }

    std::unordered_map<LogicalId, bool> is_clean;
    if (options.only_clean_nodes)
    {
        for (LogicalId node_id : topo)
        {
            const TensorNode &node = graph.getNode(node_id);
            if (node.opType == OpType::INPUT)
            {
                if (dynamic_input_set.count(node_id) > 0)
                {
                    is_clean[node_id] = false;
                }
                else
                {
                    InputDataType input_type = graph.getInputDataType(node_id);
                    if (input_type == InputDataType::RUNTIME)
                    {
                        is_clean[node_id] = false;
                    }
                    else if (options.fold_weights)
                    {
                        is_clean[node_id] = (input_type == InputDataType::CONSTANT ||
                                             input_type == InputDataType::STORAGE);
                    }
                    else
                    {
                        is_clean[node_id] = (input_type == InputDataType::CONSTANT);
                    }
                }
            }
            else
            {
                bool all_clean = true;
                for (LogicalId pid : node.child_ids)
                {
                    if (!is_clean[pid])
                    {
                        all_clean = false;
                        break;
                    }
                }
                is_clean[node_id] = all_clean;
            }
        }
    }

    ShapePropagator prop;
    for (LogicalId node_id : topo)
    {
        if (options.only_clean_nodes && !is_clean[node_id])
            continue;
        if (graph.getNode(node_id).opType == OpType::INPUT)
            continue;

        prop.inferShape(node_id, graph);
    }

    std::unordered_map<LogicalId, std::vector<uint8_t>> results;
    std::unordered_map<LogicalId, TensorView> views;

    for (LogicalId node_id : topo)
    {
        if (options.only_clean_nodes && !is_clean[node_id])
            continue;

        if (options.only_clean_nodes && store.has(node_id))
            continue;

        const TensorNode &node = graph.getNode(node_id);
        uint64_t elem_size = getDTypeSize(node.dtype);

        if (node.opType == OpType::INPUT || node.opType == OpType::CACHE)
        {
            TensorView view = makeView(node);
            if (options.force_non_contiguous)
            {
                for (auto &s : view.strides)
                    s *= 2;
            }
            views[node_id] = view;
            uint64_t buf_elements = getRequiredBufferSize(view);
            results[node_id].resize(buf_elements * elem_size, 0);

            std::vector<uint8_t> raw_bytes;
            if (options.raw_input_data && options.raw_input_data->count(node_id))
            {
                raw_bytes = options.raw_input_data->at(node_id);
            }
            else if (graph.constantStaging.count(node_id))
            {
                raw_bytes = *graph.constantStaging.at(node_id);
            }
            else if (graph.input_data_types.count(node_id) &&
                     graph.input_data_types.at(node_id) == InputDataType::STORAGE)
            {
                TensorMetadata meta = FileRegistry::get().getNodeMeta(node_id);
                uint64_t size_bytes = meta.dataOffsetEnd - meta.dataOffsetStart;
                raw_bytes.resize(size_bytes);

                std::ifstream file(meta.filePath, std::ios::binary);
                if (!file.is_open())
                {
                    Error::throw_err("[executeReferenceGraph] Failed to open model file: " + meta.filePath);
                }
                file.seekg(meta.dataOffsetStart, std::ios::beg);
                file.read(reinterpret_cast<char *>(raw_bytes.data()), size_bytes);
            }
            else if (store.has(node_id))
            {
                raw_bytes = store.read(node_id);
            }
            else
            {
                Error::throw_err("[executeReferenceGraph] Input node value not found for node: " +
                                 toString(node_id));
            }

            uint64_t num_elements = countElements(view);
            for (uint64_t i = 0; i < num_elements; ++i)
            {
                uint64_t idx = getStridedIndex(i, view.getShape(), view.strides);
                std::memcpy(results[node_id].data() + idx * elem_size, raw_bytes.data() + i * elem_size, elem_size);
            }

            store.write(node_id, view, results[node_id].data(), results[node_id].size());
            continue;
        }

        std::vector<const void *> input_ptrs;
        std::vector<TensorView> input_views;
        std::vector<TensorNode> input_nodes;
        for (LogicalId pid : node.child_ids)
        {
            if (results.find(pid) == results.end())
            {
                if (store.has(pid))
                {
                    results[pid] = store.read(pid);
                    views[pid] = makeView(graph.getNode(pid));
                }
                else
                {
                    Error::throw_err("Parent node " + std::to_string(pid.value) + " not found in results or store");
                }
            }
            input_ptrs.push_back(results[pid].data());
            input_views.push_back(views[pid]);
            TensorNode in_node = graph.getNode(pid);
            in_node.strides = views[pid].strides;
            input_nodes.push_back(in_node);
        }

        TensorView out_view_contig = makeView(node);
        TensorView out_view_non_contig = out_view_contig;
        if (options.force_non_contiguous)
        {
            for (auto &s : out_view_non_contig.strides)
                s *= 2;
        }

        TensorNode out_node_nc = node;
        bool ignore_in_ms = (node.opType != OpType::COPY_TO);
        auto refs_nc = KernelRegistry::get().findMatchingKernels(
            node.opType, node.opName, input_nodes, out_node_nc, true, MemSpace{1, HandleType::CPP}, {},
            {Engine{0, EngineType::CPU}}, false, ignore_in_ms, false, true);

        TensorView chosen_out_view;
        KernelId chosen_kernel_uid = KernelId{0};
        if (options.force_non_contiguous && !refs_nc.empty())
        {
            chosen_out_view = out_view_non_contig;
            chosen_kernel_uid = refs_nc.front();
        }
        else
        {
            TensorNode out_node_c = node;
            auto refs_c = KernelRegistry::get().findMatchingKernels(
                node.opType, node.opName, input_nodes, out_node_c, true, MemSpace{1, HandleType::CPP}, {},
                {Engine{0, EngineType::CPU}}, false, ignore_in_ms, false, true);
            if (refs_c.empty())
            {
                Error::throw_err("No reference kernel found for node " + std::to_string(node_id.value) +
                                 " op=" + toString(node.opType) +
                                 (node.opType == OpType::FUSED ? " (" + node.opName + ")" : ""));
            }
            chosen_out_view = out_view_contig;
            chosen_kernel_uid = refs_c.front();
        }

        const KernelEntry &kernel = KernelRegistry::get().getKernel(chosen_kernel_uid);

        if (kernel.is_view)
        {
            TensorView dummy_out_view(node, 0);
            kernel.inferView(input_nodes, dummy_out_view, graph);
            LogicalId parent_id = node.child_ids[0];
            results[node_id] = results[parent_id];
            chosen_out_view.strides = dummy_out_view.strides;
            chosen_out_view.offset = dummy_out_view.offset;
            views[node_id] = chosen_out_view;

            TensorView contig_view = dummy_out_view;
            contig_view.strides = calcContiguousStrides(dummy_out_view.getShape());
            std::vector<uint8_t> contig_data(countElements(contig_view) * elem_size);
            const uint8_t *src_data = results[parent_id].data() + chosen_out_view.offset;
            for (uint64_t i = 0; i < countElements(contig_view); ++i)
            {
                uint64_t src_idx = getStridedIndex(i, chosen_out_view.getShape(), chosen_out_view.strides);
                std::memcpy(contig_data.data() + i * elem_size, src_data + src_idx * elem_size, elem_size);
            }
            store.write(node_id, contig_view, contig_data.data(), contig_data.size());
            continue;
        }

        views[node_id] = chosen_out_view;
        uint64_t buf_elements = getRequiredBufferSize(chosen_out_view);
        results[node_id].resize(buf_elements * elem_size, 0);
        std::vector<void *> output_ptrs = {results[node_id].data()};
        std::vector<TensorView> output_views = {chosen_out_view};

        if (kernel.run)
        {
            kernel.run(KernelContext(input_ptrs, output_ptrs, input_views, output_views));
        }

        store.write(node_id, chosen_out_view, results[node_id].data(), results[node_id].size());
    }

    if (!options.only_clean_nodes && root_ids.size() == 1)
    {
        LogicalId root_id = root_ids[0];
        if (results.find(root_id) != results.end())
        {
            return extractFloatTensor(root_id, graph, views[root_id], results[root_id].data());
        }
    }

    return {};
}
