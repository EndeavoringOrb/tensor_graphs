#pragma once

#include <chrono>
#include <iostream>
#include <map>
#include <set>

#include "core/common/bench_utils.hpp"
#include "core/plan/planner.hpp"
#include "tests/fused.hpp"

namespace KernelBench
{
// Variable identity and ordering are part of the ABI. Shapes/strides are not
// part of a reference expression's identity; they belong to its configurations.
inline std::string describeReference(const Graph &graph, LogicalId root_id,
                                     const std::vector<LogicalId> &input_ids)
{
    for (size_t i = 0; i < input_ids.size(); ++i)
        if (root_id == input_ids[i])
            return "$" + std::to_string(i);
    const auto &node = graph.getNode(root_id);
    std::string expression = toString(node.opType);
    if (node.opType == OpType::FUSED)
        expression += ":" + node.opName;
    if (graph.constantStaging.count(root_id))
        expression += ":constant:" + node.contentHash;
    expression += "(";
    for (auto child_id : node.child_ids)
        expression += describeReference(graph, child_id, input_ids) + ",";
    return expression + ")";
}

inline std::string referenceExpression(const KernelEntry &kernel)
{
    Graph graph;
    std::vector<LogicalId> input_ids;
    for (size_t i = 0; i < kernel.min_num_inputs; ++i)
        input_ids.push_back(graph.input(kernel.dummyShapes.at(i), kernel.dtypes.at(i)));
    auto root_id = kernel.refFactory ? kernel.refFactory(input_ids, graph)
                                    : graph.allocateNode(kernel.opType, "", DType::ANY, input_ids).id;
    return describeReference(graph, root_id, input_ids);
}

inline bool bindRecord(const KernelEntry &kernel, Record &record, std::vector<std::string> *reasons = nullptr)
{
    if (record.inputShapes.size() != record.inputStrides.size() ||
        record.inputShapes.size() != record.inputDTypes.size())
        Error::throw_err("KernelBench configuration has inconsistent input metadata");
    if (record.outputShape.size() != record.outputStrides.size())
        Error::throw_err("KernelBench output shape/stride ranks differ");
    std::vector<TensorNode> inputs(record.inputShapes.size());
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        if (record.inputShapes[i].size() != record.inputStrides[i].size())
            Error::throw_err("KernelBench input shape/stride ranks differ");
        inputs[i].opType = OpType::INPUT;
        inputs[i].setShape(record.inputShapes[i]);
        inputs[i].strides = record.inputStrides[i];
        inputs[i].dtype = record.inputDTypes[i];
    }
    TensorNode output;
    output.setShape(record.outputShape);
    output.strides = record.outputStrides;
    output.dtype = record.outputDType;
    // Diagnose the same built-in checks used by KernelEntry::matches, outside execution.
    auto reject = [&](const std::string &reason) {
        if (reasons)
            reasons->push_back(reason);
        return false;
    };
    if (inputs.size() < kernel.min_num_inputs || inputs.size() > kernel.max_num_inputs)
        return reject("input_count");
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        if (!kernel.requiresContiguous.empty() &&
            kernel.requiresContiguous[std::min(i, kernel.requiresContiguous.size() - 1)] && !isContiguous(inputs[i]))
            return reject("contiguity");
    }
    HardwareBinding binding;
    if (!TopologyMapper::resolve(kernel.output_mem_space, kernel.input_mem_spaces, kernel.engines, kernel.is_view,
                                 inputs.size(), {}, {}, {}, true, true, true, binding))
        return reject("memory_space_or_engine");
    for (size_t i = 0; i < inputs.size(); ++i)
        if (!kernel.dtypes.empty() && inputs[i].dtype != kernel.dtypes[std::min(i, kernel.dtypes.size() - 1)])
            return reject("dtype");
    if (kernel.match && !kernel.match(inputs, output))
        return reject("custom_match_rejected");
    record.kernelId = kernel.uid;
    record.buildContextId = BUILD_CONTEXT_ID;
    record.hwTag = HW_TAG;
    record.output_mem_space = binding.output_mem_space;
    record.input_mem_spaces = binding.input_mem_spaces;
    record.engines = binding.engines;
    return true;
}

// Reconstruct each compiled bucket as logical reference operations. External
// cached values become inputs. Expanding fused instructions preserves the
// producer/consumer connections, allowing new fusions across instructions.
struct DiscoveryStats
{
    size_t buckets = 0;
    size_t saturation_passes = 0;
    size_t fragment_roots = 0;
};

inline std::vector<Record> discoverCalls(const std::string &cache_path, DiscoveryStats *stats = nullptr)
{
    auto cache = loadCacheFile(cache_path, true);
    if (!cache.isValid)
        Error::throw_err(cache.invalidReason);
    if (cache.compiledGraphs.empty())
        Error::throw_err("Goal cache contains no compiled buckets");
    std::cerr << "[discovery] Loaded " << cache.compiledGraphs.size() << " compiled buckets" << std::endl;
    CostModel costs(false, "");
    std::map<std::string, Record> calls;
    std::mutex calls_mutex;
    costs.callObserver = [&](const Record &record) {
        std::lock_guard<std::mutex> lock(calls_mutex);
        calls.emplace(serializeToString(record), record);
    };
    std::unordered_map<std::string, KernelId> reference_ids;
    for (const auto &[uid, kernel] : KernelRegistry::get().getAllKernels())
        if (kernel.refFactory)
            reference_ids.emplace(kernel.getName(), uid);
    costs.referenceObserver = [&](const std::string &name, const std::vector<TensorNode> &inputs,
                                  const TensorNode &output, const std::vector<std::vector<uint8_t>> &constants) {
        Record record;
        record.kernelId = reference_ids.at(name);
        record.buildContextId = BUILD_CONTEXT_ID;
        for (const auto &input : inputs)
        {
            record.inputShapes.push_back(input.getShape());
            record.inputStrides.push_back(input.strides);
            record.inputDTypes.push_back(input.dtype);
        }
        record.outputShape = output.getShape();
        record.outputStrides = output.strides;
        record.outputDType = output.dtype;
        record.inputConstants = constants;
        std::lock_guard<std::mutex> lock(calls_mutex);
        calls.emplace(serializeToString(record), record);
    };
    Settings settings;
    settings.log_cost_calls = false;
    settings.saturate_only = true;
    size_t bucket_index = 0;
    for (const auto &compiled : cache.compiledGraphs)
    {
        ++bucket_index;
        std::cerr << "[discovery] Bucket " << bucket_index << "/" << cache.compiledGraphs.size()
                  << ": expanding " << compiled.instructions.size() << " instructions" << std::endl;
        Graph graph;
        std::unordered_map<EClassId, LogicalId> logical_ids;
        std::unordered_set<EClassId> consumed;
        auto getInput = [&](EClassId physical_id) {
            if (logical_ids.count(physical_id))
                return logical_ids.at(physical_id);
            const auto &view = compiled.nodeViews.at(physical_id);
            auto id = graph.input(view.getShape(), view.dtype, view.strides);
            std::shared_ptr<std::vector<uint8_t>> data;
            if (compiled.constantStaging.count(physical_id))
                data = compiled.constantStaging.at(physical_id);
            else if (compiled.has_logical_id(physical_id))
            {
                auto old_id = compiled.get_logical_id(physical_id);
                if (compiled.constantStaging.count(EClassId{old_id.value}))
                    data = compiled.constantStaging.at(EClassId{old_id.value});
                else if (cache.constants.count(old_id))
                    data = cache.constants.at(old_id);
            }
            if (data)
            {
                graph.constantStaging[id] = data;
                graph.input_data_types[id] = InputDataType::CONSTANT;
                SHA256 hash;
                hash.update(data->data(), data->size());
                graph.getNode(id).contentHash = hash.digest();
            }
            logical_ids[physical_id] = id;
            return id;
        };
        for (const auto &instruction : compiled.instructions)
        {
            if (instruction.kernel_id == KernelId{0})
            {
                getInput(instruction.eclass_id);
                continue;
            }
            const auto &kernel = KernelRegistry::get().getKernel(instruction.kernel_id);
            std::vector<LogicalId> inputs;
            for (auto child_id : instruction.children)
            {
                inputs.push_back(getInput(child_id));
                consumed.insert(child_id);
            }
            const auto &view = compiled.nodeViews.at(instruction.eclass_id);
            if (kernel.opType == OpType::FUSED && !kernel.refFactory)
                Error::throw_err("Cannot expand fused kernel without reference factory: " + kernel.getName());
            auto root_id = kernel.refFactory
                               ? kernel.refFactory(inputs, graph)
                               : graph.allocateNode(kernel.opType, "", view.dtype, inputs, view.getShape(), view.strides).id;
            auto &node = graph.getNode(root_id);
            node.setShape(view.getShape());
            node.strides = view.strides;
            node.dtype = view.dtype;
            logical_ids[instruction.eclass_id] = root_id;
        }
        // View instructions are omitted from compiled caches, leaving fragments
        // at unresolved view boundaries. Include every fragment in ONE egraph.
        std::vector<LogicalId> roots;
        for (const auto &instruction : compiled.instructions)
        {
            if (consumed.count(instruction.eclass_id) || instruction.kernel_id == KernelId{0})
                continue;
            roots.push_back(logical_ids.at(instruction.eclass_id));
        }
        if (roots.empty())
            Error::throw_err("Compiled bucket has no executable fragment roots");
        auto started_at = std::chrono::steady_clock::now();
        auto topo = topologicalSort(roots, graph);
        std::cerr << "[discovery] Bucket " << bucket_index << ": preparing " << topo.size()
                  << " logical nodes across " << roots.size() << " fragments" << std::endl;
        Planner planner(costs, settings);
        // The reconstructed graph already represents the selected bucket's
        // physical workload. No additional dirty/needed-region injection or
        // extraction is needed. Initialize all roots, then rewrite once.
        planner.initBaseEGraph(roots.front(), graph, topo, nullptr, false);
        auto &saturated = planner.baseState;
        std::cerr << "[discovery] Bucket " << bucket_index << ": saturating once" << std::endl;
        planner.saturate(saturated, {}, false);
        std::cerr << "[discovery] Bucket " << bucket_index << ": collecting kernel matches" << std::endl;
        planner.computeENodeInfos(saturated);
        std::cerr << "[discovery] Bucket " << bucket_index << ": collected " << calls.size()
                  << " unique calls so far in "
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - started_at).count()
                  << "s" << std::endl;
        if (stats)
        {
            ++stats->buckets;
            ++stats->saturation_passes;
            stats->fragment_roots += roots.size();
        }
    }
    std::vector<Record> result;
    for (const auto &[key, record] : calls)
        result.push_back(record);
    if (result.empty())
        Error::throw_err("Goal graph saturation generated no kernel calls");
    return result;
}

inline std::vector<float> benchmarkRecord(const KernelEntry &kernel, const Record &record, uint32_t iterations)
{
    if (iterations < 3)
        Error::throw_err("KernelBench requires at least three measurement iterations");
    PreparedKernel prepared;
    prepared.prepare(kernel, record);
    prepared.updateStorageContext(kernel, record, 0);
    if (!kernel.is_view)
        prepared.run(kernel);
    prepared.synchronize();
    std::vector<float> samples;
    for (uint32_t i = 0; i < iterations; ++i)
    {
        prepared.updateStorageContext(kernel, record, i + 1);
        auto start = std::chrono::steady_clock::now();
        if (!kernel.is_view)
            prepared.run(kernel);
        prepared.synchronize();
        samples.push_back(std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count());
    }
    return samples;
}
} // namespace KernelBench
