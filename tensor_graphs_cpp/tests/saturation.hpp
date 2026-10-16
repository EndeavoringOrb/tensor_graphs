#pragma once

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "core/cost_model.hpp"
#include "core/egraph.hpp"
#include "core/graph.hpp"
#include "core/kernels.hpp"
#include "core/plan/bucket_regions.hpp"
#include "core/plan/planner.hpp"
#include "core/settings.hpp"
#include "core/shape_propagator.hpp"

inline Region makeSaturationRegion(std::initializer_list<Dim> dims)
{
    Region r;
    r.region.assign(dims.begin(), dims.end());
    return r;
}

inline void dumpSaturationGraph(const EGraph &egraph, const Graph &graph, const std::string &path)
{
    std::ofstream out(path);
    if (!out)
        Error::throw_err("Cannot open saturation dump: " + path);
    std::map<uint32_t, std::vector<uint32_t>> anchors;
    for (const auto &[id, node] : graph.nodes)
    {
        const auto owner = egraph.findEClassByLogicalId(id);
        if (owner != EClassId{})
            anchors[egraph.findConst(owner).value].push_back(id.value);
    }
    for (const auto &cls : egraph.classes)
    {
        if (egraph.findConst(cls.id) != cls.id || cls.enodes.empty())
            continue;
        auto ids = anchors[cls.id.value];
        std::sort(ids.begin(), ids.end());
        std::ostringstream label;
        label << static_cast<int>(cls.dtype) << '|' << toString(cls.mem_space) << '|'
              << toString(cls.shape) << '|' << toString(cls.strides) << '|' << cls.is_clean << '|'
              << (cls.id.value < egraph.ufSize.size() ? egraph.ufSize[cls.id.value] : 1) << '|';
        for (auto id : ids)
            label << id << ',';
        const auto staged = egraph.constantStaging.find(cls.id);
        if (staged != egraph.constantStaging.end())
            for (auto byte : *staged->second)
                label << ':' << int(byte);
        if (cls.base_eclass_id != BaseEClassId{})
            out << "B " << cls.id.value << ' ' << egraph.resolveBaseId(cls.base_eclass_id).value << '\n';
        out << "C " << cls.id.value << ' ' << std::quoted(label.str()) << '\n';
    }
    std::set<std::string> expressions;
    for (const auto &node : egraph.enodes)
    {
        std::ostringstream row, label;
        label << node.getKernelId().value << '|' << int(node.getOpType()) << '|' << node.getOpName()
              << '|' << toString(node.getShape()) << '|' << toString(node.getStrides()) << '|'
              << int(node.getDType()) << '|' << toString(node.getMemSpace()) << '|' << node.getContentHash();
        for (auto engine : node.getEngines())
            label << '|' << toString(engine);
        const auto index = &node - egraph.enodes.data();
        if (index < egraph.nodeToEClass.size())
            row << egraph.findConst(egraph.nodeToEClass[index]).value << ' ' << std::quoted(label.str());
        else
            continue;
        for (auto child : node.getChildren())
            row << ' ' << egraph.findConst(child).value;
        expressions.insert(row.str());
    }
    for (const auto &expression : expressions)
        out << "N " << expression << '\n';
}

struct SaturationBenchmarkResult
{
    double full_ms = 0.0;
    double partial_ms = 0.0;
    double total_ms = 0.0;
    size_t full_enodes = 0;
    size_t full_classes = 0;
};

inline SaturationBenchmarkResult runSaturationBenchmark(uint32_t seq_len = 7, const std::string &dump_prefix = "")
{
    Graph graph;
    auto input = graph.input({seq_len, seq_len}, DType::FLOAT32);

    // Causal mask using TRIU and NEG_INF fill (matches InfinityDomination & matchTriuMask)
    LogicalId fill_node = graph.fill(TGConstants::NEG_INF, {seq_len, seq_len});
    LogicalId k_node = graph.constant(std::vector<int32_t>{1});
    LogicalId mask = graph.triu(fill_node, k_node);

    auto scores = graph.add(input, mask);
    auto weights = graph.pow(graph.fill(std::exp(1.f), {seq_len, seq_len}), scores);
    auto sum_w = graph.sum(weights, 1);
    auto rep_sum = graph.repeat(sum_w, seq_len, 1);
    auto attn = graph.div(weights, rep_sum);
    auto root = graph.dot(attn, input);

    ShapePropagator propagator;
    propagator.inferShapeRecursive(root, graph);

    Settings settings;
    settings.records_path = "";
    CostModel costs(false, "");
    Planner planner(costs, settings);

    Bucket full;
    full.inputDirtyRegions[input] = {makeSaturationRegion({{0, seq_len}, {0, seq_len}})};
    full.outputNeededRegion = {makeSaturationRegion({{0, seq_len}, {0, seq_len}})};

    auto t0 = std::chrono::steady_clock::now();
    auto state = planner.saturateBucket(root, graph, full, true);
    auto t1 = std::chrono::steady_clock::now();

    if (!dump_prefix.empty())
        dumpSaturationGraph(state, graph, dump_prefix + "-full.txt");

    const auto region_facts = computeRegionFacts(graph, topologicalSort({root}, graph), nullptr);
    planner.fullSaturationState_ = &state;
    planner.regionFacts_ = &region_facts;

    double partial_ms = 0.0;

    for (uint32_t bucket_seq_len = 1; bucket_seq_len <= seq_len; ++bucket_seq_len)
    {
        Bucket prefill;
        prefill.inputDirtyRegions[input] = {
            makeSaturationRegion({{0, bucket_seq_len}, {0, seq_len}})};
        prefill.outputNeededRegion = {
            makeSaturationRegion({{0, bucket_seq_len}, {0, seq_len}})};
        auto prefill_t0 = std::chrono::steady_clock::now();
        auto prefill_result = planner.saturateBucket(root, graph, prefill, true, nullptr, &state);
        auto prefill_t1 = std::chrono::steady_clock::now();
        partial_ms += std::chrono::duration<double, std::milli>(prefill_t1 - prefill_t0).count();

        if (!dump_prefix.empty())
            dumpSaturationGraph(prefill_result, graph,
                                dump_prefix + "-prefill" + std::to_string(bucket_seq_len) + ".txt");

        const uint32_t row = bucket_seq_len - 1;
        Bucket decode;
        decode.inputDirtyRegions[input] = {
            makeSaturationRegion({{row, row + 1}, {0, seq_len}})};
        decode.outputNeededRegion = {
            makeSaturationRegion({{row, row + 1}, {0, seq_len}})};
        auto decode_t0 = std::chrono::steady_clock::now();
        auto decode_result = planner.saturateBucket(root, graph, decode, true, nullptr, &state);
        auto decode_t1 = std::chrono::steady_clock::now();
        partial_ms += std::chrono::duration<double, std::milli>(decode_t1 - decode_t0).count();

        if (!dump_prefix.empty())
            dumpSaturationGraph(decode_result, graph,
                                dump_prefix + "-decode" + std::to_string(bucket_seq_len) + ".txt");
    }

    double full_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    SaturationBenchmarkResult res;
    res.full_ms = full_ms;
    res.partial_ms = partial_ms;
    res.total_ms = full_ms + partial_ms;
    res.full_enodes = state.getENodes().size();
    res.full_classes = state.getClasses().size();
    return res;
}
