// Build against baseline/candidate headers with identical generated kernels.
// Define TG_TEST_COMPACT for candidates supporting class compaction. Compare
// the full and seeded dumps together with tests/compare_saturation_egraphs.py.
#include "generated/build_context.gen.hpp"
#include "generated/kernels_all.gen.hpp"
#include "tests/common.hpp"
#include <fstream>
#include <iomanip>
#include <set>

void dumpGraph(const EGraph &egraph, const Graph &graph, const std::string &path)
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
              << egraph.ufSize[cls.id.value] << '|';
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
        row << egraph.findConst(egraph.nodeToEClass[index]).value << ' ' << std::quoted(label.str());
        for (auto child : node.getChildren())
            row << ' ' << egraph.findConst(child).value;
        expressions.insert(row.str());
    }
    for (const auto &expression : expressions)
        out << "N " << expression << '\n';
}

int main(int argc, char **argv)
{
    if (argc < 2)
        Error::throw_err("usage: saturation_equivalence OUTPUT_PREFIX [seeded]");
    Graph graph;
    auto input = graph.input({4, 4}, DType::FLOAT32);
    std::vector<float> mask(16, TGConstants::NEG_INF);
    for (uint32_t row = 0; row < 4; ++row)
        for (uint32_t col = 0; col <= row; ++col)
            mask[row * 4 + col] = 0;
    auto scores = graph.add(input, graph.constant({4, 4}, mask.data(), DType::FLOAT32));
    auto weights = graph.pow(graph.fill(std::exp(1.f), {4, 4}), scores);
    auto root = graph.dot(graph.div(weights, graph.repeat(graph.sum(weights, 1), 4, 1)), input);
    ShapePropagator propagator;
    propagator.inferShapeRecursive(root, graph);
    Settings settings;
    settings.records_path = "";
    CostModel costs(false, "");
    Planner planner(costs, settings);
    Bucket full;
    full.inputDirtyRegions[input] = {makeRegion({{0, 4}, {0, 4}})};
    full.outputNeededRegion = {makeRegion({{0, 4}, {0, 4}})};
    auto state = planner.saturateBucket(root, graph, full, true);
    dumpGraph(state, graph, std::string(argv[1]) + "-full.txt");
#ifdef TG_TEST_COMPACT
    const auto root_map = state.compactClasses();
    for (auto &rule : planner.fullSaturationRules_)
        rule->remapEClassIds(root_map);
#endif
    const auto region_facts = computeRegionFacts(graph, topologicalSort({root}, graph), nullptr);
    if (argc > 2)
    {
        planner.fullSaturationState_ = &state;
        planner.regionFacts_ = &region_facts;
    }
    for (uint32_t row : {0u, 2u})
    {
        Bucket partial;
        partial.inputDirtyRegions[input] = {makeRegion({{row, row + 1}, {0, 4}})};
        partial.outputNeededRegion = {makeRegion({{row, row + 1}, {0, 4}})};
        auto result = planner.saturateBucket(root, graph, partial, true, nullptr, &state);
        dumpGraph(result, graph, std::string(argv[1]) + "-partial" + std::to_string(row) + ".txt");
    }
}
