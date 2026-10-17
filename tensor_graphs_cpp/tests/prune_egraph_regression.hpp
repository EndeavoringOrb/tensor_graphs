#pragma once

#include <iostream>
#include <string>
#include <vector>

#include "core/plan/planner.hpp"

namespace PruneEGraphRegression
{
inline void checkCascade(Planner &planner, bool shared_state)
{
    EGraph source;
    const MemSpace ram{1, HandleType::CPP};
    auto makeClass = [&]() { return source.addEClass({1}, {1}, DType::FLOAT32, ram); };
    auto addNode = [&](EClassId owner, const std::string &name, std::vector<EClassId> children) {
        const ENodeId node_id{static_cast<uint32_t>(source.getNumENodes())};
        source.addENode(owner, ENode(KernelId{0}, OpType::FUSED, name, std::move(children),
                                    {1}, {1}, DType::FLOAT32, ram, {}));
        return node_id;
    };
    const EClassId healthy = makeClass();
    const EClassId dead_leaf = makeClass();
    const EClassId first = makeClass();
    const EClassId second = makeClass();
    const EClassId mixed = makeClass();
    const EClassId root = makeClass();
    const EClassId initially_empty = makeClass();
    addNode(healthy, "healthy", {});
    const ENodeId rejected = addNode(dead_leaf, "rejected", {});
    // Repeated children must invalidate an expression only once.
    addNode(first, "first", {dead_leaf, dead_leaf});
    addNode(second, "second", {first});
    addNode(mixed, "dead alternative", {second});
    addNode(mixed, "empty alternative", {initially_empty});
    const ENodeId surviving = addNode(mixed, "live alternative", {healthy});
    addNode(root, "root", {mixed});

    if (shared_state)
    {
        source.freezeClasses();
        source.freezeENodes();
        source.freezeHashcons();
    }
    EGraph graph = source;
    std::vector<ENodeInfo> costs(graph.getNumENodes());
    for (auto &info : costs)
        info.cost = 1.0f;
    costs[rejected.value].cost = TGConstants::INF;

    planner.pruneEGraph(graph, costs);
    auto checkPruned = [&]() {
        for (EClassId id : {dead_leaf, first, second, initially_empty})
            if (!graph.getEClass(id).enodes.empty())
                Error::throw_err("[PruneEGraphRegression] dead class retained expressions");
        if (graph.getEClass(mixed).enodes != std::vector<ENodeId>{surviving})
            Error::throw_err("[PruneEGraphRegression] live alternative was lost or dead alternatives survived");
        if (graph.getEClass(healthy).enodes.size() != 1 || graph.getEClass(root).enodes.size() != 1)
            Error::throw_err("[PruneEGraphRegression] cascade removed a live branch");
    };
    checkPruned();
    planner.pruneEGraph(graph, costs);
    checkPruned();
    if (source.getEClass(dead_leaf).enodes.size() != 1 || source.getEClass(mixed).enodes.size() != 3)
        Error::throw_err("[PruneEGraphRegression] pruning mutated the source snapshot");

    graph.rebuild(true);
    if (graph.getNumENodes() != 3)
        Error::throw_err("[PruneEGraphRegression] compaction retained dead expressions");
    for (const auto &node : graph.getENodes())
        for (EClassId child : node.getChildren())
            if (graph.getEClass(graph.findConst(child)).enodes.empty())
                Error::throw_err("[PruneEGraphRegression] surviving expression references an empty class");
}

inline void run()
{
    CostModel cost_model(false, "");
    Settings settings;
    Planner planner(cost_model, settings);
    checkCascade(planner, false);
    checkCascade(planner, true);
    std::cout << "pruneEGraph regression tests passed" << std::endl;
}
} // namespace PruneEGraphRegression
