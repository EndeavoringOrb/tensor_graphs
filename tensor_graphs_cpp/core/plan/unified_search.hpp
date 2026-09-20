#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/plan/search_state.hpp"

struct ExtractionResult;

struct AgendaNode
{
    // A queued branch shares its prefix with sibling branches.  This keeps
    // the complete agenda affordable without copying the full decision path
    // into every node.
    DecisionPathRef path;
    Decision next_decision;
    float priority = TGConstants::INF;
    float lower_bound = TGConstants::INF;
    uint64_t sequence_id = 0;

    bool operator>(const AgendaNode &other) const
    {
        if (std::abs(priority - other.priority) > 1e-5f)
            return priority > other.priority;
        if (std::abs(lower_bound - other.lower_bound) > 1e-5f)
            return lower_bound > other.lower_bound;
        return sequence_id > other.sequence_id;
    }
};

class PriorityQueue
{
  public:
    bool empty() const
    {
        return queue.empty();
    }

    size_t size() const
    {
        return queue.size();
    }

    const AgendaNode &top() const
    {
        return queue.top();
    }

    void pop()
    {
        queue.pop();
    }

    void push(AgendaNode node)
    {
        // Never discard a frontier node here.  The priority is an ordering
        // heuristic; dropping a node would make the search incomplete and
        // invalidate the global-optimum guarantee.
        queue.push(std::move(node));
    }

  private:
    std::priority_queue<AgendaNode, std::vector<AgendaNode>, std::greater<AgendaNode>> queue;
};

// The phase iterators remain the authoritative implementation of each phase's
// pruning hooks.  This facade owns the unified agenda and the common incumbent /
// timeout policy, while SearchState provides the shared reversible decision API
// for callers that need arbitrary LCA jumps.
class UnifiedSearchPlanner
{
  public:
    UnifiedSearchPlanner(const EGraph &egraph, EClassId rootEClassId, const std::vector<ENodeInfo> &enodeInfos,
                         const std::unordered_map<LogicalId, EClassId> &nodeToEClass,
                         const std::unordered_set<BaseEClassId> &cachedNodes,
                         const std::unordered_map<EClassId, LogicalId> &eclassToLogical, const Settings &settings,
                         std::shared_ptr<SearchDelegate> delegate,
                         const std::unordered_set<EClassId> *cachedEClasses,
                         const std::unordered_set<EClassId> *cleanEClasses,
                         const std::unordered_map<BaseEClassId, ParallelBuffer> &preallocatedBuffers,
                         const std::unordered_map<MemSpace, uint64_t> &reducedCaps,
                         const std::unordered_map<MemSpace, uint64_t> &reservedPerMemorySpace)
        : egraph(egraph), rootEClassId(rootEClassId), enodeInfos(enodeInfos), nodeToEClass(nodeToEClass),
          cachedNodes(cachedNodes), eclassToLogical(eclassToLogical), settings(settings), delegate(std::move(delegate)),
          cachedEClasses(cachedEClasses), cleanEClasses(cleanEClasses), preallocatedBuffers(preallocatedBuffers),
          reducedCaps(reducedCaps), reservedPerMemorySpace(reservedPerMemorySpace)
    {
    }

    ExtractionResult solve(float minCompileSeconds, bool onlyDive, bool stopOnFirstValid);

  private:
    const EGraph &egraph;
    EClassId rootEClassId;
    const std::vector<ENodeInfo> &enodeInfos;
    const std::unordered_map<LogicalId, EClassId> &nodeToEClass;
    const std::unordered_set<BaseEClassId> &cachedNodes;
    const std::unordered_map<EClassId, LogicalId> &eclassToLogical;
    const Settings &settings;
    std::shared_ptr<SearchDelegate> delegate;
    const std::unordered_set<EClassId> *cachedEClasses;
    const std::unordered_set<EClassId> *cleanEClasses;
    const std::unordered_map<BaseEClassId, ParallelBuffer> &preallocatedBuffers;
    const std::unordered_map<MemSpace, uint64_t> &reducedCaps;
    const std::unordered_map<MemSpace, uint64_t> &reservedPerMemorySpace;
};
