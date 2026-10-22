// tensor_graphs_cpp/core/plan/search_state.hpp
#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include "core/common/constants.hpp"
#include "core/common/thread_pool.hpp"
#include "core/egraph.hpp"
#include "core/graph.hpp"
#include "core/logging.hpp"
#include "core/plan/domain.hpp"
#include "core/plan/enode_info.hpp"
#include "core/plan/search_volume.hpp"
#include "core/types.hpp"

struct CacheCandidate
{
    BaseEClassId base_eclass_id;
    uint64_t size_bytes = 0;
    DType dtype = DType::FLOAT32;
    MemSpace mem_space;
    uint32_t num_users = 0;
};

namespace plan
{

using ::CacheCandidate;
using ::ENodeInfo;

// Search treats reduced bucket e-graphs as immutable. Buckets that share a
// reduced representative keep a shared owner instead of copying the large
// graph into every search slot.
class BucketEGraphSet
{
    std::vector<std::shared_ptr<EGraph>> graphs_;

  public:
    size_t size() const { return graphs_.size(); }
    EGraph &operator[](size_t index) { return *graphs_.at(index); }
    const EGraph &operator[](size_t index) const { return *graphs_.at(index); }
    EGraph &at(size_t index) { return *graphs_.at(index); }
    const EGraph &at(size_t index) const { return *graphs_.at(index); }
    EGraph &back() { return *graphs_.back(); }
    const EGraph &back() const { return *graphs_.back(); }

    void emplace_back() { graphs_.push_back(std::make_shared<EGraph>()); }
    void push_back(EGraph &&graph) { graphs_.push_back(std::make_shared<EGraph>(std::move(graph))); }
    void push_back(const EGraph &graph) { graphs_.push_back(std::make_shared<EGraph>(graph)); }
    void pushShared(std::shared_ptr<EGraph> graph)
    {
        if (!graph)
            Error::throw_err("[BucketEGraphSet] cannot add a null e-graph owner");
        graphs_.push_back(std::move(graph));
    }

    BucketEGraphSet &operator=(std::initializer_list<EGraph> graphs)
    {
        graphs_.clear();
        graphs_.reserve(graphs.size());
        for (const EGraph &graph : graphs)
            push_back(graph);
        return *this;
    }
};

using VarId = uint32_t;
constexpr VarId kInvalidVarId = UINT32_MAX;

// Follow every surviving alternative, independent of current search decisions.
// Only classes outside this closure can be omitted permanently from the search.
inline std::vector<EClassId> collectReachableCids(const EGraph &egraph, EClassId root_id)
{
    root_id = egraph.findConst(root_id);
    std::unordered_set<EClassId> reached{root_id};
    std::vector<EClassId> reachable{root_id};
    for (size_t head = 0; head < reachable.size(); ++head)
    {
        for (ENodeId en_id : egraph.getEClass(reachable[head]).enodes)
        {
            for (EClassId child : egraph.getENode(en_id).getChildren())
            {
                child = egraph.findConst(child);
                if (reached.insert(child).second)
                    reachable.push_back(child);
            }
        }
    }
    std::sort(reachable.begin(), reachable.end());
    return reachable;
}

enum class VarType : uint8_t
{
    CACHED,
    SELECTED,
    START,
    OFFSET
};

struct VarInfo
{
    VarId id = kInvalidVarId;
    VarType type = VarType::SELECTED;
    std::string name;
    uint32_t bucket_idx = 0;
    EClassId eclass_id;
    VarId selection_var = kInvalidVarId;
    BaseEClassId base_eclass_id;
    MemSpace mem_space;
    uint64_t size_bytes = 0;
    uint32_t size_pages = 0;
};

// A directed Even-Shiloach tree for one bucket. Each enode contributes separate
// edges, so another possible enode can still support the same child. Levels only
// increase between backtracks; n is infinity, including for disconnected cycles.
// Queries and non-tree edge deletions are O(1); repairs take O(mn) total work
// along a deletion-only path. Backtracking restores the exact predecessor scans.
struct SelectionReachability
{
    struct Edge
    {
        uint32_t from;
        uint32_t to;
        uint32_t en_idx;
        bool active;
    };

    struct Node
    {
        VarId var_id;
        Domain selection;
        uint32_t level;
        uint32_t next_incoming = 0;
        bool queued = false;
        std::vector<uint32_t> incoming;
        std::vector<std::vector<uint32_t>> enode_edges;
        uint32_t indegree = 0;
    };

    struct UndoEntry
    {
        enum class Kind
        {
            SELECTION,
            EDGE,
            NODE
        };
        Kind kind;
        uint32_t index;
        Domain selection;
        uint32_t level = 0;
        uint32_t next_incoming = 0;
    };

    bool initialized = false;
    bool is_dag = false;
    uint32_t root = 0;
    std::unordered_map<VarId, uint32_t> node_indices;
    std::vector<Node> nodes;
    std::vector<Edge> edges;
    std::vector<UndoEntry> undo;
    std::vector<uint32_t> node_undo_epoch;
    uint32_t current_undo_epoch = 0;
    // Reused FIFO storage for update(). queued ensures at most one pending
    // entry per node, so this ring only needs one slot per node.
    std::vector<uint32_t> queue_buffer;
    std::vector<uint32_t> dag_queue;

    void initialize(const EGraph &egraph, EClassId root_id,
                    const std::unordered_map<EClassId, VarId> &selected_vars,
                    const std::vector<Domain> &domains, std::vector<VarId> &unreachable,
                    bool is_dag_mode = false)
    {
        is_dag = is_dag_mode;
        if (is_dag)
        {
            initializeDag(egraph, root_id, selected_vars, domains, unreachable);
            return;
        }
        const uint32_t infinity = static_cast<uint32_t>(selected_vars.size());
        for (const auto &[cid, var_id] : selected_vars)
        {
            node_indices.emplace(var_id, static_cast<uint32_t>(nodes.size()));
            nodes.push_back(Node{var_id, domains[var_id], infinity});
        }
        root = node_indices.at(selected_vars.at(root_id));
        for (const auto &[cid, var_id] : selected_vars)
        {
            const uint32_t from = node_indices.at(var_id);
            const EClass &cls = egraph.getEClass(cid);
            nodes[from].enode_edges.resize(cls.enodes.size());
            for (uint32_t en_idx = 0; en_idx < cls.enodes.size(); ++en_idx)
            {
                for (EClassId child : egraph.getENode(cls.enodes[en_idx]).getChildren())
                {
                    auto child_it = selected_vars.find(egraph.findConst(child));
                    if (child_it == selected_vars.end())
                        continue;
                    const uint32_t to = node_indices.at(child_it->second);
                    const uint32_t edge_id = static_cast<uint32_t>(edges.size());
                    edges.push_back(Edge{from, to, en_idx, domains[var_id].contains(en_idx + 1)});
                    nodes[from].enode_edges[en_idx].push_back(edge_id);
                    nodes[to].incoming.push_back(edge_id);
                }
            }
        }

        nodes[root].level = 0;
        std::vector<uint32_t> frontier = {root};
        for (size_t head = 0; head < frontier.size(); ++head)
        {
            const uint32_t from = frontier[head];
            for (const auto &enode_edges : nodes[from].enode_edges)
            {
                for (uint32_t edge_id : enode_edges)
                {
                    const Edge &edge = edges[edge_id];
                    if (edge.active && nodes[edge.to].level == infinity)
                    {
                        nodes[edge.to].level = nodes[from].level + 1;
                        frontier.push_back(edge.to);
                    }
                }
            }
        }
        for (uint32_t idx = 0; idx < nodes.size(); ++idx)
        {
            Node &node = nodes[idx];
            if (node.level == infinity)
            {
                unreachable.push_back(node.var_id);
            }
            else if (idx != root)
            {
                while (!supportsLevel(node.incoming[node.next_incoming], node.level))
                    ++node.next_incoming;
            }
        }
        queue_buffer.resize(nodes.size());
        node_undo_epoch.assign(nodes.size(), 0);
        initialized = true;
    }

    void initializeDag(const EGraph &egraph, EClassId root_id,
                       const std::unordered_map<EClassId, VarId> &selected_vars,
                       const std::vector<Domain> &domains, std::vector<VarId> &unreachable)
    {
        for (const auto &[cid, var_id] : selected_vars)
        {
            node_indices.emplace(var_id, static_cast<uint32_t>(nodes.size()));
            Node node;
            node.var_id = var_id;
            node.selection = domains[var_id];
            node.level = 0;
            node.indegree = 0;
            nodes.push_back(std::move(node));
        }
        root = node_indices.at(selected_vars.at(root_id));
        for (const auto &[cid, var_id] : selected_vars)
        {
            const uint32_t from = node_indices.at(var_id);
            const EClass &cls = egraph.getEClass(cid);
            nodes[from].enode_edges.resize(cls.enodes.size());
            for (uint32_t en_idx = 0; en_idx < cls.enodes.size(); ++en_idx)
            {
                for (EClassId child : egraph.getENode(cls.enodes[en_idx]).getChildren())
                {
                    auto child_it = selected_vars.find(egraph.findConst(child));
                    if (child_it == selected_vars.end())
                        continue;
                    const uint32_t to = node_indices.at(child_it->second);
                    const uint32_t edge_id = static_cast<uint32_t>(edges.size());
                    edges.push_back(Edge{from, to, en_idx, false});
                    nodes[from].enode_edges[en_idx].push_back(edge_id);
                    nodes[to].incoming.push_back(edge_id);
                }
            }
        }

        std::vector<bool> reachable(nodes.size(), false);
        reachable[root] = true;
        dag_queue.clear();
        dag_queue.push_back(root);
        for (size_t head = 0; head < dag_queue.size(); ++head)
        {
            const uint32_t from = dag_queue[head];
            const Node &node = nodes[from];
            for (uint32_t en_idx = 0; en_idx < node.enode_edges.size(); ++en_idx)
            {
                if (!node.selection.contains(static_cast<int32_t>(en_idx + 1)))
                    continue;
                for (uint32_t edge_id : node.enode_edges[en_idx])
                {
                    Edge &edge = edges[edge_id];
                    edge.active = true;
                    ++nodes[edge.to].indegree;
                    if (!reachable[edge.to])
                    {
                        reachable[edge.to] = true;
                        dag_queue.push_back(edge.to);
                    }
                }
            }
        }
        for (uint32_t idx = 0; idx < nodes.size(); ++idx)
        {
            if (!reachable[idx])
            {
                unreachable.push_back(nodes[idx].var_id);
            }
        }
        dag_queue.reserve(nodes.size());
        initialized = true;
    }

    bool supportsLevel(uint32_t edge_id, uint32_t level) const
    {
        const Edge &edge = edges[edge_id];
        return edge.active && nodes[edge.from].level + 1 == level;
    }

    bool isTreeEdge(uint32_t edge_id) const
    {
        const Edge &edge = edges[edge_id];
        const Node &node = nodes[edge.to];
        return edge.to != root && node.level < nodes.size() &&
               node.next_incoming < node.incoming.size() && node.incoming[node.next_incoming] == edge_id;
    }

    void checkForcedParent(uint32_t node_idx, const std::vector<Domain> &domains,
                           std::vector<std::pair<VarId, int32_t>> &forced_selections) const
    {
        if (node_idx == root || nodes[node_idx].indegree == 0)
            return;

        const Domain &dom = domains[nodes[node_idx].var_id];
        if (dom.contains(0))
            return;

        uint32_t active_from = UINT32_MAX;
        uint32_t active_en_idx = UINT32_MAX;
        for (uint32_t edge_id : nodes[node_idx].incoming)
        {
            if (!edges[edge_id].active)
                continue;
            if (active_from == UINT32_MAX)
            {
                active_from = edges[edge_id].from;
                active_en_idx = edges[edge_id].en_idx;
            }
            else if (edges[edge_id].from != active_from || edges[edge_id].en_idx != active_en_idx)
            {
                return;
            }
        }

        if (active_from != UINT32_MAX)
        {
            VarId p_var = nodes[active_from].var_id;
            int32_t required_val = static_cast<int32_t>(active_en_idx + 1);
            forced_selections.push_back({p_var, required_val});
        }
    }

    void update(uint32_t node_idx, const Domain &selection, std::vector<VarId> &unreachable,
                const std::vector<Domain> *domains = nullptr,
                std::vector<std::pair<VarId, int32_t>> *forced_parents = nullptr)
    {
        if (is_dag)
        {
            updateDag(node_idx, selection, unreachable, domains, forced_parents);
            return;
        }
        Node &changed = nodes[node_idx];
        ++current_undo_epoch;
        if (current_undo_epoch == 0)
        {
            std::fill(node_undo_epoch.begin(), node_undo_epoch.end(), 0);
            current_undo_epoch = 1;
        }
        undo.push_back(UndoEntry{UndoEntry::Kind::SELECTION, node_idx, changed.selection});
        size_t queue_head = 0;
        size_t queue_tail = 0;
        size_t queue_count = 0;
        auto enqueue = [&](uint32_t idx) {
            if (!nodes[idx].queued)
            {
                if (queue_count >= queue_buffer.size())
                    Error::throw_err("SelectionReachability::update: ring queue capacity exceeded (pending=" +
                                     std::to_string(queue_count) + ", nodes=" +
                                     std::to_string(nodes.size()) + ")");
                nodes[idx].queued = true;
                queue_buffer[queue_tail] = idx;
                queue_tail = (queue_tail + 1) % queue_buffer.size();
                ++queue_count;
            }
        };
        for (uint32_t en_idx = 0; en_idx < changed.enode_edges.size(); ++en_idx)
        {
            const int32_t value = static_cast<int32_t>(en_idx + 1);
            // Domain widening must happen through backtrackTo, which restores
            // edges, levels, and the last processed selection together.
            assert(!selection.contains(value) || changed.selection.contains(value));
            if (!changed.selection.contains(value) || selection.contains(value))
                continue;
            for (uint32_t edge_id : changed.enode_edges[en_idx])
            {
                undo.push_back(UndoEntry{UndoEntry::Kind::EDGE, edge_id, {}});
                edges[edge_id].active = false;
                if (isTreeEdge(edge_id))
                    enqueue(edges[edge_id].to);
            }
        }
        changed.selection = selection;

        const uint32_t infinity = static_cast<uint32_t>(nodes.size());
        while (queue_count > 0)
        {
            const uint32_t idx = queue_buffer[queue_head];
            queue_head = (queue_head + 1) % queue_buffer.size();
            --queue_count;
            Node &node = nodes[idx];
            node.queued = false;
            if (node.level == infinity)
                continue;
            const uint32_t prev_level = node.level;
            const uint32_t prev_incoming = node.next_incoming;
            while (node.next_incoming < node.incoming.size() &&
                   !supportsLevel(node.incoming[node.next_incoming], node.level))
                ++node.next_incoming;
            if (node.next_incoming == node.incoming.size())
            {
                // No support left at this level: the next supported level is
                // the smallest parent level + 1 above it (raising the level
                // one step at a time and rescanning finds the same level,
                // but costs O(levels * incoming) for an unreachable node).
                uint32_t next_level = infinity;
                for (uint32_t edge_id : node.incoming)
                {
                    const Edge &edge = edges[edge_id];
                    // A self-edge never supports a level (level + 1 != level).
                    if (!edge.active || edge.from == idx || nodes[edge.from].level >= infinity)
                        continue;
                    const uint32_t candidate = nodes[edge.from].level + 1;
                    if (candidate > node.level)
                        next_level = std::min(next_level, candidate);
                }
                node.level = next_level;
                node.next_incoming = 0;
                if (node.level < infinity)
                {
                    while (node.next_incoming < node.incoming.size() &&
                           !supportsLevel(node.incoming[node.next_incoming], node.level))
                        ++node.next_incoming;
                    if (node.next_incoming == node.incoming.size())
                        Error::throw_err("SelectionReachability::update: no support at the computed level");
                }
            }
            if (node.level == prev_level && node.next_incoming == prev_incoming)
                continue;
            if (node_undo_epoch[idx] != current_undo_epoch)
            {
                undo.push_back(UndoEntry{UndoEntry::Kind::NODE, idx, {}, prev_level, prev_incoming});
                node_undo_epoch[idx] = current_undo_epoch;
            }
            if (node.level == prev_level)
                continue;
            if (node.level == infinity)
            {
                unreachable.push_back(node.var_id);
            }
            for (const auto &enode_edges : node.enode_edges)
            {
                for (uint32_t edge_id : enode_edges)
                {
                    if (edges[edge_id].active && isTreeEdge(edge_id))
                        enqueue(edges[edge_id].to);
                }
            }
        }
    }

    void updateDag(uint32_t node_idx, const Domain &selection, std::vector<VarId> &unreachable,
                   const std::vector<Domain> *domains = nullptr,
                   std::vector<std::pair<VarId, int32_t>> *forced_parents = nullptr)
    {
        Node &changed = nodes[node_idx];
        undo.push_back(UndoEntry{UndoEntry::Kind::SELECTION, node_idx, changed.selection});
        dag_queue.clear();
        for (uint32_t en_idx = 0; en_idx < changed.enode_edges.size(); ++en_idx)
        {
            const int32_t value = static_cast<int32_t>(en_idx + 1);
            assert(!selection.contains(value) || changed.selection.contains(value));
            if (!changed.selection.contains(value) || selection.contains(value))
                continue;
            for (uint32_t edge_id : changed.enode_edges[en_idx])
            {
                if (!edges[edge_id].active)
                    continue;
                edges[edge_id].active = false;
                undo.push_back(UndoEntry{UndoEntry::Kind::EDGE, edge_id, {}});
                const uint32_t to = edges[edge_id].to;
                assert(nodes[to].indegree > 0);
                --nodes[to].indegree;
                if (to != root)
                {
                    if (nodes[to].indegree == 0)
                    {
                        unreachable.push_back(nodes[to].var_id);
                        dag_queue.push_back(to);
                    }
                    else if (forced_parents && domains)
                    {
                        checkForcedParent(to, *domains, *forced_parents);
                    }
                }
            }
        }
        changed.selection = selection;

        for (size_t head = 0; head < dag_queue.size(); ++head)
        {
            const uint32_t curr = dag_queue[head];
            for (const auto &enode_edges : nodes[curr].enode_edges)
            {
                for (uint32_t edge_id : enode_edges)
                {
                    if (!edges[edge_id].active)
                        continue;
                    edges[edge_id].active = false;
                    undo.push_back(UndoEntry{UndoEntry::Kind::EDGE, edge_id, {}});
                    const uint32_t to = edges[edge_id].to;
                    assert(nodes[to].indegree > 0);
                    --nodes[to].indegree;
                    if (to != root)
                    {
                        if (nodes[to].indegree == 0)
                        {
                            unreachable.push_back(nodes[to].var_id);
                            dag_queue.push_back(to);
                        }
                        else if (forced_parents && domains)
                        {
                            checkForcedParent(to, *domains, *forced_parents);
                        }
                    }
                }
            }
        }
    }

    void backtrackTo(size_t marker)
    {
        while (undo.size() > marker)
        {
            const UndoEntry &entry = undo.back();
            switch (entry.kind)
            {
            case UndoEntry::Kind::SELECTION:
                nodes[entry.index].selection = entry.selection;
                break;
            case UndoEntry::Kind::EDGE:
                edges[entry.index].active = true;
                if (is_dag)
                    ++nodes[edges[entry.index].to].indegree;
                break;
            case UndoEntry::Kind::NODE:
                nodes[entry.index].level = entry.level;
                nodes[entry.index].next_incoming = entry.next_incoming;
                break;
            }
            undo.pop_back();
        }
    }
};

struct DomainTrailEntry
{
    VarId var_id;
    Domain prev_domain;
};

struct ReachabilityTrailEntry
{
    uint32_t bucket_idx;
    VarId changed;
    size_t undo_marker;
    bool was_initialized;
};

using TrailEntry = std::variant<DomainTrailEntry, ReachabilityTrailEntry>;

// Point updates recompute sums in a stable order. Undoing a domain therefore
// restores the exact bound, without floating-point add/subtract drift.
struct PropagationSumTree
{
    size_t leaf_count = 1;
    std::vector<double> sums;

    void initialize(size_t count)
    {
        while (leaf_count < count)
            leaf_count *= 2;
        sums.assign(2 * leaf_count, 0.0);
    }

    void set(size_t index, double value)
    {
        index += leaf_count;
        sums[index] = value;
        while (index > 1)
        {
            index /= 2;
            sums[index] = sums[2 * index] + sums[2 * index + 1];
        }
    }

    double total() const { return sums.empty() ? 0.0 : sums[1]; }
};

struct PropagationState
{
    struct StartPrecedenceConsumer
    {
        EClassId parent_cid;
        VarId selection_var = kInvalidVarId;
        VarId start_var = kInvalidVarId;
        uint32_t total_enodes = 0;
        uint32_t dep_mask = 0;
        uint32_t view_mask = 0;
        std::vector<uint32_t> dep_indices;
        std::vector<uint32_t> view_indices;
    };
    using StartPrecedenceParent = StartPrecedenceConsumer;

    struct StartPrecedenceBucket
    {
        std::vector<std::vector<StartPrecedenceConsumer>> consumers_by_cid;
        std::vector<VarId> selected_by_cid;
        std::vector<uint32_t> visited_stamp;
        uint32_t stamp = 0;
        std::vector<EClassId> frontier;
    };

    struct WriteAfterReadClassInfo
    {
        bool is_active = false;
        bool is_view = false;
        bool is_input_or_cache = false;
        bool is_root = false;
        EClassId view_parent{UINT32_MAX};
        EClassId first_view_child{UINT32_MAX};
        EClassId next_view_sibling{UINT32_MAX};
        uint32_t view_tin = UINT32_MAX;
        uint32_t view_tout = UINT32_MAX;
        EClassId base_cid{UINT32_MAX};
        uint32_t en_idx = 0;
        ENodeId en_id{UINT32_MAX};
        int32_t start_min = -1;
        int32_t start_max = -1;
        int32_t max_reader_start_max = -1;
        std::vector<EClassId> readers;
        std::vector<EClassId> reader_dependents;
    };

    struct WriteAfterReadBucket
    {
        bool dirty = true;
        bool structure_dirty = true;
        bool spatial_dirty = true;
        bool fixed_offset_structure_dirty = true;
        bool fixed_offset_index_initialized = false;
        std::vector<WriteAfterReadClassInfo> class_info;
        std::vector<EClassId> active_cids;
        std::vector<EClassId> touched_cids;
        std::vector<EClassId> dirty_start_cids;
        std::vector<uint32_t> affected_stamp;
        uint32_t affected_epoch = 1;
        struct FixedOffsetAllocation
        {
            EClassId cid;
            MemSpace mem_space;
            uint32_t offset = 0;
            uint64_t end = 0;
            uint32_t size = 0;
            uint32_t en_idx = 0;
            ENodeId en_id;
            VarId start_var = kInvalidVarId;
            VarId offset_var = kInvalidVarId;
            bool is_view = false;
            bool is_input_or_cache = false;
            bool is_root = false;
        };
        struct SpatialIntervalNode
        {
            FixedOffsetAllocation allocation;
            uint64_t max_end = 0;
            uint64_t priority = 0;
            int32_t left = -1;
            int32_t right = -1;
        };
        struct SpatialIntervalIndex
        {
            std::vector<SpatialIntervalNode> nodes;
            std::vector<int32_t> free_nodes;
            int32_t root = -1;
        };
        std::unordered_map<MemSpace,
                           std::map<std::pair<uint32_t, uint32_t>, FixedOffsetAllocation>>
            fixed_offset_allocations;
        std::unordered_map<MemSpace, SpatialIntervalIndex> fixed_offset_spatial_index;
        std::unordered_map<VarId, std::pair<MemSpace, std::pair<uint32_t, uint32_t>>>
            fixed_offset_keys_by_var;
        std::unordered_map<MemSpace, uint32_t> max_fixed_allocation_size;
    };

    struct Alternative
    {
        VarId start_var = kInvalidVarId;
        std::vector<VarId> children;
        std::vector<std::pair<uint32_t, uint32_t>> engine_slots;
        bool is_view = false;
        OpType op_type = OpType::INPUT;
        float cost = 0.0f;
    };

    struct EngineUsage
    {
        uint32_t bucket_idx = 0;
        uint32_t slot_count = 0;
        std::unordered_set<VarId> active_starts;
        std::unordered_map<int32_t, uint32_t> fixed_starts;
        PropagationSumTree work;
    };

    struct Allocation
    {
        bool active = false;
        bool offset_fixed = false;
        int32_t start = 0;
        int32_t end = 0;
        uint32_t offset = 0;
        uint32_t size = 0;
        VarId offset_var = kInvalidVarId;
        MemSpace mem_space;
    };

    struct BucketData
    {
        std::multiset<double> engine_bounds;
        std::unordered_map<MemSpace, std::unordered_set<VarId>> allocations;
    };

    bool initialized = false;
    std::vector<VarId> owners;
    std::vector<std::vector<Alternative>> alternatives;
    std::vector<std::vector<VarId>> parents;
    std::vector<std::vector<VarId>> view_neighbors;
    std::vector<VarId> cache_vars;
    std::vector<std::vector<VarId>> cache_users;
    std::vector<std::vector<uint32_t>> cache_candidates;
    std::unordered_map<MemSpace, std::vector<uint32_t>> candidates_by_space;
    std::unordered_map<MemSpace, uint64_t> fixed_cache_bytes;
    std::unordered_set<MemSpace> cache_budget_dirty;
    std::vector<EngineUsage> engines;
    std::vector<BucketData> buckets;
    std::vector<StartPrecedenceBucket> start_precedence;
    std::vector<WriteAfterReadBucket> write_after_read;
    PropagationSumTree cost;
    std::vector<Allocation> allocations;

    // Scratch storage for stateless propagators
    std::vector<uint32_t> topo_path_visited_stamp;
    uint32_t topo_path_stamp = 0;
    std::vector<VarId> topo_path_frontier;

    std::vector<uint32_t> topo_ancestor_visited_stamp;
    uint32_t topo_ancestor_stamp = 0;
    std::vector<uint32_t> topo_affected_stamp;
    uint32_t topo_aff_stamp = 0;
    std::vector<VarId> topo_ancestor_frontier;
    std::vector<VarId> topo_affected_vars;

    std::vector<uint32_t> mem_alloc_visited_stamp;
    uint32_t mem_alloc_stamp = 0;
    std::vector<VarId> mem_alloc_frontier;
};

class SearchState
{
  private:
    // Kept incrementally so SearchEngine can detect contradictions without
    // scanning every domain after each propagator invocation.
    std::unordered_set<VarId> empty_domains;

    uint64_t domain_revision = 0;
    DomainVolumeTracker domain_volume_tracker;

    VarId last_domain_change_var = kInvalidVarId;
    Domain last_domain_change_before;
    Domain last_domain_change_after;

    // Domain changes are consumed by SearchEngine's propagator worklist.
    std::vector<VarId> dirty_domains;
    std::vector<uint32_t> var_dirty_epoch;
    uint32_t current_dirty_epoch = 1;

    // Mutable propagation data belongs to the state so copied searches retain
    // independent incremental propagation data.
    std::vector<SelectionReachability> selection_reachability;

    // Revisions let the brancher invalidate cached schedule and lifetime data
    // only when the selected graph or start times change.
    uint64_t selection_revision = 0;
    uint64_t schedule_revision = 0;
    uint64_t allocation_revision = 0;
    std::vector<std::set<std::pair<int32_t, VarId>>> fixed_offset_vars;

    void updateDomainIndexes(VarId var_id, const Domain &old_domain, const Domain &new_domain)
    {
        if (domain_volume_tracker.initialized())
            domain_volume_tracker.setDomain(var_id, new_domain);
        if (var_id >= var_infos.size())
            return;

        const VarInfo &info = var_infos[var_id];
        if (info.type != VarType::START)
            ++allocation_revision;
        if (info.type == VarType::SELECTED)
        {
            ++selection_revision;
        }
        else if (info.type == VarType::START)
        {
            ++schedule_revision;
        }
        else if (info.type == VarType::OFFSET)
        {
            if (fixed_offset_vars.size() <= info.bucket_idx)
                fixed_offset_vars.resize(info.bucket_idx + 1);
            if (old_domain.isFixed())
                fixed_offset_vars[info.bucket_idx].erase({old_domain.fixedValue(), var_id});
            if (new_domain.isFixed())
                fixed_offset_vars[info.bucket_idx].insert({new_domain.fixedValue(), var_id});
        }
        else if (info.type == VarType::CACHED)
        {
            if (old_domain.isFixed() && old_domain.fixedValue() == 1)
            {
                fixed_cache_bytes[info.mem_space] -= info.size_bytes;
                fixed_cached_vars.erase(var_id);
            }
            if (new_domain.isFixed() && new_domain.fixedValue() == 1)
            {
                fixed_cache_bytes[info.mem_space] += info.size_bytes;
                fixed_cached_vars.insert(var_id);
            }
        }
    }

    void updateEmptyDomainIndex(VarId var_id, bool was_empty, bool is_empty)
    {
        if (was_empty == is_empty)
            return;

        if (is_empty)
            empty_domains.insert(var_id);
        else
            empty_domains.erase(var_id);
    }

    void markDomainDirty(VarId var_id)
    {
        if (var_id >= var_dirty_epoch.size())
            var_dirty_epoch.resize(var_infos.size() > var_id ? var_infos.size() : var_id + 1, 0);
        if (var_dirty_epoch[var_id] != current_dirty_epoch)
        {
            var_dirty_epoch[var_id] = current_dirty_epoch;
            dirty_domains.push_back(var_id);
        }
    }

    void invalidateWriteAfterRead(VarId var_id)
    {
        if (!propagation.initialized)
            return;
        const VarInfo &info = var_infos[var_id];
        if (info.type == VarType::CACHED)
        {
            // Cache decisions change persistence in every bucket. Invalidate
            // immediately, even though full validation is batched: intervening
            // OFFSET/START checks must not read metadata from the old decision.
            for (auto &schedule : propagation.write_after_read)
            {
                schedule.dirty = true;
                schedule.structure_dirty = true;
                schedule.spatial_dirty = true;
                schedule.fixed_offset_structure_dirty = true;
                schedule.dirty_start_cids.clear();
            }
            return;
        }
        if (info.bucket_idx >= propagation.write_after_read.size())
            return;
        auto &write_after_read = propagation.write_after_read[info.bucket_idx];
        if (info.type == VarType::SELECTED)
        {
            write_after_read.dirty = true;
            write_after_read.structure_dirty = true;
            write_after_read.spatial_dirty = true;
            write_after_read.fixed_offset_structure_dirty = true;
            write_after_read.dirty_start_cids.clear();
        }
        else if (info.type == VarType::START)
        {
            write_after_read.dirty = true;
            write_after_read.dirty_start_cids.push_back(info.eclass_id);
        }
        else if (info.type == VarType::OFFSET)
        {
            write_after_read.spatial_dirty = true;
        }
    }

  public:
    std::vector<VarInfo> var_infos;
    std::vector<Domain> domains;
    std::vector<TrailEntry> trail;

    // Lookups
    std::unordered_map<BaseEClassId, VarId> cached_vars;
    std::vector<std::unordered_map<EClassId, VarId>> selected_vars;
    std::vector<std::unordered_map<EClassId, VarId>> start_vars;
    std::vector<std::unordered_map<EClassId, VarId>> offset_vars;

    // Context across all buckets
    std::vector<Bucket> buckets;
    std::vector<float> bucket_weights;
    // Incremental cost bounds maintained by the cost propagators.
    float lower_bound = 0.0f;
    std::vector<float> bucket_lower_bounds;
    std::vector<float> bucket_critical_path_lower_bounds;
    bool critical_path_lower_bound_initialized = false;
    std::vector<float> bucket_engine_work_lower_bounds;
    std::vector<std::unordered_map<Engine, float>> engine_work;
    std::vector<std::unordered_map<Engine, float>> selected_engine_work;
    bool engine_work_initialized = false;
    std::vector<float> bucket_start_run_lower_bounds;
    // Root selection-relaxation bounds (RelaxationBoundPropagator). Computed
    // over the root domains, so they hold in every node below the root.
    std::vector<float> bucket_relaxation_lower_bounds;
    BucketEGraphSet bucket_egraphs;
    std::vector<EClassId> bucket_root_ids;
    std::vector<std::vector<ENodeInfo>> bucket_enode_infos;
    std::vector<std::vector<EClassId>> reachable_cids;

    std::vector<CacheCandidate> candidates;
    std::unordered_map<MemSpace, uint64_t> mem_caps;
    std::unordered_map<MemSpace, uint32_t> page_alignments;
    std::unordered_map<MemSpace, uint32_t> preallocated_pages;
    // Set while the root primal heuristic completes a plan: every offset is
    // fixed explicitly and checked once fixed, so propagators may skip
    // forward pruning of unfixed offsets.
    bool completing_plan = false;
    // Spaces whose root offsets StaticPrivateLayoutPropagator fixed to
    // pairwise-disjoint private ranges: distinct non-alias allocations there
    // can never overlap, so memory reuse checks are vacuous for the search.
    std::unordered_set<MemSpace> private_layout_spaces;
    // Per bucket, whether every start domain was downward closed from 0 at
    // solve entry (before root propagation). Rank compression stays inside
    // such domains.
    std::vector<char> start_domains_downward_closed_at_entry;

    void recordStartDomainShapeAtEntry()
    {
        start_domains_downward_closed_at_entry.assign(buckets.size(), 1);
        for (uint32_t b = 0; b < buckets.size() && b < start_vars.size(); ++b)
            for (const auto &[cid, st_v] : start_vars[b])
            {
                const Domain &dom = domains[st_v];
                if (dom.isEmpty() || dom.getMin() != 0 || dom.size() != dom.getMax() + 1)
                    start_domains_downward_closed_at_entry[b] = 0;
            }
    }
    std::unordered_map<BaseEClassId, ParallelBuffer> preallocated_buffers;
    std::unordered_map<MemSpace, uint64_t> fixed_cache_bytes;
    std::unordered_set<VarId> fixed_cached_vars;

    bool isBaseFixedCached(BaseEClassId base_id) const
    {
        if (base_id == BaseEClassId{})
            return false;
        auto it = cached_vars.find(base_id);
        return it != cached_vars.end() && fixed_cached_vars.count(it->second);
    }

    // The graph, metadata, capacities and weights are immutable after the first
    // propagation. All derived data is owned by this state, including in copies.
    mutable PropagationState propagation;
    // The incumbent is global to this search, so backtracking must retain it.
    float best_cost = TGConstants::INF;
    // Batched bounds can be stale until all dirty buckets have been refreshed.
    bool lower_bounds_pending = false;

    void ensurePropagationState() const;
    void updatePropagationContribution(VarId var_id, bool add) const;
    int32_t selectedAlternative(VarId sel_var) const;
    bool isSelectedParent(VarId parent, VarId child) const;
    float costLowerBound() const;

    void updateLowerBoundBucket(uint32_t bucket_idx)
    {
        if (bucket_idx >= buckets.size())
            Error::throw_err("bucket_idx " + std::to_string(bucket_idx) + " out of bounds (" +
                             std::to_string(buckets.size()) + ") in updateLowerBoundBucket");
        if (bucket_lower_bounds.size() < buckets.size())
            bucket_lower_bounds.resize(buckets.size(), 0.0f);
        if (bucket_critical_path_lower_bounds.size() < buckets.size())
            bucket_critical_path_lower_bounds.resize(buckets.size(), 0.0f);
        if (bucket_engine_work_lower_bounds.size() < buckets.size())
            bucket_engine_work_lower_bounds.resize(buckets.size(), 0.0f);
        if (bucket_start_run_lower_bounds.size() < buckets.size())
            bucket_start_run_lower_bounds.resize(buckets.size(), 0.0f);
        if (bucket_relaxation_lower_bounds.size() < buckets.size())
            bucket_relaxation_lower_bounds.resize(buckets.size(), 0.0f);
        bucket_lower_bounds[bucket_idx] = std::max({bucket_critical_path_lower_bounds[bucket_idx],
                                                    bucket_engine_work_lower_bounds[bucket_idx],
                                                    bucket_start_run_lower_bounds[bucket_idx],
                                                    bucket_relaxation_lower_bounds[bucket_idx]});
        lower_bound = 0.0f;
        for (size_t b = 0; b < bucket_lower_bounds.size(); ++b)
        {
            const float weight = b < bucket_weights.size() ? bucket_weights[b] : 1.0f;
            if (weight > 0.0f)
                lower_bound += weight * bucket_lower_bounds[b];
        }
    }

    SearchState() = default;

    uint64_t getSelectionRevision() const
    {
        return selection_revision;
    }

    uint64_t getScheduleRevision() const
    {
        return schedule_revision;
    }

    uint64_t getAllocationRevision() const
    {
        return allocation_revision;
    }

    const std::set<std::pair<int32_t, VarId>> &getFixedOffsetVars(uint32_t bucket_idx) const
    {
        static const std::set<std::pair<int32_t, VarId>> empty;
        return bucket_idx < fixed_offset_vars.size() ? fixed_offset_vars[bucket_idx] : empty;
    }

    VarId addVar(VarInfo info, const Domain &initial_domain)
    {
        if (domain_volume_tracker.initialized())
            Error::throw_err("Cannot add variables after initializing search volume");
        assert(!propagation.initialized);
        VarId id = static_cast<VarId>(var_infos.size());
        info.id = id;
        var_infos.push_back(std::move(info));
        domains.push_back(initial_domain);
        updateDomainIndexes(id, Domain{}, initial_domain);
        if (initial_domain.isEmpty())
            empty_domains.insert(id);
        markDomainDirty(id);
        return id;
    }

    void initializeDomainVolume()
    {
        domain_volume_tracker.initialize(domains);
    }

    double domainLog2Volume() const
    {
        return domain_volume_tracker.log2Volume();
    }

    // Call once per bucket, after CACHE insertion and enode pruning are complete.
    // Preserve eclass/enode IDs for extraction; allocate dense VarIds only for
    // canonical classes reachable through at least one surviving alternative.
    void addBucketVariables(uint32_t b, const std::vector<EClassId> *shared_reachable = nullptr)
    {
        resizeBucketVariableMaps();
        setBucketReachable(b, shared_reachable);
        BucketVariables pending;
        buildBucketVariables(b, pending);
        commitBucketVariables(b, pending);
        indexBucketVariables(b, pending);
    }

    // Adds every bucket's variables with the ids sequential addBucketVariables
    // calls in bucket order would give. Reachable classes and variable
    // descriptions are built per bucket in parallel (buckets whose
    // shared_reachable_from[b] is an earlier bucket copy its classes);
    // variables are then added in bucket order and the per-bucket maps filled
    // in parallel.
    void addAllBucketVariables(const std::vector<uint32_t> &shared_reachable_from)
    {
        const uint32_t nb = static_cast<uint32_t>(buckets.size());
        if (shared_reachable_from.size() != nb)
            Error::throw_err("[SearchState] addAllBucketVariables needs one reachable source per bucket");
        resizeBucketVariableMaps();
        ThreadPool::get().parallel_for(nb, [&](uint32_t b) {
            if (shared_reachable_from[b] >= b)
                setBucketReachable(b, nullptr);
        });
        std::vector<BucketVariables> pending(nb);
        ThreadPool::get().parallel_for(nb, [&](uint32_t b) {
            const uint32_t from = shared_reachable_from[b];
            if (from < b)
                setBucketReachable(b, &reachable_cids[from]);
            buildBucketVariables(b, pending[b]);
        });
        for (uint32_t b = 0; b < nb; ++b)
            commitBucketVariables(b, pending[b]);
        ThreadPool::get().parallel_for(nb, [&](uint32_t b) { indexBucketVariables(b, pending[b]); });
    }

  private:
    struct BucketVariables
    {
        std::vector<VarInfo> infos; // START infos hold their selection's index in this list
        std::vector<Domain> domains;
        VarId first_id = kInvalidVarId;
    };

    void resizeBucketVariableMaps()
    {
        selected_vars.resize(buckets.size());
        start_vars.resize(buckets.size());
        offset_vars.resize(buckets.size());
        reachable_cids.resize(buckets.size());
    }

    void setBucketReachable(uint32_t b, const std::vector<EClassId> *shared_reachable)
    {
        assert(selected_vars[b].empty() && start_vars[b].empty() && offset_vars[b].empty());
        const EGraph &egraph = bucket_egraphs[b];
        const EClassId root_cid = egraph.findConst(bucket_root_ids[b]);
        bucket_root_ids[b] = root_cid;
        if (shared_reachable != nullptr)
            reachable_cids[b] = *shared_reachable;
        else
            reachable_cids[b] = collectReachableCids(egraph, root_cid);
        if (reachable_cids[b].empty() || reachable_cids[b].size() > static_cast<size_t>(INT32_MAX))
            Error::throw_err("Bucket start horizon requires between 1 and INT32_MAX reachable classes");
    }

    void buildBucketVariables(uint32_t b, BucketVariables &out) const
    {
        const EGraph &egraph = bucket_egraphs[b];
        const EClassId root_cid = bucket_root_ids[b];
        const uint32_t max_start = static_cast<uint32_t>(reachable_cids[b].size() - 1);
        out.infos.reserve(reachable_cids[b].size() * 3);
        out.domains.reserve(reachable_cids[b].size() * 3);
        auto add = [&](VarInfo info, const Domain &domain) {
            out.infos.push_back(std::move(info));
            out.domains.push_back(domain);
            return static_cast<VarId>(out.infos.size() - 1);
        };
        for (EClassId cid : reachable_cids[b])
        {
            const EClass &cls = egraph.getEClass(cid);
            uint32_t n_enodes = static_cast<uint32_t>(cls.enodes.size());
            if (n_enodes > 63)
            {
                std::stringstream error;
                error << "EClass " << cid.value << " has " << n_enodes
                      << " enodes, exceeding domain bitmask capacity (max 63).\n"
                      << toString(cls) << "\nEnode details:\n";
                for (size_t enode_idx = 0; enode_idx < cls.enodes.size(); ++enode_idx)
                {
                    const ENodeId enode_id = cls.enodes[enode_idx];
                    const ENode &enode = egraph.getENode(enode_id);
                    error << "  [" << enode_idx << "] ENode " << enode_id.value << ":\n"
                          << toString(enode) << "\n";
                }
                Error::throw_err(error.str());
            }

            // selected_<bucket_id>_<eclass_id> in {0, 1, ..., n_enodes}
            VarInfo sel_info;
            sel_info.type = VarType::SELECTED;
            sel_info.bucket_idx = b;
            sel_info.eclass_id = cid;
            sel_info.name = "selected_" + std::to_string(b) + "_" + std::to_string(cid.value);

            uint64_t mask = UINT64_MAX >> (63 - n_enodes);
            if (cid == root_cid)
            {
                mask &= ~uint64_t{1}; // Root must be selected
            }

            const VarId sel_vid = add(std::move(sel_info), Domain::makeMask(mask));

            // One dispatch start is shared by every alternative in the e-class.
            VarInfo st_info;
            st_info.type = VarType::START;
            st_info.bucket_idx = b;
            st_info.eclass_id = cid;
            st_info.selection_var = sel_vid;
            st_info.name = "start_" + std::to_string(b) + "_" + std::to_string(cid.value);

            add(std::move(st_info), Domain::makeRange(0, max_start));

            // Views may alias the preallocated region; preallocated eclasses
            // must use their physical buffer offsets.
            if (cls.mem_space.type != HandleType::STORAGE)
            {
                VarInfo off_info;
                off_info.type = VarType::OFFSET;
                off_info.bucket_idx = b;
                off_info.eclass_id = cid;
                off_info.mem_space = cls.mem_space;
                off_info.size_bytes = getStorageSpanBytes(cls.shape, cls.strides, cls.dtype);
                off_info.size_pages = bytesToPages(off_info.size_bytes, cls.mem_space);
                off_info.name = "offset_" + std::to_string(b) + "_" + std::to_string(cid.value);

                uint32_t align = getPageAlignment(cls.mem_space);
                uint64_t cap = getMemoryCap(cls.mem_space);
                uint32_t max_p = (cap > off_info.size_bytes) ? static_cast<uint32_t>((cap - off_info.size_bytes) / align) : 0;
                Domain offset_domain = Domain::makeRange(0, max_p);
                const auto prealloc_it = preallocated_buffers.find(cls.base_eclass_id);
                if (prealloc_it != preallocated_buffers.end())
                {
                    const ParallelBuffer &buffer = prealloc_it->second;
                    if (!(buffer.mem_space == cls.mem_space) || buffer.offset < 0 ||
                        buffer.offset % align != 0 || buffer.offset / align > INT32_MAX)
                    {
                        Error::throw_err("Invalid preallocated buffer offset for EClass " + std::to_string(cid.value));
                    }
                    offset_domain = Domain::makeFixed(static_cast<int32_t>(buffer.offset / align));
                }

                add(std::move(off_info), offset_domain);
            }
        }
    }


    void commitBucketVariables(uint32_t b, BucketVariables &pending)
    {
        pending.first_id = static_cast<VarId>(var_infos.size());
        LOG(INFO) << "[SearchState start horizon] bucket=" << b << " initial=[0.." << reachable_cids[b].size() - 1
                  << "] source=structurally-reachable-classes count=" << reachable_cids[b].size();
        for (size_t k = 0; k < pending.infos.size(); ++k)
        {
            VarInfo &info = pending.infos[k];
            if (info.type == VarType::START)
                info.selection_var += pending.first_id;
            if (addVar(info, pending.domains[k]) != pending.first_id + k)
                Error::throw_err("[SearchState] bucket variables were not added contiguously");
        }
    }

    void indexBucketVariables(uint32_t b, const BucketVariables &pending)
    {
        selected_vars[b].reserve(reachable_cids[b].size());
        start_vars[b].reserve(reachable_cids[b].size());
        offset_vars[b].reserve(reachable_cids[b].size());
        for (size_t k = 0; k < pending.infos.size(); ++k)
        {
            const VarInfo &info = pending.infos[k];
            const VarId id = pending.first_id + static_cast<VarId>(k);
            if (info.type == VarType::SELECTED)
                selected_vars[b][info.eclass_id] = id;
            else if (info.type == VarType::START)
                start_vars[b][info.eclass_id] = id;
            else
                offset_vars[b][info.eclass_id] = id;
        }
    }

  public:
    size_t numVars() const
    {
        return var_infos.size();
    }

    size_t getTrailMarker() const
    {
        return trail.size();
    }

#ifdef TG_PROFILE
    uint64_t start_lower_tightenings = 0;
    uint64_t start_upper_tightenings = 0;
    uint64_t selection_values_removed = 0;
#endif
    uint64_t getDomainRevision() const
    {
        return domain_revision;
    }

    VarId getLastDomainChangeVar() const
    {
        return last_domain_change_var;
    }

    const Domain &getLastDomainChangeBefore() const
    {
        return last_domain_change_before;
    }

    const Domain &getLastDomainChangeAfter() const
    {
        return last_domain_change_after;
    }

    bool hasEmptyDomain() const
    {
        return !empty_domains.empty();
    }

    VarId getEmptyDomainVar() const
    {
        return empty_domains.empty() ? kInvalidVarId : *empty_domains.begin();
    }

    void backtrackTo(size_t marker)
    {
        while (trail.size() > marker)
        {
            if (const auto *entry = std::get_if<DomainTrailEntry>(&trail.back()))
            {
                updatePropagationContribution(entry->var_id, false);
                const bool was_empty = domains[entry->var_id].isEmpty();
                updateDomainIndexes(entry->var_id, domains[entry->var_id], entry->prev_domain);
                domains[entry->var_id] = entry->prev_domain;
                invalidateWriteAfterRead(entry->var_id);
                updatePropagationContribution(entry->var_id, true);
                updateEmptyDomainIndex(entry->var_id, was_empty, entry->prev_domain.isEmpty());
                markDomainDirty(entry->var_id);
            }
            else
            {
                const auto &reachability_entry = std::get<ReachabilityTrailEntry>(trail.back());
                auto &reachability = selection_reachability[reachability_entry.bucket_idx];
                if (reachability_entry.was_initialized)
                    reachability.backtrackTo(reachability_entry.undo_marker);
                else
                    reachability = SelectionReachability{};
                markDomainDirty(reachability_entry.changed);
            }
            trail.pop_back();
        }
    }

    bool setDomain(VarId var_id, const Domain &new_domain)
    {
        if (domains[var_id] != new_domain)
        {
            last_domain_change_var = var_id;
            last_domain_change_before = domains[var_id];
            last_domain_change_after = new_domain;
            domain_revision++;
#ifdef TG_PROFILE
            if (!domains[var_id].isEmpty() && !new_domain.isEmpty())
            {
                if (var_infos[var_id].type == VarType::START)
                {
                    start_lower_tightenings += new_domain.getMin() > domains[var_id].getMin();
                    start_upper_tightenings += new_domain.getMax() < domains[var_id].getMax();
                }
                if (var_infos[var_id].type == VarType::SELECTED && new_domain.size() < domains[var_id].size())
                    selection_values_removed += domains[var_id].size() - new_domain.size();
            }
#endif
            updatePropagationContribution(var_id, false);
            const bool was_empty = domains[var_id].isEmpty();
            trail.push_back(DomainTrailEntry{var_id, domains[var_id]});
            updateDomainIndexes(var_id, domains[var_id], new_domain);
            domains[var_id] = new_domain;
            invalidateWriteAfterRead(var_id);
            updatePropagationContribution(var_id, true);
            updateEmptyDomainIndex(var_id, was_empty, new_domain.isEmpty());
            markDomainDirty(var_id);
            return true;
        }
        return false;
    }

    // Ephemeral proposals from the reachability update, consumed by the next
    // propagator. Never replayed across propagation calls or backtracking.
    std::vector<std::pair<VarId, int32_t>> pending_parent_selections;

    const SelectionReachability &selectionReachability(uint32_t bucket_idx) const
    {
        return selection_reachability.at(bucket_idx);
    }

    bool selectionReachabilityInitialized(uint32_t bucket_idx) const
    {
        return bucket_idx < selection_reachability.size() && selection_reachability[bucket_idx].initialized;
    }

    // Called only for the changed SELECTED variable. The first call builds a
    // bucket once; later calls touch removed enodes and affected tree nodes.
    void updateSelectionReachability(VarId changed, std::vector<VarId> &unreachable, bool is_dag = false,
                                     std::vector<std::pair<VarId, int32_t>> *forced_parents = nullptr)
    {
        const uint32_t bucket_idx = var_infos[changed].bucket_idx;
        if (selection_reachability.size() < buckets.size())
            selection_reachability.resize(buckets.size());
        auto &reachability = selection_reachability[bucket_idx];
        if (!reachability.initialized)
        {
            trail.push_back(ReachabilityTrailEntry{bucket_idx, changed, 0, false});
            reachability.initialize(bucket_egraphs[bucket_idx], bucket_root_ids[bucket_idx],
                                    selected_vars[bucket_idx], domains, unreachable, is_dag);
            return;
        }
        const uint32_t node_idx = reachability.node_indices.at(changed);
        if (forced_parents && !domains[changed].contains(0))
        {
            reachability.checkForcedParent(node_idx, domains, *forced_parents);
        }
        if (reachability.nodes[node_idx].selection == domains[changed])
            return;
        trail.push_back(ReachabilityTrailEntry{bucket_idx, changed, reachability.undo.size(), true});
        reachability.update(node_idx, domains[changed], unreachable, &domains, forced_parents);
    }

    // Builds every uninitialized bucket's tree at once. A bucket's tree reads
    // only its own graph and selection domains, so the trees are built in
    // parallel; trail entries are pushed in bucket order afterwards, exactly
    // as sequential first calls would. unreachable[b] receives bucket b's
    // unreachable selections; the caller applies them.
    void initializeSelectionReachability(const std::vector<VarId> &root_vars, bool is_dag,
                                         std::vector<std::vector<VarId>> &unreachable)
    {
        if (root_vars.size() != buckets.size())
            Error::throw_err("[SearchState] initializeSelectionReachability needs one root var per bucket");
        if (selection_reachability.size() < buckets.size())
            selection_reachability.resize(buckets.size());
        unreachable.assign(buckets.size(), {});
        std::vector<uint8_t> built(buckets.size(), 0);
        for (uint32_t b = 0; b < buckets.size(); ++b)
            built[b] = root_vars[b] != kInvalidVarId && !selection_reachability[b].initialized;
        ThreadPool::get().parallel_for(static_cast<uint32_t>(buckets.size()), [&](uint32_t b) {
            if (!built[b])
                return;
            selection_reachability[b].initialize(bucket_egraphs[b], bucket_root_ids[b], selected_vars[b], domains,
                                                 unreachable[b], is_dag);
        });
        for (uint32_t b = 0; b < buckets.size(); ++b)
            if (built[b])
                trail.push_back(ReachabilityTrailEntry{b, root_vars[b], 0, false});
    }

    template <typename F>
    void consumeDirtyDomains(F &&func)
    {
        if (dirty_domains.empty())
            return;
        for (VarId var_id : dirty_domains)
            func(var_id);
        dirty_domains.clear();
        ++current_dirty_epoch;
        if (current_dirty_epoch == 0)
        {
            std::fill(var_dirty_epoch.begin(), var_dirty_epoch.end(), 0);
            current_dirty_epoch = 1;
        }
    }

    std::vector<VarId> takeDirtyDomains()
    {
        std::vector<VarId> result = dirty_domains;
        dirty_domains.clear();
        ++current_dirty_epoch;
        if (current_dirty_epoch == 0)
        {
            std::fill(var_dirty_epoch.begin(), var_dirty_epoch.end(), 0);
            current_dirty_epoch = 1;
        }
        return result;
    }

    void schedulePropagation(VarId var_id)
    {
        markDomainDirty(var_id);
    }

    static uint32_t getDefaultPageAlignment(const MemSpace &ms)
    {
        if (ms.type == HandleType::CPP)
            return 64;
        if (ms.type == HandleType::CUDA)
            return 256;
        if (ms.type == HandleType::OPENCL)
            return 4096;
        return 64;
    }

    uint32_t getPageAlignment(const MemSpace &ms) const
    {
        auto it = page_alignments.find(ms);
        return (it != page_alignments.end()) ? it->second : getDefaultPageAlignment(ms);
    }

    uint64_t getMemoryCap(const MemSpace &ms) const
    {
        auto it = mem_caps.find(ms);
        return (it != mem_caps.end()) ? it->second : (1024ULL * 1024 * 1024 * 4);
    }

    uint32_t bytesToPages(uint64_t bytes, const MemSpace &ms) const
    {
        uint32_t align = getPageAlignment(ms);
        return static_cast<uint32_t>((bytes + align - 1) / align);
    }
};

} // namespace plan

#include "core/plan/search_state_propagation.hpp"
