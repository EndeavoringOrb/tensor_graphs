// tensor_graphs_cpp/core/plan/search_state.hpp
#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include "core/common/constants.hpp"
#include "core/egraph.hpp"
#include "core/graph.hpp"
#include "core/logging.hpp"
#include "core/plan/domain.hpp"
#include "core/plan/enode_info.hpp"
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
    uint32_t enode_idx = 0;
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

    void initialize(const EGraph &egraph, EClassId root_id,
                    const std::unordered_map<EClassId, VarId> &selected_vars,
                    const std::vector<Domain> &domains, std::vector<VarId> &unreachable)
    {
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
                    edges.push_back(Edge{from, to, domains[var_id].contains(en_idx + 1)});
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

    void update(uint32_t node_idx, const Domain &selection, std::vector<VarId> &unreachable)
    {
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
            while (node.level < infinity)
            {
                while (node.next_incoming < node.incoming.size() &&
                       !supportsLevel(node.incoming[node.next_incoming], node.level))
                    ++node.next_incoming;
                if (node.next_incoming < node.incoming.size())
                    break;
                ++node.level;
                node.next_incoming = 0;
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
    struct StartPrecedenceParent
    {
        EClassId parent_cid;
        uint32_t en_idx = 0;
        VarId selection_var = kInvalidVarId;
        VarId start_var = kInvalidVarId;
        bool is_view = false;
    };

    struct StartPrecedenceBucket
    {
        std::unordered_map<EClassId, std::vector<StartPrecedenceParent>> parents;
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
        EClassId base_cid{UINT32_MAX};
        uint32_t en_idx = 0;
        ENodeId en_id{UINT32_MAX};
        int32_t start_max = -1;
        int32_t max_reader_start_max = -1;
        std::vector<EClassId> readers;
    };

    struct WriteAfterReadBucket
    {
        bool dirty = true;
        std::vector<WriteAfterReadClassInfo> class_info;
        std::vector<std::vector<EClassId>> temporal_overlaps;
        std::vector<EClassId> active_cids;
        std::vector<EClassId> touched_cids;
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
        std::multiset<int32_t> finish_times;
        std::unordered_set<VarId> open_lifetimes;
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
    std::unordered_set<VarId> memory_dirty;

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

    std::vector<uint32_t> mem_affected_stamp;
    uint32_t mem_prop_aff_stamp = 0;
    std::vector<VarId> mem_affected_scratch;
};

class SearchState
{
  private:
    // Kept incrementally so SearchEngine can detect contradictions without
    // scanning every domain after each propagator invocation.
    std::unordered_set<VarId> empty_domains;

    // Domain changes are consumed by SearchEngine's propagator worklist.
    std::vector<VarId> dirty_domains;
    std::vector<uint32_t> var_dirty_epoch;
    uint32_t current_dirty_epoch = 1;

    // Mutable propagation data belongs to the state so copied searches retain
    // independent incremental propagation data.
    std::vector<SelectionReachability> selection_reachability;

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
        if (info.type != VarType::SELECTED && info.type != VarType::START)
            return;
        if (info.bucket_idx < propagation.write_after_read.size())
            propagation.write_after_read[info.bucket_idx].dirty = true;
    }

  public:
    std::vector<VarInfo> var_infos;
    std::vector<Domain> domains;
    std::vector<TrailEntry> trail;

    // Lookups
    std::unordered_map<BaseEClassId, VarId> cached_vars;
    std::vector<std::unordered_map<EClassId, VarId>> selected_vars;
    std::vector<std::unordered_map<EClassId, std::vector<VarId>>> start_vars;
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
    std::vector<EGraph> bucket_egraphs;
    std::vector<EClassId> bucket_root_ids;
    std::vector<std::unordered_set<EClassId>> bucket_clean_eclasses;
    std::vector<std::unordered_map<LogicalId, EClassId>> bucket_node_to_eclass;
    std::vector<std::unordered_map<EClassId, LogicalId>> bucket_eclass_to_logical;
    std::vector<std::vector<ENodeInfo>> bucket_enode_infos;
    std::vector<std::vector<EClassId>> reachable_cids;

    std::vector<CacheCandidate> candidates;
    std::unordered_map<MemSpace, uint64_t> mem_caps;
    std::unordered_map<MemSpace, uint32_t> page_alignments;
    std::unordered_map<MemSpace, uint32_t> preallocated_pages;
    std::unordered_map<BaseEClassId, ParallelBuffer> preallocated_buffers;

    // The graph, metadata, capacities and weights are immutable after the first
    // propagation. All derived data is owned by this state, including in copies.
    mutable PropagationState propagation;
    // The incumbent is global to this search, so backtracking must retain it.
    float best_cost = TGConstants::INF;

    void ensurePropagationState() const;
    void updatePropagationContribution(VarId var_id, bool add) const;
    void markMemoryAffected(VarId var_id) const;
    int32_t selectedAlternative(VarId sel_var) const;
    bool isSelectedParent(VarId parent, VarId child) const;
    float costLowerBound() const;

    void updateLowerBoundBucket(uint32_t bucket_idx)
    {
        if (bucket_lower_bounds.size() < buckets.size())
            bucket_lower_bounds.resize(buckets.size(), 0.0f);
        if (bucket_critical_path_lower_bounds.size() < buckets.size())
            bucket_critical_path_lower_bounds.resize(buckets.size(), 0.0f);
        if (bucket_engine_work_lower_bounds.size() < buckets.size())
            bucket_engine_work_lower_bounds.resize(buckets.size(), 0.0f);
        bucket_lower_bounds[bucket_idx] = std::max(bucket_critical_path_lower_bounds[bucket_idx],
                                                   bucket_engine_work_lower_bounds[bucket_idx]);
        lower_bound = 0.0f;
        for (size_t b = 0; b < bucket_lower_bounds.size(); ++b)
        {
            const float weight = b < bucket_weights.size() ? bucket_weights[b] : 1.0f;
            if (weight > 0.0f)
                lower_bound += weight * bucket_lower_bounds[b];
        }
    }

    SearchState() = default;

    VarId addVar(VarInfo info, const Domain &initial_domain)
    {
        assert(!propagation.initialized);
        VarId id = static_cast<VarId>(var_infos.size());
        info.id = id;
        var_infos.push_back(std::move(info));
        domains.push_back(initial_domain);
        if (initial_domain.isEmpty())
            empty_domains.insert(id);
        markDomainDirty(id);
        return id;
    }

    // Call once per bucket, after CACHE insertion and enode pruning are complete.
    // Preserve eclass/enode IDs for extraction; allocate dense VarIds only for
    // canonical classes reachable through at least one surviving alternative.
    void addBucketVariables(uint32_t b)
    {
        selected_vars.resize(buckets.size());
        start_vars.resize(buckets.size());
        offset_vars.resize(buckets.size());
        reachable_cids.resize(buckets.size());
        assert(selected_vars[b].empty() && start_vars[b].empty() && offset_vars[b].empty());

        const EGraph &egraph = bucket_egraphs[b];
        const EClassId root_cid = egraph.findConst(bucket_root_ids[b]);
        bucket_root_ids[b] = root_cid;
        reachable_cids[b] = collectReachableCids(egraph, root_cid);
        const uint32_t max_start = static_cast<uint32_t>(reachable_cids[b].size() - 1);
        for (EClassId cid : reachable_cids[b])
        {
            const EClass &cls = egraph.getEClass(cid);
            uint32_t n_enodes = static_cast<uint32_t>(cls.enodes.size());
            if (n_enodes > 31)
            {
                Error::throw_err("EClass " + std::to_string(cid.value) + " has " + std::to_string(n_enodes) +
                                 " enodes, exceeding domain bitmask capacity (max 31).");
            }

            // selected_<bucket_id>_<eclass_id> in {0, 1, ..., n_enodes}
            VarInfo sel_info;
            sel_info.type = VarType::SELECTED;
            sel_info.bucket_idx = b;
            sel_info.eclass_id = cid;
            sel_info.name = "selected_" + std::to_string(b) + "_" + std::to_string(cid.value);

            uint32_t mask = UINT32_MAX >> (31 - n_enodes);
            if (cid == root_cid)
            {
                mask &= ~1u; // Root must be selected
            }

            VarId sel_vid = addVar(sel_info, Domain::makeMask(mask));
            selected_vars[b][cid] = sel_vid;

            // Starts encode dispatch order, so N reachable classes need at most N slots.
            for (uint32_t en_idx = 0; en_idx < n_enodes; ++en_idx)
            {
                VarInfo st_info;
                st_info.type = VarType::START;
                st_info.bucket_idx = b;
                st_info.eclass_id = cid;
                st_info.enode_idx = en_idx;
                st_info.name = "start_" + std::to_string(b) + "_" + std::to_string(cid.value) + "_" +
                               std::to_string(en_idx);

                VarId st_vid = addVar(st_info, Domain::makeRange(0, max_start));
                start_vars[b][cid].push_back(st_vid);
            }

            // offset_<bucket_id>_<eclass_id> in [preallocated_pages, max_pages]
            if (cls.mem_space.type != HandleType::STORAGE)
            {
                VarInfo off_info;
                off_info.type = VarType::OFFSET;
                off_info.bucket_idx = b;
                off_info.eclass_id = cid;
                off_info.mem_space = cls.mem_space;
                off_info.size_bytes = getSizeBytes(cls.shape, cls.dtype);
                off_info.size_pages = bytesToPages(off_info.size_bytes, cls.mem_space);
                off_info.name = "offset_" + std::to_string(b) + "_" + std::to_string(cid.value);

                uint32_t align = getPageAlignment(cls.mem_space);
                uint64_t cap = getMemoryCap(cls.mem_space);
                uint32_t max_p = (cap > off_info.size_bytes) ? static_cast<uint32_t>((cap - off_info.size_bytes) / align) : 0;
                const auto prealloc_it = preallocated_pages.find(cls.mem_space);
                uint32_t min_p = prealloc_it == preallocated_pages.end() ? 0 : prealloc_it->second;

                VarId off_vid = addVar(off_info, Domain::makeRange(min_p, std::max(min_p, max_p)));
                offset_vars[b][cid] = off_vid;
            }
        }
    }

    size_t numVars() const
    {
        return var_infos.size();
    }

    size_t getTrailMarker() const
    {
        return trail.size();
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
                markMemoryAffected(entry->var_id);
                const bool was_empty = domains[entry->var_id].isEmpty();
                domains[entry->var_id] = entry->prev_domain;
                invalidateWriteAfterRead(entry->var_id);
                updatePropagationContribution(entry->var_id, true);
                markMemoryAffected(entry->var_id);
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
            updatePropagationContribution(var_id, false);
            markMemoryAffected(var_id);
            const bool was_empty = domains[var_id].isEmpty();
            trail.push_back(DomainTrailEntry{var_id, domains[var_id]});
            domains[var_id] = new_domain;
            invalidateWriteAfterRead(var_id);
            updatePropagationContribution(var_id, true);
            markMemoryAffected(var_id);
            updateEmptyDomainIndex(var_id, was_empty, new_domain.isEmpty());
            markDomainDirty(var_id);
            return true;
        }
        return false;
    }

    // Called only for the changed SELECTED variable. The first call builds a
    // bucket once; later calls touch removed enodes and affected tree nodes.
    void updateSelectionReachability(VarId changed, std::vector<VarId> &unreachable)
    {
        const uint32_t bucket_idx = var_infos[changed].bucket_idx;
        if (selection_reachability.size() < buckets.size())
            selection_reachability.resize(buckets.size());
        auto &reachability = selection_reachability[bucket_idx];
        if (!reachability.initialized)
        {
            trail.push_back(ReachabilityTrailEntry{bucket_idx, changed, 0, false});
            reachability.initialize(bucket_egraphs[bucket_idx], bucket_root_ids[bucket_idx],
                                    selected_vars[bucket_idx], domains, unreachable);
            return;
        }
        const uint32_t node_idx = reachability.node_indices.at(changed);
        if (reachability.nodes[node_idx].selection == domains[changed])
            return;
        trail.push_back(ReachabilityTrailEntry{bucket_idx, changed, reachability.undo.size(), true});
        reachability.update(node_idx, domains[changed], unreachable);
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

    uint32_t getPageAlignment(const MemSpace &ms) const
    {
        auto it = page_alignments.find(ms);
        return (it != page_alignments.end()) ? it->second : 4096;
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
