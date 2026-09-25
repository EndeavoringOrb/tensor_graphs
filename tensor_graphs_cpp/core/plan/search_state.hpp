// tensor_graphs_cpp/core/plan/search_state.hpp
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/common/constants.hpp"
#include "core/egraph.hpp"
#include "core/graph.hpp"
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

struct TrailEntry
{
    VarId var_id;
    Domain prev_domain;
};

class SearchState
{
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
    std::vector<EGraph> bucket_egraphs;
    std::vector<EClassId> bucket_root_ids;
    std::vector<std::unordered_set<EClassId>> bucket_clean_eclasses;
    std::vector<std::unordered_map<LogicalId, EClassId>> bucket_node_to_eclass;
    std::vector<std::unordered_map<EClassId, LogicalId>> bucket_eclass_to_logical;
    std::vector<std::vector<ENodeInfo>> bucket_enode_infos;

    std::vector<CacheCandidate> candidates;
    std::unordered_map<MemSpace, uint64_t> mem_caps;
    std::unordered_map<MemSpace, uint32_t> page_alignments;
    std::unordered_map<MemSpace, uint32_t> preallocated_pages;
    std::unordered_map<BaseEClassId, ParallelBuffer> preallocated_buffers;

    SearchState() = default;

    VarId addVar(VarInfo info, const Domain &initial_domain)
    {
        VarId id = static_cast<VarId>(var_infos.size());
        info.id = id;
        var_infos.push_back(std::move(info));
        domains.push_back(initial_domain);
        return id;
    }

    size_t numVars() const
    {
        return var_infos.size();
    }

    size_t getTrailMarker() const
    {
        return trail.size();
    }

    void backtrackTo(size_t marker)
    {
        while (trail.size() > marker)
        {
            const auto &entry = trail.back();
            domains[entry.var_id] = entry.prev_domain;
            trail.pop_back();
        }
    }

    bool setDomain(VarId var_id, const Domain &new_domain)
    {
        if (domains[var_id] != new_domain)
        {
            trail.push_back(TrailEntry{var_id, domains[var_id]});
            domains[var_id] = new_domain;
            return true;
        }
        return false;
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
