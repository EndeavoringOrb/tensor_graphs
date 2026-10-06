// tensor_graphs_cpp/core/plan/propagators/base_to_view_offset.hpp
#pragma once

#include "core/plan/propagators/base.hpp"
#include "core/plan/propagators/view_offset_helpers.hpp"

namespace plan
{

// BaseToViewOffsetPropagator: see docs/core/propagators.md.
class BaseToViewOffsetPropagator : public Propagator
{
    std::vector<std::unordered_map<EClassId, std::vector<EClassId>>> bucket_view_users;
    bool initialized = false;

    void ensureInitialized(const SearchState &state)
    {
        if (initialized && bucket_view_users.size() == state.buckets.size())
            return;
        ViewOffsetHelpers::buildBucketViewUsers(state, bucket_view_users);
        initialized = true;
    }

    bool propagateEClass(SearchState &state, uint32_t b, EClassId base_cid, std::vector<VarId> &worklist)
    {
        if (b >= bucket_view_users.size())
            return true;
        auto it = bucket_view_users[b].find(base_cid);
        if (it == bucket_view_users[b].end())
            return true;

        for (EClassId v_cid : it->second)
        {
            EClassId v_base_cid;
            if (ViewOffsetHelpers::getConfirmedViewBase(state, b, v_cid, v_base_cid) && v_base_cid == base_cid)
            {
                if (!ViewOffsetHelpers::intersectViewAndBase(state, b, v_cid, base_cid, worklist))
                    return false;
            }
            else
            {
                if (!ViewOffsetHelpers::filterUnviableViewEnodes(state, b, v_cid, base_cid, worklist))
                    return false;
            }
        }
        return true;
    }

  public:
    std::string name() const override
    {
        return "BaseToViewOffsetPropagator";
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        ensureInitialized(state);
        if (changed != kInvalidVarId)
        {
            const auto &info = state.var_infos[changed];
            if (info.type != VarType::OFFSET)
                return true;
            return propagateEClass(state, info.bucket_idx, info.eclass_id, worklist);
        }
        else
        {
            for (uint32_t b = 0; b < state.buckets.size(); ++b)
            {
                for (const auto &pair : state.offset_vars[b])
                {
                    if (!propagateEClass(state, b, pair.first, worklist))
                        return false;
                }
            }
        }
        return true;
    }
};

} // namespace plan
