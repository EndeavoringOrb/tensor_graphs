// tensor_graphs_cpp/core/plan/enode_info.hpp
#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "core/common/constants.hpp"
#include "core/egraph.hpp"

struct ENodeInfo
{
    float cost = TGConstants::INF;
    bool is_view = false;
    float dp_cost = TGConstants::INF;
    // Optimistic makespan of the best deduplicated DAG rooted at this enode.
    // Unlike dp_cost, this is the maximum work on any individual engine.
    float optimistic_dag_cost = TGConstants::INF;
    float dp_cp_cost = TGConstants::INF;
    float rev_cp_cost = 0.0f;
    float dp_mem = TGConstants::INF;
};

inline EClassId resolve_view_alias(EClassId id, const EGraph &egraph,
                                   const std::unordered_map<EClassId, uint32_t> &selection_map,
                                   const std::vector<ENodeInfo> &enodeInfos)
{
    auto step = [&](EClassId curr) -> EClassId {
        EClassId canon = egraph.findConst(curr);
        auto sel_it = selection_map.find(canon);
        if (sel_it == selection_map.end())
            return EClassId{UINT32_MAX};

        uint32_t sel = sel_it->second;
        const auto &enodes = egraph.getEClass(canon).enodes;
        if (sel >= enodes.size())
            return EClassId{UINT32_MAX};

        ENodeId enode_id = enodes[sel];
        if (enode_id.value < enodeInfos.size() && enodeInfos[enode_id.value].is_view)
        {
            const ENode &node = egraph.getENode(enode_id);
            if (!node.getChildren().empty())
            {
                return egraph.findConst(node.getChildren()[0]);
            }
        }
        return EClassId{UINT32_MAX};
    };

    EClassId slow = egraph.findConst(id);
    EClassId fast = slow;

    while (true)
    {
        EClassId next_slow = step(slow);
        if (next_slow.value == UINT32_MAX)
            return slow;

        EClassId next_fast1 = step(fast);
        if (next_fast1.value == UINT32_MAX)
            return fast;

        EClassId next_fast2 = step(next_fast1);
        if (next_fast2.value == UINT32_MAX)
            return next_fast1;

        slow = next_slow;
        fast = next_fast2;

        if (slow == fast)
        {
            return slow;
        }
    }
}
