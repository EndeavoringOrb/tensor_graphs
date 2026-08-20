#pragma once
#include <unordered_map>
#include <vector>

#include "core/egraph.hpp"
#include "core/plan/validators/validator.hpp"

struct CycleStepValidator : public ISelectionValidator
{
    const EGraph &egraph;
    std::vector<uint32_t> visited;
    uint32_t visited_gen = 0;

    CycleStepValidator(const EGraph &_egraph) : egraph(_egraph), visited(_egraph.getClasses().size(), 0)
    {
    }

    bool reaches(EClassId start, EClassId target, const std::unordered_map<EClassId, uint32_t> &selection_map,
                 std::vector<EClassId> &out_cycle)
    {
        if (start == target)
        {
            out_cycle.push_back(start);
            return true;
        }

        std::vector<EClassId> q;
        q.push_back(start);
        visited_gen++;
        visited[start.value] = visited_gen;

        std::unordered_map<EClassId, EClassId> came_from;

        int head = 0;
        while (head < q.size())
        {
            EClassId curr = q[head++];

            auto it = selection_map.find(curr);
            if (it == selection_map.end())
                continue;

            uint32_t sel = it->second;
            ENodeId enode_id = egraph.getEClass(curr).enodes[sel];
            for (EClassId child : egraph.getENode(enode_id).getChildren())
            {
                EClassId canon_child = egraph.findConst(child);
                if (canon_child == target)
                {
                    out_cycle.push_back(target);
                    EClassId p = curr;
                    while (true)
                    {
                        out_cycle.push_back(p);
                        if (p == start)
                            break;
                        p = came_from[p];
                    }
                    return true;
                }

                if (visited[canon_child.value] != visited_gen)
                {
                    visited[canon_child.value] = visited_gen;
                    came_from[canon_child] = curr;
                    q.push_back(canon_child);
                }
            }
        }
        return false;
    }

    bool validateStep(EClassId current, ENodeId enode_id, const std::unordered_map<EClassId, uint32_t> &selection_map,
                      std::vector<EClassId> &conflict_nodes) override
    {
        const ENode &enode = egraph.getENode(enode_id);
        for (EClassId child : enode.getChildren())
        {
            EClassId canon_child = egraph.findConst(child);
            if (selection_map.find(canon_child) != selection_map.end())
            {
                if (reaches(canon_child, current, selection_map, conflict_nodes))
                {
                    return false;
                }
            }
        }
        return true;
    }

    bool validate(const std::unordered_map<EClassId, uint32_t> &selection_map, const std::vector<EClassId> &order,
                  const std::vector<EClassId> &path, std::vector<ParallelBuffer> &buffers,
                  std::unordered_map<EClassId, BufferId> &eclass_to_buf, float &cost,
                  std::vector<EClassId> &conflict_nodes) override
    {
        return true;
    }
};

struct CycleValidator : public ISelectionValidator
{
    const EGraph &egraph;
    std::vector<uint32_t> indegree;
    std::vector<EClassId> zero_indegree;

    CycleValidator(const EGraph &_egraph) : indegree(_egraph.getClasses().size(), 0), egraph(_egraph)
    {
        zero_indegree.reserve(egraph.getClasses().size());
    }

    bool validateStep(EClassId current, ENodeId enode_id, const std::unordered_map<EClassId, uint32_t> &selection_map,
                      std::vector<EClassId> &conflict_nodes) override
    {
        return true;
    }

    bool detectCycles(const std::unordered_map<EClassId, uint32_t> &selection_map)
    {
        std::fill(indegree.begin(), indegree.end(), 0);
        for (const auto &kv : selection_map)
        {
            uint32_t sel = kv.second;
            ENodeId enode_id = egraph.getEClass(kv.first).enodes[sel];
            for (EClassId child : egraph.getENode(enode_id).getChildren())
            {
                EClassId canon_child = egraph.findConst(child);
                if (selection_map.find(canon_child) != selection_map.end())
                {
                    indegree[canon_child.value]++;
                }
            }
        }

        zero_indegree.clear();
        for (const auto &kv : selection_map)
        {
            if (indegree[kv.first.value] == 0)
            {
                zero_indegree.push_back(kv.first);
            }
        }

        uint32_t processed = 0;
        while (!zero_indegree.empty())
        {
            EClassId curr = zero_indegree.back();
            zero_indegree.pop_back();
            processed++;

            auto sel_it = selection_map.find(curr);
            if (sel_it == selection_map.end())
                continue;

            uint32_t sel = sel_it->second;
            ENodeId enode_id = egraph.getEClass(curr).enodes[sel];
            for (EClassId child : egraph.getENode(enode_id).getChildren())
            {
                child = egraph.findConst(child);
                if (selection_map.find(child) != selection_map.end())
                {
                    indegree[child.value]--;
                    if (indegree[child.value] == 0)
                    {
                        zero_indegree.push_back(child);
                    }
                }
            }
        }

        if (processed < selection_map.size())
        {
            return true;
        }
        return false;
    }

    bool validate(const std::unordered_map<EClassId, uint32_t> &selection_map, const std::vector<EClassId> &order,
                  const std::vector<EClassId> &path, std::vector<ParallelBuffer> &buffers,
                  std::unordered_map<EClassId, BufferId> &eclass_to_buf, float &cost,
                  std::vector<EClassId> &conflict_nodes) override
    {
        return !detectCycles(selection_map);
    }
};