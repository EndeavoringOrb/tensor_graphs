#include "core/egraph.hpp"
#include "core/plan/validators/validator.hpp"

struct CycleValidator : public ISelectionValidator
{
    const EGraph &egraph;
    std::vector<uint32_t> indegree;
    std::vector<EClassId> zero_indegree;

    CycleValidator(const EGraph &_egraph) : indegree(_egraph.getClasses().size(), 0), egraph(_egraph)
    {
        zero_indegree.reserve(egraph.getClasses().size());
    }

    // detectCycles — proper implementation of the stub.
    //   Kahn's algorithm restricted to the selection-induced subgraph.
    //   Returns true and sets reason="cycle" if a cycle is found.
    bool detectCycles(const std::unordered_map<EClassId, uint32_t> &selection_map, std::string &reason)
    {
        std::fill(indegree.begin(), indegree.end(), 0);
        for (const auto &kv : selection_map)
        {
            uint32_t sel = kv.second;
            ENodeId enode_id = egraph.getEClass(kv.first).enodes[sel];
            for (EClassId child : egraph.getENode(enode_id).getChildren())
            {
                indegree[egraph.findConst(child).value]++;
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

            uint32_t sel = selection_map.at(curr);
            ENodeId enode_id = egraph.getEClass(curr).enodes[sel];
            for (EClassId child : egraph.getENode(enode_id).getChildren())
            {
                child = egraph.findConst(child);
                indegree[child.value]--;
                if (indegree[child.value] == 0)
                {
                    zero_indegree.push_back(child);
                }
            }
        }

        if (processed < selection_map.size())
        {
            reason = "cycle";
            return true;
        }
        return false;
    }

    bool validate(const std::unordered_map<EClassId, uint32_t> &selection_map, const std::vector<EClassId> &order,
                  std::vector<ParallelBuffer> &buffers, std::unordered_map<EClassId, BufferId> &eclass_to_buf,
                  BufferId &overflow, float &cost, std::string &reason, bool &updated_buffers,
                  bool &updated_cost) override
    {
        return !detectCycles(selection_map, reason);
    }
};