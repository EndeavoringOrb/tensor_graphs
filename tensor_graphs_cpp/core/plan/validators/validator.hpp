#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "core/types.hpp"

// Conceptual interface
class ISelectionValidator
{
  public:
    virtual ~ISelectionValidator() = default;

    // Called during extraction (descent) when a node is assigned an enode.
    // Return false to reject this choice immediately.
    virtual bool validateStep(EClassId current, ENodeId enode_id,
                              const std::unordered_map<EClassId, uint32_t> &selection_map,
                              std::vector<EClassId> &conflict_nodes)
    {
        return true;
    }

    // Returns true if the selection map is valid under this rule's constraints.
    // If invalid, populates 'conflict_nodes' and returns false.
    virtual bool validate(const std::unordered_map<EClassId, uint32_t> &selection_map,
                          const std::vector<EClassId> &order, const std::vector<EClassId> &path,
                          std::vector<ParallelBuffer> &buffers, std::unordered_map<EClassId, BufferId> &eclass_to_buf,
                          float &cost, std::vector<EClassId> &conflict_nodes) = 0;
};