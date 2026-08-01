#pragma once

#include <unordered_map>

#include "core/types.hpp"

// Conceptual interface
class ISelectionValidator
{
  public:
    virtual ~ISelectionValidator() = default;

    // Returns true if the selection map is valid under this rule's constraints.
    // If invalid, populates 'reason' and returns false.
    virtual bool validate(const std::unordered_map<EClassId, uint32_t> &selection_map,
                          const std::vector<EClassId> &order, std::vector<ParallelBuffer> &buffers,
                          std::unordered_map<EClassId, BufferId> &eclass_to_buf, BufferId &overflow, float &cost,
                          std::string &reason, bool &updated_buffers, bool &updated_cost) = 0;
};