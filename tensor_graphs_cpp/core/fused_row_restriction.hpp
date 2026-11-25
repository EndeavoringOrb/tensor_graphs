#pragma once

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/types.hpp"

// Kept out of core/kernels.hpp: that header seeds every kernel uid (build.py
// CORE_DEPENDENCIES), so changing it would invalidate all cost records.

// How a fused kernel restricts to a box of output rows, for kernels whose row
// structure cannot be derived from their reference pattern (see
// fusedRowLocality). Restricting output rows [a, b) along `output_axis`:
// - input i with input_axis[i] >= 0 is sliced to rows [a, b) along that axis;
// - input i with input_axis[i] < 0 is passed unchanged;
// - the INT32 [1] input `offset_input` (if >= 0) gets `a` added: kernels that
//   compute positions from their local row index stay correct.
struct FusedRowRestriction
{
    uint32_t output_axis = 0;
    std::vector<int32_t> input_axis;
    int32_t offset_input = -1;

    bool operator==(const FusedRowRestriction &other) const
    {
        return output_axis == other.output_axis && input_axis == other.input_axis &&
               offset_input == other.offset_input;
    }
};

class FusedRowRestrictionRegistry
{
  public:
    static FusedRowRestrictionRegistry &get()
    {
        static FusedRowRestrictionRegistry instance;
        return instance;
    }

    void add(const std::string &opName, const FusedRowRestriction &restriction)
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (restriction.offset_input >= static_cast<int32_t>(restriction.input_axis.size()))
            Error::throw_err("[FusedRowRestrictionRegistry] offset input out of range for " + opName);
        if (restriction.offset_input >= 0 && restriction.input_axis[restriction.offset_input] >= 0)
            Error::throw_err("[FusedRowRestrictionRegistry] offset input of " + opName + " cannot also be sliced");
        auto [it, inserted] = restrictions.emplace(opName, restriction);
        if (!inserted && !(it->second == restriction))
            Error::throw_err("[FusedRowRestrictionRegistry] conflicting row restrictions for " + opName);
    }

    const FusedRowRestriction *find(const std::string &opName) const
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = restrictions.find(opName);
        return it == restrictions.end() ? nullptr : &it->second;
    }

  private:
    mutable std::mutex mutex;
    std::unordered_map<std::string, FusedRowRestriction> restrictions;
};

struct FusedRowRestrictionRegistrar
{
    FusedRowRestrictionRegistrar(const std::string &opName, const FusedRowRestriction &restriction)
    {
        FusedRowRestrictionRegistry::get().add(opName, restriction);
    }
};

