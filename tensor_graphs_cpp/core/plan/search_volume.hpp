#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "core/plan/domain.hpp"

namespace plan
{

inline double emptyLog2Volume()
{
    return -std::numeric_limits<double>::infinity();
}

// Domain::size() uses int32_t; a full signed integer interval has 2^32 values.
inline uint64_t domainCardinality(const Domain &domain)
{
    if (domain.isEmpty())
        return 0;
    if (domain.is_mask)
        return static_cast<uint64_t>(domain.size());
    return static_cast<uint64_t>(static_cast<int64_t>(domain.max_val) - domain.min_val + 1);
}

inline uint64_t domainIntersectionCardinality(const Domain &a, const Domain &b)
{
    if (a.isEmpty() || b.isEmpty())
        return 0;
    if (!a.is_mask && !b.is_mask)
    {
        const int64_t lo = std::max(a.min_val, b.min_val);
        const int64_t hi = std::min(a.max_val, b.max_val);
        return hi < lo ? 0 : static_cast<uint64_t>(hi - lo + 1);
    }
    const Domain &mask_domain = a.is_mask ? a : b;
    const Domain &other = a.is_mask ? b : a;
    uint64_t count = 0;
    for (uint32_t bit = 0; bit < 64; ++bit)
    {
        if ((mask_domain.mask & (uint64_t{1} << bit)) == 0)
            continue;
        const int64_t value = static_cast<int64_t>(mask_domain.min_val) + bit;
        if (value < std::numeric_limits<int32_t>::min() || value > std::numeric_limits<int32_t>::max())
            Error::throw_err("Mask domain exceeds signed integer bounds in search volume");
        const int64_t offset = value - other.min_val;
        const bool contained = other.is_mask
                                   ? offset >= 0 && offset < 64 && (other.mask & (uint64_t{1} << offset)) != 0
                                   : value >= other.min_val && value <= other.max_val;
        if (contained)
            ++count;
    }
    return count;
}

inline double domainLog2Volume(const Domain &domain)
{
    const uint64_t count = domainCardinality(domain);
    return count == 0 ? emptyLog2Volume() : std::log2(static_cast<double>(count));
}

inline double addLog2Volumes(double a, double b)
{
    if (a == emptyLog2Volume())
        return b;
    if (b == emptyLog2Volume())
        return a;
    const double hi = std::max(a, b);
    const double lo = std::min(a, b);
    return hi + std::log1p(std::exp2(lo - hi)) / std::log(2.0);
}

// Sum log cardinalities in a stable tree. Point updates and restoration take
// O(log variables), avoiding a whole-domain scan at each search node.
class DomainVolumeTracker
{
    size_t leaf_count = 1;
    size_t domain_count = 0;
    std::vector<double> sums;

  public:
    bool initialized() const { return !sums.empty(); }

    void initialize(const std::vector<Domain> &domains)
    {
        domain_count = domains.size();
        leaf_count = 1;
        while (leaf_count < domain_count)
            leaf_count *= 2;
        sums.assign(2 * leaf_count, 0.0);
        for (size_t i = 0; i < domains.size(); ++i)
            sums[leaf_count + i] = domainLog2Volume(domains[i]);
        for (size_t i = leaf_count; i-- > 1;)
            sums[i] = sums[2 * i] + sums[2 * i + 1];
    }

    void setDomain(size_t var_id, const Domain &domain)
    {
        if (!initialized() || var_id >= domain_count)
            Error::throw_err("Invalid domain update in search volume tracker");
        size_t index = leaf_count + var_id;
        sums[index] = domainLog2Volume(domain);
        while (index > 1)
        {
            index /= 2;
            sums[index] = sums[2 * index] + sums[2 * index + 1];
        }
    }

    double log2Volume() const
    {
        if (!initialized())
            Error::throw_err("Search volume tracker has not been initialized");
        return sums[1];
    }
};

struct SearchVolumeEvaluation
{
    double total_domain_bits = 0.0;
    double remaining_domain_bits = emptyLog2Volume();
    double log2_remaining_fraction = emptyLog2Volume();
    double eliminated_bits = std::numeric_limits<double>::infinity();
    size_t remaining_nodes = 0;

    bool exhausted() const { return remaining_nodes == 0; }
};

// Callers supply disjoint frontier boxes, including any popped active box.
// This function does not estimate feasible-assignment counts within the boxes.
inline SearchVolumeEvaluation evaluateSearchVolume(double total_domain_bits,
                                                   const std::vector<double> &remaining_box_bits)
{
    if (!std::isfinite(total_domain_bits) || total_domain_bits < 0.0)
        Error::throw_err("Search volume requires a nonempty initial domain");
    SearchVolumeEvaluation result;
    result.total_domain_bits = total_domain_bits;
    for (double box_bits : remaining_box_bits)
    {
        if (box_bits == emptyLog2Volume())
            continue;
        if (!std::isfinite(box_bits) || box_bits < 0.0)
            Error::throw_err("Invalid remaining hyperbox volume");
        result.remaining_domain_bits = addLog2Volumes(result.remaining_domain_bits, box_bits);
        ++result.remaining_nodes;
    }
    if (!result.exhausted())
    {
        const double tolerance = 1.0e-10 * std::max(1.0, total_domain_bits);
        if (result.remaining_domain_bits > total_domain_bits + tolerance)
            Error::throw_err("Remaining hyperbox union exceeds initial domain volume");
        result.log2_remaining_fraction = std::min(0.0, result.remaining_domain_bits - total_domain_bits);
        result.eliminated_bits = -result.log2_remaining_fraction;
    }
    return result;
}

} // namespace plan
