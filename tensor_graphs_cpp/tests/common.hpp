#pragma once

#include <iostream>
#include <vector>

#include "core/cost_model.hpp"
#include "core/egraph.hpp"
#include "core/graph.hpp"
#include "core/misc.hpp"
#include "core/plan/planner.hpp"
#include "core/types.hpp"

Region makeRegion(std::initializer_list<Dim> dims)
{
    Region r;
    r.region.assign(dims.begin(), dims.end());
    return r;
}

bool regionListEquals(const std::vector<Region> &actual, const std::vector<Region> &expected)
{
    const auto a = normalizeRegions(actual);
    const auto e = normalizeRegions(expected);
    if (a.size() != e.size())
        return false;
    for (uint64_t i = 0; i < a.size(); ++i)
    {
        if (!regionsMatch(a[i], e[i]))
            return false;
    }
    return true;
}

void assertRegionListEquals(const std::vector<Region> &actual, const std::vector<Region> &expected,
                            const std::string &label)
{
    if (!regionListEquals(actual, expected))
    {
        std::stringstream ss;
        ss << "[RegionTest] " << label << " expected " << encodeRegionList(expected) << " but got "
           << encodeRegionList(actual);
        Error::throw_err(ss.str());
    }
}

bool compareOutputs(const float *ref, const float *test, uint64_t elements, float eps = 1e-4f)
{
    for (uint64_t i = 0; i < elements; ++i)
    {
        if (std::abs(ref[i] - test[i]) > eps)
        {
            std::cout << "\nMismatch at index " << i << ": (ref)" << ref[i] << " != (test)" << test[i] << std::endl;
            return false;
        }
    }
    return true;
}

bool compareOutputs(const int32_t *ref, const int32_t *test, uint64_t elements, float eps = 1e-4f)
{
    for (uint64_t i = 0; i < elements; ++i)
    {
        if (ref[i] != test[i])
        {
            std::cout << "\nMismatch at index " << i << ": (ref)" << ref[i] << " != (test)" << test[i] << std::endl;
            return false;
        }
    }
    return true;
}

bool compareOutputs(const bool *ref, const bool *test, uint64_t elements, float eps = 1e-4f)
{
    for (uint64_t i = 0; i < elements; ++i)
    {
        if (ref[i] != test[i])
        {
            std::cout << "\nMismatch at index " << i << ": (ref)" << ref[i] << " != (test)" << test[i] << std::endl;
            return false;
        }
    }
    return true;
}