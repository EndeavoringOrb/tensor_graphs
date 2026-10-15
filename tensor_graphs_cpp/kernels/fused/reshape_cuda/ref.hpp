#pragma once
#include <vector>
#include <utility>
#include "core/kernels.hpp"
#include "core/types.hpp"

inline bool computeReshapeViewStrides(
    const std::vector<uint32_t> &oldShape,
    const std::vector<uint64_t> &oldStrides,
    const std::vector<uint32_t> &newShape,
    std::vector<uint64_t> &newStrides)
{
    if (countElements(oldShape) != countElements(newShape))
        return false;

    newStrides.assign(newShape.size(), 0);
    if (newShape.empty())
        return true;

    // Check if old is contiguous
    bool old_is_contig = true;
    if (!oldStrides.empty())
    {
        uint64_t expected = 1;
        for (int i = static_cast<int>(oldShape.size()) - 1; i >= 0; --i)
        {
            if (oldShape[i] == 1)
                continue;
            if (oldStrides[i] != expected)
            {
                old_is_contig = false;
                break;
            }
            expected *= oldShape[i];
        }
    }
    if (old_is_contig)
    {
        newStrides = calcContiguousStrides(newShape);
        return true;
    }

    // Check squeeze / unsqueeze (non-1 dims match exactly in order)
    std::vector<std::pair<uint32_t, uint64_t>> non_ones;
    for (size_t i = 0; i < oldShape.size(); ++i)
    {
        if (oldShape[i] != 1)
        {
            uint64_t st = (i < oldStrides.size()) ? oldStrides[i] : 1;
            non_ones.push_back({oldShape[i], st});
        }
    }
    std::vector<size_t> non_one_out_indices;
    for (size_t i = 0; i < newShape.size(); ++i)
    {
        if (newShape[i] != 1)
            non_one_out_indices.push_back(i);
    }

    if (non_ones.size() == non_one_out_indices.size())
    {
        bool match = true;
        for (size_t i = 0; i < non_ones.size(); ++i)
        {
            if (non_ones[i].first != newShape[non_one_out_indices[i]])
            {
                match = false;
                break;
            }
        }
        if (match)
        {
            for (size_t i = 0; i < non_ones.size(); ++i)
            {
                newStrides[non_one_out_indices[i]] = non_ones[i].second;
            }
            // Fill 1-dim strides from right to left
            uint64_t running_stride = 1;
            for (int i = static_cast<int>(newShape.size()) - 1; i >= 0; --i)
            {
                if (newShape[i] == 1)
                {
                    newStrides[i] = running_stride;
                }
                else
                {
                    running_stride = newStrides[i] * newShape[i];
                }
            }
            return true;
        }
    }

    return false;
}

inline LogicalId refFactoryReshapeCUDA(const std::vector<LogicalId> &inputs, Graph &g)
{
    return g.reshape(inputs[0], inputs[1]);
}

