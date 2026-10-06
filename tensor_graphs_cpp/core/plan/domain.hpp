// tensor_graphs_cpp/core/plan/domain.hpp
#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>

#include "core/types.hpp"

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif

namespace plan
{

struct Domain
{
    bool is_mask = false;
    uint32_t mask = 0;
    int32_t min_val = 0; // Represents lower bound for ranges, or base offset for masks
    int32_t max_val = 0; // Represents upper bound for ranges (unused when is_mask)

    static Domain makeMask(uint32_t m, int32_t base = 0)
    {
        Domain d;
        d.is_mask = true;
        d.mask = m;
        d.min_val = base;
        d.max_val = 0;
        return d;
    }

    static Domain makeRange(int32_t lo, int32_t hi)
    {
        Domain d;
        d.is_mask = false;
        d.min_val = lo;
        d.max_val = hi;
        return d;
    }

    static Domain makeFixed(int32_t val, bool use_mask = false, int32_t base = 0)
    {
        if (use_mask)
        {
            if (val >= base && val - base < 32)
            {
                return makeMask(1u << (val - base), base);
            }
            return makeMask(1u, val);
        }
        return makeRange(val, val);
    }

    static Domain makeEmpty(bool use_mask = false, int32_t base = 0)
    {
        if (use_mask)
        {
            return makeMask(0, base);
        }
        return makeRange(1, 0);
    }

    bool isEmpty() const
    {
        if (is_mask)
            return mask == 0;
        return min_val > max_val;
    }

    bool isFixed() const
    {
        if (is_mask)
            return mask != 0 && (mask & (mask - 1)) == 0;
        return min_val == max_val;
    }

    int32_t fixedValue() const
    {
        assert(isFixed());
        if (is_mask)
        {
#if defined(_MSC_VER) && !defined(__clang__)
            unsigned long idx = 0;
            _BitScanForward(&idx, mask);
            return min_val + static_cast<int32_t>(idx);
#else
            return min_val + __builtin_ctz(mask);
#endif
        }
        return min_val;
    }

    bool contains(int32_t val) const
    {
        if (is_mask)
        {
            uint32_t offset = static_cast<uint32_t>(val - min_val);
            if (offset >= 32u)
                return false;
            return (mask & (1u << offset)) != 0;
        }
        return val >= min_val && val <= max_val;
    }

    int32_t size() const
    {
        if (is_mask)
        {
#if defined(_MSC_VER) && !defined(__clang__)
            return static_cast<int32_t>(__popcnt(mask));
#else
            return __builtin_popcount(mask);
#endif
        }
        return (min_val <= max_val) ? (max_val - min_val + 1) : 0;
    }

    int32_t getMin() const
    {
        if (is_mask)
        {
            if (mask == 0)
                return min_val + 32;
#if defined(_MSC_VER) && !defined(__clang__)
            unsigned long idx = 0;
            _BitScanForward(&idx, mask);
            return min_val + static_cast<int32_t>(idx);
#else
            return min_val + __builtin_ctz(mask);
#endif
        }
        return min_val;
    }

    int32_t getMax() const
    {
        if (is_mask)
        {
            if (mask == 0)
                return min_val - 1;
#if defined(_MSC_VER) && !defined(__clang__)
            unsigned long idx = 0;
            _BitScanReverse(&idx, mask);
            return min_val + static_cast<int32_t>(idx);
#else
            return min_val + (31 - __builtin_clz(mask));
#endif
        }
        return max_val;
    }

    bool remove(int32_t val)
    {
        if (is_mask)
        {
            uint32_t offset = static_cast<uint32_t>(val - min_val);
            if (offset < 32u && (mask & (1u << offset)))
            {
                mask &= ~(1u << offset);
                return true;
            }
            return false;
        }
        else
        {
            if (val == min_val && min_val <= max_val)
            {
                min_val++;
                return true;
            }
            if (val == max_val && min_val <= max_val)
            {
                max_val--;
                return true;
            }
            if (val > min_val && val < max_val && (max_val - min_val < 32))
            {
                int32_t span = max_val - min_val + 1;
                uint32_t m = (span == 32) ? ~0u : ((1u << span) - 1u);
                is_mask = true;
                mask = m & ~(1u << (val - min_val));
                max_val = 0;
                return true;
            }
            return false;
        }
    }

    bool setMin(int32_t lo)
    {
        if (is_mask)
        {
            if (lo <= min_val)
                return false;
            if (lo - min_val >= 32)
            {
                if (mask == 0)
                    return false;
                mask = 0;
                return true;
            }
            uint32_t shift = static_cast<uint32_t>(lo - min_val);
            uint32_t allowed_mask = ~((1u << shift) - 1u);
            if ((mask & allowed_mask) != mask)
            {
                mask &= allowed_mask;
                return true;
            }
            return false;
        }
        else
        {
            if (lo > min_val)
            {
                min_val = lo;
                return true;
            }
            return false;
        }
    }

    bool setMax(int32_t hi)
    {
        if (is_mask)
        {
            if (hi < min_val)
            {
                if (mask == 0)
                    return false;
                mask = 0;
                return true;
            }
            if (hi - min_val >= 31)
                return false;
            uint32_t shift = static_cast<uint32_t>(hi - min_val);
            uint32_t allowed_mask = (1u << (shift + 1u)) - 1u;
            if ((mask & allowed_mask) != mask)
            {
                mask &= allowed_mask;
                return true;
            }
            return false;
        }
        else
        {
            if (hi < max_val)
            {
                max_val = hi;
                return true;
            }
            return false;
        }
    }

    bool setRange(int32_t lo, int32_t hi)
    {
        bool c1 = setMin(lo);
        bool c2 = setMax(hi);
        return c1 || c2;
    }

    bool canConvertToMask() const
    {
        return !is_mask && !isEmpty() && (max_val - min_val < 32);
    }

    bool convertToMask()
    {
        if (is_mask)
            return false;
        if (isEmpty())
        {
            is_mask = true;
            mask = 0;
            return true;
        }
        int32_t span = max_val - min_val + 1;
        if (span > 32)
            return false;
        is_mask = true;
        mask = (span == 32) ? ~0u : ((1u << span) - 1u);
        max_val = 0;
        return true;
    }

    bool convertToRange()
    {
        if (!is_mask)
            return false;
        if (isEmpty())
        {
            is_mask = false;
            min_val = 1;
            max_val = 0;
            return true;
        }
        int32_t lo = getMin();
        int32_t hi = getMax();
        int32_t expected_size = hi - lo + 1;
        if (size() != expected_size)
            return false;
        is_mask = false;
        min_val = lo;
        max_val = hi;
        return true;
    }

    bool intersectWith(const Domain &other)
    {
        if (is_mask && other.is_mask)
        {
            if (min_val == other.min_val)
            {
                uint32_t new_mask = mask & other.mask;
                if (new_mask != mask)
                {
                    mask = new_mask;
                    return true;
                }
                return false;
            }
            uint32_t new_mask = 0;
            if (min_val < other.min_val)
            {
                int32_t diff = other.min_val - min_val;
                uint32_t other_aligned = (diff < 32) ? (other.mask << diff) : 0u;
                new_mask = mask & other_aligned;
            }
            else
            {
                int32_t diff = min_val - other.min_val;
                uint32_t other_aligned = (diff < 32) ? (other.mask >> diff) : 0u;
                new_mask = mask & other_aligned;
            }
            if (new_mask != mask)
            {
                mask = new_mask;
                return true;
            }
            return false;
        }
        else if (!is_mask && !other.is_mask)
        {
            return setRange(std::max(min_val, other.min_val), std::min(max_val, other.max_val));
        }
        else if (is_mask && !other.is_mask)
        {
            return setRange(other.min_val, other.max_val);
        }
        else
        {
            return setRange(other.getMin(), other.getMax());
        }
    }

    bool operator==(const Domain &other) const
    {
        if (is_mask != other.is_mask)
            return false;
        if (is_mask)
        {
            if (mask == 0 && other.mask == 0)
                return true;
            if (min_val == other.min_val)
                return mask == other.mask;
            if (min_val < other.min_val)
            {
                int32_t diff = other.min_val - min_val;
                if (diff >= 32)
                    return false;
                return (mask >> diff) == other.mask && (mask & ((1u << diff) - 1u)) == 0;
            }
            else
            {
                int32_t diff = min_val - other.min_val;
                if (diff >= 32)
                    return false;
                return (other.mask >> diff) == mask && (other.mask & ((1u << diff) - 1u)) == 0;
            }
        }
        return min_val == other.min_val && max_val == other.max_val;
    }

    bool operator!=(const Domain &other) const
    {
        return !(*this == other);
    }

    std::string toString() const
    {
        if (isEmpty())
            return "empty";
        if (is_mask)
        {
            std::string s = "{";
            bool first = true;
            for (int i = 0; i < 32; ++i)
            {
                if (mask & (1u << i))
                {
                    if (!first)
                        s += ",";
                    s += std::to_string(min_val + i);
                    first = false;
                }
            }
            s += "}";
            return s;
        }
        return "[" + std::to_string(min_val) + ".." + std::to_string(max_val) + "]";
    }
};

} // namespace plan
