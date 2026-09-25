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
    int32_t min_val = 0;
    int32_t max_val = 0;

    static Domain makeMask(uint32_t m)
    {
        Domain d;
        d.is_mask = true;
        d.mask = m;
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

    static Domain makeFixed(int32_t val, bool use_mask = false)
    {
        if (use_mask)
        {
            if (val < 0 || val >= 32)
            {
                Error::throw_err("Domain::makeFixed mask value out of range: " + std::to_string(val) +
                                 " (expected 0..31)");
            }
            return makeMask(1u << val);
        }
        return makeRange(val, val);
    }

    static Domain makeEmpty(bool use_mask = false)
    {
        if (use_mask)
        {
            return makeMask(0);
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
        return min_val == max_val && min_val <= max_val;
    }

    int32_t fixedValue() const
    {
        assert(isFixed());
        if (is_mask)
        {
#if defined(_MSC_VER) && !defined(__clang__)
            unsigned long idx = 0;
            _BitScanForward(&idx, mask);
            return static_cast<int32_t>(idx);
#else
            return __builtin_ctz(mask);
#endif
        }
        return min_val;
    }

    bool contains(int32_t val) const
    {
        if (is_mask)
        {
            if (val < 0 || val >= 32)
                return false;
            return (mask & (1u << val)) != 0;
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
                return 32;
#if defined(_MSC_VER) && !defined(__clang__)
            unsigned long idx = 0;
            _BitScanForward(&idx, mask);
            return static_cast<int32_t>(idx);
#else
            return __builtin_ctz(mask);
#endif
        }
        return min_val;
    }

    int32_t getMax() const
    {
        if (is_mask)
        {
            if (mask == 0)
                return -1;
#if defined(_MSC_VER) && !defined(__clang__)
            unsigned long idx = 0;
            _BitScanReverse(&idx, mask);
            return static_cast<int32_t>(idx);
#else
            return 31 - __builtin_clz(mask);
#endif
        }
        return max_val;
    }

    bool remove(int32_t val)
    {
        if (is_mask)
        {
            if (val >= 0 && val < 32 && (mask & (1u << val)))
            {
                mask &= ~(1u << val);
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
            return false;
        }
    }

    bool setMin(int32_t lo)
    {
        if (is_mask)
        {
            if (lo <= 0)
                return false;
            if (lo >= 32)
            {
                if (mask == 0)
                    return false;
                mask = 0;
                return true;
            }
            uint32_t allowed_mask = ~((1u << lo) - 1);
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
            if (hi >= 31)
                return false;
            if (hi < 0)
            {
                if (mask == 0)
                    return false;
                mask = 0;
                return true;
            }
            uint32_t allowed_mask = (1u << (hi + 1)) - 1;
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

    bool intersectWith(const Domain &other)
    {
        if (is_mask && other.is_mask)
        {
            uint32_t new_mask = mask & other.mask;
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
            return mask == other.mask;
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
                    s += std::to_string(i);
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
