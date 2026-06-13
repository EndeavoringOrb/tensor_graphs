#pragma once

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <limits>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

namespace tg_random
{
    /**
     * Manipulates the input uint32_t so that the output is "random".
     * It is not actually random: the same input will result in the same output,
     * but given a certain input it is hard for an onlooker to predict the output.
     *
     * @param input A 32-bit unsigned integer
     * @return A new 32-bit unsigned integer
     */
    inline uint32_t PCG_Hash(uint32_t input)
    {
        uint32_t state = input * 747796405u + 2891336453u;
        uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        return (word >> 22u) ^ word;
    }

    /**
     * Generates a random floating-point value in the range [0, 1]
     *
     * @param seed A reference to a 32-bit unsigned integer used as the seed.
     * @return A random floating-point value in the range [0, 1].
     */
    inline float randFloat(uint32_t &seed)
    {
        seed = PCG_Hash(seed);
        return static_cast<float>(seed) / static_cast<float>(std::numeric_limits<uint32_t>::max());
    }

    /**
     * Generates a random floating-point value following a normal distribution
     *
     * @param mean A float, the mean of the normal distribution
     * @param stddev A float, the standard deviation of the normal distribution
     * @param seed A reference to a 32-bit unsigned integer used as the seed
     * @return A random floating-point value
     */
    inline float randDist(const float mean, const float stddev, uint32_t &seed)
    {
        // Generate two independent random numbers from a uniform distribution in the range (0,1)
        float u1 = randFloat(seed);
        float u2 = randFloat(seed);

        // Prevent log(0) which results in -infinity
        u1 = std::max(u1, 1e-7f);

        // Box-Muller transform to convert uniform random numbers to normal distribution
        float z0 = stddev * std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * M_PI * u2) + mean;

        return z0;
    }

    /**
     * Generates a random integer value in the range [low, high] (including both endpoints)
     *
     * @param seed A 32-bit unsigned integer used as the seed.
     * @return A random integer value in the range [low, high].
     */
    inline int randInt(uint32_t &seed, int low, int high)
    {
        return std::min(
            static_cast<int>(randFloat(seed) * static_cast<float>(high - low + 1) + static_cast<float>(low)),
            high);
    }

    /**
     * Generates a random unsigned integer value in the range [0, high] (including both endpoints)
     *
     * @param seed A 32-bit unsigned integer used as the seed.
     * @return A random integer value in the range [0, high].
     */
    inline uint32_t randUInt(uint32_t &seed, uint32_t high)
    {
        return std::min(
            static_cast<uint32_t>(randFloat(seed) * static_cast<float>(high + 1)),
            high);
    }
}