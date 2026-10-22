// tensor_graphs_cpp/core/common/malloc_tuning.hpp
#pragma once

// Planning allocates and frees many large vectors (e-graph rebuilds,
// per-bucket propagator tables) from many threads at once. With glibc's
// defaults, blocks above a dynamic threshold are mmapped and unmapped on
// every allocation, paying page faults each time, and freed heap tops are
// trimmed back to the system only to be grown again. Keeping large blocks in
// the heap and never trimming trades peak RSS for allocation speed; it does
// not change any result.
#if defined(__GLIBC__)
#include <malloc.h>

namespace tg_malloc_tuning
{
inline const bool kApplied = []() {
    constexpr int kMmapThreshold = 32 * 1024 * 1024; // glibc's largest heap block
    constexpr int kTopPad = 256 * 1024 * 1024;
    return mallopt(M_MMAP_THRESHOLD, kMmapThreshold) == 1 && mallopt(M_TRIM_THRESHOLD, -1) == 1 &&
           mallopt(M_TOP_PAD, kTopPad) == 1;
}();
} // namespace tg_malloc_tuning
#endif
