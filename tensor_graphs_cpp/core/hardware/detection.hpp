#pragma once

// --- OS Detection ---
#if defined(_WIN32) || defined(_WIN64)
#define TG_OS_WINDOWS
#elif defined(__APPLE__)
#define TG_OS_MACOS
#elif defined(__linux__)
#define TG_OS_LINUX
#endif

// --- Architecture Detection ---
#if defined(__aarch64__) || defined(_M_ARM64)
#define TG_ARCH_ARM64
#if defined(__ARM_NEON) || defined(TG_OS_WINDOWS) // Windows ARM64 always has NEON
#define TG_HAS_NEON
#endif
#elif defined(__x86_64__) || defined(_M_X64)
#define TG_ARCH_X64
#endif

// On x86, build.py uses -march=native, so this reflects the build machine's ISA.
#if defined(__AVX2__)
#define TG_HAS_AVX2
#endif
