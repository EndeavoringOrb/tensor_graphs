#pragma once

// Diagnostic sampling profiler (Windows only). Threads register themselves;
// a sampler thread periodically suspends each one, unwinds its stack and
// counts functions. Enabled by TG_SAMPLE_PROFILE=<report path>.

#ifdef _WIN32

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <string>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dbghelp.h>
#pragma comment(lib, "dbghelp.lib")

#include "core/types.hpp"
#include "core/common/windows_symbols.hpp"

class SamplingProfiler
{
  public:
    static SamplingProfiler &get()
    {
        static SamplingProfiler instance;
        return instance;
    }

    static bool enabled()
    {
        return std::getenv("TG_SAMPLE_PROFILE") != nullptr;
    }

    void start()
    {
        if (running_.exchange(true))
            return;
        stop_ = false;
        sampler_ = std::thread([this]() { loop(); });
    }

    void registerCurrentThread()
    {
        HANDLE handle = nullptr;
        if (!DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &handle,
                             THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, 0))
            Error::throw_err("[SamplingProfiler] DuplicateHandle failed");
        std::lock_guard<std::mutex> lock(threadsMtx_);
        threads_.push_back({GetCurrentThreadId(), handle});
    }

    void unregisterCurrentThread()
    {
        const DWORD id = GetCurrentThreadId();
        std::lock_guard<std::mutex> lock(threadsMtx_);
        for (size_t i = 0; i < threads_.size(); ++i)
            if (threads_[i].first == id)
            {
                CloseHandle(threads_[i].second);
                threads_.erase(threads_.begin() + i);
                return;
            }
    }

    // Stops sampling and writes the report.
    void finish(const std::string &phase)
    {
        if (!running_.exchange(false))
            return;
        stop_ = true;
        sampler_.join();
        const char *path = std::getenv("TG_SAMPLE_PROFILE");
        std::ofstream out(path, std::ios::app);
        if (!out)
            Error::throw_err(std::string("[SamplingProfiler] cannot write ") + path);
        HANDLE process = GetCurrentProcess();
        ensureWindowsSymbols();
        WindowsSymbolsLock symbol_lock;
        auto name = [&](DWORD64 address) {
            alignas(SYMBOL_INFO) char buffer[sizeof(SYMBOL_INFO) + 512];
            SYMBOL_INFO *symbol = reinterpret_cast<SYMBOL_INFO *>(buffer);
            symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
            symbol->MaxNameLen = 511;
            DWORD64 displacement = 0;
            if (SymFromAddr(process, address, &displacement, symbol))
                return std::string(symbol->Name);
            return std::string("?");
        };
        // Aggregate by function name.
        std::unordered_map<std::string, uint64_t> leafByName, inclusiveByName;
        std::unordered_map<DWORD64, std::string> names;
        auto cached = [&](DWORD64 a) -> const std::string & {
            auto it = names.find(a);
            if (it == names.end())
                it = names.emplace(a, name(a)).first;
            return it->second;
        };
        for (const auto &stack : stacks_)
        {
            if (stack.empty())
                continue;
            leafByName[cached(stack.front())]++;
            std::vector<std::string> seen;
            for (DWORD64 pc : stack)
            {
                const std::string &n = cached(pc);
                bool dup = false;
                for (const auto &s : seen)
                    dup = dup || s == n;
                if (dup)
                    continue;
                seen.push_back(n);
                inclusiveByName[n]++;
            }
        }
        auto dump = [&](const char *title, const std::unordered_map<std::string, uint64_t> &counts) {
            std::vector<std::pair<uint64_t, std::string>> sorted;
            for (const auto &[n, c] : counts)
                sorted.push_back({c, n});
            std::sort(sorted.rbegin(), sorted.rend());
            out << "== " << phase << " " << title << " (samples=" << stacks_.size() << ")\n";
            for (size_t i = 0; i < sorted.size() && i < 60; ++i)
                out << sorted[i].first << "\t" << sorted[i].second << "\n";
        };
        dump("leaf", leafByName);
        dump("inclusive", inclusiveByName);
        stacks_.clear();
    }

  private:
    static constexpr size_t kMaxDepth = 48;

    void loop()
    {
        std::vector<DWORD64> frames(kMaxDepth);
        while (!stop_)
        {
            {
                std::lock_guard<std::mutex> lock(threadsMtx_);
                for (const auto &[id, handle] : threads_)
                {
                    size_t depth = 0;
                    if (SuspendThread(handle) == static_cast<DWORD>(-1))
                        continue;
                    CONTEXT ctx{};
                    ctx.ContextFlags = CONTEXT_FULL;
                    if (GetThreadContext(handle, &ctx))
                    {
                        // No allocation while the thread is suspended: it may
                        // hold the heap lock.
                        for (; depth < kMaxDepth; ++depth)
                        {
#if defined(_M_ARM64)
                            const DWORD64 pc = ctx.Pc;
#else
                            const DWORD64 pc = ctx.Rip;
#endif
                            if (pc == 0)
                                break;
                            frames[depth] = pc;
                            DWORD64 imageBase = 0;
                            PRUNTIME_FUNCTION entry = RtlLookupFunctionEntry(pc, &imageBase, nullptr);
                            if (!entry)
                            {
#if defined(_M_ARM64)
                                ctx.Pc = ctx.Lr;
                                if (ctx.Pc == pc)
                                    break;
#else
                                ctx.Rip = *reinterpret_cast<DWORD64 *>(ctx.Rsp);
                                ctx.Rsp += 8;
#endif
                                continue;
                            }
                            PVOID handlerData = nullptr;
                            DWORD64 establisher = 0;
                            RtlVirtualUnwind(UNW_FLAG_NHANDLER, imageBase, pc, entry, &ctx, &handlerData,
                                             &establisher, nullptr);
                        }
                    }
                    ResumeThread(handle);
                    stacks_.emplace_back(frames.begin(), frames.begin() + depth);
                }
            }
            Sleep(2);
        }
    }

    std::mutex threadsMtx_;
    std::vector<std::pair<DWORD, HANDLE>> threads_;
    std::vector<std::vector<DWORD64>> stacks_;
    std::thread sampler_;
    std::atomic<bool> running_{false};
    std::atomic<bool> stop_{false};
};

#endif
