#pragma once

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dbghelp.h>
#include <cstdio>

// DbgHelp is process global and must never be entered concurrently.
inline SRWLOCK tg_symbol_lock = SRWLOCK_INIT;
inline INIT_ONCE tg_symbol_once = INIT_ONCE_STATIC_INIT;
inline BOOL CALLBACK initializeWindowsSymbols(PINIT_ONCE, PVOID, PVOID *)
{
    SymSetOptions(SYMOPT_DEFERRED_LOADS | SYMOPT_UNDNAME | SYMOPT_LOAD_LINES);
    if (!SymInitialize(GetCurrentProcess(), nullptr, TRUE))
        fprintf(stderr, "Symbol initialization failed: Windows error %lu\n", GetLastError());
    return TRUE;
}
inline void ensureWindowsSymbols()
{
    InitOnceExecuteOnce(&tg_symbol_once, initializeWindowsSymbols, nullptr, nullptr);
}
struct WindowsSymbolsLock
{
    WindowsSymbolsLock() { AcquireSRWLockExclusive(&tg_symbol_lock); }
    ~WindowsSymbolsLock() { ReleaseSRWLockExclusive(&tg_symbol_lock); }
    WindowsSymbolsLock(const WindowsSymbolsLock &) = delete;
    WindowsSymbolsLock &operator=(const WindowsSymbolsLock &) = delete;
};
#endif
