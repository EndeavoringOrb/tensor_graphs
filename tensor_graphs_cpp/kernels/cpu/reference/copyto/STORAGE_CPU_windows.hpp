#ifdef TG_OS_WINDOWS

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <io.h>
#include <windows.h>
#ifdef ERROR
#undef ERROR
#endif

#include <cstring>
#include <string>
#include <vector>

#include "core/kernels.hpp"
#include "core/types.hpp"

inline bool matchCopyTo_STORAGE_CPU(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].dtype != output.dtype)
        return false;
    if (inputs[0].getShape() != output.getShape())
        return false;
    if (!isContiguous(output))
        return false;

    return true;
}

inline void runCopyTo_STORAGE_CPU(const KernelContext &ctx)
{
    int fd = ctx.fd[0];
    if (fd < 0)
    {
        Error::throw_err("STORAGE_CPU_WINDOWS: Invalid file descriptor (" + std::to_string(fd) + ")");
    }

    uint64_t fileOffset = ctx.inViews[0].offset;
    uint64_t sizeBytes = countElements(ctx.inViews[0]) * getDTypeSize(ctx.inViews[0].dtype);
    uint8_t *dst = static_cast<uint8_t *>(ctx.outputs[0]);

    if (sizeBytes == 0)
        return;

    // Convert standard POSIX file descriptor to Win32 HANDLE
    HANDLE hFile = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    if (hFile == INVALID_HANDLE_VALUE)
    {
        Error::throw_err("STORAGE_CPU_WINDOWS: Failed to retrieve Win32 handle "
                         "from file descriptor.");
    }

    // Configure OVERLAPPED struct for stateless read at absolute offset (prevents
    // file pointer pointer clobbering)
    OVERLAPPED overlapped = {};
    overlapped.Offset = static_cast<DWORD>(fileOffset & 0xFFFFFFFF);
    overlapped.OffsetHigh = static_cast<DWORD>((fileOffset >> 32) & 0xFFFFFFFF);

    DWORD bytesRead = 0;
    if (!ReadFile(hFile, dst, static_cast<DWORD>(sizeBytes), &bytesRead, &overlapped))
    {
        DWORD err = GetLastError();
        Error::throw_err("STORAGE_CPU_WINDOWS: ReadFile failed at offset " + std::to_string(fileOffset) +
                         " with error code " + std::to_string(err));
    }

    if (bytesRead != sizeBytes)
    {
        Error::throw_err("STORAGE_CPU_WINDOWS: Incomplete read. Expected " + std::to_string(sizeBytes) +
                         " bytes, but read " + std::to_string(bytesRead));
    }
}

REGISTER_REF_KERNEL(OpType::COPY_TO, 1, 1, matchCopyTo_STORAGE_CPU, runCopyTo_STORAGE_CPU, MemSpace(1, HandleType::CPP),
                    {Engine(0, EngineType::CPU)}, {DType::ANY}, {{8, 32}}, {true},
                    {{MemSpace(0, HandleType::STORAGE)}});

#endif // TG_OS_WINDOWS