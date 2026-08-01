#pragma once

#ifndef TG_OS_WINDOWS

#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <string>
#include <vector>

#include "core/kernels.hpp"
#include "core/types.hpp"

inline bool matchCopyTo_STORAGE_CPU_POSIX(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].dtype != output.dtype)
        return false;
    if (inputs[0].getShape() != output.getShape())
        return false;
    if (!isContiguous(output))
        return false;

    return true;
}

inline void runCopyTo_STORAGE_CPU_POSIX(const KernelContext &ctx)
{
    int fd = ctx.fd[0];
    if (fd < 0)
    {
        Error::throw_err("STORAGE_CPU_POSIX: Invalid file descriptor (" + std::to_string(fd) + ")");
    }

    uint64_t fileOffset = ctx.inViews[0].offset;
    uint64_t sizeBytes = countElements(ctx.inViews[0]) * getDTypeSize(ctx.inViews[0].dtype);
    uint8_t *dst = static_cast<uint8_t *>(ctx.outputs[0]);

    if (sizeBytes == 0)
        return;

    // Use pread for thread-safe, stateless reading at an absolute offset
    ssize_t bytesRead = ::pread(fd, dst, sizeBytes, static_cast<off_t>(fileOffset));
    if (bytesRead < 0)
    {
        Error::throw_err("STORAGE_CPU_POSIX: pread failed at offset " + std::to_string(fileOffset) +
                         " with error: " + std::string(std::strerror(errno)));
    }

    if (static_cast<uint64_t>(bytesRead) != sizeBytes)
    {
        Error::throw_err("STORAGE_CPU_POSIX: Incomplete read. Expected " + std::to_string(sizeBytes) +
                         " bytes, but read " + std::to_string(bytesRead));
    }
}

REGISTER_REF_KERNEL(OpType::COPY_TO, 1, 1, matchCopyTo_STORAGE_CPU_POSIX, runCopyTo_STORAGE_CPU_POSIX,
                    MemSpace(1, HandleType::CPP),
                    {Engine(0, EngineType::CPU)},
                    {DType::ANY},
                    {{8, 32}},
                    {true},
                    {{MemSpace(0, HandleType::STORAGE)}});

#endif // !TG_OS_WINDOWS