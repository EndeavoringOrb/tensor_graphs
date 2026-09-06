#ifdef TG_USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <string>
#include "core/types.hpp"
#include "core/kernels.hpp"

namespace {
struct CublasHandleManager {
    cublasHandle_t handle = nullptr;
    CublasHandleManager() {
        cublasCreate(&handle);
    }
    ~CublasHandleManager() {
        if (handle) {
            cublasDestroy(handle);
            handle = nullptr;
        }
    }
};

inline cublasHandle_t getThreadCublasHandle() {
    static thread_local CublasHandleManager s_mgr;
    return s_mgr.handle;
}
} // namespace

inline bool matchCublasDotF32(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (output.dtype != DType::FLOAT32)
        return false;
    const auto &sA = inputs[0].getShape();
    const auto &sB = inputs[1].getShape();
    const auto &sC = output.getShape();
    if (sA.size() < 2 || sB.size() < 2 || sC.size() < 2)
        return false;
    if (sA.size() != sB.size() || sA.size() != sC.size())
        return false;
    size_t rank = sA.size();
    for (size_t i = 0; i < rank - 2; ++i)
    {
        if (sA[i] != sB[i] || sA[i] != sC[i])
            return false;
    }
    if (sA[rank - 1] != sB[rank - 2])
        return false;
    if (sC[rank - 2] != sA[rank - 2] || sC[rank - 1] != sB[rank - 1])
        return false;
    if (!isContiguous(output))
        return false;
    return true;
}

inline void runCublasDotF32(const KernelContext &ctx)
{
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.cuda_stream());
    const float *A = static_cast<const float *>(ctx.inputs[0]);
    const float *B = static_cast<const float *>(ctx.inputs[1]);
    float *C = static_cast<float *>(ctx.outputs[0]);

    const auto &viewA = ctx.inViews[0];
    const auto &viewB = ctx.inViews[1];
    const auto &viewC = ctx.outViews[0];

    size_t rank = viewA.getShape().size();
    int M = static_cast<int>(viewA.getShape()[rank - 2]);
    int K = static_cast<int>(viewA.getShape()[rank - 1]);
    int N = static_cast<int>(viewB.getShape()[rank - 1]);

    int batchCount = 1;
    for (size_t i = 0; i < rank - 2; ++i)
    {
        batchCount *= static_cast<int>(viewA.getShape()[i]);
    }

    int lda = static_cast<int>(viewA.strides[rank - 2]);
    int ldb = static_cast<int>(viewB.strides[rank - 2]);
    int ldc = static_cast<int>(viewC.strides[rank - 2]);

    long long int strideA = (batchCount > 1 && rank > 2) ? static_cast<long long int>(viewA.strides[rank - 3]) : (long long int)M * K;
    long long int strideB = (batchCount > 1 && rank > 2) ? static_cast<long long int>(viewB.strides[rank - 3]) : (long long int)K * N;
    long long int strideC = (batchCount > 1 && rank > 2) ? static_cast<long long int>(viewC.strides[rank - 3]) : (long long int)M * N;

    cublasHandle_t handle = getThreadCublasHandle();
    cublasSetStream(handle, stream);

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasStatus_t status = cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, ldb, strideB,
        A, lda, strideA,
        &beta,
        C, ldc, strideC,
        batchCount
    );

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        Error::throw_err("cublasSgemmStridedBatched failed with status " + std::to_string(status));
    }
}

inline LogicalId refFactoryCublasDotF32(const std::vector<LogicalId> &inputs, Graph &graph)
{
    if (inputs.size() != 2)
        Error::throw_err("Dot requires 2 inputs");
    return graph.dot(inputs[0], inputs[1]);
}

REGISTER_KERNEL("CuBLAS_Dot_F32", 2, 2, matchCublasDotF32, runCublasDotF32, refFactoryCublasDotF32,
    {}, MemSpace(2, HandleType::CUDA), {Engine(0, EngineType::CUDA_GPU)},
    {DType::FLOAT32, DType::FLOAT32},
    {{2, 8, 16}, {2, 16, 8}}, {true, true},
    {{MemSpace(2, HandleType::CUDA)}, {MemSpace(2, HandleType::CUDA)}});

#endif
