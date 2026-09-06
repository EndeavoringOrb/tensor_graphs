// File: tensor_graphs_cpp/core/cost_model.hpp
// TODO: Enhanced NaN protection in prediction algorithms

#pragma once
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/graph.hpp"
#include "core/kernels.hpp"
#include "core/misc.hpp"
#include "core/ops/ops.hpp"
#include "core/reference_graph_registry.hpp"
#include "core/shape_propagator.hpp"
#include "core/types.hpp"
#include "generated/build_context.gen.hpp"

#if defined(TG_USE_CUDA)
#define HW_TAG "CUDA_Enabled"
#else
#if defined(TG_OS_WINDOWS)
#define PLAT_OS_STR "Windows"
#elif defined(TG_OS_MACOS)
#define PLAT_OS_STR "macOS"
#elif defined(TG_OS_LINUX)
#define PLAT_OS_STR "Linux"
#else
#define PLAT_OS_STR "UnknownOS"
#endif

#if defined(TG_ARCH_ARM64)
#define PLAT_ARCH_STR "ARM64"
#elif defined(TG_ARCH_X64)
#define PLAT_ARCH_STR "x64"
#else
#define PLAT_ARCH_STR "UnknownArch"
#endif

#define HW_TAG PLAT_OS_STR "_" PLAT_ARCH_STR
#endif

struct Record
{
    KernelId kernelId = KernelId{0};
    uint64_t buildContextId = 0;
    std::string hwTag = HW_TAG;

    std::vector<std::vector<uint32_t>> inputShapes;
    std::vector<uint32_t> outputShape;
    std::vector<std::vector<uint64_t>> inputStrides;
    std::vector<uint64_t> outputStrides;
    std::vector<DType> inputDTypes;
    DType outputDType = DType::FLOAT32;
    std::vector<std::vector<uint8_t>> inputConstants;
    MemSpace output_mem_space = {1, HandleType::CPP};
    std::vector<Engine> engines = {Engine(0, EngineType::CPU)};
    std::vector<MemSpace> input_mem_spaces;
    float runTime = 0.0f;
};

inline void tg_serialize(BinaryWriter &bw, const Record &val)
{
    bw.write(val.kernelId);
    bw.write(val.buildContextId);
    bw.write(val.hwTag);
    bw.write(val.inputShapes);
    bw.write(val.outputShape);
    bw.write(val.inputStrides);
    bw.write(val.outputStrides);
    bw.write(val.inputDTypes);
    bw.write(val.outputDType);
    bw.write(val.inputConstants);
    bw.write(val.output_mem_space);
    bw.write(val.engines);
    bw.write(val.input_mem_spaces);
    bw.write(val.runTime);
}

inline void tg_deserialize(BinaryReader &br, Record &val)
{
    br.read(val.kernelId);
    br.read(val.buildContextId);
    br.read(val.hwTag);
    br.read(val.inputShapes);
    br.read(val.outputShape);
    br.read(val.inputStrides);
    br.read(val.outputStrides);
    br.read(val.inputDTypes);
    br.read(val.outputDType);
    br.read(val.inputConstants);
    br.read(val.output_mem_space);
    br.read(val.engines);
    br.read(val.input_mem_spaces);
    br.read(val.runTime);
}

inline double getInnerContigElements(const std::vector<uint32_t> &shape, const std::vector<uint64_t> &strides)
{
    if (shape.empty() || strides.empty() || shape.size() != strides.size())
        return 1.0;
    uint64_t contig = 1;
    uint64_t expectedStride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i)
    {
        if (shape[i] <= 1)
            continue;
        if (strides[i] == expectedStride)
        {
            contig *= shape[i];
            expectedStride *= shape[i];
        }
        else
        {
            break;
        }
    }
    return static_cast<double>(std::max<uint64_t>(1, contig));
}

inline uint64_t getInnermostStride(const std::vector<uint32_t> &shape, const std::vector<uint64_t> &strides)
{
    if (shape.empty() || strides.empty())
        return 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i)
    {
        if (shape[i] > 1)
        {
            return (i < static_cast<int>(strides.size())) ? strides[i] : 1;
        }
    }
    return 1;
}

inline bool hasZeroStride(const std::vector<uint32_t> &shape, const std::vector<uint64_t> &strides)
{
    for (size_t i = 0; i < shape.size() && i < strides.size(); ++i)
    {
        if (shape[i] > 1 && strides[i] == 0)
            return true;
    }
    return false;
}

inline bool isCommutativeOp(OpType op, const std::string &op_name)
{
    if (op == OpType::ADD || op == OpType::MUL || op == OpType::MAX)
        return true;
    if (!op_name.empty())
    {
        if (op_name.find("Add") != std::string::npos || op_name.find("add") != std::string::npos ||
            op_name.find("Mul") != std::string::npos || op_name.find("mul") != std::string::npos)
        {
            return true;
        }
    }
    return false;
}

inline double getUniqueElements(const std::vector<uint32_t> &shape, const std::vector<uint64_t> &strides)
{
    if (shape.empty())
        return 1.0;
    uint64_t count = 1;
    for (size_t i = 0; i < shape.size(); ++i)
    {
        if (i < strides.size() && strides[i] == 0 && shape[i] > 1)
            continue;
        count *= shape[i];
    }
    return static_cast<double>(std::max<uint64_t>(1, count));
}

inline uint32_t getEffectiveRank(const std::vector<uint32_t> &shape)
{
    uint32_t nonTrivial = 0;
    for (uint32_t d : shape)
    {
        if (d > 1)
            nonTrivial++;
    }
    return std::max<uint32_t>(1, nonTrivial);
}

struct CostModel
{
    struct ModelKey
    {
        KernelId kernelId;
        uint64_t numInputs;
        bool operator==(const ModelKey &o) const
        {
            return kernelId == o.kernelId && numInputs == o.numInputs;
        }
        bool operator!=(const ModelKey &o) const
        {
            return !(*this == o);
        }
    };

    struct ModelKeyHash
    {
        uint64_t operator()(const ModelKey &k) const
        {
            return std::hash<KernelId>()(k.kernelId) ^ (std::hash<uint64_t>()(k.numInputs) << 1);
        }
    };

    struct Matrix
    {
        int rows, cols;
        std::vector<double> data;
        Matrix(int r, int c) : rows(r), cols(c), data(r * c, 0.0)
        {
        }
        double &operator()(int r, int c)
        {
            return data[r * cols + c];
        }
        double operator()(int r, int c) const
        {
            return data[r * cols + c];
        }
    };

    static Matrix transpose(const Matrix &A)
    {
        Matrix B(A.cols, A.rows);
        for (int i = 0; i < A.rows; ++i)
            for (int j = 0; j < A.cols; ++j)
                B(j, i) = A(i, j);
        return B;
    }

    static Matrix multiply(const Matrix &A, const Matrix &B)
    {
        Matrix C(A.rows, B.cols);
        for (int i = 0; i < A.rows; ++i)
            for (int k = 0; k < A.cols; ++k)
                for (int j = 0; j < B.cols; ++j)
                    C(i, j) += A(i, k) * B(k, j);
        return C;
    }

    static bool invert(Matrix &A)
    {
        if (A.rows != A.cols)
            return false;
        int n = A.rows;
        Matrix I(n, n);
        for (int i = 0; i < n; ++i)
            I(i, i) = 1.0;

        // Gauss-Jordan elimination
        for (int i = 0; i < n; ++i)
        {
            double maxEl = std::abs(A(i, i));
            int pivot = i;
            for (int j = i + 1; j < n; ++j)
            {
                if (std::abs(A(j, i)) > maxEl)
                {
                    maxEl = std::abs(A(j, i));
                    pivot = j;
                }
            }
            if (maxEl < 1e-12)
                return false;

            if (pivot != i)
            {
                for (int j = 0; j < n; ++j)
                {
                    std::swap(A(i, j), A(pivot, j));
                    std::swap(I(i, j), I(pivot, j));
                }
            }

            double pivotVal = A(i, i);
            for (int j = 0; j < n; ++j)
            {
                A(i, j) /= pivotVal;
                I(i, j) /= pivotVal;
            }

            for (int k = 0; k < n; ++k)
            {
                if (k == i)
                    continue;
                double factor = A(k, i);
                for (int j = 0; j < n; ++j)
                {
                    A(k, j) -= factor * A(i, j);
                    I(k, j) -= factor * I(i, j);
                }
            }
        }
        A = I;
        return true;
    }

    struct LinearModel
    {
        std::vector<double> weights;
        std::vector<double> scale;
        bool valid = false;
        std::vector<Record> records;
        Record singleRecord;
        bool hasSingleRecord = false;
        OpType opType = OpType::INPUT;
        std::string opName = "";
        ReferenceFactory refFactory = nullptr;

        struct Candidate
        {
            double dist;
            double estTime;
        };

        float predict(const std::vector<double> &features, const WorkloadMetrics &target_w,
                      const std::vector<std::vector<uint32_t>> &in_shapes,
                      const std::vector<std::vector<uint64_t>> &in_strides, const std::vector<DType> &in_dtypes,
                      const std::vector<uint32_t> &out_shape, const std::vector<uint64_t> &out_strides,
                      DType out_dtype) const
        {
            const std::vector<Record> *recs_ptr = &records;
            std::vector<Record> single_list;
            if (recs_ptr->empty() && hasSingleRecord)
            {
                single_list.push_back(singleRecord);
                recs_ptr = &single_list;
            }

            if (recs_ptr->empty())
            {
                return 1e-6f;
            }

            double target_bytes = target_w.bytesRead + target_w.bytesWritten;

            bool is_dot = (opType == OpType::DOT) || (opName.find("Dot") != std::string::npos) ||
                          (opName.find("dot") != std::string::npos) || (opName.find("linear") != std::string::npos) ||
                          (opName.find("conv") != std::string::npos) || (opName.find("Conv") != std::string::npos) ||
                          (opName.find("gemm") != std::string::npos) || (opName.find("GEMM") != std::string::npos) ||
                          (opName.find("matmul") != std::string::npos) || (opName.find("MatMul") != std::string::npos) ||
                          (opName.find("Attention") != std::string::npos) || (opName.find("attention") != std::string::npos);

            std::vector<Candidate> candidates;
            candidates.reserve(recs_ptr->size());

            uint64_t tgt_out_stride = getInnermostStride(out_shape, out_strides);
            bool tgt_out_contig = (tgt_out_stride == 1);
            uint32_t tgt_eff_rank = getEffectiveRank(out_shape);

            for (const auto &r : *recs_ptr)
            {
                ReferenceFactory active_factory = refFactory;
                if (!active_factory && KernelRegistry::get().hasKernel(r.kernelId))
                {
                    active_factory = KernelRegistry::get().getKernel(r.kernelId).refFactory;
                }
                if (!active_factory && !opName.empty())
                {
                    const auto *ref_entry = ReferenceGraphRegistry::get().getFactory(opName);
                    if (ref_entry)
                        active_factory = ref_entry->factory;
                }

                WorkloadMetrics ref_w;
                if (active_factory)
                {
                    ref_w = computeWorkloadFromRefFactory(active_factory, r.inputShapes, r.inputDTypes,
                                                          r.outputShape, r.outputDType, r.inputConstants);
                }
                else
                {
                    ref_w = computeWorkload(opType, r.inputShapes, r.inputDTypes,
                                            r.outputShape, r.outputDType, opName, r.inputConstants);
                }

                double ref_time = std::max(1e-6, static_cast<double>(std::isnan(r.runTime) ? 1e-6f : r.runTime));
                double ratio = 1.0;
                double ref_bytes = ref_w.bytesRead + ref_w.bytesWritten;
                bool has_flops = (ref_w.flops > 0.0 && target_w.flops > 0.0);

                if (is_dot || has_flops)
                {
                    if (has_flops)
                    {
                        double flop_ratio = target_w.flops / ref_w.flops;
                        double byte_ratio = (ref_bytes > 0.0) ? (target_bytes / ref_bytes) : 1.0;
                        ratio = std::max(flop_ratio, byte_ratio);
                    }
                    else
                    {
                        ratio = (ref_bytes > 0.0) ? (target_bytes / ref_bytes) : 1.0;
                    }
                }
                else if (opType == OpType::SUM || opType == OpType::MAX || opType == OpType::ARGMAX)
                {
                    double ref_in_elems = r.inputShapes.empty() ? 1.0 : countElements(r.inputShapes[0]);
                    double target_in_elems = in_shapes.empty() ? 1.0 : countElements(in_shapes[0]);
                    ratio = (ref_in_elems > 0.0) ? (target_in_elems / ref_in_elems) : 1.0;
                }
                else
                {
                    if (ref_bytes > 0.0 && target_bytes > 0.0)
                    {
                        ratio = target_bytes / ref_bytes;
                    }
                    else
                    {
                        double ref_out = countElements(r.outputShape);
                        double target_out = countElements(out_shape);
                        ratio = (ref_out > 0.0) ? (target_out / ref_out) : 1.0;
                    }

                    // Rank & indexing arithmetic penalty adjustment
                    double ref_eff_rank = getEffectiveRank(r.outputShape);
                    if (ref_eff_rank > 0 && tgt_eff_rank > 0 && ref_eff_rank != tgt_eff_rank)
                    {
                        ratio *= (0.4 + 0.6 * (static_cast<double>(tgt_eff_rank) / ref_eff_rank));
                    }

                    // Stride and layout penalty adjustments for data movement / elementwise kernels
                    uint64_t ref_out_stride = getInnermostStride(r.outputShape, r.outputStrides);
                    bool ref_out_contig = (ref_out_stride == 1);

                    bool any_tgt_strided = (!tgt_out_contig);
                    bool any_ref_strided = (!ref_out_contig);
                    double max_tgt_cache_waste = tgt_out_contig ? 1.0 : std::min(16.0, static_cast<double>(tgt_out_stride));
                    double max_ref_cache_waste = ref_out_contig ? 1.0 : std::min(16.0, static_cast<double>(ref_out_stride));

                    bool any_tgt_zero = false;
                    bool any_ref_zero = false;

                    for (size_t i = 0; i < in_shapes.size() && i < r.inputShapes.size(); ++i)
                    {
                        uint64_t ref_in_stride = getInnermostStride(r.inputShapes[i], r.inputStrides[i]);
                        uint64_t tgt_in_stride = getInnermostStride(in_shapes[i], in_strides[i]);
                        if (ref_in_stride > 1)
                        {
                            any_ref_strided = true;
                            max_ref_cache_waste = std::max(max_ref_cache_waste, std::min(16.0, static_cast<double>(ref_in_stride)));
                        }
                        if (tgt_in_stride > 1)
                        {
                            any_tgt_strided = true;
                            max_tgt_cache_waste = std::max(max_tgt_cache_waste, std::min(16.0, static_cast<double>(tgt_in_stride)));
                        }
                        if (hasZeroStride(in_shapes[i], in_strides[i]))
                            any_tgt_zero = true;
                        if (hasZeroStride(r.inputShapes[i], r.inputStrides[i]))
                            any_ref_zero = true;
                    }

                    if (any_tgt_zero && !any_ref_zero)
                    {
                        ratio *= 0.5;
                    }
                    else if (!any_tgt_zero && any_ref_zero)
                    {
                        ratio *= 2.0;
                    }

                    if (!any_ref_strided && any_tgt_strided)
                    {
                        double loop_overhead = 6.0;
                        ratio *= (max_tgt_cache_waste * loop_overhead);
                    }
                    else if (any_ref_strided && !any_tgt_strided)
                    {
                        double loop_overhead = 6.0;
                        ratio /= (max_ref_cache_waste * loop_overhead);
                    }
                    else if (any_ref_strided && any_tgt_strided)
                    {
                        if (max_ref_cache_waste > 0.0)
                        {
                            ratio *= (max_tgt_cache_waste / max_ref_cache_waste);
                        }
                    }
                }

                double est_time = ref_time * ratio;

                double dist = 0.0;
                if (has_flops)
                {
                    dist += std::abs(std::log(std::max(1.0, target_w.flops)) - std::log(std::max(1.0, ref_w.flops)));
                    dist += 0.2 * std::abs(std::log(std::max(1.0, target_bytes)) - std::log(std::max(1.0, ref_bytes)));
                }
                else
                {
                    dist += std::abs(std::log(std::max(1.0, target_bytes)) - std::log(std::max(1.0, ref_bytes)));
                }

                // Layout and memory geometry distance
                auto calcInputPairDist = [&](size_t tgt_idx, size_t ref_idx) -> double {
                    uint64_t tgt_in_stride = getInnermostStride(in_shapes[tgt_idx], in_strides[tgt_idx]);
                    uint64_t ref_in_stride = getInnermostStride(r.inputShapes[ref_idx], r.inputStrides[ref_idx]);
                    double d = 0.0;
                    bool tgt_in_contig = (tgt_in_stride == 1);
                    bool ref_in_contig = (ref_in_stride == 1);
                    if (tgt_in_contig != ref_in_contig)
                    {
                        d += 10.0;
                    }
                    else if (!tgt_in_contig && !ref_in_contig)
                    {
                        d += 0.5 * std::abs(std::log(std::max(1.0, static_cast<double>(tgt_in_stride))) -
                                            std::log(std::max(1.0, static_cast<double>(ref_in_stride))));
                    }

                    bool tgt_in_zero = hasZeroStride(in_shapes[tgt_idx], in_strides[tgt_idx]);
                    bool ref_in_zero = hasZeroStride(r.inputShapes[ref_idx], r.inputStrides[ref_idx]);
                    if (tgt_in_zero != ref_in_zero)
                    {
                        d += 5.0;
                    }
                    return d;
                };

                bool is_comm = isCommutativeOp(opType, opName);
                if (in_shapes.size() == 2 && r.inputShapes.size() == 2 && is_comm)
                {
                    double dist_direct = calcInputPairDist(0, 0) + calcInputPairDist(1, 1);
                    double dist_swapped = calcInputPairDist(0, 1) + calcInputPairDist(1, 0);
                    dist += std::min(dist_direct, dist_swapped);
                }
                else
                {
                    for (size_t i = 0; i < in_shapes.size() && i < r.inputShapes.size(); ++i)
                    {
                        dist += calcInputPairDist(i, i);
                    }
                }

                uint64_t ref_out_stride = getInnermostStride(r.outputShape, r.outputStrides);
                bool ref_out_contig = (ref_out_stride == 1);
                if (tgt_out_contig != ref_out_contig)
                {
                    dist += 10.0;
                }
                else if (!tgt_out_contig && !ref_out_contig)
                {
                    dist += 0.5 * std::abs(std::log(std::max(1.0, static_cast<double>(tgt_out_stride))) -
                                           std::log(std::max(1.0, static_cast<double>(ref_out_stride))));
                }

                uint32_t ref_eff_rank = getEffectiveRank(r.outputShape);
                if (tgt_eff_rank != ref_eff_rank)
                {
                    dist += 0.2 * std::abs(static_cast<double>(tgt_eff_rank) - static_cast<double>(ref_eff_rank));
                }

                candidates.push_back({dist, est_time});
            }

            if (candidates.empty())
            {
                return 1e-6f;
            }

            std::sort(candidates.begin(), candidates.end(), [](const Candidate &a, const Candidate &b) {
                return a.dist < b.dist;
            });

            if (candidates[0].dist < 1e-4)
            {
                return static_cast<float>(std::max(1e-6, candidates[0].estTime));
            }

            size_t k = std::min<size_t>(candidates.size(), 3);
            double total_weight = 0.0;
            double total_weighted_time = 0.0;
            for (size_t i = 0; i < k; ++i)
            {
                double d = candidates[i].dist;
                double w = 1.0 / (d * d + 1e-4);
                total_weight += w;
                total_weighted_time += w * candidates[i].estTime;
            }

            double y = (total_weight > 0.0) ? (total_weighted_time / total_weight) : 1e-6;
            if (std::isnan(y) || std::isinf(y))
                return 1e-6f;
            return static_cast<float>(std::max(1e-6, y));
        }
    };

    std::unordered_map<KernelId, std::vector<Record>> records;
    std::unordered_map<ModelKey, LinearModel, ModelKeyHash> models;
    std::unordered_set<uint64_t> loggedCalls;
    std::ofstream callFile;
    std::mutex logMtx;
    std::atomic<bool> doneWarning{false};
    bool enableLogging = false;

    CostModel(bool logCalls = true, const std::string &recordsPath = "benchmarks/records.bin") : enableLogging(logCalls)
    {
        if (enableLogging)
        {
            initLogging();
        }
        if (!recordsPath.empty() && std::filesystem::exists(recordsPath))
        {
            load(recordsPath);
        }
    }

    void initLogging()
    {
        if (callFile.is_open())
            return;

        const std::string path = "benchmarks/calls.bin";
        std::filesystem::create_directories(std::filesystem::path(path).parent_path());
        {
            std::ifstream inFile(path, std::ios::binary);
            if (inFile.is_open())
            {
                BinaryReader br(inFile);
                while (inFile.peek() != EOF)
                {
                    Record r;
                    br.read(r);
                    r.runTime = 0.0f;
                    loggedCalls.insert(std::hash<std::string>{}(serializeToString(r)));
                }
            }
        }
        callFile.open(path, std::ios::app | std::ios::binary);
        if (!callFile.is_open())
            std::cerr << "Failed to open " << path << " for appending.\n";
    }

    void setLogging(bool enable)
    {
        enableLogging = enable;
        if (enableLogging)
        {
            initLogging();
        }
    }

    void log_call(KernelId kernelId, const std::vector<uint32_t> &outShape, const std::vector<uint64_t> &outStrides,
                  DType outDType, const std::vector<std::vector<uint32_t>> &inShapes,
                  const std::vector<std::vector<uint64_t>> &inStrides, const std::vector<DType> &inDTypes,
                  const std::vector<std::vector<uint8_t>> &inConstants)
    {
        if (!enableLogging)
            return;

        Record r;
        r.kernelId = kernelId;
        r.buildContextId = BUILD_CONTEXT_ID;
        r.hwTag = HW_TAG;
        r.inputShapes = inShapes;
        r.outputShape = outShape;
        r.inputStrides = inStrides;
        r.outputStrides = outStrides;
        r.inputDTypes = inDTypes;
        r.outputDType = outDType;
        r.inputConstants = inConstants;
        const auto &entry = KernelRegistry::get().getKernel(kernelId);
        HardwareBinding binding;
        if (TopologyMapper::resolve(entry.output_mem_space, entry.input_mem_spaces, entry.engines, entry.is_view,
                                    inShapes.size(), {}, {}, {}, true, true, true, binding))
        {
            r.output_mem_space = binding.output_mem_space;
            r.input_mem_spaces = binding.input_mem_spaces;
            r.engines = binding.engines;
        }
        else
        {
            // Preserve the old record format for kernels that cannot run on this host.
            r.output_mem_space = entry.output_mem_space;
            r.engines = entry.engines;
            r.input_mem_spaces.clear();
            for (size_t i = 0; i < inShapes.size(); ++i)
            {
                const size_t ruleIdx = std::min(i, entry.input_mem_spaces.empty() ? 0 : entry.input_mem_spaces.size() - 1);
                r.input_mem_spaces.push_back(ruleIdx < entry.input_mem_spaces.size()
                                                 ? entry.input_mem_spaces[ruleIdx]
                                                 : MemSpace{1, HandleType::CPP});
            }
        }
        r.runTime = 0.0f;

        std::string callStr = serializeToString(r);
        uint64_t callHash = std::hash<std::string>{}(callStr);

        std::lock_guard<std::mutex> lock(logMtx);
        if (loggedCalls.find(callHash) == loggedCalls.end())
        {
            loggedCalls.insert(callHash);
            if (callFile.is_open())
            {
                BinaryWriter bw(callFile);
                bw.write(r);
                callFile.flush();
            }
        }
    }

    std::vector<double> extractFeatures(const WorkloadMetrics &w, const std::vector<std::vector<uint32_t>> &inShapes,
                                        const std::vector<std::vector<uint64_t>> &inStrides,
                                        const std::vector<DType> &inDTypes, const std::vector<uint32_t> &outShape,
                                        const std::vector<uint64_t> &outStrides, const DType &outDType) const
    {
        std::vector<double> features;
        features.push_back(1.0); // Bias

        // 1. Compute & algorithmic intensity
        features.push_back(std::log(std::max(1.0, w.flops)));

        // 2. Global output complexity
        double outElements = static_cast<double>(countElements(outShape));
        double outInnerContig = getInnerContigElements(outShape, outStrides);
        double outUniqueElems = getUniqueElements(outShape, outStrides);
        uint32_t outEffRank = getEffectiveRank(outShape);

        features.push_back(std::log(std::max(1.0, outElements)));
        features.push_back(std::log(std::max(1.0, outInnerContig)));
        features.push_back(std::log(std::max(1.0, outUniqueElems)));
        features.push_back(std::log(std::max(1.0, static_cast<double>(outEffRank))));

        // 3. Per-input layout and access geometry
        for (uint64_t i = 0; i < inShapes.size(); ++i)
        {
            double elements = static_cast<double>(countElements(inShapes[i]));
            double innerContig = getInnerContigElements(inShapes[i], inStrides[i]);
            double uniqueElems = getUniqueElements(inShapes[i], inStrides[i]);
            uint32_t effRank = getEffectiveRank(inShapes[i]);

            bool isInnerZero = !inStrides[i].empty() && inStrides[i].back() == 0 && inShapes[i].back() > 1;

            features.push_back(std::log(std::max(1.0, elements)));
            features.push_back(std::log(std::max(1.0, innerContig)));
            features.push_back(std::log(std::max(1.0, uniqueElems)));
            features.push_back(std::log(std::max(1.0, static_cast<double>(effRank))));
            features.push_back(isInnerZero ? 1.0 : 0.0);
        }

        return features;
    }

    void fitModel(const ModelKey &mk, const std::vector<Record> &recs)
    {
        LinearModel model;
        if (KernelRegistry::get().hasKernel(mk.kernelId))
        {
            const auto &entry = KernelRegistry::get().getKernel(mk.kernelId);
            model.opType = entry.opType;
            model.opName = entry.opName;
            model.refFactory = entry.refFactory;
        }
        if (!model.refFactory && !model.opName.empty())
        {
            const auto *entry = ReferenceGraphRegistry::get().getFactory(model.opName);
            if (entry)
                model.refFactory = entry->factory;
        }

        model.records = recs;
        if (!recs.empty())
        {
            model.singleRecord = recs[0];
            model.hasSingleRecord = true;
        }

        models[mk] = model;
    }

    void load(std::string benchmarkPath)
    {
        records.clear();
        models.clear();
        std::ifstream file(benchmarkPath, std::ios::binary);
        if (!file.is_open())
            return;

        BinaryReader br(file);
        uint32_t total = 0, valid = 0;
        std::unordered_map<ModelKey, std::vector<Record>, ModelKeyHash> recordsByKey;

        while (file.peek() != EOF)
        {
            Record r;
            br.read(r);
            total++;
            if (r.hwTag != HW_TAG || r.buildContextId != BUILD_CONTEXT_ID ||
                !KernelRegistry::get().hasKernel(r.kernelId))
                continue;
            valid++;
            records[r.kernelId].push_back(r);

            ModelKey mk = {r.kernelId, r.inputShapes.size()};
            recordsByKey[mk].push_back(std::move(r));
        }

        std::cout << "Loaded " << valid << " valid records from " << benchmarkPath << std::endl;

        ProgressTimer timer2(recordsByKey.size(), "fitting interpolation models");
        for (const auto &kv : recordsByKey)
        {
            timer2.tick();
            fitModel(kv.first, kv.second);
        }
    }

    float estimateCost(KernelId kernelId, const std::vector<uint32_t> &outShape,
                       const std::vector<uint64_t> &outStrides, DType outDType,
                       const std::vector<std::vector<uint32_t>> &inShapes,
                       const std::vector<std::vector<uint64_t>> &inStrides, const std::vector<DType> &inDTypes,
                       const std::vector<std::vector<uint8_t>> &inConstants = {}, bool exactRecordOnly = false)
    {
        auto it = records.find(kernelId);
        if (it == records.end() || it->second.empty())
        {
            log_call(kernelId, outShape, outStrides, outDType, inShapes, inStrides, inDTypes, inConstants);

            if (!doneWarning.exchange(true, std::memory_order_relaxed))
            {
                std::cout << "\nWARNING INF COST ESTIMATION DUE TO MISSING RECORDS\n" << std::flush;
            }
            return std::numeric_limits<float>::infinity();
        }

        OpType opType = OpType::INPUT;
        std::string opName = "";
        ReferenceFactory ref_factory = nullptr;
        if (KernelRegistry::get().hasKernel(kernelId))
        {
            const auto &entry = KernelRegistry::get().getKernel(kernelId);
            opType = entry.opType;
            opName = entry.opName;
            ref_factory = entry.refFactory;
        }
        if (!ref_factory && !opName.empty())
        {
            const auto *ref_entry = ReferenceGraphRegistry::get().getFactory(opName);
            if (ref_entry)
                ref_factory = ref_entry->factory;
        }

        bool is_comm = isCommutativeOp(opType, opName);
        auto areInputConstantsMatching = [](const std::vector<std::vector<uint8_t>> &a,
                                            const std::vector<std::vector<uint8_t>> &b,
                                            size_t a_idx, size_t b_idx) -> bool {
            const auto &ca = (a_idx < a.size()) ? a[a_idx] : std::vector<uint8_t>{};
            const auto &cb = (b_idx < b.size()) ? b[b_idx] : std::vector<uint8_t>{};
            return ca == cb;
        };

        for (const auto &r : it->second)
        {
            if (r.outputShape != outShape || r.outputStrides != outStrides || r.outputDType != outDType)
                continue;

            bool match = false;
            if (r.inputShapes == inShapes && r.inputStrides == inStrides &&
                r.inputDTypes == inDTypes)
            {
                bool const_match = true;
                for (size_t i = 0; i < inShapes.size(); ++i)
                {
                    if (!areInputConstantsMatching(r.inputConstants, inConstants, i, i))
                    {
                        const_match = false;
                        break;
                    }
                }
                if (const_match)
                    match = true;
            }
            if (!match && is_comm && inShapes.size() == 2 && r.inputShapes.size() == 2)
            {
                if (r.inputShapes[0] == inShapes[1] && r.inputShapes[1] == inShapes[0] &&
                    r.inputStrides[0] == inStrides[1] && r.inputStrides[1] == inStrides[0] &&
                    r.inputDTypes[0] == inDTypes[1] && r.inputDTypes[1] == inDTypes[0] &&
                    areInputConstantsMatching(r.inputConstants, inConstants, 0, 1) &&
                    areInputConstantsMatching(r.inputConstants, inConstants, 1, 0))
                {
                    match = true;
                }
            }

            if (match)
            {
                return std::max(1e-6f, std::isnan(r.runTime) ? 1e-6f : r.runTime);
            }
        }

        if (enableLogging || exactRecordOnly)
        {
            log_call(kernelId, outShape, outStrides, outDType, inShapes, inStrides, inDTypes, inConstants);
        }

        if (exactRecordOnly)
        {
            return std::numeric_limits<float>::infinity();
        }

        WorkloadMetrics target_w;
        if (ref_factory)
        {
            target_w = computeWorkloadFromRefFactory(ref_factory, inShapes, inDTypes, outShape, outDType, inConstants);
        }
        else
        {
            target_w = computeWorkload(opType, inShapes, inDTypes, outShape, outDType, opName, inConstants);
        }

        ModelKey mk = {kernelId, inShapes.size()};
        auto model_it = models.find(mk);
        if (model_it != models.end())
        {
            auto features = extractFeatures(target_w, inShapes, inStrides, inDTypes, outShape, outStrides, outDType);
            float p = model_it->second.predict(features, target_w, inShapes, inStrides, inDTypes, outShape, outStrides,
                                               outDType);
            return std::isnan(p) ? 1e-6f : p;
        }

        LinearModel fallback_model;
        fallback_model.records = it->second;
        fallback_model.singleRecord = it->second[0];
        fallback_model.hasSingleRecord = true;
        fallback_model.opType = opType;
        fallback_model.opName = opName;
        fallback_model.refFactory = ref_factory;
        auto features = extractFeatures(target_w, inShapes, inStrides, inDTypes, outShape, outStrides, outDType);
        float p =
            fallback_model.predict(features, target_w, inShapes, inStrides, inDTypes, outShape, outStrides, outDType);
        return std::isnan(p) ? 1e-6f : p;
    }
};
