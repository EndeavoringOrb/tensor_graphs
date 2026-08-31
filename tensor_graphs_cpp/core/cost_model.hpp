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
        Record singleRecord;
        bool hasSingleRecord = false;
        OpType opType = OpType::INPUT;
        std::string opName = "";

        float predict(const std::vector<double> &features, const WorkloadMetrics &targetW,
                      const std::vector<std::vector<uint32_t>> &inShapes,
                      const std::vector<std::vector<uint64_t>> &inStrides, const std::vector<DType> &inDTypes,
                      const std::vector<uint32_t> &outShape, const std::vector<uint64_t> &outStrides,
                      DType outDType) const
        {
            if (valid && weights.size() == features.size())
            {
                double log_y = 0.0;
                for (size_t i = 0; i < weights.size(); ++i)
                {
                    double val = features[i];
                    if (scale[i] > 0.0)
                        val /= scale[i];
                    log_y += weights[i] * val;
                }
                log_y = std::clamp(log_y, -13.8, 20.0);
                double y = std::exp(log_y);
                if (std::isnan(y) || std::isinf(y)) return 1e-6f;
                return static_cast<float>(std::max(1e-6, y));
            }

            if (hasSingleRecord)
            {
                WorkloadMetrics refW = computeWorkload(opType, singleRecord.inputShapes, singleRecord.inputDTypes,
                                                       singleRecord.outputShape, singleRecord.outputDType, opName);
                double refTime = std::max(1e-6, static_cast<double>(singleRecord.runTime));
                double ratio = 1.0;

                bool isDot = (opType == OpType::DOT) || (opName.find("Dot") != std::string::npos) ||
                             (opName.find("dot") != std::string::npos) || (opName.find("linear") != std::string::npos);

                if (isDot)
                {
                    if (refW.flops > 0.0 && targetW.flops > 0.0)
                    {
                        ratio = targetW.flops / refW.flops;
                    }
                    else
                    {
                        double refBytes = refW.bytesRead + refW.bytesWritten;
                        double targetBytes = targetW.bytesRead + targetW.bytesWritten;
                        ratio = (refBytes > 0.0) ? (targetBytes / refBytes) : 1.0;
                    }
                }
                else if (opType == OpType::SUM || opType == OpType::MAX || opType == OpType::ARGMAX)
                {
                    double refInElems =
                        singleRecord.inputShapes.empty() ? 1.0 : countElements(singleRecord.inputShapes[0]);
                    double targetInElems = inShapes.empty() ? 1.0 : countElements(inShapes[0]);
                    ratio = (refInElems > 0.0) ? (targetInElems / refInElems) : 1.0;
                }
                else
                {
                    double refBytes = refW.bytesRead + refW.bytesWritten;
                    double targetBytes = targetW.bytesRead + targetW.bytesWritten;
                    if (refBytes > 0.0 && targetBytes > 0.0)
                    {
                        ratio = targetBytes / refBytes;
                    }
                    else
                    {
                        double refOut = countElements(singleRecord.outputShape);
                        double targetOut = countElements(outShape);
                        ratio = (refOut > 0.0) ? (targetOut / refOut) : 1.0;
                    }

                    // Rank & indexing arithmetic penalty adjustment
                    double refEffRank = getEffectiveRank(singleRecord.outputShape);
                    double tgtEffRank = getEffectiveRank(outShape);
                    if (refEffRank > 0 && tgtEffRank > 0 && refEffRank != tgtEffRank)
                    {
                        ratio *= (0.4 + 0.6 * (tgtEffRank / refEffRank));
                    }

                    // Stride penalty adjustment for copy/elementwise kernels
                    if (!inShapes.empty() && !singleRecord.inputShapes.empty() && !inStrides.empty() &&
                        !singleRecord.inputStrides.empty())
                    {
                        double refInnerContig =
                            getInnerContigElements(singleRecord.inputShapes[0], singleRecord.inputStrides[0]);
                        double tgtInnerContig = getInnerContigElements(inShapes[0], inStrides[0]);

                        bool refInnerZero = !singleRecord.inputStrides[0].empty() &&
                                            singleRecord.inputStrides[0].back() == 0 &&
                                            singleRecord.inputShapes[0].back() > 1;
                        bool tgtInnerZero = !inStrides[0].empty() && inStrides[0].back() == 0 && inShapes[0].back() > 1;

                        if (tgtInnerZero && !refInnerZero)
                        {
                            ratio *= std::max(2.0, std::log2(std::max(2.0, (double)inShapes[0].back())));
                        }
                        else if (!tgtInnerZero && refInnerZero)
                        {
                            ratio /=
                                std::max(2.0, std::log2(std::max(2.0, (double)singleRecord.inputShapes[0].back())));
                        }
                        else if (refInnerContig > 1.0 && tgtInnerContig <= 1.0)
                        {
                            ratio *= 4.0;
                        }
                        else if (refInnerContig <= 1.0 && tgtInnerContig > 1.0)
                        {
                            ratio /= 4.0;
                        }
                    }
                }

                double y = refTime * ratio;
                if (std::isnan(y) || std::isinf(y)) return 1e-6f;
                return static_cast<float>(std::max(1e-6, y));
            }

            return 1e-6f;
        }
    };

    std::unordered_map<KernelId, std::vector<Record>> records;
    std::unordered_map<ModelKey, LinearModel, ModelKeyHash> models;
    std::unordered_set<uint64_t> loggedCalls;
    std::ofstream callFile;
    std::mutex logMtx;
    bool doneWarning = false;
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
        r.output_mem_space = entry.output_mem_space;
        r.engines = entry.engines;
        r.input_mem_spaces.clear();
        for (size_t i = 0; i < inShapes.size(); ++i)
        {
            size_t ruleIdx = std::min(i, entry.input_mem_spaces.empty() ? 0 : entry.input_mem_spaces.size() - 1);
            if (ruleIdx < entry.input_mem_spaces.size())
            {
                r.input_mem_spaces.push_back(entry.input_mem_spaces[ruleIdx]);
            }
            else
            {
                r.input_mem_spaces.push_back(MemSpace{1, HandleType::CPP});
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
        }

        if (!recs.empty())
        {
            model.singleRecord = recs[0];
            model.hasSingleRecord = true;
        }

        if (recs.size() < 2)
        {
            models[mk] = model;
            return;
        }

        int K = static_cast<int>(recs.size());
        auto sample_w = computeWorkload(model.opType, recs[0].inputShapes, recs[0].inputDTypes, recs[0].outputShape,
                                        recs[0].outputDType, model.opName);
        auto sample_feat = extractFeatures(sample_w, recs[0].inputShapes, recs[0].inputStrides, recs[0].inputDTypes,
                                           recs[0].outputShape, recs[0].outputStrides, recs[0].outputDType);
        int D = static_cast<int>(sample_feat.size());

        Matrix X(K, D);
        Matrix Y(K, 1);

        model.scale.assign(D, 1.0);

        for (int i = 0; i < K; ++i)
        {
            auto w = computeWorkload(model.opType, recs[i].inputShapes, recs[i].inputDTypes, recs[i].outputShape,
                                     recs[i].outputDType, model.opName);
            auto feat = extractFeatures(w, recs[i].inputShapes, recs[i].inputStrides, recs[i].inputDTypes,
                                        recs[i].outputShape, recs[i].outputStrides, recs[i].outputDType);
            for (int j = 0; j < D && j < static_cast<int>(feat.size()); ++j)
            {
                X(i, j) = feat[j];
                model.scale[j] = std::max(model.scale[j], std::abs(feat[j]));
            }
            double target_time = std::max(1e-6, static_cast<double>(recs[i].runTime));
            Y(i, 0) = std::log(target_time);
        }

        for (int i = 0; i < K; ++i)
        {
            for (int j = 0; j < D; ++j)
            {
                if (model.scale[j] > 0.0)
                {
                    X(i, j) /= model.scale[j];
                }
            }
        }

        Matrix Xt = transpose(X);
        Matrix XtX = multiply(Xt, X);
        Matrix XtY = multiply(Xt, Y);

        double lambda = 1e-2;
        for (int i = 0; i < D; ++i)
        {
            XtX(i, i) += lambda;
        }

        if (invert(XtX))
        {
            Matrix W = multiply(XtX, XtY);
            model.weights.resize(D);
            model.valid = true;
            for (int i = 0; i < D; ++i)
            {
                if (std::isnan(W(i, 0)) || std::isinf(W(i, 0))) {
                    model.valid = false;
                    break;
                }
                model.weights[i] = W(i, 0);
            }
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
                       const std::vector<std::vector<uint8_t>> &inConstants, bool exactRecordOnly = false)
    {
        auto it = records.find(kernelId);
        if (it == records.end() || it->second.empty())
        {
            log_call(kernelId, outShape, outStrides, outDType, inShapes, inStrides, inDTypes, inConstants);

            if (!doneWarning)
            {
                std::cout << "\nWARNING INF COST ESTIMATION DUE TO MISSING RECORDS\n" << std::flush;
                doneWarning = true;
            }
            return std::numeric_limits<float>::infinity();
        }

        for (const auto &r : it->second)
        {
            if (r.inputShapes == inShapes && r.outputShape == outShape && r.inputStrides == inStrides &&
                r.outputStrides == outStrides && r.inputDTypes == inDTypes && r.outputDType == outDType &&
                r.inputConstants == inConstants)
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

        OpType opType = OpType::INPUT;
        std::string opName = "";
        if (KernelRegistry::get().hasKernel(kernelId))
        {
            const auto &entry = KernelRegistry::get().getKernel(kernelId);
            opType = entry.opType;
            opName = entry.opName;
        }

        WorkloadMetrics targetW = computeWorkload(opType, inShapes, inDTypes, outShape, outDType, opName);

        ModelKey mk = {kernelId, inShapes.size()};
        auto modelIt = models.find(mk);
        if (modelIt != models.end())
        {
            auto features = extractFeatures(targetW, inShapes, inStrides, inDTypes, outShape, outStrides, outDType);
            float p = modelIt->second.predict(features, targetW, inShapes, inStrides, inDTypes, outShape, outStrides, outDType);
            return std::isnan(p) ? 1e-6f : p;
        }

        LinearModel fallbackModel;
        fallbackModel.singleRecord = it->second[0];
        fallbackModel.hasSingleRecord = true;
        fallbackModel.opType = opType;
        fallbackModel.opName = opName;
        auto features = extractFeatures(targetW, inShapes, inStrides, inDTypes, outShape, outStrides, outDType);
        float p = fallbackModel.predict(features, targetW, inShapes, inStrides, inDTypes, outShape, outStrides, outDType);
        return std::isnan(p) ? 1e-6f : p;
    }
};