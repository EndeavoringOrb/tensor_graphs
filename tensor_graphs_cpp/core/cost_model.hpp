// File: tensor_graphs_cpp/core/cost_model.hpp
#pragma once
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/graph.hpp"
#include "core/kernels.hpp"
#include "core/misc.hpp"
#include "core/types.hpp"
#include "generated/build_context.gen.hpp"

// TODO: make hardware detection better
#if defined(USE_CUDA)
#define HW_TAG "CUDA_Enabled"
#else
// Determine OS String
#if defined(TG_OS_WINDOWS)
#define PLAT_OS_STR "Windows"
#elif defined(TG_OS_MACOS)
#define PLAT_OS_STR "macOS"
#elif defined(TG_OS_LINUX)
#define PLAT_OS_STR "Linux"
#else
#define PLAT_OS_STR "UnknownOS"
#endif

// Determine Arch String
#if defined(TG_ARCH_ARM64)
#define PLAT_ARCH_STR "ARM64"
#elif defined(TG_ARCH_X64)
#define PLAT_ARCH_STR "x64"
#else
#define PLAT_ARCH_STR "UnknownArch"
#endif

#define HW_TAG PLAT_OS_STR "_" PLAT_ARCH_STR
#endif

// Uncomment the following line to enable logging calls to
// `benchmarks/calls.bin`
#define TENSOR_GRAPHS_LOG_COST_CALLS

struct Record
{
    KernelId kernelId;
    uint64_t buildContextId;
    std::string hwTag;

    std::vector<std::vector<uint32_t>> inputShapes;
    std::vector<uint32_t> outputShape;
    std::vector<std::vector<uint64_t>> inputStrides;
    std::vector<uint64_t> outputStrides;
    std::vector<DType> inputDTypes;
    DType outputDType;
    std::vector<std::vector<uint8_t>> inputConstants;
    MemSpace output_mem_space;
    std::vector<Engine> engines;
    std::vector<MemSpace> input_mem_spaces;
    float runTime;
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
        double fallbackTime = 0.0;
        double fallbackElements = 1.0;

        float predict(const std::vector<double> &features, uint64_t targetElements) const
        {
            if (valid && weights.size() == features.size())
            {
                double y = 0.0;
                for (uint64_t i = 0; i < weights.size(); ++i)
                {
                    double val = features[i];
                    if (scale[i] > 0)
                        val /= scale[i];
                    y += weights[i] * val;
                }
                return static_cast<float>(std::max(1e-6, y));
            }
            // Linear Fallback
            if (fallbackElements > 0)
                return static_cast<float>(fallbackTime * (static_cast<double>(targetElements) / fallbackElements));
            return 1e-6f;
        }
    };

    std::unordered_map<KernelId, std::vector<Record>> records;
    std::unordered_map<ModelKey, LinearModel, ModelKeyHash> models;
    std::unordered_set<uint64_t> loggedCalls;
    std::ofstream callFile;
    std::mutex logMtx;
    bool doneWarning = false;

    CostModel()
    {
#ifdef TENSOR_GRAPHS_LOG_COST_CALLS
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
                    r.runTime = 0.0f; // normalize for hash
                    loggedCalls.insert(std::hash<std::string>{}(serializeToString(r)));
                }
            }
        }
        callFile.open(path, std::ios::app | std::ios::binary);
        if (!callFile.is_open())
            std::cerr << "Failed to open " << path << " for appending.\n";
#endif
    }

    void log_call(KernelId kernelId, const std::vector<uint32_t> &outShape, const std::vector<uint64_t> &outStrides,
                  DType outDType, const std::vector<std::vector<uint32_t>> &inShapes,
                  const std::vector<std::vector<uint64_t>> &inStrides, const std::vector<DType> &inDTypes,
                  const std::vector<std::vector<uint8_t>> &inConstants)
    {
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

    std::vector<double> extractFeatures(const std::vector<std::vector<uint32_t>> &inShapes,
                                        const std::vector<std::vector<uint64_t>> &inStrides,
                                        const std::vector<DType> &inDTypes, const std::vector<uint32_t> &outShape,
                                        const std::vector<uint64_t> &outStrides, const DType &outDType) const
    {
        std::vector<double> features;
        features.push_back(1.0); // Bias

        double outElements = static_cast<double>(countElements(outShape));

        double inElements = 0.0;
        for (const auto &s : inShapes)
            inElements += static_cast<double>(countElements(s));

        features.push_back(outElements);
        features.push_back(inElements);

        // Expand per input details
        for (uint64_t i = 0; i < inShapes.size(); ++i)
        {
            double elements = static_cast<double>(countElements(inShapes[i]));
            double bytes = elements * getDTypeSize(inDTypes[i]);
            bool contig = isContiguous(inStrides[i], inShapes[i]);
            features.push_back(bytes);
            features.push_back(contig ? 1.0 : 0.0);
        }

        double elements = static_cast<double>(countElements(outShape));
        double bytes = elements * getDTypeSize(outDType);
        bool contig = isContiguous(outStrides, outShape);
        features.push_back(bytes);
        features.push_back(contig ? 1.0 : 0.0);

        return features;
    }

    void fitModel(const ModelKey &mk, const std::vector<Record> &recs)
    {
        LinearModel model;

        if (!recs.empty())
        {
            model.fallbackTime = recs[0].runTime;
            uint64_t e = countElements(recs[0].outputShape);
            model.fallbackElements = e > 0 ? static_cast<double>(e) : 1.0;
        }

        if (recs.size() < 2)
        {
            models[mk] = model;
            return;
        }

        int K = static_cast<int>(recs.size());
        auto sample_feat = extractFeatures(recs[0].inputShapes, recs[0].inputStrides, recs[0].inputDTypes,
                                           recs[0].outputShape, recs[0].outputStrides, recs[0].outputDType);
        int D = static_cast<int>(sample_feat.size());

        Matrix X(K, D);
        Matrix Y(K, 1);

        model.scale.assign(D, 1.0);

        for (int i = 0; i < K; ++i)
        {
            auto feat = extractFeatures(recs[i].inputShapes, recs[i].inputStrides, recs[i].inputDTypes,
                                        recs[i].outputShape, recs[i].outputStrides, recs[i].outputDType);
            for (int j = 0; j < D && j < static_cast<int>(feat.size()); ++j)
            {
                X(i, j) = feat[j];
                model.scale[j] = std::max(model.scale[j], std::abs(feat[j]));
            }
            Y(i, 0) = recs[i].runTime;
        }

        // Apply Scaling to prevent double precision loss inside the square matrices
        for (int i = 0; i < K; ++i)
        {
            for (int j = 0; j < D; ++j)
            {
                if (model.scale[j] > 0)
                {
                    X(i, j) /= model.scale[j];
                }
            }
        }

        Matrix Xt = transpose(X);
        Matrix XtX = multiply(Xt, X);
        Matrix XtY = multiply(Xt, Y);

        // Standard Ridge regularization
        double lambda = 1e-2;
        for (int i = 0; i < D; ++i)
        {
            XtX(i, i) += lambda;
        }

        if (invert(XtX))
        {
            Matrix W = multiply(XtX, XtY);
            model.weights.resize(D);
            for (int i = 0; i < D; ++i)
            {
                model.weights[i] = W(i, 0);
            }
            model.valid = true;
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
        ProgressTimer timer(0, "loading records");
        std::unordered_map<ModelKey, std::vector<Record>, ModelKeyHash> recordsByKey;

        while (file.peek() != EOF)
        {
            timer.tick();
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
                       const std::vector<std::vector<uint8_t>> &inConstants)
    {
        auto it = records.find(kernelId);
        if (it == records.end() || it->second.empty())
        {
#ifdef TENSOR_GRAPHS_LOG_COST_CALLS
            log_call(kernelId, outShape, outStrides, outDType, inShapes, inStrides, inDTypes, inConstants);
#endif
            if (!doneWarning)
            {
                std::cout << "\nWARNING INF COST ESTIMATION DUE TO MISSING RECORDS\n" << std::flush;
                doneWarning = true;
            }
            return std::numeric_limits<float>::infinity();
        }

        // Exact match short-circuit
        for (const auto &r : it->second)
        {
            if (r.inputShapes == inShapes && r.outputShape == outShape && r.inputStrides == inStrides &&
                r.outputStrides == outStrides && r.inputDTypes == inDTypes && r.outputDType == outDType &&
                r.inputConstants == inConstants)
            {
                return std::max(1e-6f, r.runTime);
            }
        }

#ifdef TENSOR_GRAPHS_LOG_COST_CALLS
        log_call(kernelId, outShape, outStrides, outDType, inShapes, inStrides, inDTypes, inConstants);
#endif

        ModelKey mk = {kernelId, inShapes.size()};
        auto modelIt = models.find(mk);
        if (modelIt != models.end())
        {
            auto features = extractFeatures(inShapes, inStrides, inDTypes, outShape, outStrides, outDType);
            uint64_t targetElements = countElements(outShape);
            return modelIt->second.predict(features, targetElements);
        }

        return std::numeric_limits<float>::infinity();
    }
};