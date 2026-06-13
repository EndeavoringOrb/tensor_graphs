// File: tensor_graphs_cpp/core/cost_model.hpp
#pragma once
#include "core/types.hpp"
#include "core/graph.hpp"
#include "core/kernels.hpp"
#include "core/misc.hpp"
#include "generated/build_context.gen.hpp"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <iostream>
#include <cmath>
#include <limits>
#include <filesystem>
#include <mutex>
#include <algorithm>

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

// Uncomment the following line to enable logging calls to `benchmarks/calls.bin`
#define TENSOR_GRAPHS_LOG_COST_CALLS

struct Record
{
    uint64_t kernelUid;
    uint64_t buildContextId;
    std::string hwTag;

    std::vector<std::vector<uint32_t>> inputShapes;
    std::vector<std::vector<uint32_t>> outputShapes;
    std::vector<std::vector<uint64_t>> inputStrides;
    std::vector<std::vector<uint64_t>> outputStrides;
    std::vector<DType> inputDTypes;
    std::vector<DType> outputDTypes;
    std::vector<std::vector<uint8_t>> inputConstants;
    std::vector<Backend> backends;
    std::vector<std::vector<Backend>> inputBackends;
    float runTime;
};

inline void tg_serialize(BinaryWriter &bw, const Record &val)
{
    bw.write(val.kernelUid);
    bw.write(val.buildContextId);
    bw.write(val.hwTag);
    bw.write(val.inputShapes);
    bw.write(val.outputShapes);
    bw.write(val.inputStrides);
    bw.write(val.outputStrides);
    bw.write(val.inputDTypes);
    bw.write(val.outputDTypes);
    bw.write(val.inputConstants);
    bw.write(val.backends);
    bw.write(val.inputBackends);
    bw.write(val.runTime);
}

inline void tg_deserialize(BinaryReader &br, Record &val)
{
    br.read(val.kernelUid);
    br.read(val.buildContextId);
    br.read(val.hwTag);
    br.read(val.inputShapes);
    br.read(val.outputShapes);
    br.read(val.inputStrides);
    br.read(val.outputStrides);
    br.read(val.inputDTypes);
    br.read(val.outputDTypes);
    br.read(val.inputConstants);
    br.read(val.backends);
    br.read(val.inputBackends);
    br.read(val.runTime);
}

struct CostModel
{
    struct ModelKey
    {
        uint64_t kernelUid;
        size_t numInputs;
        bool operator==(const ModelKey &o) const
        {
            return kernelUid == o.kernelUid && numInputs == o.numInputs;
        }
    };

    struct ModelKeyHash
    {
        size_t operator()(const ModelKey &k) const
        {
            return std::hash<uint64_t>()(k.kernelUid) ^ (std::hash<size_t>()(k.numInputs) << 1);
        }
    };

    struct Matrix
    {
        int rows, cols;
        std::vector<double> data;
        Matrix(int r, int c) : rows(r), cols(c), data(r * c, 0.0) {}
        double &operator()(int r, int c) { return data[r * cols + c]; }
        double operator()(int r, int c) const { return data[r * cols + c]; }
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
                for (size_t i = 0; i < weights.size(); ++i)
                {
                    double val = features[i];
                    if (scale[i] > 0)
                        val /= scale[i];
                    y += weights[i] * val;
                }
                return static_cast<float>(std::max(0.0, y));
            }
            // Linear Fallback
            if (fallbackElements > 0)
                return static_cast<float>(fallbackTime * (static_cast<double>(targetElements) / fallbackElements));
            return 0.0f;
        }
    };

    std::unordered_map<uint64_t, std::vector<Record>> records;
    std::unordered_map<ModelKey, LinearModel, ModelKeyHash> models;
    std::unordered_set<size_t> loggedCalls;
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

    std::vector<double> extractFeatures(
        const std::vector<std::vector<uint32_t>> &inShapes,
        const std::vector<std::vector<uint64_t>> &inStrides,
        const std::vector<DType> &inDTypes,
        const std::vector<std::vector<uint32_t>> &outShapes,
        const std::vector<std::vector<uint64_t>> &outStrides,
        const std::vector<DType> &outDTypes) const
    {
        std::vector<double> features;
        features.push_back(1.0); // Bias

        double outElements = 0.0;
        for (const auto &s : outShapes)
            outElements += static_cast<double>(countElements(s));

        double inElements = 0.0;
        for (const auto &s : inShapes)
            inElements += static_cast<double>(countElements(s));

        features.push_back(outElements);
        features.push_back(inElements);

        // Expand per input details
        for (size_t i = 0; i < inShapes.size(); ++i)
        {
            double elements = static_cast<double>(countElements(inShapes[i]));
            double bytes = elements * getDTypeSize(inDTypes[i]);
            bool contig = isContiguous(inStrides[i], inShapes[i]);
            features.push_back(bytes);
            features.push_back(contig ? 1.0 : 0.0);
        }

        // Expand per output details
        for (size_t i = 0; i < outShapes.size(); ++i)
        {
            double elements = static_cast<double>(countElements(outShapes[i]));
            double bytes = elements * getDTypeSize(outDTypes[i]);
            bool contig = isContiguous(outStrides[i], outShapes[i]);
            features.push_back(bytes);
            features.push_back(contig ? 1.0 : 0.0);
        }

        return features;
    }

    void fitModel(const ModelKey &mk, const std::vector<Record> &recs)
    {
        LinearModel model;

        if (!recs.empty())
        {
            model.fallbackTime = recs[0].runTime;
            uint64_t e = 0;
            for (const auto &s : recs[0].outputShapes)
                e += countElements(s);
            model.fallbackElements = e > 0 ? static_cast<double>(e) : 1.0;
        }

        if (recs.size() < 2)
        {
            models[mk] = model;
            return;
        }

        int K = static_cast<int>(recs.size());
        auto sample_feat = extractFeatures(
            recs[0].inputShapes, recs[0].inputStrides, recs[0].inputDTypes,
            recs[0].outputShapes, recs[0].outputStrides, recs[0].outputDTypes);
        int D = static_cast<int>(sample_feat.size());

        Matrix X(K, D);
        Matrix Y(K, 1);

        model.scale.assign(D, 1.0);

        for (int i = 0; i < K; ++i)
        {
            auto feat = extractFeatures(
                recs[i].inputShapes, recs[i].inputStrides, recs[i].inputDTypes,
                recs[i].outputShapes, recs[i].outputStrides, recs[i].outputDTypes);
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
        ProgressTimer timer(0, "loading records ");
        std::unordered_map<ModelKey, std::vector<Record>, ModelKeyHash> recordsByKey;

        while (file.peek() != EOF)
        {
            timer.tick();
            Record r;
            br.read(r);
            total++;
            if (r.hwTag != HW_TAG || r.buildContextId != BUILD_CONTEXT_ID || !KernelRegistry::get().hasKernel(r.kernelUid))
                continue;
            valid++;
            records[r.kernelUid].push_back(r);

            ModelKey mk = {r.kernelUid, r.inputShapes.size()};
            recordsByKey[mk].push_back(std::move(r));
        }

        std::cout << "Loaded " << valid << " valid records from " << benchmarkPath << std::endl;

        ProgressTimer timer2(recordsByKey.size(), "fitting interpolation models ");
        for (const auto &kv : recordsByKey)
        {
            timer2.tick();
            fitModel(kv.first, kv.second);
        }
    }

    float estimateCost(
        uint64_t kernelUid,
        const std::vector<uint32_t> &outShape,
        const std::vector<uint64_t> &_outStrides,
        DType outDType,
        const std::vector<std::vector<uint32_t>> &inShapes,
        const std::vector<std::vector<uint64_t>> &inStrides,
        const std::vector<DType> &inDTypes,
        const std::vector<std::vector<uint8_t>> &inConstants)
    {
        std::vector<std::vector<uint32_t>> outShapes = {outShape};
        std::vector<DType> outDTypes = {outDType};
        const std::vector<std::vector<uint64_t>> outStrides = {_outStrides};

        auto it = records.find(kernelUid);
        if (it == records.end() || it->second.empty())
        {
#ifdef TENSOR_GRAPHS_LOG_COST_CALLS
            {
                Record r;
                r.kernelUid = kernelUid;
                r.buildContextId = BUILD_CONTEXT_ID;
                r.hwTag = HW_TAG;
                r.inputShapes = inShapes;
                r.outputShapes = outShapes;
                r.inputStrides = inStrides;
                r.outputStrides = outStrides;
                r.inputDTypes = inDTypes;
                r.outputDTypes = outDTypes;
                r.inputConstants = inConstants;
                const auto &entry = KernelRegistry::get().getKernel(kernelUid);
                r.backends = entry.backends;
                r.inputBackends = entry.inputBackends;
                r.runTime = 0.0f;
                std::string callStr = serializeToString(r);
                size_t callHash = std::hash<std::string>{}(callStr);

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
#endif
            if (!doneWarning)
            {
                std::cout << "\nWARNING INF COST ESTIMATION DUE TO MISSING RECORDS\n"
                          << std::flush;
                doneWarning = true;
            }
            return std::numeric_limits<float>::infinity();
        }

        // Exact match short-circuit
        for (const auto &r : it->second)
        {
            if (r.inputShapes == inShapes && r.outputShapes == outShapes &&
                r.inputStrides == inStrides && r.outputStrides == outStrides &&
                r.inputDTypes == inDTypes && r.outputDTypes == outDTypes &&
                r.inputConstants == inConstants)
            {
                return r.runTime;
            }
        }

#ifdef TENSOR_GRAPHS_LOG_COST_CALLS
        {
            Record r;
            r.kernelUid = kernelUid;
            r.buildContextId = BUILD_CONTEXT_ID;
            r.hwTag = HW_TAG;
            r.inputShapes = inShapes;
            r.outputShapes = outShapes;
            r.inputStrides = inStrides;
            r.outputStrides = outStrides;
            r.inputDTypes = inDTypes;
            r.outputDTypes = outDTypes;
            r.inputConstants = inConstants;
            const auto &entry = KernelRegistry::get().getKernel(kernelUid);
            r.backends = entry.backends;
            r.inputBackends = entry.inputBackends;
            r.runTime = 0.0f;

            std::string callStr = serializeToString(r);
            size_t callHash = std::hash<std::string>{}(callStr);

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
#endif

        ModelKey mk = {kernelUid, inShapes.size()};
        auto modelIt = models.find(mk);
        if (modelIt != models.end())
        {
            auto features = extractFeatures(inShapes, inStrides, inDTypes, outShapes, outStrides, outDTypes);
            uint64_t targetElements = countElements(outShape);
            return modelIt->second.predict(features, targetElements);
        }

        return std::numeric_limits<float>::infinity();
    }
};