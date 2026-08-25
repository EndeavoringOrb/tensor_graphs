#pragma once

#include <iostream>
#include <vector>

#include "core/cost_model.hpp"
#include "core/egraph.hpp"
#include "core/graph.hpp"
#include "core/kernels.hpp"
#include "core/misc.hpp"
#include "core/plan/planner.hpp"
#include "core/plan/rule_registry.hpp"
#include "core/settings.hpp"
#include "core/types.hpp"

inline void setupTestSettings(Settings &settings, bool enable_rules = true)
{
    for (const auto &spec : kAllRuleSpecs)
    {
        settings.set_rule_enabled(spec.category, spec.rule_name, enable_rules);
    }
}

inline Region makeRegion(std::initializer_list<Dim> dims)
{
    Region r;
    r.region.assign(dims.begin(), dims.end());
    return r;
}

inline bool regionListEquals(const std::vector<Region> &actual, const std::vector<Region> &expected)
{
    const auto a = normalizeRegions(actual);
    const auto e = normalizeRegions(expected);
    if (a.size() != e.size())
        return false;
    for (uint64_t i = 0; i < a.size(); ++i)
    {
        if (!regionsMatch(a[i], e[i]))
            return false;
    }
    return true;
}

inline void assertRegionListEquals(const std::vector<Region> &actual, const std::vector<Region> &expected,
                                   const std::string &label)
{
    if (!regionListEquals(actual, expected))
    {
        std::stringstream ss;
        ss << "[RegionTest] " << label << " expected " << encodeRegionList(expected) << " but got "
           << encodeRegionList(actual);
        Error::throw_err(ss.str());
    }
}

inline bool compareOutputs(const float *ref, const float *test, uint64_t elements, float eps = 1e-4f)
{
    for (uint64_t i = 0; i < elements; ++i)
    {
        if (std::abs(ref[i] - test[i]) > eps)
        {
            std::cout << "\nMismatch at index " << i << ": (ref)" << ref[i] << " != (test)" << test[i] << std::endl;
            return false;
        }
    }
    return true;
}

inline bool compareOutputs(const int32_t *ref, const int32_t *test, uint64_t elements, float eps = 1e-4f)
{
    for (uint64_t i = 0; i < elements; ++i)
    {
        if (ref[i] != test[i])
        {
            std::cout << "\nMismatch at index " << i << ": (ref)" << ref[i] << " != (test)" << test[i] << std::endl;
            return false;
        }
    }
    return true;
}

inline bool compareOutputs(const bool *ref, const bool *test, uint64_t elements, float eps = 1e-4f)
{
    for (uint64_t i = 0; i < elements; ++i)
    {
        if (ref[i] != test[i])
        {
            std::cout << "\nMismatch at index " << i << ": (ref)" << ref[i] << " != (test)" << test[i] << std::endl;
            return false;
        }
    }
    return true;
}

inline void populateDummyRecords(CostModel &costModel, const EGraph &egraph, float defaultRuntime = 1.0f)
{
    for (const auto &enode : egraph.getENodes())
    {
        KernelId kid = enode.getKernelId();
        if (kid.value != 0 && kid.value != UINT32_MAX && costModel.records.find(kid) == costModel.records.end())
        {
            Record r;
            r.kernelId = kid;
            r.buildContextId = BUILD_CONTEXT_ID;
            r.hwTag = HW_TAG;
            r.outputShape = enode.getShape();
            r.outputStrides = enode.getStrides();
            r.outputDType = enode.getDType();
            for (EClassId child : enode.getChildren())
            {
                const auto &cCls = egraph.getEClass(egraph.findConst(child));
                r.inputShapes.push_back(cCls.shape);
                r.inputStrides.push_back(cCls.strides);
                r.inputDTypes.push_back(cCls.dtype);
            }
            r.runTime = defaultRuntime;
            costModel.records[kid].push_back(r);
        }
    }

    for (const auto &pair : KernelRegistry::get().getAllKernels())
    {
        const auto &k = pair.second;
        if (k.uid.value != 0 && k.uid.value != UINT32_MAX && costModel.records.find(k.uid) == costModel.records.end())
        {
            Record r;
            r.kernelId = k.uid;
            r.buildContextId = BUILD_CONTEXT_ID;
            r.hwTag = HW_TAG;
            r.runTime = defaultRuntime;
            r.output_mem_space = k.output_mem_space;
            r.engines = k.engines;
            r.outputShape = k.dummyShapes.empty() ? std::vector<uint32_t>{8, 8} : k.dummyShapes[0];
            r.outputStrides = calcContiguousStrides(r.outputShape);
            r.outputDType = k.dtypes.empty() ? DType::FLOAT32 : k.dtypes[0];
            for (size_t i = 0; i < k.dummyShapes.size(); ++i)
            {
                r.inputShapes.push_back(k.dummyShapes[i]);
                r.inputStrides.push_back(calcContiguousStrides(k.dummyShapes[i]));
                r.inputDTypes.push_back(i < k.dtypes.size() ? k.dtypes[i] : DType::FLOAT32);
            }
            if (r.inputShapes.empty())
            {
                r.inputShapes = {{8, 8}};
                r.inputStrides = {{8, 1}};
                r.inputDTypes = {DType::FLOAT32};
            }
            costModel.records[k.uid].push_back(r);
        }
    }
}