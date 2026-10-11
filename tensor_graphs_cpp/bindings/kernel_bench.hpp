#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "core/common/kernel_bench.hpp"

namespace KernelBench
{
namespace py = pybind11;

inline py::dict recordToDict(const Record &record)
{
    py::dict result;
    result["kernel_id"] = record.kernelId.value;
    result["build_context_id"] = record.buildContextId;
    result["hardware_tag"] = record.hwTag;
    result["input_shapes"] = record.inputShapes;
    result["input_strides"] = record.inputStrides;
    std::vector<uint32_t> dtypes;
    for (auto dtype : record.inputDTypes)
        dtypes.push_back(static_cast<uint32_t>(dtype));
    result["input_dtypes"] = dtypes;
    result["input_constants"] = record.inputConstants;
    result["output_shape"] = record.outputShape;
    result["output_strides"] = record.outputStrides;
    result["output_dtype"] = static_cast<uint32_t>(record.outputDType);
    return result;
}

inline Record recordFromDict(const py::dict &data)
{
    Record record;
    record.inputShapes = data["input_shapes"].cast<std::vector<std::vector<uint32_t>>>();
    record.inputStrides = data["input_strides"].cast<std::vector<std::vector<uint64_t>>>();
    for (auto dtype : data["input_dtypes"].cast<std::vector<uint32_t>>())
        record.inputDTypes.push_back(static_cast<DType>(dtype));
    record.inputConstants = data["input_constants"].cast<std::vector<std::vector<uint8_t>>>();
    record.outputShape = data["output_shape"].cast<std::vector<uint32_t>>();
    record.outputStrides = data["output_strides"].cast<std::vector<uint64_t>>();
    record.outputDType = static_cast<DType>(data["output_dtype"].cast<uint32_t>());
    return record;
}

inline py::dict kernelToDict(const KernelEntry &kernel)
{
    py::dict result;
    result["kernel_id"] = kernel.uid.value;
    result["name"] = kernel.getName();
    result["is_reference"] = kernel.isReference;
    result["is_view"] = kernel.is_view;
    result["reference_expression"] = referenceExpression(kernel);
    result["dummy_shapes"] = kernel.dummyShapes;
    result["has_reference_factory"] = kernel.refFactory != nullptr;
    return result;
}

inline void registerBindings(py::module_ &module)
{
    module.def("kernel_bench_kernels", []() {
        py::list result;
        for (const auto &[uid, kernel] : KernelRegistry::get().getAllKernels())
            result.append(kernelToDict(kernel));
        return result;
    });
    module.def("kernel_bench_discover", [](const std::string &cache_path) {
        std::vector<Record> records;
        {
            py::gil_scoped_release release;
            records = discoverCalls(cache_path);
        }
        py::list result;
        for (const auto &record : records)
        {
            auto data = recordToDict(record);
            const auto &kernel = KernelRegistry::get().getKernel(record.kernelId);
            data["kernel_name"] = kernel.getName();
            data["reference_expression"] = referenceExpression(kernel);
            auto bound_record = record;
            data["kernel_matches"] = bindRecord(kernel, bound_record);
            result.append(data);
        }
        return result;
    }, py::arg("cache_path"));
    module.def("kernel_bench_matches", [](uint64_t kernel_id, const py::dict &configuration) {
        auto record = recordFromDict(configuration);
        return bindRecord(KernelRegistry::get().getKernel(KernelId{kernel_id}), record);
    }, py::arg("kernel_id"), py::arg("configuration"));
    module.def("kernel_bench_test", [](uint64_t kernel_id, const py::dict &configuration) {
        const auto &kernel = KernelRegistry::get().getKernel(KernelId{kernel_id});
        auto record = recordFromDict(configuration);
        if (!kernel.refFactory)
            Error::throw_err("KernelBench correctness testing requires a reference factory");
        if (!bindRecord(kernel, record))
            Error::throw_err("Kernel does not match the requested configuration");
        py::gil_scoped_release release;
        if (!testKernelWithRecord(kernel, record, true))
            Error::throw_err("KernelBench correctness test failed for " + kernel.getName());
    }, py::arg("kernel_id"), py::arg("configuration"));
    module.def("kernel_bench_benchmark", [](uint64_t kernel_id, const py::dict &configuration, uint32_t iterations) {
        const auto &kernel = KernelRegistry::get().getKernel(KernelId{kernel_id});
        auto record = recordFromDict(configuration);
        if (!bindRecord(kernel, record))
            Error::throw_err("Kernel does not match the requested configuration");
        py::gil_scoped_release release;
        return benchmarkRecord(kernel, record, iterations);
    }, py::arg("kernel_id"), py::arg("configuration"), py::arg("iterations") = 8);
}
} // namespace KernelBench
