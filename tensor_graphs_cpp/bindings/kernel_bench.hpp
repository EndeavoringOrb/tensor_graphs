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
    module.def("kernel_bench_write_records", [](const std::string &path, const py::list &measurements) {
        std::ofstream stream(path, std::ios::binary | std::ios::app);
        if (!stream)
            Error::throw_err("Could not open compiler records for writing: " + path);
        BinaryWriter writer(stream);
        size_t count = 0;
        for (auto item : measurements)
        {
            auto data = py::cast<py::dict>(item);
            const auto name = data["kernel_name"].cast<std::string>();
            const KernelEntry *matched = nullptr;
            for (const auto &[uid, kernel] : KernelRegistry::get().getAllKernels())
                if (kernel.getName() == name)
                {
                    if (matched)
                        Error::throw_err("Compiler record kernel name is ambiguous: " + name);
                    matched = &kernel;
                }
            if (!matched)
                Error::throw_err("Compiler record kernel is not registered: " + name);
            auto record = recordFromDict(data["configuration"].cast<py::dict>());
            if (!bindRecord(*matched, record))
                Error::throw_err("Compiler record configuration no longer matches: " + name);
            record.runTime = data["median_ms"].cast<float>();
            if (!std::isfinite(record.runTime) || record.runTime <= 0)
                Error::throw_err("Compiler record timing must be finite and positive");
            writer.write(record);
            ++count;
        }
        stream.flush();
        if (!stream)
            Error::throw_err("Could not persist compiler records: " + path);
        return count;
    }, py::arg("path"), py::arg("measurements"));
    module.def("kernel_bench_capabilities", []() {
        std::vector<std::string> capabilities;
#ifdef TG_USE_CUDA
        capabilities.push_back("TG_USE_CUDA");
#endif
#ifdef TG_USE_OPENCL
        capabilities.push_back("TG_USE_OPENCL");
#endif
#ifdef TG_HAS_AVX2
        capabilities.push_back("TG_HAS_AVX2");
#endif
#ifdef TG_HAS_NEON
        capabilities.push_back("TG_HAS_NEON");
#endif
        return capabilities;
    });
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
            std::vector<std::string> match_reasons;
            data["kernel_matches"] = bindRecord(kernel, bound_record, &match_reasons);
            data["match_reasons"] = match_reasons;
            result.append(data);
        }
        return result;
    }, py::arg("cache_path"));
    module.def("kernel_bench_match_diagnostics", [](uint64_t kernel_id, const py::dict &configuration) {
        auto record = recordFromDict(configuration);
        std::vector<std::string> reasons;
        bool matched = bindRecord(KernelRegistry::get().getKernel(KernelId{kernel_id}), record, &reasons);
        py::dict result;
        result["matched"] = matched;
        result["reasons"] = reasons;
        return result;
    }, py::arg("kernel_id"), py::arg("configuration"));
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
    module.def("kernel_bench_test_diagnostics", [](uint64_t kernel_id, const py::dict &configuration, uint32_t seed) {
        const auto &kernel = KernelRegistry::get().getKernel(KernelId{kernel_id});
        auto record = recordFromDict(configuration);
        if (!kernel.refFactory || !bindRecord(kernel, record))
            Error::throw_err("KernelBench requires a matching kernel with a reference factory");
        CorrectnessDiagnostic diagnostic;
        bool passed;
        {
            py::gil_scoped_release release;
            passed = testKernelWithRecord(kernel, record, true, &diagnostic, seed);
        }
        py::dict result;
        result["passed"] = passed;
        result["kind"] = diagnostic.kind;
        result["message"] = diagnostic.message;
        if (diagnostic.kind == "correctness_mismatch")
        {
            result["index"] = diagnostic.index;
            auto encode_value = [](float value) -> py::object {
                if (std::isfinite(value))
                    return py::float_(value);
                return py::str(std::isnan(value) ? "NaN" : value > 0 ? "Infinity" : "-Infinity");
            };
            result["reference"] = encode_value(diagnostic.reference);
            result["actual"] = encode_value(diagnostic.actual);
        }
        if (diagnostic.kind == "output_size_mismatch")
        {
            result["reference_size"] = diagnostic.reference_size;
            result["actual_size"] = diagnostic.actual_size;
        }
        py::dict tolerance;
        tolerance["absolute"] = diagnostic.epsilon;
        tolerance["rule"] = "equal or both NaN or finite absolute difference <= epsilon";
        result["tolerance"] = tolerance;
        py::dict reproduction;
        reproduction["seed"] = seed;
        reproduction["generator"] = "std::mt19937; reset per configuration; inputs in index order";
        reproduction["float_distribution"] = "std::uniform_real_distribution<float>(-1, 1); BF16 truncated from float32";
        reproduction["int32_distribution"] = "std::uniform_int_distribution<int32_t>(1, 10); CONCAT axis input overridden from shapes";
        reproduction["bool_distribution"] = "std::uniform_int_distribution<int>(0, 1)";
        reproduction["byte_distribution"] = "std::uniform_int_distribution<int>(1, 127)";
        reproduction["constants"] = "input_constants replace generated values; stride scattering follows configuration";
        reproduction["configuration"] = configuration;
        result["reproduction"] = reproduction;
        return result;
    }, py::arg("kernel_id"), py::arg("configuration"), py::arg("seed") = 42);
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
