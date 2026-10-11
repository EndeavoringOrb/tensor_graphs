#include <filesystem>
#include "generated/kernels_all.gen.hpp"
#include "core/common/kernel_bench.hpp"

inline LogicalId refFactoryKernelBenchFma(const std::vector<LogicalId> &inputs, Graph &graph)
{
    return graph.add(graph.mul(inputs.at(0), inputs.at(1)), inputs.at(2));
}

inline bool matchKernelBenchFma(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    return output.dtype == DType::FLOAT32 && inputs.at(0).getShape() == output.getShape() &&
           inputs.at(1).getShape() == output.getShape() && inputs.at(2).getShape() == output.getShape();
}

inline void runKernelBenchFma(const KernelContext &context)
{
    auto a = static_cast<const float *>(context.inputs.at(0));
    auto b = static_cast<const float *>(context.inputs.at(1));
    auto c = static_cast<const float *>(context.inputs.at(2));
    auto output = static_cast<float *>(context.outputs.at(0));
    for (uint64_t i = 0; i < countElements(context.outViews.at(0).getShape()); ++i)
        output[i] = a[i] * b[i] + c[i];
}

inline void runKernelBenchWrongFma(const KernelContext &context)
{
    auto output = static_cast<float *>(context.outputs.at(0));
    for (uint64_t i = 0; i < countElements(context.outViews.at(0).getShape()); ++i)
        output[i] = 0.0f;
}

int main(int argc, char **argv)
{
    System::get();
    KernelId fma_id{0x123456789abcdefULL};
    KernelRegistry::get().registerKernel(fma_id, OpType::FUSED, "KernelBench_Test_FMA", 3, 3,
        matchKernelBenchFma, runKernelBenchFma, refFactoryKernelBenchFma, {}, false, false, nullptr,
        MemSpace{1, HandleType::CPP}, {Engine{0, EngineType::CPU}},
        {DType::FLOAT32, DType::FLOAT32, DType::FLOAT32}, {{8}, {8}, {8}}, {true, true, true},
        {{1, HandleType::CPP}, {1, HandleType::CPP}, {1, HandleType::CPP}});
    KernelId restricted_id{0x123456789abcdf0ULL};
    KernelRegistry::get().registerKernel(restricted_id, OpType::FUSED, "KernelBench_Test_Restricted_FMA", 3, 3,
        [](const std::vector<TensorNode> &, const TensorNode &) { return false; },
        runKernelBenchFma, refFactoryKernelBenchFma, {}, false, false, nullptr,
        MemSpace{1, HandleType::CPP}, {Engine{0, EngineType::CPU}},
        {DType::FLOAT32, DType::FLOAT32, DType::FLOAT32}, {{8}, {8}, {8}}, {true, true, true},
        {{1, HandleType::CPP}, {1, HandleType::CPP}, {1, HandleType::CPP}});
    auto findReference = [](OpType op_type) {
        for (const auto &[uid, kernel] : KernelRegistry::get().getAllKernels())
            if (kernel.opType == op_type && kernel.isReference && kernel.dtypes.at(0) == DType::FLOAT32)
                return uid;
        Error::throw_err("Missing FLOAT32 reference kernel");
    };
    CompiledGraph compiled;
    for (uint32_t i = 1; i <= 5; ++i)
    {
        TensorView view;
        view.setShape({8});
        view.strides = {1};
        view.dtype = DType::FLOAT32;
        compiled.nodeViews[EClassId{i}] = view;
    }
    std::vector<float> constant_values(8, 1.0f);
    auto constant_bytes = std::make_shared<std::vector<uint8_t>>(constant_values.size() * sizeof(float));
    std::memcpy(constant_bytes->data(), constant_values.data(), constant_bytes->size());
    compiled.constantStaging[EClassId{3}] = constant_bytes;
    OpInstruction mul;
    mul.eclass_id = EClassId{4};
    mul.kernel_id = findReference(OpType::MUL);
    mul.children = {EClassId{1}, EClassId{2}};
    OpInstruction add;
    add.eclass_id = EClassId{5};
    add.kernel_id = findReference(OpType::ADD);
    add.children = {EClassId{4}, EClassId{3}};
    compiled.instructions = {mul, add};
    // A shared producer with two endpoints plus an independent fragment, as
    // produced when a compiled cache omits a view instruction at a boundary.
    compiled.nodeViews[EClassId{6}] = compiled.nodeViews.at(EClassId{5});
    OpInstruction branch = add;
    branch.eclass_id = EClassId{6};
    branch.children = {EClassId{4}, EClassId{1}};
    compiled.instructions.push_back(branch);
    for (uint32_t i = 7; i <= 9; ++i)
    {
        TensorView view;
        view.setShape({16});
        view.strides = {1};
        view.dtype = DType::FLOAT32;
        compiled.nodeViews[EClassId{i}] = view;
    }
    OpInstruction fragment = mul;
    fragment.eclass_id = EClassId{9};
    fragment.children = {EClassId{7}, EClassId{8}};
    compiled.instructions.push_back(fragment);
    auto cache_path = argc > 1 ? std::filesystem::path(argv[1])
                               : std::filesystem::temp_directory_path() / "kernel-bench-regression.bin";
    {
        std::ofstream file(cache_path, std::ios::binary);
        BinaryWriter writer(file);
        writer.write<uint8_t>(0);
        writer.write<uint32_t>(4);
        writer.write(LogicalId{5});
        writer.write(std::unordered_set<BaseEClassId>{});
        writer.write(std::vector<float>{1.0f});
        writer.write<uint8_t>(1);
        writer.write(compiled);
    }
    KernelBench::DiscoveryStats discovery_stats;
    auto calls = KernelBench::discoverCalls(cache_path.string(), &discovery_stats);
    if (discovery_stats.buckets != 1 || discovery_stats.saturation_passes != 1 || discovery_stats.fragment_roots != 3)
        Error::throw_err("Multi-fragment bucket was not saturated exactly once");
    bool found_fma = false;
    bool found_restricted_reference = false;
    bool found_independent_fragment = false;
    for (auto record : calls)
    {
        if (record.kernelId == mul.kernel_id && record.outputShape == std::vector<uint32_t>{16})
            found_independent_fragment = true;
        if (record.kernelId == restricted_id)
            found_restricted_reference = true;
        if (record.kernelId != fma_id)
            continue;
        const auto &kernel = KernelRegistry::get().getKernel(fma_id);
        if (!KernelBench::bindRecord(kernel, record) || !testKernelWithRecord(kernel, record, true))
            Error::throw_err("FMA configuration did not test correctly");
        auto wrong_kernel = kernel;
        wrong_kernel.run = runKernelBenchWrongFma;
        if (testKernelWithRecord(wrong_kernel, record, true))
            Error::throw_err("Incorrect FMA implementation passed correctness verification");
        auto samples = KernelBench::benchmarkRecord(kernel, record, 3);
        if (samples.size() != 3 || samples.at(0) <= 0)
            Error::throw_err("Missing timing samples");
        found_fma = true;
    }
    if (!found_fma)
        Error::throw_err("Unselected FMA was not discovered across selected MUL + ADD instructions");
    if (!found_restricted_reference)
        Error::throw_err("Reference expression rejected by existing kernel predicate was not discovered");
    if (!found_independent_fragment)
        Error::throw_err("Independent fragment's MUL configuration was not discovered");
    Record scalar_record;
    scalar_record.inputShapes = {{}, {}, {}};
    scalar_record.inputStrides = {{}, {}, {}};
    scalar_record.inputDTypes = {DType::FLOAT32, DType::FLOAT32, DType::FLOAT32};
    const auto &scalar_kernel = KernelRegistry::get().getKernel(fma_id);
    if (!KernelBench::bindRecord(scalar_kernel, scalar_record) ||
        !testKernelWithRecord(scalar_kernel, scalar_record, true))
        Error::throw_err("Scalar configuration failed correctness verification");
    float finite = 1.0f, nan = std::numeric_limits<float>::quiet_NaN();
    if (compareOutputs(&finite, &nan, 1))
        Error::throw_err("NaN output incorrectly passed correctness verification");
    if (argc <= 1)
        std::filesystem::remove(cache_path);
    std::cout << "KernelBench discovery, correctness, and benchmark regression passed\n";
}
