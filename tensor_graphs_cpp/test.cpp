#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <cstring>
#include <cassert>

#include "core/types.hpp"
#include "core/memory.hpp"
#include "core/graph.hpp"
#include "core/session.hpp"
#include "core/kernels.hpp"

#include "generated/kernels_all.gen.hpp"
#include "generated/build_context.gen.hpp"

// Fills an array with random uniform values from[-1.0, 1.0]
void fillRandom(float *ptr, size_t elements)
{
    static std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < elements; ++i)
    {
        ptr[i] = dist(gen);
    }
}

// Compares floating point arrays with an epsilon tolerance
bool compareOutputs(const float *a, const float *b, size_t elements, float eps = 1e-4f)
{
    for (size_t i = 0; i < elements; ++i)
    {
        if (std::abs(a[i] - b[i]) > eps)
        {
            std::cout << "\nMismatch at index " << i << ": " << a[i] << " != " << b[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main()
{
    std::cout << "Running Non-Reference Kernel Tests..." << std::endl;

    int passed = 0;
    int total = 0;

    auto executeAndCompare = [&](const std::string &name,
                                 Graph &g_ref, MemoryManager &mem_ref, const std::unordered_map<uint32_t, const void *> &in_ref, uint32_t out_ref,
                                 Graph &g_tgt, MemoryManager &mem_tgt, const std::unordered_map<uint32_t, const void *> &in_tgt, uint32_t out_tgt)
    {
        total++;
        std::cout << "Testing " << name << " ... ";

        // 1. Force the planner to ONLY use reference kernels for baseline evaluation
        KernelRegistry::get().setReferenceOnly(true);
        Session sess_ref(g_ref, mem_ref, out_ref, "");
        sess_ref.run(in_ref);
        const float *res_ref = static_cast<const float *>(sess_ref.getOutput(out_ref));
        uint64_t elements = countElements(g_ref.nodes[out_ref].shape);
        std::vector<float> ref_copy(res_ref, res_ref + elements);

        // 2. Allow the planner to use general/fused optimizations
        KernelRegistry::get().setReferenceOnly(false);
        Session sess_tgt(g_tgt, mem_tgt, out_tgt, "");
        sess_tgt.run(in_tgt);
        const float *res_tgt = static_cast<const float *>(sess_tgt.getOutput(out_tgt));

        // 3. Verify
        if (compareOutputs(ref_copy.data(), res_tgt, elements))
        {
            std::cout << "OK" << std::endl;
            passed++;
        }
        else
        {
            std::cout << "FAILED" << std::endl;
        }
    };

    // Automatically test all fused kernels against their reference graph made up of atomic kernels
    const auto &kernels = KernelRegistry::get().getAllKernels();
    for (const auto &kernel : kernels)
    {
        // Skip reference implementations and non-fused (atomic) kernels since they don't have a refFactory test pattern
        if (kernel.isReference || kernel.opType != OpType::FUSED)
        {
            continue;
        }

        if (kernel.dummyShapes.size() != kernel.numInputs)
        {
            std::cout << "Skipping " << kernel.opName << " due to missing dummy shapes in registry." << std::endl;
            continue;
        }

        Graph g_ref, g_tgt;
        MemoryManager mem_ref, mem_tgt;
        mem_ref.buffers.emplace(Backend::CPU, DeviceBuffer(1024 * 1024 * 16));
        mem_tgt.buffers.emplace(Backend::CPU, DeviceBuffer(1024 * 1024 * 16));

        // Generate data based on the sizes declared in the `REGISTER_FUSED_KERNEL` macro.
        std::vector<std::vector<float>> inputData(kernel.dummyShapes.size());
        for (size_t i = 0; i < kernel.dummyShapes.size(); ++i)
        {
            uint64_t elements = countElements(kernel.dummyShapes[i]);
            if (elements == 0)
                elements = 1;
            inputData[i].resize(elements);
            fillRandom(inputData[i].data(), elements);
        }

        auto setup = [&](Graph &g, MemoryManager &mem) -> std::pair<uint32_t, std::unordered_map<uint32_t, const void *>>
        {
            std::vector<uint32_t> inputIds;
            std::unordered_map<uint32_t, const void *> in_map;

            for (size_t i = 0; i < kernel.dummyShapes.size(); ++i)
            {
                uint32_t id = g.allocateId();
                mem.allocate(Backend::CPU, id, getSizeBytes(kernel.dummyShapes[i], DType::FLOAT32), StorageType::TRANSIENT);
                TensorView v = mem.getView(Backend::CPU, id, kernel.dummyShapes[i], DType::FLOAT32);
                g.inputWithId(id, v.shape, v.dtype, v, StorageType::TRANSIENT);
                inputIds.push_back(id);
                in_map[id] = inputData[i].data();
            }

            // Execute the factory matching pattern
            uint32_t outId = kernel.refFactory(inputIds, g);
            return {outId, in_map};
        };

        auto ref = setup(g_ref, mem_ref);
        auto tgt = setup(g_tgt, mem_tgt);

        std::string testName = kernel.opName + (kernel.inplace ? " (Inplace)" : "");
        executeAndCompare(testName, g_ref, mem_ref, ref.second, ref.first, g_tgt, mem_tgt, tgt.second, tgt.first);
    }

    std::cout << "\n----------------------" << std::endl;
    std::cout << "Tests Passed: " << passed << "/" << total << std::endl;
    std::cout << "----------------------" << std::endl;

    return 0;
}