// TODO: test all fused kernels against their reference graph made up of atomic kernels
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

// Fills an array with random uniform values from [-1.0, 1.0]
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

    // ---------------------------------------------------------
    // Add_3D_1D Fused
    // ---------------------------------------------------------
    {
        Graph g_ref, g_tgt;
        MemoryManager mem_ref, mem_tgt;
        mem_ref.buffers.emplace(Backend::CPU, DeviceBuffer(1024 * 1024 * 16));
        mem_tgt.buffers.emplace(Backend::CPU, DeviceBuffer(1024 * 1024 * 16));

        std::vector<float> d3D(24);
        fillRandom(d3D.data(), 24);
        std::vector<float> d1D(4);
        fillRandom(d1D.data(), 4);

        auto setup = [&](Graph &g, MemoryManager &mem) -> std::pair<uint32_t, std::unordered_map<uint32_t, const void *>>
        {
            uint32_t id3D = g.allocateId();
            std::vector<uint32_t> shape3D = {2, 3, 4};
            mem.allocate(Backend::CPU, id3D, getSizeBytes(shape3D, DType::FLOAT32), StorageType::TRANSIENT);
            TensorView v3D = mem.getView(Backend::CPU, id3D, shape3D, DType::FLOAT32);
            g.inputWithId(id3D, v3D.shape, v3D.dtype, v3D, StorageType::TRANSIENT);

            uint32_t id1D = g.allocateId();
            std::vector<uint32_t> shape1D = {4};
            mem.allocate(Backend::CPU, id1D, getSizeBytes(shape1D, DType::FLOAT32), StorageType::TRANSIENT);
            TensorView v1D = mem.getView(Backend::CPU, id1D, shape1D, DType::FLOAT32);
            g.inputWithId(id1D, v1D.shape, v1D.dtype, v1D, StorageType::TRANSIENT);

            uint32_t outId = ReferenceGraphRegistry::get().getFactory("Add_3D_1D")->factory({id3D, id1D}, g);
            return {outId, {{id3D, d3D.data()}, {id1D, d1D.data()}}};
        };

        auto ref = setup(g_ref, mem_ref);
        auto tgt = setup(g_tgt, mem_tgt);
        executeAndCompare("Add_3D_1D", g_ref, mem_ref, ref.second, ref.first, g_tgt, mem_tgt, tgt.second, tgt.first);
    }

    // ---------------------------------------------------------
    // Tanh Fused
    // ---------------------------------------------------------
    {
        Graph g_ref, g_tgt;
        MemoryManager mem_ref, mem_tgt;
        mem_ref.buffers.emplace(Backend::CPU, DeviceBuffer(1024 * 1024 * 16));
        mem_tgt.buffers.emplace(Backend::CPU, DeviceBuffer(1024 * 1024 * 16));

        std::vector<float> d1D(10);
        fillRandom(d1D.data(), 10);

        auto setup = [&](Graph &g, MemoryManager &mem) -> std::pair<uint32_t, std::unordered_map<uint32_t, const void *>>
        {
            uint32_t id1D = g.allocateId();
            std::vector<uint32_t> shape1D = {10};
            mem.allocate(Backend::CPU, id1D, getSizeBytes(shape1D, DType::FLOAT32), StorageType::TRANSIENT);
            TensorView v1D = mem.getView(Backend::CPU, id1D, shape1D, DType::FLOAT32);
            g.inputWithId(id1D, v1D.shape, v1D.dtype, v1D, StorageType::TRANSIENT);

            uint32_t outId = ReferenceGraphRegistry::get().getFactory("Tanh")->factory({id1D}, g);
            return {outId, {{id1D, d1D.data()}}};
        };

        auto ref = setup(g_ref, mem_ref);
        auto tgt = setup(g_tgt, mem_tgt);
        executeAndCompare("Tanh", g_ref, mem_ref, ref.second, ref.first, g_tgt, mem_tgt, tgt.second, tgt.first);
    }

    std::cout << "\n----------------------" << std::endl;
    std::cout << "Tests Passed: " << passed << "/" << total << std::endl;
    std::cout << "----------------------" << std::endl;

    return 0;
}