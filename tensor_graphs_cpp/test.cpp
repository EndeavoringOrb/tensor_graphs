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
                                 Graph &g_ref, const std::unordered_map<uint32_t, const void *> &in_ref, uint32_t out_ref,
                                 Graph &g_tgt, const std::unordered_map<uint32_t, const void *> &in_tgt, uint32_t out_tgt)
    {
        total++;
        std::cout << "Testing " << name << " ... ";

        // Provision distinct 16MB pools per execution
        MemoryManager mem_ref;
        mem_ref.buffers.emplace(Backend::CPU, DeviceBuffer(1024 * 1024 * 16));
        MemoryManager mem_tgt;
        mem_tgt.buffers.emplace(Backend::CPU, DeviceBuffer(1024 * 1024 * 16));

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
    // TEST 1: Add_3D_1D Fused (Broadcasting logic)
    // ---------------------------------------------------------
    {
        Graph g_ref, g_tgt;
        std::vector<float> d3D(24);
        fillRandom(d3D.data(), 24);
        std::vector<float> d1D(4);
        fillRandom(d1D.data(), 4);

        auto setup = [&](Graph &g) -> std::pair<uint32_t, std::unordered_map<uint32_t, const void *>>
        {
            uint32_t id3D = g.allocateId();
            TensorView v3D;
            v3D.shape = {2, 3, 4};
            v3D.strides = TensorView::calcContiguousStrides(v3D.shape);
            v3D.dtype = DType::FLOAT32;
            g.inputWithId(id3D, v3D.shape, v3D.dtype, v3D, StorageType::TRANSIENT);

            uint32_t id1D = g.allocateId();
            TensorView v1D;
            v1D.shape = {4};
            v1D.strides = TensorView::calcContiguousStrides(v1D.shape);
            v1D.dtype = DType::FLOAT32;
            g.inputWithId(id1D, v1D.shape, v1D.dtype, v1D, StorageType::TRANSIENT);

            // Ref factory inherently builds the pure reference representation via atomic operations (Repeat/Reshape -> Add)
            // But the FUSED target kernel will automatically snap to this exact pattern when planned!
            uint32_t outId = ReferenceGraphRegistry::get().getFactory("Add_3D_1D")->factory({id3D, id1D}, g);
            return {outId, {{id3D, d3D.data()}, {id1D, d1D.data()}}};
        };

        auto ref = setup(g_ref);
        auto tgt = setup(g_tgt);
        executeAndCompare("Add_3D_1D", g_ref, ref.second, ref.first, g_tgt, tgt.second, tgt.first);
    }

    // ---------------------------------------------------------
    // TEST 2: Tanh Fused
    // ---------------------------------------------------------
    {
        Graph g_ref, g_tgt;
        std::vector<float> d1D(10);
        fillRandom(d1D.data(), 10);

        auto setup = [&](Graph &g) -> std::pair<uint32_t, std::unordered_map<uint32_t, const void *>>
        {
            uint32_t id1D = g.allocateId();
            TensorView v1D;
            v1D.shape = {10};
            v1D.strides = TensorView::calcContiguousStrides(v1D.shape);
            v1D.dtype = DType::FLOAT32;
            g.inputWithId(id1D, v1D.shape, v1D.dtype, v1D, StorageType::TRANSIENT);

            uint32_t outId = ReferenceGraphRegistry::get().getFactory("Tanh")->factory({id1D}, g);
            return {outId, {{id1D, d1D.data()}}};
        };

        auto ref = setup(g_ref);
        auto tgt = setup(g_tgt);
        executeAndCompare("Tanh", g_ref, ref.second, ref.first, g_tgt, tgt.second, tgt.first);
    }

    // ---------------------------------------------------------
    // TEST 3: DOT Non-Contiguous
    // ---------------------------------------------------------
    {
        Graph g_ref, g_tgt;
        std::vector<float> baseA(24);
        fillRandom(baseA.data(), 24);
        std::vector<float> dB(40);
        fillRandom(dB.data(), 40);

        // For the reference graph to execute without error (as atomic reference DOT only accepts contiguous inputs),
        // we copy the simulated "strided" data layout exactly into a contiguous physical flat array.
        std::vector<float> dA_contig(24);
        for (int b = 0; b < 2; ++b)
            for (int m = 0; m < 3; ++m)
                for (int k = 0; k < 4; ++k)
                    dA_contig[b * 12 + m * 4 + k] = baseA[b * 12 + m * 1 + k * 3]; // Mappings strictly reflect a {12, 1, 3} stride.

        auto setupRef = [&]() -> std::pair<uint32_t, std::unordered_map<uint32_t, const void *>>
        {
            uint32_t idA = g_ref.allocateId();
            TensorView vA;
            vA.shape = {2, 3, 4};
            vA.strides = TensorView::calcContiguousStrides(vA.shape);
            vA.dtype = DType::FLOAT32;
            g_ref.inputWithId(idA, vA.shape, vA.dtype, vA, StorageType::TRANSIENT);

            uint32_t idB = g_ref.allocateId();
            TensorView vB;
            vB.shape = {2, 4, 5};
            vB.strides = TensorView::calcContiguousStrides(vB.shape);
            vB.dtype = DType::FLOAT32;
            g_ref.inputWithId(idB, vB.shape, vB.dtype, vB, StorageType::TRANSIENT);

            uint32_t outId = g_ref.dot(idA, idB);
            return {outId, {{idA, dA_contig.data()}, {idB, dB.data()}}};
        };

        auto setupTgt = [&]() -> std::pair<uint32_t, std::unordered_map<uint32_t, const void *>>
        {
            uint32_t idA = g_tgt.allocateId();
            TensorView vA;
            vA.shape = {2, 3, 4};
            vA.strides = {12, 1, 3};
            vA.dtype = DType::FLOAT32;
            g_tgt.inputWithId(idA, vA.shape, vA.dtype, vA, StorageType::TRANSIENT);

            uint32_t idB = g_tgt.allocateId();
            TensorView vB;
            vB.shape = {2, 4, 5};
            vB.strides = TensorView::calcContiguousStrides(vB.shape);
            vB.dtype = DType::FLOAT32;
            g_tgt.inputWithId(idB, vB.shape, vB.dtype, vB, StorageType::TRANSIENT);

            uint32_t outId = g_tgt.dot(idA, idB);
            // Non-contiguous kernel runs directly over the misaligned flat 'baseA' layout utilizing the overridden View Strides!
            return {outId, {{idA, baseA.data()}, {idB, dB.data()}}};
        };

        auto ref = setupRef();
        auto tgt = setupTgt();
        executeAndCompare("DOT Non-Contiguous", g_ref, ref.second, ref.first, g_tgt, tgt.second, tgt.first);
    }

    // ---------------------------------------------------------
    // TEST 4: REPEAT Inplace
    // ---------------------------------------------------------
    {
        Graph g_ref, g_tgt;
        std::vector<float> dA(6);
        fillRandom(dA.data(), 6);

        auto setup = [&](Graph &g) -> std::pair<uint32_t, std::unordered_map<uint32_t, const void *>>
        {
            uint32_t idA = g.allocateId();
            TensorView vA;
            vA.shape = {2, 3};
            vA.strides = TensorView::calcContiguousStrides(vA.shape);
            vA.dtype = DType::FLOAT32;
            g.inputWithId(idA, vA.shape, vA.dtype, vA, StorageType::TRANSIENT);

            int32_t repeats = 1;
            uint32_t repId = g.constant({1}, &repeats, DType::INT32);
            int32_t axis = 0;
            uint32_t axId = g.constant({1}, &axis, DType::INT32);

            uint32_t outId = g.repeat(idA, repId, axId);
            return {outId, {{idA, dA.data()}}};
        };

        auto ref = setup(g_ref);
        auto tgt = setup(g_tgt);
        executeAndCompare("REPEAT Inplace", g_ref, ref.second, ref.first, g_tgt, tgt.second, tgt.first);
    }

    // ---------------------------------------------------------
    // TEST 5: RESHAPE Inplace
    // ---------------------------------------------------------
    {
        Graph g_ref, g_tgt;
        std::vector<float> dA(6);
        fillRandom(dA.data(), 6);

        auto setup = [&](Graph &g) -> std::pair<uint32_t, std::unordered_map<uint32_t, const void *>>
        {
            uint32_t idA = g.allocateId();
            TensorView vA;
            vA.shape = {2, 3};
            vA.strides = TensorView::calcContiguousStrides(vA.shape);
            vA.dtype = DType::FLOAT32;
            g.inputWithId(idA, vA.shape, vA.dtype, vA, StorageType::TRANSIENT);

            int32_t shape_data[] = {3, 2};
            uint32_t shapeId = g.constant({2}, shape_data, DType::INT32);

            uint32_t outId = g.reshape(idA, shapeId);
            return {outId, {{idA, dA.data()}}};
        };

        auto ref = setup(g_ref);
        auto tgt = setup(g_tgt);
        executeAndCompare("RESHAPE Inplace", g_ref, ref.second, ref.first, g_tgt, tgt.second, tgt.first);
    }

    // ---------------------------------------------------------
    // TEST 6: ADD
    // ---------------------------------------------------------
    {
        Graph g_ref, g_tgt;
        std::vector<float> dA(6);
        fillRandom(dA.data(), 6);
        std::vector<float> dB(6);
        fillRandom(dB.data(), 6);

        auto setup = [&](Graph &g) -> std::pair<uint32_t, std::unordered_map<uint32_t, const void *>>
        {
            uint32_t idA = g.allocateId();
            TensorView vA;
            vA.shape = {2, 3};
            vA.strides = TensorView::calcContiguousStrides(vA.shape);
            vA.dtype = DType::FLOAT32;
            g.inputWithId(idA, vA.shape, vA.dtype, vA, StorageType::TRANSIENT);

            uint32_t idB = g.allocateId();
            TensorView vB;
            vB.shape = {2, 3};
            vB.strides = TensorView::calcContiguousStrides(vB.shape);
            vB.dtype = DType::FLOAT32;
            g.inputWithId(idB, vB.shape, vB.dtype, vB, StorageType::TRANSIENT);

            uint32_t outId = g.add(idA, idB);
            return {outId, {{idA, dA.data()}, {idB, dB.data()}}};
        };

        auto ref = setup(g_ref);
        auto tgt = setup(g_tgt);
        executeAndCompare("ADD", g_ref, ref.second, ref.first, g_tgt, tgt.second, tgt.first);
    }

    // ---------------------------------------------------------
    // TEST 7: MUL
    // ---------------------------------------------------------
    {
        Graph g_ref, g_tgt;
        std::vector<float> dA(6);
        fillRandom(dA.data(), 6);
        std::vector<float> dB(6);
        fillRandom(dB.data(), 6);

        auto setup = [&](Graph &g) -> std::pair<uint32_t, std::unordered_map<uint32_t, const void *>>
        {
            uint32_t idA = g.allocateId();
            TensorView vA;
            vA.shape = {2, 3};
            vA.strides = TensorView::calcContiguousStrides(vA.shape);
            vA.dtype = DType::FLOAT32;
            g.inputWithId(idA, vA.shape, vA.dtype, vA, StorageType::TRANSIENT);

            uint32_t idB = g.allocateId();
            TensorView vB;
            vB.shape = {2, 3};
            vB.strides = TensorView::calcContiguousStrides(vB.shape);
            vB.dtype = DType::FLOAT32;
            g.inputWithId(idB, vB.shape, vB.dtype, vB, StorageType::TRANSIENT);

            uint32_t outId = g.mul(idA, idB);
            return {outId, {{idA, dA.data()}, {idB, dB.data()}}};
        };

        auto ref = setup(g_ref);
        auto tgt = setup(g_tgt);
        executeAndCompare("MUL", g_ref, ref.second, ref.first, g_tgt, tgt.second, tgt.first);
    }

    // ---------------------------------------------------------
    // TEST 8: DIVIDE
    // ---------------------------------------------------------
    {
        Graph g_ref, g_tgt;
        std::vector<float> dA(6);
        fillRandom(dA.data(), 6);
        std::vector<float> dB(6);
        fillRandom(dB.data(), 6);

        auto setup = [&](Graph &g) -> std::pair<uint32_t, std::unordered_map<uint32_t, const void *>>
        {
            uint32_t idA = g.allocateId();
            TensorView vA;
            vA.shape = {2, 3};
            vA.strides = TensorView::calcContiguousStrides(vA.shape);
            vA.dtype = DType::FLOAT32;
            g.inputWithId(idA, vA.shape, vA.dtype, vA, StorageType::TRANSIENT);

            uint32_t idB = g.allocateId();
            TensorView vB;
            vB.shape = {2, 3};
            vB.strides = TensorView::calcContiguousStrides(vB.shape);
            vB.dtype = DType::FLOAT32;
            g.inputWithId(idB, vB.shape, vB.dtype, vB, StorageType::TRANSIENT);

            uint32_t outId = g.div(idA, idB);
            return {outId, {{idA, dA.data()}, {idB, dB.data()}}};
        };

        auto ref = setup(g_ref);
        auto tgt = setup(g_tgt);
        executeAndCompare("DIVIDE", g_ref, ref.second, ref.first, g_tgt, tgt.second, tgt.first);
    }

    // ---------------------------------------------------------
    // TEST 9: NEGATE
    // ---------------------------------------------------------
    {
        Graph g_ref, g_tgt;
        std::vector<float> dA(6);
        fillRandom(dA.data(), 6);

        auto setup = [&](Graph &g) -> std::pair<uint32_t, std::unordered_map<uint32_t, const void *>>
        {
            uint32_t idA = g.allocateId();
            TensorView vA;
            vA.shape = {2, 3};
            vA.strides = TensorView::calcContiguousStrides(vA.shape);
            vA.dtype = DType::FLOAT32;
            g.inputWithId(idA, vA.shape, vA.dtype, vA, StorageType::TRANSIENT);

            uint32_t outId = g.neg(idA);
            return {outId, {{idA, dA.data()}}};
        };

        auto ref = setup(g_ref);
        auto tgt = setup(g_tgt);
        executeAndCompare("NEGATE", g_ref, ref.second, ref.first, g_tgt, tgt.second, tgt.first);
    }

    // ---------------------------------------------------------
    // TEST 10: SIN
    // ---------------------------------------------------------
    {
        Graph g_ref, g_tgt;
        std::vector<float> dA(6);
        fillRandom(dA.data(), 6);

        auto setup = [&](Graph &g) -> std::pair<uint32_t, std::unordered_map<uint32_t, const void *>>
        {
            uint32_t idA = g.allocateId();
            TensorView vA;
            vA.shape = {2, 3};
            vA.strides = TensorView::calcContiguousStrides(vA.shape);
            vA.dtype = DType::FLOAT32;
            g.inputWithId(idA, vA.shape, vA.dtype, vA, StorageType::TRANSIENT);

            uint32_t outId = g.sin(idA);
            return {outId, {{idA, dA.data()}}};
        };

        auto ref = setup(g_ref);
        auto tgt = setup(g_tgt);
        executeAndCompare("SIN", g_ref, ref.second, ref.first, g_tgt, tgt.second, tgt.first);
    }

    // ---------------------------------------------------------
    // TEST 11: COS
    // ---------------------------------------------------------
    {
        Graph g_ref, g_tgt;
        std::vector<float> dA(6);
        fillRandom(dA.data(), 6);

        auto setup = [&](Graph &g) -> std::pair<uint32_t, std::unordered_map<uint32_t, const void *>>
        {
            uint32_t idA = g.allocateId();
            TensorView vA;
            vA.shape = {2, 3};
            vA.strides = TensorView::calcContiguousStrides(vA.shape);
            vA.dtype = DType::FLOAT32;
            g.inputWithId(idA, vA.shape, vA.dtype, vA, StorageType::TRANSIENT);

            uint32_t outId = g.cos(idA);
            return {outId, {{idA, dA.data()}}};
        };

        auto ref = setup(g_ref);
        auto tgt = setup(g_tgt);
        executeAndCompare("COS", g_ref, ref.second, ref.first, g_tgt, tgt.second, tgt.first);
    }

    std::cout << "\n----------------------" << std::endl;
    std::cout << "Tests Passed: " << passed << "/" << total << std::endl;
    std::cout << "----------------------" << std::endl;

    return 0;
}