#pragma once

#include "core/kernels.hpp"

inline void testStorageOutputMatching()
{
    const MemSpace ram{1, HandleType::CPP};
    const MemSpace storage{0, HandleType::STORAGE};
    KernelEntry kernel{};
    kernel.min_num_inputs = 1;
    kernel.max_num_inputs = 1;
    kernel.output_mem_space = ram;
    kernel.input_mem_spaces = {ram};
    TensorNode node;
    node.dtype = DType::FLOAT32;
    node.setShape({4});
    node.strides = {1};

    if (!kernel.matches({node}, node, ram, {ram}))
        Error::throw_err("[Regression Test Failed] Writable output rejected");
    if (kernel.matches({node}, node, storage, {ram}))
        Error::throw_err("[Regression Test Failed] Compute kernel accepted a storage output");
    // Fusion probes output backends while ignoring input placement.
    if (kernel.matches({node}, node, storage, {}, {}, false, true, true, true))
        Error::throw_err("[Regression Test Failed] Fusion accepted a storage output");
    if (!kernel.matches({node}, node, storage, {ram}, {}, true))
        Error::throw_err("[Regression Test Failed] Unconstrained output lookup rejected");

    kernel.input_mem_spaces = {storage};
    if (!kernel.matches({node}, node, ram, {storage}))
        Error::throw_err("[Regression Test Failed] Reading storage into RAM rejected");
    kernel.output_mem_space = storage;
    if (kernel.matches({node}, node, ram, {storage}, {}, true))
        Error::throw_err("[Regression Test Failed] Executable storage writer accepted");

    kernel.is_view = true;
    if (!kernel.matches({node}, node, storage, {storage}, System::get().getAvailableEngines()))
        Error::throw_err("[Regression Test Failed] Storage metadata view rejected");
}
