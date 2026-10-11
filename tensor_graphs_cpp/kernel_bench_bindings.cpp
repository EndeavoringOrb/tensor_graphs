// Small native interface: no model runners or full-model compilation.
#include "generated/kernels_all.gen.hpp"
#include "bindings/kernel_bench.hpp"

PYBIND11_MODULE(kernel_bench_native, module)
{
    KernelBench::registerBindings(module);
}
