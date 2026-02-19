from tensor_graphs.backend.registry import KernelRegistry


def check():
    kernels = KernelRegistry.get_all_kernels()
    for op_type, backends in kernels.items():
        print(f"Op: {op_type}")
        for backend, entries in backends.items():
            print(f"  Backend: {backend}")
            for num, entry in enumerate(entries):
                # entry is (backend, sigs, target_dtype, func)
                print(f"    {num} Sigs: {entry[1]}")
                print(f"    {num} Target DType: {entry[2]}")
                print(f"    {num} Inplace: {entry[3]}")


if __name__ == "__main__":
    check()
