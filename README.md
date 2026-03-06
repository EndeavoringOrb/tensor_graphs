# tensor_graphs

computes DAGs with caching on subsequent runs

Arm64 Native Tools Command Prompt for VS 2022
cl.exe /Zi /EHsc /Fe: tensor_graphs_cpp/main.exe tensor_graphs_cpp/main.cpp /Itensor_graphs_cpp /std:c++17

1. uncomment TENSOR_GRAPHS_LOG_COST_CALLS
2. run main.exe
3. run bench.exe (make sure bench.cpp is compiled with #include "kernels/*" in the exact same order as main.cpp because it uses that order to index kernels. yea I know that's a terrible way to do it, but for now it works)
4. comment out TENSOR_GRAPHS_LOG_COST_CALLS
5. delete the cache (dirty_region_caches/gemma-3-270m-cpp.jsonl)
6. run main.exe (this time it will use the benchmark runtimes to plan)

## compile flags
- `TENSOR_GRAPHS_LOG_COST_CALLS` if set, CostModel will ops that were encountered during planning but have no benchmark records
- `DEBUG` if set, enables debug stuff