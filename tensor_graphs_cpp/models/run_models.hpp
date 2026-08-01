// tensor_graphs_cpp/models/run_models.hpp
#pragma once
#include "core/graph.hpp"
#include "core/memory.hpp"
#include "models/gemma-3-270m.hpp"
#include "models/qwen-3.6-35b-a3b.hpp"

struct ModelGraphRoots
{
    std::vector<LogicalId> roots;
    std::vector<LogicalId> inputs;
};

inline ModelGraphRoots build_gemma_graph(Graph &g, MemoryManager &mem, const std::string &model_path,
                                         uint32_t max_seq_len)
{
    Gemma3ModelConfig cfg;
    LogicalId inputIdsId = g.input({1, max_seq_len}, DType::INT32);
    Gemma3Model gemma(cfg, max_seq_len, g, mem, model_path);
    return {{gemma.build_graph(inputIdsId)}, {inputIdsId}};
}

inline ModelGraphRoots build_qwen_graph(Graph &g, MemoryManager &mem, const std::string &model_path,
                                        uint32_t max_seq_len)
{
    Qwen3_6_35B_A3B_Config cfg;
    LogicalId inputIdsId = g.input({1, max_seq_len}, DType::INT32);
    Qwen3_6_35B_A3B_Model qwen(cfg, max_seq_len, g, mem, model_path);
    return {{qwen.build_graph(inputIdsId)}, {inputIdsId}};
}