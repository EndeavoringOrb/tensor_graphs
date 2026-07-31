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

inline ModelGraphRoots build_gemma_graph(Graph &g, MemoryManager &mem, const std::string &model_path)
{
    Gemma3ModelConfig cfg;
    uint32_t maxSeqLen = 8;
    LogicalId inputIdsId = g.input({1, maxSeqLen}, DType::INT32);
    Gemma3Model gemma(cfg, maxSeqLen, g, mem, model_path);
    return {{gemma.build_graph(inputIdsId)}, {inputIdsId}};
}

inline ModelGraphRoots build_qwen_graph(Graph &g, MemoryManager &mem, const std::string &model_path)
{
    Qwen3_6_35B_A3B_Config cfg;
    uint32_t maxSeqLen = 8;
    LogicalId inputIdsId = g.input({1, maxSeqLen}, DType::INT32);
    Qwen3_6_35B_A3B_Model qwen(cfg, maxSeqLen, g, mem, model_path);
    return {{qwen.build_graph(inputIdsId)}, {inputIdsId}};
}