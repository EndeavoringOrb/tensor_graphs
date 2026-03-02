#pragma once
#include "core/types.hpp"
#include "core/graph.hpp"
#include "core/memory.hpp"
#include "core/cost_model.hpp"
#include "core/planner.hpp"
#include "core/executor.hpp"
#include <unordered_map>
#include <memory>

class Session
{
private:
    Graph &graph;
    MemoryManager &memManager;
    CostModel costModel;
    CompiledGraph compiled;
    std::unique_ptr<Executor> executor;
    uint32_t rootId;
    bool isCompiled;

public:
    Session(Graph &g, MemoryManager &mem, uint32_t root)
        : graph(g), memManager(mem), rootId(root), isCompiled(false) {}

    void compile()
    {
        Planner planner(costModel, 4ULL * 1024 * 1024 * 1024); // TODO: integrate MemoryManager and planner so planner knows how much mem it has while planning

        compiled = planner.plan(rootId, graph);
        executor = std::make_unique<Executor>(compiled, memManager, graph); // TODO: don't use unique_ptr, just initialize Executor without args at first
        isCompiled = true;
    }

    void run(const std::unordered_map<uint32_t, const void *> &inputs)
    {
        if (!isCompiled)
        {
            compile();
        }
        executor->run(inputs);
    }

    const void *getOutput(uint32_t nodeId) const
    {
        const TensorNode &node = graph.nodes[nodeId];
        auto &buf = memManager.buffers.at(node.backend);
        auto it = buf.allocationMap.find(nodeId);

        // Verify node exists inside the allocator map post-execution
        if (it == buf.allocationMap.end())
            return nullptr;

        uint64_t offset = it->second->offset;
        uint64_t baseOffset = node.view.shape.empty() ? 0 : node.view.baseOffset;

        return buf.arena.data() + offset + baseOffset;
    }

    const void *getRootOutput() const
    {
        return getOutput(rootId);
    }
};