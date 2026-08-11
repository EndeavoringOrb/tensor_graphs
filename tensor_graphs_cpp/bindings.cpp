#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "core/graph.hpp"
#include "core/memory.hpp"
#include "core/plan/planner.hpp"
#include "core/plan/search_delegate.hpp"
#include "models/run_models.hpp"
#include "generated/kernels_all.gen.hpp"

namespace py = pybind11;

class PySearchDelegate : public SearchDelegate
{
  public:
    using SearchDelegate::SearchDelegate;

    void push_state() override
    {
        PYBIND11_OVERRIDE(void, SearchDelegate, push_state);
    }
    void pop_state() override
    {
        PYBIND11_OVERRIDE(void, SearchDelegate, pop_state);
    }
    std::vector<uint32_t> order_enodes(uint32_t eclass_id, const std::vector<ActionFeature> &enodes) override
    {
        PYBIND11_OVERRIDE(std::vector<uint32_t>, SearchDelegate, order_enodes, eclass_id, enodes);
    }
    std::vector<uint32_t> order_dispatch(const std::vector<ActionFeature> &ready_nodes) override
    {
        PYBIND11_OVERRIDE(std::vector<uint32_t>, SearchDelegate, order_dispatch, ready_nodes);
    }
    std::vector<uint32_t> order_malloc(const std::vector<ActionFeature> &avail_buffers) override
    {
        PYBIND11_OVERRIDE(std::vector<uint32_t>, SearchDelegate, order_malloc, avail_buffers);
    }
};

PYBIND11_MODULE(tensor_graphs, m)
{
    py::class_<ActionFeature>(m, "ActionFeature")
        .def(py::init<>())
        .def_readwrite("id", &ActionFeature::id)
        .def_readwrite("cost", &ActionFeature::cost)
        .def_readwrite("size", &ActionFeature::size)
        .def_readwrite("op_type", &ActionFeature::op_type);

    py::class_<SearchDelegate, PySearchDelegate, std::shared_ptr<SearchDelegate>>(m, "SearchDelegate")
        .def(py::init<>())
        .def("push_state", &SearchDelegate::push_state)
        .def("pop_state", &SearchDelegate::pop_state)
        .def("order_enodes", &SearchDelegate::order_enodes)
        .def("order_dispatch", &SearchDelegate::order_dispatch)
        .def("order_malloc", &SearchDelegate::order_malloc);

    m.def("plan_graph", [](const std::string &model_name, const std::string &model_path,
                           std::shared_ptr<SearchDelegate> delegate) {
        std::unordered_map<MemSpace, uint64_t> bufferSizes = {{MemSpace{1, HandleType::CPP}, 24ULL * 1024 * 1024 * 1024}};
        MemoryManager mem(bufferSizes);
        Graph g;

        uint32_t max_seq_len = 8;
        ModelGraphRoots roots;
        if (model_name == "gemma-3-270m")
        {
            roots = build_gemma_graph(g, mem, model_path, max_seq_len);
        }
        else if (model_name == "qwen-3.6-35b-a3b")
        {
            roots = build_qwen_graph(g, mem, model_path, max_seq_len);
        }
        else
        {
            throw std::runtime_error("Unknown model: " + model_name);
        }

        CostModel costModel;
        costModel.load("benchmarks/records.bin");

        Planner planner(costModel, mem.getMemCaps());

        Bucket bucket;
        bucket.outputNeededRegion = {makeFull(g.getNode(roots.roots[0]).getShape())};
        for (LogicalId inp : roots.inputs)
        {
            bucket.inputDirtyRegions[inp] = {makeFull(g.getNode(inp).getShape())};
        }

        std::unordered_map<LogicalId, MemSpace> cachedNodes;
        std::unordered_map<LogicalId, ParallelBuffer> prealloc;

        try
        {
            CompiledGraph compiled = planner.plan(roots.roots[0], g, bucket, cachedNodes, true, false, nullptr,
                                                  prealloc, 0.0f, "cost", delegate);
            return compiled.cost();
        }
        catch (const std::exception &e)
        {
            Error::throw_err(e.what());
        }
    });
}