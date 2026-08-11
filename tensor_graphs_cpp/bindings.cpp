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

    void init_egraph(const std::vector<float> &node_features, const std::vector<uint32_t> &edge_src,
                     const std::vector<uint32_t> &edge_dst) override
    {
        PYBIND11_OVERRIDE(void, SearchDelegate, init_egraph, node_features, edge_src, edge_dst);
    }

    void init_dispatch_graph(const std::vector<float> &node_features, const std::vector<uint32_t> &edge_src,
                             const std::vector<uint32_t> &edge_dst) override
    {
        PYBIND11_OVERRIDE(void, SearchDelegate, init_dispatch_graph, node_features, edge_src, edge_dst);
    }

    void init_malloc_graph(const std::vector<float> &node_features, const std::vector<uint32_t> &edge_src,
                           const std::vector<uint32_t> &edge_dst) override
    {
        PYBIND11_OVERRIDE(void, SearchDelegate, init_malloc_graph, node_features, edge_src, edge_dst);
    }

    std::vector<uint32_t> order_enodes(const std::vector<ActionFeatureExtractDispatch> &enodes) override
    {
        PYBIND11_OVERRIDE(std::vector<uint32_t>, SearchDelegate, order_enodes, enodes);
    }

    std::vector<uint32_t> order_dispatch(const std::vector<ActionFeatureExtractDispatch> &ready_nodes) override
    {
        PYBIND11_OVERRIDE(std::vector<uint32_t>, SearchDelegate, order_dispatch, ready_nodes);
    }

    std::vector<uint32_t> order_malloc(const std::vector<ActionFeatureMalloc> &avail_buffers) override
    {
        PYBIND11_OVERRIDE(std::vector<uint32_t>, SearchDelegate, order_malloc, avail_buffers);
    }
};

PYBIND11_MODULE(tensor_graphs, m)
{
    // Bind enum HandleType
    py::enum_<HandleType>(m, "HandleType")
        .value("STORAGE", HandleType::STORAGE)
        .value("CPP", HandleType::CPP)
        .value("OPENCL", HandleType::OPENCL)
        .value("CUDA", HandleType::CUDA)
        .export_values();

    // Bind struct MemSpace
    py::class_<MemSpace>(m, "MemSpace")
        .def(py::init<>())
        .def(py::init<uint32_t, HandleType>())
        .def_readwrite("idx", &MemSpace::idx)
        .def_readwrite("type", &MemSpace::type);

    // Bind struct LogicalId with __hash__ and __eq__ so pybind11 can use it as a dict key
    py::class_<LogicalId>(m, "LogicalId")
        .def(py::init<>())
        .def_readwrite("value", &LogicalId::value)
        .def("__hash__", [](const LogicalId &self) { return std::hash<LogicalId>()(self); })
        .def("__eq__", [](const LogicalId &self, const LogicalId &other) { return self == other; })
        .def("__repr__", [](const LogicalId &self) { return "LogicalId(" + std::to_string(self.value) + ")"; });

    // Bind TensorNode
    py::class_<TensorNode>(m, "TensorNode")
        .def_readonly("id", &TensorNode::id)
        .def_readonly("op_type", &TensorNode::opType)
        .def_readonly("child_ids", &TensorNode::child_ids)
        .def_property_readonly("shape", &TensorNode::getShape);

    // Bind Graph
    py::class_<Graph>(m, "Graph")
        .def_readonly("nodes", &Graph::nodes);

    // Bind ActionFeatureExtractDispatch
    py::class_<ActionFeatureExtractDispatch>(m, "ActionFeatureExtractDispatch")
        .def_readwrite("cost", &ActionFeatureExtractDispatch::cost)
        .def_readwrite("size", &ActionFeatureExtractDispatch::size)
        .def_readwrite("mem_space", &ActionFeatureExtractDispatch::mem_space)
        .def_readwrite("engine_idxs", &ActionFeatureExtractDispatch::engine_idxs)
        .def_readwrite("graph", &ActionFeatureExtractDispatch::graph);

    // Bind ActionFeatureMalloc
    py::class_<ActionFeatureMalloc>(m, "ActionFeatureMalloc")
        .def_readwrite("size", &ActionFeatureMalloc::size)
        .def_readwrite("start", &ActionFeatureMalloc::start)
        .def_readwrite("end", &ActionFeatureMalloc::end);

    // Bind SearchDelegate and PySearchDelegate
    py::class_<SearchDelegate, PySearchDelegate, std::shared_ptr<SearchDelegate>>(m, "SearchDelegate")
        .def(py::init<>())
        .def("push_state", &SearchDelegate::push_state)
        .def("pop_state", &SearchDelegate::pop_state)
        .def("init_egraph", &SearchDelegate::init_egraph)
        .def("init_dispatch_graph", &SearchDelegate::init_dispatch_graph)
        .def("init_malloc_graph", &SearchDelegate::init_malloc_graph)
        .def("order_enodes", &SearchDelegate::order_enodes)
        .def("order_dispatch", &SearchDelegate::order_dispatch)
        .def("order_malloc", &SearchDelegate::order_malloc);

    // Bind plan_graph function
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