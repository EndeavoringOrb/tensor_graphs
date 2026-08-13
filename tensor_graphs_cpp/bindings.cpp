#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "core/plan/cached_plan.hpp"
#include "core/plan/planner.hpp"
#include "core/plan/search_delegate.hpp"
#include "core/session.hpp"
#include "generated/kernels_all.gen.hpp"
#include "models/deepseek-v4-flash.hpp"
#include "models/run_models.hpp"

namespace py = pybind11;

// Pybind11 Trampoline Class for C++ SearchDelegate virtual method overrides
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

    void init_bufferize_graph(const std::vector<float> &node_features, const std::vector<uint32_t> &edge_src,
                              const std::vector<uint32_t> &edge_dst) override
    {
        PYBIND11_OVERRIDE(void, SearchDelegate, init_bufferize_graph, node_features, edge_src, edge_dst);
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

    std::vector<uint32_t> order_bufferize(const std::vector<ActionFeatureBufferize> &choices) override
    {
        PYBIND11_OVERRIDE(std::vector<uint32_t>, SearchDelegate, order_bufferize, choices);
    }

    std::vector<uint32_t> order_malloc(const std::vector<ActionFeatureMalloc> &avail_buffers) override
    {
        PYBIND11_OVERRIDE(std::vector<uint32_t>, SearchDelegate, order_malloc, avail_buffers);
    }
};

// C++ API for LLM Generation accessible to Python
class LLMSession
{
    std::unique_ptr<MemoryManager> mem;
    std::unique_ptr<Graph> g;
    std::unique_ptr<Repo> repo;
    std::unique_ptr<Session> session;
    LogicalId inputIdsId;
    LogicalId logitsId;
    uint32_t vocab_size = 0;
    uint32_t max_seq_len = 128;
    std::vector<uint32_t> prev_tokens;

  public:
    LLMSession(const std::string &model_name, const std::string &model_path,
               std::shared_ptr<SearchDelegate> delegate = nullptr, float min_compile_time = 0.0f,
               bool compile_decode_buckets = false, const std::string &cache_file = "", bool disable_caching = false)
    {
        std::unordered_map<MemSpace, uint64_t> bufferSizes = {
            {MemSpace{1, HandleType::CPP}, 16ULL * 1024 * 1024 * 1024}};

        if (HardwareCaps::get().has_opencl)
        {
            bufferSizes[MemSpace{1, HandleType::OPENCL}] = 1ULL * 1024 * 1024 * 1024;
        }

        mem = std::make_unique<MemoryManager>(bufferSizes);
        g = std::make_unique<Graph>();

        if (model_name == "gemma-3-270m")
        {
            Gemma3ModelConfig cfg;
            vocab_size = cfg.vocab_size;
            auto roots = build_gemma_graph(*g, *mem, model_path, max_seq_len);
            logitsId = roots.roots[0];
            inputIdsId = roots.inputs[0];
        }
        else if (model_name == "qwen-3.6-35b-a3b")
        {
            Qwen3_6_35B_A3B_Config cfg;
            vocab_size = cfg.vocab_size;
            auto roots = build_qwen_graph(*g, *mem, model_path, max_seq_len);
            logitsId = roots.roots[0];
            inputIdsId = roots.inputs[0];
        }
        else if (model_name == "deepseek-v4")
        {
            DeepSeekV4FlashConfig cfg;
            vocab_size = cfg.vocab_size;
            inputIdsId = g->input({1, max_seq_len}, DType::INT32);
            DeepSeekV4FlashModel model(cfg, max_seq_len, *g, *mem, model_path);
            logitsId = model.build_graph(inputIdsId);
        }
        else
        {
            throw std::runtime_error("Unknown model: " + model_name);
        }

        std::string gHash = computeGraphHash(*g, {logitsId});
        repo = std::make_unique<Repo>("benchmarks/repo_" + model_name, gHash, true);

        std::string actual_cache = cache_file;
        if (actual_cache.empty())
        {
            std::filesystem::create_directories("dirty_region_caches");
            actual_cache = "dirty_region_caches/" + model_name + "-cpp.bin";
        }

        session = std::make_unique<Session>(*g, *mem, logitsId, actual_cache, 0, repo.get(), disable_caching,
                                            min_compile_time, delegate);

        if (compile_decode_buckets)
        {
            for (uint32_t i = 0; i < max_seq_len; ++i)
            {
                std::unordered_map<LogicalId, std::vector<Region>> inputDirty;
                Region inputRegion;
                inputRegion.region = {{0, 1}, {i, i + 1}};
                inputDirty[inputIdsId] = {inputRegion};

                Region outputNeeded;
                outputNeeded.region = {{0, 1}, {i, i + 1}, {0, vocab_size}};
                session->addBucket(inputDirty, {outputNeeded});
            }
        }

        session->compile(true);
    }

    int32_t generate_step(const std::vector<uint32_t> &tokens)
    {
        if (tokens.empty() || tokens.size() >= max_seq_len)
            return -1;

        std::vector<Region> dirty_regions;
        uint32_t i = 0;
        while (i < tokens.size())
        {
            if (i >= prev_tokens.size() || tokens[i] != prev_tokens[i])
            {
                uint32_t start = i;
                while (i < tokens.size() && (i >= prev_tokens.size() || tokens[i] != prev_tokens[i]))
                {
                    i++;
                }
                uint32_t end = i;
                Region inR;
                inR.region = {{0, 1}, {start, end}};
                dirty_regions.push_back(inR);
            }
            else
            {
                i++;
            }
        }

        if (dirty_regions.empty())
        {
            uint32_t tokIdx = tokens.size() - 1;
            Region inR;
            inR.region = {{0, 1}, {tokIdx, tokIdx + 1}};
            dirty_regions.push_back(inR);
        }

        std::vector<int32_t> input_data(max_seq_len, 0);
        for (size_t k = 0; k < tokens.size(); ++k)
        {
            input_data[k] = tokens[k];
        }

        session->writeInput(inputIdsId, input_data.data(), input_data.size() * sizeof(int32_t));

        Bucket b;
        uint32_t tokIdx = tokens.size() - 1;
        Region outR;
        outR.region = {{0, 1}, {tokIdx, tokIdx + 1}, {0, vocab_size}};
        b.inputDirtyRegions = {{inputIdsId, dirty_regions}};
        b.outputNeededRegion = {outR};

        const float *device_output = static_cast<const float *>(session->run(b));

        std::vector<float> host_output;
#ifdef TG_USE_CUDA
        cudaPointerAttributes attrs;
        if (cudaPointerGetAttributes(&attrs, device_output) == cudaSuccess && attrs.type == cudaMemoryTypeDevice)
        {
            host_output.resize(vocab_size);
            cudaMemcpy(host_output.data(), device_output + tokIdx * vocab_size, vocab_size * sizeof(float),
                       cudaMemcpyDeviceToHost);
        }
        else
#endif
        {
            host_output.assign(device_output + tokIdx * vocab_size, device_output + tokIdx * vocab_size + vocab_size);
        }

        float max_val = -1e9f;
        int32_t argmax_idx = 0;
        for (uint32_t k = 0; k < vocab_size; ++k)
        {
            if (host_output[k] > max_val)
            {
                max_val = host_output[k];
                argmax_idx = k;
            }
        }

        prev_tokens = tokens;
        return argmax_idx;
    }
};

PYBIND11_MODULE(tensor_graphs, m)
{
    m.doc() = "Python bindings for TensorGraph compilation and search optimization";

    // Enums
    py::enum_<HandleType>(m, "HandleType")
        .value("STORAGE", HandleType::STORAGE)
        .value("CPP", HandleType::CPP)
        .value("OPENCL", HandleType::OPENCL)
        .value("CUDA", HandleType::CUDA)
        .export_values();

    // Value Objects
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

    py::class_<Graph>(m, "Graph").def_readonly("nodes", &Graph::nodes);

    // Action Feature Structs
    py::class_<ActionFeatureExtractDispatch>(m, "ActionFeatureExtractDispatch")
        .def_readwrite("cost", &ActionFeatureExtractDispatch::cost)
        .def_readwrite("size", &ActionFeatureExtractDispatch::size)
        .def_readwrite("mem_space", &ActionFeatureExtractDispatch::mem_space)
        .def_readwrite("engine_idxs", &ActionFeatureExtractDispatch::engine_idxs)
        .def_readwrite("graph", &ActionFeatureExtractDispatch::graph);

    py::class_<ActionFeatureBufferize>(m, "ActionFeatureBufferize")
        .def_readwrite("is_new_buffer", &ActionFeatureBufferize::is_new_buffer)
        .def_readwrite("size", &ActionFeatureBufferize::size)
        .def_readwrite("parent_size", &ActionFeatureBufferize::parent_size)
        .def_readwrite("parent_birth_time", &ActionFeatureBufferize::parent_birth_time);

    py::class_<ActionFeatureMalloc>(m, "ActionFeatureMalloc")
        .def_readwrite("size", &ActionFeatureMalloc::size)
        .def_readwrite("start", &ActionFeatureMalloc::start)
        .def_readwrite("end", &ActionFeatureMalloc::end);

    // Search Delegate
    py::class_<SearchDelegate, PySearchDelegate, std::shared_ptr<SearchDelegate>>(m, "SearchDelegate")
        .def(py::init<>())
        .def("push_state", &SearchDelegate::push_state)
        .def("pop_state", &SearchDelegate::pop_state)
        .def("init_egraph", &SearchDelegate::init_egraph)
        .def("init_dispatch_graph", &SearchDelegate::init_dispatch_graph)
        .def("init_bufferize_graph", &SearchDelegate::init_bufferize_graph)
        .def("init_malloc_graph", &SearchDelegate::init_malloc_graph)
        .def("order_enodes", &SearchDelegate::order_enodes)
        .def("order_dispatch", &SearchDelegate::order_dispatch)
        .def("order_bufferize", &SearchDelegate::order_bufferize)
        .def("order_malloc", &SearchDelegate::order_malloc);

    // Opaque handle for Caching Saturated E-Graphs
    py::class_<SaturatedEGraphContext, std::shared_ptr<SaturatedEGraphContext>>(m, "SaturatedEGraphContext")
        .def_property_readonly("num_buckets", [](const SaturatedEGraphContext &self) { return self.buckets.size(); })
        .def("setup_for_bucket", &SaturatedEGraphContext::setup_for_bucket, py::arg("bucket_idx"),
             py::arg("cachedNodes"), py::arg("strictCache"));

    m.def("build_and_saturate_egraph", &build_and_saturate_egraph, py::arg("model_name"), py::arg("model_path"),
          py::arg("log_cost_calls") = false, py::arg("compile_decode_buckets") = true);

    m.def("extract_best_from_egraph", &extract_best_from_egraph, py::arg("ctx"), py::arg("delegate"),
          py::arg("log_cost_calls") = false);

    m.def("get_cached_nodes", &get_cached_nodes, py::arg("ctx"), py::arg("delegate"),
          py::arg("log_cost_calls") = false);

    py::class_<LLMSession>(m, "LLMSession")
        .def(py::init<const std::string &, const std::string &, std::shared_ptr<SearchDelegate>, float, bool,
                      const std::string &, bool>(),
             py::arg("model_name"), py::arg("model_path"), py::arg("delegate") = nullptr,
             py::arg("min_compile_time") = 0.0f, py::arg("compile_decode_buckets") = false, py::arg("cache_file") = "",
             py::arg("disable_caching") = false)
        .def("generate_step", &LLMSession::generate_step);
}