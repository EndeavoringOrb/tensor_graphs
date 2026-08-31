#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "core/common/thread_pool.hpp"
#include "core/hardware.hpp"
#include "core/plan/cached_plan.hpp"
#include "core/plan/planner.hpp"
#include "core/plan/search_delegate.hpp"
#include "core/session.hpp"
#include "generated/kernels_all.gen.hpp"
#include "models/deepseek-v4-flash.hpp"
#include "models/krea-2-turbo.hpp"
#include "models/qwen-image-vae.hpp"
#include "models/qwen3-vl.hpp"
#include "models/run_models.hpp"

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
    void on_leaf_evaluated(float cost) override
    {
        PYBIND11_OVERRIDE(void, SearchDelegate, on_leaf_evaluated, cost);
    }

    void init_cache_graph(const std::vector<float> &node_features, const std::vector<uint32_t> &edge_src,
                          const std::vector<uint32_t> &edge_dst) override
    {
        PYBIND11_OVERRIDE(void, SearchDelegate, init_cache_graph, node_features, edge_src, edge_dst);
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

    std::vector<uint32_t> order_cache(const std::vector<ActionFeatureCache> &choices) override
    {
        PYBIND11_OVERRIDE(std::vector<uint32_t>, SearchDelegate, order_cache, choices);
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

    std::vector<uint32_t> order_frontier(const std::vector<ActionFeatureFrontier> &frontier) override
    {
        PYBIND11_OVERRIDE(std::vector<uint32_t>, SearchDelegate, order_frontier, frontier);
    }
};

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
               bool compile_decode_buckets = false, const std::string &cache_file = "", bool disable_caching = false,
               uint32_t threads = 0, bool log_cost_calls = true)
    {
        if (threads > 0)
        {
            set_num_threads(threads);
        }

        auto act_delegate = delegate ? delegate : std::make_shared<HeuristicSearchDelegate>();

        mem = std::make_unique<MemoryManager>();
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
                                            min_compile_time, act_delegate, log_cost_calls);

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

class Krea2Session
{
    std::unique_ptr<MemoryManager> mem;
    std::unique_ptr<Graph> g;
    std::unique_ptr<Repo> repo;
    std::unique_ptr<Session> session;
    LogicalId inputIdsId;
    LogicalId latentInputId;
    LogicalId imageOutputId;

    Krea2TurboConfig cfg;
    Krea2TurboVAEConfig vae_cfg;
    Qwen3VLConfig te_cfg;
    uint32_t num_steps;
    float mu_val;

  public:
    Krea2Session(const std::string &model_path, const std::string &text_encoder_path = "",
                 const std::string &vae_path = "", uint32_t height = 1024, uint32_t width = 1024,
                 uint32_t text_seq_len = 128, uint32_t steps = 8, float mu = 1.15f,
                 std::shared_ptr<SearchDelegate> delegate = nullptr, float min_compile_time = 0.0f,
                 const std::string &cache_file = "", bool disable_caching = false, uint32_t threads = 0,
                 bool log_cost_calls = true)
        : cfg(height, width, text_seq_len), vae_cfg(height, width), te_cfg(), num_steps(steps), mu_val(mu)
    {
        if (threads > 0)
        {
            set_num_threads(threads);
        }

        auto act_delegate = delegate ? delegate : std::make_shared<HeuristicSearchDelegate>();

        std::string actual_dit_path = model_path;
        if (std::filesystem::is_directory(model_path))
        {
            if (std::filesystem::exists(model_path + "/krea.safetensors"))
                actual_dit_path = model_path + "/krea.safetensors";
            else if (std::filesystem::exists(model_path + "/turbo.safetensors"))
                actual_dit_path = model_path + "/turbo.safetensors";
            else if (std::filesystem::exists(model_path + "/krea2_turbo_fp8_scaled.safetensors"))
                actual_dit_path = model_path + "/krea2_turbo_fp8_scaled.safetensors";
            else if (std::filesystem::exists(model_path + "/transformer"))
                actual_dit_path = model_path + "/transformer";
        }

        std::string actual_te_path = text_encoder_path;
        if (actual_te_path.empty())
        {
            if (std::filesystem::exists(model_path + "/text_encoder"))
                actual_te_path = model_path + "/text_encoder";
            else if (std::filesystem::exists(model_path + "/text_encoders"))
                actual_te_path = model_path + "/text_encoders";
            else if (std::filesystem::exists(model_path + "/qwen3vl_4b_bf16.safetensors"))
                actual_te_path = model_path + "/qwen3vl_4b_bf16.safetensors";
            else if (std::filesystem::exists(model_path + "/qwen3vl_4b.safetensors"))
                actual_te_path = model_path + "/qwen3vl_4b.safetensors";
            else if (std::filesystem::exists(model_path + "/qwen3vl_4b_fp8_scaled.safetensors"))
                actual_te_path = model_path + "/qwen3vl_4b_fp8_scaled.safetensors";
            else
                actual_te_path = model_path;
        }

        std::string actual_vae_path = vae_path;
        if (actual_vae_path.empty())
        {
            if (std::filesystem::exists(model_path + "/vae"))
                actual_vae_path = model_path + "/vae";
            else if (std::filesystem::exists(model_path + "/qwen_image_vae.safetensors"))
                actual_vae_path = model_path + "/qwen_image_vae.safetensors";
            else
                actual_vae_path = model_path;
        }

        mem = std::make_unique<MemoryManager>();
        g = std::make_unique<Graph>();

        auto roots = build_krea2_pipeline_graph(*g, *mem, actual_dit_path, actual_te_path, actual_vae_path, height,
                                                width, text_seq_len, steps, mu);
        imageOutputId = roots.roots[0];
        inputIdsId = roots.inputs[0];
        latentInputId = roots.inputs[1];

        std::string gHash = computeGraphHash(*g, {imageOutputId});
        repo = std::make_unique<Repo>("benchmarks/repo_krea-2-turbo-pipeline", gHash, true);

        std::string actual_cache = cache_file;
        if (actual_cache.empty())
        {
            std::filesystem::create_directories("dirty_region_caches");
            actual_cache = "dirty_region_caches/krea-2-turbo-pipeline-" + std::to_string(width) + "x" +
                           std::to_string(height) + "-s" + std::to_string(steps) + ".bin";
        }

        session = std::make_unique<Session>(*g, *mem, imageOutputId, actual_cache, 0, repo.get(), disable_caching,
                                            min_compile_time, act_delegate, log_cost_calls);
        session->compile(true);
    }

    std::vector<float> generate_image(const std::vector<int32_t> &token_ids, const std::vector<float> &latent_data)
    {
        std::vector<int32_t> padded_tokens = token_ids;
        if (padded_tokens.size() < cfg.text_seq_len)
        {
            padded_tokens.resize(cfg.text_seq_len, 0);
        }
        else if (padded_tokens.size() > cfg.text_seq_len)
        {
            padded_tokens.resize(cfg.text_seq_len);
        }

        session->writeInput(inputIdsId, padded_tokens.data(), cfg.text_seq_len * sizeof(int32_t));
        session->writeInput(latentInputId, latent_data.data(), latent_data.size() * sizeof(float));

        Bucket b;
        const float *device_output = static_cast<const float *>(session->run(b));

        uint64_t num_pixels = 1ULL * vae_cfg.in_channels * cfg.height * cfg.width;
        std::vector<float> host_output(num_pixels);

#ifdef TG_USE_CUDA
        cudaPointerAttributes attrs;
        if (cudaPointerGetAttributes(&attrs, device_output) == cudaSuccess && attrs.type == cudaMemoryTypeDevice)
        {
            cudaMemcpy(host_output.data(), device_output, num_pixels * sizeof(float), cudaMemcpyDeviceToHost);
        }
        else
#endif
        {
            std::memcpy(host_output.data(), device_output, num_pixels * sizeof(float));
        }

        return host_output;
    }
};

PYBIND11_MODULE(tensor_graphs, m)
{
    m.doc() = "Python bindings for TensorGraph compilation and search optimization";

    m.def("set_num_threads", &set_num_threads, py::arg("num_threads"),
          "Set the number of threads used by TensorGraph thread pools and parallel execution.");
    m.def("get_num_threads", &get_num_threads, "Get the current number of threads used by TensorGraph.");

    // Enums
    py::enum_<HandleType>(m, "HandleType")
        .value("STORAGE", HandleType::STORAGE)
        .value("CPP", HandleType::CPP)
        .value("OPENCL", HandleType::OPENCL)
        .value("CUDA", HandleType::CUDA)
        .export_values();

    py::enum_<DType>(m, "DType")
        .value("FLOAT32", DType::FLOAT32)
        .value("INT32", DType::INT32)
        .value("INT64", DType::INT64)
        .value("BF16", DType::BF16)
        .value("BOOL", DType::BOOL)
        .value("ANY", DType::ANY)
        .value("INT8", DType::INT8)
        .value("E2M1_PACKED_INT8", DType::E2M1_PACKED_INT8)
        .value("E2M1", DType::E2M1)
        .value("F8_E8M0", DType::F8_E8M0)
        .value("F8_E4M3", DType::F8_E4M3)
        .export_values();

    py::enum_<OpType>(m, "OpType")
        .value("INPUT", OpType::INPUT)
        .value("CACHE", OpType::CACHE)
        .value("ADD", OpType::ADD)
        .value("MUL", OpType::MUL)
        .value("DIVIDE", OpType::DIVIDE)
        .value("DOT", OpType::DOT)
        .value("SIN", OpType::SIN)
        .value("COS", OpType::COS)
        .value("NEGATE", OpType::NEGATE)
        .value("POWER", OpType::POWER)
        .value("SUM", OpType::SUM)
        .value("MAX", OpType::MAX)
        .value("RESHAPE", OpType::RESHAPE)
        .value("PERMUTE", OpType::PERMUTE)
        .value("SLICE", OpType::SLICE)
        .value("CONCAT", OpType::CONCAT)
        .value("CAST", OpType::CAST)
        .value("UNPACK", OpType::UNPACK)
        .value("REPEAT", OpType::REPEAT)
        .value("ARANGE", OpType::ARANGE)
        .value("TRIU", OpType::TRIU)
        .value("GATHER", OpType::GATHER)
        .value("FILL", OpType::FILL)
        .value("COPY_TO", OpType::COPY_TO)
        .value("IM2COL", OpType::IM2COL)
        .value("CONTIGUOUS", OpType::CONTIGUOUS)
        .value("SCATTER", OpType::SCATTER)
        .value("LOG", OpType::LOG)
        .value("ARGMAX", OpType::ARGMAX)
        .value("LT", OpType::LT)
        .value("EQ", OpType::EQ)
        .value("AND", OpType::AND)
        .value("OR", OpType::OR)
        .value("NOT", OpType::NOT)
        .value("FUSED", OpType::FUSED)
        .export_values();

    // Value Objects
    py::class_<Dim>(m, "Dim")
        .def(py::init<>())
        .def(py::init<uint32_t, uint32_t>())
        .def_readwrite("start", &Dim::start)
        .def_readwrite("stop", &Dim::stop)
        .def("__repr__",
             [](const Dim &d) { return "Dim(" + std::to_string(d.start) + ", " + std::to_string(d.stop) + ")"; });

    py::class_<Region>(m, "Region")
        .def(py::init<>())
        .def_readwrite("region", &Region::region)
        .def("__repr__", [](const Region &r) { return encodeRegion(r); });

    py::class_<Bucket>(m, "Bucket")
        .def(py::init<>())
        .def_readwrite("inputDirtyRegions", &Bucket::inputDirtyRegions)
        .def_readwrite("outputNeededRegion", &Bucket::outputNeededRegion);

    py::class_<MemSpace>(m, "MemSpace")
        .def(py::init<>())
        .def(py::init<uint32_t, HandleType>())
        .def_readwrite("idx", &MemSpace::idx)
        .def_readwrite("type", &MemSpace::type);

    py::class_<LogicalId>(m, "LogicalId")
        .def(py::init<>())
        .def_readwrite("value", &LogicalId::value)
        .def("__hash__", [](const LogicalId &self) { return std::hash<LogicalId>()(self); })
        .def("__eq__", [](const LogicalId &self, const LogicalId &other) { return self == other; })
        .def("__repr__", [](const LogicalId &self) { return "LogicalId(" + std::to_string(self.value) + ")"; });

    py::class_<TensorNode>(m, "TensorNode")
        .def_readonly("id", &TensorNode::id)
        .def_readonly("op_type", &TensorNode::opType)
        .def_readonly("dtype", &TensorNode::dtype)
        .def_readonly("child_ids", &TensorNode::child_ids)
        .def_property_readonly("shape", &TensorNode::getShape);

    py::class_<Graph>(m, "Graph")
        .def(py::init<>())
        .def_readonly("nodes", &Graph::nodes)
        .def("hasNode", &Graph::hasNode)
        .def("getNode", [](const Graph &self, LogicalId id) { return self.getNode(id); })
        .def(
            "input",
            [](Graph &self, const std::vector<uint32_t> &shape, DType dtype) { return self.input(shape, dtype); },
            py::arg("shape"), py::arg("dtype") = DType::FLOAT32)
        .def("add", [](Graph &self, LogicalId a, LogicalId b) { return self.add(a, b); })
        .def("mul", [](Graph &self, LogicalId a, LogicalId b) { return self.mul(a, b); })
        .def("div", [](Graph &self, LogicalId a, LogicalId b) { return self.div(a, b); })
        .def("dot", [](Graph &self, LogicalId a, LogicalId b) { return self.dot(a, b); })
        .def("sin", [](Graph &self, LogicalId a) { return self.sin(a); })
        .def("cos", [](Graph &self, LogicalId a) { return self.cos(a); })
        .def("neg", [](Graph &self, LogicalId a) { return self.neg(a); })
        .def("pow", [](Graph &self, LogicalId a, LogicalId b) { return self.pow(a, b); })
        .def("sum", [](Graph &self, LogicalId a, LogicalId b) { return self.sum(a, b); })
        .def("max", [](Graph &self, LogicalId a, LogicalId b) { return self.max(a, b); })
        .def("reshape",
             [](Graph &self, LogicalId a, const std::vector<int32_t> &shape) { return self.reshape(a, shape); })
        .def("permute", [](Graph &self, LogicalId a, LogicalId dims) { return self.permute(a, dims); })
        .def("slice", [](Graph &self, LogicalId a, LogicalId st, LogicalId en,
                         LogicalId step) { return self.slice(a, st, en, step); })
        .def("scatter", [](Graph &self, LogicalId t, LogicalId u, LogicalId st, LogicalId en,
                           LogicalId step) { return self.scatter(t, u, st, en, step); })
        .def("concat",
             [](Graph &self, const std::vector<LogicalId> &ids, uint32_t axis) { return self.concat(ids, axis); })
        .def("cast", [](Graph &self, LogicalId a, DType dtype) { return self.cast(a, dtype); })
        .def("repeat",
             [](Graph &self, LogicalId a, uint32_t repeats, uint32_t axis) { return self.repeat(a, repeats, axis); })
        .def("arange",
             [](Graph &self, LogicalId st, LogicalId sp, LogicalId step) { return self.arange(st, sp, step); })
        .def("triu", [](Graph &self, LogicalId a, LogicalId k) { return self.triu(a, k); })
        .def("gather", [](Graph &self, LogicalId a, LogicalId idx) { return self.gather(a, idx); })
        .def("fill",
             [](Graph &self, float value, const std::vector<uint32_t> &shape) { return self.fill(value, shape); })
        .def("constant", [](Graph &self, const std::vector<int32_t> &vals) { return self.constant(vals); })
        .def("relu", [](Graph &self, LogicalId a, const std::vector<uint32_t> &shape) { return self.relu(a, shape); })
        .def("log", [](Graph &self, LogicalId a) { return self.log(a); })
        .def("argmax", [](Graph &self, LogicalId a, LogicalId dim, LogicalId k) { return self.argmax(a, dim, k); })
        .def("lt", [](Graph &self, LogicalId a, LogicalId b) { return self.lt(a, b); })
        .def("eq", [](Graph &self, LogicalId a, LogicalId b) { return self.eq(a, b); })
        .def("logical_and", [](Graph &self, LogicalId a, LogicalId b) { return self.logical_and(a, b); })
        .def("logical_or", [](Graph &self, LogicalId a, LogicalId b) { return self.logical_or(a, b); })
        .def("logical_not", [](Graph &self, LogicalId a) { return self.logical_not(a); });

    // Action Feature Structs
    py::class_<ActionFeatureCache>(m, "ActionFeatureCache")
        .def_readwrite("is_cached", &ActionFeatureCache::is_cached)
        .def_readwrite("size", &ActionFeatureCache::size)
        .def_readwrite("num_users", &ActionFeatureCache::num_users)
        .def_readwrite("logical_id", &ActionFeatureCache::logical_id)
        .def_readwrite("mem_space", &ActionFeatureCache::mem_space)
        .def_readwrite("mem_cap", &ActionFeatureCache::mem_cap);

    py::class_<ActionFeatureExtractDispatch>(m, "ActionFeatureExtractDispatch")
        .def_readwrite("cost", &ActionFeatureExtractDispatch::cost)
        .def_readwrite("dp_cost", &ActionFeatureExtractDispatch::dp_cost)
        .def_readwrite("min_dp_cp_cost", &ActionFeatureExtractDispatch::min_dp_cp_cost)
        .def_readwrite("rev_cp_cost", &ActionFeatureExtractDispatch::rev_cp_cost)
        .def_readwrite("size", &ActionFeatureExtractDispatch::size)
        .def_readwrite("mem_space", &ActionFeatureExtractDispatch::mem_space)
        .def_readwrite("engine_idxs", &ActionFeatureExtractDispatch::engine_idxs)
        .def_readwrite("num_nodes", &ActionFeatureExtractDispatch::num_nodes)
        .def_readwrite("num_edges", &ActionFeatureExtractDispatch::num_edges)
        .def_readwrite("mem_cap", &ActionFeatureExtractDispatch::mem_cap);

    py::class_<ActionFeatureBufferize>(m, "ActionFeatureBufferize")
        .def_readwrite("is_new_buffer", &ActionFeatureBufferize::is_new_buffer)
        .def_readwrite("size", &ActionFeatureBufferize::size)
        .def_readwrite("parent_size", &ActionFeatureBufferize::parent_size)
        .def_readwrite("parent_birth_time", &ActionFeatureBufferize::parent_birth_time)
        .def_readwrite("mem_space", &ActionFeatureBufferize::mem_space)
        .def_readwrite("mem_cap", &ActionFeatureBufferize::mem_cap);

    py::class_<ActionFeatureMalloc>(m, "ActionFeatureMalloc")
        .def_readwrite("size", &ActionFeatureMalloc::size)
        .def_readwrite("start", &ActionFeatureMalloc::start)
        .def_readwrite("end", &ActionFeatureMalloc::end)
        .def_readwrite("mem_space", &ActionFeatureMalloc::mem_space)
        .def_readwrite("mem_cap", &ActionFeatureMalloc::mem_cap);

    py::class_<ActionFeatureFrontier>(m, "ActionFeatureFrontier")
        .def_readwrite("eclass_id", &ActionFeatureFrontier::eclass_id)
        .def_readwrite("num_enodes", &ActionFeatureFrontier::num_enodes)
        .def_readwrite("min_dp_cp_cost", &ActionFeatureFrontier::min_dp_cp_cost)
        .def_readwrite("min_dp_cost", &ActionFeatureFrontier::min_dp_cost)
        .def_readwrite("size", &ActionFeatureFrontier::size)
        .def_readwrite("dtype", &ActionFeatureFrontier::dtype)
        .def_readwrite("mem_space", &ActionFeatureFrontier::mem_space)
        .def_readwrite("mem_cap", &ActionFeatureFrontier::mem_cap);

    // Search Delegate
    py::class_<SearchDelegate, PySearchDelegate, std::shared_ptr<SearchDelegate>>(m, "SearchDelegate")
        .def(py::init<>())
        .def("push_state", &SearchDelegate::push_state)
        .def("pop_state", &SearchDelegate::pop_state)
        .def("on_leaf_evaluated", &SearchDelegate::on_leaf_evaluated)
        .def("init_cache_graph", &SearchDelegate::init_cache_graph)
        .def("init_egraph", &SearchDelegate::init_egraph)
        .def("init_dispatch_graph", &SearchDelegate::init_dispatch_graph)
        .def("init_bufferize_graph", &SearchDelegate::init_bufferize_graph)
        .def("init_malloc_graph", &SearchDelegate::init_malloc_graph)
        .def("order_cache", &SearchDelegate::order_cache)
        .def("order_enodes", &SearchDelegate::order_enodes)
        .def("order_dispatch", &SearchDelegate::order_dispatch)
        .def("order_bufferize", &SearchDelegate::order_bufferize)
        .def("order_malloc", &SearchDelegate::order_malloc)
        .def("order_frontier", &SearchDelegate::order_frontier);

    py::class_<HeuristicSearchDelegate, SearchDelegate, std::shared_ptr<HeuristicSearchDelegate>>(
        m, "HeuristicSearchDelegate")
        .def(py::init<>());
    m.attr("HeuristicDelegate") = m.attr("HeuristicSearchDelegate");

    // Saturated E-Graph Context & Simulations
    py::class_<SaturatedEGraphContext, std::shared_ptr<SaturatedEGraphContext>>(m, "SaturatedEGraphContext")
        .def_property_readonly("num_buckets", [](const SaturatedEGraphContext &self) { return self.buckets.size(); });

    m.def("build_and_saturate_egraph", &build_and_saturate_egraph, py::arg("model_name"), py::arg("model_path"),
          py::arg("log_cost_calls") = false, py::arg("compile_decode_buckets") = true, py::arg("max_seq_len") = 8);

    m.def("build_and_saturate_egraph_from_graph", &build_and_saturate_egraph_from_graph, py::arg("graph"),
          py::arg("root_id"), py::arg("buckets") = std::vector<Bucket>{}, py::arg("log_cost_calls") = false,
          py::arg("mem_cap_override") = 0);

    m.def("run_hierarchical_simulations", &run_hierarchical_simulations, py::arg("ctx"), py::arg("bucket_idx"),
          py::arg("delegate"), py::arg("level_simulations"), py::arg("log_cost_calls") = false,
          py::arg("min_compile_seconds") = 0.0f);

    m.def("extract_best_from_egraph", &extract_best_from_egraph, py::arg("ctx"), py::arg("delegate"),
          py::arg("log_cost_calls") = false);

    py::class_<LLMSession>(m, "LLMSession")
        .def(py::init<const std::string &, const std::string &, std::shared_ptr<SearchDelegate>, float, bool,
                      const std::string &, bool, uint32_t, bool>(),
             py::arg("model_name"), py::arg("model_path"), py::arg("delegate") = nullptr,
             py::arg("min_compile_time") = 0.0f, py::arg("compile_decode_buckets") = false, py::arg("cache_file") = "",
             py::arg("disable_caching") = false, py::arg("threads") = 0, py::arg("log_cost_calls") = true)
        .def("generate_step", &LLMSession::generate_step);

    py::class_<Krea2Session>(m, "Krea2Session")
        .def(py::init<const std::string &, const std::string &, const std::string &, uint32_t, uint32_t, uint32_t,
                      uint32_t, float, std::shared_ptr<SearchDelegate>, float, const std::string &, bool, uint32_t,
                      bool>(),
             py::arg("model_path"), py::arg("text_encoder_path") = "", py::arg("vae_path") = "",
             py::arg("height") = 1024, py::arg("width") = 1024, py::arg("text_seq_len") = 128, py::arg("steps") = 8,
             py::arg("mu") = 1.15f, py::arg("delegate") = nullptr, py::arg("min_compile_time") = 0.0f,
             py::arg("cache_file") = "", py::arg("disable_caching") = false, py::arg("threads") = 0,
             py::arg("log_cost_calls") = true)
        .def("generate_image", &Krea2Session::generate_image, py::arg("token_ids"), py::arg("latent_data"))
        .def("generate", &Krea2Session::generate_image, py::arg("token_ids"), py::arg("latent_data"));
}