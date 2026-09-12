#include "core/session.hpp"
#include "generated/kernels_all.gen.hpp"
#include "tests/constant_view_regression.hpp"

void requireFull(bool condition, const std::string &message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void testFullSession()
{
    Settings settings = Settings::get_default();
    settings.use_ortools_full = true;
    settings.cache_file = "";
    settings.records_path = "";
    settings.log_cost_calls = false;
    settings.mem_caps = {{MemSpace{1, HandleType::CPP}, 16 * 4096}};
    Graph graph;
    MemoryManager memory;
    LogicalId x = graph.input({2, 2}, DType::FLOAT32);
    LogicalId y = graph.input({2, 2}, DType::FLOAT32);
    graph.input_data_types[x] = InputDataType::RUNTIME;
    graph.input_data_types[y] = InputDataType::RUNTIME;
    LogicalId stable = graph.neg(y);
    LogicalId root = graph.add(graph.reshape(x, {4}), graph.reshape(stable, {4}));
    ShapePropagator().inferShapeRecursive(root, graph);
    Session session(graph, memory, root, settings);
    ConstantViewRegression::populateAllKernelDummyRecords(session.costModel);
    Bucket partial;
    partial.inputDirtyRegions[x] = {makeFull(graph.getNode(x).getShape())};
    partial.outputNeededRegion = {makeFull(graph.getNode(root).getShape())};
    partial.weight = 50.0f;
    session.addBucket(partial.inputDirtyRegions, partial.outputNeededRegion, partial.weight);
    session.ensureFullBucket();
    session.ensureCacheCoverage(false);
    session.isPlanned = true;
    session.compile(false);

    requireFull(session.selectedCachedNodes.count(stable), "Expected the clean intermediate to be cached");
    auto solution = nlohmann::json::parse(std::ifstream("benchmarks/ortools_full_solution.json"));
    for (size_t b = 0; b < session.cachedGraphs.size(); ++b)
    {
        const auto &compiled = session.cachedGraphs[b];
        const auto &extraction = solution["extractions"][b];
        for (const auto &inst : compiled.instructions)
        {
            const auto &key = std::to_string(inst.eclass_id.value);
            requireFull(inst.outBuffer.id.value == extraction["eclass_to_buf"][key].get<uint32_t>(),
                        "Native planner replaced the joint bufferization");
            const auto &buffers = extraction["buffers"];
            auto buffer = std::find_if(buffers.begin(), buffers.end(), [&](const auto &buf) {
                return buf["id"].template get<uint32_t>() == inst.outBuffer.id.value;
            });
            requireFull(buffer != buffers.end() && inst.outBuffer.offset == (*buffer)["offset"].get<int64_t>(),
                        "Native planner replaced the joint allocation");
            requireFull(!KernelRegistry::get().getKernel(inst.kernel_id).is_view, "View emitted as executable kernel");
        }
    }
    std::vector<float> x_data{2, 4, 6, 8}, y_data{1, 2, 3, 4};
    session.writeInput(x, x_data.data(), x_data.size() * sizeof(float));
    session.writeInput(y, y_data.data(), y_data.size() * sizeof(float));
    auto output = static_cast<const float *>(session.run(session.manualBuckets[session.fullBucketIdx], nullptr, false));
    for (size_t i = 0; i < x_data.size(); ++i)
        requireFull(std::abs(output[i] - (x_data[i] - y_data[i])) < 1e-5f, "Full bucket produced wrong output");
    for (float &value : x_data)
        value += 10;
    session.writeInput(x, x_data.data(), x_data.size() * sizeof(float));
    output = static_cast<const float *>(session.run(partial, nullptr, false));
    for (size_t i = 0; i < x_data.size(); ++i)
        requireFull(std::abs(output[i] - (x_data[i] - y_data[i])) < 1e-5f, "Cached bucket produced wrong output");
    std::cout << "Full OR-Tools session regression passed." << std::endl;
}

int main(int argc, char **argv)
{
    try
    {
        // Also exercise the production process boundary for Python fixtures on
        // hosts where OR-Tools uses a different architecture from the bindings.
        if (argc == 4 && std::string(argv[1]) == "--solve-json")
            return ortools_export::runOrtoolsSolverProcess(argv[2], argv[3]) ? 0 : 1;
        Settings settings;
        settings.load({"--use-ortools-full"});
        requireFull(settings.use_ortools_full && !settings.use_ortools, "Full solver flag was not parsed independently");
        testFullSession();
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
