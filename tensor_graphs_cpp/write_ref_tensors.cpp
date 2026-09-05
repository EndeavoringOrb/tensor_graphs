// tensor_graphs_cpp/write_ref_tensors.cpp
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/argparse.hpp"
#include "core/graph.hpp"
#include "core/kernels.hpp"
#include "core/memory.hpp"
#include "core/misc.hpp"
#include "core/repo.hpp"
#include "core/types.hpp"
#include "generated/kernels_all.gen.hpp"
#include "models/run_models.hpp"

// We use a custom version of executeReferenceGraph for clean tensors
void computeAndWriteCleanTensors(Graph &graph, const std::vector<LogicalId> &rootIds,
                                 const std::vector<LogicalId> &dynamicInputs, Repo &repo,
                                 bool fold_weights = false)
{
    std::vector<LogicalId> topo = topologicalSort(rootIds, graph);

    // Create a set for O(1) lookup of dynamic model inputs
    std::unordered_set<LogicalId> dynamicInputSet(dynamicInputs.begin(), dynamicInputs.end());

    // 1. Identify "clean" nodes
    std::unordered_map<LogicalId, bool> is_clean;
    for (LogicalId nodeId : topo)
    {
        const TensorNode &node = graph.getNode(nodeId);
        if (node.opType == OpType::INPUT)
        {
            if (dynamicInputSet.count(nodeId) > 0)
            {
                is_clean[nodeId] = false;
            }
            else
            {
                InputDataType input_data_type = graph.getInputDataType(nodeId);
                if (fold_weights)
                {
                    is_clean[nodeId] = (input_data_type == InputDataType::CONSTANT ||
                                        input_data_type == InputDataType::STORAGE);
                }
                else
                {
                    is_clean[nodeId] = (input_data_type == InputDataType::CONSTANT);
                }
            }
        }
        else
        {
            bool all_clean = true;
            for (LogicalId pid : node.child_ids)
            {
                if (!is_clean[pid])
                {
                    all_clean = false;
                    break;
                }
            }
            is_clean[nodeId] = all_clean;
        }
    }

    // 2. Infer shapes only for clean nodes
    for (LogicalId nodeId : topo)
    {
        if (!is_clean[nodeId])
            continue;
        if (graph.getNode(nodeId).opType == OpType::INPUT)
            continue;

        ShapePropagator prop;
        prop.inferShape(nodeId, graph);
    }

    std::unordered_map<uint32_t, std::vector<uint8_t>> results;
    std::unordered_map<uint32_t, TensorView> views;

    int computed = 0;
    int skipped = 0;

    for (LogicalId nodeId : topo)
    {
        if (!is_clean[nodeId])
            continue;

        if (repo.has(nodeId))
        {
            skipped++;
            continue;
        }

        const TensorNode &node = graph.getNode(nodeId);
        uint64_t elemSize = getDTypeSize(node.dtype);

        if (node.opType == OpType::INPUT || node.opType == OpType::CACHE)
        {
            TensorView view = makeView(node);
            views[nodeId.value] = view;
            uint64_t bufElements = getRequiredBufferSize(view);
            results[nodeId.value].resize(bufElements * elemSize, 0);

            std::vector<uint8_t> rawBytes;
            if (graph.constantStaging.count(nodeId))
            {
                rawBytes = *graph.constantStaging.at(nodeId);
            }
            else if (graph.input_data_types.count(nodeId) &&
                     graph.input_data_types.at(nodeId) == InputDataType::STORAGE)
            {
                // Fetch file paths and offsets registered in the FileRegistry
                TensorMetadata meta = FileRegistry::get().getNodeMeta(nodeId);
                uint64_t sizeBytes = meta.dataOffsetEnd - meta.dataOffsetStart;
                rawBytes.resize(sizeBytes);

                std::ifstream file(meta.filePath, std::ios::binary);
                if (!file.is_open())
                {
                    Error::throw_err("[computeAndWriteCleanTensors] Failed to open model file: " + meta.filePath);
                }
                file.seekg(meta.dataOffsetStart, std::ios::beg);
                file.read(reinterpret_cast<char *>(rawBytes.data()), sizeBytes);
            }
            else
            {
                Error::throw_err("[computeAndWriteCleanTensors] input node value not "
                                 "found in constantStaging");
            }

            uint64_t numElements = countElements(view);
            for (uint64_t i = 0; i < numElements; ++i)
            {
                uint64_t idx = getStridedIndex(i, view.getShape(), view.strides);
                std::memcpy(results[nodeId.value].data() + idx * elemSize, rawBytes.data() + i * elemSize, elemSize);
            }

            repo.write(nodeId, view, results[nodeId.value].data(), results[nodeId.value].size());
            computed++;
            continue;
        }

        std::vector<const void *> inputPtrs;
        std::vector<TensorView> inputViews;
        std::vector<TensorNode> inputNodes;
        for (LogicalId pid : node.child_ids)
        {
            if (results.find(pid.value) == results.end())
            {
                if (repo.has(pid))
                {
                    results[pid.value] = repo.read(pid);
                    views[pid.value] = makeView(graph.getNode(pid));
                }
                else
                {
                    Error::throw_err("Parent node " + std::to_string(pid.value) + " not found in results or repo");
                }
            }
            inputPtrs.push_back(results[pid.value].data());
            inputViews.push_back(views[pid.value]);
            TensorNode inNode = graph.getNode(pid);
            inNode.strides = views[pid.value].strides;
            inputNodes.push_back(inNode);
        }

        TensorView outViewContig = makeView(node);
        TensorNode outNodeC = node;
        bool ignore_in_ms = (node.opType != OpType::COPY_TO);
        auto refs_c = KernelRegistry::get().findMatchingKernels(
            node.opType, node.opName, inputNodes, outNodeC, true, MemSpace{1, HandleType::CPP}, {MemSpace{1, HandleType::CPP}},
            {Engine{0, EngineType::CPU}}, false, ignore_in_ms, false, true);

        if (refs_c.empty())
        {
            Error::throw_err("No reference kernel found for node " + std::to_string(nodeId.value) +
                             " op=" + toString(node.opType));
        }

        TensorView chosenOutView = outViewContig;
        KernelId chosenKernelUid = refs_c.front();
        const KernelEntry &kernel = KernelRegistry::get().getKernel(chosenKernelUid);

        if (kernel.is_view)
        {
            TensorView dummyOutView(node, 0);
            kernel.inferView(inputNodes, dummyOutView, graph);
            LogicalId parentId = node.child_ids[0];
            results[nodeId.value] = results[parentId.value];
            chosenOutView.strides = dummyOutView.strides;
            chosenOutView.offset = dummyOutView.offset;
            views[nodeId.value] = chosenOutView;

            TensorView contigView = dummyOutView;
            contigView.strides = calcContiguousStrides(dummyOutView.getShape());
            std::vector<uint8_t> contigData(countElements(contigView) * elemSize);
            const uint8_t *srcData = results[parentId.value].data() + chosenOutView.offset;
            for (uint64_t i = 0; i < countElements(contigView); ++i)
            {
                uint64_t srcIdx = getStridedIndex(i, chosenOutView.getShape(), chosenOutView.strides);
                std::memcpy(contigData.data() + i * elemSize, srcData + srcIdx * elemSize, elemSize);
            }
            repo.write(nodeId, contigView, contigData.data(), contigData.size());
            computed++;
            continue;
        }

        views[nodeId.value] = chosenOutView;
        uint64_t bufElements = getRequiredBufferSize(chosenOutView);
        results[nodeId.value].resize(bufElements * elemSize, 0);
        std::vector<void *> outputPtrs = {results[nodeId.value].data()};
        std::vector<TensorView> outputViews = {chosenOutView};

        if (kernel.run)
        {
            kernel.run(KernelContext(inputPtrs, outputPtrs, inputViews, outputViews));
        }

        repo.write(nodeId, chosenOutView, results[nodeId.value].data(), results[nodeId.value].size());
        computed++;
    }

    std::cout << "Computed " << computed << " clean tensors, skipped " << skipped << " (already in repo)." << std::endl;
}

int main(int argc, char *argv[])
{
    ArgParser parser("write_ref_tensors", "Compute and write reference/clean tensors for a model.");
    parser.add_positional("model",
                          "Name of the target model (gemma-3-270m, "
                          "qwen-3.6-35b-a3b).",
                          "gemma-3-270m");
    parser.add_positional("model_path", "Model file or directory containing model files.", "models/google/gemma-3-270m");
    parser.add_option({"--seq-len"}, "Maximum model sequence length (default: 128).", "128");
    parser.add_flag({"--fold-weights"}, "Enable folding of weights (InputDataType::STORAGE).");

    if (!parser.parse(argc, argv))
    {
        return 1;
    }

    std::string model = parser.get_positional("model");
    std::string model_path = parser.get_positional("model_path");
    bool fold_weights = parser.get_flag("--fold-weights");

    MemoryManager mem;
    Graph g;

    uint32_t max_seq_len = 128;
    try
    {
        max_seq_len = std::max(1u, static_cast<uint32_t>(std::stoul(parser.get_option("--seq-len"))));
    }
    catch (...)
    {
        std::cerr << "Invalid --seq-len value provided.\n";
        return 1;
    }
    std::cout << "Building " << model << " Graph for Reference Tensors..." << std::endl;
    ModelGraphRoots roots;

    if (model == "gemma-3-270m" || model == "gemma")
    {
        roots = build_gemma_graph(g, mem, model_path, max_seq_len);
    }
    else if (model == "qwen-3.6-35b-a3b" || model == "qwen")
    {
        roots = build_qwen_graph(g, mem, model_path, max_seq_len);
    }
    else if (model == "krea" || model == "krea-2-turbo" || model == "krea2-turbo" || model == "krea2")
    {
        roots = build_krea2_graph(g, mem, model_path);
    }
    else if (model == "vae" || model == "krea-2-turbo-vae" || model == "krea-vae" || model == "krea2-vae" ||
             model == "qwen-image-vae")
    {
        roots = build_krea2_vae_graph(g, mem, model_path);
    }
    else if (model == "qwen3-vl" || model == "qwen3-vl-bf16" || model == "qwen3vl" || model == "qwen3vl-bf16" ||
             model == "qwen3vl_4b_bf16")
    {
        roots = build_qwen3_vl_graph(g, mem, model_path);
    }
    else
    {
        std::cout << "Unknown model: " << model << std::endl;
        return 1;
    }

    std::string gHash = computeGraphHash(g, roots.roots);
    std::string repoPath = "benchmarks/repo_" + model + "-seq" + std::to_string(max_seq_len);

    std::cout << "Graph Hash: " << gHash << std::endl;
    std::cout << "Using Repo: " << repoPath << std::endl;

    Repo repo(repoPath, gHash, false);

    computeAndWriteCleanTensors(g, roots.roots, roots.inputs, repo, fold_weights);

    std::cout << "Done writing reference tensors." << std::endl;
    return 0;
}
