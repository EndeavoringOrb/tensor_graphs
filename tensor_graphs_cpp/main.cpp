#include "core/memory.hpp"
#include "core/loaders/safetensors.hpp"
#include "core/graph.hpp"
#include "core/hashing.hpp"
#include <iostream>

int main()
{
    MemoryManager memManager;
    memManager.buffers.emplace(Backend::CPU, DeviceBuffer(1024 * 1024 * 1024));

    SafetensorsLoader loader("C:/Users/aaron/CODING/tensor_graphs/resources/model.safetensors");
    std::string weightName = "model.embed_tokens.weight";

    Graph graph;

    if (loader.hasTensor(weightName))
    {
        const auto &meta = loader.getMetadata(weightName);

        // 1. User allocates for safetensors weight
        uint32_t weightId = graph.allocateId();
        uint64_t offset = memManager.allocate(Backend::CPU, weightId, meta.sizeBytes(), StorageType::PERSISTENT);

        // Load raw data and write (Sparse logic automatically kicks in because init() hasn't been called)
        std::vector<uint8_t> temp(meta.sizeBytes());
        loader.loadTensor(weightName, temp.data(), temp.size());
        memManager.write(Backend::CPU, weightId, temp.data(), temp.size());

        // 2. User gets view
        TensorView view;
        view.baseOffset = offset;
        view.shape = meta.shape;
        view.strides = TensorView::calcContiguousStrides(meta.shape);

        // 3. User creates graph input using that view
        graph.inputWithId(weightId, meta.shape, meta.dtype, view);

        std::cout << "Successfully loaded tensor '" << weightName << "' into sparse data structure." << std::endl;

        // Hash reading directly via sparse value, and unload avoiding permanent footprint during Planning
        std::string structuralHash = Hashing::getStructuralHash(weightId, graph, memManager);
        memManager.unload(Backend::CPU, weightId);
        std::cout << "Hashed and unloaded weight effectively. Hash: " << structuralHash << std::endl;

        // Finish Planning Sequence, when done, actualize Memory Arena
        memManager.buffers.at(Backend::CPU).init();
    }

    return 0;
}