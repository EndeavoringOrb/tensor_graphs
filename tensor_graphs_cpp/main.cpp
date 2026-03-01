#include <core.hpp>
#include <kernels/kernels.hpp>

int main()
{
    // 1. Setup MemoryManager with buffer sizing
    MemoryManager memManager;
    memManager.buffers.emplace(Backend::CPU, DeviceBuffer(1024 * 1024 * 1024)); // 1GB buffer

    // 2. Open and inspect SafeTensors without bringing data to memory yet
    SafetensorsLoader loader("C:/Users/aaron/CODING/tensor_graphs/resources/model.safetensors");
    std::string weightName = "model.embed_tokens.weight";

    if (loader.hasTensor(weightName))
    {
        const auto &meta = loader.getMetadata(weightName);

        // 3. Plan memory allocation (Graph building phase)
        uint32_t dummyNodeId = 0;
        uint64_t offset = memManager.allocate(Backend::CPU, dummyNodeId, meta.sizeBytes(), StorageType::PERSISTENT);

        // 4. Initialize physical memory
        memManager.buffers.at(Backend::CPU).init();

        // 5. Tell the loader to blast bytes directly to the reserved section
        uint8_t *targetPtr = memManager.buffers.at(Backend::CPU).arena.data() + offset;
        loader.loadTensor(weightName, targetPtr, meta.sizeBytes());

        std::cout << "Successfully loaded tensor '" << weightName << "' (" << meta.sizeBytes() << " bytes)." << std::endl;
    }

    return 0;
}