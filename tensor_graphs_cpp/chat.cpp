#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "core/argparse.hpp"
#include "core/graph.hpp"
#include "core/memory.hpp"
#include "core/session.hpp"
#include "core/types.hpp"
#include "models/deepseek-v4-flash.hpp"

#ifdef TG_OS_WINDOWS
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#define SHM_MAX_TOKENS 8192

#pragma pack(push, 1)
struct SharedMemoryPayload
{
    volatile int32_t state;
    int32_t input_tokens[SHM_MAX_TOKENS];
    int32_t input_len;
    int32_t output_token;
};
#pragma pack(pop)

int32_t perform_argmax(const float *logits, uint32_t vocab_size)
{
    float max_val = -1e9f;
    int32_t argmax_idx = 0;
    for (uint32_t i = 0; i < vocab_size; ++i)
    {
        if (logits[i] > max_val)
        {
            max_val = logits[i];
            argmax_idx = i;
        }
    }
    return argmax_idx;
}

int main(int argc, char *argv[])
{
    ArgParser parser("chat", "Chat interface server for DeepSeek V4 Flash.");
    parser.add_positional("model_path", "Model directory.");

    if (!parser.parse(argc, argv))
        return 1;

    std::string model_path = parser.get_positional("model_path");

    SharedMemoryPayload *shm_payload = nullptr;
#ifdef TG_OS_WINDOWS
    HANDLE hMapFile =
        CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, sizeof(SharedMemoryPayload), "tg_chat_shm");
    shm_payload =
        (SharedMemoryPayload *)MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(SharedMemoryPayload));
#else
    int shm_fd = shm_open("/tg_chat_shm", O_CREAT | O_RDWR, 0666);
    ftruncate(shm_fd, sizeof(SharedMemoryPayload));
    shm_payload =
        (SharedMemoryPayload *)mmap(NULL, sizeof(SharedMemoryPayload), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
#endif

    std::unordered_map<MemSpace, uint64_t> bufferSizes = {{MemSpace{1, HandleType::CPP}, 32ULL * 1024 * 1024 * 1024}};
    MemoryManager mem(bufferSizes);
    Graph g;

    DeepSeekV4FlashConfig cfg;
    uint32_t max_seq_len = 4096;
    LogicalId inputIdsId = g.input({1, max_seq_len}, DType::INT32);
    DeepSeekV4FlashModel model(cfg, max_seq_len, g, mem, model_path);
    LogicalId logits_id = model.build_graph(inputIdsId);

    std::string gHash = computeGraphHash(g, {logits_id});
    Repo repo("benchmarks/repo_deepseek-v4", gHash, true);
    Session session(g, mem, logits_id, "dirty_region_caches/deepseek-v4-flash.bin", 0, &repo);

    for (uint32_t i = 1; i <= max_seq_len; ++i)
    {
        std::unordered_map<LogicalId, std::vector<Region>> inputDirty;
        Region inR;
        inR.region = {{0, 1}, {i - 1, i}};
        inputDirty[inputIdsId] = {inR};

        Region outR;
        outR.region = {{0, 1}, {i - 1, i}, {0, cfg.vocab_size}};
        session.addBucket(inputDirty, {outR});
    }

    session.compile(true);

    std::cout << "[Server] Ready. Listening on shared memory..." << std::endl;
    shm_payload->state = 0;

    std::vector<int32_t> input_data(max_seq_len, 0);
    std::vector<float> host_output;

    while (true)
    {
        if (shm_payload->state == 1)
        {
            std::vector<uint32_t> tokens;
            for (int i = 0; i < shm_payload->input_len; i++)
                tokens.push_back(shm_payload->input_tokens[i]);

            for (uint32_t step = 0; step < 1000; ++step)
            { // Max 1000 generation steps
                if (tokens.size() >= max_seq_len)
                    break;

                std::fill(input_data.begin(), input_data.end(), 0);
                for (size_t i = 0; i < tokens.size(); i++)
                    input_data[i] = tokens[i];
                session.writeInput(inputIdsId, input_data.data(), input_data.size() * sizeof(int32_t));

                Bucket b;
                uint32_t tokIdx = tokens.size() - 1;
                Region inR;
                inR.region = {{0, 1}, {tokIdx, tokIdx + 1}};
                Region outR;
                outR.region = {{0, 1}, {tokIdx, tokIdx + 1}, {0, cfg.vocab_size}};
                b.inputDirtyRegions = {{inputIdsId, {inR}}};
                b.outputNeededRegion = {outR};

                const float *device_output = static_cast<const float *>(session.run(b));
                host_output.resize(cfg.vocab_size);
                std::memcpy(host_output.data(), device_output + tokIdx * cfg.vocab_size,
                            cfg.vocab_size * sizeof(float));

                int32_t next_token = perform_argmax(host_output.data(), cfg.vocab_size);
                tokens.push_back(next_token);

                shm_payload->output_token = next_token;
                shm_payload->state = 2; // Signal Python token is ready

                while (shm_payload->state == 2)
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                if (shm_payload->state == 5)
                {
                    std::cout << "[Server] Exiting..." << std::endl;
                    return 0;
                }
                if (shm_payload->state == 4)
                    break; // Finished early
            }
            shm_payload->state = 4; // Generation complete
        }
        std::this_thread::sleep_for(std::chrono::microseconds(500));
    }
    return 0;
}