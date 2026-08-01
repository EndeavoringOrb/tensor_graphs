// tensor_graphs_cpp/chat.cpp
#include <algorithm>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

#include "core/graph.hpp"
#include "core/memory.hpp"
#include "core/session.hpp"
#include "models/kimi-k3.hpp"

#ifdef TG_OS_WINDOWS
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

static constexpr int MAX_SEQ_LEN = 4096;

#pragma pack(push, 1)
struct SharedMemoryPayload
{
    volatile int32_t state;
    int32_t num_input_tokens;
    int32_t input_tokens[8192];
    int32_t output_token;
    int32_t has_image;
    int32_t grid_t;
    int32_t grid_h;
    int32_t grid_w;
    int32_t num_patches;
    // pixel_values starts directly after at offset 32800.
    float pixel_values[1];
};
#pragma pack(pop)

int32_t argmax(const float *logits, uint32_t size)
{
    float max_val = -1e9f;
    int32_t max_idx = 0;
    for (uint32_t i = 0; i < size; ++i)
    {
        if (logits[i] > max_val)
        {
            max_val = logits[i];
            max_idx = i;
        }
    }
    return max_idx;
}

int main()
{
    std::cout << "Initializing Kimi-K3 Chat Server..." << std::endl;

    // Setup Shared Memory
    size_t shm_size = 64 * 1024 * 1024; // 64MB
    SharedMemoryPayload *payload = nullptr;

#ifdef TG_OS_WINDOWS
    HANDLE hMapFile = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, shm_size, "kimi_k3_shm");
    if (!hMapFile)
        return 1;
    payload = (SharedMemoryPayload *)MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, shm_size);
#else
    int shm_fd = shm_open("/kimi_k3_shm", O_CREAT | O_RDWR, 0666);
    if (shm_fd < 0)
        return 1;
    ftruncate(shm_fd, shm_size);
    payload = (SharedMemoryPayload *)mmap(NULL, shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
#endif

    payload->state = 0;

    KimiK3Config cfg;
    std::unordered_map<MemSpace, uint64_t> bufSizes = {{MemSpace{1, HandleType::CPP}, 32ULL * 1024 * 1024 * 1024}};
    MemoryManager mem(bufSizes);

    std::cout << "Building Text Graph..." << std::endl;
    Graph text_g;
    KimiK3Model text_model(cfg, MAX_SEQ_LEN, text_g, mem, "models_meta/moonshotai/kimi-k3");
    LogicalId logits_id = text_model.build_text_graph();

    LogicalId input_ids = text_model.input_ids_id;
    LogicalId image_indices = text_model.image_indices_id;
    LogicalId image_features = text_model.image_features_id;

    Session text_session(text_g, mem, logits_id, "dirty_region_caches/kimi-k3-text.bin", 0, nullptr, false, 0.0f);
    text_session.compile();

    std::vector<int32_t> host_input_ids(MAX_SEQ_LEN, 0);
    std::vector<int32_t> host_image_indices(MAX_SEQ_LEN, 0);
    std::vector<float> host_image_features(MAX_SEQ_LEN * cfg.hidden_size, 0.0f);

    int last_t = 0, last_h = 0, last_w = 0;
    std::unique_ptr<Graph> vision_g;
    std::unique_ptr<KimiK3Model> vision_model;
    std::unique_ptr<Session> vision_session;

    std::cout << "Listening for Python UI..." << std::endl;

    while (true)
    {
        if (payload->state == 1)
        { // Python wrote inputs
            std::cout << "Received prompt (" << payload->num_input_tokens << " tokens)..." << std::endl;

            if (payload->has_image)
            {
                if (payload->grid_t != last_t || payload->grid_h != last_h || payload->grid_w != last_w)
                {
                    std::cout << "Rebuilding vision graph for " << payload->grid_t << "x" << payload->grid_h << "x"
                              << payload->grid_w << std::endl;
                    vision_g = std::make_unique<Graph>();
                    vision_model = std::make_unique<KimiK3Model>(cfg, 1, *vision_g, mem, "models/moonshotai/kimi-k3");
                    LogicalId v_out = vision_model->build_vision_graph(payload->num_patches, payload->grid_t,
                                                                       payload->grid_h, payload->grid_w);
                    vision_session =
                        std::make_unique<Session>(*vision_g, mem, v_out, "dirty_region_caches/kimi-k3-vision.bin");
                    vision_session->compile();
                    last_t = payload->grid_t;
                    last_h = payload->grid_h;
                    last_w = payload->grid_w;
                }

                // Write pixel values
                LogicalId pixel_input_id = LogicalId{0};
                vision_session->writeInput(pixel_input_id, payload->pixel_values,
                                           payload->num_patches * 3 * cfg.patch_size * cfg.patch_size * sizeof(float));
                Bucket vb;
                const float *dev_feats = static_cast<const float *>(vision_session->run(vb));
                uint32_t merged_patches = (payload->grid_h / 2) * (payload->grid_w / 2);
                std::memcpy(host_image_features.data(), dev_feats, merged_patches * cfg.hidden_size * sizeof(float));
            }

            uint32_t img_idx = 0;
            for (uint32_t i = 0; i < payload->num_input_tokens; ++i)
            {
                int32_t token = payload->input_tokens[i];
                host_input_ids[i] = token;
                if (token == 163605 && payload->has_image)
                {
                    host_image_indices[i] = img_idx++;
                }
                else
                {
                    host_image_indices[i] = 0;
                }
            }

            std::vector<int32_t> tokens(payload->input_tokens, payload->input_tokens + payload->num_input_tokens);

            for (uint32_t step = payload->num_input_tokens; step < MAX_SEQ_LEN; ++step)
            {
                text_session.writeInput(input_ids, host_input_ids.data(), host_input_ids.size() * sizeof(int32_t));
                text_session.writeInput(image_indices, host_image_indices.data(), host_image_indices.size() * sizeof(int32_t));
                text_session.writeInput(image_features, host_image_features.data(), host_image_features.size() * sizeof(float));

                Bucket b;
                if (step == payload->num_input_tokens)
                {
                    b.inputDirtyRegions[input_ids] = {Region{{{0, 1}, {0, (uint32_t)tokens.size()}}}};
                    b.inputDirtyRegions[image_indices] = {Region{{{0, 1}, {0, (uint32_t)tokens.size()}}}};
                    b.inputDirtyRegions[image_features] = {Region{{{0, (uint32_t)MAX_SEQ_LEN}, {0, cfg.hidden_size}}}};
                }
                else
                {
                    uint32_t tokIdx = tokens.size() - 1;
                    b.inputDirtyRegions[input_ids] = {Region{{{0, 1}, {tokIdx, tokIdx + 1}}}};
                    b.inputDirtyRegions[image_indices] = {Region{{{0, 1}, {tokIdx, tokIdx + 1}}}};
                }

                b.outputNeededRegion = {Region{{{0, 1}, {(uint32_t)tokens.size() - 1, (uint32_t)tokens.size()}, {0, cfg.vocab_size}}}};

                const float *logits_raw = static_cast<const float *>(text_session.run(b));
                const float *token_logits = logits_raw + ((tokens.size() - 1) * cfg.vocab_size);

                int32_t next_token = argmax(token_logits, cfg.vocab_size);
                tokens.push_back(next_token);

                if (next_token == 163586)
                    break; // EOS

                host_input_ids[tokens.size() - 1] = next_token;
                host_image_indices[tokens.size() - 1] = 0;

                payload->output_token = next_token;
                payload->state = 3;
                while (payload->state != 4)
                {
                    if (payload->state == 5)
                        exit(0);
                    std::this_thread::yield();
                }
            }
            payload->state = 5;
        }
        else if (payload->state == 5)
        {
            break;
        }
        std::this_thread::yield();
    }

    return 0;
}