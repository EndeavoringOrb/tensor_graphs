#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <algorithm>
#include <limits>
#include <memory>

namespace py = pybind11;

struct ParallelBuffer {
    uint32_t id;
    int64_t size;
    float start_time;
    float end_time;
};

bool overlapsBuf(const ParallelBuffer &a, const ParallelBuffer &b) {
    float ax = a.start_time, ay = a.end_time;
    float bx = b.start_time, by = b.end_time;
    if (bx < ax) { std::swap(ax, bx); std::swap(ay, by); }
    return bx < ay;
}

// Struct to store exact delta changes for blazing fast backtracking
struct UndoRecord {
    int action;
    int64_t prev_global_offset_max;
    int64_t prev_last_chosen_offset;
    uint32_t prev_last_chosen_id;
    std::vector<std::pair<int, int64_t>> offset_changes; // {neighbor_idx, old_offset}
};

class MallocEnv {
public:
    int N;
    uint64_t mem_cap;
    std::vector<ParallelBuffer> buffers;
    std::vector<std::vector<int>> adj;

    // State Variables
    std::vector<bool> allocated;
    std::vector<int64_t> current_offsets;
    int64_t global_offset_max;
    int64_t last_chosen_offset;
    uint32_t last_chosen_id;
    int num_allocated;
    
    // History Stack for backtracking
    std::vector<UndoRecord> history;

    MallocEnv(uint64_t cap, const std::vector<ParallelBuffer>& bufs) 
        : mem_cap(cap), buffers(bufs) {
        N = buffers.size();
        adj.resize(N);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                if (i != j && overlapsBuf(buffers[i], buffers[j])) {
                    adj[i].push_back(j);
                }
            }
        }
        reset();
    }

    void reset() {
        allocated.assign(N, false);
        current_offsets.assign(N, 0);
        global_offset_max = 0;
        last_chosen_offset = -1;
        last_chosen_id = 0;
        num_allocated = 0;
        history.clear();
    }

    std::shared_ptr<MallocEnv> clone() const {
        return std::make_shared<MallocEnv>(*this);
    }

    std::vector<bool> get_valid_actions() const {
        std::vector<bool> valid(N, false);
        
        int64_t h_min = std::numeric_limits<int64_t>::max();
        for (int i = 0; i < N; ++i) {
            if (!allocated[i]) {
                int64_t h = current_offsets[i] + buffers[i].size;
                if (h < h_min) h_min = h;
            }
        }

        for (int i = 0; i < N; ++i) {
            if (allocated[i]) continue;
            
            int64_t offset_i = current_offsets[i];
            if (offset_i < global_offset_max) continue;
            if (offset_i == last_chosen_offset && buffers[i].id < last_chosen_id) continue;
            if (offset_i >= h_min) continue;
            if (mem_cap != std::numeric_limits<uint64_t>::max() &&
                static_cast<uint64_t>(offset_i) + buffers[i].size > mem_cap) continue;

            valid[i] = true;
        }
        return valid;
    }

    // Returns a tuple: (reward, done)
    std::pair<float, bool> step(int action) {
        if (allocated[action]) return {-1.0f, true};

        // Snapshot current state for backtrack
        UndoRecord rec;
        rec.action = action;
        rec.prev_global_offset_max = global_offset_max;
        rec.prev_last_chosen_offset = last_chosen_offset;
        rec.prev_last_chosen_id = last_chosen_id;

        int64_t offset_i = current_offsets[action];
        allocated[action] = true;
        num_allocated++;
        
        global_offset_max = std::max(global_offset_max, offset_i);
        last_chosen_offset = offset_i;
        last_chosen_id = buffers[action].id;

        int64_t new_end = offset_i + buffers[action].size;
        for (int j : adj[action]) {
            if (current_offsets[j] < new_end) {
                // Record previous offset of neighbor before overwriting
                rec.offset_changes.push_back({j, current_offsets[j]});
                current_offsets[j] = new_end;
            }
        }
        history.push_back(std::move(rec));

        if (num_allocated == N) {
            return {1.0f, true}; // Fully allocated -> Global optimal success
        }

        auto valid = get_valid_actions();
        bool has_valid = false;
        for (bool v : valid) { if (v) { has_valid = true; break; } }

        if (!has_valid) {
            return {-1.0f, true}; // OOM / No valid actions -> Dead End
        }

        return {0.0f, false}; // Keep going
    }

    // Instantly restores the previous state using the history stack
    void undo() {
        if (history.empty()) return;
        
        const UndoRecord& rec = history.back();
        allocated[rec.action] = false;
        num_allocated--;
        
        global_offset_max = rec.prev_global_offset_max;
        last_chosen_offset = rec.prev_last_chosen_offset;
        last_chosen_id = rec.prev_last_chosen_id;
        
        for (const auto& change : rec.offset_changes) {
            current_offsets[change.first] = change.second;
        }
        
        history.pop_back();
    }

    // Export Nx3 state (size, current_offset, is_allocated)
    py::array_t<float> get_state() const {
        auto result = py::array_t<float>({N, 5});
        auto buf = result.request();
        float* ptr = static_cast<float*>(buf.ptr);
        for (int i = 0; i < N; ++i) {
            ptr[i*5 + 0] = static_cast<float>(buffers[i].size);
            ptr[i*5 + 1] = static_cast<float>(buffers[i].start_time);
            ptr[i*5 + 2] = static_cast<float>(buffers[i].end_time);
            ptr[i*5 + 3] = static_cast<float>(current_offsets[i]);
            ptr[i*5 + 4] = allocated[i] ? 1.0f : 0.0f;
        }
        return result;
    }
};

PYBIND11_MODULE(malloc_rl, m) {
    py::class_<ParallelBuffer>(m, "ParallelBuffer")
        .def(py::init<uint32_t, int64_t, float, float>());
        
    py::class_<MallocEnv, std::shared_ptr<MallocEnv>>(m, "MallocEnv")
        .def(py::init<uint64_t, const std::vector<ParallelBuffer>&>())
        .def("reset", &MallocEnv::reset)
        .def("clone", &MallocEnv::clone)
        .def("get_valid_actions", &MallocEnv::get_valid_actions)
        .def("step", &MallocEnv::step)
        .def("undo", &MallocEnv::undo)
        .def("get_state", &MallocEnv::get_state)
        .def_readonly("N", &MallocEnv::N)
        .def_readonly("num_allocated", &MallocEnv::num_allocated);
}