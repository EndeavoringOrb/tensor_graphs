#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "core/logging.hpp"

class ThreadPool
{
public:
    static ThreadPool &get()
    {
        static ThreadPool instance;
        return instance;
    }

    void set_num_threads(uint32_t n)
    {
        if (n == 0)
        {
            n = std::max(1U, std::thread::hardware_concurrency());
        }

        std::unique_lock<std::mutex> lock(queue_mutex);
        if (n == num_threads_ && (threads.size() == n - 1 || n <= 1))
            return;

        stop = true;
        condition.notify_all();
        lock.unlock();

        for (std::thread &worker : threads)
        {
            if (worker.joinable())
                worker.join();
        }

        lock.lock();
        threads.clear();
        while (!tasks.empty())
            tasks.pop();
        stop = false;
        num_threads_ = n;

        if (num_threads_ > 1)
        {
            for (uint32_t i = 0; i < num_threads_ - 1; ++i)
            {
                threads.emplace_back([this]
                                     {
                    while (true)
                    {
                        std::function<void()> task;
                        {
                            std::unique_lock<std::mutex> lock(this->queue_mutex);
                            this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });
                            if (this->stop && this->tasks.empty())
                                return;
                            task = std::move(this->tasks.front());
                            this->tasks.pop();
                        }
                        task();
                    } });
            }
        }
    }

    uint32_t get_num_threads() const
    {
        return num_threads_;
    }

    void parallel_for(uint32_t num_tasks, const std::function<void(uint32_t)> &task)
    {
        if (num_tasks == 0)
            return;
        if (num_tasks == 1 || num_threads_ <= 1 || threads.empty())
        {
            for (uint32_t i = 0; i < num_tasks; ++i)
            {
                task(i);
            }
            return;
        }

        struct State
        {
            std::atomic<uint32_t> counter{0};
            std::atomic<uint32_t> completed{0};
            std::function<void(uint32_t)> task;
        };
        auto state = std::make_shared<State>();
        state->task = task;

        auto worker_task = [state, num_tasks]()
        {
            while (true)
            {
                uint32_t idx = state->counter.fetch_add(1, std::memory_order_relaxed);
                if (idx >= num_tasks)
                    break;
                try
                {
                    state->task(idx);
                }
                catch (const std::exception &e)
                {
                    LOG(ERROR) << "\n[ThreadPool Error in Task " << idx << "]: " << e.what() << std::endl;
                }
                catch (...)
                {
                    LOG(ERROR) << "\n[ThreadPool Unknown Fatal Exception in Task " << idx << "]" << std::endl;
                }
                state->completed.fetch_add(1, std::memory_order_release);
            }
        };

        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            uint32_t num_to_push = std::min(static_cast<uint32_t>(threads.size()), num_tasks);
            for (uint32_t i = 0; i < num_to_push; ++i)
            {
                tasks.push(worker_task);
            }
        }
        condition.notify_all();

        // Main thread helps out
        worker_task();

        // Spin-wait yield to minimize dispatch latency
        while (state->completed.load(std::memory_order_acquire) < num_tasks)
        {
            std::this_thread::yield();
        }
    }

private:
    ThreadPool() : stop(false)
    {
        set_num_threads(0);
    }

    ~ThreadPool()
    {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread &worker : threads)
        {
            if (worker.joinable())
                worker.join();
        }
    }

    std::vector<std::thread> threads;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
    uint32_t num_threads_;
};

inline void set_num_threads(uint32_t n)
{
    ThreadPool::get().set_num_threads(n);
}

inline uint32_t get_num_threads()
{
    return ThreadPool::get().get_num_threads();
}