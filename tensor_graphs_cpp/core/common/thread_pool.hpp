// File: tensor_graphs_cpp/core/common/thread_pool.hpp
#pragma once
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

class ThreadPool
{
public:
    static ThreadPool &get()
    {
        static ThreadPool instance;
        return instance;
    }

    void parallel_for(uint32_t num_tasks, const std::function<void(uint32_t)> &task)
    {
        if (num_tasks == 0)
            return;
        if (num_tasks == 1)
        {
            task(0);
            return;
        }

        std::atomic<uint32_t> counter{0};
        std::atomic<uint32_t> completed{0};

        auto worker_task = [&]()
        {
            while (true)
            {
                uint32_t idx = counter.fetch_add(1, std::memory_order_relaxed);
                if (idx >= num_tasks)
                    break;
                task(idx);
                completed.fetch_add(1, std::memory_order_release);
            }
        };

        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            for (uint32_t i = 0; i < threads.size(); ++i)
            {
                tasks.push(worker_task);
            }
        }
        condition.notify_all();

        // Main thread helps out
        worker_task();

        // Spin-wait yield to minimize dispatch latency
        while (completed.load(std::memory_order_acquire) < num_tasks)
        {
            std::this_thread::yield();
        }
    }

private:
    ThreadPool() : stop(false)
    {
        uint32_t num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0)
            num_threads = 1;

        for (uint32_t i = 0; i < num_threads - 1; ++i)
        {
            threads.emplace_back([this]
                                 {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });
                        if (this->stop && this->tasks.empty()) return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                } });
        }
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
            worker.join();
        }
    }

    std::vector<std::thread> threads;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};