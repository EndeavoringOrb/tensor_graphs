// tensor_graphs_cpp/core/plan/selector.hpp
#pragma once

#include <memory>
#include <queue>
#include <vector>

#include "core/plan/search_node.hpp"

namespace plan
{

class Selector
{
  public:
    virtual ~Selector() = default;
    virtual void push(const std::shared_ptr<SearchNode> &node) = 0;
    virtual std::shared_ptr<SearchNode> pop() = 0;
    virtual bool empty() const = 0;
    virtual size_t size() const = 0;
    virtual void clear() = 0;
};

class PriorityQueueSelector : public Selector
{
  private:
    std::priority_queue<std::shared_ptr<SearchNode>, std::vector<std::shared_ptr<SearchNode>>, SearchNodeCompare> queue;

  public:
    void push(const std::shared_ptr<SearchNode> &node) override
    {
        queue.push(node);
    }

    std::shared_ptr<SearchNode> pop() override
    {
        if (queue.empty())
            return nullptr;
        auto top = queue.top();
        queue.pop();
        return top;
    }

    bool empty() const override
    {
        return queue.empty();
    }

    size_t size() const override
    {
        return queue.size();
    }

    void clear() override
    {
        while (!queue.empty())
            queue.pop();
    }
};

class LIFOSelector : public Selector
{
  private:
    std::vector<std::shared_ptr<SearchNode>> stack;

  public:
    void push(const std::shared_ptr<SearchNode> &node) override
    {
        stack.push_back(node);
    }

    std::shared_ptr<SearchNode> pop() override
    {
        if (stack.empty())
            return nullptr;
        auto top = stack.back();
        stack.pop_back();
        return top;
    }

    bool empty() const override
    {
        return stack.empty();
    }

    size_t size() const override
    {
        return stack.size();
    }

    void clear() override
    {
        stack.clear();
    }
};

} // namespace plan
