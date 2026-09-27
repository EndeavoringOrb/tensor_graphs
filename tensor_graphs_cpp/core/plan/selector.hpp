// tensor_graphs_cpp/core/plan/selector.hpp
#pragma once

#include <algorithm>
#include <memory>
#include <vector>

#include "core/common/constants.hpp"
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
    virtual void setIncumbent(float cost) {}
    virtual bool hasIncumbent() const { return false; }
};

class PriorityQueueSelector : public Selector
{
  private:
    SearchNodeCompare comp;
    std::vector<std::shared_ptr<SearchNode>> queue;
    float incumbent_cost = TGConstants::INF;

  public:
    void setIncumbent(float cost) override
    {
        incumbent_cost = cost;
        if (!comp.has_incumbent)
        {
            comp.has_incumbent = true;
            queue.erase(std::remove_if(queue.begin(), queue.end(),
                                       [&](const std::shared_ptr<SearchNode> &n) {
                                           return !n || n->lower_bound >= incumbent_cost;
                                       }),
                        queue.end());
            std::make_heap(queue.begin(), queue.end(), comp);
        }
    }

    bool hasIncumbent() const override
    {
        return comp.has_incumbent;
    }

    void push(const std::shared_ptr<SearchNode> &node) override
    {
        if (!node || (comp.has_incumbent && node->lower_bound >= incumbent_cost))
            return;
        queue.push_back(node);
        std::push_heap(queue.begin(), queue.end(), comp);
    }

    std::shared_ptr<SearchNode> pop() override
    {
        while (!queue.empty())
        {
            std::pop_heap(queue.begin(), queue.end(), comp);
            auto top = queue.back();
            queue.pop_back();
            if (top && (!comp.has_incumbent || top->lower_bound < incumbent_cost))
                return top;
        }
        return nullptr;
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
        queue.clear();
        comp.has_incumbent = false;
        incumbent_cost = TGConstants::INF;
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
