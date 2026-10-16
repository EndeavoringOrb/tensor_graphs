#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <unordered_set>
#include <utility>

// Copies share completed visit history. Each writer adds an independent layer
// before changing a shared layer, so seeded saturation does not copy millions
// of keys for every bucket. Layers are never mutated while shared.
template <typename Key, typename Hash = std::hash<Key>>
class SharedVisitedSet
{
    struct Layer
    {
        std::unordered_set<Key, Hash> entries;
        std::shared_ptr<const Layer> parent;
        std::size_t total_size = 0;
    };

    std::shared_ptr<Layer> layer_;

  public:
    bool contains(const Key &key) const
    {
        for (const Layer *layer = layer_.get(); layer; layer = layer->parent.get())
            if (layer->entries.count(key))
                return true;
        return false;
    }

    bool insert(Key key)
    {
        if (!layer_ || !layer_.unique())
        {
            if (contains(key))
                return false;
            auto next_layer = std::make_shared<Layer>();
            next_layer->parent = layer_;
            next_layer->total_size = size();
            layer_ = std::move(next_layer);
        }
        else
        {
            for (const Layer *parent = layer_->parent.get(); parent; parent = parent->parent.get())
                if (parent->entries.count(key))
                    return false;
        }
        const bool inserted = layer_->entries.insert(std::move(key)).second;
        if (inserted)
            ++layer_->total_size;
        return inserted;
    }

    std::size_t size() const
    {
        return layer_ ? layer_->total_size : 0;
    }

    template <typename Visitor>
    void forEach(Visitor visitor) const
    {
        for (const Layer *layer = layer_.get(); layer; layer = layer->parent.get())
            for (const Key &key : layer->entries)
                visitor(key);
    }
};
