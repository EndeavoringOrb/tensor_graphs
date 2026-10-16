#pragma once

#include <functional>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>

// Graph copies inherit immutable memo entries and write into their own layer.
// Values may refer to graph IDs, so clear the memo before renumbering those IDs.
template <typename Key, typename Value, typename Hash = std::hash<Key>>
class SharedMemo
{
    struct Layer
    {
        std::unordered_map<Key, Value, Hash> entries;
        std::shared_ptr<const Layer> parent;
    };
    std::shared_ptr<Layer> layer_;

  public:
    const Value *find(const Key &key) const
    {
        for (const Layer *layer = layer_.get(); layer; layer = layer->parent.get())
        {
            auto found = layer->entries.find(key);
            if (found != layer->entries.end())
                return &found->second;
        }
        return nullptr;
    }

    void insert(Key key, Value value)
    {
        if (find(key))
            return;
        if (!layer_ || !layer_.unique())
        {
            auto next_layer = std::make_shared<Layer>();
            next_layer->parent = layer_;
            layer_ = std::move(next_layer);
        }
        layer_->entries.emplace(std::move(key), std::move(value));
    }

    void insertOrAssign(Key key, Value value)
    {
        if (!layer_ || !layer_.unique())
        {
            auto next_layer = std::make_shared<Layer>();
            next_layer->parent = layer_;
            layer_ = std::move(next_layer);
        }
        layer_->entries.insert_or_assign(std::move(key), std::move(value));
    }

    template <typename Visitor> void forEach(Visitor visitor) const
    {
        std::unordered_set<Key, Hash> seen;
        for (const Layer *layer = layer_.get(); layer; layer = layer->parent.get())
            for (const auto &[key, value] : layer->entries)
                if (seen.insert(key).second)
                    visitor(key, value);
    }

    void clear()
    {
        layer_.reset();
    }
};
