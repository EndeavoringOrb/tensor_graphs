// tensor_graphs_cpp/core/plan/pruning.hpp
#pragma once

#include <cstddef>
#include <string>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <utility>

#ifdef TG_PROFILE
#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <unordered_map>
#endif

#define TG_PRUNING_RULE(Name)                                                                                          \
    static constexpr const char *kName = #Name;                                                                        \
    const char *name() const                                                                                           \
    {                                                                                                                  \
        return kName;                                                                                                  \
    }                                                                                                                  \
    bool enabled = true;

namespace prune
{

// =============================================================================
// Detection idiom (stdlib fundamentals TS v2 is_detected, reimplemented)
// =============================================================================
template <class Void, template <class...> class Op, class... Args> struct detector : std::false_type
{
};

template <template <class...> class Op, class... Args>
struct detector<std::void_t<Op<Args...>>, Op, Args...> : std::true_type
{
};

template <template <class...> class Op, class... Args> using is_detected = detector<void, Op, Args...>;

template <template <class...> class Op, class... Args>
inline constexpr bool is_detected_v = is_detected<Op, Args...>::value;

// =============================================================================
// Hook probes
// =============================================================================
template <class R, class Ctx> using init_expr = decltype(std::declval<R &>().init(std::declval<const Ctx &>()));

template <class R, class Node, class Ctx>
using push_expr = decltype(std::declval<R &>().on_push(std::declval<Node>(), std::declval<const Ctx &>()));

template <class R, class Node, class Ctx>
using pop_expr = decltype(std::declval<R &>().on_pop(std::declval<Node>(), std::declval<const Ctx &>()));

template <class R, class Cand, class Ctx>
using check_expr =
    decltype(std::declval<R &>().check(std::declval<Cand>(), std::size_t{}, std::declval<const Ctx &>()));

template <class R, class Ctx>
using leaf_expr = decltype(std::declval<R &>().validate_leaf(std::declval<const Ctx &>()));

template <class R> using name_expr = decltype(std::declval<const R &>().name());
template <class R> using static_kname_expr = decltype(R::kName);

template <class R, class Ctx> constexpr bool has_init_v = is_detected_v<init_expr, R, Ctx>;
template <class R, class Node, class Ctx> constexpr bool has_push_v = is_detected_v<push_expr, R, Node, Ctx>;
template <class R, class Node, class Ctx> constexpr bool has_pop_v = is_detected_v<pop_expr, R, Node, Ctx>;
template <class R, class Cand, class Ctx> constexpr bool has_check_v = is_detected_v<check_expr, R, Cand, Ctx>;
template <class R, class Ctx> constexpr bool has_leaf_v = is_detected_v<leaf_expr, R, Ctx>;
template <class R> constexpr bool has_name_v = is_detected_v<name_expr, R>;
template <class R> constexpr bool has_static_kname_v = is_detected_v<static_kname_expr, R>;

template <typename R> constexpr const char *rule_name_v()
{
    if constexpr (has_static_kname_v<R>)
        return R::kName;
    else if constexpr (has_name_v<R>)
        return R{}.name();
    else
        return typeid(R).name();
}

struct RuleSpec
{
    const char *category;
    const char *rule_name;
};

template <typename Tuple, size_t... Is>
constexpr auto make_category_specs_impl(const char *category, std::index_sequence<Is...>)
{
    using std::tuple_element_t;
    return std::array<RuleSpec, sizeof...(Is)>{RuleSpec{category, rule_name_v<tuple_element_t<Is, Tuple>>()}...};
}

template <typename Tuple> constexpr auto make_category_specs(const char *category)
{
    return make_category_specs_impl<Tuple>(category, std::make_index_sequence<std::tuple_size_v<Tuple>>{});
}

template <typename T, size_t N1, size_t N2>
constexpr std::array<T, N1 + N2> concat_arrays(const std::array<T, N1> &a, const std::array<T, N2> &b)
{
    std::array<T, N1 + N2> result{};
    for (size_t i = 0; i < N1; ++i)
        result[i] = a[i];
    for (size_t i = 0; i < N2; ++i)
        result[N1 + i] = b[i];
    return result;
}

template <typename T, size_t N1, size_t N2, typename... Rest>
constexpr auto concat_arrays(const std::array<T, N1> &a, const std::array<T, N2> &b, const Rest &...rest)
{
    return concat_arrays(concat_arrays(a, b), rest...);
}

template <typename Tuple, typename SettingsT, size_t... Is>
auto extract_enabled_states_impl(const std::string &category, const SettingsT &settings, std::index_sequence<Is...>)
{
    using std::tuple_element_t;
    return std::make_tuple(settings.is_rule_enabled(category, rule_name_v<tuple_element_t<Is, Tuple>>())...);
}

template <typename Tuple, typename SettingsT>
auto extract_enabled_states(const std::string &category, const SettingsT &settings)
{
    constexpr size_t N = std::tuple_size_v<Tuple>;
    return extract_enabled_states_impl<Tuple>(category, settings, std::make_index_sequence<N>{});
}

template <typename RulesTuple, typename BoolTuple, size_t... Is>
auto instantiate_from_bools_impl(const BoolTuple &bools, std::index_sequence<Is...>)
{
    using std::tuple_element_t;
    return std::make_tuple(tuple_element_t<Is, RulesTuple>(std::get<Is>(bools))...);
}

template <typename RulesTuple, typename BoolTuple> auto instantiate_from_bools(const BoolTuple &bools)
{
    constexpr size_t N = std::tuple_size_v<RulesTuple>;
    return instantiate_from_bools_impl<RulesTuple>(bools, std::make_index_sequence<N>{});
}

template <typename RulesTuple, typename SettingsT>
auto instantiate_rules(const std::string &category, const SettingsT &settings)
{
    auto bools = extract_enabled_states<RulesTuple>(category, settings);
    return instantiate_from_bools<RulesTuple>(bools);
}

// =============================================================================
// Profiler Infrastructure
// =============================================================================
#ifdef TG_PROFILE
struct RuleTimingStats
{
    uint64_t init_ns = 0;
    uint64_t init_calls = 0;

    uint64_t check_ns = 0;
    uint64_t check_calls = 0;
    uint64_t check_pruned = 0;

    uint64_t push_ns = 0;
    uint64_t push_calls = 0;

    uint64_t pop_ns = 0;
    uint64_t pop_calls = 0;

    uint64_t leaf_ns = 0;
    uint64_t leaf_calls = 0;
};

class PruningProfiler
{
  private:
    mutable std::mutex mtx;
    std::unordered_map<std::string, RuleTimingStats> stats_by_rule;

  public:
    static PruningProfiler &get()
    {
        static PruningProfiler instance;
        return instance;
    }

    void record_init(const std::string &name, uint64_t ns)
    {
        std::lock_guard<std::mutex> lock(mtx);
        auto &s = stats_by_rule[name];
        s.init_ns += ns;
        s.init_calls++;
    }

    void record_check(const std::string &name, uint64_t ns, bool pruned)
    {
        std::lock_guard<std::mutex> lock(mtx);
        auto &s = stats_by_rule[name];
        s.check_ns += ns;
        s.check_calls++;
        if (pruned)
            s.check_pruned++;
    }

    void record_push(const std::string &name, uint64_t ns)
    {
        std::lock_guard<std::mutex> lock(mtx);
        auto &s = stats_by_rule[name];
        s.push_ns += ns;
        s.push_calls++;
    }

    void record_pop(const std::string &name, uint64_t ns)
    {
        std::lock_guard<std::mutex> lock(mtx);
        auto &s = stats_by_rule[name];
        s.pop_ns += ns;
        s.pop_calls++;
    }

    void record_leaf(const std::string &name, uint64_t ns)
    {
        std::lock_guard<std::mutex> lock(mtx);
        auto &s = stats_by_rule[name];
        s.leaf_ns += ns;
        s.leaf_calls++;
    }

    void printSummary() const
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (stats_by_rule.empty())
            return;

        std::cout << "\n=========================================================================================\n";
        std::cout << " [Pruning Rule Profiling Report]\n";
        std::cout << "=========================================================================================\n";
        std::cout << std::left << std::setw(36) << "Rule Name" << std::right << std::setw(12) << "Check (ms)"
                  << std::setw(12) << "Check Calls" << std::setw(10) << "Pruned" << std::setw(12) << "Push/Pop(ms)"
                  << std::setw(12) << "Total (ms)\n";
        std::cout << std::string(94, '-') << "\n";

        for (const auto &[name, s] : stats_by_rule)
        {
            double check_ms = s.check_ns / 1e6;
            double push_pop_ms = (s.push_ns + s.pop_ns) / 1e6;
            double total_ms = (s.init_ns + s.check_ns + s.push_ns + s.pop_ns + s.leaf_ns) / 1e6;
            std::cout << std::left << std::setw(36) << name.substr(0, 35) << std::right << std::setw(12) << std::fixed
                      << std::setprecision(2) << check_ms << std::setw(12) << s.check_calls << std::setw(10)
                      << s.check_pruned << std::setw(12) << std::fixed << std::setprecision(2) << push_pop_ms
                      << std::setw(12) << std::fixed << std::setprecision(2) << total_ms << "\n";
        }
        std::cout << "=========================================================================================\n\n"
                  << std::flush;
    }
};

inline void printPruningProfileSummary()
{
    PruningProfiler::get().printSummary();
}
#else
inline void printPruningProfileSummary()
{
}
#endif

// =============================================================================
// PruningRuleSet -- compile-time list of rules with zero-overhead dispatch.
// =============================================================================
template <typename... Rules> struct PruningRuleSet
{
    std::tuple<Rules...> rules;

    PruningRuleSet() = default;

    explicit PruningRuleSet(std::tuple<Rules...> t) : rules(std::move(t))
    {
    }

    template <typename First, typename... Rest,
              std::enable_if_t<!std::is_same_v<std::decay_t<First>, std::tuple<Rules...>>, int> = 0>
    explicit PruningRuleSet(First &&first, Rest &&...rest)
        : rules(std::forward<First>(first), std::forward<Rest>(rest)...)
    {
    }

    static constexpr bool empty()
    {
        return sizeof...(Rules) == 0;
    }

    static constexpr size_t size()
    {
        return sizeof...(Rules);
    }

    template <class Ctx> void init(const Ctx &ctx)
    {
        std::apply([&](auto &...rs) { (init_one(rs, ctx), ...); }, rules);
    }

    template <class Node, class Ctx> void on_push(Node node, const Ctx &ctx)
    {
        std::apply([&](auto &...rs) { (push_one(rs, node, ctx), ...); }, rules);
    }

    template <class Node, class Ctx> void on_pop(Node node, const Ctx &ctx)
    {
        std::apply([&](auto &...rs) { (pop_one(rs, node, ctx), ...); }, rules);
    }

    template <class Cand, class Ctx> bool is_pruned(Cand candidate, size_t candidate_idx, const Ctx &ctx)
    {
        return std::apply([&](auto &...rs) { return (false || ... || check_one(rs, candidate, candidate_idx, ctx)); },
                          rules);
    }

    template <class Ctx> bool validate_leaf(const Ctx &ctx)
    {
        return std::apply([&](auto &...rs) { return (true && ... && leaf_one(rs, ctx)); }, rules);
    }

    std::string names() const
    {
        std::string out;
        std::apply(
            [&](const auto &...rs) {
                size_t i = 0;
                (((out += (i++ ? "," : "")), (out += name_of(rs))), ...);
            },
            rules);
        return out;
    }

  private:
    template <class R, class Ctx> static void init_one(R &rule, const Ctx &ctx)
    {
        if constexpr (has_init_v<R, Ctx>)
        {
#ifdef TG_PROFILE
            auto t0 = std::chrono::steady_clock::now();
            rule.init(ctx);
            auto t1 = std::chrono::steady_clock::now();
            PruningProfiler::get().record_init(name_of(rule),
                                               std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
#else
            rule.init(ctx);
#endif
        }
    }

    template <class R, class Node, class Ctx> static void push_one(R &rule, Node node, const Ctx &ctx)
    {
        if constexpr (has_push_v<R, Node, Ctx>)
        {
#ifdef TG_PROFILE
            auto t0 = std::chrono::steady_clock::now();
            rule.on_push(node, ctx);
            auto t1 = std::chrono::steady_clock::now();
            PruningProfiler::get().record_push(name_of(rule),
                                               std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
#else
            rule.on_push(node, ctx);
#endif
        }
    }

    template <class R, class Node, class Ctx> static void pop_one(R &rule, Node node, const Ctx &ctx)
    {
        if constexpr (has_pop_v<R, Node, Ctx>)
        {
#ifdef TG_PROFILE
            auto t0 = std::chrono::steady_clock::now();
            rule.on_pop(node, ctx);
            auto t1 = std::chrono::steady_clock::now();
            PruningProfiler::get().record_pop(name_of(rule),
                                              std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
#else
            rule.on_pop(node, ctx);
#endif
        }
    }

    template <class R, class Cand, class Ctx>
    static bool check_one(R &rule, Cand candidate, size_t candidate_idx, const Ctx &ctx)
    {
        if constexpr (has_check_v<R, Cand, Ctx>)
        {
#ifdef TG_PROFILE
            auto t0 = std::chrono::steady_clock::now();
            bool pruned = rule.check(candidate, candidate_idx, ctx);
            auto t1 = std::chrono::steady_clock::now();
            PruningProfiler::get().record_check(
                name_of(rule), std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(), pruned);
            return pruned;
#else
            return rule.check(candidate, candidate_idx, ctx);
#endif
        }
        else
            return false;
    }

    template <class R, class Ctx> static bool leaf_one(R &rule, const Ctx &ctx)
    {
        if constexpr (has_leaf_v<R, Ctx>)
        {
#ifdef TG_PROFILE
            auto t0 = std::chrono::steady_clock::now();
            bool ok = rule.validate_leaf(ctx);
            auto t1 = std::chrono::steady_clock::now();
            PruningProfiler::get().record_leaf(name_of(rule),
                                               std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
            return ok;
#else
            return rule.validate_leaf(ctx);
#endif
        }
        else
            return true;
    }

    template <class R> static std::string name_of(const R &rule)
    {
        if constexpr (has_name_v<R>)
            return rule.name();
        else
            return typeid(R).name();
    }
};

} // namespace prune