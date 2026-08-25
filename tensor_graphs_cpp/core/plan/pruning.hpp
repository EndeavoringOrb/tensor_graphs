// tensor_graphs_cpp/core/plan/pruning.hpp
//
// Unified pruning-rule infrastructure for DFS iterators.
//
// A "pruning rule" is any plain struct that defines ZERO OR MORE of the hooks
// below. Hooks are detected at compile time (C++17 detection idiom) so a rule
// only pays for what it implements -- there is no common base class, no
// vtable, and no indirection: every call is a direct, inlinable call on the
// concrete rule type held inside a std::tuple.
//
//   Hook                              When it runs                     Typical use
//   -------------------------------   ------------------------------  -----------------------------
//   void init(const Ctx&)             BEFORE the DFS starts (once)     Precompute lookup tables,
//                                                                     global invariants of the
//                                                                     search (e.g. "everything
//                                                                     runs on one engine")
//   void on_push(Node, const Ctx&)    DURING DFS, right after a        Maintain incremental state
//                                     choice is committed              (counters, stacks, ...)
//   void on_pop(Node, const Ctx&)     DURING DFS, right before the     Undo on_push exactly
//                                     choice is undone (LIFO)         (state restoration)
//   bool check(Cand, size_t, Ctx&)    DURING DFS, per candidate        Prune symmetric / dominated
//                                     before committing                branches cheaply
//   bool validate_leaf(const Ctx&)    AFTER DFS reaches a leaf         Whole-assignment checks
//                                     (complete configuration)        that cannot be tested
//                                                                     incrementally
//   const char* name() const          Debug logging only               Diagnostics
//
// Node / Cand / Ctx are iterator-specific types -- rules for different
// iterators keep their own, different signatures; nothing is forced into one
// shape. The only convention is the HOOK NAMES above and that a rule's state
// must be restorable by on_pop (push/pop calls are perfectly balanced and
// nested LIFO).
//
// Usage sketch (inside an iterator):
//
//   template <typename... Rules> struct MyIterator {
//     PruningRuleSet<Rules...> rules;
//     ...
//     rules.init(ctx);                          // once, before the search
//     if (rules.is_pruned(cand, idx, ctx)) ...  // per candidate
//     rules.on_push(node, ctx);                 // after committing
//     rules.on_pop(node, ctx);                  // before undoing
//     ok = rules.validate_leaf(ctx);            // at a leaf
//   };
//
// With an empty rule set (MyIterator<>) every call above compiles to nothing.
#pragma once

#include <cstddef>
#include <string>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <utility>

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
// Hook probes -- each probe is parameterized on the iterator's own types so
// rules with genuinely different signatures coexist under one rule set.
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

template <class R, class Ctx> constexpr bool has_init_v = is_detected_v<init_expr, R, Ctx>;
template <class R, class Node, class Ctx> constexpr bool has_push_v = is_detected_v<push_expr, R, Node, Ctx>;
template <class R, class Node, class Ctx> constexpr bool has_pop_v = is_detected_v<pop_expr, R, Node, Ctx>;
template <class R, class Cand, class Ctx> constexpr bool has_check_v = is_detected_v<check_expr, R, Cand, Ctx>;
template <class R, class Ctx> constexpr bool has_leaf_v = is_detected_v<leaf_expr, R, Ctx>;
template <class R> constexpr bool has_name_v = is_detected_v<name_expr, R>;

// =============================================================================
// PruningRuleSet -- compile-time list of rules with zero-overhead dispatch.
// =============================================================================
template <typename... Rules> struct PruningRuleSet
{
    std::tuple<Rules...> rules;

    PruningRuleSet() = default;

    template <typename... Rs, std::enable_if_t<sizeof...(Rs) == sizeof...(Rules) && (sizeof...(Rs) > 0), int> = 0>
    explicit PruningRuleSet(Rs &&...rs) : rules(std::forward<Rs>(rs)...)
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

    // ---- BEFORE DFS: one-time preparation ----------------------------------
    template <class Ctx> void init(const Ctx &ctx)
    {
        std::apply([&](auto &...rs) { (init_one(rs, ctx), ...); }, rules);
    }

    // ---- DURING DFS: incremental state maintenance (balanced, LIFO) --------
    template <class Node, class Ctx> void on_push(Node node, const Ctx &ctx)
    {
        std::apply([&](auto &...rs) { (push_one(rs, node, ctx), ...); }, rules);
    }

    template <class Node, class Ctx> void on_pop(Node node, const Ctx &ctx)
    {
        std::apply([&](auto &...rs) { (pop_one(rs, node, ctx), ...); }, rules);
    }

    // ---- DURING DFS: per-candidate pruning (short-circuits on first hit) ---
    template <class Cand, class Ctx> bool is_pruned(Cand candidate, size_t candidate_idx, const Ctx &ctx)
    {
        // Fold over || evaluates left-to-right and stops at the first true.
        return std::apply([&](auto &...rs) { return (false || ... || check_one(rs, candidate, candidate_idx, ctx)); },
                          rules);
    }

    // ---- AFTER DFS: leaf validation (short-circuits on first failure) ------
    template <class Ctx> bool validate_leaf(const Ctx &ctx)
    {
        // Fold over && evaluates left-to-right and stops at the first false.
        return std::apply([&](auto &...rs) { return (true && ... && leaf_one(rs, ctx)); }, rules);
    }

    // ---- Diagnostics --------------------------------------------------------
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
            rule.init(ctx);
    }

    template <class R, class Node, class Ctx> static void push_one(R &rule, Node node, const Ctx &ctx)
    {
        if constexpr (has_push_v<R, Node, Ctx>)
            rule.on_push(node, ctx);
    }

    template <class R, class Node, class Ctx> static void pop_one(R &rule, Node node, const Ctx &ctx)
    {
        if constexpr (has_pop_v<R, Node, Ctx>)
            rule.on_pop(node, ctx);
    }

    template <class R, class Cand, class Ctx>
    static bool check_one(R &rule, Cand candidate, size_t candidate_idx, const Ctx &ctx)
    {
        if constexpr (has_check_v<R, Cand, Ctx>)
            return rule.check(candidate, candidate_idx, ctx);
        else
            return false;
    }

    template <class R, class Ctx> static bool leaf_one(R &rule, const Ctx &ctx)
    {
        if constexpr (has_leaf_v<R, Ctx>)
            return rule.validate_leaf(ctx);
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