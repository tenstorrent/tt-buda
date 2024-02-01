// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <functional>
#include <iterator>
#include <memory>
#include <regex>
#include <type_traits>
#include <vector>

#include "graph_lib/edge.hpp"
#include "graph_lib/graph.hpp"
#include "graph_lib/node.hpp"
#include "graph_lib/node_types.hpp"

namespace tt::graphlib::query
{
inline std::string view_node_name(Node const* node) { return node->name(); }
inline std::string view_layer_name(Node const* node)
{
    return node->as<TaggedNode>()->tag_value_or("layer", std::string{});
}
inline std::string view_op_type(Node const* node)
{
    if (auto op = dynamic_cast<OpNode const*>(node); op != nullptr)
    {
        return op->op_name();
    }
    return std::string{};
}

template <typename T, typename U = T>
struct Predicate
{
    template <typename E = T, typename = std::enable_if_t<not std::is_same_v<E, U>>>
    Predicate(std::function<bool(U)> fn, std::function<U(T)> view) : fn(fn), view(view)
    {
    }

    template <typename E = T, typename = std::enable_if_t<std::is_same_v<E, U>>>
    Predicate(std::function<bool(U)> fn) : fn(fn), view(nullptr)
    {
    }

    Predicate(std::regex regex, std::function<std::string(T)> view = nullptr) :
        fn([regex](std::string r) { return std::regex_match(r, regex); }), view(view)
    {
    }

    Predicate(std::string const& regex, std::function<std::string(T)> view = nullptr) :
        Predicate(std::regex(regex), view)
    {
    }

    template <typename Iter, typename Fn, typename U1 = std::invoke_result_t<Fn, T>>
    static Predicate<T, U1> anyOf(Iter begin, Iter end, Fn view = nullptr)
    {
        return Predicate<T, U1>(
            [begin, end](U1 r)
            { return std::any_of(begin, end, [r](auto const& p) { return Predicate<U1>(p)(r); }); },
            view);
    }

    template <typename Iter, typename Fn, typename U1 = std::invoke_result_t<Fn, T>>
    static Predicate<T, U1> allOf(Iter begin, Iter end, Fn view = nullptr)
    {
        return Predicate<T, U1>(
            [begin, end](U1 r)
            { return std::all_of(begin, end, [r](auto const& p) { return Predicate<U1>(p)(r); }); },
            view);
    }

    bool operator()(T r) const
    {
        if constexpr (std::is_same_v<T, U>)
            return fn(r);
        else
        {
            auto u = view(r);
            TT_ASSERT(u.size() >= 0);
            return fn(u);
        }
    }

    Predicate<T> negate() const
    {
        auto self = *this;
        return Predicate<T>([self](T r) { return not self(r); });
    }

    template <typename U1>
    Predicate<T> operator&(Predicate<T, U1> b) const
    {
        auto a = *this;
        return Predicate<T>([a, b](T r) { return a(r) and b(r); });
    }

    template <typename U1>
    Predicate<T> operator|(Predicate<T, U1> b) const
    {
        auto a = *this;
        return Predicate<T>([a, b](T r) { return a(r) or b(r); });
    }

    // Creates a copy of this with U template parameter type erased, it does this by introducing a level of indirection
    Predicate<T> type_erased() const
    {
        if constexpr (std::is_same_v<T, U>)
        {
            return *this;
        }
        else
        {
            auto a = *this;
            return Predicate<T>([a](T r) { return a(r); });
        }
    }

    std::function<bool(U)> fn;
    std::function<U(T)> view;
};

using NodePredicate = Predicate<Node*>;

inline NodePredicate name_regex(std::string const& name_regex)
{
    return graphlib::query::Predicate<Node*, std::string>(name_regex, graphlib::query::view_node_name).type_erased();
}

inline NodePredicate layer_regex(std::string const& layer_regex)
{
    return graphlib::query::Predicate<Node*, std::string>(layer_regex, graphlib::query::view_layer_name).type_erased();
}

inline NodePredicate op_type(std::string const& t)
{
    return graphlib::query::Predicate<Node*, std::string>(
               [t](std::string o) { return t == o; }, graphlib::query::view_op_type)
        .type_erased();
}

inline NodePredicate always()
{
    return graphlib::query::Predicate<Node*>([](Node*) { return true; });
}

inline NodePredicate never()
{
    return graphlib::query::Predicate<Node*>([](Node*) { return false; });
}

template <typename IterT, typename ViewT = typename IterT::value_type>
class Filter
{
   public:
    using Reference = typename IterT::reference;
    using Value = typename IterT::value_type;

    class Iterator : public std::iterator<std::input_iterator_tag, Value>
    {
       public:
        Iterator(IterT iter, IterT end, Predicate<Value, ViewT> const& p) : iter(iter), end(end), p(&p)
        {
            this->iter = next(iter);
        }

        IterT next(IterT i) { return std::find_if(i, end, *p); }

        Iterator& operator++()
        {
            ++iter;
            iter = next(iter);
            return *this;
        }

        Iterator operator++(int)
        {
            auto r = *this;
            ++(*this);
            return r;
        }

        bool operator==(Iterator other) const { return iter == other.iter; }
        bool operator!=(Iterator other) const { return not(*this == other); }
        Reference operator*() const { return *iter; }

       private:
        IterT iter;
        IterT end;
        Predicate<Value, ViewT> const* p;
    };

    template <typename OwnedT = void>
    Filter(IterT begin, IterT end, Predicate<Value, ViewT> p, std::shared_ptr<OwnedT> owned = {}) :
        iter_begin(begin), iter_end(end), p(p), owned(std::static_pointer_cast<void>(owned))
    {
    }

    Iterator begin() { return Iterator(iter_begin, iter_end, p); }
    Iterator end() { return Iterator(iter_end, iter_end, p); }

   private:
    IterT iter_begin;
    IterT iter_end;
    Predicate<Value, ViewT> p;
    std::shared_ptr<void> owned;
};

inline Predicate<graphlib::Node*> predicate_op_node_type()
{
    return Predicate<graphlib::Node*>([](graphlib::Node* n) { return dynamic_cast<graphlib::OpNode*>(n) != nullptr; });
}

template <typename IterT, typename ViewT = typename IterT::value_type, typename OwnedT = void>
inline Filter<IterT, ViewT> make_filter(
    IterT begin, IterT end, Predicate<typename IterT::value_type, ViewT> p, std::shared_ptr<OwnedT> owned = {})
{
    return Filter<IterT, ViewT>(begin, end, p, owned);
}

template <typename IterT, typename ViewT = typename IterT::value_type, typename OwnedT = void>
inline Filter<IterT> filter_nodes(IterT begin, IterT end, Predicate<Node*> p)
{
    static_assert(std::is_same_v<typename IterT::reference, Node*&>);
    return make_filter(begin, end, p);
}

template <typename ViewT = typename std::vector<Node*>::iterator::value_type>
inline Filter<std::vector<Node*>::iterator, ViewT> filter_nodes(Graph* graph, Predicate<Node*, ViewT> p)
{
    auto nodes = std::make_shared<std::vector<Node*>>(graph->nodes());
    return make_filter(nodes->begin(), nodes->end(), p, nodes);
}

inline Filter<std::vector<Node*>::iterator, std::string> filter_nodes_by_name(
    Graph* graph, std::string const& name_regex)
{
    return filter_nodes(graph, Predicate<Node*, std::string>(name_regex, view_node_name));
}
}  // namespace tt::graphlib::query
