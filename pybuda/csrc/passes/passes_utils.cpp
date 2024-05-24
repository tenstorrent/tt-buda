// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes_utils.hpp"

#include "balancer/balancer_utils.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "utils/logger.hpp"

namespace tt {

using NodeType = graphlib::NodeType;

void optimize_tms(std::vector<graphlib::OpType> &tms) {
    if (tms.size() < 2) {
        return;
    }

    enum class Erase {
        None,
        B,
        AB,
    };

    using graphlib::OpType;
    using MatchFn = std::function<bool(OpType const &, OpType const &)>;  // return true means apply MergeFn
    using MergeFn = std::function<Erase(OpType &, OpType &)>;
    std::vector<std::pair<MatchFn, MergeFn>> rules = {
        // back to back slice or stack
        {[](OpType const &a, OpType const &b)
         { return (a.op == "hslice" or a.op == "vslice" or a.op == "hstack" or a.op == "vstack") and a.op == b.op; },
         [](OpType &a, OpType &b)
         {
             TT_ASSERT(a.attr.size() == 1 and b.attr.size() == 1);
             TT_ASSERT(std::holds_alternative<int>(a.attr[0]) and std::holds_alternative<int>(b.attr[0]));
             std::get<int>(a.attr[0]) *= std::get<int>(b.attr[0]);

             return Erase::B;
         }},

        // back to back (slice then stack) or (stack then slice)
        {[](OpType const &a, OpType const &b)
         {
             bool valid = false;
             if ((a.op == "hslice" and b.op == "hstack") or (a.op == "vslice" and b.op == "vstack") or
                 (a.op == "hstack" and b.op == "hslice") or (a.op == "vstack" and b.op == "vslice"))
             {
                 int a_factor = std::get<int>(a.attr[0]);
                 int b_factor = std::get<int>(b.attr[0]);
                 if (((a_factor >= b_factor) and (a_factor % b_factor == 0)) or
                     ((b_factor > a_factor) and (b_factor % a_factor == 0)))
                 {
                     valid = true;
                 }
             }
             return valid;
         },
         [](OpType &a, OpType &b)
         {
             TT_ASSERT(a.attr.size() == 1 and b.attr.size() == 1);
             TT_ASSERT(std::holds_alternative<int>(a.attr[0]) and std::holds_alternative<int>(b.attr[0]));
             int &a_factor = std::get<int>(a.attr[0]);
             int &b_factor = std::get<int>(b.attr[0]);
             if (a_factor == b_factor)
             {
                 // They cancel each other
                 return Erase::AB;
             }
             else if (a_factor > b_factor)
             {
                 a_factor /= b_factor;
             }
             else
             {
                 b_factor /= a_factor;
                 std::swap(a, b);
             }
             return Erase::B;
         }},

        // hoist slice above stack
        {[](OpType const &a, OpType const &b)
         {
             return ((a.op == "hstack" and b.op == "hslice") or (a.op == "vstack" and b.op == "vslice")) and
                    balancer::divisible_either_direction(std::get<int>(a.attr[0]), std::get<int>(b.attr[0]));
         },
         [](OpType &a, OpType &b)
         {
             std::swap(a, b);
             return Erase::None;
         }},

        // hoist transpose before slice
        {[](OpType const &a, OpType const &b)
         { return (a.op == "hslice" or a.op == "vslice") and b.op == "transpose"; },
         [](OpType &a, OpType &b)
         {
             // Switch slicing direction and commute
             a.op = (a.op == "hslice") ? "vslice" : "hslice";
             std::swap(a, b);
             return Erase::None;
         }},

        // hoist transpose before stack
        {[](OpType const &a, OpType const &b)
         { return (a.op == "hstack" or a.op == "vstack") and b.op == "transpose"; },
         [](OpType &a, OpType &b)
         {
             // Switch stacking direction and commute
             a.op = (a.op == "hstack") ? "vstack" : "hstack";
             std::swap(a, b);
             return Erase::None;
         }},

        // hoist transpose before broadcast
        {[](OpType const &a, OpType const &b) { return a.op == "broadcast" and b.op == "transpose"; },
         [](OpType &a, OpType &b)
         {
             int &dim = std::get<int>(a.attr[0]);
             if (dim > 1)
             {
                 TT_ASSERT(dim == 2 or dim == 3);
                 dim = (dim == 2) ? 3 : 2;
             }
             std::swap(a, b);
             return Erase::None;
         }},

        // hoist broadcast before stack
        {[](OpType const &a, OpType const &b)
         {
             int supported_bcast_dim = (a.op == "hstack") ? 2 : 3;
             return (a.op == "hstack" or a.op == "vstack") and b.op == "broadcast" and
                    std::get<int>(b.attr[0]) == supported_bcast_dim;
         },
         [](OpType &a, OpType &b)
         {
             std::swap(a, b);
             return Erase::None;
         }},

        // back to back broadcast (order c, r, z)
        {[](OpType const &a, OpType const &b) {
             return (a.op == "broadcast" and b.op == "broadcast") and
                    (std::get<int>(a.attr[0]) < std::get<int>(b.attr[0]));
         },
         [](OpType &a, OpType &b)
         {
             std::swap(a, b);
             return Erase::None;
         }},

        // back to back transpose
        {[](OpType const &a, OpType const &b) { return a.op == "transpose" and b.op == "transpose"; },
         [](OpType &, OpType &) { return Erase::AB; }},

        // slice after select
        {[](OpType const &a, OpType const &b) { return a.op == "select" and (b.op == "hslice" or b.op == "vslice"); },
         [](OpType &a, OpType &b)
         {
             TT_ASSERT(a.attr.size() == 4 and b.attr.size() == 1);
             TT_ASSERT(std::holds_alternative<int>(a.attr[1]));
             TT_ASSERT(std::holds_alternative<int>(a.attr[2]));
             TT_ASSERT(std::holds_alternative<int>(a.attr[3]));
             TT_ASSERT(std::holds_alternative<int>(b.attr[0]));

             std::get<int>(a.attr[1]) *= std::get<int>(b.attr[0]);
             std::get<int>(a.attr[2]) *= std::get<int>(b.attr[0]);
             std::get<int>(a.attr[3]) *= std::get<int>(b.attr[0]);
             std::get<int>(a.buda_attrs["index"]) *= std::get<int>(b.attr[0]);
             std::get<int>(a.buda_attrs["length"]) *= std::get<int>(b.attr[0]);
             std::get<int>(a.buda_attrs["stride"]) *= std::get<int>(b.attr[0]);
             std::swap(a, b);
             return Erase::None;
         }},

        // back to back select
        {[](OpType const &a, OpType const &b) { return false and a.op == "select" and b.op == "select"; },
         [](OpType &a, OpType &b)
         {
             TT_ASSERT(a.attr.size() == 4 and b.attr.size() == 4);
             // TODO: there are some cases of back to back select that can be merged, when a.length >= b.length
             return Erase::None;
         }},
    };

    using SingleMatchFn = std::function<bool(OpType const &)>;  // return true means apply MergeFn
    using SingleUpdateFn = std::function<bool(OpType &)>;       // return true means remove this tm
    std::pair<SingleMatchFn, SingleUpdateFn> single_rules[] = {
        // Erase stacks and slices with factors of 1
        {[](OpType const &a) {
             return (a.op == "hstack" or a.op == "vstack" or a.op == "hslice" or a.op == "vslice") and
                    std::get<int>(a.attr[0]) == 1;
         },
         [](OpType &) { return true; }},
    };

    bool any_updated = true;
    while (any_updated)
    {
        any_updated = false;
        for (auto [match_fn, update_fn] : single_rules)
        {
            for (auto iter = tms.begin(); iter != tms.end(); ++iter)
            {
                auto &tm = *iter;
                if (match_fn(tm))
                {
                    any_updated = true;
                    if (update_fn(tm))
                    {
                        tms.erase(iter);
                        break;
                    }
                }
            }
        }

        if (any_updated)
            continue;

        for (auto [match_fn, merge_fn] : rules) {
            auto iter = std::adjacent_find(tms.begin(), tms.end(), match_fn);
            if (iter != tms.end()) {
                any_updated = true;
                switch (merge_fn(*iter, *(iter + 1))) {
                    case Erase::B: {
                        tms.erase(iter + 1);
                        break;
                    }
                    case Erase::AB: {
                        tms.erase(iter, iter + 2);
                        break;
                    }
                    default: break;
                }
            }
        }
    }
}

void optimize_tms(Graph *graph) {
    for (Node *node : graph->nodes()) {
        if (node->node_type() == NodeType::kBudaOp) {
            for (auto const &edge : graph->operand_data_edges(node)) {
                auto edge_attributes = graph->get_edge_attributes(edge);
                std::vector<graphlib::OpType> &tms = edge_attributes->get_tms();
                // Collapse mergeable tms
                optimize_tms(tms);
            }
        }
    }
}

// Recalculate all node shapes from inputs
void recalculate_shapes(graphlib::Graph *graph)
{
    for (Node *n : graphlib::topological_sort(*graph))
    {
        if (n->node_type() == graphlib::NodeType::kInput)
            continue;

        graphlib::calculate_and_set_node_shape(graph, n);
    }
}

std::vector<int> get_factors(int num)
{
    std::vector<int> factors;

    while (num % 2 == 0)
    {
        factors.push_back(2);
        num /= 2;
    }

    int sqrt_num = sqrt(num);
    for (int i = 3; i <= sqrt_num; i += 2)
    {
        while (num % i == 0)
        {
            factors.push_back(i);
            num /= i;
        }
    }

    if (num > 2)
    {
        factors.push_back(num);
    }

    return factors;
}

bool check_unsupported_hw_ops(Graph *graph, bool should_throw)
{
    bool unsupported_hw_ops = false;
    py::object eval_module = py::module_::import("pybuda.op.eval.buda");
    std::string message;

    for (Node *node : graph->nodes())
    {
        // TODO: Remove this block once backend supports hconcat / vconcat
        if (node->node_type() == NodeType::kBudaNaryTM)
        {
            graphlib::BudaNaryTMNode *tm = node->as<graphlib::BudaNaryTMNode>();
            unsupported_hw_ops = true;
            message += fmt::format("{} {}\n", tm->name(), tm->op_type().op);
            continue;
        }

        if (node->node_type() != NodeType::kBudaOp)
            continue;

        graphlib::BudaOpNode *op = node->as<graphlib::BudaOpNode>();
        py::function pybuda_parallelization = eval_module.attr("get_f_pybuda_parallelization")(op->op_type_ptr());
        py::object parallelization = pybuda_parallelization(balancer::get_op_shape(graph, node));

        if (parallelization.is_none())
        {
            unsupported_hw_ops = true;

            std::string attrs;
            for (const auto &[key, val] : op->buda_attrs())
            {
                attrs = attrs + key + ": ";
                if (std::holds_alternative<bool>(val))
                {
                    attrs += std::to_string(std::get<bool>(val)) + ", ";
                }
                else if (std::holds_alternative<int>(val))
                {
                    attrs += std::to_string(std::get<int>(val)) + ", ";
                }
                else if (std::holds_alternative<float>(val))
                {
                    attrs += std::to_string(std::get<float>(val)) + ", ";
                }
                else if (std::holds_alternative<std::string>(val))
                {
                    attrs += std::get<std::string>(val) + ", ";
                }
            }
            if (attrs.length() > 1)
            {
                attrs.erase(attrs.length() - 2);
            }
            log_warning("Unsupported HW op: {} {}({})", op->name(), op->op_type().op, attrs);
            message += fmt::format("{} {}({})\n", op->name(), op->op_type().op, attrs);
        }
    }

    if (unsupported_hw_ops and should_throw)
        throw UnsupportedHWOpsError(message);

    return unsupported_hw_ops;
}

// Returns true if string is part of 2D vector of strings, false otherwise.
bool is_str_in_strings(const std::string &str, const std::vector<std::vector<std::string>> &strings)
{
    for (const std::vector<std::string> &iter : strings)
    {
        if (std::find(iter.begin(), iter.end(), str) != iter.end())
        {
            return true;
        }
    }

    return false;
}

}  // namespace tt
