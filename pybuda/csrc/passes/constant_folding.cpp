// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/constant_folding.hpp"

#include <pybind11/pybind11.h>

#include <functional>

#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "passes/commute_utils.hpp"
#include "passes/passes_utils.hpp"
#include "utils/logger.hpp"

namespace tt::passes
{
using FoldFn = bool(graphlib::Graph *, graphlib::OpNode *, graphlib::OpNode *);

static graphlib::InputNode *get_constant_input(graphlib::Graph *graph, graphlib::OpNode *binary)
{
    auto operands = graph->data_operands(binary);
    TT_ASSERT(operands.size() == 2);
    graphlib::Node *input0 = operands[0];
    graphlib::Node *input1 = operands[1];

    graphlib::InputNode *constant = dynamic_cast<graphlib::InputNode *>(input0);
    if (not constant or not constant->is_constant())
        std::swap(input0, input1);

    return dynamic_cast<graphlib::InputNode *>(input0);
}

static graphlib::OpNode *get_producer_input(graphlib::Graph *graph, graphlib::OpNode *binary)
{
    auto operands = graph->data_operands(binary);
    TT_ASSERT(operands.size() == 2);
    graphlib::Node *input0 = operands[0];
    graphlib::Node *input1 = operands[1];

    graphlib::OpNode *producer = dynamic_cast<graphlib::OpNode *>(input0);
    if (not producer)
        std::swap(input0, input1);

    return dynamic_cast<graphlib::OpNode *>(input0);
}

bool is_constant_eltwise_binary(graphlib::Graph *graph, graphlib::OpNode *binary)
{
    if (not graphlib::is_eltwise_binary(binary))
        return false;

    return bool(get_constant_input(graph, binary)) and bool(get_producer_input(graph, binary));
}

static void insert_pad_within_tile(graphlib::Graph *graph, graphlib::Edge edge, int dim, int size)
{
    graphlib::Node *producer = graph->node_by_id(edge.producer_node_id);
    graphlib::Node *consumer = graph->node_by_id(edge.consumer_node_id);
    TT_ASSERT(consumer->shape().index_in_bounds(dim));

    dim = consumer->shape().negative_index(dim);
    std::uint32_t producer_dim_size = producer->shape().index_in_bounds(dim) ? producer->shape()[dim] : 1;
    std::uint32_t consumer_dim_size = consumer->shape()[dim];

    if (producer_dim_size == consumer_dim_size)
        return;

    // Handle by fixing up existing broadcast
    bool only_bcast = true;
    auto &tms = graph->get_edge_attributes(edge)->get_tms();
    for (auto &tm : tms)
    {
        if (tm.op != "broadcast")
        {
            only_bcast = false;
            break;
        }

        int tm_dim = consumer->shape().negative_index(std::get<int>(tm.attr[0]));
        if (tm_dim == dim)
        {
            std::get<int>(tm.attr[0]) = tm_dim;
            std::get<int>(tm.attr[1]) = size;
            return;
        }
    }

    // Handle by inserting a broadcast
    bool implicit_broadcast = producer_dim_size == 1;
    if (implicit_broadcast)
    {
        tms.push_back(graphlib::OpType("broadcast", {dim, size}, {}));
        return;
    }

    TT_ASSERT(size % graphlib::Shape::BUDA_TILE_DIM == 0);

    // Handle with a pad_tile op
    auto *pad_tile = graph->add_node(
        consumer->clone("pad_tile_" + producer->name() + "_" + std::to_string(edge.edge_creation_id)),
        graph->get_subgraph_id_for_node(producer->id()))->as<graphlib::OpNode>();
    pad_tile->change_op_type(graphlib::OpType("pad_tile", {dim, (int)producer->shape()[dim]}));
    auto [incoming_edge, outgoing_edge] = graphlib::insert_node_on_edge(graph, edge, pad_tile);
    if (only_bcast)
    {
        auto &incoming_tms = graph->get_edge_attributes(incoming_edge)->get_tms();
        auto &outgoing_tms = graph->get_edge_attributes(outgoing_edge)->get_tms();
        outgoing_tms.insert(outgoing_tms.begin(), incoming_tms.begin(), incoming_tms.end());
        incoming_tms.clear();
    }
    graphlib::Shape pad_tile_shape = only_bcast ? producer->shape() : consumer->shape();
    TT_ASSERT(size == graphlib::align_up_tile((int)pad_tile_shape[dim]));
    pad_tile_shape[dim] = graphlib::align_up_tile(pad_tile_shape[dim]);
    pad_tile->set_shape(pad_tile_shape);
    graphlib::try_consteval_op(graph, pad_tile);
}

static bool try_hoist_above_narrow(graphlib::Graph *graph, graphlib::OpNode *narrow, graphlib::OpNode *consumer)
{
    if (narrow->op_name() != "narrow")
        return false;

    if (graph->user_data_edges(narrow).size() > 1)
        return false;

    auto shape = narrow->shape();
    auto attr = narrow->op_type().attr;
    TT_ASSERT(attr.size() == 4);
    int dim = shape.negative_index(std::get<int>(attr[0]));
    int start = std::get<int>(attr[1]);
    int length = std::get<int>(attr[2]);

    bool within_tile = graphlib::align_up_tile(length) == graphlib::align_up_tile((int)shape[dim]);

    if ((dim == -1 or dim == -2) and (start == 0) and within_tile)
    {
        log_trace(LogGraphCompiler, "Hoist above narrow: {} {}", narrow->name(), consumer->name());

        auto narrow_operands = graph->data_operands(narrow);
        TT_ASSERT(narrow_operands.size() == 1);
        auto narrow_operand = narrow_operands.front();

        consumer->add_golden_transform(narrow->op_type());
        consumer->set_shape(narrow_operand->shape());

        auto edges = graph->get_edges(narrow, consumer);
        TT_ASSERT(edges.size() == 1);
        graphlib::swap(
            graph,
            edges.front(),
            [graph, narrow_operand, dim](graphlib::Edge edge)
            {
                // Skip fixing up the original narrow producer
                if (edge.producer_node_id == narrow_operand->id())
                    return;

                insert_pad_within_tile(graph, edge, dim, (int)narrow_operand->shape()[dim]);
            });

        return true;
    }

    return false;
}

template <typename CommutableFn, typename SinkOpFn>
std::vector<graphlib::Node *> find_operands_commute_through(
    graphlib::Graph *graph, graphlib::Node *root, CommutableFn commutable_fn, SinkOpFn sink_op_fn)
{
    std::vector<graphlib::Node *> sinks;
    std::vector<graphlib::Node *> needs_visit = {root};
    while (not needs_visit.empty())
    {
        graphlib::Node *node = needs_visit.back();
        needs_visit.pop_back();

        if (sink_op_fn(node))
        {
            sinks.push_back(node);
        }
        else if (commutable_fn(node))
        {
            for (auto *operand : graph->data_operands(node))
            {
                needs_visit.push_back(operand);
            }
        }
        else
        {
            // If any operands, through any path isn't a sink or commutable, give up
            return {};
        }
    }
    return sinks;
}

static bool try_fold_constant_multiply_into_matmul_rhs(
    graphlib::Graph *graph, graphlib::OpNode *operand, graphlib::OpNode *multiply)
{
    // Hoists and returns true if:
    //  - op is eltwise multiply
    //  - 1 argument is a 1 dimensional constant tensor
    //  - 1 argument is a matmul with RHS parameters
    if (multiply->op_name() != "multiply")
        return false;

    std::vector<graphlib::Node *> matmuls = find_operands_commute_through(
        graph,
        operand,
        [](graphlib::Node *commutable)
        {
            graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(commutable);
            return op and (op->is_op_type("add") or op->is_op_type("nop"));
        },
        [](graphlib::Node *matmul)
        {
            graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(matmul);
            return op and op->is_dense_matmul();
        });

    if (matmuls.empty())
        return false;

    graphlib::InputNode *constant = get_constant_input(graph, multiply);
    TT_ASSERT(constant);

    auto shape = constant->shape();
    if (shape.volume() != shape[-1])
        return false;

    // Check all matmul weights can legally fold this constant
    for (graphlib::Node *matmul : matmuls)
    {
        auto matmul_operands = graph->operand_data_edges(matmul);
        TT_ASSERT(matmul_operands.size() >= 2);
        graphlib::InputNode *matmul_rhs =
            dynamic_cast<graphlib::InputNode *>(graph->node_by_id(matmul_operands[1].producer_node_id));

        if (not matmul_rhs or
            not(matmul_rhs->is_parameter() or matmul_rhs->is_constant() or matmul_rhs->is_optimizer_parameter()))
            return false;

        if (graph->enable_training() and matmul_rhs->is_parameter())
            return false;
    }

    auto constant_edges = graph->get_edges(constant, multiply);
    TT_ASSERT(constant_edges.size() == 1);
    auto constant_edge = constant_edges.front();
    auto constant_attr = graph->remove_edge(constant_edge);

    // Fold
    for (graphlib::Node *matmul : matmuls)
    {
        auto matmul_operands = graph->operand_data_edges(matmul);
        TT_ASSERT(matmul_operands.size() >= 2);
        graphlib::InputNode *matmul_rhs =
            dynamic_cast<graphlib::InputNode *>(graph->node_by_id(matmul_operands[1].producer_node_id));

        log_trace(
            LogGraphCompiler, "Fold multiply into matmul weights: {} -> {}", multiply->name(), matmul_rhs->name());

        // Fixup broadcast
        for (auto &tm : constant_attr->get_tms())
        {
            if (tm.op == "broadcast")
            {
                int tm_dim = multiply->shape().negative_index(std::get<int>(tm.attr[0]));
                if (tm_dim == -2)
                {
                    std::get<int>(tm.attr[1]) = matmul_rhs->shape()[-2];
                }
            }
        }

        auto *multiply_clone = graph->add_node(
            multiply->clone(multiply->name() + "_" + matmul_rhs->name()),
            graph->get_subgraph_id_for_node(matmul->id()));
        multiply_clone->set_shape(matmul_rhs->shape());

        auto *constant_clone = graph->add_node(
            constant->clone(constant->name() + "_" + multiply->name()),
            graph->get_subgraph_id_for_node(matmul->id()));

        // Connect matmul rhs to multiply LHS
        graphlib::insert_node_on_edge(graph, matmul_operands[1], multiply_clone);

        // Connect constant to multiply RHS
        constant_edge.producer_node_id = constant_clone->id();
        constant_edge.consumer_input_port_id = 1;
        constant_edge.consumer_node_id = multiply_clone->id();
        graph->add_edge(constant_edge, constant_attr);

        graphlib::try_consteval_op(graph, multiply_clone);
    }

    // Remove multiply from the graph, but check if constant has other consumers before removing constant
    graphlib::bypass_node(graph, multiply, true);
    if (graph->user_edges(constant).size() == 0)
    {
        graph->remove_node(constant);
    }

    return true;
}

static bool try_fold_constant_associative(graphlib::Graph *graph, graphlib::OpNode *a, graphlib::OpNode *b)
{
    if (a->op_name() != b->op_name())
        return false;

    if (a->op_name() != "multiply" and a->op_name() != "add")
        return false;

    graphlib::InputNode *a_constant = get_constant_input(graph, a);
    if (not a_constant or not a_constant->is_constant())
        return false;

    if (graph->user_data_edges(a).size() > 1)
        return false;

    log_trace(LogGraphCompiler, "Fold constant associative: {} {}", a->name(), b->name());

    graphlib::InputNode *b_constant = get_constant_input(graph, b);
    TT_ASSERT(b_constant);
    auto a_edges = graph->get_edges(a_constant, a);
    auto b_edges = graph->get_edges(b_constant, b);
    TT_ASSERT(a_edges.size() == 1);
    TT_ASSERT(b_edges.size() == 1);

    auto b_attr = graph->get_edge_attributes(b_edges.front());
    graph->remove_edge(b_edges.front());
    auto b_subgraph_id = graph->get_subgraph_id_for_node(b->id());
    b = graph->add_node(graphlib::bypass_node(graph, b, true), b_subgraph_id)->as<graphlib::OpNode>();
    insert_node_on_edge(graph, a_edges.front(), b);
    b_edges.front().consumer_node_id = b->id();
    b_edges.front().consumer_input_port_id = 1;
    graph->add_edge(b_edges.front(), b_attr);
    graphlib::try_consteval_op(graph, b);

    return true;
}

static std::vector<FoldFn *> fold_fns = {
    try_hoist_above_narrow,
    try_fold_constant_multiply_into_matmul_rhs,
    try_fold_constant_associative,
};

static bool try_fold_constant_binary_op(graphlib::Graph *graph, graphlib::OpNode *binary)
{
    if (not is_constant_eltwise_binary(graph, binary))
        return false;

    for (FoldFn *fn : fold_fns)
    {
        auto *producer = get_producer_input(graph, binary);
        TT_ASSERT(producer);
        if (fn(graph, producer, binary))
            return true;
    }

    return false;
}

void constant_folding(graphlib::Graph *graph)
{
    bool updated = true;
    while (updated)
    {
        updated = false;
        for (auto *node : graphlib::topological_sort(*graph))
        {
            graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(node);
            if (not op)
                continue;

            if (try_fold_constant_binary_op(graph, op))
            {
                updated = true;
                break;
            }
        }
    }
}
}  // namespace tt::passes
