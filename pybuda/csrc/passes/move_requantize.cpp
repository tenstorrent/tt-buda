
#include <pybind11/pybind11.h>

#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "utils/logger.hpp"
#include "python_bindings_common.hpp"
#include "graph_lib/node.hpp"
#include "graph_lib/graph.hpp"
#include "passes/move_requantize.hpp"
#include "utils/logger.hpp"
#include "passes/passes_utils.hpp"
#include "passes/commute_utils.hpp"

namespace tt::passes
{
static void set_bcast_dims(graphlib::Graph *graph, std::vector<int> &volumes, graphlib::Edge edge) {
    graph->get_edge_attributes(edge)->clear_broadcast_dims();

    for (std::size_t i = 0; i < volumes.size(); i++) {
        int volume = volumes[i];
        if (volume > 1) {
            graph->get_edge_attributes(edge)->set_broadcast_dim(i, volume, false); 
        }
    }
}

static std::vector<graphlib::Node *> find_path_to_requant(
    graphlib::Graph *graph,
    graphlib::OpNode *initial_op)
{
    std::vector<graphlib::Node *> path;

    graphlib::Node *iter = initial_op;
    auto clone_shape = initial_op->shape();

    bool found_requant = false;
    while (not found_requant)
    {   
        auto op = dynamic_cast<graphlib::OpNode *>(iter);
        if (not op)
            break;

        if (op->op_name() == "buda_requantize")
        {
            found_requant = true;
            path.push_back(op);
            break;
        }

        if (graph->data_users(op).size() > 1)
            break;

        if (not (is_elementwise(op) or op == initial_op))
            break;

        // Only commute through elementwise ops
        path.push_back(op);
        iter = graph->data_users(op)[0];
    }

    if (not found_requant)
        path.clear();

    return path;
}


void commute_through_requant(graphlib::Graph *graph, std::vector<graphlib::Node *> const &path) {
    TT_ASSERT(path.size() >= 2);
    graphlib::OpNode *first = path.front()->as<graphlib::OpNode>();
    graphlib::OpNode *last = path.back()->as<graphlib::OpNode>();
    std::pair<int, int> operand_dims;
    log_debug(LogGraphCompiler, "Commute and bypass TM through requant: {} -> {}", first->name(), last->name());
    graphlib::OpType golden_transform = first->op_type();

    graphlib::Shape commute_shape = shape_of_only_operand(graph, first);
    graphlib::Shape clone_shape = first->shape();

    for (std::size_t i = 1; i < path.size(); ++i)
    {

        graphlib::Node *producer = path[i - 1];
        graphlib::Node *consumer = path[i];
        auto consumer_df_before = consumer->output_df();

        TT_ASSERT(graph->user_data_edges(producer).size() == 1);

        // Set the shape to the desired final shape for this whole path
        if (graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(consumer))
        {   
            graphlib::OpNode *producer_as_op = dynamic_cast<graphlib::OpNode *>(producer);
            if (producer_as_op) {
                // Must change commute shape, clone shape, and golden transform if there are broadcasts on the incoming edge
                auto [commute_bcasts, clone_bcasts] = handle_shape_change_through_bcast(graph, first, producer_as_op, op, &commute_shape, &clone_shape).second;
                if (golden_transform.op == "reshape") {
                    for (std::size_t i = 0; i < golden_transform.attr.size(); i++) {
                        int current_dim = std::get<int>(golden_transform.attr[i]);
                        golden_transform.attr[i] = clone_bcasts[i]*current_dim;
                    }
                }

                graphlib::Edge between_edge = retrieve_between_edge(graph, producer, consumer);
                set_bcast_dims(graph, commute_bcasts, between_edge);                
            }

            if (is_elementwise(op))
            {
                commute_through_eltwise(op, &commute_shape, &golden_transform);
            }
            else if (is_quantization_ops(op)) {
                commute_through_quantization(op, &commute_shape, &golden_transform);
            }
            else 
            {
                TT_ASSERT(false, "Found non-elementwise and non-quant op in path to requantize");
            }
            log_trace(LogGraphCompiler, "  Op node: {} -> shape set to {}", consumer->name(), commute_shape);
        }

        // Handle nary operands (not on this `path`) 
        std::vector<graphlib::Edge> consumer_operands = graph->operand_data_edges(consumer);
        for (graphlib::Edge operand_edge : consumer_operands)
        {
            if (operand_edge.producer_node_id == producer->id())
                continue;

            convert_implicit_to_explicit_bcasts(graph, operand_edge);
            auto name = last->name() + "_operand_commute_clone" + std::to_string(operand_edge.edge_creation_id);
            graphlib::Node *clone = graph->add_node(first->clone(name), graph->get_subgraph_id_for_node(operand_edge.producer_node_id));
            graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(clone);
            log_trace(LogGraphCompiler, "  Operand commute clone: {} -> between {} and {} ", name, consumer->name(), graph->node_by_id(operand_edge.producer_node_id)->name());

            update_reshape_attr(op, commute_shape);
            clone->set_shape(commute_shape);
            log_trace(LogGraphCompiler, "  Operand commute clone shape: {}", commute_shape);


            auto [in_edge, out_edge] = insert_node_on_edge(graph, operand_edge, clone);
            // Set dataformat to match producer on operand edge
            clone->set_output_df(graph->node_by_id(in_edge.producer_node_id)->output_df());
            handle_change_rank(graph, clone);
            try_commute_bcast_through_clone(graph, op);
            if (graphlib::InputNode *input = dynamic_cast<graphlib::InputNode *>(graph->data_operands(clone)[0]))
                try_consteval_input_no_operand_forks(graph, input, true);
        }
        // Maintain df from before commute
        consumer->set_output_df(consumer_df_before);
    }


    // Insert the TM after requant op
    auto tm_ = bypass_node(graph, first, true /*remove*/);
    auto tags = tm_->as<graphlib::TaggedNode>()->get_tags();

    for (auto requant_out_edge : graph->user_data_edges(last))
    {
        auto original_tms = graph->get_edge_attributes(requant_out_edge)->get_tms();
        TT_ASSERT(original_tms.size() == 0);
        auto name = tm_->name() + "_cloned" + std::to_string(requant_out_edge.edge_creation_id);

        graphlib::Node *curr_node = graph->add_node(
            tm_->clone(name), graph->get_subgraph_id_for_node(requant_out_edge.consumer_node_id));
        curr_node->as<graphlib::TaggedNode>()->add_tags(tags);

        insert_node_on_edge(graph, requant_out_edge, curr_node);
        curr_node->set_output_df(last->output_df());

    }
}


bool move_tm_through_requantize(graphlib::Graph *graph) {

    bool attempt_update = true;
    bool updated_anything = false;
    while (attempt_update)
    {   
        // Set to false here because we want to stop looping if no update occurs
        attempt_update = false;
        for (auto *node : graphlib::topological_sort(*graph))
        {
            graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(node);
            if (not op)
                continue;

            if (op->as<graphlib::TaggedNode>()->has_tag("dont_erase"))
                continue;

            if (op->op_name() != "reshape" and op->op_name() != "transpose")
                continue;

            std::vector<graphlib::Node *> path = find_path_to_requant(graph, op);
            if (path.empty())
                continue;

            commute_through_requant(graph, path);
            attempt_update = true;
            updated_anything = true;
            recalculate_shapes(graph);
            break;
        }
    }
    return updated_anything;
}

}  // namespace tt::passes