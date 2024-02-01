// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/bind_reshape_to_io.hpp"

#include <pybind11/pybind11.h>

#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "utils/logger.hpp"

namespace tt::passes
{

static bool is_reshape(graphlib::Node const *node)
{
    graphlib::OpNode const *op = dynamic_cast<graphlib::OpNode const *>(node);
    return op and op->op_name() == "reshape";
}

static std::vector<graphlib::Node *> path_to_reshape_from_input(graphlib::Graph *graph, graphlib::Node *initial_node)
{
    std::vector<graphlib::Node *> path;
    
    log_trace(LogGraphCompiler, "Starting at node: {}", initial_node->name());
    graphlib::Node *iter = initial_node;

    bool bail = false; 
    while (not bail)
    {
        std::vector<graphlib::Node *> users = graph->data_users(iter);
        if (users.size() > 1) {
            log_trace(LogGraphCompiler, "Multiple users at {}, bailing", iter->name());
            bail = true;
        }

        std::vector<graphlib::Node *> operands = graph->data_operands(iter);
        if (operands.size() > 1) {
            bool non_broadcast_fork_found = false;
            for (std::size_t i = 0; i < operands.size(); i ++) {
                if (operands[i]->shape() == iter->shape()) {
                    if (non_broadcast_fork_found) {
                        log_trace(LogGraphCompiler, "More than one non-broadcast operand path at {}, bailing", iter->name());
                        bail = true;
                    }
                    non_broadcast_fork_found = true;
                }
                else {
                    std::vector<graphlib::OpType> tms = graph->get_edge_attributes(graph->operand_data_edges(iter)[i])->get_tms();
                    for (graphlib::OpType &op_type : tms) {
                        if ((op_type.op != "broadcast") or operands[i]->shape().volume() != 1) {
                            log_trace(LogGraphCompiler, "Found eitehr non broadcast or broadcast from volume != 1 at {}, bailing", iter->name());
                            bail = true; 
                        }
                    }
                }
            }
        }
        graphlib::OpNode *user = dynamic_cast<graphlib::OpNode *>(users[0]);
        if (not user) {
            log_trace(LogGraphCompiler, "No users at {}, bailing", iter->name());
            bail = true;
        }
        else if (is_eltwise(user)) {
            path.push_back(iter);
            log_trace(LogGraphCompiler, "Adding node to path: {}", iter->name());
        }
        else if (is_reshape(user)) {
            path.push_back(iter);
            log_trace(LogGraphCompiler, "Adding node to path: {}", iter->name());
            path.push_back(user);
            log_trace(LogGraphCompiler, "Adding reshape node to path, done searching: {}", user->name());
            break;
        }
        else {
            log_trace(LogGraphCompiler, "Not eltwise or reshape at {}, bailing", user->name());
            bail = true;
        }
        iter = user;
    }
    if (bail) {
        path.clear();
    }
    return path;
}

static void commute_eltwise_ops_to_input(graphlib::Graph *graph, std::vector<graphlib::Node *> path)
{
    graphlib::InputNode *first = path.front()->as<graphlib::InputNode>();
    graphlib::OpNode *last = path.back()->as<graphlib::OpNode>();

    TT_ASSERT(first);
    TT_ASSERT(is_reshape(last));
    graphlib::Shape original_shape = first->shape();
    graphlib::Shape reinterpret_shape = last->shape();
    graphlib::Shape commute_shape = reinterpret_shape;
    graphlib::OpType golden_transform = last->op_type();

    log_debug(LogGraphCompiler, "Reinterpret shape of input node: {} original: {} reinterprer: {}", first->name(), commute_shape, original_shape);
    first->set_shape(commute_shape);
    graphlib::RuntimeTensorTransform runtime_tensor_transform(original_shape, commute_shape);
    first->set_runtime_tensor_transform(runtime_tensor_transform);
    bypass_node(graph, last, true);

    // path is input -> eltwise -> eltwise ... -> reshape
    for (std::size_t i = 1; i < path.size() - 1; ++i) {
        graphlib::Node *node = path[i];

        // Set the shape to the desired final shape for this whole path
        log_debug(LogGraphCompiler, "Commuting: {} to: {}", node->name(), commute_shape);
        if (graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(node))
            op->add_golden_transform(golden_transform);
        node->set_shape(commute_shape);

        std::vector<graphlib::Edge> consumer_operands = graph->operand_data_edges(node);
        for (graphlib::Edge operand_edge : consumer_operands)
        {
            graph->get_edge_attributes(operand_edge)->set_tms({});
        }
        calculate_and_set_node_shape(graph, node);
    }
}

static std::vector<graphlib::Node *> search_pass_to_output_from_another_user(graphlib::Graph *graph, graphlib::Node *initial_node)
{
    std::vector<graphlib::Node *> path;

    graphlib::Node *iter = initial_node;
    while (true)
    {
        std::vector<graphlib::Node *> users = graph->data_users(iter);
        if (users.size() > 1) {
            // Check for the case that an op has multiply users, but all of them are the same op
            // and cease the search otherwise
            graphlib::Node* user_0 = users[0];
            for (auto user : users) {
                if (user != user_0) {
                    path.clear();
                    break;
                }
            }
        }

        // Check the case that initial-node is output node
        graphlib::OutputNode *output = dynamic_cast<graphlib::OutputNode *>(iter);
        if (output) {
            path.push_back(iter);
            break;
        }

        graphlib::OpNode *user = dynamic_cast<graphlib::OpNode *>(users[0]);
        if (not user) {
            graphlib::OutputNode *user_output = dynamic_cast<graphlib::OutputNode *>(users[0]);
            if (not user_output) {
                path.clear();
                break;
            }
            path.push_back(iter);
            path.push_back(users[0]);
            break;
        }
        if (is_eltwise(user)) {
            path.push_back(iter);
        }
        else {
            path.clear();
            break;
        }
        iter = user;
    }
    return path;
}

static std::vector<graphlib::Node *> path_to_reshape_from_output(graphlib::Graph *graph, graphlib::Node *initial_node)
{
    std::vector<graphlib::Node *> path;
    
    log_trace(LogGraphCompiler, "Starting at node: {}", initial_node->name());
    graphlib::Node *iter = initial_node;
    while (true)
    {
        std::vector<graphlib::Node *> operands = graph->data_operands(iter);
        int operand_index = 0;
        if (operands.size() > 1) {
            log_trace(LogGraphCompiler, "Multiple operands at {}", iter->name());
            bool non_broadcast_fork_found = false;
            for (std::size_t i = 0; i < operands.size(); i ++) {
                if (operands[i]->shape() == iter->shape()) {
                    if (non_broadcast_fork_found) {
                        log_trace(LogGraphCompiler, "More than one non-broadcast path at {}, bailing", iter->name());
                        path.clear();
                        return path;
                    }
                    operand_index = i;
                    non_broadcast_fork_found = true;
                }
                else {
                    std::vector<graphlib::OpType> tms = graph->get_edge_attributes(graph->operand_data_edges(iter)[i])->get_tms();
                    for (graphlib::OpType &op_type : tms) {
                        if ((op_type.op != "broadcast") or operands[i]->shape().volume() != 1) {
                            log_trace(LogGraphCompiler, "Found eitehr non broadcast or broadcast from volume != 1 at {}, bailing", iter->name());
                            path.clear();
                            return path;
                        }
                    }
                }
            }
        }

        graphlib::OpNode *operand = dynamic_cast<graphlib::OpNode *>(operands[operand_index]);
        if (not operand) {
            path.clear();
            log_trace(LogGraphCompiler, "No operands at {}, bailing", iter->name());
            break;
        }
        if (is_eltwise(operand)) {
            if (graph->data_users(operand).size() > 1) {
                std::vector<graphlib::Node*> users = graph->data_users(operand);
                bool other_users_commutable = true;

                // Check if the other users have pass to output nodes
                // Add the ops included in sub-pass if there is such, and cease the search otherwise
                for (auto user : users) {
                    if (user == iter)
                        continue;

                    std::vector<graphlib::Node*> pass_to_output = search_pass_to_output_from_another_user(graph, user);
                    if (pass_to_output.empty()) {
                        other_users_commutable = false;
                        break;
                    } else {
                        path.insert(path.end(), pass_to_output.begin(), pass_to_output.end());
                        log_trace(LogGraphCompiler, "Sub-pass is found between another user {} and output, Adding nodes", operand->name());
                    }
                }

                if (not other_users_commutable) {
                    path.clear();
                    log_trace(LogGraphCompiler, "Eltwise-op with multiple users, at least one of them are not commutable, {}, bailing", operand->name());
                    break;
                }
            }
            path.push_back(iter);
            log_trace(LogGraphCompiler, "Adding node to path: {}", iter->name());
        }
        else if (is_reshape(operand) and graph->data_users(operand).size() == 1) {
            path.push_back(iter);
            log_trace(LogGraphCompiler, "Adding node to path: {}", iter->name());
            path.push_back(operand);
            log_trace(LogGraphCompiler, "Adding node to path: {}", operand->name());
            break;
        }
        else {
            path.clear();
            log_trace(LogGraphCompiler, "Not eltwise or reshape with single user {}, bailing", operand->name());
            break;
        }
        iter = operand;
    }
    std::reverse(path.begin(), path.end());
    return path;
}

static void commute_eltwise_ops_to_output(graphlib::Graph *graph, std::vector<graphlib::Node *> path)
{
    graphlib::OpNode *first = path.front()->as<graphlib::OpNode>();
    graphlib::OutputNode *last = path.back()->as<graphlib::OutputNode>();

    TT_ASSERT(is_reshape(first));
    TT_ASSERT(last);
    graphlib::Node *reshape_input = graph->data_operands(first)[0];
    graphlib::Shape commute_shape = reshape_input->shape();
    graphlib::Shape original_shape = first->shape();
    graphlib::OpType golden_transform = first->op_type();
    std::string reshape_name = first->name();
    graphlib::RuntimeTensorTransform runtime_tensor_transform(commute_shape, original_shape);

    bypass_node(graph, first, true);
    // path is reshape -> eltwise -> eltwise ... -> output
    for (std::size_t i = 1; i < path.size() - 1; ++i) {
        graphlib::Node *node = path[i];

        // Set the shape to the desired final shape for this whole path
        log_debug(LogGraphCompiler, "Commuting: {} to: {}", node->name(), commute_shape);
        if (graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(node))
            op->add_golden_transform(golden_transform);
        if (graphlib::OutputNode *op = dynamic_cast<graphlib::OutputNode *>(node))
            op->set_runtime_tensor_transform(runtime_tensor_transform);
        node->set_shape(commute_shape);

        std::vector<graphlib::Edge> consumer_operands = graph->operand_data_edges(node);
        for (graphlib::Edge operand_edge : consumer_operands)
        {
            size_t num_tms = graph->get_edge_attributes(operand_edge)->get_tms().size();
            graph->get_edge_attributes(operand_edge)->set_tms({});

            // Insert reshape ops to fixed shaped (no bcast) constant input
            graphlib::InputNode *input_operand = dynamic_cast<graphlib::InputNode *>(graph->node_by_id(operand_edge.producer_node_id));
            if (input_operand and input_operand->is_constant() and num_tms == 0) {
                std::string name = reshape_name + "_commute_clone" + std::to_string(operand_edge.edge_creation_id);

                graphlib::Node *clone = graph->add_node(reshape_input->clone(name), graph->get_subgraph_id_for_node(operand_edge.producer_node_id));
                graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(clone);
                std::vector<graphlib::OpType::Attr> reshape_attr;
                for (auto d : commute_shape.as_vector()) {
                    reshape_attr.push_back(int(d));
                }
                op->change_op_type("reshape", reshape_attr);
                op->add_golden_transform(golden_transform);

                insert_node_on_edge(graph, operand_edge, clone);
                handle_change_rank(graph, clone);
                try_consteval_input_no_operand_forks(graph, input_operand, true);
            }
        }
        calculate_and_set_node_shape(graph, node);
    }
    log_debug(LogGraphCompiler, "Reinterpret shape of output node: {} original: {} reinterprer: {}", last->name(), commute_shape, original_shape);
    last->set_shape(commute_shape);
    last->set_runtime_tensor_transform(runtime_tensor_transform);
}

void bind_reshape_to_io(graphlib::Graph *graph)
{
    bool updated = true;
    while (updated)
    {
        updated = false;
        for (graphlib::Node *node : graph->nodes())
        {
            if (node->node_type() == graphlib::NodeType::kInput)
            {
                graphlib::InputNode *input = node->as<graphlib::InputNode>();
                if (input->input_type() != graphlib::InputNodeType::Activation)
                {
                    continue;
                }
                if (input->get_runtime_tensor_transform().type != graphlib::RuntimeTensorTransformType::NoTransform)
                {
                    // Can't have multiple transforms on a node
                    continue;
                }
                // search down for commutable reshapes
                std::vector<graphlib::Node *> path = path_to_reshape_from_input(graph, node);
                if (path.empty())
                {
                    continue;
                }
                commute_eltwise_ops_to_input(graph, path);
                updated = true;
                break;
            }
            else if (node->node_type() == graphlib::NodeType::kOutput)
            {
                if (node->as<graphlib::OutputNode>()->get_runtime_tensor_transform().type !=
                    graphlib::RuntimeTensorTransformType::NoTransform)
                {
                    // Can't have multiple transforms on a node
                    continue;
                }
                // serach up
                std::vector<graphlib::Node *> path = path_to_reshape_from_output(graph, node);
                if (path.empty())
                {
                    continue;
                }
                commute_eltwise_ops_to_output(graph, path);
                updated = true;
                break;
            }
        }
    }
}
}  // namespace tt::passes
