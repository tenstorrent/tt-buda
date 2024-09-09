// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "utils.hpp"

#include <functional>
#include <map>
#include <queue>
#include <unordered_set>
#include <vector>

#include "autograd/binding.hpp"
#include "graph_lib/defines.hpp"
#include "graph_lib/edge.hpp"
#include "graph_lib/graph.hpp"
#include "graph_lib/node.hpp"
#include "graph_lib/node_types.hpp"
#include "reportify/reportify.hpp"
#include "utils/logger.hpp"

namespace tt
{

namespace graphlib
{

bool is_eltwise(const OpNode *op)
{
    bool is_buda = dynamic_cast<const BudaOpNode *>(op) != nullptr;
    py::object eval_module = py::module_::import(is_buda ? "pybuda.op.eval.buda" : "pybuda.op.eval.pybuda");
    py::function is_eltwise = eval_module.attr("is_eltwise");
    // TODO: better determination of non elementwise ops
    bool is_concatenate = op->op_name() == "concatenate";
    return is_eltwise(op->op_type()).cast<bool>() and not is_concatenate;
}

bool is_eltwise_nary(const OpNode *op)
{
    bool is_buda = dynamic_cast<const BudaOpNode *>(op) != nullptr;
    py::object eval_module = py::module_::import(is_buda ? "pybuda.op.eval.buda" : "pybuda.op.eval.pybuda");
    py::function is_eltwise_nary = eval_module.attr("is_eltwise_nary");
    return is_eltwise_nary(op->op_type()).cast<bool>();
}

bool is_eltwise_unary(const OpNode *op)
{
    bool is_buda = dynamic_cast<const BudaOpNode *>(op) != nullptr;
    py::object eval_module = py::module_::import(is_buda ? "pybuda.op.eval.buda" : "pybuda.op.eval.pybuda");
    py::function is_eltwise_unary = eval_module.attr("is_eltwise_unary");
    return is_eltwise_unary(op->op_type()).cast<bool>();
}

bool is_eltwise_binary(const OpNode *op)
{
    bool is_buda = dynamic_cast<const BudaOpNode *>(op) != nullptr;
    py::object eval_module = py::module_::import(is_buda ? "pybuda.op.eval.buda" : "pybuda.op.eval.pybuda");
    py::function is_eltwise_binary = eval_module.attr("is_eltwise_binary");
    return is_eltwise_binary(op->op_type()).cast<bool>();
}

bool is_reduce_z(OpNode const *op)
{
    return (op->op_name() == "reduce" and std::get<std::string>(op->buda_attrs().at("dim")) == "z") or
           op->has_tag("reduce_z");
}

bool default_node_filter(Node *) { return true; }

static bool requires_visit(const std::unordered_map<NodeId, bool> &visited, NodeId node_id)
{
    return visited.find(node_id) == visited.end() or visited.at(node_id) == false;
}

int get_row_size_from_tile_size(TileDim tile_dim)
{
    int ret = 32;
    switch (tile_dim)
    {
        case TileDim::Dim32x32: ret = 32; break;
        case TileDim::Dim16x32: ret = 16; break;
        case TileDim::Dim32x16: ret = 32; break;
        case TileDim::Dim8x32: ret = 8; break;
        case TileDim::Dim4x32: ret = 4; break;
        case TileDim::Dim2x32: ret = 2; break;
        case TileDim::Dim1x32: ret = 1; break;
        default: TT_ASSERT(false, "Invalid tile dim");
    }
    return ret;
}

int get_col_size_from_tile_size(TileDim tile_dim)
{
    int ret = 32;
    switch (tile_dim)
    {
        case TileDim::Dim32x32: ret = 32; break;
        case TileDim::Dim16x32: ret = 32; break;
        case TileDim::Dim32x16: ret = 16; break;
        case TileDim::Dim8x32: ret = 32; break;
        case TileDim::Dim4x32: ret = 32; break;
        case TileDim::Dim2x32: ret = 32; break;
        case TileDim::Dim1x32: ret = 32; break;
        default: TT_ASSERT(false, "Invalid tile dim");
    }
    return ret;
}

TileDim get_tile_dim_from_height_width(int tile_height, int tile_width)
{
    TileDim ret = TileDim::Dim32x32;

    switch (tile_height)
    {
        case 32:
            if (tile_width == 16)
            {
                ret = TileDim::Dim32x16;
            }
            else if (tile_width == 32)
            {
                ret = TileDim::Dim32x32;
            }
            else
            {
                TT_ASSERT(false, "Invalid tile dim");
            }
            break;
        case 16: ret = TileDim::Dim16x32; break;
        case 8: ret = TileDim::Dim8x32; break;
        case 4: ret = TileDim::Dim4x32; break;
        case 2: ret = TileDim::Dim2x32; break;
        case 1: ret = TileDim::Dim1x32; break;
        default: TT_ASSERT(false, "Invalid tile dim");
    }
    return ret;
}

void validate_tile_dims(Graph *graph, graphlib::OpNode *op_node)
{
    if (graphlib::is_eltwise_binary(op_node))
    {
        auto srcA_tile_dim = graph->operands(op_node)[0]->shape().get_tile_dim();
        auto srcB_tile_dim = graph->operands(op_node)[1]->shape().get_tile_dim();
        if (srcA_tile_dim == srcB_tile_dim)
        {
            return;
        }

        // Canonicalize tile dim for binary op
        auto srcA_tile_volume = graph->operands(op_node)[0]->shape().get_tile_height() *
                                graph->operands(op_node)[0]->shape().get_tile_width();
        auto srcB_tile_volume = graph->operands(op_node)[1]->shape().get_tile_height() *
                                graph->operands(op_node)[1]->shape().get_tile_width();

        auto srcA_shape = graph->operands(op_node)[0]->shape();
        auto srcB_shape = graph->operands(op_node)[1]->shape();

        if (srcA_tile_volume > srcB_tile_volume)
        {
            graphlib::Shape trans_shape(true, Shape::Type::FREE, srcB_shape.as_vector());
            trans_shape.set_tile_dim(srcA_tile_dim);
            auto padded_srcB_shape = graphlib::Shape::to_buda(trans_shape);
            graph->operands(op_node)[1]->set_shape(padded_srcB_shape);
        }
        else if (srcA_tile_volume < srcB_tile_volume)
        {
            graphlib::Shape trans_shape(true, Shape::Type::FREE, srcA_shape.as_vector());
            trans_shape.set_tile_dim(srcB_tile_dim);
            auto padded_srcA_shape = graphlib::Shape::to_buda(trans_shape);
            graph->operands(op_node)[0]->set_shape(padded_srcA_shape);
        }
        else
        {
            // Volume match iff 32x16 and 16x32
            // Insert NOP to make sure both inputs are padded to 32x32
            TT_ASSERT(false, "Volume match but tile dims don't match");
        }

        TT_ASSERT(
            graph->operands(op_node)[0]->shape().get_tile_dim() == graph->operands(op_node)[1]->shape().get_tile_dim());
    }
    else if (op_node->is_matmul())
    {
        // check RHS matmul, set to full tile
        auto rhs = graph->operands(op_node)[1];

        if (rhs->shape().get_tile_dim() != TileDim::Dim32x32)
        {
            graphlib::Shape trans_shape(true, Shape::Type::FREE, rhs->shape().as_vector());
            trans_shape.set_tile_dim(TileDim::Dim32x32);
            auto padded_rhs_shape = graphlib::Shape::to_buda(trans_shape);
            rhs->set_shape(padded_rhs_shape);
        }
    }
    else if (op_node->op_type().op == "reduce")
    {
        auto operand = graph->operands(op_node)[0];
        if (operand->shape().get_tile_dim() != TileDim::Dim32x32)
        {
            graphlib::Shape trans_shape(true, Shape::Type::FREE, operand->shape().as_vector());
            trans_shape.set_tile_dim(TileDim::Dim32x32);
            auto padded_shape = graphlib::Shape::to_buda(trans_shape);
            operand->set_shape(padded_shape);
        }
    }
    else if (op_node->op_type().op == "embedding")
    {
        for (auto operand : graph->operands(op_node))
        {
            if (operand->shape().get_tile_dim() != TileDim::Dim32x32)
            {
                graphlib::Shape trans_shape(true, Shape::Type::FREE, operand->shape().as_vector());
                trans_shape.set_tile_dim(TileDim::Dim32x32);
                auto padded_shape = graphlib::Shape::to_buda(trans_shape);
                operand->set_shape(padded_shape);
            }
        }
    }

    return;
}

std::vector<std::vector<Node *>> topological_generations(const Graph &graph)
{
    std::vector<std::vector<Node *>> generations;

    // the first step is to discover top level nodes in the graph
    // queue up all visible nodes
    std::vector<Node *> nodes = graph.nodes();
    std::queue<Node *> node_queue;
    for (Node *node : nodes)
    {
        if (graph.is_node_visible(node))
        {
            node_queue.push(node);
        }
    }
    // vector to store top level nodes
    std::vector<Node *> top_level_nodes;
    std::unordered_map<NodeId, bool> visited{};

    std::function<void(Node *)> VisitNode = [&](Node *node)
    {
        visited[node->id()] = true;

        // count the number of operands of the node
        int num_operands = 0;
        for (const Edge &operand_edge : graph.operand_edges(node))
        {
            if (operand_edge.edge_type == EdgeType::kDataLoopback or
                operand_edge.edge_type == EdgeType::kPartialDataCopy)
            {
                continue;
            }
            else if (operand_edge.edge_type == EdgeType::kControlLoop)
            {
                continue;  // not unrolling loops, just terminate
            }
            num_operands++;

            NodeId predecessor_id = operand_edge.producer_node_id;
            Node *predecessor_node = graph.node_by_id(predecessor_id);
            if (requires_visit(visited, predecessor_id))
            {
                VisitNode(predecessor_node);
            }
        }
        if (num_operands == 0)
        {
            top_level_nodes.push_back(node);
        }
    };

    // recurse through node operands until top, then stop, and add to result
    while (not node_queue.empty())
    {
        Node *node = node_queue.front();

        if (requires_visit(visited, node->id()))
        {
            VisitNode(node);
        }
        node_queue.pop();
    }

    // now do a BFS through nodes
    std::queue<Node *> bfs_queue;

    // also store a mapping of each node to its level (or generation)
    std::unordered_map<NodeId, unsigned> node_to_level;

    // add top level nodes to the queue
    for (Node *node : top_level_nodes)
    {
        bfs_queue.push(node);
        node_to_level[node->id()] = 0;
    }

    // iterate through the queue
    // store processed nodes in a set
    std::unordered_set<NodeId> processed_nodes;
    while (not bfs_queue.empty())
    {
        Node *node = bfs_queue.front();
        bfs_queue.pop();

        // queue eligible children of this node
        for (const Edge &user_edge : graph.user_edges(node))
        {
            if (user_edge.edge_type == EdgeType::kControlLoop)
            {
                continue;  // not unrolling loops, just terminate
            }
            if (user_edge.edge_type == EdgeType::kDataLoopback or user_edge.edge_type == EdgeType::kPartialDataCopy)
            {
                continue;
            }
            NodeId user_id = user_edge.consumer_node_id;
            Node *user_node = graph.node_by_id(user_id);

            // if this node has already been processed, then skip it
            if (processed_nodes.find(user_id) != processed_nodes.end())
            {
                continue;
            }

            // if all the operands of this node already have levels, then this node will be inserted into the queue
            bool all_operands_have_levels = true;
            unsigned level = 0;
            for (const Edge &operand_edge : graph.operand_edges(user_node))
            {
                if (operand_edge.edge_type == EdgeType::kDataLoopback or
                    operand_edge.edge_type == EdgeType::kPartialDataCopy)
                {
                    continue;
                }
                else if (operand_edge.edge_type == EdgeType::kControlLoop)
                {
                    continue;  // not unrolling loops, just terminate
                }
                NodeId operand_id = operand_edge.producer_node_id;
                if (node_to_level.find(operand_id) == node_to_level.end())
                {
                    all_operands_have_levels = false;
                    break;
                }
                else
                {
                    level = std::max(level, node_to_level[operand_id]);
                }
            }
            // insert node into queue if all operands have levels
            if (all_operands_have_levels)
            {
                bfs_queue.push(user_node);
                node_to_level[user_id] = level + 1;
                // mark node as processed
                processed_nodes.insert(user_id);
            }
        }
    }

    // now that we have the levels, we can create the generations
    for (auto const &[node_id, level] : node_to_level)
    {
        if (generations.size() <= level)
        {
            generations.resize(level + 1);
        }
        generations[level].push_back(graph.node_by_id(node_id));
    }

    return generations;
}

std::vector<Node *> top_row(graphlib::Graph const *graph, std::vector<Node *> const &nodes)
{
    std::vector<Node *> sorted_nodes;

    // find the first generation that contains at least one of the nodes
    // iterate over each generation in topological_generations
    for (auto const &generation : topological_generations(*graph))
    {
        // iterate over each node to check if it belongs to this generation
        for (auto *n : nodes)
        {
            if (std::find(generation.begin(), generation.end(), n) != generation.end())
            {
                sorted_nodes.push_back(n);
            }
        }
        // if sorted_nodes is not empty, then we have found the first generation that contains at least one of the nodes
        if (sorted_nodes.size() > 0)
        {
            return sorted_nodes;
        }
    }
    return sorted_nodes;
}

std::vector<Node *> bot_row(graphlib::Graph const *graph, std::vector<Node *> const &nodes)
{
    std::vector<Node *> sorted_nodes;

    // find the last generation that contains at least one of the nodes
    // iterate over each generation in topological_generations in reverse order
    auto generations = topological_generations(*graph);
    // number of generations
    int num_generations = generations.size();

    // iterate over each generation in reverse order
    for (auto g = 0; g < num_generations; g++)
    {
        auto generation = generations[num_generations - g - 1];

        // iterate over each node to check if it belongs to this generation
        for (auto *n : nodes)
        {
            if (std::find(generation.begin(), generation.end(), n) != generation.end())
            {
                sorted_nodes.push_back(n);
            }
        }
        // if sorted_nodes is not empty, then we have found the last generation that contains at least one of the nodes
        if (sorted_nodes.size() > 0)
        {
            return sorted_nodes;
        }
    }
    return sorted_nodes;
}

std::vector<Node *> topological_sort(const Graph &graph, std::function<bool(Node *)> node_filter, bool unroll_loops)
{
    std::vector<Node *> result;
    std::unordered_map<NodeId, bool> visited{};
    std::unordered_map<Edge, int> control_loop_edge_to_iteration;

    std::vector<Node *> nodes = graph.nodes();
    std::queue<Node *> node_queue;
    for (Node *node : nodes)
    {
        if (graph.is_node_visible(node))
        {
            node_queue.push(node);
        }
    }

    std::function<void(Node *)> VisitNode = [&](Node *node)
    {
        visited[node->id()] = true;

        for (const Edge &operand_edge : graph.operand_edges(node))
        {
            if (operand_edge.edge_type == EdgeType::kDataLoopback or
                operand_edge.edge_type == EdgeType::kPartialDataCopy)
            {
                continue;
            }
            else if (operand_edge.edge_type == EdgeType::kControlLoop)
            {
                continue;  // not unrolling loops, just terminate
            }

            NodeId predecessor_id = operand_edge.producer_node_id;
            Node *predecessor_node = graph.node_by_id(predecessor_id);
            if (requires_visit(visited, predecessor_id))
            {
                VisitNode(predecessor_node);
            }
        }
        if (node_filter(node))
        {
            result.push_back(node);
        }

        if (unroll_loops)
        {
            for (const Edge &user_edge : graph.user_edges(node))
            {
                if (user_edge.edge_type == EdgeType::kControlLoop)
                {
                    auto loop_attributes = EdgeAttributes::as<LoopEdgeAttributes>(graph.get_edge_attributes(user_edge));
                    if (control_loop_edge_to_iteration.find(user_edge) == control_loop_edge_to_iteration.end())
                    {
                        control_loop_edge_to_iteration[user_edge] = 1;  // initialize loop count
                    }
                    if (control_loop_edge_to_iteration[user_edge] < loop_attributes->loop_iterations())
                    {
                        // Re-enqueue nodes in the same order they were originally intended to be processed
                        for (Node *node : nodes)
                        {
                            if (loop_attributes->is_processed_in_loop(node->id()))
                            {
                                visited[node->id()] = false;
                                node_queue.push(node);
                            }
                        }
                    }
                    control_loop_edge_to_iteration[user_edge] += 1;
                }
            }
        }
    };

    while (not node_queue.empty())
    {
        Node *node = node_queue.front();

        if (requires_visit(visited, node->id()))
        {
            VisitNode(node);
        }
        node_queue.pop();
    }
    return result;
}

void fork_subgraph(Graph *graph, Node *node) {
    // If the node passed is an input node then just fork it
    graphlib::InputNode *input = dynamic_cast<graphlib::InputNode *>(node);
    graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(node);
    if (input) {
        input->get_consteval_graph(graph, true, true); // create graph before clone so input node name is correct
        std::vector<graphlib::Edge> user_edges = graph->user_data_edges(input);
        TT_ASSERT(graph->data_operands(input).size() == 0, "Input can't have operands");
        std::vector<graphlib::Node *> removed_to_forked;
        for (int i = 0; i < (int)user_edges.size(); i++)
        {
            graphlib::Edge const &user_edge = user_edges[i];
            log_trace(
                LogConstEval,
                "fork_subgraph: cloning: {} -> {}",
                input->name(),
                graph->node_by_id(user_edge.consumer_node_id)->name());
            
            std::string clone_name = input->name() + "_subgraph_fork_clone_" + std::to_string(user_edge.edge_creation_id);
            TaggedNode *clone = graph->add_node(
                input->clone(clone_name), 
                graph->get_subgraph_id_for_node(input->id()))->as<TaggedNode>();

            clone->tag("forked_from", input->name());
            auto attr = graph->get_edge_attributes(user_edge);
            graph->remove_edge(user_edge);
            // Replace user operand_edge
            Edge new_user_edge = Edge(clone->id(), user_edge.producer_output_port_id, user_edge.consumer_node_id, user_edge.consumer_input_port_id, user_edge.edge_type);
            
            graph->add_edge(new_user_edge, attr);
            removed_to_forked.push_back(clone);
        }

        graphlib::Node *first_forked = removed_to_forked[0];
        auto removed_node = graph->remove_node(input);

        // Need to maintain original name because user can access it by name
        first_forked->set_name(removed_node->name());
        
    }
    else if (op) {
        std::vector<Edge> user_edges = graph->user_data_edges(op);
        std::vector<Edge> operand_edges = graph->operand_data_edges(op);

        // Clone this op once for every user
        for (int i = 1; i < (int)user_edges.size(); i++)
        {
            graphlib::Edge const &user_edge = user_edges[i];
            log_trace(
                LogConstEval,
                "fork_subgraph: cloning: {} -> {}",
                op->name(),
                graph->node_by_id(user_edge.consumer_node_id)->name());

            std::string clone_name = op->name() + "_subgraph_fork_clone_" + std::to_string(user_edge.edge_creation_id);
            Node *clone_op = graph->add_node(op->clone(clone_name), graph->get_subgraph_id_for_node(op->id()));

            // Copy all the operand edges
            for (int j = 0; j < (int)operand_edges.size(); j++) {
                Edge operand_edge = operand_edges[j];
                Edge new_edge = Edge(operand_edge.producer_node_id, operand_edge.producer_output_port_id, clone_op->id(), operand_edge.consumer_input_port_id, operand_edge.edge_type);
                graph->add_edge(new_edge, graph->get_edge_attributes(operand_edge));
            }

            // Replace user operand_edge
            Edge new_user_edge = Edge(clone_op->id(), i, user_edge.consumer_node_id, user_edge.consumer_input_port_id, user_edge.edge_type);
            
            graph->add_edge(new_user_edge, graph->get_edge_attributes(user_edge));
            graph->remove_edge(user_edge);
        }

        // Fork the graph of each operand
        for (auto operand_edge : graph->operand_data_edges(op)) {
            fork_subgraph(graph, graph->node_by_id(operand_edge.producer_node_id));
        }

    }
    else {
        TT_ASSERT(false, "The node passed must be an InputNode or OpNode");
    }
}

std::vector<Node *> visible_nodes(Graph const &graph, std::function<bool(Node *)> node_filter)
{
    std::vector<Node *> result;

    for (Node *node : graph.nodes())
    {
        if (graph.is_node_visible(node) and node_filter(node))
        {
            result.push_back(node);
        }
    }

    return result;
}

std::vector<Node *> reachable_nodes(
    const Graph *graph, Node *start, std::function<bool(Node *)> node_filter, bool ancenstors_only)
{
    std::vector<Node *> result;
    std::unordered_map<NodeId, bool> visited{};

    std::function<void(Node *)> VisitNode = [&](Node *node)
    {
        visited[node->id()] = true;

        for (auto operand : graph->data_operands(node))
        {
            if (requires_visit(visited, operand->id()))
            {
                VisitNode(operand);
            }
        }
        if (node->node_type() != NodeType::kInput and not ancenstors_only)
        {
            for (auto user : graph->data_users(node))
            {
                if (requires_visit(visited, user->id()))
                {
                    VisitNode(user);
                }
            }
        }
        if (node_filter(node))
        {
            result.push_back(node);
        }
    };

    VisitNode(start);

    return result;
}

// Check if there exists a data edge between the two nodes(producer, consumer )
bool check_producer_consumer(Graph *graph, Node *producer, Node *consumer, std::function<bool(Node *)> node_filter)
{
    std::vector<graphlib::Node *> rc_nodes = reachable_nodes(graph, producer, node_filter, true);

    // if there exists a dependency between the two given nodes, return true
    return (std::find(rc_nodes.begin(), rc_nodes.end(), consumer) != rc_nodes.end());
}

// Find the longest path from the graph. Optionally look for paths that don't start from ordered inputs.
// TODO: write a few unit tests
std::vector<Node *> get_longest_path(const Graph *graph, bool from_inputs_only)
{
    std::unordered_map<Node *, int> cost;
    std::unordered_map<Node *, Node *> parent_map;

    if (from_inputs_only)
    {
        // set big negative numbers on all other inputs
        for (Node *node : graph->nodes()) cost.emplace(std::make_pair(node, std::numeric_limits<int>::lowest()));
        for (Node *node : graph->ordered_module_inputs()) cost[node] = 0;
    }

    int max_distance = std::numeric_limits<int>::lowest();
    Node *max_path_output = NULL;
    for (Node *node : topological_sort(*graph))
    {
        for (Node *user : graph->data_users(node))
        {
            if (cost[user] < cost[node] + 1)
            {
                cost[user] = cost[node] + 1;
                parent_map[user] = node;
            }
            if (cost[node] > max_distance)
            {
                max_distance = cost[node];
                max_path_output = node;
            }
        }
    }

    std::vector<Node *> max_path = {max_path_output};
    while (parent_map.find(max_path_output) != parent_map.end())
    {
        max_path_output = parent_map.at(max_path_output);
        max_path.push_back(max_path_output);
    }

    std::reverse(max_path.begin(), max_path.end());

    return max_path;
}

std::vector<Node *> get_nodes_with_indegree_zero(Graph *graph)
{
    std::vector<Node *> indegree_zero_nodes;
    for (Node *node : graph->nodes())
    {
        int num_operands = 0;
        for (auto operand : graph->operands(node))
        {
            if (operand->node_type() != NodeType::kInput)
            {
                num_operands++;
            }
        }
        if (num_operands == 0)
        {
            if (node->node_type() != NodeType::kInput)
            {
                indegree_zero_nodes.push_back(node);
            }
        }
    }
    return indegree_zero_nodes;
}

std::vector<Node *> get_nodes_with_outdegree_zero(Graph *graph)
{
    std::vector<Node *> outdegree_zero_nodes;
    for (Node *node : graph->nodes())
    {
        if (graph->users(node).size() == 0)
        {
            if (node->node_type() != NodeType::kInput)
            {
                outdegree_zero_nodes.push_back(node);
            }
        }
    }
    return outdegree_zero_nodes;
}

std::vector<Node *> get_nodes_with_data_outdegree_zero(Graph *graph)
{
    std::vector<Node *> outdegree_zero_nodes;
    for (Node *node : graph->nodes())
    {
        if (graph->user_data_edges(node).size() == 0)
        {
            if (node->node_type() != NodeType::kInput)
            {
                outdegree_zero_nodes.push_back(node);
            }
        }
    }
    return outdegree_zero_nodes;
}

// Insert new node on the given edge. Node attributes will be picked up from consumer node.
std::pair<Edge, Edge> insert_node_on_edge(
    Graph *graph,
    Edge &edge,
    Node *node,
    bool inherit_consumer_attrs,
    bool remove_edge,
    std::uint32_t consumer_index,
    bool place_tms_on_outgoing)
{
    Node *consumer = graph->node_by_id(edge.consumer_node_id);
    Node *producer = graph->node_by_id(edge.producer_node_id);

    graph->copy_node_attributes(inherit_consumer_attrs ? consumer : producer, node);

    // Don't copy "gradient op" flag, since the last node is still the one accumulating
    if ((node->node_type() == NodeType::kBudaOp) || (node->node_type() == NodeType::kPyOp))
        node->as<graphlib::OpNode>()->set_gradient_op(false);

    // Create new edges
    Edge new_edge0 =
        Edge(edge.producer_node_id, edge.producer_output_port_id, node->id(), consumer_index, edge.edge_type);

    Edge new_edge1 = Edge(node->id(), 0, edge.consumer_node_id, edge.consumer_input_port_id, edge.edge_type);

    graph->add_edge(new_edge0);
    graph->add_edge(new_edge1);

    graph->copy_edge_attributes(edge, new_edge0);
    graph->copy_edge_attributes(edge, new_edge1);

    // TMs should be placed only on one of the edges.
    // Since we've copied all edge attributes (including TMs) to both edges,
    // we need to remove TMs from one of them.
    if (not place_tms_on_outgoing)
    {
        graph->get_edge_attributes(new_edge1)->set_tms({});
    }
    else
    {
        graph->get_edge_attributes(new_edge0)->set_tms({});
    }

    bool edges_added = false;
    for (Edge &e : graph->operand_edges(consumer))
    {
        // Adjust control & autograd edges
        if ((e.edge_type != EdgeType::kData) && (e.edge_type != EdgeType::kAutogradOutputToLoss) &&
            (e.edge_type != EdgeType::kAutogradInputToGradientOut) &&
            (e.edge_type != EdgeType::kAutogradFwdToGradient) && (e.edge_type != EdgeType::kAutogradFwdToRecompute)

        )
        {
            edges_added = true;
            graph->add_edge(graph->node_by_id(e.producer_node_id), node, e.producer_output_port_id, 0, e.edge_type);
        }
    }

    // If the producer was in backward (or optimizer) epoch, and there are fwd->bwd edges going to it,
    // the need to go to the new op, too
    if (not edges_added and producer->get_epoch_type() != graphlib::NodeEpochType::Forward)
    {
        for (Edge &e : graph->operand_edges(producer))
        {
            // Adjust control & autograd edges
            if ((e.edge_type == EdgeType::kAutogradFwdToBwd) || (e.edge_type == EdgeType::kAutogradFwdToOptimizer) ||
                (e.edge_type == EdgeType::kAutogradFwdToGradient))
            {
                graph->add_edge(graph->node_by_id(e.producer_node_id), node, e.producer_output_port_id, 0, e.edge_type);
            }
            // Move the kAutogradFwdToGradient edge, since we can only have one
            if (e.edge_type == EdgeType::kAutogradFwdToGradient)
            {
                graph->remove_edge(e);
            }
        }
    }
    // If the consumer of the edge we're trying to add a node on is a "recompute-node",
    // we need to also create replicated fwd->recompute edges on the newly added node.
    // this is to keep track of which nodes are considered to be "recompute".
    for (Edge &e : graph->operand_edges(consumer))
    {
        if (e.edge_type == EdgeType::kAutogradFwdToRecompute)
        {
            Node *fwd_node_being_recompute = graph->node_by_id(e.producer_node_id);
            graph->add_edge(fwd_node_being_recompute, node, e.producer_output_port_id, 0, e.edge_type);
        }
    }

    if (remove_edge)
    {
        graph->remove_edge(edge);
    }

    return std::make_pair(new_edge0, new_edge1);
}

std::tuple<BudaOpNode*, Edge, Edge> insert_nop_on_edge(Graph *graph, Edge &edge, const std::string &nop_name, bool is_buffering, bool hoist_tms, bool remove_edge)
{
    const Node *src = graph->node_by_id(edge.producer_node_id);
    const Node *dest = graph->node_by_id(edge.consumer_node_id);

    BudaOpNode *nop = graph->add_node(
        graphlib::create_node<graphlib::BudaOpNode>(nop_name, "nop"),
        graph->get_subgraph_id_for_node(src->id()));
    nop->set_shape(src->shape());
    nop->set_buffering_op(is_buffering);
    nop->as<TaggedNode>()->tag("fj_buffering_nop", is_buffering);

    nop->set_epoch_type(dest->get_epoch_type());
    nop->set_output_df(src->output_df());

    if (src->node_type() == NodeType::kBudaOp)
    {
        const BudaOpNode *src_op = src->as<BudaOpNode>();
        if (src_op->op_name() != "dequantization")
        {
            nop->set_accumulate_df(src_op->accumulate_df());
            nop->set_intermediate_df(src_op->intermediate_df());
            nop->set_math_fidelity(src_op->math_fidelity());
        }
    }

    auto [edge0, edge1] = insert_node_on_edge(graph, edge, nop, false, remove_edge, 0 /* consumer_index */, not hoist_tms);

    return std::make_tuple(nop, edge0, edge1);
}

// Copy non-data edges from old dest to new
void copy_control_edges(Graph *graph, Node *old_dest, Node *new_dest)
{
    std::vector<Node *> data_operands = graph->data_operands(old_dest);
    Node *data_operand = data_operands.at(0);

    for (Edge &e : graph->operand_edges(old_dest))
    {
        if (e.edge_type == EdgeType::kData)
        {
            continue;
        }
        Node *new_consumer = data_operand;

        if (new_consumer->node_type() != NodeType::kBudaOp)
        {
            // If `new_dest` is an OutputNode, we'll fetch it off of its data-operand since we still want to
            // copy this control edge over (consider kInputToGradient being connected to kOutput node)
            new_consumer = data_operand;
        }

        if (new_consumer->node_type() != NodeType::kBudaOp)
        {
            continue;
        }

        if ((e.edge_type == EdgeType::kAutogradFwdToBwd and
             new_consumer->get_epoch_type() != NodeEpochType::Backward) or
            (e.edge_type == EdgeType::kAutogradFwdToOptimizer and
             new_consumer->get_epoch_type() != NodeEpochType::Optimizer))
        {
            // There are cases where we're trying to connect kAutogradFwdToBwd on a Fwd consumer node which doesn't make
            // sense.
            continue;
        }

        // Copy control & autograd edges
        graph->add_edge(
            graph->node_by_id(e.producer_node_id),
            new_consumer,
            e.producer_output_port_id,
            e.consumer_input_port_id,
            e.edge_type);
    }

    for (Edge &e : graph->user_edges(old_dest))
    {
        if (e.edge_type == EdgeType::kData)
        {
            continue;
        }

        // Copy control & autograd edges
        if (e.edge_type == EdgeType::kControl)
        {
            graph->add_edge(new_dest, graph->node_by_id(e.consumer_node_id), 0, 0, e.edge_type);
        }
        else
        {
            // if it's an autograd-edge between <NODE_TO_DELETE> -> consumer, we'll reassign
            // the edge to the producer node since `new_dest` may be an output node
            graph->add_edge(data_operand, graph->node_by_id(e.consumer_node_id), 0, 0, e.edge_type);
        }
    }
}

// Copy non-data edges when removing a node
void handle_control_edges_when_removing_node(Graph *graph, Node *node_being_removed)
{
    std::vector<Edge> operand_data_edges = graph->operand_data_edges(node_being_removed);
    TT_ASSERT(
        operand_data_edges.size() == 1,
        "Tried to handle control edges, but node being removed has more than 1 operand!");

    Edge &producer_to_nbr_edge = operand_data_edges.front();
    Node *producer = graph->node_by_id(producer_to_nbr_edge.producer_node_id);

    auto is_not_data_edge = [](Edge e) { return (e.edge_type != EdgeType::kData); };
    std::vector<Edge> operand_edges = graph->operand_edges(node_being_removed, is_not_data_edge);
    std::vector<Edge> user_edges = graph->user_edges(node_being_removed, is_not_data_edge);

    // Handle operand edges
    for (Edge &o_e : operand_edges)
    {
        if (node_being_removed->is_forward())
        {
            if (o_e.edge_type == EdgeType::kControl)
            {
                for (Edge &user : graph->user_data_edges(node_being_removed))
                {
                    Edge new_edge(
                        o_e.producer_node_id,
                        o_e.producer_output_port_id,
                        user.consumer_node_id,
                        user.consumer_input_port_id,
                        o_e.edge_type);
                    graph->add_edge(new_edge);
                }
            }
            else
            {
                TT_ASSERT(false, "Unexpected edge type: {}", tt::graphlib::edge_type_to_string(o_e.edge_type));
            }
        }
        else if (node_being_removed->is_backward())
        {
            if (o_e.edge_type == EdgeType::kAutogradFwdToBwd)
            {
                // We can just "delete" this edge, i.e. not copy it
                continue;
            }
            if (o_e.edge_type == EdgeType::kAutogradFwdToGradient)
            {
                graph->add_edge(graph->node_by_id(o_e.producer_node_id), producer, o_e.edge_type);
                continue;
            }
            if (o_e.edge_type == EdgeType::kAutogradFwdToRecompute)
            {
                // We can just "delete" this edge, i.e. not copy it
                continue;
            }

            TT_ASSERT(false, "Unexpected edge type: {}", tt::graphlib::edge_type_to_string(o_e.edge_type));
        }
        else if (node_being_removed->is_optimizer())
        {
            if (o_e.edge_type == EdgeType::kAutogradFwdToOptimizer)
            {
                // We can just "delete" this edge, i.e. not copy it
                continue;
            }

            TT_ASSERT(false, "Unexpected edge type: {}", tt::graphlib::edge_type_to_string(o_e.edge_type));
        }

        // TODO: Other control edges
    }

    // Handle user edges
    for (Edge &u_e : user_edges)
    {
        if (node_being_removed->is_forward())
        {
            if (u_e.edge_type == EdgeType::kAutogradFwdToBwd)
            {
                // Push the edge to parent of node being removed
                graph->add_edge(producer, graph->node_by_id(u_e.consumer_node_id), u_e.edge_type);
                continue;
            }
            if (u_e.edge_type == EdgeType::kAutogradFwdToOptimizer)
            {
                graph->add_edge(producer, graph->node_by_id(u_e.consumer_node_id), u_e.edge_type);
                continue;
            }
            if (u_e.edge_type == EdgeType::kAutogradFwdToGradient)
            {
                // Since there will be no fwd node anymore, we can just delete this edge
                continue;
            }
            if (u_e.edge_type == EdgeType::kAutogradFwdToRecompute)
            {
                // Moving this edge from nbr(fwd)->recompute(bwd) to nbr's_parent(fwd)->recompute(bwd)
                // Not sure this makes sense though, depends what the edge is used for later on
                graph->add_edge(producer, graph->node_by_id(u_e.consumer_node_id), u_e.edge_type);
                continue;
            }

            TT_ASSERT(false, "Unexpected edge type: {}", tt::graphlib::edge_type_to_string(u_e.edge_type));
        }
        else if (node_being_removed->is_backward())
        {
            TT_ASSERT(false, "Unexpected edge type: {}", tt::graphlib::edge_type_to_string(u_e.edge_type));
        }
        else if (node_being_removed->is_optimizer())
        {
            TT_ASSERT(false, "Unexpected edge type: {}", tt::graphlib::edge_type_to_string(u_e.edge_type));
        }
        // TODO: Other control edges
    }
}

// Creates buffering queue and adds it to the graph. Returns pointer to created queue node.
// Queue inherits shape output_df, and epoch_type from producer node.
graphlib::QueueNode *create_buffering_queue(
    Graph *graph, const graphlib::Node *producer_node, const std::string name, int num_entries)
{
    TT_ASSERT(num_entries > 0, "Number of entries in queue has to be greater than 0");
    if (num_entries > graph->get_microbatch())
    {
        log_warning(
            "Wasting DRAM. Number of entries in queue is greater than microbatch size. For buffering queue the "
            "theoretical maximum number of entries is equal to microbatch size.");
    }

    // Create new queue
    std::unique_ptr<graphlib::BufferingQueueNode> queue_node_unique =
        graphlib::create_node<graphlib::BufferingQueueNode>(name, num_entries);
    queue_node_unique->set_shape(producer_node->shape());
    queue_node_unique->set_output_df(producer_node->output_df());
    queue_node_unique->set_epoch_type(producer_node->get_epoch_type());

    graphlib::QueueNode *queue =
        graph->add_node(std::move(queue_node_unique), graph->get_subgraph_id_for_node(producer_node->id()));
    return queue;
}

// Bypass queue, connecting its source to its destination. There has to be only one source for queue, and user is
// defined by user_edge.
std::unique_ptr<Node> connect_queue_src_to_queue_user(Graph *graph, Node *queue, Edge &user_edge, bool remove_queue)
{
    TT_ASSERT(queue->node_type() == NodeType::kQueue, " provided node has to be NodeType::kQueue");
    std::vector<Edge> op_edges = graph->operand_data_edges(queue);
    TT_ASSERT(op_edges.size() == 1, "connect_queue_src_to_queue_user can only be called on nodes with one operand");

    Edge src_edge = op_edges[0];
    std::vector<graphlib::OpType> operand_tms = graph->get_edge_attributes(src_edge)->get_tms();

    // if we want to remove queue at the end, we won't remove user_edge now since it will be done in
    // graph->remove_node() if we only wan't to connect queue src to its dest (determined by user_edge), we will delete
    // user_edge.
    std::shared_ptr<EdgeAttributes> user_edge_attrs =
        remove_queue ? graph->get_edge_attributes(user_edge) : graph->remove_edge(user_edge);
    std::vector<graphlib::OpType> user_tms = user_edge_attrs->get_tms();

    Edge new_edge(
        src_edge.producer_node_id,
        src_edge.producer_output_port_id,
        user_edge.consumer_node_id,
        user_edge.consumer_input_port_id,
        user_edge.edge_type);
    graph->add_edge(new_edge);

    std::vector<graphlib::OpType> new_edge_tms;
    new_edge_tms.insert(new_edge_tms.end(), operand_tms.begin(), operand_tms.end());
    new_edge_tms.insert(new_edge_tms.end(), user_tms.begin(), user_tms.end());

    auto new_edge_attributes = graph->get_edge_attributes(new_edge);
    new_edge_attributes->set_tms(new_edge_tms);
    new_edge_attributes->set_ublock_order(user_edge_attrs->get_ublock_order());

    return remove_queue ? graph->remove_node(queue) : nullptr;
}

// Bypass node, connecting its source to its destination(s). The node must only have one input operand.
// Optionally, user can provide callback on each of the newly created edges, and original edge.
std::unique_ptr<Node> bypass_node(Graph *graph, Node *node, bool remove_node, std::function<void(Edge, Edge)> callback)
{
    std::vector<Edge> op_edges = graph->operand_data_edges(node);
    TT_ASSERT(op_edges.size() == 1, "bypass_node can only be called on nodes with one operand");

    Edge src_edge = op_edges[0];
    std::vector<graphlib::OpType> operand_tms = graph->get_edge_attributes(src_edge)->get_tms();

    for (Edge &user : graph->user_data_edges(node))
    {
        std::vector<graphlib::OpType> user_tms = graph->get_edge_attributes(user)->get_tms();

        Edge new_edge(
            src_edge.producer_node_id,
            src_edge.producer_output_port_id,
            user.consumer_node_id,
            user.consumer_input_port_id,
            user.edge_type);
        graph->add_edge(new_edge);

        std::vector<graphlib::OpType> new_edge_tms;
        new_edge_tms.insert(new_edge_tms.end(), operand_tms.begin(), operand_tms.end());
        new_edge_tms.insert(new_edge_tms.end(), user_tms.begin(), user_tms.end());

        auto new_edge_attributes = graph->get_edge_attributes(new_edge);
        new_edge_attributes->set_tms(new_edge_tms);
        new_edge_attributes->set_ublock_order(graph->get_edge_attributes(user)->get_ublock_order());

        callback(new_edge, user);
    }

    handle_control_edges_when_removing_node(graph, node);

    OpNode *op_node = dynamic_cast<OpNode *>(node);
    if (op_node and op_node->is_gradient_op())
    {
        OpNode *producer_op_node = dynamic_cast<OpNode *>(graph->node_by_id(src_edge.producer_node_id));
        if (producer_op_node)
            producer_op_node->set_gradient_op();
    }

    return remove_node ? graph->remove_node(node) : nullptr;
}

// Replace node with a new one, removing the old one and reconnecting all edges as before.
// The new node must have the same number of operands, or skip_operands must be set.
void replace_node(Graph *graph, Node *original_node, Node *new_node, bool skip_operands)
{
    if (!skip_operands)
    {
        for (Edge &operand : graph->operand_data_edges(original_node))
        {
            Edge new_edge = Edge(
                operand.producer_node_id,
                operand.producer_output_port_id,
                new_node->id(),
                operand.consumer_input_port_id,
                operand.edge_type);
            graph->add_edge(new_edge);
            graph->copy_edge_attributes(operand, new_edge);
        }
    }

    for (Edge &user : graph->user_edges(original_node))
    {
        if (user.edge_type == graphlib::EdgeType::kData)
        {
            Edge new_edge = Edge(
                new_node->id(),
                (graphlib::PortId)0,
                user.consumer_node_id,
                user.consumer_input_port_id,
                user.edge_type);
            graph->add_edge(new_edge);
            graph->copy_edge_attributes(user, new_edge);
        }
    }

    copy_control_edges(graph, original_node, new_node);
    graph->copy_node_attributes(original_node, new_node);
    graph->remove_node(original_node);
}

Edge swap(Graph *graph, Edge edge, std::function<void(Edge)> operand_callback, std::function<void(Edge)> user_callback)
{
    auto replace_edge = [graph](Edge orig, Edge new_edge)
    {
        auto attr = graph->get_edge_attributes(orig);
        graph->remove_edge(orig);
        graph->add_edge(new_edge, attr);
    };

    Node *producer = graph->node_by_id(edge.producer_node_id);
    Node *consumer = graph->node_by_id(edge.consumer_node_id);
    auto producer_operands = graph->operand_data_edges(producer);
    auto consumer_users = graph->user_data_edges(consumer);

    TT_ASSERT(producer_operands.size() == 1, "swap is only compatible with unary producers");

    // Swap the orientation of the original edge
    auto swapped_edge = edge;
    std::swap(swapped_edge.producer_node_id, swapped_edge.consumer_node_id);
    swapped_edge.consumer_input_port_id = 0;
    replace_edge(edge, swapped_edge);

    // Producer operand point to consumer
    auto producer_operand = producer_operands.front();
    auto remap_producer = producer_operand;
    remap_producer.consumer_node_id = consumer->id();
    remap_producer.consumer_input_port_id = edge.consumer_input_port_id;
    replace_edge(producer_operand, remap_producer);

    for (auto const &operand : graph->operand_data_edges(consumer))
    {
        operand_callback(operand);
    }

    // Consumer users map to producer
    for (auto const &user : consumer_users)
    {
        auto new_user = user;
        new_user.producer_node_id = producer->id();
        replace_edge(user, new_user);
        user_callback(new_user);
    }

    return swapped_edge;
}

std::vector<Node *> subgraph(const Graph *graph, Node *producer, Node *consumer)
{
    bool found = false;
    std::unordered_map<Node *, std::vector<Node *>> deps;
    std::unordered_set<Node *> visited;
    std::vector<Node *> visit = {producer};
    while (not visit.empty())
    {
        Node *node = visit.back();
        visit.pop_back();
        for (Node *user : graph->data_users(node))
        {
            deps[user].push_back(node);

            if (user == consumer)
            {
                // We can stop visiting this path since we hit the consumer
                found = true;
            }
            else if (visited.find(user) == visited.end())
            {
                // Only continue to visit nodes that haven't been visited yet
                visit.push_back(user);
            }

            visited.insert(user);
        }
    }

    if (not found)
        return {};

    std::vector<Node *> sub;

    visit = deps[consumer];
    while (not visit.empty())
    {
        std::vector<Node *> next;
        for (Node *node : visit)
        {
            if (node == producer)
                continue;

            sub.push_back(node);
            auto const &d = deps.at(node);
            next.insert(next.end(), d.begin(), d.end());
        }
        std::swap(visit, next);
    }

    return sub;
}

void convert_implicit_to_explicit_bcasts(Graph *graph, Edge edge)
{
    auto edge_attr = graph->get_edge_attributes(edge);
    for (OpType &op_type : graph->get_edge_attributes(edge)->get_tms())
    {
        if (op_type.op == "broadcast")
        {
            constexpr bool explicit_bcast = true;
            std::get<bool>(op_type.attr[2]) = explicit_bcast;
        }
    }
}

graphlib::Node *cascade_nary_to_binary_op(graphlib::Graph *graph, graphlib::Node *nary_op)
{
    auto operands = graph->operand_data_edges(nary_op);
    TT_ASSERT(operands.size() >= 2, nary_op->name(), operands.size());
    if (operands.size() == 2)
        return nary_op;

    graphlib::Node *sink = graph->add_node(
        nary_op->clone(nary_op->name() + "_cascade_sink"), graph->get_subgraph_id_for_node(nary_op->id()));
    for (int i = 0; i < ((int)operands.size() / 2); ++i)
    {
        graphlib::Edge operand_a = operands[i * 2];
        graphlib::Edge operand_b = operands[i * 2 + 1];
        auto attrs_a = graph->get_edge_attributes(operand_a);
        auto attrs_b = graph->get_edge_attributes(operand_b);
        graphlib::Node *add = graph->add_node(
            nary_op->clone(nary_op->name() + "_cascade_" + std::to_string(i)),
            graph->get_subgraph_id_for_node(nary_op->id()));
        operand_a.consumer_input_port_id = 0;
        operand_a.consumer_node_id = add->id();
        operand_b.consumer_input_port_id = 1;
        operand_b.consumer_node_id = add->id();
        graph->add_edge(operand_a, attrs_a);
        graph->add_edge(operand_b, attrs_b);

        graphlib::Edge sink_edge(add->id(), 0, sink->id(), i, graphlib::EdgeType::kData);
        graph->add_edge(sink_edge);
    }

    if ((operands.size() % 2) != 0)
    {
        graphlib::Edge back = operands.back();
        graphlib::Edge sink_edge(back.producer_node_id, 0, sink->id(), operands.size() - 1, graphlib::EdgeType::kData);
        graph->add_edge(sink_edge);
    }

    for (graphlib::Edge user : graph->user_data_edges(nary_op))
    {
        user.producer_node_id = sink->id();
        graph->add_edge(user);
    }

    graph->remove_node(nary_op);
    return cascade_nary_to_binary_op(graph, sink);
}

bool swap_broadcast_dims(graphlib::Graph *graph, graphlib::Edge edge, int old_dim, int new_dim)
{
    bool swapped = false;
    auto tms = graph->get_edge_attributes(edge)->get_tms();
    std::vector<graphlib::OpType> new_tms;
    for (graphlib::OpType &op_type : tms)
    {
        if (op_type.op == "broadcast")
        {
            int dim = std::get<int>(op_type.attr[0]);
            int size = std::get<int>(op_type.attr[1]);
            bool explicit_bcast = std::get<bool>(op_type.attr[2]);
            if (dim == old_dim)
            {
                graphlib::OpType updated_bcast("broadcast", {new_dim, size, explicit_bcast});
                new_tms.push_back(updated_bcast);
                swapped = true;
            }
            else
            {
                new_tms.push_back(op_type);
            }
        }
        else
        {
            new_tms.push_back(op_type);
        }
    }
    graph->get_edge_attributes(edge)->set_tms(new_tms);
    return swapped;
}

void handle_change_rank(graphlib::Graph *graph, graphlib::Edge edge)
{
    auto get_consumer_size = [](std::uint32_t producer_size, graphlib::Node *node)
    {
        std::uint32_t consumer_size = node->shape().size();
        graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(node);
        if (not op)
            return consumer_size;
        if (op->op_name() == "reshape")
            return producer_size;
        if (op->op_name() == "squeeze")
            return (consumer_size + 1);
        if (op->op_name() == "unsqueeze")
            return (consumer_size - 1);
        return consumer_size;
    };

    auto producer_size = graph->node_by_id(edge.producer_node_id)->shape().size();
    auto consumer_size = get_consumer_size(producer_size, graph->node_by_id(edge.consumer_node_id));

    if (producer_size == consumer_size)
        return;

    graphlib::OpNode *consumer = dynamic_cast<graphlib::OpNode *>(graph->node_by_id(edge.consumer_node_id));
    if (consumer and consumer->op_type() == "embedding")
        return;

    // This is one of the few cases where we actually want to move tms downstream
    auto tms = graph->get_edge_attributes(edge)->get_tms();
    graph->get_edge_attributes(edge)->set_tms({});

    auto insert = [graph](graphlib::Edge edge, std::string op, std::uint32_t rank) -> graphlib::Edge
    {
        graphlib::Node *producer = graph->node_by_id(edge.producer_node_id);
        graphlib::Node *consumer = graph->node_by_id(edge.consumer_node_id);
        graphlib::OpNode *inherit = dynamic_cast<graphlib::OpNode *>(consumer)
                                        ? dynamic_cast<graphlib::OpNode *>(consumer)
                                        : dynamic_cast<graphlib::OpNode *>(producer);
        TT_ASSERT(inherit);
        // If there are 2 edges from the same producer to the same consumer (eg. eltwise binary op),
        // need edge_creation_id to differentiate naming.
        std::string name = producer->name() + "_" + consumer->name() + "_" + op + std::to_string(rank) + "_" +
                           std::to_string(edge.edge_creation_id);
        graphlib::OpNode *change_rank = dynamic_cast<graphlib::OpNode *>(
            graph->add_node(inherit->clone(name), graph->get_subgraph_id_for_node(producer->id())));
        TT_ASSERT(change_rank);
        auto attr = (op == "squeeze") ? std::vector<graphlib::OpType::Attr>{0}
                                      : std::vector<graphlib::OpType::Attr>{0, ((int)rank - 1)};
        change_rank->change_op_type(graphlib::OpType(op, attr));
        change_rank->set_shape(producer->shape().as_rank(rank));
        change_rank->tag("dont_erase", true);
        auto [incoming_edge, outgoing_edge] = insert_node_on_edge(graph, edge, change_rank);
        change_rank->set_output_df_from_operands(graph);
        if (try_consteval_op(graph, change_rank))
            return graph->operand_data_edges(consumer)[0];

        // Set dataformat to match producer
        change_rank->set_output_df(producer->output_df());
        return outgoing_edge;
    };

    int orig_producer_size = (int)producer_size;
    while (producer_size < consumer_size)
    {
        producer_size++;
        edge = insert(edge, "unsqueeze", producer_size);
    }

    while (producer_size > consumer_size)
    {
        producer_size--;
        TT_ASSERT(producer_size > 0);
        edge = insert(edge, "squeeze", producer_size);
    }

    int diff = (int)producer_size - orig_producer_size;
    for (OpType &op_type : tms)
    {
        if (op_type.op == "broadcast")
        {
            if (std::get<int>(op_type.attr[0]) >= 0)
                std::get<int>(op_type.attr[0]) += diff;
        }
    }
    graph->get_edge_attributes(edge)->set_tms(tms);
}

void handle_change_rank(graphlib::Graph *graph, graphlib::Node *node)
{
    for (graphlib::Edge e : graph->operand_data_edges(node)) handle_change_rank(graph, e);
    for (graphlib::Edge e : graph->user_data_edges(node)) handle_change_rank(graph, e);
}

graphlib::Edge clone_input_forking_edge(graphlib::Graph *graph, graphlib::Edge user_edge, bool allow_single_user)
{
    Node *input = graph->node_by_id(user_edge.producer_node_id);
    TT_ASSERT(input->node_type() == NodeType::kInput);
    TT_ASSERT(graph->data_operands(input).empty(), "Cannot clone a loopback input");
    TT_ASSERT(graph->data_users(input).size() > 1 or allow_single_user, "Cannot clone input that doesn't fork");
    Node *clone = graph->add_node(
        input->clone(input->name() + "_fork_clone" + std::to_string(user_edge.consumer_node_id)),
        graph->get_subgraph_id_for_node(input->id()));

    auto edge_attr = graph->get_edge_attributes(user_edge);
    graph->remove_edge(user_edge);
    graphlib::Edge new_edge(
        clone->id(),
        user_edge.producer_output_port_id,
        user_edge.consumer_node_id,
        user_edge.consumer_input_port_id,
        user_edge.edge_type);
    graph->add_edge(new_edge, edge_attr);
    return new_edge;
}

graphlib::Shape default_tm_evaluator(graphlib::OpType const &tm, graphlib::Shape shape, graphlib::IRLevel ir_level)
{
    std::vector<Shape> shapes = {shape};
    std::tuple<Shape, std::vector<DimBroadcast>> shape_data =
        get_op_shape(tm, shapes, ir_level == IRLevel::IR_BUDA, shape.get_tile_dim());
    shape = std::get<0>(shape_data);
    TT_ASSERT(std::get<1>(shape_data).size() == 0, "TMs should not cause broadcasts");
    return shape;
}

graphlib::Shape ignore_broadcast_tm_evaluator(
    graphlib::OpType const &tm, graphlib::Shape shape, graphlib::IRLevel ir_level)
{
    // Ignore unit broadcasts
    // Since we ignore bcasts slices / stacks might have incorrect factors, ignore them
    if (tm.op == "broadcast" and shape.is_unit(std::get<int>(tm.attr[0])))
        return shape;
    else if (tm.op == "hslice" and shape.is_unit(-1))
        return shape;
    else if (tm.op == "vslice" and shape.is_unit(-2))
        return shape;
    else if (tm.op == "hstack" and shape.is_unit(-3))
        return shape;
    else if (tm.op == "vstack" and shape.is_unit(-3))
        return shape;
    return default_tm_evaluator(tm, shape, ir_level);
}

graphlib::Shape post_tms_shape(
    graphlib::Shape input_shape,
    std::vector<OpType> const &tms,
    std::function<graphlib::Shape(graphlib::OpType const &, graphlib::Shape, graphlib::IRLevel)> tm_evaluator,
    graphlib::IRLevel ir_level)
{
    for (OpType const &tm : tms)
    {
        input_shape = tm_evaluator(tm, input_shape, ir_level);
    }
    return input_shape;
}

graphlib::Shape post_tms_shape(
    Graph const *graph,
    graphlib::Edge edge,
    std::function<graphlib::Shape(graphlib::OpType const &, graphlib::Shape, graphlib::IRLevel)> tm_evaluator)
{
    graphlib::Shape producer_shape = graph->node_by_id(edge.producer_node_id)->shape();
    auto const &tms = graph->get_edge_attributes(edge)->get_tms();
    return post_tms_shape(producer_shape, tms, tm_evaluator, graph->get_ir_level());
}

std::pair<int, int> get_padding(graphlib::Graph const *graph, graphlib::Node const *node)
{
    graphlib::BudaOpNode const *op = dynamic_cast<graphlib::BudaOpNode const *>(node);
    TT_ASSERT(op);
    if (not op)
        return std::make_pair(0, 0);
    for (auto user : graph->user_data_edges(op))
    {
        auto attrs = graph->get_edge_attributes(user);
        for (auto const &tm : attrs->get_tms())
        {
            if (tm.op == "buda_unpad")
            {
                int rt = std::get<int>(tm.attr[0]);
                int ct = std::get<int>(tm.attr[1]);
                return std::make_pair(rt, ct);
            }
        }
    }
    return std::make_pair(0, 0);
}

bool tms_support_kernel_broadcast(
    Shape producer_shape, std::vector<OpType> const &tms, UBlockOrder ublock_order, int ublock_ct, bool is_buda)
{
    if (not std::any_of(tms.begin(), tms.end(), [](auto const &op_type) { return op_type.op == "broadcast"; }))
        return false;

    // Kernel broadcast producer must start as 2 dimensional
    if (producer_shape.z() > 1)
        return false;

    // Anything goes when it's just a single tile
    if (producer_shape.is_single_tile())
        return true;

    // The following code asserts if this set of TMs interleaves the
    // broadcast tile order which is illegal for kernel broadcast.
    Shape shape = producer_shape;
    for (auto const &tm : tms)
    {
        if (tm.op == "broadcast")
        {
            // If we sliced and now have a zdim, a broadcast will create repeat tiles which is illegal
            if (shape.z() > 1)
                return false;

            int dim = shape.negative_index(std::get<int>(tm.attr[0]));

            // If the broadcast dim is in the same direction as the ublock order,
            // then we are repeating tiles which is illagal
            if ((not shape.is_unit(-2) and dim == -1 and (ublock_order == UBlockOrder::R or ublock_ct > 1)) or
                (not shape.is_unit(-1) and dim == -2 and ublock_order == UBlockOrder::C))
                return false;
        }
        else if (tm.op == "hslice" and not shape.is_unit(-2) and (ublock_order == UBlockOrder::R or ublock_ct > 1))
        {
            return false;
        }
        else if (tm.op == "hstack" and not shape.is_unit(-2) and (ublock_order == UBlockOrder::R or ublock_ct > 1))
        {
            return false;
        }
        else if (tm.op == "vslice" and not shape.is_unit(-1) and ublock_order == UBlockOrder::C)
        {
            return false;
        }
        else if (tm.op == "vstack" and not shape.is_unit(-1) and ublock_order == UBlockOrder::C)
        {
            return false;
        }
        else if (tm.op == "transpose")
        {
            ublock_order = flip_ublock_order(ublock_order);
        }

        shape = ::get_tm_shape(tm, shape, is_buda);
    }

    return true;
}

// Calculate node shape from operand shapes, using python callback
void calculate_and_set_node_shape(Graph *graph, Node *node)
{
    log_trace(LogGraphCompiler, "Calculate and set node shape for: {} {}", node->name(), node->get_type());
    // Apply TMs and get post-TM operand shapes
    std::vector<Shape> operand_shapes;

    // Validate / Canonicalize TileDim
    auto op_node = dynamic_cast<graphlib::OpNode *>(node);
    if (op_node)
    {
        validate_tile_dims(graph, op_node);
    }

    for (graphlib::Edge &e : graph->operand_data_edges(node))
    {
        auto operand_shape = graph->node_by_id(e.producer_node_id)->shape();
        std::vector<OpType> tms = graph->get_edge_attributes(e)->get_tms();
        for (OpType tm : tms)
        {
            std::vector<Shape> shapes = {operand_shape};
            std::tuple<Shape, std::vector<DimBroadcast>> shape_data =
                get_op_shape(tm, shapes, graph->get_ir_level() == IRLevel::IR_BUDA, operand_shape.get_tile_dim());
            operand_shape = std::get<0>(shape_data);
            TT_ASSERT(std::get<1>(shape_data).size() == 0, "TMs should not cause broadcasts");
            log_trace(LogGraphCompiler, "    TM {} {}", tm.as_string(), operand_shape);
        }
        log_trace(
            LogGraphCompiler,
            "  Operand[{}] {} {}",
            e.consumer_input_port_id,
            operand_shape,
            graph->node_by_id(e.producer_node_id)->name());
        operand_shapes.push_back(operand_shape);
    }

    if ((node->node_type() == graphlib::NodeType::kOutput) || (node->node_type() == graphlib::NodeType::kQueue))
    {
        // Graph shape from first, and only, operand
        TT_ASSERT(operand_shapes.size() == 1, "Node should have exactly one operand");
        node->set_shape(operand_shapes[0]);
        return;
    }

    if ((node->node_type() != NodeType::kPyOp) && (node->node_type() != NodeType::kBudaOp) &&
        (node->node_type() != NodeType::kBudaNaryTM))
        return;

    graphlib::OpType op_type = node->node_type() == NodeType::kBudaNaryTM
                                   ? dynamic_cast<graphlib::BudaNaryTMNode *>(node)->op_type()
                                   : dynamic_cast<graphlib::OpNode *>(node)->op_type();

    bool is_fused_op = (node->node_type() == graphlib::kBudaOp) && node->as<graphlib::BudaOpNode>()->is_fused_op();

    std::tuple<Shape, std::vector<DimBroadcast>> shape_data =
        is_fused_op
            ? get_fused_op_shape(node->as<graphlib::BudaOpNode>(), operand_shapes)
            : get_op_shape(
                  op_type, operand_shapes, graph->get_ir_level() == IRLevel::IR_BUDA, node->shape().get_tile_dim());

    log_trace(LogGraphCompiler, "  {}", std::get<0>(shape_data));
    node->set_shape(std::get<0>(shape_data));

    // Set broadcast attributes on edges
    for (graphlib::Edge &e : graph->operand_data_edges(node))
    {
        for (DimBroadcast &b : std::get<1>(shape_data))
        {
            log_trace(LogGraphCompiler, "  brcst {} {} {}", std::get<0>(b), std::get<1>(b), std::get<2>(b));

            int operand = std::get<0>(b);
            if (operand == (int)e.consumer_input_port_id)
            {
                int dim = std::get<1>(b);
                int size = std::get<2>(b);
                bool const is_buda = graph->get_ir_level() == IRLevel::IR_BUDA;
                if (is_buda and dim >= 2)
                {
                    size /= graphlib::Shape::BUDA_TILE_DIM;
                }
                graph->get_edge_attributes(e)->set_broadcast_dim(dim, size);
            }
        }
    }
}

std::vector<UBlockOrder> get_input_ublock_order(Graph const *graph, Node const *node)
{
    std::vector<UBlockOrder> ublock_order;

    std::vector<Edge> operands = graph->operand_data_edges(node);
    if (graphlib::OpNode const *op_node = dynamic_cast<graphlib::OpNode const *>(node))
    {
        if (op_node->is_matmul())
        {
            auto edge_attrs0 = graph->get_edge_attributes(operands[0]);
            auto edge_attrs1 = graph->get_edge_attributes(operands[1]);
            ublock_order.push_back(edge_attrs0->get_ublock_order());
            ublock_order.push_back(edge_attrs1->get_ublock_order());
            if (op_node->is_sparse_matmul())
            {
                ublock_order.push_back(UBlockOrder::R);
            }
        }
        else
        {
            auto edge_attrs0 = graph->get_edge_attributes(operands[0]);
            for (Edge edge : operands)
            {
                auto edge_attrs = graph->get_edge_attributes(edge);
                TT_ASSERT(edge_attrs->get_ublock_order() == edge_attrs0->get_ublock_order());
                ublock_order.push_back(edge_attrs->get_ublock_order());
            }
        }
    }
    else
    {
        // Is output or queue node
        TT_ASSERT(operands.size() == 1);
        ublock_order = {graph->get_edge_attributes(operands[0])->get_ublock_order()};
    }

    return ublock_order;
}

tt::graphlib::Node *get_input_queue_producer(Graph const *graph, tt::graphlib::InputNode const *node)
{
    std::vector<graphlib::Edge> partial_datacopy_edges = graph->operand_partial_datacopy_edges(node);
    auto producers = graph->data_operands(node);

    if (not producers.empty() and not partial_datacopy_edges.empty())
    {
        throw std::runtime_error("Input queue " + node->name() + " has both producer and partial datacopy edge!");
    }
    else if (not producers.empty())
    {
        TT_ASSERT(producers.size() == 1);
        return producers[0];
    }
    else if (not partial_datacopy_edges.empty())
    {
        std::vector<graphlib::Edge> producer_edges;
        for (auto edge : partial_datacopy_edges)
        {
            auto output_node = graph->node_by_id(edge.producer_node_id);
            TT_ASSERT(graph->operand_edges(output_node).size() == 1, "Output node should only have 1 producer");
            producer_edges.push_back(graph->operand_edges(output_node).front());
        }

        // Assert all partial datacopy producer edges have the same ublock order
        TT_ASSERT(std::all_of(
            producer_edges.begin(),
            producer_edges.end(),
            [graph, producer_edges](Edge e)
            {
                return graph->get_edge_attributes(e)->get_ublock_order() ==
                       graph->get_edge_attributes(producer_edges[0])->get_ublock_order();
            }));

        graphlib::OutputNode *output =
            graph->node_by_id(partial_datacopy_edges[0].producer_node_id)->as<graphlib::OutputNode>();
        auto output_producer = graph->data_operands(output);
        TT_ASSERT(output_producer.size() == 1);
        TT_ASSERT(output_producer[0]->node_type() == graphlib::NodeType::kBudaOp);
        return output_producer[0];
    }

    return nullptr;
}

tt::graphlib::UBlockOrder get_input_queue_ublock_order(Graph const *graph, Node const *node)
{
    UBlockOrder ublock_order = UBlockOrder::R;
    if (tt::graphlib::Node *producer = get_input_queue_producer(graph, node->as<graphlib::InputNode>()); producer)
    {
        ublock_order = get_output_ublock_order(graph, producer);
    }
    else
    {
        std::vector<tt::graphlib::Edge> consumers = graph->user_data_edges(node);
        bool all_users_transpose = std::all_of(
            consumers.begin(),
            consumers.end(),
            [graph](graphlib::Edge e) { return graph->get_edge_attributes(e)->has_tm("transpose"); });
        tt::graphlib::UBlockOrder user_ublock_order = graph->get_edge_attributes(consumers.front())->get_ublock_order();
        bool all_users_same_order = std::all_of(
            consumers.begin(),
            consumers.end(),
            [graph, user_ublock_order](graphlib::Edge e)
            { return user_ublock_order == graph->get_edge_attributes(e)->get_ublock_order(); });

        tt::graphlib::UBlockOrder q_ublock_order = all_users_same_order ? user_ublock_order : graphlib::UBlockOrder::R;
        ublock_order = all_users_transpose ? flip_ublock_order(q_ublock_order) : q_ublock_order;
    }

    return ublock_order;
}

UBlockOrder get_output_ublock_order(Graph const *graph, Node const *node)
{
    if (node->node_type() == graphlib::NodeType::kInput)
    {
        return get_input_queue_ublock_order(graph, node);
    }

    graphlib::BudaOpNode const *op_node = dynamic_cast<graphlib::BudaOpNode const *>(node);
    if (op_node and op_node->op_name() == "reduce")
    {
        return UBlockOrder::R;
    }

    return get_input_ublock_order(graph, node).back();
}

// Insert NOP on an edge with transpose TM, then flip ublock order for better streaming
// returns true if nop inserted
bool try_insert_nop_on_transpose_edge(Graph *graph, Edge &edge)
{
    auto node = graph->node_by_id(edge.consumer_node_id);
    std::vector<graphlib::OpType> tms = graph->get_edge_attributes(edge)->get_tms();
    if (tms.size() > 0 && tms[tms.size() - 1].op == "nop")
        return false;

    // even number of transposes are ok, tiles are not transposed in the end
    int transposes = 0;
    int last_transpose = 0;
    for (std::size_t i = 0; i < tms.size(); i++)
    {
        if (tms[i].op == "transpose")
        {
            transposes++;
            last_transpose = i;
        }
    }
    if (transposes % 2 == 0)
        return false;  // Even number of transposes cancel out

    // Add a NOP on the edge, and move TMs after last transpose to it
    graphlib::BudaOpNode *nop = graph->add_node(
        graphlib::create_node<graphlib::BudaOpNode>(
            node->name() + "_transpose_nop_" + std::to_string(edge.edge_creation_id), "nop"),
        graph->get_subgraph_id_for_node(node->id()));
    nop->copy_parent_op_attributes(node->as<graphlib::BudaOpNode>());

    auto [new_edge0, new_edge1] = graphlib::insert_node_on_edge(graph, edge, nop);

    int num_first_group = (last_transpose == 0) ? 1 : last_transpose;
    int num_second_group = (last_transpose == 0) ? tms.size() - 1 : 1;
    int num_third_group = tms.size() - num_first_group - 1;
    graph->get_edge_attributes(new_edge0)->set_tms(
        std::vector<graphlib::OpType>(tms.begin(), tms.begin() + num_first_group));

    // Flip the ublock order wrt the producer for more likely streaming
    graphlib::UBlockOrder producer_ublock_order =
        graphlib::get_output_ublock_order(graph, graph->node_by_id(new_edge0.producer_node_id));
    graph->get_edge_attributes(new_edge0)->set_ublock_order(graphlib::flip_ublock_order(producer_ublock_order));

    if (num_second_group > 0)
    {
        // No need to add second nop if we have only 1 transpose on position 0.
        if (last_transpose != 0)
        {
            // Assign last transpose to its own edge so it could be streamed. We might need an extra Nop for this
            // purpose.
            graphlib::BudaOpNode *nop2 = graph->add_node(
                graphlib::create_node<graphlib::BudaOpNode>(
                    node->name() + "_transpose_nop_2_" + std::to_string(edge.edge_creation_id), "nop"),
                graph->get_subgraph_id_for_node(node->id()));
            nop2->copy_parent_op_attributes(node->as<graphlib::BudaOpNode>());

            auto [mid_edge, last_edge] = graphlib::insert_node_on_edge(graph, new_edge1, nop2);
            graph->get_edge_attributes(mid_edge)->set_tms(
                std::vector<graphlib::OpType>(tms.begin() + num_first_group, tms.begin() + num_first_group + 1));

            if (num_third_group > 0)
            {
                graph->get_edge_attributes(last_edge)->set_tms(
                    std::vector<graphlib::OpType>(tms.begin() + num_first_group + 1, tms.end()));
            }
        }
        else
        {
            // Keep rest of TMs on new_edge1 if nop2 is not added.
            graph->get_edge_attributes(new_edge1)->set_tms(
                std::vector<graphlib::OpType>(tms.begin() + num_first_group, tms.end()));
        }
    }

    return true;
}

// Return a vector of pairs of optimizer parameter input nodes and optimizer key names for a given model parameter node
std::vector<std::pair<InputNode *, std::string>> get_optimizer_param_info(
    const Graph *graph, const Node *model_parameter)
{
    // If autograd has run, there will be EdgeType::kAutogradFwdToOptimizer edges. We parse through this
    // list looking for inputs that require its tensors to be populated by the python-side optimizer obj
    std::vector<std::pair<InputNode *, std::string>> ret;
    for (graphlib::Edge edge : graph->user_edges(model_parameter))
    {
        if (edge.edge_type != graphlib::EdgeType::kAutogradFwdToOptimizer)
            continue;
        if (graph->node_by_id(edge.consumer_node_id)->node_type() != NodeType::kInput)
            continue;

        graphlib::InputNode *input = graph->node_by_id(edge.consumer_node_id)->as<graphlib::InputNode>();
        if (not input->is_optimizer_parameter())
        {
            continue;
        }

        // Parse out the optimizer-param suffix string and do a lookup to get the tensor
        std::string optimizer_input_name = input->name();
        std::string::size_type optimizer_param_idx = optimizer_input_name.rfind('.');
        TT_ASSERT(
            optimizer_param_idx != std::string::npos,
            "Expecting optimizer node to have a '.<optimizer-param>' suffix identifier");

        std::string optimizer_param_key = optimizer_input_name.substr(optimizer_param_idx + 1);
        ret.push_back(std::make_pair(input, optimizer_param_key));
    }
    return ret;
}

bool is_constant_input(const Node *node)
{
    graphlib::InputNode const *input = dynamic_cast<graphlib::InputNode const *>(node);
    return input and input->is_constant();
}

bool is_recompute(const Graph *graph, const Node *node)
{
    for (const Edge &edge : graph->operand_edges(node))
    {
        if (edge.edge_type == graphlib::EdgeType::kAutogradFwdToRecompute)
        {
            return true;
        }
    }
    return false;
}

Node *get_fwd_from_recompute(const Graph *graph, const Node *node)
{
    for (const Edge &edge : graph->operand_edges(node))
    {
        if (edge.edge_type == graphlib::EdgeType::kAutogradFwdToRecompute)
        {
            return graph->node_by_id(edge.producer_node_id);
        }
    }
    return nullptr;
}

ConstEvalGraph::ConstEvalGraph(
    std::string const &name, Node *runtime_input, bool promote_input, unsigned int subgraph_id, int unique_id) :
    consteval_graph(IRLevel::IR_CONSTEVAL, name, unique_id == -1 ? Graph::generate_unique_graph_id() : unique_id),
    runtime_input(runtime_input),
    subgraph_id_(subgraph_id)
{
    TT_ASSERT(runtime_input->node_type() == NodeType::kInput);
    if (promote_input)
        promote_node(nullptr, runtime_input, runtime_input->clone());
}

std::unique_ptr<Node> ConstEvalGraph::promote_node(std::unique_ptr<Node> &&consteval_node)
{
    return promote_node(nullptr, nullptr, std::forward<std::unique_ptr<Node>>(consteval_node));
}

std::unique_ptr<Node> ConstEvalGraph::promote_node(Graph *runtime_graph, Node *runtime_node)
{
    return promote_node(runtime_graph, runtime_node, runtime_node->clone());
}

std::unique_ptr<Node> ConstEvalGraph::promote_node(
    Graph *runtime_graph, Node *runtime_node, std::unique_ptr<Node> &&consteval_node_free)
{
    TT_ASSERT(not runtime_graph or runtime_node);
    TT_ASSERT(not runtime_graph or runtime_graph->get_ir_level() == IRLevel::IR_PYBUDA);

    graph_updated_since_autograd = true;

    Node *consteval_node = consteval_graph.add_node<Node>(std::move(consteval_node_free), subgraph_id_);

    // Promoted consteval nodes are always in the forward epoch for their respective consteval graph
    // ConstEvalGraph will automatically run its own autograd and insert its own, respective BW ops
    consteval_node->set_epoch_type(NodeEpochType::Forward);

    if (consteval_output)
    {
        // Runtime input node needs to always map to the consteval graph output
        auto output_operands = consteval_graph.data_operands(consteval_output);
        TT_ASSERT(output_operands.size() == 1);
        runtime_to_consteval_map[runtime_input->id()] = output_operands[0]->id();
    }

    // Create mapping from runtime node id to consteval
    if (runtime_node)
    {
        runtime_to_consteval_map.insert({runtime_node->id(), consteval_node->id()});
    }

    // Create edges inherited from the runtime_graph
    if (runtime_graph)
    {
        for (Edge const &runtime_edge : runtime_graph->operand_data_edges(runtime_node))
        {
            auto runtime_attr = runtime_graph->get_edge_attributes(runtime_edge);
            int const_producer_id = 0;

            if (runtime_to_consteval_map.find(runtime_edge.producer_node_id) == runtime_to_consteval_map.end())
            {
                InputNode *runtime_operand =
                    dynamic_cast<InputNode *>(runtime_graph->node_by_id(runtime_edge.producer_node_id));
                TT_ASSERT(runtime_operand, "All operands of promoted nodes must be graph inputs");
                Node *consteval_operand = nullptr;
                
                // Only add the node if it doesn't already exist in the consteval graph
                if (ConstEvalGraph *nested_consteval_graph = runtime_operand->get_consteval_graph())
                    consteval_operand = graft(nested_consteval_graph->get_graph());
                else if (!consteval_graph.has_node_with_name(runtime_operand->name()))
                    consteval_operand = consteval_graph.add_node<Node>(runtime_operand->clone(), subgraph_id_);
                else
                    consteval_operand = consteval_graph.get_node_by_name(runtime_operand->name());

                // Only map the operand if it has 1 user
                if (runtime_graph->user_data_edges(runtime_operand).size() > 1)
                    const_producer_id = consteval_operand->id();
                else if (runtime_graph->user_data_edges(runtime_operand).size() == 1)
                    runtime_to_consteval_map.insert({runtime_operand->id(), consteval_operand->id()});

                runtime_graph->remove_edge(runtime_edge);
                auto users = runtime_graph->user_edges(runtime_operand);
                if (users.empty())
                    runtime_graph->remove_node(runtime_operand);
            }

            Edge consteval_edge = Edge(
                const_producer_id ? const_producer_id : runtime_to_consteval_map.at(runtime_edge.producer_node_id),
                runtime_edge.producer_output_port_id,
                runtime_to_consteval_map.at(runtime_edge.consumer_node_id),
                runtime_edge.consumer_input_port_id,
                runtime_edge.edge_type);

            consteval_graph.add_edge(consteval_edge);
            consteval_graph.get_edge_attributes(consteval_edge)->copy_from(*runtime_attr);
            runtime_attr->get_tms().clear();  // remove all operand runtime tms, they are consumed by consteval
        }
    }
    else if (dynamic_cast<graphlib::OpNode *>(consteval_node))
    {
        TT_ASSERT(consteval_output);
        // If there is no runtime graph then new consteval nodes are simply appended as the new output node
        Edge output_edge = consteval_graph.operand_data_edges(consteval_output).at(0);
        Edge new_edge(
            output_edge.producer_node_id,
            output_edge.producer_output_port_id,
            consteval_node->id(),
            0,
            EdgeType::kData);
        consteval_graph.add_edge(new_edge);
    }

    // Connect to the graph output
    if (consteval_output)
    {
        consteval_graph.remove_edge(consteval_graph.operand_data_edges(consteval_output).at(0));
    }
    else
    {
        consteval_output = consteval_graph.add_node<Node>(
            std::make_unique<OutputNode>(consteval_graph.name() + ".output"), subgraph_id_);
    }

    Edge consteval_edge(consteval_node->id(), 0, consteval_output->id(), 0, EdgeType::kData);
    consteval_graph.add_edge(consteval_edge);

    runtime_input->set_shape(consteval_node->shape());
    runtime_input->set_output_df(consteval_node->output_df());
    consteval_output->set_shape(consteval_node->shape());
    consteval_output->set_output_df(consteval_node->output_df());

    if (runtime_graph)
    {
        if (runtime_graph->operand_data_edges(runtime_node).size() == 1)
        {
            return graphlib::bypass_node(runtime_graph, runtime_node, true /*remove_node*/);
        }
    }
    return nullptr;
}

Node *ConstEvalGraph::graft(Graph *other)
{
    NodeId other_output_op_id = -1;
    std::unordered_map<NodeId, NodeId> node_id_map;
    std::vector<Node *> nodes = other->nodes();
    std::vector<Edge> edges = other->edges(EdgeType::kData);

    // Copy all nodes except for the output node
    for (Node *node : nodes)
    {
        if (node->node_type() == NodeType::kOutput)
        {
            TT_ASSERT(other_output_op_id == -1, "Only one output is supported for consteval graphs");
            other_output_op_id = other->data_operands(node)[0]->id();
            continue;
        }

        // If the graph being graft is from a common ancenstor nodes can overlap
        if (consteval_graph.has_node_with_name(node->name()))
        {
            node_id_map.insert({node->id(), consteval_graph.get_node_by_name(node->name())->id()});
            continue;
        }

        Node *new_node = consteval_graph.add_node<Node>(node->clone(), subgraph_id_);
        node_id_map.insert({node->id(), new_node->id()});
    }

    // Copy all edges except for the output edge
    for (Edge const &edge : edges)
    {
        if (edge.producer_node_id == other_output_op_id)
            continue;

        Edge new_edge(
            node_id_map.at(edge.producer_node_id),
            edge.producer_output_port_id,
            node_id_map.at(edge.consumer_node_id),
            edge.consumer_input_port_id,
            edge.edge_type);
        consteval_graph.add_edge(new_edge);
        consteval_graph.copy_edge_attributes(edge, new_edge, other);
    }

    TT_ASSERT(other_output_op_id != -1);
    TT_ASSERT(node_id_map.find(other_output_op_id) != node_id_map.end());
    Node *output = consteval_graph.node_by_id(node_id_map.at(other_output_op_id));
    return output;
}

std::unique_ptr<ConstEvalGraph> ConstEvalGraph::clone(Node *new_runtime_input, const std::string &new_input_node_name)
{
    TT_ASSERT(new_runtime_input);
    int unique_id = Graph::generate_unique_graph_id();
    std::unique_ptr<ConstEvalGraph> cloned = std::make_unique<ConstEvalGraph>(
        consteval_graph.name() + "." + std::to_string(unique_id), new_runtime_input, false, subgraph_id_, unique_id);

    consteval_graph.clone(&cloned->consteval_graph);
    cloned->needs_autograd = needs_autograd;
    cloned->ran_autograd = ran_autograd;
    cloned->graph_updated_since_autograd = graph_updated_since_autograd;

    if (consteval_output)
        cloned->consteval_output = cloned->consteval_graph.get_node_by_name(consteval_output->name());
    // Map the old ids to cloned ones
    for (auto [runtime_node_id, consteval_node_id] : runtime_to_consteval_map)
    {
        Node *consteval_node = consteval_graph.node_by_id(consteval_node_id);
        std::string node_name = consteval_node->name();

        if (consteval_node->node_type() == NodeType::kInput and new_input_node_name != "")
        {
            std::string const &old_node_name = consteval_node->name();
            cloned->consteval_graph.update_node_name(
                cloned->consteval_graph.get_node_by_name(old_node_name), new_input_node_name);
            node_name = new_input_node_name;
        }
        cloned->runtime_to_consteval_map[runtime_node_id] = cloned->consteval_graph.get_node_by_name(node_name)->id();
    }
    return cloned;
}

void ConstEvalGraph::pad_output_to_buda_dims(std::string const &name_prefix)
{
    graphlib::Node *output = get_output();
    graphlib::Shape shape = output->shape();

    for (int dim : {-1, -2})
    {
        if (shape[dim] % graphlib::Shape::BUDA_TILE_DIM != 0)
        {
            graphlib::OpType pad_tile("pad_tile", {dim, (int)shape[dim]});
            auto consteval_pad_tile = graphlib::create_node<graphlib::PyOpNode>(
                name_prefix + "_pad_tile_" + ((dim == -1) ? "c_" : "r_") + output->name(), pad_tile);
            shape[dim] = align_up_tile(shape[dim]);
            consteval_pad_tile->set_output_df(output->output_df());
            consteval_pad_tile->set_epoch_type(output->get_epoch_type());
            consteval_pad_tile->set_shape(shape);
            promote_node(std::move(consteval_pad_tile));
        }
    }
}

void ConstEvalGraph::autograd()
{
    if (not needs_autograd)
        return;

    if (ran_autograd)
    {
        // Remove BW graph and build it again from scratch
        auto bw_nodes = consteval_graph.nodes([](Node *n) { return n->get_epoch_type() == NodeEpochType::Backward; });
        for (Node *bw_node : bw_nodes)
        {
            consteval_graph.remove_node(bw_node);
        }
    }

    autograd2::autograd2_engine consteval_autograd_engine(&consteval_graph, autograd2::autograd_config{});
    consteval_autograd_engine.run();

    ran_autograd = true;
    graph_updated_since_autograd = false;
}

bool is_consteval_capable_input_type(Node *node)
{
    graphlib::InputNode *input = dynamic_cast<graphlib::InputNode *>(node);
    return input and (input->is_parameter() or input->is_constant()) and
           not node->as<graphlib::TaggedNode>()->has_tag("dont_consteval");
}

bool is_consteval_capable_op(Graph *graph, Node *node, bool allow_forks)
{
    graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(node);
    if (not op)
        return false;

    std::vector<graphlib::Node *> operands = graph->data_operands(op);

    if (not std::all_of(operands.begin(), operands.end(), is_consteval_capable_input_type))
        return false;

    bool disable_forks = not allow_forks;

    auto requires_grad = [graph](graphlib::Node *n)
    { return graph->enable_training() and n->as<graphlib::InputNode>()->requires_grad(); };

    auto fork = [graph, disable_forks, requires_grad](graphlib::Node *n)
    { return (requires_grad(n) or disable_forks) and (graph->data_users(n).size() > 1); };

    auto bcast = [graph, requires_grad](graphlib::Node *n)
    {
        bool any_bcast = false;
        for (auto e : graph->user_data_edges(n))
        {
            auto edge_attr = graph->get_edge_attributes(e);
            any_bcast |= edge_attr->has_broadcast_dims();
        }
        return requires_grad(n) and any_bcast;
    };

    if (std::any_of(operands.begin(), operands.end(), fork))
        return false;

    if (std::any_of(operands.begin(), operands.end(), bcast))
        return false;

    if (std::none_of(operands.begin(), operands.end(), requires_grad))
        return true;

    // requires_grad = true
    //   - if grad is required then we limit consteval to tm ops only
    return op->is_tm();
}

bool is_consteval_capable_input_no_operand_forks(Graph *graph, InputNode *input)
{
    if (not is_consteval_capable_input_type(input))
        return false;

    std::vector<Node *> users = graph->data_users(input);
    std::vector<Edge> user_edges = graph->user_data_edges(input);

    // If there is only one user then check if that op is consteval capable
    if (users.size() == 1)
        return is_consteval_capable_op(graph, users[0]) and graph->data_operands(users[0]).size() == 1;

    // If there are multiple users....
    // 1. All of the users must have one operand (unary ops)
    // 2. No user edge can have any tms
    // 3. All of the users must have the same op type
    // 4. All of the users must have the exact same op attrs

    if (not std::all_of(users.begin(), users.end(), [graph](Node *n) { return graph->data_operands(n).size() == 1; }))
        return false;

    if (not std::all_of(
            user_edges.begin(),
            user_edges.end(),
            [graph](Edge e) { return graph->get_edge_attributes(e)->get_tms().size() == 0; }))
        return false;

    std::vector<OpNode *> user_ops;
    for (Node *user : users)
        if (auto *op = dynamic_cast<OpNode *>(user))
            user_ops.push_back(op);
        else
            return false;

    std::string op_name = user_ops[0]->op_name();
    if (not std::all_of(user_ops.begin(), user_ops.end(), [op_name](OpNode *n) { return n->op_name() == op_name; }))
        return false;

    auto attrs = user_ops[0]->op_attrs();
    for (OpNode *op : user_ops)
        if (attrs != op->op_attrs())
            return false;

    return true;
}

std::unique_ptr<Node> try_consteval_op(Graph *graph, Node *node, bool dump_graph)
{
    if (not is_consteval_capable_op(graph, node))
        return nullptr;

    std::vector<graphlib::Node *> operands = graph->data_operands(node);
    graphlib::InputNode *input = operands[0]->as<graphlib::InputNode>();
    auto consteval_graph = input->get_consteval_graph(graph, true, true);
    auto ret_node = consteval_graph->promote_node(graph, node);

    if (dump_graph)
        reportify::dump_consteval_graph(graph->name(), input->name(), consteval_graph->get_graph());

    return ret_node;
}

bool try_consteval_input_no_operand_forks(Graph *graph, InputNode *input, bool dump_graph)
{
    if (not is_consteval_capable_input_no_operand_forks(graph, input))
        return false;

    auto consteval_graph = input->get_consteval_graph(graph, true, true);

    auto users = graph->data_users(input);

    // Thanks to is_consteval_capable_input(), we know that each user is identical (same op, same attrs, no edge tms)
    consteval_graph->promote_node(graph, users[0]);

    for (uint32_t i = 1; i < users.size(); i++) bypass_node(graph, users[i], true);

    if (dump_graph)
        reportify::dump_consteval_graph(graph->name(), input->name(), consteval_graph->get_graph());

    return true;
}

bool can_swap_operands(Graph *graph, Node *node)
{
    if (graph->data_operands(node).size() != 2)
        return false;
    if (node->node_type() == kBudaOp)
    {
        auto op = node->as<BudaOpNode>()->op_type().op;
        return ((op != "sub") && (op != "matmul"));
    }

    if (node->node_type() == kPyOp)
    {
        auto op = node->as<PyOpNode>()->op_type().op;
        return ((op != "sub") && (op != "matmul"));
    }
    return false;
}

void swap_operands(Graph *graph, Node *node)
{
    TT_ASSERT(can_swap_operands(graph, node));

    auto operand_edges = graph->operand_edges(node);

    for (Edge operand_edge : operand_edges)
    {
        Edge new_edge(operand_edge);
        new_edge.consumer_input_port_id = 1 - new_edge.consumer_input_port_id;
        graph->add_edge(new_edge);
        graph->copy_edge_attributes(operand_edge, new_edge);
        graph->remove_edge(operand_edge);
    }
}

Edge retrieve_between_edge(Graph *graph, Node *producer, Node *consumer)
{
    auto producer_user_edges = graph->user_data_edges(producer);
    Edge *edge = nullptr;
    for (auto &e : producer_user_edges)
    {
        if (e.consumer_node_id == consumer->id())
        {
            edge = &e;
            break;
        }
    }
    TT_ASSERT(edge);
    return *edge;
}

bool are_bcasts_between_ops(Graph *graph, Node *producer, Node *consumer)
{
    auto edge = retrieve_between_edge(graph, producer, consumer);
    auto edge_attr = graph->get_edge_attributes(edge);
    return edge_attr->has_broadcast_dims();
}

bool are_different_ranked_shapes_equivalent(Shape a, Shape b)
{
    auto a_vec = a.as_vector();
    auto b_vec = b.as_vector();

    // Remove all pre 1s
    std::vector<int> new_a;
    for (int i = 0; i < (int)a_vec.size(); i++)
    {
        if (a_vec[i] == 1)
        {
            a_vec.erase(a_vec.begin() + i);
            i--;
        }
        else if (a_vec[i] > 1)
            break;
    }
    for (int i = 0; i < (int)b_vec.size(); i++)
    {
        if (b_vec[i] == 1)
        {
            b_vec.erase(b_vec.begin() + i);
            i--;
        }
        else if (b_vec[i] > 1)
            break;
    }

    // Remove all post 1s
    for (int i = (int)a_vec.size() - 1; i >= 0; i--)
    {
        if (a_vec[i] == 1)
        {
            a_vec.erase(a_vec.begin() + i);
        }
        else if (a_vec[i] > 1)
            break;
    }
    for (int i = (int)b_vec.size() - 1; i >= 0; i--)
    {
        if (b_vec[i] == 1)
        {
            b_vec.erase(b_vec.begin() + i);
        }
        else if (b_vec[i] > 1)
            break;
    }

    if (a_vec.size() != b_vec.size())
        return false;

    for (int i = 0; i < (int)a_vec.size(); i++)
    {
        if (a_vec[i] != b_vec[i])
            return false;
    }
    return true;
}

// Check if this is a linked queue.
// Linked queues are output queues which have users nodes connected via partial data copy edges.
//
bool is_linked_queue(const graphlib::Graph *graph, const graphlib::Node *node)
{
    bool output_link_queue = node->node_type() == graphlib::NodeType::kOutput and
                             not graph
                                     ->user_edges(
                                         node,
                                         [](graphlib::Edge e) {
                                             return e.edge_type == graphlib::EdgeType::kPartialDataCopy or
                                                    e.edge_type == graphlib::EdgeType::kSubgraphLink;
                                         })
                                     .empty();
    bool input_link_queue = node->node_type() == graphlib::NodeType::kInput and
                            not graph
                                    ->operand_edges(
                                        node,
                                        [](graphlib::Edge e) {
                                            return e.edge_type == graphlib::EdgeType::kPartialDataCopy or
                                                   e.edge_type == graphlib::EdgeType::kSubgraphLink;
                                        })
                                    .empty();
    return output_link_queue or input_link_queue;
}

// Check whether queue is input queue on host, meaning it's data resides on host and is accessed via PCIe.
//
bool is_input_host_queue(bool input_queues_on_host, const Graph *graph, const Node *node)
{
    bool input_on_host =
        input_queues_on_host && node->as<graphlib::QueueNode>()->is_input() &&
        (node->as<graphlib::InputNode>()->is_activation() or node->as<graphlib::InputNode>()->is_loss()) &&
        not is_linked_queue(graph, node);

    return input_on_host;
}

// Check whether queue is output queue on host, meaning it's data resides on host and is transferred via PCIe.
//
bool is_output_host_queue(bool output_queues_on_host, const Graph *graph, const Node *node)
{
    bool output_on_host = output_queues_on_host && (node->node_type() == graphlib::NodeType::kOutput) &&
                          node->as<graphlib::OutputNode>()->untilize() && not is_linked_queue(graph, node);
    return output_on_host;
}

NodeGraphContainer::~NodeGraphContainer()
{
    if (remove_from_graph)
    {
        graph->remove_node(node);
    }
}

}  // namespace graphlib

}  // namespace tt
