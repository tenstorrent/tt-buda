// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/print_graph.hpp"

#include <string.h>

#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "utils/logger.hpp"

namespace tt::passes
{

void print_node(graphlib::Node *node) { std::cout << node->name() << node->shape() << std::endl; }

bool print_opnode(graphlib::Graph *graph, graphlib::Node *node)
{
    auto op = dynamic_cast<graphlib::OpNode *>(node);
    if (not op)
    {
        return false;
    }
    std::cout << op->name() << op->shape() << " = (";
    for (auto operand : graph->data_operands(op))
    {
        std::cout << operand->name() << ", ";
    }
    std::cout << ") : {OP: " << op->op_type() << "}" << std::endl;
    return true;
}

void print_graph_regular(graphlib::Graph *graph, std::string stage)
{
    auto stage_to_print = env_as_optional<std::string>("PYBUDA_PRINT_GRAPH_AT");
    if (not stage_to_print or ((stage_to_print != stage) and (stage_to_print != "ALL")))
    {
        return;
    }
    std::cout << "Graph at: " << stage << std::endl;

    // Print inputs
    std::cout << "#! INPUTS !#" << std::endl;
    std::vector<graphlib::Node *> inputs = graph->ordered_module_inputs();
    for (graphlib::Node *node : inputs)
    {
        print_node(node);
    }

    // Print graph nodes
    std::cout << "#! GRAPH !#" << std::endl;
    for (graphlib::Node *node : graphlib::topological_sort(*graph))
    {
        print_opnode(graph, node);
    }

    // Print outputs
    std::cout << "#! OUTPUTS !#" << std::endl;
    std::vector<graphlib::Node *> outputs = graph->ordered_module_outputs();
    for (graphlib::Node *node : outputs)
    {
        print_node(node);
    }

    std::cout << "Graph end" << std::endl;
}

void print_graph_viz_nodes_definitions(graphlib::Node *node)
{
    // Clean op names for GraphViz (dots are not supported)
    std::string node_name = "";
    node_name = node->name();
    std::replace(node_name.begin(), node_name.end(), '.', '_');

    // Determine node type
    std::string node_type = "";
    if (node->node_type() == graphlib::NodeType::kInput)
    {
        node_type = "input";
    }
    else if (node->node_type() == graphlib::NodeType::kOutput)
    {
        node_type = "output";
    }

    std::cout << node_name << "["
              << "label=\""
              << "name: " << node_name << "\\n"
              << "shape: " << node->shape() << "\\n"
              << "type: " << node_type << "\"]" << std::endl;
}

void print_graph_viz_op_nodes_definitions(graphlib::Node *node)
{
    auto print_bw_graph = env_as_optional<std::string>("PYBUDA_PRINT_GRAPH_VIZ_FORMAT_DIR");

    auto op = dynamic_cast<graphlib::OpNode *>(node);
    if (not op)
        return;

    // Clean op names for GraphViz (dots are not supported)
    std::string node_name = "";
    node_name = op->name();
    std::replace(node_name.begin(), node_name.end(), '.', '_');

    // Print forward or backward nodes only if defined by env var
    if (print_bw_graph && print_bw_graph == "forward_only" && node_name.rfind("bw_", 0) == 0)
        return;
    else if (print_bw_graph && print_bw_graph == "backward_only" && not(node_name.rfind("bw_", 0) == 0))
        return;

    std::cout << node_name << "["
              << "label=\""
              << "name: " << node_name << "\\n"
              << "shape: " << op->shape() << "\\n"
              << "op: " << op->op_type() << "\"]" << std::endl;
}

void print_graph_viz_op_nodes_relations(graphlib::Graph *graph, graphlib::Node *node)
{
    auto print_bw_graph = env_as_optional<std::string>("PYBUDA_PRINT_GRAPH_VIZ_FORMAT_DIR");

    auto op = dynamic_cast<graphlib::OpNode *>(node);
    if (not op)
        return;

    // Clean op names for GraphViz (dots are not supported)
    std::string node_name = "";
    node_name = op->name();
    std::replace(node_name.begin(), node_name.end(), '.', '_');

    // Print relations with each operand
    for (auto operand : graph->data_operands(op))
    {
        // Clean operands op names for GraphViz (dots are not supported)
        std::string operand_node_name = "";
        operand_node_name = operand->name();
        std::replace(operand_node_name.begin(), operand_node_name.end(), '.', '_');

        // Print forward or backward nodes only if defined by env var
        if (print_bw_graph && print_bw_graph == "forward_only" &&
            (node_name.rfind("bw_", 0) == 0 or operand_node_name.rfind("bw_", 0) == 0))
            return;
        else if (
            print_bw_graph && print_bw_graph == "backward_only" &&
            not(node_name.rfind("bw_", 0) == 0 or operand_node_name.rfind("bw_", 0) == 0))
            return;

        std::cout << operand_node_name << " -> " << node_name << std::endl;
    }
}

/**
 * @brief Print graph structure formatted to fit GraphViz visualizer
 *
 * For usage, just copy whole, or part of the output graph and past it
 * into GraphViz online console:
 * https://dreampuf.github.io/GraphvizOnline
 */
void print_graph_viz_format(graphlib::Graph *graph, std::string stage)
{
    auto stage_to_print = env_as_optional<std::string>("PYBUDA_PRINT_GRAPH_VIZ_FORMAT_AT");
    if (not stage_to_print or ((stage_to_print != stage) and (stage_to_print != "ALL")))
        return;
    std::cout << "Graph at: " << stage << std::endl;

    // GraphViz DiGraph opening
    std::cout << "DiGraph G {" << std::endl;

    // Print inputs
    std::cout << "// Inputs" << std::endl;
    std::vector<graphlib::Node *> inputs = graph->ordered_module_inputs();
    for (graphlib::Node *node : inputs)
    {
        print_graph_viz_nodes_definitions(node);
    }

    // Print outputs
    std::cout << "// Outputs" << std::endl;
    std::vector<graphlib::Node *> outputs = graph->ordered_module_outputs();
    for (graphlib::Node *node : outputs)
    {
        print_graph_viz_nodes_definitions(node);
    }

    // Print definition of op nodes
    std::cout << "// Op nodes:" << std::endl;
    for (graphlib::Node *node : graphlib::topological_sort(*graph))
    {
        print_graph_viz_op_nodes_definitions(node);
    }

    // Print relationships of op nodes
    std::cout << "// Op relationships:" << std::endl;
    for (graphlib::Node *node : graphlib::topological_sort(*graph))
    {
        print_graph_viz_op_nodes_relations(graph, node);
    }

    // GraphViz DiGraph closing
    std::cout << "}" << std::endl;

    std::cout << "Graph end" << std::endl;
}

void print_graph(graphlib::Graph *graph, std::string stage)
{
    auto regular_print = env_as_optional<std::string>("PYBUDA_PRINT_GRAPH_AT");
    auto graph_viz_format = env_as_optional<std::string>("PYBUDA_PRINT_GRAPH_VIZ_FORMAT_AT");

    if (regular_print)
    {
        print_graph_regular(graph, stage);
    }
    else if (graph_viz_format)
    {
        print_graph_viz_format(graph, stage);
    }
}

}  // namespace tt::passes
