// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/reproduce_subgraph.hpp"

#include <pybind11/pybind11.h>

#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "utils/logger.hpp"
#include "python_bindings_common.hpp"

#include "balancer/types.hpp"
#include "passes/consteval.hpp"

namespace tt::passes
{


static std::vector<graphlib::Node *> find_path_to_node(
    graphlib::Graph *graph,
    graphlib::OpNode *initial_op,
    graphlib::OpNode *final_op,
    graphlib::OpNode *from = nullptr)
{
    std::vector<graphlib::Node *> path;

    graphlib::OpNode *iter = from ? from : initial_op;

    bool found_op = false;
    while (not found_op)
    {   
        graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(iter);
        TT_ASSERT(op);
        path.push_back(op);

        std::vector<graphlib::Node *> users = graph->data_users(op);
        for (std::size_t i = 1; i < users.size(); ++i)
        {
            graphlib::OpNode *user = dynamic_cast<graphlib::OpNode *>(users[i]);
            auto fork_path = find_path_to_node(graph, initial_op, final_op, user);
            if (not fork_path.empty())
            {
                path.insert(path.end(), fork_path.begin(), fork_path.end());
                found_op = true;
                break;
            }
        }

        if(op->id() == final_op->id())
        {
            found_op = true;
            break;
        }

        TT_ASSERT(users.size() > 0);
        graphlib::OutputNode *output = dynamic_cast<graphlib::OutputNode *>(users[0]);
        if (output)
            break;
        
        iter = dynamic_cast<graphlib::OpNode *>(users[0]);
        if (not iter)
            break;
    }

    if (not found_op)
        path.clear();

    return path;
}



void reproduce_subgraph(
    graphlib::Graph *graph,
    std::string input_name,
    std::string output_name,
    std::unordered_map<std::string, py::object> intermediates,
    std::shared_ptr<balancer::BalancerSolution> balancer_solution,
    placer::PlacerSolution *placer_solution)
{
    std::vector<graphlib::Node *> nodes_to_keep;
    graphlib::OpNode *input_node;
    graphlib::OpNode *output_node;

    graphlib::Node *last_node = nullptr;
    TT_ASSERT(input_name.length() > 0, "No input node provided");
    TT_ASSERT(output_name.length() > 0, "No output node provided");

    TT_ASSERT(graph->get_node_by_name(input_name), "Input node not found");
    TT_ASSERT(graph->get_node_by_name(output_name), "Output node not found");
    input_node = dynamic_cast<graphlib::OpNode *>(graph->get_node_by_name(input_name));
    output_node = dynamic_cast<graphlib::OpNode *>(graph->get_node_by_name(output_name));

    nodes_to_keep = find_path_to_node(graph, dynamic_cast<graphlib::OpNode *>(input_node), dynamic_cast<graphlib::OpNode *>(output_node));
    TT_ASSERT(nodes_to_keep.size() > 0, "No path found between input and output nodes");
    last_node = nodes_to_keep.back();
    std::vector<graphlib::Node *> needed_inputs;
    for (auto node : nodes_to_keep)
    {
        std::vector<graphlib::Edge> operand_data_edges = graph->operand_data_edges(node);
        int i = 0;
        for (auto operand_data_edge : operand_data_edges)
        {
            auto operand = graph->node_by_id(operand_data_edge.producer_node_id);
            if (std::find(nodes_to_keep.begin(), nodes_to_keep.end(), operand) != nodes_to_keep.end())
                continue;
            // for first operand that's not an input node, connect it to the graph input, with a runtime transform
            // for subsequent nodes, create a constant input node
            if (not dynamic_cast<graphlib::InputNode *>(operand))
            {
                if (dynamic_cast<graphlib::QueueNode *>(operand))
                {
                    operand = graph->data_operands(operand)[0];
                }
                TT_ASSERT(dynamic_cast<graphlib::OpNode *>(operand), "Something went wrong");
                graphlib::Node *input;
                if (i == 0)
                {
                    input = graph->ordered_module_inputs()[0];
                    graphlib::InputNode *in_node = dynamic_cast<graphlib::InputNode *>(input);
                    graphlib::RuntimeTensorTransform runtime_tensor_transform {};
                    runtime_tensor_transform.set_constant_input_tensor(intermediates[operand->name()]);
                    in_node->set_runtime_tensor_transform(runtime_tensor_transform);
                    input->set_shape(operand->shape());
                    for (auto edge : graph->user_data_edges(input))
                    {
                        graph->remove_edge(edge);
                    }
                    balancer::BlockShape block_shape = balancer_solution->op_models[operand->name()].output_buffers[0].block_shape;
                    block_shape.mblock_m *= balancer_solution->op_models[operand->name()].grid_shape.r;
                    block_shape.mblock_n *= balancer_solution->op_models[operand->name()].grid_shape.c;
                    balancer_solution->block_shapes[input->name()] = block_shape;

                    placer_solution->name_to_queue_placement.erase(input->name());
                    i++;
                }
                else
                {
                    py::object constant_value = intermediates[operand->name()];
                    input = graph->add_node(graphlib::create_node<graphlib::ConstantInputNode>(
                            operand->name() + "_constant_input_" + std::to_string(i++),
                            make_shared_py_object(constant_value),
                            operand->shape()), graph->get_subgraph_id_for_node(node->id()));

                    tt::balancer::GridShape grid_shape = tt::balancer::GridShape(1, 1);
                    tt::balancer::BlockShape block_shape = tt::balancer::BlockShape(operand->shape(), grid_shape.r, grid_shape.c, 1, tt::balancer::UBlockShape(1, 1));
                    tt::balancer::BufferModel input_buffer_model = tt::balancer::BufferModel(block_shape, 1, input->output_df());

                    tt::balancer::OpModel input_op_model;
                    input_op_model.grid_shape = grid_shape;
                    input_op_model.op_shape.outputs.push_back(operand->shape());
                    input_op_model.output_buffers.push_back(input_buffer_model);
                    input_op_model.data_format = input->output_df();
                    input_op_model.input_prologue = false;

                    balancer_solution->op_models[node->name()] = input_op_model;
                    balancer_solution->block_shapes[node->name()] = block_shape;
                    placer_solution->input_queue_to_grid_shape.insert(
                        {node->name(),
                         tt::placer::GridShape((std::uint32_t)grid_shape.r, (std::uint32_t)grid_shape.c)});
                }
                graphlib::Edge new_edge(operand_data_edge);
                new_edge.producer_node_id = input->id();
                graph->add_edge(new_edge);
                graph->copy_edge_attributes(operand_data_edge, new_edge);
                graph->remove_edge(operand_data_edge);
                needed_inputs.push_back(input);
            }
            else
            {
                needed_inputs.push_back(operand);
            }
        }
    }
    nodes_to_keep.insert(nodes_to_keep.end(), needed_inputs.begin(), needed_inputs.end());
    for (graphlib::Node *node : graph->ordered_module_inputs())
        nodes_to_keep.push_back(node);

    int i = 0;
    for (graphlib::Node *node : graph->ordered_module_outputs())
    {
        nodes_to_keep.push_back(node);
        if (i++ == 0)
        {
            placer_solution->name_to_queue_placement.erase(node->name());
            graph->add_edge(last_node, node, graphlib::EdgeType::kData);
            graph->set_output_node_redirected(true);
            graphlib::OutputNode *out_node = dynamic_cast<graphlib::OutputNode *>(node);
            out_node->set_runtime_tensor_transform(graphlib::RuntimeTensorTransform());
            out_node->set_untilize(false);
            node->set_shape(last_node->shape());

            balancer::BlockShape block_shape = balancer_solution->op_models[last_node->name()].output_buffers[0].block_shape;
            balancer_solution->block_shapes[out_node->name()] = block_shape;
        }
    }

    for (auto node : nodes_to_keep)
    {
        placer_solution->name_to_op_placement[node->name()].global_epoch_id = 0;
    }
    for (unsigned int i = 1; i < placer_solution->num_epochs; i++)
    {
        placer_solution->epoch_id_to_chip.erase(i);
        placer_solution->epoch_id_to_op_placement.erase(i);
        placer_solution->epoch_id_to_epoch_info.erase(i);
    }
    placer_solution->num_epochs = 1;

    for (auto node : graphlib::topological_sort(*graph))
    {
        if (std::find(nodes_to_keep.begin(), nodes_to_keep.end(), node) == nodes_to_keep.end())
        {
            graph->remove_node(node);
        }
    }

}
}  // namespace tt::passes
