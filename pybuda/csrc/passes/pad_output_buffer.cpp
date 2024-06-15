// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/pad_output_buffer.hpp"


#include "graph_lib/utils.hpp"
#include "graph_lib/node_types.hpp"
#include "passes/passes_utils.hpp"

namespace tt::passes
{
// Insert buda_pad op on specified edge
void insert_buda_pad(graphlib::Graph *graph, graphlib::Edge edge, int rt_pad_amount, int ct_pad_amount, graphlib::Shape original_output_shape, graphlib::Node *output_node)
{
    // Get node from edge
    graphlib::Node *node = graph->node_by_id(edge.consumer_node_id);

    // Construct buda_pad node and insert in graph
    graphlib::OpType buda_pad_op_type("buda_pad", {rt_pad_amount, ct_pad_amount, 0});
    auto buda_pad_ref_node = graph->add_node(
        graphlib::create_node<graphlib::PyOpNode>(node->name() + "_pad", buda_pad_op_type),
        graph->get_subgraph_id_for_node(node->id()));
    graphlib::insert_node_on_edge(graph, edge, buda_pad_ref_node);

    // Add runtime transform to referent node
    graphlib::RuntimeTensorTransform runtime_tensor_transform(original_output_shape);
    output_node->as<graphlib::OutputNode>()->set_runtime_tensor_transform(runtime_tensor_transform);
}


void pad_output_buffer(graphlib::Graph *graph, const DeviceConfig &device_config)
{
    bool pad_output_buffer = env_as<bool>("PYBUDA_PAD_OUTPUT_BUFFER");
    int pad_threshold = env_as<int>("PYBUDA_PAD_OUTPUT_BUFFER_THRESHOLD_TILES");
    if (not pad_output_buffer)
        return;

    std::vector<graphlib::Node *> padded_inputs;
    // Pad all outputs to the nearest multiple of the grid size
    for (graphlib::Node *output_node: graph->nodes_by_type(graphlib::NodeType::kOutput))
    {
        // Skip partial data copy edges (past-cache link between producers/consumers)
        std::vector<graphlib::Edge> partial_datacopy_edges = graph->user_partial_datacopy_edges(output_node);
        if (not partial_datacopy_edges.empty())
            continue;

        int available_grid_r = device_config.grid_size.r;
        int available_grid_c = device_config.grid_size.c;
        graphlib::Shape original_output_shape = output_node->shape();

        // If the output shape is not divisible by the grid size, pad to nearest multiple
        int original_output_rt = graphlib::Shape::to_buda(original_output_shape).rt();
        int original_output_ct = graphlib::Shape::to_buda(original_output_shape).ct();

        if ((original_output_rt * original_output_ct) < pad_threshold)
            continue;

        int rt_pad_amount = original_output_rt < available_grid_r ? 0 : (available_grid_r - (original_output_rt % available_grid_r)) % available_grid_r;
        int ct_pad_amount = original_output_ct < available_grid_c ? 0 : (available_grid_c - (original_output_ct % available_grid_c)) % available_grid_c;

        // No padding needed
        if (rt_pad_amount == 0 and ct_pad_amount == 0)
            continue;

        // Construct new padded shape
        graphlib::Shape padded_shape = original_output_shape;
        padded_shape[-1] = original_output_shape[-1] + (ct_pad_amount * graphlib::Shape::BUDA_TILE_DIM);
        padded_shape[-2] = original_output_shape[-2] + (rt_pad_amount * graphlib::Shape::BUDA_TILE_DIM);

        // Get output edges
        std::vector<graphlib::Edge> edges = graph->operand_data_edges(output_node);
        TT_ASSERT(edges.size() == 1);
        bool padded_ct = false;

        // If un-squeeze and matmul are preceding the output, we need to pad the second matmul operand
        if (graph->node_by_id(edges[0].producer_node_id)->as<graphlib::OpNode>()->op_type().op == "unsqueeze")
        {
            edges = graph->operand_data_edges(graph->node_by_id(edges[0].producer_node_id));
            TT_ASSERT(edges.size() == 1);
        }
        if (graph->node_by_id(edges[0].producer_node_id)->as<graphlib::OpNode>()->op_type().op == "matmul")
        {
            graphlib::Node *matmul = graph->node_by_id(edges[0].producer_node_id);
            edges = graph->operand_data_edges(matmul);
            if (ct_pad_amount != 0)
            {
                padded_ct = true;
                insert_buda_pad(graph, edges[1], 0, ct_pad_amount, original_output_shape, output_node);
                graphlib::Node *producer = graph->node_by_id(edges[1].producer_node_id);
                if (producer->node_type() == graphlib::NodeType::kInput)
                {
                    padded_inputs.push_back(producer);
                }
                else if (producer->as<graphlib::OpNode>()->op_type().op == "transpose")
                {
                    if (graph->data_operands(producer)[0]->node_type() == graphlib::NodeType::kInput)
                        padded_inputs.push_back(graph->data_operands(producer)[0]);
                }
            }

        }
        // Reset C pad if matmul second operand is already padded
        if (padded_ct)
            ct_pad_amount = 0;

        // No need to pad output if matmul is already padded on C dim
        if (rt_pad_amount == 0 and ct_pad_amount == 0)
            continue;

        // Construct buda_pad node and insert in graph
        graphlib::Edge buda_pad_ref_edge = graph->operand_data_edges(output_node)[0];
        insert_buda_pad(graph, buda_pad_ref_edge, rt_pad_amount, ct_pad_amount, original_output_shape, output_node);
    }

    recalculate_shapes(graph);
    for (graphlib::Node *input : padded_inputs)
    {
        bool constevaled = true;
        while(constevaled) 
        {
            constevaled = try_consteval_input_no_operand_forks(graph, input->as<graphlib::InputNode>(), true);
        }
    }

    return;
}
} // namespace tt::passes
