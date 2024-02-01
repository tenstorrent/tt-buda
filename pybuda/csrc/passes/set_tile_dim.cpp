// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/set_tile_dim.hpp"

#include <pybind11/pybind11.h>

#include "graph_lib/utils.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/shape.hpp"

namespace tt::passes
{

int calculate_tile_size(int val)
{
    // We might not even care about large dim size 
    // that are not divisible by 32
    if (val > 32)
        return 32;

    int smallest_pad = 31;
    int current_tile_size = 32;

    std::vector<int> tile_sizes = {32, 16, 8, 4, 2, 1};

    for (auto tile_size_ : tile_sizes)
    {
        int rem = val % tile_size_;
        int pad = tile_size_ - rem;
        if (rem == 0 and smallest_pad != 0) {
            // Pick the largest tile size that divides evenly
            smallest_pad = 0;
            current_tile_size = tile_size_;
        } else if (pad <= smallest_pad) {
            // pick the tile size with smallest pad
            smallest_pad = pad;
            current_tile_size = tile_size_;
        }
    }
    return current_tile_size;
}


bool has_matmul_as_input(graphlib::Graph *graph, graphlib::Node *node)
{
    for (auto *operand : graph->operands(node))
    {
        graphlib::OpNode *op_node = dynamic_cast<graphlib::OpNode *>(operand);
        if (op_node and op_node->is_matmul())
            return true;
    }
    return false;
}

void set_tile_dim_for_nodes(graphlib::Graph *graph)
{

    // Unary ops and Matmul SrcA can have reduced tile_dim

    for (auto *node : graphlib::topological_sort(*graph))
    {


        graphlib::OpNode *op_node = dynamic_cast<graphlib::OpNode *>(node);


        if (not op_node)
            continue;

        if (not (op_node->is_matmul() or 
                    (graphlib::is_eltwise_binary(op_node) and has_matmul_as_input(graph, node))))
            continue;

        TileDim calculated_tile_dim = TileDim::Dim32x32;
        graphlib::Shape shape = op_node->shape();
        int tile_size_r = calculate_tile_size(shape[-2]); 
        int tile_size_c = 32; // Force Column to 32 for now

        calculated_tile_dim = graphlib::get_tile_dim_from_height_width(tile_size_r, tile_size_c);
        node->set_tile_dim(calculated_tile_dim);


        if (graphlib::is_eltwise_binary(op_node))
        {
            // Make sure both input to eltwise binary have same tile dim
            for (auto operand : graph->operands(node))
            {
                if (operand->shape().get_tile_dim() != calculated_tile_dim)
                {
                    operand->set_tile_dim(calculated_tile_dim);
                }
            }
        }

    }

}
}



