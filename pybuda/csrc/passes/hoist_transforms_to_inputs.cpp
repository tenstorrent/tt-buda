// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/hoist_transforms_to_inputs.hpp"

#include <pybind11/pybind11.h>

#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
// #include "utils/logger.hpp"
// #include "passes/erase_inverse_ops.hpp"

namespace tt::passes
{

void hoist_transforms_to_inputs(tt::graphlib::Graph *graph)
{
    std::vector<graphlib::Node *> removed;
    for (graphlib::Node *node : graphlib::topological_sort(*graph))
    {
        if (std::find(removed.begin(), removed.end(), node) != removed.end())
        {
            continue;
        }

        graphlib::OpNode *op_node = dynamic_cast<graphlib::OpNode *>(node);
        graphlib::TaggedNode *tagged_node = node->as<graphlib::TaggedNode>();
        if (not op_node or not tagged_node or !tagged_node->has_tag("optimize_hoist"))
        {
            continue;
        }

        auto op_type = op_node->op_type();

        std::vector<graphlib::Node *> operands = graph->data_operands(op_node);
        std::vector<graphlib::Node *> users = graph->data_users(op_node);
        TT_ASSERT(operands.size() == 1);
        graphlib::Node *operand = operands[0];

        graphlib::InputNode *input = operand->as<graphlib::InputNode>();
        TT_ASSERT(input);

        // TODO: Something (like a tag) should tell us which transform to tack on
        graphlib::RuntimeTensorTransform runtime_tensor_transform {};
        runtime_tensor_transform.type = graphlib::RuntimeTensorTransformType::Prestride;
        runtime_tensor_transform.original_shape = input->shape();
        runtime_tensor_transform.reinterpreted_shape = op_node->shape();

        runtime_tensor_transform.stride_height = std::get<int>(op_type.attr[0]);
        runtime_tensor_transform.stride_width = std::get<int>(op_type.attr[1]);
        runtime_tensor_transform.kernel_height = std::get<int>(op_type.attr[2]);
        runtime_tensor_transform.kernel_width = std::get<int>(op_type.attr[3]);

        input->set_runtime_tensor_transform(runtime_tensor_transform);
        input->set_shape(runtime_tensor_transform.reinterpreted_shape);

        for (auto *input_user : graph->data_users(input))
        {
            removed.push_back(input_user);
            bypass_node(graph, input_user, true);
        }
    }
}

}  // namespace tt::passes
