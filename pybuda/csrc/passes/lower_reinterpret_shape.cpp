// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/lower_reinterpret_shape.hpp"

#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "utils/logger.hpp"

namespace tt::passes
{
void lower_reinterpret_shape(graphlib::Graph *graph)
{
    auto is_input_and_all_users_same = [graph](graphlib::Node *operand, graphlib::OpNode *op) -> bool
    {
        graphlib::InputNode *input = dynamic_cast<graphlib::InputNode *>(operand);
        if (not input or not input->is_activation())
            return false;
        for (graphlib::Node *user : graph->data_users(operand))
        {
            graphlib::OpNode *user_op = dynamic_cast<graphlib::OpNode *>(user);
            if (not user_op or user_op->op_type() != op->op_type())
                return false;
        }
        return true;
    };

    std::vector<graphlib::Node *> removed;
    for (graphlib::Node *node : graphlib::topological_sort(*graph))
    {
        if (std::find(removed.begin(), removed.end(), node) != removed.end())
            continue;

        graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(node);
        if (not op or (op->op_name() != "reshape"))
            continue;

        auto reinterpret = op->shape();

        std::vector<graphlib::Node *> operands = graph->data_operands(op);
        std::vector<graphlib::Node *> users = graph->data_users(op);
        TT_ASSERT(operands.size() == 1);
        graphlib::Node *operand = operands[0];
        if (is_input_and_all_users_same(operand, op))
        {
            // Reinterpret with a Z causes issue disable for now
            if (reinterpret.size() > 2 and reinterpret[-3] > 1)
                continue;

            graphlib::InputNode *input = operand->as<graphlib::InputNode>();
            auto original = input->shape();
            graphlib::RuntimeTensorTransform runtime_tensor_transform(original, reinterpret);

            input->set_runtime_tensor_transform(runtime_tensor_transform);
            input->set_shape(reinterpret);

            for (auto *input_user : graph->data_users(input))
            {
                removed.push_back(input_user);
                bypass_node(graph, input_user, true);
            }
        }
        else if (users.size() == 1 and users[0]->node_type() == graphlib::NodeType::kOutput)
        {
            graphlib::OutputNode *output = users[0]->as<graphlib::OutputNode>();
            // Inherit shape from operand
            auto original = operand->shape();
            output->set_shape(original);
            graphlib::RuntimeTensorTransform runtime_tensor_transform(original, reinterpret);
            output->set_runtime_tensor_transform(runtime_tensor_transform);
            removed.push_back(op);
            bypass_node(graph, op, true);
        }
    }
}
}  // namespace tt
