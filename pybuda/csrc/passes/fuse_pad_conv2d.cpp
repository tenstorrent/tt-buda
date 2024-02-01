// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>

#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "utils/logger.hpp"
#include "python_bindings_common.hpp"

#include "passes/consteval.hpp"

namespace tt::passes
{

// Find consecutive Pad and Conv2D nodes and fuse them into a single Conv2D node.
void fuse_pad_conv2d(graphlib::Graph *graph) {

    for (auto *node : graphlib::topological_sort(*graph)) {
        graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(node);
        if (not op or op->op_name() != "pad")
            continue;

        auto attrs = op->op_attrs();
        if (std::get<int>(attrs[attrs.size() - 2]) != 0) {
            // Second last attr must be 0 (constant mode), otherwise cannot fuse into conv2d
            continue;
        }
        auto users = graph->users(node);

        bool all_users_are_conv2d = true;
        for (auto *user : users) {
            graphlib::OpNode *user_op = dynamic_cast<graphlib::OpNode *>(user);
            if (not user_op or user_op->op_name() != "conv2d") {
                all_users_are_conv2d = false;
                break;
            }
        }

        if (not all_users_are_conv2d)
            continue;

        auto pad_attrs = op->op_attrs();
        TT_ASSERT(pad_attrs.size() == 4 or pad_attrs.size() == 6);

        if (pad_attrs.size() == 4) {
            // Expand pad attributes to match conv2d attributes
            pad_attrs.insert(pad_attrs.begin() + 2, 0);
            pad_attrs.insert(pad_attrs.begin() + 3, 0);
        }
        // Add Pad to Conv2d attributes
        for (auto user : users) {
            graphlib::OpNode *user_op = dynamic_cast<graphlib::OpNode *>(user);
            auto conv_attrs = user_op->op_attrs();
            TT_ASSERT(conv_attrs.size() == 13);
            // Conv2d attributes are [stride[0],stride[1],dilation,groups,padding[0],padding[1],padding[2],padding[3],channel_last]
            int pad_idx_offset = 4;
            for (uint32_t i = 0; i < 4; i++) {
                conv_attrs[pad_idx_offset + i] = std::get<int>(pad_attrs[i]) + std::get<int>(conv_attrs[pad_idx_offset + i]);
            }
            user_op->overwrite_op_attrs(conv_attrs);
        }

        // bypass the pad node
        graphlib::bypass_node(graph, node, true);
    }
}
}  // namespace tt::passes
