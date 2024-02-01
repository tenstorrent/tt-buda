// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/erase_consecutive_reshape.hpp"

#include <pybind11/pybind11.h>

#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "utils/logger.hpp"
#include "python_bindings_common.hpp"

#include "passes/consteval.hpp"

namespace tt::passes
{


static bool is_elementwise_binary(graphlib::OpNode *op, graphlib::Graph *graph)
{
    py::object eval_module = py::module_::import("pybuda.op.eval.pybuda");
    py::function is_eltwise = eval_module.attr("is_eltwise");
    bool is_eltwise_op = is_eltwise(op->op_type()).cast<bool>();
    bool is_binary = graph->data_operands(op).size() == 2;
    return is_eltwise_op and is_binary;
}

static bool can_fuse_select_concat(
    std::map<int, std::vector<std::pair<graphlib::OpNode *, graphlib::InputNode *>>> const& map, 
    int concat_dim, graphlib::Graph *graph)
{
    if (map.empty())
        return false;

    // the lambda will only get called when the map is not empty
    // so we can safely access begin()->second
    auto const cmpWithFirst = [&](std::pair<int, std::vector<std::pair<graphlib::OpNode *, graphlib::InputNode *>>> const& i)
    {
        auto firstOp = map.begin()->second.back().first;
        auto compare_op = i.second.back().first;
        if (firstOp->op_name() != "select")
            return false;

        bool same_op = firstOp->op_name() == compare_op->op_name();
        bool same_dim = 
                (std::get<int>(firstOp->op_attrs()[0]) == std::get<int>(compare_op->op_attrs()[0])) 
                and (std::get<int>(firstOp->op_attrs()[0]) == concat_dim);
        bool same_producer = 
                (graph->data_operands(firstOp)[0] == graph->data_operands(compare_op)[0]);

        // Stride must equal to producer concat dim
        const int stride = graph->data_operands(firstOp)[0]->shape()[concat_dim];
        bool same_stride = 
                (std::get<int>(firstOp->op_attrs()[3]) == std::get<int>(compare_op->op_attrs()[3])) 
                and (std::get<int>(firstOp->op_attrs()[3]) == stride); 

        

        return same_op and same_dim and same_stride and same_producer;
    };

    bool same_op_setup = std::all_of(map.begin(), map.end(), cmpWithFirst);

    if (not same_op_setup)
        return false;

    // Check that all channels are selected
    uint32_t select_amount = 0;
    for (auto const& [key, value] : map)
    {
        select_amount += std::get<int>(value.back().first->op_attrs()[2]);
    }

    bool select_all_channels = select_amount == graph->data_operands(map.begin()->second.back().first)[0]->shape()[concat_dim];

    // Check no overlap among select indices
    std::vector<int> start_indices;
    std::vector<uint32_t> index_count(select_amount, 0);

    for (auto const& [key, value] : map)
    {
        auto start_index = std::get<int>(value.back().first->op_attrs()[1]);
        auto length = std::get<int>(value.back().first->op_attrs()[2]);
        
        for (auto i = start_index; i < start_index + length; i++) {
            index_count[i] += 1;
        }
        start_indices.push_back(start_index);
    }

    // Make sure each index is counted exactly once
    bool no_overlap = std::all_of(index_count.begin(), index_count.end(), [](uint32_t index) { return index == 1; });

    // Make sure start index are monotonically increasingh
    bool monotonically_increasing = true;
 
    for (uint i = 1; i < start_indices.size() ; i++) {
        if (start_indices[i] < start_indices[i - 1]) {
            monotonically_increasing = false;
            break;
        }
    }
    return same_op_setup and select_all_channels and no_overlap and monotonically_increasing;

}

static bool find_path_from_concat_to_select(
    graphlib::Graph *graph, std::vector<graphlib::Node *> & operands, int num_operands,
    std::map<int, std::vector<std::pair<graphlib::OpNode *, graphlib::InputNode *>>> & ops_to_fuse)
{
    bool can_fuse = true;
    for (int i = 0; i < num_operands; i++)
    {
        auto current_op = dynamic_cast<graphlib::OpNode *>(operands[i]);
        while (can_fuse) {
            if (not current_op)
            {
                can_fuse = false;
                break;
            }

            if (current_op->op_name() == "select") {
                // Found producer select, add to map and break
                ops_to_fuse[i].push_back(std::make_pair(current_op, nullptr));
                break;
            } else if (is_elementwise_binary(current_op, graph) and 
                        (current_op->op_name() == "add" or current_op->op_name() == "multiply" or current_op->op_name() == "subtract")) {
                // Found elementwise binary, add to map and continue
                // Currently only support fusion with add/mul/sub

                auto input_nodes = graph->data_operands(current_op);
                auto input_0 = dynamic_cast<graphlib::InputNode *>(input_nodes[0]);
                auto input_1 = dynamic_cast<graphlib::InputNode *>(input_nodes[1]);
                if (input_0 and input_0->is_constant())
                {
                    ops_to_fuse[i].push_back(std::make_pair(current_op, input_0));
                    // Update current_op to be the other input
                    current_op = dynamic_cast<graphlib::OpNode *>(input_nodes[1]);
                }
                else if (input_1 and input_1->is_constant())
                {
                    ops_to_fuse[i].push_back(std::make_pair(current_op, input_1));
                    // Update current_op to be the other input
                    current_op = dynamic_cast<graphlib::OpNode *>(input_nodes[0]);
                } else {
                    // Found non-constant input, break
                    can_fuse = false;
                    break;
                }
            } else {
                // Found non-select, non-elementwise binary, break
                can_fuse = false;
                break;
            }
        }

        if (not can_fuse)
            break;
    }

    return can_fuse;
}



static bool create_merge_groups(
    std::map<int, std::vector<std::pair<graphlib::OpNode *, graphlib::InputNode *>>> & ops_to_fuse,
    std::map<int, std::vector<std::pair<graphlib::OpNode *, graphlib::InputNode *>>> & merge_groups,
    std::vector<int> & merge_group_operand_indices)
{
    bool can_fuse = true;

    // Find the longest path from Select to Concat
    auto longest_path = std::max_element(ops_to_fuse.begin(), ops_to_fuse.end(),
                                         [](auto const &a, auto const &b) {
                                             return a.second.size() < b.second.size();
                                         });

    int longest_path_op_index = longest_path->first;



    // Create merge groups and place each op in the longest path into its own group
    merge_group_operand_indices.push_back(longest_path_op_index);
    for (long unsigned int i = 0; i < longest_path->second.size(); i++) {
        merge_groups[i].push_back(longest_path->second[i]);
    }

    // Loop through all other operand paths to populate merge groups
    // If an op is in the same path as the longest path, add it to the same merge group
    // TODO: Currently does not support mutually exclusive set of ops among paths
    for (auto iter : ops_to_fuse)
    {
        auto key = iter.first;
        auto value = iter.second;
        if (key == longest_path_op_index)
            continue;

        merge_group_operand_indices.push_back(key);

        // Loop through all ops along this path
        long unsigned int merge_group_idx = 0;
        for (long unsigned int i = 0; i < value.size(); i++) {
            auto op = value[i].first;
            auto inp = value[i].second;
            bool found_group = false;

            // Try finding a merge group with the same op type and input shape
            for (long unsigned int j = merge_group_idx; j < longest_path->second.size(); j++) {
                auto reference_op = merge_groups[j][0].first;
                auto reference_inp = merge_groups[j][0].second;

                if (
                    reference_op->op_name() == op->op_name() 
                    and ((reference_inp == nullptr and inp == nullptr) or inp->shape() == reference_inp->shape())
                ) {
                    // Found merge group
                    merge_groups[j].push_back(value[i]);
                    found_group = true;
                    merge_group_idx++;
                    break;
                } else {
                    // Not found merge group, continue to next merge group
                    // Because we traverse the path from producer to consumer, we can assume that
                    // the rest of the ops in this path will not match any of the merge groups above current op.
                    // Therefore, we can fill current location with nullptr and move on to next merge group.
                    merge_groups[j].push_back(std::make_pair(nullptr, nullptr));
                    merge_group_idx++;
                }
            }
            if (not found_group){
                can_fuse = false;
                break;
            }
        }
        // Fill in the rest of merge groups with nullptr
        if (merge_group_idx < longest_path->second.size()) {
            for (long unsigned int j = merge_group_idx; j < longest_path->second.size(); j++) {
                merge_groups[j].push_back(std::make_pair(nullptr, nullptr));
            }
        }
    }

    return can_fuse;
}
static bool fuse_per_channel_concat(graphlib::Graph *graph, graphlib::OpNode *concat)
{

    bool can_fuse = true;
    auto operands = graph->data_operands(concat);
    int num_operands = operands.size();
    int concat_dim = std::get<int>(concat->op_attrs()[0]);
    std::map<int, std::vector<std::pair<graphlib::OpNode *, graphlib::InputNode *>>> ops_to_fuse;

    // Find path from each operand of concat to producer select, if possible
    can_fuse &= find_path_from_concat_to_select(graph, operands, num_operands, ops_to_fuse);
    if (not can_fuse)
        return false;

    // Validate that all paths are setup in the same way
    can_fuse &= can_fuse_select_concat(ops_to_fuse, concat_dim, graph);
    if (not can_fuse)
        return false;

    auto input_to_fused_block = graph->data_operands(ops_to_fuse.begin()->second.back().first)[0];
    std::map<int, graphlib::Shape> operand_input_shapes;

    for (auto & [key, value] : ops_to_fuse)
    {
        auto select = value.back().first;
        operand_input_shapes[key] = select->shape();

        // Reverse value vector so that it is in order of producer to consumer
        std::reverse(value.begin(), value.end());
    }


    std::map<int, std::vector<std::pair<graphlib::OpNode *, graphlib::InputNode *>>> merge_groups;
    std::vector<int> merge_group_operand_indices;

    // Place parallel ops that need to be merged into their own groups
    // Paths hold ops in a depth-first order, we need to place them in a breadth-first order
    // If certain ops are missing from a path, we need to fill in the gaps with place holder nullptr
    can_fuse &= create_merge_groups(ops_to_fuse, merge_groups, merge_group_operand_indices);

    if (not can_fuse)
        return false;

    // Finally, fuse the ops
    for (auto iter : merge_groups) {
        int layer_id = iter.first;
        auto ops = iter.second;
        if (ops[0].first->op_name() == "select") {
            for (auto & [op, inp] : ops) {
                bypass_node(graph, op, true);
            }
        } else {
            // This is an eltwise binary op
            auto const_name = concat->name() + "_const_concat_" + std::to_string(layer_id);
            auto const_concat = graph->add_node(concat->clone(const_name), graph->get_subgraph_id_for_node(concat->id()));

            // Merge the group of ops together
            for (long unsigned int i = 0; i < ops.size(); i++) {
                auto op = ops[i].first;
                auto inp = ops[i].second;

                if (op != nullptr) {
                    auto current_edge = graph->get_edges(inp, op)[0];
                    auto input_ndim = inp->shape().size();

                    while (input_ndim < concat->shape().size()) {
                        // insert unsqueeze tms
                        graphlib::OpType op_type("unsqueeze", {0, (int)input_ndim});
                        graph->get_edge_attributes(current_edge)->append_tm(op_type);
                        input_ndim++;
                    }

                    graphlib::Edge edge(
                        dynamic_cast<graphlib::Node*>(inp)->id(),
                        0, 
                        dynamic_cast<graphlib::Node*>(const_concat)->id(),
                        merge_group_operand_indices[i], 
                        graphlib::EdgeType::kData);

                    // Copy over edge attributes
                    graph->add_edge(edge);
                    graph->copy_edge_attributes(current_edge, edge);
                    graph->remove_edge(current_edge);

                }
                else {
                    // This is a place holder for a missing op along the path
                    // We need to insert a dummy const node with the correct shape / value
                    // Since first op in group is guaranteed to be an eltwise binary op,
                    // we can use the op type/shape to create the dummy op.

                    auto first_op_in_group = ops[0].first;
                    float const_value;
                    if (first_op_in_group->op_name() == "add" or first_op_in_group->op_name() == "subtract")
                        const_value = 0.0;
                    else if (first_op_in_group->op_name() == "multiply")
                        const_value = 1.0;
                    else
                        throw std::runtime_error("Unsupported eltwise fusion: " + ops[0].first->op_name());

                    auto first_op_input_edge = graph->get_edges(ops[0].second, const_concat);
                    TT_ASSERT(first_op_input_edge.size() == 1);
                    auto tms = graph->get_edge_attributes(first_op_input_edge[0])->get_tms();

                    graphlib::Shape const_shape = ops[0].second->shape();
                    if (const_shape.size() == operand_input_shapes.at(merge_group_operand_indices[i]).size())
                        const_shape[concat_dim] = operand_input_shapes.at(merge_group_operand_indices[i])[concat_dim];

                    // Create a dummy const node
                    py::object eval_module = py::module_::import("pybuda.op.eval");
                    auto const_tensor = make_shared_py_object(
                            eval_module.attr("create_constant_tensor_from_tensor")
                                        (std::vector<float>{const_value}, const_shape.as_vector(), false, ops[0].second->output_df()));
                    auto const_node = graph->add_node(graphlib::create_node<graphlib::ConstantInputNode>(
                            "created_const_op" + std::to_string(layer_id) + "_" + std::to_string(i), 
                            const_tensor,
                            const_shape), graph->get_subgraph_id_for_node(first_op_in_group->id()));
                    const_node->set_shape(const_shape);
                    const_node->set_output_df(ops[0].second->output_df());

                    // Copy over edge attributes from the first op in the group
                    graphlib::Edge edge(
                        dynamic_cast<graphlib::Node*>(const_node)->id(),
                        0, 
                        dynamic_cast<graphlib::Node*>(const_concat)->id(),
                        merge_group_operand_indices[i], 
                        graphlib::EdgeType::kData);
                    graph->add_edge(edge);
                    graph->get_edge_attributes(edge)->set_tms(tms);
                }
            }

            // Update eltwise op
            auto eltwise_op = ops[0].first;
            auto eltwise_name = concat->name() + "_" + eltwise_op->name() + "_" + std::to_string(layer_id);
            auto eltwise = graph->add_node(eltwise_op->clone(eltwise_name), graph->get_subgraph_id_for_node(eltwise_op->id()));
            graph->add_edge(const_concat, eltwise, graphlib::EdgeType::kData);
            graph->add_edge(input_to_fused_block, eltwise, graphlib::EdgeType::kData);
            input_to_fused_block = eltwise;

            for (auto & [op, inp] : ops) 
            {
                if (op != nullptr)
                    graph->remove_node(op);
            }
        }
    }

    // Remove the concat node
    auto concat_users = graph->data_users(concat);
    for (auto user : concat_users)
    {
        graph->add_edge(input_to_fused_block, user, graphlib::EdgeType::kData);
    }
    graph->remove_node(concat);

    return can_fuse;

}


void fuse_per_channel_ops(graphlib::Graph *graph)
{
    bool eliminated_concat = false;
    bool updated = true;
    while (updated)
    {
        updated = false;
        for (auto *node : graphlib::topological_sort(*graph))
        {
            graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(node);
            if (not op or op->op_name() != "concatenate")
                continue;

            if (fuse_per_channel_concat(graph, op))
            {
                updated = true;
                eliminated_concat = true;
                run_consteval_graph_pass(graph);
                break;
            }
        }
    }
    // If concats are eliminated, input constants are joined, thus it's not enough to 
    // call consteval on the node, the entire pass needs to run
    if (eliminated_concat)
    {
        run_consteval_graph_pass(graph);
    }
}
}  // namespace tt::passes
