// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/move_select_after_matmul_optional.hpp"

#include <pybind11/pybind11.h>

#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "utils/logger.hpp"
#include "passes/passes_utils.hpp"
#include "passes/commute_utils.hpp"
#include "reportify/reportify.hpp"

namespace tt::passes
{

static bool is_large_aligned_select(graphlib::Node const *node, int threshold)
{
    graphlib::OpNode const *op = dynamic_cast<graphlib::OpNode const *>(node);
    if (op == nullptr or op->op_name() != "select")
        return false;

    graphlib::OpType const& op_type = op->op_type();
    int length = std::get<int>(op_type.attr[2]);
    return  (length >= threshold); 
}

static bool is_vslice(graphlib::Node const *node)
{
    graphlib::OpNode const *op = dynamic_cast<graphlib::OpNode const *>(node);
    return op and op->op_name() == "vslice";
}

static bool all_users_matmul(std::vector<graphlib::Node *> const &users)
{
    for (auto user : users)
    {
        graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(user);
    if (op->op_name() != "matmul")
        return false;
    }

    return true;
}

static bool is_vslice_select_vstack(graphlib::OpNode *op, graphlib::OpNode *prev_op)
{
    if (op->op_name() == "vstack")
        return true;
    if (prev_op == nullptr)
        return false;
    if (prev_op->op_name() == "vstack" and op->op_name() == "select")
        return true;
    if (prev_op->op_name() == "select" and op->op_name() == "vslice")
        return true;
    return false;
}

static bool shape_match(graphlib::Node const *node, graphlib::Shape const &sh, int concat_dim)
{
    auto node_shape = node->shape();
    if (node_shape.size() != sh.size())
        return false;

    for (size_t i = 0; i < sh.size(); ++i)
    {
        if (i == size_t(concat_dim))
            continue;
 
        if (node_shape[i] != sh[i])
            return false;
    }

    return true;
}

static bool have_consecutive_even_mm_as_src_nodes(graphlib::Graph *graph, graphlib::OpNode *initial_op)
{
    // select op is supposed to have single operand
    graphlib::Node *iter = graph->data_operands(initial_op)[0];
    int src_mm_nodes_count = 0;
    while (true)
    {
        if (iter->node_type() == graphlib::NodeType::kInput)
            break;

        graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(iter);
        TT_ASSERT(op);

        std::vector<graphlib::Node *> operands = graph->data_operands(op);

        // skip if op-type is slice or stack
        auto op_name = op->op_name();
        if (op_name == "hslice" or op_name == "hstack" or op_name == "vslice" or op_name == "vstack" or op_name == "reduce_max")
        {
            iter = operands[0]; 
            continue;
        }
 
        // something not matmul is the source nodes of the select op
        if (op_name != "sparse_matmul" and op_name != "matmul")
            break;

        // consider the case when source node is sparse-matmul or matmul
        src_mm_nodes_count++;
        if (op_name == "sparse_matmul")
            iter = operands[1];
        else
            iter = operands[0];
    }
    return (src_mm_nodes_count % 2 == 0);
}

static std::vector<graphlib::Node *> other_concat_operands_commutable(
    graphlib::Graph *graph, 
    graphlib::Node *concat_op, 
    graphlib::Node *prev_node, 
    std::vector<graphlib::Node *> const &path,
    std::vector<graphlib::Node *> &main_pending,
    int &main_r,
    int &main_c)
{
    std::vector<graphlib::Node *> sub;

    // retrieve the shape of input to original select op (without decompose)
    graphlib::Shape select_input_shape = graph->data_operands(path[0])[0]->shape();
    auto concat_attrs = dynamic_cast<graphlib::OpNode const *>(concat_op)->op_attrs();
    int concat_dim = std::get<int>(concat_attrs[0]);
    if (concat_dim < 0) 
        concat_dim += concat_op->shape().size(); 

    // iterate through the concat's operands 
    std::vector<graphlib::Node *> concat_operands = graph->data_operands(concat_op);
    std::vector<graphlib::Node *> tmp_pending;
    int tmp_r_sum = 0, tmp_c_sum = 0;
    for (auto operand : concat_operands)
    {
        // skip the operand from the same path as moving select op
        if (operand == prev_node)
            continue;

        graphlib::Node * iter = operand; 
        graphlib::OpNode *prev_op = nullptr;
        int tmp_c = 0, tmp_r = 0;
        while (true)
        {
            // check the op's output shape
            if (shape_match(iter, select_input_shape, concat_dim)) 
            {
                sub.push_back(iter);
                break;
            }

            graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(iter);
            if (not op)
                return {};    

            // check #users
            std::vector<graphlib::Node *> op_users = graph->data_users(iter);
            if (op_users.size() > 1)
                return {};

            // cannot commute through matmul
            if (op->op_name() == "matmul")
                return {};

            // check operands
            graphlib::Node *op_operand;
            std::vector<graphlib::Node *> op_operands = graph->data_operands(iter);        
            if (op_operands.size() > 2)
            {
                return {};     
            }
            else if (op_operands.size() == 2)
            {
                graphlib::InputNode *constant_operand = dynamic_cast<graphlib::InputNode *>(op_operands[0]);
                if (not constant_operand or not constant_operand->is_constant())
                    constant_operand = dynamic_cast<graphlib::InputNode *>(op_operands[1]);

                if (constant_operand and constant_operand->is_constant()) // one of them is const node
                    op_operand = (constant_operand == op_operands[0]) ? op_operands[1] : op_operands[0];
                else
                    return {};
            }
            else // # operands == 1
            {
                if (is_elementwise(op) or is_vslice_select_vstack(op, prev_op))
                    op_operand = op_operands[0];
                else if (op->op_name() == "pad_tile")
                {
                    int dim = std::get<int>(op->op_type().attr[0]);
                    int pad = std::get<int>(op->op_type().attr[1]);
                    if (dim < 0)
                        dim += op->shape().size();

                    if (dim < 2 or (dim == 3 and pad == tmp_c) or (dim == 2 and pad == tmp_r))
                    {
                        tmp_c = (dim == 3) ? pad : tmp_c;
                        tmp_r = (dim == 2) ? pad : tmp_r;
                        if (dim >= 2)
                            tmp_pending.push_back(op);
                        op_operand = op_operands[0];
                    }
                    else
                        return {};
                }
                else if (op->op_name() == "narrow")
                {
                    int dim = std::get<int>(op->op_type().attr[0]);
                    int org = std::get<int>(op->op_type().attr[2]);
                    if (dim < 0)
                        dim += op->shape().size();

                    tmp_c = (dim == 3) ? org : tmp_c;
                    tmp_r = (dim == 2) ? org : tmp_r;
                    if (dim >= 2)
                        tmp_pending.push_back(op);
                    op_operand = op_operands[0];
                }
                else
                    return {};
            }

            // move onto next op
            prev_op = op;
            iter = op_operand;
        }

        if ((tmp_c != main_c and concat_dim != 3) or (tmp_r != main_r and concat_dim != 2))
        {
            return {};
        }

        tmp_r_sum += tmp_r;
        tmp_c_sum += tmp_c;
    }

    // tmp_c == main_c and tmp_r == main_r for all operands
    main_pending.insert(main_pending.end(), tmp_pending.begin(), tmp_pending.end());
    main_c += ((concat_dim == 3) ? tmp_c_sum : 0);
    main_r += ((concat_dim == 2) ? tmp_r_sum : 0);

    return sub; // all commutable
}

static void commute_binary_op(graphlib::Graph *graph, graphlib::Node *op)
{
    auto operand_edges = graph->operand_data_edges(op);
    graphlib::Shape shape = op->shape();
    graphlib::Shape input_shape = (graph->data_operands(op)[0]->shape() != shape) ? graph->data_operands(op)[0]->shape() : graph->data_operands(op)[1]->shape(); 

    // Check bcast
    for (auto op_edge : operand_edges)
    {
        auto edge_attrs = graph->get_edge_attributes(op_edge);
        std::vector<graphlib::OpType> &tms = edge_attrs->get_tms();
        for (size_t i = 0; i < tms.size(); ++i)
        {
            if (tms[i].op == "broadcast")
            {
                auto bcast_attr = tms[i].attr;
                int dim = std::get<int>(bcast_attr[0]);
                int factor = std::get<int>(bcast_attr[1]);
                if (dim < 0)
                    dim += shape.size();

                int target_dim = input_shape[dim];
                if (target_dim != factor)
                {
                    bcast_attr[1] = target_dim;
                    tms[i].attr = bcast_attr;
                }
            }
        }
    }
}

static void commute_concat_op(
    graphlib::Graph *graph, 
    std::vector<graphlib::Node *> &path, 
    std::vector<graphlib::Node *> &subpath,
    int concat_index)
{
    graphlib::Node *concat_op = path[concat_index]; 

    // Look for vslice -> select -> vstack sequence if exists 
    for (auto init_op : subpath)
    {
    graphlib::Node *iter = init_op;
        graphlib::OpNode *vslice_op = nullptr, *select_op = nullptr, *vstack_op = nullptr; 
        while (iter != concat_op)
        {
            // Retrieve next op
            auto next = graph->data_users(iter)[0];

            // Check the op-type name if applicable
            graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(iter);
            if (not op)
                iter = next;
            if (op->op_name() == "vslice")
                vslice_op = op;
            else if (op->op_name() == "select" and vslice_op)
                select_op = op;
            else if (op->op_name() == "vstack" and select_op)
                vstack_op = op;
            else
            {
                vslice_op = nullptr;
                select_op = nullptr;
                vstack_op = nullptr;
            }             

            // Hoist another splice-seq on the other concat operand branchs
            if (vslice_op and select_op and vstack_op)
            {
                auto old_edge = graph->user_data_edges(vstack_op)[0];
                auto new_edge = graphlib::Edge(
                      graph->data_operands(vslice_op)[0]->id(),
                      0,
                      old_edge.consumer_node_id,
                      old_edge.consumer_input_port_id,
                      old_edge.edge_type);
                graph->add_edge(new_edge);
                graph->copy_edge_attributes(old_edge, new_edge);
                graph->remove_edge(old_edge);
                graph->remove_node(vslice_op);
                graph->remove_node(select_op);
                graph->remove_node(vstack_op);
                vslice_op = nullptr;
                select_op = nullptr;
                vstack_op = nullptr;
            }
            else
            {
                // binary-commute
                if (graph->data_operands(iter).size() == 2 and op->op_name() != "matmul")
                {
                    commute_binary_op(graph, iter);
                }

                // Re-calculate the op shape
                graphlib::calculate_and_set_node_shape(graph, op);
            } 

            // Move to next op
            iter = next;
        } 
    }
}

static std::pair<std::vector<graphlib::Node *>, std::vector<std::vector<graphlib::Node *>>> find_path_from_select_to_matmul_op(
    graphlib::Graph *graph,
    graphlib::Node *initial_op)
{
    std::vector<graphlib::Node *> path; 
    std::vector<std::vector<graphlib::Node *>> subpath;
    std::vector<graphlib::Node *> pending_padding;

    // path to find: vslice -> select -> vstack -> (...) -> matmul -> (random op)    
    // vslice -> select is fixed and already checked
    path.push_back(initial_op);     // vslice, already checked
    graphlib::Node *next_node = graph->data_users(initial_op)[0];
    path.push_back(next_node);        // select, already checked
 
    // traverse the graph until bumping into matmul op
    graphlib::Node *iter = graph->data_users(next_node)[0];
    int org_shape_r = 0;
    int org_shape_c = 0;
    while (true)
    {
        graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(iter);
        TT_ASSERT(op);

        // op node check
        auto op_name = op->op_name();
        if (op_name == "vstack" and path.size() == 2)
        {
            path.push_back(op);
        }
        else if (op_name == "matmul" or op_name == "sparse_matmul")
        {
            path.push_back(op);
            break;
        }
        else if (op_name == "pad_tile")
        {
            int dim = std::get<int>(op->op_type().attr[0]);
            if (dim < 0)
                dim += op->shape().size();
            int padded_shape = std::get<int>(op->op_type().attr[1]);
            int org_shape = graph->data_operands(op)[0]->shape()[dim];

            if (dim == 3)
            {
                if ((padded_shape == org_shape_c) or (padded_shape == org_shape))
                {
                    org_shape_c = 0;
                    pending_padding.push_back(op);
                }
                else 
                {
                    path.clear();
                    break;
                }
            }
            else if (dim == 2)
            {
                if ((padded_shape == org_shape_r) or (padded_shape == org_shape))
                {
                    org_shape_r = 0;
                    pending_padding.push_back(op);
                }
                else 
                {
                    path.clear();
                    break;
                }        
            }
            else 
            {
                   path.push_back(op); 
            }
        }
        else if (op_name == "narrow")
        {
            int dim = std::get<int>(op->op_type().attr[0]);
            if (dim < 0)
                dim += op->shape().size();

            if (dim == 3)
            {
                org_shape_c = std::get<int>(op->op_type().attr[2]);
                pending_padding.push_back(op);
            }
            else if (dim == 2)
            {
                org_shape_r = std::get<int>(op->op_type().attr[2]);
                pending_padding.push_back(op);
            }
            else 
            {
                path.push_back(op); 
            }
        } 
        else if (op_name == "concatenate")
        {
            path.push_back(op);        
        }
        else 
        {
            path.clear();
            break;
        }

        // Check user nodes
        std::vector<graphlib::Node *> users = graph->data_users(iter);
        if (users.size() > 1)
        {
            if (not all_users_matmul(users))
            {
                path.clear();
                break;
            }
        }
 
        graphlib::OpNode *user_op = dynamic_cast<graphlib::OpNode *>(users[0]); 
        if (user_op and user_op->op_name() == "concatenate")
        {
            std::vector<graphlib::Node *> sub = other_concat_operands_commutable(
                graph,
                users[0],
                iter,
                path,
                pending_padding,
                org_shape_r,
                org_shape_c); /* <concat op> <src node of concat op> <path vector> */
            if (sub.empty())
            {
                path.clear();
                break;
            }
            subpath.push_back(sub);
        }

        // move onto next op
        iter = users[0];
    }

    // org_shape set by narrow ops should be cancelled out by subsequent pad-tile ops
    if (org_shape_r != 0 or org_shape_c != 0)
    {
        path.clear();
    }

    // the path at least should be length = 4: vslice -> select -> vstack -> matmul
    if (path.size() < 4)
    {
        path.clear();
    }

    // Remove all pad/narrow ops inbetween, if those cancel out
    if (path.size() > 0)
    {
        for (graphlib::Node *padding_node : pending_padding)
        {
            //graph->remove_node(padding_node);
            graphlib::bypass_node(graph, padding_node, true);
        }
    }

    return std::make_pair(path, subpath);
}

static void commute(
    graphlib::Graph *graph, 
    std::vector<graphlib::Node *> &path,
    std::vector<std::vector<graphlib::Node *>> &subpath)
{
    graphlib::OpNode *first_vslice = path.front()->as<graphlib::OpNode>();
    graphlib::Node *vslice_input = graph->data_operands(first_vslice)[0];
    graphlib::Shape vslice_input_shape = vslice_input->shape();
    graphlib::OpNode *matmul = path.back()->as<graphlib::OpNode>();
    graphlib::Shape matmul_shape = matmul->shape();
    size_t path_length = path.size();
    size_t subpath_idx = 0;

    // original path is:          (vslice-input op) -> vslice -> select -> vstack -> (...) -> matmul -> (random op)
    // path would be changed to:  (vslice-input op) -> (...) -> matmul -> vslice -> select -> vstack -> (random op)
    // 1. Connect input of vslice op to the first op of (...) -> matmul sequence
    // 2. Update the shape of ops in (...) to vslice-input
    // 3. Update the output shape of matmul op to reflect the op-shape before select op  # (1, 1, 5344, 80) --> (1,1,6144,80)
    // 4. Connect vslice -> select -> vstack sequence to the output of the matmul op, and update the output shape accordingly

    // Re-Connect vslice's input op to the first subseuqent op(s)
    graphlib::OpNode *first_subsequent_op = path[3]->as<graphlib::OpNode>();
    std::vector<graphlib::Edge> to_matmul_edges = graph->user_data_edges(path[path_length-2]);
    bool need_multiple_edges = (to_matmul_edges.size() > 1) and (path_length == 4);
    auto old_edge = graph->operand_data_edges(first_subsequent_op)[0];
    auto new_edge = graphlib::Edge(
            vslice_input->id(),
            0,
            first_subsequent_op->id(),
            (first_subsequent_op->is_matmul()) ? ((matmul->is_sparse_matmul()) ? 1 : 0) : 0,
            old_edge.edge_type);
    graph->add_edge(new_edge);
    graph->copy_edge_attributes(old_edge, new_edge);
    graph->remove_edge(old_edge);

    // Update the shape of ops in (...) sequence
    for (size_t i = 3; i < path_length-1; ++i)
    {
        graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(path[i]);
        if (op and op->op_name() == "concatenate")
        {
            TT_ASSERT(subpath.size() > subpath_idx);
            commute_concat_op(graph, path, subpath[subpath_idx], i);
            subpath_idx++;
        }
        graphlib::calculate_and_set_node_shape(graph, path[i]);
    }

    // Move vslice -> select -> vstack ops after each matmul op
    for (auto to_matmul_edge : to_matmul_edges)
    {
        graphlib::Node *matmul_op = graph->node_by_id(to_matmul_edge.consumer_node_id);

        if (need_multiple_edges and matmul_op != first_subsequent_op)
        {
            auto _new_edge = graphlib::Edge(
                    vslice_input->id(),
                    0,
                    matmul_op->id(),
                    0, //(matmul_op->is_sparse_matmul()) ? 1 : 0,
                    graphlib::EdgeType::kData);
                graph->add_edge(_new_edge);
            graph->remove_edge(to_matmul_edge);
        }

        graphlib::calculate_and_set_node_shape(graph, matmul_op);

        // Iterate through the mamtul user edge
        for (auto matmul_user_edge : graph->user_data_edges(matmul_op))
        {
            // Clone nodes
            graphlib::Node *vslice_clone = graph->add_node(
                path[0]->clone(path[0]->name() + "_splice_commute_clone" + matmul_op->name() + std::to_string(matmul_user_edge.edge_creation_id)),
                graph->get_subgraph_id_for_node(matmul_op->id()));
            graphlib::Node *select_clone = graph->add_node(
                path[1]->clone(path[1]->name() + "_splice_commute_clone" + matmul_op->name() + std::to_string(matmul_user_edge.edge_creation_id)),
                graph->get_subgraph_id_for_node(matmul_op->id()));
            graphlib::Node *vstack_clone = graph->add_node(
                path[2]->clone(path[2]->name() + "_splice_commute_clone" + matmul_op->name() + std::to_string(matmul_user_edge.edge_creation_id)),
                graph->get_subgraph_id_for_node(matmul_op->id()));

            // Connect output of the matmul to vslice
            auto new_edge = graphlib::Edge(
                matmul_op->id(),
                0,
                vslice_clone->id(),
                0,
                matmul_user_edge.edge_type);
            graph->add_edge(new_edge);
            graph->get_edge_attributes(new_edge)->set_ublock_order(graph->get_edge_attributes(matmul_user_edge)->get_ublock_order());
            graphlib::calculate_and_set_node_shape(graph, vslice_clone);

            // Connect middle nodes
            graph->add_edge(vslice_clone, select_clone);
            graphlib::calculate_and_set_node_shape(graph, select_clone);
            graph->add_edge(select_clone, vstack_clone);

            // Conncect output of vstack to the op that was originally connected to the output of matmul op
            new_edge = graphlib::Edge(
                vstack_clone->id(),
                0,
                matmul_user_edge.consumer_node_id,
                matmul_user_edge.consumer_input_port_id,
                matmul_user_edge.edge_type);
            graph->add_edge(new_edge);
            graph->get_edge_attributes(new_edge)->set_ublock_order(graph->get_edge_attributes(matmul_user_edge)->get_ublock_order());
            graph->remove_edge(matmul_user_edge);

            // Recalculate shape
            graphlib::calculate_and_set_node_shape(graph, vstack_clone);
        }
    }

    // Remove redundant edges/nodes
    graph->remove_node(path[0]);
    graph->remove_node(path[1]);
    graph->remove_node(path[2]);
}

// temporarily apply the pass only inside of moving-splice pass (it seems some tests intentionally use identical ops)
// TODO: remove the same pass from buda_pass.cpp for now
static void internal_merge_identical_user_ops(graphlib::Graph *graph) {
    std::unordered_set<graphlib::Node *> removed_nodes;
    for (auto *node : graphlib::topological_sort(*graph)) {
        if (removed_nodes.find(node) != removed_nodes.end())
            continue;

        graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(node);
        if (not op)
            continue;
 
        std::vector<graphlib::Node *> users = graph->data_users(node);
        if (users.size() > 1)
        {
            std::unordered_map<std::string, graphlib::Node *> seen_users;
            for (graphlib::Node *user : users)
            {
                graphlib::OpNode *user_op = dynamic_cast<graphlib::OpNode *>(user);
                if (not user_op)
                    continue;
 
                std::string user_key = user_op->op_type().as_string();
                auto identical_op = seen_users.find(user_key);
                if (identical_op == seen_users.end())
                {
                    seen_users.insert({user_key, user});
                }
                else
                {
                    // identical user node is found, if it's unary, merge it
                    graphlib::Node *identical_op_ptr = seen_users[user_key];
                    if (graph->data_operands(user).size() == 1 and user_op->op_type().op != "buffer")
                    {
                        replace_node(graph, user, identical_op_ptr, true);
                        removed_nodes.insert(user);
                    }
                }
            }
        }
    }
}

void move_select_after_matmul_optional(graphlib::Graph *graph)
{
    if (getenv("PYBUDA_MANUAL_SPLICE_DECOMP_TH") == nullptr)
        return;
    int manual_splice_decomp_th = env_as<int>("PYBUDA_MANUAL_SPLICE_DECOMP_TH");

    // 0. merge indentical ops
    internal_merge_identical_user_ops(graph);

    // Purpose: move a select that was inserted by PAD_SPARSE_MM override to after matmul op for easier streaming later
    // 1. Find a select op that:
    //     (1) has selecting length longer than threshold
    //     (2) has tile-aligned index, length, and stride
    //     (3) has odd number of consecutive matmuls as most recent source nodes
    // 2. Find a pass that consists of (random op) -> vslice -> select -> vstack -> (...) -> matmul -> (random op)
    // 3. Commute the path to be (random op) -> (...) -> matmul -> vslice -> select -> stack -> (random op)

    bool attempt_update = true;
    while (attempt_update)
    {
        // Set to false here because we want to stop looping if no update occurs
        attempt_update = false;
        for (auto *node : graphlib::topological_sort(*graph))
        {
            graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(node);
            if (not op)
                continue;

            if (not is_large_aligned_select(op, manual_splice_decomp_th))
                continue;

            if (have_consecutive_even_mm_as_src_nodes(graph, op))
                continue;

            // select op should have only one operand
            graphlib::Node *operand = graph->data_operands(node)[0];
            if (not is_vslice(operand))
                continue;

            auto [path, subpath] = find_path_from_select_to_matmul_op(graph, operand);
            if (path.empty())
                continue;

            commute(graph, path, subpath);
            attempt_update = true;
            break;
        }
    }

}
}  // namespace tt::passes
