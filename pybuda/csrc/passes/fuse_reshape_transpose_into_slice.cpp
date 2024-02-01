// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/fuse_reshape_transpose_into_slice.hpp"

namespace tt::passes
{

void fuse_reshape_transpose_pairs_into_slice_or_stack_tm(graphlib::Graph *graph)
{
    log_debug(LogGraphCompiler, "Running pass to commute and fuse slice/stack compatible ops");

    bool updated = true;
    while (updated)
    {
        updated = false;
        for (auto *node : graphlib::topological_sort(*graph))
        {
            graphlib::OpNode *reference_op = dynamic_cast<graphlib::OpNode *>(node);
            if (not reference_op)
                continue;

            if ((reference_op->op_name() != "reshape") and (reference_op->op_name() != "transpose"))
                continue;

            StackSliceOpType ref_op_type = StackSliceOpType::None;
            if (reference_op->op_name() == "reshape")
                ref_op_type = StackSliceOpType::HSlice;

            if (reference_op->op_name() == "transpose")
                ref_op_type = StackSliceOpType::HStack;

            // Find valid horizontal candidates:
            // - reshape + transpose => hslice
            // - transpose + reshape => hstack
            graphlib::OpNode *candidate = find_valid_candidate(ref_op_type, graph, reference_op);
            if (not candidate)
            {
                log_debug(LogGraphCompiler, "Didn't find valid candidates for node: {}", reference_op->name());

                // Checking if reference reshapes are valid vertical candidates:
                // - reshape => vslice
                // - reshape => vstack
                if (reference_op->op_name() == "reshape")
                {
                    convert_reshape_into_vslice_or_vstack_if_possible(graph, reference_op);
                }
                continue;
            }

            // Check whether we can commute through all forks of the suitable pair
            if (not valid_commute_through_forks(graph, reference_op, candidate))
            {
                log_debug(
                    LogGraphCompiler,
                    "No valid commute path for nodes: {} -> {}",
                    reference_op->name(),
                    candidate->name());
                continue;
            }

            // Commute transpose through all forks up to the reshape + cancel out transposes
            commute_through_forks(ref_op_type, graph, reference_op, candidate);

            // Fuse reshape + transpose into hslice
            fuse_into_slice_or_stack(ref_op_type, graph, reference_op);

            updated = true;
            break;
        }
    }
}

graphlib::OpNode *find_valid_candidate(
    StackSliceOpType ref_op_type, graphlib::Graph *graph, graphlib::OpNode *initial_op)
{
    if (ref_op_type == StackSliceOpType::HSlice)
    {
        return find_valid_candidate_for_hslice(graph, initial_op);
    }
    else if (ref_op_type == StackSliceOpType::HStack)
    {
        return find_valid_candidate_for_hstack(graph, initial_op);
    }
    else
    {
        log_debug(LogGraphCompiler, "Invalid PyBuda Stack/Slice type: {}", ref_op_type);
        return nullptr;
    }
}

graphlib::OpNode *find_valid_candidate_for_hslice(
    graphlib::Graph *graph, graphlib::OpNode *initial_op, graphlib::OpNode *reference_op)
{
    if ((reference_op) and (reference_op->op_name() != "transpose") and (not can_commute_past_operand(reference_op)))
        return nullptr;

    // First call only
    if (not reference_op)
        reference_op = initial_op;

    // Return op if it's valid and compatible
    if ((reference_op->op_name() == "transpose") and is_hslice_compatible(graph, initial_op, reference_op))
        return reference_op;

    // Traverse forked edges
    std::vector<graphlib::Node *> user_nodes = graph->data_users(reference_op);
    for (std::size_t i = 0; i < user_nodes.size(); i++)
    {
        graphlib::OpNode *user_node = dynamic_cast<graphlib::OpNode *>(user_nodes[i]);
        if (not user_node)
            return nullptr;

        // Recursively traverse in order to find valid candidate
        graphlib::OpNode *valid_candidate = find_valid_candidate_for_hslice(graph, initial_op, user_node);
        if (valid_candidate != nullptr)
        {
            return valid_candidate;
        }
        else
        {
            return nullptr;
        }
    }

    return nullptr;
}

bool is_hslice_compatible(graphlib::Graph *graph, graphlib::OpNode *a, graphlib::OpNode *b)
{
    // Only reshape and transpose ops are supported
    if ((a->op_name() != "reshape") or (b->op_name() != "transpose"))
        return false;

    // Supporting only 3D and 4D tensors
    int reshape_shape_size = a->shape().size();
    if (reshape_shape_size != 3 and reshape_shape_size != 4)
        return false;

    // Canonical shape of reshape operand
    std::vector<graphlib::Node *> reshape_operands = graph->data_operands(a);
    if (reshape_operands.size() != 1)
        return false;
    std::vector<std::uint32_t> reshape_operand_shape = reshape_operands[0]->shape().canonical().as_vector();

    // Canonical shape of reshape
    auto reshape_shape = a->shape().canonical().as_vector();

    // Canonical shape of transpose
    auto transpose_shape = b->shape().canonical().as_vector();

    // Valid transpose attributes (T & R dim). Using +4 for negative indexing as this pass
    // works on canonical shapes
    int b_dim0 = b->op_type().get_attr_as<int>("dim0");
    int b_dim1 = b->op_type().get_attr_as<int>("dim1");
    if (b_dim0 < 0)
        b_dim0 += 4;
    if (b_dim1 < 0)
        b_dim1 += 4;
    if (not((b_dim0 == 1 or b_dim0 == 2) and (b_dim1 == 1 or b_dim1 == 2)))
        return false;

    // Ignoring batched inputs for now
    if (reshape_shape[0] != 1)
        return false;

    // Check if column is tile-dim aligned. As hslice is TM op, C and R
    // should be tile dim aligned during lowering. Therefore, it's ether skip
    // or pad to the tile-dim aligned shape.
    if (transpose_shape[3] % graphlib::Shape::BUDA_TILE_DIM != 0)
        return false;

    // Valid hslice shape
    int hslice_factor = reshape_operand_shape[3] / transpose_shape[3];
    int hslice_z_dim_size = reshape_operand_shape[1] * hslice_factor;
    if (hslice_factor == 1 or hslice_z_dim_size != (int)transpose_shape[1])
        return false;

    if ((reshape_shape[1] == transpose_shape[2]) and (reshape_shape[2] == transpose_shape[1]))
        return true;

    return false;
}

graphlib::OpNode *find_valid_candidate_for_hstack(
    graphlib::Graph *graph, graphlib::OpNode *initial_op, graphlib::OpNode *reference_op)
{
    if ((reference_op) and (reference_op->op_name() != "reshape") and (not can_commute_past_operand(reference_op)))
        return nullptr;

    // First call only
    if (not reference_op)
        reference_op = initial_op;

    // Return op if it's valid and compatible
    if ((reference_op->op_name() == "reshape") and is_hstack_compatible(initial_op, reference_op))
        return reference_op;

    // Traverse forked edges
    std::vector<graphlib::Node *> user_nodes = graph->data_users(reference_op);
    for (std::size_t i = 0; i < user_nodes.size(); i++)
    {
        graphlib::OpNode *user_node = dynamic_cast<graphlib::OpNode *>(user_nodes[i]);
        if (not user_node)
            return nullptr;

        // Recursively traverse in order to find valid candidate
        graphlib::OpNode *valid_candidate = find_valid_candidate_for_hstack(graph, initial_op, user_node);
        if (valid_candidate != nullptr)
        {
            return valid_candidate;
        }
        else
        {
            return nullptr;
        }
    }

    return nullptr;
}

bool is_hstack_compatible(graphlib::OpNode *a, graphlib::OpNode *b)
{
    // Only reshape and transpose ops are supported
    if ((a->op_name() != "transpose") or (b->op_name() != "reshape"))
        return false;

    // Supporting only 3D and 4D tensors
    int reshape_shape_size = b->shape().size();
    if (reshape_shape_size != 3 and reshape_shape_size != 4)
        return false;

    // Canonical shape of transpose
    auto transpose_shape = a->shape().canonical().as_vector();

    // Canonical shape of reshape
    auto reshape_shape = b->shape().canonical().as_vector();

    // Valid transpose attributes (T & R dim). Using +4 for negative indexing as this pass
    // works on canonical shapes
    int a_dim0 = a->op_type().get_attr_as<int>("dim0");
    int a_dim1 = a->op_type().get_attr_as<int>("dim1");
    if (a_dim0 < 0)
        a_dim0 += 4;
    if (a_dim1 < 0)
        a_dim1 += 4;
    if (not((a_dim0 == 1 or a_dim0 == 2) and (a_dim1 == 1 or a_dim1 == 2)))
        return false;

    // Ignoring batched inputs for now
    if (transpose_shape[0] != 1)
        return false;

    // Check if column is tile-dim aligned. As hslice is TM op, C and R
    // should be tile dim aligned during lowering. Therefore, it's ether skip
    // or pad to the tile-dim aligned shape.
    if (transpose_shape[3] % graphlib::Shape::BUDA_TILE_DIM != 0)
        return false;

    if (reshape_shape[3] == transpose_shape[2] * transpose_shape[3])
        return true;

    return false;
}

bool valid_commute_through_forks(
    graphlib::Graph *graph, graphlib::OpNode *firstOp, graphlib::OpNode *lastOp, graphlib::OpNode *commuteOp)
{
    if (firstOp == commuteOp)
        return true;

    if (commuteOp)
        if (not can_commute_past_operand(commuteOp))
            return false;

    // Get latest node data operands
    if (not commuteOp)
        commuteOp = lastOp;
    std::vector<graphlib::Node *> operand_nodes = graph->data_operands(commuteOp);

    // Traverse operand edges from latest commutable op
    bool valid_commute = true;
    for (std::size_t i = 0; i < operand_nodes.size(); i++)
    {
        graphlib::OpNode *operand_node = dynamic_cast<graphlib::OpNode *>(operand_nodes[i]);
        if (not operand_node)
            continue;

        // Recursively traverse through all paths
        valid_commute &= valid_commute_through_forks(graph, firstOp, lastOp, operand_node);
    }

    return valid_commute;
}

bool can_commute_past_operand(graphlib::OpNode *op)
{
    // Element-wise ops are allowed as they are not tackling with Z dim
    py::object eval_module = py::module_::import("pybuda.op.eval.pybuda");
    py::function is_eltwise_fun = eval_module.attr("is_eltwise");
    bool can_commute = is_eltwise_fun(op->op_type()).cast<bool>();

    // Reduce sum is allowed if not reducing over Z dim (only R and C)
    if (op->op_name() == "reduce_sum")
    {
        // Get dimension over which to do reduce
        TT_ASSERT(op->op_attrs().size() == 1);
        int dim_to_reduce_over = std::get<int>(op->op_attrs()[0]);

        // Convert to positive indexing
        if (dim_to_reduce_over < 0)
            dim_to_reduce_over += op->shape().size();

        can_commute = (dim_to_reduce_over > 1);
    }

    return can_commute;
}

void commute_through_forks(
    StackSliceOpType ref_op_type, graphlib::Graph *graph, graphlib::OpNode *firstOp, graphlib::Node *lastOp)
{
    if (ref_op_type == StackSliceOpType::HSlice)
    {
        commute_through_hslice_forks(graph, firstOp, lastOp);
    }
    else if (ref_op_type == StackSliceOpType::HStack)
    {
        commute_through_hstack_forks(graph, firstOp, lastOp);
    }
    else
    {
        log_debug(LogGraphCompiler, "Invalid PyBuda Stack/Slice type: {}", ref_op_type);
    }
}

void commute_through_hslice_forks(graphlib::Graph *graph, graphlib::OpNode *firstOp, graphlib::Node *lastOp)
{
    // Commute over all operand nodes
    std::vector<graphlib::Node *> operand_nodes = graph->data_operands(lastOp);
    for (graphlib::Node *operand_node : operand_nodes)
    {
        if (firstOp->id() == operand_node->id())
            break;

        if ((operand_node->shape() == lastOp->shape()) and
            (dynamic_cast<graphlib::OpNode *>(operand_node)->op_type().op == "transpose"))
        {
            bypass_node(graph, operand_node, true);
            break;
        }

        // Update shape of commutable operands along the path
        update_shape_during_commute(graph, operand_node, lastOp);

        // Fetch operand node operand edges and commute
        std::vector<graphlib::Edge> operand_edges = graph->operand_data_edges(operand_node);
        for (graphlib::Edge operand_edge : operand_edges)
        {
            convert_implicit_to_explicit_bcasts(graph, operand_edge);
            auto name = lastOp->name() + "_operand_commute_clone" + std::to_string(operand_edge.edge_creation_id);
            graphlib::Node *clone_node = graph->add_node(lastOp->clone(name), graph->get_subgraph_id_for_node(lastOp->id()));
            insert_node_on_edge(graph, operand_edge, clone_node);
            handle_change_rank(graph, clone_node);

            commute_through_hslice_forks(graph, firstOp, clone_node);
        }

        auto change_rank = [graph](graphlib::Edge new_edge, graphlib::Edge) { handle_change_rank(graph, new_edge); };
        bypass_node(graph, lastOp, true, change_rank);
    }
}

void commute_through_hstack_forks(graphlib::Graph *graph, graphlib::OpNode *firstOp, graphlib::Node *lastOp)
{
    // Commute over all operand nodes
    std::vector<graphlib::Node *> operand_nodes = graph->data_operands(lastOp);
    for (graphlib::Node *operand_node : operand_nodes)
    {
        if (firstOp->id() == operand_node->id())
            break;

        if ((operand_node->shape() == lastOp->shape()) and
            (dynamic_cast<graphlib::OpNode *>(operand_node)->op_type().op == "reshape"))
        {
            bypass_node(graph, operand_node, true);
            break;
        }

        // Update shape of commutable operands along the path
        update_shape_during_commute(graph, operand_node, lastOp);

        // Fetch operand node operand edges and commute
        std::vector<graphlib::Edge> operand_edges = graph->operand_data_edges(operand_node);
        for (graphlib::Edge operand_edge : operand_edges)
        {
            convert_implicit_to_explicit_bcasts(graph, operand_edge);
            auto name = lastOp->name() + "_operand_commute_clone" + std::to_string(operand_edge.edge_creation_id);
            graphlib::Node *clone_node = graph->add_node(lastOp->clone(name), graph->get_subgraph_id_for_node(lastOp->id()));
            insert_node_on_edge(graph, operand_edge, clone_node);
            handle_change_rank(graph, clone_node);

            commute_through_hstack_forks(graph, firstOp, clone_node);
        }

        auto change_rank = [graph](graphlib::Edge new_edge, graphlib::Edge) { handle_change_rank(graph, new_edge); };
        bypass_node(graph, lastOp, true, change_rank);
    }
}

void update_shape_during_commute(graphlib::Graph *graph, graphlib::Node *operand_node, graphlib::Node *lastOp)
{
    graphlib::Shape commute_shape = lastOp->shape();

    // Handle operands
    if (graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(operand_node))
    {
        // Use Python API to check if op is element-wise
        py::object eval_module = py::module_::import("pybuda.op.eval.pybuda");
        py::function is_eltwise_fun = eval_module.attr("is_eltwise");
        bool is_eltwise = is_eltwise_fun(op->op_type()).cast<bool>();
        // Handle element-wise operands
        if (is_eltwise)
            operand_node->set_shape(commute_shape);
        // Handle special case operands
        else
        {
            // Calculate and set reduce_sum shape (validated in pass before commute)
            if (op->op_type().op == "reduce_sum")
            {
                int dim_to_reduce_over = std::get<int>(op->op_attrs()[0]);
                if (dim_to_reduce_over < 0)
                {
                    dim_to_reduce_over += op->shape().size();
                }
                commute_shape[dim_to_reduce_over] = 1;
                operand_node->set_shape(commute_shape);
            }
        }
        op->add_golden_transform(dynamic_cast<graphlib::OpNode *>(lastOp)->op_type());
        // Handle constants
    }
    else if (dynamic_cast<graphlib::ConstantInputNode *>(operand_node))
    {
        // Set 4d dim for constant for proper TM recalculation
        // operand_node->set_shape(op->shape().canonical());

        // Clear TMs from constant user edges so they can be recalculated afterwards
        for (graphlib::Edge user_edge : graph->user_data_edges(operand_node))
        {
            // Clear TMs as they'll be recalculated again
            graph->get_edge_attributes(user_edge)->get_tms().clear();

            // Handle for hslice as transpose is last op
            graphlib::OpNode *lastOpNode = dynamic_cast<graphlib::OpNode *>(lastOp);
            if (lastOpNode->op_type().op == "transpose")
            {
                // Properly apply transpose on constants
                auto name = lastOp->name() + "_constant_commute_clone" + std::to_string(user_edge.edge_creation_id);
                graphlib::Node *clone_node = graph->add_node(lastOp->clone(name), graph->get_subgraph_id_for_node(lastOp->id()));
                insert_node_on_edge(graph, user_edge, clone_node);
                calculate_and_set_node_shape(graph, clone_node);
                handle_change_rank(graph, clone_node);
                try_consteval_op(graph, clone_node);
            }
            // Handle constant change rank as until this points most of the constants are in 2D shape
            handle_change_rank(graph, user_edge);
        }
        // Handle other
    }
    else
        return;
}

void fuse_into_slice_or_stack(StackSliceOpType ref_op_type, graphlib::Graph *graph, graphlib::OpNode *firstOp)
{
    if (ref_op_type == StackSliceOpType::HSlice)
    {
        fuse_into_hslice(graph, firstOp);
    }
    else if (ref_op_type == StackSliceOpType::HStack)
    {
        fuse_into_hstack(graph, firstOp);
    }
    else
    {
        log_debug(LogGraphCompiler, "Invalid PyBuda Stack/Slice type: {}", ref_op_type);
    }
}

void fuse_into_hslice(graphlib::Graph *graph, graphlib::OpNode *firstOp)
{
    std::vector<graphlib::Node *> user_nodes = graph->data_users(firstOp);
    std::string user_node_name = user_nodes[0]->name();
    user_node_name = user_node_name.substr(0, user_node_name.find("_operand_commute_clone"));

    // Get transpose dimensions of interest
    graphlib::OpNode *transposeNode = dynamic_cast<graphlib::OpNode *>(user_nodes[0]);

    // Canonical shape of transpose
    auto transpose_shape = transposeNode->shape().canonical().as_vector();

    // Get number of slices (T dim as constrain for valid hslice is for T dim to be 1 at start)
    int hslice_num_slices = transpose_shape[1];

    // Bypass connected transposes
    for (graphlib::Node *user_node : user_nodes) bypass_node(graph, user_node, true);

    // Replace reshape with proper hslice
    std::string name = firstOp->name() + "_" + user_node_name + "_fused_into_hslice";
    graphlib::OpType op_type("hslice", {hslice_num_slices});
    auto hslice_node = graph->add_node(
        std::make_unique<graphlib::PyOpNode>(name, op_type), graph->get_subgraph_id_for_node(firstOp->id()));
    replace_node(graph, firstOp, hslice_node, false);
    calculate_and_set_node_shape(graph, hslice_node);
}

void fuse_into_hstack(graphlib::Graph *graph, graphlib::OpNode *firstOp)
{
    std::vector<graphlib::Node *> user_nodes = graph->data_users(firstOp);
    std::string user_node_name = user_nodes[0]->name();
    user_node_name = user_node_name.substr(0, user_node_name.find("_operand_commute_clone"));

    // Get relevant reshape shape
    graphlib::OpNode *reshapeNode = dynamic_cast<graphlib::OpNode *>(user_nodes[0]);

    // Canonical shape of reshape
    auto reshape_shape = reshapeNode->shape().canonical().as_vector();

    // Determine hstack num slices
    int hstack_num_slices = static_cast<int>(reshape_shape[3] / firstOp->shape()[-1]);

    // Bypass connected transposes
    for (graphlib::Node *user_node : user_nodes) bypass_node(graph, user_node, true);

    // Replace reshape with proper hstack
    std::string name = firstOp->name() + "_" + user_node_name + "_fused_into_hstack";
    graphlib::OpType op_type("hstack", {hstack_num_slices});
    auto hstack_node = graph->add_node(
        std::make_unique<graphlib::PyOpNode>(name, op_type), graph->get_subgraph_id_for_node(firstOp->id()));
    replace_node(graph, firstOp, hstack_node, false);
    calculate_and_set_node_shape(graph, hstack_node);
}

void convert_reshape_into_vslice_or_vstack_if_possible(graphlib::Graph *graph, graphlib::OpNode *reference_op)
{
    // Supporting only 3D and 4D tensors
    int reshape_shape_size = reference_op->shape().size();
    if (reshape_shape_size != 3 and reshape_shape_size != 4)
        return;

    // Canonical shape of reshape
    auto reshape_shape = reference_op->shape().canonical().as_vector();

    // Canonical shape of reshape operand
    std::vector<graphlib::Edge> operand_edges = graph->operand_data_edges(reference_op);
    if (operand_edges.size() != 1)
        return;
    auto operand_shape = graphlib::post_tms_shape(graph, operand_edges[0]).canonical().as_vector();

    // Ignoring batched inputs for now
    if ((reshape_shape[0] != 1) or (operand_shape[0] != 1))
        return;

    // Last dim shouldn't change
    if (reshape_shape[3] != operand_shape[3])
        return;

    // For both V slice and stack, R dim should be multiply of 32. As stack/slice
    // op need tile-dim granularity, pre-requirement is tile-dim divisibility must
    // match in order for certain reshapes to be decomposed into V stack/slice.
    if ((operand_shape[2] % graphlib::Shape::BUDA_TILE_DIM != 0) or
        (reshape_shape[2] % graphlib::Shape::BUDA_TILE_DIM != 0))
        return;

    // If reshape is a V slice
    if ((reshape_shape[1] > operand_shape[1]) and
        (reshape_shape[1] == (operand_shape[1] * (operand_shape[2] / reshape_shape[2]))))
        convert_reshape_into_vslice(graph, reference_op, reshape_shape, operand_shape);
    // If reshape is a V stack
    else if (
        (operand_shape[1] > reshape_shape[1]) and
        (reshape_shape[2] == (operand_shape[2] * (reshape_shape[2] / operand_shape[2]))))
        convert_reshape_into_vstack(graph, reference_op, reshape_shape, operand_shape);
    else
        return;
}

void convert_reshape_into_vslice(
    graphlib::Graph *graph,
    graphlib::OpNode *reference_op,
    std::vector<uint32_t> reshape_shape,
    std::vector<uint32_t> operand_shape)
{
    // Valid slice
    int vslice_num_slices = static_cast<int>(operand_shape[2] / reshape_shape[2]);
    if ((vslice_num_slices <= 1) or (reshape_shape[2] != (operand_shape[2] / vslice_num_slices)))
        return;

    // Replace reshape with vslice
    std::string name = reference_op->name() + "_fused_vslice";
    graphlib::OpType op_type("vslice", {vslice_num_slices});
    auto vslice_node = graph->add_node(
        std::make_unique<graphlib::PyOpNode>(name, op_type), graph->get_subgraph_id_for_node(reference_op->id()));
    replace_node(graph, reference_op, vslice_node, false);
    calculate_and_set_node_shape(graph, vslice_node);
    vslice_node->set_shape(vslice_node->shape().as_rank(reshape_shape.size()));
    handle_change_rank(graph, vslice_node);
}

void convert_reshape_into_vstack(
    graphlib::Graph *graph,
    graphlib::OpNode *reference_op,
    std::vector<uint32_t> reshape_shape,
    std::vector<uint32_t> operand_shape)
{
    // Valid slice
    int vstack_num_slices = static_cast<int>(reshape_shape[2] / operand_shape[2]);
    if ((vstack_num_slices <= 1) or (reshape_shape[2] != (operand_shape[2] * vstack_num_slices)))
        return;

    // Replace reshape with vstack
    std::string name = reference_op->name() + "_fused_vstack";
    graphlib::OpType op_type("vstack", {vstack_num_slices});
    auto vstack_node = graph->add_node(
        std::make_unique<graphlib::PyOpNode>(name, op_type), graph->get_subgraph_id_for_node(reference_op->id()));
    replace_node(graph, reference_op, vstack_node, false);
    calculate_and_set_node_shape(graph, vstack_node);
    vstack_node->set_shape(vstack_node->shape().as_rank(reshape_shape.size()));
    handle_change_rank(graph, vstack_node);
}

}  // namespace tt::passes
