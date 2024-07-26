// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/commute_utils.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "passes/passes_utils.hpp"


namespace tt::passes {

int volume_above(std::vector<std::uint32_t> shape, int dim)
{
    int volume = 1;
    for (int i = 0; i < dim; i++)
        volume *= shape[i];

    return volume;
}

int volume_below(std::vector<std::uint32_t> shape, int dim)
{
    int volume = 1;
    for (size_t i = dim; i < shape.size(); i++)
        volume *= shape[i];

    return volume;
}

std::tuple<bool, int> can_commute_reshape_through_dim(graphlib::Shape input_shape, graphlib::Shape output_shape, int dim, bool commute_up) 
{
    bool can_commute = false;
    int new_dim = -1;

    auto input_shape_vec = input_shape.as_vector();
    auto output_shape_vec = output_shape.as_vector();

    if (commute_up)
    {
        auto temp = input_shape_vec;
        input_shape_vec = output_shape_vec;
        output_shape_vec = temp;
    }
    for (size_t i = 0; i < input_shape_vec.size(); i++)
    {
        if (input_shape_vec[i] == output_shape_vec[dim])
        {
            // check whether volume above and below matching dim is the same
            if ((volume_above(input_shape_vec, i) == volume_above(output_shape_vec, dim)) and
                (volume_below(input_shape_vec, i) == volume_below(output_shape_vec, dim)))
                {
                    can_commute = true;
                    new_dim = i;
                    break;
                }
        }
    }
    return std::make_tuple(can_commute, new_dim);
}

std::tuple<bool, int> can_commute_through_dim(graphlib::OpNode *initial_op, graphlib::Graph *graph, int dim, bool commute_up)
{
    bool can_reduce = false;
    int new_dim = -1;
    if (initial_op->op_name() == "reshape")
    {
        auto input_shape = graph->data_operands(initial_op)[0]->shape();
        auto output_shape = initial_op->shape();
        auto result = can_commute_reshape_through_dim(input_shape, output_shape, dim, commute_up);
        can_reduce = std::get<0>(result);
        new_dim = std::get<1>(result);
    }
    else if (initial_op->op_name() == "transpose")
    {
        can_reduce = true;
        int dim0 = initial_op->op_type().get_attr_as<int>("dim0");
        if (dim0 < 0)
        {
            dim0 += initial_op->shape().size();
        }
        int dim1 = initial_op->op_type().get_attr_as<int>("dim1");
        if (dim1 < 0)
        {
            dim1 += initial_op->shape().size();
        }

        if (dim == dim0)
            new_dim = dim1;
        else if (dim == dim1)
            new_dim = dim0;
        else
            new_dim = dim;
    }
    return std::make_tuple(can_reduce, new_dim);
}

bool match_unsqueeze(graphlib::OpType const &a, graphlib::OpType const &b) { 
    bool fns_match = a.op == "unsqueeze" and b.op == "squeeze"; 

    if (not fns_match)
        return false;

    return std::get<int>(a.attr[0]) == std::get<int>(b.attr[0]);
}

bool match_squeeze(graphlib::OpType const &a, graphlib::OpType const &b) { 
    bool fns_match = a.op == "squeeze" and b.op == "unsqueeze";

    if (not fns_match)
        return false;

    return std::get<int>(a.attr[0]) == std::get<int>(b.attr[0]);
}

bool match_reshape(graphlib::OpType const &a, graphlib::OpType const &) { return a.op == "reshape"; }

bool match_transpose(graphlib::OpType const &a, graphlib::OpType const &b)
{
    if (a.op != "transpose")
        return false;

    int a_dim0 = a.get_attr_as<int>("dim0");
    int a_dim1 = a.get_attr_as<int>("dim1");
    if (a_dim0 > a_dim1)
        std::swap(a_dim0, a_dim1);

    int b_dim0 = b.get_attr_as<int>("dim0");
    int b_dim1 = b.get_attr_as<int>("dim1");
    if (b_dim0 > b_dim1)
        std::swap(b_dim0, b_dim1);

    return (a_dim0 == b_dim0) and (a_dim1 == b_dim1);
}

size_t total_broadcast_volume(graphlib::Graph *graph, graphlib::Edge edge)
{
    auto tms = graph->get_edge_attributes(edge)->get_tms();
    size_t volume = 1;
    for (graphlib::OpType &op_type : tms)
    {
        if (op_type.op == "broadcast")
        {
            volume *= std::get<int>(op_type.attr[1]);
        }
    }
    return volume;
}

std::pair<bool, int> are_inverse_with_broadcast(graphlib::Shape shape_a, graphlib::Shape shape_b, size_t broadcast_volume)
{
    bool are_inverse_with_broadcast = true;
    if (shape_a.size() != shape_b.size() or shape_a == shape_b)
    {
        return std::make_pair(false, -1);
    }
    int broadcast_dim = -1;
    for (size_t i = 0; i < shape_b.size(); i++)
    {
        if ((shape_b[i] != shape_a[i]) and
            ((shape_b[i] != broadcast_volume) or (shape_a[i] != 1)))
            are_inverse_with_broadcast = false;
        else if ((shape_b[i] == broadcast_volume) and (shape_a[i] == 1))
        {
            if (broadcast_dim != -1)
                are_inverse_with_broadcast = false;
            broadcast_dim = i;
        }
    }
    return std::make_pair(are_inverse_with_broadcast, broadcast_dim);
}

graphlib::Shape shape_of_only_operand(graphlib::Graph *graph, graphlib::OpNode *op)
{
        std::vector<graphlib::Node *> operands = graph->data_operands(op);
        TT_ASSERT(operands.size() == 1);
        graphlib::Node *operand = operands[0];
        return op->shape_of_operand(graph, operand);
}

bool are_compatible_ops(graphlib::Graph *graph, graphlib::OpNode *a, graphlib::OpNode *b, graphlib::Shape *updated_shape, bool check_inverse)
{
    py::object eval_module = py::module_::import("pybuda.op.eval.pybuda");
    py::function is_tm = eval_module.attr("is_tm");

    if (a == b)
        return (not check_inverse);

    std::vector<graphlib::Node *> operands = graph->data_operands(a);
    if (operands.size() != 1)
        return false;

    graphlib::Node *operand = operands[0];
    auto operand_shape = a->shape_of_operand(graph, operand);
    if (updated_shape)
    {
        operand_shape = *updated_shape;
    }

    // Inverse tms have to be the same op, except for unsqueeze/squeeze case
    bool are_compatible_tms = 
        is_tm(a->op_type()).cast<bool>() and 
        ((a->op_name() == b->op_name()) or
        ((a->op_name() == "unsqueeze" and b->op_name() == "squeeze") or (a->op_name() == "squeeze" and b->op_name() == "unsqueeze")));

    if (not are_compatible_tms)
        return false;

    auto shape_to_check_on_b = check_inverse ? b->shape() : shape_of_only_operand(graph, b);
    bool is_inverse = are_compatible_tms & (operand_shape == shape_to_check_on_b);
    auto operand_edges = graph->operand_data_edges(b);
    is_inverse |= are_inverse_with_broadcast(operand_shape, shape_to_check_on_b, total_broadcast_volume(graph, operand_edges[0])).first;
    is_inverse &= not b->as<graphlib::TaggedNode>()->tag_value_or("dont_erase", false);
    if (not is_inverse)
        return false;

    for (auto [name, match_fn] : match_fns)
    {
        if (match_fn(a->op_type(), b->op_type()))
            return true;
    }

    return false;
}

bool commute_through_select(
    graphlib::Graph* graph, 
    graphlib::OpNode* op, 
    graphlib::OpNode* initial_op, 
    graphlib::Node* producer,
    graphlib::Shape *commute_shape, 
    graphlib::Shape *clone_shape, 
    bool check_only,
    bool *retain_operand_dim,
    std::pair<int, int> *operand_dims,
    graphlib::OpType *golden_transform,
    bool commute_up) 
{    
    int select_dim = std::get<int>(op->op_attrs()[0]);

    // Convert to positive indexing
    if (select_dim < 0)
        select_dim += op->shape().size();

    // setting commute_up here doesnt work when commuting through concat/select. 
    // This is because we actually want different behaviour from can_commute_through_dim
    // depending on whether we are trying to insert clones above the concat OR if we
    // are trying to commute down, but need to commute back up when checkingthat all
    // producer forks have an equivalent to the initial_op.
    // TODO: Should just remove that logic entirely from can_commute_through_dim

    // Find matching dim in commute and clone shape
    bool select_commute_up = clone_shape ? commute_up : false;
    int matching_in_commute_shape = -1;
    int matching_in_clone_shape = -1;
    int matching_in_op_shape = -1;

    for (int i = (int)commute_shape->size() - 1; i >= 0; i--)
    {
        if (clone_shape) {
            for (int j = (int)clone_shape->size() - 1; j >= 0; j--)
            {
                if ((*commute_shape)[i] == (*clone_shape)[j] and (((*commute_shape)[i] == 1 and matching_in_commute_shape == -1) or (*commute_shape)[i] != 1))
                {
                    matching_in_commute_shape = i;
                    matching_in_clone_shape = j;
                    break;
                }
            }
        }

        for (int j = (int)op->shape().size() - 1; j >= 0; j--)
        {
            if ((*commute_shape)[i] == op->shape()[j] and (((*commute_shape)[i] == 1 and matching_in_commute_shape == -1) or (*commute_shape)[i] != 1))
            {
                matching_in_commute_shape = i;
                matching_in_op_shape = j;
                break;
            }
        }
    }


    if (matching_in_commute_shape == -1) {
        TT_ASSERT(check_only, "Commute is impossible, must just be checking if commute is possible");
        return false;
    }
    if (clone_shape) {
        if (matching_in_commute_shape == select_dim)
            select_commute_up = true;
    }
    auto [dim_unaffected_by_commute, new_dim] = can_commute_through_dim(initial_op, graph, select_dim, select_commute_up);
    bool can_commute = dim_unaffected_by_commute;
    // dim_involved_in_commute is whether the select dim is unchanged by the commute
    // i.e. Unaffected: reshape (1, 1, 128, 64) to (1, 128, 8, 8) -> select(1, 0, 32, 128) -> (1, 32, 8, 8)
    // i.e. Affected:   reshape (1, 1, 32, 1024) to (1, 32, 32, 32) ->select(-1, 0, 16, 32) -> (1, 32, 32, 16)
    //      In the Affected case, the commute shape should become (1, 1, 32, 512) after the select op. 
    //      The clone shape should be just (1, 32, 32, 16). The op attrs do not need to change.

    if (not dim_unaffected_by_commute)
    {    
        if ((clone_shape and (matching_in_clone_shape - (int)clone_shape->size()) < (matching_in_commute_shape - (int)commute_shape->size())) 
        or ((not clone_shape) and (matching_in_op_shape - (int)op->shape().size()) < (matching_in_commute_shape - (int)commute_shape->size())))
        {
            // Functionality not implemented for commute up yet
            if (not commute_up and select_dim == (int)(op->shape().size() - 1))
            {
                graphlib::Shape updated_commute_shape = *commute_shape;
                updated_commute_shape[commute_shape->size() - 1] = op->shape().as_vector()[select_dim] * op->shape()[select_dim - 1];
                *commute_shape = updated_commute_shape;

                if (clone_shape)
                {
                    graphlib::Shape updated_clone_shape = *clone_shape;
                    updated_clone_shape = op->shape();
                    *clone_shape = updated_clone_shape;
                }
                
                new_dim = select_dim;
                can_commute = true;
            }
            if (commute_up and select_dim == (int)(op->shape().size() - 1)) 
            {
                graphlib::Shape updated_commute_shape = *commute_shape;
                updated_commute_shape[commute_shape->size() - 1] = producer->shape().as_vector()[select_dim] * producer->shape()[select_dim - 1];
                *commute_shape = updated_commute_shape;

                if (clone_shape)
                {
                    graphlib::Shape updated_clone_shape = *clone_shape;
                    updated_clone_shape = producer->shape();
                    *clone_shape = updated_clone_shape;
                }

                new_dim = select_dim;
                can_commute = true;
            }
        }
    }
    
    if (dim_unaffected_by_commute)
    {
        graphlib::Shape updated_commute_shape = *commute_shape;
        if (producer and commute_up)
            updated_commute_shape[new_dim] = producer->shape().as_vector()[select_dim];
        else
            updated_commute_shape[new_dim] = op->shape().as_vector()[select_dim];
        *commute_shape = updated_commute_shape;

        if (clone_shape != nullptr)
        {
            graphlib::Shape updated_clone_shape = *clone_shape;
            if (producer and commute_up)
                updated_clone_shape[select_dim] = producer->shape().as_vector()[select_dim];
            else
                updated_clone_shape[select_dim] = op->shape().as_vector()[select_dim];
            *clone_shape = updated_clone_shape;
        }
    }

    if (check_only) {
        return can_commute;
    }
    

    TT_ASSERT(can_commute, "Cannot commute through op if can_commute is false");
    TT_ASSERT(retain_operand_dim, "retain_operand_dim must be set");
    TT_ASSERT(operand_dims, "operand_dims must be set");
    TT_ASSERT(golden_transform, "golden_transform must be set");

    // Perform the actual commute
    auto attr = op->op_attrs();

    std::get<int>(attr[0]) = new_dim - commute_shape->size();
    
    auto concat_shape = *commute_shape;
    int concat_output_len = op->shape().as_vector()[select_dim];
    op->set_shape(concat_shape);

    *retain_operand_dim = true;                    
    *operand_dims = std::make_pair(select_dim, new_dim);

    auto concat_golden_transform = *golden_transform;

    if (concat_golden_transform.op == "reshape")
        concat_golden_transform.attr[select_dim] = concat_output_len;
    op->add_golden_transform(concat_golden_transform);
    op->overwrite_op_attrs(attr);

    *commute_shape = concat_shape;
    *golden_transform = concat_golden_transform;
    (*clone_shape)[select_dim] = concat_output_len;

    log_trace(LogGraphCompiler, "  Select node: {} -> dim changed from {} to {}", op->name(), select_dim, new_dim);

    return true;
}

bool can_commute_through_select(
    graphlib::Graph* graph,
    graphlib::OpNode* op,
    graphlib::OpNode* initial_op,
    graphlib::Node* producer,
    graphlib::Shape *commute_shape,
    graphlib::Shape *clone_shape,
    bool commute_up)
{
    return commute_through_select(graph, op, initial_op, producer, commute_shape, clone_shape, true, nullptr, nullptr, nullptr, commute_up);             
}


bool commute_through_concat(
    graphlib::Graph* graph, 
    graphlib::OpNode* op, 
    graphlib::OpNode* initial_op, 
    graphlib::Node* producer,
    graphlib::Shape *commute_shape, 
    graphlib::Shape *clone_shape, 
    bool check_only,
    bool *retain_operand_dim,
    std::pair<int, int> *operand_dims,
    graphlib::OpType *golden_transform,
    bool commute_up) 
{    
    if (op->op_name() == "concatenate")
        TT_ASSERT(op->op_attrs().size() == 1);

    int concat_dim = std::get<int>(op->op_attrs()[0]);
    if (concat_dim < 0)
        concat_dim += op->shape().size();

    // setting commute_up here doesnt work when commuting through concat/select. 
    // This is because we actually want different behaviour from can_commute_through_dim
    // depending on whether we are trying to insert clones above the concat OR if we
    // are trying to commute down, but need to commute back up when checkingthat all
    // producer forks have an equivalent to the initial_op.
    // TODO: Should just remove that logic entirely from can_commute_through_dim

    // Find matching dim in commute and clone shape
    bool concat_commute_up = clone_shape ? commute_up : false;
    if (clone_shape) {
        int matching_in_commute_shape = -1;
        int matching_in_clone_shape = -1;
        for (int i = (int)commute_shape->size() - 1; i >= 0; i--)
        {
            if (matching_in_clone_shape != -1)
                break;
            for (int j = (int)clone_shape->size() - 1; j >= 0; j--)
            {
                if ((*commute_shape)[i] == (*clone_shape)[j])
                {
                    matching_in_commute_shape = i;
                    matching_in_clone_shape = j;
                    break;
                }
            }
        }

        if (matching_in_commute_shape == -1 or matching_in_clone_shape == -1) {
            TT_ASSERT(check_only, "Commute is impossible, must just be checking if commute is possible");
            return false;
        }

        if (matching_in_commute_shape == concat_dim)
            concat_commute_up = true;
    }
    auto [can_commute, new_dim] = can_commute_through_dim(initial_op, graph, concat_dim, concat_commute_up);
    concat_dim -= op->shape().size();
    if (can_commute and new_dim != 0)
    {
        graphlib::Shape updated_commute_shape = *commute_shape;
        if (producer and commute_up)
            updated_commute_shape[new_dim] = producer->shape()[concat_dim];
        else
            updated_commute_shape[new_dim] = op->shape()[concat_dim];
        *commute_shape = updated_commute_shape;

        if (clone_shape != nullptr)
        {
            graphlib::Shape updated_clone_shape = *clone_shape;
            if (producer and commute_up)
                updated_clone_shape[concat_dim] = producer->shape()[concat_dim];
            else
                updated_clone_shape[concat_dim] = op->shape()[concat_dim];
            *clone_shape = updated_clone_shape;
        }
    }
    else
        can_commute = false;

    if (check_only) {
        return can_commute;
    }
    

    TT_ASSERT(can_commute, "Cannot commute through op if can_commute is false");
    TT_ASSERT(retain_operand_dim, "retain_operand_dim must be set");
    TT_ASSERT(operand_dims, "operand_dims must be set");
    TT_ASSERT(golden_transform, "golden_transform must be set");

    // Perform the actual commute
    auto attr = op->op_attrs();

    if (new_dim >= 0)
        std::get<int>(attr[0]) = new_dim - commute_shape->size();
    else
        std::get<int>(attr[0]) = new_dim;
    
    auto concat_shape = *commute_shape;
    int concat_output_len = op->shape()[concat_dim];
    op->set_shape(concat_shape);

    *retain_operand_dim = true;                    
    *operand_dims = std::make_pair(concat_dim, new_dim);

    auto concat_golden_transform = *golden_transform;

    if (concat_golden_transform.op == "reshape")
        concat_golden_transform.attr[concat_dim >= 0 ? concat_dim : concat_dim + concat_golden_transform.attr.size()] = concat_output_len;
    op->add_golden_transform(concat_golden_transform);
    op->overwrite_op_attrs(attr);

    *commute_shape = concat_shape;
    *golden_transform = concat_golden_transform;
    (*clone_shape)[concat_dim] = concat_output_len;

    log_trace(LogGraphCompiler, "  Concat node: {} -> dim changed from {} to {}", op->name(), concat_dim, new_dim);

    return true;
}

bool can_commute_through_concat(
    graphlib::Graph* graph,
    graphlib::OpNode* op,
    graphlib::OpNode* initial_op,
    graphlib::Node* producer,
    graphlib::Shape *commute_shape,
    graphlib::Shape *clone_shape,
    bool commute_up)
{
    return commute_through_concat(graph, op, initial_op, producer, commute_shape, clone_shape, true, nullptr, nullptr, nullptr, commute_up);             
}

bool commute_through_reduce(
    graphlib::Graph* graph, 
    graphlib::OpNode* op, 
    graphlib::OpNode* initial_op, 
    graphlib::OpNode* producer,
    graphlib::Node* next,
    graphlib::Shape *commute_shape, 
    graphlib::Shape *clone_shape, 
    bool check_only,
    bool *retain_operand_dim,
    std::pair<int, int> *operand_dims,
    graphlib::OpType *golden_transform,
    bool commute_up) 
{   
    TT_ASSERT(op->op_attrs().size() == 1);
    int reduce_dim = std::get<int>(op->op_attrs()[0]);

    // Convert to positive indexing
    if (reduce_dim < 0)
        reduce_dim += op->shape().size();

    int original_reduce_dim = reduce_dim;
    // Check to see if this op has a user that is the same kind of reduce 
    bool can_commute = false;

    auto op_users = graph->data_users(op);
    auto op_operands = graph->data_operands(op);

    auto next_nodes = op_users;
    auto prev_nodes = op_operands;
    if (commute_up) {
        next_nodes = op_operands;
        prev_nodes = op_users;
    }

    // NOTE: We only allow this commute to happen if the transpose we are attempting to commute is Int32
    // This is because this commute can reduce perf on some models. The reason we allow it if the df is
    // Int32 is because Int32 transpose is not feasible on silicon, and so we must allow efforts to 
    // commute these transposes out of quantized regions to go forward.
    if (not commute_up and initial_op->op_name() == "transpose" and initial_op->output_df() == tt::DataFormat::Int32) {
        int dim0 = initial_op->op_type().get_attr_as<int>("dim0");
        int dim1 = initial_op->op_type().get_attr_as<int>("dim1");

        if (dim0 < 0)
            dim0 += initial_op->shape().size();

        if (dim1 < 0)
            dim1 += initial_op->shape().size();

        if (dim0 == reduce_dim)
        {
            reduce_dim = dim1;
        }
        else if (dim1 == reduce_dim) {
            reduce_dim = dim0;
        }

        if (reduce_dim < (int)op->shape().size()) {
            can_commute = true;
            auto reduce_shape = op->shape();
            *clone_shape = reduce_shape;
            reduce_shape[reduce_dim] = op->shape()[original_reduce_dim];
            reduce_shape[original_reduce_dim] = op->shape()[reduce_dim];
            *commute_shape = reduce_shape;
        }
        
    }
    else {

    
        for (graphlib::Node* next_node : next_nodes) {
            graphlib::OpNode *next_op = dynamic_cast<graphlib::OpNode*>(next_node);
            if (next_op == nullptr)
                continue;

            // Check if the next op is a reduce, and the same type of reduce
            if (next_op->op_name() != op->op_name())
                continue;

            auto compare_shape = check_only ? graph->data_operands(op)[0]->shape() : *clone_shape;

            int next_reduce_dim = std::get<int>(next_op->op_attrs()[0]);
            // Convert to positive indexing
            if (next_reduce_dim < 0)
                next_reduce_dim += next_op->shape().size();

            int min_reduce_dim = std::min(reduce_dim, next_reduce_dim);
            int max_reduce_dim = std::max(reduce_dim, next_reduce_dim);
            int commute_max_reduce_dim = max_reduce_dim - (op->shape().size() - commute_shape->size()); // Adjust for commute shape

            // This avoids the case where the reshape unflattens y into z. i.e (1, 1, 64, 4096) -> (1, 32, 2, 4096)
            if (not commute_up)
            {
                if ((*commute_shape)[commute_max_reduce_dim] != compare_shape[min_reduce_dim] * compare_shape[max_reduce_dim]) {
                    TT_ASSERT(check_only, "Cannot perform commute if commute is not possible");
                    can_commute = false;
                    break;
                }
            }
            else {
                if ((*commute_shape)[commute_max_reduce_dim] != 1) {
                    TT_ASSERT(check_only, "Cannot perform commute if commute is not possible");
                    can_commute = false;
                    break;
                }
            }
            // If the next op is the same kind of reduce, and the reduce dim is one off, skip, next op we handle this case 
            if (next_reduce_dim == reduce_dim+1 or next_reduce_dim == reduce_dim-1) {
                can_commute = true;
                break;
            }
        }

        // Check to see if previous op is reduce
        for(graphlib::Node* prev_node : prev_nodes)
        {
            graphlib::OpNode *prev_op = dynamic_cast<graphlib::OpNode*>(prev_node);
            if (prev_op == nullptr)
                continue;

            if (prev_op->op_name() == op->op_name())
            {
                TT_ASSERT(prev_op->op_attrs().size() == 1);
                int prev_reduce_dim = std::get<int>(prev_op->op_attrs()[0]);
                // Convert to positive indexing
                if (prev_reduce_dim < 0)
                    prev_reduce_dim += op->shape().size();

                // If the previous op is the same kind of reduce, and the reduce dim is one off, then we can determine the commute shape after both ops
                if (prev_reduce_dim == reduce_dim+1 or prev_reduce_dim == reduce_dim-1)
                {
                    auto commute_dim = (uint32_t) std::max(prev_reduce_dim, reduce_dim);
                    auto commute_vec = commute_shape->as_vector();
                    while (commute_dim >= commute_vec.size())  
                        commute_vec.push_back(1);
                    *commute_shape = graphlib::Shape::create(commute_vec);
                    if (commute_up)
                        (*commute_shape)[commute_dim] = producer->shape()[reduce_dim] * producer->shape()[prev_reduce_dim];
                    else
                        (*commute_shape)[commute_dim] = 1;
                    if (clone_shape != nullptr) {
                        (*clone_shape)[reduce_dim] = 1;
                        (*clone_shape)[prev_reduce_dim] = 1;
                    }
                    can_commute = true;
                }
            }
        }
        
        if (not can_commute)
        {
            auto [can_commute, new_dim] = can_commute_through_dim(initial_op, graph, reduce_dim, commute_up);
            if (can_commute)
            {
                graphlib::Shape updated_commute_shape = *commute_shape;
                if (producer)
                {
                    TT_ASSERT(commute_up, "Should only be using producer for shape if commuting up");
                    updated_commute_shape[new_dim] = producer->shape().as_vector()[reduce_dim];
                }
                else
                {
                    updated_commute_shape[new_dim] = op->shape().as_vector()[reduce_dim];
                }
                *commute_shape = updated_commute_shape;
                if (clone_shape != nullptr)
                {
                    graphlib::Shape updated_clone_shape = *clone_shape;
                    if (producer)
                    {
                        TT_ASSERT(commute_up, "Should only be using producer for shape if commuting up");
                        updated_clone_shape[reduce_dim] = producer->shape().as_vector()[reduce_dim];
                    }
                    else
                    {
                        updated_clone_shape[reduce_dim] = op->shape().as_vector()[reduce_dim];
                    }
                    *clone_shape = updated_clone_shape;
                }
            }
        }
    }

    if (check_only)
        return can_commute;

    TT_ASSERT(can_commute, "Cannot commute through op if can_commute is false");
    TT_ASSERT(retain_operand_dim, "retain_operand_dim must be set");
    TT_ASSERT(operand_dims, "operand_dims must be set");
    TT_ASSERT(golden_transform, "golden_transform must be set");
    TT_ASSERT(next, "next must be set");
    TT_ASSERT(not commute_up, "Cannot perform commute upwards");

    if (not commute_up and initial_op->op_name() == "transpose") {
        auto op_attr = op->op_attrs();
        auto reduce_shape = op->shape();

        *clone_shape = reduce_shape;
        reduce_shape[reduce_dim] = op->shape()[original_reduce_dim];
        reduce_shape[original_reduce_dim] = op->shape()[reduce_dim];
        *commute_shape = reduce_shape;

        op_attr[0] = reduce_dim;
        op->add_golden_transform(*golden_transform);
        op->overwrite_op_attrs(op_attr);
        op->set_shape(reduce_shape);
        return true;
    }

    if (graphlib::OpNode *next_as_op = dynamic_cast<graphlib::OpNode *>(next)) {
        if (op->op_name() == next_as_op->op_name()) {
            return true;
        }
    }

    if (producer->op_name() == op->op_name()) {
        auto op_attr = op->op_attrs();
        auto producer_attr = producer->op_attrs();

        int op_reduce_dim = std::get<int>(op_attr[0]);
        if (op_reduce_dim < 0)
            op_reduce_dim += clone_shape->size();

        int producer_reduce_dim = std::get<int>(producer_attr[0]);
        if (producer_reduce_dim < 0)
            producer_reduce_dim += clone_shape->size();

        int new_op_dim = std::max(op_reduce_dim, producer_reduce_dim);
        std::get<int>(op_attr[0]) = new_op_dim - commute_shape->size();

        auto reduce_shape = *commute_shape;
        auto reduce_vec = reduce_shape.as_vector();
        while ((uint32_t)new_op_dim >= reduce_vec.size())  
            reduce_vec.push_back(1);
        reduce_shape = graphlib::Shape::create(reduce_vec);

        auto reduce_golden_transform = *golden_transform;
        if (reduce_golden_transform.op == "reshape")
        {
            reduce_golden_transform.attr[op_reduce_dim] = 1;
            reduce_golden_transform.attr[producer_reduce_dim] = 1;
        }

        op->add_golden_transform(reduce_golden_transform);
        op->overwrite_op_attrs(op_attr);

        *commute_shape = reduce_shape;
        *golden_transform = reduce_golden_transform;
        (*clone_shape)[op_reduce_dim] = 1;
        (*clone_shape)[producer_reduce_dim] = 1;

        producer->change_op_type("nop");
    }
    else 
    {
        auto attr = op->op_attrs();
        int reduce_dim = std::get<int>(attr[0]);

        int orig_op_dims = op->shape().size();
        if (reduce_dim < 0)
        {
            reduce_dim += orig_op_dims;
        }
        int new_dim = std::get<1>(can_commute_through_dim(initial_op, graph, reduce_dim));
        std::get<int>(attr[0]) = new_dim - commute_shape->size();
        
        auto reduce_shape = *commute_shape;
        reduce_shape[new_dim] = 1;
        op->set_shape(reduce_shape);

        auto reduce_golden_transform = *golden_transform;
        if (reduce_golden_transform.op == "reshape")
            reduce_golden_transform.attr[reduce_dim] = 1;

        op->add_golden_transform(reduce_golden_transform);

        op->overwrite_op_attrs(attr);

        *commute_shape = reduce_shape;
        *golden_transform = reduce_golden_transform;
        (*clone_shape)[reduce_dim] = 1;
        log_trace(LogGraphCompiler, "  Reduce node: {} -> dim changed from {} to {}", op->name(), reduce_dim, new_dim);
    }

    return true;
}

bool can_commute_through_reduce(
    graphlib::Graph* graph, 
    graphlib::OpNode* op, 
    graphlib::OpNode* initial_op, 
    graphlib::OpNode* producer,
    graphlib::Shape* commute_shape,
    graphlib::Shape* clone_shape,
    bool commute_up)
{
    return commute_through_reduce(graph, op, initial_op, producer, nullptr, commute_shape, clone_shape, true, nullptr, nullptr, nullptr, commute_up);
}

bool commute_through_eltwise(
    graphlib::OpNode* op,
    graphlib::Shape *commute_shape,
    graphlib::OpType *golden_transform)
{
    TT_ASSERT(is_elementwise(op), "op must be an eltwise op");
    op->set_shape(*commute_shape);
    op->add_golden_transform(*golden_transform);
    return true;
}

bool commute_through_squeeze(
    graphlib::OpNode* op,
    graphlib::OpNode* initial_op,
    graphlib::Shape* commute_shape,
    graphlib::Shape* clone_shape,
    graphlib::OpType* golden_transform,
    bool commute_up,
    bool check_only) 
{   
    TT_ASSERT(op->op_name() == "squeeze", "Op is not a squeeze op");
    if (commute_up)
        return false;

    // Only commute transpose through squeeze for now
    if (initial_op->op_name() != "transpose")
        return false;

    std::vector<graphlib::OpType::Attr> op_attrs = op->op_attrs();

    // Commute only if squeeze dim is 0 for now
    if (std::get<int>(op_attrs[0]) != 0)
        return false;

    if ((*commute_shape)[0] != 1)
        return false;

    if (check_only)
        return true;

    auto updated_commute_shape = commute_shape->as_vector();
    updated_commute_shape.erase(updated_commute_shape.begin());
    *commute_shape = graphlib::Shape::create(updated_commute_shape);
    op->set_shape(*commute_shape);
    op->add_golden_transform(*golden_transform);

    auto updated_clone_shape = clone_shape->as_vector();
    updated_clone_shape.erase(updated_clone_shape.begin());
    *clone_shape = graphlib::Shape::create(updated_clone_shape);

    return true;
}

bool can_commute_through_squeeze(
    graphlib::OpNode* op,
    graphlib::OpNode* initial_op,
    graphlib::Shape* commute_shape,
    graphlib::Shape* clone_shape,
    bool commute_up) 
{
    return commute_through_squeeze(op, initial_op, commute_shape, clone_shape, nullptr, commute_up, true);
}


bool commute_through_quantization(
    graphlib::OpNode* op, 
    graphlib::OpNode* initial_op,
    bool check_only,
    graphlib::Shape* commute_shape,
    graphlib::OpType* golden_transform,
    bool commute_up)
{
    TT_ASSERT(is_quantization_ops(op), "op must be an quantization op");
    (void)commute_up; // Avoid compiler warning.
    int axis = std::get<int>(op->op_attrs()[1]);
    int new_axis = axis;
    bool can_commute = false;

    if (initial_op->op_type().op == "reshape") {
        
        // axis of quantization must have the same volume to the left and right of it
        if (new_axis < 0)
            new_axis += op->shape().size();

        // check if axis moved to the right (or in the same place)
        while (new_axis < (int)commute_shape->size()) {
            if ((*commute_shape)[new_axis] == op->shape()[axis]) {
                if (volume_above(commute_shape->as_vector(), new_axis) == volume_above(op->shape().as_vector(), axis)
                    and volume_below(commute_shape->as_vector(), new_axis) == volume_below(op->shape().as_vector(), axis)) {
                    can_commute = true;
                }
                break;
            }
            new_axis++;
        }
        if (not can_commute) {
            new_axis = axis-1;
            while (new_axis >= 0) {
                if ((*commute_shape)[new_axis] == op->shape()[axis]) {
                    if (volume_above(commute_shape->as_vector(), new_axis) == volume_above(op->shape().as_vector(), axis)
                        and volume_below(commute_shape->as_vector(), new_axis) == volume_below(op->shape().as_vector(), axis)) {
                        can_commute = true;
                    }
                    break;
                }
                new_axis--;
            }
        } 
    } 
    else if (initial_op->op_type().op == "transpose") 
    {   
        can_commute = true;
        if (new_axis < 0)
            new_axis += op->shape().size();
        
        int transpose_dim0 = initial_op->op_type().get_attr_as<int>("dim0");
        int transpose_dim1 = initial_op->op_type().get_attr_as<int>("dim1");
        if (transpose_dim0 < 0)
            transpose_dim0 += commute_shape->size();

        if (transpose_dim1 < 0)
            transpose_dim1 += commute_shape->size();

        if (new_axis == transpose_dim0)
            new_axis = transpose_dim1;
        else if (new_axis == transpose_dim1)
            new_axis = transpose_dim0;
    }
        
    if (check_only)
        return can_commute;

    TT_ASSERT(can_commute, "Should not have called this if it is incommutable.");

    std::vector<graphlib::OpType::Attr> op_attrs = op->op_attrs();
    op_attrs[1] = new_axis;
    op->set_shape(*commute_shape);
    op->overwrite_op_attrs(op_attrs);
    op->add_golden_transform(*golden_transform);
    return true;
}

bool can_commute_through_quantize(
    graphlib::OpNode* op, 
    graphlib::OpNode* initial_op,
    graphlib::Shape* commute_shape,
    bool commute_up)
{
    return commute_through_quantization(op, initial_op, true, commute_shape, nullptr, commute_up);
}

bool is_elementwise(graphlib::OpNode *op)
{
    py::object eval_module = py::module_::import("pybuda.op.eval.pybuda");
    py::function is_eltwise = eval_module.attr("is_eltwise");
    return is_eltwise(op->op_type()).cast<bool>();
}

bool is_quantization_ops(graphlib::OpNode *op)
{
    return op->op_name() == "buda_quantize" or op->op_name() == "buda_dequantize" or op->op_name() == "buda_requantize" 
           or op->op_name() == "quantize" or op->op_name() == "dequantize" or op->op_name() == "requantize" ;
}


bool can_commute_past_op(
    graphlib::OpNode *op,
    graphlib::OpNode *initial_op,
    graphlib::Graph *graph,
    graphlib::Shape *commute_shape,
    graphlib::Shape *clone_shape,
    bool commute_up,
    graphlib::Node *producer)
{
    if (op->op_name() == "reduce_avg" or op->op_name() == "reduce_sum")
    {
        graphlib::OpNode *producer_as_op = dynamic_cast<graphlib::OpNode *>(producer);
        bool can_commute = can_commute_through_reduce(graph, op, initial_op, producer_as_op, commute_shape, clone_shape, commute_up);
        return can_commute;
    }
    else if (op->op_name() == "concatenate")
    {
        bool can_commute = can_commute_through_concat(graph, op, initial_op, producer, commute_shape, clone_shape, commute_up);
        return can_commute;
    }
    else if (op->op_name() == "select")
    {
        bool can_commute = can_commute_through_select(graph, op, initial_op, producer, commute_shape, clone_shape, commute_up);
        return can_commute;
    }
    else if (is_quantization_ops(op)) {
        bool can_commute = can_commute_through_quantize(op, initial_op, commute_shape, commute_up);
        return can_commute;
    } else if (op->op_name() == "squeeze") {
        return can_commute_through_squeeze(op, initial_op, commute_shape, clone_shape, commute_up);
    }

    return (is_elementwise(op) and op->op_name() != "interleave");
}

void update_reshape_attr(graphlib::OpNode *reshape, graphlib::Shape new_shape)
{
    if(reshape->op_name() != "reshape")
        return;
    std::vector<graphlib::OpType::Attr> attr;
    for (size_t i=0; i < new_shape.size(); i++)
    {
        attr.push_back((int)new_shape[i]);
    }
    reshape->overwrite_op_attrs(attr);
}

std::pair<bool, std::pair<std::vector<int>, std::vector<int>>> handle_shape_change_through_bcast(
    graphlib::Graph *graph, 
    graphlib::OpNode *initial_op, 
    graphlib::Node *producer, 
    graphlib::OpNode *consumer, 
    graphlib::Shape *commute_shape, 
    graphlib::Shape *clone_shape) 
{   
    graphlib::Edge edge = retrieve_between_edge(graph, producer, consumer);

    auto tms = graph->get_edge_attributes(edge)->get_tms();

    std::vector<int> total_bcast_on_commute_shape(commute_shape->size(), 1);
    std::vector<int> total_bcast_on_clone_shape(clone_shape->size(), 1);
    bool can_commute = true;
    for (graphlib::OpType &op_type : tms) {
        if (op_type.op == "broadcast")
        {
            int bcast_dim = std::get<int>(op_type.attr[0]);
            int volume = std::get<int>(op_type.attr[1]);
            if (bcast_dim < 0)
                bcast_dim += clone_shape->size();
            auto [can_commute_bcast_through_dim, new_dim] = can_commute_through_dim(initial_op, graph, bcast_dim);
            can_commute = can_commute and can_commute_bcast_through_dim;
            // It is possible that we may be accumulating broadcasts, so use *=
            if (can_commute)
            {
                (*commute_shape)[new_dim] *= volume;
                total_bcast_on_commute_shape[new_dim] *= volume;
                if (clone_shape != nullptr)
                {
                    (*clone_shape)[bcast_dim] *= volume;
                    total_bcast_on_clone_shape[bcast_dim] *= volume;
                }
            }        
        }
    }
    return std::make_pair(can_commute, std::make_pair(total_bcast_on_commute_shape, total_bcast_on_clone_shape));
}

void add_or_compound_bcast(graphlib::Graph *graph, graphlib::Edge edge, int dim, int volume)
{
    auto tms = graph->get_edge_attributes(edge)->get_tms();
    graph->get_edge_attributes(edge)->clear_broadcast_dims();

    if (tms.size() == 0) {
        graph->get_edge_attributes(edge)->set_broadcast_dim(dim, volume, false);
        return;
    }
    
    bool compounded_bcast = false;
    for (graphlib::OpType &op_type : tms)
    {
        if (op_type.op == "broadcast")
        {
            int existing_dim = std::get<int>(op_type.attr[0]);

            int existing_volume = std::get<int>(op_type.attr[1]);
            if (existing_dim == dim)
            {
                compounded_bcast = true;
                existing_volume *= volume;
            }

            graph->get_edge_attributes(edge)->set_broadcast_dim(existing_dim, existing_volume, std::get<bool>(op_type.attr[2]));
        }
    }

    if (not compounded_bcast) {
        graph->get_edge_attributes(edge)->set_broadcast_dim(dim, volume, false);
    }
}

void restore_bcast_on_condition(graphlib::Graph *graph, graphlib::Edge edge, std::vector<graphlib::OpType> orig_tms, graphlib::Shape operand_shape, std::function<bool(graphlib::Shape, int)> eval_condition)
{
    // auto tms = graph->get_edge_attributes(edge)->get_tms();
    for (graphlib::OpType &op_type : orig_tms) 
    {
        if (op_type.op == "broadcast") 
        {   
            int dim = std::get<int>(op_type.attr[0]);
            int volume = std::get<int>(op_type.attr[1]);
            if (dim < 0)
                dim += operand_shape.size();

            if (eval_condition(operand_shape, dim))
                continue;
            
            add_or_compound_bcast(graph, edge, dim, volume);
        }
    }
}

bool try_commute_bcast_through_clone(graphlib::Graph *graph, graphlib::OpNode *node) 
{
    graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(node);
    if (not op)
        return false;

    if (not (op->op_name() == "transpose" or op->op_name() == "reshape"))
        return false;

    auto operand_edge = graph->operand_data_edges(node)[0];
    auto user_edge = graph->user_data_edges(op)[0];
    auto operand = graph->node_by_id(operand_edge.producer_node_id);
    auto tms = graph->get_edge_attributes(operand_edge)->get_tms();

    // Determine shape of input after broadcasts
    auto operand_shape = operand->shape();
    auto original_operand_shape = operand->shape();
    auto op_shape_as_vec = operand_shape.as_vector();
    while (op_shape_as_vec.size() < node->shape().size())
        op_shape_as_vec.insert(op_shape_as_vec.begin(), 1);
    operand_shape = graphlib::Shape::create(op_shape_as_vec);

    bool has_bcasts = false;
    int total_bcast_volume = 1;
    int num_bcasts = 0;
    std::vector<int> bcast_dims;
    for (graphlib::OpType &op_type : tms) 
    {
        if (op_type.op == "broadcast") 
        {   
            has_bcasts = true;
            num_bcasts++;
            int dim = std::get<int>(op_type.attr[0]);
            bcast_dims.push_back(dim);
            int volume = std::get<int>(op_type.attr[1]);
            if (dim < 0)
            {
                while (-dim > (int)operand_shape.size()) {
                    op_shape_as_vec = operand_shape.as_vector();
                    op_shape_as_vec.insert(op_shape_as_vec.begin(), 1);
                    operand_shape = graphlib::Shape::create(op_shape_as_vec);
                }
                dim += operand_shape.size();
            }
                
            operand_shape[dim] = volume;
            total_bcast_volume *= volume;
        }
    }

    // True because its a tm and we succesfully hoisted the tm through the bcasts on the edge
    // .... which is none
    if (not has_bcasts)
        return true;

    // Find equivalent dim in reshape
    int matching_in_operand = -1;
    int matching_in_op = -1;

    for (int i = (int)operand_shape.size() - 1; i >= 0; i--) {
        if (matching_in_operand != -1)
            break;
        for (int j = (int)op->shape().size() - 1; j >= 0; j--){
            if (operand_shape[i] == op->shape()[j] and operand_shape[i] != 1) {
                matching_in_operand = i;
                matching_in_op = j;
                break;
            }
        }
    }
    if (matching_in_operand == -1 or matching_in_op == -1)
        return false;

    if (op->op_name() == "reshape")
    {
        if (volume_above(operand_shape.as_vector(), matching_in_operand) == volume_above(op->shape().as_vector(), matching_in_op) and
            volume_below(operand_shape.as_vector(), matching_in_operand) == volume_below(op->shape().as_vector(), matching_in_op) and (operand->shape().size() != op->shape().size() || operand->shape()[1] != op->shape()[1])) {
            // We can commute the broadcast through the reshape
            // Only one broadcast now

            
            auto operand_swapped_shape = operand_shape;
            auto tmp = operand_swapped_shape[matching_in_operand];
            operand_swapped_shape[matching_in_operand] = operand_swapped_shape[matching_in_op];
            operand_swapped_shape[matching_in_op] = tmp;

            int matching_in_operand_neg = matching_in_operand - (int)operand_shape.size();
            int matching_in_op_neg = matching_in_op - (int)op->shape().size();

            bool is_transpose_reshape = operand_swapped_shape == op->shape() and operand_shape != op->shape() and matching_in_operand != matching_in_op;
            bool is_flatten_channel_first = ((uint32_t)matching_in_op == op->shape().size() - 2) and (matching_in_operand_neg == matching_in_op_neg - 1) and (not is_transpose_reshape);
            bool is_flatten_channel_last =  ((uint32_t)matching_in_op == op->shape().size() - 1) and (matching_in_operand_neg == matching_in_op_neg) and (not is_transpose_reshape) and (not is_flatten_channel_first)
                                            and (total_bcast_volume == volume_above(op->shape().as_vector(), matching_in_op)) and (total_bcast_volume == volume_above(operand_shape.as_vector(), matching_in_operand));

            bool is_single_bcast_unnafected_by_reshape = num_bcasts == 1 and (volume_below(op->shape().as_vector(), matching_in_op + 1) == volume_below(operand_shape.as_vector(), matching_in_operand + 1)) 
                                                         and (volume_above(op->shape().as_vector(), matching_in_op) == volume_above(operand_shape.as_vector(), matching_in_operand))
                                                         and (matching_in_op_neg == matching_in_operand_neg) and (bcast_dims[0] == matching_in_op_neg or bcast_dims[0] == matching_in_op);
            bool is_nop_reshape_after_bcast = are_different_ranked_shapes_equivalent(op->shape(), operand_shape);
            // i.e its a reshape op but the before/after is (1, 64, 1, 1) -> (1, 1, 64, 1) or similar

            if (is_transpose_reshape) {
                int bcast_dim = std::get<int>(tms[0].attr[0]);
                if (bcast_dim < 0)
                    bcast_dim += operand_shape.size();
                int bcast_volume = op->shape()[bcast_dim];

                // Set new reshape shape
                auto new_reshape_shape = op->shape();
                auto attr = op->op_attrs();
                std::get<int>(attr[bcast_dim]) = 1;
                new_reshape_shape[bcast_dim] = 1;
                op->set_shape(new_reshape_shape);
                op->overwrite_op_attrs(attr);

                // Remove the broadcasts from operand
                graph->get_edge_attributes(operand_edge)->clear_broadcast_dims();

                // Set broadcasts on user edge
                add_or_compound_bcast(graph, user_edge, bcast_dim, bcast_volume);

                // Place back any unaffected broadcasts
                std::function<bool(graphlib::Shape, int)> restore_on_transpose_condition = [matching_in_operand](graphlib::Shape operand_shape, int dim) {
                    // Avoid compiler warnings
                    (void) operand_shape;
                    return dim != matching_in_operand;
                };
                restore_bcast_on_condition(graph, operand_edge, tms, operand_shape, restore_on_transpose_condition);
            }
            // I.e (1, 64, 224, 224) -> (1, 1, 64, 50176)
            else if (is_flatten_channel_first) {
                int bcast_dim = -1;
                int bcast_volume = op->shape()[op->shape().size() - 1];

                // Set new reshape shape
                auto new_reshape_shape = op->shape();
                auto attr = op->op_attrs();
                std::get<int>(attr[op->shape().size() - 1]) = 1;
                new_reshape_shape[op->shape().size() - 1] = 1;
                op->set_shape(new_reshape_shape);
                op->overwrite_op_attrs(attr);
                
                // Remove the broadcasts from operand
                graph->get_edge_attributes(operand_edge)->clear_broadcast_dims();

                // Set broadcasts on user edge
                add_or_compound_bcast(graph, user_edge, bcast_dim, bcast_volume);

                // Place back any unaffected broadcasts
                std::function<bool(graphlib::Shape, int)> restore_on_channel_first_flatten_condition = [](graphlib::Shape operand_shape, int dim) { 
                    return (uint32_t)dim == operand_shape.size()-1 or (uint32_t)dim == operand_shape.size()-2;
                };
                restore_bcast_on_condition(graph, operand_edge, tms, operand_shape, restore_on_channel_first_flatten_condition);
            }
            else if (is_flatten_channel_last) {

                int bcast_dim = -2;
                int bcast_volume = op->shape()[op->shape().size() - 2];

                // Set new reshape shape
                auto new_reshape_shape = op->shape();
                auto attr = op->op_attrs();
                std::get<int>(attr[op->shape().size() - 2]) = 1;
                new_reshape_shape[op->shape().size() - 2] = 1;
                op->set_shape(new_reshape_shape);
                op->overwrite_op_attrs(attr);
                
                // Remove the broadcasts from operand
                graph->get_edge_attributes(operand_edge)->clear_broadcast_dims();

                // Set broadcasts on user edge
                add_or_compound_bcast(graph, user_edge, bcast_dim, bcast_volume);

                // Place back any unaffected broadcasts
                std::function<bool(graphlib::Shape, int)> restore_on_channel_last_flatten_condition = [](graphlib::Shape operand_shape, int dim) { 
                    return (uint32_t)dim == operand_shape.size()-2 or (uint32_t)dim == operand_shape.size()-3;
                };
                restore_bcast_on_condition(graph, operand_edge, tms, operand_shape, restore_on_channel_last_flatten_condition);
            }
            else if (is_single_bcast_unnafected_by_reshape) {
                // i.e (1, 1, 32, 1024) -> (1, 32, 1, 1024). Where the 1024 is just broadcasted from (1, 1, 32, 1)
                int bcast_dim = matching_in_op_neg;
                int bcast_volume = op->shape()[op->shape().size() + bcast_dim];

                // Set new reshape shape
                auto new_reshape_shape = op->shape();
                auto attr = op->op_attrs();
                std::get<int>(attr[op->shape().size() + bcast_dim]) = 1;
                new_reshape_shape[op->shape().size() + bcast_dim] = 1;
                op->set_shape(new_reshape_shape);
                op->overwrite_op_attrs(attr);

                // Remove the broadcasts from operand
                graph->get_edge_attributes(operand_edge)->clear_broadcast_dims();

                // Set broadcasts on user edge
                add_or_compound_bcast(graph, user_edge, bcast_dim, bcast_volume);
            }
            else if (is_nop_reshape_after_bcast) {
                // Remove the broadcasts from operand
                graph->get_edge_attributes(operand_edge)->clear_broadcast_dims();
                auto new_shape = original_operand_shape.as_vector();
 
                while (new_shape.size() > op->shape().size())
                {
                    TT_ASSERT(new_shape[0] == 1);
                    new_shape.erase(new_shape.begin());
                }

                op->set_shape(graphlib::Shape::create(new_shape));
                update_reshape_attr(op, op->shape());

                for (auto &t : tms) {
                    if (t.op == "broadcast") {
                        int dim = std::get<int>(t.attr[0]);
                        if (dim >= 0) {
                            dim -= operand_shape.size();
                        }
                        int volume = std::get<int>(t.attr[1]);
                        add_or_compound_bcast(graph, user_edge, dim, volume);
                    }
                }
            }
            else
                return false;
        }
        else
            return false;
        return true;
    }
    else if (op->op_name() == "transpose")
    {
        auto &tms = graph->get_edge_attributes(operand_edge)->get_tms();
        auto updated_shape = node->shape();

        std::vector<int> erase_tms;
        for (int i = 0; i < (int)tms.size(); ++i)
        {
            graphlib::OpType op_type = tms[i];
            if (op_type.op == "broadcast")
            {
                int dim = std::get<int>(op_type.attr[0]);
                if (dim > 0)
                    dim -= operand_shape.size();

                int operand_dim_size = operand->shape()[dim];
                // Hoist the transpose before the bcasts
                if (dim == -1 or dim == -2)
                {
                    erase_tms.push_back(i);
                    dim = (dim == -1) ? -2 : -1;  // flip the dim
                    updated_shape[dim] = operand_dim_size;
                    std::get<int>(op_type.attr[0]) = dim;
                    for (auto user_edge : graph->user_data_edges(op))
                        graph->get_edge_attributes(user_edge)->prepend_tm(op_type);
                }
            }
        }

        node->set_shape(updated_shape);

        // Erase tms in reverse order so indices remain ordered
        for (auto iter = erase_tms.rbegin(); iter != erase_tms.rend(); ++iter) tms.erase(tms.begin() + *iter);
        return true;
    }
    return false;
}

bool all_producer_forks_have_equivalent(
    graphlib::Graph *graph,
    graphlib::OpNode *initial_op,
    graphlib::Shape commute_shape,
    graphlib::OpNode *from)
{
    graphlib::OpNode *iter = from ? from : initial_op;

    bool found_equivalent = false;
    while (not found_equivalent)
    {
        graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(iter);
        TT_ASSERT(op);

        found_equivalent = are_compatible_ops(graph, initial_op, op, &commute_shape, false);

        bool all_forks_have_equivalent = true;
        std::vector<graphlib::Node *> operands = graph->data_operands(op);
        bool can_commute = can_commute_past_op(op, initial_op, graph, &commute_shape, nullptr, true, operands[0]);
        
        for (std::size_t i = 1; (i < operands.size()) and all_forks_have_equivalent; ++i)
        {
            graphlib::InputNode *input = dynamic_cast<graphlib::InputNode *>(operands[i]);
            if (input and (input->is_constant() or input->is_parameter()))
                continue;
            graphlib::OpNode *operand = dynamic_cast<graphlib::OpNode *>(operands[i]);
            auto commute_shape_copy = commute_shape;
            can_commute_past_op(op, initial_op, graph, &commute_shape_copy, nullptr, true, operands[i]);
            all_forks_have_equivalent &= operand and all_producer_forks_have_equivalent(graph, initial_op, commute_shape_copy, operand);
        }

       
        if (found_equivalent and all_forks_have_equivalent)
        {
            return true;
        }
        else if (not can_commute or not all_forks_have_equivalent)
        {
            return false;
        }
        iter = dynamic_cast<graphlib::OpNode *>(operands[0]);
        if (not iter)
            break;
    }

    return false;
}

} // namespace tt::passes
