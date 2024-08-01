// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/fuse_redundant_tm_sequence.hpp"
#include "passes/commute_utils.hpp"

using tt::LogTMFusion;
namespace tt::passes
{
 
std::pair<graphlib::OpType, graphlib::Shape> generate_inverse_info(graphlib::Graph *graph, graphlib::OpNode *op) {
    
    TT_ASSERT(op->op_name() == "reshape" or op->op_name() == "transpose", "Op must be reshape or transpose.");

    if (op->op_name() == "reshape") {
        graphlib::Shape inverse_shape = op->shape_of_operand(graph, graph->data_operands(op)[0]);
        graphlib::OpType inverse_op_type = op->op_type();
        std::vector<BudaOpAttr> new_attrs;
        for (uint32_t dim : inverse_shape) {
            new_attrs.push_back((int)dim);
        }

        inverse_op_type.attr = new_attrs;
        return std::make_pair(inverse_op_type, inverse_shape);
    }
    else {
        graphlib::Shape inverse_shape = op->shape_of_operand(graph, graph->data_operands(op)[0]);
        // Attrs remain the same for inverse transpose
        return std::make_pair(op->op_type(), inverse_shape);
    }
}

void clone_tm_on_all_user_forks(graphlib::Graph *graph, graphlib::OpNode *tm) {
    TT_ASSERT(graph->data_users(tm).size() > 1);

    for (auto user_edge : graph->user_data_edges(tm)) {
        std::string clone_name = tm->name() + "_redundant_tm_pattern_tm_user_fork_clone" + std::to_string(user_edge.edge_creation_id);
        graphlib::Node *clone = graph->add_node(tm->clone(clone_name), graph->get_subgraph_id_for_node(tm->id()));
        insert_node_on_edge(graph, user_edge, clone);
    }
    log_debug(LogGraphCompiler, "Moving forks of TM: {} to operand.", tm->name());
    bypass_node(graph, tm, true);
}

void clone_tms_on_forks(graphlib::Graph *graph, std::vector<graphlib::OpNode *> tms_to_move) {
    // Iterate through these backwards
    for (int i = tms_to_move.size()-1; i >= 0; i--) {
        graphlib::OpNode *tm = tms_to_move[i];
        clone_tm_on_all_user_forks(graph, tm);
    }
}

bool swap_down_through_eltwise(graphlib::Graph *graph, graphlib::OpNode *tm) {
    std::vector<graphlib::Node *> users = graph->data_users(tm);
    if (users.size() > 1) {
        return false;
    }

    py::object eval_module = py::module_::import("pybuda.op.eval.pybuda");
    py::function is_tm = eval_module.attr("is_tm");

    TT_ASSERT(is_tm(tm->op_type()).cast<bool>());

    graphlib::OpNode *user = users[0]->as<graphlib::OpNode>();

    TT_ASSERT(user, "User must be elteise or quantization op.");

    if (not (user and (is_eltwise(user) or is_quantization_ops(user))))
        return false;

    // Don't want to check the operand that is tm;
    std::vector<graphlib::Edge> user_operands = graph->operand_data_edges(user, [tm](graphlib::Edge edge) {
        return edge.producer_node_id != tm->id();
    });
    std::vector<graphlib::Edge> user_users = graph->user_data_edges(user);

    // Place an inverse on each operand
    auto [inverse_op_type, inverse_shape] = generate_inverse_info(graph, tm);
    for (auto operand_edge : user_operands) {
        std::string inverse_clone_name = tm->name() + "_redundant_tm_pattern_tm_commute_operand_clone" + std::to_string(operand_edge.edge_creation_id);
        graphlib::OpNode *inverse_clone = graph->add_node(tm->clone(inverse_clone_name), graph->get_subgraph_id_for_node(tm->id()))->as<graphlib::OpNode>();
        inverse_clone->overwrite_op_attrs(inverse_op_type.attr);
        inverse_clone->set_shape(inverse_shape);
        insert_node_on_edge(graph, operand_edge, inverse_clone);
        inverse_clone->set_output_df_from_operands(graph);

        
        if (graph->data_operands(inverse_clone)[0]->node_type() == graphlib::NodeType::kInput)
        {
            try_consteval_op(graph, inverse_clone, true);
        }
    }

    // Move node down and attach all users 
    graphlib::Edge old_tm_user_edge = retrieve_between_edge(graph, tm, user);
    graphlib::Edge new_edge = graphlib::Edge(graph->data_operands(tm)[0]->id(), old_tm_user_edge.producer_output_port_id, user->id(), old_tm_user_edge.consumer_input_port_id, old_tm_user_edge.edge_type);
    graph->add_edge(new_edge);
    graphlib::Edge old_tm_operand_edge = retrieve_between_edge(graph, graph->data_operands(tm)[0], tm);
    graph->remove_edge(old_tm_operand_edge);
    graph->remove_edge(old_tm_user_edge);

    for (uint32_t i = 0; i < user_users.size(); i++) {
        graphlib::Edge new_user_user_edge = graphlib::Edge(tm->id(), i, user_users[i].consumer_node_id, user_users[i].consumer_input_port_id, user_users[i].edge_type);
        graph->remove_edge(user_users[i]);
        graph->add_edge(new_user_user_edge);
    }

    graphlib::Edge new_user_tm_edge = graphlib::Edge(user->id(), 0, tm->id(), 0, new_edge.edge_type);
    graph->add_edge(new_user_tm_edge);
    user->set_shape(inverse_shape);

    user->add_golden_transform(tm->op_type());

    return true;
}

void move_down_through_eltwise(graphlib::Graph *graph, std::vector<graphlib::OpNode *> tms_to_move) {
    // Iterate through these backwards
    for (int i = tms_to_move.size()-1; i >= 0; i--) {
        graphlib::OpNode *tm = tms_to_move[i];

        while (swap_down_through_eltwise(graph, tm));
    }
}

bool equivalent_pattern(const TMPattern& pattern1, const TMPattern& pattern2) {
    if (pattern1.size() != pattern2.size())
        return false;

    for (uint32_t i = 0; i < pattern1.size(); i++) {
        if (pattern1[i].op_name != pattern2[i].op_name)
            return false;

        if (pattern1[i].check_attrs and pattern2[i].check_attrs) {
            // If both side want to check attrs, then they must be the same
            TT_ASSERT(pattern1[i].attrs.size() == pattern2[i].attrs.size());
            TT_ASSERT(pattern1[i].op_name == "transpose", "Only support attrs check for transpose op");

            bool dim1_match_dim1 = pattern1[i].attrs[0] == pattern2[i].attrs[0];
            bool dim1_match_dim2 = pattern1[i].attrs[0] == pattern2[i].attrs[1];

            bool dim2_match_dim1 = pattern1[i].attrs[1] == pattern2[i].attrs[0];
            bool dim2_match_dim2 = pattern1[i].attrs[1] == pattern2[i].attrs[1];

            if (not ((dim1_match_dim1 and dim2_match_dim2) or (dim2_match_dim1 and dim1_match_dim2)))
                return false;
        }
    }

    return true;
}

std::vector<std::pair<graphlib::Shape, TMPattern>> replacement_output_shape(graphlib::Shape input_shape, const std::vector<TMPattern>& patterns) {
    py::object eval_module = py::module_::import("pybuda.op.eval.pybuda");

    std::vector<std::pair<graphlib::Shape, TMPattern>> shapes_patterns;

    for (TMPattern pattern : patterns) {
        for (uint32_t i = 0; i < pattern.size(); i++) {
            auto op_name = pattern[i].op_name;
            auto attrs = pattern[i].attrs;

            py::function pybuda_shape = eval_module.attr("get_f_pybuda_shape")(pattern[i].as_op_type());
            std::vector<std::vector<std::uint32_t>> operand_tuples;
            operand_tuples.push_back(input_shape.as_vector());
            py::tuple ret = pybuda_shape(operand_tuples);
            graphlib::Shape shape = graphlib::Shape::create(ret[0].cast<std::vector<std::uint32_t>>());
            input_shape = shape;
        }
        shapes_patterns.push_back(std::make_pair(input_shape, pattern));
    }
    return shapes_patterns;
}

std::string pattern_to_string(const TMPattern& pattern) {
    std::stringstream ss;
    for (uint i = 0; i < pattern.size(); i++) {
       ss << pattern[i].op_name;
       if (pattern[i].attrs.size() > 0) {
            ss << "(";
            for (auto attr : pattern[i].attrs) {
                ss << attr << ",";
            }
            ss << ")";
       }
       if (i < pattern.size() - 1)
            ss << "-->";
    }
    return ss.str();
}

bool replace_pattern_with_new_pattern(
    tt::graphlib::Graph* graph,
    const TMPattern& current_pattern, 
    const TMPattern& replace_pattern, 
    graphlib::Node *sequence_producer, 
    std::vector<graphlib::OpNode *> pattern_sequence) {

    log_debug(LogTMFusion, "Trying to replace pattern from {} to {}.", pattern_to_string(current_pattern), pattern_to_string(replace_pattern));

    bool multiple_user = false;
    std::vector<graphlib::Node *> users;
    graphlib::Node * fuse_node = nullptr;
    graphlib::Node *terminal_node = pattern_sequence.back();
    pattern_sequence.pop_back();

    // Check whether the matched pattern has multiple user or not
    // if there are multiple user at the end of the pattern matched node and
    // multiple user are same op and same shape
    // then the matched pattern can be fused by using replace pattern
    // and other user nodes are connected to the fused op.
    for (auto current_node : pattern_sequence) {
        users = graph->users(current_node);
        if (users.size() > 1) {
            bool user_is_terminal_node = std::find(users.begin(), users.end(), terminal_node) != users.end();

            // If there is a fork in the middle of the matched pattern, we cannot fuse TMs
            if (!user_is_terminal_node) {
                log_debug(LogTMFusion, "There is a fork in the middle of the matched pattern - cannot fuse tms.");
                return false;
            }

            OpType op_type = terminal_node->as<graphlib::OpNode>()->op_type();
            for (auto& user : users) {

                if (user->node_type() != graphlib::NodeType::kPyOp) {
                    // All users should be PyOps
                    return false;
                }

                if (user->as<graphlib::OpNode>()->op_type().op != op_type.op) {
                    // All users should be the same op
                    log_debug(LogTMFusion, "There is a user at the end of the matched pattern which is different op - cannot fuse tms.");
                    return false;
                }

                if (user->shape() != terminal_node->shape()) {
                    // All users should have the same shape
                    log_debug(LogTMFusion, "There is a user at the end of the matched pattern which is same op but different shape - cannot fuse tms.");
                    return false;
                }
            }
            multiple_user = true;
            break;

        }
    }

    // remove the edges of the users if it is same op and same shape
    if (multiple_user) {
        for (auto& user : users) {
            if (user != terminal_node) {
                auto edge_to_remove = graph->get_edges(pattern_sequence.back(), user)[0];
                graph->remove_edge(edge_to_remove);
            }
        }
    }
    // Bypass all nodes until the end of the current pattern
    // remove old pattern
   for (auto current_node : pattern_sequence) {
        TT_ASSERT(graph->users(current_node).size() == 1);
        bypass_node(graph, current_node, true);
    }

    TT_ASSERT(graph->get_edges(sequence_producer, terminal_node).size() == 1);
    auto current_edge = graph->get_edges(sequence_producer, terminal_node)[0];
    for (uint i = 0; i < replace_pattern.size(); i++) {
        auto op = replace_pattern[i];
        std::string name = sequence_producer->name() + "_fused_tm_op_" + std::to_string(current_edge.edge_creation_id);
        auto new_node = graph->add_node(
            std::make_unique<graphlib::PyOpNode>(name, op.as_op_type()), graph->get_subgraph_id_for_node(sequence_producer->id()));
        fuse_node = new_node;
        auto [new_in_edge, new_out_edge] = graphlib::insert_node_on_edge(graph, current_edge, new_node);
        current_edge = new_out_edge;
    }

    // Remove the final node
    bypass_node(graph, terminal_node, true);

    // connect the edge of the users to the fused op
    if (multiple_user) {
        for (auto& user : users){
            if (user != terminal_node)
                graph->add_edge(fuse_node, user);
        }
    }

    recalculate_shapes(graph);
    log_info(LogTMFusion, "Found replaceable TM sequence. Fuse from {} tms into {} tms.", current_pattern.size(), replace_pattern.size());
    return true;
}


bool fuse_tm_sequences(tt::graphlib::Graph* graph,TMPatternPairs& pattern_map) {

    // Want to match the largest patterns first
    std::sort(pattern_map.begin(), pattern_map.end(), [](const std::pair<TMPattern, std::vector<TMPattern>> &a, const std::pair<TMPattern, std::vector<TMPattern>> &b) {
        return a.first.size() > b.first.size();
    });

    bool updated = true;
    py::object eval_module = py::module_::import("pybuda.op.eval.pybuda");
    py::function is_tm = eval_module.attr("is_tm");
    bool updated_anything = false;
    while (updated) {
        updated = false;

        // Loop through pre-defined TM patterns
        for (auto & pattern : pattern_map) {
            auto search_pattern = pattern.first;
            auto replace_patterns = pattern.second;

            TMPattern current_pattern;
            std::vector<graphlib::OpNode*> pattern_sequence;
            graphlib::Node * sequence_producer = nullptr;
            graphlib::Shape sequence_input_shape;
            bool potential_prefix = true;
            graphlib::OpNode *blocking_eltwise = nullptr;
            graphlib::OpNode *forked_tm = nullptr;

            // Topological traversal to find the search pattern
            for (auto *node : graphlib::topological_sort(*graph))
            {
                graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(node);
                if (not op)
                    continue;

                if (op->as<graphlib::TaggedNode>()->tag_value_or("dont_erase", false))
                    continue;

                if ((is_eltwise(op) or is_quantization_ops(op)) and current_pattern.size() > 0){
                    blocking_eltwise = op;
                    pattern_sequence.push_back(op);
                    continue;
                }

                if (not is_tm(op->op_type()).cast<bool>())
                {
                    // Clear and try find another viable candidate
                    current_pattern.clear();
                    pattern_sequence.clear();
                    sequence_producer = nullptr;
                    potential_prefix = true;
                    blocking_eltwise = nullptr;
                    forked_tm = nullptr;
                    continue;
                }

                if (current_pattern.size() == 0) {
                    // Found a potential starting point of the pattern
                    sequence_input_shape = graph->operands(op)[0]->shape();
                    sequence_producer = graph->operands(op)[0];
                }

                if (graph->data_users(op).size() > 1) {
                    forked_tm = op;
                }

                current_pattern.emplace_back(op->op_type(), true);
                pattern_sequence.push_back(op);

                // Check for match
                if (current_pattern.size() > search_pattern.size()) {
                    // Clear and try find another viable candidate
                    current_pattern.clear();
                    pattern_sequence.clear();
                    sequence_producer = nullptr;
                    blocking_eltwise = nullptr;
                    forked_tm = nullptr;
                    potential_prefix = true;
                    continue;

                } else if (current_pattern.size() < search_pattern.size()) {
                    // Check if current pattern is a prefix of search pattern
                    for (uint32_t i = 0; i < current_pattern.size(); i++) {
                        if (current_pattern[i].op_name != search_pattern[i].op_name) {
                            potential_prefix = false;
                            break;
                        }
                    }

                    if (not potential_prefix) {
                        // Try find another viable candidate
                        current_pattern.clear();
                        pattern_sequence.clear();
                        sequence_producer = nullptr;
                        blocking_eltwise = nullptr;
                        forked_tm = nullptr;
                        potential_prefix = true;
                        continue;
                    }
                } else if (current_pattern.size() == search_pattern.size()) {
                    // Check if current pattern matches search pattern
                    bool same_pattern = equivalent_pattern(current_pattern, search_pattern);


                    if (forked_tm and forked_tm != pattern_sequence.back() and same_pattern) {
                        bool forked_tm_after_eltwise = false;
                        std::vector<graphlib::OpNode *> tms_to_move;
                        for (graphlib::OpNode *seq_op : pattern_sequence) {
                            if (not is_tm(seq_op->op_type()).cast<bool>()) {
                                forked_tm_after_eltwise = true;
                                break;
                            }
                            TT_ASSERT(is_tm(seq_op->op_type()).cast<bool>());
                            
                            tms_to_move.push_back(seq_op);
                            if (seq_op == forked_tm)
                                break;
                        }
                        if (not forked_tm_after_eltwise)
                        {
                            clone_tms_on_forks(graph, tms_to_move);
                            current_pattern.clear();
                            pattern_sequence.clear();
                            sequence_producer = nullptr;
                            blocking_eltwise = nullptr;
                            forked_tm = nullptr;
                            updated = true;
                            updated_anything = true;
                            potential_prefix = true;
                            continue;
                        }
                    }

                    // Check if there is a blocking eltwise, if there is then move all tms in the pattern above down
                    if (blocking_eltwise and same_pattern) {
                        std::vector<graphlib::OpNode *> tms_to_move;
                        for (graphlib::OpNode *seq_op : pattern_sequence) {
                            if (seq_op == blocking_eltwise)
                                break;

                            if (is_tm(seq_op->op_type()).cast<bool>())
                                tms_to_move.push_back(seq_op);
                        }
                        move_down_through_eltwise(graph, tms_to_move);
                        current_pattern.clear();
                        pattern_sequence.clear();
                        sequence_producer = nullptr;
                        blocking_eltwise = nullptr;
                        forked_tm = nullptr;
                        updated = true;
                        updated_anything = true;
                        potential_prefix = true;
                        continue;
                    }

                    // Verify i/o shape by calling pybuda shape function
                    auto shapes_patterns = replacement_output_shape(sequence_input_shape, replace_patterns);
                    
                    bool found_match = false;
                    TMPattern matching_pattern;
                    for (auto shape_pattern : shapes_patterns) {
                        graphlib::Shape output_shape = shape_pattern.first;
                        TMPattern pattern = shape_pattern.second;
                        if (output_shape == op->shape()) {
                            found_match = true;
                            matching_pattern = pattern;
                            break;
                        }
                    }

                    // Make sure output shape is the same after replacement
                    if (same_pattern and found_match) {
                        // Replace current pattern with replace pattern
                        bool is_pattern_replaced = replace_pattern_with_new_pattern(graph, current_pattern, matching_pattern, sequence_producer, pattern_sequence);
                        // Break and reset
                        current_pattern.clear();
                        pattern_sequence.clear();
                        sequence_producer = nullptr;
                        blocking_eltwise = nullptr;
                        updated = is_pattern_replaced;
                        if (is_pattern_replaced)
                            updated_anything = is_pattern_replaced;
                        potential_prefix = true;
                        continue;
                    }
                }
            }
        }
    }
    return updated_anything;

}

}  // namespace tt::passes
