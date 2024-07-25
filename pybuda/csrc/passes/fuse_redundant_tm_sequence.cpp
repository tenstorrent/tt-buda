// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/fuse_redundant_tm_sequence.hpp"


using tt::LogTMFusion;
namespace tt::passes
{

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
            for (uint32_t j = 0; j < 2; j++) {
                if (pattern1[i].attrs[j] != pattern2[i].attrs[j])
                    return false;
            }
        }
    }
    return true;
}

graphlib::Shape replacement_output_shape(graphlib::Shape input_shape, const TMPattern& pattern) {
    py::object eval_module = py::module_::import("pybuda.op.eval.pybuda");

    for (uint i = 0; i < pattern.size(); i++) {
        auto op_name = pattern[i].op_name;
        auto attrs = pattern[i].attrs;

        py::function pybuda_shape = eval_module.attr("get_f_pybuda_shape")(pattern[i].as_op_type());
        std::vector<std::vector<std::uint32_t>> operand_tuples;
        operand_tuples.push_back(input_shape.as_vector());
        py::tuple ret = pybuda_shape(operand_tuples);
        graphlib::Shape shape = graphlib::Shape::create(ret[0].cast<std::vector<std::uint32_t>>());
        input_shape = shape;
    }
    return input_shape;
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
    graphlib::Node * terminal_node) {

    log_debug(LogTMFusion, "Trying to replace pattern from {} to {}.", pattern_to_string(current_pattern), pattern_to_string(replace_pattern));

    bool multiple_user = false;
    std::vector<graphlib::Node *> users;
    graphlib::Node * fuse_node = nullptr;

    // Check whether the matched pattern has multiple user or not
    // if there are multiple user at the end of the pattern matched node and
    // multiple user are same op and same shape
    // then the matched pattern can be fused by using replace pattern
    // and other user nodes are connected to the fused op.
    auto current_node = graph->users(sequence_producer)[0];
    while (current_node != terminal_node) {
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
        current_node = users[0];
    }

    // remove the edges of the users if it is same op and same shape
    if (multiple_user) {
        for (auto& user : users) {
            if (user != terminal_node) {
                auto edge_to_remove = graph->get_edges(current_node, user)[0];
                graph->remove_edge(edge_to_remove);
            }
        }
    }
    // Bypass all nodes until the end of the current pattern
    current_node = graph->users(sequence_producer)[0];

    // remove old pattern
    while (current_node != terminal_node) {
        TT_ASSERT(graph->users(current_node).size() == 1);
        auto next_node = graph->users(current_node)[0];
        bypass_node(graph, current_node, true);
        current_node = next_node;
    }

    TT_ASSERT(graph->get_edges(sequence_producer, terminal_node).size() == 1);
    auto current_edge = graph->get_edges(sequence_producer, terminal_node)[0];
    for (uint i = 0; i < replace_pattern.size(); i++) {
        auto op = replace_pattern[i];
        std::string name = sequence_producer->name() + "_fused_tm_op_" + std::to_string(i);
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

    bool updated = true;
    py::object eval_module = py::module_::import("pybuda.op.eval.pybuda");
    py::function is_tm = eval_module.attr("is_tm");
    bool updated_anything = false;
    while (updated) {
        updated = false;

        // Loop through pre-defined TM patterns
        for (auto & pattern : pattern_map) {
            auto search_pattern = pattern.first;
            auto replace_pattern = pattern.second;

            TMPattern current_pattern;
            graphlib::Node * sequence_producer = nullptr;
            graphlib::Shape sequence_input_shape;
            bool potential_prefix = true;

            // Topological traversal to find the search pattern
            for (auto *node : graphlib::topological_sort(*graph))
            {
                graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(node);
                if (not op)
                    continue;

                if (op->as<graphlib::TaggedNode>()->tag_value_or("dont_erase", false))
                    continue;

                if (not is_tm(op->op_type()).cast<bool>())
                {
                    // Clear and try find another viable candidate
                    current_pattern.clear();
                    sequence_producer = nullptr;
                    potential_prefix = true;
                    continue;
                }

                if (current_pattern.size() == 0) {
                    // Found a potential starting point of the pattern
                    sequence_input_shape = graph->operands(op)[0]->shape();
                    sequence_producer = graph->operands(op)[0];
                }

                current_pattern.emplace_back(op->op_type(), true);

                // Check for match
                if (current_pattern.size() > search_pattern.size()) {
                    // Clear and try find another viable candidate
                    current_pattern.clear();
                    sequence_producer = nullptr;
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
                        sequence_producer = nullptr;
                        potential_prefix = true;
                        continue;
                    }
                } else if (current_pattern.size() == search_pattern.size()) {
                    // Check if current pattern matches search pattern
                    bool same_pattern = equivalent_pattern(current_pattern, search_pattern);

                    // Verify i/o shape by calling pybuda shape function
                    graphlib::Shape output_shape = replacement_output_shape(sequence_input_shape, replace_pattern);

                    // Make sure output shape is the same after replacement
                    bool same_shape = output_shape == op->shape();
                    if (same_pattern and same_shape) {
                        // Replace current pattern with replace pattern
                        bool is_pattern_replaced = replace_pattern_with_new_pattern(graph, current_pattern, replace_pattern, sequence_producer, node);
                        // Break and reset
                        current_pattern.clear();
                        sequence_producer = nullptr;
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
