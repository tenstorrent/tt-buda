// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <exception>

#include "graph_lib/graph.hpp"
#include "graph_lib/node.hpp"
#include "graph_lib/node_types.hpp"

namespace tt
{

using Graph = graphlib::Graph;
using Node = graphlib::Node;

struct UnsupportedHWOpsError : public std::exception
{
    std::string e;
    UnsupportedHWOpsError(std::string const &e) : e(e) {}
    UnsupportedHWOpsError(graphlib::OpType const &op_type) : e(op_type.as_string()) {}
    virtual char const *what() const noexcept override { return e.c_str(); }
};

// Run TM optimizations again, balancer might introduce tms so this needs to happen post placer
void optimize_tms(std::vector<graphlib::OpType> &tms);
void optimize_tms(Graph *graph);

// Recalculate all node shapes from inputs
void recalculate_shapes(graphlib::Graph *graph);

std::vector<int> get_factors(int num);  // !!! This function is unused !!!

bool check_unsupported_hw_ops(graphlib::Graph *graph, bool should_throw = false);

// Returns true if string is part of 2D vector of strings, false otherwise.
bool is_str_in_strings(const std::string &str, const std::vector<std::vector<std::string>> &strings);
}  // namespace tt
