// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <utility>

#include "third_party/json/json.hpp"
#include "pattern_matcher.hpp"

using json = nlohmann::json;
// fwd declare
namespace tt::graphlib {
    class Graph;
}

using Graph = tt::graphlib::Graph;

namespace pattern_matcher {

struct MatchResult {
    bool is_subgraph_pattern_found;
    bool is_subgraph_loopable;
    SubgraphPatternMatchMappings subgraph_matches;
};

// 1. Discover the largest subgraph in the graph containing exactly `num_matches_to_search`
// 2. If we can "roll" the subgraph, we compact the graph with a subgraph + loops.
std::pair<Graph*, MatchResult> lower_pybuda_to_pattern_matcher(Graph* graph, int num_matches_to_search);
MatchResult lower_json_to_pattern_matcher(json json_graph, int num_matches_to_search);

} // namespace pattern_matcher
