// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "pattern_matcher/python_bindings.hpp"
#include "pattern_matcher/boost_lowering.hpp"

#include "graph_lib/graph.hpp"
#include "pybind11_json.hpp"

namespace tt {

void PatternMatcherModule(py::module &m_pattern_matcher) {
    using namespace pattern_matcher;

    py::class_<pattern_matcher::MatchResult>(m_pattern_matcher, "MatchResult")
        .def_readwrite("is_subgraph_pattern_found", &MatchResult::is_subgraph_pattern_found)
        .def_readwrite("is_subgraph_loopable", &MatchResult::is_subgraph_loopable)
        .def_readwrite("subgraph_matches", &MatchResult::subgraph_matches);

    m_pattern_matcher.def("lower_pybuda_to_pattern_matcher", &lower_pybuda_to_pattern_matcher);
    m_pattern_matcher.def("lower_json_to_pattern_matcher", &lower_json_to_pattern_matcher);
}

} // namespace tt

