// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

namespace tt {

// Compare two pytorch tensors, and return true if they are equal
bool compare_tensors(std::shared_ptr<void> tensor0, std::shared_ptr<void> tensor1);

namespace graphlib {

class Graph;
class Node;

// Calculate node shape from operand shapes, using python callback
void calculate_and_set_node_shape(Graph *graph, Node *node);

}

void GraphModule(py::module &m_graph);

}


