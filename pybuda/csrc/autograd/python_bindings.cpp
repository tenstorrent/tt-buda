// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "autograd/python_bindings.hpp"

#include "autograd/autograd.hpp"
#include "graph_lib/defines.hpp"
#include "graph_lib/graph.hpp"
#include "graph_lib/node_types.hpp"
#include "python_bindings_common.hpp"

namespace tt {

static bool has_newstyle_interface(std::string const &op_name, bool is_buda)
{
    py::object eval_module =
        is_buda ? py::module_::import("pybuda.op.eval.buda") : py::module_::import("pybuda.op.eval.pybuda");
    return eval_module.attr("has_newstyle_interface")(op_name).cast<bool>();
}

void AutogradModule(py::module &m_autograd)
{
    py::class_<autograd2::autograd_config>(m_autograd, "AutogradConfig")
        .def(py::init<bool, py::object>(),
                py::arg("recompute") = false,
                py::arg("optimizer") = py::none());

    py::class_<autograd2::autograd2_engine>(m_autograd, "AutogradEngine")
      .def(py::init([](graphlib::Graph* graph, const autograd2::autograd_config& cfg) {
            return std::make_unique<autograd2::autograd2_engine>(graph, cfg);
        }))
      .def("run", &autograd2::autograd2_engine::run);

    py::class_<tt::autograd2::autograd_context>(m_autograd, "AutogradContext")
        .def(
            "op",
            [](tt::autograd2::autograd_context &self,
               std::variant<std::string, py::object> const &type,
               std::vector<tt::autograd2::NodeContext> operands,
               std::vector<graphlib::OpType::Attr> attributes)
            {
                graphlib::OpType op_type = std::holds_alternative<std::string>(type)
                                               ? graphlib::OpType(std::get<std::string>(type), attributes)
                                               : std::get<py::object>(type).attr("op_type").cast<graphlib::OpType>();

                if (std::holds_alternative<std::string>(type))
                    TT_LOG_ASSERT(
                        not has_newstyle_interface(std::get<std::string>(type), false),
                        "Error autograd a type with old OpType interface, expects new OpType interface {}",
                        std::get<std::string>(type));
                if (self.epoch_type == graphlib::NodeEpochType::Backward)
                {
                    return self.autograd->create_op(
                        op_type,
                        operands,
                        self.current_fwd_op,
                        self.operand,
                        self.created_op_index++);
                }
                else if (self.epoch_type == graphlib::NodeEpochType::Optimizer)
                {
                    return self.autograd->create_optimizer_op(
                        op_type,
                        operands,
                        self.current_fwd_op,
                        self.operand,
                        self.created_op_index++);
                }

                throw std::runtime_error("Expected autograd_context.epoch_type to be Backward or Optimizer");
            },
            py::arg("type"),
            py::arg("operands"),
            py::arg("attributes") = std::vector<graphlib::OpType::Attr>())
        .def(
            "create_optimizer_op",
            [](tt::autograd2::autograd_context &self,
               std::string type,
               std::vector<tt::autograd2::NodeContext> operands,
               std::vector<graphlib::OpType::Attr> attributes)
            {
                return self.autograd->create_optimizer_op(
                    graphlib::OpType(type, attributes),
                    operands,
                    self.current_fwd_op,
                    self.operand,
                    self.created_op_index++);
            },
            py::arg("type"),
            py::arg("operands"),
            py::arg("attributes") = std::vector<graphlib::OpType::Attr>())
        .def(
            "constant",
            [](tt::autograd2::autograd_context &self, int value)
            {
                return self.autograd->create_constant(
                    self.current_fwd_op, self.operand, value, self.created_op_index++, self.epoch_type);
            })
        .def(
            "constant",
            [](tt::autograd2::autograd_context &self, float value)
            {
                return self.autograd->create_constant(
                    self.current_fwd_op, self.operand, value, self.created_op_index++, self.epoch_type);
            })
        .def(
            "input",
            [](tt::autograd2::autograd_context &self,
               std::string input_name,
               std::vector<std::uint32_t> tensor_shape,
               bool copy_consteval_operations,
               bool disable_consteval)
            {
                // Note requires_grad = False
                return self.autograd->create_input(
                    self.current_fwd_op,
                    self.operand,
                    self.created_op_index++,
                    self.epoch_type,
                    input_name,
                    tensor_shape,
                    copy_consteval_operations,
                    disable_consteval);
            },
            py::arg("input_name"),
            py::arg("tensor_shape"),
            py::kw_only(),
            py::arg("copy_consteval_operations") = false,
            py::arg("disable_consteval") = false)
        .def(
            "loopback",
            [](tt::autograd2::autograd_context &self,
               tt::autograd2::NodeContext const &original,
               tt::autograd2::NodeContext const &update)
            {
                graphlib::Graph *graph = self.autograd->get_graph();
                graph->add_edge(
                    graph->node_by_id(original.id),
                    graph->node_by_id(update.id),
                    graphlib::PortId(0),
                    graphlib::PortId(0),
                    graphlib::EdgeType::kDataLoopback);
            })
        .def(
            "tensor",
            [](tt::autograd2::autograd_context &self, py::object tensor)
            {
                return self.autograd->create_constant(
                    self.current_fwd_op,
                    self.operand,
                    make_shared_py_object(tensor),
                    graphlib::Shape::create(tensor.attr("shape").cast<std::vector<std::uint32_t>>()),
                    self.created_op_index++,
                    self.epoch_type);
            })
        .def(
            "get_pytorch_tensor",
            [](tt::autograd2::autograd_context &self, tt::autograd2::NodeContext const &node)
            {
                graphlib::ConstantInputNode *cnode =
                    dynamic_cast<graphlib::ConstantInputNode *>(self.autograd->get_graph()->node_by_id(node.id));
                TT_ASSERT(cnode && cnode->is_tensor(), "Only use for ConstantInputNode of type tensor");
                return borrow_shared_py_object(cnode->tensor());
            })
        .def(
            "get_shape",
            [](tt::autograd2::autograd_context &self, tt::autograd2::NodeContext &node)
            { return self.autograd->get_graph()->node_by_id(node.id)->shape().as_vector(); })
        .def(
            "get_operands",
            [](tt::autograd2::autograd_context &self, tt::autograd2::NodeContext const &node)
            {
                graphlib::Graph *graph = self.autograd->get_graph();
                std::vector<tt::autograd2::Node *> operands = graph->data_operands(graph->node_by_id(node.id));
                std::vector<tt::autograd2::NodeContext *> operand_contexts;
                for (auto it = operands.begin(); it != operands.end(); ++it)
                {
                    tt::autograd2::NodeContext *op_context = new tt::autograd2::NodeContext(*it);
                    operand_contexts.push_back(op_context);
                }
                return operand_contexts;
            });
}

} // namespace tt
