// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "graph_lib/python_bindings.hpp"

#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <optional>

#include "autograd/autograd.hpp"
#include "balancer/balancer.hpp"
#include "graph_lib/graph.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/query.hpp"
#include "graph_lib/utils.hpp"
#include "json.hpp"
#include "passes/fuse_ops.hpp"
#include "pybind11_json.hpp"
#include "python_bindings_common.hpp"
#include "reportify/reportify.hpp"
#include "utils/logger.hpp"
#include "utils/raw_ptr.hpp"

using json = nlohmann::json;

PYBIND11_DECLARE_HOLDER_TYPE(T, tt::raw_ptr<T>);

namespace tt {

using Graph = graphlib::Graph;
using Node = graphlib::Node;
using NodeType = graphlib::NodeType;
using Shape = graphlib::Shape;

// Ideally, this would be in graphlib... but being in a different shared object causes import errors at runtime :(
std::tuple<std::vector<py::object>,std::unordered_map<std::string, py::object>,std::vector<py::object>,std::unordered_map<std::string, py::object>,std::unordered_map<std::string, py::object>>
eval_graph(
    Graph *graph,
    const std::vector<py::object> &inputs,
    const std::unordered_map<std::string, py::object> &parameters,
    py::object tt_device,
    const std::unordered_map<int, py::object> &intermediate_golden_tensors,
    const std::vector<py::object> &losses,
    const std::vector<py::object> &targets,
    std::shared_ptr<balancer::BalancerSolution> balancer_solution,
    float relative_atol,
    float pcc,
    std::string const &dump_tensors_path,
    bool allow_modified_shapes,
    bool return_intermediates);

py::object get_constant_input_value(graphlib::Node *node, bool is_buda);
py::object eval_input(
    Node *node,
    std::unordered_map<std::string, py::object> inputs,
    bool is_buda,
    graphlib::NodeEpochType epoch_type = graphlib::NodeEpochType::Forward);
py::object eval_input_bw(Node *node, py::object inputs, bool is_buda);

void GraphModule(py::module &m_graph)
{
    py::class_<Graph>(m_graph, "Graph")
        .def(py::init([](std::string name) { return std::make_unique<Graph>(graphlib::IRLevel::IR_PYBUDA, name); }))
        .def("clone", [](Graph &self) { return self.clone(); })
        .def("get_node_name", [](const Graph &self, const graphlib::NodeId id) { return self.node_by_id(id)->name(); })
        .def("get_name", [](const Graph &self) { return self.name(); })
        .def(
            "get_node_id", [](const Graph &self, const std::string &name) { return self.get_node_by_name(name)->id(); })
        .def("has_node_with_id", &Graph::has_node_with_id)
        .def("set_enable_training", &Graph::set_enable_training)
        .def("enable_training", &Graph::enable_training)
        .def("set_microbatch", &Graph::set_microbatch)
        .def("get_microbatch", &Graph::get_microbatch)
        .def("nodes", [](const Graph &self) { 
            std::vector<graphlib::Node*> nodes = self.nodes();
            std::vector<std::string> names;
            std::transform(nodes.begin(), nodes.end(), std::back_inserter(names), [](graphlib::Node* node) {
                return node->name();
            });
            return names;
        })
        .def("get_ordered_input_names", &Graph::get_ordered_input_names)
        .def("get_ordered_intermediate_names", &Graph::get_ordered_intermediate_names)
        .def("get_ordered_output_names", &Graph::get_ordered_output_names)
        .def("get_ordered_target_names", &Graph::get_ordered_target_names)
        .def("get_ordered_input_gradient_names", &Graph::get_ordered_input_gradient_names)
        .def("get_ordered_output_gradient_names", &Graph::get_ordered_output_gradient_names)
        .def("get_ordered_input_requires_grad", &Graph::get_ordered_input_requires_grad)
        .def("get_ordered_output_requires_grad", &Graph::get_ordered_output_requires_grad)
        .def("get_ordered_input_subgraph_indices", &Graph::get_ordered_input_subgraph_indices)
        .def("get_ordered_target_subgraph_indices", &Graph::get_ordered_target_subgraph_indices)
        .def("get_ordered_output_subgraph_indices", &Graph::get_ordered_output_subgraph_indices)
        .def("get_constant_names", &Graph::get_constant_names)
        .def(
            "get_constant_nodes",
            &Graph::get_constant_nodes,
            py::return_value_policy::reference,
            py::arg("recurse") = false)
        .def("get_subgraph_id_for_node", &Graph::get_subgraph_id_for_node)
        .def("get_parameter_nodes", &Graph::get_parameter_nodes, py::return_value_policy::reference)
        .def("register_module_inputs", &Graph::register_module_inputs, py::arg("module_inputs"), py::arg("append") = false)
        .def("register_module_outputs", &Graph::register_module_outputs, py::arg("module_outputs"), py::arg("requires_grad"), py::arg("append") = false)
        .def("register_module_targets", &Graph::register_module_targets)
        .def("get_ordered_input_shapes", &Graph::get_ordered_input_shapes)
        .def("get_ordered_output_shapes", &Graph::get_ordered_output_shapes)
        .def("output_node_redirected", &Graph::get_output_node_redirected)
        .def("get_ordered_target_shapes", &Graph::get_ordered_target_shapes)
        .def("get_ordered_intermediate_shapes", &Graph::get_ordered_intermediate_shapes)
        .def("get_tile_broadcast_dims_for_input", &Graph::get_tile_broadcast_dims_for_input)
        .def("get_tile_broadcast_dims_for_bw_input", &Graph::get_tile_broadcast_dims_for_bw_input)
        .def("get_tile_broadcast_dims_for_target", &Graph::get_tile_broadcast_dims_for_target)
        .def("get_ordered_input_tile_dims", &Graph::get_ordered_input_tile_dims)
        .def("get_ordered_parameter_tile_dims", &Graph::get_ordered_parameter_tile_dims)
        .def("get_ordered_constant_tile_dims", &Graph::get_ordered_constant_tile_dims)
        .def(
            "get_input_runtime_tensor_transforms",
            [](Graph *graph) -> std::vector<graphlib::RuntimeTensorTransform>
            {
                std::vector<graphlib::RuntimeTensorTransform> transforms;
                auto inputs = graph->ordered_module_inputs();
                for (int i = 0; i < (int)inputs.size(); ++i)
                {
                    transforms.push_back(inputs[i]->as<graphlib::InputNode>()->get_runtime_tensor_transform());
                }
                return transforms;
            })
        .def(
            "get_constant_input_runtime_tensor_transform_constants",
            [](Graph *graph) -> std::vector<std::tuple<std::string, py::object>>
            {
                std::vector<std::tuple<std::string, py::object>> constants;
                auto inputs = graph->ordered_module_inputs();
                for (int i = 0; i < (int)inputs.size(); ++i)
                {
                    graphlib::RuntimeTensorTransform transform = inputs[i]->as<graphlib::InputNode>()->get_runtime_tensor_transform();
                    if (transform.type == graphlib::RuntimeTensorTransformType::ConstantInput)
                    {
                        constants.push_back(std::make_tuple(inputs[i]->name(), transform.get_constant_input_tensor()));
                    }
                }
                return constants;
            })
        .def(
            "get_output_runtime_tensor_transforms",
            [](Graph *graph) -> std::vector<graphlib::RuntimeTensorTransform>
            {
                std::vector<graphlib::RuntimeTensorTransform> transforms;
                auto outputs = graph->ordered_module_outputs();
                for (int i = 0; i < (int)outputs.size(); ++i)
                {
                    tt::graphlib::RuntimeTensorTransform transform = outputs[i]->as<graphlib::OutputNode>()->get_runtime_tensor_transform();
                    transform.swap_original_and_reinterpreted_shapes();
                    transforms.push_back(transform);
                }
                return transforms;
            })
        // Return information about fused ops and their schedule. Currently used purely for test verification,
        // i.e. to ensure that fusing occured exactly in the way that was expected
        .def(
            "get_fused_ops",
            [](Graph *graph)
            {
                std::vector<std::tuple<
                    std::uint32_t,                         // inputs
                    std::vector<std::vector<std::string>>  // schedules
                    >>
                    ret;

                for (Node *node : graph->nodes())
                {
                    if (node->node_type() != graphlib::kBudaOp)
                        continue;
                    graphlib::BudaOpNode *op = node->as<graphlib::BudaOpNode>();
                    if (!op->is_fused_op())
                        continue;

                    auto f = op->get_fused_op();
                    auto s = f->get_schedules();
                    std::vector<std::vector<std::string>> schedules;
                    for (auto sch : s)
                    {
                        std::vector<std::string> schedule;
                        for (FusedSubOp subop : sch.ops) schedule.push_back(subop.op_type.op);
                        schedules.push_back(schedule);
                    }

                    std::tuple<
                        std::uint32_t,                         // inputs
                        std::vector<std::vector<std::string>>  // schedules
                        >
                        op_data = {f->get_input_count(), schedules};
                    ret.push_back(op_data);
                }
                return ret;
            });
    py::class_<Shape> shape(m_graph, "Shape");

    shape.def_property_readonly("v", [](Shape const &shape) { return shape[-5]; })
        .def_property_readonly("w", [](Shape const &shape) { return shape[-4]; })
        .def_property_readonly("z", [](Shape const &shape) { return shape[-3]; })
        .def_property_readonly("r", [](Shape const &shape) { return shape[-2]; })
        .def_property_readonly("c", [](Shape const &shape) { return shape[-1]; })
        .def_property_readonly("rt", &Shape::rt)
        .def_property_readonly("ct", &Shape::ct)
        .def("get_tile_dim", &Shape::get_tile_dim)
        .def("get_tile_height", &Shape::get_tile_height)
        .def("get_tile_width", &Shape::get_tile_width)
        .def_static("create", &Shape::create, py::arg("values"))
        .def_static(
            "create_buda", py::overload_cast<std::vector<std::uint32_t>, int, int>(&Shape::create_buda))
        .def_static(
            "create_with_type_from_other", Shape::create_with_type_from_other, py::arg("other"), py::arg("values"))
        .def("len", [](Shape const &shape) { return shape.as_vector().size(); })
        .def("__len__", [](Shape const &shape) { return shape.as_vector().size(); })
        .def("__iter__", [](Shape &shape) {
            return py::make_iterator(shape.begin(), shape.end()); 
        }, py::keep_alive<0, 1>())
        .def("as_list", [](Shape const &shape) { return shape.as_vector(); })
        .def("__getitem__", [](Shape const &shape, int idx) { return shape[idx]; })
        .def("__setitem__", [](Shape &shape, int idx, std::uint32_t val) { shape[idx] = val; })
        .def(
            "__repr__",
            [](Shape const &shape)
            {
                std::stringstream ss;
                ss << shape;
                return ss.str();
            })
        .def(py::self == py::self)
        .def(py::pickle(
            [](const Shape &shape) {  // __getstate__
                std::tuple<bool, int, std::vector<std::uint32_t>> pickle_data = shape.get_pickle_data();
                return py::make_tuple(std::get<0>(pickle_data), std::get<1>(pickle_data), std::get<2>(pickle_data));
            },
            [](py::tuple t) {  // __setstate__
                if (t.size() != 3)
                    throw std::runtime_error("Shape: Invalid state!");

                return Shape::create_from_pickled(
                    t[0].cast<bool>(),
                    t[1].cast<int>(),
                    t[2].cast<std::vector<std::uint32_t>>());
            }))
        .def("__get_pickle_data", &Shape::get_pickle_data)
        .def_static("__create_from_pickled", &Shape::create_from_pickled)
        .def("to_json", [](Shape const &shape) { json j = shape; return j; })
        .def_static("from_json", [](json j) { return j.get<Shape>(); });

    py::enum_<Shape::Type>(shape, "Type")
        .value("FREE", Shape::Type::FREE)
        .value("BUDA", Shape::Type::BUDA)
        .export_values();

    py::class_<tt::Node, tt::raw_ptr<tt::Node>>(m_graph, "Node")
        .def_property_readonly("id", &Node::id)
        .def_property_readonly("name", &Node::name)
        .def_property_readonly("node_type", &Node::node_type)
        .def_property_readonly("shape", &Node::shape)
        .def_property_readonly("output_df", &Node::output_df);

    py::class_<tt::graphlib::InputNode, tt::raw_ptr<tt::graphlib::InputNode>>(m_graph, "InputNode")
        .def_property_readonly("id", &Node::id)
        .def_property_readonly("name", &Node::name)
        .def_property_readonly("node_type", &Node::node_type)
        .def_property_readonly("shape", &Node::shape)
        .def_property_readonly("output_df", &Node::output_df);

    py::class_<graphlib::NodeContext>(m_graph, "NodeContext")
        .def_readonly("id", &graphlib::NodeContext::id)
        .def_readonly("name", &graphlib::NodeContext::name)
        .def_readonly("node_type", &graphlib::NodeContext::type)
        .def_readonly("shape", &graphlib::NodeContext::shape)
        .def_readonly("unbroadcast_shape", &graphlib::NodeContext::unbroadcast_shape)
        .def_readonly("output_df", &graphlib::NodeContext::output_df);

    py::class_<tt::graphlib::OpType, tt::raw_ptr<tt::graphlib::OpType>>(m_graph, "OpType")
        .def(
            py::init([](std::string const &op,
                        std::vector<tt::graphlib::OpType::Attr> const &attr,
                        tt::graphlib::OpType::Attrs const &named_attrs)
                     { return graphlib::OpType(op, attr, {}, named_attrs); }),
            py::arg("op"),
            py::arg("attr") = std::vector<tt::graphlib::OpType::Attr>{},
            py::arg("named_attrs") = tt::graphlib::OpType::Attrs{})
        .def_readonly("op", &tt::graphlib::OpType::op)
        .def_readonly("attr", &tt::graphlib::OpType::attr)
        .def_readonly("named_attrs", &tt::graphlib::OpType::named_attrs)
        .def_readonly("buda_attrs", &tt::graphlib::OpType::buda_attrs)
        .def(
            "set_buda_attr",
            [](tt::graphlib::OpType &op_type, std::string const &name, tt::graphlib::OpType::Attr const &attr)
            { op_type.buda_attrs[name] = attr; })
        .def(
            "__getattr__",
            [](tt::graphlib::OpType const &op_type, std::string const &name) { return op_type.get_attr(name); })
        .def(
            "__setattr__",
            [](tt::graphlib::OpType &op_type, std::string const &name, tt::graphlib::OpType::Attr value)
            { return op_type.set_attr(name, value); })
        .def("__repr__", [](tt::graphlib::OpType const &op_type) { return op_type.as_string(); });

    py::enum_<tt::graphlib::NodeType>(m_graph, "NodeType")
        .value("kInput", tt::graphlib::NodeType::kInput)
        .value("kOutput", tt::graphlib::NodeType::kOutput)
        .value("kPyOp", tt::graphlib::NodeType::kPyOp)
        .value("kBudaOp", tt::graphlib::NodeType::kBudaOp)
        .value("kBudaNaryTM", tt::graphlib::NodeType::kBudaNaryTM)
        .value("kQueue", tt::graphlib::NodeType::kQueue)
        .export_values();

    py::enum_<tt::graphlib::UBlockOrder>(m_graph, "UBlockOrder")
        .value("R", tt::graphlib::UBlockOrder::R)
        .value("C", tt::graphlib::UBlockOrder::C)
        .export_values();

    py::enum_<tt::graphlib::RuntimeTensorTransformType>(m_graph, "RuntimeTensorTransformType")
        .value("NoTransform", tt::graphlib::RuntimeTensorTransformType::NoTransform)
        .value("ReinterpretShape", tt::graphlib::RuntimeTensorTransformType::ReinterpretShape)
        .value("Prestride", tt::graphlib::RuntimeTensorTransformType::Prestride)
        .value("EmbeddingIndex", tt::graphlib::RuntimeTensorTransformType::EmbeddingIndex)
        .value("ConstantInput", tt::graphlib::RuntimeTensorTransformType::ConstantInput)
        .value("Unpad", tt::graphlib::RuntimeTensorTransformType::Unpad)
        .value("Concatenate", tt::graphlib::RuntimeTensorTransformType::Concatenate)
        .export_values();

    py::class_<tt::graphlib::RuntimeTensorTransform>(m_graph, "RuntimeTensorTransform")
        .def(py::init<>())
        .def("EmbeddingIndex", &tt::graphlib::RuntimeTensorTransform::EmbeddingIndex)
        .def_readwrite("type", &tt::graphlib::RuntimeTensorTransform::type)
        .def_readwrite("original_shape", &tt::graphlib::RuntimeTensorTransform::original_shape)
        .def_readwrite("reinterpreted_shape", &tt::graphlib::RuntimeTensorTransform::reinterpreted_shape)
        .def_readwrite("unpadded_shape", &tt::graphlib::RuntimeTensorTransform::unpadded_shape)
        .def_readwrite("stride_height", &tt::graphlib::RuntimeTensorTransform::stride_height)
        .def_readwrite("stride_width", &tt::graphlib::RuntimeTensorTransform::stride_width)
        .def_readwrite("kernel_height", &tt::graphlib::RuntimeTensorTransform::kernel_height)
        .def_readwrite("kernel_width", &tt::graphlib::RuntimeTensorTransform::kernel_width)
        .def_readwrite("concat_group", &tt::graphlib::RuntimeTensorTransform::concat_group)
        .def_readwrite("concat_index", &tt::graphlib::RuntimeTensorTransform::concat_index)
        .def_readwrite("concat_dim", &tt::graphlib::RuntimeTensorTransform::concat_dim)
        .def(py::pickle(
            [](const tt::graphlib::RuntimeTensorTransform &transform) {  // __getstate__
                return py::make_tuple(
                    transform.type,
                    transform.original_shape,
                    transform.reinterpreted_shape,
                    transform.unpadded_shape,
                    transform.stride_height,
                    transform.stride_width,
                    transform.concat_group,
                    transform.concat_index,
                    transform.concat_dim
                );
            },
            [](py::tuple t) {  // __setstate__
                if (t.size() != 9)
                    throw std::runtime_error("tt::graphlib::RuntimeTensorTransform: Invalid state!");

                // TODO: Covers only ReinterpretShape
                tt::graphlib::RuntimeTensorTransform transform {};
                transform.type = t[0].cast<tt::graphlib::RuntimeTensorTransformType>();
                transform.original_shape = t[1].cast<tt::graphlib::Shape>();
                transform.reinterpreted_shape = t[2].cast<tt::graphlib::Shape>();
                transform.unpadded_shape = t[3].cast<tt::graphlib::Shape>();
                transform.stride_height = t[4].cast<int>();
                transform.stride_width = t[5].cast<int>();
                transform.concat_group = t[6].cast<int>();
                transform.concat_index = t[7].cast<int>();
                transform.concat_dim = t[8].cast<int>();

                return transform;
            }))
        .def("to_json", [](tt::graphlib::RuntimeTensorTransform const& rtt) { json j = rtt; return j; })
        .def("from_json", [](json const& j) { return j.get<tt::graphlib::RuntimeTensorTransform>(); });

    m_graph.def(
        "create_op_node",
        [](Graph *graph,
           const std::string &name,
           const graphlib::OpType &op_type,
           const std::vector<std::uint32_t> &shape,
           tt::DataFormat data_format,
           const int subgraph_index,
           graphlib::TagHints tags)
        {
            auto node = graph->add_node(graphlib::create_node<graphlib::PyOpNode>(name, op_type), subgraph_index);
            node->set_shape(Shape::create(shape));
            node->set_output_df(data_format);
            node->as<graphlib::TaggedNode>()->tag("original_op_name", name);
            node->as<graphlib::TaggedNode>()->tag("original_op_type", op_type.op);
            node->as<graphlib::TaggedNode>()->add_tags(tags);
            return node->id();
        });
    m_graph.def("create_parameter_input", 
    [](Graph *graph, const std::string &name, const std::vector<std::uint32_t> &shape, bool requires_grad, tt::DataFormat data_format, const int subgraph_index) {
        auto node = graph->add_node(graphlib::create_node<graphlib::InputNode>(
                    name, 
                    graphlib::InputNodeType::Parameter, 
                    requires_grad), subgraph_index);
        node->set_shape(Shape::create(shape));
        node->set_output_df(data_format);
        node->as<graphlib::TaggedNode>()->tag("original_op_name", name);
        return node->id();
    });
    m_graph.def("create_activation_input", 
    [](Graph *graph, const std::string &name, const std::vector<std::uint32_t> &shape, bool requires_grad, tt::DataFormat data_format, const int subgraph_index) {
        auto node = graph->add_node(graphlib::create_node<graphlib::InputNode>(
                    name,
                    graphlib::InputNodeType::Activation, 
                    requires_grad), subgraph_index);
        node->set_shape(Shape::create(shape));
        node->set_output_df(data_format);
        node->as<graphlib::TaggedNode>()->tag("original_op_name", name);
        return node->id();
    });
    m_graph.def("create_target_input", 
    [](Graph *graph, const std::string &name, const std::vector<std::uint32_t> &shape, bool requires_grad, tt::DataFormat data_format, const int subgraph_index) {
        auto node = graph->add_node(graphlib::create_node<graphlib::InputNode>(
                    name,
                    graphlib::InputNodeType::Target, 
                    requires_grad), subgraph_index);
        node->set_shape(Shape::create(shape));
        node->set_output_df(data_format);
        node->as<graphlib::TaggedNode>()->tag("original_op_name", name);
        return node->id();
    });
    m_graph.def("create_constant_input", [](Graph *graph, const std::string &name, float constant_value, tt::DataFormat data_format, const int subgraph_index) {
        auto node = graph->add_node(graphlib::create_node<graphlib::ConstantInputNode>(
                    name,
                    constant_value), subgraph_index);
        node->set_shape(Shape::create({1}));
        node->set_output_df(data_format);
        return node->id();
    });
    m_graph.def("create_constant_input", 
    [](Graph *graph, const std::string &name, py::object constant_value, const std::vector<std::uint32_t> &shape, tt::DataFormat data_format, const int subgraph_index) {
        auto node = graph->add_node(graphlib::create_node<graphlib::ConstantInputNode>(
                    name,
                    make_shared_py_object(constant_value),
                    Shape::create(shape)), subgraph_index);
        node->set_output_df(data_format);
        return node->id();
    });
    m_graph.def("create_output", 
    [](Graph *graph, const std::string &name, const std::vector<std::uint32_t> &shape, tt::DataFormat data_format, bool is_loss_output, const int subgraph_index) {
        auto node = graph->add_node(graphlib::create_node<graphlib::OutputNode>(name), subgraph_index);
        node->set_shape(Shape::create(shape));
        node->set_output_df(data_format);
        if (is_loss_output) node->set_loss_output();
        return node->id();
    });

    m_graph.def("get_constant_input_value", &get_constant_input_value);

    m_graph.def(
        "get_shape_for_node",
        [](Graph *graph, const std::string &name) -> std::vector<std::uint32_t>
        {
            auto node = graph->get_node_by_name(name);
            return node->shape().as_vector();
        });

    m_graph.def("create_data_edge", [](
          Graph *graph,
          const graphlib::NodeId start,
          int out_port_id,
          const graphlib::NodeId end,
          int in_port_id,
          std::vector<py::tuple> operand_broadcast)
    {
        graphlib::Edge edge(start, (graphlib::PortId)out_port_id, end, (graphlib::PortId)in_port_id, graphlib::EdgeType::kData);
        graph->add_edge(edge);
        std::shared_ptr<graphlib::EdgeAttributes> attr = graph->get_edge_attributes(edge);

        for (const py::tuple &broadcast : operand_broadcast) {
            if (in_port_id == broadcast[0].cast<int>()) {
                int dim = broadcast[1].cast<int>();
                int size = broadcast[2].cast<int>();
                attr->set_broadcast_dim(dim, size);
            }
        }
    });

    m_graph.def("add_partial_datacopy_edge", [](
          Graph *graph,
          const graphlib::NodeId start,
          int out_port_id,
          const graphlib::NodeId end,
          int in_port_id)
    {
        graphlib::Edge edge(start, (graphlib::PortId)out_port_id, end, (graphlib::PortId)in_port_id, graphlib::EdgeType::kPartialDataCopy);
        graph->add_edge(edge);
        // Disable consteval for partial datacopy inputs
        graphlib::Node *input = graph->node_by_id(end);
        input->as<graphlib::TaggedNode>()->tag("dont_consteval", "true");
    });

    m_graph.def("create_control_edge", [](
          Graph *graph,
          const graphlib::NodeId start,
          int out_port_id,
          const graphlib::NodeId end,
          int in_port_id)
    {
        graph->add_edge(graphlib::Edge(start, (graphlib::PortId)out_port_id, end, (graphlib::PortId)in_port_id, graphlib::EdgeType::kControl));
    });

    m_graph.def("get_optimizer_param_info", [](
          Graph *graph,
          const std::string &param_name)
    {
        TT_ASSERT(graph->has_node_with_name(param_name),
                    "Module contains parameter name: " + param_name + " which doesn't exist in the graph.");
        Node *node = graph->get_node_by_name(param_name);
        return graphlib::get_optimizer_param_info(graph, node);
    }, py::return_value_policy::reference);

    m_graph.def("get_intermediate_tensors",
        [](Graph *graph,
            const std::vector<py::object> &inputs,
            const std::unordered_map<std::string, py::object> &parameters,
            py::object tt_device,
            float relative_atol, 
            float pcc,
            const std::unordered_map<int, py::object> &intermediate_golden_tensors,
            const std::vector<py::object> &losses,
            const std::vector<py::object> &targets,
            std::shared_ptr<balancer::BalancerSolution> balancer_solution,
            std::string const& dump_tensors_path,
            bool allow_modified_shapes) {
                
        auto [ret, fwd_to_gradient_mapping, bwd_gradients, updated_parameter_mapping, intermediate_tensors] =  eval_graph(graph, inputs, parameters, tt_device, intermediate_golden_tensors, losses, targets, balancer_solution, relative_atol, pcc, dump_tensors_path, allow_modified_shapes, true);
        return intermediate_tensors;
    },
        py::arg("graph"),
        py::arg("inputs"),
        py::arg("parameters"),
        py::arg("tt_device"),
        py::arg("relative_atol"),
        py::arg("pcc"),
        py::arg("intermediate_golden_tensors") = std::unordered_map<int, py::object>(),
        py::arg("losses") = std::vector<py::object>(),
        py::arg("targets") = std::vector<py::object>(),
        py::arg("balancer_solution") = nullptr,
        py::arg("dump_tensors_path") = "",
        py::arg("allow_modified_shapes") = false
    );


    m_graph.def("eval",
        [](Graph *graph,
            const std::vector<py::object> &inputs,
            const std::unordered_map<std::string, py::object> &parameters,
            py::object tt_device,
            float relative_atol, 
            float pcc,
            const std::unordered_map<int, py::object> &intermediate_golden_tensors,
            const std::vector<py::object> &losses,
            const std::vector<py::object> &targets,
            std::shared_ptr<balancer::BalancerSolution> balancer_solution,
            std::string const& dump_tensors_path,
            bool allow_modified_shapes) {
        auto ret =  eval_graph(graph, inputs, parameters, tt_device, intermediate_golden_tensors, losses, targets, balancer_solution, relative_atol, pcc, dump_tensors_path, allow_modified_shapes, false);
        return std::make_tuple(std::get<0>(ret), std::get<1>(ret), std::get<2>(ret), std::get<3>(ret));

    },
        py::arg("graph"),
        py::arg("inputs"),
        py::arg("parameters"),
        py::arg("tt_device"),
        py::arg("relative_atol"),
        py::arg("pcc"),
        py::arg("intermediate_golden_tensors") = std::unordered_map<int, py::object>(),
        py::arg("losses") = std::vector<py::object>(),
        py::arg("targets") = std::vector<py::object>(),
        py::arg("balancer_solution") = nullptr,
        py::arg("dump_tensors_path") = "",
        py::arg("allow_modified_shapes") = false
    );

    m_graph.def("remove_node", [](Graph *graph, const graphlib::NodeId id) {
        graph->remove_node(id);
        return;
    });

    m_graph.def("record_consteval_operations", [](Graph *graph) {
        std::unordered_map<std::string, std::optional<json>> recorded_consteval_operations;

        for (Node *node : tt::graphlib::topological_sort(*graph))
        {
            if (node->node_type() == NodeType::kInput)
            {
                graphlib::InputNode *input = node->as<graphlib::InputNode>();

                if (input->get_consteval_graph()) 
                {
                    recorded_consteval_operations[node->name()] = 
                        reportify::create_json_for_graph(input->get_consteval_graph()->get_graph());
                } 
                else
                {
                    recorded_consteval_operations[node->name()] = std::nullopt;
                }
            }
        }
        return recorded_consteval_operations;
    });

    // Query
    py::module_ m_graph_query = m_graph.def_submodule("query", "Submodule defining pybuda graph queries");

    py::class_<graphlib::query::NodePredicate>(m_graph_query, "NodePredicate")
        .def(
            "__or__",
            [](graphlib::query::NodePredicate const &a, graphlib::query::NodePredicate const &b) { return a | b; })
        .def(
            "__and__",
            [](graphlib::query::NodePredicate const &a, graphlib::query::NodePredicate const &b) { return a & b; })
        .def("negate", [](graphlib::query::NodePredicate const &a) { return a.negate(); });

    m_graph_query.def("name_regex", graphlib::query::name_regex);
    m_graph_query.def("layer_regex", graphlib::query::layer_regex);
    m_graph_query.def("op_type", graphlib::query::op_type);
}

py::object eval_relu(py::object tensor, graphlib::OpType type);

py::object eval_op(graphlib::OpType type, std::vector<py::object> inputs, graphlib::IRLevel ir_level, bool evaluate_output_relu = true) {

    py::object eval_module;

    switch (ir_level) {
        case graphlib::IRLevel::IR_PYBUDA: eval_module = py::module_::import("pybuda.op.eval.pybuda"); break;
        case graphlib::IRLevel::IR_BUDA: eval_module = py::module_::import("pybuda.op.eval.buda"); break;
        case graphlib::IRLevel::IR_CONSTEVAL: eval_module = py::module_::import("pybuda.op.eval.pybuda"); break;
    }

    py::function pybuda_eval = eval_module.attr("get_f_pybuda_eval")(type);

    log_trace(LogEval, "  eval_op: {}", type);
    bool has_requant = type.buda_attrs.find("requant") != type.buda_attrs.end() and std::get<bool>(type.buda_attrs.at("requant"));

    std::vector<py::object> inputs_;
    if (has_requant) {
        inputs_.assign(inputs.begin(), inputs.end()); 
        inputs_.erase(inputs_.end() - 1); // skip requantization input (last input)
    } else {
        inputs_ = inputs;
    }

    py::object result = pybuda_eval(inputs_);
    
    py::object common_module = py::module_::import("pybuda.op.eval");
    common_module.attr("eval_debug_print")(type.op, inputs, result);

    if (has_requant and ir_level == graphlib::IRLevel::IR_BUDA and type.op == "matmul")
    {
        std::vector<py::object> requant_inps = {result, inputs.back()};
        graphlib::OpType requant("requantization", {type.buda_attrs.at("zero_point")});
        auto requant_eval = eval_module.attr("get_f_pybuda_eval")(requant);
        result = requant_eval(requant_inps);
    }

    if (evaluate_output_relu)
        result = eval_relu(result, type);

    return result;
}

py::object eval_relu(py::object tensor, graphlib::OpType type)
{

    auto relu_match = type.buda_attrs.find("relu_en");
    if (relu_match != type.buda_attrs.end()) {
        std::vector<py::object> inputs;
        inputs.push_back(tensor);
        float relu_threshold = (type.buda_attrs.find("relu_threshold") != type.buda_attrs.end())
                                   ? std::get<float>(type.buda_attrs["relu_threshold"])
                                   : 0.0;
        string relu_mode = (type.buda_attrs.find("relu_mode") != type.buda_attrs.end())
                                   ? std::get<string>(type.buda_attrs["relu_mode"])
                                   : "min";

        graphlib::OpType relu("relu", {relu_threshold, relu_mode});
        tensor = eval_op(relu, inputs, graphlib::IRLevel::IR_PYBUDA);
    }
    return tensor;
}

py::object eval_fused_op(std::shared_ptr<FusedOp> fused_op, std::vector<py::object> inputs)
{
    std::unordered_map<std::uint32_t, py::object> buffers;
    std::optional<py::object> dest = std::nullopt;
    for (auto schedule : fused_op->get_schedules())
    {
        for (auto sub_op : schedule.ops)
        {
            std::vector<py::object> sub_op_inputs;

            for (FusedSubOpInput i : sub_op.inputs)
            {
                if (i.type == FusedSubOpInput::InputType::INPUT)  {
                    TT_ASSERT(i.index < inputs.size(), "Refering to input that doesn't exist for fused op");
                    sub_op_inputs.push_back(inputs.at(i.index));
                }
                else if (i.type == FusedSubOpInput::InputType::DEST) {
                    TT_ASSERT(dest.has_value());
                    sub_op_inputs.push_back(dest.value());
                    dest = std::nullopt; // done with reuse
                }
                else {
                    auto it = buffers.find(i.index);
                    TT_ASSERT(it != buffers.end(), "Referring to intermediate buffer that doesn't exist");
                    sub_op_inputs.push_back(it->second);
                }

                // In case the input for this sub_op is from the input buffer,
                // we don't need to apply any tms (they were applied before this method).
                if (i.type == FusedSubOpInput::InputType::INPUT)
                    continue;

                auto input = sub_op_inputs.back();

                // Apply needed tms...
                if (i.has_tile_broadcast())
                {
                    int tile_broadcast_dim = i.tile_broadcast.first ? 2 : 3;
                    graphlib::OpType op = graphlib::OpType("tile_broadcast", {tile_broadcast_dim});
                    input = eval_op(op, {input}, graphlib::IRLevel::IR_BUDA);
                }

                if (i.has_broadcast())
                {
                    int broadcast_dim = i.broadcast.first;
                    int broadcast_factor = i.broadcast.second;

                    graphlib::OpType op = graphlib::OpType("broadcast", {broadcast_dim, broadcast_factor});
                    input = eval_op(op, {input}, graphlib::IRLevel::IR_BUDA);
                }

                sub_op_inputs.pop_back();
                sub_op_inputs.emplace_back(input);
            }
            
            py::object result = eval_op(sub_op.op_type, sub_op_inputs, graphlib::IRLevel::IR_BUDA);

            if (sub_op.output_type == FusedSubOp::OutputType::OUTPUT)
                return result;
            else if (sub_op.output_type == FusedSubOp::OutputType::DEST)
                dest = result;
            else {
                // intermed buffer
                if (buffers.count((std::uint32_t)sub_op.output_buffer) == 0)
                    buffers.insert(std::make_pair((std::uint32_t)sub_op.output_buffer, result));
                else
                    buffers[(std::uint32_t)sub_op.output_buffer] = result;
            }
        }
    }
    TT_THROW("Evaluated the full fused op, but haven't reached the output.");
    return py::none();
}

py::object eval_t_streaming_tms(
    py::object tensor,
    graphlib::Graph *graph,
    graphlib::Node *node,
    std::shared_ptr<balancer::BalancerSolution> balancer_solution,
    std::string const &dir)
{
    if (not balancer_solution)
    {
        return tensor;
    }

    auto match = balancer_solution->op_models.find(node->name());
    if (match != balancer_solution->op_models.end())
    {
        std::vector<graphlib::OpType> t_streaming_tms = calculate_t_streaming_tms(graph, node, match->second);
        if (not t_streaming_tms.empty())
        {
            log_trace(LogEval, "{} t streaming: {}", dir, node->name());
        }
        for (auto tm : t_streaming_tms)
        {
            tensor = eval_op(tm, {tensor}, graph->get_ir_level());
        }
    }

    return tensor;
}

py::object eval_t_streaming_tms(
    py::object tensor,
    graphlib::Graph *graph,
    graphlib::Node *node,
    std::shared_ptr<balancer::BalancerSolution> balancer_solution)
{
    return eval_t_streaming_tms(tensor, graph, node, balancer_solution, "Redo");
}

py::object eval_golden_transforms(graphlib::Node *node, py::object tensor, bool eval_for_output = false)
{
    graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(node);
    if (not op or op->get_golden_transforms().empty())
        return tensor;

    log_trace(LogEval, "Undo golden transforms: {}", node->name());
    for (graphlib::OpType const &op_type : op->get_golden_transforms())
    {
        // Don't eval reshapes on output as its already done by reinterpret shape.
        // Don't eval transpose on output as transposed tensor should be passed to output node.
        //
        if (!eval_for_output || (op_type.op != "reshape" && op_type.op != "transpose"))
        {
            tensor = eval_op(op_type, {tensor}, graphlib::IRLevel::IR_PYBUDA);
        }
    }

    return tensor;
}

void eval_partial_datacopy_golden_transforms(
    std::vector<py::object> &ret, graphlib::Node *output_node, py::object output_tensor)
{
    auto output_index =
        output_node->as<graphlib::OutputNode>()->get_partial_datacopy_golden_output_index().value_or((int)ret.size());
    auto const &golden_transforms = output_node->as<graphlib::OutputNode>()->get_golden_transforms();

    if (output_index >= (int)ret.size())
        ret.resize(output_index + 1);

    for (auto const &op_type : golden_transforms)
    {
        output_tensor = eval_op(op_type, {output_tensor}, graphlib::IRLevel::IR_PYBUDA);
    }

    if (ret.at(output_index).ptr() == nullptr)
    {
        ret.at(output_index) = output_tensor;
    }
    else
    {
        graphlib::OpType overlay("add");
        ret.at(output_index) = eval_op(overlay, {ret.at(output_index), output_tensor}, graphlib::IRLevel::IR_PYBUDA);
    }
}

bool compare_tensor_to_golden(const std::string &name, const py::object &golden, const py::object &calculated, 
        float relative_atol, float pcc, graphlib::IRLevel ir_level, bool warning_only = false)
{
    py::object eval_module = py::module_::import("pybuda.op.eval");
    bool is_buda = ir_level == graphlib::IRLevel::IR_BUDA;

    if (pcc == 0.0) 
        return eval_module.attr("compare_tensor_to_golden")(name, golden, calculated, is_buda, 
                py::none(), /* rtol */
                py::none(), /* atol */
                py::none(), /* pcc */
                warning_only, 
                relative_atol,
                py::none() /* verify_cfg */).cast<bool>();

    return eval_module.attr("compare_tensor_to_golden")(name, golden, calculated, is_buda, 
                py::none(), /* rtol */
                py::none(), /* atol */
                pcc,
                warning_only, 
                relative_atol,
                py::none() /* verify_cfg */).cast<bool>();
}

py::object create_constant_tensor(float constant_value, std::pair<int, int> constant_dims, bool is_buda, DataFormat df) {
    py::object eval_module = py::module_::import("pybuda.op.eval");
    return eval_module.attr("create_constant_tensor_from_value")(constant_value, constant_dims, is_buda, df);
}

py::object create_constant_tensor(const std::vector<float> &tile_value, bool is_buda, DataFormat df) {
    py::object eval_module = py::module_::import("pybuda.op.eval");
    return eval_module.attr("create_constant_tensor_from_tile")(tile_value, is_buda, df);
}

py::object create_constant_tensor(const std::vector<float> &tensor_value, const Shape &tensor_shape, bool is_buda, tt::DataFormat df)
{
    py::object eval_module = py::module_::import("pybuda.op.eval");
    return eval_module.attr("create_constant_tensor_from_tensor")(tensor_value, tensor_shape.as_vector(), is_buda, df);
}


void dump_tensor(py::object tensor, std::string filename) {
    py::object eval_module = py::module_::import("pybuda.op.eval");
    eval_module.attr("dump_tensor")(tensor, filename);
}

// Evaluate TMs
py::object eval_tms(py::object tensor, const std::vector<graphlib::OpType> &tms, graphlib::IRLevel ir_level) {
    for (graphlib::OpType tm : tms) {
        std::vector<py::object> inputs;
        inputs.push_back(tensor);
        tensor = eval_op(tm, inputs, ir_level);
    }
    return tensor;
}

std::vector<py::object> eval_operand_tms(
    graphlib::Graph *graph,
    graphlib::Node *node,
    std::unordered_map<graphlib::NodeId, std::vector<py::object>> const &node_outputs) {
    std::vector<py::object> inputs;
    for (graphlib::Edge &input_edge : graph->operand_data_edges(node)) {
        TT_LOG_ASSERT(
            node_outputs.find(input_edge.producer_node_id) != node_outputs.end(),
            "Producer node output not in map, either hasn't run yet or doesn't exist in graph, node: {} producer: {}",
            node->name(),
            graph->node_by_id(input_edge.producer_node_id)->name());
        py::object tensor = node_outputs.at(input_edge.producer_node_id)[input_edge.producer_output_port_id];
        graphlib::Node *operand = graph->node_by_id(input_edge.producer_node_id);
        log_trace(
            LogEval, "  Operand[{}]: {} {}", input_edge.consumer_input_port_id, operand->name(), operand->shape());
        tensor = eval_tms(tensor, graph->get_edge_attributes(input_edge)->get_tms(), graph->get_ir_level());
        inputs.push_back(tensor);
    }
    return inputs;
}

py::object consteval_input(
    Node *runtime_node,
    std::unordered_map<std::string, py::object> inputs,
    bool is_buda,
    graphlib::NodeEpochType node_epoch_type)
{
    TT_ASSERT(
        node_epoch_type == graphlib::NodeEpochType::Forward or node_epoch_type == graphlib::NodeEpochType::Backward);
    TT_ASSERT(runtime_node->node_type() == NodeType::kInput);
    graphlib::InputNode *runtime_input = runtime_node->as<graphlib::InputNode>();

    TT_ASSERT(runtime_input->get_consteval_graph());
    graphlib::Graph *consteval_graph = runtime_input->get_consteval_graph()->get_graph();

    log_trace(
        LogConstEval,
        "ConstEval graph: {} Epoch: {} input: {}",
        consteval_graph->name(),
        node_epoch_type_to_string(node_epoch_type),
        runtime_node->name());
    TT_ASSERT(consteval_graph->get_ir_level() == graphlib::IRLevel::IR_CONSTEVAL);

    py::object tensor_module;
    py::function narrow_buda_tensor_to_pytorch;
    py::function pad_pytorch_tensor_to_buda;
    if (is_buda) {
        tensor_module = py::module_::import("pybuda.tensor");
        narrow_buda_tensor_to_pytorch = tensor_module.attr("narrow_buda_tensor_to_pytorch");
        pad_pytorch_tensor_to_buda = tensor_module.attr("pad_pytorch_tensor_to_buda");
    }

    if (node_epoch_type == graphlib::NodeEpochType::Backward)
    {
        std::vector<graphlib::Node *> loss_input =
            consteval_graph->users(runtime_input->get_consteval_graph()->get_output());
        TT_ASSERT(node_epoch_type != graphlib::NodeEpochType::Backward or loss_input.size() == 1);
        inputs[loss_input.at(0)->name()] = inputs.at(runtime_node->name());
    }

    py::object output;
    std::unordered_map<graphlib::NodeId, std::vector<py::object>> node_outputs;
    for (Node *node : tt::graphlib::topological_sort(
             *consteval_graph, [node_epoch_type](Node *n) { return n->get_epoch_type() == node_epoch_type; })) {
        log_trace(LogConstEval, "ConstEval node: {} - {} - Shape{}", node->name(), node->get_type(), node->shape());

        if (node->node_type() == graphlib::NodeType::kInput) {
            py::object input_value = py::none();

            auto input_node = node->as<graphlib::InputNode>();
            if (input_node->is_constant())
            {
                input_value = get_constant_input_value(node, is_buda);
            }
            else
            {
                input_value = inputs.at(node->name());
            }

            TT_ASSERT(not input_value.is_none());

            if (is_buda) {
                input_value = narrow_buda_tensor_to_pytorch(input_value, node->shape().as_vector());
            }
            node_outputs.insert({node->id(), {input_value}});
            continue;
        }

        if (node->node_type() == graphlib::NodeType::kOutput or node->node_type() == graphlib::NodeType::kQueue) {
            continue;
        }

        TT_ASSERT(
            node->node_type() == graphlib::NodeType::kPyOp,
            "Must be kPyOp at this point",
            node->name(),
            node->node_type());
        graphlib::OpNode *op_node = node->as<graphlib::OpNode>();

        auto type = op_node->op_type();
        auto relu_match = type.buda_attrs.find("relu");
        TT_ASSERT(relu_match == type.buda_attrs.end(), "ConstEval doesn't support relu");

        std::vector<py::object> inputs = eval_operand_tms(consteval_graph, node, node_outputs);
        output = eval_op(op_node->op_type(), inputs, consteval_graph->get_ir_level());

        node_outputs[op_node->id()].push_back(output);
    }

    TT_ASSERT(output.ptr() != nullptr, runtime_node->name());

    if (is_buda) {
        output = pad_pytorch_tensor_to_buda(output, std::vector<int>{});
    }

    return output;
}

py::object eval_input(
    Node *node, std::unordered_map<std::string, py::object> inputs, bool is_buda, graphlib::NodeEpochType epoch_type)
{
    graphlib::InputNode *input = node->as<graphlib::InputNode>();

    if (input->get_consteval_graph())
    {
        return consteval_input(node, inputs, is_buda, epoch_type);
    }
    else if (is_buda)
    {
        auto tensor_module = py::module_::import("pybuda.tensor");
        auto pad_pytorch_tensor_to_buda = tensor_module.attr("pad_pytorch_tensor_to_buda");
        return pad_pytorch_tensor_to_buda(inputs.at(node->name()), input->get_tile_broadcast_dims());
    }

    return inputs.at(node->name());
}

py::object eval_input_bw(Node *node, py::object input_value, bool is_buda)
{
    std::unordered_map<std::string, py::object> inputs = {{node->name(), input_value}};
    return eval_input(node, inputs, is_buda, graphlib::NodeEpochType::Backward);
}

py::object eval_reinterpret_shape(Graph *graph, Node *node, py::object input_value, bool flip = false)
{
    graphlib::InputNode *input = dynamic_cast<graphlib::InputNode *>(node);
    graphlib::OutputNode *output = dynamic_cast<graphlib::OutputNode *>(node);
    TT_ASSERT(input or output);

    graphlib::RuntimeTensorTransform runtime_tensor_transform =
        input ? input->get_runtime_tensor_transform() : output->get_runtime_tensor_transform();
    if (runtime_tensor_transform.type != graphlib::RuntimeTensorTransformType::ReinterpretShape)
    {
        return input_value;
    }

    if (flip)
    {
        runtime_tensor_transform.swap_original_and_reinterpreted_shapes();
    }

    log_trace(
        LogEval,
        "Eval reinterpret shape {}: {} -> {}",
        node->name(),
        node->shape(),
        runtime_tensor_transform.reinterpreted_shape);
    const bool is_buda = graph->get_ir_level() == graphlib::IRLevel::IR_BUDA;
    std::vector<graphlib::OpType::Attr> attr;
    if (is_buda)
    {
        auto original = runtime_tensor_transform.original_shape.canonical();
        auto reinterpreted = runtime_tensor_transform.reinterpreted_shape.canonical();
        for (size_t i = 0; i < original.size(); ++i)
            attr.push_back((int)original[i]); 
        for (size_t i = 0; i < reinterpreted.size(); ++i)
            attr.push_back((int)reinterpreted[i]);
    }
    else
    {
        auto vec = runtime_tensor_transform.reinterpreted_shape.as_vector();
        for (auto dim : vec) attr.emplace_back((int)dim);
    }

    graphlib::OpType reinterpret_shape("reshape", attr);
    return eval_op(reinterpret_shape, {input_value}, graph->get_ir_level());
}

py::object eval_embedding_index(Graph *, Node *, py::object input_value) { return input_value; }

py::object eval_constant_input(Graph *, Node *node, py::object input_value)
{
    graphlib::InputNode *input = dynamic_cast<graphlib::InputNode *>(node);

    if (!input)
    {
        return input_value;
    }

    graphlib::RuntimeTensorTransform runtime_tensor_transform = input->get_runtime_tensor_transform();
    if (runtime_tensor_transform.type != graphlib::RuntimeTensorTransformType::ConstantInput)
    {
        return input_value;
    }
    return runtime_tensor_transform.get_constant_input_tensor();
}

py::object eval_unpad(Graph *graph, Node *node, py::object input_value)
{
    graphlib::OutputNode *output_node = dynamic_cast<graphlib::OutputNode *>(node);
    if (!output_node)
        return input_value;

    graphlib::RuntimeTensorTransform runtime_tensor_transform = output_node->get_runtime_tensor_transform();
    if (runtime_tensor_transform.type != graphlib::RuntimeTensorTransformType::Unpad)
        return input_value;

    // Determine buda_unpad attributes based on original and padded shape
    graphlib::Shape original_shape = runtime_tensor_transform.unpadded_shape;
    int orig_c = original_shape[-1];
    int orig_r = original_shape[-2];

    graphlib::Shape padded_shape = node->shape();
    int pad_ct = (padded_shape[-1] / graphlib::Shape::BUDA_TILE_DIM) - graphlib::Shape::to_buda(original_shape).ct();
    if (pad_ct < 0)
        pad_ct = 0;
    int pad_rt = (padded_shape[-2] / graphlib::Shape::BUDA_TILE_DIM) - graphlib::Shape::to_buda(original_shape).rt();
    if (pad_rt < 0)
        pad_rt = 0;

    // Populate attributes and construct buda_unpad op for evaluation
    std::vector<graphlib::OpType::Attr> attr;
    attr.emplace_back(pad_rt);
    attr.emplace_back(pad_ct);
    attr.emplace_back(orig_r);
    attr.emplace_back(orig_c);

    graphlib::OpType unpad("buda_unpad", attr);
    return eval_op(unpad, {input_value}, graph->get_ir_level());
}

py::object eval_prestride(Graph *graph, Node *node, py::object input_value)
{
    graphlib::InputNode *input = dynamic_cast<graphlib::InputNode *>(node);

    if (!input)
    {
        return input_value;
    }

    graphlib::RuntimeTensorTransform runtime_tensor_transform = input->get_runtime_tensor_transform();
    if (runtime_tensor_transform.type != graphlib::RuntimeTensorTransformType::Prestride)
    {
        return input_value;
    }

    log_trace(LogEval, "Eval prestride {}: {} -> {}", node->name(), node->shape(), runtime_tensor_transform.reinterpreted_shape);
    std::vector<graphlib::OpType::Attr> attr;

    attr.emplace_back(runtime_tensor_transform.stride_height);
    attr.emplace_back(runtime_tensor_transform.stride_width);
    attr.emplace_back(runtime_tensor_transform.kernel_height);
    attr.emplace_back(runtime_tensor_transform.kernel_width);
    attr.emplace_back(static_cast<int>(runtime_tensor_transform.original_shape[-2]));
    attr.emplace_back(static_cast<int>(runtime_tensor_transform.original_shape[-1]));

    graphlib::OpType prestride_act("conv2d_prestride_act", attr);
    return eval_op(prestride_act, {input_value}, graph->get_ir_level());
}

py::object eval_concatenate(Graph *graph, std::vector<Node *> nodes, std::vector<py::object> input_values, size_t output_index)
{
    graphlib::OutputNode *output = dynamic_cast<graphlib::OutputNode *>(nodes[output_index]);

    if (!output)
    {
        return input_values[output_index];
    }

    graphlib::RuntimeTensorTransform runtime_tensor_transform = output->get_runtime_tensor_transform();
    if (runtime_tensor_transform.type != graphlib::RuntimeTensorTransformType::Concatenate)
    {
        return input_values[output_index];
    }

    Node *node = nodes[output_index];
    log_trace(LogEval, "Eval concatenate {}: {}", node->name(), node->shape());
    std::vector<graphlib::OpType::Attr> attr;

    attr.emplace_back(runtime_tensor_transform.concat_dim);

    std::vector<py::object> concat_inputs;
    // There won't be too many of these, do a nested loop for simplicity
    int conat_group = runtime_tensor_transform.concat_group;
    for (size_t i = 0; i < nodes.size(); i++)
    {
        for (Node *output_node : nodes)
        {
            graphlib::OutputNode *output = dynamic_cast<graphlib::OutputNode *>(output_node);
            if (!output)
            {
                continue;
            }

            graphlib::RuntimeTensorTransform runtime_tensor_transform = output->get_runtime_tensor_transform();
            if (runtime_tensor_transform.type != graphlib::RuntimeTensorTransformType::Concatenate)
            {
                continue;
            }

            if (runtime_tensor_transform.concat_group == conat_group and (size_t)runtime_tensor_transform.concat_index == i)
            {
                concat_inputs.push_back(input_values[i]);
                if (graph->get_ir_level() == graphlib::IRLevel::IR_BUDA)
                {
                    int length_at_dim = nodes[i]->shape()[runtime_tensor_transform.concat_dim];
                    attr.emplace_back(length_at_dim);
                }
            }
        }
    }

    graphlib::OpType prestride_act("concatenate", attr);
    return eval_op(prestride_act, concat_inputs, graph->get_ir_level());
}

std::vector<py::object> eval_runtime_tensor_transform(Graph *graph, std::vector<Node *> nodes, std::vector<py::object> input_values, bool flip = false)
{
    std::vector <py::object> ret;
    TT_ASSERT(nodes.size() == input_values.size());
    for (size_t i = 0; i < nodes.size(); i++)
    {
        Node *node = nodes[i];
        py::object input_value = input_values[i];

        graphlib::InputNode *input = dynamic_cast<graphlib::InputNode *>(node);
        graphlib::OutputNode *output = dynamic_cast<graphlib::OutputNode *>(node);
        TT_ASSERT(input or output);

        graphlib::RuntimeTensorTransform runtime_tensor_transform =
            input ? input->get_runtime_tensor_transform() : output->get_runtime_tensor_transform();

        switch (runtime_tensor_transform.type)
        {
            case graphlib::RuntimeTensorTransformType::Prestride:
                ret.push_back(eval_prestride(graph, node, input_value));
                break;
            case graphlib::RuntimeTensorTransformType::ReinterpretShape:
                ret.push_back(eval_reinterpret_shape(graph, node, input_value, flip));
                break;
            case graphlib::RuntimeTensorTransformType::EmbeddingIndex:
                ret.push_back(eval_embedding_index(graph, node, input_value));
                break;
            case graphlib::RuntimeTensorTransformType::ConstantInput:
                ret.push_back(eval_constant_input(graph, node, input_value));
                break;
            case graphlib::RuntimeTensorTransformType::Unpad:
                ret.push_back(eval_unpad(graph, node, input_value));
                break;
            case graphlib::RuntimeTensorTransformType::Concatenate:
                // Concatenate will join multiple outputs togeter, pass full vectors
                ret.push_back(eval_concatenate(graph, nodes, input_values, i));
                break;
            default: ret.push_back(input_value);
        }
    }
    return ret;
}

bool compare_tensors(std::shared_ptr<void> tensor0, std::shared_ptr<void> tensor1)
{
    py::object tensor_module = py::module_::import("pybuda.tensor");
    py::function compare_tensors_func = tensor_module.attr("compare_tensors");
    auto tensor0_pt = borrow_shared_py_object(tensor0);
    auto tensor1_pt = borrow_shared_py_object(tensor1);
    return compare_tensors_func(tensor0_pt, tensor1_pt).cast<bool>();
}

py::object get_constant_input_value(graphlib::Node *node, bool is_buda)
{
    TT_ASSERT(node->as<graphlib::InputNode>()->is_constant());
    graphlib::ConstantInputNode *cnode = node->as<graphlib::ConstantInputNode>();

    if (cnode->is_single_value()) {
        auto constant_value = cnode->constant_value();
        auto constant_dims = cnode->constant_dims();
        return create_constant_tensor(constant_value, constant_dims, is_buda, node->output_df());
    } else if (cnode->is_single_tile()) {
        auto constant_tile = cnode->tile_value();
        return create_constant_tensor(constant_tile, is_buda, node->output_df());
    } else if (cnode->is_tensor()) {
        auto tensor = borrow_shared_py_object(cnode->tensor());
        if (is_buda) {
            py::object tensor_module = py::module_::import("pybuda.tensor");
            py::function pad_pytorch_tensor_to_buda = tensor_module.attr("pad_pytorch_tensor_to_buda");
            tensor = pad_pytorch_tensor_to_buda(
                tensor,
                node->as<graphlib::InputNode>()->get_tile_broadcast_dims(),
                false, // squeeze
                1, // microbatch
                node->shape().get_tile_height(),
                node->shape().get_tile_width());
        }
        return tensor;
    }

    throw std::runtime_error("Shouldn't reach here.");
}

bool is_gradient_comparison_valid(Graph* graph, const graphlib::Edge& gradient_edge) {
    Node* forward_node = graph->node_by_id(gradient_edge.producer_node_id);
    Node* bwd_node_producing_gradient = graph->node_by_id(gradient_edge.consumer_node_id);

    bool is_gradient_comparison_valid = forward_node->shape() == bwd_node_producing_gradient->shape();
    log_trace(tt::LogTest, "forward_node:({}:{}), gradient_node: ({}:{})",
            forward_node->name(), forward_node->shape().as_string(),
            bwd_node_producing_gradient->name(), bwd_node_producing_gradient->shape().as_string());

    for (const auto& user_edge : graph->user_data_edges(bwd_node_producing_gradient)) {
        auto edge_attributes = graph->get_edge_attributes(user_edge);
        is_gradient_comparison_valid &= (not edge_attributes->has_tms());
    }

    return is_gradient_comparison_valid;
}

static std::unordered_map<std::string, py::object> get_graph_input_mapping(
    Graph *graph, const std::unordered_map<std::string, py::object> &parameters, py::object optimizer)
{
    const bool is_buda = graph->get_ir_level() == graphlib::IRLevel::IR_BUDA;
    std::unordered_map<std::string, py::object> graph_inputs = parameters;

    for (Node *node : tt::graphlib::topological_sort(*graph))
    {
        graphlib::InputNode *input = dynamic_cast<graphlib::InputNode *>(node);
        if (not input)
            continue;

        if (input->is_constant())
        {
            graph_inputs.insert({input->name(), get_constant_input_value(input, is_buda)});
        }
        else if (input->is_optimizer_parameter())
        {
            TT_ASSERT(not optimizer.is_none());

            std::vector<graphlib::Edge> optimizer_edges = graph->operand_edges(
                input, [](const auto &edge) { return edge.edge_type == graphlib::EdgeType::kAutogradFwdToOptimizer; });
            TT_ASSERT(optimizer_edges.size() == 1);

            std::string param_name = graph->node_by_id(optimizer_edges[0].producer_node_id)->name();
            py::object optimizer_params = optimizer.attr("get_optimizer_params")(param_name, is_buda);
            if (optimizer_params.is_none())
                continue;

            // Parse out the optimizer-param suffix string and do a lookup to get the tensor
            std::string optimizer_input_name = input->name();
            std::string::size_type optimizer_param_idx = optimizer_input_name.rfind('.');
            TT_ASSERT(
                optimizer_param_idx != std::string::npos,
                "Expecting optimizer node to have a '.<optimizer-param>' suffix identifier");

            std::string optimizer_param_key = optimizer_input_name.substr(optimizer_param_idx + 1);
            py::object opt_tensor = optimizer_params.attr("get")(optimizer_param_key);
            if (opt_tensor.is_none())
                log_fatal("optimizer_param key: {} not found for node: {}", optimizer_param_key, optimizer_input_name);

            graph_inputs.insert({input->name(), opt_tensor.attr("value")().cast<py::object>()});
        }
    }

    return graph_inputs;
}

// Evaluate graph with given inputs, and return list of outputs. If intermediate golden tensors are
// provided, compare each matching node ID
// Returns:
//   - graph outputs (ordered)
//   - map of parameter names to parameter gradients
//   - backward gradients on inputs, where they had requires_grad set (ordered by inputs)
//   - intermediate values for nodes in intermediate_to_return
std::tuple<
    std::vector<py::object>,
    std::unordered_map<std::string, py::object>,
    std::vector<py::object>,
    std::unordered_map<std::string, py::object>,
    std::unordered_map<std::string, py::object>>
eval_graph(
    Graph *graph,
    const std::vector<py::object> &inputs,
    const std::unordered_map<std::string, py::object> &parameters,
    py::object tt_device,
    const std::unordered_map<int, py::object> &intermediate_golden_tensors,
    const std::vector<py::object> &losses,
    const std::vector<py::object> &targets,
    std::shared_ptr<balancer::BalancerSolution> balancer_solution,
    float relative_atol,
    float pcc,
    std::string const &dump_tensors_path,
    bool allow_modified_shapes,
    bool return_intermediates)
{
    log_debug(LogEval, "Eval graph: {}", graph->name());

    std::unordered_map<graphlib::NodeId, std::vector<py::object>> node_outputs;
    std::unordered_map<std::string, py::object> fwd_to_gradient_mapping;
    std::unordered_map<std::string, py::object> input_to_gradient_mapping;
    std::unordered_map<std::string, py::object> updated_parameter_mapping;
    std::unordered_map<std::string, py::object> intermediate_tensors;

    const bool is_buda = graph->get_ir_level() == graphlib::IRLevel::IR_BUDA;

    auto optimizer = tt_device.attr("get_optimizer")();
    std::unordered_map<std::string, py::object> graph_inputs = get_graph_input_mapping(graph, parameters, optimizer);

    // Populate parameters and constant tensor mapping automatically that were created during compile
    for (Node *node : tt::graphlib::topological_sort(*graph)) {
        graphlib::InputNode *input = dynamic_cast<graphlib::InputNode *>(node);
        if (not input)
            continue;

        if (input->is_constant())
        {
            log_debug(tt::LogTest, "Populating constant: {}", node->name());
            node_outputs[node->id()].push_back(eval_input(node, graph_inputs, is_buda));
        }
        else if (input->is_parameter())
        {
            std::string param_name = node->as<graphlib::InputNode>()->get_fractured_parameter_mapping();
            if (param_name.empty())
                param_name = node->name();
            log_debug(tt::LogTest, "Populating module parameter: {}", param_name);
            node_outputs[node->id()].push_back(eval_input(node, graph_inputs, is_buda));
        }
        else if (input->is_optimizer_parameter())
        {
            log_debug(tt::LogTest, "Populating optimizer parameter: {}", node->name());
            node_outputs[node->id()].push_back(eval_input(node, graph_inputs, is_buda));
        }
    }

    // Populate loss tensor mapping
    if (losses.size() > 0) {
        size_t losses_index = 0;
        std::vector<std::string> output_gradient_names = graph->get_ordered_output_gradient_names();
        TT_ASSERT(output_gradient_names.size() == losses.size(), "The number of output gradient ports (" + std::to_string(output_gradient_names.size()) +
                                ") and losses (" + std::to_string(losses.size()) + ") should match.");

        std::vector<Node *> nodes;
        for (auto name : output_gradient_names)
            nodes.push_back(graph->get_node_by_name(name));
        std::vector<py::object> loss_tensors = eval_runtime_tensor_transform(graph, nodes, losses);
        for (std::string loss_name : graph->get_ordered_output_gradient_names())
        {
            py::object loss = loss_tensors.at(losses_index);
            Node *node = graph->get_node_by_name(loss_name);
            TT_ASSERT( (node->node_type() == NodeType::kInput) && node->as<graphlib::InputNode>()->is_loss(), 
                    "Expected that this node is a loss");
            log_debug(tt::LogTest, "Populating bwd loss: {}", node->name());
            node_outputs[node->id()].push_back(loss);
            ++losses_index;
        }
    }

    // Populate target tensor mapping
    if (targets.size() > 0) {
        size_t targets_index = 0;
        std::vector<std::string> target_inputs = graph->get_ordered_target_names();
        TT_ASSERT(target_inputs.size() == targets.size(), "The number of target inputs (" + std::to_string(target_inputs.size()) +
                                ") and targets (" + std::to_string(targets.size()) + ") should match.");
        for (std::string target : target_inputs)
        {
            Node *node = graph->get_node_by_name(target);
            TT_ASSERT( (node->node_type() == NodeType::kInput) && node->as<graphlib::InputNode>()->is_target(), 
                    "Expected that this node is a target input");
            log_debug(tt::LogTest, "Populating target input: {}", node->name());
            node_outputs[node->id()].push_back(targets.at(targets_index++));
        }
    }

    // Populate Pybuda input tensor mapping
    int input_index = 0;
    std::vector<py::object> input_tensors = eval_runtime_tensor_transform(graph, graph->ordered_module_inputs(), inputs);
    for (Node *node : graph->ordered_module_inputs()) {
        bool ignore_shape = node->as<graphlib::InputNode>()->get_runtime_tensor_transform().type ==
                            graphlib::RuntimeTensorTransformType::EmbeddingIndex;
        py::object input = input_tensors.at(input_index);

        // Evaluate tile_broadcast after runtime_tensor_transform
        if (graph->get_tile_broadcast_dims_for_input(input_index).size() > 0) {
            log_debug(tt::LogTest, "Evaluating tile_broadcast on input node {}", node->name());

            std::vector<graphlib::OpType::Attr> attr;
            for (auto dim : graph->get_tile_broadcast_dims_for_input(input_index)) {
                attr.emplace_back((int)dim);
                graphlib::OpType tile_bcast("tile_broadcast", attr);
                input = eval_op(tile_bcast, {input}, graph->get_ir_level());
                attr.clear();
            };
        }
        std::vector<std::uint32_t> shape = input.attr("shape").cast<std::vector<std::uint32_t>>();
        if (!ignore_shape && !allow_modified_shapes && (shape != node->shape().as_vector()))
        {
            throw std::runtime_error("Input " + std::to_string(input_index) + "'s shape is incorrect. " +
                    "Expected: " + node->shape().as_string() + ", got: " + Shape::create(shape).as_string());
        }
        node_outputs[node->id()].push_back(input);
        log_debug(tt::LogTest, "Populating module input: {}", node->name());
        input_index++;
    }

    for (Node *node : tt::graphlib::topological_sort(*graph)) {

        if (node->node_type() == NodeType::kInput)
            continue;

        if (node->node_type() == NodeType::kOutput)
            continue;

        if (node->node_type() == NodeType::kQueue) {
            // Copy input to output. Output to queue should never have TMs, only queue to next op will.
            graphlib::Edge input_edge = graph->operand_data_edges(node)[0];
            py::object tensor = node_outputs.at(input_edge.producer_node_id)[input_edge.producer_output_port_id];
            TT_ASSERT(graph->get_edge_attributes(input_edge)->get_tms().size() == 0, "Edges to queues should never have TMs");
            node_outputs[node->id()].push_back(tensor);
            continue;
        }

        if (node->node_type() == NodeType::kBudaNaryTM)
        {
            std::vector<py::object> inputs = eval_operand_tms(graph, node, node_outputs);
            py::object obj =
                eval_op(node->as<graphlib::BudaNaryTMNode>()->op_type(), inputs, graph->get_ir_level(), false);
            node_outputs[node->id()].push_back(obj);
            continue;
        }

        if ((node->node_type() != NodeType::kPyOp) && (node->node_type() != NodeType::kBudaOp))
            continue;

        graphlib::OpNode *op_node = node->as<graphlib::OpNode>();

        try {

            log_trace(LogEval, "Eval node: {} - {} - Shape{}", op_node->name(), op_node->op_type(), op_node->shape());

            std::vector<py::object> inputs = eval_operand_tms(graph, node, node_outputs);

            bool is_fused_op = (node->node_type() == graphlib::kBudaOp) && node->as<graphlib::BudaOpNode>()->is_fused_op();
            py::object obj = 
                is_fused_op ? eval_fused_op(node->as<graphlib::BudaOpNode>()->get_fused_op(), inputs) :
                              eval_op(op_node->op_type(), inputs, graph->get_ir_level(), false); // Don't Eval relu for intermediate checking

            auto gradient_edges = graph->operand_edges(node, 
                [](const auto& edge) { return edge.edge_type == graphlib::EdgeType::kAutogradFwdToGradient; });

            for (Node* user : graph->data_users(node)) {
                if (user->node_type() == graphlib::NodeType::kQueue and
                    user->as<graphlib::QueueNode>()->is_grad_accumulator()) {
                    // inherit grad edges for eval
                    auto gradient_edges_from_grad_queue = graph->operand_edges(user, 
                        [](const auto& edge) { return edge.edge_type == graphlib::EdgeType::kAutogradFwdToGradient; });
                    for (const auto& gradient_edge : gradient_edges_from_grad_queue) {
                        gradient_edges.push_back(gradient_edge);
                    }
                }
            }

            auto golden_node_id  = (graph->get_ir_level() == graphlib::IRLevel::IR_BUDA) ? node->pybuda_id() : node->id();
            if (op_node->has_golden_id()) {
                golden_node_id = op_node->golden_id(); // if a different intermediate node is used as a reference...
            }
            auto golden = intermediate_golden_tensors.find(golden_node_id);
            if (golden != intermediate_golden_tensors.end()) {
                // Intermediate checks are for debug only, so setting `warning only` to not fail tests for minor
                // mismatches on intermediates
                py::object calculated = eval_golden_transforms(node, obj);
                compare_tensor_to_golden(
                    node->name(), golden->second, calculated, relative_atol, pcc, graph->get_ir_level(), true);
            } else {
                // Check if there's a gradient to check
                if (gradient_edges.size() > 0) {
                    Node* producer = graph->node_by_id(gradient_edges.at(0).producer_node_id);
                    auto node_id  = (graph->get_ir_level() == graphlib::IRLevel::IR_BUDA) ? producer->pybuda_id() : producer->id();
                    auto golden_fwd = intermediate_golden_tensors.find(node_id);
                    if (golden_fwd != intermediate_golden_tensors.end()) {
                        bool is_valid = is_gradient_comparison_valid(graph, gradient_edges.at(0));
                        if (is_valid) {
                            py::object grad = golden_fwd->second.attr("grad");
                            if (!grad.is(py::none())) {
                                py::object calculated = eval_golden_transforms(producer, obj);
                                compare_tensor_to_golden(
                                    node->name() + " from " + producer->name(),
                                    grad,
                                    calculated,
                                    relative_atol,
                                    pcc,
                                    graph->get_ir_level());
                            }
                        } else {
                            log_debug(
                                tt::LogTest,
                                "Skipping Gradient Check: {} from {} because TMs on edge make the comparison invalid",
                                node->name(),
                                producer->name()
                            );
                        }
                    }
                }
            }

            // Eval relu after checking intermediate tensors
            obj = eval_relu(obj, op_node->op_type());
            node_outputs[node->id()].push_back(obj);

            std::vector<graphlib::Edge> loopback_edges = graph->user_edges(node,
                [](const auto& edge) { return edge.edge_type == graphlib::EdgeType::kDataLoopback; });
            for (const auto& loopback_edge : loopback_edges) {
                Node* consumer_node = graph->node_by_id(loopback_edge.consumer_node_id);
                if (consumer_node->node_type() == NodeType::kInput) {
                    updated_parameter_mapping[consumer_node->name()] = eval_input_bw(consumer_node, obj, is_buda);
                }
            }

            // Pick out gradient ops
            if (op_node->is_gradient_op()) 
            {
                for (graphlib::Edge gradient_edge : gradient_edges)
                {
                    Node *producer = graph->node_by_id(gradient_edge.producer_node_id);
                    py::object ret = eval_golden_transforms(node, obj);
                    if (producer->node_type() == NodeType::kInput) {
                        auto operands = graph->data_operands(producer);
                        if (operands.size() == 1)
                        {
                            graphlib::Node *optimizer = operands[0];
                            ret = eval_t_streaming_tms(ret, graph, optimizer, balancer_solution);
                        }
                        ret = eval_input_bw(producer, ret, is_buda);
                        ret = eval_runtime_tensor_transform(graph, {producer}, {ret}, true).at(0);
                    }

                    fwd_to_gradient_mapping[producer->name()] = ret;
                    log_debug(tt::LogTest, "Populating gradient op map: {}, gradient:{}", producer->name(), op_node->name());
                }
            }

        } catch (std::out_of_range &e) {
            throw std::runtime_error("Eval is missing inputs for " + node->name() + ", something went wrong.");
        } catch (py::error_already_set &e) {
            throw std::runtime_error(
                "Encountered python error while evaluating node " + node->name() + " (" + node->get_type() + "): \n" +
                std::string(e.what()));
        }

        if (return_intermediates)
        {
            log_debug(tt::LogTest, "Populating intermediate: {}", node->name());
            intermediate_tensors[node->name()] = node_outputs[node->id()].at(0);
        }
    }

    for (Node *node : tt::graphlib::topological_sort(*graph)) {
        if (node->node_type() == NodeType::kOutput) {
            std::vector<Node*> operands = graph->data_operands(node);

            TT_ASSERT(operands.size() == 1);
            node_outputs[node->id()] = node_outputs.at(operands[0]->id());
        }
    }

    for (Node *node : tt::graphlib::topological_sort(*graph)) {
        if (node->node_type() == NodeType::kInput and node->as<graphlib::InputNode>()->is_activation())
        {
            auto gradient_user_edges = graph->user_edges(node,
                [](const auto& edge) { return edge.edge_type == graphlib::EdgeType::kAutogradFwdToGradient; });

            for (const auto& gradient_user_edge : gradient_user_edges) {
                py::object gradient = node_outputs.at(gradient_user_edge.consumer_node_id).at(0);
                gradient = eval_golden_transforms(graph->node_by_id(gradient_user_edge.consumer_node_id), gradient);
                gradient = eval_input_bw(node, gradient, is_buda);
                gradient = eval_runtime_tensor_transform(graph, {node}, {gradient}, true).at(0);
                input_to_gradient_mapping[node->name()] = gradient;
            }
        }
    }

    std::vector<py::object> golden_transforms_outputs;
    auto module_outputs = graph->ordered_module_outputs();

    for (Node *output_node : graph->ordered_module_outputs()) {
        TT_ASSERT(output_node->node_type() == NodeType::kOutput);
        std::vector<Node *> operands = graph->data_operands(output_node);
        TT_ASSERT(operands.size() == 1);
        const std::vector<py::object>& output_tensors = node_outputs.at(output_node->id());
        for (auto output_tensor : output_tensors) {
            output_tensor = eval_golden_transforms(operands[0], output_tensor, /* eval_for_output */ true);
            golden_transforms_outputs.push_back(output_tensor);
        }
        log_debug(tt::LogTest, "Populating module output: {}", output_node->name());
    }
    std::vector<py::object> ret = eval_runtime_tensor_transform(graph, module_outputs, golden_transforms_outputs);

    for (Node *output_node : graph->ordered_partial_datacopy_outputs())
    {
        TT_ASSERT(
            output_node->node_type() == NodeType::kOutput and
            output_node->get_epoch_type() == graphlib::NodeEpochType::Forward);

        auto user_edges = graph->user_edges(output_node);
        TT_ASSERT(user_edges.size() == 1);
        TT_ASSERT(user_edges[0].edge_type == graphlib::EdgeType::kPartialDataCopy);

        auto user_shape = graph->node_by_id(user_edges[0].consumer_node_id)->shape().as_rank(4).as_vector();
        auto output_shape = output_node->shape().as_rank(4).as_vector();

        for (long unsigned int i = 0; i < user_shape.size(); i++)
        {
            TT_ASSERT(user_shape[i] == output_shape[i] or user_shape[i] % output_shape[i] == 0);
        }
        std::vector<Node *> operands = graph->data_operands(output_node);
        TT_ASSERT(operands.size() == 1);
        const std::vector<py::object> &output_tensors = node_outputs.at(output_node->id());
        TT_ASSERT(output_tensors.size() == 1);

        eval_partial_datacopy_golden_transforms(ret, output_node, output_tensors[0]);
        log_debug(tt::LogTest, "Populating partial data copy output: {}", output_node->name());
    }

    std::vector<py::object> bwd_gradients;
    for (Node *node : graph->ordered_module_inputs()) {
        TT_ASSERT(node->node_type() == NodeType::kInput);
        if (node->as<graphlib::InputNode>()->requires_grad())
        {
            if (input_to_gradient_mapping.find(node->name()) == input_to_gradient_mapping.end())
                continue;
            bwd_gradients.push_back(eval_input_bw(node, input_to_gradient_mapping.at(node->name()), is_buda));
        }
    }

    if (!dump_tensors_path.empty()) {
        for (auto &[node_id, output_tensor] : node_outputs) {
            auto node = graph->node_by_id(node_id);
            if ((node->node_type() == NodeType::kPyOp) || (node->node_type() == NodeType::kBudaOp)) {
                dump_tensor(output_tensor.at(0), dump_tensors_path + "/" + "intermediates." + node->name());
            }  else if (node->node_type() == NodeType::kInput) {
                if (updated_parameter_mapping.find(node->name()) != updated_parameter_mapping.end()) {
                    dump_tensor(updated_parameter_mapping[node->name()], dump_tensors_path + "/" + node->name());
                } else {
                    dump_tensor(output_tensor.at(0), dump_tensors_path + "/" + node->name());
                }
            } else {
                dump_tensor(output_tensor.at(0), dump_tensors_path + "/" + node->name());
            }
        }
    }
    return {ret, fwd_to_gradient_mapping, bwd_gradients, updated_parameter_mapping, intermediate_tensors};
}

}  // namespace tt
