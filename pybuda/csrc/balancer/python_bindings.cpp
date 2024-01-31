// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "balancer/python_bindings.hpp"

#include <sstream>

#include "balancer/balancer.hpp"
#include "balancer/exceptions.hpp"
#include "balancer/policies/policy_utils.hpp"
#include "balancer/python_interface.hpp"
#include "balancer/balancer_utils.hpp"
#include "graph_lib/utils.hpp"
#include "placer/placer.hpp"
#include "passes/fuse_ops.hpp"

#include "third_party/json/json.hpp"

using namespace tt::balancer;

template <typename T>
inline std::optional<std::array<T, 2>> pair_as_array(std::optional<std::pair<T, T>> const& p)
{
    if (not p)
        return std::nullopt;
    return std::array<T, 2>{p->first, p->second};
}

template <typename T>
inline std::optional<std::pair<T, T>> array_as_pair(std::optional<std::array<T, 2>> const& p)
{
    if (not p)
        return std::nullopt;
    return std::make_pair((*p)[0], (*p)[1]);
}

void BalancerModule(py::module &m_balancer) {
    py::class_<BalancerSolution, std::shared_ptr<BalancerSolution>>(m_balancer, "BalancerSolution")
        .def_readonly("placer_solution", &BalancerSolution::placer_solution)
        .def_readonly("op_models", &BalancerSolution::op_models)
        .def_readonly("output_host_tms", &BalancerSolution::output_host_tms)
        .def("cut_edges_as_override", [](BalancerSolution const& s, tt::graphlib::Graph* graph) {
            std::vector<std::tuple<std::string, std::string, int>> edges;
            edges.reserve(s.graph_solver_cut_edges.size());
            for (auto [edge, a] : s.graph_solver_cut_edges)
            {
                edges.push_back(std::make_tuple(
                    graph->node_by_id(edge.producer_node_id)->name(),
                    graph->node_by_id(edge.consumer_node_id)->name(),
                    edge.consumer_input_port_id));
            }
            return edges;
        });

    py::enum_<PolicyType>(m_balancer, "PolicyType")
        .value("MinimizeGrid", PolicyType::MinimizeGrid)
        .value("Random", PolicyType::Random)
        .value("NLP", PolicyType::NLP)
        .value("CNN", PolicyType::CNN)
        .value("Ribbon", PolicyType::Ribbon)
        .export_values();

    py::class_<GridShape>(m_balancer, "GridShape")
        .def_readonly("r", &GridShape::r)
        .def_readonly("c", &GridShape::c)
        .def("__eq__", [](GridShape const& a, GridShape const& b) { return a == b; })
        .def(
            "__eq__",
            [](GridShape const& a, std::pair<int, int> const& b) { return a == GridShape(b.first, b.second); })
        .def("__repr__", [](GridShape const& a) {
            std::stringstream ss;
            ss << a;
            return ss.str();
        });

    using OpOverrideTypes = std::variant<std::optional<bool>, std::string, std::optional<int>, std::optional<std::array<int, 2>>, std::optional<std::map<std::uint32_t, std::uint32_t>>>;
    py::class_<OpOverride>(m_balancer, "OpOverride")
        .def(py::init<>())
        .def_readwrite("grid_shape", &OpOverride::grid_shape)
        .def_readwrite("force_dram_parameters", &OpOverride::force_dram_parameters)
        .def_readwrite("t_stream_dir", &OpOverride::t_stream_dir)
        .def_readwrite("t_stream_shape", &OpOverride::t_stream_shape)
        .def_readwrite("fracture_factor", &OpOverride::fracture_factor)
        .def_readwrite("u_kt", &OpOverride::u_kt)
        .def_readwrite("input_buffer_multiplier", &OpOverride::input_buffer_multiplier)
        .def_readwrite("output_buffer_multiplier", &OpOverride::output_buffer_multiplier)
        .def(py::pickle(
            [](const OpOverride& p) {  // __getstate__
                return py::make_tuple(
                    p.grid_shape,
                    p.force_dram_parameters,
                    p.t_stream_dir,
                    p.t_stream_shape,
                    p.fracture_factor,
                    p.u_kt,
                    p.input_buffer_multiplier,
                    p.output_buffer_multiplier);
            },
            [](py::tuple t) {  // __setstate__
                if (t.size() != 8)
                {
                    throw std::runtime_error("OpOverride: Invalid state!");
                }

                OpOverride p;
                p.grid_shape = t[0].cast<std::optional<std::pair<int, int>>>();
                p.force_dram_parameters = t[1].cast<std::optional<bool>>();
                p.t_stream_dir = t[2].cast<std::string>();
                p.t_stream_shape = t[3].cast<std::optional<std::pair<int, int>>>();
                p.fracture_factor = t[4].cast<std::optional<int>>();
                p.u_kt = t[5].cast<std::optional<int>>();
                p.input_buffer_multiplier = t[6].cast<std::optional<std::map<std::uint32_t, std::uint32_t>>>();
                p.output_buffer_multiplier = t[7].cast<std::optional<int>>();
                return p;
            }))
        .def(
            "to_json",
            [](OpOverride const& op_override)
            {
                std::unordered_map<std::string, OpOverrideTypes> d;
                d["grid_shape"] = pair_as_array(op_override.grid_shape);
                d["force_dram_parameters"] = op_override.force_dram_parameters;
                d["t_stream_dir"] = op_override.t_stream_dir;
                d["t_stream_shape"] = pair_as_array(op_override.t_stream_shape);
                d["fracture_factor"] = op_override.fracture_factor;
                d["u_kt"] = op_override.u_kt;
                d["input_buffer_multiplier"] = op_override.input_buffer_multiplier;
                d["output_buffer_multiplier"] = op_override.output_buffer_multiplier;
                return d;
            })
        .def(
            "from_json",
            [](std::unordered_map<std::string, OpOverrideTypes> const& d)
            {
                OpOverride op_override;
                if (auto match = d.find("grid_shape");
                    match != d.end() && std::holds_alternative<std::optional<std::array<int, 2>>>(match->second))
                    op_override.grid_shape = array_as_pair(std::get<std::optional<std::array<int, 2>>>(match->second));
                if (auto match = d.find("force_dram_parameters"); match != d.end())
                    op_override.force_dram_parameters = std::get<std::optional<bool>>(match->second);
                if (auto match = d.find("t_stream_dir"); match != d.end())
                    op_override.t_stream_dir = std::get<std::string>(match->second);
                if (auto match = d.find("t_stream_shape");
                    match != d.end() && std::holds_alternative<std::optional<std::array<int, 2>>>(match->second))
                    op_override.t_stream_shape =
                        array_as_pair(std::get<std::optional<std::array<int, 2>>>(match->second));
                if (auto match = d.find("fracture_factor");
                    match != d.end() and std::holds_alternative<std::optional<int>>(match->second))
                    op_override.fracture_factor = std::get<std::optional<int>>(match->second);
                if (auto match = d.find("u_kt");
                    match != d.end() and std::holds_alternative<std::optional<int>>(match->second))
                    op_override.fracture_factor = std::get<std::optional<int>>(match->second);
                if (auto match = d.find("input_buffer_multiplier");
                    match != d.end() &&
                    std::holds_alternative<std::optional<std::map<std::uint32_t, std::uint32_t>>>(match->second))
                    op_override.input_buffer_multiplier =
                        std::get<std::optional<std::map<std::uint32_t, std::uint32_t>>>(match->second);
                if (auto match = d.find("output_buffer_multiplier");
                    match != d.end() && std::holds_alternative<std::optional<int>>(match->second))
                    op_override.output_buffer_multiplier = std::get<std::optional<int>>(match->second);
                return op_override;
            });

    py::enum_<tt::placer::ChipPlacementPolicy>(m_balancer, "ChipPlacementPolicy")
        .value("MMIO_LAST", tt::placer::ChipPlacementPolicy::MMIO_LAST)
        .value("SNAKE", tt::placer::ChipPlacementPolicy::SNAKE)
        .export_values();

    py::enum_<legalizer::GraphSolverSelfCutType>(m_balancer, "GraphSolverSelfCutType")
        .value("None", legalizer::GraphSolverSelfCutType::None)
        .value("ConsumerOperandDataEdgesFirst", legalizer::GraphSolverSelfCutType::ConsumerOperandDataEdgesFirst)
        .value("ProducerUserDataEdgesFirst", legalizer::GraphSolverSelfCutType::ProducerUserDataEdgesFirst)
        .value("FastCut", legalizer::GraphSolverSelfCutType::FastCut)
        .export_values();

    py::class_<BalancerConfig>(m_balancer, "BalancerConfig")
        .def(
            py::init<
                tt::DeviceConfig,
                tt::scheduler::SchedulerConfig,
                PolicyType,
                int,
                std::vector<std::uint32_t>,
                tt::placer::ChipPlacementPolicy,
                bool,
                bool,
                bool,
                bool,
                bool,
                bool,
                std::unordered_map<std::string, OpOverride>,
                std::vector<std::vector<std::string>>,
                std::vector<std::vector<std::string>>,
                std::unordered_map<std::string, std::uint32_t>,
                std::unordered_map<std::string, ::tt::placer::PlacerOpOverride>,
                bool,
                legalizer::GraphSolverSelfCutType,
                bool,
                bool,
                bool>(),
            py::arg("device_config"),
            py::arg("scheduler_config"),
            py::arg("policy_type") = PolicyType::NLP,
            py::arg("random_policy_seed") = 0,
            py::arg("chip_ids") = std::vector<std::uint32_t>{0},
            py::arg("chip_placement_policy") = tt::placer::ChipPlacementPolicy::MMIO_LAST,
            py::arg("default_dram_parameters") = false,
            py::arg("skip_l1_usage_validation") = false,
            py::arg("enable_t_streaming") = false,
            py::arg("manual_t_streaming") = false,
            py::arg("input_queues_on_host") = true,
            py::arg("output_queues_on_host") = true,
            py::arg("op_overrides") = std::unordered_map<std::string, OpOverride>{},
            py::arg("op_names_to_epoch_break") = std::vector<std::vector<std::string>>{},
            py::arg("op_names_to_chip_break") = std::vector<std::vector<std::string>>{},
            py::arg("op_names_to_chip_id_assignment") = std::unordered_map<std::string, std::uint32_t>{},
            py::arg("op_name_to_placer_overrides") = std::unordered_map<std::string, ::tt::placer::PlacerOpOverride>{},
            py::arg("enable_auto_transposing_placement") = false,
            py::arg("graph_solver_self_cut_type") = legalizer::GraphSolverSelfCutType::None,
            py::arg("use_interactive_placer") = true,
            py::arg("enable_enumerate_u_kt") = true,
            py::arg("enable_single_buffer_fallback") = false)
        .def_readwrite("device_config", &BalancerConfig::device_config)
        .def_readwrite("scheduler_config", &BalancerConfig::scheduler_config)
        .def_readwrite("policy_type", &BalancerConfig::policy_type)
        .def_readwrite("random_policy_seed", &BalancerConfig::random_policy_seed)
        .def_readwrite("chip_ids", &BalancerConfig::chip_ids)
        .def_readwrite("default_dram_parameters", &BalancerConfig::default_dram_parameters)
        .def_readwrite("skip_l1_usage_validation", &BalancerConfig::skip_l1_usage_validation)
        .def_readwrite("enable_t_streaming", &BalancerConfig::enable_t_streaming)
        .def_readwrite("manual_t_streaming", &BalancerConfig::manual_t_streaming)
        .def_readwrite("input_queues_on_host", &BalancerConfig::input_queues_on_host)
        .def_readwrite("output_queues_on_host", &BalancerConfig::output_queues_on_host)
        .def_readwrite("op_overrides", &BalancerConfig::op_overrides)
        .def_readwrite("op_names_to_epoch_break", &BalancerConfig::op_names_to_epoch_break)
        .def_readwrite("op_names_to_chip_break", &BalancerConfig::op_names_to_chip_break)
        .def_readwrite("op_names_to_chip_id_assignment", &BalancerConfig::op_names_to_chip_break)
        .def_readwrite("op_name_to_placer_overrides", &BalancerConfig::op_name_to_placer_overrides)
        .def_readwrite("enable_auto_transposing_placement", &BalancerConfig::enable_auto_transposing_placement)
        .def_readwrite("graph_solver_self_cut_type", &BalancerConfig::graph_solver_self_cut_type)
        .def_readwrite("use_interactive_placer", &BalancerConfig::use_interactive_placer)
        .def_readwrite("enable_enumerate_u_kt", &BalancerConfig::enable_enumerate_u_kt)
        .def_readwrite("enable_single_buffer_fallback", &BalancerConfig::enable_single_buffer_fallback)
        .def_readwrite("fork_join_tiles_treshold", &BalancerConfig::fork_join_tiles_treshold)
        .def_readwrite("target_cycles_offset", &BalancerConfig::target_cycles_offset);

    py::class_<TensorShape>(m_balancer, "TensorShape")
        .def_readonly("w", &TensorShape::w)
        .def_readonly("z", &TensorShape::z)
        .def_readonly("rt", &TensorShape::rt)
        .def_readonly("ct", &TensorShape::ct)
        .def("__getitem__", [](TensorShape const& shape, int i) { return shape[i]; })
        .def("__setitem__", [](TensorShape& shape, int i, int val) { shape[i] = val; })
        .def(
            "__repr__",
            [](TensorShape const& a)
            {
                std::stringstream ss;
                ss << a;
                return ss.str();
            });

    py::class_<OpShape>(m_balancer, "OpShape")
        .def(
            py::init(
                [](std::vector<std::tuple<int, int, int, int>> const& input_shapes,
                   std::tuple<int, int, int, int> const& output_shape,
                   bool scalar_dims = true)
                {
                    auto tile_dim = [scalar_dims](int scalar_or_tile)
                    { return scalar_dims ? (scalar_or_tile / tt::graphlib::Shape::BUDA_TILE_DIM) : scalar_or_tile; };
                    std::vector<TensorShape> tensor_input_shapes;
                    for (auto [w, z, r, c] : input_shapes)
                        tensor_input_shapes.emplace_back(w, z, tile_dim(r), tile_dim(c));
                    auto [w, z, r, c] = output_shape;
                    return OpShape(
                        tensor_input_shapes, tensor_input_shapes, {TensorShape(w, z, tile_dim(r), tile_dim(c))});
                }),
            py::arg("input_shapes"),
            py::arg("output_shape"),
            py::arg("scalar_dims") = true)
        .def_readonly("inputs", &OpShape::inputs)
        .def_readonly("outputs", &OpShape::outputs)
        .def(
            "__repr__",
            [](OpShape const& a)
            {
                std::stringstream ss;
                ss << a;
                return ss.str();
            });

    py::class_<UBlockShape>(m_balancer, "UBlockShape")
        .def_readonly("rt", &UBlockShape::rt)
        .def_readonly("ct", &UBlockShape::ct)
        .def("volume", [](UBlockShape const& a) { return a.volume(); })
        .def("__eq__", [](UBlockShape const& a, UBlockShape const& b) { return a == b; })
        .def(
            "__eq__",
            [](UBlockShape const& a, std::pair<int, int> const& b) { return a.rt == b.first and a.ct == b.second; })
        .def("__repr__", [](UBlockShape const& a) {
            std::stringstream ss;
            ss << a;
            return ss.str();
        });

    py::class_<BlockShape>(m_balancer, "BlockShape")
        .def_readonly("t", &BlockShape::t)
        .def_readonly("mblock_m", &BlockShape::mblock_m)
        .def_readonly("mblock_n", &BlockShape::mblock_n)
        .def_readonly("ublock", &BlockShape::ublock)
        .def("volume", &BlockShape::volume)
        .def("buffer_tiles", &BlockShape::buffer_tiles)
        .def("__eq__", [](BlockShape const& a, BlockShape const& b) { return a == b; })
        .def("__repr__", [](BlockShape const& a) {
            std::stringstream ss;
            ss << a;
            return ss.str();
        });

    py::enum_<TStreamDir::Value>(m_balancer, "TStreamDir")
        .value("R", TStreamDir::Value::R)
        .value("C", TStreamDir::Value::C)
        .value("RZ", TStreamDir::Value::RZ)
        .value("CZ", TStreamDir::Value::CZ)
        .export_values();

    py::class_<TStreamFactor>(m_balancer, "TStreamFactor")
        .def_readonly("r", &TStreamFactor::r)
        .def_readonly("c", &TStreamFactor::c)
        .def_property_readonly("dir", [](TStreamFactor const& a) { return a.dir.v; })
        .def("__eq__", [](TStreamFactor const& a, TStreamFactor const& b) { return a == b; })
        .def("__repr__", [](TStreamFactor const& a) {
            std::stringstream ss;
            ss << a;
            return ss.str();
        });

    py::class_<BufferModel>(m_balancer, "BufferModel")
        .def_readonly("block_shape", &BufferModel::block_shape)
        .def_readonly("l1_size_tiles", &BufferModel::l1_size_tiles)
        .def_readonly("data_format", &BufferModel::data_format)
        .def("__repr__", [](BufferModel const& a) {
            std::stringstream ss;
            ss << a;
            return ss.str();
        });

    py::class_<OpModel>(m_balancer, "OpModel")
        .def_readonly("grid_shape", &OpModel::grid_shape)
        .def_readonly("op_shape", &OpModel::op_shape)
        .def("op_type", &OpModel::op_type)
        .def("buda_op_attrs", &OpModel::buda_op_attrs)
        .def("get_reduce_dim", &OpModel::get_reduce_dim)
        .def_readonly("data_format", &OpModel::data_format)
        .def("math_fidelity", &OpModel::math_fidelity)
        .def_readonly("t_stream_factor", &OpModel::t_stream_factor)
        .def_readonly("fracture_factor", &OpModel::fracture_factor)
        .def_readonly("sparse_indices", &OpModel::sparse_indices)
        .def_readonly("input_buffers", &OpModel::input_buffers)
        .def_readonly("output_buffers", &OpModel::output_buffers)
        .def_readonly("parameter_buffers", &OpModel::parameter_buffers)
        .def_readonly("is_sparse_matmul", &OpModel::is_sparse_matmul)
        .def_readonly("nz_tiles", &OpModel::nz_tiles)
        .def_readonly("nz_ublocks", &OpModel::nz_ublocks)
        .def_readonly("nz_strips", &OpModel::nz_strips)
        .def("block_shape", &OpModel::block_shape)
        .def("__repr__", [](OpModel const& a) {
            std::stringstream ss;
            ss << a;
            return ss.str();
        });

    py::class_<FusedSubOpModel>(m_balancer, "FusedSubOpModel")
        .def_readonly("type", &FusedSubOpModel::type)
        .def_readonly("mblock_m", &FusedSubOpModel::mblock_m)
        .def_readonly("mblock_n", &FusedSubOpModel::mblock_n)
        .def_readonly("ublock_rt", &FusedSubOpModel::ublock_rt)
        .def_readonly("ublock_ct", &FusedSubOpModel::ublock_ct)
        .def_readonly("mblock_k", &FusedSubOpModel::mblock_k)
        .def_readonly("ublock_kt", &FusedSubOpModel::ublock_kt)
        .def_readonly("reduce_dim", &FusedSubOpModel::reduce_dim)
        .def_readonly("has_dest_input", &FusedSubOpModel::has_dest_input)
        .def_readonly("has_dest_output", &FusedSubOpModel::has_dest_output)
        .def("__repr__", [](FusedSubOpModel const& a) {
            std::stringstream ss;
            ss << a;
            return ss.str();
        });

    using OutputHostTMTypes = std::variant<bool, int>;
    py::class_<OutputHostTM>(m_balancer, "OutputHostTM")
        .def_readonly("hstack_factor", &OutputHostTM::hstack_factor)
        .def_readonly("vstack_factor", &OutputHostTM::vstack_factor)
        .def_readonly("row_major", &OutputHostTM::row_major)
        .def(py::pickle(
            [](const OutputHostTM &p) { // __getstate__
                return py::make_tuple(
                    p.hstack_factor,
                    p.vstack_factor,
                    p.row_major
                );
            },
          [](py::tuple t) { // __setstate__
              if (t.size() != 3)
                  throw std::runtime_error("OutputHostTM: Invalid state!");

                OutputHostTM p = {
                    .hstack_factor = t[0].cast<int>(),
                    .vstack_factor = t[1].cast<int>(),
                    .row_major = t[2].cast<bool>()
                };

                return p;
            }))
        .def(
            "to_json",
            [](OutputHostTM const& output_host_tm) {
                std::unordered_map<std::string, OutputHostTMTypes> d;
                d["hstack_factor"] = output_host_tm.hstack_factor;
                d["vstack_factor"] = output_host_tm.vstack_factor;
                d["row_major"] = output_host_tm.row_major;
                return d;
            })
        .def_static("from_json", [](std::unordered_map<std::string, OutputHostTMTypes> const& d) {
            OutputHostTM output_host_tm;
            if (auto match = d.find("hstack_factor");
                match != d.end() && std::holds_alternative<int>(match->second))
                output_host_tm.hstack_factor = std::get<int>(match->second);
            if (auto match = d.find("vstack_factor"); match != d.end())
                output_host_tm.vstack_factor = std::get<int>(match->second);
            if (auto match = d.find("row_major"); match != d.end())
                    output_host_tm.row_major = std::get<bool>(match->second);
            return output_host_tm;
        });

    py::class_<FactorizedInt::Constant>(m_balancer, "Constant").def(py::init<int>(), py::arg("value"));

    py::class_<FactorizedInt>(m_balancer, "FactorizedInt")
        .def(py::init<>())
        .def(py::init<int>(), py::arg("max_val"))
        .def(py::init<std::pair<int, int>>(), py::arg("range"))
        .def(py::init<FactorizedInt::Constant>(), py::arg("constant"))
        .def("__and__", &FactorizedInt::operator&)
        .def("__or__", &FactorizedInt::operator|)
        .def("__sub__", &FactorizedInt::operator-)
        .def("__mul__", &FactorizedInt::operator*)
        .def("__div__", &FactorizedInt::operator/)
        .def_property_readonly("factors", &FactorizedInt::get_factors)
        .def_property_readonly("max_factor", &FactorizedInt::get_max_factor);

    m_balancer.def(
        "policy_from_string", &policy_from_string, "Returns policy type from string", py::arg("policy_type_str"));

    m_balancer.def(
        "graph_solver_self_cut_type_from_string",
        &legalizer::graph_solver_self_cut_type_from_string,
        "Returns graph solver self cut type from string",
        py::arg("graph_solver_self_cut_type_from_string_str"));

    m_balancer.def(
        "can_use_interactive_placer",
        &can_use_interactive_placer,
        "Returns whether provided policy can use interactive placer",
        py::arg("policy_type"));

    m_balancer.def(
        "chip_placement_policy_from_string",
        &tt::placer::chip_placement_policy_from_string,
        "Returns how chip ids will be ordered in placement",
        py::arg("chip_placement_policy_str"));

}

// python_interface.hpp implementation
namespace tt::balancer
{

std::pair<int, int> get_parallelization(
    Graph const* graph, OpNode const* node, int fracture_factor, bool sparse_buffer_enable)
{
    auto eval_module = py::module_::import("pybuda.op.eval.buda");
    py::function pybuda_parallelization = eval_module.attr("get_f_pybuda_parallelization")(node->op_type_ptr());

    auto op_shape = get_op_shape(graph, node);
    if ( (node->node_type() == graphlib::kBudaOp) && node->as<graphlib::BudaOpNode>()->is_fused_op())
    {
        // For the purposes of parallelization, the output op shape is the greatest common divisor of outputs 
        // of all fused op shapes
        auto fused_op = node->as<graphlib::BudaOpNode>()->get_fused_op();

        const auto &schedules = fused_op->get_schedules();
        balancer::TensorShape gcd_shape = schedules[0].ops[0].op_shape.outputs[0];
        for (const auto &schedule : schedules)
            for (const auto &op: schedule.ops)
            {
                gcd_shape.w = gcd(gcd_shape.w, op.op_shape.outputs[0].w);
                gcd_shape.z = gcd(gcd_shape.z, op.op_shape.outputs[0].z);
                gcd_shape.rt = gcd(gcd_shape.rt, op.op_shape.outputs[0].rt);
                gcd_shape.ct = gcd(gcd_shape.ct, op.op_shape.outputs[0].ct);

                // We need to take all fused op post tm inputs into account too.
                for (std::uint32_t input_id = 0; input_id < op.inputs.size(); input_id++)
                {
                    if (op.inputs[input_id].type != FusedSubOpInput::InputType::INPUT)
                        continue;

                    gcd_shape.w = gcd(gcd_shape.w, op.op_shape.inputs[input_id].w);
                    gcd_shape.z = gcd(gcd_shape.z, op.op_shape.inputs[input_id].z);
                    gcd_shape.rt = gcd(gcd_shape.rt, op.op_shape.inputs[input_id].rt);
                    gcd_shape.ct = gcd(gcd_shape.ct, op.op_shape.inputs[input_id].ct);
                }
            }

        log_trace(LogBalancer, "OpOverriding output shape for parallelization of {} to ({},{},{},{})", 
                node->name(),
                gcd_shape.w,
                gcd_shape.z,
                gcd_shape.rt,
                gcd_shape.ct);
        op_shape.outputs[0] = gcd_shape;
    }
    else if (node->as<graphlib::BudaOpNode>()->is_sparse_matmul() and sparse_buffer_enable)
    {
        int bcast_factor =
            graph->data_operands(node)[0]->as<graphlib::ConstantInputNode>()->get_sparse_buda().bcast_factor;
        auto [r, c] = pybuda_parallelization(op_shape, fracture_factor).cast<std::pair<int, int>>();
        TT_ASSERT((r % bcast_factor) == 0);
        return std::make_pair(r / bcast_factor, c);
    }
    return pybuda_parallelization(op_shape, fracture_factor).cast<std::pair<int, int>>();
}

int get_execution_cycles(std::string const& arch_name, OpModel const& op_model, bool theoretical, std::vector<FusedSubOpModel> const& sub_op_models)
{
    auto eval_module = py::module_::import("pybuda.op.eval.buda");
    py::function pybuda_op_execution_cycles =
        eval_module.attr("get_f_pybuda_execution_cycles")(op_model.buda_op_node->op_type_ptr());
    if (op_model.buda_op_node->op_type() == "matmul")
    {
        // Theoretical execution cycles are only applicable to matmuls
        return pybuda_op_execution_cycles(arch_name, op_model, theoretical).cast<int>();
    }

    if (op_model.fused_op() != nullptr)
    {
        return pybuda_op_execution_cycles(arch_name, op_model, sub_op_models).cast<int>();
    }

    return pybuda_op_execution_cycles(arch_name, op_model).cast<int>();
}

}  // namespace tt::balancer
