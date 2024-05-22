// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "third_party/json/pybind11_json.hpp"

#include <sstream>
namespace py = pybind11;

#include "autograd/python_bindings.hpp"
#include "backend_api/backend_api.hpp"
#include "balancer/python_bindings.hpp"
#include "buda_passes.hpp"
#include "graph_lib/graph.hpp"
#include "graph_lib/python_bindings.hpp"
#include "lower_to_buda/common.hpp"
#include "lower_to_buda/netlist.hpp"
#include "passes/amp.hpp"
#include "passes/consteval.hpp"
#include "passes/fork_join.hpp"
#include "passes/fracture.hpp"
#include "passes/passes_utils.hpp"
#include "passes/placer_buda_passes.hpp"
#include "passes/python_bindings.hpp"
#include "passes/link_past_cache_ios.hpp"
#include "passes/move_index_to_mm_weights.hpp"
#include "pattern_matcher/python_bindings.hpp"
#include "placer/python_bindings.hpp"
#include "python_bindings_common.hpp"
#include "reportify/reportify.hpp"
#include "scheduler/python_bindings.hpp"
#include "shared_utils/sparse_matmul_utils.hpp"
#include "utils/ordered_associative_containers/ordered_map.hpp"
#include "tt_torch_device/python_bindings.hpp"

#include "utils/signal_handlers.hpp"

namespace tt {

PYBIND11_MODULE(_C, m) {

    // Register signal handlers when loading pybuda module.
    static SignalHandlers signal_handlers;

    m.attr("__name__") = "pybuda._C";
    m.doc() = "python bindings to pybuda framwork";

    m.attr("VERSION") = py::int_(1);

    m.attr("k_dim") = py::int_(passes::k_dim);

    py::enum_<tt::DataFormat>(m, "DataFormat")
        .value("Float32", tt::DataFormat::Float32)
        .value("Float16", tt::DataFormat::Float16)
        .value("Bfp8", tt::DataFormat::Bfp8)
        .value("Bfp4", tt::DataFormat::Bfp4)
        .value("Bfp2", tt::DataFormat::Bfp2)
        .value("Float16_b", tt::DataFormat::Float16_b)
        .value("Bfp8_b", tt::DataFormat::Bfp8_b)
        .value("Bfp4_b", tt::DataFormat::Bfp4_b)
        .value("Bfp2_b", tt::DataFormat::Bfp2_b)
        .value("Lf8", tt::DataFormat::Lf8)
        .value("UInt16", tt::DataFormat::UInt16)
        .value("Int8", tt::DataFormat::Int8)
        .value("RawUInt8", tt::DataFormat::RawUInt8)
        .value("RawUInt16", tt::DataFormat::RawUInt16)
        .value("RawUInt32", tt::DataFormat::RawUInt32)
        .value("Int32", tt::DataFormat::Int32)
        .value("Invalid", tt::DataFormat::Invalid)
        .export_values()
        .def(
            "to_json",
            [](tt::DataFormat df) {
                std::stringstream ss;
                ss << df;
                return ss.str();
            })
        .def("from_json", [](std::string const &encoded) {
            static std::unordered_map<std::string, tt::DataFormat> decode = {
                {"Float32", tt::DataFormat::Float32},
                {"Float16", tt::DataFormat::Float16},
                {"Bfp8", tt::DataFormat::Bfp8},
                {"Bfp4", tt::DataFormat::Bfp4},
                {"Bfp2", tt::DataFormat::Bfp2},
                {"Float16_b", tt::DataFormat::Float16_b},
                {"Bfp8_b", tt::DataFormat::Bfp8_b},
                {"Bfp4_b", tt::DataFormat::Bfp4_b},
                {"Bfp2_b", tt::DataFormat::Bfp2_b},
                {"Lf8", tt::DataFormat::Lf8},
                {"UInt16", tt::DataFormat::UInt16},
                {"Int8", tt::DataFormat::Int8},
                {"RawUInt8", tt::DataFormat::RawUInt8},
                {"RawUInt16", tt::DataFormat::RawUInt16},
                {"RawUInt32", tt::DataFormat::RawUInt32},
                {"Int32", tt::DataFormat::Int32},
                {"Invalid", tt::DataFormat::Invalid},
            };
            return decode.at(encoded);
        });

    py::module_ m_graph = m.def_submodule("graph", "Submodule defining pybuda graph functions");
    GraphModule(m_graph);

    py::module_ m_autograd = m.def_submodule("autograd", "Submodule defining autograd_engine.");
    AutogradModule(m_autograd);

    py::module_ m_scheduler = m.def_submodule("scheduler", "Submodule defining scheduling of ops on device.");
    SchedulerModule(m_scheduler);

    py::module_ m_placer = m.def_submodule("placer", "Submodule defining placer functions for placing ops onto epoch/chips");
    PlacerModule(m_placer);

    py::module_ m_balancer = m.def_submodule("balancer", "Submodule balancing ops onto device");
    BalancerModule(m_balancer);

    py::module_ m_pattern_matcher = m.def_submodule("pattern_matcher", "Submodule for discovering repeated subgraph structures");
    PatternMatcherModule(m_pattern_matcher);

    py::module_ m_backend = m.def_submodule("backend_api", "API to Buda Backend");
    tt::backend_api::BackendModule(m_backend);

    py::module_ m_passes = m.def_submodule("passes", "API to Buda Passes");
    PassesModule(m_passes);

    py::module_ m_torch_device = m.def_submodule("torch_device", "TT Torch Device");
    TorchDeviceModule(m_torch_device);

    py::class_<BudaNetlistConfig>(m, "BudaNetlistConfig")
        .def(py::init<>());

    py::class_<BudaNetlist>(m, "BudaNetlist")
        .def(py::init<>())
        .def("dump_to_yaml", &BudaNetlist::dump_to_yaml)
        .def("append_comment", &BudaNetlist::append_comment);

    py::class_<DramQueueConfigOverride>(m, "DramQueueConfigOverride")
        .def(py::init<std::optional<std::uint32_t>, std::optional<std::uint32_t>>())
        .def(py::pickle(
            [](const DramQueueConfigOverride &p) {  // __getstate__
                return py::make_tuple(p.chip_id, p.channel);
            },
            [](py::tuple t) {  // __setstate__
                if (t.size() != 2)
                    throw std::runtime_error("DramQueueConfigOverride: Invalid state!");

                DramQueueConfigOverride p(
                    t[0].cast<std::optional<std::uint32_t>>(), t[1].cast<std::optional<std::uint32_t>>());
                return p;
            }))
        .def(
            "to_json",
            [](const DramQueueConfigOverride &p)
            {
                std::unordered_map<std::string, std::optional<std::uint32_t>> d;
                d["chip_id"] = p.chip_id;
                d["channel"] = p.channel;
                return d;
            })
        .def(
            "from_json",
            [](std::unordered_map<std::string, std::optional<std::uint32_t>> const &d)
            {
                DramQueueConfigOverride queue_override;
                if (auto match = d.find("chip_id"); match != d.end())
                    queue_override.chip_id = match->second;
                if (auto match = d.find("channel"); match != d.end())
                    queue_override.channel = match->second;
                return queue_override;
            }),
        py::arg("chip_id"), py::arg("channel");

    py::class_<PostPlacerConfig>(m, "PostPlacerConfig")
        .def(
            py::init<
                DeviceConfig const &,
                std::uint32_t,
                std::uint32_t,
                bool,
                bool,
                bool,
                DramQueueMap,
                std::uint32_t,
                std::uint32_t,
                bool,
                placer::DRAMPlacementAlgorithm>(),
            py::arg("device_config"),
            py::arg("microbatch_size"),
            py::arg("microbatch_count"),
            py::arg("enable_t_streaming"),
            py::arg("input_queues_on_host"),
            py::arg("output_queues_on_host"),
            py::arg("manual_dram_queue_placement"),
            py::arg("output_queue_multiplier"),
            py::arg("input_queue_multiplier"),
            py::arg("enable_cross_chip_buffering"),
            py::arg("placement_algorithm"));

    py::class_<InsertionInstruction,PyInsertionInstruction , std::shared_ptr<InsertionInstruction>>(m, "InsertionInstruction")
        .def(
            py::init<
                InstructionType,
                std::string,
                std::string,
                bool,
                std::optional<std::uint32_t>,
                std::optional<std::uint32_t>,
                bool>(),
            py::arg("instr_type"),
            py::arg("src"),
            py::arg("dest"),
            py::arg("hoist_tms"),
            py::arg("input_id") = std::nullopt,
            py::arg("fork_id") = std::nullopt,
            py::arg("user_defined") = false)
        .def("unique_id", &InsertionInstruction::unique_id)
        .def("insert", &InsertionInstruction::insert);

    using NopInsertionFields = std::variant<std::string, bool, std::uint32_t, std::optional<std::uint32_t>>;
    py::class_<NopInsertionInstruction, InsertionInstruction, std::shared_ptr<NopInsertionInstruction>>(
        m, "NopInsertionInstruction")
        .def(
            py::init<
                std::string,
                std::string,
                bool,
                std::uint32_t,
                std::optional<std::uint32_t>,
                std::optional<std::uint32_t>,
                bool,
                bool,
                bool,
                bool,
                bool>(),
            py::arg("src"),
            py::arg("dest"),
            py::arg("hoist_tms"),
            py::arg("nop_count") = 1,
            py::arg("input_id") = std::nullopt,
            py::arg("fork_id") = std::nullopt,
            py::arg("user_defined") = false,
            py::arg("mergeable") = false,
            py::arg("daisy_chain") = false,
            py::arg("request_merge") = false,
            py::arg("is_fj_buffering") = false)
        .def(py::pickle(
            [](const NopInsertionInstruction &p) {  // __getstate__
                return py::make_tuple(
                    p.src, p.dest, p.hoist_tms, p.nop_count, p.input_id, p.fork_id, p.user_defined, p.mergeable, p.daisy_chain, p.request_merge, p.is_fj_buffering);
            },
            [](py::tuple t) {  // __setstate__
                if (t.size() != 11)
                    throw std::runtime_error("Invalid state!");

                NopInsertionInstruction p(
                    t[0].cast<std::string>(),
                    t[1].cast<std::string>(),
                    t[2].cast<bool>(),
                    t[3].cast<std::uint32_t>(),
                    t[4].cast<std::optional<std::uint32_t>>(),
                    t[5].cast<std::optional<std::uint32_t>>(),
                    t[6].cast<bool>(),
                    t[7].cast<bool>(),
                    t[8].cast<bool>(),
                    t[9].cast<bool>(),
                    t[10].cast<bool>());
                return p;
            }))
        .def(
            "to_json",
            [](const NopInsertionInstruction &p)
            {
                std::unordered_map<std::string, NopInsertionFields> d;
                d["src"] = p.src;
                d["dest"] = p.dest;
                d["hoist_tms"] = p.hoist_tms;
                d["nop_count"] = p.nop_count;
                d["input_id"] = p.input_id;
                d["fork_id"] = p.fork_id;
                d["user_defined"] = p.user_defined;
                d["mergeable"] = p.mergeable;
                d["daisy_chain"] = p.daisy_chain;
                d["request_merge"] = p.request_merge;
                d["is_fj_buffering"] = p.is_fj_buffering;
                return d;
            })
        .def(
            "from_json",
            [](std::unordered_map<std::string, NopInsertionFields> const &d)
            {
                std::string src{};
                std::string dest{};
                bool hoist_tms{false};
                std::uint32_t nop_count{1};
                std::optional<std::uint32_t> input_id{std::nullopt};
                std::optional<std::uint32_t> fork_id{std::nullopt};
                bool user_defined{false};
                bool mergeable{false};
                bool daisy_chain{false};
                bool request_merge{false};
                bool is_fj_buffering{false};

                if (auto match = d.find("src"); match != d.end())
                    src = std::get<std::string>(match->second);
                if (auto match = d.find("dest"); match != d.end())
                    dest = std::get<std::string>(match->second);
                if (auto match = d.find("hoist_tms"); match != d.end())
                    hoist_tms = std::get<bool>(match->second);
                if (auto match = d.find("nop_count"); match != d.end())
                    nop_count = std::get<std::uint32_t>(match->second);
                if (auto match = d.find("input_id");
                    match != d.end() && std::holds_alternative<std::optional<std::uint32_t>>(match->second))
                    input_id = std::get<std::optional<std::uint32_t>>(match->second);
                if (auto match = d.find("fork_id");
                    match != d.end() && std::holds_alternative<std::optional<std::uint32_t>>(match->second))
                    fork_id = std::get<std::optional<std::uint32_t>>(match->second);
                if (auto match = d.find("user_defined"); match != d.end())
                    user_defined = std::get<bool>(match->second);
                if (auto match = d.find("mergeable"); match != d.end())
                    mergeable = std::get<bool>(match->second);
                if (auto match = d.find("daisy_chain"); match != d.end())
                    daisy_chain = std::get<bool>(match->second);
                if (auto match = d.find("request_merge"); match != d.end())
                    request_merge = std::get<bool>(match->second);
                if (auto match = d.find("is_fj_buffering"); match != d.end())
                    is_fj_buffering = std::get<bool>(match->second);

                return NopInsertionInstruction(
                    src, dest, hoist_tms, nop_count, input_id, fork_id, user_defined, mergeable, daisy_chain, request_merge, is_fj_buffering);
            })
        .def("unique_id", &NopInsertionInstruction::unique_id);

    py::class_<QueueInsertionInstruction, InsertionInstruction, std::shared_ptr<QueueInsertionInstruction>>(
        m, "QueueInsertionInstruction")
        .def(
            py::init<
                std::string,
                std::string,
                bool,
                int,
                std::uint32_t,
                std::optional<std::uint32_t>,
                std::optional<std::uint32_t>,
                bool>(),
            py::arg("src"),
            py::arg("dest"),
            py::arg("hoist_tms"),
            py::arg("num_entries"),
            py::arg("queue_size"),
            py::arg("input_id") = std::nullopt,
            py::arg("fork_id") = std::nullopt,
            py::arg("user_defined") = false)
        .def(py::pickle(
            [](const QueueInsertionInstruction &p) {  // __getstate__
                return py::make_tuple(
                    p.src, p.dest, p.hoist_tms, p.num_entries, p.queue_size, p.input_id, p.fork_id, p.user_defined);
            },
            [](py::tuple t) {  // __setstate__
                if (t.size() != 8)
                    throw std::runtime_error("Invalid state!");

                QueueInsertionInstruction p(
                    t[0].cast<std::string>(),
                    t[1].cast<std::string>(),
                    t[2].cast<bool>(),
                    t[3].cast<int>(),
                    t[4].cast<std::uint32_t>(),
                    t[5].cast<std::optional<std::uint32_t>>(),
                    t[6].cast<std::optional<std::uint32_t>>(),
                    t[7].cast<bool>());
                return p;
            }))
        .def("unique_id", &QueueInsertionInstruction::unique_id);

    py::class_<tt::placer::Blocks>(m, "Blocks").def(py::init<>());
    py::class_<tt::placer::Block>(m, "Block").def(py::init<>());

    py::class_<PostPlacerResults>(m, "PostPlacerResults")
        .def_readonly("perf_model_results", &PostPlacerResults::perf_model_results)
        .def_readonly("ins_instructions", &PostPlacerResults::ins_instructions)
        .def_readonly("allocated_blocks", &PostPlacerResults::allocated_blocks)
        .def_readonly("current_host_address", &PostPlacerResults::current_host_address);

    py::enum_<tt::MathFidelity>(m, "MathFidelity")
        .value("LoFi", tt::MathFidelity::LoFi)
        .value("HiFi2", tt::MathFidelity::HiFi2)
        .value("HiFi3", tt::MathFidelity::HiFi3)
        .value("HiFi4", tt::MathFidelity::HiFi4)
        .value("Invalid", tt::MathFidelity::Invalid)
        .export_values()
        .def(
            "to_json",
            [](tt::MathFidelity df)
            {
                std::stringstream ss;
                ss << df;
                return ss.str();
            })
        .def(
            "from_json",
            [](std::string const &encoded)
            {
                static std::unordered_map<std::string, tt::MathFidelity> decode = {
                    {"LoFi", tt::MathFidelity::LoFi},
                    {"HiFi2", tt::MathFidelity::HiFi2},
                    {"HiFi3", tt::MathFidelity::HiFi3},
                    {"HiFi4", tt::MathFidelity::HiFi4},
                    {"Invalid", tt::MathFidelity::Invalid},
                };
                return decode.at(encoded);
            });

    py::register_exception<UnsupportedHWOpsError>(m, "UnsupportedHWOpsError");

    m.def("link_past_cache_ios", &passes::link_past_cache_ios);
    m.def("move_index_to_mm_weights", &passes::move_index_to_mm_weights);
    m.def("run_post_initial_graph_passes", &run_post_initial_graph_passes);
    m.def("run_optimization_graph_passes", &run_optimization_graph_passes);
    m.def("run_post_optimize_decompose_graph_passes", &run_post_optimize_decompose_graph_passes);
    m.def("run_consteval_graph_pass", &passes::run_consteval_graph_pass);
    m.def("run_post_autograd_graph_passes", &run_post_autograd_graph_passes);
    m.def(
        "run_pre_placer_buda_passes",
        &run_pre_placer_buda_passes,
        py::arg("graph"),
        py::arg("scheduler_config"),
        py::arg("device_config"),
        py::arg("chip_ids") = std::vector<std::uint32_t>{0},
        py::arg("op_names_to_chip_break") = placer::PredicatesToBreaks(),
        py::arg("op_names_to_epoch_break") = placer::PredicatesToBreaks(),
        py::arg("op_names_dont_fuse") = std::vector<std::string>{},
        py::arg("op_names_manual_fuse") = std::vector<std::string>{},
        py::arg("fracture_chip_id_assignments") = passes::FractureChipIdAssignments{},
        py::arg("default_df_override") = std::optional<DataFormat>{},
        py::arg("default_accumulate_df") = std::optional<DataFormat>{},
        py::arg("enable_broadcast_splitting") = false,
        py::arg("fp32_fallback") = DataFormat::Float16_b,
        py::arg("default_math_fidelity") = MathFidelity::HiFi3,
        py::arg("enable_auto_fusing") = false,
        py::arg("amp_level") = 0,
        py::arg("enable_recompute") = 0,
        py::arg("output_queues_on_host") = true,
        py::arg("input_queues_on_host") = true,
        py::arg("ins_instructions") = tt::
            ordered_map<InsInstructionUniqueId, std::shared_ptr<InsertionInstruction>, InsInstructionUniqueIdHash>{},
        py::arg("insert_queues") = std::vector<std::tuple<std::string, std::string, int>>{},
        py::arg("amp_properties") = std::vector<AMPNodeProperties>{},
        py::arg("op_intermediates_to_save") = std::vector<std::string>{},
        py::arg("use_interactive_placer") = true,
        py::arg("enable_device_tilize") = false);
    m.def(
        "is_subset_of_instructions",
        &is_subset_of_instructions,
        py::arg("ins_instructions") = tt::
            ordered_map<InsInstructionUniqueId, std::shared_ptr<InsertionInstruction>, InsInstructionUniqueIdHash>{},
        py::arg("previous_instructions") = tt::
            ordered_map<InsInstructionUniqueId, std::shared_ptr<InsertionInstruction>, InsInstructionUniqueIdHash>{});
    m.def("run_post_placer_buda_passes", &run_post_placer_buda_passes);
    m.def("run_pre_netlist_generation_buda_passes", &run_pre_netlist_generation_buda_passes);
    m.def("run_placer_buda_passes", &passes::run_placer_buda_passes);
    m.def("run_pre_lowering_passes", &run_pre_lowering_passes);
    m.def("lower_to_buda_netlist", &lower_to_buda_netlist,
    py::arg("graph"),
    py::arg("graph_name"),
    py::arg("placer_solution"),
    py::arg("balancer_solution"),
    py::arg("chip_ids"),
    py::arg("device_config"),
    py::arg("enable_forked_dram_inputs")=false);
    m.def("merge_netlists", &merge_netlists);

    m.def("dump_graph", [](
        const tt::graphlib::Graph *graph, 
        std::string test_name, 
        std::string graph_name, 
        const tt::placer::PlacerSolution *placer_solution,
        std::shared_ptr<tt::balancer::BalancerSolution> balancer_solution)
    {
        tt::reportify::dump_graph(test_name, graph_name, graph, placer_solution, balancer_solution);
    },
        py::arg("graph"),
        py::arg("test_name"),
        py::arg("graph_name"),
        py::arg("placer_solution") = nullptr,
        py::arg("balancer_solution") = nullptr
    );
    m.def("dump_epoch_type_graphs", [](
        const tt::graphlib::Graph *graph, 
        std::string test_name, 
        std::string graph_name, 
        const tt::placer::PlacerSolution *placer_solution,
        std::shared_ptr<tt::balancer::BalancerSolution> balancer_solution)
    {
        tt::reportify::dump_epoch_type_graphs(test_name, graph_name, graph, placer_solution, balancer_solution);
    },
        py::arg("graph"),
        py::arg("test_name"),
        py::arg("graph_name"),
        py::arg("placer_solution") = nullptr,
        py::arg("balancer_solution") = nullptr
    );
    m.def("dump_epoch_id_graphs", [](
        const tt::graphlib::Graph *graph, 
        std::string test_name, 
        std::string graph_name, 
        const tt::placer::PlacerSolution *placer_solution,
        std::shared_ptr<tt::balancer::BalancerSolution> balancer_solution)
    {
        tt::reportify::dump_epoch_id_graphs(test_name, graph_name, graph, placer_solution, balancer_solution);
    },
        py::arg("graph"),
        py::arg("test_name"),
        py::arg("graph_name"),
        py::arg("placer_solution"),
        py::arg("balancer_solution") = nullptr
    );

    py::enum_<tt::graphlib::NodeEpochType>(m, "NodeEpochType")
        .value("Forward", tt::graphlib::NodeEpochType::Forward)
        .value("Backward", tt::graphlib::NodeEpochType::Backward)
        .value("Optimizer", tt::graphlib::NodeEpochType::Optimizer)
        .export_values();

    py::class_<tt::sparse::SparseCOO>(m, "SparseCOO")
        .def(
            py::init<
                std::vector<std::int64_t>,
                std::vector<std::int64_t>,
                std::vector<float>,
                std::vector<std::int64_t>>(),
            py::arg("rows"),
            py::arg("cols"),
            py::arg("vals"),
            py::arg("shape"))
        .def_readonly("shape", &sparse::SparseCOO::shape)
        .def_readonly("rows", &sparse::SparseCOO::rows)
        .def_readonly("cols", &sparse::SparseCOO::cols)
        .def_readonly("vals", &sparse::SparseCOO::vals);

    py::class_<tt::sparse::SparseBUDA>(m, "SparseBUDA")
        .def_readonly("sparse_indices", &sparse::SparseBUDA::sparse_indices)
        .def_readonly("sparse_shape", &sparse::SparseBUDA::sparse_shape)
        .def_readonly("zdim", &sparse::SparseBUDA::zdim)
        .def_readonly("bcast_factor", &sparse::SparseBUDA::bcast_factor)
        .def("get_sparse_tile_ptr_bits", &sparse::SparseBUDA::get_sparse_tile_ptr_bits)
        .def("get_sparse_ublock_idx_bits", &sparse::SparseBUDA::get_sparse_ublock_idx_bits)
        .def("get_sparse_tiles_and_encodings", [](tt::sparse::SparseBUDA &self, int grid_r) {
            return self.get_sparse_tiles_and_encodings(grid_r);
        });

    // m.def("compress_sparse_tensor", &sparse::compress_sparse_tensor);
    m.def("compress_sparse_tensor_and_strip_info", &sparse::compress_sparse_tensor_and_strip_info);

    py::class_<tt::passes::AMPNodeProperties>(m, "AMPNodeProperties")
        .def(
            py::init<
                std::optional<std::string>,
                std::optional<tt::graphlib::NodeEpochType>,
                std::optional<tt::DataFormat>,
                std::optional<tt::DataFormat>,
                std::optional<tt::DataFormat>,
                std::optional<tt::MathFidelity>,
                std::optional<std::string>,
                std::optional<tt::passes::InputDfConfig>,
                std::optional<bool>,
                std::optional<std::vector<std::pair<std::uint32_t, std::uint32_t>>>>(),
            py::arg("op_type") = std::nullopt,
            py::arg("epoch_type") = std::nullopt,
            py::arg("output_df") = std::nullopt,
            py::arg("intermediate_df") = std::nullopt,
            py::arg("accumulate_df") = std::nullopt,
            py::arg("math_fidelity") = std::nullopt,
            py::arg("name_regex_match") = std::nullopt,
            py::arg("input_df") = std::nullopt,
            py::arg("is_gradient_op") = std::nullopt,
            py::arg("input_parameter_indices_to_optimize") = std::nullopt)
        .def_readonly("op_type", &tt::passes::AMPNodeProperties::op_type)
        .def_readonly("epoch_type", &tt::passes::AMPNodeProperties::epoch_type)
        .def_readonly("output_df", &tt::passes::AMPNodeProperties::output_df)
        .def_readonly("intermediate_df", &tt::passes::AMPNodeProperties::intermediate_df)
        .def_readonly("accumulate_df", &tt::passes::AMPNodeProperties::accumulate_df)
        .def_readonly("math_fidelity", &tt::passes::AMPNodeProperties::math_fidelity)
        .def_readonly("input_df", &tt::passes::AMPNodeProperties::input_df)
        .def_readonly("is_gradient_op", &tt::passes::AMPNodeProperties::is_gradient_op)
        .def_readonly("name_regex_match", &tt::passes::AMPNodeProperties::name_regex_match)
        .def_readonly(
            "input_parameter_indices_to_optimize", &tt::passes::AMPNodeProperties::input_parameter_indices_to_optimize)
        .def("to_json", [](tt::passes::AMPNodeProperties const& properties) {
            nlohmann::json j = properties;
            return j;
        })
        .def("from_json", [](nlohmann::json const& j) {
            return j.template get<tt::passes::AMPNodeProperties>();
        })
        .def(py::pickle(
            [](const tt::passes::AMPNodeProperties &p) {  // __getstate__
                return py::make_tuple(
                    p.op_type,
                    p.epoch_type,
                    p.output_df,
                    p.intermediate_df,
                    p.accumulate_df,
                    p.math_fidelity,
                    p.name_regex_match,
                    p.input_df,
                    p.is_gradient_op,
                    p.input_parameter_indices_to_optimize);
            },
            [](py::tuple t) {  // __setstate__
                if (t.size() != 10)
                    throw std::runtime_error("Invalid state!");

                tt::passes::AMPNodeProperties p(
                    t[0].cast<std::optional<std::string>>(),
                    t[1].cast<std::optional<tt::graphlib::NodeEpochType>>(),
                    t[2].cast<std::optional<tt::DataFormat>>(),
                    t[3].cast<std::optional<tt::DataFormat>>(),
                    t[4].cast<std::optional<tt::DataFormat>>(),
                    t[5].cast<std::optional<tt::MathFidelity>>(),
                    t[6].cast<std::optional<std::string>>(),
                    t[7].cast<std::optional<tt::passes::InputDfConfig>>(),
                    t[8].cast<std::optional<bool>>(),
                    t[9].cast<std::optional<vector<std::pair<std::uint32_t, std::uint32_t>>>>());
                return p;
            }));
}

}
