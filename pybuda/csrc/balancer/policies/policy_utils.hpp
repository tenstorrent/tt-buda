// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <random>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "balancer/balancer.hpp"
#include "balancer/balancer_utils.hpp"
#include "balancer/policies/policy_types.hpp"
#include "utils/logger.hpp"

namespace tt::placer
{
class InteractivePlacer;
}

namespace tt
{
// last bool in InsInstructionUniqueId represents instruction field mergeable. Tells if nop is mergeable (when two nops
// have the same producer and are flagged as mergeable, they are merged into one that feeds all of their consummers)
using InsInstructionUniqueId = std::tuple<std::string, std::string, std::uint32_t, std::uint32_t, bool, bool>;
struct InsInstructionUniqueIdHash;
struct InsertionInstruction;
}  // namespace tt

namespace tt::balancer
{

struct OpModelPair
{
    OpModel model;
    const graphlib::BudaOpNode* op;
};

class EpochSolution
{
   public:
    std::unordered_set<const tt::graphlib::Node*> current_epoch_nodes;
    std::unordered_set<const tt::graphlib::Node*> current_epoch_ops;

   private:
    std::uint32_t ribbon_size;
    std::vector<OpModelPair> ops;
    float utilization;
    const BalancerConfig* balancer_config;
    const Graph* graph;
    int dram_readers_core_count;
    int dram_writers_core_count;
    int pcie_readers_core_count;
    int pcie_writers_core_count;
    int epoch_target_cycles;
    mutable int pipeline_cycles;

    float evaluate() const;
    void recalc_nodes();

   public:
    EpochSolution(
        std::uint32_t ribbon_size,
        const BalancerConfig* balancer_config,
        std::vector<OpModelPair>& ops,
        const Graph* graph,
        int epoch_target_cycles) :
        ribbon_size(ribbon_size),
        ops(ops),
        utilization(0.0f),
        balancer_config(balancer_config),
        graph(graph),
        epoch_target_cycles(epoch_target_cycles)
    {
        recalc_nodes();
        utilization = evaluate();
    }

    void update_model(std::uint32_t index, const OpModel& model)
    {
        ops[index].model = model;
        recalc_nodes();
        utilization = evaluate();
    }

    void set_op_count(std::size_t op_count)
    {
        ops.resize(op_count);
        recalc_nodes();
        utilization = evaluate();
    }

    void print() const;
    float get_score() const { return utilization; }
    const BalancerConfig* get_balancer_config() const { return balancer_config; }
    const std::vector<OpModelPair>& get_ops() const { return ops; }
    std::uint32_t get_ribbon_size() const { return ribbon_size; }
    const std::unordered_set<const tt::graphlib::Node*>& get_current_epoch_ops() { return current_epoch_ops; }
    const std::unordered_set<const tt::graphlib::Node*>& get_current_epoch_nodes() { return current_epoch_nodes; }
    int get_pipeline_cycles() const { return pipeline_cycles; }
};

struct EpochCost
{
    int setup_cycles = 0;
    int runtime_cycles = 0;
};

OpModelMap to_op_model_map(OpModels const& selected_op_models);

placer::PlacerSolution run_placer(
    Graph const* graph, const BalancerConfig& config, OpModelMap const& selected_op_models);

void dump_balancer_placer_data(
    Graph const* graph,
    std::vector<std::uint32_t> chip_ids,
    tt::placer::PlacerSolution const& placer_solution,
    OpModelMap const& op_model_map,
    std::ostream& of,
    const std::string& arch_name);

std::vector<EpochCost> calculate_epoch_costs(
    placer::PlacerSolution const& placer_solution, OpModelMap const& selected_op_models, std::string const& arch_name);

inline int epoch_costs_sum(std::vector<EpochCost> const& epoch_costs)
{
    int sum = 0;
    for (EpochCost cost : epoch_costs) sum += cost.setup_cycles + cost.runtime_cycles;
    return sum;
}

inline PolicyType policy_from_string(std::string const& s)
{
    if (s == "MaximizeTMinimizeGrid")
        return PolicyType::MaximizeTMinimizeGrid;
    else if (s == "MinimizeGrid")
        return PolicyType::MinimizeGrid;
    else if (s == "Random")
        return PolicyType::Random;
    else if (s == "NLP")
        return PolicyType::NLP;
    else if (s == "CNN")
        return PolicyType::CNN;
    else if (s == "Ribbon")
        return PolicyType::Ribbon;
    else if (s == "default")  // default policy
        return PolicyType::NLP;

    log_error(LogBalancer, "Failed to parse policy from string: {}", s);
    log_error(LogBalancer, "Falling back to PolicyType::MinimizeGrid");

    return PolicyType::MinimizeGrid;
}

inline std::string policy_to_string(PolicyType p)
{
    switch (p)
    {
        case PolicyType::MinimizeGrid: return "MinimizeGrid";
        case PolicyType::Random: return "Random";
        case PolicyType::NLP: return "NLP";
        case PolicyType::CNN: return "CNN";
        case PolicyType::Ribbon: return "Ribbon";
        default: break;
    }
    return "Unknown";
}

void epoch_or_chip_break_remove_processed_nodes(
    const Graph* graph,
    std::vector<std::vector<std::string>>& op_names_to_epoch_or_chip_break,
    const std::unordered_set<const tt::graphlib::Node*>& processed_nodes);

std::pair<scheduler::Schedule, std::unordered_set<string>> policy_run_scheduler(
    graphlib::Graph const* graph,
    BalancerConfig const& config,
    const std::unordered_set<const tt::graphlib::Node*>& processed_nodes,
    const tt::scheduler::Schedule& processed_schedule,
    std::vector<tt::scheduler::Schedule>& op_names_to_epoch_break);

std::tuple<std::vector<std::string>, std::unordered_set<string>, std::unordered_set<string>> policy_run_scheduler(
    graphlib::Graph const* graph,
    BalancerConfig const& config,
    const std::unordered_set<const tt::graphlib::Node*>& processed_nodes,
    const tt::scheduler::Schedule& processed_schedule,
    std::vector<std::vector<std::string>>& op_names_to_epoch_break,
    std::vector<tt::scheduler::Schedule>& op_names_to_chip_break);

int get_limiter_cycles(
    const OpModel& op_model,
    const Graph* graph,
    const BalancerConfig& balancer_config,
    const int dram_access_core_count = 0,
    const int pcie_access_core_count = 0,
    const std::unordered_set<const tt::graphlib::Node*>* current_epoch_nodes = nullptr,
    bool invalidate_cached = false);

int get_limiter_cycles(
    const OpModel& op_model,
    const Graph* graph,
    const DeviceConfig& device_config,
    const bool input_queues_on_host,
    const bool output_queues_on_host,
    const int dram_access_core_count = 0,
    const int pcie_access_core_count = 0,
    const std::unordered_set<const tt::graphlib::Node*>* current_epoch_nodes = nullptr,
    bool invalidate_cached = false);

bool is_output_write_to_dram_over_target(
    const OpModel& op_model, const DeviceConfig& device_config, const int target_exec_cycles);

void cut_graph_solver_epoch(
    const graphlib::Graph* graph, placer::InteractivePlacer& placer, legalizer::GraphSolver& graph_solver);

void validate_solution(const std::vector<std::string>& scheduled_ops, const placer::PlacerSolution& placer_solution);

std::unordered_set<const tt::graphlib::Node*> calculate_current_epoch_nodes(
    const Graph* graph, const std::unordered_set<const tt::graphlib::Node*>& current_epoch_ops);

void set_op_model_for_node(
    legalizer::GraphSolver& graph_solver,
    const graphlib::Node* node,
    const OpModel& selected_op_model,
    std::string const& arch_name);

void set_op_model_for_node_ribbon(
    legalizer::GraphSolver& graph_solver,
    const graphlib::Node* op,
    const OpModel& selected_op_model,
    std::uint32_t current_ribbon_size);

int ribbon_buffering_factor(const OpModel& op_model);

bool is_matmul(const graphlib::BudaOpNode* op);

bool prologue_ok(const OpModel& op_model);
bool ukt_ok(const OpModel& op_model);
bool mblock_size_ok(const OpModel& op_model);
bool is_candidate_better_than_current(
    const OpModel& current,
    const OpModel& candidate,
    const Graph* graph,
    int ribbon_size,
    int target_exec_cycles,
    const BalancerConfig& device_config);

std::uint32_t pick_ribbon_size(
    std::uint32_t start_index,
    std::uint32_t end_index,
    const Graph* graph,
    const legalizer::GraphSolver& graph_solver,
    const std::vector<std::string>& scheduled_ops,
    std::uint32_t device_rows);

void cut_graph_solver_ribbon(
    const graphlib::Graph* graph,
    const graphlib::Node* op,
    placer::InteractivePlacer& placer,
    legalizer::GraphSolver& graph_solver);

std::pair<std::uint32_t, std::uint32_t> get_next_ribbon_change_op(
    const graphlib::Graph* graph,
    std::uint32_t current_index,
    const std::vector<std::string>& scheduled_ops,
    std::uint32_t current_matmul_dim_r = 0);

bool can_bind_sparse_dense_matmul_pair(
    const Graph* graph,
    const graphlib::BudaOpNode* sparse_op,
    OpModel const& sparse_op_model,
    const graphlib::BudaOpNode* dense_op,
    OpModel const& dense_op_model,
    placer::InteractivePlacer const& interactive_placer,
    bool allow_transpose);

bool close_to_target(std::uint32_t test, std::uint32_t target);

bool validate_sparse_matmul_model(
    const graphlib::BudaOpNode* op,
    const OpModel& op_model,
    const graphlib::Graph* graph,
    std::unordered_set<std::uint64_t>& validated_cache);

bool can_fit_on_single_epoch(
    tt::placer::InteractivePlacer& ip_fittment_tester,
    const std::string& op_name_1,
    const tt::balancer::GridShape& op_shape_1,
    const std::string& op_name_2,
    const tt::balancer::GridShape& op_shape_2,
    bool enable_transpose = true);

std::optional<placer::CoordRange> place_sparse_dense_pair(
    const graphlib::BudaOpNode* op,
    const OpModel* prefered_op_model,
    const graphlib::BudaOpNode* dense_matmul_op,
    const OpModel* prefered_op_model_dense,
    tt::placer::InteractivePlacer& interactive_placer,
    tt::placer::InteractivePlacer& ip_fittment_tester,
    bool& sparse_dense_pair);

int calculate_target_cycles_for_ribbon_size(
    const graphlib::Graph* graph,
    const BalancerConfig& config,
    legalizer::GraphSolver& graph_solver,
    tt::placer::InteractivePlacer& interactive_placer,
    tt::placer::InteractivePlacer& ip_fittment_tester,
    const std::uint32_t ribbon_size,
    std::unordered_set<std::uint64_t>& validated_cache,
    const std::vector<std::string>& scheduled_ops,
    const std::unordered_set<string>& epoch_break_ops,
    const graphlib::NodeEpochType current_epoch_type,
    const std::uint32_t placed_op_index,
    const int epoch_target_cycles);

template <typename T>
const OpModel* pick_preferred_op_model(
    const graphlib::Graph* graph,
    const BalancerConfig& config,
    const T& current_graph_solver,
    const graphlib::BudaOpNode* op,
    const std::uint32_t ribbon_size,
    std::unordered_set<std::uint64_t>& validated_cache,
    const int target_cycles)
{
    auto op_models = current_graph_solver.at(op);
    const OpModel* prefered_op_model = nullptr;
    for (const auto& op_model : op_models)
    {
        log_trace(
            LogBalancer,
            "    Examining Grid: {}, {}, stream: {}",
            op_model.grid_shape.r,
            op_model.grid_shape.c,
            op_model.t_stream_factor);

        // If it is sparse matmul op skip op model that can't be encoded.
        if (op->is_sparse_matmul())
        {
            if (!validate_sparse_matmul_model(op, op_model, graph, validated_cache))
            {
                log_trace(
                    LogBalancer,
                    "    Invalid sparse matmul op model. Grid: {}, {}, stream: {}",
                    op_model.grid_shape.r,
                    op_model.grid_shape.c,
                    op_model.t_stream_factor);
                continue;
            }
        }

        // If it is first valid op model select it.
        if (nullptr == prefered_op_model)
        {
            prefered_op_model = &op_model;
            continue;
        }

        // If we already have valid op model selected compare it with new one and select better.
        if (is_candidate_better_than_current(*prefered_op_model, op_model, graph, ribbon_size, target_cycles, config))
        {
            prefered_op_model = &op_model;
        }
    }

    return prefered_op_model;
}

template <typename T>
OpModel select_best_op_model_ribbon(
    const T& current_graph_solver,
    const graphlib::BudaOpNode* op,
    const std::uint32_t current_ribbon_size,
    const BalancerConfig& config,
    const graphlib::Graph* graph,
    std::unordered_set<std::uint64_t>& validated_cache,
    const int target_cycles)
{
    log_trace(LogBalancer, "  Selecting best op_model for {}. Choices:", op->name());
    const OpModel* selected_op_model = pick_preferred_op_model(
        graph, config, current_graph_solver, op, current_ribbon_size, validated_cache, target_cycles);

    TT_ASSERT(nullptr != selected_op_model, "No valid op_models for operation: ", op->name());

    return *selected_op_model;
}

bool buffer_graph(
    Graph* graph,
    tt::ordered_map<
        tt::InsInstructionUniqueId,
        std::shared_ptr<tt::InsertionInstruction>,
        tt::InsInstructionUniqueIdHash>& inst,
    legalizer::GraphSolver& graph_solver);

void score_solution(const std::vector<EpochSolution>& solutions, const DeviceConfig& device_config);
}  // namespace tt::balancer
