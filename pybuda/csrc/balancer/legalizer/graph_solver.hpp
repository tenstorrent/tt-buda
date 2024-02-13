// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <bitset>
#include <memory>
#include <unordered_map>
#include <vector>

#include "balancer/balancer_cache_collection.hpp"
#include "balancer/balancer_config.hpp"
#include "balancer/exceptions.hpp"
#include "balancer/legalizer/constraints.hpp"
#include "balancer/legalizer/graph_solver_types.hpp"
#include "balancer/types.hpp"
#include "graph_lib/graph.hpp"
#include "graph_lib/node.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "utils/logger.hpp"
#include "utils/profile.hpp"
#include "utils/small_vector.hpp"

namespace tt::balancer::legalizer
{

struct GraphSolverSolution
{
    OpModels selected_op_models;
    CutEdges cut_edges;

    GraphSolverSolution() = default;
    GraphSolverSolution(const OpModels& selected_op_models, const CutEdges& cut_edges) :
        selected_op_models(selected_op_models), cut_edges(cut_edges)
    {
    }
};

inline GraphSolverSelfCutType graph_solver_self_cut_type_from_string(std::string const& s)
{
    if ("None" == s)
        return GraphSolverSelfCutType::None;
    else if ("ConsumerOperandDataEdgesFirst" == s)
        return GraphSolverSelfCutType::ConsumerOperandDataEdgesFirst;
    else if ("ProducerUserDataEdgesFirst" == s)
        return GraphSolverSelfCutType::ProducerUserDataEdgesFirst;
    else if ("FastCut" == s)
        return GraphSolverSelfCutType::FastCut;

    log_error(LogGraphSolver, "Failed to parse graph solver self cut type from string: {}", s);
    log_error(LogGraphSolver, "Falling back to GraphSolverSelfCutType::None");

    return GraphSolverSelfCutType::None;
}

inline std::string graph_solver_self_cut_type_to_string(GraphSolverSelfCutType gssct)
{
    switch (gssct)
    {
        case GraphSolverSelfCutType::None: return "None";
        case GraphSolverSelfCutType::ConsumerOperandDataEdgesFirst: return "ConsumerOperandDataEdgesFirst";
        case GraphSolverSelfCutType::ProducerUserDataEdgesFirst: return "ProducerUserDataEdgesFirst";
        case GraphSolverSelfCutType::FastCut: return "FastCut";
        default: break;
    }

    return "Unknown";
}

class GraphSolver
{
   private:
    static constexpr std::size_t kNumBitsetBits = 1024;
    using Bitset = std::bitset<kNumBitsetBits>;
    static Bitset kBitsetAll;
    static constexpr Bitset kBitsetNone = Bitset{};

   public:
    struct ConstraintInfo
    {
        static constexpr int kPageSize = 20;

        struct Page
        {
            std::vector<std::int64_t> node_id_order;
            std::unordered_map<std::string, OpModel const&> id_to_op_models;
            std::unordered_map<std::string, std::vector<std::uint64_t>> node_id_to_op_model_ids;
            std::unordered_map<std::string, std::vector<std::tuple<std::uint16_t, std::uint16_t>>> edge_to_path_sets;
            std::unordered_map<std::string, std::uint64_t> failure_reason_ids;
        };

        std::string graph_name;
        std::vector<Page> pages;
        std::vector<std::uint64_t> op_model_selection;
        std::unordered_map<std::string, std::string> node_id_to_name;
        std::unordered_map<std::string, std::pair<int, int>> node_name_to_page;
        GraphSolver* gs_owner_cache = nullptr;

        bool needs_to_be_recomputed(const GraphSolver* graph_solver) const { return graph_solver != gs_owner_cache; }
    };

    struct RemainingOpModels
    {
        class Iterator : public std::iterator<std::input_iterator_tag, OpModel const>
        {
            std::uint64_t i = 0;
            std::vector<OpModel> const* p = nullptr;
            Bitset mask = 0;

           private:
            void next_valid()
            {
                if (mask == Bitset{})
                {
                    i = p->size();
                    return;
                }

                while (mask.any() and not mask[i]) ++i;
                mask.reset(i);
            }

           public:
            Iterator(std::vector<OpModel> const* p, const Bitset& mask, std::uint64_t i = 0) : i(i), p(p), mask(mask)
            {
                next_valid();
            }

            Iterator& operator++()
            {
                next_valid();
                return *this;
            }

            Iterator operator++(int)
            {
                auto r = *this;
                next_valid();
                return r;
            }

            bool operator==(Iterator other) const { return (p == other.p) and (i == other.i); }
            bool operator!=(Iterator other) const { return not(*this == other); }
            reference operator*() const { return (*p)[i]; }
        };

        RemainingOpModels(std::vector<OpModel> const& p, const Bitset& mask) : p(&p), mask(mask) {}

        Iterator begin() const { return Iterator(p, mask); }
        Iterator end() const { return Iterator(p, 0, std::min(kNumBitsetBits, p->size())); }
        size_t size() const { return mask.count(); }

        std::vector<OpModel> const* p = nullptr;
        Bitset mask = 0;
    };

    LegalOpModels const& legal_op_models_no_buffering() const;
    GraphSolverSolution const finish();
    void recompute_legal_op_models_on_cut(std::unordered_set<graphlib::Node*>& nodes_to_legalize);
    void recompute_legal_op_models(std::unordered_set<graphlib::Node*>& nodes_to_legalize);
    std::vector<graphlib::Node*> buffer(std::vector<BufferInfo>& buffer_edges);

   private:
    static Bitset bitset(std::uint64_t bit)
    {
        Bitset b;
        b.set(bit);
        return b;
    }
    // is `a` a subset of `b`
    static bool is_subset(const Bitset& a, const Bitset& b) { return a == (a & b); }

    using PathSetId = int;
    using BitsetId = int;

    class NodePathsProcessor
    {
       public:
        void add_node(const graphlib::Node* node)
        {
            if (control_set.count(node) == 0)
            {
                queue.push_back(node);
                control_set.insert(node);
            }
        }

        void process(GraphSolver* graph_solver)
        {
            while (!queue.empty())
            {
                const graphlib::Node* node = queue.back();
                queue.pop_back();
                control_set.erase(node);

                auto operand_path_sets = graph_solver->get_operand_path_sets_pts(node);
                auto user_path_sets = graph_solver->get_user_path_sets_pts(node);
                for (auto path_set : operand_path_sets)
                {
                    path_set->update_node_processor(graph_solver->bitsets, this);
                }
                for (auto path_set : user_path_sets)
                {
                    path_set->update_node_processor(graph_solver->bitsets, this);
                }
            }
        }

       private:
        std::vector<graphlib::Node const*> queue;
        std::unordered_set<graphlib::Node const*> control_set;
    };

    struct Path
    {
        std::uint16_t producer_id = 0;
        std::uint16_t consumer_id = 0;
        EdgeCost cost;

        Path() = default;
        Path(std::uint16_t producer_id, std::uint16_t consumer_id, EdgeCost cost) :
            producer_id(producer_id), consumer_id(consumer_id), cost(cost)
        {
        }
    };

    class PathSet
    {
       public:
        using Paths = SmallVector<Path, 16>;

        PathSet(
            BitsetId producer_set_id,
            BitsetId consumer_set_id,
            graphlib::Node* producer_node,
            graphlib::Node* consumer_node,
            Paths const& paths) :
            producer_set_id(producer_set_id),
            consumer_set_id(consumer_set_id),
            producer_node(producer_node),
            consumer_node(consumer_node),
            paths(paths)
        {
        }

        Bitset get_producer_set(const std::vector<Bitset>& bitsets) const { return bitsets[producer_set_id]; }
        Bitset get_consumer_set(const std::vector<Bitset>& bitsets) const { return bitsets[consumer_set_id]; }

        Paths const& get_paths() const { return paths; }
        Paths* get_paths_pt() { return &paths; }

        template <typename F>
        typename Paths::ConstIterator max_cost(F f) const
        {
            typename Paths::ConstIterator result = nullptr;

            for (auto iter = paths.begin(); iter != paths.end(); ++iter)
            {
                if (!result or f(result->cost, iter->cost))
                {
                    result = iter;
                }
            }

            return result;
        }

        template <bool is_operand, typename F>
        typename Paths::ConstIterator min_cost(F f, const std::uint16_t index) const
        {
            typename Paths::ConstIterator result = nullptr;

            for (auto iter = paths.begin(); iter != paths.end(); ++iter)
            {
                if ((is_operand and index == iter->consumer_id) or (!is_operand and index == iter->producer_id))
                {
                    if (!result or f(iter->cost, result->cost))
                    {
                        result = iter;
                    }
                }
            }

            return result;
        }

        bool erase(typename Paths::ConstIterator pos, std::vector<Bitset>& bitsets)
        {
            *const_cast<typename Paths::Iterator>(pos) = paths.back();
            paths.pop_back();
            return update(bitsets);
        }

        bool empty(const std::vector<Bitset>& bitsets) const
        {
            return paths.empty() or (bitsets[producer_set_id] == 0) or (bitsets[consumer_set_id] == 0);
        }

        bool update(std::vector<Bitset>& bitsets)
        {
            Bitset valid_producer_set = 0;
            Bitset valid_consumer_set = 0;
            Bitset producer = bitsets[producer_set_id];
            Bitset consumer = bitsets[consumer_set_id];

            for (std::size_t i = 0; i < paths.size(); i++)
            {
                Path const& path = paths[i];
                if (consumer[path.consumer_id] and producer[path.producer_id])
                {
                    valid_producer_set.set(path.producer_id);
                    valid_consumer_set.set(path.consumer_id);
                }
                else
                {
                    paths[i] = paths.back();
                    paths.pop_back();
                    i--;
                }
            }

            bool is_producer_sub = is_subset(producer, valid_producer_set);
            bool is_consumer_sub = is_subset(consumer, valid_consumer_set);
            bool unchanged = is_producer_sub and is_consumer_sub;

            if (!unchanged)
            {
                bitsets[producer_set_id] &= valid_producer_set;
                bitsets[consumer_set_id] &= valid_consumer_set;
            }

            return not unchanged;
        }

        void update_node_processor(std::vector<Bitset>& bitsets, NodePathsProcessor* node_processor)
        {
            Bitset valid_producer_set = 0;
            Bitset valid_consumer_set = 0;
            Bitset producer = bitsets[producer_set_id];
            Bitset consumer = bitsets[consumer_set_id];
            for (std::size_t i = 0; i < paths.size(); i++)
            {
                Path const& path = paths[i];
                if (consumer[path.consumer_id] and producer[path.producer_id])
                {
                    valid_producer_set.set(path.producer_id);
                    valid_consumer_set.set(path.consumer_id);
                }
                else
                {
                    paths[i] = paths.back();
                    paths.pop_back();
                    i--;
                }
            }

            if (!is_subset(producer, valid_producer_set))
            {
                node_processor->add_node(consumer_node);
                node_processor->add_node(producer_node);
                bitsets[producer_set_id] &= valid_producer_set;
            }

            if (!is_subset(consumer, valid_consumer_set))
            {
                node_processor->add_node(producer_node);
                node_processor->add_node(consumer_node);
                bitsets[consumer_set_id] &= valid_consumer_set;
            }
        }

        const graphlib::Node* get_producer_node() const { return producer_node; }
        const graphlib::Node* get_consumer_node() const { return consumer_node; }

       private:
       private:
        BitsetId producer_set_id = -1;
        BitsetId consumer_set_id = -1;
        graphlib::Node* producer_node = nullptr;
        graphlib::Node* consumer_node = nullptr;
        Paths paths;
    };

    const std::vector<OpModel>& get_legal_op_models(graphlib::Node const* node) const;
    void reset(bool partial_reset_allowed = false);
    void invalidate_suboptimal_op_models(const std::vector<graphlib::Node*>& nodes);
    void invalidate_streaming_into_output(const std::vector<graphlib::Node*>& nodes);
    void invalidate_suboptimal_op_models_for_op(
        const graphlib::BudaOpNode* node, GraphSolverOpModelInvalidationStrategyTier tier);

    struct SharedData
    {
       public:
        std::unique_ptr<Constraint> constraint;
        std::unordered_map<std::uint64_t, const std::pair<const EdgeCost, const ConstraintFailureReason>>
            constraint_result_cache;

       private:
        LegalOpModels legal_op_models;
        bool graph_solving_finished;
        std::unordered_map<const tt::graphlib::Node*, std::vector<std::vector<tt::balancer::OpModel>>>
            recomputed_legal_op_models;

       public:
        SharedData(std::unique_ptr<Constraint>&& constraint, LegalOpModels const& legal_op_models) :
            constraint(std::move(constraint)), legal_op_models(legal_op_models), graph_solving_finished(false)
        {
        }

        friend const std::vector<OpModel>& GraphSolver::get_legal_op_models(graphlib::Node const* node) const;
        friend void GraphSolver::reset(bool partial_reset_allowed);
        friend LegalOpModels const& GraphSolver::legal_op_models_no_buffering() const;
        friend GraphSolverSolution const GraphSolver::finish();
        friend void GraphSolver::recompute_legal_op_models(std::unordered_set<graphlib::Node*>& nodes_to_legalize);
        friend std::vector<graphlib::Node*> GraphSolver::buffer(std::vector<BufferInfo>& buffer_edges);
    };

    PathSet& get_path_set(const graphlib::Edge& edge);
    PathSet const& get_path_set(const graphlib::Edge& edge) const;

    PathSet* get_path_set_pt(const graphlib::Edge& edge);
    SmallVector<PathSet*> get_operand_path_sets_pts(graphlib::Node const* node);
    SmallVector<PathSet*> get_user_path_sets_pts(graphlib::Node const* node);

    void log_bitset(graphlib::Node const* node, const Bitset& set) const;
    template <bool kOperand>
    void handle_cumulative_paths_error(
        PathSet const& path_set,
        const Bitset& debug_snapshot,
        graphlib::Node const* producer,
        graphlib::Node const* consumer);
    template <bool kOperand, typename CostFns>
    bool apply_cumulative_costs(
        tt::SmallVector<PathSet*> const& path_sets, graphlib::Node const* node, CostFns cost_fns);
    void handle_no_paths_left_on_update(
        bool invoked_by_set, const std::string& root_node_name, const std::string& current_node_name);
    void update_solver(graphlib::Node const* root, bool expand_root = true, bool invoked_by_set = false);

    Bitset* get_bitset(graphlib::NodeId node_id);
    Bitset const* get_bitset(graphlib::NodeId node_id) const;
    Bitset* get_or_insert_bitset(graphlib::NodeId node_id, const Bitset& init);

    void throw_error_for_edge(graphlib::Edge edge);
    void resolve(bool partial_reset_allowed = false);
    bool resolve_step(const bool self_cut_allowed);
    std::vector<graphlib::Edge> get_epoch_type_switch_cut_edges();
    void update_constraint_info();
    bool self_cut(graphlib::Node* producer_node, graphlib::Node* consumer_node);
    void register_virtual_node(graphlib::Node* buffer_nop);
    void insert_virtual_queue(
        graphlib::Edge& edge, const graphlib::Node* src, const graphlib::Node* dest, bool is_e2e_queue = false);
    void resolve_step_postprocess(const std::vector<graphlib::Node*>& nodes);

    GraphSolver(
        graphlib::Graph* graph,
        std::unique_ptr<Constraint>&& constraint,
        LegalOpModels const& legal_op_models,
        BalancerConfig const& balancer_config,
        std::shared_ptr<tt::balancer::BalancerCacheCollection> balancer_cache_collection,
        std::vector<graphlib::Edge> const& cut_edges,
        bool use_op_model_recalculation_on_cut,
        bool resolve_on_create = true);

   public:
    template <typename ConstraintT>
    static GraphSolver create(
        graphlib::Graph* graph,
        LegalOpModels const& legal_op_models,
        BalancerConfig const& balancer_config,
        std::shared_ptr<tt::balancer::BalancerCacheCollection> balancer_cache_collection,
        bool use_op_model_recalculation_on_cut,
        std::vector<graphlib::Edge> const& cut_edges = {},
        bool resolve_on_create = true)
    {
        return GraphSolver(
            graph,
            std::make_unique<ConstraintT>(balancer_config.device_config, balancer_cache_collection),
            legal_op_models,
            balancer_config,
            balancer_cache_collection,
            cut_edges,
            use_op_model_recalculation_on_cut,
            resolve_on_create);
    }

    RemainingOpModels at(graphlib::Node const* node) const;
    const BalancerConfig& get_balancer_config() const { return balancer_config; }
    std::shared_ptr<BalancerCacheCollection> get_balancer_cache_collection() const { return balancer_cache_collection; }
    OpModels* get_selected_op_models_for_buffering(
        std::unordered_set<const tt::graphlib::Node*> const& current_epoch_ops);
    void set(graphlib::Node const* node, OpModel const& op_model, bool skip_update = false);
    void cut(std::vector<graphlib::Edge> const& edge, bool epoch_cut = false);
    std::unique_ptr<graphlib::GraphTraversalContext> get_graph_traversal_context();
    std::unique_ptr<graphlib::GraphTraversalContext> get_graph_epoch_traversal_context(
        const std::unordered_set<const graphlib::Node*>* epoch_nodes);
    const CutEdges& get_cut_edges() const { return cut_edges; }
    const OpModels& get_selected_op_models() const { return selected_op_models; }
    void invalidate_suboptimal_op_models(int invalidation_strategy);
    void set_filter_grid_size(graphlib::Node const* node, OpModel const& role_op_model);
#ifdef DEBUG
    void compute_edge_elimination_debug_info(
        graphlib::Edge& edge,
        Bitset* producer_bitset,
        Bitset* consumer_bitset,
        Bitset& edge_producer_bitset,
        Bitset& edge_consumer_bitset,
        std::vector<OpModel>& producer_op_models_debug,
        std::vector<OpModel>& consumer_op_models_debug,
        std::uint64_t producer_count,
        std::uint64_t consumer_count,
        EdgeConstraintDebugInfo& edge_constraint_debug_info,
        EdgeConstraintDebugInfo& graph_constraint_debug_info);
#endif
    ConstraintInfo const& get_constraint_info() const { return *constraint_info_ptr.get(); }

   private:
    graphlib::Graph* graph;
    std::shared_ptr<SharedData> shared_data;
    BalancerConfig const& balancer_config;
    std::shared_ptr<balancer::BalancerCacheCollection> balancer_cache_collection;
    std::vector<PathSet> path_sets;
    std::vector<Bitset> bitsets;
    OpModels selected_op_models;
    CutEdges cut_edges;
    std::unordered_map<const graphlib::Node*, int> op_model_recompute_version;
    std::unordered_set<const graphlib::Node*> virtual_nodes;
    std::unordered_set<graphlib::Edge> edges_to_ignore;
    std::vector<graphlib::Edge> edges_pending_removal;
    std::vector<std::shared_ptr<graphlib::NodeGraphContainer>> virtual_nodes_management;
    std::unordered_map<graphlib::Edge, PathSetId> path_set_ids;
    std::unordered_map<graphlib::NodeId, BitsetId> bitset_ids;
    std::unordered_map<graphlib::NodeId, Bitset> op_disabled_bitset_cache;
    bool use_op_model_recalculation_on_cut;
    std::unordered_map<std::string, ConstraintFailureReason> failure_reasons;
    std::shared_ptr<ConstraintInfo> constraint_info_ptr;
    int suboptimal_opmodel_invalidation_strategy = 0;
    bool single_core_ip_mode;
};

}  // namespace tt::balancer::legalizer
