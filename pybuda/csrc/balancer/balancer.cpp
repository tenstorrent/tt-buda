// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "balancer/balancer.hpp"

#include <algorithm>
#include <experimental/filesystem>
#include <fstream>
#include <optional>
#include <unordered_set>

#include "balancer/balancer_cache_collection.hpp"
#include "balancer/legalizer/legalizer.hpp"
#include "balancer/policies/policies.hpp"
#include "balancer/policies/policy_types.hpp"
#include "balancer/policies/policy_utils.hpp"
#include "balancer/python_interface.hpp"
#include "graph_lib/node_types.hpp"
#include "passes/passes_utils.hpp"
#include "placer/epoch_placer.hpp"
#include "placer/placer.hpp"
#include "python_bindings_common.hpp"

using NodeType = tt::graphlib::NodeType;

namespace tt::balancer
{

std::ostream& operator<<(std::ostream& stream, PolicyType policy_type)
{
    switch (policy_type)
    {
        case PolicyType::MaximizeTMinimizeGrid: stream << "PolicyType::MaximizeTMinimizeGrid"; break;
        case PolicyType::MinimizeGrid: stream << "PolicyType::MinimizeGrid"; break;
        case PolicyType::Random: stream << "PolicyType::Random"; break;
        case PolicyType::NLP: stream << "PolicyType::NLP"; break;
        case PolicyType::CNN: stream << "PolicyType::CNN"; break;
        case PolicyType::Ribbon: stream << "PolicyType::Ribbon"; break;
        default: stream << "PolicyType::Unknown"; break;
    }
    return stream;
}

std::ostream& operator<<(std::ostream& os, BalancerConfig const& config)
{
    os << "BalancerConfig{" << std::endl;
    os << "  .device_config.arch_name = " << config.device_config.arch_name << std::endl;
    os << "  .device_config.device_yaml = " << config.device_config.device_yaml << std::endl;
    os << "  .policy_type = " << config.policy_type << std::endl;
    os << "  .random_policy_seed = " << config.random_policy_seed << std::endl;
    os << "  .num_chips = " << config.chip_ids.size() << std::endl;
    os << "  .skip_l1_usage_validation = " << config.skip_l1_usage_validation << std::endl;
    os << "  .enable_t_streaming = " << config.enable_t_streaming << std::endl;
    os << "}";
    return os;
}

static std::vector<graphlib::Edge> cut_edges(BalancerConfig const& config, graphlib::Graph const* graph)
{
    // Prevent duplicates coming from config.
    //
    std::unordered_set<graphlib::Edge> edges;

    for (auto const& epoch : config.op_names_to_epoch_break)
    {
        for (auto const& op_name : epoch)
        {
            if (not graph->has_node_with_name(op_name))
                continue;
            graphlib::Node* consumer = graph->get_node_by_name(op_name);
            for (auto edge : graph->operand_data_edges(consumer))
            {
                if (graph->node_by_id(edge.producer_node_id)->node_type() != NodeType::kBudaOp)
                    continue;

                if (edges.count(edge) == 0)
                {
                    edges.insert(edge);
                }
            }
        }
    }

    return std::vector<graphlib::Edge>(edges.begin(), edges.end());
}

legalizer::GraphSolver get_graph_solver(
    BalancerConfig const& config,
    std::shared_ptr<BalancerCacheCollection> cache_collection,
    graphlib::Graph* graph,
    LegalOpModels const& legal_op_models,
    bool use_op_model_recalculation)
{
    if (config.device_config.is_grayskull())
    {
        return legalizer::GraphSolver::create<legalizer::GrayskullConstraint>(
            graph, legal_op_models, config, cache_collection, use_op_model_recalculation, cut_edges(config, graph));
    }
    else if (config.device_config.is_wormhole())
    {
        return legalizer::GraphSolver::create<legalizer::WormholeConstraint>(
            graph, legal_op_models, config, cache_collection, use_op_model_recalculation, cut_edges(config, graph));
    }

    log_fatal("Unknown device arch name: {}", config.device_config.arch_name);
}

static void add_broadcasts_for_sparse_inputs_0_2(
    graphlib::Graph const* graph, graphlib::Node const* node, OpModel const& op_model)
{
    // If fracture factor > 1, inputs have already been properly sized to target eltwise pipes scenario
    // The reason they can't be handled here as well is that fractured convs have different inputs 0 and
    // 2 for each c-dim core
    if (op_model.fracture_factor > 1)
    {
        return;
    }

    // No need to broadcast in this case
    if (op_model.grid_shape.c == 1)
    {
        return;
    }

    std::vector<Edge> in_edges = graph->operand_data_edges(node);
    TT_ASSERT(in_edges.size() == 3 || in_edges.size() == 4);  // 4 with bias

    std::shared_ptr<tt::graphlib::EdgeAttributes> input0_edge_attrs = graph->get_edge_attributes(in_edges[0]);
    std::shared_ptr<tt::graphlib::EdgeAttributes> input2_edge_attrs = graph->get_edge_attributes(in_edges[2]);

    std::vector<tt::graphlib::OpType> input0_edge_tms = input0_edge_attrs->get_tms();
    std::vector<tt::graphlib::OpType> input2_edge_tms = input2_edge_attrs->get_tms();

    // Append tms
    tt::graphlib::OpType broadcast_to_append = graphlib::OpType("broadcast", {3, op_model.grid_shape.c, true}, {});

    input0_edge_tms.push_back(broadcast_to_append);
    input2_edge_tms.push_back(broadcast_to_append);

    input0_edge_attrs->set_tms(input0_edge_tms);
    input2_edge_attrs->set_tms(input2_edge_tms);
}

static void insert_sparse_fracturing_tms(
    graphlib::Graph const* graph, graphlib::Node const* node, OpModel const& op_model)
{
    // Out
    std::vector<Edge> out_edges = graph->user_data_edges(node);
    TT_ASSERT(out_edges.size() == 1);
    std::shared_ptr<tt::graphlib::EdgeAttributes> out_edge_attrs = graph->get_edge_attributes(out_edges[0]);
    std::vector<tt::graphlib::OpType> out_edge_tms = out_edge_attrs->get_tms();

    std::vector<tt::graphlib::OpType> prepend_tms = {
        // Instead of adding hslice(f) + vstack(f), we could just add an hslice(f) and divide the existing vslice's
        // split factor by f (where f = fracture_factor), but the optimize_tms() pass does this for us
        graphlib::OpType("hslice", {op_model.fracture_factor}, {}),
        graphlib::OpType("vstack", {op_model.fracture_factor}, {})};
    out_edge_tms.insert(out_edge_tms.begin(), prepend_tms.begin(), prepend_tms.end());
    out_edge_attrs->set_tms(out_edge_tms);

    // In1
    std::vector<Edge> in_edges = graph->operand_data_edges(node);
    TT_ASSERT(in_edges.size() == 3);
    std::shared_ptr<tt::graphlib::EdgeAttributes> in1_edge_attrs = graph->get_edge_attributes(in_edges[1]);
    std::vector<tt::graphlib::OpType> in1_edge_tms = in1_edge_attrs->get_tms();

    std::vector<tt::graphlib::OpType> append_tms = {
        graphlib::OpType("broadcast", {3, op_model.fracture_factor, true}, {})};
    in1_edge_tms.insert(in1_edge_tms.end(), append_tms.begin(), append_tms.end());
    in1_edge_attrs->set_tms(in1_edge_tms);
}

static void insert_sparse_buffer_op_tms(
    Graph const* graph, graphlib::OpNode* op, balancer::OpModel const& op_model, int fracture_factor)
{
    TT_ASSERT(op_model.has_sparse_buffer());
    TT_ASSERT(fracture_factor == 1);

    if (op_model.grid_shape.r == 1)
        return;

    auto users = graph->user_data_edges(op);
    graphlib::BudaOpNode* user = dynamic_cast<graphlib::BudaOpNode*>(graph->node_by_id(users[0].consumer_node_id));
    bool is_reduce_z = user->op_name() == "reduce" and std::get<std::string>(user->buda_attrs().at("dim")) == "z";
    bool is_matmul = user->is_matmul() and not user->is_sparse_matmul();

    if (not is_reduce_z and not is_matmul)
        return;

    TT_ASSERT(users.size() == 1, "Unsupported, multiple users with different views");

    auto edge_attrs = graph->get_edge_attributes(users[0]);
    auto& tms = edge_attrs->get_tms();
    if (tms.empty())
        return;

    TT_ASSERT(not is_matmul or tms.size() == 2, op->name(), user->name(), tms.size());
    TT_ASSERT(not is_reduce_z or tms.size() == 1, op->name(), user->name(), tms.size());
    if (is_matmul)
    {
        TT_ASSERT(tms[0].op == "vslice");
        TT_ASSERT(tms[1].op == "hstack");
        int vslice_factor = std::get<int>(tms[0].attr[0]);
        int hstack_factor = std::get<int>(tms[1].attr[0]);
        TT_ASSERT(vslice_factor == hstack_factor);
        TT_ASSERT(vslice_factor > 1);
        int factor = vslice_factor;
        int grid_r = op_model.grid_shape.r;
        tms.clear();
        tms.push_back(graphlib::OpType("vslice", {grid_r * factor}, {}));
        tms.push_back(graphlib::OpType("hstack", {factor}, {}));
        tms.push_back(graphlib::OpType("vstack", {grid_r}, {}));
    }
    else if (is_reduce_z)
    {
        TT_ASSERT(false, "This path results in illegal tms, cannot have slice after stack");
        TT_ASSERT(tms[0].op == "vslice");
        int vslice_factor = std::get<int>(tms[0].attr[0]);
        TT_ASSERT(vslice_factor > 1);
        int factor = vslice_factor;
        int grid_r = op_model.grid_shape.r;
        tms.clear();
        tms.push_back(graphlib::OpType("vslice", {grid_r * factor}, {}));
        tms.push_back(graphlib::OpType("hstack", {factor}, {}));
        tms.push_back(graphlib::OpType("vstack", {grid_r}, {}));
        tms.push_back(graphlib::OpType("hslice", {factor}, {}));
    }
}

// Layout dataflow reorders the output buffer of sparse matmul in a way
// such that each row of cores between a sparse/consumer pair has a 1to1
// mapping of tiles and avoids inefficient gathers.  This function erases
// the existing TMs along this path and replaces them with "per row core"
// equivalent set of TMs.  This often results in more complicated TMs, but
// much simpler pipes
static void insert_sparse_dataflow_tms(
    graphlib::Graph const* graph, graphlib::Node const* node, OpModel const& op_model)
{
    for (Edge user : graph->user_data_edges(node))
    {
        auto& tms = graph->get_edge_attributes(user)->get_tms();
        TT_ASSERT(tms.size() >= 1 and tms.size() <= 3);

        bool needs_stack = false;
        int row_slice = op_model.grid_shape.r * op_model.block_shape().rt();
        bool backwards = tms.front().op == "transpose";
        int factor = 0;

        if (backwards)
        {
            TT_ASSERT(tms.size() >= 2);
            TT_ASSERT(tms[1].op == "hslice");
            factor = std::get<int>(tms[1].attr[0]);
            if (tms.size() > 2)
            {
                TT_ASSERT(tms[2].op == "vstack");
                TT_ASSERT(factor == std::get<int>(tms[2].attr[0]));
                needs_stack = true;
            }
        }
        else
        {
            TT_ASSERT(tms[0].op == "vslice");
            factor = std::get<int>(tms[0].attr[0]);
            if (tms.size() > 1)
            {
                TT_ASSERT(tms[1].op == "hstack");
                TT_ASSERT(factor == std::get<int>(tms[1].attr[0]));
                needs_stack = true;
            }
        }
        TT_ASSERT(factor > 1);
        TT_ASSERT(row_slice > 1);

        tms.clear();

        if (backwards)
        {
            tms.push_back(graphlib::OpType("transpose", {}, {}, {{"dim0", 2}, {"dim1", 3}, {"z_dim_slice", -1}}));
            tms.push_back(graphlib::OpType("hslice", {row_slice}, {}));
            tms.push_back(graphlib::OpType("vstack", {factor}, {}));
            tms.push_back(graphlib::OpType("hstack", {row_slice / factor}, {}));
            if (not needs_stack)
                tms.push_back(graphlib::OpType("vslice", {factor}, {}));
        }
        else
        {
            tms.push_back(graphlib::OpType("vslice", {row_slice}, {}));
            tms.push_back(graphlib::OpType("hstack", {factor}, {}));
            tms.push_back(graphlib::OpType("vstack", {row_slice / factor}, {}));
            if (not needs_stack)
                tms.push_back(graphlib::OpType("hslice", {factor}, {}));
        }
        optimize_tms(tms);
    }
}

void print_perf_input_data(
    tt::sparse::EncodingTiles const& buda_indices_all_rows, int sparse_ublock_idx_bits, balancer::OpModel const& op_model)
{
    constexpr int TILE_DIM = tt::sparse::TILE_DIM;
    using IndexType = std::remove_extent_t<decltype(tt::sparse::strip_info_struct::F::index_array)>;
    const int ublock_tile_index_bytes = 16 - sparse_ublock_idx_bits;
    const int grid_r = buda_indices_all_rows.size();

    fmt::print("~~ Node: {}\n", op_model.buda_op_node->name());
    fmt::print("~~~~ t: {}\n", op_model.t_stream_factor.t());
    fmt::print("~~~~ mblock_m: {}\n", op_model.block_shape().mblock_m);
    fmt::print("~~~~ mblock_n: {}\n", op_model.block_shape().mblock_n);
    fmt::print("~~~~ ublock_rt: {}\n", op_model.ublock_shape().rt);
    fmt::print("~~~~ ublock_ct: {}\n", op_model.ublock_shape().ct);
    fmt::print("~~~~ u_kt: {}\n", op_model.input_buffers[0].block_shape.ublock.ct);
    fmt::print(
        "~~~~ m_k (total strips): {}\n",
        op_model.op_shape.inputs[1].rt / op_model.input_buffers[1].block_shape.ublock.rt);

    for (int curr_r = 0; curr_r < grid_r; curr_r++)
    {
        std::vector<std::int32_t> buda_indices = buda_indices_all_rows[curr_r];
        std::uint8_t const* base_ptr = reinterpret_cast<std::uint8_t const*>(buda_indices.data());
        TT_ASSERT((int)buda_indices.size() % (TILE_DIM * TILE_DIM) == 0);

        int cnt_nz_tiles = 0;
        int cnt_nz_ublocks = 0;
        int cnt_nz_strips = 0;

        for (int tile_id = 0; tile_id < (int)(buda_indices.size() / (TILE_DIM * TILE_DIM)); ++tile_id)
        {
            tt::sparse::strip_info_struct const* info = reinterpret_cast<tt::sparse::strip_info_struct const*>(
                base_ptr + tile_id * (TILE_DIM * TILE_DIM * sizeof(std::uint32_t)));

            bool done = false;
            while (not done)
            {
                if (info->f.nz_ublocks > 0)
                {
                    cnt_nz_strips++;
                }

                cnt_nz_ublocks += info->f.nz_ublocks;

                int i = 0;
                for (int ublock_i = 0; ublock_i < info->f.nz_ublocks; ++ublock_i)
                {
                    IndexType encoded = info->f.index_array[i++];
                    IndexType nz_tiles_in_ublock = encoded >> sparse_ublock_idx_bits;
                    nz_tiles_in_ublock =
                        (nz_tiles_in_ublock == 0u) ? (1u << ublock_tile_index_bytes) : nz_tiles_in_ublock;
                    cnt_nz_tiles += nz_tiles_in_ublock;
                    i += nz_tiles_in_ublock;
                }

                done = info->f.last_strip_in_tile;
                info = reinterpret_cast<tt::sparse::strip_info_struct const*>(
                    reinterpret_cast<std::uint8_t const*>(info) + sizeof(tt::sparse::strip_info_struct) +
                    i * sizeof(IndexType));
            }
        }

        fmt::print("~~~~~~ grid_r {}\n", curr_r);
        fmt::print("~~~~~~~~ cnt_nz_tiles: {}\n", cnt_nz_tiles);
        fmt::print("~~~~~~~~ cnt_nz_ublocks: {}\n", cnt_nz_ublocks);
        fmt::print("~~~~~~~~ cnt_nz_strips: {}\n", cnt_nz_strips);
    }
}

void update_ops_on_selected_op_models(graphlib::Graph const* graph, OpModels const& op_models)
{
    for (Node* node : graph->nodes())
    {
        if (node->node_type() == NodeType::kBudaOp)
        {
            graphlib::OpNode* op = node->as<graphlib::OpNode>();
            graphlib::OpType type = op->op_type();
            if (op->is_sparse_matmul())
            {
                TT_LOG_ASSERT(op_models.count(node) > 0, "Missing op model for node: {}", node->name());
                balancer::OpModel op_model = op_models.at(node);

                int grid_r = op_model.grid_shape.r;
                int u_rt = op_model.output_buffers[0].block_shape.ublock.rt;
                int u_kt = op_model.input_buffers[1].block_shape.ublock.rt;
                int u_ct = op_model.output_buffers[0].block_shape.ublock.ct;
                bool has_buffer_op = op_model.has_sparse_buffer();
                bool force_buffer_op_layout = env_as<bool>("PYBUDA_FORCE_SPARSE_BUFFER_LAYOUT");
                bool buffer_op_layout = has_buffer_op or force_buffer_op_layout;
                const sparse::SparseBUDA& sparse_buda =
                    graph->data_operands(node)[0]->as<graphlib::ConstantInputNode>()->get_sparse_buda();
                auto layout = sparse::SparseBUDA::create_layout(
                    buffer_op_layout, op_model.t_stream_factor.dir.z_major(), op_model.fracture_factor);

                std::string visualize_sparse_path =
                    env_as<bool>("PYBUDA_VISUALIZE_SPARSE") ? "sparse_" + op->name() + ".png" : "";
                auto [sparse, encodings, sparse_s, encodings_s, num_strips_per_row] =
                    sparse_buda.get_sparse_tiles_and_encodings(
                        grid_r,
                        op_model.t_stream_factor.r,
                        op_model.t_stream_factor.c,
                        u_rt,
                        u_kt,
                        op_model.fracture_factor,
                        layout,
                        visualize_sparse_path);
                int sparse_tile_ptr_bits =
                    sparse_buda.get_sparse_tile_ptr_bits(grid_r, op_model.t_stream_factor.r, u_rt);
                int sparse_ublock_idx_bits =
                    sparse_buda.get_sparse_ublock_idx_bits(grid_r, op_model.t_stream_factor.r, u_rt);

                if (env_as<bool>("PYBUDA_SPARSE_PRINT_PERF_INPUT_DATA"))
                {
                    print_perf_input_data(encodings, sparse_tile_ptr_bits, op_model);
                }

                auto sparse_utils_module = py::module_::import("pybuda.op.eval.sparse_utils");
                py::function shapeify = sparse_utils_module.attr("shapeify_sparse_tiles_and_encodings");

                // Overwrite input tensors
                auto [sp, enc] = shapeify(sparse, encodings, grid_r, op_model.fracture_factor)
                                     .cast<std::pair<py::object, py::object>>();
                graphlib::ConstantInputNode* cin0 = graph->data_operands(node)[0]->as<graphlib::ConstantInputNode>();
                graphlib::ConstantInputNode* cin2 = graph->data_operands(node)[2]->as<graphlib::ConstantInputNode>();
                cin0->set_tensor_handle(make_shared_py_object(sp));
                cin2->set_tensor_handle(make_shared_py_object(enc));

                // tenstorrent/budabackend#1234
                // tenstorrent/pybuda#504
                // Due to BBE-imposed constraint, we can't have more that 2 operands multicasting
                // BBE changed behavior so that inputs 0&2 use eltwise style pipes instead of row multicast
                // On FE side, we add a broadcast in c-dim to account for this diff
                add_broadcasts_for_sparse_inputs_0_2(graph, node, op_model);

                // Overwrite op attributes
                auto op_attrs = op->op_attrs();
                TT_ASSERT(op_attrs.size() == 15);
                op_attrs[2] = sparse_tile_ptr_bits;
                op_attrs[7] = op_model.fracture_factor;
                op_attrs[8] = u_rt;
                op_attrs[9] = u_kt;
                op_attrs[10] = u_ct;
                op_attrs[11] = op_model.grid_shape.c;
                op_attrs[12] = op_model.t_stream_factor.r;
                op_attrs[13] = op_model.t_stream_factor.c;
                op_attrs[14] = sparse_ublock_idx_bits;
                op->overwrite_op_attrs(op_attrs);

                // Overwrite buda attributes
                auto buda_attrs = op->buda_attrs();
                buda_attrs["num_sparse_tiles"] =
                    static_cast<int>(sparse_s[3] / sparse::TILE_DIM / op_model.fracture_factor);
                buda_attrs["num_index_tiles"] =
                    static_cast<int>(encodings_s[3] / sparse::TILE_DIM / op_model.fracture_factor);
                buda_attrs["sparse_tile_ptr_bits"] = sparse_tile_ptr_bits;
                buda_attrs["sparse_ublock_idx_bits"] = sparse_ublock_idx_bits;
                buda_attrs["fracture_factor"] = op_model.fracture_factor;
                if (has_buffer_op)
                {
                    TT_ASSERT((op_model.grid_shape.c % 2) == 0);
                    std::vector<int> num_nz_strips;
                    int grid_c = op_model.grid_shape.c / 2;
                    int grid_volume = op_model.grid_shape.r * grid_c;
                    num_nz_strips.resize(grid_volume);
                    for (int i = 0; i < grid_volume; ++i) num_nz_strips[i] = num_strips_per_row[i / grid_c];
                    buda_attrs["num_nz_strips"] = num_nz_strips;
                    buda_attrs["act_buffered"] = true;
                }
                op->overwrite_buda_attrs(buda_attrs);

                // Overwrite op attributes
                auto attr = op->op_attrs();
                std::get<int>(attr[2]) = sparse_tile_ptr_bits;
                op->overwrite_op_attrs(attr);

                log_trace(LogBalancer, "  Sparse node {}:", node->name());
                log_trace(LogBalancer, "    {}", op_model.grid_shape);
                log_trace(
                    LogBalancer, "    Num sparse tiles per core: {}:", std::get<int>(buda_attrs["num_sparse_tiles"]));
                log_trace(
                    LogBalancer, "    Num index tiles per core: {}:", std::get<int>(buda_attrs["num_index_tiles"]));

                graph->data_operands(node)[0]->set_shape(graphlib::Shape::create_buda(sparse_s));
                graph->data_operands(node)[2]->set_shape(graphlib::Shape::create_buda(encodings_s));

                if (op_model.fracture_factor > 1)
                {
                    // Update node shape to account for fracture factor
                    tt::graphlib::Shape shape = node->shape().canonical();
                    tt::graphlib::Shape new_shape = graphlib::Shape::create_buda(
                        shape.as_vector()[0],
                        shape.as_vector()[1],
                        shape.as_vector()[2] / op_model.fracture_factor,
                        shape.as_vector()[3] * op_model.fracture_factor);
                    TT_ASSERT(shape.volume() == new_shape.volume());
                    node->set_shape(new_shape);

                    // Insert tms to account for fracture factor
                    insert_sparse_fracturing_tms(graph, node, op_model);
                }

                log_trace(LogBalancer, "Sparse layout {}: {}", op->name(), layout);
                switch (layout)
                {
                    case sparse::SparseBUDA::Layout::BufferOp:
                    {
                        insert_sparse_buffer_op_tms(graph, op, op_model, op_model.fracture_factor);
                        break;
                    }
                    case sparse::SparseBUDA::Layout::ZMajorDataflow:
                    {
                        TT_ASSERT(op_model.fracture_factor == 1);
                        insert_sparse_dataflow_tms(graph, op, op_model);
                        break;
                    }
                    default:
                    {
                        break;
                    }
                }
            }
            else if (type.op == "embedding")
            {
                balancer::OpModel const& op_model = op_models.at(node);

                auto* embedding_table = graph->data_operands(op)[0]->as<graphlib::InputNode>();
                embedding_table->set_layout(BudaQueueLayout::Flat);

                // Reconfigure shape for raw tilized layout
                auto* embedding_indices = graph->data_operands(op)[1]->as<graphlib::InputNode>();
                auto indices_shape = embedding_indices->shape();
                TT_ASSERT(indices_shape[-2] == graphlib::Shape::BUDA_TILE_DIM);
                indices_shape[-2] = indices_shape[-2] * op_model.grid_shape.r;
                indices_shape[-1] = graphlib::align_up_tile(
                    indices_shape[-1] / (op_model.grid_shape.r * graphlib::Shape::BUDA_TILE_DIM));
                embedding_indices->set_shape(indices_shape);

                // Convert num_indices to be per core
                int num_indices = std::get<int>(type.buda_attrs.at("num_indices"));
                num_indices = graphlib::align_up_tile(num_indices);
                TT_ASSERT(num_indices % op_model.grid_shape.r == 0);
                std::get<int>(type.buda_attrs.at("num_indices")) = num_indices / op_model.grid_shape.r;

                op->change_op_type(type);
            }
            else if (type.op == "dropout")
            {
                // Overwrite op attributes
                TT_LOG_ASSERT(op_models.count(node) > 0, "Missing op model for node: {}", node->name());
                balancer::OpModel op_model = op_models.at(node);

                auto attr = op->op_attrs();
                attr[5] = op_model.t_stream_factor.r;
                attr[6] = op_model.t_stream_factor.c;
                attr[7] = op_model.t_stream_factor.dir.r();
                attr[8] = op_model.t_stream_factor.dir.z_major();
                op->overwrite_op_attrs(attr);
            }
            else if (type.op == "splice")
            {
                // Update op attributes
                TT_LOG_ASSERT(op_models.count(node) > 0, "Missing op model for node: {}", node->name());
                balancer::OpModel op_model = op_models.at(node);
                graphlib::UBlockOrder ublock_order = get_output_ublock_order(graph, node);
                op->py_attr<void>(
                    "update_ranges",
                    (ublock_order == graphlib::UBlockOrder::R),  // ublock_is_row_order
                    op_model.ublock_shape().rt,
                    op_model.ublock_shape().ct,
                    op_model.grid_shape.r,
                    op_model.grid_shape.c,
                    op_model.t_stream_factor.r,
                    op_model.t_stream_factor.c);
            }
            else if (type.op == "tilizer")
            {
                auto* input = graph->data_operands(op)[0]->as<graphlib::InputNode>();
                input->set_layout(BudaQueueLayout::Flat);
            }
        }
    }
}

static void insert_input_queues(
    placer::PlacerSolution& placer_solution, const Graph* graph, const OpModelMap& op_models)
{
    // Add input queues to the placer solution
    for (auto [node_name, op_model] : op_models)
    {
        Node* node = graph->get_node_by_name(node_name);
        switch (node->node_type())
        {
            case NodeType::kInput:
            {
                placer_solution.input_queue_to_grid_shape.insert(
                    {node_name,
                     tt::placer::GridShape(
                         (std::uint32_t)op_model.grid_shape.r, (std::uint32_t)op_model.grid_shape.c)});
                break;
            }
            default: break;
        }
    }
}

static std::tuple<OpModelMap, BlockShapeMap, OutputHostTMMap, CutEdges> balancer_passes(
    Graph* graph,
    BalancerConfig& config,
    std::shared_ptr<BalancerCacheCollection> cache_collection,
    std::optional<placer::PlacerSolution>& placer_solution)
{
    log_debug(LogBalancer, "{}", config);
    LegalOpModels valid_op_models = legalizer::get_legal_op_models(graph, config, cache_collection);

    auto graph_solver = get_graph_solver(config, cache_collection, graph, valid_op_models);

    legalizer::GraphSolverSolution graph_solver_solution = run_policy(graph, config, graph_solver, placer_solution);

    update_ops_on_selected_op_models(graph, graph_solver_solution.selected_op_models);

    auto ret = legalizer::resolve_block_shapes(graph, config, graph_solver_solution);

    if (placer_solution.has_value())
        insert_input_queues(placer_solution.value(), graph, std::get<0>(ret));

    return ret;
}

std::shared_ptr<BalancerSolution> run_balancer_and_placer(
    Graph* graph, BalancerConfig& config, std::shared_ptr<BalancerCacheCollection> cache_collection)
{
    log_info("Running Balancer with Policy: {}", config.policy_type);
    PROFILE_SCOPE();

    // New epoch-by-epoch placement loop
    if (config.epoch_by_epoch)
        return placer::run_epoch_placer(&graph, config, cache_collection);

    std::optional<placer::PlacerSolution> opt_placer_solution = std::nullopt;
    auto const& [op_models, block_shape_map, output_host_tms, cut_edges] =
        balancer_passes(graph, config, cache_collection, opt_placer_solution);

    TT_ASSERT(
        graph->virtual_node_count() == 0,
        "After balancer passes are complete we should not have virtual nodes in graph anymore.");

    auto placer_solution =
        opt_placer_solution.has_value() ? opt_placer_solution.value() : run_placer(graph, config, op_models);
    dump_balancer_placer_data(
        graph, config.chip_ids, placer_solution, op_models, std::cout, config.device_config.arch_name);

    return std::make_shared<BalancerSolution>(placer_solution, op_models, block_shape_map, output_host_tms, cut_edges);
}

}  // namespace tt::balancer
