// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "balancer/legalizer/legalizer.hpp"

#include <algorithm>
#include <unordered_set>

#include "autograd/binding.hpp"
#include "balancer/balancer.hpp"
#include "balancer/balancer_cache_collection.hpp"
#include "balancer/balancer_utils.hpp"
#include "balancer/exceptions.hpp"
#include "balancer/python_interface.hpp"
#include "balancer/types.hpp"
#include "graph_lib/defines.hpp"
#include "graph_lib/node.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "passes/fuse_ops.hpp"
#include "passes/print_graph.hpp"
#include "passes/t_stream.hpp"
#include "shared_utils/sparse_matmul_utils.hpp"
#include "utils/logger.hpp"

using NodeType = tt::graphlib::NodeType;
using PortId = tt::graphlib::PortId;
using Edge = tt::graphlib::Edge;

namespace tt::balancer
{

std::ostream& operator<<(std::ostream& os, OpOverride const& op_override)
{
    os << "OpOverride{";
    if (op_override.grid_shape)
        os << " .grid_shape = (" << op_override.grid_shape->first << ", " << op_override.grid_shape->second << ")";
    if (op_override.force_dram_parameters)
        os << " .force_dram_parameters = true";
    if (not op_override.t_stream_dir.empty())
        os << " .t_stream_dir = " << op_override.t_stream_dir;
    if (op_override.t_stream_shape)
        os << " .t_stream_shape = (" << op_override.t_stream_shape->first << ", " << op_override.t_stream_shape->second
           << ")";
    os << " }";
    return os;
}

void OpOverride::apply(
    FactorizedShape& grid_pars,
    bool& force_dram_parameters_out,
    std::vector<TStreamDir>& t_stream_dirs,
    FactorizedShape& overridden_streaming_pars,
    bool& enable_t_streaming,
    const std::string& op_name)
{
    log_debug(LogBalancer, "  {}", *this);

    if (grid_shape)
    {
        auto [r, c] = grid_shape.value();
        grid_pars = grid_pars & FactorizedShape(Parallelization(r, c));
        if (grid_pars.empty())
        {
            log_fatal(
                LogBalancer,
                "Illegal grid shape chosen for op '{}' override, grid_shape: {}",
                op_name,
                GridShape(r, c));
        }
    }

    if (force_dram_parameters.has_value())
    {
        force_dram_parameters_out = force_dram_parameters.value();
    }

    if (t_stream_shape.has_value())
    {
        auto [r, c] = t_stream_shape.value();
        overridden_streaming_pars = FactorizedShape(FactorizedInt::Constant(r), FactorizedInt::Constant(c));
        enable_t_streaming = true;
    }

    if (t_stream_dir == "r")
        t_stream_dirs = {TStreamDir::R};
    else if (t_stream_dir == "c")
        t_stream_dirs = {TStreamDir::C};
    else if (t_stream_dir == "rz")
        t_stream_dirs = {TStreamDir::RZ};
    else if (t_stream_dir == "cz")
        t_stream_dirs = {TStreamDir::CZ};
    else if (t_stream_dir == "n")
        enable_t_streaming = false;
}

std::optional<int> OpOverride::get_fracture_factor()
{
    if (this->fracture_factor.has_value())
    {
        return this->fracture_factor.value();
    }

    return {};
}

std::optional<int> OpOverride::get_u_kt()
{
    if (this->u_kt.has_value())
    {
        return this->u_kt;
    }

    return {};
}

}  // namespace tt::balancer

namespace tt::balancer::legalizer
{

bool validate_sparse_matmul_model(const graphlib::Graph* graph, const graphlib::BudaOpNode* op, const OpModel& op_model)
{
    TT_ASSERT(op->is_sparse_matmul());

    int grid_r = op_model.grid_shape.r;
    int u_rt = op_model.output_buffers[0].block_shape.ublock.rt;
    int u_kt = op_model.input_buffers[1].block_shape.ublock.rt;
    const sparse::SparseBUDA& sparse_buda =
        graph->data_operands(op)[0]->as<graphlib::ConstantInputNode>()->get_sparse_buda();

    int sparse_tile_ptr_bits = sparse_buda.get_sparse_tile_ptr_bits(grid_r, op_model.t_stream_factor.r, u_rt);
    int sparse_ublock_idx_bits = sparse_buda.get_sparse_ublock_idx_bits(grid_r, op_model.t_stream_factor.r, u_rt);
    if (sparse_tile_ptr_bits < 0 or sparse_ublock_idx_bits < 0)
    {
        return false;
    }

    // Calculate bits needed for ublock tile indices (u_rt + u_kt separately encoded)
    //
    constexpr int kMaxBits = 16;
    int u_rt_bits = sparse::get_u_rt_encoding_bits(u_rt);
    int u_kt_bits = sparse::get_u_kt_encoding_bits(u_kt);
    if (sparse_tile_ptr_bits + u_rt_bits + u_kt_bits > kMaxBits)
    {
        return false;
    }

    return true;
}

static bool edge_tms_consume_rz_major(Graph const* graph, graphlib::Edge edge)
{
    // Must come from sparse matmul
    // Must do vslice + hstack pattern
    Node* node = graph->node_by_id(edge.producer_node_id);
    if (node->node_type() == NodeType::kQueue)
    {
        node = graph->data_operands(node).back();
    }

    graphlib::OpNode* op = dynamic_cast<graphlib::OpNode*>(node);
    if (not(op and op->is_sparse_matmul()))
        return false;

    auto const& tms = graph->get_edge_attributes(edge)->get_tms();
    if (tms.empty())
        return false;

    int internal_slice_stack_factor = 1;
    for (auto const& tm : tms)
    {
        if (tm.op == "vslice")
        {
            internal_slice_stack_factor *= std::get<int>(tm.attr[0]);
        }
        else if (tm.op == "hstack")
        {
            if (internal_slice_stack_factor % std::get<int>(tm.attr[0]) == 0)
                internal_slice_stack_factor /= std::get<int>(tm.attr[0]);
            else
                internal_slice_stack_factor = 0;
        }
        else
        {
            internal_slice_stack_factor = 0;
        }
    }
    return internal_slice_stack_factor == 1;
}

static std::vector<TStreamDir> get_legal_streaming_dirs(Graph const* graph, graphlib::BudaOpNode const* op_node)
{
    auto operands = graph->operand_data_edges(op_node);
    bool has_z = std::any_of(
        operands.begin(), operands.end(), [graph](Edge edge) { return post_tms_shape(graph, edge).z() > 1; });
    bool tms_consume_rz_major = std::any_of(
        operands.begin(), operands.end(), [graph](Edge edge) { return edge_tms_consume_rz_major(graph, edge); });
    bool sparse_matmul_bcast_factor =
        (op_node->is_sparse_matmul() and
         graph->data_operands(op_node)[0]->as<graphlib::ConstantInputNode>()->get_sparse_buda().bcast_factor > 1);
    bool is_reduce_z = graphlib::is_reduce_z(op_node);
    if (is_reduce_z or tms_consume_rz_major)
        return {TStreamDir::R};
    else if (has_z or sparse_matmul_bcast_factor)
        return {TStreamDir::R, TStreamDir::C, TStreamDir::RZ};
    else
        return {TStreamDir::R, TStreamDir::C};
}

static FactorizedInt get_fracture_factorization(
    graphlib::Graph const* graph, graphlib::BudaOpNode const* op_node, std::optional<OpOverride> op_override)
{
    bool fracturization_disable = env_as<bool>("PYBUDA_FRACTURIZATION_DISABLE");
    if (fracturization_disable)
        return FactorizedInt(1);

    if (not op_node->is_sparse_matmul())
        return FactorizedInt(1);

    FactorizedInt fracture_factorization(
        graph->data_operands(op_node)[0]->as<graphlib::ConstantInputNode>()->get_sparse_buda().fracture_factor);

    if (op_override)
    {
        if (auto fracture_factor = op_override->get_fracture_factor())
        {
            fracture_factorization =
                fracture_factorization & FactorizedInt(FactorizedInt::Constant(fracture_factor.value()));

            if (fracture_factorization.empty())
            {
                log_fatal(
                    LogBalancer, "Illegal fracture factor chose for override, factor: {}", fracture_factor.value());
            }
        }
    }

    return fracture_factorization;
}

std::optional<int> get_output_buffer_override(
    graphlib::BudaOpNode const* op_node, std::optional<OpOverride> op_override)
{
    if (op_override and op_override->output_buffer_multiplier)
    {
        log_warning(
            LogBalancer,
            "Internal Override: User is overriding output buffer factor for op {} to {}",
            op_node->op_name(),
            op_override->output_buffer_multiplier.value());
        return op_override->output_buffer_multiplier.value();
    }
    return {};
}

std::map<std::uint32_t, std::uint32_t> get_min_input_buffer_multiplier_overrides(std::optional<OpOverride> op_override)
{
    if (op_override and op_override->input_buffer_multiplier)
    {
        return op_override->input_buffer_multiplier.value();
    }
    return {};
}

static int get_u_kt(std::optional<OpOverride> op_override)
{
    if (op_override)
    {
        if (auto u_kt = op_override->get_u_kt())
        {
            return u_kt.value();
        }
    }

    return 0;
}

static int get_output_buffer_factor(
    graphlib::BudaOpNode const*, int calculated_user_buffer_factor, std::optional<int> output_buffer_factor_override)
{
    int output_buffer_factor = calculated_user_buffer_factor * 2;  // double buffer
    if (output_buffer_factor_override)
    {
        output_buffer_factor = output_buffer_factor_override.value();
    }
    return output_buffer_factor;
}

static std::pair<FactorizedShape, LegalSparseUKts> calculate_streaming_pars(
    Graph const* graph,
    graphlib::BudaOpNode const* op_node,
    Parallelization grid_par,
    FactorizedShape all_pars,
    TStreamDir dir,
    int fracture_factor)
{
    bool is_reduce_z = graphlib::is_reduce_z(op_node);
    int operand_z_dim = graph->operands(op_node)[0]->shape().z();

    if (is_reduce_z and operand_z_dim != 1)
        return std::make_pair(FactorizedShape(1, 1), LegalSparseUKts{});

    if (op_node->is_embedding())
        return std::make_pair(FactorizedShape(1, 1), LegalSparseUKts{});

    if (op_node->is_sparse_matmul() and dir.r())
    {
        // Get lhs sparse tensor
        sparse::SparseBUDA& sparse_buda =
            graph->data_operands(op_node)[0]->as<graphlib::ConstantInputNode>()->get_sparse_buda();
        std::vector<tt::sparse::SparseCOO>& sparse_zs = sparse_buda.sparse_zs;
        auto layout = sparse::SparseBUDA::create_layout(dir.z_major(), fracture_factor);
        int bcast_factor =
            (layout == sparse::SparseBUDA::Layout::ZMajor || layout == sparse::SparseBUDA::Layout::ZMajorDataflow)
                ? sparse_buda.bcast_factor
                : 1;

        // Each potential t needs to evenly divide output's r-dim but also in1's r-dim (in0's c-dim)
        std::vector<graphlib::Edge> operands = graph->operand_data_edges(op_node);
        TT_ASSERT(operands.size() == 3);
        graphlib::Shape rhs_shape = post_tms_shape(graph, operands[1]);
        FactorizedInt inner_dim(rhs_shape.rt());
        FactorizedInt total_t = all_pars.r / FactorizedInt::Constant(grid_par.r * bcast_factor);

        LegalSparseUKts r_para_to_legal_u_kts = sparse::SparseBUDA::get_par_t_values(
            grid_par.r,
            total_t.get_factors(),
            sparse_zs,
            inner_dim.get_factors(),
            sparse_buda.bcast_factor,
            fracture_factor,
            layout);

        std::vector<int> r_para;
        r_para.reserve(r_para_to_legal_u_kts.size());
        for (auto const& [para, u_kts] : r_para_to_legal_u_kts) r_para.push_back(para);
        std::sort(r_para.begin(), r_para.end());

        log_trace(LogBalancer, "Streaming dir: {} -- Found t values: {} // {}", dir, fmt::join(r_para, ", "), total_t);

        return std::make_pair(FactorizedShape(FactorizedInt(r_para.begin(), r_para.end()), 1), r_para_to_legal_u_kts);
    }
    else if (op_node->is_sparse_matmul() and dir.c() and fracture_factor > 1)
    {
        return std::make_pair(FactorizedShape(1, 1), LegalSparseUKts{});
    }

    FactorizedInt r(FactorizedInt::FactorRange(grid_par.r, all_pars.r.get_max_factor()));
    FactorizedInt c(FactorizedInt::FactorRange(grid_par.c, all_pars.c.get_max_factor()));
    r = r / FactorizedInt::Constant(grid_par.r);
    c = c / FactorizedInt::Constant(grid_par.c);

    if (dir.r())
    {
        c = 1;
    }
    else if (dir.c())
    {
        r = 1;
    }

    if (op_node->is_fused_op())
    {
        auto fused_op = op_node->get_fused_op();
        if (fused_op->has_matmul_op())
        {
            r = 1;
            c = 1;
        }

        if (fused_op->has_broadcast_c())
        {
            c = 1;
        }

        if (fused_op->has_reduce_op())
        {
            std::uint32_t dim = fused_op->get_reduce_dim();
            if (dim == 2)
                r = 1;
            else if (dim == 3)
                c = 1;
        }
    }

    return std::make_pair(FactorizedShape(r, c), LegalSparseUKts{});
}

static bool streaming_unsupported_op(const std::string& op_type_name)
{
    // TODO Consider adding enum for all OPs and converting to switch-case.
    //
    if ("embedding" == op_type_name)
    {
        return true;
    }

    return false;
}

static std::pair<FactorizedShape, LegalSparseUKts> calculate_streaming_pars(
    Graph const* graph,
    graphlib::BudaOpNode const* op_node,
    Parallelization grid_par,
    FactorizedShape all_pars,
    TStreamDir dir,
    FactorizedShape overridden_pars,
    bool enable_t_streaming,
    int fracture_factor)
{
    if (not enable_t_streaming or streaming_unsupported_op(op_node->op_type().op))
    {
        return std::make_pair(FactorizedShape(1, 1), LegalSparseUKts{});
    }

    auto [streaming_pars, legal_sparse_u_kts] =
        calculate_streaming_pars(graph, op_node, grid_par, all_pars, dir, fracture_factor);

    if (not overridden_pars.empty())
        streaming_pars = streaming_pars & overridden_pars;

    return std::make_pair(streaming_pars, legal_sparse_u_kts);
}

static std::vector<int> enumerate_factored_u_kts(OpModel const& op_model, int user_overriden_u_kt, bool enabled)
{
    if (not enabled)
        return {};

    // If u_kt is user-overriden, then don't test all possible u_kts
    if (user_overriden_u_kt > 0)
        return {};

    if (op_model.op_type() != "matmul")
        return {};

    auto factors = FactorizedInt(op_model.input_buffers[1].block_shape.ublock.rt).get_factors();
    TT_ASSERT(not factors.empty());
    factors.pop_back();  // The initial op model holds the last factor
    return factors;
}

// Remove legacy path, once fork/join hangs are removed:
//   tenstorrent/pybuda#1697
static UBlockShape calculate_ublock_shape_legacy(
    OpShape op_shape,
    Parallelization par,
    std::size_t dst_size_tiles,
    UBlockOrder ublock_order,
    const OpType& op_type,
    bool is_splice,
    bool is_sparse_matmul,
    bool is_embedding,
    bool is_tilize)
{
    // 2 * 4 = Half Dest
    constexpr int kMaxUBlockR = 2;
    constexpr int kMaxUBlockC = 4;
    auto max_pot_multiple = [](int a) -> int { return (1 << __builtin_ctz(a)); };
    auto is_pot = [](int a) { return (a & (a - 1)) == 0; };

    int max_ublock_volume = (int)dst_size_tiles;
    TT_ASSERT(is_pot(max_ublock_volume));
    TT_ASSERT(is_pot(kMaxUBlockR));
    TT_ASSERT(is_pot(kMaxUBlockC));

    UBlockShape ublock;

    TensorShape tensor = op_shape.outputs[0];
    TT_ASSERT(tensor.rt % par.r == 0);
    TT_ASSERT(tensor.ct % par.c == 0);
    int block_rt = tensor.rt / par.r;
    int block_ct = tensor.ct / par.c;

    if (is_splice)
    {
        for (auto input : op_shape.inputs)
        {
            block_rt = gcd(input.rt, block_rt);
            block_ct = gcd(input.ct, block_ct);
        }

        // Splice ublock size must be a factor of length and stride
        int dim = op_type.get_attr_as<int>("dim");
        for (auto [index, num_tile_length, num_tile_stride] :
             op_type.get_attr_as<std::vector<std::tuple<int, int, int>>>("canonical_ranges"))
        {
            if (dim == 2)
            {
                if (index > 0)
                    block_rt = gcd(index, block_rt);
                block_rt = gcd(index + num_tile_length, block_rt);
                block_rt = gcd(index + num_tile_stride, block_rt);
            }
            else if (dim == 3)
            {
                if (index > 0)
                    block_ct = gcd(index, block_ct);
                block_ct = gcd(index + num_tile_length, block_ct);
                block_ct = gcd(index + num_tile_stride, block_ct);
            }
        }
    }

    if (is_sparse_matmul)
    {
        // For sparse matmul we use a different ublock heurisctic to always maximize its volume
        int r_major_ublock_r = FactorizedInt(block_rt).get_nearest_factor_le(max_ublock_volume);
        int r_major_ublock_c = FactorizedInt(block_ct).get_nearest_factor_le(max_ublock_volume / r_major_ublock_r);
        int c_major_ublock_c = FactorizedInt(block_ct).get_nearest_factor_le(max_ublock_volume);
        int c_major_ublock_r = FactorizedInt(block_rt).get_nearest_factor_le(max_ublock_volume / c_major_ublock_c);
        bool r_major = (r_major_ublock_r * r_major_ublock_c) > (c_major_ublock_r * c_major_ublock_c);
        ublock.rt = r_major ? r_major_ublock_r : c_major_ublock_r;
        ublock.ct = r_major ? r_major_ublock_c : c_major_ublock_c;
        return ublock;
    }

    int max_ublock_r = is_embedding || is_tilize ? 1 : std::min(kMaxUBlockR, max_ublock_volume);
    int max_ublock_c = std::min(kMaxUBlockC, max_ublock_volume);

    // Maximize ublock, precidence to anti-ublock order
    if (ublock_order == UBlockOrder::C)
    {
        ublock.rt = std::min(max_pot_multiple(block_rt), max_ublock_r);
        ublock.ct = std::min({max_pot_multiple(block_ct), max_ublock_volume / ublock.rt, max_ublock_c});
    }
    else
    {
        ublock.ct = std::min(max_pot_multiple(block_ct), max_ublock_c);
        ublock.rt = std::min({max_pot_multiple(block_rt), max_ublock_volume / ublock.ct, max_ublock_r});
    }

    TT_ASSERT(block_rt % ublock.rt == 0);
    TT_ASSERT(block_ct % ublock.ct == 0);
    return ublock;
}

static std::pair<UBlockShape, std::unordered_map<std::string, balancer::UBlockShape>> calculate_ublock_shape_legacy(
    OpShape op_shape,
    Parallelization total_par,
    std::size_t dst_size_tiles,
    UBlockOrder ublock_order,
    graphlib::BudaOpNode const* op_node)
{
    UBlockShape ublock = calculate_ublock_shape_legacy(
        op_shape,
        total_par,
        dst_size_tiles,
        ublock_order,
        op_node->op_type(),
        op_node->op_name() == "splice",
        op_node->is_sparse_matmul(),
        op_node->is_embedding(),
        op_node->is_tilize());

    std::unordered_map<std::string, balancer::UBlockShape> fused_op_ublock_shape;
    if (op_node->is_fused_op())
    {
        auto fused_op = op_node->get_fused_op();
        for (auto const& sch : fused_op->get_schedules())
        {
            for (auto const& op : sch.ops)
            {
                fused_op_ublock_shape.insert(std::make_pair(
                    op.name,
                    calculate_ublock_shape_legacy(
                        op.op_shape, total_par, dst_size_tiles, ublock_order, op.op_type, false, false, false, false)));
            }
        }
    }

    return std::make_pair(ublock, fused_op_ublock_shape);
}

static FactorizedShape calculate_ublock_shape(
    OpShape op_shape, Parallelization par, std::size_t dst_size_tiles, const OpType& op_type)
{
    TensorShape tensor = op_shape.outputs[0];
    TT_ASSERT(tensor.rt % par.r == 0);
    TT_ASSERT(tensor.ct % par.c == 0);
    int block_rt = tensor.rt / par.r;
    int block_ct = tensor.ct / par.c;
    int max_ublock_volume = static_cast<int>(dst_size_tiles);
    TT_ASSERT(max_ublock_volume > 0);

    if (op_type.op == "splice")
    {
        for (auto input : op_shape.inputs)
        {
            block_rt = gcd(input.rt, block_rt);
            block_ct = gcd(input.ct, block_ct);
        }

        // Splice ublock size must be a factor of length and stride
        int dim = op_type.get_attr_as<int>("dim");
        for (auto [index, num_tile_length, num_tile_stride] :
             op_type.get_attr_as<std::vector<std::tuple<int, int, int>>>("canonical_ranges"))
        {
            if (dim == 2)
            {
                block_rt = gcd(index, block_rt);
                block_rt = gcd(index + num_tile_length, block_rt);
                block_rt = gcd(index + num_tile_stride, block_rt);
            }
            else if (dim == 3)
            {
                block_ct = gcd(index, block_ct);
                block_ct = gcd(index + num_tile_length, block_ct);
                block_ct = gcd(index + num_tile_stride, block_ct);
            }
        }
    }

    bool is_embedding = op_type.op == "embedding";
    bool is_tilize = op_type.op == "tilizer";
    int max_ublock_r = (is_embedding or is_tilize) ? 1 : max_ublock_volume;
    int max_ublock_c = max_ublock_volume;
    max_ublock_r = FactorizedInt(block_rt).get_nearest_factor_le(max_ublock_r);
    max_ublock_c = FactorizedInt(block_ct).get_nearest_factor_le(max_ublock_c);
    return FactorizedShape(max_ublock_r, max_ublock_c);
}

static std::pair<UBlockShape, std::unordered_map<std::string, balancer::UBlockShape>> calculate_ublock_shape(
    OpShape op_shape,
    Parallelization total_par,
    std::size_t dst_size_tiles,
    UBlockOrder ublock_order,
    graphlib::BudaOpNode const* op_node)
{
    if (env_as<bool>("PYBUDA_LEGACY_UBLOCK_SHAPE"))
        return calculate_ublock_shape_legacy(op_shape, total_par, dst_size_tiles, ublock_order, op_node);

    FactorizedShape ublock_factors = calculate_ublock_shape(op_shape, total_par, dst_size_tiles, op_node->op_type());

    // All subops + top level op of fused op must have the same ublock shape
    if (op_node->is_fused_op())
    {
        auto fused_op = op_node->get_fused_op();
        for (auto const& sch : fused_op->get_schedules())
        {
            for (auto const& op : sch.ops)
            {
                FactorizedShape sub_op_ublock_factors =
                    calculate_ublock_shape(op.op_shape, total_par, dst_size_tiles, op.op_type);
                ublock_factors = ublock_factors & sub_op_ublock_factors;
            }
        }
    }

    // always maximize its volume
    UBlockShape ublock(1, 1);
    for (auto candidate : ublock_factors)
    {
        if (candidate.volume() > (int)dst_size_tiles)
            continue;

        // It's generally better to bias one dimension, either r major or c major so that back to back ops with
        // similar tensor shapes are more likely end up with the same ublock shape and reduce reblocking
        // Arbitrarily bias r-major, i.e. wider ublocks
        bool r_major_bias = (candidate.volume() == ublock.volume() and candidate.c > ublock.ct);
        if (candidate.volume() > ublock.volume() or r_major_bias)
            ublock = UBlockShape(candidate.r, candidate.c);
    }

    // All subops + top level op of fused op must have the same ublock shape
    std::unordered_map<std::string, balancer::UBlockShape> fused_op_ublock_shape;
    if (op_node->is_fused_op())
    {
        auto fused_op = op_node->get_fused_op();
        for (auto const& sch : fused_op->get_schedules())
        {
            for (auto const& op : sch.ops)
            {
                fused_op_ublock_shape[op.name] = ublock;
            }
        }
    }

    return std::make_pair(ublock, fused_op_ublock_shape);
}

static std::tuple<int, bool, bool, bool> calculate_user_buffer_factor(
    Graph const* graph, graphlib::BudaOpNode const* op_node, UBlockOrder ublock_order, OpModel op_model)
{
    //
    // Returns a tuple (factor, can_stream, is_legal_stack_for_grid)
    //   Used as a multiplier on the mblock to denote how many mblocks we need to buffer
    //   if (can_stream == true) then we are allowed to slice the mblock into t
    //

    bool can_stream_due_to_operands = true;
    bool can_stream_due_to_users = true;
    bool is_legal_stack_for_grid = true;
    TStreamFactor t_stream_factor = op_model.t_stream_factor;
    std::vector<Edge> operands = graph->operand_data_edges(op_node);
    for (Edge operand : operands)
    {
        auto edge_attrs = graph->get_edge_attributes(operand);
        int hstack_factor = 1;
        int vstack_factor = 1;
        int hslice_factor = 1;
        int vslice_factor = 1;

        for (graphlib::OpType const& tm : edge_attrs->get_tms())
        {
            if (tm.op == "hslice")
            {
                int slice_factor = std::get<int>(tm.attr[0]);
                hslice_factor *= slice_factor;
            }
            else if (tm.op == "vslice")
            {
                int slice_factor = std::get<int>(tm.attr[0]);
                vslice_factor *= slice_factor;
            }
            else if (tm.op == "hstack")
            {
                int stack_factor = std::get<int>(tm.attr[0]);
                hstack_factor *= stack_factor;
            }
            else if (tm.op == "vstack")
            {
                int stack_factor = std::get<int>(tm.attr[0]);
                vstack_factor *= stack_factor;
            }
        }

        int total_stack_factor = hstack_factor * vstack_factor;
        int total_slice_factor = hslice_factor * vslice_factor;
        int stack_factor = total_stack_factor / total_slice_factor;
        if (stack_factor > 1)
        {
            can_stream_due_to_operands &=
                t_stream_factor.dir.z_major() or
                (divisible_either_direction(vstack_factor, vslice_factor * t_stream_factor.r) and
                 divisible_either_direction(hstack_factor, hslice_factor * t_stream_factor.c) and
                 (not op_node->is_matmul() or
                  ((op_node->is_matmul() and operand.consumer_input_port_id == 0 and t_stream_factor.dir.r()) or
                   (op_node->is_matmul() and operand.consumer_input_port_id == 1 and t_stream_factor.dir.c()))));

            int grid_dim = (t_stream_factor.dir == TStreamDir::R) ? op_model.grid_shape.r : op_model.grid_shape.c;
            is_legal_stack_for_grid &= divisible_either_direction(total_stack_factor, grid_dim);
        }
    }

    std::vector<Edge> users = graph->user_data_edges(op_node);
    int buffer_factor = 1;
    for (Edge user : users)
    {
        graphlib::Node* user_node = graph->node_by_id(user.consumer_node_id);

        // For now, disable streaming for loopback, it can cause the gradient queue and parameter
        // to stream differently and therefore have a different shape.  We need to support consteval
        // on gradient queues or some other solution.
        if (user_node->node_type() == graphlib::NodeType::kInput)
        {
            can_stream_due_to_users = false;
        }

        // Only applies to users on the same epoch
        if (user_node->get_epoch_type() != op_node->get_epoch_type())
        {
            continue;
        }

        // Can always stream through queues
        if (user_node->node_type() == graphlib::NodeType::kQueue)
        {
            continue;
        }

        auto shape = op_node->shape();
        auto edge_attrs = graph->get_edge_attributes(user);
        int total_stack_factor = 1;
        int total_slice_factor = 1;
        bool needs_stack_factor = false;
        bool needs_slice_factor = false;

        for (graphlib::OpType const& tm : edge_attrs->get_tms())
        {
            if (tm.op == "hslice" or tm.op == "vslice")
            {
                int slice_factor = std::get<int>(tm.attr[0]);
                needs_slice_factor |=
                    ((tm.op == "hslice" and (ublock_order == UBlockOrder::R or t_stream_factor.dir.r()) and
                      shape.rt() > 1) or
                     (tm.op == "vslice" and (ublock_order == UBlockOrder::C or t_stream_factor.dir.c()) and
                      shape.ct() > 1));
                can_stream_due_to_users &=
                    t_stream_factor.dir.z_major() or
                    (not needs_slice_factor and divisible_either_direction(slice_factor, t_stream_factor.t()));
                total_slice_factor *= slice_factor;
            }
            else if (tm.op == "hstack" or tm.op == "vstack")
            {
                int stack_factor = std::get<int>(tm.attr[0]);
                needs_stack_factor |=
                    ((tm.op == "hstack" and (ublock_order == UBlockOrder::R or t_stream_factor.dir.r()) and
                      shape.rt() > 1) or
                     (tm.op == "vstack" and (ublock_order == UBlockOrder::C or t_stream_factor.dir.c()) and
                      shape.ct() > 1));
                can_stream_due_to_users &=
                    t_stream_factor.dir.z_major() or
                    (not needs_stack_factor and divisible_either_direction(stack_factor, t_stream_factor.t()));
                total_stack_factor *= stack_factor;
            }
            else if (tm.op == "transpose")
            {
                can_stream_due_to_users &= (t_stream_factor.r == 1 or t_stream_factor.c == 1);
                auto producer_ublock_order = ublock_order;
                auto consumer_ublock_order = edge_attrs->get_ublock_order();

                // Check if user feeds graph output queue
                auto consumer_users = graph->users(user_node);
                bool feeds_graph_output_queue =
                    consumer_users.size() == 1 and consumer_users[0]->node_type() == graphlib::NodeType::kOutput;
                bool producer_is_one_tile_wide_or_tall = (shape.rt() == 1 or shape.ct() == 1);

                // Ublock order needs to swap through transpose except for directly feeding graph output queue or matmul
                // (matmul requires ublock order)
                can_stream_due_to_users &= (producer_ublock_order != consumer_ublock_order) or producer_is_one_tile_wide_or_tall or
                              feeds_graph_output_queue;
            }
            else if (tm.op == "broadcast")
            {
                int dim = std::get<int>(tm.attr[0]);
                if (dim == 3 and ublock_order == UBlockOrder::C)
                {
                    can_stream_due_to_users = false;
                }
                else if (dim == 2 and ublock_order == UBlockOrder::R)
                {
                    can_stream_due_to_users = false;
                }
                else
                {
                    // Cannot stream period if bcast on z
                    can_stream_due_to_users = false;
                }
            }
            else if (tm.op == "buda_unpad")
            {
                int r_pad = std::get<int>(tm.attr[0]);
                int c_pad = std::get<int>(tm.attr[1]);
                if ((r_pad and t_stream_factor.dir.c()) or (c_pad and t_stream_factor.dir.r()))
                {
                    can_stream_due_to_users = false;
                }
            }

            shape = ::get_tm_shape(tm, shape, true);
        }

        if (user_node->as<graphlib::TaggedNode>()->has_tag("padding_nop"))
        {
            can_stream_due_to_users = false;
        }

        auto is_partial_datacopy_edge = [](Edge e) { return (e.edge_type == graphlib::EdgeType::kPartialDataCopy); };
        std::vector<graphlib::Edge> partial_datacopy_edges = graph->user_edges(user_node, is_partial_datacopy_edge);
        if (user_node->node_type() == graphlib::NodeType::kOutput and partial_datacopy_edges.empty())
        {
            // Host runtime outputs cannot support undoing z major order
            can_stream_due_to_users &= not t_stream_factor.dir.z_major() and not env_as<bool>("PYBUDA_DISABLE_STREAM_OUTPUT");
        }

        int stack_factor = (total_stack_factor / total_slice_factor) * needs_stack_factor;
        int slice_factor = (total_slice_factor / total_stack_factor) * needs_slice_factor;
        if (stack_factor > 1 or slice_factor > 1)
            can_stream_due_to_users = false;

        buffer_factor = std::max(buffer_factor, stack_factor);
    }

    return std::make_tuple(buffer_factor, can_stream_due_to_operands, can_stream_due_to_users, is_legal_stack_for_grid);
}

static bool legal_t_streaming(BlockShape block_shape, TStreamFactor t_stream_factor, UBlockOrder ublock_order)
{
    if (t_stream_factor.none())
        return true;

    if (t_stream_factor.dir.is_ublock_order(ublock_order))
    {
        return t_stream_factor.dir.r() ? ((block_shape.mblock_m == 1) or (t_stream_factor.c == 1))
                                       : ((block_shape.mblock_n == 1) or (t_stream_factor.r == 1));
    }
    else
    {
        // For now this is over-constrained, this is allowed to simply be `return true`.
        // Below is to reduce combinations / probably reduces chances of limiting matmul
        return t_stream_factor.dir.r() ? (t_stream_factor.c == 1) : (t_stream_factor.r == 1);
    }
}

static std::vector<BufferModel> calculate_output_buffer_models_for_grid(
    OpShape op_shape,
    GridShape selected_grid,
    UBlockShape ublock,
    TStreamFactor t_stream_factor,
    int output_buffer_factor,
    bool is_gradient_op,
    DataFormat output_df)
{
    std::vector<BufferModel> output_buffers;

    for (int output_idx = 0; output_idx < (int)op_shape.outputs.size(); ++output_idx)
    {
        // Block shape is determined by the combination of
        // (how much we parallelized into the grid) * (how much we parallelized into t)
        int par_r = selected_grid.r * t_stream_factor.r;
        int par_c = selected_grid.c * t_stream_factor.c;
        int par_t = t_stream_factor.t();
        TT_ASSERT(par_r and par_c and par_t, par_r, par_c, par_t);
        // All outputs need to have the same dim & parallelization
        TensorShape output = op_shape.outputs[output_idx];
        BlockShape block_shape(output, par_r, par_c, par_t, ublock);
        output_buffers.emplace_back(block_shape, is_gradient_op ? 1 : output_buffer_factor, output_df);
    }

    return output_buffers;
}

// Helper method to check if the given node is an input node that represents
// a parameter, constant, or optimizer parameter.
// Returns true for input node where data doesn't change for each microbatch item.
static bool is_input_node_parameter_or_constant(const graphlib::Node* node)
{
    return node->node_type() == graphlib::NodeType::kInput and
           (node->as<graphlib::InputNode>()->is_parameter() or node->as<graphlib::InputNode>()->is_constant() or
            node->as<graphlib::InputNode>()->is_optimizer_parameter());
}

// Calculates the required parameter buffer size for parameters depending on two prefetch scenarios:
// 1. Pre-TM parameter prefetch:
//    - parameter buffer: calculated for parameters in Pre-TM shape, distributed equally across the op grid
//    - input buffer: calculated later in the `calculate_input_buffer_models` function
// 2. Post-TM parameter prefetch (optimization that uses more l1 memory in some cases, thus it is not always used):
//    - parameter buffer: calculated for parameters in Post-TM shape, grid size is calculated 
//      in 'op_model.get_input_grid_shape' method
//    - input buffer: not needed as the TMs are already evaluated and the kernel directly reads
//      from the parameter buffer
//
static BufferModel calculate_parameter_buffer_model_for_grid(
    const OpModel& op_model,
    std::size_t input_idx,
    DataFormat parameter_df,
    bool is_post_tm_prefetch)
{
    const TensorShape& pre_tm_shape = op_model.op_shape.producer_shapes[input_idx];
    const TensorShape& post_tm_shape = op_model.op_shape.inputs[input_idx];
    TensorShape parameter_shape = is_post_tm_prefetch ? post_tm_shape : pre_tm_shape;
    GridShape parameter_grid = is_post_tm_prefetch ? op_model.get_input_grid_shape(input_idx) : op_model.grid_shape;

    // Ensure that parameter_shape is divisible by grid_r/grid_c in the respective dimensions, as there are cases where:
    // - parameter_shape.rt % parameter_grid.r != 0 or parameter_shape.ct % parameter_grid.c != 0
    // - parameter_shape.rt < parameter_grid.r or parameter_shape.ct < parameter_grid.c
    //
    int grid_r = FactorizedInt(parameter_shape.rt).get_nearest_factor_le(parameter_grid.r);
    int grid_c = FactorizedInt(parameter_shape.ct).get_nearest_factor_le(parameter_grid.c);

    BlockShape parameter_block_shape(parameter_shape, grid_r, grid_c, 1, UBlockShape(1, 1));
    return BufferModel(parameter_block_shape, 1, parameter_df, is_post_tm_prefetch);
}

// Allocate parameter buffers using the Pre-TM prefetch type to minimize l1 memory usage
//
static std::vector<BufferModel> calculate_parameter_buffer_models_for_grid(
    const balancer::OpModel& op_model,
    const std::vector<graphlib::Node*>& operands,
    bool force_dram_parameters)
{
    std::vector<BufferModel> parameter_buffers(operands.size());
    if (force_dram_parameters) 
    {
        return parameter_buffers;
    }

    for (std::size_t input_idx = 0; input_idx < operands.size(); ++input_idx)
    {
        if (is_input_node_parameter_or_constant(operands[input_idx]))
        {
            parameter_buffers[input_idx] = calculate_parameter_buffer_model_for_grid(
                op_model,
                input_idx,
                operands[input_idx]->output_df(),
                false /*is_post_tm_prefetch*/);
        }
    }

    return parameter_buffers;
}

// Attempt to switch parameter buffers from Pre-TM to Post-TM prefetch type if they fit in l1
//
static void try_promote_post_tm_parameter_prefetch(
    balancer::OpModel& op_model,
    const std::vector<graphlib::Node*>& operands,
    std::size_t l1_usable_size,
    bool force_dram_parameters)
{
    if (force_dram_parameters) 
    {
        return;
    }

    const graphlib::BudaOpNode* op_node = op_model.buda_op_node;
    if (op_node->is_splice() or op_node->is_reduce() or op_node->is_embedding()) 
    {
        // These op types are not currently supported for Post-TM parameter prefetch optimization
        //
        return;
    }

    for (std::size_t input_idx = 0; input_idx < operands.size(); ++input_idx)
    {
        if (!is_input_node_parameter_or_constant(operands[input_idx]))
        { 
            continue;
        }
        
        if (op_model.parameter_buffers[input_idx].is_unrolled()) 
        {
            // The parameter is already unrolled, meaning the Post-TM shape has been calculated
            //
            continue;
        }           

        if (op_model.input_buffers[input_idx].kernel_broadcast_tiles) 
        {
            // If the parameter is kernel broadcast, it is already in Post-TM shape
            //
            continue;
        }

        BufferModel post_tm_param_buffer = calculate_parameter_buffer_model_for_grid(
            op_model,
            input_idx,
            operands[input_idx]->output_df(),
            true /*is_post_tm_prefetch*/);

        bool constexpr kIncludeT = true;
        std::size_t param_memory_usage = 
            op_model.parameter_buffers[input_idx].size_bytes(kIncludeT) + op_model.input_buffers[input_idx].size_bytes();
        std::size_t adjusted_memory_usage = 
            op_model.get_l1_memory_usage() - param_memory_usage + post_tm_param_buffer.size_bytes(kIncludeT);
        if (adjusted_memory_usage <= l1_usable_size) 
        {
            op_model.parameter_buffers[input_idx] = post_tm_param_buffer;
            op_model.input_buffers[input_idx].l1_size_tiles = 0;
        }
    }
}

static std::vector<BufferModel> calculate_intermediate_buffer_models_for_grid(
    graphlib::BudaOpNode const* op,
    BufferModel const& output_buffer,
    FusedOp const* fused_op,
    std::unordered_map<std::string, balancer::UBlockShape> const& fused_op_ublock_shape)
{
    std::vector<BufferModel> intermediate_buffers;
    bool intermediate_alias_output = op->intermediate_df() == output_buffer.data_format;
    bool is_reduce_z = graphlib::is_reduce_z(op);
    bool needs_intermediate_buffer_allocation =
        ((op->is_gradient_op() or op->is_matmul() or is_reduce_z) and not intermediate_alias_output);
    if (fused_op)
    {
        std::vector<int> mapped_intermediate_buffers;
        for (FusedSchedule const& schedule : fused_op->get_schedules())
        {
            for (FusedSubOp const& op : schedule.ops)
            {
                // TODO: do we need to handle dest here?
                if ((op.output_type == FusedSubOp::OutputType::INTERMED) &&
                    std::find(
                        mapped_intermediate_buffers.begin(), mapped_intermediate_buffers.end(), op.output_buffer) ==
                        mapped_intermediate_buffers.end())
                {
                    TT_ASSERT(op.op_shape.outputs.size() == 1);
                    mapped_intermediate_buffers.push_back(op.output_buffer);
                    BlockShape block_shape(1, 1, 1, fused_op_ublock_shape.at(op.name));  // ublock buffered
                    intermediate_buffers.emplace_back(block_shape, 1, op.output_df);
                }
            }
        }
    }
    else if (needs_intermediate_buffer_allocation)
    {
        intermediate_buffers.emplace_back(output_buffer.block_shape, 1, op->intermediate_df());
    }
    return intermediate_buffers;
}

static std::pair<std::uint32_t, bool> calculate_input_multiplier(
    const std::map<std::uint32_t, std::uint32_t>& input_multiplier_overrides,
    std::uint32_t operand,
    const TensorShape& shape,
    const UBlockShape& ublock)
{
    constexpr std::uint32_t DEFAULT_INPUT_MULTIPLIER = 2;
    if (auto input_multiplier_override_it = input_multiplier_overrides.find(operand);
        input_multiplier_override_it != input_multiplier_overrides.end())
    {
        log_debug(
            LogBalancer,
            "Using input multiplier override for operand {}: {}",
            operand,
            input_multiplier_override_it->second);
        return {
            std::max(input_multiplier_override_it->second, DEFAULT_INPUT_MULTIPLIER) * shape.ct * shape.rt /
                ublock.volume(),
            true};
    }
    return {DEFAULT_INPUT_MULTIPLIER, false};
}

static TensorShape calculate_effective_input_buffer_shape(
    Graph const* graph,
    graphlib::Edge edge,
    TStreamFactor producer_t_stream_factor,
    TStreamFactor consumer_t_stream_factor)
{
    TT_ASSERT(
        producer_t_stream_factor.none() or consumer_t_stream_factor.none(),
        "This function only handles one or the other not both");

    auto shape = graph->node_by_id(edge.producer_node_id)->shape();
    auto edge_attrs = graph->get_edge_attributes(edge);

    // Special eval that clamps if we over slice
    auto tm_shape = [](graphlib::OpType tm, graphlib::Shape const& shape) -> graphlib::Shape
    {
        if (tm.op == "vslice" and
            (std::get<int>(tm.attr[0]) > (int)shape.rt() or ((int)shape.rt() % std::get<int>(tm.attr[0]) != 0)))
        {
            std::get<int>(tm.attr[0]) = (int)shape.rt();
        }
        if (tm.op == "hslice" and
            (std::get<int>(tm.attr[0]) > (int)shape.ct() or ((int)shape.ct() % std::get<int>(tm.attr[0]) != 0)))
        {
            std::get<int>(tm.attr[0]) = (int)shape.ct();
        }
        return ::get_tm_shape(tm, shape, true);
    };

    shape = tm_shape(graphlib::OpType("vslice", {producer_t_stream_factor.r}, {}), shape);
    shape = tm_shape(graphlib::OpType("hslice", {producer_t_stream_factor.c}, {}), shape);

    int internal_slice_stack_factor = 1;
    for (graphlib::OpType const& tm : edge_attrs->get_tms())
    {
        bool eval_tm = true;

        if (tm.op == "hslice" or tm.op == "vslice")
        {
            int slice_factor = std::get<int>(tm.attr[0]);
            internal_slice_stack_factor *= slice_factor;
        }
        else if (tm.op == "hstack" or tm.op == "vstack")
        {
            int stack_factor = std::get<int>(tm.attr[0]);
            eval_tm = ((internal_slice_stack_factor % stack_factor) == 0);
            if (eval_tm)
                internal_slice_stack_factor /= stack_factor;
        }

        if (eval_tm)
            shape = tm_shape(tm, shape);
    }

    shape = tm_shape(graphlib::OpType("vslice", {consumer_t_stream_factor.r}, {}), shape);
    shape = tm_shape(graphlib::OpType("hslice", {consumer_t_stream_factor.c}, {}), shape);

    return shape;
}

static std::unordered_map<graphlib::NodeId, TensorShape> calculate_effective_input_buffer_shapes_for_users(
    graphlib::Graph const* graph, graphlib::Node const* node, TStreamFactor t_stream_factor)
{
    std::unordered_map<graphlib::NodeId, TensorShape> effective_input_buffer_shape_for_user;
    for (auto edge : graph->user_data_edges(node))
    {
        effective_input_buffer_shape_for_user[edge.consumer_node_id] =
            calculate_effective_input_buffer_shape(graph, edge, t_stream_factor, TStreamFactor());
    }
    return effective_input_buffer_shape_for_user;
}

static int calculate_max_u_kt_sparse(
    graphlib::Graph const* graph, graphlib::BudaOpNode const* op_node, int u_kt_override)
{
    TT_ASSERT(op_node->is_sparse_matmul());
    if (u_kt_override)
        return u_kt_override;

    std::vector<graphlib::Edge> operands = graph->operand_data_edges(op_node);
    TT_ASSERT(operands.size() == 3);
    TensorShape rhs_shape =
        calculate_effective_input_buffer_shape(graph, operands[1], TStreamFactor(), TStreamFactor());
    return rhs_shape.rt;
}

static int calculate_max_u_kt(
    graphlib::Graph const* graph, graphlib::BudaOpNode const* op_node, TStreamFactor t_stream_factor, int u_kt_override)
{
    TT_ASSERT(op_node->is_matmul());
    if (u_kt_override)
        return u_kt_override;
    std::vector<graphlib::Edge> operands = graph->operand_data_edges(op_node);
    TT_ASSERT(operands.size() >= 2);
    TensorShape lhs_shape = calculate_effective_input_buffer_shape(
        graph, operands[0], TStreamFactor(), t_stream_factor.dir.r() ? t_stream_factor : TStreamFactor());
    TensorShape rhs_shape = calculate_effective_input_buffer_shape(
        graph, operands[1], TStreamFactor(), t_stream_factor.dir.c() ? t_stream_factor : TStreamFactor());
    return int(std::min(lhs_shape.ct, rhs_shape.rt));
}

static std::vector<BufferModel> calculate_matmul_input_buffer_models_for_l1_budget(
    graphlib::Graph const* graph,
    const graphlib::BudaOpNode* op_node,
    GridShape grid_shape,
    OpShape op_shape,
    std::vector<graphlib::Node*> const& operands,
    BlockShape const& output_block_shape,
    TStreamFactor const& t_stream_factor,
    std::size_t input_l1_buffer_space,
    int,
    std::unordered_map<std::string, balancer::UBlockShape>,
    LegalSparseUKts const&,
    int u_kt_override,
    const std::map<std::uint32_t, std::uint32_t>& input_multiplier_overrides)
{
    TT_ASSERT(op_shape.inputs.size() >= 2 && op_shape.inputs.size() <= 4);
    TT_ASSERT(operands.size() >= 2 && operands.size() <= 4);
    std::vector<BufferModel> input_buffers;
    DataFormat input0_df = operands[0]->output_df();
    DataFormat input1_df = operands[1]->output_df();
    TensorShape input0 = op_shape.inputs[0];
    TensorShape input1 = op_shape.inputs[1];
    int max_u_kt = calculate_max_u_kt(graph, op_node, t_stream_factor, u_kt_override);

    auto buda_attrs = op_node->buda_attrs();
    bool has_requant = buda_attrs.find("requant") != buda_attrs.end() and std::get<bool>(buda_attrs.at("requant"));
    std::optional<BufferModel> fused_bias;
    std::optional<BufferModel> fused_requant;

    if (has_requant)
    {
        TT_ASSERT(operands.size() >= 3);
        TensorShape input2 = op_shape.inputs[2];
        DataFormat input2_df = operands[2]->output_df();
        UBlockShape input2_ublock = output_block_shape.ublock;
        BlockShape input2_block_shape(input2, GridShape(1, grid_shape.c), 1, 1, input2_ublock);
        auto [input_buffer_multiplier, override_enabled] =
            calculate_input_multiplier(input_multiplier_overrides, 2, input2, input2_ublock);

        if (operands.size() == 3)
        {
            // Just dequant
            fused_requant = BufferModel(input2_block_shape, input_buffer_multiplier, input2_df, override_enabled);

            // Carve out fused operand before calculating u_kt
            if (fused_requant->size_bytes() <= input_l1_buffer_space)
                input_l1_buffer_space -= fused_requant->size_bytes();
        }
        else if (operands.size() == 4)
        {
            // Bias + dequant
            fused_bias = BufferModel(input2_block_shape, input_buffer_multiplier, input2_df, override_enabled);

            // Carve out fused operand before calculating u_kt
            if (fused_bias->size_bytes() <= input_l1_buffer_space)
                input_l1_buffer_space -= fused_bias->size_bytes();

            TensorShape input3 = op_shape.inputs[3];
            DataFormat input3_df = operands[3]->output_df();
            UBlockShape input3_ublock = output_block_shape.ublock;
            BlockShape input3_block_shape(input3, GridShape(1, grid_shape.c), 1, 1, input3_ublock);
            auto [input_buffer_multiplier, override_enabled] =
                calculate_input_multiplier(input_multiplier_overrides, 3, input3, input3_ublock);
            fused_requant = BufferModel(input3_block_shape, input_buffer_multiplier, input3_df, override_enabled);

            // Carve out fused operand before calculating u_kt
            if (fused_requant->size_bytes() <= input_l1_buffer_space)
                input_l1_buffer_space -= fused_requant->size_bytes();
        }
    }
    else
    {
        if (operands.size() == 3)
        {
            // fused bias
            TensorShape input2 = op_shape.inputs[2];
            DataFormat input2_df = operands[2]->output_df();
            UBlockShape input2_ublock = output_block_shape.ublock;
            BlockShape input2_block_shape(input2, GridShape(1, grid_shape.c), 1, 1, input2_ublock);
            auto [input_buffer_multiplier, override_enabled] =
                calculate_input_multiplier(input_multiplier_overrides, 2, input2, input2_ublock);
            fused_bias = BufferModel(input2_block_shape, input_buffer_multiplier, input2_df, override_enabled);

            // Carve out fused operand before calculating u_kt
            if (fused_bias->size_bytes() <= input_l1_buffer_space)
                input_l1_buffer_space -= fused_bias->size_bytes();
        }
    }

    // 1 outer strip of tiles per input double buffered
    int input0_outer_dim_bytes =
        output_block_shape.mblock_m * output_block_shape.ublock.rt * tile_size_bytes(input0_df) * 2;
    int input1_outer_dim_bytes =
        output_block_shape.mblock_n * output_block_shape.ublock.ct * tile_size_bytes(input1_df) * 2;

    bool minimize_op0 = false;
    bool minimize_op1 = false;

    if (!env_as<bool>("PYBUDA_DISABLE_MIN_MATMUL_BUFFER"))
    {
        // Minimize one of the buffers - whichever strip is bigger if input1 is alowed through switch
        if ((input1_outer_dim_bytes > input0_outer_dim_bytes) && env_as<bool>("PYBUDA_MIN_MATMUL_BUFFER_ALLOW_IN1"))
        {
            input1_outer_dim_bytes /= output_block_shape.mblock_n;
            minimize_op1 = true;
        }
        else
        {
            input0_outer_dim_bytes /= output_block_shape.mblock_m;
            minimize_op0 = true;
        }
    }
    std::size_t k_factor =
        std::max(std::size_t(1), input_l1_buffer_space / (input0_outer_dim_bytes + input1_outer_dim_bytes));

    TT_ASSERT(k_factor <= INT_MAX);
    int u_kt = FactorizedInt(max_u_kt).get_nearest_factor_le(static_cast<int>(k_factor));

    UBlockShape input0_ublock(output_block_shape.ublock.rt, u_kt);
    UBlockShape input1_ublock(u_kt, output_block_shape.ublock.ct);

    BlockShape input0_block_shape(
        input0, GridShape(grid_shape.r, 1), minimize_op0 ? 1 : output_block_shape.mblock_m, 1, input0_ublock);
    BlockShape input1_block_shape(
        input1, GridShape(1, grid_shape.c), 1, minimize_op1 ? 1 : output_block_shape.mblock_n, input1_ublock);

    TT_ASSERT(u_kt != 0);
    auto [input0_buffer_multiplier, override_enabled0] =
        calculate_input_multiplier(input_multiplier_overrides, 0, input0, input0_ublock);
    input_buffers.emplace_back(input0_block_shape, input0_buffer_multiplier, input0_df, override_enabled0);
    input_buffers[0].minimize_input_buffer = minimize_op0;

    auto [input1_buffer_multiplier, override_enabled1] =
        calculate_input_multiplier(input_multiplier_overrides, 1, input1, input1_ublock);
    input_buffers.emplace_back(input1_block_shape, input1_buffer_multiplier, input1_df, override_enabled1);
    input_buffers[1].minimize_input_buffer = minimize_op1;

    if (fused_bias)
    {
        fused_bias->buffer_factor = std::max((std::uint32_t)fused_bias->buffer_factor / u_kt, (std::uint32_t)2);
        input_buffers.push_back(*fused_bias);
    }
    if (fused_requant)
    {
        fused_requant->buffer_factor = std::max((std::uint32_t)fused_requant->buffer_factor / u_kt, (std::uint32_t)2);
        input_buffers.push_back(*fused_requant);
    }

    return input_buffers;
}

static std::vector<BufferModel> calculate_sparse_matmul_input_buffer_models_for_l1_budget(
    graphlib::Graph const* graph,
    const graphlib::BudaOpNode* op_node,
    GridShape grid_shape,
    OpShape op_shape,
    std::vector<graphlib::Node*> const& operands,
    BlockShape const& output_block_shape,
    TStreamFactor const& t_stream_factor,
    std::size_t input_l1_buffer_space,
    int fracture_factor,
    std::unordered_map<std::string, balancer::UBlockShape>,
    LegalSparseUKts const& legal_sparse_u_kts,
    int u_kt_override,
    const std::map<std::uint32_t, std::uint32_t>& input_multiplier_overrides)
{
    DataFormat input0_df = operands[0]->output_df();
    DataFormat input1_df = operands[1]->output_df();
    DataFormat input2_df = operands[2]->output_df();
    TensorShape input0 = op_shape.inputs[0];
    TensorShape input1 = op_shape.inputs[1];
    TensorShape input2 = op_shape.inputs[2];

    TT_ASSERT(op_shape.inputs.size() == 3);
    TT_ASSERT(operands.size() == 3);

    BlockShape sparse_block_shape(input0, GridShape(grid_shape.r, 1), 1, 1, UBlockShape(1, 1));
    BlockShape index_block_shape(input2, GridShape(grid_shape.r, 1), 1, 1, UBlockShape(1, 1));

    BufferModel buffer_model0 = BufferModel(sparse_block_shape, 1, input0_df);
    BufferModel buffer_model2 = BufferModel(index_block_shape, 1, input2_df);

    // Sparse MM will access the parameter buffer allocation for in0 so no need to allocate an input buffer for it.  We
    // will keep the input buffer info for other bits of code to reference shapes
    buffer_model0.l1_size_tiles = 0;

    int leftover_l1_space = input_l1_buffer_space - buffer_model0.size_bytes() - buffer_model2.size_bytes();

    // Calculate max u_kt given leftover l1 space
    auto [input1_buffer_multiplier, override_enabled] =
        calculate_input_multiplier(input_multiplier_overrides, 1, input1, UBlockShape(1, output_block_shape.ublock.ct));
    BlockShape input_block_shape_ukt1(
        input1,
        GridShape(1, grid_shape.c / fracture_factor),
        1,
        output_block_shape.mblock_n,
        UBlockShape(1, output_block_shape.ublock.ct));
    int min_buffer_mem =
        BufferModel(input_block_shape_ukt1, input1_buffer_multiplier, input1_df, override_enabled).size_bytes();

    // Find max u_kt given input dims
    int max_u_kt_dimensionwise = calculate_max_u_kt_sparse(graph, op_node, u_kt_override);
    TT_ASSERT(max_u_kt_dimensionwise > 0);

    // Additional limits on u_kt (memory- and encoding- imposed constraints)
    const sparse::SparseBUDA& sparse_buda =
        graph->data_operands(op_node)[0]->as<graphlib::ConstantInputNode>()->get_sparse_buda();
    int max_u_kt_memorywise = (int)(leftover_l1_space / min_buffer_mem);  // sparse op
    int max_u_kt_encodingwise = sparse_buda.get_max_u_kt(grid_shape.r, t_stream_factor.r, output_block_shape.ublock.rt);
    std::size_t k_factor = std::max(1, max_u_kt_memorywise);
    TT_ASSERT(k_factor <= INT_MAX);
    auto legal_factors = FactorizedInt(max_u_kt_dimensionwise);
    TT_ASSERT(not legal_factors.empty());

    TT_ASSERT(not t_stream_factor.is_streaming_r() or not legal_sparse_u_kts.empty());
    if (t_stream_factor.is_streaming_r())
    {
        std::vector<int> const& legal_u_kts = legal_sparse_u_kts.at(t_stream_factor.r);
        TT_ASSERT(not legal_u_kts.empty());
        legal_factors = legal_factors & FactorizedInt(legal_u_kts.begin(), legal_u_kts.end());
        if (legal_factors.empty())
            return {};
    }

    if (static_cast<int>(k_factor) < legal_factors.get_min_factor())
        return {};
    int u_kt_memorywise = legal_factors.get_nearest_factor_le(static_cast<int>(k_factor));  // Limit u_kt by memory

    if (static_cast<int>(max_u_kt_encodingwise) < legal_factors.get_min_factor())
        return {};
    int u_kt_encodingwise =
        legal_factors.get_nearest_factor_le(static_cast<int>(max_u_kt_encodingwise));  // Limit u_kt by encoding limits

    // u_kt is now the min of memory/encoding limits
    int u_kt = std::max(1, std::min(u_kt_memorywise, u_kt_encodingwise));

    // Recreate buffer model with new u_kt
    BlockShape input_block_shape = BlockShape(
        input1,
        GridShape(1, grid_shape.c / fracture_factor),
        1,
        output_block_shape.mblock_n,
        UBlockShape(u_kt, output_block_shape.ublock.ct));
    BufferModel buffer_model1 = BufferModel(input_block_shape, input1_buffer_multiplier, input1_df);
    
    return {buffer_model0, buffer_model1, buffer_model2};
}

static std::vector<BufferModel> calculate_depthwise_input_buffer_models_for_l1_budget(
    graphlib::Graph const*,
    const graphlib::BudaOpNode*,
    GridShape grid_shape,
    OpShape op_shape,
    std::vector<graphlib::Node*> const& operands,
    BlockShape const& output_block_shape,
    TStreamFactor const&,
    std::size_t input_l1_buffer_space,
    int,
    std::unordered_map<std::string, balancer::UBlockShape>,
    LegalSparseUKts const&,
    int,
    const std::map<std::uint32_t, std::uint32_t>& input_multiplier_overrides)
{
    TT_ASSERT(op_shape.inputs.size() >= 2 && op_shape.inputs.size() <= 3);
    TT_ASSERT(operands.size() >= 2 && operands.size() <= 3);
    std::vector<BufferModel> input_buffers;
    DataFormat input0_df = operands[0]->output_df();
    DataFormat input1_df = operands[1]->output_df();
    TensorShape input0 = op_shape.inputs[0];
    TensorShape input1 = op_shape.inputs[1];
    int u_kt = 1;  // HLK-imposed limit

    std::optional<BufferModel> fused_bias;
    if (operands.size() == 3)
    {
        // fused bias
        TensorShape input2 = op_shape.inputs[2];
        DataFormat input2_df = operands[2]->output_df();
        UBlockShape input2_ublock = output_block_shape.ublock;
        BlockShape input2_block_shape(input2, GridShape(1, grid_shape.c), 1, 1, input2_ublock);
        auto [input_buffer_multiplier, override_enabled] =
            calculate_input_multiplier(input_multiplier_overrides, 2, input2, input2_ublock);
        fused_bias = BufferModel(input2_block_shape, input_buffer_multiplier, input2_df, override_enabled);

        // Carve out fused operand before calculating u_kt
        if (fused_bias->size_bytes() <= input_l1_buffer_space)
            input_l1_buffer_space -= fused_bias->size_bytes();
    }

    // HLK doesn't support minimizing op1, but op0 is okay
    bool minimize_op0 = not env_as<bool>("PYBUDA_DISABLE_MIN_DEPTHWISE_BUFFER", false);

    UBlockShape input0_ublock(output_block_shape.ublock.rt, output_block_shape.ublock.ct);
    UBlockShape input1_ublock(u_kt, output_block_shape.ublock.ct);

    BlockShape input0_block_shape(
        input0, GridShape(grid_shape.r, 1), minimize_op0 ? 1 : output_block_shape.mblock_m, 1, input0_ublock);
    BlockShape input1_block_shape(input1, GridShape(1, grid_shape.c), 1, output_block_shape.mblock_n, input1_ublock);

    auto [input0_buffer_multiplier, override_enabled0] =
        calculate_input_multiplier(input_multiplier_overrides, 0, input0, input0_ublock);
    input_buffers.emplace_back(input0_block_shape, input0_buffer_multiplier, input0_df, override_enabled0);
    input_buffers[0].minimize_input_buffer = minimize_op0;

    auto [input1_buffer_multiplier, override_enabled1] =
        calculate_input_multiplier(input_multiplier_overrides, 1, input1, input1_ublock);
    input_buffers.emplace_back(input1_block_shape, input1_buffer_multiplier, input1_df, override_enabled1);
    input_buffers[1].minimize_input_buffer = false;  // HLK-imposed limit

    if (fused_bias)
    {
        fused_bias->buffer_factor = std::max((std::uint32_t)fused_bias->buffer_factor / u_kt, (std::uint32_t)2);
        input_buffers.push_back(*fused_bias);
    }

    return input_buffers;
}

static std::vector<BufferModel> calculate_eltwise_input_buffer_models_for_l1_budget(
    graphlib::Graph const*,
    const graphlib::BudaOpNode*,
    GridShape grid_shape,
    OpShape op_shape,
    std::vector<graphlib::Node*> const& operands,
    BlockShape const& output_block_shape,
    TStreamFactor const&,
    std::size_t,
    int,
    std::unordered_map<std::string, balancer::UBlockShape>,
    LegalSparseUKts const&,
    int,
    const std::map<std::uint32_t, std::uint32_t>& input_multiplier_overrides)
{
    TT_ASSERT(op_shape.inputs.size() == operands.size());

    std::vector<BufferModel> input_buffers;
    for (int input_idx = 0; input_idx < (int)op_shape.inputs.size(); ++input_idx)
    {
        TensorShape const& input = op_shape.inputs[input_idx];
        BlockShape input_block_shape(input, grid_shape, 1, 1, output_block_shape.ublock);
        auto [input_buffer_multiplier, override_enabled] =
            calculate_input_multiplier(input_multiplier_overrides, input_idx, input, output_block_shape.ublock);
        input_buffers.emplace_back(
            input_block_shape, input_buffer_multiplier, operands[input_idx]->output_df(), override_enabled);
    }
    return input_buffers;
}

static std::vector<BufferModel> calculate_reduce_input_buffer_models_for_l1_budget(
    graphlib::Graph const* graph,
    const graphlib::BudaOpNode* op_node,
    GridShape grid_shape,
    OpShape op_shape,
    std::vector<graphlib::Node*> const& operands,
    BlockShape const& output_block_shape,
    TStreamFactor const& t_stream_factor,
    std::size_t input_l1_buffer_space,
    int,
    std::unordered_map<std::string, balancer::UBlockShape>,
    LegalSparseUKts const&,
    int,
    const std::map<std::uint32_t, std::uint32_t>& input_multiplier_overrides)
{
    const graphlib::OpType& op_type = op_node->op_type();
    TT_ASSERT(op_type.op == "reduce");
    TT_ASSERT(op_shape.inputs.size() == operands.size());
    TT_ASSERT(op_shape.inputs.size() == 1);

    if (std::get<std::string>(op_type.buda_attrs.at("dim")) == "z")
    {
        return calculate_eltwise_input_buffer_models_for_l1_budget(
            graph,
            op_node,
            grid_shape,
            op_shape,
            operands,
            output_block_shape,
            t_stream_factor,
            input_l1_buffer_space,
            1,
            {},
            {},
            0,
            input_multiplier_overrides);
    }

    auto calc_u_kt = [](std::size_t input_l1_buffer_space,
                        int reduce_dim_tiles,
                        int non_reduce_dim_ublock_tiles,
                        DataFormat df) -> int
    {
        std::size_t non_reduce_dim_bytes = non_reduce_dim_ublock_tiles * tile_size_bytes(df);
        std::size_t k_factor = std::max(std::size_t(1), input_l1_buffer_space / (non_reduce_dim_bytes * 2));
        int u_kt = FactorizedInt(reduce_dim_tiles).get_nearest_factor_le(static_cast<int>(k_factor));
        return u_kt;
    };

    graphlib::Node* operand = operands[0];
    TensorShape input = op_shape.inputs[0];
    UBlockShape input_ublock;
    if (std::get<std::string>(op_type.buda_attrs.at("dim")) == "r")
    {
        int u_kt = calc_u_kt(input_l1_buffer_space, input.rt, output_block_shape.ublock.ct, operand->output_df());
        input_ublock = UBlockShape(u_kt, output_block_shape.ublock.ct);
    }
    else
    {
        TT_ASSERT(std::get<std::string>(op_type.buda_attrs.at("dim")) == "c");
        int u_kt = calc_u_kt(input_l1_buffer_space, input.ct, output_block_shape.ublock.rt, operand->output_df());
        input_ublock = UBlockShape(output_block_shape.ublock.rt, u_kt);
    }

    BlockShape input_block_shape(input, grid_shape, 1, 1, input_ublock);
    auto [input_buffer_multiplier, override_enabled] =
        calculate_input_multiplier(input_multiplier_overrides, 0, input, input_ublock);
    return {BufferModel(input_block_shape, input_buffer_multiplier, operand->output_df(), override_enabled)};
}

static std::vector<BufferModel> calculate_embedding_input_buffer_models_for_l1_budget(
    graphlib::Graph const*,
    const graphlib::BudaOpNode* op_node,
    GridShape grid_shape,
    OpShape op_shape,
    std::vector<graphlib::Node*> const& operands,
    BlockShape const& output_block_shape,
    TStreamFactor const&,
    std::size_t,
    int,
    std::unordered_map<std::string, balancer::UBlockShape>,
    LegalSparseUKts const&,
    int,
    const std::map<std::uint32_t, std::uint32_t>&)
{
    const graphlib::OpType& op_type = op_node->op_type();
    TT_ASSERT(op_type.op == "embedding");
    TT_ASSERT(op_shape.inputs.size() == operands.size());
    TT_ASSERT(op_shape.inputs.size() == 2);

    BlockShape embedding_table_block_shape(
        op_shape.inputs[0], GridShape(1, grid_shape.c), 1, 1, UBlockShape(1, output_block_shape.ublock.ct));
    BufferModel embedding_table(embedding_table_block_shape, 2, operands[1]->output_df());

    TensorShape indices_shape = op_shape.inputs[1];
    TT_ASSERT((indices_shape.ct % grid_shape.r) == 0);
    indices_shape.rt = indices_shape.rt * grid_shape.r;
    indices_shape.ct = indices_shape.ct / grid_shape.r;
    BlockShape indices_block_shape(indices_shape, GridShape(grid_shape.r, 1), 1, 1, UBlockShape(1, 1));
    BufferModel indices(indices_block_shape, 2, operands[1]->output_df());

    return {embedding_table, indices};
}

static std::vector<BufferModel> calculate_fused_input_buffer_models_for_l1_budget(
    graphlib::Graph const*,
    const graphlib::BudaOpNode* op_node,
    GridShape grid_shape,
    OpShape op_shape,
    std::vector<graphlib::Node*> const& operands,
    BlockShape const&,
    TStreamFactor const&,
    std::size_t,
    int,
    std::unordered_map<std::string, balancer::UBlockShape> fused_op_ublock_shape,
    LegalSparseUKts const&,
    int,
    const std::map<std::uint32_t, std::uint32_t>& input_multiplier_overrides)
{
    TT_ASSERT(op_shape.inputs.size() == operands.size());

    std::vector<BufferModel> input_buffers;
    std::vector<bool> visited(op_shape.inputs.size(), false);
    for (FusedSchedule const& schedule : op_node->get_fused_op()->get_schedules())
    {
        for (FusedSubOp const& sub_op : schedule.ops)
        {
            // Id of input for this specific sub op.
            int input_id = -1;
            for (FusedSubOpInput const& sub_input : sub_op.inputs)
            {
                input_id++;

                if (sub_input.type != FusedSubOpInput::InputType::INPUT)
                    continue;

                TT_ASSERT(sub_input.index < op_shape.inputs.size());
                TT_ASSERT(visited[sub_input.index] == false);

                if (visited[sub_input.index])
                    continue;
                visited[sub_input.index] = true;

                TensorShape const& input = op_shape.inputs[sub_input.index];
                UBlockShape ublock_shape = fused_op_ublock_shape.at(sub_op.name);

                // Special case for fused matmul op.
                if (sub_op.op_type == "matmul")
                {
                    int u_kt = 1;  // The only legal u_kt for matmul sub-ops
                    if (input_id == 0)
                        ublock_shape.ct = u_kt;
                    else
                        ublock_shape.rt = u_kt;
                }

                BlockShape input_block_shape(input, grid_shape, 1, 1, ublock_shape);
                auto [input_buffer_multiplier, override_enabled] =
                    calculate_input_multiplier(input_multiplier_overrides, sub_input.index, input, ublock_shape);
                input_buffers.emplace_back(
                    input_block_shape,
                    input_buffer_multiplier,
                    operands[sub_input.index]->output_df(),
                    override_enabled);
            }
        }
    }
    TT_ASSERT(input_buffers.size() == op_shape.inputs.size());

    return input_buffers;
}

template <typename... Args>
static std::vector<BufferModel> calculate_input_buffer_models(
    graphlib::Graph const* graph, const graphlib::BudaOpNode* op_node, Args... args)
{
    if (op_node->is_fused_op())
    {
        return calculate_fused_input_buffer_models_for_l1_budget(graph, op_node, args...);
    }
    else if (op_node->is_matmul() and not op_node->is_sparse_matmul() and not op_node->is_depthwise_matmul())
    {
        return calculate_matmul_input_buffer_models_for_l1_budget(graph, op_node, args...);
    }
    else if (op_node->is_sparse_matmul())
    {
        return calculate_sparse_matmul_input_buffer_models_for_l1_budget(graph, op_node, args...);
    }
    else if (op_node->is_depthwise_matmul())
    {
        return calculate_depthwise_input_buffer_models_for_l1_budget(graph, op_node, args...);
    }
    else if (op_node->is_reduce())
    {
        return calculate_reduce_input_buffer_models_for_l1_budget(graph, op_node, args...);
    }
    else if (op_node->is_embedding())
    {
        return calculate_embedding_input_buffer_models_for_l1_budget(graph, op_node, args...);
    }

    return calculate_eltwise_input_buffer_models_for_l1_budget(graph, op_node, args...);
}

// Returns the length of the pattern for kernel broadcast, by creating the pipe for the consumer and producer, and
// analyzing the addresses of tiles being sent
//
int get_kernel_broadcast_len(
    std::unordered_map<Pipe, int>* const kb_cache,
    graphlib::Graph const* graph,
    balancer::OpModel const& op_model,
    graphlib::Edge const& edge,
    graphlib::EdgeAttributes* edge_attr,
    std::vector<OpType> const& tms)
{
    log_trace(LogKernelBroadcast, "  get_kernel_broadcast_len, operand index: {}", edge.consumer_input_port_id);

    graphlib::Node* producer = graph->node_by_id(edge.producer_node_id);

    // Don't kernel broadcast if producer is a buda op
    //
    if (producer->node_type() == graphlib::NodeType::kBudaOp)
    {
        return 0;
    }

    // Don't kernel broadcast from producers with shape.z > 1
    //
    if (producer->shape().z() > 1)
    {
        // This constraint is imposed by net2pipe, the test below errors out if this constraint is removed
        //
        //   pybuda/test/test_constraints.py::test_stream_stacking_transpose
        //     ERROR: TM ERROR (producer = input_1_mm0, consumer = mm0): with kernel broadcast that's not per-t,
        //     producer must have t = 1 and buf_size_mb = 1 or 2
        //
        return 0;
    }

    // Don't kernel broadcast from buffering and e2e queues
    //
    if (producer->node_type() == graphlib::NodeType::kQueue and
        (producer->as<graphlib::QueueNode>()->queue_type() == graphlib::QueueNodeType::Buffering or
         producer->as<graphlib::QueueNode>()->queue_type() == graphlib::QueueNodeType::EpochToEpoch))
    {
        // This helps us avoid user inserted queues and virtual queues, all of which can end up having t > 1
        //
        return 0;
    }

    // If there's no broadcasts, there's nothing to kernel broadcast
    //
    if (not std::any_of(tms.begin(), tms.end(), [](auto const& op_type) { return op_type.op == "broadcast"; }))
    {
        return 0;
    }

    // If a producer is a single tile or if each consumer core "sees" a single tile from producer, we can return early
    // out and return a pattern length of 1
    // There are some other edge cases where the pattern will be of length 1, but those are harder to detect for early
    // out scenarios. We delegate the responsibility of finding those to pattern detection
    //
    if (producer->shape().is_single_tile())
    {
        return 1;
    }

    // Producer cannot be a buda op
    // OpModel must belong to a buda op
    //
    TT_ASSERT(producer->node_type() != graphlib::NodeType::kBudaOp);
    TT_ASSERT(op_model.buda_op_node);

    // Producer tile layout is not relevant, so we simplify by using default/trivial values
    //
    TileLayout producer_tile_layout = TileLayout(
        GridShape(1, 1),                                                        // this can always be 1x1
        BlockShape(producer->shape(), 1, 1, 1, UBlockShape(1, 1)).canonical(),  // doesn't matter what the dims are
        graphlib::UBlockOrder::R,  // this doesn't matter, so always set to R
        Padding());                // queues don't have padding on them

    // As matmul doesn't have eltwise-like pipes, but rather multicasts in some dimensions, we need to adjust the grid
    // shape of the consumer to not divide a dimension like an eltwise-style pipe would
    //
    GridShape consumer_grid_shape = op_model.grid_shape;
    if (op_model.buda_op_node->is_matmul())
    {
        consumer_grid_shape = edge.consumer_input_port_id == 0 ? GridShape(consumer_grid_shape.r, 1)
                                                               : GridShape(1, consumer_grid_shape.c);
    }

    // Consumer tile layout - describes the input buffer
    //
    TileLayout consumer_tile_layout = TileLayout(
        consumer_grid_shape,
        op_model.input_buffers[edge.consumer_input_port_id].block_shape.canonical(),
        edge_attr->get_ublock_order(),
        Padding());  // padding isn't relevant for input buffer of a consumer op which has padding set

    // Create the pipe
    //
    Pipe pipe(
        producer_tile_layout,
        1,  // producer_out_buf_mb is not relevant for kernel broadcast
        tms,
        consumer_tile_layout);

    // Check if pipe exists in cache
    //
    if (kb_cache)
    {
        auto match = kb_cache->find(pipe);
        if (match != kb_cache->end())
        {
            log_trace(LogKernelBroadcast, "    Found in cache - len: {}", match->second);
            return match->second;
        }
    }

    int pattern_len = detect_repetitive_pattern(kb_cache, pipe);

    return pattern_len;
}

static void try_promote_kernel_broadcast_inputs(
    std::unordered_map<Pipe, int>* const kb_cache,
    graphlib::Graph const* graph,
    graphlib::OpNode const* op_node,
    std::size_t l1_usable_size,
    OpModel& op_model,
    bool force_dram_parameters)
{
    // Check if kernel broadcasting is disabled
    //
    static const bool disable_kernel_broadcast = env_as<bool>("PYBUDA_DISABLE_KERNEL_BROADCAST");
    if (disable_kernel_broadcast)
    {
        return;
    }

    // Embedding, tilize and reduce ops don't support kernel broadcasting
    //
    if (op_node->is_embedding() || op_node->is_tilize() || op_node->is_reduce())
    {
        return;
    }

    log_trace(
        LogKernelBroadcast,
        "try_promote_kernel_broadcast_inputs, op: {}, op model id: {:8}",
        op_node->name(),
        op_model.id.id);

    // Check each edge for kernel broadcasting
    //
    for (graphlib::Edge const& edge : graph->operand_data_edges(op_node))
    {
        // Sparse matmul's in0 is always fully prologued
        //
        if (op_node->is_sparse_matmul() and edge.consumer_input_port_id == 0)
        {
            continue;
        }

        graphlib::Node const* producer = graph->node_by_id(edge.producer_node_id);

        // Do not kernel broadcast params/consts when the force_dram_parameter is set to true.
        // In this case, we don't want to prologue, but stream from DRAM.
        if(force_dram_parameters and is_input_node_parameter_or_constant(producer))
        {
            continue;
        }

        auto attr = graph->get_edge_attributes(edge);
        auto tms = attr->get_tms();
        insert_t_stream_tms(op_node, tms, op_model.t_stream_factor, TStreamFactor{}, edge.consumer_input_port_id);

        static const bool use_legacy_kernel_broadcast_path = env_as<bool>("PYBUDA_LEGACY_KERNEL_BROADCAST");
        if (use_legacy_kernel_broadcast_path)
        {
            log_trace(LogKernelBroadcast, "  Using legacy path...");
            if (not tms_support_kernel_broadcast(
                    producer->shape(), tms, attr->get_ublock_order(), op_model.block_shape().ublock.ct))
                continue;

            bool single_tile = producer->shape().is_single_tile();
            TensorShape shape = post_tms_shape(producer->shape(), tms, graphlib::ignore_broadcast_tm_evaluator);
            int input_idx = edge.consumer_input_port_id;
            bool is_prologue = bool(op_model.parameter_buffers[input_idx]);
            int per_core_rt = round_up_div(
                shape.rt, (op_node->is_matmul() and edge.consumer_input_port_id == 1) ? 1 : op_model.grid_shape.r);
            int per_core_ct = round_up_div(
                shape.ct, (op_node->is_matmul() and edge.consumer_input_port_id == 0) ? 1 : op_model.grid_shape.c);
            UBlockShape ublock = single_tile ? UBlockShape(1, 1) : op_model.input_buffers[input_idx].block_shape.ublock;
            int t = shape.z;
            int mblock_m = round_up_div(per_core_rt, ublock.rt);
            int mblock_n = round_up_div(per_core_ct, ublock.ct);
            BlockShape block_shape(t, mblock_m, mblock_n, ublock);
            BufferModel l1_buffer_model(block_shape, is_prologue ? 1 : 2, producer->output_df());
            // Kernel always wants programming like it is single buffered
            BufferModel kernel_buffer_model(block_shape, 1, producer->output_df());

            static const bool include_t =
                use_legacy_kernel_broadcast_path;  // we don't actually want to include t in the size calculation, but
                                                   // we did use it in legacy path, keeping it for bwd compatibility
            std::size_t current_input_size =
                op_model.input_buffers[input_idx].size_bytes() + op_model.parameter_buffers[input_idx].size_bytes();
            TT_ASSERT(current_input_size <= op_model.get_l1_memory_usage());
            std::size_t adjusted_memory_usage =
                op_model.get_l1_memory_usage() - current_input_size + l1_buffer_model.size_bytes(include_t);
            if (adjusted_memory_usage <= l1_usable_size)
            {
                // Clobber the input/param buffer's allocation size with adjusted kernel broadcast size / zero to
                // reflect their new L1 footprint. Leave the blocking information intact so canonical form checks work
                // as is
                op_model.input_buffers[input_idx].kernel_broadcast_tiles = kernel_buffer_model.size_tiles(include_t);
                op_model.input_buffers[input_idx].l1_size_tiles = l1_buffer_model.size_tiles(include_t);
                op_model.parameter_buffers[input_idx].l1_size_tiles = 0;
            }

            continue;
        }

        // Default kernel broadcast path (non-legacy)

        // Get kernel_broadcast len (0 if no pattern)
        //
        int kb_len = get_kernel_broadcast_len(kb_cache, graph, op_model, edge, attr.get(), tms);
        if (not kb_len)
        {
            continue;
        }

        const int input_idx = edge.consumer_input_port_id;

        // This is the number of tiles that a single consumer core will need to "see" from producer op in order to
        // produce a single mblock of output
        //
        const int producer_tiles_single_mblock =
            op_model.input_buffers[input_idx].block_shape.canonical().volume_no_t();

        // kb_len should be *no bigger* than producer_tiles_single_mblock
        //
        TT_ASSERT(
            kb_len <= producer_tiles_single_mblock,
            "kb_len: {}, producer_tiles_single_mblock: {}",
            kb_len,
            producer_tiles_single_mblock);

        // If kernel broadcast fits into L1, set it on the input buffer
        // Account buffering for kernel broadcast inputs
        // Use single buffering for prologue inputs
        int l1_kb_buffer_len = kb_len;

        // Adjust for double buffering when the input is not a prologue, such as an activation
        if (!is_input_node_parameter_or_constant(producer))
        {
            l1_kb_buffer_len *= 2;
        }

        int kb_mem_footprint = l1_kb_buffer_len * tile_size_bytes(producer->output_df());
        std::size_t current_input_size =
            op_model.input_buffers[input_idx].size_bytes() + op_model.parameter_buffers[input_idx].size_bytes(true /*include_t*/);
        TT_ASSERT(current_input_size <= op_model.get_l1_memory_usage());
        std::size_t adjusted_memory_usage = op_model.get_l1_memory_usage() - current_input_size + kb_mem_footprint;
        if (adjusted_memory_usage <= l1_usable_size)
        {
            // Change the l1_size_tiles property of input buffer to reflect memory footprint of kernel broadcast.
            // Set l1_size_tiles to reflect double/single buffering.
            // Additionally, change prologue buffer to 0 since it is no longer needed.
            // Leaving the blocking information intact so canonical form checks work as is.
            //
            op_model.input_buffers[input_idx].kernel_broadcast_tiles = kb_len;
            op_model.input_buffers[input_idx].l1_size_tiles = l1_kb_buffer_len;
            op_model.parameter_buffers[input_idx].l1_size_tiles = 0;

            log_trace(
                LogKernelBroadcast,
                "  Kernel broadcast detected on op {}, op model id: {}, operand id {}, kernel broadcast length: {}",
                op_node->name(),
                op_model.id.id,
                input_idx,
                kb_len);

            TT_ASSERT(op_model.get_l1_memory_usage() == adjusted_memory_usage);
        }
    }
}

static std::optional<std::size_t> find_max_parameter_buffer_l1_user(const OpModel& op_model)
{
    if (op_model.buda_op_node->is_sparse_matmul())
    {
        // Only encodings can be streamed from dram, sparse tiles cannot
        constexpr std::size_t kEncodingsParameterIndex = 2;
        if (op_model.parameter_buffers.size() > kEncodingsParameterIndex and op_model.parameter_buffers[kEncodingsParameterIndex])
        {
            return std::optional<std::size_t>(kEncodingsParameterIndex);
        }

        return std::nullopt;
    }

    std::size_t max = 0;
    std::size_t max_idx = 0;
    for (std::size_t i = 0; i < op_model.parameter_buffers.size(); ++i)
    {
        bool constexpr kIncludeT = true;
        const BufferModel& parameter_buffer = op_model.parameter_buffers[i];
        if (parameter_buffer and max < parameter_buffer.size_bytes(kIncludeT))
        {
            max = parameter_buffer.size_bytes(kIncludeT);
            max_idx = i;
        }
    }

    return max ? std::optional<std::size_t>(max_idx) : std::nullopt;
}

static std::vector<BufferModel> upsize_output_buffer(
    graphlib::Graph const* graph,
    std::vector<BufferModel> output_buffers,
    std::size_t l1_remaining_size,
    bool is_gradient_op)
{
    int factor = int(l1_remaining_size / output_buffers[0].size_bytes());
    if (factor < 2 or is_gradient_op)
        return output_buffers;
    int microbatch = graph->get_microbatch();
    factor = FactorizedInt(microbatch).get_nearest_factor_le(factor);
    if (divisible_either_direction(output_buffers[0].block_shape.t, factor))
        output_buffers[0].buffer_factor *= factor;
    return output_buffers;
}

static std::pair<OpModelFailureReason, std::string> validate_memory_requirements(
    OpModel const& op_model, std::size_t l1_usable_size, std::size_t dram_channel_capacity)
{
    std::size_t buffer_usage_bytes = op_model.get_l1_memory_usage();
    if (buffer_usage_bytes > l1_usable_size)
    {
        return std::make_pair(
            L1UsageOverMaxLimit, fmt::format("L1 Usage[{}] > L1 Max[{}]", buffer_usage_bytes, l1_usable_size));
    }

    for (BufferModel const& dram_buffer : op_model.dram_buffers)
    {
        constexpr bool include_t = true;
        if (dram_buffer.size_bytes(include_t) > dram_channel_capacity)
        {
            return make_pair(
                ExceededDramChannelCapacity,
                fmt::format(
                    "Exceeded DRAM channel capacity: Buffer Usage[{}] DRAM Channel[{}]",
                    dram_buffer.size_bytes(include_t),
                    dram_channel_capacity));
        }
    }

    return std::make_pair(NoFailure, "");
}

static bool unpadding_producer_macroblock(Graph const* graph, graphlib::OpNode const* op_node, BlockShape block_shape)
{
    // Constraint 1. Unpadding must be less than the producer grid's macroblock size in both r & c dimensions.
    //       - unpad_rt < producer_mb_r * producer_ub_r and unpad_ct < producer_mb_c * producer_ub_c
    //       - This constraint ensures that every core within the producer grid's output kernel buffer is popped.
    //         Otherwise, data will backpressure and the system will hang.
    //       - This should be a rational constraint because if the data is never read, we should never be producing the
    //       data.

    // Extract macroblock and microblock sizes from the block shape
    int producer_mb_r = block_shape.mblock_m;
    int producer_mb_c = block_shape.mblock_n;
    int producer_ub_r = block_shape.ublock.rt;
    int producer_ub_c = block_shape.ublock.ct;

    // Iterate through outgoing edges to get unpad nodes for the particular op node
    std::vector<tt::graphlib::Edge> outgoing_edges = graph->user_data_edges(op_node);
    for (graphlib::Edge outgoing_edge : outgoing_edges)
    {
        vector<OpType> tms = graph->get_edge_attributes(outgoing_edge)->get_tms();
        for (OpType op_type : tms)
        {
            if (op_type.op == "buda_unpad")
            {
                int unpad_rt = std::get<int>(op_type.buda_attrs["rt"]);
                int unpad_ct = std::get<int>(op_type.buda_attrs["ct"]);
                if (unpad_rt >= producer_mb_r * producer_ub_r || unpad_ct >= producer_mb_c * producer_ub_c)
                {
                    return false;
                }

                // We break on first buda_unpad node because we assume that there is only one buda_unpad TM per outgoing
                // edge
                break;
            }
        }
    }

    return true;
}

static bool padding_consumer_macroblock(Graph const* graph, graphlib::OpNode const* op_node, BlockShape block_shape)
{
    // Padding must be less than the consumer grid's macroblock size in both r & c dimensions.
    //      - pad_rt < consumer_mb_r * consumer_ub_r and pad_ct < consumer_mb_c * consumer_ub_c
    //      - This constraint ensures that every core in the consumer grid is producing some functional data.
    //      - This should be a rational constraint because if we are padding more than an additional macro block,
    //        this means that we are using cores to compute only padding and that we could have satisfied divisibility
    //        constraints with a smaller padding.

    // Extract macroblock and microblock sizes from the block shape
    int consumer_mb_r = block_shape.mblock_m;
    int consumer_mb_c = block_shape.mblock_n;
    int consumer_ub_r = block_shape.ublock.rt;
    int consumer_ub_c = block_shape.ublock.ct;

    // Iterate through incoming edges to get pad nodes for the particular op node
    std::vector<graphlib::Edge> incoming_edges = graph->operand_data_edges(op_node);
    for (graphlib::Edge incoming_edge : incoming_edges)
    {
        vector<OpType> tms = graph->get_edge_attributes(incoming_edge)->get_tms();
        for (OpType op_type : tms)
        {
            if (op_type.op == "buda_pad")
            {
                // Extract padding from the op type
                int pad_rt = std::get<int>(op_type.buda_attrs["rt"]);
                int pad_ct = std::get<int>(op_type.buda_attrs["ct"]);
                if (pad_rt >= consumer_mb_r * consumer_ub_r || pad_ct >= consumer_mb_c * consumer_ub_c)
                {
                    return false;
                }

                // We break on first buda_pad node because we expect only one buda_pad TM per op node edge
                break;
            }
        }
    }

    return true;
}

static bool padding_multiple_pre_stack(Graph const* graph, graphlib::OpNode const* op_node)
{
    // If stacking TMs without full t buffering are used on the data transformation path,
    // padding must be a multiple of the pre-stacked dimension in the dimension of the stack.
    //      - pad_rt % pre_stack_rt == 0 if vstack
    //      - pad_ct % pre_stack_ct == 0 if hstack
    //      - This is a hard constraint due to the nature of the underlying output scatter pipes that implement
    //      stacking.

    // Iterate through incoming edges to get pad nodes for the particular op node
    // with aim to check if pad is a multiple of the pre-stacked dimension.
    std::vector<graphlib::Edge> incoming_edges = graph->operand_data_edges(op_node);
    for (graphlib::Edge incoming_edge : incoming_edges)
    {
        // Get producer node and its shape
        graphlib::NodeId incoming_node_id = incoming_edge.producer_node_id;
        Node* incoming_node = graph->node_by_id(incoming_node_id);
        Shape shape = incoming_node->shape();  // init shape of the operation before PAD

        // Init pre-stack dimension values
        int pre_stack_rt = 1;
        int pre_stack_ct = 1;

        vector<OpType> tms = graph->get_edge_attributes(incoming_edge)->get_tms();
        // Iterate through the all TMs
        for (int i = 0; i < (int)tms.size() - 1; i++)
        {
            shape = ::get_tm_shape(tms[i], shape, true);

            // Check if TM sequence contains stack operation
            if (tms[i].op == "vstack" || tms[i].op == "hstack")
            {
                // Check if padding is after stack
                if (tms[i + 1].op == "buda_pad")
                {
                    std::vector<std::uint32_t> shape_vect = shape.as_vector();
                    std::uint32_t shape_size = shape.size();
                    if (tms[i].op == "vstack")
                    {
                        pre_stack_rt = shape[shape_size - 1];
                    }
                    else if (tms[i].op == "hstack")
                    {
                        pre_stack_ct = shape[shape_size - 2];
                    }

                    // Check if the pad is a multiple of the pre-stacked dimension
                    int pad_rt = std::get<int>(tms[i + 1].buda_attrs["rt"]);
                    int pad_ct = std::get<int>(tms[i + 1].buda_attrs["ct"]);
                    if (pad_rt % pre_stack_rt != 0 || pad_ct % pre_stack_ct != 0)
                    {
                        return false;
                    }
                }
            }
        }
    }

    return true;
}

// Padding constraints
static bool check_padding_constraints(Graph const* graph, graphlib::OpNode const* op_node, BlockShape block_shape)
{
    bool padding_constraints_satisfied = true;

    // Constraint 1
    padding_constraints_satisfied &= unpadding_producer_macroblock(graph, op_node, block_shape);
    // Constraint 2
    padding_constraints_satisfied &= padding_consumer_macroblock(graph, op_node, block_shape);
    // Constraint 3
    padding_constraints_satisfied &= padding_multiple_pre_stack(graph, op_node);

    return padding_constraints_satisfied;
}

static std::pair<OpModel, OpModelFailureReason> calculate_op_model_impl(
    Graph const* graph,
    std::shared_ptr<BalancerCacheCollection> cache_collection,
    graphlib::BudaOpNode const* op_node,
    GridShape selected_grid,
    TStreamFactor t_stream_factor,
    UBlockOrder ublock_order,
    bool force_dram_parameters,
    std::size_t dst_size_tiles,
    std::size_t l1_usable_size,
    std::size_t dram_channel_capacity,
    std::string& customFailureMessage,
    int fracture_factor,
    LegalSparseUKts const& legal_sparse_u_kts,
    int u_kt_override,
    std::map<std::uint32_t, std::uint32_t> const& min_input_buffer_factor_overrides,
    std::optional<int> output_buffer_factor_override,
    bool fallback_single_buffer)
{
    OpModel op_model;

    // If sparse matmul, in0 and in2 shapes depend on u_rt and u_kt. However, we don't know what they are at this point
    // so we assume worst-case (largest shapes) and update later when the values are known.
    //
    op_model.op_shape = get_op_shape(
        graph,
        op_node,
        selected_grid,
        /* u_kt */ 1,
        /* u_rt */ 1,  // u_kt and u_rt both set to 1, this provides worst-case scenario for in0/in2 L1 footprints
        t_stream_factor,
        fracture_factor,
        /* calculate_sparse_in0_in2_shapes */ true);  // Need in0/in2 shapes to calculate their L1 footprint!
    op_model.grid_shape = selected_grid;
    op_model.buda_op_node = op_node;
    op_model.data_format = op_node->output_df();
    op_model.t_stream_factor = t_stream_factor;
    op_model.fracture_factor = fracture_factor;
    auto [pad_rt, pad_ct] = graphlib::get_padding(graph, op_node);
    op_model.padding = Padding(pad_rt, pad_ct);

    int streaming_threshold = env_as<int>("PYBUDA_SUPRESS_T_FACTOR_MM", 0);
    if (streaming_threshold and op_node->is_matmul())
    {
        if (t_stream_factor.t() > streaming_threshold)
            return std::make_pair(op_model, IllegalStreaming);
    }

    TT_ASSERT(op_model.op_shape.outputs.size() == 1, "Currently we only support 1 output for 1 ublock shape below");
    Parallelization total_par(selected_grid.r * t_stream_factor.r, selected_grid.c * t_stream_factor.c);
    UBlockShape ublock;
    std::tie(ublock, op_model.fused_op_ublock_shape) =
        calculate_ublock_shape(op_model.op_shape, total_par, dst_size_tiles, ublock_order, op_node);

    // Calculate output_buffer_factor (buf_size_mb)
    auto [calculated_user_buffer_factor, operand_access_allows_streaming, user_access_allows_streaming, is_legal_stack_for_grid] =
        calculate_user_buffer_factor(graph, op_node, ublock_order, op_model);

    if (not is_legal_stack_for_grid)
        return std::make_pair(op_model, IllegalStackForGrid);

    int output_buffer_factor =
        get_output_buffer_factor(op_node, calculated_user_buffer_factor, output_buffer_factor_override);

    if (not op_model.t_stream_factor.none())
    {
        if (not operand_access_allows_streaming && not user_access_allows_streaming)
        {
            // if both operand_access_allows_streaming and user_access_allows_streaming are
            // false, then we use OperandrAndUserAccessPreventsStreaming
            return std::make_pair(op_model, OperandAndUserAccessPreventsStreaming);
        }
        else if (not operand_access_allows_streaming)
        {
            return std::make_pair(op_model, OperandAccessPreventsStreaming);
        }
        else if (not user_access_allows_streaming)
        {
            return std::make_pair(op_model, UserAccessPreventsStreaming);
        }
    }

    op_model.effective_input_buffer_shape_for_user =
        calculate_effective_input_buffer_shapes_for_users(graph, op_node, t_stream_factor);

    // Calculate output buffer shape
    op_model.output_buffers = calculate_output_buffer_models_for_grid(
        op_model.op_shape,
        selected_grid,
        ublock,
        op_model.t_stream_factor,
        output_buffer_factor,
        op_node->is_gradient_op(),
        op_node->output_df());

    if (not legal_t_streaming(op_model.output_buffers[0].block_shape, op_model.t_stream_factor, ublock_order))
        return std::make_pair(op_model, IllegalStreaming);

    // Calculate parameter buffer shapes
    std::vector<graphlib::Node*> operands = graph->data_operands(op_node);
    op_model.parameter_buffers = calculate_parameter_buffer_models_for_grid(
        op_model,
        operands,
        force_dram_parameters);

    // Calculate intermediate buffer shapes
    TT_ASSERT(op_model.output_buffers.size() == 1);
    op_model.intermediate_buffers = calculate_intermediate_buffer_models_for_grid(
        op_node, op_model.output_buffers[0], op_model.fused_op().get(), op_model.fused_op_ublock_shape);

    // Calculate input buffer shapes

    // After output + parameters have been allocated try to pick input buffers that'll fit in the remaining space
    // Try with parameters in L1 first time around, then fallback to streaming the parameters if we can't fit
    // Note: sparse matmul must be able to fit in0 and in2 as parameters
    BlockShape const& output_block_shape = op_model.output_buffers[0].block_shape;

    bool padding_constraints_satisfied = check_padding_constraints(graph, op_node, output_block_shape);
    if (not padding_constraints_satisfied)
        return std::make_pair(op_model, PaddingConstraintsNotSatisfied);

    int fallback_loop_count = op_node->is_sparse_matmul() ? 3 : op_model.num_parameter_buffers() + 2;
    for (int i = 0; i < fallback_loop_count; ++i)
    {
        bool try_fallback = (i >= 1);
        if (try_fallback) 
        {
            // If we exceed available memory in L1 we can first try to decrease memory usage of the param buffers
            // or fallback to the single buffered output buffer.
            //
            std::optional<std::size_t> potential_max_l1_user = find_max_parameter_buffer_l1_user(op_model);
            if (potential_max_l1_user)
            {
                // Stream the parameter from DRAM.
                //
                std::size_t max_l1_user = *potential_max_l1_user;
                if (op_model.dram_buffers.empty())
                    op_model.dram_buffers.resize(op_model.parameter_buffers.size());
                op_model.dram_buffers[max_l1_user] = op_model.parameter_buffers[max_l1_user];
                op_model.parameter_buffers[max_l1_user] = BufferModel{};
                log_trace(
                    LogBalancer,
                    "{}: cannot fit parameters in L1, fallback to streaming at input index[{}] usage[{}/{}]",
                    op_node->name(),
                    max_l1_user,
                    op_model.get_l1_memory_usage(),
                    l1_usable_size);
            }
            else if (!output_buffer_factor_override and fallback_single_buffer)
            {
                // Make the output buffer single buffered.
                //
                TT_ASSERT(op_model.output_buffers[0].buffer_factor % 2 == 0);
                TT_ASSERT(op_model.output_buffers[0].l1_size_tiles % 2 == 0);
                op_model.output_buffers[0].buffer_factor /= 2;
                op_model.output_buffers[0].l1_size_tiles /= 2;
                fallback_single_buffer = false;
                log_trace(
                    LogBalancer,
                    "{}: cannot fit output buffer in L1, fallback to single buffer usage[{}/{}]",
                    op_node->name(),
                    op_model.get_l1_memory_usage(),
                    l1_usable_size);
            }
        }

        if (op_model.get_l1_memory_usage() >= l1_usable_size)
            continue;

        std::size_t input_l1_buffer_space = l1_usable_size - op_model.get_l1_memory_usage();
        op_model.input_buffers = calculate_input_buffer_models(
            graph,
            op_node,
            op_model.grid_shape,
            op_model.op_shape,
            operands,
            output_block_shape,
            t_stream_factor,
            input_l1_buffer_space,
            fracture_factor,
            op_model.fused_op_ublock_shape,
            legal_sparse_u_kts,
            u_kt_override,
            min_input_buffer_factor_overrides);

        if (op_model.input_buffers.empty())
            break;  // Change this to continue, causes fallout:
                    // tenstorrent/pybuda#1243

        if (op_node->is_sparse_matmul())
        {
            // in2 operand's shape of sparse matmul depends on chosen u_rt and u_kt, we update it here after both u_rt
            // and u_kt have been chosen
            //
            op_model.op_shape = get_op_shape(
                graph,
                op_node,
                selected_grid,
                ublock.rt,
                op_model.input_buffers[1].block_shape.ublock.rt,
                t_stream_factor,
                fracture_factor,
                /* calculate_sparse_in0_in2_shapes */ true);
        }

        try_promote_kernel_broadcast_inputs(
            &cache_collection->pipe_to_kb_len_cache, graph, op_node, l1_usable_size, op_model, force_dram_parameters);

        if (op_model.get_l1_memory_usage() <= l1_usable_size)
            break;
    }

    if (op_model.input_buffers.empty())
        return std::make_pair(op_model, InputBufferAllocationFailure);

    if (!env_as<bool>("PYBUDA_DISABLE_UNROLLED_PARAMETERS", false)) 
    {
        // Try to use the remaining l1 memory to change prefetch type of some parameter buffers to Post-TM.
        // It means that TMs / reblocking on prologue parameter inputs will be pre-evaluated and fully unrolled in 
        // l1 if space permits.
        // This can have beneficial performance implications because it trivializes the kernel read pattern,
        // i.e. all data is exactly in order.
        //
        try_promote_post_tm_parameter_prefetch(op_model, operands, l1_usable_size, force_dram_parameters);
    }

    if (env_as<bool>("PYBUDA_ENABLE_OUTPUT_BUFFER_UPSIZING"))
    {
        std::size_t l1_remaining_size = l1_usable_size - op_model.get_l1_memory_usage();
        op_model.output_buffers =
            upsize_output_buffer(graph, op_model.output_buffers, l1_remaining_size, op_node->is_gradient_op());
    }

    auto operand_edges = graph->operand_data_edges(op_node);
    bool is_reduce_z = graphlib::is_reduce_z(op_node);
    op_model.is_sparse_matmul = op_node->is_sparse_matmul();
    op_model.consumes_rz_major = std::any_of(
                                     operand_edges.begin(),
                                     operand_edges.end(),
                                     [graph](Edge edge) { return edge_tms_consume_rz_major(graph, edge); }) or
                                 is_reduce_z;

    if (op_model.is_sparse_matmul)
    {
        // Check if sparse matmul is valid
        //
        if (!validate_sparse_matmul_model(graph, op_node, op_model))
        {
            return std::make_pair(op_model, IllegalSparseMatmul);
        }

        // Append SparseBUDA object to OpModel
        //
        const sparse::SparseBUDA& sparse_buda =
            graph->data_operands(op_node)[0]->as<graphlib::ConstantInputNode>()->get_sparse_buda();
        op_model.sparse_indices = sparse_buda.sparse_indices.size();
        op_model.sparse_buda = &sparse_buda;
    }

    auto [failedOpMemoryRequirementReason, customMemoryReaquirementReasonMessage] =
        validate_memory_requirements(op_model, l1_usable_size, dram_channel_capacity);

    customFailureMessage = customMemoryReaquirementReasonMessage;
    return std::make_pair(op_model, failedOpMemoryRequirementReason);
}

std::pair<OpModel, OpModelFailureReason> calculate_op_model(
    Graph const* graph,
    std::shared_ptr<BalancerCacheCollection> cache_collection,
    graphlib::BudaOpNode const* op_node,
    GridShape selected_grid,
    TStreamFactor t_stream_factor,
    UBlockOrder ublock_order,
    bool force_dram_parameters,
    std::size_t dst_size_tiles,
    std::size_t l1_usable_size,
    std::size_t dram_channel_capacity,
    std::string& customFailureMessage,
    int fracture_factor,
    LegalSparseUKts const& legal_sparse_u_kts,
    int u_kt_override,
    std::map<std::uint32_t, std::uint32_t> const& min_input_buffer_factor_overrides,
    std::optional<int> output_buffer_factor_override,
    bool fallback_single_buffer)
{
    while (true)
    {
        auto [op_model, failure_reason] = calculate_op_model_impl(
            graph,
            cache_collection,
            op_node,
            selected_grid,
            t_stream_factor,
            ublock_order,
            force_dram_parameters,
            dst_size_tiles,
            l1_usable_size,
            dram_channel_capacity,
            customFailureMessage,
            fracture_factor,
            legal_sparse_u_kts,
            u_kt_override,
            min_input_buffer_factor_overrides,
            output_buffer_factor_override,
            fallback_single_buffer);

        if ((failure_reason == L1UsageOverMaxLimit or failure_reason == InputBufferAllocationFailure) and
            dst_size_tiles > 1)
        {
            dst_size_tiles /= 2;
        }
        else
        {
            return std::make_pair(op_model, failure_reason);
        }
    }
}

// Calculate legal OpModels for a graph.
// Optionally override can be passed in via nodes_to_legalize to only calculate OpModels for specified set of nodes.
//
LegalOpModels get_legal_op_models(
    Graph const* graph,
    BalancerConfig const& config,
    std::shared_ptr<BalancerCacheCollection> cache_collection,
    std::unordered_set<graphlib::Node*>* nodes_to_legalize)
{
    PROFILE_SCOPE();
#ifdef DEBUG
    BudaOpNodeLegalizerFailureInfo op_graph_debug_info;
    bool enable_legalizer_detailed_debugging = env_as<bool>("PYBUDA_LEGALIZER_DETAILED_DEBUGGING");
    std::string node_name_leg_debug = env_as<std::string>("PYBUDA_LEGALIZER_DEBUG_NODE_NAME");
#endif

    std::unordered_map<Node*, const BudaOpNodeLegalizerFailureInfo> nodes_without_legal_op_model;
    LegalOpModels valid_op_models;
    FactorizedShape device_grid(
        FactorizedInt::Factorial(config.device_config.grid_size.r),
        FactorizedInt::Factorial(config.device_config.grid_size.c));

    // Nebula is harvested in Nebula+Galaxy setup, but the device_grid is for
    // unharvested galaxy
    FactorizedShape harvested_device_grid(
        FactorizedInt::Factorial(config.device_config.get_harvested_nebula_galaxy_grid().r),
        FactorizedInt::Factorial(config.device_config.get_harvested_nebula_galaxy_grid().c));

    for (Node* node : tt::graphlib::topological_sort(*graph))
    {
        if (node->node_type() != NodeType::kBudaOp)
        {
            continue;
        }

        graphlib::BudaOpNode const* op_node = static_cast<graphlib::BudaOpNode const*>(node);

        if (nullptr != nodes_to_legalize and nodes_to_legalize->count(node) == 0)
        {
            continue;
        }

        BudaOpNodeLegalizerFailureInfo failure_info;

#ifdef DEBUG
        graphlib::BudaOpNode* debug_op_node = nullptr;
        if (enable_legalizer_detailed_debugging)
        {
            debug_op_node = const_cast<graphlib::BudaOpNode*>(op_node);
            debug_op_node->leg_debug_info = std::make_shared<BudaOpNodeLegalizerFailureInfo>();
        }
#endif

        auto op_override = config.get_op_override(node->name());
        FactorizedInt fracture_factorization = get_fracture_factorization(graph, op_node, op_override);
        std::optional<int> output_buffer_override = get_output_buffer_override(op_node, op_override);
        std::map<std::uint32_t, std::uint32_t> input_buffer_multipliers =
            get_min_input_buffer_multiplier_overrides(op_override);
        int user_overriden_u_kt = get_u_kt(op_override);
        UBlockOrder ublock_order = get_output_ublock_order(graph, op_node);
        bool fallback_single_buffer = config.enable_single_buffer_fallback;
        // Support for full dst mode was removed by backend:
        //   tenstorrent/budabackend#1543
        // Follow up for re-enablement:
        //   tenstorrent/budabackend#2098
        bool full_dst_mode = false and op_node->is_sparse_matmul() and env_as<bool>("PYBUDA_MAXIMIZE_SPARSE_UBLOCK");
        std::size_t dst_size_tiles = calculate_dst_size_tiles(
            config.device_config.get_dst_size(),
            op_node->accumulate_df(),
            op_node->shape().get_tile_volume(),
            full_dst_mode ? 1 : 2);

        std::vector<OpModel> valid_grids;
        for (int fracture_factor : fracture_factorization.get_factors())
        {
            // all_pars can extend beyond the device grid, used to express t-streaming
            auto all_pars = FactorizedShape(get_parallelization(graph, op_node, fracture_factor));
            all_pars.c = all_pars.c.keep_factors_divisible_by(
                FactorizedInt::Constant(fracture_factor));  // remove invalid factors
            // TODO: each op's parallelization() should define FactorizedShape, instead of returning a 2-tuple, in order
            // to avoid having the line above (which is specific to sparse mm)
            auto grid_pars = all_pars & device_grid;
            bool force_dram_parameters = config.default_dram_parameters;
            FactorizedShape overridden_streaming_pars;

            // output ops will be placed on Nebula hence they should fit a harvested grid
            if (env_as<bool>("PYBUDA_NEBULA_GALAXY_PLACER"))
            {
                auto consumers = graph->users(op_node);
                bool feeds_graph_output_queue = std::any_of(
                    consumers.begin(),
                    consumers.end(),
                    [](Node* n) { return n->node_type() == graphlib::NodeType::kOutput; });
                if (feeds_graph_output_queue)
                {
                    grid_pars = grid_pars & harvested_device_grid;
                }
            }

            std::vector<TStreamDir> streaming_dirs = get_legal_streaming_dirs(graph, op_node);

            log_debug(LogBalancer, "Calculate legal op models for node {} {}:", node->name(), node->get_type());

            bool override_enable_t_streaming = not config.manual_t_streaming;
            if (auto op_override = config.get_op_override(node->name()))
                op_override->apply(
                    grid_pars,
                    force_dram_parameters,
                    streaming_dirs,
                    overridden_streaming_pars,
                    override_enable_t_streaming,
                    node->name());

            bool enable_t_streaming = config.enable_t_streaming and override_enable_t_streaming and
                                      (not node->as<graphlib::TaggedNode>()->has_tag("padding_nop"));

            log_trace(LogBalancer, "  Grids:");
            for (Parallelization grid_par : grid_pars)
            {
                bool did_non_streaming = false;
                for (auto streaming_dir : streaming_dirs)
                {
                    auto [streaming_pars, legal_sparse_u_kts] = calculate_streaming_pars(
                        graph,
                        op_node,
                        grid_par,
                        all_pars,
                        streaming_dir,
                        overridden_streaming_pars,
                        enable_t_streaming,
                        fracture_factor);

                    for (auto streaming_par : streaming_pars)
                    {
                        if (did_non_streaming and streaming_par == Parallelization(1, 1))
                            continue;  // We already covered this case with TStreamDir::R, i.e. non-streaming
                        did_non_streaming |= (streaming_par == Parallelization(1, 1));

                        std::string customFailureMessage;

                        auto [op_model, failure_reason] = calculate_op_model(
                            graph,
                            cache_collection,
                            op_node,
                            grid_par,
                            TStreamFactor(streaming_dir, streaming_par),
                            ublock_order,
                            force_dram_parameters,
                            dst_size_tiles,
                            config.device_config.get_l1_usable_size(),
                            config.device_config.get_dram_channel_capacity(),
                            customFailureMessage,
                            fracture_factor,
                            legal_sparse_u_kts,
                            user_overriden_u_kt,
                            input_buffer_multipliers,
                            output_buffer_override,
                            fallback_single_buffer);

                        if (NoFailure == failure_reason)
                        {
                            valid_grids.push_back(op_model);

                            for (int u_kt_override :
                                 enumerate_factored_u_kts(op_model, user_overriden_u_kt, config.enable_enumerate_u_kt))
                            {
                                auto [factored_u_kt_op_model, factored_u_kt_failure_reason] = calculate_op_model(
                                    graph,
                                    cache_collection,
                                    op_node,
                                    grid_par,
                                    TStreamFactor(streaming_dir, streaming_par),
                                    ublock_order,
                                    force_dram_parameters,
                                    dst_size_tiles,
                                    config.device_config.get_l1_usable_size(),
                                    config.device_config.get_dram_channel_capacity(),
                                    customFailureMessage,
                                    fracture_factor,
                                    legal_sparse_u_kts,
                                    u_kt_override,
                                    input_buffer_multipliers,
                                    output_buffer_override,
                                    fallback_single_buffer);
                                if (factored_u_kt_failure_reason == NoFailure)
                                    valid_grids.push_back(factored_u_kt_op_model);
                            }

                            log_trace(
                                LogBalancer,
                                "    {} {:<32} {} Legalizer Valid",
                                op_node->name(),
                                GridShape(grid_par),
                                TStreamFactor(streaming_dir, streaming_par));
                            log_trace(LogBalancer, "      L1: {:<16}", op_model.get_l1_memory_usage());
                            log_trace(
                                LogBalancer,
                                "      Cycles: {:<16}",
                                op_model.get_execution_cycles(config.device_config.arch_name));
                            log_trace(LogBalancer, "{}", op_model);
                        }
                        else
                        {
                            log_trace(
                                LogBalancer,
                                "    {} {:<26} {} Legalizer Failed: {}",
                                op_node->name(),
                                GridShape(grid_par),
                                TStreamFactor(streaming_dir, streaming_par),
                                customFailureMessage.empty() ? OpModelFailureReasonMessages[failure_reason]
                                                             : customFailureMessage);
                            log_trace(LogBalancer, "{}", op_model);
                            failure_info.recordOpModelFailure(failure_reason);
                        }

#ifdef DEBUG
                        if (enable_legalizer_detailed_debugging)
                        {
                            debug_op_node->leg_debug_info->recordOpModelFailure(failure_reason);
                        }

                        op_graph_debug_info.recordOpModelFailure(failure_reason);
#endif
                    }
                }
            }
        }

#ifdef DEBUG
        if (enable_legalizer_detailed_debugging)
        {
            if (node_name_leg_debug == node->name() or node_name_leg_debug.empty())
            {
                log_debug(
                    LogBalancer,
                    "OpModel failure statistics for node: {} {} {}",
                    node->name(),
                    node->get_type(),
                    node->shape());
                log_debug(LogBalancer, debug_op_node->leg_debug_info->toString().c_str());
            }
        }
#endif

        log_debug(LogBalancer, "Total op models for node: {} {}", node->name(), valid_grids.size());
        if (valid_grids.empty())
        {
            nodes_without_legal_op_model.emplace(node, failure_info);
            std::uint32_t buffer_alloc_cnt = nodes_without_legal_op_model[node].getOpModelFailureCountByType(
                OpModelFailureReason::InputBufferAllocationFailure);
            std::uint32_t user_access_cnt = nodes_without_legal_op_model[node].getOpModelFailureCountByType(
                OpModelFailureReason::UserAccessPreventsStreaming);
            std::uint32_t operand_access_cnt = nodes_without_legal_op_model[node].getOpModelFailureCountByType(
                OpModelFailureReason::OperandAccessPreventsStreaming);
            std::uint32_t operand_and_user_access_cnt = nodes_without_legal_op_model[node].getOpModelFailureCountByType(
                OpModelFailureReason::OperandAndUserAccessPreventsStreaming);
            log_warning(
                LogBalancer,
                "No valid grids found for node: {} {} {}, buffer_alloc_cnt {},  user_access_cnt {} , "
                "operand_access_cnt {}, operand_and_user_access_cnt {}",
                node->name(),
                node->get_type(),
                node->shape(),
                buffer_alloc_cnt,
                user_access_cnt,
                operand_access_cnt,
                operand_and_user_access_cnt);
        }
        valid_op_models.emplace(node, valid_grids);
    }

#ifdef DEBUG
    log_debug(LogBalancer, "OpModel failure statistics for whole graph:");
    log_debug(LogBalancer, op_graph_debug_info.toString().c_str());
#endif
    if (nodes_without_legal_op_model.size() > 0)
    {
        std::size_t nodes_without_legal_op_model_count = nodes_without_legal_op_model.size();
        throw BalancerError(
            fmt::format("{} Nodes have no valid grids, exiting", nodes_without_legal_op_model_count),
            BalancerError::NoValidGrid(std::move(nodes_without_legal_op_model)));
    }

    return valid_op_models;
}

static OpModel create_input_queue_op_model(
    TensorShape input_shape, GridShape grid_shape, BlockShape block_shape, DataFormat data_format, bool prologue)
{
    BufferModel input_buffer_model;
    input_buffer_model.block_shape = block_shape;
    input_buffer_model.buffer_factor = 1;
    input_buffer_model.l1_size_tiles = input_buffer_model.block_shape.volume();
    input_buffer_model.data_format = data_format;

    OpModel input_op_model;
    input_op_model.grid_shape = grid_shape;
    input_op_model.op_shape.outputs.push_back(input_shape);
    input_op_model.output_buffers.push_back(input_buffer_model);
    input_op_model.data_format = data_format;
    input_op_model.input_prologue = prologue;

    return input_op_model;
}

static void resolve_input_queue_block_shapes(Graph const* graph, BalancerConfig const& config, OpModelMap& op_models)
{
    auto compatible_queue_grid_for_users = [](TensorShape const& input_shape,
                                              std::vector<OpModel const*> const& users,
                                              bool parameter = false) -> GridShape
    {
        GridShape grid_shape = users[0]->grid_shape;
        for (OpModel const* user_op_model : users)
        {
            GridShape user_grid_shape = user_op_model->grid_shape;
            bool user_is_matmul = (user_op_model->op_type() == "matmul");
            grid_shape.r = std::min(grid_shape.r, user_grid_shape.r);
            grid_shape.c = std::min(
                grid_shape.c,
                (user_is_matmul && !parameter)
                    ? 1
                    : user_grid_shape.c);  // for matmul, only one column reads, so giving it more only hurts it
        }

        int grid_r = FactorizedInt(input_shape.rt).get_nearest_factor_le(grid_shape.r);
        int grid_c = FactorizedInt(input_shape.ct).get_nearest_factor_le(grid_shape.c);
        return GridShape(grid_r, grid_c);
    };

    auto compatible_queue_ublock_for_users = [](TensorShape const& input_shape,
                                                GridShape grid_shape,
                                                std::vector<graphlib::Edge> const& user_edges,
                                                std::vector<OpModel const*> const& users) -> UBlockShape
    {
        TT_ASSERT(not user_edges.empty());
        TT_ASSERT(not users.empty());
        TT_ASSERT(user_edges.size() == users.size());
        // For now just take the first user, unclear what's best for all users
        graphlib::Edge user_edge = user_edges.front();
        OpModel const* user_op_model = users.front();
        UBlockShape ublock = user_op_model->input_buffers[user_edge.consumer_input_port_id].block_shape.ublock;

        // Clamp ublock to tensor shape, needed if bcasting
        TT_ASSERT((input_shape.rt % grid_shape.r) == 0);
        TT_ASSERT((input_shape.ct % grid_shape.c) == 0);
        int par_r = input_shape.rt / grid_shape.r;
        int par_c = input_shape.ct / grid_shape.c;
        ublock.rt = gcd(ublock.rt, par_r);
        ublock.ct = gcd(ublock.ct, par_c);

        return ublock;
    };

    // when enabled, we won't force the input-activations to be blocked to 1x1
    bool enable_reblock_input_activations = env_as<bool>("PYBUDA_REBLOCK_INPUT_ACT");
    const std::uint32_t reblock_input_max_size =
        64;  // reblock small inputs smaller than this, regardless of enable switch

    for (Node* node : graph->nodes())
    {
        switch (node->node_type())
        {
            case NodeType::kInput:
            {
                static constexpr int kMaxPrefetchBufStreams = 24;

                GridShape grid_shape;
                BlockShape block_shape;
                graphlib::Shape shape = node->shape();
                TensorShape input_shape(shape);
                graphlib::InputNode* input = dynamic_cast<graphlib::InputNode*>(node);
                std::vector<graphlib::Node*> data_loopback = graph->data_operands(node);
                std::vector<graphlib::Edge> user_edges = graph->user_data_edges(node);
                std::vector<OpModel const*> users;
                std::vector<OpModel const*> prologue_users;

                auto is_partial_datacopy_edge = [](Edge e)
                { return (e.edge_type == graphlib::EdgeType::kPartialDataCopy); };
                std::vector<graphlib::Edge> partial_datacopy_edges =
                    graph->operand_edges(node, is_partial_datacopy_edge);
                for (auto edge : user_edges)
                {
                    graphlib::Node* user = graph->node_by_id(edge.consumer_node_id);
                    OpModel const& user_op_model = op_models.at(user->name());
                    users.push_back(&user_op_model);
                    if (user_op_model.parameter_buffers[edge.consumer_input_port_id])
                    {
                        prologue_users.push_back(&user_op_model);
                    }
                }
                TT_ASSERT(not users.empty(), "Node {} has no users", node->name());
                bool all_users_prologue = prologue_users.size() == users.size();
                bool is_embedding_table =
                    input->is_parameter() and
                    graph->node_by_id(user_edges.front().consumer_node_id)->as<graphlib::OpNode>()->is_embedding();

                auto users_tilize = graph->data_users(input);

                bool is_tilize_op_input = std::any_of(
                    users_tilize.begin(),
                    users_tilize.end(),
                    [](auto* n)
                    {
                        graphlib::OpNode* op_node = dynamic_cast<graphlib::OpNode*>(n);
                        return op_node->is_tilize();
                    });
                //
                // Each branch must initialize grid_shape and block_shape
                //
                if (is_embedding_table || is_tilize_op_input)
                {
                    TT_ASSERT(!is_embedding_table || users.size() == 1);
                    TT_ASSERT(!is_embedding_table || user_edges.size() == 1);
                    TT_ASSERT(user_edges.front().consumer_input_port_id == 0);
                    // Embedding table constraints
                    //   - prologue = false
                    //   - grid_r must = 1 for now
                    //   - grid_c must = op.grid_c
                    //   - mblock_m must = [1, 1]
                    all_users_prologue = false;
                    OpModel const& op_model = *users.front();
                    grid_shape.r = 1;
                    grid_shape.c = op_model.grid_shape.c;
                    TT_ASSERT(input->shape().ct() % grid_shape.c == 0);

                    if (is_embedding_table)
                    {
                        block_shape =
                            BlockShape(1, 1, 1, UBlockShape(input->shape().rt(), input->shape().ct() / grid_shape.c));
                    }
                    else if (is_tilize_op_input)
                    {
                        block_shape = BlockShape(input_shape, 1, 1, 1, UBlockShape(1, op_model.ublock_shape().ct));
                    }
                }

                else if (not partial_datacopy_edges.empty())
                {
                    // op model for partial datacopy inputs is determined by output that feeds it
                    auto* output_node = graph->node_by_id(partial_datacopy_edges.front().producer_node_id);
                    auto output_operands = graph->data_operands(output_node);
                    TT_ASSERT(output_operands.size() == 1);
                    auto* writeback_op = output_operands.front();
                    OpModel const& op_model = op_models.at(writeback_op->name());
                    grid_shape = op_model.grid_shape;
                    block_shape = op_model.block_shape();
                    for (auto edge : partial_datacopy_edges)
                    {
                        auto other_output = graph->node_by_id(edge.producer_node_id);
                        auto other_writeback_op = graph->data_operands(other_output).front();
                        OpModel const& other_op_model = op_models.at(other_writeback_op->name());
                        TT_ASSERT(
                            other_op_model.grid_shape == grid_shape,
                            "Partial datacopy grid shape mismatch on {} and {}",
                            writeback_op->name(),
                            other_output->name());
                        bool block_shapes_match = other_op_model.block_shape().mblock_m == block_shape.mblock_m and
                                                  other_op_model.block_shape().mblock_n == block_shape.mblock_n and
                                                  other_op_model.block_shape().ublock == block_shape.ublock;
                        TT_ASSERT(
                            block_shapes_match,
                            "Partial datacopy block shape mismatch on (note, t's don't have to match)",
                            writeback_op->name(),
                            other_op_model.block_shape(),
                            other_output->name(),
                            block_shape);
                    }

                    // Update read-view with t multiplier
                    TT_ASSERT(node->shape().volume() % output_node->shape().volume() == 0);
                    size_t multiplier = node->shape().volume() / output_node->shape().volume();
                    block_shape.t *= multiplier;
                }
                else if (not data_loopback.empty())
                {
                    // If an optimizer node writes to this input (kDataLoopback) then we need to inherit its blockshape
                    auto node = data_loopback[0];
                    if (node->node_type() == NodeType::kOutput)
                    {
                        node = graph->data_operands(node)[0];
                    }
                    OpModel const& op_model = op_models.at(node->name());
                    grid_shape = op_model.grid_shape;
                    block_shape = op_model.block_shape();

                    // Users need to be at least as big as the optimizer op writing to it because otherwise the
                    // parameters wouldn't be able to fit on their core grid. This can be enforced by the balancer
                    // policies, but for now we assert.
                    for (OpModel const* user_op_model : prologue_users)
                    {
                        GridShape user_grid_shape = user_op_model->grid_shape;
                        if (user_grid_shape.r < grid_shape.r or user_grid_shape.c < grid_shape.c)
                        {
                            log_debug(
                                LogBalancer,
                                "Optimizer grid for input exceeds consumer op grid dims: {} optimizer({}) user({})",
                                node->name(),
                                grid_shape,
                                user_grid_shape);
                            log_debug(LogBalancer, "  Fallback to stream parameters: {}", node->name());
                            all_users_prologue = false;
                        }
                    }
                }
                else if (input and (input->is_parameter() or input->is_optimizer_parameter() or input->is_constant()))
                {
                    // If it's a parameter, we need the grid shape of the smallest consumer grid dims
                    grid_shape = compatible_queue_grid_for_users(input_shape, users, true /*parameter*/);
                    UBlockShape ublock = compatible_queue_ublock_for_users(input_shape, grid_shape, user_edges, users);
                    block_shape = BlockShape(input_shape, grid_shape.r, grid_shape.c, 1, ublock);

                    // Test to make sure that after placing all ops that reference this prologue buffer still fit in L1
                    // Fallback to streaming the param buffer
                    if (all_users_prologue and prologue_users.size() > 1)
                    {
                        int idx = 0;
                        for (OpModel const* user_op_model_ptr : prologue_users)
                        {
                            // Take a copy to test if we fit in L1 with updated parameter grid blocking
                            OpModel user_op_model = *user_op_model_ptr;
                            Edge edge = user_edges[idx++];

                            // Only replace the parameter buffer model if not kernel broadcast, we've
                            // already determined that the entire buffer can fit in this core's L1
                            bool is_kernel_broadcast =
                                user_op_model.input_buffers[edge.consumer_input_port_id].kernel_broadcast_tiles > 0;
                            if (not is_kernel_broadcast)
                                user_op_model.parameter_buffers[edge.consumer_input_port_id] =
                                    BufferModel(block_shape, 1, graph->node_by_id(edge.producer_node_id)->output_df());

                            bool out_of_memory =
                                user_op_model.get_l1_memory_usage() > config.device_config.get_l1_usable_size();
                            int num_prefetch_streams = 0;
                            auto user = graph->node_by_id(edge.consumer_node_id);
                            for (auto operand_edge : graph->operand_data_edges(user))
                            {
                                if (user_op_model.parameter_buffers[operand_edge.consumer_input_port_id])
                                {
                                    auto operand_shape = graph->node_by_id(operand_edge.producer_node_id)->shape();
                                    std::vector<OpModel const*> operand_users;
                                    for (auto operand_user_node :
                                         graph->data_users(graph->node_by_id(operand_edge.producer_node_id)))
                                    {
                                        operand_users.push_back(&op_models.at(operand_user_node->name()));
                                    }
                                    auto operand_grid_shape = compatible_queue_grid_for_users(
                                        operand_shape, operand_users, true /*parameter*/);

                                    num_prefetch_streams +=
                                        (round_up_div(user_op_model.grid_shape.r, operand_grid_shape.r) *
                                         round_up_div(user_op_model.grid_shape.c, operand_grid_shape.c));
                                }
                            }
                            bool out_of_prefetch_streams = num_prefetch_streams > kMaxPrefetchBufStreams;

                            if (out_of_memory or out_of_prefetch_streams)
                            {
                                // tenstorrent/pybuda#390
                                // TT_ASSERT(prologue_users.size() > 1, "Single user should alway fit in L1, unless op
                                // model calculation changed");

                                log_debug(
                                    LogBalancer,
                                    "Smallest consumer grid shape forces other parameter consumer to fall out of L1, "
                                    "prologue_users[{}] out_of_memory[{}] out_of_prefetch_streams[{}]",
                                    prologue_users.size(),
                                    out_of_memory,
                                    out_of_prefetch_streams);
                                log_debug(LogBalancer, "  Fallback to stream parameters: {}", node->name());
                                all_users_prologue = false;
                                break;
                            }
                        }
                    }
                }
                else if (
                    (enable_reblock_input_activations or
                     (node->shape().rt() * node->shape().ct() <= reblock_input_max_size)) and
                    input and input->is_activation())
                {
                    // If it's activation, we'll arbitrarily pick the smallest grid shape
                    grid_shape = compatible_queue_grid_for_users(input_shape, users);
                    UBlockShape ublock = compatible_queue_ublock_for_users(input_shape, grid_shape, user_edges, users);
                    block_shape = BlockShape(input_shape, grid_shape.r, grid_shape.c, 1, ublock);
                }
                else
                {
                    // We can choose anything for ordinary input, so 1x1 grid/ublock for now (to support bcast shapes)
                    grid_shape = GridShape(1, 1);
                    block_shape = BlockShape(input_shape, grid_shape.r, grid_shape.c, 1, UBlockShape(1, 1));

                    bool exceeds_dram_channel_size = (block_shape.volume() * tile_size_bytes(node->output_df())) >
                                                     config.device_config.get_dram_channel_capacity();
                    if (exceeds_dram_channel_size)
                    {
                        FactorizedShape legal_grid_shapes = FactorizedShape(input_shape.rt, input_shape.ct);
                        FactorizedShape::Iterator legal_grid_shapes_iter = legal_grid_shapes.begin();
                        bool init = true;
                        while (exceeds_dram_channel_size and legal_grid_shapes_iter != legal_grid_shapes.end())
                        {
                            if (init)
                            {
                                grid_shape = compatible_queue_grid_for_users(input_shape, users);
                                init = false;
                            }
                            else
                            {
                                grid_shape = GridShape(*legal_grid_shapes_iter++);
                            }

                            block_shape = BlockShape(input_shape, grid_shape.r, grid_shape.c, 1, UBlockShape(1, 1));
                            exceeds_dram_channel_size = (block_shape.volume() * tile_size_bytes(node->output_df())) >
                                                        config.device_config.get_dram_channel_capacity();
                        }

                        TT_ASSERT(
                            not exceeds_dram_channel_size,
                            "Could not find queue grid size large enough to fit queue into dram");
                    }
                }

                OpModel op_model = create_input_queue_op_model(
                    input_shape, grid_shape, block_shape, node->output_df(), all_users_prologue);
                op_models.emplace(node->name(), op_model);
                break;
            }
            default: break;
        }
    }
}

std::tuple<OpModelMap, BlockShapeMap, OutputHostTMMap, CutEdges> resolve_block_shapes(
    Graph const* graph, BalancerConfig const& config, GraphSolverSolution const& graph_solver_solution)
{
    log_debug(LogBalancer, "Resolve block shapes:");
    OpModelMap op_models;
    OutputHostTMMap output_host_tms;

    for (Node* node : graph->nodes())
    {
        if (node->node_type() != NodeType::kBudaOp)
        {
            continue;
        }
        TT_LOG_ASSERT(
            graph_solver_solution.selected_op_models.count(node) > 0, "Missing op model for node {}", node->name());
        op_models.emplace(node->name(), graph_solver_solution.selected_op_models.at(node));
    }

    resolve_input_queue_block_shapes(graph, config, op_models);

    BlockShapeMap block_shape_map;
    for (Node* node : tt::graphlib::topological_sort(*graph))
    {
        auto is_partial_datacopy_edge = [](Edge e) { return (e.edge_type == graphlib::EdgeType::kPartialDataCopy); };
        std::vector<graphlib::Edge> partial_datacopy_operand_edges =
            graph->operand_edges(node, is_partial_datacopy_edge);

        BlockShape block_shape;
        switch (node->node_type())
        {
            case NodeType::kInput:
            {
                block_shape = op_models.at(node->name()).block_shape();
                break;
            }
            case NodeType::kOutput:
            {
                // Scale the block based on the operand's grid shape, since output queue is always on one "core" (host)
                std::vector<Node*> operands = graph->data_operands(node);
                TT_ASSERT(operands.size() == 1);
                Node* operand = operands[0];
                OpModel const& operand_op_model = op_models.at(operand->name());
                BlockShape operand_block_shape = operand_op_model.block_shape();
                GridShape operand_grid = operand_op_model.grid_shape;

                block_shape = operand_block_shape;
                std::vector<graphlib::Edge> partial_datacopy_edges = graph->user_edges(node, is_partial_datacopy_edge);

                if (not operand_op_model.t_stream_factor.none())
                {
                    OutputHostTM tm;
                    tm.hstack_factor = operand_op_model.t_stream_factor.c;
                    tm.vstack_factor = operand_op_model.t_stream_factor.r;
                    tm.row_major = operand_op_model.t_stream_factor.dir.r();
                    output_host_tms.emplace(node->name(), tm);
                    if (not tm.row_major or tm.hstack_factor > 1)
                    {
                        node->as<graphlib::OutputNode>()->set_untilize(false);
                    }
                }

                if (config.output_queues_on_host and node->as<graphlib::OutputNode>()->untilize() and
                    partial_datacopy_edges.empty())
                {
                    block_shape.mblock_m *= (operand_grid.r);
                    block_shape.mblock_n *= (operand_grid.c);
                }

                log_debug(LogBalancer, "  kOutput {:64} {} inherit: {}", node->name(), block_shape, operand->name());
                break;
            }
            case NodeType::kQueue:
            {
                std::vector<Node*> operands = graph->data_operands(node);
                TT_ASSERT(operands.size() == 1);
                Node* operand = operands[0];
                OpModel const& operand_op_model = op_models.at(operand->name());
                block_shape = operand_op_model.block_shape();
                if (not operand_op_model.t_stream_factor.none())
                {
                    OutputHostTM tm;
                    tm.hstack_factor = operand_op_model.t_stream_factor.c;
                    tm.vstack_factor = operand_op_model.t_stream_factor.r;
                    tm.row_major = operand_op_model.t_stream_factor.dir.r();
                    output_host_tms.emplace(node->name(), tm);
                }
                log_debug(LogBalancer, "  kQueue {:64} {} inherit: {}", node->name(), block_shape, operand->name());
                break;
            }
            case NodeType::kBudaOp:
            {
                OpModel& op_model = op_models.at(node->name());
                block_shape = op_model.block_shape();
                break;
            }
            case NodeType::kBudaNaryTM:
            {
                break;
            }
            default:
            {
                TT_ASSERT(false, "Unhandled node_type", node->node_type());
                break;
            }
        }

        log_debug(LogBalancer, "  {:64} {} {}", node->name(), block_shape, node->shape());
        block_shape_map.emplace(node->name(), block_shape);
    }

    return std::make_tuple(op_models, block_shape_map, output_host_tms, graph_solver_solution.cut_edges);
}

}  // namespace tt::balancer::legalizer
