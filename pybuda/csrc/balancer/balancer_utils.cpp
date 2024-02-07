// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "balancer_utils.hpp"

#include <cstdint>
#include <unordered_map>

#include "passes/t_stream.hpp"
#include "utils/hash_combine.hpp"
#include "utils/logger.hpp"
#include "utils/profile.hpp"

using NodeType = tt::graphlib::NodeType;
using UBlockOrder = tt::graphlib::UBlockOrder;

namespace tt::balancer
{

// Returns OpShape
//
// Sparse matmul in0 and in2 shapes are costly to calculate, but are often not needed - flag
// calculate_sparse_in0_in2_shapes is used to control whether to calculate them
//
OpShape get_op_shape(
    Graph const* graph,
    Node const* node,
    GridShape grid_shape,
    int u_rt,
    int u_kt,
    TStreamFactor t_stream_factor,
    int fracture_factor,
    bool calculate_sparse_in0_in2_shapes)
{
    std::vector<TensorShape> producer_shapes;
    std::vector<TensorShape> input_shapes;
    std::vector<TensorShape> output_shapes;

    const graphlib::OpNode* op_node = node->as<graphlib::OpNode>();

    if (op_node->is_sparse_matmul() and calculate_sparse_in0_in2_shapes)
    {
        std::vector<graphlib::Node*> data_operands = graph->data_operands(node);
        graphlib::ConstantInputNode* cin = data_operands[0]->as<graphlib::ConstantInputNode>();
        const sparse::SparseBUDA& sparse_buda = cin->get_sparse_buda();

        int sparse_ct = -1;
        int encodings_ct = -1;

        // Below functions are estimates, but are very accurate for the purposes of calculating the shapes
        // - To improve estimates for sparse tiles, we should include the sparse layout - this would make the numbers
        // exact
        // - To improve estimates for encoding tiles, we should include the sparse layout but there's still some very
        // minor encoding details that are not modelled here - however, we're talking few bytes of difference
        sparse_ct = sparse_buda.get_sparse_tiles_per_core_estimate(grid_shape.r, t_stream_factor.r);
        encodings_ct = sparse_buda.get_encoding_tiles_per_core_estimate(grid_shape.r, t_stream_factor.r, u_rt, u_kt);

        // in0
        input_shapes.emplace_back(1, 1, grid_shape.r, sparse_ct);

        // in1
        graphlib::Edge in1_edge = graph->operand_data_edges(node).at(1);
        graphlib::Shape in1_shape = data_operands[1]->shape();
        std::vector<graphlib::OpType> tms = graph->get_edge_attributes(in1_edge)->get_tms();
        insert_t_stream_tms(
            node->as<graphlib::OpNode>(), tms, t_stream_factor, TStreamFactor{}, in1_edge.consumer_input_port_id);
        input_shapes.emplace_back(post_tms_shape(in1_shape, tms));

        // in2
        input_shapes.emplace_back(1, 1, grid_shape.r, encodings_ct);

        // out
        graphlib::Shape out_shape = node->shape().canonical();
        graphlib::Shape new_shape = graphlib::Shape::create_buda(
            out_shape.as_vector()[0],
            out_shape.as_vector()[1],
            out_shape.as_vector()[2] / fracture_factor,
            out_shape.as_vector()[3] * fracture_factor);
        output_shapes.emplace_back(new_shape);

        producer_shapes.resize(3);
        producer_shapes[0] = input_shapes[0];
        producer_shapes[1] = in1_shape;
        producer_shapes[2] = input_shapes[2];

        return OpShape(producer_shapes, input_shapes, output_shapes);
    }
    else
    {
        for (graphlib::Edge edge : graph->operand_data_edges(node))
        {
            graphlib::Shape producer_shape = graph->node_by_id(edge.producer_node_id)->shape();
            producer_shapes.emplace_back(producer_shape);
            std::vector<graphlib::OpType> tms = graph->get_edge_attributes(edge)->get_tms();
            insert_t_stream_tms(
                node->as<graphlib::OpNode>(), tms, t_stream_factor, TStreamFactor{}, edge.consumer_input_port_id);
            input_shapes.emplace_back(post_tms_shape(producer_shape, tms));
        }

        return OpShape(producer_shapes, input_shapes, {TensorShape(node->shape())});
    }
}

std::vector<graphlib::OpType> calculate_t_streaming_tms(Graph const*, Node const*, OpModel const& op_model)
{
    if (op_model.t_stream_factor.none())
    {
        return {};
    }

    std::vector<graphlib::OpType> tms;
    if (op_model.t_stream_factor.dir.r())
    {
        if (op_model.t_stream_factor.r > 1)
        {
            tms.push_back(graphlib::OpType("vslice", {op_model.t_stream_factor.r}, {}));
        }
        if (op_model.t_stream_factor.c > 1)
        {
            tms.push_back(graphlib::OpType("hslice", {op_model.t_stream_factor.c}, {}));
        }
    }
    else
    {
        if (op_model.t_stream_factor.c > 1)
        {
            tms.push_back(graphlib::OpType("hslice", {op_model.t_stream_factor.c}, {}));
        }
        if (op_model.t_stream_factor.r > 1)
        {
            tms.push_back(graphlib::OpType("vslice", {op_model.t_stream_factor.r}, {}));
        }
    }
    return tms;
}

std::pair<CanCoord, TensorShape> map_inverse_tms(
    CanCoord coord, TensorShape shape, std::vector<graphlib::OpType> const& tms)
{
    for (auto iter = tms.rbegin(); iter != tms.rend(); ++iter)
    {
        graphlib::OpType const& tm = *iter;
        // TODO: this string comparison is a little ridiculous, we should adopt union graphlib::TM for faster TM eval
        //       but for now it makes this routine ~10x faster and measurable in constraint solving time
        switch (tm.op[0])
        {
            case 'b':  // broadcast
            {
                if (tm.op[1] == 'u')  // buda_pad
                {
                    int rt = std::get<int>(tm.attr[0]);
                    int ct = std::get<int>(tm.attr[1]);
                    if (tm.op[5] == 'p')  // pad
                    {
                        shape.rt -= rt;
                        shape.ct -= ct;
                        coord.rt = std::min(coord.rt, shape.rt - 1);
                        coord.ct = std::min(coord.ct, shape.ct - 1);
                    }
                    else  // unpad
                    {
                        shape.rt += rt;
                        shape.ct += ct;
                    }
                }
                else
                {
                    int dim = std::get<int>(tm.attr[0]);
                    int factor = std::get<int>(tm.attr[1]);
                    switch (dim)
                    {
                        case 1:
                        {
                            shape.z /= factor;
                            coord.t %= shape.z;
                            break;
                        }
                        case 2:
                        {
                            shape.rt /= factor;
                            coord.rt %= shape.rt;
                            break;
                        }
                        case 3:
                        {
                            shape.ct /= factor;
                            coord.ct %= shape.ct;
                            break;
                        }
                        default:
                        {
                            log_fatal(LogBalancer, "Unsupported broadcast dim: {}", dim);
                            break;
                        }
                    }
                }
                break;
            }
            case 'v':
            {
                if (tm.op[2] == 'l')  // vslice
                {
                    int factor = std::get<int>(tm.attr[0]);
                    TT_ASSERT(shape.z % factor == 0);
                    coord.rt = (coord.t % factor) * shape.rt + coord.rt;
                    coord.t /= factor;
                    shape.z /= factor;
                    shape.rt *= factor;
                    TT_ASSERT(coord.t < shape.z);
                    TT_ASSERT(coord.rt < shape.rt);
                }
                else  // vstack
                {
                    int factor = std::get<int>(tm.attr[0]);
                    TT_ASSERT(shape.rt % factor == 0);
                    shape.rt /= factor;
                    shape.z *= factor;
                    coord.t = coord.t * factor + coord.rt / shape.rt;
                    coord.rt %= shape.rt;
                    TT_ASSERT(coord.t < shape.z);
                    TT_ASSERT(coord.rt < shape.rt);
                }
                break;
            }
            case 'h':
            {
                if (tm.op[2] == 'l')  // hslice
                {
                    int factor = std::get<int>(tm.attr[0]);
                    TT_ASSERT(shape.z % factor == 0);
                    coord.ct = (coord.t % factor) * shape.ct + coord.ct;
                    coord.t /= factor;
                    shape.z /= factor;
                    shape.ct *= factor;
                    TT_ASSERT(coord.t < shape.z);
                    TT_ASSERT(coord.ct < shape.ct);
                }
                else  // hstack
                {
                    int factor = std::get<int>(tm.attr[0]);
                    TT_ASSERT(shape.ct % factor == 0);
                    shape.ct /= factor;
                    shape.z *= factor;
                    coord.t = coord.t * factor + coord.ct / shape.ct;
                    coord.ct %= shape.ct;
                    TT_ASSERT(coord.t < shape.z);
                    TT_ASSERT(coord.ct < shape.ct);
                }
                break;
            }
            case 't':  // transpose
            {
                if (tm.op[1] == 'i')  // tile_broadcast
                    break;
                std::swap(shape.rt, shape.ct);
                std::swap(coord.rt, coord.ct);
                break;
            }
            default:
            {
                TT_ASSERT(false, "Unhandled case map_inverse_tms");
                break;
            }
        }
    }

    return std::make_pair(coord, shape);
}

// Checks whether a pattern of length cycle_len exists in the vector of linear coordinates
//
bool is_pattern(std::vector<LinCoord> const& vec, int cycle_len)
{
    int vec_len = vec.size();
    if (vec_len < cycle_len)
    {
        return false;
    }

    for (int i = 0; i < vec_len - cycle_len; i++)
    {
        if (vec[i] != vec[i + cycle_len])
            return false;
    }

    return true;
}

// Calculates addresses of tiles of producer, that a single consumer core receives, and returns the length of the
// repeating pattern, if one exists. Otherwise returns 0.
//
int detect_repetitive_pattern(std::unordered_map<Pipe, int>* const kb_cache, Pipe const& pipe)
{
    TensorShape consumer_shape = pipe.consumer_layout.shape();

    const int block_volume = pipe.consumer_layout.block_shape.volume();
    const int block_volume_no_t = pipe.consumer_layout.block_shape.volume_no_t();

    log_trace(LogKernelBroadcast, "      block_volume: {:15}", block_volume);
    log_trace(LogKernelBroadcast, "      block_volume_no_t: {:10}", block_volume_no_t);

    // We can only check for the first core - if there is a pattern, it will be the same on all cores
    //
    constexpr int grid_r = 0;
    constexpr int grid_c = 0;

    // Structure to keep track of producer addresses that a single consumer core sees
    //
    std::vector<LinCoord> producer_addresses;
    producer_addresses.resize(block_volume);

    // Generate addresses of producer tiles that a single consumer core sees
    //
    for (int block_offset = 0; block_offset < block_volume; ++block_offset)
    {
        // Walk the consumer tile order linearly
        //
        LinCoord consumer_linear(grid_r, grid_c, block_offset);
        CanCoord consumer_coord = pipe.consumer_layout.map(consumer_linear);

        // Map consumer tile position to producer tile origin
        //
        auto [producer_coord, p_shape] = map_inverse_tms(consumer_coord, consumer_shape, pipe.tms);
        LinCoord producer_linear = pipe.producer_layout.map(producer_coord);

        producer_addresses[block_offset] = producer_linear;
    }

    // Check if the producer addresses are repetitive
    //
    // Lower bound: 1
    //   this bound can be improved, but there's several edge cases, let's do it only if compile time is an
    //   issue
    // Upper bound: block_volume_no_t
    //   no need to go above a single mblock
    //
    for (int pattern_candidate = 1; pattern_candidate <= block_volume_no_t; pattern_candidate++)
    {
        if (is_pattern(producer_addresses, pattern_candidate))
        {
            log_trace(LogKernelBroadcast, "      found pattern of len: {:10}", pattern_candidate);
            if (kb_cache)
            {
                kb_cache->insert({pipe, pattern_candidate});
            }
            return pattern_candidate;
        }
    }

    // We haven't found a pattern, add cache entry and return 0 length
    //
    log_trace(LogKernelBroadcast, "      pattern not found...");
    if (kb_cache)
    {
        kb_cache->insert({pipe, 0});
    }
    return 0;  // no pattern
}

inline int ordered(GridCoord coord, GridShape shape) { return shape.c * coord.r + coord.c; }

ResourceUsage get_edge_resource_usage(std::unordered_map<Pipe, ResourceUsage>& pipe_to_ru_cache, Pipe pipe)
{
    auto match = pipe_to_ru_cache.find(pipe);
    if (match != pipe_to_ru_cache.end())
    {
        return match->second;
    }

    ResourceUsage usage;

    struct ProducerPhase
    {
        LinCoord prev;
        int first_t_phases = 0;
        int first_repeat = 0;
        int contiguous = 1;
        int phases = 0;
    };

    TensorShape consumer_shape = pipe.consumer_layout.shape();
    LinCoord prev_producer_linear;
    SmallVector<std::uint64_t, 8> unique_consumer_grids;
    unique_consumer_grids.resize(pipe.producer_layout.grid_shape.volume());
    SmallVector<std::uint64_t, 8> unique_producer_grids;
    unique_producer_grids.resize(pipe.consumer_layout.grid_shape.volume());
    SmallVector<ProducerPhase, 8> producer_phases;
    producer_phases.resize(pipe.producer_layout.grid_shape.volume());

    GridShape consumer_grid = pipe.consumer_layout.grid_shape;
    int block_volume = pipe.consumer_layout.block_shape.volume();
    int producer_block_volume = pipe.producer_layout.block_shape.volume();

    bool monotonic_producer_ts = true;
    for (int grid_r = 0; grid_r < consumer_grid.r; ++grid_r)
    {
        for (int grid_c = 0; grid_c < consumer_grid.c; ++grid_c)
        {
            GridCoord consumer_grid_coord(grid_r, grid_c);
            int consumer_core_phases = 0;
            int first_t_consumer_core_phases = 0;
            int prev_producer_t = 0;
            for (int block_offset = 0; block_offset < block_volume; ++block_offset)
            {
                // Walk the consumer tile order linearly
                LinCoord consumer_linear(grid_r, grid_c, block_offset);
                CanCoord consumer_coord = pipe.consumer_layout.map(consumer_linear);
                // Map consumer tile position to producer tile origin
                auto [producer_coord, p_shape] = map_inverse_tms(consumer_coord, consumer_shape, pipe.tms);
                LinCoord producer_linear = pipe.producer_layout.map(producer_coord);
                // Check if this tile comes from the same grid coordinate, if not we need a new phase
                bool consumer_contiguous = prev_producer_linear.next().grid_coord() == producer_linear.grid_coord();
                consumer_core_phases += int(not consumer_contiguous);
                prev_producer_linear = producer_linear;

                // Check if we go backwards in t
                if (producer_coord.t == 0)
                    first_t_consumer_core_phases = consumer_core_phases;
                monotonic_producer_ts &= prev_producer_t <= producer_coord.t;
                prev_producer_t = producer_coord.t;

                int producer_grid_idx = ordered(producer_linear.grid_coord(), pipe.producer_layout.grid_shape);
                int consumer_grid_idx = ordered(consumer_grid_coord, pipe.consumer_layout.grid_shape);
                std::uint64_t& consumer_grid_mask = unique_consumer_grids[producer_grid_idx];
                if (consumer_grid_idx < 64)
                    consumer_grid_mask |= (1llu << std::uint64_t(consumer_grid_idx));
                else
                    consumer_grid_mask |= (consumer_grid_mask + 1llu);

                std::uint64_t& producer_grid_mask = unique_producer_grids[consumer_grid_idx];
                if (producer_grid_idx < 64)
                    producer_grid_mask |= (1llu << std::uint64_t(producer_grid_idx));
                else
                    producer_grid_mask |= (producer_grid_mask + 1llu);

                // Calculate producer phases
                ProducerPhase& producer_phase = producer_phases[producer_grid_idx];
                bool producer_contiguous = (producer_phase.prev.next() == producer_linear and consumer_contiguous);
                producer_phase.contiguous += int(producer_contiguous);
                producer_phase.phases += int(not producer_contiguous);
                if (not producer_phase.first_repeat and producer_phase.contiguous == producer_block_volume)
                    producer_phase.first_repeat = producer_phase.phases;
                if (producer_coord.t == 0)
                    producer_phase.first_t_phases = producer_phase.phases;
                producer_phase.prev = (not producer_phase.prev.valid() or producer_phase.prev.next() == producer_linear)
                                          ? producer_linear
                                          : LinCoord{};
            }

            // If the tile read order never went backwards in t, this can be turned into a loop
            if (monotonic_producer_ts)
                consumer_core_phases = first_t_consumer_core_phases;

            usage.consumer_phases = std::max(usage.consumer_phases, consumer_core_phases);
        }
    }

    for (std::uint64_t mask : unique_consumer_grids)
    {
        usage.producer_fan_out = std::max(usage.producer_fan_out, __builtin_popcountll(mask));
    }

    for (std::uint64_t mask : unique_producer_grids)
    {
        usage.consumer_fan_in = std::max(usage.consumer_fan_in, __builtin_popcountll(mask));
    }

    for (ProducerPhase producer_phase : producer_phases)
    {
        usage.producer_phases = std::max(
            usage.producer_phases,
            producer_phase.first_repeat ? producer_phase.first_repeat
            : monotonic_producer_ts     ? producer_phase.first_t_phases
                                        : producer_phase.phases);
    }

    usage.producer_phases *= std::min(2, pipe.producer_out_buf_mb);
    usage.consumer_phases *= std::min(2, pipe.producer_out_buf_mb);

    pipe_to_ru_cache.insert({pipe, usage});
    return usage;
}

ResourceUsage get_edge_resource_usage(
    Graph const* graph,
    std::unordered_map<Pipe, ResourceUsage>& pipe_to_ru_cache,
    graphlib::Edge edge,
    OpModel const& producer_op_model,
    OpModel const& consumer_op_model,
    bool is_queue)
{
    graphlib::Node const* producer_node = graph->node_by_id(edge.producer_node_id);
    graphlib::OpNode const* consumer_node =
        dynamic_cast<graphlib::OpNode const*>(graph->node_by_id(edge.consumer_node_id));

    auto edge_attr = graph->get_edge_attributes(edge);
    auto tms = edge_attr->get_tms();

    insert_t_stream_tms(
        consumer_node,
        tms,
        consumer_op_model.t_stream_factor,
        producer_op_model.t_stream_factor,
        edge.consumer_input_port_id,
        is_queue);

    GridShape producer_grid_shape = producer_op_model.grid_shape;
    BlockShape producer_block_shape = producer_op_model.output_buffers[0].block_shape.canonical();
    if (producer_op_model.fracture_factor > 1)
    {
        TT_ASSERT(producer_grid_shape.c % producer_op_model.fracture_factor == 0);
        producer_grid_shape.r *= producer_op_model.fracture_factor;
        producer_grid_shape.c /= producer_op_model.fracture_factor;
    }

    GridShape consumer_grid_shape = consumer_op_model.get_input_grid_shape(edge.consumer_input_port_id);
    BlockShape consumer_block_shape =
        consumer_op_model.input_buffers[edge.consumer_input_port_id].block_shape.canonical();
    if (consumer_node->is_matmul())
        consumer_grid_shape = edge.consumer_input_port_id == 0 ? GridShape(consumer_grid_shape.r, 1)
                                                               : GridShape(1, consumer_grid_shape.c);

    Pipe pipe(
        TileLayout(
            producer_grid_shape,
            producer_block_shape,
            get_output_ublock_order(graph, producer_node),
            producer_op_model.padding),
        producer_op_model.output_buffers[0].buffer_factor,
        tms,
        TileLayout(
            consumer_grid_shape, consumer_block_shape, edge_attr->get_ublock_order(), consumer_op_model.padding));

    try
    {
        return get_edge_resource_usage(pipe_to_ru_cache, pipe);
    }
    catch (...)
    {
        log_error("{} -> {}[{}]", producer_node->name(), consumer_node->name(), edge.consumer_input_port_id);
        log_error("producer {}", producer_op_model);
        log_error("consumer {}", consumer_op_model);
        log_error("Test\n{}", pipe);
        throw;
    }
}

ResourceUsage get_edge_resource_usage_simple(
    Graph const*,
    graphlib::Edge edge,
    OpModel const& producer_op_model,
    OpModel const& consumer_op_model,
    bool is_queue)
{
    ResourceUsage usage;

    if (is_queue)
    {
        auto producer_grid_shape = producer_op_model.grid_shape;
        bool matmul_lhs = consumer_op_model.buda_op_node->is_matmul() and edge.consumer_input_port_id == 0;
        bool matmul_rhs = consumer_op_model.buda_op_node->is_matmul() and edge.consumer_input_port_id == 1;
        int fork_factor_r =
            matmul_rhs ? producer_grid_shape.r : round_up_div(producer_grid_shape.r, consumer_op_model.grid_shape.r);
        int fork_factor_c =
            matmul_lhs ? producer_grid_shape.c : round_up_div(producer_grid_shape.c, consumer_op_model.grid_shape.c);
        usage.consumer_fan_in = fork_factor_r * fork_factor_c;
    }
    else
    {
        int producer_grid_volume = producer_op_model.grid_shape.volume();
        int consumer_grid_volume = 0;
        if (consumer_op_model.buda_op_node->is_matmul() and edge.consumer_input_port_id < 2)
        {
            consumer_grid_volume =
                (edge.consumer_input_port_id == 1) ? consumer_op_model.grid_shape.c : consumer_op_model.grid_shape.r;
        }
        else
        {
            consumer_grid_volume = consumer_op_model.grid_shape.volume();
        }
        usage.producer_fan_out = round_up_div(consumer_grid_volume, producer_grid_volume);
    }
    return usage;
}

std::tuple<uint32_t, uint32_t, uint32_t> get_sparse_matmul_metadata(balancer::OpModel const& op_model)
{
    int grid_r = op_model.grid_shape.r;
    int u_rt = op_model.output_buffers[0].block_shape.ublock.rt;
    int u_kt = op_model.input_buffers[1].block_shape.ublock.rt;
    int t_factor_c = op_model.t_stream_factor.c;
    int t_factor_r = op_model.t_stream_factor.r;
    const sparse::SparseBUDA& sparse_buda = *(op_model.sparse_buda);
    auto layout = sparse::SparseBUDA::create_layout(
        op_model.has_sparse_buffer() or env_as<bool>("PYBUDA_FORCE_SPARSE_BUFFER_LAYOUT"),
        op_model.t_stream_factor.dir.z_major(),
        op_model.fracture_factor);
    int bcast_factor = sparse_buda.bcast_factor;
    int zdim = sparse_buda.sparse_zs.size();

    // Initialize tiles/ublocks/strips counter
    int sum_nz_tiles = 0;
    int sum_nz_ublocks = 0;
    int sum_nz_strips = 0;
    constexpr int TILE_DIM = tt::sparse::TILE_DIM;

    struct CounterEntry
    {
        std::unordered_set<uint64_t> rt_ct_cmb;
        std::unordered_set<uint64_t> ubc_ubr_cmb;
        std::unordered_set<int> ubc_idxs;
        int smallest_rt;
        CounterEntry() : smallest_rt(INT_MAX){};
    };

    std::vector<CounterEntry> counters;
    int slice_count = grid_r * t_factor_r;

    // Iterate throufh all sparse tensors
    for (int z = 0; z < zdim; z++)
    {
        auto sparse = sparse_buda.sparse_zs[z];

        // Take stat of the sparseCOO
        int dflow_factor = (layout == sparse::SparseBUDA::Layout::ZMajorDataflow)
                               ? (sparse.rt() / grid_r / t_factor_r / bcast_factor)
                               : 1;
        int num_slices = (layout == tt::sparse::SparseBUDA::Layout::Default)
                             ? grid_r * t_factor_r
                             : grid_r * t_factor_r * bcast_factor * dflow_factor;
        std::int64_t slice_height = sparse.shape[0] / num_slices;

        std::vector<CounterEntry> ret(num_slices);
        for (size_t idx = 0; idx < sparse.rows.size(); idx++)
        {
            // Count nonzero tiles/ublocks/strips in the SparseCOO
            int ret_slice_idx = -1, rt = -1;
            if (layout == tt::sparse::SparseBUDA::Layout::Default)
            {
                ret_slice_idx = sparse.rows[idx] / slice_height;
                rt = (sparse.rows[idx] % slice_height) / TILE_DIM;
            }
            else if (layout == tt::sparse::SparseBUDA::Layout::ZMajor)
            {
                int slice_idx = sparse.rows[idx] / slice_height;
                int inner_idx = (slice_idx / slice_count) * grid_r + (slice_idx % grid_r);
                int slice_inner_idx = inner_idx % bcast_factor;
                ret_slice_idx = (slice_idx % (grid_r * t_factor_r)) / grid_r * grid_r + (inner_idx / bcast_factor);
                int new_rows = (sparse.rows[idx] % slice_height) + slice_height * slice_inner_idx;
                rt = new_rows / TILE_DIM;
            }
            else
            {
                TT_ASSERT(
                    layout == sparse::SparseBUDA::Layout::BufferOp or
                    layout == sparse::SparseBUDA::Layout::ZMajorDataflow);
                if (layout == sparse::SparseBUDA::Layout::ZMajorDataflow and
                    ((sparse.rt() / grid_r / t_factor_r) % bcast_factor != 0))
                    continue;

                int slice_idx = sparse.rows[idx] / slice_height;
                int inner_idx = (slice_idx % (dflow_factor * grid_r)) * bcast_factor +
                                (slice_idx / (dflow_factor * grid_r * t_factor_r));
                int slice_inner_idx = inner_idx % (bcast_factor * dflow_factor);
                ret_slice_idx = (slice_idx / (dflow_factor * grid_r)) % t_factor_r * grid_r +
                                (inner_idx / (bcast_factor * dflow_factor));
                int new_rows = (sparse.rows[idx] % slice_height) + slice_height * slice_inner_idx;
                rt = new_rows / TILE_DIM;
            }
            int ct = sparse.cols[idx] / TILE_DIM;
            int ubr_idx = rt / u_rt;
            int ubc_idx = ct / u_kt;
            uint64_t rt_ct_key = (uint64_t(rt) << 32) | (ct & 0x0FFFF);
            uint64_t ubc_ubr_key = (uint64_t(ubc_idx) << 32) | (ubr_idx & 0x0FFFF);

            // Add the metadata to counting struct
            CounterEntry& e = ret[ret_slice_idx];
            e.rt_ct_cmb.insert(rt_ct_key);
            e.ubc_ubr_cmb.insert(ubc_ubr_key);
            e.ubc_idxs.insert(ubc_idx);
            if (rt < ret[ret_slice_idx].smallest_rt)
                ret[ret_slice_idx].smallest_rt = rt;
        }

        // Count tiles, ublocks, strips
        for (int idx = 0; idx < slice_count; idx++)
        {
            const CounterEntry& e = ret[idx];
            sum_nz_tiles += e.rt_ct_cmb.size();
            sum_nz_ublocks += e.ubc_ubr_cmb.size();
            sum_nz_strips += e.ubc_idxs.size();
            if (e.smallest_rt >= 1 and e.smallest_rt < INT_MAX)
            {
                sum_nz_tiles++;
                sum_nz_ublocks++;
            }
        }
    }

    sum_nz_tiles *= t_factor_c;
    sum_nz_ublocks *= t_factor_c;
    sum_nz_strips *= t_factor_c;
    return std::make_tuple<>(sum_nz_tiles, sum_nz_ublocks, sum_nz_strips);
}

}  // namespace tt::balancer
