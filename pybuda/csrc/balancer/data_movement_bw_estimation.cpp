// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "balancer/data_movement_bw_estimation.hpp"

#include <memory>
#include <stdexcept>
#include <vector>

#include "balancer/balancer_utils.hpp"
#include "balancer/bandwidth_bucket.hpp"
#include "balancer/bandwidth_estimator_impl.hpp"
#include "balancer/python_interface.hpp"
#include "balancer/types.hpp"
#include "passes/t_stream.hpp"
#include "utils/assert.hpp"

namespace tt
{
namespace balancer
{

TileLayout get_producer_tile_layout(const Graph* graph, const Edge& edge, const OpModel& producer_op_model)
{
    graphlib::Node const* producer_node = graph->node_by_id(edge.producer_node_id);
    GridShape producer_grid_shape = producer_op_model.grid_shape;
    BlockShape producer_block_shape = producer_op_model.output_buffers[0].block_shape.canonical();
    if (producer_op_model.fracture_factor > 1)
    {
        TT_ASSERT(producer_grid_shape.c % producer_op_model.fracture_factor == 0);
        producer_grid_shape.r *= producer_op_model.fracture_factor;
        producer_grid_shape.c /= producer_op_model.fracture_factor;
    }

    return TileLayout(producer_grid_shape, producer_block_shape, get_output_ublock_order(graph, producer_node));
}

TileLayout get_consumer_tile_layout(
    const Graph* graph, const Edge& edge, const OpModel& consumer_op_model, InputType input_type)
{
    GridShape consumer_grid_shape = consumer_op_model.grid_shape;
    BlockShape consumer_block_shape =
        consumer_op_model.input_buffers[edge.consumer_input_port_id].block_shape.canonical();

    // When consumer is matmul, it receives data on a column or row of cores and then multicasts along the other
    // dimension. We need to adjust the grid shape accordingly.
    if (input_type != InputType::Eltwise)
    {
        consumer_grid_shape = input_type == InputType::MatmulRow ? GridShape(consumer_grid_shape.r, 1)
                                                                 : GridShape(1, consumer_grid_shape.c);
    }

    return TileLayout(consumer_grid_shape, consumer_block_shape, graph->get_edge_attributes(edge)->get_ublock_order());
}

vector<OpType> insert_t_stream_tms_wrapper(
    const Graph* graph,
    const Edge& edge,
    const OpModel& producer_op_model,
    const OpModel& consumer_op_model,
    bool is_queue)
{
    graphlib::OpNode const* consumer_node = dynamic_cast<OpNode const*>(graph->node_by_id(edge.consumer_node_id));
    auto edge_attr = graph->get_edge_attributes(edge);
    vector<OpType> tms = edge_attr->get_tms();

    insert_t_stream_tms(
        consumer_node,
        tms,
        consumer_op_model.t_stream_factor,
        producer_op_model.t_stream_factor,
        edge.consumer_input_port_id,
        is_queue);
    return tms;
}

InputType get_input_type(const Graph* graph, const Edge& edge)
{
    graphlib::OpNode const* consumer_node = dynamic_cast<OpNode const*>(graph->node_by_id(edge.consumer_node_id));
    if (consumer_node->is_matmul())
    {
        return edge.consumer_input_port_id == 0 ? InputType::MatmulRow : InputType::MatmulColumn;
    }
    return InputType::Eltwise;
}

three_d_array_tile_src_map prepare_tile_map(const TileLayout& producer_layout, const int producer_out_buf_mb)
{
    return three_d_array_tile_src_map(
        "producer",
        "consumer",
        producer_layout.t(),
        producer_layout.block_shape.ublock.rt,
        producer_layout.block_shape.ublock.ct,
        producer_layout.block_shape.mblock_m,
        producer_layout.block_shape.mblock_n,
        producer_layout.grid_shape.r,
        producer_layout.grid_shape.c,
        producer_out_buf_mb / 2,
        producer_layout.ublock_order == graphlib::UBlockOrder::R);
}

void apply_pybuda_tms_to_tile_map(three_d_array_tile_src_map& tile_map, const std::vector<graphlib::OpType>& tms)
{
    for (graphlib::OpType const& tm : tms)
    {
        if (tm.op == "tile_broadcast")
        {
            continue;
        }

        if (tm.op == "broadcast")
        {
            int dim = std::get<int>(tm.attr[0]);
            std::string dims[] = {"w", "z", "r", "c"};
            tile_map = tile_map.apply_tm(dims[dim] + "_" + tm.op, {std::get<int>(tm.attr[1])});
        }
        else if (tm.op == "buda_pad")
        {
            int rt = std::get<int>(tm.attr[0]);
            int ct = std::get<int>(tm.attr[1]);
            tile_map = tile_map.pad(rt, ct);
        }
        else if (tm.op == "buda_unpad")
        {
            int rt = std::get<int>(tm.attr[0]);
            int ct = std::get<int>(tm.attr[1]);
            tile_map = tile_map.unpad(rt, ct);
        }
        else if (tm.op == "transpose")
        {
            tile_map = tile_map.apply_tm(tm.op, {});
        }
        else
        {
            tile_map = tile_map.apply_tm(tm.op, {std::get<int>(tm.attr[0])});
        }
    }
}

int get_consumer_fanin_from_tile_maps(
    const GridShape& producer_grid_shape, const consumer_to_producer_tile_map& consumer_tile_map)
{
    int consumer_fanin = 0;
    for (const auto& [_, pipe] : consumer_tile_map.pipes)
    {
        auto ordered = [&producer_grid_shape](int r, int c) { return producer_grid_shape.c * r + c; };
        unordered_set<int> producer_cores;
        for (const tile_to_core_index_map& producer_tile_coord : pipe.tile_map)
        {
            producer_cores.insert(ordered(producer_tile_coord.core_r, producer_tile_coord.core_c));
        }
        consumer_fanin = std::max(consumer_fanin, (int)producer_cores.size());
    }

    return consumer_fanin;
}

bool is_producer_scatter(TileLayout producer_layout, int scatter_granularity, int producer_effective_buf_size_mb)
{
    const int mblock_tiles = producer_layout.volume(false /* include_t */);
    const int buffered_tiles = mblock_tiles * producer_effective_buf_size_mb;

    return scatter_granularity != buffered_tiles;
}

int approximate_scatter_packer_num_phases(
    const int scatter_granularity, const int tiles_per_input, const int producer_effective_buf_size_mb)
{
    return tiles_per_input / scatter_granularity * producer_effective_buf_size_mb;
}

// This function copies the logic from:
// int Net2Pipe::get_op_kernel_input_tile_clear_granularity(tt_op_info op_info, int index)
int get_unpacker_kernel_clear_granularity(
    const graphlib::OpNode* consumer_op_node,
    const BlockShape& consumer_block_shape,
    const int input_ordinal,
    const int kernel_broadcast_tiles)
{
    if (kernel_broadcast_tiles > 0)
    {
        return kernel_broadcast_tiles;
    }

    const int ublock_size_tiles = consumer_block_shape.ublock.rt * consumer_block_shape.ublock.ct;

    if (consumer_op_node->is_sparse_matmul())
    {
        switch (input_ordinal)
        {
            case 0:
                // Special input for sparse matmul.
                return std::get<int>(consumer_op_node->buda_attrs().at("num_sparse_tiles"));
            case 1:
                // Regular matmul column input.
                return ublock_size_tiles * consumer_block_shape.mblock_n;
            case 2:
                // Special encoding input for sparse matmul.
                return 1;
            default: return ublock_size_tiles;
        }
    }

    if (consumer_op_node->is_matmul())
    {
        switch (input_ordinal)
        {
            case 0:
                // Matmul row input.
                return ublock_size_tiles * consumer_block_shape.mblock_m;
            case 1:
                // Matmul column input.
                return ublock_size_tiles * consumer_block_shape.mblock_n;
            default: return ublock_size_tiles;
        }
    }

    if (consumer_op_node->is_op_type("buffer"))
    {
        switch (input_ordinal)
        {
            case 0:
                // Special input for buffer op (taken over from net2pipe).
                return ublock_size_tiles * consumer_block_shape.mblock_n;
            default: return 1;
        }
    }

    // TODO handle embedding, tilizer ops.

    // Reduce, eltwise are covered by ublock_size_tiles.
    return ublock_size_tiles;
}

int calculate_unpacker_buffer_size_bytes(const int kernel_clear_granularity, const int tile_size_bytes)
{
    const int min_kernel_buffer_size = kernel_clear_granularity * tile_size_bytes;
    int min_double_buffer_size = 2 * min_kernel_buffer_size;

    constexpr int _32KB = 32 * 1024;
    if (min_double_buffer_size >= _32KB)
    {
        return min_double_buffer_size;
    }

    int num_increments = (_32KB - min_double_buffer_size + min_kernel_buffer_size - 1) / min_kernel_buffer_size;

    return min_double_buffer_size + num_increments * min_kernel_buffer_size;
}

int calculate_buf_space_available_ack_thr(const int unpacker_buf_size_tiles, const int tiles_per_input)
{
    switch (unpacker_buf_size_tiles / tiles_per_input)
    {
        case 0: return 0;
        case 1:
        case 2: return 1;
        case 3:
        case 4:
        case 5: return 2;
        default: return 3;
    }
}

OpToOpConnectionModel OpToOpConnectionModel::create_op_to_op_connection_model(
    const TileLayout& producer,
    const TileLayout& consumer,
    const int producer_out_buf_mb,
    const int kernel_broadcast_tiles,
    const std::vector<graphlib::OpType>& tms,
    const InputType input_type)
{
    three_d_array_tile_src_map tile_map = prepare_tile_map(producer, producer_out_buf_mb);

    apply_pybuda_tms_to_tile_map(tile_map, tms);

    consumer_to_producer_tile_map consumer_tile_map;
    if (input_type == InputType::Eltwise)
    {
        consumer_tile_map = tile_map.get_op_eltwise_input(
            kernel_broadcast_tiles,
            false,  // kernel_broadcast_per_t, ignore for now as it seems not to be used.
            consumer.t(),
            consumer.block_shape.ublock.rt,
            consumer.block_shape.ublock.ct,
            consumer.block_shape.mblock_m,
            consumer.block_shape.mblock_n,
            consumer.grid_shape.r,
            consumer.grid_shape.c,
            consumer.ublock_order == graphlib::UBlockOrder::R);
    }
    else if (input_type == InputType::MatmulRow)
    {
        consumer_tile_map = tile_map.get_op_matmul_row_input(
            kernel_broadcast_tiles,
            false,  // kernel_broadcast_per_t, ignore for now as it seems not to be used.
            consumer.t(),
            consumer.block_shape.ublock.rt,
            consumer.block_shape.ublock.ct,
            consumer.block_shape.mblock_m,
            consumer.block_shape.mblock_n,
            consumer.grid_shape.r,
            consumer.grid_shape.c);
    }
    else if (input_type == InputType::MatmulColumn)
    {
        consumer_tile_map = tile_map.get_op_matmul_col_input(
            kernel_broadcast_tiles,
            false,  // kernel_broadcast_per_t, ignore for now as it seems not to be used.
            consumer.t(),
            consumer.block_shape.ublock.rt,
            consumer.block_shape.ublock.ct,
            consumer.block_shape.mblock_m,
            consumer.block_shape.mblock_n,
            consumer.grid_shape.r,
            consumer.grid_shape.c);
    }

    OpToOpConnectionModel result;
    result.set_scatter_granularity(consumer_tile_map.scatter_granularity);
    result.set_tiles_per_input(consumer.block_shape.volume());
    result.set_consumer_multicast(false);
    result.set_producer_fanout(consumer_tile_map.max_producer_core_fan_out());
    result.set_consumer_fanin(get_consumer_fanin_from_tile_maps(producer.grid_shape, consumer_tile_map));

    const int producer_effective_buf_size_mb = producer_out_buf_mb > 1 ? producer_out_buf_mb / 2 : 1;
    const bool producer_scatter_packer =
        consumer_tile_map.producer_tiles_out_of_order ||
        is_producer_scatter(producer, consumer_tile_map.scatter_granularity, producer_effective_buf_size_mb);
    result.set_scatter_pack(producer_scatter_packer);

    // Approximate number of phases for the packer side. This is number of phases in a single firmware loop. This number
    // could be lower in reality if some phases are merged together.
    const int packer_num_phases =
        producer_scatter_packer
            ? approximate_scatter_packer_num_phases(
                  result.get_scatter_granularity(), producer.block_shape.volume_no_t(), producer_effective_buf_size_mb)
            : 1 /* legacy pack doesn't care much about phases */;
    result.set_packer_num_phases(packer_num_phases);

    return result;
}

OpToOpConnectionModel OpToOpConnectionModel::create_op_to_op_connection_model(
    const TileLayout& producer_layout,
    const TileLayout& consumer_layout,
    const int producer_out_buf_mb,
    const vector<graphlib::OpType>& tms)
{
    TensorShape consumer_shape = consumer_layout.shape();
    LinCoord prev_producer_linear = LinCoord(0, 0, 0);

    map<int, set<int>> producer_core_to_consumer_cores;
    map<int, set<int>> consumer_core_to_producer_cores;

    GridShape consumer_grid = consumer_layout.grid_shape;
    const int block_volume = consumer_layout.block_shape.volume();

    auto ordered = [](GridCoord coord, GridShape shape) { return shape.c * coord.r + coord.c; };

    // Initialize to max possible value.
    int scatter_granularity = block_volume;
    bool scatter_pack = false;

    int producer_effective_buf_size_mb = producer_out_buf_mb > 1 ? producer_out_buf_mb / 2 : 1;
    int producer_tmblock_multiplier = 1;

    // Number of tiles buffered for a single output (excluding double buffering).
    int producer_buffered_tiles_single_out =
        producer_layout.volume(false /* include_t */) / producer_layout.grid_shape.volume();

    if (producer_effective_buf_size_mb > producer_layout.t())
    {
        assert(producer_effective_buf_size_mb % producer_layout.t() == 0);
        producer_tmblock_multiplier = producer_effective_buf_size_mb / producer_layout.t();
        producer_buffered_tiles_single_out *= producer_tmblock_multiplier;
    }

    for (int grid_r = 0; grid_r < consumer_grid.r; ++grid_r)
    {
        for (int grid_c = 0; grid_c < consumer_grid.c; ++grid_c)
        {
            GridCoord consumer_grid_coord(grid_r, grid_c);

            // How many contiguous tiles are coming to the current consumer core.
            int cur_core_contiguous_tiles = 0;

            for (int block_offset = 0; block_offset < block_volume; ++block_offset)
            {
                // Walk the consumer tile order linearly
                LinCoord consumer_linear(grid_r, grid_c, block_offset);
                CanCoord consumer_coord = consumer_layout.map(consumer_linear);
                // Map consumer tile position to producer tile origin
                auto [producer_coord, p_shape] = map_inverse_tms(consumer_coord, consumer_shape, tms);
                LinCoord producer_linear = producer_layout.map(producer_coord);

                int producer_grid_idx = ordered(producer_linear.grid_coord(), producer_layout.grid_shape);
                int consumer_grid_idx = ordered(consumer_grid_coord, consumer_layout.grid_shape);

                producer_core_to_consumer_cores[producer_grid_idx].insert(consumer_grid_idx);
                consumer_core_to_producer_cores[consumer_grid_idx].insert(producer_grid_idx);

                // If the first block offset, we'll skip contiguous check.
                bool producer_tiles_contiguous = prev_producer_linear.next() == producer_linear;
                if (block_offset > 0 && cur_core_contiguous_tiles > 0 && !producer_tiles_contiguous)
                {
                    scatter_granularity = std::gcd(scatter_granularity, cur_core_contiguous_tiles);
                    scatter_granularity = std::gcd(scatter_granularity, producer_linear.address());
                    cur_core_contiguous_tiles = 0;
                }

                cur_core_contiguous_tiles++;
                prev_producer_linear = producer_linear;
            }

            scatter_granularity = std::gcd(scatter_granularity, cur_core_contiguous_tiles);

            if (scatter_granularity == block_volume)
            {
                // Full block can be sent in one go. We can extend the scatter granularity to the producer buffer size.
                scatter_granularity *= producer_tmblock_multiplier;
            }

            if (scatter_granularity < producer_buffered_tiles_single_out)
            {
                // This is not 100% accuracte, as in some cases we can have pipe-scattering where the scatter
                // granularity is equal to the mblock size. But then, scatter_pack should be true.
                scatter_pack = true;
            }
        }
    }

    int producer_fanout = 0;
    for (const auto& [_, cores] : producer_core_to_consumer_cores)
    {
        producer_fanout = std::max(producer_fanout, (int)cores.size());
    }

    int consumer_fanin = 0;
    for (const auto& [_, cores] : consumer_core_to_producer_cores)
    {
        consumer_fanin = std::max(consumer_fanin, (int)cores.size());
    }

    return OpToOpConnectionModel(
        scatter_granularity,
        block_volume,
        scatter_pack,
        approximate_scatter_packer_num_phases(
            scatter_granularity, producer_layout.block_shape.volume_no_t(), producer_effective_buf_size_mb),
        false, /* consumer_multicast */
        producer_fanout,
        consumer_fanin);
}

OpToOpConnectionModel::ConnectionType OpToOpConnectionModel::get_connection_type() const
{
    if (producer_fanout_ == 1 && consumer_fanin_ == 1)
    {
        return ConnectionType::DirectConnection;
    }
    else if (producer_fanout_ > 1 && consumer_fanin_ == 1)
    {
        return ConnectionType::ForkedProducer;
    }
    else if (producer_fanout_ == 1 && consumer_fanin_ > 1)
    {
        return ConnectionType::GatheredConsumer;
    }
    else if (producer_fanout_ > 1 && consumer_fanin_ > 1)
    {
        return ConnectionType::ForkAndGatherCombo;
    }
    else
    {
        return ConnectionType::Unknown;
    }
}

std::unique_ptr<Estimator> EstimatorFactory::get_estimator(
    OpToOpConnectionModel::ConnectionType connection_type, const Estimator::Features& features)
{
    switch (connection_type)
    {
        case OpToOpConnectionModel::ConnectionType::DirectConnection:
            return std::make_unique<DirectConnectionEstimator>(features);
        case OpToOpConnectionModel::ConnectionType::ForkedProducer:
            return std::make_unique<ForkedProducerEstimator>(features);
        case OpToOpConnectionModel::ConnectionType::GatheredConsumer:
            return std::make_unique<GatheredConsumerEstimator>(features);
        case OpToOpConnectionModel::ConnectionType::ForkAndGatherCombo:
            return std::make_unique<ForkAndGatherComboEstimator>(features);
        case OpToOpConnectionModel::ConnectionType::DramDirect: return std::make_unique<DramDirectEstimator>(features);
        case OpToOpConnectionModel::ConnectionType::Unknown: return std::make_unique<Estimator>(features);
        default: TT_ASSERT(false, "Unexpected connection type");
    }
    return nullptr;
}

//----------------------------------------------------------------------------------------------------------------------

BandwidthBucket DirectConnectionEstimator::estimate_bandwidth_impl() const
{
    return estimate_direct_connection(
        features_.get_unpacker_buffer_size_bytes(),
        features_.get_kernel_clear_granularity(),
        features_.get_buf_space_available_ack_thr(),
        features_.get_epoch_tiles(),
        features_.get_tile_size(),
        features_.get_packer_buffer_size_bytes(),
        features_.get_packer_scatter_gather_num_tiles(),
        features_.get_packer_num_phases(),
        features_.get_scatter_pack());
}

BandwidthBucket ForkedProducerEstimator::estimate_bandwidth_impl() const
{
    return estimate_forked_connection(
        features_.get_epoch_tiles(),
        features_.get_tile_size(),
        features_.get_packer_buffer_size_bytes(),
        features_.get_packer_scatter_gather_num_tiles(),
        features_.get_producer_fanout());
}

BandwidthBucket GatheredConsumerEstimator::estimate_bandwidth_impl() const
{
    return estimate_gather_connection(
        features_.get_epoch_tiles(),
        features_.get_tile_size(),
        features_.get_packer_scatter_gather_num_tiles(),
        features_.get_consumer_fanin());
}

BandwidthBucket ForkAndGatherComboEstimator::estimate_bandwidth_impl() const
{
    /*
    Forking and gathering is a combination of forked producer and gathered consumer. We can estimate the bandwidth by
    taking the gather bandwidth without fork and then apply following formula depending on the fork factor:
        fork factor: 2, f(x) = -0.033x^2 + 1.426x + -3.978
        fork factor: 3, f(x) = -0.030x^2 + 1.286x + -3.678
        fork factor: 4, f(x) = -0.033x^2 + 1.259x + -3.735
        fork factor: 5, f(x) = -0.026x^2 + 1.054x + -2.798
        fork factor: 6, f(x) = -0.026x^2 + 0.960x + -2.543
        fork factor: 7, f(x) = -0.021x^2 + 0.767x + -1.722
        fork factor: 8, f(x) = -0.017x^2 + 0.633x + -1.734

    This formula is approximation for the effect of fork factor on the bandwidth. The formula is derived from the data
    collected on tests with and without forking on the same gather cases.
    */
    EstimatorFactory factory;
    const BandwidthBucket gather_bandwidth_bucket =
        factory.get_estimator(OpToOpConnectionModel::ConnectionType::GatheredConsumer, features_)->estimate_bandwidth();

    if (gather_bandwidth_bucket.get_bucket_index() == BandwidthBucket::BucketIndex::k0to4)
    {
        // If the gather bandwidth is in the lowest bucket, we return the same bucket since forking can not improve the
        // bandwidth.
        return gather_bandwidth_bucket;
    }

    const double gather_bandwidth = gather_bandwidth_bucket.get_bandwidth();
    const int fork_factor = features_.get_producer_fanout();

    // implement formula from the above comment
    double bandwidth = 0;
    switch (fork_factor)
    {
        case 2: bandwidth = -0.033 * gather_bandwidth * gather_bandwidth + 1.426 * gather_bandwidth - 3.978; break;
        case 3: bandwidth = -0.030 * gather_bandwidth * gather_bandwidth + 1.286 * gather_bandwidth - 3.678; break;
        case 4: bandwidth = -0.033 * gather_bandwidth * gather_bandwidth + 1.259 * gather_bandwidth - 3.735; break;
        case 5: bandwidth = -0.026 * gather_bandwidth * gather_bandwidth + 1.054 * gather_bandwidth - 2.798; break;
        case 6: bandwidth = -0.026 * gather_bandwidth * gather_bandwidth + 0.960 * gather_bandwidth - 2.543; break;
        case 7: bandwidth = -0.021 * gather_bandwidth * gather_bandwidth + 0.767 * gather_bandwidth - 1.722; break;
        case 8: bandwidth = -0.017 * gather_bandwidth * gather_bandwidth + 0.633 * gather_bandwidth - 1.734; break;
        default: bandwidth = gather_bandwidth;
    }
    return BandwidthBucket(bandwidth);
}

BandwidthBucket DramDirectEstimator::estimate_bandwidth_impl() const
{
    return BandwidthBucket(BandwidthBucket::BucketIndex::k12to16);
}

//----------------------------------------------------------------------------------------------------------------------

BandwidthBucket get_bandwidth_estimation(
    const Graph* graph,
    const Edge& edge,
    const OpModel& producer_op_model,
    const OpModel& consumer_op_model,
    bool is_queue)
{
    vector<OpType> tms = insert_t_stream_tms_wrapper(graph, edge, producer_op_model, consumer_op_model, is_queue);
    InputType input_type = get_input_type(graph, edge);

    // Check multicast on the original grid shape.
    bool is_multicast = (input_type == InputType::MatmulRow && consumer_op_model.grid_shape.c > 1) ||
                        (input_type == InputType::MatmulColumn && consumer_op_model.grid_shape.r > 1);

    TileLayout producer_layout = get_producer_tile_layout(graph, edge, producer_op_model);
    TileLayout consumer_layout = get_consumer_tile_layout(graph, edge, consumer_op_model, input_type);

    DataFormat data_format = consumer_op_model.input_buffers[edge.consumer_input_port_id].data_format;

    const int tile_size_in_bytes = tile_size_bytes(data_format);

    const int producer_out_buf_mb = producer_op_model.output_buffers[0].buffer_factor;

    const int kernel_broadcast_tiles =
        consumer_op_model.input_buffers[edge.consumer_input_port_id].kernel_broadcast_tiles;

    const int kernel_clear_granularity = get_unpacker_kernel_clear_granularity(
        dynamic_cast<OpNode const*>(graph->node_by_id(edge.consumer_node_id)),
        consumer_op_model.block_shape(),
        edge.consumer_input_port_id,
        kernel_broadcast_tiles);

    const int unpacker_buffer_size_bytes =
        calculate_unpacker_buffer_size_bytes(kernel_clear_granularity, tile_size_in_bytes);

    OpToOpConnectionModel op_to_op_connection_model;

    // There are cases where the connection model cannot be created, e.g. when the producer and consumer shapes are
    // incompatible. In such cases, we return the lowest bandwidth bucket.
    // TODO: check why these models even exist after legalizer.
    try
    {
        op_to_op_connection_model = OpToOpConnectionModel::create_op_to_op_connection_model(
            producer_layout, consumer_layout, producer_out_buf_mb, kernel_broadcast_tiles, tms, input_type);
    }
    catch (const std::runtime_error& e)
    {
        return BandwidthBucket(BandwidthBucket::BucketIndex::k0to4);
    }

    EstimatorFactory factory;
    Estimator::Features features;

    // TODO move multicast feature to OpToOpConnectionModel.
    features.set_features(op_to_op_connection_model);
    features.set_consumer_multicast(is_multicast);

    features.set_unpacker_buffer_size_bytes(unpacker_buffer_size_bytes);
    features.set_kernel_clear_granularity(kernel_clear_granularity);
    features.set_buf_space_available_ack_thr(calculate_buf_space_available_ack_thr(
        unpacker_buffer_size_bytes / tile_size_in_bytes, features.get_tiles_per_input()));

    features.set_epoch_tiles(features.get_tiles_per_input() * graph->get_microbatch());
    features.set_tile_size(tile_size_in_bytes);
    features.set_packer_buffer_size_bytes(
        producer_layout.block_shape.buffer_tiles(producer_out_buf_mb) * tile_size_in_bytes);

    std::unique_ptr<Estimator> estimator =
        factory.get_estimator(op_to_op_connection_model.get_connection_type(), features);

    return estimator->estimate_bandwidth();
}

//----------------------------------------------------------------------------------------------------------------------

}  // namespace balancer
}  // namespace tt
