
// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <vector>

#include "autograd/binding.hpp"
#include "balancer/bandwidth_bucket.hpp"
#include "balancer/types.hpp"
#include "graph_lib/graph.hpp"
#include "graph_lib/node_types.hpp"
#include "third_party/budabackend/src/net2pipe/inc/tile_maps.h"

using Graph = tt::graphlib::Graph;
using Node = tt::graphlib::Node;
using Edge = tt::graphlib::Edge;
using OpType = tt::graphlib::OpType;

namespace tt
{
namespace balancer
{

//----------------------------------------------------------------------------------------------------------------------

// Returns bandwidth estimation for a given edge between two ops given by their models.
BandwidthBucket get_bandwidth_estimation(
    const Graph* graph,
    const Edge& edge,
    const OpModel& producer_op_model,
    const OpModel& consumer_op_model,
    bool is_queue);

//----------------------------------------------------------------------------------------------------------------------

enum class InputType
{
    Eltwise,
    MatmulRow,
    MatmulColumn,
};

InputType get_input_type(const Graph* graph, const Edge& edge);

// Helper functions to carve out necessary information from the graph and op models.

// Returns tms to be inserted for a given edge.
vector<OpType> insert_t_stream_tms_wrapper(
    const Graph* graph,
    const Edge& edge,
    const OpModel& producer_op_model,
    const OpModel& consumer_op_model,
    bool is_queue);

// Returns the tile layout for the producer and consumer ops.
TileLayout get_producer_tile_layout(const Graph* graph, const Edge& edge, const OpModel& producer_op_model);
TileLayout get_consumer_tile_layout(
    const Graph* graph, const Edge& edge, const OpModel& consumer_op_model, InputType input_type);

// Returns whether the producer's buffer is scatter or not.
bool is_producer_scatter(TileLayout producer_layout, int scatter_granularity, int producer_effective_buf_size_mb);

// Returns the expected number of phases for the scatter packer.
int approximate_scatter_packer_num_phases(
    const int scatter_granularity, const int tiles_per_input, const int producer_effective_buf_size_mb);

// Returns the number of tiles consumer kernel will clear at the time.
int get_unpacker_kernel_clear_granularity(
    const graphlib::OpNode* consumer_op_node,
    const BlockShape& consumer_block_shape,
    const int input_ordinal,
    const int kernel_broadcast_tiles);

// Returns consumer's unpacker buffer size given the kernel clear granularity and tile size.
int calculate_unpacker_buffer_size_bytes(const int kernel_clear_granularity, const int tile_size_bytes);

// Returns ack threshold for the consumer side streams. This is a HW parameter, but is needed for bandwidth estimation.
int calculate_buf_space_available_ack_thr(const int unpacker_buf_size_tiles, const int tiles_per_input);

// Returns number of producer cores that send data to the consumer.
int get_consumer_fanin_from_tile_maps(
    const GridShape& producer_grid_shape, const consumer_to_producer_tile_map& consumer_tile_map);

// Returns tile map object for the given producer.
three_d_array_tile_src_map prepare_tile_map(const TileLayout& producer_tile_layout, const int producer_out_buf_mb);

// Applies the tms to the tile map.
void apply_pybuda_tms_to_tile_map(three_d_array_tile_src_map& tile_map, const std::vector<OpType>& tms);

//----------------------------------------------------------------------------------------------------------------------

// Model that dictates type of connection in HW between two ops.
class OpToOpConnectionModel
{
   public:
    enum class ConnectionType
    {
        DirectConnection = 0,
        ForkedProducer,
        GatheredConsumer,
        ForkAndGatherCombo,
        DramDirect,
        DramGather,
        Unknown
    };

    OpToOpConnectionModel() {}

    // Implementation using tile maps lib from net2pipe.
    static OpToOpConnectionModel create_op_to_op_connection_model(
        const TileLayout& producer,
        const TileLayout& consumer,
        const int producer_out_buf_mb,
        const int kernel_broadcast_tiles,
        const std::vector<OpType>& tms,
        const InputType input_type);

    // Implementation using tile layouts only.
    static OpToOpConnectionModel create_op_to_op_connection_model(
        const TileLayout& producer_layout,
        const TileLayout& consumer_layout,
        const int producer_out_buf_mb,
        const vector<OpType>& tms);

    OpToOpConnectionModel(
        int scatter_granularity,
        int tiles_per_input,
        int scatter_pack,
        int packer_num_phases,
        bool consumer_multicast,
        int producer_fanout,
        int consumer_fanin) :
        scatter_granularity_(scatter_granularity),
        tiles_per_input_(tiles_per_input),
        scatter_pack_(scatter_pack),
        packer_num_phases_(packer_num_phases),
        consumer_multicast_(consumer_multicast),
        producer_fanout_(producer_fanout),
        consumer_fanin_(consumer_fanin)
    {
    }

    ConnectionType get_connection_type() const;

    int get_scatter_granularity() const { return scatter_granularity_; }
    int get_tiles_per_input() const { return tiles_per_input_; }
    int get_scatter_pack() const { return scatter_pack_; }
    int get_packer_num_phases() const { return packer_num_phases_; }
    bool get_consumer_multicast() const { return consumer_multicast_; }
    int get_producer_fanout() const { return producer_fanout_; }
    int get_consumer_fanin() const { return consumer_fanin_; }

    void set_scatter_granularity(int scatter_granularity) { scatter_granularity_ = scatter_granularity; }
    void set_tiles_per_input(int tiles_per_input) { tiles_per_input_ = tiles_per_input; }
    void set_scatter_pack(int scatter_pack) { scatter_pack_ = scatter_pack; }
    void set_packer_num_phases(int packer_num_phases) { packer_num_phases_ = packer_num_phases; }
    void set_consumer_multicast(bool consumer_multicast) { consumer_multicast_ = consumer_multicast; }
    void set_producer_fanout(int producer_fanout) { producer_fanout_ = producer_fanout; }
    void set_consumer_fanin(int consumer_fanin) { consumer_fanin_ = consumer_fanin; }

   private:
    int scatter_granularity_;
    int tiles_per_input_;
    int scatter_pack_;
    int packer_num_phases_;
    bool consumer_multicast_;

    int producer_fanout_;
    int consumer_fanin_;
};

//----------------------------------------------------------------------------------------------------------------------

// Estimator class that estimates bandwidth based on the given features.
class Estimator
{
   public:
    class Features
    {
       public:
        void set_unpacker_buffer_size_bytes(int unpacker_buffer_size_bytes)
        {
            unpacker_buffer_size_bytes_ = unpacker_buffer_size_bytes;
        }
        void set_kernel_clear_granularity(int kernel_clear_granularity)
        {
            kernel_clear_granularity_ = kernel_clear_granularity;
        }
        void set_buf_space_available_ack_thr(int buf_space_available_ack_thr)
        {
            buf_space_available_ack_thr_ = buf_space_available_ack_thr;
        }
        void set_epoch_tiles(int epoch_tiles) { epoch_tiles_ = epoch_tiles; }
        void set_tiles_per_input(int tiles_per_input) { tiles_per_input_ = tiles_per_input; }
        void set_tile_size(int tile_size) { tile_size_ = tile_size; }
        void set_packer_buffer_size_bytes(int packer_buffer_size_bytes)
        {
            packer_buffer_size_bytes_ = packer_buffer_size_bytes;
        }
        void set_packer_scatter_gather_num_tiles(int packer_scatter_gather_num_tiles)
        {
            packer_scatter_gather_num_tiles_ = packer_scatter_gather_num_tiles;
        }
        void set_packer_num_phases(int packer_num_phases) { packer_num_phases_ = packer_num_phases; }
        void set_scatter_pack(bool scatter_pack) { scatter_pack_ = scatter_pack; }
        void set_producer_fanout(int producer_fanout) { producer_fanout_ = producer_fanout; }
        void set_consumer_fanin(int consumer_fanin) { consumer_fanin_ = consumer_fanin; }
        void set_consumer_multicast(bool consumer_multicast) { consumer_multicast_ = consumer_multicast; }

        int get_unpacker_buffer_size_bytes() const { return unpacker_buffer_size_bytes_; }
        int get_kernel_clear_granularity() const { return kernel_clear_granularity_; }
        int get_buf_space_available_ack_thr() const { return buf_space_available_ack_thr_; }
        int get_epoch_tiles() const { return epoch_tiles_; }
        int get_tiles_per_input() const { return tiles_per_input_; }
        int get_tile_size() const { return tile_size_; }
        int get_packer_buffer_size_bytes() const { return packer_buffer_size_bytes_; }
        int get_packer_scatter_gather_num_tiles() const { return packer_scatter_gather_num_tiles_; }
        int get_packer_num_phases() const { return packer_num_phases_; }
        bool get_scatter_pack() const { return scatter_pack_; }
        int get_producer_fanout() const { return producer_fanout_; }
        int get_consumer_fanin() const { return consumer_fanin_; }
        bool get_consumer_multicast() const { return consumer_multicast_; }

        Features() :
            unpacker_buffer_size_bytes_(0),
            kernel_clear_granularity_(0),
            buf_space_available_ack_thr_(0),
            epoch_tiles_(0),
            tiles_per_input_(0),
            tile_size_(0),
            packer_buffer_size_bytes_(0),
            packer_scatter_gather_num_tiles_(0),
            packer_num_phases_(0),
            scatter_pack_(false),
            producer_fanout_(0),
            consumer_fanin_(0),
            consumer_multicast_(false)
        {
        }

        Features(
            int unpacker_buffer_size_bytes,
            int kernel_clear_granularity,
            int buf_space_available_ack_thr,
            int epoch_tiles,
            int tiles_per_input,
            int tile_size,
            int packer_buffer_size_bytes,
            int packer_scatter_gather_num_tiles,
            int packer_num_phases,
            bool scatter_pack,
            int producer_fanout,
            int consumer_fanin,
            bool consumer_multicast) :
            unpacker_buffer_size_bytes_(unpacker_buffer_size_bytes),
            kernel_clear_granularity_(kernel_clear_granularity),
            buf_space_available_ack_thr_(buf_space_available_ack_thr),
            epoch_tiles_(epoch_tiles),
            tiles_per_input_(tiles_per_input),
            tile_size_(tile_size),
            packer_buffer_size_bytes_(packer_buffer_size_bytes),
            packer_scatter_gather_num_tiles_(packer_scatter_gather_num_tiles),
            packer_num_phases_(packer_num_phases),
            scatter_pack_(scatter_pack),
            producer_fanout_(producer_fanout),
            consumer_fanin_(consumer_fanin),
            consumer_multicast_(consumer_multicast)
        {
        }

        void set_features(const OpToOpConnectionModel& op_to_op_connection_model)
        {
            packer_scatter_gather_num_tiles_ = op_to_op_connection_model.get_scatter_granularity();
            scatter_pack_ = op_to_op_connection_model.get_scatter_pack();
            producer_fanout_ = op_to_op_connection_model.get_producer_fanout();
            consumer_fanin_ = op_to_op_connection_model.get_consumer_fanin();
            consumer_multicast_ = op_to_op_connection_model.get_consumer_multicast();
            tiles_per_input_ = op_to_op_connection_model.get_tiles_per_input();
            packer_num_phases_ = op_to_op_connection_model.get_packer_num_phases();
        }

        void set_features()
        {
            // not set yet
            // unpacker_buffer_size_bytes_ = 0; << unpacker buf size, derived from kernel clear granularity
            // kernel_clear_granularity_ = 0;   << derived from op model and op type
            // buf_space_available_ack_thr_ = 0; << derived from tiles per input and buf size
            // epoch_tiles_ = 0;                << derived from microbatch size and tiles per input
            // tile_size_ = 0;                  << derived from op model data format
            // packer_buffer_size_bytes_ = 0;   << derived from producer tile layout and buf out mb
            // packer_num_phases_ = 0;          << derived from packer scatter gather num tiles and tile map, tile_map
            // has a function to return this but complexity is high potentially
        }

       private:
        int unpacker_buffer_size_bytes_;
        int kernel_clear_granularity_;
        int buf_space_available_ack_thr_;
        int epoch_tiles_;
        int tiles_per_input_;
        int tile_size_;
        int packer_buffer_size_bytes_;
        int packer_scatter_gather_num_tiles_;
        int packer_num_phases_;
        bool scatter_pack_;
        int producer_fanout_;
        int consumer_fanin_;
        bool consumer_multicast_;
    };

    Estimator(const Features& features) : features_(features) {}

    virtual ~Estimator() = default;

    BandwidthBucket estimate_bandwidth() const { return estimate_bandwidth_impl(); }

   protected:
    virtual BandwidthBucket estimate_bandwidth_impl() const
    {
        return BandwidthBucket(BandwidthBucket::BucketIndex::k12to16);
    }

    Features features_;
};

class DirectConnectionEstimator : public Estimator
{
   public:
    DirectConnectionEstimator(const Features& features) : Estimator(features) {}

   private:
    BandwidthBucket estimate_bandwidth_impl() const override;
};

class ForkedProducerEstimator : public Estimator
{
   public:
    ForkedProducerEstimator(const Features& features) : Estimator(features) {}

   private:
    BandwidthBucket estimate_bandwidth_impl() const override;
};

class GatheredConsumerEstimator : public Estimator
{
   public:
    GatheredConsumerEstimator(const Features& features) : Estimator(features) {}

   private:
    BandwidthBucket estimate_bandwidth_impl() const override;
};

class ForkAndGatherComboEstimator : public Estimator
{
   public:
    ForkAndGatherComboEstimator(const Features& features) : Estimator(features) {}

   private:
    BandwidthBucket estimate_bandwidth_impl() const override;
};

class DramDirectEstimator : public Estimator
{
   public:
    DramDirectEstimator(const Features& features) : Estimator(features) {}

   private:
    BandwidthBucket estimate_bandwidth_impl() const override;
};

//----------------------------------------------------------------------------------------------------------------------

class EstimatorFactory
{
   public:
    static std::unique_ptr<Estimator> get_estimator(
        OpToOpConnectionModel::ConnectionType connection_type, const Estimator::Features& features);
};

//----------------------------------------------------------------------------------------------------------------------

}  // namespace balancer
}  // namespace tt