
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

class OpToOpConnectionModel;

// Class which models HW configuration of dram read pipe.
struct DramReadPipeHWConfiguration
{
    int dram_scatter_chunk_size_tiles;
    int dram_buf_read_chunk_size_tiles;
    int dram_receiving_stream_buffer_size_tiles;
};

//----------------------------------------------------------------------------------------------------------------------

// Returns bandwidth estimation for a given edge between two ops given by their models.
BandwidthBucket get_bandwidth_estimation(
    const Graph* graph,
    const Edge& edge,
    const OpModel& producer_op_model,
    const OpModel& consumer_op_model,
    bool is_queue,
    bool decompose_t_stream);

// Returns HW configuration of dram read pipe.
DramReadPipeHWConfiguration get_dram_read_pipe_hw_configuration(
    const Graph* graph,
    const Edge& edge,
    const OpModel& producer_op_model,
    const OpModel& consumer_op_model,
    const int dram_io_available_space_bytes);

DramReadPipeHWConfiguration get_dram_read_pipe_hw_configuration(
    const int scatter_granularity,
    const int kernel_clear_granularity,
    const int producer_tiles_per_input,
    const int consumer_tiles_per_input,
    const bool is_consumer_multicast,
    const int unpacker_buffer_size_bytes,
    const int tile_size,
    const int dram_io_available_space_bytes);

// Scales dram read BW based on the number of readers of the dram buffer.
float scale_dram_read_bandwidth_wrt_fork_factor(const float bw_without_fork, const float fork_factor);

// Returns total number fo tiles that flow through the input buffer of the consumer node perf input in a graph.
int get_op_input_tiles_per_input(const OpModel& op_model, const int input_index);

// Returns total number of tiles that flow through the input buffer of the consumer op in an entire epoch.
int get_op_input_epoch_tiles(const Graph* graph, const OpModel& op_model, const int input_index);

//----------------------------------------------------------------------------------------------------------------------

enum class InputType
{
    Eltwise,
    MatmulRow,
    MatmulColumn,
};

// Returns the type of the producer -> consumer edge.
InputType get_input_type(const Graph* graph, const Edge& edge);

// Helper functions to carve out necessary information from the graph and op models.

// Returns tms to be inserted for a given edge.
vector<OpType> get_tms_on_graph_edge(
    const Graph* graph,
    const Edge& edge,
    const OpModel& producer_op_model,
    const OpModel& consumer_op_model,
    bool is_queue,
    bool decompose_t_stream);

// Returns the tile layout for the producer and consumer ops.
TileLayout get_producer_tile_layout(const Graph* graph, const Edge& edge, const OpModel& producer_op_model);
TileLayout get_consumer_tile_layout(
    const Graph* graph, const Edge& edge, const OpModel& consumer_op_model, InputType input_type);

// Returns how many macroblocks a producer node buffers at output (not taking double-buffering into account).
int get_producer_out_buf_mb(bool is_queue, const BufferModel& buffer_model);

// Returns whether the producer's buffer is scatter or not.
bool is_scatter_producer(TileLayout producer_layout, int scatter_granularity, int producer_effective_buf_size_mb);

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
int calculate_buf_space_available_ack_thr(const int unpacker_buf_size_tiles, const int scatter_gather_num_tiles);

// Returns number of producer cores that send data to the consumer.
int get_consumer_fanin_from_tile_maps(
    const GridShape& producer_grid_shape, const consumer_to_producer_tile_map& consumer_tile_map);

// Returns tile map object for the given producer.
three_d_array_tile_src_map prepare_tile_map(
    const TileLayout& producer_layout,
    const int producer_out_buf_mb,
    const std::string& producer_name,
    const std::string& consumer_name);

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
        DramRead,
        Unknown
    };

    OpToOpConnectionModel(TileLayout&& producer_layout, TileLayout&& consumer_layout) :
        producer_layout_(std::move(producer_layout)), consumer_layout_(std::move(consumer_layout))
    {
    }

    // Implementation using tile maps lib from net2pipe.
    static OpToOpConnectionModel create_op_to_op_connection_model(
        TileLayout&& producer,
        TileLayout&& consumer,
        const Graph* graph,
        const Edge& edge,
        const int producer_out_buf_mb,
        const int kernel_broadcast_tiles,
        const int kernel_clear_granularity,
        const std::vector<OpType>& tms,
        const InputType input_type,
        const bool is_producer_queue,
        const bool is_multicast);

    ConnectionType get_connection_type() const;

    int get_scatter_granularity() const { return scatter_granularity_; }
    int get_producer_tiles_per_input() const { return producer_tiles_per_input_; }
    bool is_producer_scatter() const { return is_producer_scatter_; }
    int get_producer_num_phases() const { return producer_num_phases_; }
    int get_producer_fanout() const { return producer_fanout_; }
    bool is_producer_queue() const { return is_producer_queue_; }
    bool is_consumer_multicast() const { return consumer_multicast_; }
    int get_consumer_tiles_per_input() const { return consumer_tiles_per_input_; }
    int get_consumer_fanin() const { return consumer_fanin_; }
    int get_kernel_clear_granularity() const { return kernel_clear_granularity_; }
    const TileLayout& get_producer_tile_layout() const { return producer_layout_; }
    const TileLayout& get_consumer_tile_layout() const { return consumer_layout_; }

    void set_scatter_granularity(int val) { scatter_granularity_ = val; }
    void set_producer_tiles_per_input(int val) { producer_tiles_per_input_ = val; }
    void set_is_producer_scatter(bool val) { is_producer_scatter_ = val; }
    void set_producer_num_phases(int val) { producer_num_phases_ = val; }
    void set_producer_fanout(int val) { producer_fanout_ = val; }
    void set_is_producer_queue(bool val) { is_producer_queue_ = val; }
    void set_consumer_multicast(bool val) { consumer_multicast_ = val; }
    void set_consumer_tiles_per_input(int val) { consumer_tiles_per_input_ = val; }
    void set_consumer_fanin(int val) { consumer_fanin_ = val; }
    void set_kernel_clear_granularity(int val) { kernel_clear_granularity_ = val; }

   private:
    // Producer side fields.
    TileLayout producer_layout_;
    int scatter_granularity_;
    int producer_tiles_per_input_;
    int is_producer_scatter_;
    int producer_num_phases_;
    int producer_fanout_;
    bool is_producer_queue_;

    // Consumer side fields.
    TileLayout consumer_layout_;
    bool consumer_multicast_;
    int consumer_tiles_per_input_;
    int consumer_fanin_;
    int kernel_clear_granularity_;
};

// Creates a connection model between producer and consumer nodes.
OpToOpConnectionModel get_producer_consumer_connection_model(
    const Graph* graph,
    const Edge& edge,
    const OpModel& producer_op_model,
    const OpModel& consumer_op_model,
    bool is_queue,
    bool decompose_t_stream);

// Returns a consumer tile map w.r.t. to the given producer.
consumer_to_producer_tile_map get_consumer_tile_map(
    const TileLayout& producer,
    const TileLayout& consumer,
    const Graph* graph,
    const Edge& edge,
    const int producer_out_buf_mb,
    const int kernel_broadcast_tiles,
    const std::vector<graphlib::OpType>& tms,
    const InputType input_type);

//----------------------------------------------------------------------------------------------------------------------

// Estimator class that estimates bandwidth based on the given features.
class Estimator
{
   public:
    class Features
    {
       public:
        Features() {}

        static Features from_connection_model(const OpToOpConnectionModel& op_to_op_connection_model)
        {
            Features features;
            features.set_features(op_to_op_connection_model);

            return features;
        }

        void set_features(const OpToOpConnectionModel& op_to_op_connection_model)
        {
            scatter_gather_num_tiles_ = op_to_op_connection_model.get_scatter_granularity();
            producer_tiles_per_input_ = op_to_op_connection_model.get_producer_tiles_per_input();
            producer_num_phases_ = op_to_op_connection_model.get_producer_num_phases();
            producer_fan_out_ = op_to_op_connection_model.get_producer_fanout();
            is_producer_scatter_ = op_to_op_connection_model.is_producer_scatter();

            consumer_tiles_per_input_ = op_to_op_connection_model.get_consumer_tiles_per_input();
            consumer_fanin_ = op_to_op_connection_model.get_consumer_fanin();
            consumer_multicast_ = op_to_op_connection_model.is_consumer_multicast();
        }

        int get_scatter_gather_num_tiles() const { return scatter_gather_num_tiles_; }
        int get_producer_epoch_tiles() const { return producer_epoch_tiles_; }
        int get_producer_tiles_per_input() const { return producer_tiles_per_input_; }
        int get_producer_buffer_size_bytes() const { return producer_buffer_size_bytes_; }
        int get_producer_num_phases() const { return producer_num_phases_; }
        int get_producer_fan_out() const { return producer_fan_out_; }
        bool is_producer_scatter() const { return is_producer_scatter_; }
        int get_unpacker_buffer_size_bytes() const { return unpacker_buffer_size_bytes_; }
        int get_kernel_clear_granularity() const { return kernel_clear_granularity_; }
        int get_buf_space_available_ack_thr() const { return buf_space_available_ack_thr_; }
        int get_consumer_epoch_tiles() const { return consumer_epoch_tiles_; }
        int get_consumer_tiles_per_input() const { return consumer_tiles_per_input_; }
        int get_consumer_fanin() const { return consumer_fanin_; }
        bool is_consumer_multicast() const { return consumer_multicast_; }
        int get_tile_size() const { return tile_size_; }

        void set_scatter_gather_num_tiles(int val) { scatter_gather_num_tiles_ = val; }
        void set_producer_epoch_tiles(int val) { producer_epoch_tiles_ = val; }
        void set_producer_tiles_per_input(int val) { producer_tiles_per_input_ = val; }
        void set_producer_buffer_size_bytes(int val) { producer_buffer_size_bytes_ = val; }
        void set_producer_num_phases(int val) { producer_num_phases_ = val; }
        void set_producer_fan_out(int val) { producer_fan_out_ = val; }
        void set_producer_scatter(bool val) { is_producer_scatter_ = val; }
        void set_unpacker_buffer_size_bytes(int val) { unpacker_buffer_size_bytes_ = val; }
        void set_kernel_clear_granularity(int val) { kernel_clear_granularity_ = val; }
        void set_buf_space_available_ack_thr(int val) { buf_space_available_ack_thr_ = val; }
        void set_consumer_epoch_tiles(int val) { consumer_epoch_tiles_ = val; }
        void set_consumer_tiles_per_input(int val) { consumer_tiles_per_input_ = val; }
        void set_consumer_fanin(int val) { consumer_fanin_ = val; }
        void set_consumer_multicast(bool val) { consumer_multicast_ = val; }
        void set_tile_size(int val) { tile_size_ = val; }

       private:
        // Producer side features.
        int scatter_gather_num_tiles_ = 0;
        int producer_epoch_tiles_ = 0;
        int producer_tiles_per_input_ = 0;
        int producer_buffer_size_bytes_ = 0;
        int producer_num_phases_ = 0;
        int producer_fan_out_ = 0;
        bool is_producer_scatter_ = false;

        // Consumer side features.
        int unpacker_buffer_size_bytes_ = 0;
        int kernel_clear_granularity_ = 0;
        int buf_space_available_ack_thr_ = 0;
        int consumer_epoch_tiles_ = 0;
        int consumer_tiles_per_input_ = 0;
        int consumer_fanin_ = 0;
        bool consumer_multicast_ = false;

        // Common features.
        int tile_size_ = 0;
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

class DramReadEstimator : public Estimator
{
   public:
    DramReadEstimator(const Features& features) : Estimator(features) {}

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