// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <initializer_list>
#include <map>
#include <memory>
#include <ostream>
#include <set>
#include <vector>

#include "lower_to_buda/common.hpp"
#include "output_host_tm_types.hpp"
#include "placer/placer.hpp"
#include "shared_utils/sparse_matmul_utils.hpp"

namespace tt::graphlib
{
class Graph;
class Node;
class Shape;
struct Edge;
class BudaOpNode;
enum class EdgeType;
using EdgeUniqueId = std::tuple<NodeId, PortId, NodeId, PortId, EdgeType>;
}  // namespace tt::graphlib

namespace tt
{
class FusedOp;
}

namespace tt::balancer
{
struct UniqueId
{
    static std::uint64_t next_id;
    std::uint64_t id = 0;
    UniqueId() : id(next_id++) {}
    UniqueId(std::uint64_t id) : id(id) {}
    bool operator==(UniqueId other) const { return id == other.id; };
};

struct TensorShape
{
    int w = 0;
    int z = 0;
    int rt = 0;
    int ct = 0;

    TensorShape() = default;
    TensorShape(int w, int z, int rt, int ct) : w(w), z(z), rt(rt), ct(ct) {}
    TensorShape(graphlib::Shape const &shape);
    inline int volume_in_tiles() const { return w * z * rt * ct; }
    inline bool operator==(TensorShape o) const { return w == o.w and z == o.z and rt == o.rt and ct == o.ct; }
    inline bool operator!=(TensorShape o) const { return not(*this == o); }
    int const &operator[](int index) const;
    int &operator[](int index);
};

struct OpShape
{
    std::vector<TensorShape> producer_shapes;
    std::vector<TensorShape> inputs;
    std::vector<TensorShape> outputs;

    OpShape() = default;
    OpShape(
        std::vector<TensorShape> const &producer_shapes,
        std::vector<TensorShape> const &inputs,
        std::vector<TensorShape> const &outputs) :
        producer_shapes(producer_shapes), inputs(inputs), outputs(outputs)
    {
    }
};

struct Parallelization
{
    int r = 0;
    int c = 0;

    Parallelization() = default;
    Parallelization(int r, int c) : r(r), c(c) {}

    inline bool operator==(Parallelization o) const { return (r == o.r) and (c == o.c); }
    inline bool operator!=(Parallelization o) const { return not(*this == o); }
    inline int volume() const { return r * c; }
    static Parallelization from_array(std::array<int, 2> array) { return Parallelization(array[0], array[1]); }
};

struct GridShape
{
    int r = 0;
    int c = 0;

    GridShape() = default;
    GridShape(int r, int c) : r(r), c(c) {}
    GridShape(Parallelization p) : r(p.r), c(p.c) {}

    inline bool operator==(GridShape o) const { return (r == o.r) and (c == o.c); }
    inline bool operator!=(GridShape o) const { return not(*this == o); }
    inline bool square() const { return r == c; }
    inline int volume() const { return r * c; }
    inline GridShape transposed() const { return GridShape(c, r); }
    static GridShape from_array(std::array<int, 2> array) { return GridShape(array[0], array[1]); }
};

struct UBlockShape
{
    int rt = 0;
    int ct = 0;

    UBlockShape() = default;
    UBlockShape(int rt, int ct) : rt(rt), ct(ct) {}
    UBlockShape(std::pair<int, int> shape) : rt(shape.first), ct(shape.second) {}
    inline bool operator==(UBlockShape o) const { return (rt == o.rt) and (ct == o.ct); }
    inline bool operator!=(UBlockShape o) const { return !(*this == o); }
    inline int volume() const { return rt * ct; }
};

struct BlockShape
{
    int t = 0;
    int tblock_m = 0;
    int tblock_n = 0;
    int mblock_m = 0;
    int mblock_n = 0;
    UBlockShape ublock;

    BlockShape() = default;
    BlockShape(TensorShape tensor_shape, int par_r, int par_c, int par_t, UBlockShape ublock);
    BlockShape(TensorShape tensor_shape, GridShape grid_shape, int mblock_m, int mblock_n, UBlockShape ublock);
    BlockShape(int t, int mblock_m, int mblock_n, UBlockShape ublock);
    bool operator==(BlockShape o) const;
    bool operator!=(BlockShape o) const;
    inline int m() const { return tblock_m * mblock_m; }
    inline int n() const { return tblock_n * mblock_n; }
    inline int rt() const { return tblock_m * mblock_m * ublock.rt; }
    inline int ct() const { return tblock_n * mblock_n * ublock.ct; }
    inline int buffered_rt() const { return mblock_m * ublock.rt; }
    inline int buffered_ct() const { return mblock_n * ublock.ct; }
    int volume() const;
    int volume_no_t() const;
    int buffer_tiles(int buffer_factor = 2) const;
    void set_ublock_shape(UBlockShape new_ublock);
    BlockShape canonical() const { return BlockShape(t, mblock_m * tblock_m, mblock_n * tblock_n, ublock); }
    BudaBlocks as_buda_blocks() const;
};

struct TStreamDir
{
    enum Value
    {
        R,
        C,
        RZ,
        CZ,
    } v;

    TStreamDir(Value v) : v(v) {}
    static TStreamDir Transposed(TStreamDir o)
    {
        Value v;
        switch (o.v)
        {
            case R: v = C; break;
            case C: v = R; break;
            case RZ: v = CZ; break;
            case CZ: v = RZ; break;
            default: v = R; break;
        }
        return TStreamDir(v);
    }

    inline bool operator==(TStreamDir o) const { return v == o.v; }
    inline bool operator!=(TStreamDir o) const { return not(*this == o); }
    inline bool primary_dir_compatible(TStreamDir o) const { return (r() == o.r()) and (c() == o.c()); }
    inline bool is_ublock_order(graphlib::UBlockOrder ublock_order) const
    {
        return (r() == (ublock_order == graphlib::UBlockOrder::R)) and
               (c() == (ublock_order == graphlib::UBlockOrder::C));
    }
    inline bool r() const { return v == R or v == RZ; }
    inline bool c() const { return v == C or v == CZ; }
    inline bool z_major() const { return v == RZ or v == CZ; }
};

struct TStreamFactor
{
    TStreamDir dir = TStreamDir::R;
    int r = 1;
    int c = 1;

    TStreamFactor() = default;
    TStreamFactor(TStreamDir dir, Parallelization p) : dir(dir), r(p.r), c(p.c) {}
    TStreamFactor(TStreamDir dir, int r, int c) : dir(dir), r(r), c(c) {}
    static TStreamFactor Transposed(TStreamFactor o) { return TStreamFactor(TStreamDir::Transposed(o.dir), o.c, o.r); }
    inline int t() const { return r * c; }
    inline bool none() const { return r == 1 and c == 1; }
    inline bool is_streaming() const { return not none(); }
    inline bool is_streaming_r() const { return is_streaming() and dir.r(); }
    inline bool is_streaming_c() const { return is_streaming() and dir.c(); }
    inline bool operator==(TStreamFactor o) const { return dir == o.dir and r == o.r and c == o.c; }
    inline bool operator!=(TStreamFactor o) const { return not(*this == o); }
    inline bool compatible_consumer(TStreamFactor consumer, bool is_sparse_mm, bool consumes_rz_major) const
    {
        bool allowed_none = (none() and not consumes_rz_major) or (not dir.z_major() and consumer.none());
        return allowed_none or
               (dir.primary_dir_compatible(consumer.dir) and (r == consumer.r or is_sparse_mm) and c == consumer.c);
    }
};

struct BufferModel
{
    BlockShape block_shape;
    int buffer_factor = 0;
    std::size_t l1_size_tiles = 0;
    bool size_tiles_override = false;  // set to pass l1_size_tiles to netlist
    DataFormat data_format;
    bool minimize_input_buffer = false;  // only buffer 2 ublocks for matmul
    int kernel_broadcast_tiles = 0;

    BufferModel() = default;
    BufferModel(BlockShape block_shape, int buffer_factor, DataFormat data_format, bool size_tiles_override = false) :
        block_shape(block_shape),
        buffer_factor(buffer_factor),
        l1_size_tiles(block_shape.buffer_tiles(buffer_factor)),
        size_tiles_override(size_tiles_override),
        data_format(data_format)
    {
    }
    std::size_t size_tiles(bool include_t = false) const;
    std::size_t size_bytes(bool include_t = false) const;
    std::size_t single_buffered_size_tiles() const;
    std::size_t single_buffered_size_bytes() const;
    std::size_t total_size_bytes() const;
    inline bool is_unrolled() const { return size_tiles_override; }
    operator bool() const { return buffer_factor > 0; }
    bool operator==(BufferModel const &other) const
    {
        return block_shape == other.block_shape
        and buffer_factor == other.buffer_factor
        and l1_size_tiles == other.l1_size_tiles
        and data_format == other.data_format
        and size_tiles_override == other.size_tiles_override
        and minimize_input_buffer == other.minimize_input_buffer
        and kernel_broadcast_tiles == other.kernel_broadcast_tiles;
    }
};

struct Padding
{
    int rt = 0;
    int ct = 0;

    Padding() = default;
    Padding(int rt, int ct) : rt(rt), ct(ct) {}
    bool operator==(const Padding &p) const { return rt == p.rt and ct == p.ct; }
};

// Do not add new fields to OpModel as it is very perf sensitive structure.
// In case you really need to add something talk to nsmith/nobradovic.
//
struct OpModel
{
    struct SparseMetadata {
        std::vector<int> nz_tiles;
        std::vector<int> nz_ublocks;
        std::vector<int> nz_strips;

        bool operator==(SparseMetadata const &other) const
        {
            return nz_tiles == other.nz_tiles and nz_ublocks == other.nz_ublocks and nz_strips == other.nz_strips;
        }

        SparseMetadata(int grid_r)
        {
            nz_tiles.resize(grid_r, 0);
            nz_ublocks.resize(grid_r, 0);
            nz_strips.resize(grid_r, 0);
        }
    };

    UniqueId id;
    GridShape grid_shape;
    OpShape op_shape;
    const graphlib::BudaOpNode *buda_op_node = nullptr;
    DataFormat data_format;
    bool input_prologue = false;
    bool is_sparse_matmul = false;
    bool consumes_rz_major = false;
    const sparse::SparseBUDA *sparse_buda = nullptr;            // sparse-matmul specific
    std::shared_ptr<const SparseMetadata> sparse_metadata = nullptr;  // sparse-matmul specific
    // ^ using shared_ptr (vs unique_ptr) to allow for implicit copy construction of OpModel
    TStreamFactor t_stream_factor;
    int fracture_factor;
    int sparse_indices;
    Padding padding;
    int overlay_size = 0;  // Op-level override for overlay blob size in Bytes, value 0 maps to default size, which is
                           // currently 65536 (64 kB)
    std::vector<BufferModel> input_buffers;
    std::vector<BufferModel> output_buffers;
    std::vector<BufferModel> parameter_buffers;
    std::vector<BufferModel> intermediate_buffers;
    std::vector<BufferModel> dram_buffers;
    std::unordered_map<std::string, balancer::UBlockShape> fused_op_ublock_shape;
    std::unordered_map<graphlib::NodeId, TensorShape> effective_input_buffer_shape_for_user;
    mutable int cached_execution_cycles = 0;
#ifdef DEBUG
    graphlib::EdgeUniqueId eliminating_edge;
    std::unordered_set<std::uint64_t> op_model_valid_pair_id;
#endif

    inline BlockShape block_shape() const { return output_buffers.at(0).block_shape; }
    inline UBlockShape ublock_shape() const { return output_buffers.at(0).block_shape.ublock; }
    inline void set_ublock_shape(UBlockShape ublock)
    {
        for (BufferModel &output_buffer : output_buffers)
        {
            output_buffer.block_shape.set_ublock_shape(ublock);
        }
    }
    std::size_t get_l1_memory_usage() const;
    void invalidate_cached_execution_cycles() const { cached_execution_cycles = 0; }
    int get_execution_cycles(
        std::string const &arch_name, bool theoretical = false, bool invalidate_cached = false) const;
    int get_output_buffer_factor() const { return output_buffers.at(0).buffer_factor; }
    bool has_parameter_buffers() const
    {
        return std::any_of(parameter_buffers.begin(), parameter_buffers.end(), [](auto b) { return bool(b); });
    }
    int num_parameter_buffers() const
    {
        return std::count_if(parameter_buffers.begin(), parameter_buffers.end(), [](auto b) { return bool(b); });
    }
    bool is_streaming() const { return not t_stream_factor.none(); }
    Parallelization parallelization() const
    {
        return Parallelization(grid_shape.r * t_stream_factor.r, grid_shape.c * t_stream_factor.c);
    }

    const std::string &op_type() const;
    MathFidelity math_fidelity() const;
    bool is_gradient_op() const;
    bool is_matmul() const;
    std::shared_ptr<FusedOp> fused_op() const;

    const std::string get_reduce_dim() const;
    const BudaOpAttrs buda_op_attrs() const;

    GridShape get_input_grid_shape(int input_idx) const
    {
        return buda_op_node->is_matmul() ? (input_idx == 0 ? GridShape(grid_shape.r, 1) : GridShape(1, grid_shape.c))
                                         : grid_shape;
    }
    int get_input_bytes(int input_idx) const
    {
        return input_buffers[input_idx].total_size_bytes() * get_input_grid_shape(input_idx).volume();
    }
    int get_param_bytes(int input_idx) const
    {
        return parameter_buffers[input_idx].total_size_bytes() * get_input_grid_shape(input_idx).volume();
    }
    int get_dram_bytes(int input_idx) const
    {
        return dram_buffers[input_idx].total_size_bytes() * get_input_grid_shape(input_idx).volume();
    }
    int get_output_bytes() const { return output_buffers[0].total_size_bytes() * grid_shape.volume(); }
    int get_total_param_bytes() const
    {
        int total_param_bytes = 0;
        for (auto const &parameter_buffer : parameter_buffers) total_param_bytes += parameter_buffer.total_size_bytes();
        return total_param_bytes;
    }
    int get_total_dram_bytes() const
    {
        int total_dram_bytes = 0;
        for (auto const &dram_buffer : dram_buffers) total_dram_bytes += dram_buffer.total_size_bytes();
        return total_dram_bytes;
    }
    float get_input_bytes_per_cycle(int input_idx, std::string const &arch_name) const
    {
        return static_cast<float>(get_input_bytes(input_idx)) / get_execution_cycles(arch_name);
    }
    float get_output_bytes_per_cycle(std::string const &arch_name) const
    {
        return static_cast<float>(get_output_bytes()) / get_execution_cycles(arch_name);
    }

    // PyBind is acting strange with std::shared_ptr, and there seem to be some bugs reported on this, doing this for
    // now...
    const SparseMetadata get_sparse_metadata()
    {
        return *sparse_metadata.get();
    }

    bool operator==(OpModel const &other) const { return id == other.id; }

    // This function is used to compare two OpModels for similarity. It is used for caching mechanisms.
    //
    bool is_similar(OpModel const &other) const
    {
        return buda_op_node == other.buda_op_node
        and grid_shape == other.grid_shape
        and t_stream_factor == other.t_stream_factor
        and fracture_factor == other.fracture_factor
        and input_prologue == other.input_prologue
        and padding == other.padding
        and input_buffers == other.input_buffers
        and output_buffers == other.output_buffers
        and is_similar_sparse_metadata(other);
    }

    TensorShape get_out_shape(bool post_t_stream = true) const
    {
        TensorShape out_shape = op_shape.outputs[0];
        if (post_t_stream)
        {
            return TensorShape(
                out_shape.w,
                out_shape.z * t_stream_factor.t(),
                out_shape.rt / t_stream_factor.r,
                out_shape.ct / t_stream_factor.c);
        }
        else
        {
            return out_shape;
        }
    }

   private:
    int get_execution_cycles_uncached(std::string const &arch_name, bool theoretical = false) const;
    bool is_similar_sparse_metadata(OpModel const &other) const
    {
        if (sparse_metadata == nullptr and other.sparse_metadata == nullptr)
        {
            return true;
        }

        if (sparse_metadata == nullptr or other.sparse_metadata == nullptr)
        {
            return false;
        }

        return *sparse_metadata == *other.sparse_metadata;
    }
};

using LegalOpModels = std::unordered_map<graphlib::Node const *, std::vector<OpModel>>;
using OpModels = std::unordered_map<graphlib::Node const *, OpModel>;
using OpModelMap = std::unordered_map<std::string, OpModel>;
using BlockShapeMap = std::unordered_map<std::string, BlockShape>;
using CutEdges = std::unordered_map<graphlib::Edge, bool>;

struct FusedSubOpModel
{
    std::string type;
    int mblock_m;
    int mblock_n;
    int ublock_rt;
    int ublock_ct;
    int mblock_k = 0;
    int ublock_kt = 0;
    std::string reduce_dim = "";
    bool has_dest_input = false;
    bool has_dest_output = false;
};

class FactorizedInt
{
   public:
    using Factors = int;
    using FactorRange = std::pair<int, int>;
    struct Constant
    {
        Constant(int v) : v(v) {}
        int v;
    };
    struct Factorial
    {
        Factorial(int max, int multiplier = 1) : max(max), multiplier(multiplier) {}
        int max;
        int multiplier;
    };

   public:
    FactorizedInt() = default;
    // Inclusive ranges
    FactorizedInt(Factors max_val);
    FactorizedInt(FactorRange r);
    template <typename Iterator>
    FactorizedInt(Iterator begin, Iterator end) : factors(begin, end)
    {
    }
    FactorizedInt(Constant s);
    FactorizedInt(Factorial f);

    int value() const;
    int get_min_factor() const;
    int get_max_factor() const;
    int get_nearest_factor_le(int integer) const;
    std::vector<int> const &get_factors() const;
    FactorizedInt keep_factors_divisible_by(FactorizedInt const &other) const;

    // Set intersection
    FactorizedInt operator&(FactorizedInt const &other) const;
    // Set union
    FactorizedInt operator|(FactorizedInt const &other) const;
    // Set difference
    FactorizedInt operator-(FactorizedInt const &other) const;

    // Multiply
    FactorizedInt operator*(FactorizedInt const &other) const;
    // Divide
    FactorizedInt operator/(FactorizedInt const &other) const;

    bool operator==(FactorizedInt const &other) const;
    bool overlaps(FactorizedInt const &other) const;
    bool contains(int v) const;
    bool is_singleton() const;
    inline bool empty() const { return factors.empty(); }

   private:
    static std::vector<int> factorize(int min_val, int max_val);

   private:
    std::vector<int> factors;
};

struct FactorizedShape
{
    using Constant = FactorizedInt::Constant;

    FactorizedInt r;
    FactorizedInt c;

    FactorizedShape() = default;
    FactorizedShape(graphlib::Shape const &shape);
    FactorizedShape(std::pair<int, int> shape);
    FactorizedShape(Parallelization par);
    FactorizedShape(FactorizedInt r, FactorizedInt c);

    std::pair<int, int> get_min_factor() const { return std::make_pair(r.get_min_factor(), c.get_min_factor()); }
    std::pair<int, int> get_max_factor() const { return std::make_pair(r.get_max_factor(), c.get_max_factor()); }

    // Set intersection on r & c independently
    FactorizedShape operator&(FactorizedShape const &other) const;
    // Set union on r & c independently
    FactorizedShape operator|(FactorizedShape const &other) const;
    // Set difference on r & c independently
    FactorizedShape operator-(FactorizedShape const &other) const;
    // Random access into set
    Parallelization operator[](int idx) const;

    bool operator==(FactorizedShape const &other) const;
    bool empty() const;
    std::size_t size() const;
    bool is_subset_of(FactorizedShape const &other) const;
    bool is_superset_of(FactorizedShape const &other) const;
    bool is_singleton() const;

    class Iterator
        : public std::iterator<std::input_iterator_tag, Parallelization, int, Parallelization const *, Parallelization>
    {
        int i = 0;
        FactorizedShape const *p;

       public:
        Iterator(FactorizedShape const *p);
        Iterator(FactorizedShape const *p, int i);

        Iterator &operator++();
        Iterator operator++(int);
        bool operator==(Iterator other) const;
        bool operator!=(Iterator other) const;
        reference operator*() const;
    };

    Iterator begin() const;
    Iterator end() const;
};

using GridCoord = GridShape;

struct CanCoord
{
    int w = 0;
    int t = 0;
    int rt = 0;
    int ct = 0;
    CanCoord(int w, int t, int rt, int ct) : w(w), t(t), rt(rt), ct(ct) {}
    CanCoord(int t, int rt, int ct) : w(0), t(t), rt(rt), ct(ct) {}
    bool operator==(CanCoord o) const { return w == o.w and t == o.t and rt == o.rt and ct == o.ct; }
};

union LinCoord
{
    struct
    {
        std::uint64_t grid_r : 16;
        std::uint64_t grid_c : 16;
        std::uint64_t address : 32;
    } v;
    std::uint64_t bits;

    LinCoord()
    {
        v.grid_r = std::numeric_limits<std::uint16_t>::max();
        v.grid_c = std::numeric_limits<std::uint16_t>::max();
        v.address = 0;
    }
    LinCoord(int grid_r, int grid_c, int address)
    {
        v.grid_r = grid_r;
        v.grid_c = grid_c;
        v.address = address;
    }
    LinCoord(GridCoord grid, int address) : LinCoord(grid.r, grid.c, address) {}
    int grid_r() const { return (int)v.grid_r; }
    int grid_c() const { return (int)v.grid_c; }
    GridCoord grid_coord() const { return GridCoord(grid_r(), grid_c()); }
    int address() const { return (int)v.address; }
    LinCoord next() const
    {
        auto n = *this;
        ++n.v.address;
        return n;
    }
    bool operator==(LinCoord o) const { return bits == o.bits; }
    bool operator!=(LinCoord o) const { return bits != o.bits; }
    bool valid() const
    {
        return not(
            v.grid_r == std::numeric_limits<std::uint16_t>::max() and
            v.grid_c == std::numeric_limits<std::uint16_t>::max() and v.address == 0);
    }

   private:
    LinCoord(std::uint64_t bits) : bits(bits) {}
};

struct TileLayout
{
    GridShape grid_shape;
    BlockShape block_shape;
    graphlib::UBlockOrder ublock_order;
    Padding padding;

    TileLayout(
        GridShape grid_shape, BlockShape block_shape, graphlib::UBlockOrder ublock_order, Padding padding = Padding()) :
        grid_shape(grid_shape), block_shape(block_shape), ublock_order(ublock_order), padding(padding)
    {
    }

    bool operator==(TileLayout const &other) const
    {
        return grid_shape == other.grid_shape and block_shape == other.block_shape and
               ublock_order == other.ublock_order;
    }
    inline TensorShape shape() const { return TensorShape(w(), t(), rt(), ct()); }
    inline int w() const { return 1; }
    inline int t() const { return block_shape.t; }
    inline int rt() const { return grid_shape.r * block_shape.rt(); }
    inline int ct() const { return grid_shape.c * block_shape.ct(); }
    inline int volume(bool include_t = true, bool include_padding = false) const
    {
        return (include_t ? t() : 1) * (rt() - int(not include_padding) * padding.rt) *
               (ct() - int(not include_padding) * padding.ct);
    }
    LinCoord operator[](int idx) const;
    LinCoord map(CanCoord can_coord) const;
    CanCoord map(LinCoord lin_coord) const;
    GridCoord grid_coord(CanCoord can_coord) const;
    GridCoord grid_coord(LinCoord lin_coord) const;
};

struct Pipe
{
    TileLayout producer_layout;
    TileLayout consumer_layout;
    std::vector<graphlib::OpType> tms;
    int producer_out_buf_mb;

    Pipe(
        TileLayout producer_layout,
        int producer_out_buf_mb,
        std::vector<graphlib::OpType> tms,
        TileLayout consumer_layout) :
        producer_layout(producer_layout),
        consumer_layout(consumer_layout),
        tms(tms),
        producer_out_buf_mb(producer_out_buf_mb)
    {
    }

    bool operator==(Pipe const &other) const
    {
        return producer_layout == other.producer_layout and consumer_layout == other.consumer_layout and
               tms == other.tms and producer_out_buf_mb == other.producer_out_buf_mb;
    }
};

struct ResourceUsage
{
    int producer_fan_out = 0;
    int consumer_fan_in = 0;
    int producer_phases = 0;
    int consumer_phases = 0;
};

struct OpCycleEstimates
{
    int kernel_cycles = 0;

    std::vector<float> input_bw_estimates;
    std::vector<int> memory_read_cycles;

    std::vector<float> output_bw_estimates;
    std::vector<int> memory_write_cycles;

    int calculate_op_limiter_cycles() const
    {
        int limiter_cycles = kernel_cycles;
        for (int in_memory_read_cycle : memory_read_cycles)
        {
            limiter_cycles = std::max(limiter_cycles, in_memory_read_cycle);
        }
        for (int out_memory_write_cycle : memory_write_cycles)
        {
            limiter_cycles = std::max(limiter_cycles, out_memory_write_cycle);
        }

        return limiter_cycles;
    }
    
};

inline std::ostream &operator<<(std::ostream &os, CanCoord const &coord)
{
    os << "CanCoord{.w = " << coord.w << ", .t = " << coord.t << ", .rt = " << coord.rt << ", .ct = " << coord.ct
       << "}";
    return os;
}

inline std::ostream &operator<<(std::ostream &os, LinCoord const &coord)
{
    os << "LinCoord{.grid_r = " << coord.grid_r() << ", .grid_c = " << coord.grid_c()
       << ", .address = " << coord.address() << "}";
    return os;
}

inline std::ostream &operator<<(std::ostream &os, TileLayout const &layout)
{
    os << "TileLayout(GridShape(" << layout.grid_shape.r << ", " << layout.grid_shape.c << "), BlockShape("
       << layout.block_shape.t << ", " << layout.block_shape.m() << ", " << layout.block_shape.n() << ", UBlockShape("
       << layout.block_shape.ublock.rt << ", " << layout.block_shape.ublock.ct << ")), "
       << (layout.ublock_order == graphlib::UBlockOrder::R ? "graphlib::UBlockOrder::R" : "graphlib::UBlockOrder::C");
    if (layout.padding.rt or layout.padding.ct)
        os << ", Padding(" << layout.padding.rt << ", " << layout.padding.ct << ")";
    os << ")";
    return os;
}

inline std::ostream &operator<<(std::ostream &os, Pipe const &pipe)
{
    os << "Pipe(" << pipe.producer_layout << ", " << pipe.producer_out_buf_mb << ", {";
    for (auto const &tm : pipe.tms) os << tm << ", ";
    os << "}, " << pipe.consumer_layout << ")";
    return os;
}

inline std::ostream &operator<<(std::ostream &os, TensorShape const &tensor_shape)
{
    os << "TensorShape{.w = " << tensor_shape.w << ", .z = " << tensor_shape.z << ", .rt = " << tensor_shape.rt
       << ", .ct = " << tensor_shape.ct << "}";
    return os;
}

inline std::ostream &ostream_with_indent(std::ostream &os, OpShape const &op_shape, char const *indent = "")
{
    os << indent << "OpShape{" << std::endl;
    os << indent << "  .inputs = {" << std::endl;
    int i = 0;
    for (TensorShape const &input : op_shape.inputs)
    {
        os << indent << "    [" << i++ << "] = " << input << std::endl;
    }
    os << indent << "  }," << std::endl;
    os << indent << "  .outputs = {" << std::endl;
    i = 0;
    for (TensorShape const &output : op_shape.outputs)
    {
        os << indent << "    [" << i++ << "] = " << output << std::endl;
    }
    os << indent << "  }," << std::endl;
    os << indent << "}";
    return os;
}

inline std::ostream &operator<<(std::ostream &os, OpShape const &op_shape) { return ostream_with_indent(os, op_shape); }

inline std::ostream &operator<<(std::ostream &os, GridShape const &grid_shape)
{
    os << "GridShape{.r = " << grid_shape.r << ", .c = " << grid_shape.c << "}";
    return os;
}

inline std::ostream &operator<<(std::ostream &os, Parallelization const &grid_shape)
{
    os << "Parallelization{.r = " << grid_shape.r << ", .c = " << grid_shape.c << "}";
    return os;
}

inline std::ostream &operator<<(std::ostream &os, UBlockShape const &ublock)
{
    os << "UBlockShape{.rt = " << ublock.rt << ", .ct = " << ublock.ct << "}";
    return os;
}

inline std::ostream &operator<<(std::ostream &os, BlockShape const &block_shape)
{
    os << "BlockShape{.t = " << block_shape.t << ", .mblock_m = " << (block_shape.mblock_m * block_shape.tblock_m)
       << ", .mblock_n = " << (block_shape.mblock_n * block_shape.tblock_n) << ", .ublock = " << block_shape.ublock
       << "}";
    return os;
}

inline std::ostream &operator<<(std::ostream &os, TStreamDir const &dir)
{
    switch (dir.v)
    {
        case TStreamDir::R: os << "TStreamDir::R"; break;
        case TStreamDir::C: os << "TStreamDir::C"; break;
        case TStreamDir::RZ: os << "TStreamDir::RZ"; break;
        case TStreamDir::CZ: os << "TStreamDir::CZ"; break;
        default: os << "TStreamDir::Unknown"; break;
    }
    return os;
}

inline std::ostream &operator<<(std::ostream &os, TStreamFactor const &tsf)
{
    os << "TStreamFactor{.dir = " << tsf.dir << ", .r = " << tsf.r << ", .c = " << tsf.c << "}";
    return os;
}

inline std::ostream &operator<<(std::ostream &os, BufferModel const &buffer_model)
{
    os << "BufferModel{.block_shape = " << buffer_model.block_shape
       << ", .buffer_factor = " << buffer_model.buffer_factor << ", .l1_size_tiles = " << buffer_model.l1_size_tiles
       << ", .data_format = " << buffer_model.data_format << "}";
    return os;
}

inline std::ostream &ostream_with_indent(
    std::ostream &os, std::vector<BufferModel> const &buffer_models, char const *indent = "")
{
    os << "{" << std::endl;
    int i = 0;
    for (BufferModel const &buffer_model : buffer_models)
    {
        if (not buffer_model)
            continue;
        os << indent << "    [" << i++ << "] = " << buffer_model << std::endl;
    }
    os << indent << "  }," << std::endl;
    return os;
}

inline std::ostream &ostream_with_indent(std::ostream &os, OpModel const &op_model, char const *indent = "")
{
    os << indent << "OpModel{" << std::endl;
    os << indent << "  .id = " << op_model.id.id << std::endl;
    os << indent << "  .grid_shape = " << op_model.grid_shape << std::endl;
    os << indent << "  .op_shape = ";
    ostream_with_indent(os, op_model.op_shape, (std::string("  ") + indent).c_str()) << std::endl;
    os << indent << "  .op_type = " << op_model.op_type() << "," << std::endl;
    os << indent << "  .data_format = " << op_model.data_format << "," << std::endl;
    os << indent << "  .math_fidelity = " << op_model.math_fidelity() << "," << std::endl;
    os << indent << "  .t_stream_factor = " << op_model.t_stream_factor << "," << std::endl;
    os << indent << "  .fracture_factor = " << op_model.fracture_factor << "," << std::endl;
    os << indent << "  .cached_execution_cycles = " << op_model.cached_execution_cycles << "," << std::endl;
    os << indent << "  .input_buffers = ";
    ostream_with_indent(os, op_model.input_buffers, indent);
    os << indent << "  .output_buffers = ";
    ostream_with_indent(os, op_model.output_buffers, indent);
    os << indent << "  .parameter_buffers = ";
    ostream_with_indent(os, op_model.parameter_buffers, indent);
    os << indent << "  .intermediate_buffers = ";
    ostream_with_indent(os, op_model.intermediate_buffers, indent);
    os << indent << "  .dram_buffers = ";
    ostream_with_indent(os, op_model.dram_buffers, indent);
    os << indent << "}";
    return os;
}

inline std::ostream &operator<<(std::ostream &os, OpModel const &op_model) { return ostream_with_indent(os, op_model); }

inline std::ostream &operator<<(std::ostream &os, OpModel::SparseMetadata const &sparse_metadata)
{
    os << "SparseMetadata{.nz_tiles = {";
    for (int nz_tile : sparse_metadata.nz_tiles) os << nz_tile << ", ";
    os << "}, .nz_ublocks = {";
    for (int nz_ublock : sparse_metadata.nz_ublocks) os << nz_ublock << ", ";
    os << "}, .nz_strips = {";
    for (int nz_strip : sparse_metadata.nz_strips) os << nz_strip << ", ";
    os << "}}";
    return os;
}

inline std::ostream &operator<<(std::ostream &os, FusedSubOpModel const &sub_op_model)
{
    os << "FusedSubOpModel{" << std::endl;
    os << "  .type = " << sub_op_model.type << std::endl;
    os << "  .mblock_m = " << sub_op_model.mblock_m << std::endl;
    os << "  .mblock_n = " << sub_op_model.mblock_n << std::endl;
    os << "  .ublock_rt = " << sub_op_model.ublock_rt << std::endl;
    os << "  .ublock_ct = " << sub_op_model.ublock_ct << std::endl;
    os << "  .mblock_k = " << sub_op_model.mblock_k << std::endl;
    os << "  .ublock_kt = " << sub_op_model.ublock_kt << std::endl;
    os << "  .reduce_dim = " << sub_op_model.reduce_dim << std::endl;
    os << "  .has_dest_input = " << sub_op_model.has_dest_input << std::endl;
    os << "  .has_dest_output = " << sub_op_model.has_dest_output << std::endl;
    os << "}";
    return os;
}

inline std::ostream &operator<<(std::ostream &os, FactorizedInt const &fi)
{
    os << "{";
    for (auto i : fi.get_factors()) os << i << ", ";
    os << "}";
    return os;
}

inline std::ostream &operator<<(std::ostream &os, FactorizedShape const &gp)
{
    os << "{";
    for (Parallelization parallelization : gp)
    {
        os << parallelization << ", ";
    }
    os << "}";
    return os;
}

inline std::ostream &operator<<(std::ostream &os, ResourceUsage const &ru)
{
    os << "ResourceUsage{.producer_fan_out=" << ru.producer_fan_out << ", .consumer_fan_in=" << ru.consumer_fan_in
       << ", .producer_phases=" << ru.producer_phases << ", .consumer_phases=" << ru.consumer_phases << "}";
    return os;
}

}  // namespace tt::balancer

namespace std
{
template <>
struct hash<tt::balancer::TileLayout>
{
    std::size_t operator()(tt::balancer::TileLayout const &layout) const
    {
        std::size_t seed = 0;
        // intentionally exclude edge_creation_id from the hash
        tt::hash_combine(seed, static_cast<size_t>(layout.grid_shape.r));
        tt::hash_combine(seed, static_cast<size_t>(layout.grid_shape.c));
        tt::hash_combine(seed, static_cast<size_t>(layout.block_shape.t));
        tt::hash_combine(seed, static_cast<size_t>(layout.block_shape.tblock_m));
        tt::hash_combine(seed, static_cast<size_t>(layout.block_shape.tblock_n));
        tt::hash_combine(seed, static_cast<size_t>(layout.block_shape.mblock_m));
        tt::hash_combine(seed, static_cast<size_t>(layout.block_shape.mblock_n));
        tt::hash_combine(seed, static_cast<size_t>(layout.block_shape.ublock.rt));
        tt::hash_combine(seed, static_cast<size_t>(layout.block_shape.ublock.ct));
        tt::hash_combine(seed, static_cast<size_t>(layout.ublock_order));
        return seed;
    }
};

template <>
struct hash<tt::balancer::Pipe>
{
    std::size_t operator()(tt::balancer::Pipe const &pipe) const
    {
        std::size_t seed = 0;
        // intentionally exclude edge_creation_id from the hash
        tt::hash_combine(seed, hash<tt::balancer::TileLayout>{}(pipe.producer_layout));
        tt::hash_combine(seed, hash<tt::balancer::TileLayout>{}(pipe.consumer_layout));
        tt::hash_combine(seed, static_cast<size_t>(pipe.tms.size()));
        tt::hash_combine(seed, static_cast<size_t>(pipe.producer_out_buf_mb));
        return seed;
    }
};
}  // namespace std