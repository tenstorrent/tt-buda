// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <limits>
#include <set>
#include <vector>

#include "utils/assert.hpp"
#include "utils/logger.hpp"

namespace tt::sparse
{

constexpr uint32_t TILE_DIM = 32;

// third_party/budabackend/ops/mm_bare_structs.hpp
union strip_info_struct
{
    struct
    {
        std::uint16_t enc[2];
        std::uint16_t nz_ublocks;
        std::uint16_t index_array[];
    } val;

    struct F
    {
        std::uint16_t index_l : 16;
        std::uint16_t index_h : 14;
        std::uint16_t last_strip_in_row : 1;
        std::uint16_t last_strip_in_tile : 1;
        std::uint16_t nz_ublocks;
        std::uint16_t index_array[];
    } f;

    std::uint32_t strip_index() const { return (std::uint32_t(f.index_h) << 16u) | std::uint32_t(f.index_l); }
    void set_strip_index(std::uint32_t strip_index)
    {
        TT_ASSERT(strip_index < (1u << 30u));
        f.index_h = strip_index >> 16u;
        f.index_l = std::uint16_t(strip_index & ((1u << 16u) - 1u));
    }

    strip_info_struct() { __builtin_memset(this, 0, sizeof(*this)); }

    strip_info_struct(
        std::uint32_t strip_index,
        std::uint16_t nz_ublocks,
        bool last_strip_in_row = false,
        bool last_strip_in_tile = false)
    {
        TT_ASSERT(strip_index < (1u << 30u));
        f.index_h = strip_index >> 16u;
        f.index_l = std::uint16_t(strip_index & ((1u << 16u) - 1u));
        f.last_strip_in_row = last_strip_in_row;
        f.last_strip_in_tile = last_strip_in_tile;
        f.nz_ublocks = nz_ublocks;
    }
};

union EncodedIndex
{
    struct
    {
        std::uint32_t value : 16;
        std::uint32_t unique_id : 12;
        std::uint32_t rsvd : 1;
        std::uint32_t loop_continue : 1;
        std::uint32_t loop_start : 1;
        std::uint32_t last_tile : 1;
    };
    std::uint32_t bits;

    EncodedIndex() : bits(0) {}
};

struct Bounds
{
    std::int64_t min = std::numeric_limits<std::int64_t>::max();
    std::int64_t max = std::numeric_limits<std::int64_t>::min();

    Bounds() = default;
    Bounds(std::int64_t min, std::int64_t max) : min(min), max(max) {}

    void extend(Bounds other)
    {
        min = std::min(min, other.min);
        max = std::max(max, other.max);
    }

    void extend(std::int64_t minmax) { extend(Bounds(minmax, minmax)); }
};

struct SparseCOO
{
    std::vector<std::int64_t> rows;
    std::vector<std::int64_t> cols;
    std::vector<float> vals;
    std::vector<std::int64_t> shape;
    Bounds col_bounds;

    SparseCOO(
        std::vector<std::int64_t> const& rows,
        std::vector<std::int64_t> const& cols,
        std::vector<float> const& vals,
        std::vector<std::int64_t> shape) :
        rows(rows), cols(cols), vals(vals), shape(shape), col_bounds(cols.front(), cols.back())
    {
        // is_sorted() returns true if non-descending order
        if (std::is_sorted(rows.begin(), rows.end()))
        {
            this->sorted_order = SortOrder::ROW_MAJOR;
        }
    }

    SparseCOO(std::vector<std::int64_t> shape) : shape(shape) {}

    int rt() const { return (shape[0] + TILE_DIM - 1) / TILE_DIM; }
    int ct() const { return (shape[1] + TILE_DIM - 1) / TILE_DIM; }

    enum SortOrder
    {
        UNSORTED,
        ROW_MAJOR,
        COLUMN_MAJOR,
    };

    void dump() const
    {
        int rt_dim = (shape[0] + TILE_DIM - 1) / TILE_DIM;
        int ct_dim = (shape[1] + TILE_DIM - 1) / TILE_DIM;
        int idx = 0;
        for (int i = 0; i < (int)rows.size(); ++i)
        {
            int rt = (int)rows[i] / TILE_DIM;
            int ct = (int)cols[i] / TILE_DIM;

            for (; idx <= (rt * ct_dim + ct); ++idx)
            {
                char const* tile = (idx == (rt * ct_dim + ct)) ? "1" : ".";
                char const* sep = (((idx + 1) % ct_dim) == 0) ? "\n" : " ";
                fmt::print("{}{}", tile, sep);
            }
        }
        while (idx++ < (rt_dim * ct_dim)) fmt::print(".{}", ((idx % ct_dim) == 0) ? "\n" : " ");
    }

    // Vertically slice a SparseCOO tensor
    //
    std::vector<SparseCOO> vslice(int num_slices) const
    {
        TT_ASSERT(shape[0] % num_slices == 0);

        if (num_slices == 1)
        {
            return {*this};
        }

        std::int64_t slice_height = shape[0] / num_slices;

        std::vector<SparseCOO> ret(num_slices, SparseCOO({this->shape[0] / num_slices, this->shape[1]}));

        // Calculate total count of indices upfront, in order to reserve vector space once
        //
        std::vector<int> cache(num_slices, 0);
        for (size_t idx = 0; idx < this->rows.size(); idx++)
        {
            int slice_idx = this->rows[idx] / slice_height;
            cache[slice_idx]++;
        }
        for (size_t idx = 0; idx < cache.size(); idx++)
        {
            ret[idx].rows.reserve(cache[idx]);
            ret[idx].cols.reserve(cache[idx]);
            ret[idx].vals.reserve(cache[idx]);
        }

        for (size_t idx = 0; idx < this->rows.size(); idx++)
        {
            int slice_idx = this->rows[idx] / slice_height;
            ret[slice_idx].rows.push_back(this->rows[idx] % slice_height);
            ret[slice_idx].cols.push_back(this->cols[idx]);
            ret[slice_idx].vals.push_back(this->vals[idx]);
            ret[slice_idx].col_bounds.extend(this->cols[idx]);
        }

        return ret;
    }

    template <typename Iter>
    static SparseCOO vcat(Iter begin, Iter end)
    {
        TT_ASSERT(begin != end);
        auto shape = begin->shape;
        TT_ASSERT(shape.size() == 2);
        shape[0] *= static_cast<std::int64_t>(std::distance(begin, end));
        SparseCOO ret(shape);

        // Calculate total count of indices upfront, in order to reserve vector space once
        //
        std::uint64_t total_count_indices = 0;
        for (auto iter = begin; iter != end; ++iter)
        {
            total_count_indices += iter->vals.size();
        }

        // Early out if empty
        //
        if (total_count_indices == 0)
        {
            return ret;
        }

        // Cols and Vals get reserved, while Rows get resized - we manually manage the Rows indices in the loop below
        //
        ret.rows.resize(total_count_indices);
        ret.cols.reserve(total_count_indices);
        ret.vals.reserve(total_count_indices);

        std::int64_t row_offset = 0;
        std::int64_t indices_previously_added = 0;
        for (auto iter = begin; iter != end; ++iter)
        {
            SparseCOO const& coo = *iter;
            TT_ASSERT(coo.shape == begin->shape);

            // Nothing to update if `coo` is empty
            //
            if (coo.vals.empty())
            {
                row_offset += coo.shape[0];
                continue;
            }

            for (std::int64_t i = 0; i < static_cast<std::int64_t>(coo.rows.size()); ++i)
            {
                ret.rows[indices_previously_added++] = coo.rows[i] + row_offset;
            }
            ret.cols.insert(ret.cols.end(), coo.cols.begin(), coo.cols.end());
            ret.vals.insert(ret.vals.end(), coo.vals.begin(), coo.vals.end());
            ret.col_bounds.extend(coo.col_bounds);
            row_offset += coo.shape[0];
        }

        TT_ASSERT(ret.rows.size() == total_count_indices);
        TT_ASSERT(ret.cols.size() == total_count_indices);
        TT_ASSERT(ret.vals.size() == total_count_indices);

        return ret;
    }

    static SparseCOO vcat(std::vector<SparseCOO> const& coos) { return vcat(coos.begin(), coos.end()); }

    void sort(SortOrder sort_order)
    {
        TT_ASSERT(sort_order != SortOrder::UNSORTED, "Can't sort in UNSORTED sort order");

        // If already sorted sort_order way, we can just return
        if (sort_order == this->sorted_order)
        {
            return;
        }

        sort_(sort_order);
        sorted_order = sort_order;
    }

   private:
    SortOrder sorted_order = SortOrder::UNSORTED;

    struct RowColVal
    {
        std::int64_t row;
        std::int64_t col;
        float val;
    };

    // Sorts the COO matrix in either row-major or column-major order
    //
    void sort_(bool row_major)
    {
        // If this method ever becomes memory hungry (or slow due to memory allocations), an alternative approach would
        // be something like this: https://stackoverflow.com/a/17074810/4030496

        // Zip rows, cols, and vals together
        //
        std::vector<RowColVal> zipped;
        zipped.reserve(rows.size());
        for (size_t idx = 0; idx < rows.size(); idx++)
        {
            zipped.push_back(RowColVal{rows[idx], cols[idx], vals[idx]});
        }

        // Sort either row-major or column-major
        //
        if (row_major)
        {
            std::sort(
                zipped.begin(),
                zipped.end(),
                [](const auto& lhs, const auto& rhs)
                { return lhs.row == rhs.row ? lhs.col < rhs.col : lhs.row < rhs.row; });
        }
        else
        {
            std::sort(
                zipped.begin(),
                zipped.end(),
                [](const auto& lhs, const auto& rhs)
                { return lhs.col == rhs.col ? lhs.row < rhs.row : lhs.col < rhs.col; });
        }

        // Update original rows, cols, and vals
        //
        for (size_t idx = 0; idx < rows.size(); idx++)
        {
            rows[idx] = zipped[idx].row;
            cols[idx] = zipped[idx].col;
            vals[idx] = zipped[idx].val;
            col_bounds.extend(cols[idx]);
        }
    }
};

struct SparseIndex
{
    int unique_id = 0;
    int ct = 0;
    int rt = 0;
    int z = 0;
    int zdim = 0;

    SparseIndex(std::uint32_t unique_id, int ct, int rt, int z, int zdim) :
        unique_id(unique_id), ct(ct), rt(rt), z(z), zdim(zdim){};

    int ubr_idx(int u_rt) const { return rt / u_rt; }
    int ubc_idx(int u_kt) const { return ct / u_kt; }
};

using SparseTiles = std::vector<std::vector<float>>;
using EncodingTiles = std::vector<std::vector<std::int32_t>>;
struct SparseBUDA
{
   public:
    enum class Layout
    {
        Default,         // Default layout
        ZMajor,          // Z-major layout, e.g. RZ streaming - go thru all Zs first, for a given R slice
        ZMajorDataflow,  // Z-major layout, special cased for sparse->dense dataflow (same as ZMajor, but slice
                         // vertically down to single tile)
    };

    // A little more on layouts...
    //
    // Layout dictates what the sparse tensor of the sparse matmul will look like. Sparse tensors get divided into
    // chunks, to accomodate for parallelization across cores and t-streaming. Depending on what the chunks look like,
    // and which core receives which chunk, the performance profile of the sparse matmul can change - additionally,
    // whether the parallelization is legal can also change.
    //
    // Sparse matmul ops are used for various scenarios, but most often they're there as building blocks of
    // convolutions. Sparse tensors of such ops have a specific pattern that looks something like this (e.g. for a 2x2
    // convolution):
    // [
    //   1 0 0 0
    //   0 1 0 0
    //   0 0 1 0
    //   0 0 0 1
    //   1 0 0 0
    //   0 1 0 0
    //   0 0 1 0
    //   0 0 0 1
    //   1 0 0 0
    //   0 1 0 0
    //   0 0 1 0
    //   0 0 0 1
    //   1 0 0 0
    //   0 1 0 0
    //   0 0 1 0
    //   0 0 0 1
    // ]
    // So they will look like a set of diagonal matrices, each one representing a single kernel point of the
    // convolution.
    //
    // Currently, there are 3 variants of the layout: Default, ZMajor, and ZMajorDataflow.
    //
    // Default:
    //   In this layout, the sparse tensor is in its original shape.
    //
    // ZMajor:
    //   In this layout, the goal is to eliminate serialization in execution between cores. It is easier to explain the
    //   transformation first, and then show why it eliminates serialization.
    //
    //   The transformation is as follows:
    //     - Let's say we have 2x1 cores for this sparse op and t=2. The top core will handle the first two kernel
    //     points while the second core will handle the last two kernel points - this is by design of the layout. In t=0
    //     the top core will get the top parts of the sparse tensor's first two kernel points, which is two 2x4x4
    //     pieces. The bottom core will get the top parts of the sparse tensor's last two kernel points, which is also
    //     two 2x4x4 pieces. So for t=0, the first core's tensor will look like this:
    //       [
    //         1 0 0 0  <- first row of the first kernel point
    //         0 1 0 0  <- second row of the first kernel point
    //         1 0 0 0  <- first row of the second kernel point
    //         0 1 0 0  <- second row of the second kernel point
    //       ]
    //     - And the second core, for t=0, it's tensor will look like this:
    //       [
    //         1 0 0 0  <- first row of the third kernel point
    //         0 1 0 0  <- second row of the third kernel point
    //         1 0 0 0  <- first row of the fourth kernel point
    //         0 1 0 0  <- second row of the fourth kernel point
    //       ]
    //     - If we were to continue this for t=1, the first core's tensor would look like this:
    //       [
    //         0 0 1 0  <- third row of the first kernel point
    //         0 0 0 1  <- fourth row of the first kernel point
    //         0 0 1 0  <- third row of the second kernel point
    //         0 0 0 1  <- fourth row of the second kernel point
    //       ]
    //     - And the second core, for t=1, it's tensor will look like this:
    //       [
    //         0 0 1 0  <- third row of the third kernel point
    //         0 0 0 1  <- fourth row of the third kernel point
    //         0 0 1 0  <- third row of the fourth kernel point
    //         0 0 0 1  <- fourth row of the fourth kernel point
    //       ]
    //
    //   From the perspective of the sparse matmul, where these sparse tensors are left operands, the ZMajor layout
    //   makes it so that all cores, in a given point in time, are working on the same inner dimension of the matmul. If
    //   we think about how the right operand is buffered (horizontal strips), and keep in mind that all cores receive
    //   the full right operand tensor, this layout makes it so that all cores are working on the same strip of the
    //   right operand tensor, which removes any serialization in execution.
    //
    //   Naming comes from the fact that the sparse tensor is first vertically sliced by a factor of kernel points,
    //   which is in this case a vslice(4). Then, a core will work on a set of Zs for a given R slice, so in a way, if
    //   we fix the height that a core is reading (in this case 2-tile high), we go through the tensor in Z-major order,
    //   hence the name ZMajor for the layout.
    //
    // ZMajorDataflow:
    //   This layout is a special case of ZMajor, where the sparse tensor is vertically sliced all the way down to a
    //   single tile. The goal of this layout is to improve the dataflow between the sparse and dense matmuls - if we
    //   place a sparse&dense pair of ops next to each other, with the same height of cores, as we usually do, this
    //   layout will enable dataflow thru NOC to use direct pipes, which means that each sparse core will send its
    //   output directly and only to its corresponding core of dense matmul, which is right next to the sparse core.
    //   Hence the "Dataflow" in the name.
    //
    //   Using the sparse tensor from the Default layout, the transformation to ZMajorDataflow is as follows:
    //     - Similar to ZMajor, we can first imagine that the sparse tensor is vertically sliced by a factor of kernel
    //     points, which is in this case a vslice(4), so the tensor goes from 16x4 to a 4x4x4. Then, imagine that the
    //     tensor is read completely in Z-major order, tile by tile. Let's say we have 2x1 cores for this sparse op and
    //     t=2. First core, for t=0, will get these tiles:
    //       [
    //         1 0 0 0  <- first tile of the first kernel point
    //         1 0 0 0  <- first tile of the second kernel point
    //         1 0 0 0  <- first tile of the third kernel point
    //         1 0 0 0  <- first tile of the fourth kernel point
    //       ]
    //     - And the second core, for t=0, it's tensor will look like this:
    //       [
    //         0 1 0 0  <- second tile of the first kernel point
    //         0 1 0 0  <- second tile of the second kernel point
    //         0 1 0 0  <- second tile of the third kernel point
    //         0 1 0 0  <- second tile of the fourth kernel point
    //       ]
    //     - If we were to continue this for t=1, the first core's tensor would look like this:
    //       [
    //         0 0 1 0  <- third tile of the first kernel point
    //         0 0 1 0  <- third tile of the second kernel point
    //         0 0 1 0  <- third tile of the third kernel point
    //         0 0 1 0  <- third tile of the fourth kernel point
    //       ]
    //     - And the second core, for t=1, it's tensor will look like this:
    //       [
    //         0 0 0 1  <- fourth tile of the first kernel point
    //         0 0 0 1  <- fourth tile of the second kernel point
    //         0 0 0 1  <- fourth tile of the third kernel point
    //         0 0 0 1  <- fourth tile of the fourth kernel point
    //       ]
    //
    //   The main difference between ZMajor and ZMajorDataflow is that ZMajorDataflow is vertically sliced all the way
    //   down to a single tile, which enables the dataflow between the sparse and dense matmuls to use direct pipes.
    //   However, by doing this transformation, there is a side-effect on the output, the rows of the output are all
    //   mixed up, i.e. they're not in the order that the dense matmul "expects" them to be in. To correct this, a set
    //   of TM ops is applied to the output of the sparse matmul, which will reorder the rows back to the correct order.
    //   This set of TM ops might look complicated, but when worked out, it just makes it so that each sparse matmul
    //   core sends data only to its corresponding dense matmul core, which is right next to it.

    static Layout create_layout(bool z_major, int fracture_factor);

    std::vector<SparseCOO> sparse_zs;
    std::vector<SparseIndex> sparse_indices;
    std::vector<std::int64_t> sparse_shape;
    std::vector<float> sparse_uniq_tiles;
    int zdim = 0;
    int bcast_factor = 0;
    int fracture_factor = 1;

    SparseBUDA() = default;
    SparseBUDA(
        std::vector<SparseCOO> sparse_zs,
        std::vector<SparseIndex> sparse_indices,
        std::vector<std::int64_t> sparse_shape,
        std::vector<float> tiles,
        int zdim,
        int bcast_factor,
        int fracture_factor);

    int get_sparse_tiles_per_core_estimate(int grid_r, int t_factor_r) const;
    int get_encoding_tiles_per_core_estimate(int grid_r, int t_factor_r, int u_rt, int u_kt) const;
    int get_sparse_tile_ptr_bits(int grid_r, int t_factor_r, int u_rt = 1) const;
    int get_sparse_ublock_idx_bits(int grid_r, int t_factor_r, int u_rt = 1) const;
    int get_max_u_kt(int grid_r, int t_factor_r, int u_rt, int sparse_tile_ptr_bits = 0) const;

    std::tuple<SparseTiles, EncodingTiles, std::vector<uint32_t>, std::vector<uint32_t>, std::vector<int>>
    get_sparse_tiles_and_encodings(
        int grid_r,
        int t_factor_r = 1,
        int t_factor_c = 1,
        int u_rt = 1,
        int u_kt = 1,
        int fracture_factor = 1,
        Layout layout = Layout::Default,
        std::string const& visualize_sparse_path = "") const;

    // Returns a map with keys of legal t parallelizations and values of legal u_kts
    static std::unordered_map<int, std::vector<int>> get_par_t_values(
        int grid_r,
        std::vector<int> potential_ts,
        std::vector<SparseCOO>& sparse_zs,
        std::vector<int> u_kts,
        int bcast_factor,
        int fracture_factor,
        Layout layout);

   private:
    static constexpr std::uint32_t kMaxUblocksR = 8192;
    static constexpr std::uint32_t kMaxSparseTiles = 4096;
    static constexpr std::uint32_t kMaxGridR = 10;  // TODO: parameterize based on chip arch
    static constexpr std::uint32_t kMaxSparseIndexValue = 1 << 30;
    static constexpr std::uint32_t kMaxNZUblocks = 1 << 16;
};

// SparseBUDA compress_sparse_tensor(std::vector<SparseCOO> const& sparse_zs);
SparseBUDA compress_sparse_tensor_and_strip_info(
    std::vector<SparseCOO> const& sparse_zs, int bcast_factor, int fracture_factor);

int get_u_rt_encoding_bits(int u_rt);
int get_u_kt_encoding_bits(int u_kt);

std::ostream& operator<<(std::ostream& out, const SparseCOO::SortOrder& sort_order);
std::ostream& operator<<(std::ostream& out, SparseBUDA::Layout layout);

}  // namespace tt::sparse
