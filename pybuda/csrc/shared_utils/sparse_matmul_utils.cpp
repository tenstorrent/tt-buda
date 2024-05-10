// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "sparse_matmul_utils.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "balancer/types.hpp"
#include "python_bindings_common.hpp"
#include "utils/assert.hpp"

namespace tt::sparse
{

using TileKey = std::string;

struct SparseEntry
{
    std::int32_t r;
    std::int32_t c;
    float v;

    SparseEntry(std::int32_t r, std::int32_t c, float v) : r(r), c(c), v(v) {}
    char const* get() const { return reinterpret_cast<char const*>(this); }
};

static bool comp_zcr_ublocked(SparseIndex const& a, SparseIndex const& b, int u_rt, int u_kt)
{
    // 1: Sort by z
    if (a.z == b.z)
    {
        int a_ub_c = a.ct / u_kt;
        int b_ub_c = b.ct / u_kt;

        // 2: Sort by strip index
        if (a_ub_c == b_ub_c)
        {
            int a_ub_r = a.rt / u_rt;
            int b_ub_r = b.rt / u_rt;

            // 3: Sort by ublock row
            if (a_ub_r == b_ub_r)
            {
                // 4: Sort tiles within ublock in row-major fashion
                if (a.rt == b.rt)
                {
                    return a.ct < b.ct;
                }

                return a.rt < b.rt;
            }

            return a_ub_r < b_ub_r;
        }

        return a_ub_c < b_ub_c;
    }

    return a.z < b.z;
}

std::ostream& operator<<(std::ostream& out, SparseBUDA::Layout layout)
{
    switch (layout)
    {
        case SparseBUDA::Layout::Default: out << "SparseBUDA::Layout::Default"; break;
        case SparseBUDA::Layout::ZMajor: out << "SparseBUDA::Layout::ZMajor"; break;
        case SparseBUDA::Layout::ZMajorDataflow: out << "SparseBUDA::Layout::ZMajorDataflow"; break;
        default: out << "SparseBUDA::Layout::Unknown"; break;
    }
    return out;
}

int get_u_rt_encoding_bits(int u_rt)
{
    // u_rt_bits can be 0
    int u_rt_bits = 32 - __builtin_clz(u_rt);
    u_rt_bits -= ((u_rt & (u_rt - 1)) == 0);  // power of two check
    return u_rt_bits;
}

int get_u_kt_encoding_bits(int u_kt)
{
    int u_kt_bits = 32 - __builtin_clz(u_kt);
    u_kt_bits -= ((u_kt & (u_kt - 1)) == 0);  // power of two check
    u_kt_bits = std::max(u_kt_bits, 1);       // u_kt must be bigger than 0
    return u_kt_bits;
}

struct Loop
{
    int start = 0;
    int size = 0;
    int count = 0;

    Loop() = default;
    Loop(int start, int size, int count) : start(start), size(size), count(count) {}
    int distance() const { return size * count; }
    int end() const { return start + size * count; }
    bool contains(int i) const { return i >= start and i < end(); }
    bool overlaps(Loop const& other) const
    {
        return contains(other.start) or contains(other.end()) or other.contains(start) or other.contains(end());
    }
    bool operator<(Loop const& other) const
    {
        return end() < other.end() or (end() == other.end() and size < other.size);
    }
};

std::ostream& operator<<(std::ostream& out, Loop const& loop)
{
    out << "Loop{" << loop.start << ", " << loop.size << ", " << loop.count << "}";
    return out;
}

static std::pair<std::vector<SparseIndex>, std::vector<float>> compress_unique_tiles(
    std::vector<SparseCOO> const& sparse_zs)
{
    TileKey zero_tile;
    int zdim = (int)sparse_zs.size();
    std::unordered_map<TileKey, int> unique_id_map;
    std::vector<SparseIndex> indices;
    std::vector<float> tiles;

    auto add_tile = [&unique_id_map, &tiles](TileKey const& tile) -> int
    {
        int id = 0;
        auto match = unique_id_map.find(tile);
        if (match != unique_id_map.end())
        {
            id = match->second;
        }
        else
        {
            id = (int)tiles.size() / (TILE_DIM * TILE_DIM);
            unique_id_map[tile] = id;
            int base = (int)tiles.size();
            tiles.insert(tiles.end(), TILE_DIM * TILE_DIM, 0.f);
            SparseEntry const* begin = reinterpret_cast<SparseEntry const*>(tile.data());
            SparseEntry const* end = begin + (tile.size() / sizeof(SparseEntry));
            for (SparseEntry const* entry = begin; entry != end; ++entry)
            {
                int index = base + entry->r * TILE_DIM + entry->c;
                tiles[index] = entry->v;
            }
        }
        return id;
    };

    add_tile(zero_tile);

    if (env_as<bool>("PYBUDA_SPARSE_NO_MATH"))
        return std::make_pair(indices, tiles);

    for (int z = 0; z < zdim; ++z)
    {
        SparseCOO const& sparse = sparse_zs[z];
        int next_rt = 1;
        std::unordered_map<int, TileKey> ct_to_tile_keys;

        for (int i = 0; i < (int)sparse.rows.size(); ++i)
        {
            int rt = (int)sparse.rows[i] / TILE_DIM;
            int ct = (int)sparse.cols[i] / TILE_DIM;
            int r = (int)sparse.rows[i] - (rt * TILE_DIM);
            int c = (int)sparse.cols[i] - (ct * TILE_DIM);
            SparseEntry e(r, c, sparse.vals[i]);

            if (rt < next_rt)
            {
                TileKey& tile_key = ct_to_tile_keys[ct];
                tile_key.append(e.get(), sizeof(SparseEntry));
            }
            else
            {
                // Insert zero tile on empty rows
                if (ct_to_tile_keys.empty())
                {
                    int irt = next_rt - 1;
                    int ict = ct;  // TODO: is there an optimal column to place it on?
                    indices.emplace_back(0, ict, irt, z, zdim);
                }

                for (auto const& [ict, tile_key] : ct_to_tile_keys)
                {
                    int tile_id = add_tile(tile_key);
                    int irt = next_rt - 1;
                    indices.emplace_back(tile_id, ict, irt, z, zdim);
                }

                next_rt = rt + 1;
                ct_to_tile_keys.clear();
                TileKey& tile_key = ct_to_tile_keys[ct];
                tile_key.append(e.get(), sizeof(SparseEntry));
            }
        }

        for (auto const& [ict, tile_key] : ct_to_tile_keys)
        {
            int tile_id = add_tile(tile_key);
            int irt = next_rt - 1;
            indices.emplace_back(tile_id, ict, irt, z, zdim);
        }
    }

    return std::make_pair(indices, tiles);
}

class StripAllocator
{
    using IndexType = std::remove_extent_t<decltype(strip_info_struct::F::index_array)>;
    static constexpr std::size_t kTileSize = TILE_DIM * TILE_DIM * sizeof(std::uint32_t);

   public:
    void push_strip(strip_info_struct const& info, std::vector<IndexType> strip_indices)
    {
        std::size_t total_size = sizeof(strip_info_struct) + strip_indices.size() * sizeof(IndexType);
        TT_ASSERT(total_size < kTileSize, "This can theoretically happen, but L1 mem reqs should protect against it");

        if ((strip_offset + total_size) > allocation_size_bytes())
        {
            if (prev_strip_ptr)
                prev_strip_ptr->f.last_strip_in_tile = true;
            strip_offset = allocate_tile();
        }

        std::uint8_t* strip_ptr = base_ptr() + strip_offset;
        prev_strip_ptr = reinterpret_cast<strip_info_struct*>(strip_ptr);
        TT_ASSERT(
            (reinterpret_cast<std::uintptr_t>(strip_ptr) % alignof(strip_info_struct)) == 0,
            reinterpret_cast<std::uintptr_t>(strip_ptr),
            alignof(strip_info_struct));
        memcpy(strip_ptr, &info, sizeof(info));
        strip_ptr += sizeof(info);
        memcpy(strip_ptr, strip_indices.data(), strip_indices.size() * sizeof(IndexType));
        strip_offset += total_size;
        ++num_strips;
    }

    void repeat(int n, int num_strips, int sparse_ublock_idx_bits)
    {
        if (n <= 1 or not prev_strip_ptr)
            return;

        prev_strip_ptr->f.last_strip_in_row = true;

        // Calculate the new size
        //   1. See if we can fit within the existing allocation
        //   2. If not, we span tiles so we just duplicate at tile granularity.
        //      Otherwise we'd have to go back through and calculate exactly
        //      where strips begin/end and update prev_strip_ptr's accordingly
        //      which adds a lot of complexity and bug potential.
        std::size_t orig_size = strip_offset;
        std::size_t new_size = orig_size * n;
        std::uint8_t* end_ptr = base_ptr() + orig_size;
        std::ptrdiff_t from_end = end_ptr - reinterpret_cast<std::uint8_t*>(prev_strip_ptr);
        TT_ASSERT(from_end >= 0);
        if (new_size > allocation_size_bytes())
        {
            prev_strip_ptr->f.last_strip_in_tile = true;

            orig_size = allocation_size_bytes();
            new_size = orig_size * n;
            end_ptr = base_ptr() + orig_size;
            from_end = end_ptr - reinterpret_cast<std::uint8_t*>(prev_strip_ptr);
            buda_strips.resize(new_size / sizeof(std::uint32_t), 0);
        }

        // Duplicate the section N-1 times
        for (int i = 1; i < n; ++i)
        {
            std::uint8_t* src = base_ptr();
            std::uint8_t* dst = base_ptr() + orig_size * i;
            memcpy(dst, src, orig_size);
            patch_strip_indices(dst, orig_size, num_strips * i, sparse_ublock_idx_bits);
        }

        // Update the prev_strip_ptr to point at the new end
        TT_ASSERT(static_cast<std::size_t>(from_end) <= new_size);
        strip_offset = new_size;
        prev_strip_ptr = reinterpret_cast<strip_info_struct*>(base_ptr() + strip_offset - from_end);
        num_strips *= n;

        TT_ASSERT(
            (reinterpret_cast<std::uintptr_t>(prev_strip_ptr) % alignof(strip_info_struct)) == 0,
            reinterpret_cast<std::uintptr_t>(prev_strip_ptr),
            alignof(strip_info_struct));
    }

    std::pair<std::vector<std::int32_t>, int> finish_buda_strips() const
    {
        if (prev_strip_ptr)
        {
            prev_strip_ptr->f.last_strip_in_row = true;
            prev_strip_ptr->f.last_strip_in_tile = true;
        }
        return std::make_pair(buda_strips, num_strips);
    }

   private:
    std::uint8_t* base_ptr() { return reinterpret_cast<std::uint8_t*>(buda_strips.data()); }
    std::size_t allocation_size_bytes() const { return buda_strips.size() * sizeof(std::uint32_t); }
    std::size_t allocate_tile()
    {
        auto start = allocation_size_bytes();
        buda_strips.resize(buda_strips.size() + kTileSize / sizeof(std::uint32_t), 0);
        return start;
    }

    static void patch_strip_indices(
        std::uint8_t* base, std::size_t size, std::size_t strip_offset, int sparse_ublock_idx_bits)
    {
        using IndexType = std::remove_extent_t<decltype(strip_info_struct::F::index_array)>;

        constexpr std::size_t kElemSize = sizeof(std::uint32_t);
        constexpr int kTileElems = TILE_DIM * TILE_DIM;
        constexpr int kTileSizeBytes = kTileElems * kElemSize;
        int num_tiles = static_cast<int>((size + kTileSizeBytes - 1) / kTileSizeBytes);
        int ublock_tile_index_bytes = 16 - sparse_ublock_idx_bits;

        for (int tile_id = 0; tile_id < num_tiles; ++tile_id)
        {
            strip_info_struct* info = reinterpret_cast<strip_info_struct*>(base + tile_id * kTileSizeBytes);

            bool done = false;
            while (not done and reinterpret_cast<std::uint8_t*>(info) < (base + size))
            {
                TT_ASSERT(
                    (reinterpret_cast<std::uintptr_t>(info) % alignof(strip_info_struct)) == 0,
                    reinterpret_cast<std::uintptr_t>(info),
                    alignof(strip_info_struct));

                // This entire function body is all to facilitate this single
                // line which just bumps the strip index by the new offset
                info->set_strip_index(info->strip_index() + strip_offset);

                int i = 0;
                for (int ublock_i = 0; ublock_i < info->f.nz_ublocks; ++ublock_i)
                {
                    IndexType encoded = info->f.index_array[i++];
                    IndexType nz_tiles_in_ublock = encoded >> sparse_ublock_idx_bits;
                    nz_tiles_in_ublock =
                        (nz_tiles_in_ublock == 0u) ? (1u << ublock_tile_index_bytes) : nz_tiles_in_ublock;
                    i += nz_tiles_in_ublock;
                }

                done = info->f.last_strip_in_tile;
                info = reinterpret_cast<strip_info_struct*>(
                    reinterpret_cast<std::uint8_t*>(info) + sizeof(strip_info_struct) + i * sizeof(IndexType));
            }
        }
    }

   private:
    std::vector<std::int32_t> buda_strips;
    std::size_t strip_offset = 0;
    strip_info_struct* prev_strip_ptr = nullptr;
    int num_strips = 0;
};

static std::pair<std::vector<std::int32_t>, int> encode_strips(
    std::vector<SparseIndex> const& indices,
    std::int64_t dimz,
    std::int64_t dimc,
    int u_rt,
    int u_kt,
    int sparse_tile_ptr_bits,
    int sparse_ublock_idx_bits,
    int t_factor_r,
    int t_factor_c)
{
    if (env_as<bool>("PYBUDA_SPARSE_MM_ENCODE_ALL_STRIPS"))
    {
        throw std::runtime_error("Encoding all strips for sparse matmul is unsupported currently.");  // svuckovic
    }

    // Remove when this is resolved: tenstorrent/pybuda#842
    bool allow_illegal_sparse_pars = env_as<bool>("PYBUDA_ALLOW_ILLEGAL_SPARSE_PARS");

    std::int64_t kt = (dimc + TILE_DIM - 1) / TILE_DIM;
    std::int64_t m_k = kt / u_kt;

    // Calculate bits needed for ublock (u_rt + u_kt separately encoded)
    int u_rt_bits = get_u_rt_encoding_bits(u_rt);
    int u_kt_bits = get_u_kt_encoding_bits(u_kt);
    int ublock_tile_index_bits = 16 - sparse_tile_ptr_bits;
    TT_ASSERT(u_rt_bits + u_kt_bits <= ublock_tile_index_bits);
    int nz_tiles_in_ublock_bits = 16 - sparse_ublock_idx_bits;

    using IndexType = std::remove_extent_t<decltype(strip_info_struct::F::index_array)>;

    // Encodes ublock header
    // 16b total
    // Example:
    //   - sparse_ublock_idx_bits = 7
    //   - nz_tiles_in_ublock_bits = 16 - sparse_ublock_idx_bits = 9
    // If we look at the 16 bits for the above example, it would look like this:
    //                     MSB [nnnnnnnnn|sssssss] LSB
    //                          ^^^^^^^^^ ^^^^^^^
    //                              |        |
    //    nz_tiles_in_ublock_bits <-|        |
    //     sparse_ublock_idx_bits <----------|
    //
    auto encode_ublock_header = [sparse_ublock_idx_bits, nz_tiles_in_ublock_bits](
                                    IndexType current_ublock_index, IndexType nz_tiles_in_ublock) -> IndexType
    {
        // Check if bits exceeded for current_ublock_index
        //
        if (current_ublock_index >= (1 << sparse_ublock_idx_bits))
        {
            throw std::runtime_error(
                fmt::format("UBlock index {} exceeds {} bit encoding", current_ublock_index, sparse_ublock_idx_bits));
        }

        // Check if bits exceeded for nz_tile_in_ublock
        // Note: if nz_tiles_in_ublock is (1 << nz_tiles_in_ublock_bits), this is legal, we encode it with 0
        //
        if (nz_tiles_in_ublock > (1 << nz_tiles_in_ublock_bits))
        {
            throw std::runtime_error(fmt::format(
                "UBlock index {} exceeds {} bit encoding", current_ublock_index, sparse_ublock_idx_bits));
        }

        // Use 0 to represent (1 << nz_tiles_in_ublock_bits)
        nz_tiles_in_ublock = (nz_tiles_in_ublock == (1 << nz_tiles_in_ublock_bits)) ? 0 : nz_tiles_in_ublock;

        IndexType encoded = 0;
        encoded |= nz_tiles_in_ublock << sparse_ublock_idx_bits;
        encoded |= current_ublock_index;
        return encoded;
    };

    // Encodes indices of in0
    // 16b total
    // Example:
    //   - sparse_tile_ptr_bits = 5
    //   - ublock_tile_index_bits = 16 - sparse_tile_ptr_bits = 11
    //   - u_rt_bits = 3
    //   - u_kt_bits = 6
    // If we look at the 16 bits for the above example, it would look like this:
    //                     MSB [sssss|xx|rrr|kkkkkk] LSB
    //                          ^^^^^ ^^ ^^^ ^^^^^^
    //                            |   |   |    |
    //     sparse_tile_ptr_bits <-|   |   |    |
    //              unused bits <-----|   |    |
    //                u_rt_bits <---------|    |
    //                u_kt_bits <--------------|
    //
    // Note: ublock_tile_index_bits is a union of (u_rt_bits,  u_kt_bits, unused bits)
    //
    auto encode_index_pair = [sparse_tile_ptr_bits, ublock_tile_index_bits, u_kt_bits](
                                 IndexType in0, IndexType in0_rt, IndexType in0_ct) -> IndexType
    {
        // Check that sparse tile ptr index (in0) fits in the number of bits we have (sparse_tile_ptr_bits)
        //
        if (in0 >= (1u << sparse_tile_ptr_bits))
        {
            throw std::runtime_error(fmt::format("in0 exceeds {} bit sparse encoding", sparse_tile_ptr_bits));
        }

        IndexType encoded = 0;
        encoded |= in0 << ublock_tile_index_bits;
        encoded |= in0_rt << u_kt_bits;
        encoded |= in0_ct;
        return encoded;
    };

    StripAllocator allocator;
    std::vector<IndexType> strip_indices;
    int indices_len = (int)indices.size();
    int curr_idx = 0;
    int prev_z = 0;
    int curr_strip_index = 0;
    int prev_strip_index = 0;
    std::uint16_t nz_ublocks_in_strip = 0;
    while (curr_idx < indices_len)
    {
        int start_idx = curr_idx;

        // Get all indices in current ublock
        curr_idx++;
        while (curr_idx < indices_len && indices[curr_idx].z == indices[curr_idx - 1].z &&
               indices[curr_idx].ubr_idx(u_rt) == indices[curr_idx - 1].ubr_idx(u_rt) &&
               indices[curr_idx].ubc_idx(u_kt) == indices[curr_idx - 1].ubc_idx(u_kt))
        {
            curr_idx++;
        }

        int curr_z = indices[start_idx].z;
        int orig_z = curr_z / t_factor_r;
        curr_strip_index = m_k * orig_z + indices[start_idx].ubc_idx(u_kt);
        TT_ASSERT(
            allow_illegal_sparse_pars or curr_strip_index >= prev_strip_index,
            "Strip index goes backward in t:",
            curr_strip_index,
            prev_strip_index);
        prev_strip_index = curr_strip_index;
        int curr_ublock_r = indices[start_idx].ubr_idx(u_rt);
        int next_z = (curr_idx < indices_len) ? indices[curr_idx].z : curr_z;
        int next_strip_index = (curr_idx < indices_len) ? indices[curr_idx].ubc_idx(u_kt) : -1;

        // Add missing Zs
        for (; prev_z < curr_z; ++prev_z)
        {
            allocator.push_strip(strip_info_struct(curr_strip_index, 0, true), {});
        }

        // Encode ublock
        nz_ublocks_in_strip++;
        int nz_tiles = curr_idx - start_idx;
        strip_indices.push_back(encode_ublock_header(curr_ublock_r, nz_tiles));
        for (int idx = start_idx; idx < curr_idx; idx++)
        {
            SparseIndex const& si = indices[idx];
            strip_indices.push_back(encode_index_pair(indices[idx].unique_id, si.rt % u_rt, si.ct % u_kt));
        }

        // On strip change...
        if (curr_strip_index != next_strip_index or curr_z != next_z)
        {
            bool last_index = curr_idx >= indices_len;
            bool is_last_strip_in_row = curr_z != next_z or last_index;
            allocator.push_strip(
                strip_info_struct(curr_strip_index, nz_ublocks_in_strip, is_last_strip_in_row), strip_indices);

            // Reset state for next strip
            nz_ublocks_in_strip = 0;
            strip_indices.clear();

            prev_z += int(is_last_strip_in_row);
        }
    }

    // Add missing Zs
    for (; prev_z < (dimz * t_factor_r); ++prev_z)
    {
        // Wait for the last input strip to push the final z
        int orig_z = prev_z / t_factor_r;
        curr_strip_index = m_k * (orig_z + 1) - 1;
        allocator.push_strip(strip_info_struct(curr_strip_index, 0, true), {});
    }

    allocator.repeat(t_factor_c, m_k * dimz, sparse_ublock_idx_bits);

    return allocator.finish_buda_strips();
}

static void print_info_indices(
    std::vector<std::int32_t> const& buda_indices, int sparse_ublock_idx_bits)
{
    using IndexType = std::remove_extent_t<decltype(strip_info_struct::F::index_array)>;
    int ublock_tile_index_bytes = 16 - sparse_ublock_idx_bits;
    std::uint8_t const* base_ptr = reinterpret_cast<std::uint8_t const*>(buda_indices.data());
    TT_ASSERT((int)buda_indices.size() % (TILE_DIM * TILE_DIM) == 0);
    for (int tile_id = 0; tile_id < (int)(buda_indices.size() / (TILE_DIM * TILE_DIM)); ++tile_id)
    {
        fmt::print("tile[{}]\n", tile_id);
        strip_info_struct const* info = reinterpret_cast<strip_info_struct const*>(
            base_ptr + tile_id * (TILE_DIM * TILE_DIM * sizeof(std::uint32_t)));

        bool done = false;
        while (not done)
        {
            fmt::print(
                "  [0x{:04x}] strip_info_struct{{ .strip_index = {}, nz_ublocks = {}, .last_strip_in_row = {}, "
                ".last_strip_in_tile "
                "= {} }}\n",
                std::ptrdiff_t((std::uint8_t const*)info - base_ptr),
                info->strip_index(),
                info->f.nz_ublocks,
                info->f.last_strip_in_row,
                info->f.last_strip_in_tile);

            int i = 0;
            for (int ublock_i = 0; ublock_i < info->f.nz_ublocks; ++ublock_i)
            {
                IndexType encoded = info->f.index_array[i++];
                IndexType ublock_index = encoded & ((1u << sparse_ublock_idx_bits) - 1u);
                IndexType nz_tiles_in_ublock = encoded >> sparse_ublock_idx_bits;
                nz_tiles_in_ublock = (nz_tiles_in_ublock == 0u) ? (1u << ublock_tile_index_bytes) : nz_tiles_in_ublock;
                fmt::print("    ublock_index[{}]\n", ublock_index);
                fmt::print("    nz_tiles_in_ublock[{}]\n", nz_tiles_in_ublock);
                for (int j = 0; j < nz_tiles_in_ublock; ++j)
                {
                    encoded = info->f.index_array[i++];
                    IndexType in0 = encoded >> ublock_tile_index_bytes;
                    IndexType in1 = encoded & ((1u << ublock_tile_index_bytes) - 1);
                    fmt::print("      in0[{}] in1[{}]\n", in0, in1);
                }
            }

            done = info->f.last_strip_in_tile;
            info = reinterpret_cast<strip_info_struct const*>(
                reinterpret_cast<std::uint8_t const*>(info) + sizeof(strip_info_struct) + i * sizeof(IndexType));
        }
    }
}

SparseBUDA::SparseBUDA(
    std::vector<SparseCOO> sparse_zs,
    std::vector<SparseIndex> sparse_indices,
    std::vector<std::int64_t> sparse_shape,
    std::vector<float> sparse_uniq_tiles,
    int zdim,
    int bcast_factor,
    int fracture_factor) :
    sparse_zs(sparse_zs),
    sparse_indices(sparse_indices),
    sparse_shape(sparse_shape),
    sparse_uniq_tiles(sparse_uniq_tiles),
    zdim(zdim),
    bcast_factor(bcast_factor),
    fracture_factor(fracture_factor)
{
    TT_ASSERT(sparse_shape.size() == 2, "Expected sparse_shape to have dim length of 2");

    TT_ASSERT(sparse_shape[1] % TILE_DIM == 0);
    std::uint32_t sparse_ct = sparse_shape[1] / TILE_DIM;
    TT_ASSERT(sparse_ct < SparseBUDA::kMaxSparseIndexValue, "Sparse matrix too wide");
}

enum EncodingBitErrors
{
    MaxSparseTilesExceeded = -1,
    MaxUBlocksRExceeded = -2
};

// Returns negative value if failed
//
int SparseBUDA::get_sparse_tile_ptr_bits(int grid_r, int t_factor_r, int u_rt) const
{
    // TODO: num_sparse_tiles should be calculated per core, and max should be used as the result of this fn
    // TODO: also account for fracture factor!

    std::uint32_t num_sparse_tiles = (std::uint32_t)(sparse_uniq_tiles.size() / (TILE_DIM * TILE_DIM));
    TT_ASSERT(num_sparse_tiles > 0);
    if (num_sparse_tiles > SparseBUDA::kMaxSparseTiles)
    {
        return MaxSparseTilesExceeded;
    }

    // TODO: This can be divided by fracture factor
    std::uint32_t max_ublocks_r = this->sparse_shape[0] / (TILE_DIM * grid_r * t_factor_r * u_rt);
    if (max_ublocks_r > SparseBUDA::kMaxUblocksR)
    {
        return MaxUBlocksRExceeded;
    }

    std::uint32_t max_num = num_sparse_tiles;
    int num_lz = 32 - __builtin_clz(max_num);
    return num_lz;
}

// Returns negative value if failed
//
int SparseBUDA::get_sparse_ublock_idx_bits(int grid_r, int t_factor_r, int u_rt) const
{
    // TODO: num_sparse_tiles should be calculated per core, and max should be used as the result of this fn
    // TODO: also account for fracture factor!

    std::uint32_t num_sparse_tiles = (std::uint32_t)(sparse_uniq_tiles.size() / (TILE_DIM * TILE_DIM));
    TT_ASSERT(num_sparse_tiles > 0);
    if (num_sparse_tiles > SparseBUDA::kMaxSparseTiles)
    {
        return MaxSparseTilesExceeded;
    }

    // TODO: This can be divided by fracture factor
    std::uint32_t max_ublocks_r = this->sparse_shape[0] / (TILE_DIM * grid_r * t_factor_r * u_rt);
    if (max_ublocks_r > SparseBUDA::kMaxUblocksR)
    {
        return MaxUBlocksRExceeded;
    }

    std::uint32_t max_num = std::max(num_sparse_tiles, max_ublocks_r);
    int num_lz = 32 - __builtin_clz(max_num);
    return num_lz;
}


int SparseBUDA::get_max_u_kt(int grid_r, int t_factor_r, int u_rt, int sparse_tile_ptr_bits) const
{
    int ublock_bits = 16;
    if (sparse_tile_ptr_bits == 0)
    {
        sparse_tile_ptr_bits = this->get_sparse_tile_ptr_bits(grid_r, t_factor_r, u_rt);
    }

    ublock_bits -= sparse_tile_ptr_bits;
    TT_ASSERT(ublock_bits > 0);
    int u_rt_bits = get_u_rt_encoding_bits(u_rt);
    ublock_bits -= u_rt_bits;
    TT_ASSERT(ublock_bits > 0);
    return (1 << ublock_bits);
}

SparseBUDA::Layout SparseBUDA::create_layout(bool z_major, int fracture_factor)
{
    Layout layout = Layout::Default;
    if (z_major and (fracture_factor == 1) and not env_as<bool>("PYBUDA_SPARSE_DISABLE_LAYOUT_DATAFLOW"))
        layout = Layout::ZMajorDataflow;
    else if (z_major)
        layout = Layout::ZMajor;
    return layout;
}

static std::vector<SparseCOO> vslice_layout(
    SparseCOO const& sparse, int grid_r, int t_factor_r, int bcast_factor, SparseBUDA::Layout layout)
{
    if (layout == SparseBUDA::Layout::Default)
        return sparse.vslice(grid_r * t_factor_r);

    if (layout == SparseBUDA::Layout::ZMajorDataflow and ((sparse.rt() / grid_r / t_factor_r) % bcast_factor != 0))
        return {};

    int dflow_factor =
        (layout == SparseBUDA::Layout::ZMajorDataflow) ? (sparse.rt() / grid_r / t_factor_r / bcast_factor) : 1;
    std::vector<SparseCOO> vsliced = sparse.vslice(grid_r * t_factor_r * bcast_factor * dflow_factor);
    std::vector<SparseCOO> slices;
    slices.reserve(grid_r * t_factor_r);

    for (int t = 0; t < t_factor_r; t++)
    {
        std::vector<SparseCOO> b_slices;
        b_slices.reserve(grid_r * dflow_factor * bcast_factor);

        if (layout == SparseBUDA::Layout::ZMajorDataflow)
        {
            for (int r = 0; r < grid_r; r++)
            {
                for (int d = 0; d < dflow_factor; d++)
                {
                    for (int b = 0; b < bcast_factor; b++)
                    {
                        int idx =
                            b * t_factor_r * grid_r * dflow_factor + t * grid_r * dflow_factor + r * dflow_factor + d;
                        b_slices.push_back(vsliced[idx]);
                    }
                }
            }
        }
        else
        {
            TT_ASSERT(layout == SparseBUDA::Layout::ZMajor);
            for (int b = 0; b < bcast_factor; b++)
            {
                for (int r = 0; r < grid_r; r++)
                {
                    int idx = b * t_factor_r * grid_r + t * grid_r + r;
                    b_slices.push_back(vsliced[idx]);
                }
            }
        }

        for (int r = 0; r < grid_r; r++)
        {
            auto begin = b_slices.begin() + (r * bcast_factor * dflow_factor);
            slices.push_back(SparseCOO::vcat(begin, begin + bcast_factor * dflow_factor));
        }
    }

    return slices;
}

std::unordered_map<int, std::vector<int>> SparseBUDA::get_par_t_values(
    int grid_r,
    std::vector<int> potential_ts,
    std::vector<SparseCOO>& sparse_zs,
    std::vector<int> u_kts,
    int bcast_factor,
    int fracture_factor,
    Layout layout)
{
    TT_ASSERT(!sparse_zs.empty(), "Expected z >= 1");
    TT_ASSERT(sparse_zs[0].shape.size() == 2, "Expected sparse z to have 2 dims");

    log_trace(tt::LogGraphCompiler, "Potential ts: {}", fmt::join(potential_ts, ","));
    if (potential_ts.empty() or (potential_ts.size() == 1 and potential_ts[0] == 1))
    {
        return {{1, {}}};
    }

    // Remove when this is resolved: tenstorrent/pybuda#842
    bool allow_illegal_sparse_pars = env_as<bool>("PYBUDA_ALLOW_ILLEGAL_SPARSE_PARS");

    // Sort descending, we want to test the bigger ts first
    // If a t is valid, all its factors are valid too - we can skip testing them
    std::sort(potential_ts.begin(), potential_ts.end(), [](const int a, const int b) { return a > b; });

    // Fracture factor is like having multiple grid_r's in flight
    int virtual_grid_r = grid_r * fracture_factor;

    std::set<int> ts;
    std::unordered_map<int, std::vector<int>> t_to_legal_u_kts = {{1, u_kts}};
    for (size_t idx = 0; idx < sparse_zs.size(); idx++)
    {
        SparseCOO& sparse = sparse_zs[idx];
        sparse.sort(SparseCOO::SortOrder::COLUMN_MAJOR);

        std::set<int> curr_ts = {1};

        for (int t : potential_ts)
        {
            if (curr_ts.count(t))
            {
                continue;
            }

            std::vector<SparseCOO> slices = vslice_layout(sparse, virtual_grid_r, t, bcast_factor, layout);

            // slice.empty() means we failed to create the desired layout for the given parameters
            if (slices.empty())
                continue;

            std::vector<int> legal_u_kts = u_kts;
            int slices_per_core = slices.size() / virtual_grid_r;

            for (int g_r = 0; g_r < virtual_grid_r; g_r++)
            {
                for (int slice_idx = 0; slice_idx < slices_per_core - 1; slice_idx++)
                {
                    int next_idx = slice_idx + 1;
                    const SparseCOO* curr_slice = &slices[slice_idx * virtual_grid_r + g_r];
                    const SparseCOO* next_slice = &slices[next_idx * virtual_grid_r + g_r];

                    while (next_slice->vals.empty() and ++next_idx < slices_per_core)
                        next_slice = &slices[next_idx * virtual_grid_r + g_r];

                    if (curr_slice->vals.empty() or next_slice->vals.empty())
                    {
                        // Special case, there's no values in curr slice
                        continue;
                    }

                    std::int64_t curr_col_bounds_max =
                        allow_illegal_sparse_pars ? curr_slice->cols.back() : curr_slice->col_bounds.max;
                    std::int64_t next_col_bounds_min =
                        allow_illegal_sparse_pars ? next_slice->cols.front() : next_slice->col_bounds.min;

                    // Next slice's first column needs to be in a strip that's after the last strip in current slice
                    std::vector<std::size_t> remove_u_kts;
                    remove_u_kts.reserve(legal_u_kts.size());
                    for (std::size_t i = 0; i < legal_u_kts.size(); ++i)
                    {
                        int u_kt = legal_u_kts[i];
                        if ((curr_col_bounds_max / TILE_DIM) / u_kt > (next_col_bounds_min / TILE_DIM) / u_kt)
                        {
                            remove_u_kts.push_back(i);
                        }
                    }
                    for (auto iter = remove_u_kts.rbegin(); iter != remove_u_kts.rend(); ++iter)
                        legal_u_kts.erase(legal_u_kts.begin() + *iter);
                    if (legal_u_kts.empty())
                        break;
                }
                if (legal_u_kts.empty())
                    break;
            }

            bool is_valid_t = not legal_u_kts.empty();
            if (is_valid_t)
            {
                auto match = t_to_legal_u_kts.find(t);
                if (match == t_to_legal_u_kts.end())
                {
                    t_to_legal_u_kts[t] = legal_u_kts;
                }
                else
                {
                    std::vector<int> intersection;
                    std::set_intersection(
                        legal_u_kts.begin(),
                        legal_u_kts.end(),
                        match->second.begin(),
                        match->second.end(),
                        std::back_inserter(intersection));
                    match->second = intersection;
                }

                if (not t_to_legal_u_kts[t].empty())
                    curr_ts.insert(t);
            }
        }

        if (idx == 0)
        {
            ts.insert(curr_ts.begin(), curr_ts.end());
        }
        else
        {
            // Emulating an in-place set-intersection, C++ doesn't have a great solution here :/
            std::set<int> intersection;
            std::set_intersection(
                ts.begin(), ts.end(), curr_ts.begin(), curr_ts.end(), std::inserter(intersection, intersection.end()));
            ts = intersection;

            if (ts.size() == 1)
            {
                return {{1, {}}};
            }
        }
    }

    // Erase ts from the map that might have been intersected out
    for (int t : ts)
    {
        auto match = t_to_legal_u_kts.find(t);
        if (match != t_to_legal_u_kts.end() and match->second.empty())
            t_to_legal_u_kts.erase(match);
    }
    TT_ASSERT(not t_to_legal_u_kts.empty());

    return t_to_legal_u_kts;
}

std::tuple<SparseTiles, EncodingTiles, std::vector<uint32_t>, std::vector<uint32_t>, std::vector<int>>
SparseBUDA::get_sparse_tiles_and_encodings(
    int grid_r,
    int t_factor_r,
    int t_factor_c,
    int u_rt,
    int u_kt,
    int fracture_factor,
    Layout layout,
    std::string const& visualize_sparse_path) const
{
    int sparse_tile_ptr_bits = get_sparse_tile_ptr_bits(grid_r, t_factor_r, u_rt);
    int sparse_ublock_idx_bits = get_sparse_ublock_idx_bits(grid_r, t_factor_r, u_rt);

    TT_ASSERT(sparse_tile_ptr_bits > 0 and sparse_ublock_idx_bits > 0);

    // Calculate bits needed for ublock (u_rt + u_kt separately encoded)
    int u_rt_bits = get_u_rt_encoding_bits(u_rt);
    int u_kt_bits = get_u_kt_encoding_bits(u_kt);
    TT_ASSERT(sparse_tile_ptr_bits + u_rt_bits + u_kt_bits <= 16);

    std::function<bool(tt::sparse::SparseIndex const&, tt::sparse::SparseIndex const&)> sp_indices_cmp_fn =
        [u_rt, u_kt](SparseIndex const& a, SparseIndex const& b) { return comp_zcr_ublocked(a, b, u_rt, u_kt); };

    int zdim = this->sparse_zs.size();
    // Fracture factor is like having multiple grid_r's in flight
    int virtual_grid_r = grid_r * fracture_factor;

    SparseTiles sparse_tiles;
    EncodingTiles buda_indices;
    std::vector<int> num_strips_per_row;

    std::vector<SparseCOO> slice_ztr;  // |z * t * r * b|
    slice_ztr.reserve(zdim * t_factor_r * virtual_grid_r * ((layout == Layout::Default) ? 1 : bcast_factor));
    for (int z = 0; z < zdim; z++)
    {
        auto sparse = sparse_zs[z];
        sparse.sort(SparseCOO::SortOrder::ROW_MAJOR);

        std::vector<SparseCOO> slices = vslice_layout(sparse, virtual_grid_r, t_factor_r, bcast_factor, layout);
        TT_ASSERT(not slices.empty());
        slice_ztr.insert(slice_ztr.end(), slices.begin(), slices.end());
    }

    std::vector<SparseCOO> visualize_sparse_tensors;
    for (int g_r = 0; g_r < virtual_grid_r; g_r++)
    {
        std::vector<SparseCOO> curr_slice_ts;
        for (size_t idx = g_r; idx < slice_ztr.size(); idx += virtual_grid_r)
        {
            curr_slice_ts.push_back(slice_ztr[idx]);
        }

        if (not visualize_sparse_path.empty())
        {
            visualize_sparse_tensors.insert(visualize_sparse_tensors.end(), curr_slice_ts.begin(), curr_slice_ts.end());
        }

        auto [curr_sparse_indices, curr_tiles] = compress_unique_tiles(curr_slice_ts);
        std::sort(curr_sparse_indices.begin(), curr_sparse_indices.end(), sp_indices_cmp_fn);

        sparse_tiles.push_back(curr_tiles);
        auto [encoded, num_strips] = encode_strips(
            curr_sparse_indices,
            zdim,
            curr_slice_ts[0].shape[1],
            u_rt,
            u_kt,
            sparse_tile_ptr_bits,
            sparse_ublock_idx_bits,
            t_factor_r,
            t_factor_c);
        buda_indices.push_back(encoded);
        num_strips_per_row.push_back(num_strips);
        if (env_as<bool>("PYBUDA_SPARSE_PRINT_INDICES"))
        {
            fmt::print("Grid_r[{}] {} {}\n", g_r, layout, t_factor_r);
            print_info_indices(buda_indices.back(), sparse_ublock_idx_bits);
        }
    }

    if (not visualize_sparse_path.empty())
    {
        namespace py = pybind11;
        auto sparse_utils_module = py::module_::import("pybuda.op.eval.sparse_utils");
        py::function visualize_sparse = sparse_utils_module.attr("visualize_sparse");
        visualize_sparse(visualize_sparse_tensors, visualize_sparse_path, virtual_grid_r, zdim * t_factor_r);
    }

    // Pad sparse tiles
    uint32_t sparse_max_len = static_cast<uint32_t>(std::max_element(
                                                        sparse_tiles.begin(),
                                                        sparse_tiles.end(),
                                                        [](const std::vector<float>& lhs, const std::vector<float>& rhs)
                                                        { return lhs.size() < rhs.size(); })
                                                        ->size());
    for (size_t idx = 0; idx < sparse_tiles.size(); idx++)
    {
        sparse_tiles[idx].insert(sparse_tiles[idx].end(), sparse_max_len - sparse_tiles[idx].size(), 0);
    }

    // Pad buda indices
    uint32_t indices_max_len =
        static_cast<uint32_t>(std::max_element(
                                  buda_indices.begin(),
                                  buda_indices.end(),
                                  [](const std::vector<std::int32_t>& lhs, const std::vector<std::int32_t>& rhs)
                                  { return lhs.size() < rhs.size(); })
                                  ->size());
    // round up to make divisible by 1024
    indices_max_len = (indices_max_len + TILE_DIM * TILE_DIM - 1) / (TILE_DIM * TILE_DIM) * (TILE_DIM * TILE_DIM);
    for (size_t idx = 0; idx < buda_indices.size(); idx++)
    {
        buda_indices[idx].insert(buda_indices[idx].end(), indices_max_len - buda_indices[idx].size(), 0);
    }

    std::vector<uint32_t> sparse_shape = {1, 1, grid_r * TILE_DIM, sparse_max_len * fracture_factor / TILE_DIM};
    std::vector<uint32_t> encodings_shape = {1, 1, grid_r * TILE_DIM, indices_max_len * fracture_factor / TILE_DIM};

    return std::make_tuple<>(sparse_tiles, buda_indices, sparse_shape, encodings_shape, num_strips_per_row);
}

int SparseBUDA::get_encoding_tiles_per_core_estimate(int grid_r, int t_factor_r, int u_rt, int u_kt) const
{
    // strip index (with last_* bits)   4b
    // number of ublocks                2b
    // [
    //   ublock_index (lower sparse_tile_bits) + num_matmuls     2b (shared)
    //   [
    //       tile_index + sparse_index (upper sparse_tile_bits)  2b (shared)
    //   ]
    // ]

    // ublock_index -> 10, num_matmuls -> 6
    // tile_index -> 6, sparse_index -> 10

    bool encode_all_strips = env_as<bool>("PYBUDA_SPARSE_MM_ENCODE_ALL_STRIPS");

    int tile_bytes = TILE_DIM * TILE_DIM * 4;  // Using RawUInt32 for encoding tiles
    TT_ASSERT(sparse_shape[0] % TILE_DIM == 0);
    TT_ASSERT(sparse_shape[1] % TILE_DIM == 0);
    int sparse_rt = sparse_shape[0] / TILE_DIM;
    int sparse_ct = sparse_shape[1] / TILE_DIM;
    int n_strips = sparse_ct / u_kt;
    int zt = this->zdim * t_factor_r;
    std::vector<int> space(grid_r);

    TT_ASSERT(sparse_rt % grid_r == 0);
    TT_ASSERT((sparse_rt / grid_r) % u_rt == 0);
    std::vector<std::vector<std::set<int>>> used_ublocks(
        grid_r, std::vector<std::set<int>>(zt));     // [curr_grid_row, curr_zt] -> set<ublocks with nz matmuls>
    std::vector<int> num_nonzero_strips(grid_r, 0);  // [curr_grid_row] -> num nz strips
    std::vector<std::vector<std::map<int, int>>> num_nz_ublocks(
        grid_r, std::vector<std::map<int, int>>(zt));  // [curr_grid_row, curr_zt, strip idx] -> num nz blocks
    std::vector<std::vector<std::map<int, int>>> num_nz_tiles(
        grid_r, std::vector<std::map<int, int>>(zt));  // [curr_grid_row, curr_zt, strip idx] -> num nz blocks

    int t_height = sparse_rt / t_factor_r;
    for (size_t i_nzt = 0; i_nzt < this->sparse_indices.size(); i_nzt++)
    {
        int z_ind = this->sparse_indices[i_nzt].z;
        int t_ind = this->sparse_indices[i_nzt].rt / t_height;
        int zt_ind = z_ind * t_factor_r + t_ind;
        int rt_in_t = this->sparse_indices[i_nzt].rt % t_height;
        int rt_cnt_per_core = t_height / grid_r;
        int strip_ind = this->sparse_indices[i_nzt].ct / u_kt;
        int ublock_row_in_core = (rt_in_t % rt_cnt_per_core) / u_rt;
        int core_index = rt_in_t / rt_cnt_per_core;
        num_nz_tiles[core_index][zt_ind][strip_ind]++;
        int block_key = ublock_row_in_core * n_strips + strip_ind;

        if (used_ublocks[core_index][zt_ind].find(block_key) == used_ublocks[core_index][zt_ind].end())
        {
            used_ublocks[core_index][zt_ind].insert(block_key);
            num_nz_ublocks[core_index][zt_ind][strip_ind]++;
        }
    }

    int increment;
    int max_space = 0;
    for (int idx_r = 0; idx_r < grid_r; idx_r++)
    {
        for (int idx_zt = 0; idx_zt < zt; idx_zt++)
        {
            int prev_strip_idx = -1;
            bool first_strip_encoded = false;
            for (auto const& [key, val] : num_nz_ublocks[idx_r][idx_zt])
            {
                if (encode_all_strips)
                {
                    while (prev_strip_idx + 1 < key)
                    {
                        if (space[idx_r] % tile_bytes + 6 > tile_bytes)
                        {
                            space[idx_r] = (space[idx_r] / tile_bytes + 1) * tile_bytes;
                        }

                        int leftover_strips_to_add = (tile_bytes - (space[idx_r] % tile_bytes)) / 6;
                        int dist = key - prev_strip_idx - 1;
                        space[idx_r] += std::min(leftover_strips_to_add, dist) * 6;
                        prev_strip_idx += std::min(leftover_strips_to_add, dist);
                    }
                }
                if (key == 0)
                {
                    first_strip_encoded = true;
                }

                increment = 6 + 2 * (val + num_nz_tiles[idx_r][idx_zt][key]);

                if (increment < tile_bytes)
                {
                    if (space[idx_r] % tile_bytes + increment > tile_bytes)
                    {
                        space[idx_r] = (space[idx_r] / tile_bytes + 1) * tile_bytes;
                    }
                    space[idx_r] += increment;
                }
                else
                {
                    throw std::runtime_error(fmt::format(
                        "Expected strip encoding to fit single tile of size {}, got {}.", tile_bytes, increment));
                }
                prev_strip_idx = key;  // update prev strip idx
            }

            if (encode_all_strips)
            {
                while (prev_strip_idx + 1 < n_strips)
                {
                    if (space[idx_r] % tile_bytes + 6 > tile_bytes)
                    {
                        space[idx_r] = (space[idx_r] / tile_bytes + 1) * tile_bytes;
                    }

                    int leftover_strips_to_add = (tile_bytes - (space[idx_r] % tile_bytes)) / 6;
                    int dist = n_strips - prev_strip_idx - 1;
                    space[idx_r] += std::min(leftover_strips_to_add, dist) * 6;
                    prev_strip_idx += std::min(leftover_strips_to_add, dist);
                }
            }
            else
            {
                int added_space = 0;  // this due to the fact that we have to encode first and last strip even if
                                      // encode_all_strips is false
                if (prev_strip_idx + 1 < n_strips)
                {
                    added_space += 6;
                }
                if (!first_strip_encoded)
                {
                    added_space += 6;
                }
                if (space[idx_r] % tile_bytes + added_space > tile_bytes)
                {
                    space[idx_r] = (space[idx_r] / tile_bytes + 1) * tile_bytes;
                }
                space[idx_r] += added_space;
            }
        }

        if (space[idx_r] > max_space)
        {
            max_space = space[idx_r];
        }
    }

    return (max_space + tile_bytes - 1) / tile_bytes;
}

int SparseBUDA::get_sparse_tiles_per_core_estimate(int grid_r, int t_factor_r) const
{
    TT_ASSERT(this->sparse_shape[0] / TILE_DIM >= grid_r * t_factor_r);
    TT_ASSERT(this->sparse_shape[0] / TILE_DIM % (grid_r * t_factor_r) == 0);
    TT_ASSERT(this->sparse_shape[0] / TILE_DIM % t_factor_r == 0);

    int t_slice_height = this->sparse_shape[0] / grid_r / t_factor_r / TILE_DIM;
    int rt_slice_height = this->sparse_shape[0] / t_factor_r / TILE_DIM;

    std::vector<std::set<int>> uniq_tiles_per_r = std::vector<std::set<int>>(grid_r);
    for (int idx = 0; idx < grid_r; idx++)
    {
        uniq_tiles_per_r[idx].insert(0);  // Add zero-tiles always
    }
    for (size_t idx = 0; idx < this->sparse_indices.size(); idx++)
    {
        const SparseIndex& si = this->sparse_indices[idx];

        int core_r = (si.rt % rt_slice_height) / t_slice_height;
        uniq_tiles_per_r[core_r].insert(si.unique_id);
    }

    return std::max_element(
               uniq_tiles_per_r.begin(),
               uniq_tiles_per_r.end(),
               [](const std::set<int>& lhs, const std::set<int>& rhs) { return lhs.size() < rhs.size(); })
        ->size();
}

SparseBUDA compress_sparse_tensor_and_strip_info(
    std::vector<SparseCOO> const& sparse_zs, int bcast_factor, int fracture_factor)
{
    int zdim = (int)sparse_zs.size();
    auto [sparse_indices, tiles] = compress_unique_tiles(sparse_zs);
    return SparseBUDA(sparse_zs, sparse_indices, sparse_zs[0].shape, tiles, zdim, bcast_factor, fracture_factor);
}

std::ostream& operator<<(std::ostream& out, const SparseCOO::SortOrder& sort_order)
{
    switch (sort_order)
    {
        case SparseCOO::SortOrder::UNSORTED: out << "SortOrder::UNSORTED"; break;
        case SparseCOO::SortOrder::ROW_MAJOR: out << "SortOrder::ROW_MAJOR"; break;
        case SparseCOO::SortOrder::COLUMN_MAJOR: out << "SortOrder::COLUMN_MAJOR"; break;
        default: throw std::runtime_error(fmt::format("Unexpected sort_order: {}", int(sort_order)));
    }
    return out;
}

}  // namespace tt::sparse
