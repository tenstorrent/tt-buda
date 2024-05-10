// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "balancer/types.hpp"

#include <climits>
#include <cmath>

#include "balancer/balancer_utils.hpp"
#include "balancer/python_interface.hpp"
#include "graph_lib/node_types.hpp"
#include "lower_to_buda/common.hpp"
#include "passes/fuse_ops.hpp"
#include "utils/assert.hpp"
#include "utils/logger.hpp"

namespace tt::balancer
{
std::uint64_t UniqueId::next_id = 0;

TensorShape::TensorShape(graphlib::Shape const &shape) :
    w((int)shape.w()), z((int)shape.z()), rt((int)shape.rt()), ct((int)shape.ct())
{
}

int const &TensorShape::operator[](int i) const { return (*const_cast<TensorShape *>(this))[i]; }

int &TensorShape::operator[](int i)
{
    if (i < 0)
        i += 4;
    TT_ASSERT(i <= 3);
    TT_ASSERT(i >= 0);
    switch (i)
    {
        case 0: return w;
        case 1: return z;
        case 2: return rt;
        case 3: return ct;
        default: TT_ASSERT(false); return w;
    }
}

//
// BlockShape
//

BlockShape::BlockShape(TensorShape tensor_shape, int par_r, int par_c, int par_t, UBlockShape ublock) :
    t(tensor_shape.z * par_t),
    tblock_m(1),
    tblock_n(1),
    mblock_m((tensor_shape.rt / par_r) / ublock.rt),
    mblock_n((tensor_shape.ct / par_c) / ublock.ct),
    ublock(ublock)
{
    TT_LOG_ASSERT(
        (tensor_shape.rt % (par_r * ublock.rt)) == 0,
        "Not divisible on R: {} / ({} * {})",
        tensor_shape.rt,
        par_r,
        ublock.rt);
    TT_LOG_ASSERT(
        (tensor_shape.ct % (par_c * ublock.ct)) == 0,
        "Not divisible on C: {} / ({} * {})",
        tensor_shape.ct,
        par_c,
        ublock.ct);
    TT_ASSERT(mblock_m > 0, "Invalid ublock provided", tensor_shape, par_r, par_c, par_t, ublock);
    TT_ASSERT(mblock_n > 0, "Invalid ublock provided", tensor_shape, par_r, par_c, par_t, ublock);
}

BlockShape::BlockShape(TensorShape tensor_shape, GridShape grid_shape, int mblock_m, int mblock_n, UBlockShape ublock) :
    mblock_m(mblock_m), mblock_n(mblock_n), ublock(ublock)
{
    TT_ASSERT((tensor_shape.rt % mblock_m) == 0);
    TT_ASSERT((tensor_shape.ct % mblock_n) == 0);
    TT_ASSERT((tensor_shape.rt % (mblock_m * ublock.rt)) == 0);
    TT_ASSERT((tensor_shape.ct % (mblock_n * ublock.ct)) == 0);
    TT_ASSERT((tensor_shape.rt % (grid_shape.r * mblock_m * ublock.rt)) == 0);
    TT_ASSERT((tensor_shape.ct % (grid_shape.c * mblock_n * ublock.ct)) == 0);
    TT_ASSERT(((tensor_shape.z * tensor_shape.rt * tensor_shape.ct) % (mblock_m * mblock_n * ublock.volume())) == 0);

    tblock_m = tensor_shape.rt / (grid_shape.r * mblock_m * ublock.rt);
    tblock_n = tensor_shape.ct / (grid_shape.c * mblock_n * ublock.ct);
    t = tensor_shape.z;
}

BlockShape::BlockShape(int t, int mblock_m, int mblock_n, UBlockShape ublock) :
    t(t), tblock_m(1), tblock_n(1), mblock_m(mblock_m), mblock_n(mblock_n), ublock(ublock)
{
}

bool BlockShape::operator==(BlockShape o) const
{
    return (t == o.t) and (tblock_m == o.tblock_m) and (tblock_n == o.tblock_n) and (mblock_m == o.mblock_m) and
           (mblock_n == o.mblock_n) and (ublock == o.ublock);
}

bool BlockShape::operator!=(BlockShape o) const { return !(*this == o); }

int BlockShape::volume() const { return t * tblock_m * tblock_n * mblock_m * mblock_n * ublock.volume(); }
int BlockShape::volume_no_t() const { return mblock_m * mblock_n * ublock.volume(); }

int BlockShape::buffer_tiles(int buffer_factor) const { return buffer_factor * mblock_m * mblock_n * ublock.volume(); }

void BlockShape::set_ublock_shape(UBlockShape new_ublock)
{
    // canonicalize shape first
    mblock_m *= ublock.rt;
    mblock_n *= ublock.ct;
    ublock = {1, 1};

    TT_ASSERT((mblock_m % new_ublock.rt) == 0, *this, new_ublock);
    TT_ASSERT((mblock_n % new_ublock.ct) == 0, *this, new_ublock);

    mblock_m /= new_ublock.rt;
    mblock_n /= new_ublock.ct;
    ublock = new_ublock;
}

BudaBlocks BlockShape::as_buda_blocks() const
{
    BudaBlocks blocks;
    TT_ASSERT(tblock_m == 1 and tblock_n == 1);
    blocks.z = t;
    blocks.ublock_rt = ublock.rt;
    blocks.ublock_ct = ublock.ct;
    blocks.mblock_m = mblock_m;
    blocks.mblock_n = mblock_n;
    return blocks;
}

//
// BufferModel
//
std::size_t BufferModel::size_tiles(bool include_t) const { return l1_size_tiles * (include_t ? block_shape.t : 1); }

std::size_t BufferModel::size_bytes(bool include_t) const
{
    return size_tiles(include_t) * tile_size_bytes(data_format);
}

std::size_t BufferModel::single_buffered_size_tiles() const { return block_shape.buffer_tiles(1); }

std::size_t BufferModel::single_buffered_size_bytes() const
{
    return single_buffered_size_tiles() * tile_size_bytes(data_format);
}

std::size_t BufferModel::total_size_bytes() const { return block_shape.volume() * tile_size_bytes(data_format); }

//
// OpModel
//
std::size_t OpModel::get_l1_memory_usage() const
{
    std::size_t usage = 0;

    for (BufferModel const &buffer_model : input_buffers)
    {
        usage += buffer_model.size_bytes();
    }

    for (BufferModel const &buffer_model : output_buffers)
    {
        bool const include_t = is_gradient_op();
        usage += buffer_model.size_bytes(include_t);
    }

    for (BufferModel const &buffer_model : parameter_buffers)
    {
        constexpr bool include_t = true;
        usage += buffer_model.size_bytes(include_t);
    }

    for (BufferModel const &buffer_model : intermediate_buffers)
    {
        usage += buffer_model.size_bytes();
    }

    // Note: global overlay blob override (TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE) is not included in the usage
    // calculation - when querying L1 space that FE can use, L1 space is reduced by the value of this global overlay
    // blob override
    //
    static constexpr std::int32_t bbe_reserved_blob_size = 64 * 1024;  // 64 kB
    if (overlay_size > bbe_reserved_blob_size)
    {
        usage += overlay_size - bbe_reserved_blob_size;
    }

    return usage;
}

const std::string &OpModel::op_type() const { return buda_op_node->op_name(); }

MathFidelity OpModel::math_fidelity() const
{
    return buda_op_node != nullptr ? buda_op_node->math_fidelity() : MathFidelity::Invalid;
}

bool OpModel::is_gradient_op() const { return buda_op_node->is_gradient_op(); }

bool OpModel::is_matmul() const { return buda_op_node->is_matmul(); }

std::shared_ptr<FusedOp> OpModel::fused_op() const
{
    return buda_op_node->is_fused_op() ? buda_op_node->get_fused_op() : nullptr;
}

const std::string OpModel::get_reduce_dim() const
{
    if (op_type() == "reduce")
    {
        return std::get<std::string>(buda_op_node->op_type().buda_attrs.at("dim"));
    }
    else
    {
        return "";
    }
}

const BudaOpAttrs OpModel::buda_op_attrs() const
{
    TT_ASSERT(buda_op_node, "Called on non-buda op!");
    return buda_op_node->op_type().buda_attrs;
}

int OpModel::get_execution_cycles_uncached(std::string const &arch_name, bool theoretical) const
{
    std::shared_ptr<FusedOp> fused_op = this->fused_op();

    // Calculate sparse matmul metadata and write into OpModel's SparseMetadata struct
    if (env_as<bool>("PYBUDA_TEMP_ENABLE_NEW_SPARSE_ESTIMATES", false) and this->is_sparse_matmul and
        this->sparse_metadata == nullptr)
    {
        auto *p_this = const_cast<OpModel *>(this);
        p_this->sparse_metadata = get_sparse_matmul_metadata(*this);
    }

    if (fused_op == nullptr)
    {
        return tt::balancer::get_execution_cycles(arch_name, *this, theoretical);
    }

    if (env_as<bool>("PYBUDA_TEMP_ENABLE_NEW_FUSED_ESTIMATES", false))
    {
        // to obtain the execution cycles for fused op, we are calculating cycles for each subop, so
        // we need to prepare necessary information and pass it inside the FusedSubOpModel object
        std::vector<FusedSubOpModel> sub_op_models;
        for (const auto &schedule : fused_op->get_schedules())
        {
            for (const auto &sub_op : schedule.ops)
            {
                FusedSubOpModel &sub_op_model = sub_op_models.emplace_back();
                sub_op_model.type = sub_op.op_type.op;

                sub_op_model.mblock_m = sub_op.op_shape.outputs[0].rt /
                                        (grid_shape.r * fused_op_ublock_shape.at(sub_op.name).rt * t_stream_factor.r);
                sub_op_model.mblock_n = sub_op.op_shape.outputs[0].ct /
                                        (grid_shape.c * fused_op_ublock_shape.at(sub_op.name).ct * t_stream_factor.c);
                sub_op_model.ublock_rt = fused_op_ublock_shape.at(sub_op.name).rt;
                sub_op_model.ublock_ct = fused_op_ublock_shape.at(sub_op.name).ct;

                if (sub_op_model.type == "matmul")
                {
                    if (sub_op.inputs[1].type == FusedSubOpInput::INPUT)
                    {
                        sub_op_model.ublock_kt = input_buffers[sub_op.inputs[1].index].block_shape.ublock.rt;
                        sub_op_model.mblock_k = input_buffers[sub_op.inputs[1].index].block_shape.mblock_m;
                    }
                    else if (sub_op.inputs[1].type == FusedSubOpInput::INTERMED)
                    {
                        sub_op_model.ublock_kt = intermediate_buffers[sub_op.inputs[1].index].block_shape.ublock.rt;
                        sub_op_model.mblock_k = intermediate_buffers[sub_op.inputs[1].index].block_shape.mblock_m;
                    }
                }
                else if (sub_op_model.type == "reduce")
                {
                    sub_op_model.reduce_dim = std::get<std::string>(sub_op.get_sub_op_buda_attr().at("dim"));
                }
                else
                {
                    // other ops can use dest register as input/output and it impacts the number of cycles;
                    // matmul and reduce cannot use dest on input or output
                    sub_op_model.has_dest_input =
                        sub_op.inputs[0].type == FusedSubOpInput::DEST ||
                        (sub_op.inputs.size() == 2 && sub_op.inputs[1].type == FusedSubOpInput::DEST);
                    sub_op_model.has_dest_output = sub_op.output_type == FusedSubOp::DEST;
                }
            }
        }

        return tt::balancer::get_execution_cycles(arch_name, *this, theoretical, sub_op_models);
    }

    // Go through schedules and add up execution cycles for each op
    std::uint32_t execution_cycles = 0;
    // std::cout << "Fused op " << fused_op->id() << std::endl;
    for (auto schedule : fused_op->get_schedules())
    {
        for (auto sub_op : schedule.ops)
        {
            // We don't have the right op model for each sub op... so we'll do a quick and dirty "per tile"
            // calculation. This is mostly ok, because matmuls in fused ops always have one tile * more tiles, so
            // it's very eltwise-like... we can count the number of tiles and multiple with some number

            // TODO: add approx flag to OpModel
            bool exp_approx = env_as<bool>("PYBUDA_EXP_APPROX");
            std::unordered_map<std::string, std::uint32_t> op_weights = {
                {"exp", exp_approx ? 357 : 700},
                {"gelu", 286},
                {"gelu_derivative", exp_approx ? 1500 : 3116},
                {"log", 1413},
                {"nop", 56},
                {"buffer", 56},
                {"reciprocal", exp_approx ? 606 : 915},
                {"sigmoid", 168},
                {"sqrt", 159},
                {"add", 20},
                {"multiply", 20},
                {"sub", 20},
                {"matmul", 40},
            };

            std::uint32_t tiles = (float)sub_op.op_shape.outputs[0].ct * sub_op.op_shape.outputs[0].rt *
                                  sub_op.op_shape.outputs[0].z / grid_shape.volume();

            if (sub_op.op_type.op == "matmul")
            {
                tiles = (float)sub_op.op_shape.outputs[0].z * sub_op.op_shape.inputs[0].rt *
                        sub_op.op_shape.inputs[0].ct * sub_op.op_shape.inputs[1].ct / grid_shape.volume();
            }

            std::uint32_t tile_weight = 40;  // some placeholder for other ops
            auto it = op_weights.find(sub_op.op_type.op);
            if (it != op_weights.end())
                tile_weight = it->second;

            if (sub_op.op_type.op == "matmul" || sub_op.op_type.op == "multiply")
            {
                switch (this->math_fidelity())
                {
                    case tt::MathFidelity::HiFi2: tile_weight *= 2; break;
                    case tt::MathFidelity::HiFi3: tile_weight *= 3; break;
                    case tt::MathFidelity::HiFi4: tile_weight *= 4; break;
                    default: break;
                }
            }

            // int sub_op_cycles = tt::balancer::get_execution_cycles(sub_op.op_type.op, sub_op_model);
            std::uint32_t sub_op_cycles = tiles * tile_weight;
            execution_cycles += sub_op_cycles;
            // std::cout << "  add sub_op " << sub_op.name << " / " << sub_op.op_type.op << " cycles " << sub_op_cycles
            // << ", total " << execution_cycles << std::endl;
        }
    }

    // Multiply cycle count estimate to be conservative
    std::uint32_t fused_op_cycle_multiplier = env_as<int>("PYBUDA_FUSED_OP_MULTIPLIER", 1);

    execution_cycles *= fused_op_cycle_multiplier;

    return execution_cycles;
}

int OpModel::get_execution_cycles(std::string const &arch_name, bool theoretical, bool invalidate_cached) const
{
    // Do not cache theoretical cycles otherwise we'd need to maintain multiple cache entries, one for w/ theoretical
    // cycles and one without. Theoretical cycles is only used in a few places so it didn't seem worth the additional
    // complexity to add a separate caching mechanism for it.
    if (theoretical)
        return get_execution_cycles_uncached(arch_name, theoretical);

    if (invalidate_cached)
        invalidate_cached_execution_cycles();

    if (cached_execution_cycles)
        return cached_execution_cycles;

    cached_execution_cycles = get_execution_cycles_uncached(arch_name, theoretical);
    return cached_execution_cycles;
}

//
// FactorizedInt
//

FactorizedInt::FactorizedInt(Factors max_val) : factors(factorize(1, max_val)) {}

FactorizedInt::FactorizedInt(FactorRange r) : factors(factorize(std::max(1, r.first), r.second)) {}

FactorizedInt::FactorizedInt(Constant s) : factors(factorize(s.v, s.v)) {}

FactorizedInt::FactorizedInt(Factorial f)
{
    for (int i = 1; i <= f.max; ++i)
    {
        factors.push_back(i * f.multiplier);
    }
}

int FactorizedInt::value() const
{
    TT_ASSERT(is_singleton());
    return factors.back();
}

int FactorizedInt::get_min_factor() const { return factors.front(); }
int FactorizedInt::get_max_factor() const { return factors.back(); }

int FactorizedInt::get_nearest_factor_le(int integer) const
{
    // Find nearest factor less than or equal to integer
    int nearest = factors[0];
    TT_ASSERT(nearest <= integer);
    for (auto factor : factors)
    {
        if (factor > integer)
        {
            break;
        }
        nearest = factor;
    }
    return nearest;
}

std::vector<int> const &FactorizedInt::get_factors() const { return factors; }

FactorizedInt FactorizedInt::keep_factors_divisible_by(FactorizedInt const &other) const
{
    TT_ASSERT(other.is_singleton());

    FactorizedInt ret{};

    for (size_t i = 0; i < this->factors.size(); i++)
    {
        if (this->factors[i] % other.value() == 0)
        {
            ret.factors.push_back(this->factors[i]);
        }
    }

    return ret;
}

FactorizedInt FactorizedInt::operator&(FactorizedInt const &other) const
{
    FactorizedInt intersection;
    std::set_intersection(
        factors.begin(),
        factors.end(),
        other.factors.begin(),
        other.factors.end(),
        std::back_inserter(intersection.factors));
    return intersection;
}

FactorizedInt FactorizedInt::operator|(FactorizedInt const &other) const
{
    FactorizedInt intersection;
    std::set_union(
        factors.begin(),
        factors.end(),
        other.factors.begin(),
        other.factors.end(),
        std::back_inserter(intersection.factors));
    return intersection;
}

FactorizedInt FactorizedInt::operator-(FactorizedInt const &other) const
{
    FactorizedInt intersection;
    std::set_difference(
        factors.begin(),
        factors.end(),
        other.factors.begin(),
        other.factors.end(),
        std::back_inserter(intersection.factors));
    return intersection;
}

FactorizedInt FactorizedInt::operator*(FactorizedInt const &other) const
{
    TT_ASSERT(other.is_singleton(), "Currently only support singletons");
    FactorizedInt result = *this;
    for (int &f : result.factors)
    {
        f *= other.factors.back();
    }
    return result;
}

FactorizedInt FactorizedInt::operator/(FactorizedInt const &other) const
{
    TT_ASSERT(other.is_singleton(), "Currently only support singletons");
    FactorizedInt result;
    for (int f : factors)
    {
        if (f >= other.factors.back() and (f % other.factors.back()) == 0)
            result.factors.push_back(f / other.factors.back());
    }
    return result;
}

bool FactorizedInt::operator==(FactorizedInt const &other) const { return factors == other.factors; }

bool FactorizedInt::overlaps(FactorizedInt const &other) const
{
    auto iter_a = factors.begin();
    auto iter_b = other.factors.begin();
    while (iter_a != factors.end() and iter_b != other.factors.end())
    {
        if (*iter_a == *iter_b)
            return true;
        if (*iter_a < *iter_b)
            ++iter_a;
        else
            ++iter_b;
    }
    return false;
}

bool FactorizedInt::contains(int v) const
{
    for (int f : factors)
        if (v == f)
            return true;
    return false;
}

bool FactorizedInt::is_singleton() const { return factors.size() == 1; }

std::vector<int> FactorizedInt::factorize(int min_val, int max_val)
{
    std::vector<int> factors;
    for (int i = min_val; i <= max_val; ++i)
    {
        if ((max_val % i) == 0 and (i % min_val) == 0)
            factors.push_back(i);
    }
    return factors;
}

//
// FactorizedShape
//
FactorizedShape::FactorizedShape(graphlib::Shape const &shape) : r(shape.rt()), c(shape.ct()) {}

FactorizedShape::FactorizedShape(std::pair<int, int> shape) : r(shape.first), c(shape.second) {}

FactorizedShape::FactorizedShape(Parallelization par) : r(Constant((int)par.r)), c(Constant((int)par.c)) {}

FactorizedShape::FactorizedShape(FactorizedInt r, FactorizedInt c) : r(r), c(c) {}

FactorizedShape FactorizedShape::operator&(FactorizedShape const &other) const
{
    return FactorizedShape(r & other.r, c & other.c);
}

FactorizedShape FactorizedShape::operator|(FactorizedShape const &other) const
{
    return FactorizedShape(r | other.r, c | other.c);
}

FactorizedShape FactorizedShape::operator-(FactorizedShape const &other) const
{
    return FactorizedShape(r - other.r, c - other.c);
}

Parallelization FactorizedShape::operator[](int idx) const
{
    std::vector<int> const &r_factors = r.get_factors();
    std::vector<int> const &c_factors = c.get_factors();
    int ridx = idx / (int)c_factors.size();
    int cidx = idx % (int)c_factors.size();
    return Parallelization(r_factors[ridx], c_factors[cidx]);
}

bool FactorizedShape::operator==(FactorizedShape const &other) const { return (r == other.r) and (c == other.c); }

bool FactorizedShape::empty() const { return r.get_factors().empty() or c.get_factors().empty(); }

std::size_t FactorizedShape::size() const { return r.get_factors().size() * c.get_factors().size(); }

bool FactorizedShape::is_subset_of(FactorizedShape const &other) const { return (*this & other) == *this; }

bool FactorizedShape::is_superset_of(FactorizedShape const &other) const { return (*this & other) == other; }

bool FactorizedShape::is_singleton() const { return r.is_singleton() and c.is_singleton(); }

FactorizedShape::Iterator::Iterator(FactorizedShape const *p) : p(p) {}

FactorizedShape::Iterator::Iterator(FactorizedShape const *p, int i) : i(i), p(p) {}

FactorizedShape::Iterator &FactorizedShape::Iterator::operator++()
{
    ++i;
    return *this;
}

FactorizedShape::Iterator FactorizedShape::Iterator::operator++(int)
{
    auto retval = *this;
    ++(*this);
    return retval;
}

bool FactorizedShape::Iterator::operator==(Iterator other) const { return (p == other.p) and (i == other.i); }

bool FactorizedShape::Iterator::operator!=(Iterator other) const { return !(*this == other); }

FactorizedShape::Iterator::reference FactorizedShape::Iterator::operator*() const { return (*p)[i]; }

FactorizedShape::Iterator FactorizedShape::begin() const { return Iterator(this); }

FactorizedShape::Iterator FactorizedShape::end() const { return Iterator(this, (int)size()); }

//
// TileLayout
//
LinCoord TileLayout::operator[](int idx) const
{
    int idx_t = idx / (rt() * ct());
    int idx_rt = (idx % (rt() * ct())) / ct();
    int idx_ct = idx % ct();
    return map(CanCoord(idx_t, idx_rt, idx_ct));
}

LinCoord TileLayout::map(CanCoord can_coord) const
{
    auto [w, t, r, c] = can_coord;
    TT_ASSERT(t < this->t());
    TT_ASSERT(r < this->rt());
    TT_ASSERT(c < this->ct());
    UBlockShape ublock_coord(r % block_shape.ublock.rt, c % block_shape.ublock.ct);
    int m = (r / block_shape.ublock.rt) % block_shape.mblock_m;
    int n = (c / block_shape.ublock.ct) % block_shape.mblock_n;
    BlockShape mblock_coord(t, m, n, ublock_coord);
    GridShape grid_coord(r / block_shape.rt(), c / block_shape.ct());
    int mblock_volume = block_shape.volume_no_t();
    int ublock_volume = block_shape.ublock.volume();
    int t_linear = t * mblock_volume;
    int mblock_linear = (ublock_order == graphlib::UBlockOrder::R)
                            ? m * block_shape.mblock_n * ublock_volume + n * ublock_volume
                            : n * block_shape.mblock_m * ublock_volume + m * ublock_volume;
    int ublock_linear = ublock_coord.rt * block_shape.ublock.ct + ublock_coord.ct;
    return LinCoord(grid_coord, t_linear + mblock_linear + ublock_linear);
}

CanCoord TileLayout::map(LinCoord lin_coord) const
{
    GridCoord grid_coord(lin_coord.grid_r(), lin_coord.grid_c());
    int offset = lin_coord.address();
    int mblock_offset = offset % block_shape.volume_no_t();
    int ublock_offset = mblock_offset % block_shape.ublock.volume();
    int num_ublocks = mblock_offset / block_shape.ublock.volume();
    int t = offset / block_shape.volume_no_t();
    int m = (ublock_order == graphlib::UBlockOrder::R) ? num_ublocks / block_shape.mblock_n
                                                       : num_ublocks % block_shape.mblock_m;
    int n = (ublock_order == graphlib::UBlockOrder::R) ? num_ublocks % block_shape.mblock_n
                                                       : num_ublocks / block_shape.mblock_m;
    UBlockShape ublock_coord(ublock_offset / block_shape.ublock.ct, ublock_offset % block_shape.ublock.ct);
    int r = grid_coord.r * block_shape.rt() + m * block_shape.ublock.rt + ublock_coord.rt;
    int c = grid_coord.c * block_shape.ct() + n * block_shape.ublock.ct + ublock_coord.ct;
    return CanCoord(t, r, c);
}

GridCoord TileLayout::grid_coord(CanCoord can_coord) const
{
    return GridCoord(can_coord.rt / block_shape.rt(), can_coord.ct / block_shape.ct());
}

GridCoord TileLayout::grid_coord(LinCoord lin_coord) const { return lin_coord.grid_coord(); }

}  // namespace tt::balancer
