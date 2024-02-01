// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <optional>
#include <ostream>
#include <vector>

#include "utils/assert.hpp"

namespace tt::passes
{
//
// NDSlice
// Represents an N dimensionally sliced tensor
//
// On top of maintaining the order and factors of each sliced dimension,
// it also maintains a consistent total ordering of slices which is useful
// for mapping slices to an index in a flat list of fractured nodes.
//
// All dim values use negative indexing.
//
class NDSlice
{
   public:
    static bool are_multiples(int a, int b) { return (a % b == 0) or (b % a == 0); }
    static int multiple(int a, int b)
    {
        TT_ASSERT(are_multiples(a, b));
        return a > b ? a / b : b / a;
    }

    constexpr static int k_dim = ::tt::passes::k_dim;

    enum class Spec
    {
        Explicit,
        Inferred,
    };

    struct Slice
    {
        using SliceIdx = std::tuple<int, int, int>;  // dim, index, factor

        int total_index = 0;
        std::vector<SliceIdx> indices;

        Slice() = default;
        Slice(int total_index, NDSlice const& nd_slice) : total_index(total_index)
        {
            int v = nd_slice.volume();
            for (int j = 0; j < (int)nd_slice.factors.size(); ++j)
            {
                v /= nd_slice.factors[j];
                int slice_index = (total_index / v) % nd_slice.factors[j];
                indices.push_back(std::make_tuple(nd_slice.dims[j], slice_index, nd_slice.factors[j]));
            }
        }

        Slice view(NDSlice const& nd_slice) const
        {
            Slice v;
            for (auto [dim, index, factor] : indices)
            {
                if (nd_slice.has_dim(dim))
                {
                    TT_ASSERT(factor % nd_slice.get_factor(dim) == 0);
                    int multiple = factor / nd_slice.get_factor(dim);
                    v.indices.push_back(std::make_tuple(dim, index / multiple, nd_slice.get_factor(dim)));
                }
            }

            v.calculate_index(nd_slice);
            return v;
        }

        Slice operand_view(NDSlice const& operand, int input_idx, bool is_matmul) const
        {
            Slice v;
            for (auto [dim, index, factor] : indices)
            {
                if (is_matmul and input_idx == 0 and dim == -1)
                    continue;
                if (is_matmul and input_idx == 1 and dim == -2)
                    continue;

                if (dim == k_dim)
                {
                    TT_ASSERT(is_matmul);
                    dim = (input_idx == 0) ? -1 : -2;
                }

                if (operand.has_dim(dim))
                {
                    TT_ASSERT(factor == operand.get_factor(dim), *this, operand);
                    v.indices.push_back(std::make_tuple(dim, index, factor));
                }
            }
            v.calculate_index(operand);
            return v;
        }

       private:
        void calculate_index(NDSlice const& nd_slice)
        {
            std::sort(
                indices.begin(),
                indices.end(),
                [&nd_slice](auto const& a, auto const& b)
                { return nd_slice.index_of(std::get<0>(a)) < nd_slice.index_of(std::get<0>(b)); });

            int volume = 1;
            for (int i = (int)indices.size() - 1; i >= 0; --i)
            {
                auto [dim, j, factor] = indices[i];
                total_index += j * volume;
                volume *= factor;
            }
        }
    };

    NDSlice() = default;
    NDSlice(std::vector<int> const& in_dims, std::vector<int> const& in_factors, Spec spec, int rank = 0) : spec(spec)
    {
        TT_ASSERT(in_dims.size() == in_factors.size());
        dims.reserve(in_dims.size());
        factors.reserve(in_factors.size());
        for (std::size_t i = 0; i < in_dims.size(); ++i)
        {
            if (in_factors[i] <= 1)
                continue;

            int dim = in_dims[i];
            int factor = in_factors[i];
            if (dim > 0)
                dim -= rank;
            TT_ASSERT(dim < 0);
            dims.push_back(dim);
            factors.push_back(factor);
        }
    }

    NDSlice trunc(int dim) const
    {
        auto dim_iter = std::find(dims.begin(), dims.end(), dim);
        TT_ASSERT(dim_iter != dims.end(), "Dim not found in trunc");
        auto dis = std::distance(dims.begin(), dim_iter);
        std::vector<int> cut_dims(dims.begin(), dim_iter + 1);
        std::vector<int> cut_factors(factors.begin(), factors.begin() + dis + 1);
        return NDSlice(cut_dims, cut_factors, spec);
    }

    NDSlice remove_dim(int dim) const
    {
        std::optional<int> index = index_of(dim);
        if (not index)
            return *this;
        auto removed_dims = dims;
        auto removed_factors = factors;
        removed_dims.erase(removed_dims.begin() + *index);
        removed_factors.erase(removed_factors.begin() + *index);
        return NDSlice(removed_dims, removed_factors, spec);
    }

    NDSlice replace_dim(int dim, int replacement_dim) const
    {
        std::optional<int> index = index_of(dim);
        if (not index)
            return *this;
        auto replaced_dims = dims;
        replaced_dims[*index] = replacement_dim;
        return NDSlice(replaced_dims, factors, spec);
    }

    NDSlice replace_factor(int dim, int factor) const
    {
        if (factor == 1)
            return remove_dim(dim);
        std::optional<int> index = index_of(dim);
        auto replaced_dims = dims;
        auto replaced_factors = factors;
        if (not index)
        {
            replaced_dims.push_back(dim);
            replaced_factors.push_back(factor);
        }
        else
        {
            replaced_factors[*index] = factor;
        }
        return NDSlice(replaced_dims, replaced_factors, spec);
    }

    bool operator==(NDSlice const& other) const { return (dims == other.dims) and (factors == other.factors); }
    bool operator!=(NDSlice const& other) const { return not(*this == other); }

    std::optional<int> index_of(int dim) const
    {
        TT_ASSERT(dim < 0);
        auto match = std::find(dims.begin(), dims.end(), dim);
        return match != dims.end() ? std::optional<int>((int)std::distance(dims.begin(), match)) : std::nullopt;
    }

    std::vector<int> const& get_dims() const { return dims; }
    std::vector<int> get_dims_reversed() const { return std::vector<int>(dims.rbegin(), dims.rend()); }

    bool has_dim(int dim) const { return index_of(dim).has_value(); }

    int get_factor(int dim) const
    {
        auto i = index_of(dim);
        return i ? factors[*i] : 1;
    }

    graphlib::Shape get_shape(graphlib::Shape shape) const
    {
        for (int i = 0; i < (int)dims.size(); ++i)
        {
            if (dims[i] == k_dim)
                continue;
            TT_ASSERT(shape[dims[i]] % factors[i] == 0, shape, dims[i], factors[i]);
            shape[dims[i]] /= factors[i];
        }
        return shape;
    }

    std::vector<Slice> get_slices() const
    {
        std::vector<Slice> slices;
        for (int i = 0; i < volume(); ++i)
        {
            slices.emplace_back(i, *this);
        }
        return slices;
    }

    int volume() const
    {
        int v = 1;
        for (int f : factors) v *= f;
        return v;
    }

    NDSlice operator/(NDSlice const& other) const
    {
        NDSlice result;
        for (auto dim : dims)
        {
            if (other.has_dim(dim))
            {
                int factor = get_factor(dim) / other.get_factor(dim);
                if (factor > 1)
                {
                    result.dims.push_back(dim);
                    result.factors.push_back(factor);
                }
            }
            else
            {
                result.dims.push_back(dim);
                result.factors.push_back(get_factor(dim));
            }
        }
        return result;
    }

    bool empty() const { return dims.empty(); }
    bool is_explicit() const { return spec == Spec::Explicit; }
    bool is_inferred() const { return spec == Spec::Inferred; }
    void set_spec(Spec new_spec) { spec = new_spec; }

   private:
    std::vector<int> dims;
    std::vector<int> factors;
    Spec spec = Spec::Inferred;
};

inline std::ostream& operator<<(std::ostream& out, NDSlice const& ff)
{
    out << "NDSlice{";
    for (int dim : ff.get_dims())
        out << "(" << ((dim == NDSlice::k_dim) ? "k" : std::to_string(dim)) << ", " << ff.get_factor(dim) << "),";
    out << "}";
    return out;
}

inline std::ostream& operator<<(std::ostream& out, NDSlice::Slice const& c)
{
    out << "NDSlice::Slice{";
    out << "index=" << c.total_index << ",";
    for (auto [dim, index, factor] : c.indices)
        out << "(" << ((dim == NDSlice::k_dim) ? "k" : std::to_string(dim)) << ", " << index << ", " << factor << "),";
    out << "}";
    return out;
}
}  // namespace tt::passes
