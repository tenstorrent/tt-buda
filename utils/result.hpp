// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#define UTILS_RESULT_INCLUDE_PYBIND11 1

#if defined(UTILS_RESULT_INCLUDE_PYBIND11) && (UTILS_RESULT_INCLUDE_PYBIND11 == 1)
#include <pybind11/stl.h>
#endif

#include <variant>

#include "utils/assert.hpp"

namespace tt
{
template <typename T, typename E>
struct Result : public std::variant<T, E>
{
    using Variant = std::variant<T, E>;
    using std::variant<T, E>::variant;

    bool is_valid() const { return std::holds_alternative<T>(*this); }
    bool is_error() const { return std::holds_alternative<E>(*this); }

    T& get()
    {
        TT_ASSERT(std::holds_alternative<T>(*this));
        return std::get<T>(*this);
    };

    T const& get() const
    {
        TT_ASSERT(std::holds_alternative<T>(*this));
        return std::get<T>(*this);
    }

    E& get_error()
    {
        TT_ASSERT(std::holds_alternative<E>(*this));
        return std::get<E>(*this);
    };

    E const& get_error() const
    {
        TT_ASSERT(std::holds_alternative<E>(*this));
        return std::get<E>(*this);
    }
};
}  // namespace tt

#if defined(UTILS_RESULT_INCLUDE_PYBIND11) && (UTILS_RESULT_INCLUDE_PYBIND11 == 1)
namespace pybind11::detail {
template <typename T, typename E>
struct type_caster<tt::Result<T, E>>
{
    bool load(handle src, bool convert)
    {
        auto result_caster = make_caster<T>();
        if (result_caster.load(src, convert))
        {
            value = cast_op<T>(result_caster);
            return true;
        }
        auto error_caster = make_caster<E>();
        if (error_caster.load(src, convert))
        {
            value = cast_op<E>(result_caster);
            return true;
        }
        return false;
    }

    static handle cast(tt::Result<T, E>&& src, return_value_policy policy, handle parent)
    {
        if (src.is_error())
            return type_caster<E>::cast(src.get_error(), policy, parent);
        return type_caster<T>::cast(src.get(), policy, parent);
    }

    using Type = tt::Result<T, E>;
    PYBIND11_TYPE_CASTER(Type, _("Union[") + detail::concat(make_caster<T>::name, make_caster<E>::name) + _("]"));
};
}
#endif
