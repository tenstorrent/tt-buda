// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "utils/ordered_associative_containers/ordered_map.hpp"

namespace py = pybind11;

template <typename Key, typename Value, typename Hash, typename Equal, typename Alloc>
struct pybind11::detail::type_caster<tt::ordered_map<Key, Value, Hash, Equal, Alloc>>
    : map_caster<tt::ordered_map<Key, Value, Hash, Equal, Alloc>, Key, Value> {};

namespace tt {

inline std::shared_ptr<void> make_shared_py_object(py::object obj) {
    return std::shared_ptr<void>(static_cast<void *>(obj.release().ptr()), [](void *ptr) {
        py::reinterpret_steal<py::object>(py::handle(static_cast<PyObject *>(ptr)));
    });
}

inline py::object borrow_shared_py_object(std::shared_ptr<void> handle) {
    return py::reinterpret_borrow<py::object>(py::handle(static_cast<PyObject *>(handle.get())));
}

}  // namespace tt
