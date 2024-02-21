// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "tt_torch_device/python_bindings.hpp"
#include "tt_torch_device/tt_device.hpp"
#include "pybuda/csrc/python_bindings_common.hpp"


namespace tt {

void TorchDeviceModule(py::module &m_torch_device)
{
    m_torch_device.def("get_default_device", &tt::get_default_tt_device, py::return_value_policy::reference);
    m_torch_device.def("get_available_devices", []() { return tt::get_available_tt_devices(); });

    py::class_<tt::PyBudaTensorDesc>(m_torch_device, "PyBudaTensorDesc")
        .def(
            py::init<
                std::string const&,
                std::vector<std::int64_t> const&,
                std::int64_t,
                std::optional<torch::Tensor>>(),
            py::arg("name"),
            py::arg("shape"),
            py::arg("ptr") = -1,
            py::arg("constant") = std::nullopt)
            .def_readonly("name", &tt::PyBudaTensorDesc::name)
            .def_readonly("shape", &tt::PyBudaTensorDesc::shape)
            .def_readonly("ptr", &tt::PyBudaTensorDesc::ptr)
            .def_readonly("constant", &tt::PyBudaTensorDesc::constant);

    py::class_<tt::Program>(m_torch_device, "Program")
        .def(
            py::init<std::string const&, std::map<std::string, std::string> const&>(),
            py::arg("name"),
            py::arg("params"));

    py::class_<tt::CompileRequest>(m_torch_device, "CompileRequest")
        .def(
            py::init<
                std::string const&,
                std::string const&,
                tt::tt_backend_config const&,
                std::vector<tt::PyBudaTensorDesc> const&,
                std::map<int, std::vector<std::string>> const&,
                std::map<int, std::vector<std::vector<int>>> const&,
                std::vector<tt::PyBudaTensorDesc> const&,
                std::vector<tt::PyBudaTensorDesc> const&,
                std::vector<tt::PyBudaTensorDesc> const&,
                std::map<int, std::vector<std::string>> const&>(),
            py::arg("netlist_path"),
            py::arg("output_dir"),
            py::arg("backend_config"),
            py::arg("inputs"),
            py::arg("input_runtime_transforms"),
            py::arg("input_tile_bcast_dims"),
            py::arg("constants"),
            py::arg("parameters"),
            py::arg("outputs"),
            py::arg("output_runtime_transforms"));

    py::class_<tt::Workload, std::shared_ptr<tt::Workload>>(m_torch_device, "Workload")
        .def_readonly("inputs", &tt::Workload::inputs)
        .def_readonly("constants", &tt::Workload::constants)
        .def_readonly("parameters", &tt::Workload::parameters)
        .def_readonly("outputs", &tt::Workload::outputs);

    py::class_<tt::TTDevice>(m_torch_device, "TTDevice")
        .def_readonly("backend", &tt::TTDevice::backend)
        .def_readonly("type", &tt::TTDevice::type)
        .def_readonly("arch", &tt::TTDevice::arch)
        .def_readonly("mmio", &tt::TTDevice::mmio)
        .def_readonly("index", &tt::TTDevice::index)
        .def_readonly("soc_desc_yaml", &tt::TTDevice::soc_desc_yaml)
        .def_property_readonly("cluster_yaml", &tt::get_device_cluster_yaml)
        .def("torch_device", &tt::torch_device)
        .def("str", &tt::to_string)
        .def("__str__", &tt::to_string)
        .def("compile", &tt::compile)
        .def("dispatch", &tt::dispatch);

    m_torch_device.def(
        "push_tensor", 
        &tt::push_tensor,
        py::arg("backend"),
        py::arg("desc"),
        py::arg("tensor"),
        py::arg("info") = "",
        py::arg("ptr") = std::optional<int>{});

    m_torch_device.def("is_created_on_device", tt::is_created_on_device);
    m_torch_device.def("original_shape", tt::original_shape);
    m_torch_device.def("unique_id", tt::unique_id);
}



}
