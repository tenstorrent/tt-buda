// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include <unordered_map>
#include "yaml-cpp/yaml.h"

#include "backend_api/backend_api.hpp"
#include "backend_api/device_config.hpp"
#include "utils/assert.hpp"

#include "netlist/tt_backend.hpp"
#include "netlist/tt_backend_api.hpp"
#include "common/env_lib.hpp"

namespace tt
{
template <typename T>
T DeviceConfig::get(std::string const &param, const bool system_level_command) const
{
    std::string key_value = (system_level_command) ? ("system-" + param) : (arch_name + "-" + param);
    std::string value;
    if (system_level_command and this->cached_system_level_params.size() > 0)
    {
        value = this->cached_system_level_params.at(key_value);
    }
    else 
    {
        value = ::tt::backend::get_backend_param(
            key_value,
            this->device_yaml,
            this->cluster_config_yaml,
            this->runtime_params_yaml,
            this->store_backend_db_to_yaml);
    }

    if constexpr (std::is_same_v<T, std::string>)
    {
        return value;
    }
    else if constexpr (std::is_same_v<T, std::uint32_t>)
    {
        return std::stoul(value, 0, 0);
    }
    else if constexpr (std::is_same_v<T, int>)
    {
        return std::stoi(value, 0, 0);
    }
    else if constexpr (std::is_same_v<T, std::uint64_t>)
    {
        return std::stoull(value, 0, 0);
    }
    else if constexpr (std::is_same_v<T, bool>)
    {
        return static_cast<bool>(std::stoi(value, 0, 0));
    }
    else if constexpr (std::is_same_v<T, CoreCoord>)
    {
        auto delimeter = value.find("-");
        auto x_str = value.substr(0, delimeter);
        auto y_str = value.substr(delimeter + 1, std::string::npos);
        return CoreCoord(std::stoi(x_str, 0, 0), std::stoi(y_str, 0, 0));
    }
    else if constexpr (std::is_same_v<T, DeviceGrid>)
    {
        auto delimeter = value.find("-");
        auto c_str = value.substr(0, delimeter);
        auto r_str = value.substr(delimeter + 1, std::string::npos);
        return DeviceGrid(std::stoi(r_str, 0, 0), std::stoi(c_str, 0, 0));
    }
    else if constexpr (std::is_same_v<T, std::string>)
    {
        return value;
    }
    else if constexpr (std::is_same_v<T, std::vector<int>>)
    {
        // Chips with mmio are serialized separated by a dash (eg. '1-2-3')
        std::vector<int> chips_with_mmio;

        // Split string and extract chip ids
        size_t delimeter = 0;
        while((delimeter = value.find("-")) != std::string::npos)
        {
            std::string curr_str = value.substr(0, delimeter);
            chips_with_mmio.push_back(std::stoi(curr_str, 0, 0));
            value.erase(0, delimeter + 1);
        }

        return chips_with_mmio;
    }
    else if constexpr (std::is_same_v<T, std::unordered_map<uint32_t, EthCoord>>)
    {
        // Chip locations are serialized separated by a dash (eg. '0,0,0,-1,1,0,-')
        std::unordered_map<uint32_t, EthCoord> chip_locations;
        std::vector<std::string> temporary_buffer;

        // Split string into temporary buffer for additional processing
        size_t delimeter = 0;
        while((delimeter = value.find("-")) != std::string::npos)
        {
            std::string curr_str = value.substr(0, delimeter);
            temporary_buffer.push_back(curr_str);
            value.erase(0, delimeter + 1);
        } 

        // Loop through temporary buffer and extract information
        for (std::string chip_location : temporary_buffer)
        {
            // Split string into chip id and chip location portions
            size_t delimeter = 0;
            std::vector<int> extracted_values;
            while((delimeter = chip_location.find(",")) != std::string::npos)
            {
                std::string curr_str = chip_location.substr(0, delimeter);
                extracted_values.push_back(std::stoi(curr_str, 0, 0));
                chip_location.erase(0, delimeter + 1);
            }

            // Add chip location to map
            chip_locations.insert(
                {extracted_values.at(0),
                 EthCoord(
                     extracted_values.at(1), extracted_values.at(2), extracted_values.at(3), extracted_values.at(4))});
        }

        return chip_locations;
    }
    else if constexpr (std::is_same_v<T, std::unordered_map<uint32_t, std::unordered_map<uint32_t, std::tuple<uint32_t, uint32_t>>>>)
    {
        // Ethernet connections are serialized separated by a dash (eg. '0,0,1,0,-0,1,1,1,-')
        std::unordered_map<uint32_t, std::unordered_map<uint32_t, std::tuple<uint32_t, uint32_t>>> ethernet_connections;
        std::vector<std::string> temporary_buffer;

        // Split string into temporary buffer for additional processing
        size_t delimeter = 0;
        while((delimeter = value.find("-")) != std::string::npos)
        {
            std::string curr_str = value.substr(0, delimeter);
            temporary_buffer.push_back(curr_str);
            value.erase(0, delimeter + 1);
        }

        // Loop through temporary buffer and extract information
        for (std::string eth_connection : temporary_buffer)
        {
            // Split string and collect values
            size_t delimeter = 0;
            std::vector<uint32_t> extracted_values;
            while((delimeter = eth_connection.find(",")) != std::string::npos)
            {
                std::string curr_str = eth_connection.substr(0, delimeter);
                extracted_values.push_back(std::stoul(curr_str, 0, 0));
                eth_connection.erase(0, delimeter + 1);
            }

            // Add values to map
            if(ethernet_connections.find(extracted_values[0]) == ethernet_connections.end()) {
                ethernet_connections[extracted_values[0]] = {};
            }
            ethernet_connections[extracted_values[0]][extracted_values[1]] = std::tuple<uint32_t, uint32_t>(extracted_values[2], extracted_values[3]);
        }

        return ethernet_connections;
    }
    else
    {
        static_assert(false_type_t<T>, "No specialization for type");
    }
}

// explicit instantiations
template std::string DeviceConfig::get<std::string>(std::string const &, const bool) const;
template std::uint32_t DeviceConfig::get<std::uint32_t>(std::string const &, const bool) const;
template std::uint64_t DeviceConfig::get<std::uint64_t>(std::string const &, const bool) const;
template int DeviceConfig::get<int>(std::string const &, const bool) const;
template bool DeviceConfig::get<bool>(std::string const &, const bool) const;
template CoreCoord DeviceConfig::get<CoreCoord>(std::string const &, const bool) const;
template std::vector<int> DeviceConfig::get<std::vector<int>>(std::string const &, const bool) const;
template std::unordered_map<uint32_t, EthCoord> DeviceConfig::get<std::unordered_map<uint32_t, EthCoord>>(
    std::string const &, const bool) const;
template std::unordered_map<uint32_t, std::unordered_map<uint32_t, std::tuple<uint32_t, uint32_t>>> DeviceConfig::get<std::unordered_map<uint32_t, std::unordered_map<uint32_t, std::tuple<uint32_t, uint32_t>>>>(std::string const &, const bool) const;

// temporarily added, until FE consumes a commit that includes equivalent parsing in BBE
std::unordered_map<std::string, std::string> load_cached_sys_param(std::string yaml_file)
{
    std::unordered_map<std::string, std::string> cache;
    YAML::Node reference_param_desc = YAML::LoadFile(yaml_file);
    for (auto it = reference_param_desc["system_level_params"].begin(); it != reference_param_desc["system_level_params"].end(); it++)
        cache[(it -> first).as<std::string>()] = (it -> second).as<std::string>();
    return cache;
}

void DeviceConfig::load_system_level_params()
{
    auto silicon_devices = tt::backend::detect_available_devices();
    if (silicon_devices.size() == 0) // compute machine
        this->cached_system_level_params = load_cached_sys_param(this->runtime_params_yaml);
}

std::vector<std::uint32_t> DeviceConfig::get_harvested_cfg() const
{
    auto silicon_devices = tt::backend::detect_available_devices();
    if (silicon_devices.size() == 0 and this->runtime_params_yaml.empty())
        return std::vector<std::uint32_t>(chip_ids.size(), 0);  // assume same harvesting-config among all chips for non-silicon backend

    std::vector<std::uint32_t> ret;
    for (auto i : chip_ids)
    {
        std::string cmd = "device";
        cmd += std::to_string(i);
        cmd += "-harvesting_mask";
        uint32_t num = get<std::uint32_t>(cmd, true);
        ret.push_back(num);
    }
    return ret;
}

}  // namespace tt

namespace tt::backend_api
{
using tt_backend_config = tt::tt_backend_config;
using tt_compile_result = tt::tt_compile_result;

void python_handle_refchange(const void *handle_ptr, bool allocate)
{
    py::handle handle((PyObject *)handle_ptr);
    if (allocate)
        handle.inc_ref();
    else
        handle.dec_ref();
}

void BackendModule(py::module &m_backend) {


    py::class_<tt_backend_config>(m_backend, "BackendConfig")
        .def(py::init([](
                  tt::DEVICE backend_type, 
                  tt::ARCH backend_device, 
                  tt::DEVICE_MODE device_mode, 
                  int opt_level,
                  const std::string &output_dir,
                  const std::string &soc_descriptor_path,
                  const std::string &cluster_descriptor_path) {

            auto cfg = tt_backend_config{
                .type = backend_type, 
                .arch = backend_device, 
                .mode = device_mode,
                .output_dir = output_dir,
                .soc_descriptor_path = soc_descriptor_path,
                .cluster_descriptor_path = cluster_descriptor_path};

            char *env_opt_level = getenv("TT_BACKEND_OPT_LEVEL");
            if (env_opt_level) {
                cfg.optimization_level = atoi(env_opt_level);
            }
            else {
                cfg.optimization_level = opt_level;
            }
            if (backend_type == tt::DEVICE::Golden) {
                cfg.ignore_data_format_precision = true; // run backend at full precision by default (on Golden)
            }
            return cfg;
        }))
        .def("set_golden_ignore_df_precision", [](tt_backend_config &self, bool ignore_data_format_precision) {
            self.ignore_data_format_precision = ignore_data_format_precision;
        })
        .def("set_performance_trace_args", [](tt_backend_config &self, std::string args) {
            self.perf_desc_args = args;
        })
        .def("set_runtime_args", [](tt_backend_config &self, std::string args) {
            self.runtime_args = args;
        });

    m_backend.def("get_golden_config", []() {
        tt_backend_config cfg = {tt::DEVICE::Golden, tt::ARCH::GRAYSKULL};
        return cfg;
    });

    py::enum_<tt::DEVICE>(m_backend, "BackendType")
        .value("Golden", tt::DEVICE::Golden)
        .value("Model", tt::DEVICE::Model)
        .value("Silicon", tt::DEVICE::Silicon)
        .value("NoBackend", tt::DEVICE::Invalid)
        .def_static("from_string", &tt::get_device_from_string)
        .def("to_json", [](const tt::DEVICE backend_type) {
            switch (backend_type)
            {
                case tt::DEVICE::Golden: return "Golden";
                case tt::DEVICE::Model: return "Model";
                case tt::DEVICE::Silicon: return "Silicon";
                case tt::DEVICE::Invalid: return "Invalid";
                default: break;
            }
            return "Invalid";
        })
        .def("from_json", [](std::string const& encoded) {
            static std::unordered_map<std::string, tt::DEVICE> decode = {
                {"Golden", tt::DEVICE::Golden},
                {"Model", tt::DEVICE::Model},
                {"Silicon", tt::DEVICE::Silicon},
                {"NoBackend", tt::DEVICE::Invalid},
            };
            return decode.at(encoded);
        });

    py::enum_<tt::IO_TYPE>(m_backend, "IOType")
        .value("Queue", tt::IO_TYPE::Queue)
        .value("RandomAccess", tt::IO_TYPE::RandomAccess)
        .value("Invalid", tt::IO_TYPE::Invalid);

    py::enum_<tt::IO_LAYOUT>(m_backend, "IOLayout")
        .value("Tilized", tt::IO_LAYOUT::Tilized)
        .value("Flat", tt::IO_LAYOUT::Flat)
        .value("Invalid", tt::IO_LAYOUT::Invalid);

    py::enum_<tt::ARCH>(m_backend, "BackendDevice")
        .value("Grayskull", tt::ARCH::GRAYSKULL)
        .value("Wormhole", tt::ARCH::WORMHOLE)
        .value("Wormhole_B0", tt::ARCH::WORMHOLE_B0)
        .value("Invalid", tt::ARCH::Invalid)
        .def("to_string", &tt::get_string_lowercase)
        .def_static("from_string", &tt::get_arch_from_string)
        .def("to_json", [](const tt::ARCH backend_device) {
            switch (backend_device)
            {
                case tt::ARCH::GRAYSKULL: return "Grayskull";
                case tt::ARCH::WORMHOLE: return "Wormhole";
                case tt::ARCH::WORMHOLE_B0: return "Wormhole_B0";
                case tt::ARCH::Invalid: return "Invalid";
                default: break;
            }
            return "Invalid";
        })
        .def("from_json", [](std::string const& encoded) {
            static std::unordered_map<std::string, tt::ARCH> decode = {
                {"Grayskull", tt::ARCH::GRAYSKULL},
                {"Wormhole", tt::ARCH::WORMHOLE},
                {"Wormhole_B0", tt::ARCH::WORMHOLE_B0},
                {"Invalid", tt::ARCH::Invalid},
            };
            return decode.at(encoded);
        });

    py::enum_<tt::DEVICE_MODE>(m_backend, "DeviceMode")
        .value("CompileAndRun", tt::DEVICE_MODE::CompileAndRun)
        .value("CompileOnly", tt::DEVICE_MODE::CompileOnly)
        .value("RunOnly", tt::DEVICE_MODE::RunOnly)
        .def(
            "to_json",
            [](tt::DEVICE_MODE d) {
                switch (d)
                {
                    case tt::DEVICE_MODE::CompileAndRun: return "CompileAndRun";
                    case tt::DEVICE_MODE::CompileOnly: return "CompileOnly";
                    case tt::DEVICE_MODE::RunOnly: return "RunOnly";
                    default: break;
                }
                return "Invalid";
            })
        .def("from_json", [](std::string const &encoded) {
            static std::unordered_map<std::string, tt::DEVICE_MODE> decode = {
                {"CompileAndRun", tt::DEVICE_MODE::CompileAndRun},
                {"CompileOnly", tt::DEVICE_MODE::CompileOnly},
                {"RunOnly", tt::DEVICE_MODE::RunOnly},
            };
            return decode.at(encoded);
        });

    py::class_<tt::Stride>(m_backend, "StrideDescriptor")
        .def(py::init<>())
        .def_readwrite("xy_offsets", &tt::Stride::xy_offsets)
        .def_readwrite("stride", &tt::Stride::stride)
        .def(py::pickle(
            [](const tt::Stride &s) { // __getstate__
                return py::make_tuple(
                    s.xy_offsets,
                    s.stride);
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 2)
                {
                    throw std::runtime_error("Invalid state for tt::Stride");
                }

                tt::Stride s;

                s.xy_offsets = t[0].cast<std::vector<std::pair<int, int>>>(); 
                s.stride = t[1].cast<int>();

                return s;
            }
        ));

    py::class_<tt::tt_dram_io_desc>(m_backend, "DramIODesc")
        .def_property_readonly("name", [](tt::tt_dram_io_desc &self) { return self.queue_name; })
        .def_property_readonly("data_format", [](tt::tt_dram_io_desc &self) { return self.bufq_target_format; })
        .def_readwrite("bufq_grid_dim_r", &tt::tt_dram_io_desc::bufq_grid_dim_r)
        .def_readwrite("bufq_grid_dim_c", &tt::tt_dram_io_desc::bufq_grid_dim_c)
        .def_readwrite("ublock_rt", &tt::tt_dram_io_desc::ublock_rt)
        .def_readwrite("ublock_ct", &tt::tt_dram_io_desc::ublock_ct)
        .def_readwrite("mblock_m", &tt::tt_dram_io_desc::mblock_m)
        .def_readwrite("mblock_n", &tt::tt_dram_io_desc::mblock_n)
        .def_readwrite("tile_height", &tt::tt_dram_io_desc::tile_height)
        .def_readwrite("tile_width", &tt::tt_dram_io_desc::tile_width)
        .def_readwrite("t", &tt::tt_dram_io_desc::t)
        .def_readwrite("hstack_factor", &tt::tt_dram_io_desc::hstack_factor)
        .def_readwrite("vstack_factor", &tt::tt_dram_io_desc::vstack_factor)
        .def_readwrite("stack_row_major", &tt::tt_dram_io_desc::stack_row_major)
        .def_readwrite("s_descriptor", &tt::tt_dram_io_desc::s_descriptor)
        .def_readwrite("input_count", &tt::tt_dram_io_desc::input_count)
        .def_readwrite("netlist_path", &tt::tt_dram_io_desc::netlist_path)
        .def(py::pickle(
            [](const tt::tt_dram_io_desc &p) {  // __getstate__
                return py::make_tuple(
                    p.netlist_path,
                    p.queue_name,
                    p.bufq_grid_dim_r,
                    p.bufq_grid_dim_c,
                    p.bufq_num_slots,
                    p.ublock_rt,
                    p.ublock_ct,
                    p.mblock_m,
                    p.mblock_n,
                    p.tile_height,
                    p.tile_width,
                    p.t,
                    p.input_count,
                    p.hstack_factor,
                    p.vstack_factor,
                    p.stack_row_major,
                    p.bufq_target_format,
                    p.bufq_start_addr_channel,
                    p.bufq_entry_size,
                    p.io_type,
                    p.s_descriptor,
                    p.backend_type,
                    p.layout);
            },
            [](py::tuple t) {  // __setstate__
                if (t.size() != 23)
                    throw std::runtime_error("tt::tt_dram_io_desc: Invalid state!");

                tt::tt_dram_io_desc p;
                p.netlist_path = t[0].cast<std::string>();
                p.queue_name = t[1].cast<std::string>();
                p.bufq_grid_dim_r = t[2].cast<std::uint32_t>();
                p.bufq_grid_dim_c = t[3].cast<std::uint32_t>();
                p.bufq_num_slots = t[4].cast<std::uint32_t>();
                p.ublock_rt = t[5].cast<std::uint32_t>();
                p.ublock_ct = t[6].cast<std::uint32_t>();
                p.mblock_m = t[7].cast<std::uint32_t>();
                p.mblock_n = t[8].cast<std::uint32_t>();
                p.tile_height = t[9].cast<std::uint32_t>();
                p.tile_width = t[10].cast<std::uint32_t>();
                p.t = t[11].cast<std::uint32_t>();
                p.input_count = t[12].cast<std::uint32_t>();
                p.hstack_factor = t[13].cast<std::uint32_t>();
                p.vstack_factor = t[14].cast<std::uint32_t>();
                p.stack_row_major = t[15].cast<std::uint32_t>();
                p.bufq_target_format = t[16].cast<DataFormat>();
                p.bufq_start_addr_channel = t[17].cast<std::vector<std::pair<std::uint32_t, std::uint16_t>>>();
                p.bufq_entry_size = t[18].cast<std::uint32_t>();
                p.io_type = t[19].cast<IO_TYPE>();
                p.s_descriptor = t[20].cast<tt::Stride>();
                p.backend_type = t[21].cast<DEVICE>();
                p.layout = t[22].cast<IO_LAYOUT>();

                TT_ASSERT(
                    tt::backend::translate_addresses(p) == tt::DEVICE_STATUS_CODE::Success,
                    "Failed to translate addresses for " + p.queue_name);
                return p;
            }));

    py::class_<tt::tt_PytorchTensorDesc>(m_backend, "PytorchTensorDesc", py::buffer_protocol())
        .def(py::init([]() {
            return tt_PytorchTensorDesc();
        }))
        .def(py::init([](py::object pytorch_tensor, std::uint32_t itemsize, tt::DataFormat format,
                        std::uint32_t dim,
                        std::array<std::uint32_t, 4> shape,
                        std::array<std::uint32_t, 4> strides) {

            auto ptr = pytorch_tensor.attr("data_ptr")().cast<std::uint64_t>();
            py::handle handle = pytorch_tensor.release();

            return tt_PytorchTensorDesc(
                (void *)ptr, itemsize, format, shape, strides, dim, (void*)handle.ptr(), python_handle_refchange);
        }))
        .def(py::init([](void *buffer, std::uint32_t itemsize, tt::DataFormat format,
                        std::uint32_t dim,
                        std::array<std::uint32_t, 4> shape,
                        std::array<std::uint32_t, 4> strides) {

            return tt_PytorchTensorDesc(buffer, itemsize, format, shape, strides, dim);
        }))
        .def_readwrite("itemsize", &tt_PytorchTensorDesc::itemsize)
        .def_readwrite("format", &tt_PytorchTensorDesc::format)
        .def_readwrite("shape", &tt_PytorchTensorDesc::shape)
        .def_readwrite("strides", &tt_PytorchTensorDesc::strides)
        .def_readwrite("dim", &tt_PytorchTensorDesc::dim)
        .def("print", [](tt::tt_PytorchTensorDesc &self) {
            std::cout << "Descriptor: ptr=" << (std::uint64_t)self.ptr << 
                    ", itemsize=" << self.itemsize <<
                    ", format =" << (int)self.format <<
                    ", dim =" << self.dim <<
                    ", shape =" << self.shape[0] << "," << self.shape[1] << "," << self.shape[2] << "," << self.shape[3] <<
                    ", strides =" << self.strides[0] << "," << self.strides[1] << "," << self.strides[2] << "," << self.strides[3] << std::endl;
        })
        .def_buffer([](tt::tt_PytorchTensorDesc &desc) -> py::buffer_info {

           // Mostly irrelevant since we'll be turning this into torch tensor with its
           // own format. However, this could cause numpy to interpret the data wrong
           std::string data_format = py::format_descriptor<float>::format();
           return py::buffer_info(
                const_cast<void *>(desc.ptr),
                desc.itemsize,
                data_format,
                4,
                desc.shape,
                desc.strides);
         })
         .def(py::pickle(
           [](const tt::tt_PytorchTensorDesc &t) { // __getstate__
              return py::make_tuple(
                  reinterpret_cast<std::uintptr_t>(t.ptr),
                  t.itemsize,
                  t.format,
                  t.shape,
                  t.strides,
                  t.dim);
           },
           [](py::tuple t) { // __setstate__
              if (t.size() != 6)
                  throw std::runtime_error("tt::tt_PytorchTensorDesc: Invalid state!");

              tt::tt_PytorchTensorDesc p;
              p.ptr = reinterpret_cast<const void*>(t[0].cast<std::uintptr_t>());
              p.itemsize = t[1].cast<std::uint32_t>();
              p.format = t[2].cast<DataFormat>();
              p.shape = t[3].cast<std::array<std::uint32_t, 4>>();
              p.strides = t[4].cast<std::array<std::uint32_t, 4>>();
              p.dim = t[5].cast<std::uint32_t>();
              return p;
           }
          ));

    py::class_<tt::tt_TilizedTensorDesc>(m_backend, "TilizedTensorDesc")
        .def(py::init<>())
        .def_readwrite("num_buffers", &tt::tt_TilizedTensorDesc::num_buffers)
        .def_readwrite("buf_size_bytes", &tt::tt_TilizedTensorDesc::buf_size_bytes)
        .def_readwrite("format", &tt::tt_TilizedTensorDesc::format)
        .def("print", [](tt::tt_TilizedTensorDesc &self) {
            std::cout << "Descriptor: ptr=" << (std::uint64_t)self.ptr << 
                    ", num_buffers=" << self.num_buffers <<
                    ", buf_size_bytes=" << (int)self.buf_size_bytes <<
                    ", format =" << self.format;
        })
        .def(py::pickle(
           [](const tt::tt_TilizedTensorDesc &t) { // __getstate__
              return py::make_tuple(
                  t.num_buffers,
                  t.buf_size_bytes,
                  t.format);
           },
           [](py::tuple t) { // __setstate__
              if (t.size() != 3)
                  throw std::runtime_error("tt::tt_TilizedTensorDesc: Invalid state!");

              return tt::tt_TilizedTensorDesc(
                nullptr,
                t[0].cast<std::uint32_t>(),
                t[1].cast<std::uint32_t>(),
                t[2].cast<DataFormat>()
              );
           }
        ));


    py::class_<param::DeviceDesc>(m_backend, "BackendDeviceDesc")
        .def(py::init<>())
        .def_readonly("arch", &param::DeviceDesc::arch)
        .def_readonly("soc_desc_yaml", &param::DeviceDesc::soc_desc_yaml)
        .def_readonly("mmio", &param::DeviceDesc::mmio)
        .def_readonly("harvesting_mask", &param::DeviceDesc::harvesting_mask);

    py::class_<tt_op_model_desc>(m_backend, "OpModelDesc")
        .def(py::init<>())
        .def_readwrite("type", &tt_op_model_desc::type)
        .def_readwrite("arch", &tt_op_model_desc::arch)
        .def_readwrite("data_format", &tt_op_model_desc::data_format)
        .def_readwrite("math_fidelity", &tt_op_model_desc::math_fidelity)
        .def_readwrite("t", &tt_op_model_desc::t)
        .def_readwrite("mblock_m", &tt_op_model_desc::mblock_m)
        .def_readwrite("mblock_n", &tt_op_model_desc::mblock_n)
        .def_readwrite("ublock_rt", &tt_op_model_desc::ublock_rt)
        .def_readwrite("ublock_ct", &tt_op_model_desc::ublock_ct)
        .def_readwrite("mblock_k", &tt_op_model_desc::mblock_k)
        .def_readwrite("ublock_kt", &tt_op_model_desc::ublock_kt)
        .def_readwrite("sparse_indices", &tt_op_model_desc::sparse_indices)
        .def_readwrite("sparse_nz_ublocks", &tt_op_model_desc::sparse_nz_ublocks)
        .def_readwrite("sparse_nz_strips", &tt_op_model_desc::sparse_nz_strips)
        .def_readwrite("approx_mode", &tt_op_model_desc::approx_mode)
        .def_readwrite("op_attr", &tt_op_model_desc::op_attr)
        .def_readwrite("reduce_z", &tt_op_model_desc::reduce_z);

    py::enum_<tt::DEVICE_STATUS_CODE>(m_backend, "BackendStatusCode")
        .value("Success", tt::DEVICE_STATUS_CODE::Success)
        .value("RuntimeError", tt::DEVICE_STATUS_CODE::RuntimeError)
        .value("TimeoutError", tt::DEVICE_STATUS_CODE::TimeoutError);

    py::enum_<tt::COMPILE_FAILURE>(m_backend, "BackendCompileFailure")
        .value("BriscCompile", tt::COMPILE_FAILURE::BriscCompile)
        .value("EriscCompile",tt::COMPILE_FAILURE::EriscCompile)
        .value("NriscCompile",tt::COMPILE_FAILURE::NriscCompile)
        .value("Net2Pipe",tt::COMPILE_FAILURE::Net2Pipe)
        .value("PipeGen",tt::COMPILE_FAILURE::PipeGen)
        .value("BlobGen",tt::COMPILE_FAILURE::BlobGen)
        .value("L1Size",tt::COMPILE_FAILURE::L1Size)
        .value("OverlaySize",tt::COMPILE_FAILURE::OverlaySize)
        .value("Invalid",tt::COMPILE_FAILURE::Invalid);

    py::class_<tt_compile_result>(m_backend, "BackendCompileResult")
        .def(py::init<>())
        .def_readwrite("success", &tt_compile_result::success)
        .def_readwrite("failure_type", &tt_compile_result::failure_type)
        .def_readwrite("failure_message", &tt_compile_result::failure_message)
        .def_readwrite("failure_target",&tt_compile_result::failure_target)
        .def_readwrite("device_id",&tt_compile_result::device_id)
        .def_readwrite("temporal_epoch_id", &tt_compile_result::temporal_epoch_id)
        .def_readwrite("logical_core_x",&tt_compile_result::logical_core_x)
        .def_readwrite("logical_core_y", &tt_compile_result::logical_core_y)
        .def_readwrite("extra_size_bytes", &tt_compile_result::extra_size_bytes);
            
        
    py::class_<tt_backend, std::shared_ptr<tt_backend>>(m_backend, "BackendApi")
        .def(py::init(py::overload_cast<const std::string&, const tt::tt_backend_config&>(&tt_backend::create)))
        .def("initialize", py::overload_cast<>(&tt_backend::initialize), py::call_guard<py::gil_scoped_release>())
        .def("initialize", py::overload_cast<tt::tt_compile_result*>(&tt_backend::initialize), py::call_guard<py::gil_scoped_release>())
        .def("finish", &tt_backend::finish)
        .def("run_program", &tt_backend::run_program, py::call_guard<py::gil_scoped_release>())
        .def("wait_for_idle", &tt_backend::wait_for_idle, py::call_guard<py::gil_scoped_release>())

        .def("get_queue_descriptor", &tt_backend::get_queue_descriptor);

    // Explicitly release the backend pointer
    m_backend.def("release_backend_ptr", [](std::shared_ptr<tt_backend> backend) {
        backend.reset();
    });

    m_backend.def(
        "clear_backend_param_cache",
        &tt::backend::clear_backend_param_cache_v2);

    m_backend.def("get_op_model_execution_cycles", &tt::backend::get_op_model_execution_cycles);
    m_backend.def("get_op_model_param", &tt::backend::get_op_model_param);
    
    m_backend.def(
        "push_input", 
        py::overload_cast<
            const tt::tt_dram_io_desc&,
            const tt::tt_PytorchTensorDesc&,
            const bool, const int, const int>(&tt::backend::push_input), py::call_guard<py::gil_scoped_release>());
    m_backend.def(
        "push_input", 
        py::overload_cast<
            const tt::tt_dram_io_desc&,
            const tt::tt_TilizedTensorDesc&,
            const int, const int>(&tt::backend::push_input), py::call_guard<py::gil_scoped_release>());
    m_backend.def("pop_output", &tt::backend::pop_output, py::call_guard<py::gil_scoped_release>());
    m_backend.def("get_output", &tt::backend::get_output, py::call_guard<py::gil_scoped_release>());
    m_backend.def("free_tensor", &tt::backend::free_tensor<tt::tt_PytorchTensorDesc>);
    m_backend.def("free_tensor", &tt::backend::free_tensor<tt::tt_TilizedTensorDesc>);
    m_backend.def("tilize_tensor", &tt::backend::tilize_tensor);
    m_backend.def("binarize_tensor", &tt::backend::binarize_tensor<tt::tt_PytorchTensorDesc>);
    m_backend.def("binarize_tensor", &tt::backend::binarize_tensor<tt::tt_TilizedTensorDesc>);
    m_backend.def("debinarize_tensor", &tt::backend::debinarize_tensor<tt::tt_PytorchTensorDesc>);
    m_backend.def("debinarize_tensor", &tt::backend::debinarize_tensor<tt::tt_TilizedTensorDesc>);

    m_backend.def(
        "get_io_size_in_bytes",
        &tt::backend::get_io_size_in_bytes,
        py::arg("data_formati"),
        py::arg("is_untilizesd"),
        py::arg("ublock_ct"),
        py::arg("ublock_rt"),
        py::arg("mblock_m"),
        py::arg("mblock_n"),
        py::arg("t"),
        py::arg("entries"),
        py::arg("tile_height") = 32,
        py::arg("tile_width") = 32);
    m_backend.def("get_next_aligned_address", &tt::backend::get_next_aligned_address);

    m_backend.def("translate_addresses", &tt::backend::translate_addresses, py::call_guard<py::gil_scoped_release>());
    m_backend.def("get_format_from_string", &tt::get_format_from_string, py::arg("format_str"));
    m_backend.def(
        "detect_available_silicon_devices", &tt::backend::detect_available_devices, py::arg("only_detect_mmio") = true);
    m_backend.def(
        "get_device_descs_for_available_devices",
        &tt::backend::get_device_descs_for_available_devices,
        py::arg("out_dir") = std::string("./tt_build"));

    m_backend.def(
        "get_custom_device_desc",
        &tt::backend::get_custom_device_desc,
        py::arg("arch") = tt::ARCH::Invalid,
        py::arg("mmio") = false,
        py::arg("harvesting_mask") = 0u,
        py::arg("grid_dim") = std::make_pair(0, 0),
        py::arg("out_dir") = std::string("./tt_build"));
    m_backend.def("get_device_cluster_yaml", &tt::backend::get_device_cluster_yaml_v2, py::arg("out_dir"));
    m_backend.def("initialize_child_process", &tt::backend::initialize_child_process);
    m_backend.def("finish_child_process", &tt::backend::finish_child_process);
    m_backend.def("load_cached_sys_param", &tt::load_cached_sys_param);

    py::class_<DeviceGrid>(m_backend, "DeviceGrid")
        .def(py::init<std::pair<int, int>>())
        .def_readonly("r", &DeviceGrid::r)
        .def_readonly("c", &DeviceGrid::c);

    py::class_<DeviceConfig>(m_backend, "DeviceConfig")
        .def(py::init<
             std::string,
             std::string,
             std::string,
             std::string,
             std::string,
             bool,
             std::vector<std::uint32_t>>())
        .def(py::init<
             std::string,
             std::string,
             std::string,
             std::string,
             std::string,
             bool,
             std::vector<std::tuple<std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t>>>())
        .def("get_harvested_cfg", &DeviceConfig::get_harvested_cfg)
        .def("get_ethernet_connections", &DeviceConfig::get_ethernet_connections)
        .def("get_dram_backend_reserved_max", &DeviceConfig::get_dram_backend_reserved_max)
        .def("get_dram_num_channels", &DeviceConfig::get_dram_num_channels)
        .def("get_dram_channel_capacity", &DeviceConfig::get_dram_channel_capacity)
        .def("get_host_memory_channel_start_address", &DeviceConfig::get_host_memory_channel_start_address)
        .def("get_host_memory_num_channels", &DeviceConfig::get_host_memory_num_channels)
        .def("get_host_memory_channel_size", &DeviceConfig::get_host_memory_channel_size)
        .def_property_readonly(
            "arch", [](DeviceConfig const &dc) -> tt::ARCH { return get_arch_from_string(dc.arch_name); })
        .def_readonly("arch_name", &DeviceConfig::arch_name)
        .def_readonly("device_yaml", &DeviceConfig::device_yaml)
        .def_readonly("cluster_config_yaml", &DeviceConfig::cluster_config_yaml)
        .def_readonly("backend_type", &DeviceConfig::backend_type)
        .def_readonly("grid_size", &DeviceConfig::grid_size)
        .def_readonly("chip_ids", &DeviceConfig::chip_ids);
}
}  // namespace tt::backend_api
