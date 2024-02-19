// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "pybuda/csrc/tt_torch_device/tt_device.hpp"

#include <map>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "pybuda/csrc/balancer/output_host_tm_types.hpp"
#include "third_party/budabackend/netlist/tt_backend.hpp"
#include "third_party/budabackend/netlist/tt_backend_api.hpp"
#include "utils/assert.hpp"
#include "utils/env.hpp"
#include "utils/logger.hpp"

namespace tt
{
struct RunPrograms
{
    std::vector<Program> programs;
    std::vector<torch::Tensor> inputs;
    std::unordered_map<std::string, torch::Tensor> parameters;
};

struct Barrier
{
};

using Command = std::variant<Barrier, RunPrograms>;

struct CommandQueue
{
    Workload* workload = nullptr;
    std::vector<Command> commands;
};

std::shared_ptr<Workload> compile(TTDevice& device, CompileRequest const& compile_request)
{
    TT_ASSERT(device.arch == compile_request.backend_config.arch);

    std::shared_ptr<Workload> workload = std::make_shared<Workload>(
        tt_backend::create(compile_request.netlist_path, compile_request.backend_config),
        compile_request.output_dir,
        compile_request.inputs,
        compile_request.constants,
        compile_request.parameters,
        compile_request.outputs);

    register__ordered_input_runtime_transforms(compile_request.input_runtime_transforms);
    device.input_runtime_transforms = compile_request.input_runtime_transforms;
    device.input_tile_bcast_dims = compile_request.input_tile_bcast_dims;
    device.output_runtime_transforms = compile_request.output_runtime_transforms;
    
    tt::tt_compile_result result;
    if (workload->backend->initialize(&result) != DEVICE_STATUS_CODE::Success)
        log_fatal(LogTTDevice, "Backend compile failed: {}", tt::get_string(result));

    workload->initialized = true;

    return workload;
}

static DataFormat torch_scalar_type_to_df(torch::ScalarType st)
{
    switch (st)
    {
        case torch::ScalarType::Byte: return DataFormat::Int8;
        case torch::ScalarType::Char: return DataFormat::Int8;
        case torch::ScalarType::Short: return DataFormat::UInt16;
        case torch::ScalarType::Int: return DataFormat::RawUInt32;
        case torch::ScalarType::Long: return DataFormat::RawUInt32;
        case torch::ScalarType::Half: return DataFormat::Float16;
        case torch::ScalarType::Float: return DataFormat::Float32;
        // case torch::ScalarType::Double:
        // case torch::ScalarType::ComplexHalf:
        // case torch::ScalarType::ComplexFloat:
        // case torch::ScalarType::ComplexDouble:
        // case torch::ScalarType::Bool:
        case torch::ScalarType::BFloat16: return DataFormat::Float16_b;
        default: break;
    }

    log_fatal(LogTTDevice, "Unhandled dtype {}", st);
}

static void free_tt_PytorchTensorDesc(void* ctx)
{
    tt_PytorchTensorDesc* desc = static_cast<tt_PytorchTensorDesc*>(ctx);
    backend::free_tensor(*desc);
    delete desc;
}

static tt_PytorchTensorDesc to_pytorch_tensor_desc(torch::Tensor const& tensor)
{
    // TT_ASSERT(tensor.is_contiguous());
    TT_ASSERT(tensor.dim() <= 4);
    TT_ASSERT(tensor.strides().size() == tensor.sizes().size());

    std::int64_t dim = (std::int64_t)tensor.sizes().size();
    TT_ASSERT(dim > 0);

    std::size_t scalar_size = tensor.element_size();
    std::array<std::uint32_t, PY_TENSOR_DIMS> shape = {1, 1, kTileDim, kTileDim};
    std::array<std::uint32_t, PY_TENSOR_DIMS> strides = {0, 0, 0, 0};

    int i = PY_TENSOR_DIMS - dim;
    for (auto s : tensor.sizes())
    {
        shape[i] = i >= 2 ? align_up_tile(s) : s;
        ++i;
    }

    i = PY_TENSOR_DIMS - 1;
    for (int j = dim - 1; j >= 0; --j, --i)
    {
        strides[i] = tensor.strides()[j] * scalar_size;
    }

    // Special case where dim == 1
    if (i >= 2)
    {
        strides[i] = kTileDim * scalar_size;
        --i;
    }

    int last = i + 1;
    while (i >= 0)
    {
        strides[i] = strides[last];
        --i;
    }

    return tt_PytorchTensorDesc(
        tensor.data_ptr(), tensor.element_size(), torch_scalar_type_to_df(tensor.scalar_type()), shape, strides, 4);
}

std::unordered_set<std::string> pushed;
void push_tensor(
    //tt_backend& backend,
    tt_dram_io_desc queue_desc,
    PyBudaTensorDesc const& desc,
    torch::Tensor const& tensor,
    std::string const& info)
{
    /*if (pushed.find(desc.name) != pushed.end())
    {
        std::cout << "Already pushed " << desc.name << std::endl;
        return;
    }
    
    // Only record if the name doesn't start with arg
    if (desc.name.find("arg") != 0)
    {
        pushed.insert(desc.name);
    }*/

    log_debug(
        LogTTDevice,
        "Pushing tensor({})[{}][{}] to device[{}]",
        tensor.data_ptr(),
        desc.name,
        tensor.scalar_type(),
        tensor.device());

    (void)info;
    // if (tensor.device().type() != TT)
    //     log_fatal(
    //         LogTTDevice,
    //         "Tensor is not resident on submitted device (forgot to call tensor.to(\"tt\")?) {}: device[{}] {}",
    //         desc.name,
    //         tensor.device(),
    //         info);

    //tt_dram_io_desc queue_desc = backend.get_queue_descriptor(desc.name);
    backend::translate_addresses(queue_desc);
    tt_PytorchTensorDesc tensor_desc = to_pytorch_tensor_desc(tensor);
    constexpr int kDefaultTimeoutSec = 10;
    constexpr bool push_one = false;
    auto status = backend::push_input(queue_desc, tensor_desc, push_one, kDefaultTimeoutSec, desc.ptr);
    if (status != DEVICE_STATUS_CODE::Success)
        log_fatal(LogTTDevice, "Failed to push tensor: {} {}", desc.name, status);
}

static torch::Tensor pop_tensor(tt_backend& backend, PyBudaTensorDesc const& desc, tt::balancer::OutputHostTM const& output_host_tm)
{
    log_debug(LogTTDevice, "Popping tensor[{}]", desc.name);

    tt_PytorchTensorDesc tensor_desc;
    tt_dram_io_desc queue_desc = backend.get_queue_descriptor(desc.name);
    backend::translate_addresses(queue_desc);
    queue_desc.hstack_factor = output_host_tm.hstack_factor;
    queue_desc.vstack_factor = output_host_tm.vstack_factor;
    queue_desc.stack_row_major = output_host_tm.row_major;

    constexpr bool pop_one = false;
    int timeout_in_seconds = 600;

    auto status = backend::get_output(queue_desc, tensor_desc, pop_one, timeout_in_seconds, desc.ptr);
    if (status != DEVICE_STATUS_CODE::Success)
        log_fatal(LogTTDevice, "Failed to get_output: {} {}", desc.name, status);

    // TODO: cannot call on RAM
    status = backend::pop_output(queue_desc, pop_one, timeout_in_seconds);
    if (status != DEVICE_STATUS_CODE::Success)
        log_fatal(LogTTDevice, "Failed to pop_output: {} {}", desc.name, status);

    torch::Tensor ret = from_pytorch_tensor_desc(tensor_desc, desc.shape, free_tt_PytorchTensorDesc);
    
    return ret;
}

std::vector<torch::Tensor> dispatch(
    TTDevice const& device,
    std::shared_ptr<Workload> workload,
    std::vector<Program> const& programs,
    std::vector<torch::Tensor> const& inputs,
    tt::balancer::OutputHostTMMap const& output_host_tms)
{
    bool expected = false;
    if (device.context->initialized.compare_exchange_strong(
            expected, true, std::memory_order_relaxed, std::memory_order_relaxed))
    {
        backend::initialize_child_process(workload->output_dir);
    }

    int input_idx = 0;
    // if input hasn't been transformed (first time running) we need to transform it now
    TTMetaData *input_meta;
    std::vector<const void*> copied_inputs = get_copied_inputs();
    // TT_ASSERT(copied_inputs.size() == inputs.size());
    for (auto const& desc : workload->inputs)
    {
        torch::Tensor const& input = inputs.at(input_idx);
        auto impl = input.unsafeGetTensorImpl();
        input_meta = dynamic_cast<TTMetaData*>(impl->get_backend_meta());
        TT_ASSERT (input_meta != nullptr);
        if (!input_meta->runtime_transformed)
        {
            std::string runtime_transform = device.input_runtime_transforms.at(input_idx);
            std::vector<int> tile_bcast_dims = device.input_tile_bcast_dims.at(input_idx);
            auto [transformed_input, q_updated] = eval_runtime_transform(input.to(torch::kCPU), runtime_transform, tile_bcast_dims, workload->backend->get_queue_descriptor(desc.name));
            input_meta->runtime_transformed = true;
            push_tensor(q_updated, desc, transformed_input, fmt::format("input[{}]", input_idx));
        }
        else
        {
            push_tensor(workload->backend->get_queue_descriptor(desc.name), desc, input, fmt::format("input[{}]", input_idx));
        }
        // TT_ASSERT(copied_inputs.at(input_idx) == input.const_data_ptr(), "Incorrect input pointer, input tensors need to be copied to device in the same order as they'll be consumed");
        ++input_idx;
    }

    for (Program const& program : programs)
    {
        auto status = workload->backend->run_program(program.name, program.parameters);
        if (status != DEVICE_STATUS_CODE::Success)
            log_fatal(LogTTDevice, "Failed to run_program: {} {}", program.name, status);
    }

    std::vector<torch::Tensor> outputs;
    outputs.reserve(workload->outputs.size());
    for (size_t i = 0; i < workload->outputs.size(); ++i)
    {
        PyBudaTensorDesc const& desc = workload->outputs.at(i);
        tt::balancer::OutputHostTM output_host_tm = tt::balancer::OutputHostTM(); 
        if (output_host_tms.count(desc.name))
            output_host_tm = output_host_tms.at(desc.name);

        torch::Tensor output = pop_tensor(*workload->backend, desc, output_host_tm);
        std::string runtime_transform = device.output_runtime_transforms.at(i);
        register_output_runtime_transform(output, runtime_transform);
        outputs.emplace_back(output);
    }
    return outputs;
}

std::vector<TTDevice> query_available_tt_devices()
{
    static std::shared_ptr<TTContext> context = std::make_shared<TTContext>();
    std::vector<TTDevice> d;
    auto available_devices = backend::get_device_descs_for_available_devices();
    if (available_devices.empty())
    {
        constexpr bool mmio = true;
        ARCH arch = env_as<bool>("GOLDEN_WORMHOLE_B0") ? ARCH::WORMHOLE_B0 : ARCH::GRAYSKULL;
        auto desc = backend::get_custom_device_desc(arch, mmio);
        d.emplace_back(DEVICE::Golden, arch, desc.soc_desc_yaml, desc.mmio, 0, context);
    }
    else
    {
        int index = 0;
        for (auto desc : available_devices)
        {
            d.emplace_back(DEVICE::Silicon, desc.arch, desc.soc_desc_yaml, desc.mmio, index++, context);
        }
    }

    if (d.empty())
        log_fatal(LogTTDevice, "No available devices detected (To run with golden device, set PYBUDA_DEVMODE=1)");

    log_debug(LogTTDevice, "Available devices:");
    for (int i = 0; i < (int)d.size(); ++i) log_debug(LogTTDevice, "  [{}] {} {}", i, d[i].type, d[i].arch);
    return d;
}

std::string get_device_cluster_yaml(TTDevice const&) { return backend::get_device_cluster_yaml(); }

std::string to_string(TTDevice const& d)
{
    return device_type_name(TT, true /*lower_case*/) + ":" + std::to_string(d.index);
}

torch::Device torch_device(TTDevice const& d) { return torch_device_at_index(d.index); }

TTContext::~TTContext()
{
    if (initialized.load(std::memory_order_relaxed))
        backend::finish_child_process();
}

std::tuple<torch::Tensor, tt_dram_io_desc> eval_runtime_transform(
    const torch::Tensor& tensor,
    std::string transform,
    std::vector<int> &tile_bcast_dims,
    tt_dram_io_desc q)
{
    py::object py_tensor = py::reinterpret_steal<py::object>(THPVariable_Wrap(tensor));

    PyGILState_STATE gstate=PyGILState_Ensure();
    auto module = py::module_::import("pybuda.tensor");
    py::function eval_transform = module.attr("eval_runtime_transform");
    py::tuple py_result = eval_transform(transform, py_tensor, q, tile_bcast_dims);
    PyGILState_Release(gstate);
    torch::Tensor torch_tensor = THPVariable_Unpack(static_cast<PyObject *>(py_result[0].ptr()));
    tt_dram_io_desc q_updated = py_result[1].cast<tt_dram_io_desc>();
    return std::make_pair(torch_tensor, q_updated);
}

torch::Tensor narrow_to_pytorch(const torch::Tensor& tensor, std::string transform)
{
    //TODO
    py::object py_tensor = py::reinterpret_steal<py::object>(THPVariable_Wrap(tensor));

    PyGILState_STATE gstate=PyGILState_Ensure();
    auto module = py::module_::import("pybuda.tensor");
    py::function eval_transform = module.attr("eval_runtime_transform"); //TODO: update
    py::object py_result = eval_transform(transform, py_tensor);
    PyGILState_Release(gstate);
    torch::Tensor torch_tensor = THPVariable_Unpack(static_cast<PyObject *>(py_result.ptr()));
    return torch_tensor;
}

bool is_created_on_device(const torch::Tensor& tensor)
{
    auto impl = tensor.unsafeGetTensorImpl();
    TTMetaData* meta = dynamic_cast<TTMetaData*>(impl->get_backend_meta());
    TT_ASSERT(meta != nullptr);
    return meta->created_on_device;
}
std::vector<size_t> original_shape(const torch::Tensor& tensor)
{
    auto impl = tensor.unsafeGetTensorImpl();
    TTMetaData* meta = dynamic_cast<TTMetaData*>(impl->get_backend_meta());
    TT_ASSERT(meta != nullptr);
    std::vector<size_t> shape;
    for (auto s : meta->original_shape)
        shape.push_back(s);

    return shape;
}

}  // namespace tt
