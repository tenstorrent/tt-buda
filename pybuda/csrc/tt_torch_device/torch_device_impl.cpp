// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#define STRIP_ERROR_MESSAGES
#include <ATen/Context.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/InferSize.h>
#include <ATen/TensorUtils.h>
#include <torch/library.h>

#include <utility>

#include "pybuda/csrc/tt_torch_device/tt_device.hpp"
#include "pybuda/csrc/tt_torch_device/python_bindings.hpp"
#include "utils/assert.hpp"
#include "utils/logger.hpp"

namespace tt
{
// There are dummy enums defined in pytorch, like PrivateUse1 that can be used
// for bringing up new device types.  Eventually we should mainline an enum for
// Tenstorrent and not have to use the dummy device slot.
constexpr inline c10::DispatchKey DispatchKeyTT = c10::DispatchKey::PrivateUse1;

// TorchDevice interposes potentially many underlying HWDevices
class TorchDeviceImpl : public c10::impl::DeviceGuardImplInterface
{
   public:
    TorchDeviceImpl(std::vector<TTDevice> const& tt_devices) : tt_devices(tt_devices) {}

    // Torch overrides
    virtual c10::DeviceType type() const override { return TT; }
    virtual c10::Device exchangeDevice(c10::Device d) const override
    {
        TT_ASSERT(d.index() < (int)tt_devices.size());
        std::swap(current_device, d);
        return d;
    }
    virtual c10::Device getDevice() const override { return current_device; }
    virtual void setDevice(c10::Device d) const override
    {
        TT_ASSERT(d.index() < (int)tt_devices.size());
        current_device = d;
    }
    virtual void uncheckedSetDevice(c10::Device d) const noexcept override
    {
        TT_ASSERT(d.index() < (int)tt_devices.size());
        current_device = d;
    }
    virtual c10::Stream getStream(c10::Device d) const noexcept override
    {
        return c10::Stream(c10::Stream::UNSAFE, d, 0);
    }
    virtual c10::Stream getDefaultStream(c10::Device d) const override { return getStream(d); }
    virtual c10::Stream exchangeStream(c10::Stream) const noexcept override { return getStream(current_device); }
    virtual at::DeviceIndex deviceCount() const noexcept override
    {
        return static_cast<at::DeviceIndex>(tt_devices.size());
    }
    virtual bool queryStream(const c10::Stream&) const override { return false; }
    virtual void synchronizeStream(const c10::Stream&) const override {}

    // TT specific
    static TorchDeviceImpl& get()
    {
        static TorchDeviceImpl tt_device_impl(query_available_tt_devices());

        if (env_as<bool>("PYBUDA_DEVMODE")) {
            // If we are in dev mode, we want to mark all devices as golden
            for (auto & dev : tt_device_impl.tt_devices) {
                dev.type = tt::DEVICE::Golden;
            }
        }
        return tt_device_impl;
    }
    std::int64_t get_index() { return current_device.index(); }

    int get_next_unique_id() { return next_id++; }

    TTDevice getTTDevice() const
    {
        TT_ASSERT(current_device.index() < (int)tt_devices.size());
        return tt_devices[current_device.index()];
    }

    const TTDevice& getDefaultTTDevice() const
    {
        TT_ASSERT(not tt_devices.empty());
        return tt_devices.front();
    }

    std::vector<TTDevice> getTTDevices() const { return tt_devices; }

    std::map<const void*, std::string> registered_output_transforms;
    std::vector<std::string> ordered_input_trasforms;
    std::vector<const void*> copied_inputs;

   private:
    mutable c10::Device current_device = c10::Device(TT, 0);
    mutable c10::Stream current_stream = c10::Stream(c10::Stream::UNSAFE, c10::Device(TT, 0), 0);
    std::vector<TTDevice> tt_devices;
    int next_id = 0;
};

// register backend
c10::impl::DeviceGuardImplRegistrar tt_device_impl_reg(TT, &TorchDeviceImpl::get());

const TTDevice& get_default_tt_device() { return TorchDeviceImpl::get().getDefaultTTDevice();}
std::vector<TTDevice> get_available_tt_devices() { return TorchDeviceImpl::get().getTTDevices(); }

struct Mallocator final : public c10::Allocator
{
    virtual c10::DataPtr allocate(size_t n) const
    {
        void* ptr = std::calloc(n, 1);
        return c10::DataPtr(ptr, nullptr, std::free, c10::Device(TT, 0));
    }

    static c10::Allocator* get()
    {
        static std::unique_ptr<Mallocator> mallocator = std::make_unique<Mallocator>();
        return mallocator.get();
    }
};


void fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack)
{
    PyGILState_STATE gstate=PyGILState_Ensure();
    log_debug(LogTorchDevice, "Fallback {}\n", op.operator_name());
    PyGILState_Release(gstate);
    at::native::cpu_fallback(op, stack);
}

static torch::ScalarType df_to_torch_scalar_type(DataFormat df)
{
    switch (df)
    {
        case DataFormat::Int8: return torch::ScalarType::Byte;
        case DataFormat::UInt16: return torch::ScalarType::Short;
        case DataFormat::RawUInt32: return torch::ScalarType::Int;
        case DataFormat::Int32: return torch::ScalarType::Int;
        case DataFormat::Float16: return torch::ScalarType::Half;
        case DataFormat::Float32: return torch::ScalarType::Float;
        case DataFormat::Float16_b: return torch::ScalarType::BFloat16;
        default: break;
    }

    log_fatal(LogTTDevice, "Unhandled dtype {}", df);
}

static std::pair<std::vector<std::int64_t>, std::size_t> calculate_stride_size(
    torch::IntArrayRef size, torch::IntArrayRef stride, std::size_t scalar_size)
{
    TT_ASSERT(size.size() == stride.size());

    if (size.empty())
        return std::make_pair(std::vector<std::int64_t>{}, 0);

    std::vector<std::int64_t> buda_stride(stride.size());
    std::int64_t dim = (std::int64_t)size.size();

    std::size_t size_bytes = 1;
    for (std::int64_t i = dim - 1; i >= 0; --i)
    {
        buda_stride[i] = size_bytes;
        size_bytes *= (i - dim) >= -2 ? align_up_tile(size[i]) : size[i];
    }

    // Special case for single dim, we need to pad out the rows in the allocation
    if (dim == 1)
        size_bytes *= kTileDim;

    return std::make_pair(buda_stride, size_bytes * scalar_size);
}

void register_output_runtime_transform(torch::Tensor const& tensor, std::string transform)
{
    TorchDeviceImpl::get().registered_output_transforms[tensor.const_data_ptr()] = transform;
}

void register__ordered_input_runtime_transforms(std::vector<std::string> input_transforms)
{
    TorchDeviceImpl::get().ordered_input_trasforms = input_transforms;
}

std::vector<const void*> get_copied_inputs()
{
    auto ret = TorchDeviceImpl::get().copied_inputs;
    TorchDeviceImpl::get().copied_inputs.clear();
    return ret;
}

std::string get_runtime_transform(torch::Tensor const& tensor, bool input)
{
    if (not input)
    {
        TT_ASSERT(TorchDeviceImpl::get().registered_output_transforms.count(tensor.const_data_ptr()));
        return TorchDeviceImpl::get().registered_output_transforms[tensor.const_data_ptr()];
    }
    else
    {
        TT_ASSERT(TorchDeviceImpl::get().ordered_input_trasforms.size(), "Please do not copy inputs to device until compilation has finished, first run can be performed on CPU tensors");
        size_t input_index = TorchDeviceImpl::get().copied_inputs.size();
        TT_ASSERT(input_index < TorchDeviceImpl::get().ordered_input_trasforms.size());
        TorchDeviceImpl::get().copied_inputs.push_back(tensor.const_data_ptr());
        return TorchDeviceImpl::get().ordered_input_trasforms[input_index];
    }
}

torch::Tensor from_pytorch_tensor_desc(
    tt_PytorchTensorDesc const& desc, std::vector<std::int64_t> const& shape, FreePytorchTensorDescFn* free_fn)
{
    std::int64_t elemsize = static_cast<std::int64_t>(desc.itemsize);
    std::vector<std::int64_t> strides = {
        static_cast<std::int64_t>(desc.strides[0]) / elemsize,
        static_cast<std::int64_t>(desc.strides[1]) / elemsize,
        static_cast<std::int64_t>(desc.strides[2]) / elemsize,
        static_cast<std::int64_t>(desc.strides[3]) / elemsize,
    };
    std::vector<std::int64_t> aligned_shape;
    size_t dim = 0;
    for (auto s : shape)
    {
        if (shape.size() <= 2 or dim < shape.size() - 2)
            aligned_shape.push_back(s);
        else
            aligned_shape.push_back(align_up_tile(s));
        dim++;
    }
    TT_ASSERT(shape.size() <= strides.size());

    while (strides.size() > shape.size()) strides.erase(strides.begin());

    torch::ScalarType type = df_to_torch_scalar_type(desc.format);
    std::int64_t size_bytes = strides.front() * elemsize;

    tt_PytorchTensorDesc* ctx = new tt_PytorchTensorDesc(desc);
    c10::Storage storage(
        c10::Storage::use_byte_size_t(),
        size_bytes,
        at::DataPtr(const_cast<void*>(desc.ptr), static_cast<void*>(ctx), free_fn, at::Device(TT, TorchDeviceImpl::get().get_index())));

    c10::DispatchKeySet dispatch_keyset = c10::DispatchKeySet{DispatchKeyTT};
    c10::intrusive_ptr<c10::TensorImpl> impl = c10::make_intrusive<c10::TensorImpl>(
        std::move(storage), dispatch_keyset, caffe2::TypeMeta::fromScalarType(type));

    impl->set_sizes_and_strides(torch::IntArrayRef(aligned_shape), torch::IntArrayRef(strides));

    c10::intrusive_ptr<c10::BackendMeta> backend_meta{std::unique_ptr<c10::BackendMeta>(new TTMetaData())};
    TTMetaData *tt_meta = dynamic_cast<TTMetaData*>(backend_meta.get());
    tt_meta->runtime_transformed = false;
    tt_meta->created_on_device = true;
    tt_meta->unique_output_id = TorchDeviceImpl::get().get_next_unique_id();
    impl->set_backend_meta(backend_meta);

    return torch::Tensor::wrap_tensor_impl(impl);;
}

torch::Device torch_device_at_index(std::int64_t index) { return torch::Device(TT, index); }

std::string device_type_name(c10::DeviceType type, bool lower_case)
{
    return DeviceTypeName(type, lower_case);
}

torch::Tensor empty_strided(
    torch::IntArrayRef size,
    torch::IntArrayRef stride,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> /*layout*/,
    c10::optional<at::Device> /*device*/,
    c10::optional<bool> /*pin_memory*/)
{
    c10::ScalarType type = dtype ? *dtype : c10::kFloat;
    auto [buda_stride, size_bytes] = calculate_stride_size(size, stride, scalarTypeToTypeMeta(type).itemsize());

    if (size.size() == 1 and size[0] == 0)
    {
        buda_stride = {0};
    }

    c10::Storage storage(c10::Storage::use_byte_size_t(), size_bytes, Mallocator::get());

    c10::DispatchKeySet dispatch_keyset = c10::DispatchKeySet{DispatchKeyTT};
    c10::intrusive_ptr<c10::TensorImpl> impl = c10::make_intrusive<c10::TensorImpl>(
        std::move(storage), dispatch_keyset, caffe2::TypeMeta::fromScalarType(type));

    impl->set_sizes_and_strides(torch::IntArrayRef(size), torch::IntArrayRef(buda_stride));

    c10::intrusive_ptr<c10::BackendMeta> backend_meta{std::unique_ptr<c10::BackendMeta>(new TTMetaData())};
    TTMetaData *tt_meta = dynamic_cast<TTMetaData*>(backend_meta.get());
    tt_meta->original_shape = size;
    tt_meta->runtime_transformed = false;
    impl->set_backend_meta(backend_meta);
    
    auto t = torch::Tensor::wrap_tensor_impl(impl);
    PyGILState_STATE gstate=PyGILState_Ensure();
    log_trace(
        LogTorchDevice,
        "empty_strided tensor({})[{}, {}] [{}, {}, {}]",
        impl->data(),
        size,
        stride,
        t.device(),
        t.sizes(),
        t.strides());
    PyGILState_Release(gstate);
    return t;
}

torch::Tensor empty(
    torch::IntArrayRef size,
    c10::optional<c10::ScalarType> dtype,
    c10::optional<c10::Layout> layout,
    c10::optional<c10::Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<c10::MemoryFormat> /*memory_format*/)
{
    PyGILState_STATE gstate=PyGILState_Ensure();
    log_trace(
        LogTorchDevice,
        "empty size {}",
        size);
    PyGILState_Release(gstate);
    std::vector<std::int64_t> stride(size.size(), 1);

    std::int64_t curr_stride = 1;
    for (std::int64_t i = (std::int64_t)size.size() - 1; i >= 0; --i)
    {
        stride[i] = curr_stride;
        curr_stride *= size[i];
    }

    if (size.size() == 1 and size[0] == 0)
        stride = {0};

    return empty_strided(size, stride, dtype, layout, device, pin_memory);
}

// torch::Tensor to(
//     const torch::Tensor& self,
//     c10::optional<c10::ScalarType> dtype,
//     c10::optional<c10::Layout> layout,
//     c10::optional<c10::Device> device,
//     c10::optional<bool> pin_memory,
//     bool non_blocking,
//     bool copy,
//     c10::optional<c10::MemoryFormat> optional_memory_format
// ) 
// {
//     (void)dtype;
//     (void)layout;
//     (void)device;
//     (void)pin_memory;
//     (void)non_blocking;
//     (void)copy;
//     (void)optional_memory_format;
//     return self;
// }
// torch::Tensor _to_copy(
//     const torch::Tensor& self,
//     c10::optional<c10::ScalarType> dtype,
//     c10::optional<c10::Layout> layout,
//     c10::optional<c10::Device> device,
//     c10::optional<bool> pin_memory,
//     bool non_blocking,
//     c10::optional<c10::MemoryFormat> optional_memory_format)
// {
//     (void)dtype;
//     (void)layout;
//     (void)device;
//     (void)pin_memory;
//     (void)non_blocking;
//     (void)optional_memory_format;
//     //TODO: Implement me
//     return self;
// }

torch::Tensor _copy_from(const torch::Tensor& self, const torch::Tensor& dest, bool non_blocking)
{
    PyGILState_STATE gstate=PyGILState_Ensure();
    log_trace(
        LogTorchDevice,
        "copy_from self tensor({})[{} {} {}] dest tensor({})[{} {} {}] non_blocking[{}]",
        self.getIntrusivePtr()->data(),
        self.device(),
        self.sizes(),
        self.strides(),
        dest.getIntrusivePtr()->data(),
        dest.device(),
        dest.sizes(),
        dest.strides(),
        non_blocking);
    PyGILState_Release(gstate);

    TT_ASSERT(!non_blocking);
    TT_ASSERT(self.sizes() == dest.sizes());

    if (self.sizes().empty() or self.sizes().front() == 0)
        return dest;

    void* dest_tensor_data = dest.mutable_data_ptr();
    const void* self_tensor_data = self.getIntrusivePtr()->data();
    std::size_t dest_nbytes = dest.getIntrusivePtr()->storage().nbytes();
    std::size_t self_nbytes = self.getIntrusivePtr()->storage().nbytes();
    TT_ASSERT(self.strides().back() == 1);
    TT_ASSERT(dest.strides().back() == 1);
    if (self.device().is_cpu() and dest.device().type() == TT)
    {
        // pad to buda
        
        // std::string transform = get_runtime_transform(dest, true);
        // torch::Tensor evaled = eval_runtime_transform(self, transform);
        // self_tensor_data = evaled.getIntrusivePtr()->data();
        TT_ASSERT(self_nbytes <= dest_nbytes, self_nbytes, dest_nbytes);
        std::memcpy(dest_tensor_data, self_tensor_data, self_nbytes);
        // TODO
        // barrier dest
        // dest = self
    }
    else if (self.device().type() == TT and dest.device().is_cpu())
    {
        // narrow to pytorch

        TT_ASSERT(dest_nbytes <= self_nbytes, dest_nbytes, self_nbytes);
        std::memcpy(dest_tensor_data, self_tensor_data, dest_nbytes);

        
        // std::string transform = get_runtime_transform(self, false);
        // // TODO: Check for emptry transform to avoid the memcpy
        // torch::Tensor evaled = eval_runtime_transform(dest, transform);
        // self_tensor_data = evaled.getIntrusivePtr()->data();
        // std::memcpy(dest_tensor_data, self_tensor_data, dest_nbytes);
        // TODO
        // barrier self
        // dest = self
    }
    else if (self.device().type() == TT and dest.device().type() == TT)
    {
        // blit
        // or
        // barrier self
        // barrier dest
        // dest = self
        
        //log_fatal(
        //    "Unsupported (for now) _copy_from TTDevice[{}] to TTDevice[{}]",
        //    self.device().index(),
        //    dest.device().index());
        auto self_num_items = self.numel();
        auto dest_num_items = dest.numel();
        TT_ASSERT(self_num_items == dest_num_items, self_num_items, dest_num_items);
        std::memcpy(dest_tensor_data, self_tensor_data, self_nbytes);
    }
    else
    {
        log_fatal(
            "Unsupported _copy_from for supplied device combination: self[{}] dest[{}]",
            self.device().type(),
            dest.device().type());
    }

    return dest;
}

torch::Tensor _copy_from_and_resize(const torch::Tensor& self, const torch::Tensor& dest)
{
    PyGILState_STATE gstate=PyGILState_Ensure();
    log_debug(LogTorchDevice, "copy_from_and_resize");
    PyGILState_Release(gstate);
    if (dest.device().type() == TT)
    {
        auto [buda_stride, size_bytes] = calculate_stride_size(self.sizes(), self.strides(), self.dtype().itemsize());
        dest.getIntrusivePtr()->set_sizes_and_strides(self.sizes(), buda_stride);
    }

    return ::tt::_copy_from(self, dest, false);
}

torch::Tensor _reshape_alias(torch::Tensor const& self, c10::IntArrayRef size, c10::IntArrayRef stride)
{
    PyGILState_STATE gstate=PyGILState_Ensure();
    log_debug(LogTorchDevice, "reshape_alias");
    log_trace(
        LogTorchDevice,
        "reshape_alias tensor({})[{} {} {}] reshape[{} {}]",
        self.getIntrusivePtr()->data(),
        self.device(),
        self.sizes(),
        self.strides(),
        size,
        stride);
    PyGILState_Release(gstate);

    if (self.device().type() == TT)
    {
        auto [buda_stride, size_bytes] = calculate_stride_size(size, stride, self.dtype().itemsize());
        self.getIntrusivePtr()->set_sizes_and_strides(size, buda_stride);
    }
    else
    {
        self.getIntrusivePtr()->set_sizes_and_strides(size, stride);
    }
    return self;
}

torch::Tensor as_strided(const torch::Tensor & self, c10::IntArrayRef size, c10::IntArrayRef stride, c10::optional<int64_t> storage_offset)
{
    PyGILState_STATE gstate=PyGILState_Ensure();
    log_trace(LogTorchDevice, "as_strided tensor({})[{} {} {}] size[{}] stride[{}]",
        self.getIntrusivePtr()->data(),
        self.device(),
        self.sizes(),
        self.strides(),
        size,
        stride);
    PyGILState_Release(gstate);
    if (self.device().type() == TT)
    {
        auto [buda_stride, size_bytes] = calculate_stride_size(size, stride, self.dtype().itemsize());
        // TT_ASSERT(not storage_offset, "Unhandled");
        self.getIntrusivePtr()->set_sizes_and_strides(size, buda_stride);
    }
    else
    {
        self.getIntrusivePtr()->set_sizes_and_strides(size, stride);
    }

    if (storage_offset)
        self.getIntrusivePtr()->set_storage_offset(*storage_offset);

    return self.detach();
}

torch::Tensor & index_outf(const torch::Tensor &self, const c10::List<c10::optional<torch::Tensor>> & indices, torch::Tensor & out)
{
    PyGILState_STATE gstate=PyGILState_Ensure();
    log_trace(LogTorchDevice, "index_out tensor({})[{} {} {}]",
        out.getIntrusivePtr()->data(),
        out.device(),
        out.sizes(),
        out.strides());
    PyGILState_Release(gstate);

    auto cpu = self.to(torch::kCPU);

    // copy indices to cpu
    c10::List<c10::optional<torch::Tensor>> cpu_indices;
    for (c10::optional<torch::Tensor> index : indices)
    {
        if (index and index->defined())
        {
            cpu_indices.push_back(index->to(torch::kCPU));
        }
        else
        {
            cpu_indices.push_back(torch::Tensor());
        }
    }
    cpu.index(cpu_indices);
    out = cpu.to(self.device());
    
    return out;
}

torch::Tensor view(const torch::Tensor &self, const c10::IntArrayRef size)
{
    PyGILState_STATE gstate=PyGILState_Ensure();
    log_trace(LogTorchDevice, "view tensor({})[{} {} {}] [{}]",
        self.getIntrusivePtr()->data(),
        self.device(),
        self.sizes(),
        self.strides(),
        size);

    PyGILState_Release(gstate);
    at::DimVector inferred_size = at::infer_size_dv(size, self.numel());
    c10::optional<at::DimVector>  stride = at::detail::computeStride(self.sizes(),
                                            self.strides(),
                                            inferred_size);


    torch::Tensor ret = at::detail::make_tensor<torch::TensorImpl>(
      c10::TensorImpl::VIEW, c10::Storage(self.storage()), self.key_set(), self.dtype());

    auto* ret_tmp_ = ret.unsafeGetTensorImpl();
    ret_tmp_->set_storage_offset(self.storage_offset());
    if (stride)
        ret_tmp_->set_sizes_and_strides(torch::IntArrayRef(inferred_size), torch::IntArrayRef(*stride));
    else
        TT_ASSERT(false, "Unhandled");

    return ret;
}
}  // namespace tt

bool ops_registered = false;
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    if (ops_registered)
        return;
    ops_registered = true;
    m.impl("aten::empty.memory_format", &tt::empty);
    m.impl("aten::empty_strided", &tt::empty_strided);
    m.impl("aten::_copy_from", &tt::_copy_from);
    // m.impl("aten::_to_copy", &tt::_to_copy);
    // m.impl("aten::to", &tt::to);
    m.impl("aten::_copy_from_and_resize", &tt::_copy_from_and_resize);
    m.impl("aten::_reshape_alias", &tt::_reshape_alias);
    // m.impl("aten::as_strided", &tt::as_strided);
    m.impl("aten::index.Tensor_out", &tt::index_outf);
    m.impl("aten::view", &tt::view);
}

bool fallback_registered = false;
TORCH_LIBRARY_IMPL(_, PrivateUse1, m)
{
    if (fallback_registered)
        return;
    fallback_registered = true;
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&tt::fallback>());
}
