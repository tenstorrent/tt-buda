# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Verify modules, or pipeline of modules, using the backend
"""
import os

from typing import Tuple, Dict, List, Optional
import queue

from loguru import logger
from pybuda.tvm_utils import map_tf_dtype_to_pt, flatten_inputs
import torch
import tensorflow as tf

import pybuda
from pybuda import Module
from ..tensor import Tensor, TensorShape, to_pt_tensors, to_buda_tensors, remove_microbatch
from .config import VerifyConfig, should_waive_gradient, _get_global_verify_config
from pybuda._C.backend_api import BackendType, DeviceMode
from ..module import PyTorchModule, Module, PyBudaModule
from ..config import CompilerConfig, _get_global_compiler_config, CompileDepth, _set_global_compiler_config
from ..parameter import Parameter
from ..device import Device
from .cpueval import cpueval_inference, cpueval_training
from .utils import CPUCombiner
from ..pybudaglobal import get_devices


def pybuda_override_veto_gen_overrides_json(override_file_path, compiler_cfg, new_g_compiler_config, removed_env_based_configurations):
    import json
    from deepdiff import DeepDiff

    # Create veto compile file if doesn't exist
    veto_compile_output = {}
    if not os.path.exists(override_file_path):
        with open(override_file_path, "w") as f:
            f.write(json.dumps(veto_compile_output))

    # Read veto compile file
    with open(override_file_path, "r") as f:
        veto_compile_output = json.loads(f.read())

    # Append test name to the compile output
    test_command = os.environ["PYBUDA_OVERRIDES_VETO_CUSTOM_SETUP"]
    if test_command not in veto_compile_output:
        veto_compile_output[test_command] = {}
        veto_compile_output[test_command]["cnf"] = {}
        veto_compile_output[test_command]["env"] = {}
    else:
        assert False

    # Collect general compiler configurations
    compiler_cfg_diff = DeepDiff(new_g_compiler_config, compiler_cfg)
    if "values_changed" in compiler_cfg_diff:
        for changed_cfg_data in compiler_cfg_diff['values_changed']:
            updated_value = compiler_cfg_diff['values_changed'][changed_cfg_data]['new_value']

            # Special case name handling
            if "root.compile_depth" in changed_cfg_data:
                if changed_cfg_data == "root.compile_depth.name":
                    continue
                elif changed_cfg_data == "root.compile_depth.value":
                    changed_cfg_data = "root.compile_depth"
            
            veto_compile_output[test_command]["cnf"][changed_cfg_data.split('root.')[-1]] = updated_value

    # env_var_config_state = {key: value for key, value in os.environ.items() if key.startswith(('PYBUDA_', 'TT_BACKEND_')) and key != "PYBUDA_OVERRIDES_VETO" and key != "PYBUDA_OVERRIDES_VETO_CUSTOM_SETUP"}
    # env_var_diff = DeepDiff(current_env_var_config_state, env_var_config_state)
    if len(removed_env_based_configurations) > 0:
        for key, val in removed_env_based_configurations.items():
            veto_compile_output[test_command]["env"][key] = val
            
    # Write veto compile file
    with open(override_file_path, "w") as f:
        f.write(json.dumps(veto_compile_output))

def pybuda_override_veto(compiler_cfg):
    import json

    # 1. Tackle with global compiler configurations
    new_g_compiler_config = CompilerConfig()

    # Override the default compiler config with the user specified config
    logger.info("Overriding general compiler configs (ones not specified are all removed):")
    for key, value in json.loads(os.environ["PYBUDA_OVERRIDES_VETO"])["general_conf"].items():
        if value is not None and value != "":
            logger.info("  Overriding '{}' key with '{}'", key, value)
            new_g_compiler_config.__setattr__(key, value)
        elif value is not None and value == "":
            current_value = getattr(compiler_cfg, key)
            logger.info("  Using default key '{}' with '{}' value", key, current_value)
            new_g_compiler_config.__setattr__(key, current_value)
        else:
            assert False, "Shouldn't hit this case"
    _set_global_compiler_config(new_g_compiler_config)

    # 2. Tackle with compiler configurations set through environment
    # variables

    # Get currently set compiler configurations as environment variables
    initial_env_var_config_state =  json.loads(os.environ["PYBUDA_OVERRIDES_VETO"])["environ_conf"]
    current_env_var_config_state = {key: value for key, value in os.environ.items() if key.startswith(('PYBUDA_', 'TT_BACKEND_')) and key != "PYBUDA_OVERRIDES_VETO" and key != "PYBUDA_OVERRIDES_VETO_CUSTOM_SETUP"}

    # Remove and update reference env configs
    removed_env_based_configurations = {}
    logger.info("Overriding env var compiler configs:")
    for key, value in current_env_var_config_state.items():
        if key not in initial_env_var_config_state:
            logger.info("  Removing '{}' key from env var config", key)
            removed_env_based_configurations[key] = value
            del os.environ[key]
        elif key in initial_env_var_config_state and initial_env_var_config_state[key] != "" and initial_env_var_config_state[key] != value:
            logger.info("  Overriding '{}' key with '{}'", key, initial_env_var_config_state[key])
            os.environ[key] = initial_env_var_config_state[key]
        elif key in initial_env_var_config_state and initial_env_var_config_state[key] == "":
            logger.info("  Using default key '{}' with '{}' value", key, value)
            os.environ[key] = value
        else:
            logger.info("  Keeping '{}' key as '{}' value", key, value)

    # Add new env configs
    for key, value in initial_env_var_config_state.items():
        os.environ[key] = value
        if value == "":
            continue
        if key not in current_env_var_config_state:
            logger.info("  Adding '{}' key with '{}' value", key, value)
            os.environ[key] = value

    # 3. Generate overrides json that contain all non-default overrides set
    if "PYBUDA_OVERRIDES_VETO_CUSTOM_SETUP" in os.environ:
        override_file_path = "override_veto_compile_output.json"
        pybuda_override_veto_gen_overrides_json(override_file_path, compiler_cfg, new_g_compiler_config, removed_env_based_configurations)
        
    return new_g_compiler_config


def _generate_random_inputs(input_shapes: List[Tuple], input_params: List[Dict], verify_cfg: VerifyConfig, uniform_inputs: bool, inputs_centered_on_zero: bool) -> List[Tuple[Tensor, ...]]:
    """
    Generate random inputs with shapes and input parameters provided
    """
    inputs = []
    for _ in range(verify_cfg.total_number_of_inputs()):

        def create_tensor(shape, i):
            dtype = torch.float32
            requires_grad = True
            dev_data_format = None
            if len(input_params) > i:
                if "data_format" in input_params[i]:
                    dtype = input_params[i]["data_format"]
                if "requires_grad" in input_params[i]:
                    requires_grad = input_params[i]["requires_grad"]
                if "dev_data_format" in input_params[i]:
                    dev_data_format = input_params[i]["dev_data_format"]

            if type(dtype) == tf.DType:
                dtype = map_tf_dtype_to_pt(dtype)

            if dtype in [torch.int8, torch.int, torch.int64]:
                return Tensor.create_from_torch(torch.randint(high=25000, size=shape, dtype=dtype), dev_data_format=dev_data_format)
            
            # avoid zeros
            if uniform_inputs:
                t = torch.rand(*shape, dtype=dtype)
                if inputs_centered_on_zero:
                    t = t - 0.5
            else:
                mean = 0.0 if inputs_centered_on_zero else 0.5
                t = torch.abs(torch.normal(mean=mean, std=0.1, size=shape, dtype=dtype)) + 0.00001
                t = t.detach()
            t.requires_grad = requires_grad
            return Tensor.create_from_torch(t)
                        
        inputs.append(tuple(create_tensor(shape, i) for i, shape in enumerate(input_shapes)))

    return inputs

def _translate_framework_modules(
        modules: List[Module], 
        device_types: List[str], 
        module_to_device: Dict[int, int], 
        pipe_inputs: Tuple[Tensor, ...],
        dev_count: int,
        verify_cfg: VerifyConfig,
        compiler_cfg: CompilerConfig):
    """
    Translate any framework modules to PyBuda python
    """
    modules_copy = []
    i = 0
    while i < len(modules):
        module = modules[i]
        if verify_cfg.verify_pipeline_result_vs_framework:
            modules_copy.append(module)

        pt_inputs = to_pt_tensors(pipe_inputs)
        if device_types[module_to_device[i]] == "CPUDevice":
            pipe_inputs = module.forward(*pt_inputs)
        elif not isinstance(module, PyBudaModule) and device_types[module_to_device[i]] == "TTDevice":
            from pybuda.tvm_to_python import generate_pybuda_module
            translated_modules, translated_device_types, _ = generate_pybuda_module(module, pt_inputs, compiler_cfg, module.name, verify_cfg)
            modules[i:i+1] = translated_modules
            device_types[module_to_device[i]:module_to_device[i]+1] = translated_device_types
            if len(translated_modules) > 1:
                extra_devices = len(translated_modules) - 1
                dev_count += extra_devices
                updated_module_to_device = {}
                for k, v in module_to_device.items():
                    if k < i:
                        updated_module_to_device[k] = v
                    elif k == i:
                        for new_device in range(len(translated_modules)):
                            updated_module_to_device[k + new_device] = v + new_device
                    else:
                        updated_module_to_device[k + extra_devices] = v + extra_devices
                module_to_device = updated_module_to_device
                i += extra_devices
        i += 1

    return modules, device_types, module_to_device, dev_count, modules_copy

            


def _update_and_randomize_params(
        modules: List[Module], 
        dev_count: int,
        module_to_device: Dict[int, int], 
        all_parameters: List[List[Parameter]],
        params_centered_on_zero: bool,
        scale_params: float,
        uniform_inputs: bool) -> List[Dict[str, torch.Tensor]]:
    """
    Randomize parameters, and pick up new ones from changes in compilcation
    """
    params_changed = False
    for m in modules:
        params_changed = params_changed or any([param not in all_parameters[0] for param in m.get_parameters()])
    
    if params_changed:
        new_all_parameters: List[List[Parameter]] = [ [] for _ in range(dev_count) ]
        for i, m in enumerate(modules):
            if hasattr(m.device, "get_parameters"):
                new_all_parameters[module_to_device[i]].extend(m.device.get_parameters())
            else:
                new_all_parameters[module_to_device[i]].extend(m.get_parameters())
        all_parameters = new_all_parameters

    # Randomize (if values are not set already), and save parameters
    all_saved_parameters: List[Dict[str, torch.Tensor]] = []
    for dev_index in range(dev_count):
        all_saved_parameters.append( {} )
        for p in all_parameters[dev_index]:
            if not p.has_value():
                if uniform_inputs:
                    t = torch.rand(*p.shape.get_pytorch_shape(), dtype=p.pt_data_format)
                    if params_centered_on_zero:
                        t -= 0.5
                else:
                    if params_centered_on_zero:
                        t = torch.normal(mean=0.0, std=0.1, size=p.shape.get_pytorch_shape(), dtype=p.pt_data_format)
                    else:
                        t = torch.abs(torch.normal(mean=0.5, std=0.1, size=p.shape.get_pytorch_shape(), dtype=p.pt_data_format)) + 0.00001
                assert scale_params > 0
                t /= scale_params

                # TODO: add data types, control over requires_grad
                t = t.detach()
                t.requires_grad = True
                p.set_value(t)
            else:
                t = p.value()
            #saved_t = t.type(torch.bfloat16)
            #saved_t = saved_t.detach()
            saved_t = t.clone().detach()
            
            saved_t.requires_grad = t.requires_grad
            all_saved_parameters[dev_index][p.get_name()] = saved_t
    
    return all_saved_parameters

def _update_parameters_from_checkpoint(all_saved_parameters: List[Dict[str, torch.Tensor]], device_index: int, checkpoint: Dict[str, Tensor]):
    """
    Updated saved parameters from device checkpoint, bringing the golden back into "sync"
    """
    device_parameters = all_saved_parameters[device_index]
    for name in device_parameters:
        assert name in checkpoint
        t = checkpoint[name].value().detach()
        #t = t.type(torch.float32)
        t.requires_grad = True
        # NB: we want to update the parameter in-place so any saved optimizer state is retained
        all_saved_parameters[device_index][name].data.copy_(t.data)

def _create_devices(modules: List[Module], device_types: List[str], module_to_device: Dict[int, int], sample_inputs: List[Tensor] ,verify_cfg: VerifyConfig, all_parameters: List[List[Parameter]], loss_module: Module) -> List[Device]:
    """
    Create devices to run the modules on
    """
    def _wrap_inputs(inputs):
        if not isinstance(inputs, (list, tuple)):
            inputs = (inputs, )
        return inputs
    
    inputs = sample_inputs
    from ..cpudevice import CPUDevice
    from ..gpudevice import GPUDevice
    devices: List[Device] = []
    for index, device_type in enumerate(device_types):
        module = modules[module_to_device[index]]
        if verify_cfg.test_kind.is_training():
            if verify_cfg.optimizer:
                optimizer_type = pybuda.optimizers.get_optimizer_type_from_string(verify_cfg.optimizer["type"])
                optimizer = optimizer_type(
                    parameters=all_parameters[index], **verify_cfg.optimizer["params"]
                )
            else:
                optimizer = None

            scheduler = verify_cfg.scheduler["type"](
               optimizer
            ) if verify_cfg.scheduler else None

            if device_type == "CPUDevice" or device_type == "GPUDevice":
                def optimizer_f(m: torch.nn.Module):
                    assert verify_cfg.optimizer["type"] == "sgd"
                    lr = verify_cfg.optimizer["params"]["learning_rate"]
                    opt_klass = torch.optim.SGD
                    if isinstance(m, torch.nn.Module):
                        opt_klass = torch.optim.SGD
                        return opt_klass(m.parameters(), lr=lr)
                    elif isinstance(m, (tf.keras.Model, tf.keras.layers.Layer)):
                        opt_klass = tf.keras.optimizers.legacy.SGD
                        return opt_klass(lr=lr)
                    else:
                        return opt_klass(m, lr=lr)
                
                def scheduler_f(o):
                    if scheduler is None:
                        return 
                    elif isinstance(o, torch.optim.Optimizer):
                        return scheduler.get_pytorch_scheduler(o)
                    else:
                        logger.warning("Scheduler not yet supported for other types of optimizers")
            
            if device_type == "CPUDevice":
                tt0 = CPUDevice(f"cpu{index}", optimizer_f=optimizer_f, scheduler_f=scheduler_f)
                devices.append(tt0)
            elif device_type == "GPUDevice":
                tt0 = GPUDevice(f"gpu{index}", optimizer_f=optimizer_f, scheduler_f=scheduler_f)
                devices.append(tt0)
            elif device_type == "TTDevice":
                if verify_cfg.devmode == DeviceMode.RunOnly:
                    tt0 = pybuda.TTDevice.load_image(img_path=verify_cfg.tti_path)
                else:
                    tt0 = pybuda.TTDevice(f"tt{index}", devtype=verify_cfg.devtype, arch=verify_cfg.arch, device_mode=verify_cfg.devmode, optimizer=optimizer, scheduler=scheduler, num_chips=verify_cfg.num_chips, chip_ids=verify_cfg.chip_ids, fp32_fallback=verify_cfg.fp32_fallback)
                devices.append(tt0)
            elif device_type == "-":
                assert len(devices) > 0, "At least one device must be chosen before '-' is used"
            else:
                raise RuntimeError("Unsupported device type")

            if optimizer:
                optimizer.set_optimizer_parameters()

        else:
            if device_type == "CPUDevice":
                inputs = to_pt_tensors(inputs)
                input_dtypes = [inp.dtype for inp in inputs]
                tt0 = CPUDevice(f"cpu{index}", input_dtypes=input_dtypes)
                devices.append(tt0)
            elif device_type == "GPUDevice":
                inputs = to_pt_tensors(inputs)
                tt0 = GPUDevice(f"gpu{index}")
                devices.append(tt0)
            elif device_type == "TTDevice":
                inputs = to_buda_tensors(inputs)
                if verify_cfg.devmode == DeviceMode.RunOnly:
                    tt0 = pybuda.TTDevice.load_image(img_path=verify_cfg.tti_path)
                else:
                    tt0 = pybuda.TTDevice(f"tt{index}", devtype=verify_cfg.devtype, arch=verify_cfg.arch, device_mode=verify_cfg.devmode, num_chips=verify_cfg.num_chips, chip_ids=verify_cfg.chip_ids, fp32_fallback=verify_cfg.fp32_fallback)
                devices.append(tt0)
            elif device_type == "-":
                assert len(devices) > 0, "At least one device must be chosen before '-' is used"
            else:
                raise RuntimeError("Unsupported device type")

            inputs = _wrap_inputs(module.forward(*inputs))

    for index, module in enumerate(modules):
        target_device = devices[module_to_device[index]]
        if verify_cfg.devmode == DeviceMode.RunOnly and isinstance(target_device, pybuda.TTDevice):
            continue
        target_device.place_module(module)

    if verify_cfg.test_kind.is_training() and loss_module is not None:
        if not (verify_cfg.devmode == DeviceMode.RunOnly and isinstance(devices[-1], pybuda.TTDevice)):
            devices[-1].place_loss_module(loss_module)
    return devices

def _setup_training(devices: List[Device], first_inputs: Tuple[Tensor], verify_cfg: VerifyConfig, add_loss: bool) -> List[Tensor]:
    """
    Create loss device / module, generate random targets.
    """
    from ..cpudevice import CPUDevice
    from ..gpudevice import GPUDevice
    target_tensors = None
    outputs = remove_microbatch(first_inputs)
    for d in devices:
        if type(d) == CPUDevice:
            outputs = to_pt_tensors(outputs)
            outputs = d._modules_forward(*outputs)
        elif type(d) == GPUDevice:
            outputs = to_pt_tensors(outputs)
            outputs = (output.to(d.device) for output in outputs)
            outputs = d._modules_forward(*outputs)
            outputs = (output.to("cpu") for output in outputs)
        elif isinstance(d, pybuda.TTDevice):
            outputs = to_buda_tensors(outputs)
            _, outputs, _, _, target_tensors = d.generate_graph(*outputs, return_intermediate="False", graph_name="PreVerify", compiler_cfg=CompilerConfig(), trace_only=True, verify_cfg=verify_cfg)
        else:
            raise RuntimeError(f"Unsupported device type: {type(d)}")
    output_shapes = [out.shape for out in outputs]

    # Create loss modules
    if add_loss:
        # Need to add CPU loss calculation
        cpu = CPUDevice("cpu_loss", 
                optimizer_f = None, # No parameters in cpu modules
                scheduler_f = None)

        identity = PyTorchModule("combiner0", CPUCombiner())
        cpu.place_module(identity)

        loss_module = PyTorchModule("l1loss", torch.nn.L1Loss())
        cpu.place_loss_module(loss_module)


    # Generate random targets
    targets = []
    for _ in range(verify_cfg.total_number_of_inputs()):
        if add_loss:
            # If we must add a loss (i.e no loss module is provided), CPUCombiner
            # sill pad all outputs to the shape of the largest output, and return the
            # sum of those tensors. And so the target must match that shape too.
            target = []
            def max_shape(shapes):
                mshp = [1]*4
                for i in range(-1, -4, -1):
                    mx = 1
                    for shape in shapes:
                        if len(shape) < -i:
                            continue
                        if shape[i] > mx:
                            mx = shape[i]
                    mshp[i] = mx
                return tuple(mshp)

            output_shape = max_shape(output_shapes)
            if isinstance(output_shape, TensorShape):
                output_shape = output_shape.get_pytorch_shape()
            target.append(torch.normal(mean=0.0, std=0.5, size=output_shape))
            cpu.push_to_target_inputs(tuple(target))
        else:
            assert len(target_tensors) > 0, "Target tensors are missing even though loss module exists on this device"
            target = [torch.normal(mean=0.0, std=0.5, size=target_tensor.shape.get_pytorch_shape()) for target_tensor in target_tensors]
            devices[-1].push_to_target_inputs(tuple(target))

        targets.append(target)

    return targets

def _verify_training(
        devices: List[Device],
        inputs: List[Tuple[Tensor, ...]], 
        targets: List[Tensor], 
        all_saved_parameters: List[Dict[str, torch.Tensor]], 
        checkpoint_q: queue.Queue,
        verify_cfg: VerifyConfig):
    """
    Verify training results vs. cpu eval
    """
    assert len(inputs) % verify_cfg.steps == 0, "Total inputs should be divisible by number of steps"
    step_size = len(inputs) // (verify_cfg.steps * verify_cfg.epochs)
    
    assert step_size % verify_cfg.accumulation_steps == 0, "Step size should be divisible by number of accumulation_steps"
    acc_step_size = step_size // verify_cfg.accumulation_steps

    from pybuda.op.eval import compare_tensor_to_golden # avoid circular import
    test_pass = True
    fail_on_first = "SHOW_ALL_FAILS" not in os.environ

    opt_lr = verify_cfg.optimizer['params']['learning_rate'] if verify_cfg.optimizer else None

    optimizer_index = 0
    try:
        for epoch in range(verify_cfg.epochs):
            for step in range(verify_cfg.steps):
                logger.debug("Running cpueval training for step {}", step)
                step_inputs = inputs[step*step_size+epoch*verify_cfg.steps*step_size : (step+1)*step_size+epoch*verify_cfg.steps*step_size]
                step_targets = targets[step*step_size+epoch*verify_cfg.steps*step_size : (step+1)*step_size+epoch*verify_cfg.steps*step_size]

                eval_res = cpueval_training(step_inputs, all_saved_parameters, step_targets, verify_cfg.sequential, verify_cfg.scale_loss, lr=opt_lr if epoch==0 else None)

                # Check input gradients
                if verify_cfg.enable_input_gradient_checking:
                    logger.debug("Verifying input gradients")
                    for iter in range(step_size):
                        res = verify_cfg._input_gradient_queue.get()
                        for i, (golden, result) in enumerate(zip(eval_res.devices[0].grad[iter].inputs, res)):

                            if isinstance(result, torch.Tensor):
                                result = Tensor.create_from_torch(result)

                            test_pass &= compare_tensor_to_golden(f"Iteration {iter} - Input {i} gradient", golden, result.value().type(golden.dtype), is_buda=True, verify_cfg=verify_cfg)
                            if fail_on_first:
                                assert test_pass, f"Data mismatch on iteration {iter} - Input {i} gradient"

                # Check parameter gradients
                if verify_cfg.enable_parameter_gradient_checking:
                    logger.debug("Verifying parameter gradients")
                    for acc_step in range(step_size // acc_step_size):
                        res = verify_cfg._parameter_gradient_queue.get()
                        for device_index, _ in enumerate(devices):
                            gradients = res[device_index]
                
                            for name in gradients:
                                calc = gradients[name].value()
                                # cpu eval gives us gradients on every step, but run_training gives us one on each
                                # accumulation step. So, we need to compare the last one in the step.
                                grad_index = (acc_step + 1) * acc_step_size - 1
                                golden = eval_res.devices[device_index].grad[grad_index].parameters[name]

                                warning_only = should_waive_gradient(name, verify_cfg)
                                test_pass &= compare_tensor_to_golden(f"Acc step {acc_step}, device {device_index} - Gradient for parameter {name}", golden, calc.type(golden.dtype), 
                                            is_buda=True, verify_cfg=verify_cfg,
                                            warning_only=warning_only)
                                if fail_on_first:
                                    assert test_pass, f"Data mismatch on acc step {acc_step}, device {device_index} - Gradient for parameter {name}"

                # Check weights (checkpoints and last)
                optimizer_index += 1
                if (step == verify_cfg.steps - 1) or (verify_cfg.checkpoint_interval > 0 and optimizer_index % verify_cfg.checkpoint_interval == 0):
                    checkpoint_name = "Final" if (step == verify_cfg.steps - 1) else f"Optimizer step {optimizer_index}"
                    logger.debug("Verifying parameter checkpoint {}", checkpoint_name)
                    checkpoint = checkpoint_q.get()
                    for device_index, _ in enumerate(devices):
                        param_checkpoint = checkpoint[device_index]
            
                        for name in param_checkpoint:
                            calc = param_checkpoint[name].value()
                            golden = eval_res.devices[device_index].final_parameters[name]
                            warning_only = should_waive_gradient(name, verify_cfg)
                            test_pass &= compare_tensor_to_golden(f"{checkpoint_name} parameter {name}, device {device_index}", golden, calc,
                                        is_buda=True, verify_cfg=verify_cfg,
                                        warning_only=warning_only)
                            if fail_on_first:
                                assert test_pass, f"Data mismatch on {checkpoint_name} parameter {name}, device {device_index}"

                        if step != verify_cfg.steps - 1:
                            # Not the last. Let's sync weights back to cpueval to continue to track
                            logger.debug("Syncing device param checkpoint to pytorch golden")
                            _update_parameters_from_checkpoint(all_saved_parameters, device_index, param_checkpoint)

            # epoch boundary scheduler step
            devices = get_devices()
            for d in devices:
                scheduler = d.get_pytorch_scheduler()
                if scheduler:
                    scheduler.step()

    finally:
        assert test_pass, f"Training data mismatch detected"


def get_framework_pipeline_outputs(inputs: Tuple[Tensor, ...], modules: List[Module]):
    i = 0
    while i < len(modules):
        pt_inputs = to_pt_tensors(inputs)

        inputs = modules[i].cpu_eval_forward(*pt_inputs)
        i += 1

    return inputs

def _verify_inference(
        inputs: List[Tuple[Tensor, ...]], 
        all_saved_parameters: List[Dict[str, torch.Tensor]], 
        result_q: queue.Queue, 
        verify_cfg: VerifyConfig,
        modules_copy: List[Module],
        inputs_copy: List[Tuple[Tensor, ...]],):
    """
    Verify inference vs cpueval
    """
    from pybuda.op.eval import compare_tensor_to_golden, calculate_pcc # avoid circular import
    test_pass = True
    fail_on_first = "SHOW_ALL_FAILS" not in os.environ
    contains_framework_module = any([not isinstance(x, PyBudaModule) for x in modules_copy])
    if verify_cfg.devmode == DeviceMode.RunOnly:
        # device._compile_output.initial_graph is not archived in TTI,
        # check if cpu-evaluated outputs are cached and loaded from TTI 
        devices = get_devices()
        tt_device = devices[0] if isinstance(devices[0], pybuda.TTDevice) else devices[1]
        assert tt_device._compiled_graph_state.cpueval_outputs, "cpueval-output are not loaded from TTI although device is set to RunOnly mode, exit"
        loaded_eval_outputs = list(list([tt_device._compiled_graph_state.cpueval_outputs]))

    try: 
        for iter, single_input in enumerate(inputs):
            result = result_q.get()
            if verify_cfg.devmode == DeviceMode.RunOnly:
                eval_outputs = loaded_eval_outputs[iter]
                if not isinstance(eval_outputs[0], list):
                    eval_outputs[0] = [eval_outputs[0]]
            else:
                eval_outputs = cpueval_inference([single_input], all_saved_parameters, verify_cfg.sequential)

            if verify_cfg.override_module_outptus is not None:
                contains_framework_module = False
                eval_outputs = verify_cfg.override_module_outptus

            if contains_framework_module:
                framework_outputs = get_framework_pipeline_outputs(inputs_copy[iter], modules_copy)
                assert len(framework_outputs) == len(result), "Number of framework outputs doesn't match number of Buda outputs"

            for i, (eval_out, result_out) in enumerate(zip(eval_outputs[0], result)):
                if contains_framework_module and verify_cfg.enabled:
                    test_pass &= compare_tensor_to_golden(f"Iteration {iter} - Framework Output {i}", framework_outputs[i], result_out.value(), is_buda=True, verify_cfg=verify_cfg)

                    if fail_on_first:
                        pcc_value = calculate_pcc(framework_outputs[i], result_out.value().to(framework_outputs[i].dtype))
                        assert test_pass, f"Data mismatch on iteration {iter} - Eval Output {i}. PCC got {pcc_value}, required={verify_cfg.pcc}"

                # Temporary workaround for microbatch dim being different between backend and frontend
                eval_out = eval_out.reshape(result_out.shape.get_pytorch_shape())

                if verify_cfg.enabled:
                    test_pass &= compare_tensor_to_golden(f"Iteration {iter} - Eval Output {i}", eval_out, result_out.value(),
                            is_buda=True, verify_cfg=verify_cfg)

                    if fail_on_first:
                        pcc_value = calculate_pcc(eval_out, result_out.value().to(eval_out.dtype))
                        assert test_pass, f"Data mismatch on iteration {iter} - Eval Output {i}. PCC got {pcc_value}, required={verify_cfg.pcc}"

    finally: 
        assert test_pass, f"Data mismatch detected"

def verify_module_pipeline(modules: List[Module], input_shapes: List[Tuple], verify_cfg: VerifyConfig, 
        input_params: List[Dict] = [], device_types = ["TTDevice"], params_centered_on_zero: bool = False, 
        scale_params: float = 1.0, inputs: Optional[List[Tuple[Tensor]]] = None,
        loss_module: Optional[Module] = None, uniform_inputs: bool = False, inputs_centered_on_zero: bool = False) -> List[Device]:
    """
    Test a pipeline of modules, with given verification requirements in verify_cfg, and given input shapes.

    This can do full inference and training testing on graph-level, golden, model, silicon, etc. 
    compared to pytorch equivalent model.

    input_params can be used to specify dtype and requires_grad for inputs

    Parameters
    ----------
    modules: List[Module]
        Pipeline of modules to test

    input_shapes: List[Tuple]
        List of input shapes to feed into the first module

    verify_cfg: VerifyConfig
        Verification parameters

    inputs_params: List[Dict]
        Optional parameters for each of the inputs - such as requires_grad, data_format, dev_data_format

    device_types: List[str] 
        List of device types (TTDevice, CPUDevice) to place modules on. Modules will be matched with list
        of devices 1-to-1, and if list of modules is longer than list of devices, all subsequent modules will
        be placed on the last device. If device type string is "-", the module will be placed on the previous
        device. This can't be the first device type.

    params_centered_on_zero: bool
        If set, parameters will be picked in -0.5, 0.5 range instead of 0, 1

    scale_params: float
        Divide parameters with this value

    inputs: List[Tuple[Tensor]]
        Optional list of model activations to run
        
    loss_module: Optional[Module]
        Optional loss module for training. If none is set (and training is enabled), pytorch L1Loss will be
        added, and a CPUDevice will be instantiated to calculate it.

    uniform_inputs: bool (default False)
        If set, random inputs will be uniformly distributed from 0 to 1 (i.e. default torch.rand)

    inputs_centered_on_zero: bool (default False)
        If set, random inputs will be cenetred around zero

    Returns
    -------
    List[Device, ...]
        List of generated devices with modules placed on them


    """
    
    if verify_cfg.enabled and verify_cfg.test_kind.is_training() and not verify_cfg.sequential:
        logger.warning("Concurrent training is NOT stable in the verify flow. Likely to run into errors during _verify_training")

    compiler_cfg = _get_global_compiler_config()
    
    if "PYBUDA_OVERRIDES_VETO" in os.environ:
        compiler_cfg = pybuda_override_veto(compiler_cfg)
        
        if "PYBUDA_OVERRIDES_VETO_CUSTOM_SETUP" in os.environ:
            return

    assert verify_cfg is not None, "VerifyConfig must be provided for verify_module flow"
    force_full = bool(int(os.environ.get("PYBUDA_FORCE_FULL_COMPILE_DEPTH", "0")))
    if force_full:
        compiler_cfg.compile_depth = CompileDepth.FULL
    run_backend = verify_cfg.devtype != BackendType.NoBackend and compiler_cfg.compile_depth == CompileDepth.FULL and verify_cfg.devmode != DeviceMode.CompileOnly

    # Figure out module placement so we can extract parameters
    module_to_device = {}
    dev_index = 0
    for i, m in enumerate(modules):
        module_to_device[i] = dev_index
        if i < len(device_types) - 1:
            if device_types[i] != "-":
                dev_index += 1

    dev_count = dev_index + 1

    # Generate inputs, if they are not already provided
    if inputs is None:
        inputs = _generate_random_inputs(input_shapes, input_params, verify_cfg, uniform_inputs, inputs_centered_on_zero)
    else:
        for i in range(verify_cfg.total_number_of_inputs()):
            inputs[i] = pybuda.tensor.to_buda_tensors(to_pt_tensors(inputs[i]))

    # Translate framework module if needed
    if compiler_cfg.compile_tvm_to_python:
        single_microbatch_inputs = remove_microbatch(inputs[0])
        import copy
        inputs_copy = copy.deepcopy(inputs)
        modules, device_types, module_to_device, dev_count, modules_copy = _translate_framework_modules(modules, device_types, module_to_device, single_microbatch_inputs, dev_count, verify_cfg, compiler_cfg)

        # Pybuda module will only accept flattened inputs
        for i in range(len(inputs)):
            flattened, _, _ = flatten_inputs(to_pt_tensors(inputs[i]))
            inputs[i] = pybuda.tensor.to_buda_tensors(flattened)

    # Extract parameters
    all_parameters: List[List[Parameter]] = [ [] for _ in range(dev_count) ]
    for i, m in enumerate(modules):
        all_parameters[module_to_device[i]].extend(m.get_parameters())

    # Create devices / place modules
    devices = _create_devices(modules, device_types, module_to_device, remove_microbatch(inputs[0]), verify_cfg, all_parameters, loss_module)

    all_saved_parameters = _update_and_randomize_params(modules, dev_count, module_to_device, all_parameters, params_centered_on_zero, scale_params, uniform_inputs)

    if verify_cfg.test_kind.is_training():
        all_saved_parameters.append( {} ) # one for cpu loss

    if not run_backend:
        # Compile and exit
        from ..compile import pybuda_compile
        compiler_cfg.enable_training = verify_cfg.test_kind.is_training()
        for epoch_break in verify_cfg.epoch_breaks:
            compiler_cfg.place_on_new_epoch(epoch_break)

        compiled_devices = []
        with torch.no_grad():
            single_microbatch_inputs = remove_microbatch(inputs[0])
            pipe_inputs = single_microbatch_inputs
 
        if verify_cfg.devmode == DeviceMode.CompileOnly:
            # Compile first
            from ..run import initialize_pipeline
            initialize_pipeline(
                training=verify_cfg.test_kind.is_training(),
                sample_inputs=to_buda_tensors(to_pt_tensors(pipe_inputs)),
                _sequential=True,
                _verify_cfg=verify_cfg,
                _device_mode=verify_cfg.devmode,
            )

            # Pre-calculate cpu-evauated outputs, append it to compiled_graph_state
            cpueval_outputs = []
            for single_input in inputs:
                eval_output = cpueval_inference([single_input], all_saved_parameters, verify_cfg.sequential)
                cpueval_outputs.append(eval_output[0][0])

        for device in devices:
            if isinstance(device, pybuda.cpudevice.CPUDevice):
                for module in device.modules:
                    pipe_inputs = module.forward(*to_pt_tensors(pipe_inputs))

                # Cast outputs from CPU-fallback to FP32
                pipe_inputs = [t.float() for t in pipe_inputs]
            else:
                if verify_cfg.devmode == DeviceMode.CompileOnly:
                    # Note: below does not re-compile the model (already compiled above), but just generate TTI
                    device.compile_to_image(
                        img_path=verify_cfg.tti_path,
                        training=verify_cfg.test_kind.is_training(),
                        sample_inputs= to_buda_tensors(to_pt_tensors(pipe_inputs)),
                        cpueval_outputs=cpueval_outputs,
                    )
                else:
                    device._compile_output = pybuda_compile(device, device.modules[0].get_name(), *to_buda_tensors(to_pt_tensors(pipe_inputs)),
                        compiler_cfg=compiler_cfg,
                        verify_cfg=verify_cfg)
                pipe_inputs = device._compile_output.outputs

        return devices
    # Generate the graph, to get output shapes (if training)
    # TODO for device pipeline
    if verify_cfg.test_kind.is_training():
        targets = _setup_training(devices, inputs[0], verify_cfg, loss_module is None)

        # Push inputs
        # TODO: concurrent pushing in a thread?
        for input in inputs:
            devices[0].push_to_inputs(*input)

        pybuda.set_configuration_options(enable_recompute=verify_cfg.test_kind.is_recompute())
        checkpoint_q = pybuda.run_training(
                epochs=verify_cfg.epochs, 
                steps=verify_cfg.steps, 
                accumulation_steps=verify_cfg.accumulation_steps, 
                microbatch_count=verify_cfg.microbatch_count, 
                _sequential=verify_cfg.sequential, 
                checkpoint_interval=verify_cfg.checkpoint_interval,
                _perf_trace=False,
                _verify_cfg=verify_cfg)

        assert not pybuda.error_raised(), "Error during training"

        # if parameters were updated as part of compile (by tvm), update them
        # FIXME: can't pick up new parameters here, since we already ran training and got them changed!
        #all_saved_parameters = update_and_randomize_params(all_parameters)
        #all_saved_parameters.append( {} ) # one for cpu loss

        _verify_training(devices, inputs, targets, all_saved_parameters, checkpoint_q, verify_cfg)

    else:

        # Push inputs
        # TODO: concurrent pushing in a thread?
        for input in inputs:
            devices[0].push_to_inputs(*input)

        result_q = pybuda.run_inference(_sequential=verify_cfg.sequential, _verify_cfg=verify_cfg, input_count=len(inputs))
        assert not pybuda.error_raised(), "Error during inference"
    
        # if parameters were updated as part of compile (by tvm), update them
        all_saved_parameters = _update_and_randomize_params(modules, dev_count, module_to_device, all_parameters, params_centered_on_zero, scale_params, uniform_inputs)

        _verify_inference(inputs, all_saved_parameters, result_q, verify_cfg, modules_copy, inputs_copy)

    return devices


def verify_module(module: Module, input_shapes: List[Tuple], verify_cfg: VerifyConfig, input_params: List[Dict] = [],
        device_type: str = "TTDevice", params_centered_on_zero: bool = False, scale_params: float = 1.0,
        inputs: Optional[List[Tuple[Tensor]]] = None,
        loss_module: Optional[Module] = None, uniform_inputs: bool = False, inputs_centered_on_zero: bool = False):
    """
    Test a module on one device, with given verification requirements in verify_cfg, and given input shapes.

    This can do full inference and training testing on graph-level, golden, model, silicon, etc. 
    compared to pytorch equivalent model.

    input_params can be used to specify dtype and requires_grad for inputs

    Parameters
    ----------
    module: Module
        Module to test

    input_shapes: List[Tuple]
        List of input shapes to feed into the first module

    verify_cfg: VerifyConfig
        Verification parameters

    inputs_params: List[Dict]
        Optional parameters for each of the inputs - such as requires_grad, data_format, dev_data_format

    device_type: str 
        Device type (TTDevice, CPUDevice) to place module on.

    params_centered_on_zero: bool
        If set, parameters will be picked in -0.5, 0.5 range instead of 0, 1

    scale_params: float
        Divide parameters with this value

    inputs: List[Tuple[Tensor]]
        Optional list of model activations to run
        
    loss_module: Optional[Module]
        Optional loss module for training. If none is set (and training is enabled), pytorch L1Loss will be
        added, and a CPUDevice will be instantiated to calculate it.

    uniform_inputs: bool (default False)
        If set, random inputs will be uniformly distributed from 0 to 1 (i.e. default torch.rand)

    inputs_centered_on_zero: bool (default False)
        If set, random inputs will be cenetred around zero
    
    Returns
    -------
    List[Device, ...]
        List of generated devices with modules placed on them


    """
    return verify_module_pipeline([module], input_shapes, verify_cfg, input_params, [device_type], params_centered_on_zero, scale_params, inputs, loss_module, uniform_inputs, inputs_centered_on_zero)

