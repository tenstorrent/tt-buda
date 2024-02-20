#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import inspect
from loguru import logger
from typing import List, Tuple, Union, Optional, Dict
import time
import threading
import queue
import socket

import pybuda
import torch

from benchmark.common import get_models, df_from_str, mf_from_str, trace_from_str
from pybuda._C.backend_api import BackendDevice, BackendType

# Resolve imports for functional models
import sys
sys.path.insert(1, 'pybuda')

# models
import benchmark.models.bert
import benchmark.models.deit
import benchmark.models.hrnet
import benchmark.models.inception_v4
import benchmark.models.mobilenet_v1
import benchmark.models.mobilenet_v2
import benchmark.models.mobilenet_v2_timm
import benchmark.models.mobilenet_v3_timm
import benchmark.models.openpose_body
import benchmark.models.openpose_hand
import benchmark.models.other
import benchmark.models.resnet
import benchmark.models.t5
import benchmark.models.unet
import benchmark.models.unit
import benchmark.models.vit
import benchmark.models.vovnet_v1
import benchmark.models.vovnet_v2
import benchmark.models.whisper
import benchmark.models.yolo_v3
import benchmark.models.yolo_v5

import benchmark.models.custom.custom_resnet_highres
import benchmark.models.custom.custom_vit_highres


def single_thread_generative_model_run(args, first_device, last_device, inputs, targets, output_q, num_tokens_to_generate, first_current_index, pad_token_id, write_index):
    print("Executing in single-threaded generative model mode")

    if args.training:
        assert False, "Training currently not supported in single-threaded mode"

    from pybuda.pybudaglobal import TILE_DIM
    # input_ids, encoder_attention_mask, input_length, decoder_inpu_ids, decoder_attention_mask,  
    # first_current_index, tokenizer.pad_token_id, 
    input_ids = inputs[0]
    encoder_attention_mask = inputs[1]
    decoder_input_ids = inputs[2]
    decoder_attention_mask = inputs[3]
    is_text_inputs = (first_current_index is not None)    

    print_start_info()

    start_time = time.time()

    first_device.set_active_subgraph(0)
    if is_text_inputs:
        first_device.push_to_inputs((input_ids, encoder_attention_mask)) 
    else:
        first_device.push_to_inputs((input_ids,))
    pybuda.run_forward()
    ans = output_q.get()
    encoder_last_hidden_state = ans[0].value().detach()
    generated_tokens = []

    current_token_index = 0 
    for _ in range(num_tokens_to_generate):  
        if current_token_index == 0:
            first_device.set_active_subgraph(1)
            generate_inputs = (decoder_input_ids, decoder_attention_mask, encoder_last_hidden_state, encoder_attention_mask)
            first_device.push_to_inputs(generate_inputs)
            pybuda.run_generate(input_count=args.loop_count, write_index=write_index)
            ans = output_q.get()
        else:
            if current_token_index == 1:
                start_time1 = time.time()
            first_device.set_active_subgraph(2)
            generate_inputs = (decoder_input_ids, decoder_attention_mask, encoder_attention_mask)
            first_device.push_to_inputs(generate_inputs)
            pybuda.run_generate(input_count=args.loop_count, write_index=write_index)
            ans = output_q.get()

        if is_text_inputs or current_token_index < 2:
            current_token_index += 1

        if is_text_inputs:        
            lm_head_out = ans[0].value().detach()
            next_token = torch.argmax(lm_head_out[0, (current_token_index-1) % TILE_DIM])
            generated_tokens.append(next_token)
 
            if current_token_index % TILE_DIM == 0:
                past_cache_pages = current_token_index // TILE_DIM
                # after one page of past cache, we have to rotate. 
                first_device.set_active_subgraph(3)
                pybuda.run_generate(input_count=0, write_index=0)

                pages_current = 1
                decoder_attention_mask[0, -(past_cache_pages + pages_current) * TILE_DIM:] = 1
                decoder_attention_mask[0, first_current_index:] = 0
                decoder_input_ids[0, :] = pad_token_id

            decoder_input_ids[0, current_token_index % TILE_DIM] = next_token
            decoder_attention_mask[0, first_current_index + (current_token_index % TILE_DIM)] = 1

    end_time = time.time()

    return start_time, start_time1, end_time

def single_thread_run(args, first_device, last_device, inputs, targets, output_q, num_tokens_to_generate):
    print("Executing in single-threaded mode")

    if args.training:
        assert False, "Training currently not supported in single-threaded mode"

    print_start_info()

    start_time = time.time()

    if num_tokens_to_generate:
        for _ in range(num_tokens_to_generate):
            first_device.push_to_inputs(inputs)
            pybuda.run_generate(input_count=args.loop_count, write_index=0)
            output_q.get()
    else:
        first_device.push_to_inputs(inputs)
        pybuda.run_forward(input_count=args.loop_count)
        output_q.get()

    end_time = time.time()

    return start_time, end_time


def multi_thread_run(args, first_device, last_device, inputs, targets, output_q, num_tokens_to_generate):
    print("Executing in multi-threaded mode")
    
    #import pdb; pdb.set_trace()

    #
    # Prepare a thread pushing inputs
    #
    def push_inputs_thread():
        loop_count = num_tokens_to_generate if num_tokens_to_generate else args.loop_count
        for _ in range(loop_count):
            if pybuda.error_raised():
                print(" * Aborting input thread due to error")
                return
            first_device.push_to_inputs(inputs)
            if args.training:
                last_device.push_to_target_inputs(targets)

    #
    # Start a thread popping outputs
    #
    def pop_outputs_thread(output_q):
        loop_count = num_tokens_to_generate if num_tokens_to_generate else args.loop_count
        for _ in range(loop_count):
            while True:
                try:
                    output_q.get(timeout=1)
                    break # got data, break out of forever loop
                except queue.Empty as _:
                    if pybuda.error_raised():
                        print(" * Aborting output thread due to error")
                        return

    #
    # Define input and output threads
    #
    input_thread = threading.Thread(target=push_inputs_thread)
    output = output_q if not args.training else pybuda.get_loss_queue()
    output_thread = threading.Thread(target=pop_outputs_thread, args=(output, ))
    output_thread.start()

    #
    # Sync - Make sure all process setup, compile, etc. is done
    #
    pybuda.sync()

    #
    # Run
    #
    input_thread.start()
    time.sleep(2) # Let the input thread start up and transfer initial data, reaching something like "steady state"

    print_start_info()

    start_time = time.time()
    
    if args.training:
        print(f'loop_count: {args.loop_count} gives {args.loop_count//args.microbatch_count} training batches of [fwd,bwd]', flush=True)
        # TODO: use microbatch count / accumulation steps, depending on number of devices in pipeline
        for _ in range(args.loop_count//args.microbatch_count):
            pybuda.run_forward(input_count=args.microbatch_count)
            pybuda.run_backward(input_count=args.microbatch_count)
    else:
        if num_tokens_to_generate:
            for _ in range(num_tokens_to_generate):
                pybuda.run_generate(input_count=args.loop_count, write_index=0)
        else:
            pybuda.run_forward(input_count=args.loop_count)

    input_thread.join()
    output_thread.join()
    if args.training:
        pybuda.sync() # wait for the last backward to finish

    end_time = time.time()
    
    return start_time, end_time


def print_start_info():
    print("*****************************************************")
    print(" Starting benchmark at ", time.asctime( time.localtime(time.time()) ))
    print("*****************************************************")


def run(
        args,
        duts: Dict[str, Union[pybuda.PyTorchModule, pybuda.PyBudaModule]],
        inputs: Tuple[torch.Tensor, ...],
        targets: Tuple[torch.Tensor, ...],
        other: Dict[str, object]):

    # Emulate runs on harvested machines
    from pybuda._C.backend_api import BackendDevice
    available_devices = pybuda.detect_available_devices()
    if available_devices and not args.galaxy:
        if available_devices[0] == BackendDevice.Wormhole_B0:
            os.environ["PYBUDA_FORCE_EMULATE_HARVESTED"] = "1"

    # Set default configuration type
    pybuda.config.set_configuration_options(default_df_override=df_from_str(args.dataformat), backend_runtime_params_path=args.runtime_params_yaml, device_config=args.device_config)

    # Override push timeout on slow runs
    os.environ["TT_BACKEND_PUSH_TIMEOUT"] = "500"

    if args.training:
        assert len(targets) > 0, "Targets must be supplied for training"

    first_device = None
    last_device = None
    if "cpu-pre" in duts:
        assert isinstance(duts["cpu-pre"], pybuda.PyTorchModule)
        cpu_pre = pybuda.CPUDevice("cpu-pre", module=duts["cpu-pre"])
        first_device = cpu_pre

    assert args.arch
    arch = BackendDevice.from_string(args.arch) 
    devtype = BackendType.from_string(args.device.title()) if args.device else None

    assert "tt" in duts
    if args.save_tti:
        tt = pybuda.TTDevice("tt0", module=duts["tt"], fp32_fallback=df_from_str(args.dataformat), num_chips=args.chips, arch=arch, devtype=devtype)
    elif args.load_tti:
        img = pybuda.TTDeviceImage.load_from_disk(args.load_tti)
        img.info()
        tt = pybuda.TTDevice.load_image(img=img)
    elif args.galaxy:
        tt = pybuda.TTDevice("tt0", module=duts["tt"], arch=arch, devtype=devtype, fp32_fallback=df_from_str(args.dataformat), chip_ids=[0, 11, 10, 9, 8, 7, 19, 20, 21, 22, 23, 24, 6, 5, 14, 13, 12, 16, 15, 3, 4, 26, 25, 32, 31, 30, 29, 28, 27, 1, 2, 18, 17])
    else:
        tt = pybuda.TTDevice("tt0", module=duts["tt"], arch=arch, devtype=devtype, fp32_fallback=df_from_str(args.dataformat), num_chips=args.chips)

    if first_device is None:
        first_device = tt
    last_device = tt

    cpu_post = None
    if "cpu-post" in duts:
        assert isinstance(duts["cpu-post"], pybuda.PyTorchModule)
        cpu_post = pybuda.CPUDevice("cpu-post", module=duts["cpu-post"])
        last_device = cpu_post

    if "cpu-loss" in duts:
        assert isinstance(duts["cpu-loss"], pybuda.PyTorchModule)
        if cpu_post is None: # no cpu-post module
            identity = pybuda.PyTorchModule("identity0", torch.nn.Identity())
            cpu_post = pybuda.CPUDevice("cpu-post", module=identity)

        cpu_post.place_loss_module(duts["cpu-loss"])
        last_device = cpu_post

    enable_auto_fusing = "PYBUDA_DISABLE_FUSE_OPS" not in os.environ
    if args.perf_analysis:
        if "PYBUDA_OP_PERF" not in os.environ:
            os.environ["PYBUDA_OP_PERF"] = "1"
        if "TT_BACKEND_PERF_ANALYZER" not in os.environ:
            os.environ["TT_BACKEND_PERF_ANALYZER"] = "1"

    pybuda.set_configuration_options(
            math_fidelity=mf_from_str(args.math_fidelity),
            performance_trace=trace_from_str(args.trace),
            backend_opt_level=args.backend_opt_level,
            enable_auto_fusing=enable_auto_fusing,
            enable_recompute=args.recompute,
            enable_auto_transposing_placement=args.auto_transpose)

    # Define compile inputs and max length of generative models (applicable for Whisper, T5 and 
    # similar generative models with past-cache mechanics)
    compile_inputs = inputs
    num_tokens_to_generate = None
    first_current_index = None
    pad_token_id = None
    write_index = 0
    if bool(other):
        if "compile_inputs" in other:
            compile_inputs = other["compile_inputs"]
        if "max_length" in other:
            num_tokens_to_generate = other["max_length"]
        if "first_current_index" in other:  # TODO
            first_current_index = other["first_current_index"]
        if "pad_token_id" in other:  # TODO
            pad_token_id = other["pad_token_id"]
        if "write_index" in other: # TODO
            write_index = other["write_index"]
    if num_tokens_to_generate:
        args.loop_count = 1

    if args.chips == 0:
        args.chips = len(pybuda.detect_available_devices())
    if args.chips == 0:
        raise RuntimeError("No tenstorrent devices found.")

    if args.loop_count == 0:
        args.loop_count = 1 if args.perf_analysis else 15 * args.microbatch_count * (args.chips + len(duts) - 1)
    args.microbatch = inputs[0].shape[0]

    assert args.loop_count % args.microbatch_count == 0, "loop_count must be a multiple of microbatch_count"
    print(f'Using loop_count: {args.loop_count} microbatch_count: {args.microbatch_count} microbatch: {args.microbatch} chips: {args.chips}')

    # TODO: For silicon device runs, it seems that the `tt` from user-side is not
    # the one being used with api calls like pybuda.run_forward(..). We'll fetch
    # the arch from the first device-type available
    device_list = pybuda.detect_available_devices()
    arch = device_list[0] if len(device_list) > 0 else tt.arch

    #
    # Compile, and start
    #
    if args.save_tti:
        tt.compile_to_image(img_path=args.save_tti, training=args.training, sample_inputs=compile_inputs, sample_targets=targets)
        exit(0)

    output_q = pybuda.initialize_pipeline(training=args.training, sample_inputs=compile_inputs, microbatch_count=args.microbatch_count, _verify_cfg=pybuda.VerifyConfig.disabled(), sample_targets=targets)

    if args.single_thread:
        if args.generative:
            start_time, start_time1, end_time = single_thread_generative_model_run(args, first_device, last_device, inputs, targets, output_q, num_tokens_to_generate, first_current_index, pad_token_id, write_index)
        else:
            start_time, end_time = single_thread_run(args, first_device, last_device, inputs, targets, output_q, num_tokens_to_generate)
    else:
        start_time, end_time = multi_thread_run(args, first_device, last_device, inputs, targets, output_q, num_tokens_to_generate)

    if pybuda.error_raised():
        print("*********************************")
        print(" Error raised, aborting benchmark")
        print("*********************************")
        return {
            "total_time": 0,
            "total_samples": 0,
            "samples_per_sec": 0,
            "args": vars(args),
            "arch": str(arch),
            "machine_name": socket.gethostname(),
            "error": "Error raised, aborting benchmark"
        }


    print("*****************************************************")
    print(" Ending benchmark at ", time.asctime( time.localtime(end_time) ))

    total_time = end_time - start_time
    if num_tokens_to_generate:
        total_samples = args.loop_count * args.microbatch * num_tokens_to_generate
        print(f" Total time for {num_tokens_to_generate} tokens: {total_time:.4f}")
        print(f" Tokens/s: {(num_tokens_to_generate / total_time):.1f}")
    else:
        total_samples = args.loop_count * args.microbatch
        print(f" Total time for {total_samples} inputs: {total_time:.4f}")
        print(f" Samples/s: {(total_samples / total_time):.1f}")
    print("*****************************************************")

    return {
        "total_time": total_time,
        "total_samples": total_samples,
        "samples_per_sec": total_samples / total_time,
        "args": vars(args),
        "arch": str(arch),
        "machine_name": socket.gethostname()
    }

if __name__ == "__main__":
    
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark a model on TT hardware')
    parser.add_argument('-m', '--model', help='Model to benchmark (i.e. bert)')
    # TODO parser.add_argument('-b', '--block', help='Block within model to benchmark, if supported. (i.e. self-attention)')
    parser.add_argument('-c',   '--config', help='Model configuration to benchmark (i.e. tiny, base, large)')
    parser.add_argument('-t',   '--training', action='store_true', help='Benchmark training')
    parser.add_argument('-df',  '--dataformat', choices=['Fp32', 'Fp16', 'Fp16_b', 'Bfp8', 'Bfp8_b', 'Bfp4', 'Bfp4_b'], default='Bfp8_b', help='Set data format')
    parser.add_argument('-mf',  '--math_fidelity', choices=['LoFi', 'HiFi2', 'HiFi3', 'HiFi4'], default='LoFi', help='Set math fidelity')
    parser.add_argument('-opt', '--backend_opt_level', choices=[0, 1, 2, 3, 4], default=4, type=int, help='Set backend optimization level')
    parser.add_argument(        '--loop_count', default=32, type=int, help='Set the number of times to loop through the model. By default, it will be 5x the number of chips.')
    parser.add_argument(        '--microbatch_count', default=1, type=int, help='Set the number of times to loop within each program.')
    parser.add_argument('-mb',  '--microbatch', default=64, type=int, help='The microbatch size to run the benchmark on. The model should set its own reasonable default if no microbatch is forced here.')
    parser.add_argument(        '--chips', default=1, type=int, help='Number of chips to run benchmark on. 0 to run on all available.')
    parser.add_argument(        '--recompute', action='store_true', help='Enable recompute in training')
    parser.add_argument(        '--layers', default=0, type=int, help='Number of layers to run on models where this is applicable (i.e. nlp encoders/decoders)')
    parser.add_argument(        '--trace', default="none", choices=["none", "light", "verbose"], help='Performance trace to be generated during the run.')
    parser.add_argument('-l',   '--list', action='store_true', help='List available models and configurations')
    parser.add_argument('-e',   '--env', default="", help='List of environment variable settings, i.e. "PYBUDA_OPT1=1 PYBUDA_OP2=1" to run with.')
    parser.add_argument('-o',   '--output', help='Output json file to write results to, optionally. If file already exists, results will be appended.')
    parser.add_argument(        '--disable_output', default=0, type=int, choices=[0, 1], help='Disables the generation of the output json file')
    parser.add_argument(        '--load_tti', default="", type=str, help='Skip compile and load from TTI-archive configured for silicon (specify path to TTI).')
    parser.add_argument(        '--save_tti', default="", type=str, help='Save compilation for TTDevice into a TTI-archive configured for silicon to file and exit program. (speciy path to save to).')
    parser.add_argument(        '--arch', choices=['grayskull', 'wormhole', 'wormhole_b0'], default=None, help='Set arch for offline TTI compilation.')
    parser.add_argument(        '--device', choices=['silicon', 'golden', 'model'], default=None, help='Set device.')
    parser.add_argument(        '--runtime_params_yaml', default=None, help='Set runtime params yaml for offline compile of WH devices.')
    parser.add_argument(        '--device-config', choices=['galaxy', 'wh_n150', 'wh_n300', 'gs_e150', 'gs_e300'], default=None, type=str, help='Runtime params yaml for offline compile of WH devices would be configured based on that.')
    parser.add_argument(        '--auto_transpose', action='store_true', help='Enable auto-transpose on placement')
    parser.add_argument('-bp',  '--balancer-policy', choices=['default', 'CNN', 'Ribbon', 'NLP'], default='default', help='Set balancer policy.')
    parser.add_argument(        '--perf_analysis', action='store_true', help='Enable backend perf analyzer and op estimates in compiler')
    parser.add_argument(        '--single-thread', action='store_true', help='Run benchmark models in single thread')
    parser.add_argument(        '--generative', action='store_true', help='Run benchmark models in single thread with targeting generative model')
    parser.add_argument(        '--galaxy', action='store_true', help='Run benchmark models on a neb+galaxy backend')

    args = parser.parse_args()

    models = get_models()

    # TODO(jchu): temporary workaround since code-generated python modules expect cwd in sys path
    import sys
    sys.path.append('.')

    if args.list:
        print("\nAvailable models:\n")
        for m in models:
            print(" - ", m.ljust(30), "configs: ", models[m]["configs"])
        print("\n")
        exit(0)

    if not args.model:
        print("\nModel must be specified.\n\n")
        print(parser.print_help())
        exit(1)

    if not args.model in models:
        print("Invalid model name. Available models: ")
        print(list(models.keys()))
        exit(1)

    if args.config:
        if not args.config in models[args.model]["configs"]:
            print("Invalid configuration for model ", args.model, ". Available configurations:")
            print(models[args.model]["configs"])
            exit(1)
    elif len(models[args.model]["configs"]) > 1:
        print("Model ", args.model, " has more than one configuration, you have to choose one:")
        print(models[args.model]["configs"])
        exit(1)
    
    if args.load_tti and args.save_tti:
        print("Specify only one of `--load_tti` or `--save-tti`")
        exit(1)

    if args.load_tti:
        print(f"Loading TTDevice from TTI specified at: {args.load_tti}")
    if args.save_tti:
        print(f"Saving TTDevice Image to: {args.save_tti}")
    
    pybuda.pybuda_reset()

    if args.env != "":
        envs = args.env.split(" ")
        for e in envs:
            if "=" not in e:
                name = e
                value = "1"
            else:
                name, value = e.split("=")
            os.environ[name] = value

    # Set bert multichip placement policy
    if args.model == "bert" and args.chips > 1:
        os.environ["PYBUDA_MULTICHIP_BERT"] = str(args.chips)

    kwargs = {"training": args.training, "microbatch": args.microbatch, "data_type": args.dataformat}

    device_list = pybuda.detect_available_devices()
    if device_list:
        args.arch = device_list[0].name.lower()
    elif not args.arch:
        raise RuntimeError("On a machine without a silicon device, --arch must be specified to save a TTI file.")

    kwargs["arch"] = args.arch.lower()
    kwargs["devtype"] = args.device.lower() if args.device else None

    func = models[args.model]["func"]
    available_parameters = inspect.signature(func).parameters
    for p in available_parameters:
        if p == "config":
            if args.config is None:
                assert len(models[args.model]["configs"]) == 1
                kwargs["config"] = models[args.model]["configs"][0]
            else:
                kwargs["config"] = args.config
        elif p == "force_num_layers":
            kwargs["force_num_layers"] = args.layers
        elif p == "arch":
            if args.arch == "":
                kwargs["arch"] = "grayskull"  # default
            else:
                kwargs["arch"] = args.arch

    # Balancer policy can be set thru benchmark script (as argument) and within the test itself
    # If the benchmark script is set to "default", the test will be able to override it, otherwise the script's balancer policy will have priority
    if args.balancer_policy != "default":
        compiler_cfg = pybuda.config._get_global_compiler_config()
        compiler_cfg.balancer_policy = args.balancer_policy

    model_config = models[args.model]["func"](**kwargs)

    if model_config is None:
        print("The model configuration is empty. ")
        exit(0)
        
    duts, inputs, targets, other = model_config
    implied_microbatch = inputs[0].shape[0]
    if (implied_microbatch != args.microbatch):
        logger.warning(f"Model configuration implies microbatch size of {implied_microbatch}, but command line specifies {args.microbatch}. Overriding microbatch size to {implied_microbatch}.")
    try:
        result = run(args, duts, inputs, targets, other)

    except RuntimeError as e:
        result = {
            "args": vars(args),
            "samples_per_sec": 0.0,
            "error": str(e),
            "machine_name": socket.gethostname()
        }
        print("Error encountered while running benchmark: ", e)

    import subprocess
    short_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    date = subprocess.check_output(['git', 'show', '-s', '--format=%cd', "--date=format:%y-%m-%d", 'HEAD']).decode('ascii').strip()
    result["hash"] = short_hash
    result["date"] = date
    print(result)

    if args.output and not args.disable_output:
        out_file = args.output
        if not out_file.endswith(".json"):
            out_file += ".json"
        import json, os

        all_results = []
        if os.path.exists(out_file):
            try:
                with open(out_file, "r") as f:
                    print("Reading in ", out_file, " with previous data")
                    all_results = json.loads(f.read())
            except Exception as e:
                print("Failed to load previous results, Will not overwrite, but create a different output file.")
                out_file = "post_error_" + out_file

        all_results.append(result)
        with open(out_file, "w") as f:
            f.write(json.dumps(all_results))

        print("Written out ", out_file, " with summary")
