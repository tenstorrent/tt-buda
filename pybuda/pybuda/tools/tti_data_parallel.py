#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pybuda
import pybuda.backend
import torch
import time
import os
import queue
import threading
import shutil
from typing import Iterable, Optional, Dict, List, Tuple, Union, Any
import pybuda
from loguru import logger
from dataclasses import dataclass, field
from enum import Enum

OUTPUT_TTI_NAME = "parallel_tti_run.tti"
class RunMode(Enum):
    FORWARD = 1
    GENERATIVE = 2
    
@dataclass
class ForwardRunInputs:
    inputs: Iterable[torch.Tensor] = None
        
    @staticmethod
    def get_inputs_per_card(all_inputs: "ForwardRunInputs", num_cards: int) -> List["ForwardRunInputs"]:
        run_inputs_per_card = split_tensor_batch(all_inputs.inputs, num_cards)
        inputs_per_card: List[ForwardRunInputs] = []
        for card_index in range(num_cards):
            inputs_per_card.append(
                ForwardRunInputs(
                    inputs=run_inputs_per_card[card_index]
                )
            )
        return inputs_per_card
        
@dataclass
class GenerativeRunInputs:
    compile_inputs: Iterable[torch.Tensor] = None
    run_inputs: Iterable[torch.Tensor] = None
    num_tokens_to_generate: int = None
    write_index: int = 0
    first_current_index: Optional[int] = None
    pad_token_id: Optional[int] = None
    
    def __post_init__(self):
        assert self.compile_inputs
        assert self.run_inputs
        assert self.num_tokens_to_generate
            
    
    @staticmethod
    def get_inputs_per_card(all_inputs: "GenerativeRunInputs", num_cards: int) -> List["GenerativeRunInputs"]:
        # autograd does not support crossing process boundaries, this is an issue for whisper
        # detach all input tensors from compute graph to bypass this issue
        compile_inputs_per_card = detach_all_tensors(split_tensor_batch(all_inputs.compile_inputs, num_cards))
        run_inputs_per_card = detach_all_tensors(split_tensor_batch(all_inputs.run_inputs, num_cards))
        
        inputs_per_card: List[GenerativeRunInputs] = []
        for card_index in range(num_cards):
            inputs_per_card.append(
                GenerativeRunInputs(
                    compile_inputs=compile_inputs_per_card[card_index],
                    run_inputs=run_inputs_per_card[card_index],
                    num_tokens_to_generate=all_inputs.num_tokens_to_generate,
                    write_index=all_inputs.write_index,
                    first_current_index=all_inputs.first_current_index,
                    pad_token_id=all_inputs.pad_token_id,
                )
            )
            
        return inputs_per_card


@dataclass
class ForwardRunConfig:
    chip_ids: List[int] = field(default_factory=list)
    inputs: ForwardRunInputs = None
    tti_path: str = ""
    loop_count: int = 0
    
    def __post_init__(self):
        assert self.chip_ids
        assert self.inputs
        assert self.tti_path
        assert self.loop_count
    
    def inputs_for_compile(self):
        return self.inputs.inputs
    
    def inputs_for_run(self):
        return self.inputs.inputs
    
    
@dataclass
class GenerativeRunConfig:
    chip_ids: List[int] = field(default_factory=list)
    inputs: GenerativeRunInputs = None
    tti_path: str = ""
    
    def __post_init__(self):
        assert self.chip_ids
        assert self.inputs
        assert self.tti_path
    
    def inputs_for_compile(self):
        return self.inputs.compile_inputs
    
    def inputs_for_run(self):
        return self.inputs.run_inputs
    
@dataclass
class RunEvents:
    # Set by the child process when its done running
    done_event: torch.multiprocessing.Event = None
    
    # Set by the main process when the process can be terminated
    kill_event: torch.multiprocessing.Event = None
    
    # Set by the child process when the process has started
    # In a pytest environment, pre-process-start, we run various setup functions
    # including create-ethernet-map
    process_start_event: torch.multiprocessing.Event = None
    
    # Set by the first child process after it has finished loading (unzipping) the tti
    tti_first_load_event: torch.multiprocessing.Event = None
    
    # Optional: Set by the main process to synchronize the start of the run across processes
    run_event: Optional[torch.multiprocessing.Event] = None
    
    # Optional: Set by the child process after it has finished initializing pipeline
    initialize_completed_event: Optional[torch.multiprocessing.Event] = None
    
    def __post_init__(self):
        assert self.done_event
        assert self.kill_event
        assert self.process_start_event

    def wait_for_initialize_complete(self):
        if self.initialize_completed_event:
            self.initialize_completed_event.wait()
    
    def wait_for_run_complete(self):
        self.done_event.wait()
        
@dataclass
class RunOutputs:
    # Contains the outputs of the run
    output_tensors_path: Optional[str] = ""
    
    # Contains the start and end time of the run in tuple format (start_time, end_time)
    perf_q: Optional[torch.multiprocessing.Queue] = None
    
    def get_output_tensors(self):
        if self.output_tensors_path:
            return torch.load(self.output_tensors_path)
        return None
        
    def get_start_end_time(self):
        if self.perf_q is not None:
            return self.perf_q.get()
        return None

@dataclass
class RunResult:
    # Merged outputs from all devices
    # For forward, outer dim is loop count, inner dim is output tensors per loop
    # For generative, outer dim is device count, inner dim is generated tokens per device
    outputs: List[List[torch.Tensor]] = None
    
    # Device id to start time
    per_card_start_time: Dict[int, float] = None
    
    # Device id to end time
    per_card_end_time: Dict[int, float] = None
    
    def __post_init__(self):
        assert self.per_card_start_time.keys() == self.per_card_end_time.keys()
        
    def get_per_card_runtime(self):
        per_card_runtime = {}
        for device_id in self.per_card_start_time.keys():
            per_card_runtime[device_id] = self.per_card_end_time[device_id] - self.per_card_start_time[device_id]
            
        return per_card_runtime
    
    def get_earliest_start(self):
        return min(self.per_card_start_time.values())
    
    def get_latest_end(self):
        return max(self.per_card_end_time.values())
    
    def get_total_runtime(self):
        return self.get_latest_end() - self.get_earliest_start()

# Namespace for forward run APIs
class ForwardRun:
    # Runs the tti on a single device and gathers outputs
    @staticmethod
    def _multi_thread_forward_run(config: ForwardRunConfig, events: RunEvents, output_wrapper: RunOutputs):
        # Create ethernet map runs at the beginning of every process in a pytest environment
        # Create ethernet map is not process safe
        events.process_start_event.set()
        
        tt0 = pybuda.TTDevice.load_image(img_path=config.tti_path, device_id_overrides=config.chip_ids)
        
        # For the first device process, set the event to notify the main process the tti has been unzipped
        # So that the main process can launch other processes
        # Prevents processes from racing to unzip the tti
        if events.tti_first_load_event:
            events.tti_first_load_event.set()
            
        device_output_q = pybuda.initialize_pipeline(training=False, sample_inputs=config.inputs_for_compile())
        all_outputs = []
        
        def push_inputs_thread(tt_device: pybuda.TTDevice, inputs, loop_count: int):
            for _ in range(loop_count):
                if pybuda.error_raised():
                    print(" * Aborting input thread due to error")
                    return
                tt_device.push_to_inputs(inputs)
                
        def pop_outputs_thread(output_q, all_outputs, loop_count: int):
            for _ in range(loop_count):
                while True:
                    try:
                        outputs = output_q.get(timeout=1)
                        all_outputs.append(outputs)
                        break
                    except queue.Empty as _:
                        if pybuda.error_raised():
                            print(" * Aborting output thread due to error")
                            return
                        
        if events.initialize_completed_event:
            events.initialize_completed_event.set()

        all_outputs = []
        output_thread = threading.Thread(target=pop_outputs_thread, args=(device_output_q, all_outputs, config.loop_count))
        input_thread = threading.Thread(target=push_inputs_thread, args=(tt0, config.inputs_for_run(), config.loop_count))
        
        # mimicking pybuda/test/benchmark/benchmark.py
        # Wait for this event to be set and start running
        if events.run_event:
            events.run_event.wait()
            
        output_thread.start()
        input_thread.start()
        time.sleep(2)  # Let the input thread start up and transfer initial data, reaching something like "steady state"
        
        start = time.time()
        
        pybuda.run_forward(input_count=config.loop_count)
        
        input_thread.join()
        output_thread.join()
        
        end = time.time()
        
        if output_wrapper.output_tensors_path:
            all_outputs_torch = []
            for outputs in all_outputs:
                all_outputs_torch.append([output.to_pytorch() for output in outputs])
                
            logger.info(f"Saving outputs temporarily to {output_wrapper.output_tensors_path} for main process to pick up, this may take a while for large outputs")
            torch.save(all_outputs_torch, output_wrapper.output_tensors_path)
        
        if output_wrapper.perf_q:
            output_wrapper.perf_q.put((start, end))
        
        pybuda.shutdown()
        
        # Reading tensors from queues requires this process to be alive
        # Set done_event to notify the main process that outputs can be read
        # Wait for kill_event to terminate the process
        events.done_event.set()
        events.kill_event.wait()

    @staticmethod
    def _create_run_result(
         # List of outputs per card, per loop
        outputs_per_card: List[List[List[torch.tensor]]], 
        per_card_runtime: Dict[int, Tuple[float, float]]
    ):
        # Merge the outputs from all devices
        num_cards = len(outputs_per_card)
        num_loops = len(outputs_per_card[0])
        
        # when running with n300 data parallel, the outputs are further split into two
        # for example if the output of the module should be [tensor(256, 1000)], it will be split into [tensor(128, 1000), tensor(128, 1000)]
        # thus, we need to merge these outputs back into [tensor(256, 1000)]
        if os.environ.get("PYBUDA_N300_DATA_PARALLEL", "0") == "1":
            for card_index in range(num_cards):
                for loop_idx in range(num_loops):
                    total_num_output_tensors = len(outputs_per_card[card_index][loop_idx])
                    assert total_num_output_tensors % 2 == 0, "Number of output tensors in n300 data parallel should be even"
                    merged_outputs = []
                    # Step over the outputs, merge every adjacent pair
                    for tensor_idx in range(0, total_num_output_tensors, 2):
                        merged_output = torch.cat([outputs_per_card[card_index][loop_idx][tensor_idx], outputs_per_card[card_index][loop_idx][tensor_idx + 1]], dim=0)
                        merged_outputs.append(merged_output)
                    outputs_per_card[card_index][loop_idx] = merged_outputs
                    
        single_loop_output_len = len(outputs_per_card[0][0])
        
        assert len(per_card_runtime) == num_cards
        
        merged_outputs_per_loop = []
        for loop_idx in range(num_loops):
            merged_outputs_this_loop = []
            for output_idx in range(single_loop_output_len):
                output_per_card = [outputs_per_card[card_index][loop_idx][output_idx] for card_index in range(num_cards)]
                merged_outputs_this_loop.append(torch.cat(output_per_card, dim=0))
            merged_outputs_per_loop.append(merged_outputs_this_loop)
                
        per_card_start_time = {device_id: start_end[0] for device_id, start_end in per_card_runtime.items()}
        per_card_end_time = {device_id: start_end[1] for device_id, start_end in per_card_runtime.items()}
        
        return RunResult(merged_outputs_per_loop, per_card_start_time, per_card_end_time)

# Namespace for generative run APIs
class GenerativeRun:
    @staticmethod
    def _single_thread_generative_model_run(config: GenerativeRunConfig, events: RunEvents, output_wrapper: RunOutputs):
        # Create ethernet map runs at the beginning of every process in a pytest environment
        # Create ethernet map is not process safe
        events.process_start_event.set()
        
        from pybuda.pybudaglobal import TILE_DIM
        compile_inputs = config.inputs_for_compile()
        run_inputs = config.inputs_for_run()
        
        first_device = pybuda.TTDevice.load_image(img_path=config.tti_path, device_id_overrides=config.chip_ids)
        
        # For the first device process, set the event to notify the main process the tti has been unzipped
        # So that the main process can launch other processes
        # Prevents processes from racing to unzip the tti
        if events.tti_first_load_event:
            events.tti_first_load_event.set()
            
        output_q = pybuda.initialize_pipeline(training=False, sample_inputs=compile_inputs)
        
        if events.initialize_completed_event:
            events.initialize_completed_event.set()
        
        first_current_index = config.inputs.first_current_index
        pad_token_id = config.inputs.pad_token_id
        write_index = config.inputs.write_index
        loop_count = 1
        num_tokens_to_generate = config.inputs.num_tokens_to_generate
        
        input_ids = run_inputs[0]
        encoder_attention_mask = run_inputs[1]
        decoder_input_ids = run_inputs[2]
        decoder_attention_mask = run_inputs[3]
        is_text_inputs = (first_current_index is not None)
        
        
        if events.run_event:
            events.run_event.wait()
            
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
                pybuda.run_generate(input_count=loop_count, write_index=write_index)
                ans = output_q.get()
            else:
                if current_token_index == 1:
                    start_time1 = time.time()
                first_device.set_active_subgraph(2)
                generate_inputs = (decoder_input_ids, decoder_attention_mask, encoder_attention_mask)
                first_device.push_to_inputs(generate_inputs)
                pybuda.run_generate(input_count=loop_count, write_index=write_index)
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
        
        if output_wrapper.output_tensors_path:
            torch.save(generated_tokens, output_wrapper.output_tensors_path)
            
        if output_wrapper.perf_q:
            output_wrapper.perf_q.put((start_time, end_time))
            
        pybuda.shutdown()

        events.done_event.set()
        events.kill_event.wait()

    # TODO: Implement output merging for n300 data-parallel generative runs once its supported
    @staticmethod
    def _create_run_result(
        # List of outputs per card
        # each inner list is the list of generated tokens of that card, of length num_tokens_to_generate
        outputs_per_card: List[List[torch.tensor]], 
        per_card_runtime: Dict[int, Tuple[float, float]]
    ):
        per_card_start_time = {device_id: start_end[0] for device_id, start_end in per_card_runtime.items()}
        per_card_end_time = {device_id: start_end[1] for device_id, start_end in per_card_runtime.items()}
        
        return RunResult(outputs_per_card, per_card_start_time, per_card_end_time)
    
def _encode_chip_ids(chip_ids: List[int]) -> str:
    return "_".join([str(chip_id) for chip_id in chip_ids])

def _initialize_tti_image(
    output_dir: str,
    precompiled_tti_path: Optional[str] = None,
):
    # copy tti over to the output directory if it isn't already there
    precompiled_tti_path = os.path.realpath(precompiled_tti_path)
    precompiled_tti_name = os.path.basename(precompiled_tti_path)
    image_path = os.path.join(output_dir, precompiled_tti_name)
    if os.path.abspath(precompiled_tti_path) != os.path.abspath(image_path):
        shutil.copy(precompiled_tti_path, image_path)
            
    return image_path

def _run(
    run_mode: RunMode,
    configs: Union[List[ForwardRunConfig], List[GenerativeRunConfig]],
    output_dir: str,
    sync_at_run_start: bool,
    rm_tmp_dirs: bool,
):
    procs = []
    device_ids_per_card = [config.chip_ids for config in configs]
    num_cards = len(device_ids_per_card)
    
    mp_context = torch.multiprocessing.get_context('spawn')
    all_events: List[RunEvents] = []
    all_output_wrappers: List[RunOutputs] = []
    # Shared events 
    kill_event = mp_context.Event()
    run_event = mp_context.Event() if sync_at_run_start else None

    if run_mode == RunMode.FORWARD:
        runner = ForwardRun._multi_thread_forward_run
        
    elif run_mode == RunMode.GENERATIVE:
        runner = GenerativeRun._single_thread_generative_model_run
    
    # Temporary directories for each device to dump intermediates such as outputs
    tmp_dirs = [os.path.join(output_dir, f"tmp_device_{_encode_chip_ids(chip_ids)}") for chip_ids in device_ids_per_card]
    for tmp_dir in tmp_dirs:
        os.makedirs(tmp_dir, exist_ok=True)
    
    for card_index, config in enumerate(configs):
        events = RunEvents(
            run_event=run_event,
            kill_event=kill_event,
            process_start_event=mp_context.Event(),
            done_event=mp_context.Event(),
            tti_first_load_event=mp_context.Event() if card_index == 0 else None,
            initialize_completed_event=mp_context.Event() if sync_at_run_start else None,
        )
        output_wrapper = RunOutputs(
            output_tensors_path=os.path.join(tmp_dirs[card_index], f"output_tensors_{_encode_chip_ids(config.chip_ids)}.pth"),
            perf_q=mp_context.Queue(),
        )
        all_events.append(events)
        all_output_wrappers.append(output_wrapper)
        p = mp_context.Process(
            target=runner, 
            args=(config, events, output_wrapper)
        )
        p.start()
        procs.append(p)
        events.process_start_event.wait()
        if events.tti_first_load_event:
            events.tti_first_load_event.wait()

    if sync_at_run_start:
        for device_events in all_events:
            device_events.wait_for_initialize_complete()
            
        logger.info(f"Initialize completed on all {num_cards} cards, launching run")
        run_event.set()
    
    for device_events in all_events:
        device_events.wait_for_run_complete()
    
    outputs_per_card = [output_wrapper.get_output_tensors() for output_wrapper in all_output_wrappers]
    per_card_start_end = {i: all_output_wrappers[i].get_start_end_time() for i in range(num_cards)}
    
    # Terminate the processes after reading the outputs
    kill_event.set()
    for proc_id, p in enumerate(procs):
        p.join()
        logger.info(f"Devices {device_ids_per_card[proc_id]} finished run successfully")

    # Clean up intermediate directories
    if rm_tmp_dirs:
        logger.info("Cleaning up temporary directories")
        for tmp_dir in tmp_dirs:
            shutil.rmtree(tmp_dir)
    
    if run_mode == RunMode.FORWARD:
        run_result: RunResult = ForwardRun._create_run_result(outputs_per_card, per_card_start_end) 
    elif run_mode == RunMode.GENERATIVE:
        run_result: RunResult = GenerativeRun._create_run_result(outputs_per_card, per_card_start_end)
        
    return run_result
    
def split_tensor_batch(input_data, num_cards: int):
    '''
    Splits tensors in input data recursively
    If input_data = ((tensor1, tensor2), tensor3) and we have 2 cards
    returns [
        [[first_half_tensor1, first_half_tensor2], first_half_tensor3]],
        [[second_half_tensor1, second_half_tensor2], second_half_tensor3]]
    ]
    '''
    inputs_per_card = [[] for _ in range(num_cards)]
    def _split_tensors(input_data, containers: List[List[Any]]):
        num_cards = len(containers)
        if isinstance(input_data, torch.Tensor):
            assert input_data.shape[0] % num_cards == 0, "Number of cards must divide the total batch size evenly"
            input_split = torch.tensor_split(input_data, num_cards, dim=0)
            for card_index in range(num_cards):
                containers[card_index] = input_split[card_index]
        
        elif isinstance(input_data, (list, tuple)):
            for data in input_data:
                new_containers = [[] for _ in range(num_cards)]
                _split_tensors(data, new_containers)
                for card_index in range(num_cards):
                    containers[card_index].append(new_containers[card_index])
            
        else:
            raise TypeError("Input data should contain list, tuple or torch tensor only")
    
    _split_tensors(input_data, inputs_per_card)
    return inputs_per_card
    
def detach_all_tensors(data):
    if isinstance(data, torch.Tensor):
        return data.detach()

    elif isinstance(data, list):
        for i in range(len(data)):
            data[i] = detach_all_tensors(data[i])
            
    else:
        raise TypeError("Input data should contain list or torch tensor only")
    
    return data
            
def run_tti_data_parallel(
    arch: pybuda.BackendDevice,
    device_ids: List[List[int]],
    run_mode: RunMode,
    inputs: Union[ForwardRunInputs, GenerativeRunInputs],
    sync_at_run_start: bool = False,
    rm_tmp_dirs: bool = True,
    precompiled_tti_path: str = None,
    output_dir: str = "./device_images",
    num_loops: Optional[int] = None,
) -> "RunResult":
    '''
    User-facing API. Run a tti on multiple cards in parallel.
    Arguments: 
    - arch: Architecture of the devices.
    - device_ids: List of device ids to run the tti on, each sublist should start with mmio-mapped device id.
    - run_mode: Mode to run on. Currently supports forward and generative runs.
    - inputs: List of inputs to run the tti on.
    - sync_at_run_start: If True, the processes will wait until all processes are ready to run before starting the run.
    - rm_tmp_dirs: If True, remove all temporary directories created for each card.
    - precompiled_tti_path: Path to a precompiled tti image to run on the cards.
    - output_dir: Directory to store the ttis as well as the unzipped tti directories. If it doesn't exist, one will be created.
        If precompiled_tti_path is provided, the tti will be copied to this directory.
    - num_loops: Number of loops to run the tti. For generative runs, this will be hardcoded to 1.
    Returns:
    - RunResult object containing the merged outputs and start/end times of the run on each card.
    '''
    assert arch in [pybuda.BackendDevice.Wormhole_B0, pybuda.BackendDevice.Grayskull], "Unsupported device architecture"
    assert precompiled_tti_path
    if len(device_ids[0]) > 1:
        assert os.environ.get("PYBUDA_N300_DATA_PARALLEL", "0") == "1", "Only support multi-device override in N300 data parallel mode"
    
    if arch == pybuda.BackendDevice.Wormhole_B0 and os.environ.get("PYBUDA_FORCE_THREADS", "0") != "1":
        logger.warning("PYBUDA_FORCE_THREADS is not set, this may cause errors when running on multiple devices due to parallel execution of create-ethernet-map")
    
    output_dir = os.path.realpath(output_dir)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image_path = _initialize_tti_image(
        output_dir=output_dir,
        precompiled_tti_path=precompiled_tti_path,
    )
    
    if run_mode == RunMode.FORWARD:
        assert isinstance(inputs, ForwardRunInputs)
        inputs_per_card = ForwardRunInputs.get_inputs_per_card(inputs, len(device_ids))
        configs: List[ForwardRunConfig] = [
            ForwardRunConfig(
                chip_ids=devices,
                inputs=inputs_per_card[card],
                tti_path=image_path,
                loop_count=num_loops,
            ) for card, devices in enumerate(device_ids)
        ]
        
    elif run_mode == RunMode.GENERATIVE:
        assert isinstance(inputs, GenerativeRunInputs)
        inputs_per_card = GenerativeRunInputs.get_inputs_per_card(inputs, len(device_ids))          
        image_path = _initialize_tti_image(
            output_dir=output_dir,
            precompiled_tti_path=precompiled_tti_path,
        )
        configs: List[GenerativeRunConfig] = [
            GenerativeRunConfig(
                chip_ids=devices,
                inputs=inputs_per_card[card],
                tti_path=image_path,
            ) for card, devices in enumerate(device_ids)
        ]
        
    else:
        raise TypeError("Invalid run mode provided. Supported modes are FORWARD and GENERATIVE.")
    
    run_result: RunResult = _run(run_mode=run_mode, configs=configs, output_dir=output_dir, sync_at_run_start=sync_at_run_start, rm_tmp_dirs=rm_tmp_dirs)
    
    return run_result