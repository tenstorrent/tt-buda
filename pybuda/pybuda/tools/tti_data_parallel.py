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
import traceback
import shutil
from typing import Iterable, Optional, Dict, List, Tuple, Union, Any
import pybuda
from loguru import logger
from dataclasses import dataclass, field
from enum import Enum

OUTPUT_TTI_NAME = "parallel_tti_run.tti"

class Status(Enum):
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"
    
class RunMode(Enum):
    FORWARD = "FORWARD"
    GENERATIVE = "GENERATIVE"
    
class RunnerState(Enum):
    UNINITIALIZED = "UNINITIALIZED",
    INITIALIZED = "INITIALIZED",
    SHUTDOWN = "SHUTDOWN"
    
@dataclass
class ForwardInputs:
    run_inputs: Iterable[torch.Tensor] = None
        
    def __len__(self):
        return len(self.run_inputs)
    
    def __getitem__(self, index):
        return self.run_inputs[index]

    @staticmethod
    def split_inputs_per_card(all_inputs: "ForwardInputs", num_cards: int) -> List["ForwardInputs"]:
        inputs_per_card = split_tensor_batch(all_inputs.run_inputs, num_cards)
        return inputs_per_card
    
@dataclass
class GenerativeInputs:
    run_inputs: Iterable[torch.Tensor] = None
    num_tokens_to_generate: int = None
    write_index: int = 0
    first_current_index: Optional[int] = None
    pad_token_id: Optional[int] = None
    
    def __post_init__(self):
        assert self.run_inputs
        assert self.num_tokens_to_generate
        
    def __len__(self):
        return len(self.run_inputs)
    
    def __getitem__(self, index):
        return self.run_inputs[index]
    
    @staticmethod
    def split_inputs_per_card(all_inputs: "GenerativeInputs", num_cards: int) -> List["GenerativeInputs"]:
        # autograd does not support crossing process boundaries, this is an issue for whisper
        # detach all input tensors from compute graph to bypass this issue
        run_inputs_per_card = detach_all_tensors(split_tensor_batch(all_inputs.run_inputs, num_cards))
        inputs_per_card: List[GenerativeInputs] = []
        for card_index in range(num_cards):
            inputs_per_card.append(
                GenerativeInputs(
                    run_inputs=run_inputs_per_card[card_index],
                    num_tokens_to_generate=all_inputs.num_tokens_to_generate,
                    write_index=all_inputs.write_index,
                    first_current_index=all_inputs.first_current_index,
                    pad_token_id=all_inputs.pad_token_id,
                )
            )
            
        return inputs_per_card

@dataclass
class CompileConfigForward:
    chip_ids: List[int] = field(default_factory=list)
    compile_inputs: Iterable[torch.Tensor] = None
    tti_path: str = ""
    # Follow flow of benchmark.py: give push inputs a 2 second head start
    benchmark_perf: bool = False
    
    def __post_init__(self):
        assert self.chip_ids
        assert self.compile_inputs
        assert self.tti_path
    
@dataclass
class CompileConfigGenerative:
    chip_ids: List[int] = field(default_factory=list)
    compile_inputs: Iterable[torch.Tensor] = None
    tti_path: str = ""
    
    def __post_init__(self):
        assert self.chip_ids
        assert self.compile_inputs
        assert self.tti_path
    
@dataclass
class ProcessEvents:
    # Set by the child process when its done running
    done_event: torch.multiprocessing.Event = None
    
    # Set by the main process to synchronize the start of the run across processes
    run_event: torch.multiprocessing.Event = None
    
    # Set by the child process after it has finished initializing pipeline
    initialize_completed_event: torch.multiprocessing.Event = None
    
    # Shared event between all processes, set by the main process when the process can be terminated
    kill_event: torch.multiprocessing.Event = None
    
    # Shared event between all processes, set by any process that raised an error
    error_event: torch.multiprocessing.Event = None
    
    def __post_init__(self):
        assert self.done_event
        assert self.kill_event
        assert self.run_event
        assert self.initialize_completed_event


    @staticmethod
    def wait_for_event(target_event: torch.multiprocessing.Event, error_event: torch.multiprocessing.Event, timeout=10) -> Status:
        while True:
            if target_event.wait(timeout=timeout):
                return Status.SUCCESS
            
            if error_event.is_set():
                return Status.ERROR

@dataclass
class ProcessQueues:
    input_queue: torch.multiprocessing.Queue = None
    output_queue: torch.multiprocessing.Queue = None
    perf_queue: torch.multiprocessing.Queue = None
    config_queue: torch.multiprocessing.Queue = None

    def __post_init__(self):
        assert self.input_queue and self.output_queue and self.perf_queue
        
    def push_inputs(self, inputs: List[torch.Tensor]):
        self.input_queue.put(inputs)
        
    def pop_outputs(self, timeout=10):
        return self.output_queue.get(timeout=timeout)

    def get_start_end_time(self, timeout=120):
        return self.perf_queue.get(timeout=timeout)
    
    def get_next_config(self, timeout=120):
        return self.config_queue.get(timeout=timeout)

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
    def _multi_thread_forward_run(compile_config: CompileConfigForward, events: ProcessEvents, queues: ProcessQueues):
        
        tt0 = pybuda.TTDevice.load_image(img_path=compile_config.tti_path, device_id_overrides=compile_config.chip_ids)
            
        device_output_q = pybuda.initialize_pipeline(training=False, sample_inputs=compile_config.compile_inputs)
        
        def push_inputs_thread(tt_device: pybuda.TTDevice, main_process_input_q, loop_count: int):
            for _ in range(loop_count):
                if pybuda.error_raised():
                    print(" * Aborting input thread due to error")
                    return
                inputs = main_process_input_q.get(timeout=60)
                tt_device.push_to_inputs(inputs)
                
        def pop_outputs_thread(device_output_q, main_process_output_q, loop_count: int):
            for _ in range(loop_count):
                while True:
                    try:
                        outputs = device_output_q.get(timeout=1)
                        main_process_output_q.put([output.to_pytorch() for output in outputs])
                        break
                    except queue.Empty as _:
                        if pybuda.error_raised():
                            print(" * Aborting output thread due to error")
                            return
        
        events.initialize_completed_event.set()
        
        while not events.kill_event.is_set() and not events.error_event.is_set():
            if not events.run_event.wait(timeout=1):
                continue
            
            loop_count = queues.get_next_config()
            input_thread = threading.Thread(target=push_inputs_thread, args=(tt0, queues.input_queue, loop_count))
            output_thread = threading.Thread(target=pop_outputs_thread, args=(device_output_q, queues.output_queue, loop_count))

            input_thread.start()
            output_thread.start()
            
            if compile_config.benchmark_perf:
                time.sleep(2) # Let the input thread start up and transfer initial data, reaching something like "steady state"
            
            start = time.time()
            
            pybuda.run_forward(input_count=loop_count)
            
            input_thread.join()
            output_thread.join()
            
            end = time.time()
            
            queues.perf_queue.put((start, end))
            events.run_event.clear()
            events.done_event.set()

    @staticmethod
    def multi_thread_forward_run(compile_config: CompileConfigForward, events: ProcessEvents, queues: ProcessQueues):
        try:
            ForwardRun._multi_thread_forward_run(compile_config, events, queues)
            
        except Exception as e:
            logger.error(f"Process running on chips {compile_config.chip_ids} raised an exception: {str(e)}")
            print(traceback.format_exc())
            events.error_event.set()
            
        finally:
            pybuda.shutdown()
            
    @staticmethod
    def create_run_result(
         # List of outputs per card_index, per loop
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
    
    # Currently all generative models are tested with batch 1 in benchmark.py. These models may not work with larger batch.
    # Assume batch 1 inputs for now
    @staticmethod
    def _single_thread_generative_model_run(compile_config: CompileConfigGenerative, events: ProcessEvents, queues: ProcessQueues):
        
        from pybuda.pybudaglobal import TILE_DIM
        
        first_device = pybuda.TTDevice.load_image(img_path=compile_config.tti_path, device_id_overrides=compile_config.chip_ids)
            
        output_q = pybuda.initialize_pipeline(training=False, sample_inputs=compile_config.compile_inputs)
        
        events.initialize_completed_event.set()
        while not events.kill_event.is_set() and not events.error_event.is_set():
            if not events.run_event.wait(timeout=1):
                continue
    
            run_inputs: GenerativeInputs = queues.input_queue.get(timeout=60)
            first_current_index = run_inputs.first_current_index
            pad_token_id = run_inputs.pad_token_id
            write_index = run_inputs.write_index
            loop_count = 1
            num_tokens_to_generate = run_inputs.num_tokens_to_generate
            
            input_ids = run_inputs[0]
            encoder_attention_mask = run_inputs[1]
            decoder_input_ids = run_inputs[2]
            decoder_attention_mask = run_inputs[3]
            is_text_inputs = (first_current_index is not None)
                
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
            
            queues.output_queue.put(generated_tokens)
                
            queues.perf_queue.put((start_time, end_time))
            
            events.run_event.clear()
            events.done_event.set()
        
    @staticmethod
    def single_thread_generative_model_run(compile_config: CompileConfigGenerative, events: ProcessEvents, queues: ProcessQueues):
    # TODO: Implement output merging for n300 data-parallel generative runs once its supported
        try:
            GenerativeRun._single_thread_generative_model_run(compile_config, events, queues)
            
        except Exception as e:
            logger.error(f"Process running on chips {compile_config.chip_ids} raised an exception: {str(e)}")
            print(traceback.format_exc())
            events.error_event.set()
            
        finally:
            pybuda.shutdown()
            
    @staticmethod
    def create_run_result(
        # List of outputs per card_index
        # each inner list is the list of generated tokens of that card_index, of length num_tokens_to_generate
        outputs_per_card: List[List[torch.tensor]], 
        per_card_runtime: Dict[int, Tuple[float, float]]
    ):
        per_card_start_time = {device_id: start_end[0] for device_id, start_end in per_card_runtime.items()}
        per_card_end_time = {device_id: start_end[1] for device_id, start_end in per_card_runtime.items()}
        
        return RunResult(outputs_per_card, per_card_start_time, per_card_end_time)
    
class MultiCardRunner:
    def __init__(
        self, 
        run_mode: RunMode,
        # Configs per card_index
        device_ids: List[List[int]],
        output_dir: str,
        name: str = "MultiCardRunner"
    ):
        assert len(device_ids) > 0
        assert run_mode in [RunMode.FORWARD, RunMode.GENERATIVE]
        assert output_dir
        self.name = name
        self._state: RunnerState = RunnerState.UNINITIALIZED
        self._run_mode: RunMode = run_mode
        self._num_cards: int = len(device_ids)
        self._device_ids: List[List[int]] = device_ids
        self._output_dir: str = output_dir

        self._mp_context = None
        self._processes: List[torch.multiprocessing.Process] = None
        self._all_events: List[ProcessEvents] = []
        self._all_queues: List[ProcessQueues] = []
        
        # Shared events between all processes
        self._kill_event = None
        self._error_event = None
        
    def initialize(self, compile_configs: Union[List[CompileConfigForward], List[CompileConfigGenerative]]) -> None:
        assert self._state == RunnerState.UNINITIALIZED, "Can't re-initialize a MultiCardRunner"
        
        self._mp_context = torch.multiprocessing.get_context('spawn')
        self._processes: List[torch.multiprocessing.Process] = []
        
        # init shared events 
        self._kill_event = self._mp_context.Event()
        self._error_event = self._mp_context.Event()

        if self._run_mode == RunMode.FORWARD:
            runner_function = ForwardRun.multi_thread_forward_run
            
        elif self._run_mode == RunMode.GENERATIVE:
            runner_function = GenerativeRun.single_thread_generative_model_run
        
        for card_index, config in enumerate(compile_configs):
            events = ProcessEvents(
                run_event=self._mp_context.Event(),
                done_event=self._mp_context.Event(),
                initialize_completed_event=self._mp_context.Event(),
                kill_event=self._kill_event,
                error_event=self._error_event
            )
            queues = ProcessQueues(
                input_queue=self._mp_context.Queue(),
                output_queue=self._mp_context.Queue(),
                perf_queue=self._mp_context.Queue(),
                config_queue=self._mp_context.Queue(),
            )
            
            self._all_events.append(events)
            self._all_queues.append(queues)
            
            p = self._mp_context.Process(
                target=runner_function, 
                args=(config, events, queues)
            )
            p.start()
            self._processes.append(p)
            status = ProcessEvents.wait_for_event(events.initialize_completed_event, events.error_event)
            if status == Status.ERROR:
                self._terminate_and_report()
        
        self._state = RunnerState.INITIALIZED
        logger.info(f"{self.name}: Initialize completed on all {self._num_cards} cards")
    
    def _run_forward(self, all_inputs: ForwardInputs) -> RunResult:
        num_loops = len(all_inputs)
        inputs_per_card: List[ForwardInputs] = ForwardInputs.split_inputs_per_card(all_inputs, self._num_cards)
        outputs_per_card = [[] for _ in range(self._num_cards)]
        
        def pop_outputs(card_index: int, num_loops: int, error_event: torch.multiprocessing.Event):
            for _ in range(num_loops):
                while True:
                    try:
                        outputs = self._all_queues[card_index].pop_outputs(timeout=10)
                        outputs_per_card[card_index].append(outputs)
                        break
                    except queue.Empty as _:
                        if error_event.is_set():
                            return
        
        pop_output_threads: List[threading.Thread] = []
        for card_index in range(self._num_cards):
            t = threading.Thread(target=pop_outputs, args=(card_index, num_loops, self._error_event))
            t.start()
            pop_output_threads.append(t)
            self._assert(not self._all_events[card_index].run_event.is_set(), "Unexpected run event set before starting run")
            self._all_events[card_index].run_event.set()
            self._all_queues[card_index].config_queue.put(num_loops)
            
        for i in range(num_loops):
            if self._error_event.is_set():
                break
            for card_index, forward_inputs in enumerate(inputs_per_card):
                self._all_queues[card_index].push_inputs(forward_inputs[i])
                if self._error_event.is_set():
                    break
  
        self._wait_for_all_processes_done()
        
        for t in pop_output_threads:
            t.join()
        
        if self._error_event.is_set():
            self._terminate_and_report()
        
        per_card_start_end = {card_index: self._all_queues[card_index].get_start_end_time() for card_index in range(self._num_cards)}
    
        run_result: RunResult = ForwardRun.create_run_result(outputs_per_card, per_card_start_end)
        
        return run_result
    
    def _run_generate(self, all_inputs: GenerativeInputs) -> RunResult:
        inputs_per_card = GenerativeInputs.split_inputs_per_card(all_inputs, self._num_cards)
        outputs_per_card = [[] for _ in range(self._num_cards)]
        
        # TODO: Maybe update this after we confirm that multi-batch works for generative run
        def pop_outputs(card_index: int, error_event: torch.multiprocessing.Event):
            while True:
                try:
                    outputs = self._all_queues[card_index].pop_outputs(timeout=10)
                    outputs_per_card[card_index] = outputs
                    break
                except queue.Empty as _:
                    if error_event.is_set():
                        return
        
        pop_output_threads: List[threading.Thread] = []
        for card_index in range(self._num_cards):
            t = threading.Thread(target=pop_outputs, args=(card_index, self._error_event))
            t.start()
            pop_output_threads.append(t)
            self._assert(not self._all_events[card_index].run_event.is_set(), "Unexpected run event set before starting run")
            self._all_events[card_index].run_event.set()
        
        # Assuming batch 1, push all inputs at once
        # Since batch size is 1, this shouldn't blow up the queue
        for card_index, generative_inputs in enumerate(inputs_per_card):
            self._all_queues[card_index].push_inputs(generative_inputs)
            if self._error_event.is_set():
                break

        self._wait_for_all_processes_done()

        for t in pop_output_threads:
            t.join()
        
        if self._error_event.is_set():
            self._terminate_and_report()
        
        per_card_start_end = {card_index: self._all_queues[card_index].get_start_end_time() for card_index in range(self._num_cards)}
    
        run_result: RunResult = GenerativeRun.create_run_result(outputs_per_card, per_card_start_end)
        
        return run_result

    def run(
        self, 
        all_inputs: Union[ForwardInputs, GenerativeInputs], 
    ) -> RunResult:
        assert self._state == RunnerState.INITIALIZED
        logger.info(f"{self.name}: Launching {self._run_mode.value} run")
        if self._run_mode == RunMode.FORWARD:                
            self._assert(isinstance(all_inputs, ForwardInputs), "Expected ForwardInputs for forward run")
            result: RunResult = self._run_forward(all_inputs)
        
        elif self._run_mode == RunMode.GENERATIVE:
            self._assert(isinstance(all_inputs, GenerativeInputs), "Expected GenerativeInputs for generative run")
            result: RunResult = self._run_generate(all_inputs)
            
        return result
    

    def _wait_for_all_processes_done(self) -> Status:
        for events in self._all_events:
            status = ProcessEvents.wait_for_event(events.done_event, events.error_event)
            if status == Status.SUCCESS:
                # Unset the done event for the next run
                events.done_event.clear()
            elif status == Status.ERROR:
                break
            
        return status

    def _assert(self, cond, message: str = "") -> None:
        if cond:
            return
        logger.error(f"{self.name}: error on main process with message {message}")
        self._error_event.set()
        self._terminate_and_report()

    def _terminate_and_report(self) -> None:
        self.shutdown()
        raise RuntimeError(f"{self.name}: Aborted due to error raised on one or more processes.")
        
    def shutdown(self) -> None:
        self._state = RunnerState.SHUTDOWN
        self._kill_event.set()
        for p in self._processes:
            p.join()
    
        self._processes = []
        self._all_events = []
        self._all_queues = []
        self._kill_event = None
        self._error_event = None
        self._mp_context = None
    
def _encode_chip_ids(chip_ids: List[int]) -> str:
    return "_".join([str(chip_id) for chip_id in chip_ids])

def _initialize_tti_image(
    output_dir: str,
    precompiled_tti_path: Optional[str] = None,
) -> str:
    # copy tti over to the output directory if it isn't already there
    precompiled_tti_path = os.path.realpath(precompiled_tti_path)
    precompiled_tti_name = os.path.basename(precompiled_tti_path)
    image_path = os.path.join(output_dir, precompiled_tti_name)
    if os.path.abspath(precompiled_tti_path) != os.path.abspath(image_path):
        shutil.copy(precompiled_tti_path, image_path)
            
    return image_path

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

def initialize_multicard_runner(
    arch: pybuda.BackendDevice,
    device_ids: List[List[int]],
    run_mode: RunMode,
    compile_inputs: Iterable[torch.Tensor],
    precompiled_tti_path: str = None,
    output_dir: str = "./device_images",
    benchmark_perf: bool = False
) -> MultiCardRunner:
    '''
    Arguments: 
    - arch: Architecture of the devices.
    - device_ids: List of device ids to run the tti on, each sublist should start with mmio-mapped device id.
    - run_mode: Mode to run on. Currently supports forward and generative runs.
    - compile_inputs: List of sample inputs to be used for compilation.
    - precompiled_tti_path: Path to a precompiled tti image to run on the cards.
    - output_dir: Directory to store the ttis as well as the unzipped tti directories. If it doesn't exist, one will be created.
        If precompiled_tti_path is provided, the tti will be copied to this directory.
    - benchmark_perf: For internal perf analysis, to mimic the behaviour of benchmark.py for forward runs
    Returns:
    - MultiCardRunner object that the user can use to run on the targeted device_ids.
    '''
    assert arch in [pybuda.BackendDevice.Wormhole_B0, pybuda.BackendDevice.Grayskull], "Unsupported device architecture"
    assert precompiled_tti_path
    if len(device_ids[0]) > 1:
        assert os.environ.get("PYBUDA_N300_DATA_PARALLEL", "0") == "1", "Only support multi-device override in N300 data parallel mode"
    
    if arch == pybuda.BackendDevice.Wormhole_B0 and os.environ.get("PYBUDA_FORCE_THREADS", "0") != "1":
        logger.warning("PYBUDA_FORCE_THREADS is not set, this may cause errors when running on multiple devices due to parallel execution of create-ethernet-map")
    
    # initialize output directory
    output_dir = os.path.realpath(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # copy tti over to output directory if it doesn't exist
    image_path = _initialize_tti_image(
        output_dir=output_dir,
        precompiled_tti_path=precompiled_tti_path,
    )
    
    # Create per-card compile configs
    if run_mode == RunMode.FORWARD:
        compile_inputs_per_card = split_tensor_batch(compile_inputs, len(device_ids))
        compile_configs: List[CompileConfigForward] = [
            CompileConfigForward(
                chip_ids=devices,
                compile_inputs=compile_inputs_per_card[card_index],
                tti_path=image_path,
                benchmark_perf=benchmark_perf,
            ) for card_index, devices in enumerate(device_ids)
        ]
    
    elif run_mode == RunMode.GENERATIVE:
        compile_inputs_per_card = detach_all_tensors(split_tensor_batch(compile_inputs, len(device_ids)))
        compile_configs: List[CompileConfigGenerative] = [
            CompileConfigGenerative(
                chip_ids=devices,
                compile_inputs=compile_inputs_per_card[card_index],
                tti_path=image_path,
            ) for card_index, devices in enumerate(device_ids)
        ]
        
    else:
        raise ValueError("Invalid run mode provided. Supported modes are FORWARD and GENERATIVE.")
    
    runner: MultiCardRunner = MultiCardRunner(
        run_mode=run_mode,
        device_ids=device_ids,
        output_dir=output_dir
    )
    
    runner.initialize(compile_configs=compile_configs)
    
    return runner

