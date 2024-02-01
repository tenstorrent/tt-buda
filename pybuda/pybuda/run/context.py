# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Global context for current inference/training runs
"""

from typing import Dict, Optional, List

import threading
import torch.multiprocessing as mp
from multiprocessing.synchronize import Event as EventClass
from multiprocessing.synchronize import Barrier as BarrierClass

g_run_contexts: Dict[str, "RunContext"] = {}
g_current_context: Optional["RunContext"] = None


class RunContext:

    def __init__(self, name):
        self.name = name
        self.active = False
        self.training: Optional[bool] = None
        self.shutdown_event: Optional[EventClass] = None
        self.final_barrier: Optional[BarrierClass] = None
        self.input_gradient_queue: Optional[mp.Queue] = None
        self.output_queue: Optional[mp.Queue] = None
        self.checkpoint_queue: Optional[mp.Queue] = None
        self.intermediates_queue: Optional[mp.Queue] = None
        self.processes: List[mp.Process] = []
        self.loop_thread: Optional[threading.Thread] = None
        self.error: bool = False

        ## For hacked version of FW looping. Remove when Hacked FW looping is removed
        self.global_input_index = 0

    @classmethod
    def create_new(cls, 
            training: bool,
            shutdown_event: EventClass,
            final_barrier: BarrierClass,
            name: str = "pybuda_default") -> "RunContext":
        """ 
        Create a new context, register it and make it current
        """
        global g_current_context, g_run_contexts
        if name in g_run_contexts:
            raise RuntimeError("Trying to create a new context when one already exists with the same name")

        ctx = RunContext(name)
        ctx.active = True
        ctx.training = training
        ctx.shutdown_event = shutdown_event
        ctx.final_barrier = final_barrier

        g_current_context = ctx
        g_run_contexts[name] = ctx
        return ctx

def get_current_context() -> Optional[RunContext]:
    """
    Get current run context, or None if there isn't one
    """
    return g_current_context

def clear_current_context():
    global g_current_context, g_run_contexts
    if g_current_context is None:
        return

    del g_run_contexts[g_current_context.name]
    g_current_context = None

def context_reset():
    global g_current_context, g_run_contexts
    g_run_contexts = {}
    g_current_context = None

