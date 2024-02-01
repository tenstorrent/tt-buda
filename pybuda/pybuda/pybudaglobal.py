# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
'Singleton' that holds pointers to all devices, modules, multi-processing queues, and other globally used data.

Devices and modules register with PyBudaGlobal as they are created
"""

from typing import Tuple
import os
import queue

from loguru import logger

from pybuda._C.backend_api import BackendType

devices = []     # Ordered list of devices running in a pipeline
modules = []

# Tile dimension in buda. All dimensions must be divisible by this.
TILE_DIM = 32 

# CoreGrid type
CoreGrid = Tuple[int, int]

# Keep track of state changes (devices, modules)
g_state_changed = True

# Are we actively tracing a graph, allows forwarding through pybuda modules without creating ops with unique names
g_tracing = False

# ID used to uniquefy nodes when no names are provided
g_unique_node_id = -1

from pyinstrument import Profiler
profiler = Profiler() if "PYBUDA_PROFILE" in os.environ else None

# If true, various defaults will revert to development mode - like, debug logging,
# default device will be model instead of silicon, etc.
def PYBUDA_DEVMODE():
    return "PYBUDA_DEVMODE" in os.environ

def set_device_pipeline(devs: Tuple["Device"]):
    """
    Devices are placed in a pipeline in the order of their creation. To create a specific order, use 
    `set_device_pipeline` and provide the ordered list.
    """
    global devices
    devices = []

    from .device import Device
    for d in devs:
        if not isinstance(d, Device):
            raise RuntimeError("Only Device types are allowed. Got: " + str(type(d)))
        devices.append(d)
    set_state_changed()

def register_device(d: "Device"):
    global devices
    devices.append(d)
    set_state_changed()

def register_module(m: "Module"):
    global modules
    modules.append(m)
    set_state_changed()

def get_devices():
    return devices

def get_tenstorrent_device():
    from pybuda.ttdevice import TTDevice
    for device in devices:
        if isinstance(device, TTDevice):
            return device
    return None

def pybuda_reset():
    """
    Clears global list of devices and modules. Only needed in special circumstances, like testing.
    """
    global devices
    global modules
    global optimizers

    for d in devices:
        d.shutdown_device()

    devices = []
    modules = []
    
    from pybuda.config import _clear_global_compiler_config
    _clear_global_compiler_config()
    from pybuda.run.context import context_reset
    context_reset()
    set_state_changed()

def state_changed() -> bool:
    """
    Return false if no new modules or devices, or changes to any, have occured since the last run
    """
    return g_state_changed

def set_state_changed():
    """
    Indicate that a device/module state has changed, and we'll need to recompile on the next run
    """
    global g_state_changed
    g_state_changed = True

def clear_state_changed():
    """
    Indicate that a run has started, and state is clean
    """
    global g_state_changed
    g_state_changed = False

def tracing() -> bool:
    """
    Has a graph trace has started, and unique op names should be generated
    """
    return g_tracing

def start_tracing():
    """
    Indicate that a graph trace has started, and unique op names should be generated
    """
    global g_tracing
    g_tracing = True

def stop_tracing():
    """
    Indicate that a graph trace has ended, pybuda graph can be forwarded without generating unique names
    """
    global g_tracing
    g_tracing = False

def get_unique_node_id():
    """
    ID used to uniquefy nodes when no names are provided
    """
    global g_unique_node_id
    g_unique_node_id += 1
    return g_unique_node_id

def reset_unique_node_id():
    """
    Reset ID used to uniquefy nodes
    """
    global g_unique_node_id
    g_unique_node_id = -1


def lazy_trace_data(data):
    """
    Lazy logger trace of large data
    """
    logger.opt(lazy=True).trace("{x}", x=lambda: data)

def is_silicon(devtype: BackendType):
    """
    Returns true if the device is a "silicon-like" - i.e. a silicon device or versim
    """
    #return devtype in [BackendType.Versim, BackendType.Silicon]
    return False

def align_up_tile(v):
    v -= 1
    return v - (v % TILE_DIM) + TILE_DIM

def round_up_div(n, d):
    return (n + d - 1) // d

def is_tile_dim_aligned(t):
    return t.shape[-1] % TILE_DIM == 0 and t.shape[-2] % TILE_DIM == 0

def create_queue(mp_context = None) -> queue.Queue:
    """
    Create a multi-processing queue, or if force sequential is set, a regular queue
    """
    if "PYBUDA_FORCE_SEQUENTIAL" not in os.environ and os.environ.get("PYBUDA_FORCE_THREADS", "0") == "0":
        assert mp_context is not None, "Must provide mp_context"
        q = mp_context.Queue()
        q.cancel_join_thread()
        return q

    return queue.Queue()

