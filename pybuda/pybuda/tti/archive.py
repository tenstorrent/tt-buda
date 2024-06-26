# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import shutil
import importlib
import subprocess
import inspect
import packaging
import struct
import sys
import tempfile
import time
import functools
import re
import yaml
from loguru import logger
import pathlib

from collections.abc import Iterable
from pybuda.config import _set_global_compiler_config, TTIDumpFormat
from pybuda.module import PyBudaModule
from pybuda.tensor import pytorch_tensor_to_tensor_desc, tensor_desc_to_pytorch_tensor
from pybuda.utils import generate_hash, get_current_pytest, write_buda_envs_configs
from pybuda.tti.utils import (
    compute_file_checksum,
    write_checksum_to_file,
    read_checksum_from_file,
)

import torch
import json
import pickle
from typing import Dict, List, Optional, Union, Match
from pybuda.optimizers import Optimizer
from pybuda.backend import BackendAPI
from pybuda._C.backend_api import (
    BackendType,
    PytorchTensorDesc,
    TilizedTensorDesc,
    binarize_tensor,
    debinarize_tensor,
    tilize_tensor,
    get_device_cluster_yaml,
)
from pybuda._C import DataFormat


def is_version_at_least(v, *, min_version="1.1.0"):
    return packaging.version.parse(v) >= packaging.version.parse(min_version)

def load_tensor_from_disk(filepath, value):
    if filepath.endswith(TTIDumpFormat.BACKEND_TILIZED.extension()):
        desc = TilizedTensorDesc()
        desc.format = DataFormat.from_json(value["format"])
        desc.num_buffers = value["num_buffers"]
        desc.buf_size_bytes = value["buf_size_bytes"]

    else:
        desc = PytorchTensorDesc()
        desc.itemsize = value["itemsize"]
        desc.format = DataFormat.from_json(value["format"])
        desc.shape = value["shape"]
        desc.strides = value["strides"]
    debinarize_tensor(desc, filepath)
    return desc


class TTDeviceImageJsonEncoder(json.JSONEncoder):
    DTYPE_TO_BIN_FORMAT = {
        torch.half: "f",
        torch.float16: "f",
        torch.bfloat16: "f",
        torch.float32: "f",
        torch.int: "i",
        torch.int32: "i",
        torch.short: "h",
        torch.int16: "h",
        torch.int8: "b",
        torch.uint8: "B",
    }
    @staticmethod
    def encode_descriptor(filename: str, tensor_desc: PytorchTensorDesc, tilized_tensor_desc: Optional[TilizedTensorDesc] = None):
        encoding = {
            "bin": filename,
            "itemsize": tensor_desc.itemsize,
            "shape": tensor_desc.shape,
            "strides": tensor_desc.strides,
            "dim": tensor_desc.dim,
            "format": tensor_desc.format,
        }
        if tilized_tensor_desc:
            encoding.update({
                "format": tilized_tensor_desc.format,
                "num_buffers": tilized_tensor_desc.num_buffers,
                "buf_size_bytes": tilized_tensor_desc.buf_size_bytes,
            })
        return encoding

    @staticmethod
    def rehash_as_pickled_object(
        d, key, object_value, filename_encoding, base_directory
    ):
        with open(os.path.join(base_directory, filename_encoding), "wb") as pkl_file:
            pickle.dump(object_value, pkl_file, pickle.HIGHEST_PROTOCOL)
        d[key] = filename_encoding

    @staticmethod
    def rehash_tensor_as_pickled_object(d, key, object_value, base_directory):
        filename_encoding = os.path.join(
            "tensors", f"torch.Tensor.{key}.pkl".replace("/", "_")
        )
        TTDeviceImageJsonEncoder.rehash_as_pickled_object(
            d, key, object_value, filename_encoding, base_directory
        )

    @staticmethod
    def rehash_tensor_as_bin_object(d, key, object_value, base_directory, tti_dump_format=Optional[TTIDumpFormat], backend_api: Optional[BackendAPI] = None):
        filename_encoding = os.path.join(
            "tensors", f"torch.Tensor.{key}.{tti_dump_format.extension()}".replace("/", "_")
        )

        assert isinstance(
            object_value, torch.Tensor
        ), "rehash_tensor_as_bin_object expects a torch.Tensor"

        from .tti import TTDeviceImage

        tensor = object_value.contiguous()  # contiguous row-major memory layout
        if is_version_at_least(TTDeviceImage.TTI_VERSION, min_version="1.1.0"):
            qdesc = backend_api.be_api.get_queue_descriptor(key)
            tensor_desc = pytorch_tensor_to_tensor_desc(tensor)
            tilized_tensor_desc = tilize_tensor(qdesc, tensor_desc) if tti_dump_format == TTIDumpFormat.BACKEND_TILIZED else None
            desc_to_binarize = tilized_tensor_desc if tilized_tensor_desc else tensor_desc

            binarize_tensor(desc_to_binarize, os.path.join(base_directory, filename_encoding))
            d[key] = TTDeviceImageJsonEncoder.encode_descriptor(filename_encoding, tensor_desc, tilized_tensor_desc)

        else:
            tensor_desc = pytorch_tensor_to_tensor_desc(tensor)
            fmt = TTDeviceImageJsonEncoder.DTYPE_TO_BIN_FORMAT[object_value.dtype]
            with open(
                os.path.join(base_directory, filename_encoding), "wb"
            ) as bin_file:
                for val in tensor.ravel().tolist():
                    bin_file.write(struct.pack(fmt, val))
            d[key] = TTDeviceImageJsonEncoder.encode_descriptor(filename_encoding, desc, tilized_tensor_desc)

    @staticmethod
    def preprocess_keys(d, base_directory: str, tti_dump_format: Optional[TTIDumpFormat] = None, backend_api: Optional[BackendAPI] = None):
        """Convert a dict's keys to strings if they are not."""
        kvs = list(d.items())
        for key, value in kvs:
            if not isinstance(key, str) and isinstance(key, torch.dtype):
                d[str(key)] = value
                del d[key]
            elif isinstance(value, torch.Tensor) or (
                isinstance(value, Iterable)
                and any(isinstance(sub_value, torch.Tensor) for sub_value in value)
            ):
                use_backend_format = tti_dump_format in (TTIDumpFormat.BACKEND, TTIDumpFormat.BACKEND_TILIZED)
                if use_backend_format and key != "cpueval_outputs":
                    TTDeviceImageJsonEncoder.rehash_tensor_as_bin_object(
                        d, key, value, base_directory, tti_dump_format=tti_dump_format, backend_api=backend_api
                    )
                else:
                    TTDeviceImageJsonEncoder.rehash_tensor_as_pickled_object(
                        d, key, value, base_directory
                    )
            elif isinstance(value, Optimizer):
                pkl_filepath = f"Optimizer.{value.get_type()}.pkl"
                TTDeviceImageJsonEncoder.rehash_as_pickled_object(
                    d, key, value, pkl_filepath, base_directory
                )
            elif isinstance(value, dict):
                d[key] = TTDeviceImageJsonEncoder.preprocess_keys(
                    value, base_directory, tti_dump_format, backend_api
                )

        return d

    def default(self, obj):
        if hasattr(obj, "to_json"):
            return obj.to_json()
        try:
            return json.JSONEncoder.default(self, obj)
        except TypeError as e:
            raise RuntimeError(
                f"JSON Serialization failed: {e}. {obj.__class__.__name__}.to_json(..) method needs to be implemented. Object={obj}."
            )


class TTDeviceImageJsonDecoder(json.JSONDecoder):
    DICT_KEY_DECODING = {
        "torch.": torch,
    }
    DATA_FORMAT_TO_DTYPE = {
        DataFormat.Float32: torch.float32,
        DataFormat.Float16_b: torch.bfloat16,
        DataFormat.Float16: torch.float16,
        DataFormat.RawUInt32: torch.int,
        DataFormat.RawUInt16: torch.int16,
        DataFormat.RawUInt8: torch.uint8,
        DataFormat.Int8: torch.int8,
    }

    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, dct):
        return dct

    @staticmethod
    def rehash_as_tensor(value, directory):
        from .tti import TTDeviceImage

        if is_version_at_least(TTDeviceImage.TTI_VERSION, min_version="1.1.0"):
            filepath= value["bin"]
            lazy_load_callable  = functools.partial(load_tensor_from_disk, os.path.join(directory, filepath), value)
            return lazy_load_callable
        else:
            dtype = TTDeviceImageJsonDecoder.DATA_FORMAT_TO_DTYPE[
                DataFormat.from_json(value["format"])
            ]
            fmt = TTDeviceImageJsonEncoder.DTYPE_TO_BIN_FORMAT[dtype]
            itemsize = struct.calcsize(fmt)
            tensor_data = []
            bin_filepath = value["bin"]
            with open(os.path.join(directory, bin_filepath), "rb") as bin_file:
                while True:
                    bytes_data = bin_file.read(itemsize)
                    if not bytes_data:
                        break
                    (val,) = struct.unpack(fmt, bytes_data)
                    tensor_data.append(val)

            tensor = torch.tensor(tensor_data, dtype=dtype).reshape(*value["shape"])
            return tensor

    @staticmethod
    def postprocess_keys(d, directory):
        """Convert a encoded dict's keys to back to original type ."""
        kvs = list(d.items())
        for key, value in kvs:
            if isinstance(d[key], dict):
                if "bin" in d[key]:
                    d[key] = TTDeviceImageJsonDecoder.rehash_as_tensor(value, directory)
                else:
                    value = TTDeviceImageJsonDecoder.postprocess_keys(value, directory)

            # convert nonstring to string if needed
            for (
                encoded_string,
                decoded_type,
            ) in TTDeviceImageJsonDecoder.DICT_KEY_DECODING.items():
                if isinstance(key, str) and key.startswith(encoded_string):
                    decoded_type = getattr(decoded_type, key[len(encoded_string) :])
                    d[decoded_type] = value
                    del d[key]
            if isinstance(value, str) and value.endswith(".pkl"):
                with open(os.path.join(directory, value), "rb") as pkl_file:
                    d[key] = pickle.load(pkl_file)

        return d


class TTIArchive:
    TTI_UNZIPPED_DIR_NAME = "unzipped_tti"

    @staticmethod
    def _create_tti_archive(device_images_directory: str, device_img_path: str) -> None:
        tti_absolute_file_path = os.path.realpath(device_img_path)

        try:
            subprocess.run(
                [
                    "tar",
                    "-cf",
                    tti_absolute_file_path,
                    "-C",
                    device_images_directory,
                    TTIArchive.TTI_UNZIPPED_DIR_NAME,
                ]
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed with error {e}")

    @staticmethod
    def _copy_backend_build_files(*, src_dir: str, dst_dir: str):
        logger.info(
            "TTDeviceImage: copying backend build files from {} to {}", src_dir, dst_dir
        )
        os.makedirs(src_dir, exist_ok=True)
        try:
            if not src_dir.endswith("/"):
                src_dir += "/"
            cmd = [
                "rsync",
                "-a",
                "--exclude=*.log",
                "--exclude=blob.yaml",
                "--exclude=*.d",
                src_dir,
                dst_dir,
            ]
            logger.info("Running command: {}", " ".join(cmd))
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed with error {e}")

    @staticmethod
    def _copy_netlist_yaml(*, netlist_yaml: str, dst_dir: str):
        shutil.copy(src=netlist_yaml, dst=dst_dir)

    @staticmethod
    def _copy_module_file(*, module_file, dst_dir):
        src = os.path.relpath(module_file, start=os.curdir)
        dst = os.path.join(dst_dir, src)
        dstfolder = os.path.dirname(dst)
        if not os.path.exists(dstfolder):
            os.makedirs(dstfolder, exist_ok=True)
        shutil.copy(src=src, dst=dst)

    @staticmethod
    def _get_device_img_path(device_img_path_override: Optional[str] = None):
        if device_img_path_override:
            device_img_path = device_img_path_override
        else:
            DEFAULT_DEVICE_PATH = (
                f"device_images/tt_{generate_hash(get_current_pytest())}.tti"
            )
            device_img_path = DEFAULT_DEVICE_PATH
        return device_img_path
    
    @staticmethod 
    def _device_override_str(device_id_overrides: List[int]) -> str:
        return "_".join([str(device_id) for device_id in device_id_overrides])
    
    @staticmethod
    def _get_override_netlist_path(original_netlist_path: str, device_id_overrides: List[int]) -> str:
        device_override_str = TTIArchive._device_override_str(device_id_overrides)
        return f"{os.path.splitext(original_netlist_path)[0]}_override_device_{device_override_str}.yaml"
    
    @staticmethod
    def _get_override_backend_output_path(oringal_binaries_path: str, device_id_overrides: List[int]) -> str:
        device_override_str = TTIArchive._device_override_str(device_id_overrides)
        return f"{oringal_binaries_path}_override_device_{device_override_str}"
    
    @staticmethod
    def _get_original_device_to_new_device_map(netlist_file_path: str, device_id_overrides: List[int]):
        with open(netlist_file_path, "r") as netlist_file:
            netlist_str = netlist_file.read()
        
        single_target_device_pattern = re.compile(r'\btarget_device:\s*(\d+)\b')
        multi_target_device_pattern = re.compile(r'target_device:\s*\[(\d+),\s*(\d+)\]')
        
        if len(device_id_overrides) == 1:
            m = single_target_device_pattern.search(netlist_str)
            assert m, "Expected single target_device in netlist"
            original_device = int(m.group(1))
            return {original_device: device_id_overrides[0]}

        m = multi_target_device_pattern.search(netlist_str)
        assert m, "Expected multi-target_device in netlist"
        original_devices = (int(m.group(1)), int(m.group(2)))
        return {old_device: new_device for old_device, new_device in zip(original_devices, device_id_overrides)}
    
    @staticmethod
    def _update_n300_dp_trisc_firmware_directories(
        netlist_path: str, 
        old_device_to_new_device_map: Dict[int, int]
    ) -> None:
        override_backend_outdir = os.path.dirname(netlist_path)
        with open(netlist_path, "r") as netlist_file:
            netlist_map = yaml.safe_load(netlist_file)
            
        device_suffix_pattern = re.compile(r'\.(\d+)$')
        all_graphs = list(netlist_map["graphs"].keys())
        
        temp_sub_dirs = []
        for sub_dir in os.listdir(override_backend_outdir):
            m = device_suffix_pattern.search(sub_dir)
            # Locate trisc firmware directories
            if m and any([graph in sub_dir for graph in all_graphs]):
                old_device_id = int(m.group(1))
                new_device_id = old_device_to_new_device_map[old_device_id]
                # temp name to prevent name collisions
                temp_sub_dir = device_suffix_pattern.sub(f".{new_device_id}_temp", sub_dir)
                temp_sub_dirs.append(temp_sub_dir)
                os.rename(os.path.join(override_backend_outdir, sub_dir), os.path.join(override_backend_outdir, temp_sub_dir))
        
        # Finalize temporary directory names
        device_suffix_pattern_temp = re.compile(r'\.(\d+)_temp$')
        for temp_sub_dir in temp_sub_dirs:
            new_sub_dir = device_suffix_pattern_temp.sub(r".\1", temp_sub_dir)
            os.rename(os.path.join(override_backend_outdir, temp_sub_dir), os.path.join(override_backend_outdir, new_sub_dir))
        
    @staticmethod
    def _update_n300_dp_nops_in_netlist_string(netlist_str: str, old_device_to_new_device_map: Dict[int, int]) -> str:
        netlist_map = yaml.safe_load(netlist_str)
        dp_nop_pattern = re.compile(r'dp_nop\.(\d+)$')
        device_suffix_pattern = re.compile(r'\.(\d+)$')
        
        def override_handler(m: Match, old_device_to_new_device_map: Dict[int, int]):
            matched_string = m.group(0)
            old_device_id = int(device_suffix_pattern.search(matched_string).group(1))
            new_device_id = old_device_to_new_device_map[old_device_id]
            return device_suffix_pattern.sub(f".{new_device_id}", matched_string)

        fields_to_override = set()
        graphs_map = netlist_map["graphs"]
        
        for graph_name, ops_map in graphs_map.items():
            for op_name, op_configs in ops_map.items():
                if not dp_nop_pattern.search(op_name):
                    continue
                fields_to_override.add(op_name)
                op_inputs = op_configs["inputs"]
                for op_input in op_inputs:
                    if not device_suffix_pattern.search(op_input):
                        continue
                    fields_to_override.add(op_input)
                
        new_netlist_str = re.sub("|".join(fields_to_override), lambda m: override_handler(m, old_device_to_new_device_map), netlist_str)
        return new_netlist_str
    
    @staticmethod
    def _update_n300_dp_compiled_graph_state(
        compiled_graph_state: "CompiledGraphState",
        old_device_to_new_device_map: Dict[int, int],
    ) -> None:
        device_suffix_pattern = re.compile(r'\.(\d+)$')
        def update_device_id_suffix(items: Union[Dict[str, str], List[str]]):
            assert isinstance(items, (dict, list)), "Expected items to be a dict or list"
            if isinstance(items, dict):
                updated_items: Dict[str, str] = {}
                for k, v in items.items():
                    old_device_id = int(device_suffix_pattern.search(k).group(1))
                    new_device_id = old_device_to_new_device_map[old_device_id]
                    updated_items[device_suffix_pattern.sub(f".{new_device_id}", k)] = v
                return updated_items
            else:
                updated_items: List[str] = []
                for item in items:
                    old_device_id = int(device_suffix_pattern.search(item).group(1))
                    new_device_id = old_device_to_new_device_map[old_device_id]
                    updated_items.append(device_suffix_pattern.sub(f".{new_device_id}", item))
                return updated_items

        compiled_graph_state.input_to_tile_dims = update_device_id_suffix(compiled_graph_state.input_to_tile_dims)

        compiled_graph_state.post_const_eval_constants = update_device_id_suffix(compiled_graph_state.post_const_eval_constants)
        compiled_graph_state.post_const_eval_parameters = update_device_id_suffix(compiled_graph_state.post_const_eval_parameters)

        compiled_graph_state.ordered_constant_node_names = update_device_id_suffix(compiled_graph_state.ordered_constant_node_names)
        compiled_graph_state.ordered_parameter_node_names = update_device_id_suffix(compiled_graph_state.ordered_parameter_node_names)

        compiled_graph_state.ordered_input_names = update_device_id_suffix(compiled_graph_state.ordered_input_names)
        compiled_graph_state.ordered_output_names = update_device_id_suffix(compiled_graph_state.ordered_output_names)
        
    # Keep consistent with epoch_loader.cpp::update_overlay_binary
    @staticmethod
    def _update_overlay_binary_hex_filenames(
        override_backend_output_dir: str,
        old_device_to_new_device_map: Dict[int, int],
    ) -> None:
        def rename_blob_file_temp(m: Match, old_device_to_new_device_map: Dict[int, int]):
            old_device_id = int(m.group(2))
            new_device_id = old_device_to_new_device_map[old_device_id]
            return f'pipegen_epoch{m.group(1)}_{new_device_id}_{m.group(3)}_{m.group(4)}_temp.hex'
        
        def rename_blob_file_final(m: Match):
            return f'pipegen_epoch{m.group(1)}_{m.group(2)}_{m.group(3)}_{m.group(4)}.hex'
        
        temporal_epoch_dir_re = re.compile(r"^temporal_epoch_\d+$")
        # (temporal_epoch)_(chip_id)_(route_r)_(route_c)
        overlay_blob_hex_re = re.compile(r"^pipegen_epoch(\d+)_(\d+)_(\d+)_(\d+).hex$")
        temp_overlay_blob_hex_re = re.compile(r"^pipegen_epoch(\d+)_(\d+)_(\d+)_(\d+)_temp.hex$")
        
        temporal_epoch_dirs = [os.path.join(override_backend_output_dir, epoch_dir) for epoch_dir in os.listdir(override_backend_output_dir) if temporal_epoch_dir_re.match(epoch_dir)]
        for temporal_epoch_dir in temporal_epoch_dirs:
            blobs_dir = os.path.join(temporal_epoch_dir, "overlay", "blobs")
            if not os.path.isdir(blobs_dir):
                continue
            temp_blob_files: List[str] = []
            for blob_hex_name in os.listdir(blobs_dir):
                # Rename blob files to a temporary name to prevent name collisions
                if overlay_blob_hex_re.match(blob_hex_name):
                    temp_blob_hex_name = overlay_blob_hex_re.sub(lambda m: rename_blob_file_temp(m, old_device_to_new_device_map), blob_hex_name)
                    os.rename(os.path.join(blobs_dir, blob_hex_name), os.path.join(blobs_dir, temp_blob_hex_name))
                    temp_blob_files.append(temp_blob_hex_name)
            
            # After all device ids have been updated, rename the temporary blob files to the final name
            for temp_blob_file in temp_blob_files:
                final_blob_hex_name = temp_overlay_blob_hex_re.sub(rename_blob_file_final, temp_blob_file)
                os.rename(os.path.join(blobs_dir, temp_blob_file), os.path.join(blobs_dir, final_blob_hex_name))

    # Keep consistent with epoch_loader.cpp::populate_queue_to_core_map_from_net2pipe
    @staticmethod
    def _update_producer_consumer_queue_yaml(
        override_backend_output_dir: str,
        old_device_to_new_device_map: Dict[int, int],
    ) -> None:
        
        def override_queue_target_device(m: Match, old_device_to_new_device_map: Dict[int, int]):
            old_device_id = int(m.group(1))
            new_device_id = old_device_to_new_device_map[old_device_id]
            return f"queue_target_device: {new_device_id}"
        
        def override_chip_id(m: Match, old_device_to_new_device_map: Dict[int, int]):
            old_device_id = int(m.group(1))
            new_device_id = old_device_to_new_device_map[old_device_id]
            return f"chip_id: {new_device_id}"
        
        temporal_epoch_dir_re = re.compile(r"^temporal_epoch_\d+$")
        chip_id_pattern = re.compile(r'chip_id:\s*(\d+)')
        queue_target_device_pattern = re.compile(r'queue_target_device:\s*(\d+)')

        temporal_epoch_dirs = [os.path.join(override_backend_output_dir, epoch_dir) for epoch_dir in os.listdir(override_backend_output_dir) if temporal_epoch_dir_re.match(epoch_dir)]
        
        for temporal_epoch_dir in temporal_epoch_dirs:
            overlay_dir = os.path.join(temporal_epoch_dir, "overlay")
            queue_to_consumer_path = os.path.join(overlay_dir, "queue_to_consumer.yaml")
            queue_to_producer_path = os.path.join(overlay_dir, "queue_to_producer.yaml")
            if os.path.isfile(queue_to_consumer_path):
                with open(queue_to_consumer_path, "r") as old_q_consumer_file:
                    queue_to_consumer_str = old_q_consumer_file.read()
                    
                queue_to_consumer_str_override = chip_id_pattern.sub(lambda m: override_chip_id(m, old_device_to_new_device_map), queue_to_consumer_str)
                queue_to_consumer_str_override = queue_target_device_pattern.sub(lambda m: override_queue_target_device(m, old_device_to_new_device_map), queue_to_consumer_str_override)
                
                with open(queue_to_consumer_path, "w") as new_q_consumer_file:
                    new_q_consumer_file.write(queue_to_consumer_str_override)
                    
            if os.path.isfile(queue_to_producer_path):
                with open(queue_to_producer_path, "r") as old_q_producer_file:
                    queue_to_producer_str = old_q_producer_file.read()
                    
                queue_to_producer_str_override = chip_id_pattern.sub(lambda m: override_chip_id(m, old_device_to_new_device_map), queue_to_producer_str)
                queue_to_producer_str_override = queue_target_device_pattern.sub(lambda m: override_queue_target_device(m, old_device_to_new_device_map), queue_to_producer_str_override)
                
                with open(queue_to_producer_path, "w") as new_q_producer_file:
                    new_q_producer_file.write(queue_to_producer_str_override)
   
    @staticmethod
    def _update_overlay_blob_dir_with_override_device_id(
        override_backend_output_dir: str,
        old_device_to_new_device_map: Dict[int, int],
    ) -> None:
        # Update chip_id in overlay blob hex file names
        TTIArchive._update_overlay_binary_hex_filenames(override_backend_output_dir, old_device_to_new_device_map)
        
        # Update device id in queue_to_consumer.yaml and queue_to_producer.yaml
        TTIArchive._update_producer_consumer_queue_yaml(override_backend_output_dir, old_device_to_new_device_map)
        

    @staticmethod
    def _create_device_override_netlist_yaml(original_netlist_path: str, device_id_overrides: List[int]) -> str:
        
        def replace_target_device_with_new_device(m: Match, old_device_to_new_device_map: Dict[int, int]):
            old_device = int(m.group(1))
            new_device = old_device_to_new_device_map[old_device]
            return f"target_device: {new_device}"
        
        single_device_pattern = re.compile(r'\btarget_device:\s*(\d+)\b')
        # Can't use yaml library here, netlist needs to be in specific format
        new_netlist_path = TTIArchive._get_override_netlist_path(original_netlist_path, device_id_overrides)
        
        if os.path.exists(new_netlist_path):
            return new_netlist_path
        
        with open(original_netlist_path, "r") as netlist_file:
            netlist_str_override = netlist_file.read()
        
        # Create a map that maps original device to new device
        old_device_to_new_device_map = TTIArchive._get_original_device_to_new_device_map(original_netlist_path, device_id_overrides)
        
        # Override single device ids with its new device equivalent
        netlist_str_override = single_device_pattern.sub(lambda m: replace_target_device_with_new_device(m, old_device_to_new_device_map), netlist_str_override)
        
        if len(device_id_overrides) > 1:
            assert os.environ.get("PYBUDA_N300_DATA_PARALLEL", "0") == "1", "Only support multi-device override in N300 data parallel mode"
            assert len(device_id_overrides) == 2, "Only support 1 or 2 device overrides"
            multi_target_device_pattern = re.compile(r'target_device:\s*\[(\d+),\s*(\d+)\]')
            
            # Override multi device arrays with new device ids
            netlist_str_override = multi_target_device_pattern.sub(rf"target_device: {device_id_overrides}", netlist_str_override)
            netlist_str_override = TTIArchive._update_n300_dp_nops_in_netlist_string(netlist_str_override, old_device_to_new_device_map)
            
        with open(new_netlist_path, "w") as new_netlist_file:
            new_netlist_file.write(netlist_str_override)
            
        return new_netlist_path
    
    @staticmethod
    def _update_cluster_desc_yaml(cluster_desc_path: str) -> str:
        if os.path.exists(cluster_desc_path):
            os.remove(cluster_desc_path)
        cluster_output_dir = os.path.dirname(cluster_desc_path)
        new_cluster_descriptor_path = get_device_cluster_yaml(cluster_output_dir)
        return new_cluster_descriptor_path
    
    @staticmethod
    def _update_runtime_data_yaml_with_override_device_id(
        new_backend_output_dir: str,
        old_device_to_new_device_map: Dict[int, int],
    ) -> None:
        device_id_overrides = list(old_device_to_new_device_map.values())
        
        # Update runtime data yaml
        new_runtime_yaml_path = os.path.join(new_backend_output_dir, "runtime_data.yaml")
        with open(new_runtime_yaml_path, "r") as f:
            new_runtime_data = yaml.safe_load(f)
            
        # Update worker_grid_sizes_per_chip
        old_worker_grid_sizes_per_chip = new_runtime_data["worker_grid_sizes_per_chip"]
        new_worker_grid_sizes_per_chip = {}
        assert len(old_worker_grid_sizes_per_chip) == len(device_id_overrides), f"Num devices mismatch between runtime data worker_grid_sizes_per_chip and devices to override"
        
        for old_device_id in old_worker_grid_sizes_per_chip:
            new_device_id = old_device_to_new_device_map[old_device_id]
            new_worker_grid_sizes_per_chip[new_device_id] = old_worker_grid_sizes_per_chip[old_device_id]
            
        new_runtime_data["worker_grid_sizes_per_chip"] = new_worker_grid_sizes_per_chip
            
        # Update harvested_rows_per_chip
        # Set the harvesting mask to be the same as what we used for the original device ids
        # If the grid size is not the same as what runtime detects during run
        # runtime will use the actual harvesting mask, and overlay will be recompiled implicitly by runtime
        old_harvested_rows_per_chip = new_runtime_data["harvested_rows_per_chip"]
        new_harvested_rows_per_chip = {}
        assert len(old_harvested_rows_per_chip) == len(device_id_overrides), f"Num devices mismatch between runtime data harvested_rows_per_chip and devices to override"
        
        for old_device_id in old_harvested_rows_per_chip:
            new_device_id = old_device_to_new_device_map[old_device_id]
            new_harvested_rows_per_chip[new_device_id] = old_harvested_rows_per_chip[old_device_id]
            
        new_runtime_data["harvested_rows_per_chip"] = new_harvested_rows_per_chip
        
        with open(new_runtime_yaml_path, 'w') as f:
            yaml.safe_dump(new_runtime_data, f)
    
    @staticmethod
    def _create_device_override_backend_output_dir(
        original_backend_output_dir: str,
        original_netlist_path: str,
        device_id_overrides: List[int]
    ) -> str:
        new_backend_output_dir = TTIArchive._get_override_backend_output_path(original_backend_output_dir, device_id_overrides)
        if os.path.exists(new_backend_output_dir):
            logger.info("TTDeviceImage: Using existing device override binaries directory {}", new_backend_output_dir)
            return new_backend_output_dir
        
        # Make a copy of the original binaries directory
        shutil.copytree(original_backend_output_dir, new_backend_output_dir)

        # Remove the original netlist and copy over the override netlist to the new binaries directory
        original_netlist_name = os.path.basename(original_netlist_path)
        os.remove(os.path.join(new_backend_output_dir, original_netlist_name))
        
        # Path to the netlist file override directly under unzipped_tti directory
        override_netlist_path = TTIArchive._get_override_netlist_path(original_netlist_path, device_id_overrides)
        override_netlist_name = os.path.basename(override_netlist_path)
        
        override_netlist_path_in_backend_outdir = os.path.join(new_backend_output_dir, override_netlist_name)
        
        TTIArchive._copy_netlist_yaml(netlist_yaml=override_netlist_path, dst_dir=override_netlist_path_in_backend_outdir)

        old_device_to_new_device_map = TTIArchive._get_original_device_to_new_device_map(original_netlist_path, device_id_overrides)
        
        # Update runtime data yaml
        TTIArchive._update_runtime_data_yaml_with_override_device_id(new_backend_output_dir, old_device_to_new_device_map)
        
        # Update relative files in the overlay output directories
        TTIArchive._update_overlay_blob_dir_with_override_device_id(new_backend_output_dir, old_device_to_new_device_map)
        
        if os.environ.get("PYBUDA_N300_DATA_PARALLEL", "0") == "1":
            # Update device id suffix in trisc firmware directories
            TTIArchive._update_n300_dp_trisc_firmware_directories(override_netlist_path_in_backend_outdir, old_device_to_new_device_map)

        return new_backend_output_dir
    
    @staticmethod
    def get_instantiate_modules(
        module_name_to_metadata: Dict[str, Dict[str, str]], unzipped_tti_directory: str
    ) -> List[PyBudaModule]:
        instantiated_modules = []
        for name, metadata in module_name_to_metadata.items():
            unzipped_directory_module_path = os.path.join(
                unzipped_tti_directory, "module_files", metadata["module_file_path"]
            )
            expected_module_path = os.path.join(
                os.getcwd(), metadata["module_file_path"]
            )

            # if module does not exist during execution, we'll copy over the module file to import location
            if not pathlib.Path(expected_module_path).is_file():
                logger.info(
                    "TTDeviceImage: copying module file from {} to {}",
                    unzipped_tti_directory,
                    expected_module_path,
                )
                os.makedirs(os.path.dirname(expected_module_path), exist_ok=True)
                shutil.copy(unzipped_directory_module_path, expected_module_path)

            # Create a new module object and load
            module = importlib.import_module(metadata["module"])

            # Fetch class from module and instantiate a new object from the class
            obj_class = getattr(module, metadata["class"])
            instantiated_modules.append(obj_class(name))

        return instantiated_modules

    @staticmethod
    def construct_device_image(unzipped_tti_directory: str, device_id_overrides: Optional[List[int]] = None) -> "TTDeviceImage":
        from .tti import TTDeviceImage

        device_image = None
        with open(
            os.path.join(unzipped_tti_directory, "device.json"), "r"
        ) as json_file:
            device_image_dict = json.load(json_file, cls=TTDeviceImageJsonDecoder)
                
            TTDeviceImageJsonDecoder.postprocess_keys(
                device_image_dict, unzipped_tti_directory
            )

            try:
                device_image = TTDeviceImage.from_dict(device_image_dict)
            except KeyError as e:
                raise ValueError(
                    f"TTI failed to deserialize. TTDeviceImage not contain key: {e}. TTI recompilation required."
                )

            if device_id_overrides is not None:
                if os.environ.get("PYBUDA_N300_DATA_PARALLEL", "0") == "1" or len(device_id_overrides) == 1:
                    # In n300 data parallel mode, the chip_ids of the device image is of length 1 as well
                    assert len(device_image.chip_ids) == 1, "Cannot override multi-device TTI image with single device"
                    device_image.chip_ids = [device_id_overrides[0]]
                    
                else:
                    device_image.chip_ids = device_id_overrides
                
            sys.path.append(
                "."
            )  # We need this line because the tvm->python code path does and pickle requires a match
            device_image.modules = TTIArchive.get_instantiate_modules(
                device_image.module_name_to_metadata, unzipped_tti_directory
            )
                
            netlist_file_basename = os.path.basename(
                device_image.compiled_graph_state.netlist_filename
            )
            
            original_netlist_file_path = os.path.join(
                unzipped_tti_directory, netlist_file_basename
            )
            
            netlist_file_path = original_netlist_file_path
            
            if device_id_overrides is not None:
                netlist_file_path = TTIArchive._create_device_override_netlist_yaml(netlist_file_path, device_id_overrides)
            
            device_image.compiled_graph_state.netlist_filename = netlist_file_path

            # Update the device image compiled_graph_state with the new device ids
            if device_id_overrides is not None and os.environ.get("PYBUDA_N300_DATA_PARALLEL", "0") == "1":
                old_device_to_new_device_map = TTIArchive._get_original_device_to_new_device_map(original_netlist_file_path, device_id_overrides)
                TTIArchive._update_n300_dp_compiled_graph_state(device_image.compiled_graph_state, old_device_to_new_device_map)
                    
        return device_image, original_netlist_file_path

    @staticmethod
    def load_from_disk(tti_file_path: str, device_id_overrides: Optional[List[int]] = None) -> "TTDeviceImage":
        tti_file_path = TTIArchive._get_device_img_path(tti_file_path)
        absolute_device_image_path = os.path.realpath(tti_file_path)
        logger.info("TTDeviceImage::loading from {}", absolute_device_image_path)
        absolute_device_image_directory = os.path.dirname(absolute_device_image_path)
        unzipped_tti_directory = os.path.join(
            absolute_device_image_directory, TTIArchive.TTI_UNZIPPED_DIR_NAME
        )

        def contains_matching_checksum(tti_checksum) -> bool:
            directory_checksum = read_checksum_from_file(
                os.path.join(unzipped_tti_directory, "checksum.txt")
            )
            return tti_checksum == directory_checksum

        tti_checksum = compute_file_checksum(absolute_device_image_path)
        found_matching_checksum = contains_matching_checksum(tti_checksum)
        
        if found_matching_checksum:
            logger.info(
                f"TTI: Netlist checksum matches - populating TTDevice from pre-existing dir {unzipped_tti_directory}"
            )
        else:
            logger.info(
                f"TTI: No matching checksum found - extracting TTI to dir {os.path.realpath(unzipped_tti_directory)} "
            )
            shutil.rmtree(unzipped_tti_directory, ignore_errors=True)
            try:
                subprocess.run(
                    ["tar", "-xf", tti_file_path, "-C", absolute_device_image_directory]
                )
            except subprocess.CalledProcessError as e:
                logger.error(f"Command failed with error {e}")
            write_checksum_to_file(
                tti_checksum, os.path.join(unzipped_tti_directory, "checksum.txt")
            )

        device_image, original_netlist_file_path = TTIArchive.construct_device_image(unzipped_tti_directory, device_id_overrides)
            
        if device_image.compiler_cfg.backend_cluster_descriptor_path:
            device_image.compiler_cfg.backend_cluster_descriptor_path = os.path.join(
                absolute_device_image_directory,
                device_image.compiler_cfg.backend_cluster_descriptor_path
            )
            # If we unzipped the tti for the first time and we are overriding devices
            # Regenerate the cluster descriptor
            if device_id_overrides is not None and not found_matching_checksum:
                new_cluster_desc_path = TTIArchive._update_cluster_desc_yaml(device_image.compiler_cfg.backend_cluster_descriptor_path)
                device_image.compiler_cfg.backend_cluster_descriptor_path = new_cluster_desc_path
            
        if device_image.compiler_cfg.backend_output_dir:
            device_image.compiler_cfg.backend_output_dir = os.path.join(
                absolute_device_image_directory,
                device_image.compiler_cfg.backend_output_dir,
            )
            if device_id_overrides is not None:
                device_image.compiler_cfg.backend_output_dir = TTIArchive._create_device_override_backend_output_dir(
                    device_image.compiler_cfg.backend_output_dir,
                    original_netlist_file_path,
                    device_id_overrides,
                )

        if device_image.compiler_cfg.backend_runtime_params_path:
            device_image.compiler_cfg.backend_runtime_params_path = os.path.join(
                absolute_device_image_directory,
                device_image.compiler_cfg.backend_runtime_params_path
            )

        if device_image.compiler_cfg.backend_device_descriptor_path: 
            device_image.compiler_cfg.backend_device_descriptor_path = os.path.join(
                absolute_device_image_directory,
                device_image.compiler_cfg.backend_device_descriptor_path
            )
            
            
        _set_global_compiler_config(device_image.compiler_cfg)

        return device_image

    @staticmethod
    def encode_archive_base_path(full_path: str, original_base_path: str, modified_base_path: str):
        if full_path:
            return full_path.replace(original_base_path, modified_base_path)
        return full_path


    @staticmethod
    def save_to_disk(
        device_image: "TTDeviceImage", device_img_path_override: Optional[str] = None, backend_api: Optional[BackendAPI] = None
    ):
        from .tti import TTDeviceImage

        device_img_path = TTIArchive._get_device_img_path(device_img_path_override)
        logger.info("TTI: Saving device image to {}", device_img_path)

        start_time = time.time()
        dst_relative_directory_tti = os.path.dirname(device_img_path)
        os.makedirs(os.path.realpath(dst_relative_directory_tti), exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp:
            src_tti_directory_to_zip = os.path.join(
                tmp, TTIArchive.TTI_UNZIPPED_DIR_NAME
            )
            os.makedirs(src_tti_directory_to_zip, exist_ok=True)

            copy_start = time.time()
            relative_backend_output_dir = os.path.join(
                TTIArchive.TTI_UNZIPPED_DIR_NAME, "backend_build_binaries"
            )
            TTIArchive._copy_backend_build_files(
                src_dir=device_image.compiler_cfg.backend_output_dir,
                dst_dir=os.path.join(
                    src_tti_directory_to_zip, "backend_build_binaries"
                ),
            )
            logger.debug(
                "TTI: Copying backend build files took {} seconds",
                time.time() - copy_start,
            )
            
            current_backend_output_dir = device_image.compiler_cfg.backend_output_dir
            device_image.compiler_cfg.backend_output_dir = relative_backend_output_dir
            device_image.compiler_cfg.backend_device_descriptor_path = TTIArchive.encode_archive_base_path(
                device_image.compiler_cfg.backend_device_descriptor_path,
                current_backend_output_dir,
                relative_backend_output_dir
            )
            device_image.compiler_cfg.backend_cluster_descriptor_path = TTIArchive.encode_archive_base_path(
                device_image.compiler_cfg.backend_cluster_descriptor_path,
                current_backend_output_dir,
                relative_backend_output_dir
            )
            device_image.compiler_cfg.backend_runtime_params_path = TTIArchive.encode_archive_base_path(
                device_image.compiler_cfg.backend_runtime_params_path,
                current_backend_output_dir,
                relative_backend_output_dir
            )

            netlist_path = device_image.compiled_graph_state.netlist_filename
            netlist_file_basename = os.path.basename(netlist_path)
            device_image.compiled_graph_state.netlist_filename = os.path.join(
                relative_backend_output_dir, netlist_file_basename
            )

            tensors_directory = os.path.join(src_tti_directory_to_zip, "tensors")
            os.makedirs(tensors_directory, exist_ok=True)

            with open(os.path.join(src_tti_directory_to_zip, "device.json"), "w") as f:
                device_image_state_dict = TTDeviceImage.to_dict(device_image)
                del device_image_state_dict["modules"]
                TTDeviceImageJsonEncoder.preprocess_keys(
                    device_image_state_dict,
                    src_tti_directory_to_zip,
                    device_image.compiler_cfg.tti_dump_format,
                    backend_api=backend_api,
                )
                device_image_state_json = json.dumps(
                    device_image_state_dict,
                    cls=TTDeviceImageJsonEncoder,
                    indent=4,
                    skipkeys=True,
                )
                f.write(device_image_state_json)

            module_files_directory = os.path.join(
                src_tti_directory_to_zip, "module_files"
            )
            os.makedirs(module_files_directory, exist_ok=True)

            for pybuda_module in device_image.modules:
                module_file = inspect.getfile(pybuda_module.__class__)
                TTIArchive._copy_module_file(
                    module_file=module_file, dst_dir=module_files_directory
                )

                TTIArchive._copy_netlist_yaml(
                    netlist_yaml=netlist_path, dst_dir=src_tti_directory_to_zip
                )
                write_buda_envs_configs(src_tti_directory_to_zip)
            TTIArchive._create_tti_archive(tmp, device_img_path)
        logger.info(
            "TTI: Saving device image took {} seconds", time.time() - start_time
        )
