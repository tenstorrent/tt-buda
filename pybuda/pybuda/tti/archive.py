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
from typing import Dict, List, Optional, Union
from pybuda.optimizers import Optimizer
from pybuda.backend import BackendAPI
from pybuda._C.backend_api import (
    BackendType,
    PytorchTensorDesc,
    TilizedTensorDesc,
    binarize_tensor,
    debinarize_tensor,
    tilize_tensor,
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
            # Calculate device image path based on the current test
            img_path_to_hash = get_current_pytest()
            # TT images are different for different device configurations, hence try
            # to get device-config argument to incorporate it as a part of the hash
            if "PYTEST_CURRENT_TEST" in os.environ:
                cnt = 0
                for argv in sys.argv:
                    if argv.startswith("--device-config"):
                        img_path_to_hash = f"{img_path_to_hash}_{sys.argv[cnt+1]}"
                        break
                    cnt = cnt + 1

            DEFAULT_DEVICE_PATH = (
                f"device_images/tt_{generate_hash(img_path_to_hash)}.tti"
            )
            device_img_path = DEFAULT_DEVICE_PATH
        return device_img_path

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
    def construct_device_image(unzipped_tti_directory: str) -> "TTDeviceImage":
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

            sys.path.append(
                "."
            )  # We need this line because the tvm->python code path does and pickle requires a match
            device_image.modules = TTIArchive.get_instantiate_modules(
                device_image.module_name_to_metadata, unzipped_tti_directory
            )
            netlist_file_basename = os.path.basename(
                device_image.compiled_graph_state.netlist_filename
            )
            device_image.compiled_graph_state.netlist_filename = os.path.join(
                unzipped_tti_directory, netlist_file_basename
            )

        return device_image

    @staticmethod
    def load_from_disk(tti_file_path: str) -> "TTDeviceImage":
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
        if contains_matching_checksum(tti_checksum):
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

        device_image = TTIArchive.construct_device_image(unzipped_tti_directory)
        device_image.compiler_cfg.backend_output_dir = os.path.join(
            absolute_device_image_directory,
            device_image.compiler_cfg.backend_output_dir,
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
        if device_image.compiler_cfg.backend_cluster_descriptor_path:
            device_image.compiler_cfg.backend_cluster_descriptor_path = os.path.join(
                absolute_device_image_directory,
                device_image.compiler_cfg.backend_cluster_descriptor_path
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
