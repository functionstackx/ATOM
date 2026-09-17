# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# This code is adapted from https://github.com/ROCm/vllm/blob/main/vllm/model_executor/model_loader/weight_utils.py

import fnmatch
import json
import logging
import os
import time
from glob import glob
from typing import Any, List, Optional, Union

import huggingface_hub.constants
import torch
from huggingface_hub import HfFileSystem, snapshot_download
from tqdm.auto import tqdm

from atom.utils import envs
from atom.utils.modelscope import get_lock, maybe_download_from_modelscope

logger = logging.getLogger(__name__)


def enable_hf_transfer():
    """automatically activates hf_transfer"""
    if "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ:
        try:
            # enable hf hub transfer if available
            import hf_transfer  # type: ignore # noqa

            huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER = True
        except ImportError:
            pass


enable_hf_transfer()


class DisabledTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)


def download_weights_from_hf(
    model_name_or_path: str,
    cache_dir: Optional[str],
    allow_patterns: list[str],
    revision: Optional[str] = None,
    ignore_patterns: Optional[Union[str, list[str]]] = None,
) -> str:
    """Download model weights from Hugging Face Hub, or ModelScope when enabled.

    Args:
        model_name_or_path (str): The model name or path.
        cache_dir (Optional[str]): The cache directory to store the model
            weights. If None, will use HF defaults.
        allow_patterns (list[str]): The allowed patterns for the
            weight files. Files matched by any of the patterns will be
            downloaded.
        revision (Optional[str]): The revision of the model.
        ignore_patterns (Optional[Union[str, list[str]]]): The patterns to
            filter out the weight files. Files matched by any of the patterns
            will be ignored.

    Returns:
        str: The path to the downloaded model weights.
    """
    if envs.ATOM_USE_MODELSCOPE:
        # Resolve before HfFileSystem is constructed; ModelScope-only repo IDs
        # must never be queried on Hugging Face. Include the shard index used
        # by filter_duplicate_safetensors_files.
        folder = maybe_download_from_modelscope(
            model_name_or_path,
            cache_dir=cache_dir,
            revision=revision,
            allow_patterns=[*allow_patterns, "*.safetensors.index.json"],
            ignore_patterns=ignore_patterns,
        )
        # Some ModelScope SDK versions return the snapshot directory in
        # offline mode without checking whether the requested files exist.
        # A metadata-only cache must not produce an empty weight iterator.
        if not any(glob(os.path.join(folder, pattern)) for pattern in allow_patterns):
            raise RuntimeError(
                f"No model weights matching {allow_patterns} found in {folder}. "
                "The ModelScope cache may contain only metadata; download the "
                "weights before enabling HF_HUB_OFFLINE."
            )
        return folder
    local_only = huggingface_hub.constants.HF_HUB_OFFLINE
    if not local_only:
        # Before we download we look at that is available:
        fs = HfFileSystem()
        file_list = fs.ls(model_name_or_path, detail=False, revision=revision)

        # depending on what is available we download different things
        for pattern in allow_patterns:
            matching = fnmatch.filter(file_list, pattern)
            if len(matching) > 0:
                allow_patterns = [pattern]
                break

    logger.info("Using model weights format %s", allow_patterns)
    # Use file lock to prevent multiple processes from
    # downloading the same model weights at the same time.
    with get_lock(model_name_or_path, cache_dir):
        start_time = time.perf_counter()
        hf_folder = snapshot_download(
            model_name_or_path,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            cache_dir=cache_dir,
            tqdm_class=DisabledTqdm,
            revision=revision,
            local_files_only=local_only,
        )
        time_taken = time.perf_counter() - start_time
        if time_taken > 0.5:
            logger.info(
                "Time spent downloading weights for %s: %.6f seconds",
                model_name_or_path,
                time_taken,
            )
    return hf_folder


def set_weight_attrs(
    weight: torch.Tensor,
    weight_attrs: Optional[dict[str, Any]],
):
    """Set attributes on a weight tensor.

    This method is used to set attributes on a weight tensor. This method
    will not overwrite existing attributes.

    Args:
        weight: The weight tensor.
        weight_attrs: A dictionary of attributes to set on the weight tensor.
    """
    if weight_attrs is None:
        return
    for key, value in weight_attrs.items():
        assert not hasattr(weight, key), f"Overwriting existing tensor attribute: {key}"

        # NOTE(woosuk): During weight loading, we often do something like:
        # narrowed_tensor = param.data.narrow(0, offset, len)
        # narrowed_tensor.copy_(real_weight)
        # expecting narrowed_tensor and param.data to share the same storage.
        # However, on TPUs, narrowed_tensor will lazily propagate to the base
        # tensor, which is param.data, leading to the redundant memory usage.
        # This sometimes causes OOM errors during model loading. To avoid this,
        # we sync the param tensor after its weight loader is called.
        # TODO(woosuk): Remove this hack once we have a better solution.
        setattr(weight, key, value)


def filter_duplicate_safetensors_files(
    hf_weights_files: List[str], hf_folder: str, index_file: str
) -> List[str]:
    # model.safetensors.index.json is a mapping from keys in the
    # torch state_dict to safetensors file holding that weight.
    index_file_name = os.path.join(hf_folder, index_file)
    if not os.path.isfile(index_file_name):
        return hf_weights_files

    # Iterate through the weight_map (weight_name: safetensors files)
    # to identify weights that we should use.
    with open(index_file_name) as f:
        weight_map = json.load(f)["weight_map"]
    weight_files_in_index = set()
    for weight_name in weight_map:
        weight_files_in_index.add(os.path.join(hf_folder, weight_map[weight_name]))
    # Filter out any fields that are not found in the index file.
    hf_weights_files = [f for f in hf_weights_files if f in weight_files_in_index]
    return hf_weights_files
