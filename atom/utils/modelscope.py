# SPDX-License-Identifier: MIT
"""Optional ModelScope downloads, shared by serving and benchmarking."""

import hashlib
import os
import tempfile
from pathlib import Path

import filelock
import huggingface_hub.constants

from atom.utils import envs

# Config/tokenizer/processor discovery must not download checkpoint tensors.
WEIGHT_PATTERNS = [
    "*.pt",
    "*.pth",
    "*.bin",
    "*.safetensors",
    "*.ckpt",
    "*.h5",
    "*.msgpack",
    "*.gguf",
    "*.onnx",
]


def get_lock(model_name_or_path: str | Path, cache_dir: str | None = None):
    lock_dir = cache_dir or tempfile.gettempdir()
    os.makedirs(lock_dir, exist_ok=True)
    model_name = str(model_name_or_path).replace("/", "-")
    hash_name = hashlib.sha256(model_name.encode()).hexdigest()
    return filelock.FileLock(
        os.path.join(lock_dir, hash_name + model_name + ".lock"), mode=0o666
    )


def maybe_download_from_modelscope(
    model: str,
    cache_dir: str | None = None,
    revision: str | None = None,
    allow_patterns: str | list[str] | None = None,
    ignore_patterns: str | list[str] | None = None,
    local_files_only: bool | None = None,
) -> str:
    """Resolve a remote repo to a local snapshot when ModelScope is enabled.

    Like vLLM/SGLang, import the optional SDK only for remote ModelScope
    requests. Leave local paths and the default Hugging Face path unchanged.
    Always ask the SDK to resolve remote IDs, even when a cache directory
    exists: it may contain only metadata or a different revision.
    """
    if not envs.ATOM_USE_MODELSCOPE or os.path.exists(model):
        return model
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "ModelScope support requires the optional dependency. "
            "Install it with `pip install 'modelscope>=1.18.1'` "
            "or `pip install '.[modelscope]'` from the ATOM checkout."
        ) from exc

    offline = huggingface_hub.constants.HF_HUB_OFFLINE
    with get_lock(model, cache_dir):
        return snapshot_download(
            model_id=model,
            cache_dir=cache_dir,
            revision=revision,
            allow_patterns=allow_patterns,
            ignore_file_pattern=ignore_patterns,
            local_files_only=offline or bool(local_files_only),
        )


def get_model_metadata_path(model: str, **kwargs) -> str:
    """Resolve config, tokenizer, processor and remote-code files, not weights."""
    return maybe_download_from_modelscope(
        model, ignore_patterns=WEIGHT_PATTERNS, **kwargs
    )
