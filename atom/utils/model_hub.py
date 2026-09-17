# SPDX-License-Identifier: MIT
"""Backend-neutral model loading.

Callers use these wrappers without checking hub environment variables. Backend
selection, optional SDK imports and hub-specific cache behavior live here.
Hugging Face metadata loading remains delegated to Transformers.
"""

import fnmatch
import hashlib
import logging
import os
import tempfile
import time
from glob import glob
from pathlib import Path

import filelock
import huggingface_hub.constants
from huggingface_hub import HfFileSystem, hf_hub_download, snapshot_download
from tqdm.auto import tqdm

from atom.utils import envs

logger = logging.getLogger(__name__)
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


class _DisabledTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)


class _HuggingFaceHub:
    @staticmethod
    def metadata(model, **kwargs):
        # Preserve Transformers' own config/tokenizer discovery and caching.
        return model

    @staticmethod
    def snapshot(model, **kwargs):
        kwargs["local_files_only"] = huggingface_hub.constants.HF_HUB_OFFLINE or bool(
            kwargs.get("local_files_only")
        )
        return snapshot_download(model, **kwargs)

    def weights(self, model, cache_dir, allow_patterns, revision, ignore_patterns):
        if "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ:
            try:
                import hf_transfer  # type: ignore # noqa: F401

                huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER = True
            except ImportError:
                pass
        if not huggingface_hub.constants.HF_HUB_OFFLINE:
            files = HfFileSystem().ls(model, detail=False, revision=revision)
            for pattern in allow_patterns:
                if fnmatch.filter(files, pattern):
                    allow_patterns = [pattern]
                    break
        logger.info("Using model weights format %s", allow_patterns)
        with get_lock(model, cache_dir):
            start = time.perf_counter()
            folder = self.snapshot(
                model,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                cache_dir=cache_dir,
                tqdm_class=_DisabledTqdm,
                revision=revision,
            )
            elapsed = time.perf_counter() - start
            if elapsed > 0.5:
                logger.info(
                    "Time spent downloading weights for %s: %.6f seconds",
                    model,
                    elapsed,
                )
        return folder

    @staticmethod
    def file(model, filename, **kwargs):
        kwargs["local_files_only"] = huggingface_hub.constants.HF_HUB_OFFLINE or bool(
            kwargs.get("local_files_only")
        )
        return hf_hub_download(model, filename, **kwargs)

    @staticmethod
    def cached_path(model):
        # Custom encoder discovery historically probes the HF cache only.
        try:
            return snapshot_download(model, local_files_only=True, allow_patterns=[])
        except Exception:  # noqa: BLE001
            # Optional cache discovery must not fail startup.
            return model


class _ModelScopeHub:
    @staticmethod
    def snapshot(
        model,
        cache_dir=None,
        revision=None,
        allow_patterns=None,
        ignore_patterns=None,
        local_files_only=None,
    ):
        try:
            from modelscope.hub.snapshot_download import snapshot_download
        except ImportError as exc:
            raise ImportError(
                "ModelScope support requires the optional dependency. "
                "Install it with `pip install 'modelscope>=1.18.1'` "
                "or `pip install '.[modelscope]'` from the ATOM checkout."
            ) from exc
        with get_lock(model, cache_dir):
            return snapshot_download(
                model_id=model,
                cache_dir=cache_dir,
                revision=revision,
                allow_patterns=allow_patterns,
                ignore_file_pattern=ignore_patterns,
                local_files_only=(
                    huggingface_hub.constants.HF_HUB_OFFLINE or bool(local_files_only)
                ),
            )

    def metadata(self, model, **kwargs):
        return self.snapshot(model, ignore_patterns=WEIGHT_PATTERNS, **kwargs)

    def weights(self, model, cache_dir, allow_patterns, revision, ignore_patterns):
        folder = self.snapshot(
            model,
            cache_dir=cache_dir,
            revision=revision,
            allow_patterns=[*allow_patterns, "*.safetensors.index.json"],
            ignore_patterns=ignore_patterns,
        )
        # The SDK can return a metadata-only snapshot during an offline read.
        if not any(glob(os.path.join(folder, pattern)) for pattern in allow_patterns):
            raise RuntimeError(
                f"No model weights matching {allow_patterns} found in {folder}. "
                "The ModelScope cache may contain only metadata; download the "
                "weights before enabling HF_HUB_OFFLINE."
            )
        return folder

    def file(self, model, filename, **kwargs):
        folder = self.snapshot(model, allow_patterns=[filename], **kwargs)
        path = os.path.join(folder, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{filename} is missing from snapshot {folder}")
        return path

    def cached_path(self, model):
        # ModelScope-only repositories need their metadata snapshot resolved.
        return self.metadata(model)


_HF_HUB = _HuggingFaceHub()
_MS_HUB = _ModelScopeHub()


def _get_backend():
    """The only backend-selection branch used by native model loading."""
    return _MS_HUB if envs.ATOM_USE_MODELSCOPE else _HF_HUB


def get_model_metadata_path(model: str, **kwargs) -> str:
    """Resolve metadata without downloading tensors; preserve local paths."""
    if os.path.exists(model):
        return model
    return _get_backend().metadata(model, **kwargs)


def download_model_snapshot(model: str, **kwargs) -> str:
    """Download selected files through the configured hub."""
    if os.path.exists(model):
        return model
    return _get_backend().snapshot(model, **kwargs)


def download_model_weights(
    model: str,
    cache_dir: str | None,
    allow_patterns: list[str],
    revision: str | None = None,
    ignore_patterns: str | list[str] | None = None,
) -> str:
    """Resolve model weights through the selected hub, including its cache policy."""
    if os.path.isdir(model):
        return model
    return _get_backend().weights(
        model, cache_dir, allow_patterns, revision, ignore_patterns
    )


def get_model_file(model: str, filename: str, **kwargs) -> str:
    """Resolve one model file without fetching checkpoint tensors."""
    if os.path.isdir(model):
        return os.path.join(model, filename)
    return _get_backend().file(model, filename, **kwargs)


def get_cached_model_path(model: str) -> str:
    """Find the snapshot used for optional custom-encoder discovery."""
    if os.path.isdir(model):
        return model
    return _get_backend().cached_path(model)
