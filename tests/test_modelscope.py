# SPDX-License-Identifier: MIT
"""CPU tests for ModelScope routing; no network or ModelScope SDK required."""

import importlib
import json
import sys
from types import ModuleType
from unittest.mock import Mock

import filelock
import huggingface_hub.constants
import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from atom import config as atom_config
from atom.model_loader import weight_utils
from atom.utils import envs
from atom.utils.modelscope import (
    WEIGHT_PATTERNS,
    get_lock,
    get_model_metadata_path,
    maybe_download_from_modelscope,
)


@pytest.fixture(autouse=True)
def hub_environment(monkeypatch):
    monkeypatch.delenv("ATOM_USE_MODELSCOPE", raising=False)
    monkeypatch.delenv("VLLM_USE_MODELSCOPE", raising=False)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", False)


@pytest.fixture
def ms_download(monkeypatch, tmp_path):
    monkeypatch.setenv("ATOM_USE_MODELSCOPE", "1")
    module = ModuleType("modelscope.hub.snapshot_download")
    module.snapshot_download = Mock(return_value=str(tmp_path))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module.snapshot_download


@pytest.mark.parametrize("value", ["1", "true", "TRUE", " True "])
def test_enable_values(monkeypatch, value):
    monkeypatch.setenv("ATOM_USE_MODELSCOPE", value)
    assert envs.ATOM_USE_MODELSCOPE


@pytest.mark.parametrize("value", ["0", "false", "", "yes"])
def test_disable_values(monkeypatch, value):
    monkeypatch.setenv("ATOM_USE_MODELSCOPE", value)
    assert not envs.ATOM_USE_MODELSCOPE


def test_default_and_legacy_alias(monkeypatch):
    assert not envs.ATOM_USE_MODELSCOPE
    monkeypatch.setenv("VLLM_USE_MODELSCOPE", "true")
    assert envs.ATOM_USE_MODELSCOPE
    monkeypatch.setenv("ATOM_USE_MODELSCOPE", "0")
    assert not envs.ATOM_USE_MODELSCOPE


def test_disabled_does_not_import_sdk(monkeypatch):
    monkeypatch.setitem(sys.modules, "modelscope.hub.snapshot_download", None)
    assert maybe_download_from_modelscope("org/model") == "org/model"


def test_local_directory_does_not_import_sdk(monkeypatch, tmp_path):
    monkeypatch.setenv("ATOM_USE_MODELSCOPE", "1")
    monkeypatch.setitem(sys.modules, "modelscope.hub.snapshot_download", None)
    assert get_model_metadata_path(str(tmp_path)) == str(tmp_path)


def test_missing_sdk_has_install_hint(monkeypatch):
    monkeypatch.setenv("ATOM_USE_MODELSCOPE", "1")
    monkeypatch.setitem(sys.modules, "modelscope.hub.snapshot_download", None)
    with pytest.raises(ImportError, match="pip install"):
        maybe_download_from_modelscope("org/model")


def test_download_arguments_and_lock(ms_download, tmp_path):
    cache_dir = tmp_path / "new" / "cache"
    maybe_download_from_modelscope(
        "org/model",
        cache_dir=str(cache_dir),
        revision="release-1",
        allow_patterns=["*.safetensors"],
        ignore_patterns=["original/*"],
    )
    ms_download.assert_called_once_with(
        model_id="org/model",
        cache_dir=str(cache_dir),
        revision="release-1",
        allow_patterns=["*.safetensors"],
        ignore_file_pattern=["original/*"],
        local_files_only=False,
    )
    assert list(cache_dir.glob("*.lock"))
    assert (
        get_lock("org/model", str(cache_dir)).lock_file
        == get_lock("org/model", str(cache_dir)).lock_file
    )
    # Keep the pre-existing ATOM lock naming scheme, and verify exclusion.
    with get_lock("org/model", str(cache_dir)), pytest.raises(filelock.Timeout):
        get_lock("org/model", str(cache_dir)).acquire(timeout=0)


@pytest.mark.parametrize(
    "offline,explicit", [(True, None), (True, False), (False, True)]
)
def test_offline_forwarded(ms_download, monkeypatch, offline, explicit):
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", offline)
    maybe_download_from_modelscope("org/model", local_files_only=explicit)
    assert ms_download.call_args.kwargs["local_files_only"] is True


def test_metadata_excludes_weights(ms_download):
    get_model_metadata_path("org/model")
    assert ms_download.call_args.kwargs["ignore_file_pattern"] == WEIGHT_PATTERNS
    assert ms_download.call_args.kwargs["allow_patterns"] is None


def test_partial_cache_still_resolves_weights(ms_download, tmp_path, monkeypatch):
    (tmp_path / "org" / "model").mkdir(parents=True)
    monkeypatch.setenv("MODELSCOPE_CACHE", str(tmp_path))
    get_model_metadata_path("org/model")
    maybe_download_from_modelscope("org/model", allow_patterns=["*.safetensors"])
    assert ms_download.call_count == 2
    assert ms_download.call_args.kwargs["allow_patterns"] == ["*.safetensors"]


def test_failures_do_not_fall_back_to_hf(ms_download, monkeypatch):
    hf = Mock(side_effect=AssertionError("Hugging Face must not be called"))
    monkeypatch.setattr(weight_utils, "HfFileSystem", hf)
    monkeypatch.setattr(weight_utils, "snapshot_download", hf)
    ms_download.side_effect = FileNotFoundError("offline cache miss")
    with pytest.raises(FileNotFoundError, match="offline cache miss"):
        weight_utils.download_weights_from_hf("org/model", None, ["*.safetensors"])
    hf.assert_not_called()


def test_weight_download_routes_before_hf(ms_download, monkeypatch, tmp_path):
    (tmp_path / "model.safetensors").touch()
    hf = Mock(side_effect=AssertionError("Hugging Face must not be called"))
    monkeypatch.setattr(weight_utils, "HfFileSystem", hf)
    monkeypatch.setattr(weight_utils, "snapshot_download", hf)
    assert weight_utils.download_weights_from_hf(
        "org/model", None, ["*.safetensors"], "release-1", ["original/*"]
    ) == str(tmp_path)
    assert ms_download.call_args.kwargs["allow_patterns"] == [
        "*.safetensors",
        "*.safetensors.index.json",
    ]
    assert ms_download.call_args.kwargs["revision"] == "release-1"
    assert ms_download.call_args.kwargs["ignore_file_pattern"] == ["original/*"]
    hf.assert_not_called()


def test_metadata_only_weight_cache_is_rejected(ms_download, tmp_path, monkeypatch):
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(RuntimeError, match="cache may contain only metadata"):
        weight_utils.download_weights_from_hf("org/model", None, ["*.safetensors"])


def test_hugging_face_default_unchanged(monkeypatch, tmp_path):
    fs = Mock()
    fs.ls.return_value = ["org/model/model.safetensors"]
    monkeypatch.setattr(weight_utils, "HfFileSystem", Mock(return_value=fs))
    download = Mock(return_value=str(tmp_path))
    monkeypatch.setattr(weight_utils, "snapshot_download", download)
    assert weight_utils.download_weights_from_hf(
        "org/model", None, ["*.safetensors"], revision="main"
    ) == str(tmp_path)
    fs.ls.assert_called_once_with("org/model", detail=False, revision="main")
    assert download.call_args.kwargs["revision"] == "main"


@pytest.fixture
def metadata_repo(ms_download, tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "vocab_size": 2,
            }
        )
    )
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": 1}))
    backend = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
    PreTrainedTokenizerFast(tokenizer_object=backend).save_pretrained(tmp_path)
    return tmp_path


def test_config_generation_and_draft_read_local_snapshot(metadata_repo):
    config = atom_config.get_hf_config("modelscope-only/model")
    assert config.hidden_size == 16
    assert atom_config.get_generation_config("modelscope-only/model").eos_token_id == 1
    # Use a registry-independent config format supported by this ATOM revision.
    draft_type = next(iter(atom_config._PLAIN_CONFIG_MODEL_TYPES))
    (metadata_repo / "config.json").write_text(
        json.dumps({"model_type": draft_type, "hidden_size": 32})
    )
    assert atom_config.get_hf_config("draft-only/model").hidden_size == 32


def test_native_config_preserves_repo_id_and_skips_weights(
    metadata_repo, ms_download, monkeypatch
):
    # Quantization setup requires AITER; model/config loading itself does not.
    monkeypatch.setattr(atom_config, "QuantizationConfig", Mock())
    config = atom_config.Config("modelscope-only/model", load_dummy="empty")
    assert config.model == "modelscope-only/model"
    assert config.hf_config.hidden_size == 16
    assert all(
        call.kwargs["ignore_file_pattern"] == WEIGHT_PATTERNS
        for call in ms_download.call_args_list
    )


def test_separate_speculative_config(metadata_repo, monkeypatch):
    # Acceptance scheduling imports a Triton kernel unrelated to hub routing.
    monkeypatch.setattr(
        atom_config.SpeculativeConfig,
        "_resolve_synthetic_acceptance",
        lambda self: None,
    )
    config = atom_config.SpeculativeConfig(
        model="draft-only/model", method="eagle3", num_speculative_tokens=1
    )
    assert config.model == "draft-only/model"
    assert config.draft_model_hf_config.hidden_size == 16


def test_native_tokenizer_resolves_both_loaders(metadata_repo, monkeypatch):
    from atom.model_engine import llm_engine

    # The tiny test tokenizer cannot round-trip the native probe, so this
    # exercises both AutoTokenizer and the PreTrainedTokenizerFast fallback.
    fallback = Mock(wraps=PreTrainedTokenizerFast.from_pretrained)
    monkeypatch.setattr(llm_engine.PreTrainedTokenizerFast, "from_pretrained", fallback)
    tokenizer = llm_engine._load_tokenizer("modelscope-only/model")
    assert tokenizer.encode("hello") == [1]
    fallback.assert_called_with(str(metadata_repo))
    assert all(call.args[0] == str(metadata_repo) for call in fallback.call_args_list)


def test_benchmark_tokenizer_uses_local_snapshot(metadata_repo):
    from atom.benchmarks.backend_request_func import get_tokenizer

    tokenizer = get_tokenizer("modelscope-only/model")
    assert tokenizer.encode("hello") == [1]


def test_chat_encoder_resolves_modelscope(metadata_repo):
    from atom.entrypoints.openai.chat_encoders import _resolve_model_path

    assert _resolve_model_path("modelscope-only/model") == str(metadata_repo)


def test_real_weight_iterator_uses_index(ms_download, tmp_path, monkeypatch):
    from atom.model_loader.weight_iterator import safetensors_weights_iterator

    monkeypatch.setenv("ATOM_LOADER_PREFETCH", "0")
    save_file({"layer.weight": torch.ones(2)}, tmp_path / "model.safetensors")
    save_file({"unwanted.weight": torch.zeros(2)}, tmp_path / "extra.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.weight": "model.safetensors"}})
    )
    result = dict(safetensors_weights_iterator("modelscope-only/model"))
    assert list(result) == ["layer.weight"]
    torch.testing.assert_close(result["layer.weight"], torch.ones(2))


def test_sdk_can_be_absent_during_module_import(monkeypatch):
    monkeypatch.setitem(sys.modules, "modelscope.hub.snapshot_download", None)
    import atom.utils.modelscope as module

    importlib.reload(module)
