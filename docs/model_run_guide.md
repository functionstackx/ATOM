# Model run guide

Ready-to-use commands for serving models on ATOM with AMD Instinct MI355X / MI300X GPUs. Each model recipe below is validated in nightly CI.

## Quick start

```bash
# Pull the latest ATOM container
docker pull rocm/atom:latest

# Start the container
docker run -it --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host --shm-size=16G \
  --privileged --cap-add=SYS_PTRACE \
  -e HF_TOKEN=$HF_TOKEN \
  -p 8000:8000 \
  rocm/atom:latest
```

## Download models from ModelScope

Install the optional SDK, then enable ModelScope for native ATOM serving:

```bash
pip install 'modelscope>=1.18.1'
# From a source checkout, pip install -e '.[modelscope]' also installs the extra.
ATOM_USE_MODELSCOPE=1 python -m atom.entrypoints.openai_server \
  --model Qwen/Qwen3-0.6B
```

`ATOM_USE_MODELSCOPE` accepts `1` or `true` (case-insensitive). Hugging Face
remains the default. The existing benchmark flag `VLLM_USE_MODELSCOPE` is a
fallback alias; an explicit `ATOM_USE_MODELSCOPE` value takes precedence.
Use the same setting in the benchmark client if it loads a remote tokenizer.

Config, generation config, tokenizer, multimodal processor, custom chat encoder
and speculative-draft config reads resolve to local ModelScope snapshots without
downloading weights. The native weight loader downloads safetensors and their
shard index separately. Local paths bypass ModelScope, including directories
cloned from ModelScope with Git/LFS. Model IDs returned by the server are not
rewritten to cache paths. Remote code still requires the existing trust setting.

The SDK uses its default cache or `MODELSCOPE_CACHE`. Set `HF_HUB_OFFLINE=1`
before starting ATOM to use cached files only; missing cached files raise an
error rather than falling back to Hugging Face. A directory containing only
metadata is not treated as a complete checkpoint.

For the vLLM and SGLang plugin frontends, use their own hub settings
(`VLLM_USE_MODELSCOPE` and `SGLANG_USE_MODELSCOPE`, respectively). This option
covers ATOM's native LLM path, not its separate diffusion pipelines.

## Supported models

| Model | Type | Precision | TP | Recipe |
|-------|------|-----------|-----|--------|
| DeepSeek-R1-0528 | MoE + MLA | FP8 / MXFP4 | 8 | [recipes/DeepSeek-R1.md](https://github.com/ROCm/ATOM/blob/main/recipes/DeepSeek-R1.md) |
| GLM-5 | MoE + MLA | FP8 | 8 | [recipes/GLM-5.md](https://github.com/ROCm/ATOM/blob/main/recipes/GLM-5.md) |
| GPT-OSS-120B | MoE | FP8 | 1 | [recipes/GPT-OSS.md](https://github.com/ROCm/ATOM/blob/main/recipes/GPT-OSS.md) |
| Kimi-K2.5/K2.7 | MoE | MXFP4 | 4 | [recipes/Kimi-K2.md](https://github.com/ROCm/ATOM/blob/main/recipes/Kimi-K2.md) |
| Kimi-K2-Thinking | MoE | FP8 | 8 | [recipes/Kimi-K2-Thinking.md](https://github.com/ROCm/ATOM/blob/main/recipes/Kimi-K2-Thinking.md) |
| Qwen3-235B | MoE | FP8 | 8 | [recipes/Qwen3-235b.md](https://github.com/ROCm/ATOM/blob/main/recipes/Qwen3-235b.md) |
| Qwen3-Next | MoE | FP8 | 8 | [recipes/Qwen3-Next.md](https://github.com/ROCm/ATOM/blob/main/recipes/Qwen3-Next.md) |
| Qwen3.8-Flash-Next | MoE + GDN + QSA | BF16 | 2 × 288 GiB (TP2+EP) | [recipes/Qwen3.8-Flash-Next.md](https://github.com/ROCm/ATOM/blob/main/recipes/Qwen3.8-Flash-Next.md) |

### vLLM plugin backend

ATOM also runs as a vLLM plugin backend. See recipes under [recipes/atom_vllm/](https://github.com/ROCm/ATOM/tree/main/recipes/atom_vllm/) for vLLM-integrated serving.

## Nightly CI benchmark configurations

The nightly CI sweeps these configurations for every model:

| ISL | OSL | Concurrency Levels |
|-----|-----|--------------------|
| 1024 | 1024 | 1, 2, 4, 8, 16, 32, 64, 128, 256 |
| 8192 | 1024 | 1, 2, 4, 8, 16, 32, 64, 128, 256 |

Run a benchmark against a running ATOM server:

```bash
python -m atom.benchmarks.benchmark_serving \
  --model <model_name_or_path> \
  --backend vllm --base-url http://localhost:8000 \
  --dataset-name random \
  --random-input-len 1024 --random-output-len 1024 \
  --max-concurrency 128 --num-prompts 1280 \
  --random-range-ratio 0.8 \
  --request-rate inf --ignore-eos
```

Key parameters:
- `--random-range-ratio 0.8` — adds ±20% jitter to sequence lengths
- `--num-prompts` — typically `concurrency × 10`
- `--request-rate inf` — closed-loop benchmarking (no inter-request delay)
- `--ignore-eos` — forces full output length generation

## Live dashboard

Nightly benchmark results are published to the [ATOM Benchmark Dashboard](https://rocm.github.io/ATOM/benchmark-dashboard/).

Competitive comparison (MI355X vs B200/B300) is available on the [AI Frameworks Dashboard](https://rocm.github.io/AI-Frameworks-Dashboard/atom-benchmark/).
