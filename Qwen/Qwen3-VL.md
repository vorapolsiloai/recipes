# Qwen3-VL Usage Guide
[Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) is the most powerful vision-language model in the Qwen series to date created by Alibaba Cloud. 

This generation delivers comprehensive upgrades across the board: superior text understanding & generation, deeper visual perception & reasoning, extended context length, enhanced spatial and video dynamics comprehension, and stronger agent interaction capabilities.

Available in Dense and MoE architectures that scale from edge to cloud, with Instruct and reasoning‑enhanced Thinking editions for flexible, on‑demand deployment.


## Installing vLLM

```bash
uv venv
source .venv/bin/activate

# Install vLLM >=0.11.0
uv pip install -U vllm

# Install Qwen-VL utility library (recommended for offline inference)
uv pip install qwen-vl-utils==0.0.14
```


## Running Qwen3-VL


### Qwen3-VL-235B-A22B-Instruct
This is the Qwen3-VL flagship MoE model, which requires a minimum of 8 GPUs, each with at least 80 GB of memory (e.g., A100, H100, or H200). On some types of hardware the model may not launch successfully with its default setting. Recommended approaches by hardware type are:

- **H100 with `fp8`**: Use FP8 checkpoint for optimal memory efficiency.
- **A100 & H100 with `bfloat16`**: Either reduce `--max-model-len` or restrict inference to images only.
- **H200 & B200**: Run the model out of the box, supporting full context length and concurrent image and video processing.

See sections below for detailed launch arguments for each configuration. We are actively working on optimizations and the recommended ways to launch the model will be updated accordingly.

<details>
<summary>H100 (Image + Video Inference, FP8)</summary>

```bash
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 \
  --tensor-parallel-size 8 \
  --mm-encoder-tp-mode data \
  --enable-expert-parallel \
  --async-scheduling
```

</details>


<details>
<summary>H100 (Image Inference, FP8, TP4)</summary>

```bash
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 \
  --tensor-parallel-size 4 \
  --limit-mm-per-prompt.video 0 \
  --async-scheduling \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 128
```

</details>


<details>
<summary>A100 & H100 (Image Inference, BF16)</summary>

```bash
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct \
  --tensor-parallel-size 8 \
  --limit-mm-per-prompt.video 0 \
  --async-scheduling
```

</details>


<details>
<summary>A100 & H100 (Image + Video Inference, BF16)</summary>

```bash
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct \
  --tensor-parallel-size 8 \
  --max-model-len 128000 \
  --async-scheduling
```

</details>


<details>
<summary>H200 & B200</summary>

```bash
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct \
  --tensor-parallel-size 8 \
  --mm-encoder-tp-mode data \
  --async-scheduling
```

</details>

> ℹ️ **Note**  
> Qwen3-VL-235B-A22B-Instruct also excels on text-only tasks, ranking as the [#1 open model on text by lmarena.ai](https://x.com/arena/status/1973151703563460942) at the time this guide was created.  
> You can enable text-only mode by passing `--limit-mm-per-prompt.video 0 --limit-mm-per-prompt.image 0`, which skips the vision encoder and multimodal profiling to free up memory for additional KV cache.


### Configuration Tips
- It's highly recommended to specify `--limit-mm-per-prompt.video 0` if your inference server will only process image inputs since enabling video inputs consumes more memory reserved for long video embeddings. Alternatively, you can skip memory profiling for multimodal inputs by `--skip-mm-profiling` and lower `--gpu-memory-utilization` accordingly at your own risk.
- To avoid undesirable CPU contention, it's recommended to limit the number of threads allocated to preprocessing by setting the environment variable `OMP_NUM_THREADS=1`. This is particulaly useful and shows significant throughput improvement when deploying multiple vLLM instances on the same host.
- You can set `--max-model-len` to preserve memory. By default the model's context length is 262K, but `--max-model-len 128000` is good for most scenarios.
- Specifying `--async-scheduling` improves the overall system performance by overlapping scheduling overhead with the decoding process. **Note: With vLLM >= 0.11.1, compatibility has been improved for structured output and sampling with penalties, but it may still be incompatible with speculative decoding (features merged but not yet released).** Check the latest releases for continued improvements.
- Specifying `--mm-encoder-tp-mode data` deploys the vision encoder in a data-parallel fashion for better performance. This is because the vision encoder is very small, thus tensor parallelism brings little gain but incurs significant communication overhead. Enabling this feature does consume additional memory and may require adjustment on `--gpu-memory-utilization`.
- If your workload involves mostly **unique** multimodal inputs only, it is recommended to pass `--mm-processor-cache-gb 0` to avoid caching overhead. Otherwise, specifying `--mm-processor-cache-type shm` enables this experimental feature which utilizes host shared memory to cache preprocessed input images and/or videos which shows better performance at a high TP setting.
- vLLM supports Expert Parallelism (EP) via `--enable-expert-parallel`, which allows experts in MoE models to be deployed on separate GPUs for better throughput. Check out [Expert Parallelism Deployment](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment.html) for more details.
- You can use [benchmark_moe](https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py) to perform MoE Triton kernel tuning for your hardware.
- You can further extend the model's context window with `YaRN` by passing `--rope-scaling '{"rope_type":"yarn","factor":3.0,"original_max_position_embeddings": 262144,"mrope_section":[24,20,20],"mrope_interleaved": true}' --max-model-len 1000000`


### Benchmark on VisionArena-Chat Dataset

Once the server for the `Qwen3-VL-235B-A22B-Instruct` model is running, open another terminal and run the benchmark client:

```bash
vllm bench serve \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --model Qwen/Qwen3-VL-235B-A22B-Instruct \
  --dataset-name hf \
  --dataset-path lmarena-ai/VisionArena-Chat \
  --num-prompts 1000 \
  --request-rate 20
```

### Consume the OpenAI API Compatible Server
```python
import time
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
    timeout=3600
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://ofasys-multimodal-wlcb-3-toshanghai.oss-accelerate.aliyuncs.com/wpf272043/keepme/image/receipt.png"
                }
            },
            {
                "type": "text",
                "text": "Read all the text in the image."
            }
        ]
    }
]

start = time.time()
response = client.chat.completions.create(
    model="Qwen/Qwen3-VL-235B-A22B-Instruct",
    messages=messages,
    max_tokens=2048
)
print(f"Response costs: {time.time() - start:.2f}s")
print(f"Generated text: {response.choices[0].message.content}")
```

For more usage examples, check out the [vLLM user guide for multimodal models](https://docs.vllm.ai/en/latest/features/multimodal_inputs.html) and the [official Qwen3-VL GitHub Repository](https://github.com/QwenLM/Qwen3-VL)!



## AMD GPU Support
Recommended approaches by hardware type are:


MI300X/MI325X/MI355X 

Please follow the steps here to install and run Qwen3-VL models on AMD MI300X/MI325X/MI355X GPU.

### Step 1: Installing vLLM (AMD ROCm Backend: MI300X, MI325X, MI355X) 
 > Note: The vLLM wheel for ROCm requires Python 3.12, ROCm 7.0, and glibc >= 2.35. If your environment does not meet these requirements, please use the Docker-based setup as described in the [documentation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/#pre-built-images).  
 ```bash 
 uv venv 
 source .venv/bin/activate 
 uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/
 ```



### Step 2: Start the vLLM server

Run the vllm online serving

#### Inside the working dir, create a new directory named `miopen` .
```shell
mkdir "$(pwd)/miopen"
```

### BF16 


```shell
MIOPEN_USER_DB_PATH="$(pwd)/miopen" \
MIOPEN_FIND_MODE=FAST \
VLLM_ROCM_USE_AITER=1 \
SAFETENSORS_FAST_GPU=1 \
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct \
--tensor-parallel 4 \
--mm-encoder-tp-mode data 
```

### FP8 

```shell

MIOPEN_USER_DB_PATH="$(pwd)/miopen" \
MIOPEN_FIND_MODE=FAST \
VLLM_USE_V1=1 \
VLLM_ROCM_USE_AITER=1 \
SAFETENSORS_FAST_GPU=1 \
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 \
--tensor-parallel  4 \
--mm-encoder-tp-mode "data" 

```
### Step 3: Run Benchmark
```shell
 vllm bench serve \
  --model Qwen/Qwen3-VL-235B-A22B-Instruct \
  --dataset-name random \
  --random-input-len 8192 \
  --random-output-len 1024 \
  --request-rate 10000 \
  --num-prompts 16 \
  --ignore-eos 
```

### EAGLE3 Speculative Decoding (MI300X, FP8)

Speculative decoding lowers time-per-output-token (TPOT) by drafting several
tokens per step with a small draft model and verifying them in a single target
forward pass. This configuration is verified on **AMD Instinct MI300X**
(gfx942 / CDNA3) with the FP8 target and the public EAGLE3 draft:

| Role | Hugging Face model id |
|------|------------------------|
| **Target** | `Qwen/Qwen3-VL-235B-A22B-Instruct-FP8` |
| **Draft (EAGLE3)** | `RedHatAI/Qwen3-VL-235B-A22B-Instruct-speculator.eagle3` |

> ⚠️ **Required on ROCm: `--attention-backend ROCM_AITER_FA`.**
> On vLLM v0.21+ the default ROCm attention selection picks `ROCM_ATTN`, whose
> HIP paged-decode kernel falls back to Triton for Qwen3-235B's KV head sizes.
> The speculative path runs the draft + verify forwards every decode step, so
> this penalty is amplified and SD can end up **slower** than the baseline.
> Selecting AITER FlashAttention (`ROCM_AITER_FA`) restores the fast path and
> roughly **halves SD TPOT**. Tracking:
> [vllm-project/vllm#46596](https://github.com/vllm-project/vllm/issues/46596).

#### Start the server (FP8 target + EAGLE3 draft, TP8)

Use the [`vllm/vllm-openai-rocm`](https://hub.docker.com/r/vllm/vllm-openai-rocm)
image (v0.21.0+; v0.23.0 recommended) or the ROCm wheel from Step 1. Launch on a
single 8-GPU MI300X node:

```shell
export VLLM_USE_V1="1"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_ROCM_USE_AITER="1"
export VLLM_ROCM_USE_AITER_MHA="1"
export VLLM_ROCM_USE_AITER_RMSNORM="0"
export VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT="0"   # keep 0 for the speculative-decoding path
export VLLM_RPC_TIMEOUT="300000"

vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.94 \
  --distributed-executor-backend mp \
  --enable-chunked-prefill \
  --max-model-len 16384 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 8192 \
  --mm-encoder-tp-mode data \
  --enable-expert-parallel \
  --attention-backend ROCM_AITER_FA \
  --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["-rms_norm"],"pass_config":{"fuse_norm_quant":false}}' \
  --speculative-config '{"model":"RedHatAI/Qwen3-VL-235B-A22B-Instruct-speculator.eagle3","method":"eagle3","num_speculative_tokens":4}'
```

During generation the server logs periodic
`SpecDecoding metrics: Mean acceptance length: <N>` lines — the average number
of tokens accepted per step (higher is better; ~3.0 on real multimodal prompts,
lower on synthetic data).

**Tuning notes**
- `--attention-backend ROCM_AITER_FA` is the single most important flag for SD on
  ROCm — do not omit it (see the warning above).
- `num_speculative_tokens: 4` is a good default for this EAGLE3 draft; lower it if
  acceptance is poor on your workload.
- Keep `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=0` for speculative decoding (the `=1`
  shuffled layout is a dense-model AITER tuning knob, not used here).
- SD helps most at low–moderate concurrency; at very high concurrency the target
  is already compute-bound and the SD benefit shrinks.
- For image-only serving add `--limit-mm-per-prompt '{"image":1,"video":0}'`.

#### Run the benchmark

From a separate terminal, with the **same** target id as `--model` (the draft is
server-side only):

```shell
vllm bench serve \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --model Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 \
  --dataset-name random-mm \
  --num-prompts 1000 \
  --num-warmups 10 \
  --max-concurrency 8 \
  --random-input-len 1024 \
  --random-output-len 512 \
  --random-mm-base-items-per-request 1 \
  --random-mm-limit-mm-per-prompt '{"image": 1, "video": 0}' \
  --random-mm-bucket-config '{(512, 512, 1): 1.0}' \
  --ignore-eos
```

> ℹ️ `random-mm` uses random tokens, giving a **low** acceptance length (~1.8)
> that understates SD's benefit. For representative numbers, benchmark with real
> multimodal prompts (e.g. a VisionArena/MMMU-style dataset), where acceptance
> length reaches ~3.0.

#### Measured impact (MI300X, FP8, EAGLE3, real MMMU prompts)

Effect of `--attention-backend ROCM_AITER_FA` vs the default `ROCM_ATTN`
selection (same config otherwise) — SD decode TPOT (ms) / output throughput (tok/s):

| concurrency | default `ROCM_ATTN` | + `ROCM_AITER_FA` |
|----|----|----|
| 1  | 10.65 ms / 89 tok/s  | **5.31 ms / 197 tok/s** |
| 4  | 13.89 ms / 225 tok/s | **6.60 ms / 596 tok/s** |
| 8  | 18.64 ms / 393 tok/s | **8.24 ms / 798 tok/s** |
| 16 | 24.67 ms / 399 tok/s | **11.12 ms / 1023 tok/s** |

Acceptance length is unchanged by the attention backend — the speedup is pure
attention-execution efficiency.

### EAGLE3 Speculative Decoding (MI350X / MI355X, FP8)

The same target + EAGLE3 draft re-tuned on **AMD Instinct MI350X / MI355X**
(gfx950 / CDNA4, 288 GB/GPU). Two things shift the optimum relative to MI300X:

- **Deeper draft.** `num_speculative_tokens: 5` (vs 4 on MI300X) gives the best
  TPOT — CDNA4's faster attention makes the extra draft/verify work pay off.
- **More KV-cache headroom.** With 288 GB/GPU the engine reports ~300× the
  concurrency needed for a 16 384-token window, so running batches can scale well
  past `--max-num-seqs 32`. Use `--max-num-seqs 64 --max-num-batched-tokens 16384`
  to ride that headroom at high concurrency.

`--attention-backend ROCM_AITER_FA` remains mandatory
([vllm-project/vllm#46596](https://github.com/vllm-project/vllm/issues/46596)).

#### Start the server (FP8 target + EAGLE3 draft, TP8)

```shell
export VLLM_USE_V1="1"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_ROCM_USE_AITER="1"
export VLLM_ROCM_USE_AITER_MHA="1"
export VLLM_ROCM_USE_AITER_RMSNORM="0"
export VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT="0"
export VLLM_RPC_TIMEOUT="300000"

vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.94 \
  --distributed-executor-backend mp \
  --enable-chunked-prefill \
  --max-model-len 16384 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 16384 \
  --mm-encoder-tp-mode data \
  --enable-expert-parallel \
  --attention-backend ROCM_AITER_FA \
  --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["-rms_norm"],"pass_config":{"fuse_norm_quant":false}}' \
  --speculative-config '{"model":"RedHatAI/Qwen3-VL-235B-A22B-Instruct-speculator.eagle3","method":"eagle3","num_speculative_tokens":5}'
```

#### Measured impact (MI350X, FP8, EAGLE3, padded-ISL MMMU, ISL 1024 / OSL 512)

EAGLE3 SD (`num_speculative_tokens: 5`) vs the **same target with speculative
decoding off** on MI350X. Decode TPOT (ms) / output throughput (tok/s):

| concurrency | baseline (no SD) | + EAGLE3 SD | TPOT speedup |
|----|----|----|----|
| 1  | 11.26 ms / 87 tok/s   | **5.34 ms / 182 tok/s**  | 2.11× |
| 4  | 11.81 ms / 327 tok/s  | **8.76 ms / 435 tok/s**  | 1.35× |
| 8  | 12.54 ms / 611 tok/s  | **9.82 ms / 773 tok/s**  | 1.28× |
| 16 | 15.47 ms / 999 tok/s  | **11.10 ms / 1379 tok/s** | 1.39× |

With the larger batch config (`--max-num-seqs 64 --max-num-batched-tokens
16384`) SD keeps scaling on the 288 GB GPUs — concurrency 32: **2352 tok/s**
(12.76 ms), concurrency 64: **3656 tok/s** (15.29 ms), both ahead of the no-SD
baseline (1890 / 3348 tok/s). Mean acceptance length on these real multimodal
prompts is ~2.3–2.4 (single-stream ~3.0).


  
