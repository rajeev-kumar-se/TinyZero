# TinyZero

> **⚠️ Deprecation Notice:** This repo is no longer actively maintained. For running RL experiments, please directly use the latest [veRL](https://github.com/volcengine/verl) library.
> For the archived original documentation, see [OLD_README.md](./OLD_README.md).

![image](cover.png)

TinyZero is a reproduction of [DeepSeek R1 Zero](https://github.com/deepseek-ai/DeepSeek-R1) in countdown and multiplication tasks. We built upon [veRL](https://github.com/volcengine/verl).

Through RL, the 3B base LM develops self-verification and search abilities all on its own.

You can experience the Aha moment yourself for < $30.

Twitter thread: https://x.com/jiayi_pirate/status/1882839370505621655

Full experiment log: https://wandb.ai/jiayipan/TinyZero

> 📢: We release [Adaptive Parallel Reasoning](https://github.com/Parallel-Reasoning/APR), where we explore a new dimension in scaling reasoning models.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [High-Level System Diagram](#high-level-system-diagram)
  - [Core Components](#core-components)
  - [Training Dataflow (PPO Loop)](#training-dataflow-ppo-loop)
- [Repository Structure](#repository-structure)
- [Libraries & Dependencies](#libraries--dependencies)
  - [Core ML & Deep Learning](#1-core-ml--deep-learning)
  - [Distributed Computing & Parallelism](#2-distributed-computing--parallelism)
  - [Inference Engine](#3-inference-engine)
  - [Model Hub & Tokenization](#4-model-hub--tokenization)
  - [Configuration & Experiment Tracking](#5-configuration--experiment-tracking)
  - [Data Processing](#6-data-processing)
  - [Utilities](#7-utilities)
- [Installation](#installation)
- [Countdown Task](#countdown-task)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

---

## Overview

TinyZero demonstrates that a **3B-parameter base LLM** can develop emergent reasoning capabilities—self-verification, backtracking, and search—through **pure reinforcement learning** (no supervised fine-tuning on chain-of-thought data). The project focuses on two arithmetic reasoning tasks:

| Task | Description | Reward |
|------|-------------|--------|
| **Countdown** | Given N numbers, create an equation equaling a target using +, −, ×, ÷ | Rule-based: format correctness + answer correctness |
| **Multiplication** | Multiply two numbers | Rule-based: exact match |

The RL training uses **PPO** (Proximal Policy Optimization) with optional **GRPO** (Group Relative Policy Optimization) as the advantage estimator, orchestrated through Ray-based distributed workers.

---

## Architecture

### High-Level System Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Driver Process (CPU)                            │
│                                                                        │
│  ┌──────────────┐   ┌─────────────────┐   ┌─────────────────────────┐  │
│  │  Hydra Config │──▶│  RayPPOTrainer   │──▶│  Advantage Computation  │  │
│  │  (YAML)       │   │  (Orchestrator)  │   │  (GAE / GRPO)           │  │
│  └──────────────┘   └────────┬────────┘   └─────────────────────────┘  │
│                              │                                          │
│                    ┌─────────┼─────────────────────┐                   │
│                    │  Ray RPC (Single Controller)   │                   │
│                    └─────────┼─────────────────────┘                   │
└──────────────────────────────┼──────────────────────────────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
┌──────────────────┐ ┌──────────────────┐ ┌─────────────────┐
│  ActorRollout    │ │  Critic Worker   │ │  Ref Policy     │
│  Worker (GPU)    │ │  (GPU)           │ │  Worker (GPU)   │
│                  │ │                  │ │                 │
│ ┌──────────────┐ │ │ ┌──────────────┐ │ │ ┌─────────────┐ │
│ │ Actor (FSDP) │ │ │ │ Critic Model │ │ │ │ Frozen Copy │ │
│ │ Policy Update│ │ │ │ Value Head   │ │ │ │ of Actor    │ │
│ └──────────────┘ │ │ │ (FSDP)       │ │ │ │ (FSDP)      │ │
│ ┌──────────────┐ │ │ └──────────────┘ │ │ └─────────────┘ │
│ │ Rollout      │ │ └──────────────────┘ └─────────────────┘
│ │ (vLLM)       │ │
│ │ Generation   │ │
│ └──────────────┘ │
└──────────────────┘
```

### Core Components

#### 1. Single Controller Architecture (`verl/single_controller/`)
The system uses a **Ray-based single controller** pattern where:
- A **driver process** (CPU) orchestrates the entire training loop via RPC calls.
- **Worker groups** are spawned on GPU nodes as Ray actors.
- Workers are **co-located** via `create_colocated_worker_cls()` — multiple roles (actor, rollout, ref) can share the same GPU process to maximize memory utilization.

#### 2. Hybrid Engine (`verl/workers/fsdp_workers.py`)
The `ActorRolloutRefWorker` is a unified worker that serves triple duty:
- **Actor**: The policy model being trained (FSDP-sharded, with AdamW optimizer).
- **Rollout**: Generates sequences using vLLM for fast autoregressive decoding.
- **Reference Policy**: A frozen copy of the initial model for KL divergence computation.

On each training step, the worker switches between training mode (FSDP) and inference mode (vLLM) via a custom **Sharding Manager** that handles weight synchronization between the two engines.

#### 3. PPO Training Loop (`verl/trainer/ppo/`)
The `RayPPOTrainer` implements the full PPO loop:
1. **Generate** rollouts (vLLM).
2. **Compute** reference log-probs (frozen model).
3. **Compute** value estimates (critic model).
4. **Score** responses (rule-based reward functions).
5. **Compute** advantages (GAE or GRPO).
6. **Update** critic (value loss).
7. **Update** actor (clipped policy gradient loss + entropy bonus).

#### 4. Reward System (`verl/utils/reward_score/`)
Rule-based reward functions for each task:
- **Countdown**: Validates equation format, checks that all numbers are used exactly once, evaluates arithmetic correctness.
- **Multiply**: Exact-match comparison of the model's answer to the product.
- **GSM8K / MATH**: Parse-and-compare for standard math benchmarks.

#### 5. Data Protocol (`verl/protocol.py`)
`DataProto` is the universal data exchange format — a wrapper around PyTorch `TensorDict` that supports:
- Tensor batches (prompts, responses, log-probs, values, rewards)
- Non-tensor metadata (ground truth, data source labels)
- Serialization, chunking, concatenation, reordering for distributed dispatch

### Training Dataflow (PPO Loop)

```
Data Parquet ──▶ RLHFDataset ──▶ DataLoader
                                     │
                                     ▼
                           ┌─────────────────┐
                           │  prompt batch    │
                           └────────┬────────┘
                                    │
                     ┌──────────────▼──────────────┐
                     │   ActorRollout.generate()    │
                     │   (vLLM autoregressive gen)  │
                     └──────────────┬──────────────┘
                                    │
                     ┌──────────────▼──────────────┐
                     │  RefPolicy.compute_log_prob()│
                     │  (frozen model forward pass) │
                     └──────────────┬──────────────┘
                                    │
                     ┌──────────────▼──────────────┐
                     │  Critic.compute_values()     │
                     │  (value head forward pass)   │
                     └──────────────┬──────────────┘
                                    │
                     ┌──────────────▼──────────────┐
                     │  RewardManager.__call__()    │
                     │  (rule-based scoring)        │
                     └──────────────┬──────────────┘
                                    │
                     ┌──────────────▼──────────────┐
                     │  compute_advantage()         │
                     │  (GAE or GRPO on driver CPU) │
                     └──────────────┬──────────────┘
                                    │
                ┌───────────────────┼───────────────────┐
                ▼                                       ▼
    ┌───────────────────┐                  ┌───────────────────┐
    │ Critic.update()    │                  │ Actor.update()     │
    │ (value loss)       │                  │ (PPO clipped loss) │
    └───────────────────┘                  └───────────────────┘
```

---

## Repository Structure

```
TinyZero/
├── verl/                            # Core library (fork of veRL)
│   ├── protocol.py                  # DataProto — universal data exchange format
│   ├── models/                      # Model registry & custom architectures
│   │   ├── registry.py              # Model support checking (rmpad, etc.)
│   │   ├── llama/                   # Llama-specific implementations
│   │   └── transformers/            # HuggingFace monkey-patches for rmpad/SP
│   ├── trainer/                     # Training orchestration
│   │   ├── main_ppo.py              # Main entry point (reward manager + config)
│   │   ├── fsdp_sft_trainer.py      # Supervised fine-tuning trainer
│   │   ├── main_generation.py       # Standalone generation script
│   │   ├── config/                  # Hydra YAML configs (PPO, Megatron, SFT)
│   │   └── ppo/
│   │       ├── core_algos.py        # PPO/GRPO core: GAE, policy loss, value loss, KL
│   │       └── ray_trainer.py       # RayPPOTrainer — full training loop
│   ├── workers/                     # Distributed worker implementations
│   │   ├── fsdp_workers.py          # FSDP-based Actor, Critic, RewardModel, Ref workers
│   │   ├── megatron_workers.py      # Megatron-LM based workers (alternative backend)
│   │   ├── actor/                   # Actor policy (dp_actor, megatron_actor)
│   │   ├── critic/                  # Critic value head (dp_critic, megatron_critic)
│   │   ├── rollout/                 # Sequence generation engines
│   │   │   ├── vllm_rollout/        # vLLM-based fast inference
│   │   │   └── hf_rollout.py        # HuggingFace generate() fallback
│   │   ├── reward_model/            # Neural reward model worker
│   │   └── sharding_manager/        # Weight sync between FSDP ↔ vLLM/Megatron
│   ├── single_controller/           # Ray-based orchestration layer
│   │   ├── base/                    # Abstract Worker, dispatch decorators
│   │   └── ray/                     # RayResourcePool, RayWorkerGroup, co-location
│   ├── utils/                       # Shared utilities
│   │   ├── reward_score/            # Rule-based reward functions
│   │   │   ├── countdown.py         # Countdown task scoring
│   │   │   ├── multiply.py          # Multiplication task scoring
│   │   │   ├── gsm8k.py             # GSM8K benchmark scoring
│   │   │   └── math.py              # MATH benchmark scoring
│   │   ├── fsdp_utils.py            # FSDP wrapping, offloading, checkpointing
│   │   ├── torch_functional.py      # Masked ops, entropy, log-prob utilities
│   │   ├── seqlen_balancing.py      # Load-balanced sequence partitioning
│   │   ├── tracking.py              # W&B / console metric logging
│   │   ├── dataset/                 # RLHF dataset & collation
│   │   ├── megatron/                # Megatron-LM utilities
│   │   └── ...
│   └── third_party/
│       └── vllm/                    # Vendored vLLM patches
├── examples/
│   ├── data_preprocess/             # Dataset generation scripts
│   │   ├── countdown.py             # Countdown task dataset builder
│   │   ├── multiply.py              # Multiplication dataset builder
│   │   ├── gsm8k.py                 # GSM8K preprocessing
│   │   └── ...
│   ├── ppo_trainer/                 # PPO training scripts for various models
│   ├── grpo_trainer/                # GRPO training scripts
│   ├── sft/                         # Supervised fine-tuning examples
│   └── generation/                  # Standalone inference examples
├── scripts/
│   └── train_tiny_zero.sh           # Main training launcher script
├── tests/                           # Unit & integration tests
├── docker/                          # Container definitions
├── docs/                            # Documentation source
├── pyproject.toml                   # Project metadata & dependencies (PEP 621)
├── setup.py                         # Fallback installation script
└── requirements.txt                 # Pinned dependencies
```

---

## Libraries & Dependencies

### 1. Core ML & Deep Learning

| Library | Version | Purpose |
|---------|---------|---------|
| **PyTorch** | `2.4.0` (CUDA 12.1) | Core tensor computation, autograd, model training, `FSDP` for distributed sharding, `AdamW` optimizer, mixed-precision training |
| **TensorDict** | `< 0.6` | Structured dictionary of tensors used as the backbone of `DataProto` — enables batched tensor manipulation, indexing, chunking, and serialization across the entire pipeline |
| **Flash Attention 2** | latest | High-performance fused attention kernel (`flash_attention_2`) used by all models via HuggingFace's `attn_implementation='flash_attention_2'` — critical for memory-efficient long-sequence training |
| **NumPy** | latest | Non-tensor metadata handling in `DataProto.non_tensor_batch`, reward score computation, array-based sequence balancing |

### 2. Distributed Computing & Parallelism

| Library | Version | Purpose |
|---------|---------|---------|
| **Ray** | latest | Distributed orchestration backbone — `RayResourcePool` manages GPU allocation, `RayWorkerGroup` spawns worker actors, `ray.remote` decorators enable RPC between driver and workers. Co-location support allows multiple roles on the same GPU. |
| **PyTorch FSDP** | (built-in) | `FullyShardedDataParallel` wraps actor, critic, and reference models for ZeRO-3 style parameter sharding. Supports mixed-precision, gradient checkpointing, and CPU offloading (param, grad, optimizer states). |
| **NCCL** | (built-in) | Backend for `torch.distributed` inter-GPU collective communications (all-reduce, all-gather) |
| **PyTorch DeviceMesh** | (built-in) | Manages multi-dimensional parallelism: FSDP sharding dimension + Ulysses sequence parallelism dimension + vLLM tensor parallelism dimension |

### 3. Inference Engine

| Library | Version | Purpose |
|---------|---------|---------|
| **vLLM** | `≤ 0.6.3` | High-throughput autoregressive text generation for the rollout phase. Provides PagedAttention for efficient KV-cache memory management. The `FSDPVLLMShardingManager` handles weight synchronization between the FSDP training engine and the vLLM inference engine. |
| **xFormers** | (via vLLM) | `VLLM_ATTENTION_BACKEND=XFORMERS` — attention backend for vLLM on certain GPUs |

### 4. Model Hub & Tokenization

| Library | Version | Purpose |
|---------|---------|---------|
| **HuggingFace Transformers** | `< 4.48` | Model loading via `AutoModelForCausalLM` (actor) and `AutoModelForTokenClassification` (critic value head). Provides `AutoConfig`, `AutoTokenizer`, gradient checkpointing, and `save_pretrained` for checkpoints. |
| **HuggingFace Accelerate** | latest | Utilities for device placement, mixed-precision context managers, and model loading helpers |
| **HuggingFace Datasets** | latest | Loading and preprocessing datasets (e.g., `Jiayi-Pan/Countdown-Tasks-3to4` from the Hub), `.map()` transformations, Parquet export |

### 5. Configuration & Experiment Tracking

| Library | Version | Purpose |
|---------|---------|---------|
| **Hydra** (`hydra-core`) | latest | Hierarchical YAML configuration system — `@hydra.main(config_path='config', config_name='ppo_trainer')` loads and resolves the full training config with variable interpolation (e.g., `${actor_rollout_ref.model.path}`) |
| **OmegaConf** | (via Hydra) | Structured config manipulation: `OmegaConf.resolve()`, `open_dict()`, `to_container()` for dynamic config mutation during runtime |
| **Weights & Biases** (`wandb`) | latest | Experiment tracking and metric logging — training curves, reward scores, KL divergence, MFU, timing metrics. Configured via `trainer.logger=['wandb']` |

### 6. Data Processing

| Library | Version | Purpose |
|---------|---------|---------|
| **Pandas** | latest | Parquet file I/O for training/validation datasets |
| **PyArrow** | (via datasets) | Backend for efficient columnar data storage in Parquet format |
| **tqdm** | (via datasets) | Progress bars during dataset generation |

### 7. Utilities

| Library | Version | Purpose |
|---------|---------|---------|
| **codetiming** | latest | `Timer` context manager for precise per-section timing metrics (generation, ref computation, critic update, actor update) reported to W&B |
| **dill** | latest | Extended serialization (beyond pickle) for complex Python objects sent between Ray workers |
| **pybind11** | latest | C++ extension binding support for custom CUDA kernels and optimized operations |
| **Matplotlib** | latest | Visualization of training curves and results (quality-of-life) |
| **IPython** | latest | Interactive notebook support for the tutorial (`verl_getting_started.ipynb`) |

---

## Installation

```bash
conda create -n zero python=3.9
# install torch [or you can skip this step and let vllm install the correct version for you]
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# install vllm
pip3 install vllm==0.6.3 # or you can install 0.5.4, 0.4.2 and 0.3.1
pip3 install ray

# verl
pip install -e .

# flash attention 2
pip3 install flash-attn --no-build-isolation
# quality of life
pip install wandb IPython matplotlib
```

## Countdown task

**Data Preparation**
```bash
conda activate zero
python ./examples/data_preprocess/countdown.py --local_dir {path_to_your_dataset}
```

### Run Training
```bash
conda activate zero
```

For the following code, if you see out-of-VRAM, try adding `critic.model.enable_gradient_checkpointing=True` to the script, and check out the discussion [here](https://github.com/Jiayi-Pan/TinyZero/issues/5#issuecomment-2624161643).

**Single GPU**


Works for model <= 1.5B. For Qwen2.5-0.5B base, we know it fails to learn reasoning.

```bash
export N_GPUS=1
export BASE_MODEL={path_to_your_model}
export DATA_DIR={path_to_your_dataset}
export ROLLOUT_TP_SIZE=1
export EXPERIMENT_NAME=countdown-qwen2.5-0.5b
export VLLM_ATTENTION_BACKEND=XFORMERS

bash ./scripts/train_tiny_zero.sh
```

**3B+ model**
In this case, the base model is able to develop sophisticated reasoning skills.
```bash
export N_GPUS=2
export BASE_MODEL={path_to_your_model}
export DATA_DIR={path_to_your_dataset}
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=countdown-qwen2.5-3b
export VLLM_ATTENTION_BACKEND=XFORMERS

bash ./scripts/train_tiny_zero.sh
```

### Instruct Ablation
We experiment with Qwen-2.5-3B Instruct too.
**Data Preparation**
To follow chat template, we need to reprocess the data:
```bash
conda activate zero
python examples/data_preprocess/countdown.py --template_type=qwen-instruct --local_dir={path_to_your_dataset}
```

**Training**
```bash
export N_GPUS=2
export BASE_MODEL={path_to_your_model}
export DATA_DIR={path_to_your_dataset}
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=countdown-qwen2.5-3b-instruct
export VLLM_ATTENTION_BACKEND=XFORMERS

bash ./scripts/train_tiny_zero.sh
```

## Acknowledgements
* We run our experiments based on [veRL](https://github.com/volcengine/verl).
* We use Qwen2.5 series base model [Qwen2.5](https://github.com/QwenLM/Qwen2.5).

## Citation
```
@misc{tinyzero,
author       = {Jiayi Pan and Junjie Zhang and Xingyao Wang and Lifan Yuan and Hao Peng and Alane Suhr},
title        = {TinyZero},
howpublished = {https://github.com/Jiayi-Pan/TinyZero},
note         = {Accessed: 2025-01-24},
year         = {2025}
}
```
