## Introduction
`Qwen3.Ink.Cpp` is a study-oriented repository that reproduces the **Qwen3-8B** model using pure C++.  

This project integrates quantization methods and optimized QGEMM(Quantized GEMM) kernels in [GGML](https://github.com/ggml-org/ggml), along with the SIMD-aware weight packing strategy proposed in [AWQ](https://arxiv.org/abs/2306.00978). By combining these techniques, it endeavors to achieve the best of both worlds.

Compared to the implementation in [llama.cpp](https://github.com/ggml-org/llama.cpp) for `Qwen3-8B`, this project delivers better performance during the **prompting phase**, while maintaining comparable (slightly lower) performance during the **autoregressive generation phase**.

[Performance comparison video](https://github.com/user-attachments/assets/9a211f9e-cb5b-4086-a9ee-6ca9985661f4)

---

## Installation

Before setting up the environment, clone the repository and its submodules:

```fish
git clone https://github.com/jacksonsc007/Qwen3.Ink.Cpp.git  
cd Qwen3.Ink.Cpp
git submodule update --init --recursive
```

### Python Environment

We recommend using [`uv`](https://docs.astral.sh/uv/) to set up the Python environment quickly:

```fish
uv sync
```

---

## Deployment

### Performing Quantization on FP32/FP16 Models

Follow the instructions in the notebook:
- `save_A81W41_quantized_weight-qwen3_8b.ipynb`

### Building Qwen3 with C++

Run the build script:

```fish
bash build_qwen.sh
```

### Chatting with the Model

Once built, you can start interacting with the model:

```fish
build/chat
```

---

## Specification

Here is an overview of the essential files:

- `evaluate-qwen3_8b_W4.ipynb`  
  Evaluates the impact of **weight-only quantization** on Qwen3's performance using perplexity on `wikitext` and benchmarking on GSM8K.

- `evaluate-qwen3_8b_A8W4.ipynb`  
  Evaluates the impact of **activation and weight quantization** on Qwen3's performance using perplexity on `wikitext` and benchmarking on GSM8K.

- `save_A81W41_quantized_weight-qwen3_8b.ipynb`  
  Applies A81W41 quantization to the FP32 model and saves the quantized weights and metadata to disk.

- `save_A80W40_quantized_weight-qwen3_8b.ipynb`  
  Applies A80W40 quantization to the FP32 model and saves the quantized weights and metadata to disk.

- `quantize_methods.py`  
  Contains the core quantization methods used throughout the repository.

---

## Acknowledgements

This repository was heavily inspired by and built upon the following resources:

### Courses & Tutorials:
1. [TinyML and Efficient Deep Learning Computing](https://hanlab.mit.edu/courses/2024-fall-65940)  
2. [TinyChatEngine](https://github.com/mit-han-lab/TinyChatEngine)  
3. [TinychatTutorial](https://github.com/mit-han-lab/tinychat-tutorial)  

Much credit goes to [Professor Han's](https://hanlab.mit.edu/songhan) for his open-source sprit and wonderfull lectures.

### Foundational Repositories:
1. [sgemm.c](https://github.com/salykova/sgemm.c.git)  
2. [GGML](https://github.com/ggml-org/ggml)  
3. [llama.cpp](https://github.com/ggml-org/llama.cpp)  
4. [AWQ](https://github.com/mit-han-lab/llm-awq)  