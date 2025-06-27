## Introduction
`Qwen3.Ink.Cpp` is a study-oriented repository that reproduces the **Qwen3-8B** model using pure C++.  

This project integrates quantization methods and optimized QGEMM(Quantized GEMM) kernels in [GGML](https://github.com/ggml-org/ggml), along with the SIMD-aware weight packing strategy proposed in [AWQ](https://arxiv.org/abs/2306.00978). By combining these techniques, it endeavors to achieve the best of both worlds.

In contrast to the implementation in [llama.cpp](https://github.com/ggml-org/llama.cpp) for `Qwen3-8B`, this project delivers better performance during the **prompt phase**, while maintaining comparable (slightly lower) performance during the **autoregressive generation phase**.

> [Warning]
> Please note that the benchmark results are highly dependent on my personal hardwares and may not be reproducible on other hardwares.

[Performance comparison video](https://github.com/user-attachments/assets/9a211f9e-cb5b-4086-a9ee-6ca9985661f4)

### Benchmark
#### Experiment Settings

| Component       | Specification              |
|-----------------|----------------------------|
| Operating System| Ubuntu 22.04               |
| CPU             | Intel Core i5-13600KF      |
| DRAM       | 3600 MT/s, Dual-channel                  |

#### Performance

| model        |  quantization scheme | perpelexity | gsm8k | 
| ------------ |  ------------------- | ----------- | ----- | 
| qwen3-8b     |  fp16                | 10.97       | 87.57 | 
| qwen3-8b     |  w41                 | 12.08       | 86.20 | 
| qwen3-8b     |  w40                 | 11.73       | 84.99 | 
| qwen3-8b     |  w4z                 | 11.49       | 85.52 | 
| qwen3-8b     |  A80W41              | 12.06       | 87.03 | 
| qwen3-8b     |  A80W4z              | 11.53       | 85.29 | 
| qwen3-8b-awq |  -                   | 11.52       | 86.35 | 

#### Efficiency


| model                                    | mem | backend | threads | test  | t/s          |
| ---------------------------------------- | --------------------- | ------- | ------- | ----- | ------------ |
| qwen3 8B Q4_0                            | 8.6 GiB               | CPU     | 16      | pp322 | 60.36 ± 0.23 |
| qwen3 8B Q4_0                            | 8.6 GiB               | CPU     | 16      | tg128 | 10.40 ± 0.00 |
| qwen3 8B A81Q41-repack-FP16_FP32_mix ink | 14.0 GiB              | CPU     | 16      | pp322 | 134.35       |
| qwen3 8B A81Q41-repack-FP16_FP32_mix ink | 14.0 GiB              | CPU     | 16      | tg128 | 9.26         |


- We leveraged `llama-bench` to benchmark llama.cpp models, and our own code along with custom input to benchmark our custom model.  It is not a strict comparison.
- `tg128` stands for generation of 128 tokens in autoregreesive generation phase. `pp322` denotes processing an input prompt with 322 tokens.

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
- [`save_A81W41_quantized_weight-qwen3_8b.ipynb`](https://github.com/jacksonsc007/Qwen3.Ink.Cpp/blob/qwen3_int4_fp16_fp32_mix_finetuned/notebooks/save_A81W41_quantized_weight-qwen3_8b.ipynb)

### Building Qwen3 with C++

Run the build script:

```fish
bash build_qwen.sh
```

### Chatting with the Model  ε≡٩(๑>₃<)۶

Once built, you can start interacting with the model:

```fish
build/chat
```

---

## Specification

Here is an overview of the essential files:

| File Name                                                                                                                                                                                    | Description                                                                                                                                                     |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`evaluate-qwen3_8b_W4.ipynb`](https://github.com/jacksonsc007/Qwen3.Ink.Cpp/blob/qwen3_int4_fp16_fp32_mix_finetuned/notebooks/evaluate-qwen3_8b_W4.ipynb)                                   | Evaluates the impact of **weight-only quantization** on Qwen3's performance. Perplexity on `wikitext` and the benchmark result on GSM8K are reported.           |
| [`evaluate-qwen3_8b_A8W4.ipynb`](https://github.com/jacksonsc007/Qwen3.Ink.Cpp/blob/qwen3_int4_fp16_fp32_mix_finetuned/notebooks/evaluate-qwen3_8b_A8W4.ipynb)                               | Evaluates the impact of **activation and weight quantization** on Qwen3's performance. Perplexity on `wikitext` and the benchmark result on GSM8K are reported. |
| [`save_A81W41_quantized_weight-qwen3_8b.ipynb`](https://github.com/jacksonsc007/Qwen3.Ink.Cpp/blob/qwen3_int4_fp16_fp32_mix_finetuned/notebooks/save_A81W41_quantized_weight-qwen3_8b.ipynb) | Applies A81W41 quantization to the FP32 model and saves the quantized weights and metadata to disk.                                                             |
| [`save_A80W40_quantized_weight-qwen3_8b.ipynb`](https://github.com/jacksonsc007/Qwen3.Ink.Cpp/blob/qwen3_int4_fp16_fp32_mix_finetuned/notebooks/save_A80W40_quantized_weight-qwen3_8b.ipynb) | Applies A80W40 quantization to the FP32 model and saves the quantized weights and metadata to disk.                                                             |
| [`quantize_methods.py`](https://github.com/jacksonsc007/Qwen3.Ink.Cpp/blob/qwen3_int4_fp16_fp32_mix_finetuned/notebooks/quantize_methods.py)                                                 | Contains the core quantization methods used throughout the repository.                                                                                          |

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
5. [quantized-gemm](https://github.com/jacksonsc007/quantized-gemm)

### Additional Resources
Much appreciation to [Ｒｅ：ゼロから始める異世界生活](https://ncode.syosetu.com/n2267be/) for providing the benchmark text during the development.
