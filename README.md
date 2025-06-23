# Qwen3.Ink.Cpp
`Qwen3.Ink.Cpp` is an self-contained, study-oriented repository, which reproduce Qwen3-8b model based on pure C++.

<video width="640" height="360" controls>  
  <source src="assets/videos/comparison.mp4" type="video/mp4">
  Your browser does not support MP4 video.
</video>

# Specification
- evaluate-qwen3_8b_W4.ipynb
Assess the impact of **weight-only quantization** on Qwen3's performance, by evaluating its perplexity on `wikitext`  and benchmarked it on GSM8k.

- evaluate-qwen3_8b_A8W4.ipynb
Assess the impact of **activation and weight quantization** on Qwen3's performance, by evaluating its perplexity on `wikitext`  and benchmarked it on GSM8k.

- save_A81W41_quantized_weight-qwen3_8b.ipynb
Apply A81W41 quantization on FP32 model and save the quantized weights and other meta information to local disk.

- save_A80W40_quantized_weight-qwen3_8b.ipynb
Apply A80W40 quantization on FP32 model and save the quantized weights and other meta information to local disk.

- quantize_methods.py
This library houses the de facto quantization methods used in the repository.