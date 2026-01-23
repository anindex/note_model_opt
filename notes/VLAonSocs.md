# Vision-Language-Action (VLA) Models on SoCs

> **Extending the "Other-than-CUDA" Landscape with Practical Deployment Paths**

**Author:** Hung T. Ho, Hanoi  
**Last Updated:** January 2026

---

## Table of Contents

- [Purpose of This README](#purpose-of-this-readme)
- [What Is a VLA in Deployment Terms?](#what-is-a-vla-in-deployment-terms)
- [SoCs Covered](#socs-covered)
- [VLA Feasibility on SoCs](#vla-feasibility-on-socs)
  - [VLMs & LLM Feasibility on SoCs](#vlms--llm-feasibility-on-socs)
  - [Diffusion Feasibility on SoCs](#diffusion-feasibility-on-socs)
  - [VLA Feasibility on SoCs](#vla-feasibility-on-socs-1)
- [Small-Scale VLAs for SoC Deployment](#small-scale-vlas-for-soc-deployment)
  - [Common Architectures](#common-architectures)
  - [Optimization Strategies](#optimization-strategies)
    - [Quantization](#quantization)
    - [Token Reduction](#token-reduction)
    - [Distillation](#distillation)
    - [Dynamic Inference](#dynamic-inference)
  - [Practical Stack for Optimizing Models](#practical-stack-for-optimizing-models)

---
---

## Purpose of This README

This document extends the original **"ML Training & Inference on SoCs: the Other-than-CUDA Landscape"** note.

### The original note:

- Maps the ecosystem layers (APIs → kernels → runtimes → vendor SDKs)
- Lists key frameworks and SDKs
- Explains where performance work happens on SoCs

### What it does not explicitly answer:

- How to deploy Vision-Language-Action (VLA) models on each SoC
- Which SoCs can realistically run LLMs vs only VLMs
- What model formats, runtimes, and vendor toolchains are actually used in practice
- Where vendor-specific compilation and profiling must happen

**This README fills those gaps without repeating the original content.**

---

## What Is a VLA in Deployment Terms?

A Vision-Language-Action model is **not a single monolithic network** when deployed on SoCs.

In practice, it is a **heterogeneous pipeline**:

```
Vision encoder (camera → embeddings)
    ↓
Language or reasoning core (instruction grounding)
    ↓
Action head (robot control outputs)
```

### Each part:

- Has different numerical sensitivity
- Maps to different accelerators
- Often runs at different frequencies

> **This is why SoC deployment decisions must be model-aware, not just framework-aware.**

---

## SoCs Covered

This README provides deployment guidance for all SoCs referenced in the original note:
- NVIDIA Jetson Orin (Nvidia / GPU) 
- Qualcomm Snapdragon / QRB / RB-class / IQ-series
- Apple Silicon (iPhone, iPad, Mac)

---

## VLA Feasibility on SoCs

**A critical deployment distinction for Vision-Language-Action models:**

### VLMs & LLM Feasibility on SoCs
#### Platform Notes

These platforms can run **VLM & LLMs** and examples:

- **Apple M-series devices** using Core ML and the Apple Neural Engine:
    - https://github.com/mlc-ai/mlc-llm/blob/main/docs/deploy/ios.rst
    - https://gist.github.com/othyn/42e67d7b6116d88d6c9c83e7d84b20c0
    - https://github.com/ml-explore/mlx-lm
    - https://github.com/vllm-project/vllm-metal
    
- **Qualcomm Snapdragon 8-class** and robotics SKUs using Hexagon NPU via QNN
    - https://github.com/quic/ai-hub-apps/tree/main/tutorials/llm_on_genie
    - https://github.com/quic/ai-hub-models/blob/main/qai_hub_models/models/qwen2_5_7b_instruct/README.md
- **NVIDIA Jetson Orin** (CUDA-based, included for completeness)
    - https://www.jetson-ai-lab.com/models/
    - https://docs.nvidia.com/jetson/jps/inference-services/vlm.html

- **Rockchip RKNN**: 
    - https://github.com/airockchip/rknn-llm


### Diffusion Feasibility on SoCs
#### Platform Notes

- **Qualcomm Snapdragon / AI Engine Direct (QNN):**  
    - https://github.com/quic/ai-hub-models/tree/main/qai_hub_models/models/stable_diffusion_v2_1

- **Apple Silicon (M-series):**
    - https://github.com/apple/ml-stable-diffusion

- **NVIDIA Jetson Orin** (CUDA-based, included for completeness)
    - https://www.jetson-ai-lab.com/archive/tutorial_stable-diffusion.html

### VLA Feasibility on SoCs
#### Platform Notes

- **NVIDIA Jetson Orin** (CUDA-based, included for completeness)
    - https://www.jetson-ai-lab.com/archive/openvla.html#inference-simulation
---

## Small-Scale VLAs for SoC Deployment

For edge deployment (Robotics, Mobile), standard 7B+ VLAs are often too heavy. "Small VLAs" optimize for:
1.  **Low Latency**: >10Hz loop times for real-time control.
2.  **Low VRAM**: Fitting within 8GB-16GB shared memory (Jetson Orin Nano, iPad, Snapdragon).

### Common Architectures

- **SmolVLA**: [[Blog]](https://huggingface.co/blog/smolvla) [[Code]](https://github.com/huggingface/lerobot)
    - *A lightweight VLA specifically designed for efficiency. Uses highly compressed vision encoders (e.g., SigLIP) and smaller LLM backbones (e.g., Vicuna-7B, Phi-3, or TinyLlama)*

- **NanoVLA**: [[Paper]](https://arxiv.org/abs/2510.25122)
    - *Routing Decoupled Vision-Language Understanding for Nano-sized Generalist Robotic Policies*

- **TinyVLA**: [[Paper]](https://arxiv.org/abs/2409.12514) [[Code]](https://github.com/liyaxuanliyaxuan/TinyVLA) [[Web]](https://tiny-vla.github.io)
    - *Towards Fast, Data-Efficient Vision-Language-Action Models for Robotic Manipulation*

- **RoboMamba**: [[Paper]](https://arxiv.org/abs/2406.04339) [[Web]](https://sites.google.com/view/robomamba-web)
    - *RoboMamba: Efficient Vision-Language-Action Model for Robotic Reasoning and Manipulation*

- **MoLe-VLA**: [[Paper]](https://arxiv.org/abs/2503.20384) [[Code]](https://github.com/RoyZry98/MoLe-VLA-Pytorch) [[Web]](https://sites.google.com/view/mole-vla)
    - *Dynamic Layer-skipping Vision Language Action Model via Mixture-of-Layers for Efficient Robot Manipulation*


### Optimization Strategies
1.  **Quantization**:
    - **LLM/VLM Backbone**: 4-bit (AWQ, GPTQ) or 2-bit quantization significantly reduces memory bandwidth requirements.
    - **Action Tokenization**:
        - **VQ-VLA**: [[Paper]](https://arxiv.org/abs/2507.01016) [[Web]](https://xiaoxiao0406.github.io/vqvla.github.io) [[Code]](https://github.com/xiaoxiao0406/VQ-VLA)
            - *Improving Vision-Language-Action Models via Scaling Vector-Quantized Action Tokenizers*

2.  **Token Reduction**:
    - **Pruning/Caching**: Removing less important visual tokens before the LLM stage.
        - **VLA-Pruner**: [[Paper]](https://arxiv.org/abs/2511.16449)
            - *Temporal-Aware Dual-Level Visual Token Pruning for Efficient Vision-Language-Action Inference*
        
        - **SpecPrune-VLA**: [[Paper]](https://arxiv.org/abs/2509.05614)
            - *Accelerating Vision-Language-Action Models via Action-Aware Self-Speculative Pruning*

        - **FastDriveVLA**: [[Paper]](https://arxiv.org/abs/2507.23318)
            - *Efficient End-to-End Driving via Plug-and-Play Reconstruction-based Token Pruning*

        - **SP-VLA**: [[Paper]](https://arxiv.org/abs/2506.12723)
            - *A Joint Model Scheduling and Token Pruning Approach for VLA Model Acceleration*

        - **VLA-Cache**: [[Paper]](https://arxiv.org/pdf/2502.02175) [[Code]](https://github.com/siyuhsu/vla-cache)
            - *Efficient Vision-Language-Action Manipulation via Adaptive Token Caching*
        
        - **FAST**: [[Paper]](https://arxiv.org/pdf/2501.09747) [[Code]](https://github.com/openvla/openvla) [[Web]](https://www.pi.website/research/fast)
            - *Efficient Action Tokenization for Vision-Language-Action Models*
            
        - **EfficientVLA**: [[Paper]](https://arxiv.org/abs/2506.10100)
            - *Training-Free Acceleration and Compression for Vision-Language-Action Models*
    - **Compression**: Projecting visual embeddings into a smaller latent space.
        - **Compressor-VLA**: [[Paper]](https://arxiv.org/abs/2511.18950)
            - *Instruction-Guided Visual Token Compression for Efficient Robotic Manipulation*

3.  **Distillation**: Training smaller "student" VLAs to mimic the logic of larger "teacher" models (like OpenVLA-7B).
    - **Refined Policy Distillation**: [[Paper]](https://arxiv.org/abs/2503.05833) [[Code]](https://github.com/RobotControlStack/vlagents) [[Web]](https://refined-policy-distillation.github.io)
        - *Refined Policy Distillation: From VLA Generalists to RL Experts*

    - **CEED-VLA**: [[Paper]](https://www.arxiv.org/pdf/2506.13725) [[Code]](https://github.com/OpenHelix-Team/CEED-VLA) [[Web]](https://irpn-eai.github.io/CEED-VLA/)
        - *Consistency Vision-Language-Action Model with Early-Exit Decoding*
    
    - **MoLe-VLA**: [[Paper]](https://arxiv.org/abs/2503.20384) [[Code]](https://github.com/RoyZry98/MoLe-VLA-Pytorch) [[Web]](https://sites.google.com/view/mole-vla)
        - *Dynamic Layer-skipping Vision Language Action Model via Mixture-of-Layers for Efficient Robot Manipulation*

    - **ONE-STEP DIFFUSION POLICY**: [[Paper]](https://arxiv.org/pdf/2410.21257) [[Web]](https://research.nvidia.com/labs/dir/onedp/)
        - *FAST VISUOMOTOR POLICIES VIA DIFFUSION DISTILLATION*

4.  **Dynamic Inference**:
    - **CEED-VLA**: [[Paper]](https://www.arxiv.org/pdf/2506.13725) [[Code]](https://github.com/OpenHelix-Team/CEED-VLA) [[Web]](https://irpn-eai.github.io/CEED-VLA/)
        - *Consistency Vision-Language-Action Model with Early-Exit Decoding*

    - **PD-VLA**: [[Paper]](https://arxiv.org/pdf/2503.02310)
        - *Accelerating Vision-Language-Action Model Integrated with Action Chunking via Parallel Decoding*
    
    - **Astra**: [[Paper]](https://arxiv.org/pdf/2408.01147)
        - *Efficient Transformer Architecture and Contrastive Dynamics Learning for Embodied Instruction Following*

### Practical Stack for Optimizing Models
- **Frameworks**:
    - **Pruna**: [[Code]](https://github.com/PrunaAI/pruna)
        - *For stacking optimizations (quantization + compilation) on PyTorch models.*
    - **NVIDIA ModelOpt**: [[Code]](https://github.com/NVIDIA/Model-Optimizer)
        - *Optimization suite for NVIDIA GPUs (Jetson/Orin), including PTQ, QAT, and sparsity.*
    - **Infinigence**: [[Code]](https://github.com/infinigence)
        - *High-performance inference solutions.*
