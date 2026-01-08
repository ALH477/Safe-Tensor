# AI Inference Environment (NixOS)

This repository provides a comprehensive Nix flake for deploying and running AI inference workloads. It supports both **Stable Diffusion** (image generation) and **LLMs** (text generation) using the SafeTensors format and CUDA acceleration.

## Copyright and License

Copyright (c) 2026, DeMoD LLC. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of DeMoD LLC nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

---

## Features

* **Dual Modality**: Support for Image (Diffusers) and Text (Transformers) inference.
* **Format Security**: Strictly prefers `.safetensors` to avoid pickle-based vulnerabilities.
* **CUDA Optimized**: Auto-detects NVIDIA GPUs and enables fp16 precision, attention slicing, and 4-bit LLM quantization.
* **Unified Web UI**: A tabbed Gradio interface for interacting with both models.
* **NixOS Integration**: Includes a module for deploying as a systemd service.

## Prerequisites

1.  **Nix Package Manager** (Flakes enabled).
2.  **NVIDIA GPU** with proprietary drivers enabled in your host Nix config (`allowUnfree = true`).
3.  **HuggingFace Token**: Some models (like Llama 2/3) require a token. Log in via `huggingface-cli login` inside the dev shell if needed.

## Usage

### 1. Development Shell

Enter the environment to access python scripts and system dependencies directly:

```bash
nix develop
