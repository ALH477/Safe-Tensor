# HydraMesh DCF: Distributed AI Inference Cluster

The HydraMesh Distributed AI Inference Cluster is a high-performance, low-latency framework designed for distributed data exchange and real-time AI synchronization. It implements a Distributed Actor System that utilizes the 17-byte binary UDP transport protocol defined by the DeMoD Communications Framework (DCF) v5.0.0. This version ensures hardware-agnostic deployments, supporting resources from edge devices to high-performance GPU clusters on NixOS.

---

## Copyright and License

Copyright (c) 2026 DeMoD LLC. All rights reserved. This software is licensed under the BSD-3-Clause License. The underlying DCF design specifications are provided under the GPL-3.0 License. This ensures transparency and community-driven development while protecting the proprietary implementations of the shim and driver logic.

---

## System Architecture

The environment implements a Distributed DCF Mesh Orchestrator, which effectively decouples high-level routing from intensive GPU computation.

The Ingress Layer uses the DCF Standalone Shim, a high-performance bidirectional UDP bridge written in Rust, which serves as the entry point for external clients. The Routing Layer consists of the Head Controller, a central router that manages the worker registry via UDP heartbeats and load-balances tasks across the mesh using 17-byte DCF messages. The Inference Layer is comprised of Worker Nodes, which are GPU-accelerated endpoints that execute LLM and Stable Diffusion tasks using 4-bit quantization and RAM offloading for hardware efficiency.

---

## Sourcing Models

The system is designed to pull models dynamically using your Hugging Face credentials, which is essential for accessing gated models like Llama-3 or Mistral. To configure this, you must obtain a "Read" access token from your Hugging Face settings and provide it to the cluster through the `HF_TOKEN` environment variable or the `hfToken` Nix option.

For local hosting, the cluster supports loading custom `.safetensors` files directly from a designated directory. Users should place their model checkpoints in the `./local-models` folder, and the worker node will use the `from_single_file` loader to map weights into memory with high security and efficiency.

---

## Deployment Artifacts (alh477)

The flake generates OCI-compliant containers under the `alh477` namespace to suit various hardware requirements.

The `alh477/mesh-head` image is a lightweight cluster router designed to run on any standard CPU. For NVIDIA users, the `alh477/mesh-worker-nvidia` image provides an inference engine optimized for CUDA. AMD users can utilize the `alh477/mesh-worker-rocm` image, which is built specifically for ROCm backends. Finally, the `alh477/mesh-worker-cpu` image offers a universal, hardware-agnostic inference engine that runs on any machine without requiring specific GPU drivers.

---

## Bare-Metal All-in-One (ISO)

The All-in-One ISO is a bootable NixOS environment designed for field deployment or rapid development. It creates a portable "Swiss Army Knife" that pre-configures the entire toolchain, including Rust, Python, and CUDA, along with the project source code.

When booted, the ISO automatically initializes kernel drivers for both NVIDIA and AMD hardware. It allows the machine to join the cluster as a worker or act as a head node without requiring a persistent OS installation. Developers can also use this environment for real-time coding and performance profiling using included tools like `nvtop` and `rust-analyzer`.

---

## Protocol Definition (17-Byte Header)

All internal cluster traffic adheres to the DCF 17-byte handshakeless binary header to ensure wire compatibility and minimal overhead.

The structure begins with a 1-byte `msg_type` (0x01 for Heartbeat, 0x02 for Task, 0x03 for Result), followed by a 4-byte Big-Endian `sequence` identifier for client tracking. An 8-byte Big-Endian `timestamp` records microseconds since the Epoch. The header concludes with a 4-byte Big-Endian `payload_len`, which specifies the length of the subsequent data payload.

---

## Resource Management and Stability

Professional stability is ensured through aggressive resource management strategies. RAM Offloading prevents Out-Of-Memory (OOM) crashes by spilling model weights from VRAM to System RAM when capacity is reached.

The NixOS module applies hardened isolation using `ProtectSystem` and `PrivateTmp` to secure the inference environment. Furthermore, workers are assigned an `OOMScoreAdjust` of -500, which prioritizes the inference process and protects it from kernel termination during heavy load spikes. The Head Node also implements self-healing by pruning workers that fail to send heartbeats within a 10-second window.

---

## Developer Experience (DevX)

The project provides a dedicated development shell that can be accessed via `nix develop` or `nix-shell`. This environment is equipped with auto-detection logic that scans the host system for NVIDIA or AMD drivers before falling back to a CPU-only mode.

The shell pre-loads all necessary compilers, ML libraries, and hardware-specific SDKs. It includes aliases and environment variables that point to the correct dynamic libraries, ensuring that tools like `bitsandbytes` function correctly across different hardware backends. This unified interface allows developers to switch between local testing and cluster deployment seamlessly.
