# HydraMesh DCF: Distributed AI Inference Cluster

## Overview

The HydraMesh Distributed AI Inference Cluster is a high-performance, low-latency framework designed for distributed data exchange and real-time AI synchronization. This project implements a **Distributed Actor System** utilizing the 17-byte binary UDP transport protocol defined by the **DeMoD Communications Framework (DCF) v5.0.0**.

The system is optimized for hardware-agnostic deployments, supporting everything from resource-constrained edge devices to high-performance GPU clusters on NixOS.

---

## Copyright and License

**Copyright (c) 2026 DeMoD LLC. All rights reserved.** This software is licensed under the **BSD-3-Clause License**. The underlying DCF design specifications are provided under the **GPL-3.0 License**.

---

## Use Cases and Instructions

### 1. External Ingress (The Shim)

**Use Case:** Bridging external game clients or IoT devices to the internal HydraMesh cluster. The Shim acts as the secure, high-speed entry point that handles binary serialization.

**How to Use:**

* The Shim is provided as a containerized Rust application.
* **Ingress Port:** 9999 (UDP).
* **Target:** Should point to the Head Node's Ingress Port (7777).
* **Command:** `docker run -e SHIM_NODE_TARGET=192.168.1.50:7777 -p 9999:9999/udp demod/dcf-shim`.

### 2. Cluster Routing (Head Node)

**Use Case:** Managing the `WorkerRegistry` and load-balancing inference tasks across all active GPU workers using a round-robin strategy.

**How to Use:**

* **Container:** `nix build .#container-head` and load into Docker.
* **Role:** Set the role to `head`. It will listen for internal worker heartbeats on port 7778 and client traffic on 7777.
* **Status:** Monitor the Ray Dashboard (if enabled) or the Head logs to see worker registration.

### 3. GPU Inference (Worker Node)

**Use Case:** Performing heavy Large Language Model (LLM) or Stable Diffusion generation. Workers utilize **4-bit quantization** and **System RAM Offloading** to prevent crashes on limited hardware.

**How to Use:**

* **Container:** `nix build .#container-worker` and load into Docker.
* **Role:** Set to `worker` and provide the `headIp`.
* **Hardware:** Requires NVIDIA Container Toolkit. Ensure the worker has access to `/run/opengl-driver/lib` for CUDA linking.

### 4. Portable All-in-One (The ISO)

**Use Case:** A "Swiss Army Knife" for field deployment or rapid development. Boot any GPU-capable machine into a full NixOS environment pre-configured with the entire toolchain (Rust, Python, CUDA) and the local source code.

**How to Use:**

* **Build:** `nix build .#iso`.
* **Flash:** Use `dd` or BalenaEtcher to flash the resulting `.iso` to a USB drive.
* **Boot:** Boot from USB. Log in with user `nixos` and password `hydra`.
* **Dev:** The ISO includes `rust-analyzer`, `python-lsp`, and `nvtop` for immediate performance profiling and coding.

---

## Protocol Specification

The system utilizes the DCF 17-byte handshakeless binary header for all internal cluster communication:

| Offset | Field | Type | Description |
| --- | --- | --- | --- |
| 0 | `msg_type` | u8 | 0x01: Heartbeat, 0x02: Task, 0x03: Result 

 |
| 1 | `sequence` | u32 | Big-Endian packet identifier |
| 5 | `timestamp` | u64 | Microseconds since Epoch 

 |
| 13 | `payload_len` | u32 | Length of subsequent data |

---

## Resource Management

* 
**System RAM Offloading**: If VRAM is exhausted, the system spills model weights to DRAM via the `offload_folder` configuration.


* **OOM Prioritization**: The NixOS module sets `OOMScoreAdjust = -500`, ensuring the Linux kernel preserves the inference process during memory spikes.
* 
**Self-Healing**: The Head Node prunes silent workers within 10 seconds based on RTT and heartbeat metrics.



---

## Developer Experience (DevX)

To start a local development environment with all compilers and ML libraries:

```bash
nix develop

```

This shell provides the **HydraMesh SDK** components, including the `dcf-plugin-manager` logic for dynamic transport loading
