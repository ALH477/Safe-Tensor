# AI Inference Cluster (HydraMesh/DCF Edition)

This repository provides a production-ready, distributed AI inference environment built on the **DeMoD Communications Framework (DCF)** design principles. It utilizes a high-performance, 17-byte binary UDP transport for inter-node communication, achieving the low-latency targets required for real-time applications.

---

## Copyright and License

Copyright (c) 2026, DeMoD LLC. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of DeMoD LLC nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

---

## System Architecture

The environment implements a **Distributed DCF Mesh Orchestrator**, decoupling high-level routing from intensive GPU computation.

### Core Components

* **DCF Standalone Shim**: A high-performance bidirectional UDP bridge written in Rust, serving as the ingress point for external clients.
* 
**Head Controller**: The central router that manages the `WorkerRegistry` via UDP heartbeats and load-balances tasks across the mesh using 17-byte DCF messages.


* 
**Worker Nodes**: GPU-accelerated inference endpoints that execute LLM and Stable Diffusion tasks, utilizing 4-bit quantization and RAM offloading for hardware efficiency.



---

## Technical Specifications

| Component | Specification |
| --- | --- |
| **Protocol** | DCF Binary UDP (17-byte header) 

 |
| **Header Format** | `Type(u8)` |
| **Target Latency** | <1ms local exchange 

 |
| **ML Framework** | PyTorch / Hugging Face Transformers & Diffusers |
| **NixOS Support** | Native Flake with Systemd Hardening & Resource Limits |

---

## Cluster Protocol Definition

The cluster utilizes the following message types for orchestration:

* 
**MSG_HEARTBEAT (0x01)**: Sent by workers to the Head Node every 2 seconds to maintain active registration.


* 
**MSG_TASK (0x02)**: Sent by the Head Node to workers containing the inference payload.


* 
**MSG_RESULT (0x03)**: Sent by workers back to the Head Node upon successful inference completion.



---

## NixOS Deployment

### 1. Enable the Cluster Module

Add the HydraMesh module to your system configuration:

```nix
{
  services.hydramesh = {
    enable = true;
    role = "head"; # Or "worker"
    headIp = "100.x.y.z"; # IP of the controller node
    hfToken = "your_huggingface_token";
  };
}

```

### 2. Network Configuration

The framework requires the following UDP ports to be accessible within your mesh network:

* **7777**: External Client Ingress (Head Node)
* **7778**: Internal Worker Bus (Head Node)
* **7779**: Internal Task Bus (Worker Nodes)

---

## Resource Management

To ensure professional stability, the system incorporates:

* **System RAM Offloading**: Weights automatically spill to DRAM if VRAM is exceeded, preventing process termination.
* **OOM Score Adjustment**: The inference daemon is prioritized by the kernel to prevent it from being killed during memory spikes.
* 
**Self-Healing**: The Head Node automatically prunes workers that fail to send a heartbeat within 10 seconds, rerouting subsequent tasks.



---

## Building and Development

### Generate Dependencies

```bash
nix develop

```

### Run Standalone Shim (Ingress)

```bash
SHIM_INGRESS_PORT=9999 SHIM_NODE_TARGET=127.0.0.1:7777 ./target/release/dcf-shim
