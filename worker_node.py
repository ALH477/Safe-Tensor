#!/usr/bin/env python3
"""
HydraMesh Worker Node v5.1.0
GPU/CPU inference worker for the DCF AI Inference Cluster

Copyright (c) 2026 DeMoD LLC. All rights reserved.
Licensed under BSD-3-Clause.
"""

import argparse
import time
import signal
import threading
import os
import sys
import json
import io
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

# ═══════════════════════════════════════════════════════════════════════════════
# Environment & Device Detection
# ═══════════════════════════════════════════════════════════════════════════════

def detect_device() -> str:
    """
    Auto-detect the best available compute device.
    
    Priority: TORCH_DEVICE env → CUDA → ROCm → CPU
    """
    # Allow explicit override
    env_device = os.getenv("TORCH_DEVICE", "").lower()
    if env_device in ("cuda", "rocm", "cpu"):
        return "cuda" if env_device == "rocm" else env_device  # ROCm uses CUDA API
    
    try:
        import torch
        if torch.cuda.is_available():
            # Check if this is actually ROCm (HIP)
            if hasattr(torch.version, 'hip') and torch.version.hip:
                return "cuda"  # ROCm presents as CUDA
            return "cuda"
    except ImportError:
        pass
    
    return "cpu"


DEVICE = detect_device()

# Now import ML libraries (after device detection to set env vars properly)
import torch
from dcf_common import (
    EventDrivenUDPSocket, DCFMessage, MessageType, ErrorCode,
    NodeMetrics, chunk_payload, setup_logging, validate_payload,
    DEFAULT_HEARTBEAT_INTERVAL
)

logger = setup_logging("WORKER")


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

LISTEN_PORT = int(os.getenv("DCF_WORKER_PORT", "7779"))
HEALTH_PORT = int(os.getenv("DCF_WORKER_HEALTH_PORT", "8081"))
HEARTBEAT_INTERVAL = float(os.getenv("DCF_HEARTBEAT_INTERVAL", str(DEFAULT_HEARTBEAT_INTERVAL)))

# Model defaults
DEFAULT_SD_MODEL = os.getenv("SD_MODEL_ID", "runwayml/stable-diffusion-v1-5")
DEFAULT_INFERENCE_STEPS = int(os.getenv("SD_INFERENCE_STEPS", "25"))
DEFAULT_GUIDANCE_SCALE = float(os.getenv("SD_GUIDANCE_SCALE", "7.5"))


# ═══════════════════════════════════════════════════════════════════════════════
# Inference Engine Abstraction
# ═══════════════════════════════════════════════════════════════════════════════

class InferenceEngine:
    """
    Abstract base for inference engines.
    
    Supports multiple backends and model types.
    """
    
    def __init__(self, device: str):
        self.device = device
        self.model_loaded = False
        self._lock = threading.Lock()
    
    def load(self) -> bool:
        """Load the model. Returns success status."""
        raise NotImplementedError
    
    def run(self, prompt: str, **kwargs) -> bytes:
        """Run inference. Returns result bytes."""
        raise NotImplementedError
    
    def unload(self):
        """Unload model and free resources."""
        raise NotImplementedError
    
    def get_info(self) -> dict:
        """Get engine info for metrics."""
        return {"device": self.device, "loaded": self.model_loaded}


class StableDiffusionEngine(InferenceEngine):
    """
    Stable Diffusion inference engine with multi-backend support.
    
    Supports:
    - HuggingFace model IDs (e.g., "runwayml/stable-diffusion-v1-5")
    - Local .safetensors files
    - CUDA, ROCm, and CPU backends
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = DEVICE):
        super().__init__(device)
        self.model_path = model_path or DEFAULT_SD_MODEL
        self.pipe = None
        self._dtype = torch.float16 if device == "cuda" else torch.float32
        
    def load(self) -> bool:
        """Load the Stable Diffusion pipeline."""
        from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
        
        try:
            logger.info(f"Loading model: {self.model_path}")
            logger.info(f"Device: {self.device} | Dtype: {self._dtype}")
            
            # Determine loading method
            if self._is_local_file():
                self.pipe = self._load_from_file()
            else:
                self.pipe = self._load_from_hub()
            
            # Optimize scheduler
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )
            
            # Apply optimizations
            self._apply_optimizations()
            
            self.model_loaded = True
            logger.info("✓ Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to load model: {e}")
            return False
    
    def _is_local_file(self) -> bool:
        """Check if model_path is a local file."""
        path = Path(self.model_path)
        return path.exists() and path.suffix in ('.safetensors', '.ckpt', '.bin')
    
    def _load_from_file(self):
        """Load from local safetensors file."""
        from diffusers import StableDiffusionPipeline
        
        logger.info(f"Loading from local file: {self.model_path}")
        pipe = StableDiffusionPipeline.from_single_file(
            self.model_path,
            torch_dtype=self._dtype,
            use_safetensors=True,
            load_safety_checker=False
        )
        return pipe.to(self.device)
    
    def _load_from_hub(self):
        """Load from HuggingFace Hub."""
        from diffusers import StableDiffusionPipeline
        
        logger.info(f"Loading from HuggingFace: {self.model_path}")
        pipe = StableDiffusionPipeline.from_pretrained(
            self.model_path,
            torch_dtype=self._dtype,
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False
        )
        return pipe.to(self.device)
    
    def _apply_optimizations(self):
        """Apply device-specific optimizations."""
        if self.device == "cuda":
            # Memory optimizations for GPU
            self.pipe.enable_attention_slicing()
            
            # Enable CPU offload for large models on limited VRAM
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if vram_gb < 8:
                logger.info(f"Low VRAM ({vram_gb:.1f}GB), enabling CPU offload")
                self.pipe.enable_sequential_cpu_offload()
            
            # Try xformers if available
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                logger.info("✓ xformers enabled")
            except Exception:
                pass
    
    def run(self, prompt: str, **kwargs) -> bytes:
        """Generate image from prompt. Returns PNG bytes."""
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        steps = kwargs.get('steps', DEFAULT_INFERENCE_STEPS)
        guidance = kwargs.get('guidance', DEFAULT_GUIDANCE_SCALE)
        
        with self._lock:
            with torch.inference_mode():
                result = self.pipe(
                    prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance
                )
                image = result.images[0]
        
        # Convert to PNG bytes
        buffer = io.BytesIO()
        image.save(buffer, format='PNG', optimize=True)
        return buffer.getvalue()
    
    def unload(self):
        """Free GPU memory."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            if self.device == "cuda":
                torch.cuda.empty_cache()
            self.model_loaded = False
            logger.info("Model unloaded")
    
    def get_info(self) -> dict:
        info = super().get_info()
        info.update({
            "model": self.model_path,
            "type": "stable-diffusion",
            "dtype": str(self._dtype)
        })
        if self.device == "cuda":
            info["vram_allocated_mb"] = round(torch.cuda.memory_allocated() / (1024**2), 1)
        return info


# ═══════════════════════════════════════════════════════════════════════════════
# Task Processor
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TaskResult:
    """Result of processing a task."""
    success: bool
    payload: bytes
    error_code: ErrorCode = ErrorCode.OK
    error_message: str = ""
    duration_ms: float = 0.0


def parse_task_payload(payload: bytes) -> Dict[str, Any]:
    """
    Parse task payload.
    
    Supports:
    - Plain text (prompt only)
    - JSON with parameters
    """
    text = payload.decode('utf-8').strip()
    
    # Try JSON first
    if text.startswith('{'):
        try:
            data = json.loads(text)
            if 'prompt' not in data:
                raise ValueError("JSON payload missing 'prompt' field")
            return data
        except json.JSONDecodeError:
            pass
    
    # Plain text prompt
    return {'prompt': text}


# ═══════════════════════════════════════════════════════════════════════════════
# Health Server
# ═══════════════════════════════════════════════════════════════════════════════

class WorkerHealthHandler(BaseHTTPRequestHandler):
    """HTTP health endpoint for worker."""
    
    engine: InferenceEngine = None
    metrics: NodeMetrics = None
    busy: bool = False
    
    def log_message(self, format, *args):
        pass
    
    def do_GET(self):
        if self.path == "/health":
            healthy = self.engine and self.engine.model_loaded
            status = 200 if healthy else 503
            self._send_json({
                "status": "healthy" if healthy else "unhealthy",
                "model_loaded": healthy,
                "busy": self.busy
            }, status)
        elif self.path == "/metrics":
            data = {
                "worker": self.metrics.to_dict() if self.metrics else {},
                "engine": self.engine.get_info() if self.engine else {}
            }
            self._send_json(data)
        else:
            self._send_json({"error": "Not found"}, 404)
    
    def _send_json(self, data: dict, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())


# ═══════════════════════════════════════════════════════════════════════════════
# Worker Node
# ═══════════════════════════════════════════════════════════════════════════════

class WorkerNode:
    """
    HydraMesh inference worker.
    
    Connects to head controller, processes inference tasks,
    and maintains health metrics.
    """
    
    def __init__(self, head_ip: str, head_port: int = 7778, 
                 model_path: Optional[str] = None):
        self.head_addr = (head_ip, head_port)
        self.metrics = NodeMetrics()
        
        # Initialize engine
        self.engine = StableDiffusionEngine(model_path=model_path, device=DEVICE)
        
        # Network
        self.sock = EventDrivenUDPSocket(LISTEN_PORT)
        
        # State
        self._shutdown = threading.Event()
        self._busy = False
        self._sequence = 0
        
        # Threads
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
    
    def start(self):
        """Start the worker node."""
        logger.info("═" * 60)
        logger.info("  HydraMesh Worker Node v5.1.0")
        logger.info("═" * 60)
        logger.info(f"  Device:      {DEVICE.upper()}")
        logger.info(f"  Listen Port: UDP {LISTEN_PORT}")
        logger.info(f"  Health Port: HTTP {HEALTH_PORT}")
        logger.info(f"  Head Node:   {self.head_addr[0]}:{self.head_addr[1]}")
        logger.info("═" * 60)
        
        # Load model
        if not self.engine.load():
            logger.error("Failed to load model, exiting")
            sys.exit(1)
        
        # Start health server
        self._start_health_server()
        
        # Start heartbeat
        self._heartbeat_thread.start()
        
        logger.info("✓ Worker online and ready for tasks")
        
        # Main loop
        self._run()
    
    def shutdown(self):
        """Graceful shutdown."""
        logger.info("Initiating shutdown...")
        self._shutdown.set()
        self.engine.unload()
        self.sock.close()
        logger.info("Shutdown complete")
    
    def _start_health_server(self):
        """Start HTTP health server."""
        WorkerHealthHandler.engine = self.engine
        WorkerHealthHandler.metrics = self.metrics
        
        def update_busy():
            WorkerHealthHandler.busy = self._busy
        
        server = HTTPServer(("0.0.0.0", HEALTH_PORT), WorkerHealthHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        logger.info(f"Health server on http://0.0.0.0:{HEALTH_PORT}")
    
    def _heartbeat_loop(self):
        """Send periodic heartbeats to head."""
        while not self._shutdown.is_set():
            try:
                self._sequence += 1
                msg = DCFMessage.heartbeat(self._sequence, LISTEN_PORT)
                self.sock.send(msg, self.head_addr)
                self.metrics.messages_sent += 1
            except Exception as e:
                logger.warning(f"Heartbeat failed: {e}")
            
            # Sleep in small increments for responsive shutdown
            for _ in range(int(HEARTBEAT_INTERVAL * 10)):
                if self._shutdown.is_set():
                    break
                time.sleep(0.1)
    
    def _run(self):
        """Main event loop."""
        while not self._shutdown.is_set():
            result = self.sock.recv(timeout=0.1)
            if result:
                msg, addr = result
                self.metrics.messages_received += 1
                self.metrics.bytes_received += len(msg.payload)
                
                if msg.msg_type == MessageType.TASK:
                    self._handle_task(msg)
                elif msg.msg_type == MessageType.SHUTDOWN:
                    logger.info("Received shutdown signal from head")
                    self.shutdown()
    
    def _handle_task(self, msg: DCFMessage):
        """Process an inference task."""
        self._busy = True
        start_time = time.time()
        
        logger.info(f"← Task {msg.sequence} received ({len(msg.payload)} bytes)")
        
        try:
            # Validate payload
            is_valid, error = validate_payload(msg.payload)
            if not is_valid:
                self._send_error(msg.sequence, ErrorCode.INVALID_PAYLOAD, error)
                return
            
            # Parse task
            task_data = parse_task_payload(msg.payload)
            prompt = task_data['prompt']
            
            logger.info(f"  Generating: \"{prompt[:50]}{'...' if len(prompt) > 50 else ''}\"")
            
            # Run inference
            result_bytes = self.engine.run(
                prompt,
                steps=task_data.get('steps', DEFAULT_INFERENCE_STEPS),
                guidance=task_data.get('guidance', DEFAULT_GUIDANCE_SCALE)
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Send result (chunked if necessary)
            self._send_result(msg.sequence, result_bytes)
            
            self.metrics.tasks_processed += 1
            self.metrics.record_latency(duration_ms)
            
            logger.info(f"✓ Task {msg.sequence} complete ({duration_ms:.0f}ms, {len(result_bytes)} bytes)")
            
        except Exception as e:
            logger.exception(f"✗ Task {msg.sequence} failed: {e}")
            self._send_error(msg.sequence, ErrorCode.INFERENCE_FAILED, str(e))
            self.metrics.tasks_failed += 1
        
        finally:
            self._busy = False
    
    def _send_result(self, sequence: int, payload: bytes):
        """Send result back to head, chunking if necessary."""
        chunks = chunk_payload(payload, sequence)
        for chunk in chunks:
            self.sock.send(chunk, self.head_addr)
            self.metrics.messages_sent += 1
            self.metrics.bytes_sent += len(chunk.payload)
    
    def _send_error(self, sequence: int, code: ErrorCode, message: str):
        """Send error back to head."""
        msg = DCFMessage.error(sequence, code, message)
        self.sock.send(msg, self.head_addr)
        self.metrics.messages_sent += 1


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="HydraMesh Worker Node - GPU/CPU Inference Worker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Connect to local head with HuggingFace model
  python worker_node.py --head-ip 127.0.0.1

  # Use local safetensors file
  python worker_node.py --head-ip 192.168.1.10 --model-path /models/sd-v1-5.safetensors

  # Force CPU mode
  TORCH_DEVICE=cpu python worker_node.py --head-ip 127.0.0.1

Environment Variables:
  TORCH_DEVICE          Force device (cuda/cpu)
  HF_TOKEN              HuggingFace token for gated models
  SD_MODEL_ID           Default model ID
  SD_INFERENCE_STEPS    Default inference steps
  SD_GUIDANCE_SCALE     Default CFG scale
        """
    )
    
    parser.add_argument(
        "--head-ip", 
        required=True,
        help="IP address of the head controller"
    )
    parser.add_argument(
        "--head-port",
        type=int,
        default=7778,
        help="Port of head controller worker bus (default: 7778)"
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to model (HuggingFace ID or local .safetensors file)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=LISTEN_PORT,
        help=f"UDP port to listen on (default: {LISTEN_PORT})"
    )
    
    args = parser.parse_args()
    
    # Update global port if specified
    global LISTEN_PORT
    LISTEN_PORT = args.port
    
    worker = WorkerNode(
        head_ip=args.head_ip,
        head_port=args.head_port,
        model_path=args.model_path
    )
    
    # Signal handlers
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        worker.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        worker.start()
    except KeyboardInterrupt:
        worker.shutdown()
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
