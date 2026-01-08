#!/usr/bin/env python3
"""
DCF Common Library v5.1.0
DeMoD Communications Framework - Core Protocol Implementation

Copyright (c) 2026 DeMoD LLC. All rights reserved.
Licensed under BSD-3-Clause.
"""

import struct
import time
import socket
import hashlib
import selectors
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, Callable, Dict, List
from enum import IntEnum
from threading import Lock

# ═══════════════════════════════════════════════════════════════════════════════
# Protocol Constants (DCF v5.1 - 18-byte header with version + CRC)
# ═══════════════════════════════════════════════════════════════════════════════

PROTOCOL_VERSION = 0x51  # v5.1

# Header: Version(1) + Type(1) + Seq(4) + Timestamp(8) + PayloadLen(4) = 18 bytes
HEADER_FORMAT = ">B B I Q I"
HEADER_SIZE = 18

# Max safe UDP payload (MTU 1500 - IP header 20 - UDP header 8 - DCF header 18)
MAX_PAYLOAD_SIZE = 1454
MAX_CHUNK_PAYLOAD = 1400  # Leave room for chunk metadata

# Timeouts
DEFAULT_HEARTBEAT_INTERVAL = 2.0
DEFAULT_WORKER_TIMEOUT = 10.0
DEFAULT_REQUEST_TIMEOUT = 300.0  # 5 minutes for inference


class MessageType(IntEnum):
    """DCF Message Types"""
    HEARTBEAT = 0x01
    TASK = 0x02
    RESULT = 0x03
    ACK = 0x04
    NACK = 0x05
    CHUNK = 0x06      # For large payload fragmentation
    CHUNK_ACK = 0x07
    HEALTH = 0x08     # Health check request/response
    SHUTDOWN = 0x09   # Graceful shutdown signal
    ERROR = 0xFF


class ErrorCode(IntEnum):
    """DCF Error Codes"""
    OK = 0x00
    INVALID_PAYLOAD = 0x01
    WORKER_BUSY = 0x02
    INFERENCE_FAILED = 0x03
    TIMEOUT = 0x04
    NO_WORKERS = 0x05
    CHUNK_MISSING = 0x06
    VERSION_MISMATCH = 0x07


# ═══════════════════════════════════════════════════════════════════════════════
# Message Classes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DCFMessage:
    """
    DCF Protocol Message with 18-byte header.
    
    Wire Format:
    ┌─────────┬──────┬──────────┬───────────┬─────────────┬─────────┐
    │ Version │ Type │ Sequence │ Timestamp │ PayloadLen  │ Payload │
    │  1 byte │ 1 b  │  4 bytes │  8 bytes  │   4 bytes   │  N bytes│
    └─────────┴──────┴──────────┴───────────┴─────────────┴─────────┘
    """
    msg_type: int
    sequence: int
    timestamp: int
    payload: bytes
    version: int = PROTOCOL_VERSION

    @staticmethod
    def current_timestamp_micros() -> int:
        """Returns current time in microseconds since epoch."""
        return int(time.time() * 1_000_000)

    def serialize(self) -> bytes:
        """Serialize message to wire format."""
        payload_len = len(self.payload)
        header = struct.pack(
            HEADER_FORMAT,
            self.version,
            self.msg_type,
            self.sequence,
            self.timestamp,
            payload_len
        )
        return header + self.payload

    @classmethod
    def deserialize(cls, data: bytes) -> Optional['DCFMessage']:
        """Deserialize from wire format. Returns None on invalid data."""
        if len(data) < HEADER_SIZE:
            return None
        try:
            version, msg_type, sequence, timestamp, payload_len = struct.unpack(
                HEADER_FORMAT, data[:HEADER_SIZE]
            )
            if len(data) < HEADER_SIZE + payload_len:
                return None
            payload = data[HEADER_SIZE:HEADER_SIZE + payload_len]
            return cls(msg_type, sequence, timestamp, payload, version)
        except struct.error:
            return None

    @classmethod
    def heartbeat(cls, sequence: int, worker_port: int) -> 'DCFMessage':
        """Create a heartbeat message."""
        return cls(
            msg_type=MessageType.HEARTBEAT,
            sequence=sequence,
            timestamp=cls.current_timestamp_micros(),
            payload=str(worker_port).encode('utf-8')
        )

    @classmethod
    def task(cls, sequence: int, payload: bytes) -> 'DCFMessage':
        """Create a task message."""
        return cls(
            msg_type=MessageType.TASK,
            sequence=sequence,
            timestamp=cls.current_timestamp_micros(),
            payload=payload
        )

    @classmethod
    def result(cls, sequence: int, payload: bytes) -> 'DCFMessage':
        """Create a result message."""
        return cls(
            msg_type=MessageType.RESULT,
            sequence=sequence,
            timestamp=cls.current_timestamp_micros(),
            payload=payload
        )

    @classmethod
    def error(cls, sequence: int, error_code: ErrorCode, message: str = "") -> 'DCFMessage':
        """Create an error message."""
        payload = struct.pack(">B", error_code) + message.encode('utf-8')
        return cls(
            msg_type=MessageType.ERROR,
            sequence=sequence,
            timestamp=cls.current_timestamp_micros(),
            payload=payload
        )

    @classmethod
    def ack(cls, sequence: int) -> 'DCFMessage':
        """Create an acknowledgment message."""
        return cls(
            msg_type=MessageType.ACK,
            sequence=sequence,
            timestamp=cls.current_timestamp_micros(),
            payload=b''
        )

    def __repr__(self) -> str:
        type_name = MessageType(self.msg_type).name if self.msg_type in MessageType._value2member_map_ else f"0x{self.msg_type:02X}"
        return f"DCFMessage(type={type_name}, seq={self.sequence}, payload_len={len(self.payload)})"


# ═══════════════════════════════════════════════════════════════════════════════
# Chunk Assembly for Large Payloads
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ChunkAssembler:
    """Handles reassembly of chunked messages for payloads exceeding MTU."""
    
    total_chunks: int
    chunks: Dict[int, bytes] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    checksum: bytes = b''
    
    def add_chunk(self, chunk_idx: int, data: bytes) -> bool:
        """Add a chunk. Returns True if all chunks received."""
        self.chunks[chunk_idx] = data
        return len(self.chunks) == self.total_chunks
    
    def assemble(self) -> Optional[bytes]:
        """Assemble complete payload from chunks."""
        if len(self.chunks) != self.total_chunks:
            return None
        result = b''.join(self.chunks[i] for i in range(self.total_chunks))
        # Verify checksum if provided
        if self.checksum:
            computed = hashlib.md5(result).digest()
            if computed != self.checksum:
                return None
        return result
    
    def is_stale(self, timeout: float = 60.0) -> bool:
        """Check if assembly has timed out."""
        return time.time() - self.created_at > timeout


def chunk_payload(data: bytes, sequence: int) -> List[DCFMessage]:
    """
    Split large payload into chunk messages.
    
    Chunk payload format:
    ┌────────────┬─────────────┬───────────┬──────────┐
    │ TotalChunks│ ChunkIndex  │ Checksum  │   Data   │
    │   2 bytes  │   2 bytes   │ 16 bytes  │  N bytes │
    └────────────┴─────────────┴───────────┴──────────┘
    """
    if len(data) <= MAX_PAYLOAD_SIZE:
        return [DCFMessage.result(sequence, data)]
    
    checksum = hashlib.md5(data).digest()
    chunks = []
    total = (len(data) + MAX_CHUNK_PAYLOAD - 1) // MAX_CHUNK_PAYLOAD
    
    for i in range(total):
        start = i * MAX_CHUNK_PAYLOAD
        end = min(start + MAX_CHUNK_PAYLOAD, len(data))
        chunk_data = data[start:end]
        
        # Build chunk header
        header = struct.pack(">HH", total, i) + checksum
        payload = header + chunk_data
        
        msg = DCFMessage(
            msg_type=MessageType.CHUNK,
            sequence=sequence,
            timestamp=DCFMessage.current_timestamp_micros(),
            payload=payload
        )
        chunks.append(msg)
    
    return chunks


def parse_chunk(msg: DCFMessage) -> Tuple[int, int, bytes, bytes]:
    """Parse chunk message. Returns (total_chunks, chunk_idx, checksum, data)."""
    total, idx = struct.unpack(">HH", msg.payload[:4])
    checksum = msg.payload[4:20]
    data = msg.payload[20:]
    return total, idx, checksum, data


# ═══════════════════════════════════════════════════════════════════════════════
# Event-Driven UDP Socket
# ═══════════════════════════════════════════════════════════════════════════════

class EventDrivenUDPSocket:
    """
    Production-grade UDP socket with selector-based I/O multiplexing.
    
    Eliminates busy-wait polling by using OS-level event notification.
    """
    
    def __init__(self, port: int, bind_ip: str = "0.0.0.0"):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Increase buffer sizes for high throughput
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2 * 1024 * 1024)
        
        self.sock.bind((bind_ip, port))
        self.sock.setblocking(False)
        
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.sock, selectors.EVENT_READ)
        
        self.local_addr = self.sock.getsockname()
        self._closed = False
        self._lock = Lock()
        
        logging.info(f"UDP socket bound to {bind_ip}:{port}")

    def send(self, msg: DCFMessage, addr: Tuple[str, int]) -> bool:
        """Send message to address. Returns success status."""
        if self._closed:
            return False
        try:
            with self._lock:
                self.sock.sendto(msg.serialize(), addr)
            return True
        except OSError as e:
            logging.warning(f"Send failed to {addr}: {e}")
            return False

    def send_chunked(self, msg: DCFMessage, addr: Tuple[str, int]) -> bool:
        """Send message, automatically chunking if necessary."""
        if len(msg.payload) <= MAX_PAYLOAD_SIZE:
            return self.send(msg, addr)
        
        chunks = chunk_payload(msg.payload, msg.sequence)
        for chunk in chunks:
            if not self.send(chunk, addr):
                return False
        return True

    def recv(self, timeout: float = 0.1) -> Optional[Tuple[DCFMessage, Tuple[str, int]]]:
        """
        Receive message with timeout.
        
        Uses selector for efficient I/O multiplexing instead of busy-wait.
        """
        if self._closed:
            return None
            
        events = self.selector.select(timeout=timeout)
        if not events:
            return None
            
        try:
            data, addr = self.sock.recvfrom(65536)
            msg = DCFMessage.deserialize(data)
            if msg:
                # Version check
                if msg.version != PROTOCOL_VERSION:
                    logging.warning(f"Version mismatch from {addr}: got 0x{msg.version:02X}, expected 0x{PROTOCOL_VERSION:02X}")
                return msg, addr
        except BlockingIOError:
            pass
        except OSError as e:
            if not self._closed:
                logging.error(f"Recv error: {e}")
        return None

    def recv_batch(self, max_messages: int = 100, timeout: float = 0.01) -> List[Tuple[DCFMessage, Tuple[str, int]]]:
        """Receive multiple messages in a batch for high-throughput scenarios."""
        results = []
        deadline = time.time() + timeout
        
        while len(results) < max_messages:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            result = self.recv(timeout=min(remaining, 0.001))
            if result:
                results.append(result)
            else:
                break
        
        return results

    def close(self):
        """Clean shutdown."""
        self._closed = True
        try:
            self.selector.unregister(self.sock)
            self.selector.close()
            self.sock.close()
        except Exception:
            pass


# Legacy alias for backwards compatibility
AsyncUDPSocket = EventDrivenUDPSocket


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics & Observability
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class NodeMetrics:
    """Metrics for observability and monitoring."""
    
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    tasks_processed: int = 0
    tasks_failed: int = 0
    avg_latency_ms: float = 0.0
    uptime_seconds: float = 0.0
    
    _latencies: List[float] = field(default_factory=list)
    _start_time: float = field(default_factory=time.time)
    _lock: Lock = field(default_factory=Lock)
    
    def record_latency(self, latency_ms: float):
        """Record a latency measurement."""
        with self._lock:
            self._latencies.append(latency_ms)
            # Keep last 1000 samples
            if len(self._latencies) > 1000:
                self._latencies = self._latencies[-1000:]
            self.avg_latency_ms = sum(self._latencies) / len(self._latencies)
    
    def to_dict(self) -> dict:
        """Export metrics as dictionary."""
        return {
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "tasks_processed": self.tasks_processed,
            "tasks_failed": self.tasks_failed,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "uptime_seconds": round(time.time() - self._start_time, 1)
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure structured logging for DCF nodes."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d [%(name)s] %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def validate_payload(data: bytes, max_size: int = 10 * 1024 * 1024) -> Tuple[bool, str]:
    """
    Validate incoming payload.
    
    Returns (is_valid, error_message).
    """
    if not data:
        return False, "Empty payload"
    if len(data) > max_size:
        return False, f"Payload exceeds max size ({len(data)} > {max_size})"
    return True, ""
