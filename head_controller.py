import time
import logging
import threading
from dcf_common import *

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s [HEAD] %(message)s')
logger = logging.getLogger("HeadNode")

# Ports
PORT_CLIENT_INGRESS = 7777  # From Shim/Client
PORT_WORKER_BUS     = 7778  # Internal Mesh

class WorkerRegistry:
    def __init__(self):
        self.workers = {} # { (ip, port): last_seen_ts }
        self.lock = threading.Lock()
        self.rr_index = 0

    def register(self, addr):
        with self.lock:
            if addr not in self.workers:
                logger.info(f"New Worker Registered: {addr}")
            self.workers[addr] = time.time()

    def get_next_worker(self):
        """Round-robin selection of active workers"""
        with self.lock:
            # Prune dead workers (>10s silence)
            now = time.time()
            active = [addr for addr, ts in self.workers.items() if now - ts < 10]
            
            if not active:
                return None
            
            self.rr_index = (self.rr_index + 1) % len(active)
            return active[self.rr_index]

def main():
    # 1. Start Sockets
    sock_client = AsyncUDPSocket(PORT_CLIENT_INGRESS)
    sock_internal = AsyncUDPSocket(PORT_WORKER_BUS)
    
    registry = WorkerRegistry()
    
    # Map Sequence ID -> Client Address (to route responses back)
    # Dict[int, Tuple[str, int]]
    request_map = {} 

    logger.info("HydraMesh Head Online. Waiting for traffic...")

    while True:
        # --- A. Handle Internal Traffic (Workers) ---
        packet = sock_internal.recv()
        if packet:
            msg, addr = packet
            
            if msg.msg_type == MSG_HEARTBEAT:
                # Payload contains the worker's LISTEN port (e.g., 7779)
                # addr is (ip, ephemeral_port). We need (ip, 7779)
                try:
                    worker_port = int(msg.payload.decode())
                    worker_addr = (addr[0], worker_port)
                    registry.register(worker_addr)
                except:
                    pass

            elif msg.msg_type == MSG_RESULT:
                # Worker finished a task.
                # Look up who asked for this sequence
                client_addr = request_map.pop(msg.sequence, None)
                if client_addr:
                    # Forward back to Client (shim)
                    # Convert to AUDIO type (as expected by shim) or keep generic
                    resp = DCFMessage(MSG_TASK, msg.sequence, msg.timestamp, msg.payload)
                    sock_client.send(resp, client_addr)
                    logger.info(f"Task {msg.sequence} completed -> {client_addr}")
                else:
                    logger.warning(f"Received result for unknown sequence: {msg.sequence}")

        # --- B. Handle External Traffic (Clients) ---
        packet = sock_client.recv()
        if packet:
            msg, client_addr = packet
            
            # 1. Store route for return path
            request_map[msg.sequence] = client_addr
            
            # 2. Select Worker
            worker_addr = registry.get_next_worker()
            
            if worker_addr:
                # 3. Forward to Worker
                # We reuse the sequence ID to track it
                fwd_msg = DCFMessage(MSG_TASK, msg.sequence, msg.timestamp, msg.payload)
                
                # Worker listens on 7779, so we send there.
                # Note: worker_addr is already (ip, 7779) from registry
                sock_internal.send(fwd_msg, worker_addr)
                logger.info(f"Routed Task {msg.sequence} -> {worker_addr}")
            else:
                logger.error("No workers available! Dropping packet.")

        # Sleep briefly to prevent 100% CPU usage on idle
        time.sleep(0.0001)

if __name__ == "__main__":
    main()
