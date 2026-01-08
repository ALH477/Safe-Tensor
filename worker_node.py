import argparse
import time
import logging
import threading
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from dcf_common import *

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [WORKER] %(message)s')
logger = logging.getLogger("WorkerNode")

# Config
LISTEN_PORT = 7779
LLM_MODEL_ID = "mistralai/Mistral-7B-v0.1"

class InferenceEngine:
    def __init__(self):
        logger.info(f"Loading Model: {LLM_MODEL_ID}...")
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_ID, quantization_config=quant, device_map="auto"
        )
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        logger.info("Model Loaded Successfully.")

    def run(self, prompt: str):
        # Simple generation params
        res = self.pipe(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)
        return res[0]['generated_text']

def heartbeat_loop(sock, head_ip, head_port):
    """Sends 'I am alive' packets to Head every 2s"""
    target = (head_ip, head_port)
    while True:
        # Payload is our listening port so Head knows where to send tasks
        payload = str(LISTEN_PORT).encode()
        msg = DCFMessage(MSG_HEARTBEAT, 0, DCFMessage.current_timestamp_micros(), payload)
        sock.send(msg, target)
        time.sleep(2.0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-ip", required=True, help="IP address of Head Node")
    args = parser.parse_args()

    # 1. Initialize Engine
    engine = InferenceEngine()

    # 2. Setup Network
    sock = AsyncUDPSocket(LISTEN_PORT)
    
    # 3. Start Heartbeat (Background)
    # Head listens for workers on 7778
    hb_thread = threading.Thread(target=heartbeat_loop, args=(sock, args.head_ip, 7778), daemon=True)
    hb_thread.start()

    logger.info(f"Worker Online. Listening on {LISTEN_PORT}. Head at {args.head_ip}:7778")

    # 4. Event Loop
    while True:
        packet = sock.recv()
        if packet:
            msg, _ = packet # We don't care about sender addr, we reply to Head
            
            if msg.msg_type == MSG_TASK:
                try:
                    prompt = msg.payload.decode('utf-8')
                    logger.info(f"Processing Task {msg.sequence}: {prompt[:20]}...")
                    
                    # Run Inference (Blocking)
                    start_ts = time.time()
                    result_text = engine.run(prompt)
                    duration = time.time() - start_ts
                    
                    # Send Result
                    payload = result_text.encode('utf-8')
                    resp = DCFMessage(MSG_RESULT, msg.sequence, DCFMessage.current_timestamp_micros(), payload)
                    
                    # Reply to Head (Port 7778)
                    sock.send(resp, (args.head_ip, 7778))
                    logger.info(f"Task {msg.sequence} Done ({duration:.2f}s). Result Sent.")
                    
                except Exception as e:
                    logger.error(f"Inference Failed: {e}")
                    # Optional: Send MSG_ERROR back
        
        time.sleep(0.001)

if __name__ == "__main__":
    main()
