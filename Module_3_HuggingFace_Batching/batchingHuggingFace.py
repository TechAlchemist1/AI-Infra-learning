from collections import deque
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
)

model.eval()

# an array for the batch
batchedRequest = deque()
waitingQueue = deque()
BATCH_CAPACITY = 10
#request object
class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 40


@app.get("/")
def root():
    return {"message": "LLM server is running"}

def batch():
    


# create a watcher of the batch
def batchWatcher():
    # determine if batch is the size needed
    while 1:

        if len(waitingQueue) >= BATCH_CAPACITY:
            batch()
        time.sleep(0.01)

@app.post('/spice')
def spice(request: GenerateRequest):
    # take this users request and make a job 








# a single one now lets do a batch
@app.post("/generate")
def generate(request: GenerateRequest):

    total_start = time.perf_counter()


    # -------------------------
    # 1. TOKENIZATION
    # -------------------------

    tokenize_start = time.perf_counter()

    inputs = tokenizer(
        request.prompt,
        return_tensors="pt"
    )

    tokenize_end = time.perf_counter()

    tokenize_time = tokenize_end - tokenize_start


    # -------------------------
    # 2. GENERATION
    # -------------------------

    generation_start = time.perf_counter()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=request.max_new_tokens,
            do_sample=False
        )

    generation_end = time.perf_counter()

    generation_time = generation_end - generation_start


    # -------------------------
    # 3. REMOVE PROMPT TOKENS
    # -------------------------

    input_length = inputs["input_ids"].shape[1]

    generated_ids = output_ids[0][input_length:]


    # -------------------------
    # 4. DECODING
    # -------------------------

    decode_start = time.perf_counter()

    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    )

    decode_end = time.perf_counter()

    decode_time = decode_end - decode_start


    # -------------------------
    # 5. TOTAL TIME
    # -------------------------

    total_end = time.perf_counter()

    total_time = total_end - total_start


    return {
        "response": generated_text,

        "metrics": {
            "tokenization_seconds": round(tokenize_time, 4),
            "generation_seconds": round(generation_time, 4),
            "decode_seconds": round(decode_time, 4),
            "total_seconds": round(total_time, 4),

            "input_tokens": input_length,
            "output_tokens": len(generated_ids)
        }
    }