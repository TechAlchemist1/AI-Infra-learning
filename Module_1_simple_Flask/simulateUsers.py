import concurrent.futures
import requests
import time

URL = "http://127.0.0.1:5000/spice"
NS = 1_000_000_000

def send_request(i):
    start = time.time()

    response = requests.post(
        URL,
        json={"prompt": "Explain what a load balancer does in 30 words."}
    )

    latency = time.time() - start

    return i, latency, response.json()

users = 50

with concurrent.futures.ThreadPoolExecutor(max_workers=users) as executor:
    futures = [
        executor.submit(send_request, i)
        for i in range(users)
    ]

    header = (
        f"{'user':<6}{'latency':>9}{'server':>9}{'http':>7}{'queue':>8}"
        f"{'load':>7}{'prefill':>9}{'decode':>8}{'in':>6}{'out':>6}{'tok/s':>8}"
    )
    print(header)
    print("-" * len(header))

    for future in concurrent.futures.as_completed(futures):
        i, latency, data = future.result()

        # backend.py returns total_duration in raw nanoseconds while the
        # other *_duration fields are already seconds, so convert it here
        server = data["total_duration"] / NS

        # ollama starts its total_duration clock when the request ARRIVES,
        # not when the scheduler gives it a slot -- so any time spent
        # waiting for a slot is already inside `server`. that makes
        # latency - server just the local HTTP round-trip, not queue time.
        http = latency - server

        load = data["load_duration"]
        prefill = data["prompt_duration"]
        decode = data["decode_duration"]

        # the real queue signal: whatever ollama spent on this request that
        # it did NOT attribute to loading, prefill, or decode. with
        # OLLAMA_NUM_PARALLEL=1 this grows by ~one service time per queue
        # position. note it is a residual, so it also absorbs unitemized
        # overhead (templating, tokenize/detokenize) -- run with users = 1
        # to measure that floor and subtract it.
        queue = server - (load + prefill + decode)

        decode_rate = data["output_tokens"] / decode if decode else 0

        print(
            f"{i:<6}{latency:>8.2f}s{server:>8.2f}s{http:>6.2f}s{queue:>7.2f}s"
            f"{load:>6.2f}s{prefill:>8.2f}s"
            f"{decode:>7.2f}s{data['prompt_tokens']:>6}"
            f"{data['output_tokens']:>6}{decode_rate:>8.1f}"
        )