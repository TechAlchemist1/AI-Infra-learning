"""Single-user baseline benchmark for the local Ollama model.

Measures what one user, one request at a time, actually gets: cold-start
load time, prefill throughput, and decode throughput, across a few prompt
sizes. This is the number to beat before adding any multi-user serving
techniques (batching, queuing, concurrent workers, etc).

Calls the `ollama` client directly rather than going through the Flask
/spice route -- localhost HTTP + JSON overhead is ~1ms against multi-second
generations, and calling ollama directly lets us pass tuning options
(keep_alive, num_thread, ...) without changing backend.py.

Usage:
    python benchmarks/baseline.py --label baseline
    python benchmarks/baseline.py --label optimized --num-thread 8 --keep-alive 30m
"""

import argparse
import csv
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from ollama import ChatResponse, chat

RESULTS_DIR = Path(__file__).parent / "results"
HISTORY_CSV = Path(__file__).parent / "history.csv"
MODEL = "qwen2.5:3b"
NS = 1_000_000_000

PROMPTS = {
    "short": (
        "Explain what a load balancer does."
    ),
    "medium": (
        "A backend system receives HTTP requests from many users. "
        "The system has three identical application servers behind a load balancer. "
        "Each server can process requests independently, but traffic is currently "
        "distributed unevenly and one server often becomes overloaded while the "
        "others remain underutilized. "
        "Explain what a load balancer does in this system."
    ),
    "long": (
        "A backend system receives HTTP requests from thousands of users. "
        "Traffic enters through an API gateway and is then routed toward three "
        "identical application servers. Each server runs the same application code "
        "and can independently process user requests. The servers communicate with "
        "a shared database and expose health-check endpoints. During periods of high "
        "traffic, the current routing mechanism frequently sends too many requests "
        "to one server while the other servers still have available capacity. This "
        "causes increased latency, request failures, and poor overall resource "
        "utilization. The engineering team wants traffic distributed more evenly, "
        "unhealthy servers to stop receiving requests, and new replicas to begin "
        "receiving traffic automatically when added. "
        "Explain what a load balancer does in this system."
    ),
}

# Fixed output cap so every trial decodes the same number of tokens --
# isolates prefill (input length) as the only thing that varies between
# short/medium/long, instead of letting decode length confound the numbers.
BASE_OPTIONS = {"num_predict": 50}

RAW_METRICS = [
    "latency", "total_duration", "load_duration",
    "prompt_tokens", "prompt_duration",
    "output_tokens", "decode_duration",
]
DERIVED_METRICS = ["decode_tokens_per_sec", "prefill_tokens_per_sec"]
ALL_METRICS = RAW_METRICS + DERIVED_METRICS


def call(prompt: str, options: dict, keep_alive) -> dict:
    start = time.time()
    response: ChatResponse = chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options=options,
        keep_alive=keep_alive,
    )
    latency = time.time() - start

    row = {
        "latency": latency,
        "total_duration": response["total_duration"] / NS,
        "load_duration": response["load_duration"] / NS,
        "prompt_tokens": response["prompt_eval_count"],
        "prompt_duration": response["prompt_eval_duration"] / NS,
        "output_tokens": response["eval_count"],
        "decode_duration": response["eval_duration"] / NS,
    }
    row["decode_tokens_per_sec"] = (
        row["output_tokens"] / row["decode_duration"] if row["decode_duration"] else 0.0
    )
    row["prefill_tokens_per_sec"] = (
        row["prompt_tokens"] / row["prompt_duration"] if row["prompt_duration"] else 0.0
    )
    return row


def summarize(values: list) -> dict:
    values = sorted(values)
    n = len(values)
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": values[0],
        "max": values[-1],
        "p95": values[min(n - 1, int(n * 0.95))],
    }


def run(trials: int, label: str, options: dict, keep_alive) -> dict:
    print(f"\n=== Run: {label} (options={options}, keep_alive={keep_alive}) ===")

    print("cold-start call (model likely not resident)...")
    cold = call(next(iter(PROMPTS.values())), options, keep_alive)
    print(f"  load_duration={cold['load_duration']:.2f}s total={cold['total_duration']:.2f}s")

    raw = {name: [] for name in PROMPTS}
    for name, prompt in PROMPTS.items():
        for i in range(trials):
            row = call(prompt, options, keep_alive)
            raw[name].append(row)
            print(
                f"  [{name}] trial {i + 1}/{trials}: "
                f"latency={row['latency']:.2f}s decode={row['decode_tokens_per_sec']:.1f} tok/s "
                f"prefill={row['prefill_tokens_per_sec']:.1f} tok/s"
            )

    aggregate = {
        name: {metric: summarize([r[metric] for r in rows]) for metric in ALL_METRICS}
        for name, rows in raw.items()
    }

    return {
        "label": label,
        "model": MODEL,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trials_per_prompt": trials,
        "options": options,
        "keep_alive": keep_alive,
        "cold_start": cold,
        "raw": raw,
        "aggregate": aggregate,
    }


def print_summary(result: dict):
    print(f"\n--- Summary: {result['label']} ---")
    print(f"cold start load_duration: {result['cold_start']['load_duration']:.2f}s")
    for name, agg in result["aggregate"].items():
        print(f"[{name}]")
        print(
            f"  latency (s):    mean={agg['latency']['mean']:.2f} "
            f"median={agg['latency']['median']:.2f} p95={agg['latency']['p95']:.2f}"
        )
        print(
            f"  decode tok/s:   mean={agg['decode_tokens_per_sec']['mean']:.1f} "
            f"p95={agg['decode_tokens_per_sec']['p95']:.1f}"
        )
        print(
            f"  prefill tok/s:  mean={agg['prefill_tokens_per_sec']['mean']:.1f}"
        )


def save_json(result: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fname = RESULTS_DIR / f"{result['label']}_{int(time.time())}.json"
    fname.write_text(json.dumps(result, indent=2))
    print(f"\nSaved raw results -> {fname}")
    return fname


def append_history(result: dict):
    is_new = not HISTORY_CSV.exists()
    fields = [
        "timestamp", "label", "model", "options", "keep_alive",
        "cold_start_load_s",
        "short_latency_mean", "short_decode_tok_s",
        "medium_latency_mean", "medium_decode_tok_s",
        "long_latency_mean", "long_decode_tok_s",
    ]
    with open(HISTORY_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if is_new:
            writer.writeheader()
        writer.writerow({
            "timestamp": result["timestamp"],
            "label": result["label"],
            "model": result["model"],
            "options": json.dumps(result["options"]),
            "keep_alive": result["keep_alive"],
            "cold_start_load_s": f"{result['cold_start']['load_duration']:.3f}",
            "short_latency_mean": f"{result['aggregate']['short']['latency']['mean']:.3f}",
            "short_decode_tok_s": f"{result['aggregate']['short']['decode_tokens_per_sec']['mean']:.1f}",
            "medium_latency_mean": f"{result['aggregate']['medium']['latency']['mean']:.3f}",
            "medium_decode_tok_s": f"{result['aggregate']['medium']['decode_tokens_per_sec']['mean']:.1f}",
            "long_latency_mean": f"{result['aggregate']['long']['latency']['mean']:.3f}",
            "long_decode_tok_s": f"{result['aggregate']['long']['decode_tokens_per_sec']['mean']:.1f}",
        })
    print(f"Appended summary -> {HISTORY_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--num-thread", type=int, default=None)
    parser.add_argument("--keep-alive", default=None, help='e.g. "30m", "-1" to keep forever')
    args = parser.parse_args()

    options = dict(BASE_OPTIONS)
    if args.num_thread is not None:
        options["num_thread"] = args.num_thread

    result = run(args.trials, args.label, options, args.keep_alive)
    print_summary(result)
    save_json(result)
    append_history(result)
