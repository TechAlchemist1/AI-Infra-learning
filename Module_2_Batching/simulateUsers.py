"""Concurrent HTTP load generator for the batching Flask server.

Fires N users at POST /spice all at once and reports what each one actually
experienced. Deliberately uses only the response fields that BOTH backends
return (output_tokens), so the exact same script can be pointed at:

    Module_1_simple_Flask/backend.py   -- no queue, straight to ollama
    Module_2_Batching/batching.py      -- queue + batch scheduler

which is what makes the two runs comparable.

Everything else is measured client-side with a wall clock. That is on purpose:
server-reported timers (ollama's total_duration, etc) start when ollama gets
the request, so they cannot see time a job spent sitting in OUR waitingQueue.
The wall clock can.

Usage:
    # terminal 1
    python batching.py
    # terminal 2
    python simulateUsers.py --users 20
    python simulateUsers.py --users 20 --url http://127.0.0.1:5000/spice
"""

import argparse
import concurrent.futures
import statistics
import time

import requests

PROMPT = "Explain what a load balancer does in 30 words."


def send_request(i: int, url: str, prompt: str) -> dict:
    """One user. Records when it was sent so we can see queue effects."""
    submitted = time.time()
    try:
        response = requests.post(url, json={"prompt": prompt}, timeout=600)
        response.raise_for_status()
        data = response.json()
        error = None
    except Exception as exc:
        data, error = {}, repr(exc)

    finished = time.time()
    return {
        "i": i,
        "submitted": submitted,
        "finished": finished,
        "latency": finished - submitted,
        # both backends return these two; everything else differs
        "prompt_tokens": data.get("prompt_tokens"),
        "output_tokens": data.get("output_tokens"),
        "error": error,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=20,
                        help="concurrent requests fired at once")
    parser.add_argument("--url", default="http://127.0.0.1:5000/spice")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="server's MAXBATCHSIZE; only used to warn about "
                             "requests that would never form a full batch")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--label", default="batched")
    args = parser.parse_args()

    # batching.py only drains the queue once len(waitingQueue) >= MAXBATCHSIZE
    # and never flushes a partial batch, so a leftover remainder blocks on
    # jobEvent.done.wait() forever. Fail loudly instead of looking hung.
    if args.batch_size > 0 and args.users % args.batch_size:
        leftover = args.users % args.batch_size
        print(f"WARNING: {args.users} users with batch size {args.batch_size} "
              f"leaves {leftover} request(s) that never fill a batch.")
        print("         Those will hang until timeout. Use a multiple of "
              f"{args.batch_size}, or pass --batch-size 0 for an unbatched "
              "backend like Module 1.")
        print()

    print(f"firing {args.users} concurrent users at {args.url} ...")

    wall_start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.users) as pool:
        futures = [
            pool.submit(send_request, i, args.url, args.prompt)
            for i in range(args.users)
        ]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    wall = time.time() - wall_start

    report(results, wall, args)


def report(results: list, wall: float, args):
    ok = [r for r in results if r["error"] is None]
    failed = [r for r in results if r["error"] is not None]

    # completion order is the interesting part: with a batch scheduler users
    # finish in clumps of MAXBATCHSIZE, and each clump's latency is offset by
    # one full batch service time -- that stair-step IS the queueing cost.
    ordered = sorted(ok, key=lambda r: r["finished"])
    t0 = min(r["submitted"] for r in results) if results else 0.0

    header = f"{'rank':<6}{'user':>6}{'latency':>10}{'finished_at':>13}{'out_tok':>9}{'tok/s':>8}"
    print()
    print(header)
    print("-" * len(header))
    for rank, r in enumerate(ordered, 1):
        out = r["output_tokens"]
        rate = out / r["latency"] if out and r["latency"] else 0.0
        print(f"{rank:<6}{r['i']:>6}{r['latency']:>9.2f}s"
              f"{r['finished'] - t0:>12.2f}s"
              f"{out if out is not None else '-':>9}{rate:>8.1f}")

    if failed:
        print(f"\n{len(failed)} request(s) failed:")
        for r in failed[:5]:
            print(f"  user {r['i']}: {r['error']}")

    if not ok:
        print("\nno successful requests -- nothing to summarize")
        return

    latencies = sorted(r["latency"] for r in ok)
    total_out = sum(r["output_tokens"] or 0 for r in ok)

    print(f"\n--- {args.label}: {len(ok)}/{len(results)} ok ---")
    print(f"latency (s):  mean={statistics.mean(latencies):.2f} "
          f"median={statistics.median(latencies):.2f} "
          f"min={latencies[0]:.2f} max={latencies[-1]:.2f} "
          f"p95={latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]:.2f}")
    print(f"wall time:    {wall:.2f}s")
    print(f"throughput:   {len(ok) / wall:.2f} req/s, "
          f"{total_out / wall:.1f} output tok/s aggregate")
    print(f"output tokens: {total_out} total "
          f"({total_out / len(ok):.1f} avg/request)")


if __name__ == "__main__":
    main()

