#!/usr/bin/env python3
"""Measure a llama.cpp server on this pair of Sparks under the reference protocol.

The workload is the one the vLLM numbers in docs/targets.md are quoted from, so a
number produced here can be set against them directly: chat completion, the shared
BST coding prompt, temperature 0.7, seed 1234, thinking off, 512-token generations,
per-stream and aggregate throughput at each concurrency level.

Why a script and not a shell loop: single passes on this rig swing by tens of percent
(this file's own bench records show one arm at 27.5-35.4 tok/s over six runs), and the
per-connection graph counters in the server log read very differently between runs of
one configuration. A number is only worth quoting with its spread, so every level is
run --reps times and reported as a median with its range.

  scripts/bench-spark.py --base http://127.0.0.1:8081/v1 --reps 3
  scripts/bench-spark.py --base http://127.0.0.1:8000/v1 --model deepseek-v4-flash \
      --reps 3 --levels 1 3 5 6 --label refg

Stdlib only. Run it on the node that hosts the server, never from the laptop: latency
over the network skews the per-stream figure.
"""
import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

GOLDEN_PROMPT = (
    "Write a Python binary search tree with insert, delete, and inorder "
    "traversal; explain each method."
)

GATES = [("france", "The capital of France is", 6), ("9x8", "9x8=", 6)]


def one_request(base, model, prompt, max_tokens, temp, seed, thinking, chat):
    if chat:
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temp,
            "chat_template_kwargs": {"thinking": thinking},
            "seed": seed,
        }
        url = f"{base}/chat/completions"
    else:
        body = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
                "temperature": 0.0, "seed": seed}
        url = f"{base}/completions"
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.loads(r.read().decode())
    return d, time.time() - t0


def sweep(args, level):
    def run(i):
        seed = None if args.seed is None else args.seed + i
        return one_request(args.base, args.model, args.prompt, args.max_tokens,
                           args.temp, seed, args.thinking, args.chat)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=level) as ex:
        res = list(ex.map(run, range(level)))
    wall = time.time() - t0
    toks = sum(d.get("usage", {}).get("completion_tokens", 0) for d, _ in res)
    return toks / wall, wall, toks


def gates(args):
    """Two greedy questions the model must answer correctly; see the bench records."""
    out = []
    for name, prompt, n in GATES:
        d, _ = one_request(args.base, args.model, prompt, n,
                           0.0, None, args.thinking, chat=False)
        text = d["choices"][0].get("text") or d["choices"][0]["message"]["content"]
        out.append((name, text))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8081/v1")
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--prompt", default=GOLDEN_PROMPT)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--levels", nargs="+", type=int, default=[1, 3, 5, 6])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warm", type=int, default=1, help="warm passes before the timed ones")
    ap.add_argument("--completions", action="store_true", help="raw completions, greedy")
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    args.chat = not args.completions

    print(f"label={args.label} base={args.base} model={args.model} "
          f"max_tokens={args.max_tokens} temp={args.temp} seed={args.seed} "
          f"thinking={args.thinking} reps={args.reps}")

    for name, text in gates(args):
        print(f"gate_{name}: {text!r}")

    for _ in range(args.warm):
        for level in args.levels:
            sweep(args, level)

    for level in args.levels:
        got = []
        for _ in range(args.reps):
            per_stream, wall, toks = sweep(args, level)
            got.append((per_stream, wall, toks))
        ps = sorted(x[0] for x in got)
        median = statistics.median(ps)
        print(f"c{level:<3} median={median:6.1f} tok/s  min={ps[0]:6.1f}  max={ps[-1]:6.1f}  "
              f"spread={100 * (ps[-1] - ps[0]) / median:4.1f} %  "
              f"walls={' '.join(f'{x[1]:.1f}' for x in sorted(got, key=lambda y: y[1]))}")


if __name__ == "__main__":
    main()
