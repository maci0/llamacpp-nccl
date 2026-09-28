#!/usr/bin/env python3
"""Join the op replay times with the op multiplicities to get a per-pass budget.

Usage: budget.py <export.log> <replay.log> [--top N]

The export log holds MULT lines with the multiplicity of each unique op in the
generation graph. The replay log holds one timing per unique op. Product of the
two, aggregated, is what a pass spends in each op family.
"""

import re
import sys
from collections import defaultdict

MULT_RE = re.compile(r"^MULT (\d+) (\S+) (\d+) (\d+) (\d+) (\d+) (\d+)$")
REPLAY_RE = re.compile(
    r"(\w+)\(name=[^,]*,type=(\w+),ne=\[(-?\d+),(-?\d+),(-?\d+),(-?\d+)\][^\n]*\)")
TIME_RE = re.compile(r"([\d.]+) us/run")
KB_RE = re.compile(r"([\d.]+) kB/run")

# the exporter prints the output tensor's type as its enum value
GGML_TYPE_NAMES = {0: "f32", 1: "f16", 2: "q4_0", 3: "q4_1", 8: "q8_0", 24: "i8",
                   25: "i16", 26: "i32", 27: "i64", 28: "f64", 30: "bf16", 39: "mxfp4"}


def parse_mult(path):
    out = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = MULT_RE.match(line.strip())
            if not m:
                continue
            count, op, ty = int(m.group(1)), m.group(2), int(m.group(3))
            ty_name = GGML_TYPE_NAMES.get(ty)
            if ty_name is None:
                continue
            ne = tuple(int(m.group(i)) for i in range(4, 8))
            key = (op, ty_name, ne)
            out[key] = out.get(key, 0) + count
    return out


def parse_replay(path):
    out = {}
    pending = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = REPLAY_RE.search(line)
            if m:
                pending = (m.group(1), m.group(2), tuple(int(m.group(i)) for i in range(3, 7)))
                continue
            t = TIME_RE.search(line)
            if t and pending is not None:
                key = pending
                pending = None
                # keep the fastest measurement when a case appears twice
                us = float(t.group(1))
                if key not in out or us < out[key]:
                    out[key] = us
    return out


def parse_replay_kb(path):
    """per-signature bytes per run, from the kB/run column of the perf console output"""
    out = {}
    pending = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = REPLAY_RE.search(line)
            if m:
                pending = (m.group(1), m.group(2), tuple(int(m.group(i)) for i in range(3, 7)))
                continue
            t = KB_RE.search(line)
            if t and pending is not None:
                key = pending
                pending = None
                kb = float(t.group(1))
                if key not in out or kb < out[key]:
                    out[key] = kb
    return out


def bytes_budget(export, replay, top):
    mult = parse_mult(export)
    kbs = parse_replay_kb(replay)

    by_op = defaultdict(float)
    by_op_nodes = defaultdict(int)
    total_kb = 0.0
    missing = 0
    missing_nodes = 0
    rows = []
    for key, count in mult.items():
        kb = kbs.get(key)
        if kb is None:
            missing += 1
            missing_nodes += count
            continue
        kb_pass = count * kb
        rows.append((kb_pass, count, key))
        by_op[key[0]] += kb_pass
        by_op_nodes[key[0]] += count
        total_kb += kb_pass

    print("traffic of one exported graph, from bytes/run x multiplicities")
    print("(%d signatures, %d without a replay line)" % (len(mult), missing))
    print()
    print("%-18s %8s %12s %10s" % ("op family", "count", "GB/pass", "% of sum"))
    for op, kb in sorted(by_op.items(), key=lambda kv: -kv[1])[:top]:
        print("%-18s %8d %12.2f %9.1f %%" % (op, by_op_nodes[op], kb / 1024.0 / 1024.0,
                                             100.0 * kb / total_kb))
    print()
    print("sum over matched ops: %.2f GB per pass" % (total_kb / 1024.0 / 1024.0))
    if missing_nodes:
        print("note: %d nodes have no replay line" % missing_nodes)
    print()
    print("largest single signatures:")
    for kb, count, key in sorted(rows, key=lambda r: -r[0])[:top]:
        op, ty, ne = key
        print("  %10.1f MB  x%-6d %-14s ne=[%d,%d,%d,%d]" % (kb / 1024.0, count, op, ne[0], ne[1],
                                                              ne[2], ne[3]))


def main():
    export, replay = sys.argv[1], sys.argv[2]
    top = 15
    if "--top" in sys.argv:
        top = int(sys.argv[sys.argv.index("--top") + 1])
    if "--bytes" in sys.argv:
        bytes_budget(export, replay, top)
        return

    mult = parse_mult(export)
    times = parse_replay(replay)

    by_op = defaultdict(float)
    by_op_nodes = defaultdict(int)
    total = 0.0
    unmatched_us = 0.0
    unmatched_nodes = 0
    missing = 0
    rows = []
    for key, count in mult.items():
        us = times.get(key)
        if us is None:
            missing += 1
            unmatched_nodes += count
            continue
        timed = count * us / 1000.0  # ms per pass
        rows.append((timed, count, key))
        by_op[key[0]] += timed
        by_op_nodes[key[0]] += count
        total += timed

    print("pass budget from %d op signatures (%d in the graph, %d without a timing)"
          % (len(mult), sum(mult.values()), missing))
    print()
    print("%-18s %8s %10s %10s" % ("op family", "count", "ms/pass", "% of sum"))
    for op, ms in sorted(by_op.items(), key=lambda kv: -kv[1])[:top]:
        print("%-18s %8d %10.1f %9.1f %%" % (op, by_op_nodes[op], ms, 100.0 * ms / total))
    print()
    print("sum over timed ops: %.1f ms per pass" % total)
    print()
    print("largest single signatures:")
    for ms, count, key in sorted(rows, key=lambda r: -r[0])[:top]:
        op, ty, ne = key
        print("  %8.1f ms  x%-5d %-14s ne=[%d,%d,%d,%d]" % (ms, count, op, ne[0], ne[1], ne[2], ne[3]))
    if unmatched_nodes:
        print()
        print("note: %d nodes have no timing (unsupported or skipped cases), %.1f ms unaccounted"
              % (unmatched_nodes, unmatched_us))


if __name__ == "__main__":
    main()
