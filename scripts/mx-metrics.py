#!/usr/bin/env python3
"""Summarise the GPU metric tables of an nsys sqlite export.

Usage: mx-metrics.py <report.sqlite>

Prints the metric-table schema, then the metrics whose names mention the memory system, with their
mean value over the capture. Nothing here is nsys-specific beyond the table layout, which this script
discovers rather than assumes.
"""

import sqlite3
import sys
from collections import defaultdict


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        return
    con = sqlite3.connect(sys.argv[1])
    cur = con.cursor()

    tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    metric_tables = [t for t in tables if "METRIC" in t.upper()]
    print("metric tables:", metric_tables)

    for t in metric_tables:
        cols = [c[1] for c in cur.execute("PRAGMA table_info(%s)" % t)]
        n = cur.execute("SELECT COUNT(*) FROM %s" % t).fetchone()[0]
        print(" ", t, cols, n)

    # the usual layout: GPU_METRICS holds (typeId, metricId, value) and a lookup table names them
    lookups = [t for t in tables if "METRIC_NAME" in t.upper() or "TARGET_INFO_GPU_METRIC" in t.upper()]
    print("name lookup tables:", lookups)
    for t in lookups:
        cols = [c[1] for c in cur.execute("PRAGMA table_info(%s)" % t)]
        print(" ", t, cols)
        for row in cur.execute("SELECT * FROM %s LIMIT 8" % t):
            print("   ", row)

    gpu_metrics = [t for t in tables if t.upper() == "GPU_METRICS"]
    if not gpu_metrics:
        return
    name_table = None
    for candidate in ("TARGET_INFO_GPU_METRICS", "GPU_METRIC_NAMES", "METRIC_NAMES"):
        if candidate in tables:
            name_table = candidate
            break
    if name_table is None:
        print("no name table found")
        return

    names = {}
    for mid, mname in cur.execute("SELECT metricId, name FROM %s" % name_table):
        names[mid] = mname

    acc = defaultdict(lambda: [0.0, 0, 0.0])
    for type_id, metric_id, value in cur.execute("SELECT typeId, metricId, value FROM GPU_METRICS"):
        name = names.get(metric_id)
        if name is None:
            continue
        entry = acc[name]
        entry[0] += value
        entry[1] += 1
        entry[2] = value  # last sample

    keys = [k for k in sorted(acc) if any(s in k for s in ("dram", "lts", "l1tex", "sm__", "gpu__",
                                                           "gpc__", "l2"))]
    print()
    print("%d distinct metrics, %d of them memory or throughput" % (len(acc), len(keys)))
    print("%-78s %12s" % ("metric", "mean"))
    for k in keys:
        total, n, last = acc[k]
        print("%-78s %12.4f" % (k, total / n))


if __name__ == "__main__":
    main()
