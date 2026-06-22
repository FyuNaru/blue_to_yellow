#!/usr/bin/env python3
"""Analyze [mainflow_mem] logs emitted by verl_main_flow_memtrack.diff.

Example:
    python scripts/analyze_mainflow_mem.py train.log
    python scripts/analyze_mainflow_mem.py train.log --rank 0 --csv-prefix out/mainflow
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


LINE_RE = re.compile(
    r"\[mainflow_mem\]\s+"
    r"point=(?P<point>\S+)\s+"
    r"step=(?P<step>\d+)\s+"
    r"pid=(?P<pid>\d+)\s+"
    r"rank=(?P<rank>\S+)\s+"
    r"local_rank=(?P<local_rank>\S+)\s+"
    r"free=(?P<free>[0-9.]+)\s+GiB\s+"
    r"used=(?P<used>[0-9.]+)\s+GiB\s+"
    r"total=(?P<total>[0-9.]+)\s+GiB"
    r"(?:\s+delta_prev=(?P<delta_prev>-?[0-9.]+)\s+GiB\s+delta_step=(?P<delta_step>-?[0-9.]+)\s+GiB)?"
)


KNOWN_POINT_ORDER = [
    "compute_log_prob_begin",
    "compute_log_prob_end",
    "update_actor_begin",
    "update_actor_end",
    "update_weights_begin",
    "rollout_mode_begin",
    "before_wake_up_weights",
]


@dataclass
class Record:
    step: int
    rank: str
    local_rank: str
    pid: int
    point: str
    free: float
    used: float
    total: float
    delta_prev: float | None
    delta_step: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze [mainflow_mem] logs and summarize point usage and interval deltas by step/rank."
    )
    parser.add_argument("logfile", type=Path, help="Path to the training log file.")
    parser.add_argument("--rank", help="Only include one rank, e.g. 0.")
    parser.add_argument("--local-rank", help="Only include one local_rank, e.g. 0.")
    parser.add_argument("--pid", type=int, help="Only include one pid.")
    parser.add_argument(
        "--csv-prefix",
        type=Path,
        help="If set, write '<prefix>.points.csv' and '<prefix>.intervals.csv'.",
    )
    parser.add_argument(
        "--show-all-points",
        action="store_true",
        help="Print every discovered point instead of only the known ordered points first.",
    )
    return parser.parse_args()


def parse_records(logfile: Path) -> list[Record]:
    records: list[Record] = []
    with logfile.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            match = LINE_RE.search(line)
            if not match:
                continue
            groups = match.groupdict()
            records.append(
                Record(
                    step=int(groups["step"]),
                    rank=groups["rank"],
                    local_rank=groups["local_rank"],
                    pid=int(groups["pid"]),
                    point=groups["point"],
                    free=float(groups["free"]),
                    used=float(groups["used"]),
                    total=float(groups["total"]),
                    delta_prev=float(groups["delta_prev"]) if groups["delta_prev"] is not None else None,
                    delta_step=float(groups["delta_step"]) if groups["delta_step"] is not None else None,
                )
            )
    return records


def point_order(records: list[Record], show_all_points: bool) -> list[str]:
    seen = {record.point for record in records}
    ordered = [point for point in KNOWN_POINT_ORDER if point in seen]
    if show_all_points:
        extras = sorted(seen - set(ordered))
        ordered.extend(extras)
    return ordered


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate_points(records: list[Record]) -> dict[tuple[int, str, str], dict[str, dict[str, float]]]:
    grouped: dict[tuple[int, str, str], dict[str, list[Record]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        grouped[(record.step, record.rank, record.local_rank)][record.point].append(record)

    result: dict[tuple[int, str, str], dict[str, dict[str, float]]] = {}
    for key, point_map in grouped.items():
        result[key] = {}
        for point, items in point_map.items():
            result[key][point] = {
                "avg_used": average([item.used for item in items]),
                "avg_free": average([item.free for item in items]),
                "avg_total": average([item.total for item in items]),
                "avg_delta_prev": average([item.delta_prev for item in items if item.delta_prev is not None]),
                "avg_delta_step": average([item.delta_step for item in items if item.delta_step is not None]),
                "samples": float(len(items)),
            }
    return result


def aggregate_points_by_step(
    aggregated: dict[tuple[int, str, str], dict[str, dict[str, float]]],
) -> dict[int, dict[str, dict[str, float]]]:
    by_step_raw: dict[int, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for (step, _rank, _local_rank), point_map in aggregated.items():
        for point, stats in point_map.items():
            by_step_raw[step][point]["avg_used"].append(stats["avg_used"])
            by_step_raw[step][point]["avg_free"].append(stats["avg_free"])
            by_step_raw[step][point]["avg_total"].append(stats["avg_total"])
            by_step_raw[step][point]["samples"].append(stats["samples"])

    by_step: dict[int, dict[str, dict[str, float]]] = {}
    for step, point_map in by_step_raw.items():
        by_step[step] = {}
        for point, stats in point_map.items():
            by_step[step][point] = {
                "avg_used": average(stats["avg_used"]),
                "avg_free": average(stats["avg_free"]),
                "avg_total": average(stats["avg_total"]),
                "contributors": float(len(stats["avg_used"])),
                "samples": sum(stats["samples"]),
            }
    return by_step


def compute_intervals(
    aggregated: dict[tuple[int, str, str], dict[str, dict[str, float]]],
    ordered_points: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (step, rank, local_rank), point_map in sorted(aggregated.items()):
        prev_point = None
        prev_used = None
        for point in ordered_points:
            if point not in point_map:
                continue
            used = point_map[point]["avg_used"]
            if prev_point is not None and prev_used is not None:
                rows.append(
                    {
                        "step": step,
                        "rank": rank,
                        "local_rank": local_rank,
                        "interval": f"{prev_point}->{point}",
                        "start_point": prev_point,
                        "end_point": point,
                        "start_used_gib": prev_used,
                        "end_used_gib": used,
                        "delta_interval_gib": used - prev_used,
                    }
                )
            prev_point = point
            prev_used = used
    return rows


def aggregate_intervals_by_step(interval_rows: list[dict[str, object]]) -> dict[int, dict[str, float]]:
    raw: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in interval_rows:
        raw[int(row["step"])][str(row["interval"])].append(float(row["delta_interval_gib"]))

    result: dict[int, dict[str, float]] = {}
    for step, interval_map in raw.items():
        result[step] = {}
        for interval, values in interval_map.items():
            result[step][interval] = average(values)
    return result


def print_points_table(
    aggregated: dict[tuple[int, str, str], dict[str, dict[str, float]]],
    ordered_points: list[str],
) -> None:
    print("== Point Avg Used (GiB) ==")
    header = ["step", "rank", "local_rank"] + ordered_points
    print("\t".join(header))
    for (step, rank, local_rank), point_map in sorted(aggregated.items()):
        row = [str(step), rank, local_rank]
        for point in ordered_points:
            value = point_map.get(point, {}).get("avg_used")
            row.append("" if value is None else f"{value:.2f}")
        print("\t".join(row))
    print()


def print_step_points_table(
    aggregated_by_step: dict[int, dict[str, dict[str, float]]],
    ordered_points: list[str],
) -> None:
    print("== Step Avg Used Across All Ranks (GiB) ==")
    header = ["step"] + ordered_points
    print("\t".join(header))
    for step in sorted(aggregated_by_step):
        point_map = aggregated_by_step[step]
        row = [str(step)]
        for point in ordered_points:
            value = point_map.get(point, {}).get("avg_used")
            row.append("" if value is None else f"{value:.2f}")
        print("\t".join(row))
    print()


def print_interval_table(interval_rows: list[dict[str, object]], ordered_points: list[str]) -> None:
    print("== Interval Avg Delta (GiB) ==")
    interval_order = []
    for idx in range(1, len(ordered_points)):
        interval_order.append(f"{ordered_points[idx - 1]}->{ordered_points[idx]}")
    by_key: dict[tuple[int, str, str], dict[str, float]] = defaultdict(dict)
    for row in interval_rows:
        by_key[(int(row["step"]), str(row["rank"]), str(row["local_rank"]))][str(row["interval"])] = float(
            row["delta_interval_gib"]
        )

    header = ["step", "rank", "local_rank"] + interval_order
    print("\t".join(header))
    for key in sorted(by_key):
        step, rank, local_rank = key
        row = [str(step), rank, local_rank]
        for interval in interval_order:
            value = by_key[key].get(interval)
            row.append("" if value is None else f"{value:.2f}")
        print("\t".join(row))
    print()


def print_step_interval_table(
    intervals_by_step: dict[int, dict[str, float]],
    ordered_points: list[str],
) -> None:
    print("== Step Avg Delta Across All Ranks (GiB) ==")
    interval_order = [f"{ordered_points[idx - 1]}->{ordered_points[idx]}" for idx in range(1, len(ordered_points))]
    header = ["step"] + interval_order
    print("\t".join(header))
    for step in sorted(intervals_by_step):
        interval_map = intervals_by_step[step]
        row = [str(step)]
        for interval in interval_order:
            value = interval_map.get(interval)
            row.append("" if value is None else f"{value:.2f}")
        print("\t".join(row))
    print()


def write_csvs(
    prefix: Path,
    aggregated: dict[tuple[int, str, str], dict[str, dict[str, float]]],
    interval_rows: list[dict[str, object]],
) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    points_path = prefix.with_suffix(".points.csv")
    intervals_path = prefix.with_suffix(".intervals.csv")
    step_points_path = prefix.with_suffix(".step_points.csv")
    step_intervals_path = prefix.with_suffix(".step_intervals.csv")

    with points_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "step",
                "rank",
                "local_rank",
                "point",
                "avg_used_gib",
                "avg_free_gib",
                "avg_total_gib",
                "avg_delta_prev_gib",
                "avg_delta_step_gib",
                "samples",
            ]
        )
        for (step, rank, local_rank), point_map in sorted(aggregated.items()):
            for point, stats in sorted(point_map.items()):
                writer.writerow(
                    [
                        step,
                        rank,
                        local_rank,
                        point,
                        f"{stats['avg_used']:.6f}",
                        f"{stats['avg_free']:.6f}",
                        f"{stats['avg_total']:.6f}",
                        f"{stats['avg_delta_prev']:.6f}",
                        f"{stats['avg_delta_step']:.6f}",
                        int(stats["samples"]),
                    ]
                )

    with intervals_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "step",
                "rank",
                "local_rank",
                "interval",
                "start_point",
                "end_point",
                "start_used_gib",
                "end_used_gib",
                "delta_interval_gib",
            ],
        )
        writer.writeheader()
        for row in interval_rows:
            out = dict(row)
            out["start_used_gib"] = f"{float(out['start_used_gib']):.6f}"
            out["end_used_gib"] = f"{float(out['end_used_gib']):.6f}"
            out["delta_interval_gib"] = f"{float(out['delta_interval_gib']):.6f}"
            writer.writerow(out)

    aggregated_by_step = aggregate_points_by_step(aggregated)
    intervals_by_step = aggregate_intervals_by_step(interval_rows)

    with step_points_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "point", "avg_used_gib", "avg_free_gib", "avg_total_gib", "contributors", "samples"])
        for step, point_map in sorted(aggregated_by_step.items()):
            for point, stats in sorted(point_map.items()):
                writer.writerow(
                    [
                        step,
                        point,
                        f"{stats['avg_used']:.6f}",
                        f"{stats['avg_free']:.6f}",
                        f"{stats['avg_total']:.6f}",
                        int(stats["contributors"]),
                        int(stats["samples"]),
                    ]
                )

    with step_intervals_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "interval", "avg_delta_interval_gib"])
        for step, interval_map in sorted(intervals_by_step.items()):
            for interval, value in sorted(interval_map.items()):
                writer.writerow([step, interval, f"{value:.6f}"])

    print(f"Wrote {points_path}")
    print(f"Wrote {intervals_path}")
    print(f"Wrote {step_points_path}")
    print(f"Wrote {step_intervals_path}")


def main() -> int:
    args = parse_args()
    records = parse_records(args.logfile)
    if args.rank is not None:
        records = [record for record in records if record.rank == args.rank]
    if args.local_rank is not None:
        records = [record for record in records if record.local_rank == args.local_rank]
    if args.pid is not None:
        records = [record for record in records if record.pid == args.pid]

    if not records:
        print("No [mainflow_mem] records found with the given filters.")
        return 1

    ordered_points = point_order(records, args.show_all_points)
    aggregated = aggregate_points(records)
    interval_rows = compute_intervals(aggregated, ordered_points)
    aggregated_by_step = aggregate_points_by_step(aggregated)
    intervals_by_step = aggregate_intervals_by_step(interval_rows)

    print_step_points_table(aggregated_by_step, ordered_points)
    print_step_interval_table(intervals_by_step, ordered_points)
    print_points_table(aggregated, ordered_points)
    print_interval_table(interval_rows, ordered_points)

    if args.csv_prefix is not None:
        write_csvs(args.csv_prefix, aggregated, interval_rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
