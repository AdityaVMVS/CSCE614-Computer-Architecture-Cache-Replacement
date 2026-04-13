#!/usr/bin/env python
import argparse
import csv
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import math

CORES = ["westmere-{}".format(i) for i in range(8)]
POLICIES = ["LRU", "LFU", "SRRIP", "PACIPV", "Rand", "NRU", "SHIP", "DRRIP", "TreeLRU"]
BENCHMARKS = [
    "bzip2", 
    "calculix", 
    "cactusADM", 
    "gcc", 
    "hmmer", 
    "lbm", 
    "libquantum", 
    "mcf", 
    "namd", 
    "sjeng", 
    "soplex",
    "xalan",
    "blackscholes_8c_simlarge",
    "bodytrack_8c_simlarge",
    "canneal_8c_simlarge",
    "fluidanimate_8c_simlarge",
    "streamcluster_8c_simlarge",
    "swaptions_8c_simlarge", 
    "x264_8c_simlarge"
]

############################################################
# PARSERS
############################################################

def parse_core_blocks(lines):
    core_stats = {c: {"cycles": 0, "ccycles": 0, "instrs": 0} for c in CORES}
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        m = re.match(r'^(\s*)westmere-(\d+):', line)
        if m:
            indent = len(m.group(1))
            core_idx = int(m.group(2))
            core_name = f"westmere-{core_idx}"
            j = i + 1
            while j < n:
                next_line = lines[j]
                leading = len(next_line) - len(next_line.lstrip(' '))
                if next_line.strip() == "":
                    j += 1
                    continue
                if leading <= indent:
                    break

                m_cycles = re.search(r'cycles\s*:\s*([0-9]+)', next_line)
                if m_cycles:
                    core_stats[core_name]["cycles"] += int(m_cycles.group(1))

                m_ccycles = re.search(r'cCycles\s*:\s*([0-9]+)', next_line)
                if m_ccycles:
                    core_stats[core_name]["ccycles"] += int(m_ccycles.group(1))

                m_instrs = re.search(r'instrs\s*:\s*([0-9]+)', next_line)
                if m_instrs:
                    core_stats[core_name]["instrs"] += int(m_instrs.group(1))

                j += 1

            i = j
        else:
            i += 1

    return core_stats


def parse_l3_misses(lines):
    misses_by_core = {c: 0 for c in CORES}
    l3_start_idx = None

    for idx, line in enumerate(lines):
        if re.match(r'^\s*l3:\s*#\s*Cache\s+stats', line):
            l3_start_idx = idx
            break

    if l3_start_idx is None:
        return misses_by_core

    i = l3_start_idx + 1
    n = len(lines)

    while i < n:
        line = lines[i].strip()
        # Match both l3-X and l3-XbY patterns
        m = re.match(r'^l3-(\d+)(?:b(\d+))?:', line)
        if m:
            # For l3-0b0 format, use the bank number (group 2)
            # For l3-0 format, use the cache number (group 1)
            if m.group(2) is not None:
                # Multi-threaded: l3-XbY where Y is the core
                core_id = int(m.group(2))
            else:
                # Single-threaded: l3-X where X is the core
                core_id = int(m.group(1))
            
            core_name = f"westmere-{core_id}"
            indent = len(lines[i]) - len(lines[i].lstrip(' '))
            block_misses = 0
            j = i + 1

            while j < n:
                nxt = lines[j]
                if nxt.strip() == "":
                    j += 1
                    continue
                leading = len(nxt) - len(nxt.lstrip(' '))
                if leading <= indent:
                    break

                for key in ["mGETS", "mGETXIM", "mGETXSM"]:
                    mm = re.search(rf'{key}\s*[:=]\s*([0-9]+)', nxt)
                    if mm:
                        block_misses += int(mm.group(1))

                j += 1

            if 0 <= core_id <= 7:
                misses_by_core[core_name] += block_misses

            i = j

        else:
            # Stop if we hit another top-level section
            if re.match(r'^[^\s].+:\s*$', line):
                break
            i += 1

    return misses_by_core


def parse_file(path):
    with open(path, "r", errors="ignore") as fh:
        lines = [ln.rstrip("\n") for ln in fh.readlines()]
    cores = parse_core_blocks(lines)
    l3 = parse_l3_misses(lines)
    return cores, l3


def compute_metrics(zsim_path):
    """
    Parse a single zsim.out and return all core-level metrics and aggregates.
    """
    core_stats, l3_misses = parse_file(zsim_path)

    cycles = []
    instrs = []
    ipc = []
    misses = []
    mpki = []

    for c in CORES:
        cyc = core_stats[c]["cycles"] + core_stats[c]["ccycles"]
        ins = core_stats[c]["instrs"]
        mis = l3_misses[c]

        cycles.append(cyc)
        instrs.append(ins)
        misses.append(mis)

        ipc.append(ins / cyc if cyc > 0 else 0)
        mpki.append((mis / ins) * 1000 if ins > 0 else 0)

    total_cycles = max(cycles) if any(instrs[i] > 0 for i in range(len(CORES))) else sum(cycles)
    active_ipc = [ipc[i] for i in range(len(CORES)) if instrs[i] > 0]
    active_mpki = [mpki[i] for i in range(len(CORES)) if instrs[i] > 0]

    avg_ipc = geometric_mean(active_ipc) if active_ipc else 0
    avg_mpki = sum(active_mpki) / len(active_mpki) if active_mpki else 0

    return {
        "cycles": cycles,
        "instrs": instrs,
        "ipc": ipc,
        "misses": misses,
        "mpki": mpki,
        "total_cycles": total_cycles,
        "avg_ipc": avg_ipc,
        "avg_mpki": avg_mpki,
    }

############################################################
# PLOT BAR Graphs for IPC MPKI and WS
############################################################

def plot_bar(metric_name, values, label, save_dir, xlabel=""):
    """
    Creates a bar graph for a metric across categories.
    values: dict {category: value}
    """
    categories = list(values.keys())
    metrics = list(values.values())

    plt.figure(figsize=(12, 6))
    plt.bar(categories, metrics)
    plt.xticks(rotation=45, fontsize=8)
    plt.ylabel(metric_label(metric_name))
    if xlabel:
        plt.xlabel(xlabel)
    plt.title(f"{metric_name} Comparison for {label}")
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()

    out_path = os.path.join(save_dir, f"{label}_{metric_name}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved plot: {out_path}")


def aggregate_by_group(metric_by_bench, benches):
    """
    Given metric_by_bench: {bench: {policy: value}}, compute per-policy average over provided benches.
    """
    aggregated = {}
    for policy in POLICIES:
        vals = [
            metric_by_bench[bench][policy]
            for bench in benches
            if bench in metric_by_bench and policy in metric_by_bench[bench]
        ]
        if vals:
            aggregated[policy] = sum(vals) / len(vals)
    return aggregated


def metric_label(metric_name):
    """
    Return a metric label with units for plotting.
    """
    units = {
        "IPC": "IPC (instr/cycle)",
        "MPKI": "MPKI (misses/1K instr)",
        "WS": "Weighted Speedup (relative)",
    }
    return units.get(metric_name, metric_name)


def geometric_mean(values):
    vals = [v for v in values if v > 0]
    if not vals:
        return 0.0
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def plot_grouped_bars(metric_name, bench_to_policy, label, save_dir):
    """
    Grouped bar chart where each benchmark has bars for all policies.
    bench_to_policy: {bench: {policy: value}}
    """
    benches = [b for b in BENCHMARKS if b in bench_to_policy]
    if not benches:
        return

    num_policies = len(POLICIES)
    width = 0.8 / num_policies
    x_positions = list(range(len(benches)))

    plt.figure(figsize=(max(12, len(benches) * 0.6), 6))
    for i, policy in enumerate(POLICIES):
        offsets = [x + (i - (num_policies - 1) / 2) * width for x in x_positions]
        values = [bench_to_policy[bench].get(policy, 0) for bench in benches]
        plt.bar(offsets, values, width=width, label=policy)

    plt.xticks(x_positions, benches, rotation=60, fontsize=8)
    plt.ylabel(metric_label(metric_name))
    plt.xlabel("Benchmark")
    plt.title(f"{metric_name} by Policy across {label}")
    plt.legend(
        title="Policy",
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0.2,
        frameon=True,
        ncol=1,
    )
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout(rect=[0, 0, 0.93, 1])

    out_path = os.path.join(save_dir, f"{label}_{metric_name}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved plot: {out_path}")


############################################################
# MAIN
############################################################

def parse_args():
    parser = argparse.ArgumentParser(description="Parse zsim outputs into CSV + plots.")
    script_root = Path(__file__).resolve()
    repo_root = script_root.parents[2]

    parser.add_argument(
        "--input-root",
        default=str(repo_root / "simulation" / "zsim" / "outputs"),
        help="Root directory containing policy/benchmark/zsim.out hierarchy",
    )
    parser.add_argument(
        "--parsed-dir",
        default=str(repo_root / "analysis" / "parsed_outputs"),
        help="Where to write per-benchmark CSV files",
    )
    parser.add_argument(
        "--plots-dir",
        default=str(repo_root / "analysis" / "plots"),
        help="Where to write plots (one set per policy)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    parsed_root = Path(args.parsed_dir).expanduser().resolve()
    plots_dir = Path(args.plots_dir).expanduser().resolve()

    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    # Clean previous outputs
    for target in (parsed_root, plots_dir):
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)

    header = (
        ["policy", "benchmark"] +
        [f"cycles_{c}" for c in CORES] +
        [f"instrs_{c}" for c in CORES] +
        [f"ipc_{c}" for c in CORES] +
        [f"misses_{c}" for c in CORES] +
        [f"mpki_{c}" for c in CORES] +
        ["total_cycles_all", "avg_ipc_all", "avg_mpki_all",
         "WS"]  # WS only
    )

    os.makedirs(parsed_root, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Build LRU baseline map for WS
    lru_ipc = {}
    for bench in BENCHMARKS:
        lru_path = input_root / "LRU" / bench / "zsim.out"
        if not lru_path.is_file():
            print(f"WARNING: LRU baseline missing for benchmark '{bench}', WS will be 0")
            continue
        metrics = compute_metrics(str(lru_path))
        lru_ipc[bench] = metrics["avg_ipc"]

    rows_by_policy = defaultdict(dict)
    ipc_by_bench = defaultdict(dict)
    mpki_by_bench = defaultdict(dict)
    ws_by_bench = defaultdict(dict)

    for policy in POLICIES:
        for bench in BENCHMARKS:
            zsim_path = input_root / policy / bench / "zsim.out"
            if not zsim_path.is_file():
                print(f"WARNING: no zsim.out for policy '{policy}' benchmark '{bench}', skipping")
                continue

            metrics = compute_metrics(str(zsim_path))
            lru_base = lru_ipc.get(bench)
            if lru_base and lru_base > 0:
                ws = metrics["avg_ipc"] / lru_base
            else:
                ws = 1.0 if policy == "LRU" and metrics["avg_ipc"] > 0 else 0.0

            row = (
                [policy, bench] +
                metrics["cycles"] + metrics["instrs"] + metrics["ipc"] +
                metrics["misses"] + metrics["mpki"] +
                [metrics["total_cycles"], metrics["avg_ipc"], metrics["avg_mpki"], ws]
            )

            rows_by_policy[policy][bench] = row
            ipc_by_bench[bench][policy] = metrics["avg_ipc"]
            mpki_by_bench[bench][policy] = metrics["avg_mpki"]
            ws_by_bench[bench][policy] = ws
            print(f"Parsed {policy}/{bench}")

    # Write per-policy CSV files
    for policy in POLICIES:
        bench_rows = rows_by_policy.get(policy, {})
        if not bench_rows:
            print(f"NOTE: no data collected for policy '{policy}', skipping write")
            continue

        policy_dir = parsed_root / policy
        policy_dir.mkdir(parents=True, exist_ok=True)
        out_path = policy_dir / "data.csv"

        with open(out_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            for bench in BENCHMARKS:
                if bench in bench_rows:
                    writer.writerow(bench_rows[bench])

        print(f"Wrote {out_path}")

    ############################################################
    # GEOMETRIC MEAN IPC + SPEEDUPS
    ############################################################

    spec_benches = [b for b in BENCHMARKS if "_8c_simlarge" not in b]
    parsec_benches = [b for b in BENCHMARKS if "_8c_simlarge" in b]

    geomean_ipc = {}
    geomean_spec_ipc = {}
    geomean_parsec_ipc = {}
    spec_avg_mpki = {}
    parsec_avg_mpki = {}

    for policy in POLICIES:
        vals = [
            ipc_by_bench[b][policy]
            for b in BENCHMARKS
            if b in ipc_by_bench and policy in ipc_by_bench[b]
        ]
        geomean_ipc[policy] = geometric_mean(vals) if vals else 0.0

        spec_vals = [
            ipc_by_bench[b][policy]
            for b in spec_benches
            if b in ipc_by_bench and policy in ipc_by_bench[b]
        ]
        geomean_spec_ipc[policy] = geometric_mean(spec_vals) if spec_vals else 0.0

        parsec_vals = [
            ipc_by_bench[b][policy]
            for b in parsec_benches
            if b in ipc_by_bench and policy in ipc_by_bench[b]
        ]
        geomean_parsec_ipc[policy] = geometric_mean(parsec_vals) if parsec_vals else 0.0

        spec_mpki_vals = [
            mpki_by_bench[b][policy]
            for b in spec_benches
            if b in mpki_by_bench and policy in mpki_by_bench[b]
        ]
        spec_avg_mpki[policy] = sum(spec_mpki_vals) / len(spec_mpki_vals) if spec_mpki_vals else 0.0

        parsec_mpki_vals = [
            mpki_by_bench[b][policy]
            for b in parsec_benches
            if b in mpki_by_bench and policy in mpki_by_bench[b]
        ]
        parsec_avg_mpki[policy] = sum(parsec_mpki_vals) / len(parsec_mpki_vals) if parsec_mpki_vals else 0.0

    # Calculate speedups relative to LRU
    spec_speedup = {}
    parsec_speedup = {}
    lru_spec_ipc = geomean_spec_ipc.get("LRU", 0.0)
    lru_parsec_ipc = geomean_parsec_ipc.get("LRU", 0.0)

    for policy in POLICIES:
        if lru_spec_ipc > 0:
            spec_speedup[policy] = geomean_spec_ipc[policy] / lru_spec_ipc
        else:
            spec_speedup[policy] = 0.0

        if lru_parsec_ipc > 0:
            parsec_speedup[policy] = geomean_parsec_ipc[policy] / lru_parsec_ipc
        else:
            parsec_speedup[policy] = 0.0

    print("\n=== GEOMEAN IPC (SPEC) ===")
    for p in POLICIES:
        print(f"{p:8s} : {geomean_spec_ipc[p]:.4f}")

    print("\n=== GEOMEAN IPC (PARSEC) ===")
    for p in POLICIES:
        print(f"{p:8s} : {geomean_parsec_ipc[p]:.4f}")

    print("\n=== SPEC SPEEDUP VS LRU ===")
    for p in POLICIES:
        print(f"{p:8s} : {spec_speedup[p]:.4f}")

    print("\n=== PARSEC SPEEDUP VS LRU ===")
    for p in POLICIES:
        print(f"{p:8s} : {parsec_speedup[p]:.4f}")

    print("\n=== AVERAGE MPKI (SPEC) ===")
    for p in POLICIES:
        print(f"{p:8s} : {spec_avg_mpki[p]:.4f}")

    print("\n=== AVERAGE MPKI (PARSEC) ===")
    for p in POLICIES:
        print(f"{p:8s} : {parsec_avg_mpki[p]:.4f}")

    # Write summary CSV
    results_dir = parsed_root.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_path = results_dir / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "policy",
            "geomean_spec_ipc",
            "spec_speedup_vs_lru",
            "spec_avg_mpki",
            "geomean_parsec_ipc",
            "parsec_speedup_vs_lru",
            "parsec_avg_mpki",
        ])
        for p in POLICIES:
            writer.writerow([
                p,
                geomean_spec_ipc[p],
                spec_speedup[p],
                spec_avg_mpki[p],
                geomean_parsec_ipc[p],
                parsec_speedup[p],
                parsec_avg_mpki[p],
            ])

    print(f"\nSummary written to {summary_path}")

    ############################################################
    # BAR PLOTS: GMIPC + SPEEDUPS
    ############################################################

    def plot_simple_bar(title, data_dict, out_file, ylabel=None):
        plt.figure(figsize=(10, 5))
        plt.bar(list(data_dict.keys()), list(data_dict.values()))
        plt.xticks(rotation=45)
        if ylabel:
            plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(axis="y", linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(out_file, dpi=200)
        plt.close()
        print(f"Saved plot: {out_file}")

    plot_simple_bar(
        "Geometric Mean IPC per Policy (SPEC)",
        geomean_spec_ipc,
        plots_dir / "geomean_ipc_spec.png",
        ylabel="IPC"
    )

    plot_simple_bar(
        "Geometric Mean IPC per Policy (PARSEC)",
        geomean_parsec_ipc,
        plots_dir / "geomean_ipc_parsec.png",
        ylabel="IPC"
    )

    plot_simple_bar(
        "SPEC Speedup vs LRU",
        spec_speedup,
        plots_dir / "spec_speedup_vs_lru.png",
        ylabel="Speedup (relative to LRU)"
    )

    plot_simple_bar(
        "PARSEC Speedup vs LRU",
        parsec_speedup,
        plots_dir / "parsec_speedup_vs_lru.png",
        ylabel="Speedup (relative to LRU)"
    )

    plot_simple_bar(
        "Average MPKI per Policy (SPEC)",
        spec_avg_mpki,
        plots_dir / "avg_mpki_spec.png",
        ylabel="MPKI (misses per 1K instructions)"
    )

    plot_simple_bar(
        "Average MPKI per Policy (PARSEC)",
        parsec_avg_mpki,
        plots_dir / "avg_mpki_parsec.png",
        ylabel="MPKI (misses per 1K instructions)"
    )

    # Create grouped bar plots comparing policies across benchmarks within SPEC and PARSEC groups
    for group_name, benches in [("SPEC", spec_benches), ("PARSEC", parsec_benches)]:
        ipc_group = {b: ipc_by_bench[b] for b in benches if ipc_by_bench[b]}
        mpki_group = {b: mpki_by_bench[b] for b in benches if mpki_by_bench[b]}
        ws_group = {b: ws_by_bench[b] for b in benches if ws_by_bench[b]}

        if not ipc_group:
            print(f"NOTE: no data for group '{group_name}', skipping plots")
            continue

        plot_grouped_bars("IPC", ipc_group, group_name, str(plots_dir))
        plot_grouped_bars("MPKI", mpki_group, group_name, str(plots_dir))
        plot_grouped_bars("WS", ws_group, group_name, str(plots_dir))

    print("Done.")


if __name__ == "__main__":
    main()
