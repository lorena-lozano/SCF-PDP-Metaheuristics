import os
import pandas as pd
import numpy as np

# Reuse shared helpers from the general analysis script
from helper_report.analyze_all_results import (
    load_summary,
    aggregate_metrics,
    plot_bar_by_algorithm,
)

OUTPUT_DIR = "solutions/analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LS_VARIANTS = {
    "ls_N1_best":  "solutions/ls_N1_best/results_summary.csv",
    "ls_N1_first": "solutions/ls_N1_first/results_summary.csv",
    "ls_N2_best":  "solutions/ls_N2_best/results_summary.csv",
    "ls_N2_first": "solutions/ls_N2_first/results_summary.csv",
    "ls_N3_best":  "solutions/ls_N3_best/results_summary.csv",
    "ls_N3_first": "solutions/ls_N3_first/results_summary.csv",
}


def sorted_sizes(size_values):
    def to_int(s):
        try:
            return int(s)
        except ValueError:
            return 0

    unique = sorted({str(s) for s in size_values}, key=to_int)
    return unique


def main():
    all_results = []

    # Load all LS variants
    for algo_label, csv_path in LS_VARIANTS.items():
        if not os.path.exists(csv_path):
            print(f"[WARNING] File not found for {algo_label}: {csv_path} (skipping)")
            continue

        try:
            print(f"Loading {algo_label} from {csv_path} ...")
            # family = "local_search"
            df_algo = load_summary(csv_path, algo_label=algo_label, family="local_search")
            all_results.append(df_algo)
        except Exception as e:
            print(f"[ERROR] Loading {algo_label}: {e}")

    if not all_results:
        print("No Local Search data loaded. Check LS_VARIANTS paths and CSVs.")
        return

    # Concatenate raw LS results (all sizes, all instances)
    results_df = pd.concat(all_results, ignore_index=True)

    raw_csv = os.path.join(OUTPUT_DIR, "ls_raw_results_all_sizes.csv")
    results_df.to_csv(raw_csv, index=False)
    print(f"\n[INFO] Raw Local Search results (all sizes) saved to: {raw_csv}")

    # Aggregate metrics by (Size, Algorithm, Family)
    agg_df = aggregate_metrics(results_df)

    print("\n===== LOCAL SEARCH – AGGREGATED COMPARISON (all sizes) =====")
    print(agg_df.to_string(index=False))

    agg_csv = os.path.join(OUTPUT_DIR, "ls_aggregated_all_sizes.csv")
    agg_df.to_csv(agg_csv, index=False)
    print(f"\n[INFO] Aggregated Local Search metrics saved to: {agg_csv}")

    # Per-size comparisons: for every size that appears in the data
    sizes = sorted_sizes(agg_df["Size"])

    for size in sizes:
        print(f"\n--- Plots for Local Search – size {size} ---")

        # Average objective per LS variant, for this size
        plot_bar_by_algorithm(
            df=agg_df,
            size_filter=size,
            metric="Avg_objective",
            ylabel="Average objective (feasible only)",
            title=f"Local Search variants – average objective (size {size})",
            filename=f"ls_avg_obj_size{size}.png",
        )

        # Average runtime per LS variant, for this size
        plot_bar_by_algorithm(
            df=agg_df,
            size_filter=size,
            metric="Avg_runtime_s",
            ylabel="Average runtime (s)",
            title=f"Local Search variants – average runtime (size {size})",
            filename=f"ls_avg_runtime_size{size}.png",
        )


if __name__ == "__main__":
    main()
