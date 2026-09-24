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

# -------------------------------------------------------------------
# GRASP variants and their summary CSVs (from grasp_experiments.py)
# -------------------------------------------------------------------
GRASP_VARIANTS = {
    "grasp_N1_best":  "solutions/grasp_N1_best/results_summary_50.csv",
    "grasp_N1_first": "solutions/grasp_N1_first/results_summary_50.csv",
    "grasp_N2_best":  "solutions/grasp_N2_best/results_summary_50.csv",
    "grasp_N2_first": "solutions/grasp_N2_first/results_summary_50.csv",
    "grasp_N3_best":  "solutions/grasp_N3_best/results_summary_50.csv",
    "grasp_N3_first": "solutions/grasp_N3_first/results_summary_50.csv",
    "grasp_N4_best":  "solutions/grasp_N4_best/results_summary_50.csv",
    "grasp_N4_first": "solutions/grasp_N4_first/results_summary_50.csv",
}


def main():
    all_results = []

    # 1) Load all GRASP variants
    for algo_label, csv_path in GRASP_VARIANTS.items():
        if not os.path.exists(csv_path):
            print(f"[WARNING] File not found for {algo_label}: {csv_path} (skipping)")
            continue

        try:
            print(f"Loading {algo_label} from {csv_path} ...")
            # family="GRASP" so we can later filter by it if needed
            df_algo = load_summary(csv_path, algo_label=algo_label, family="GRASP")
            all_results.append(df_algo)
        except Exception as e:
            print(f"[ERROR] Loading {algo_label}: {e}")

    if not all_results:
        print("No GRASP data loaded. Check GRASP_VARIANTS paths.")
        return

    # 2) Concatenate raw GRASP results
    results_df = pd.concat(all_results, ignore_index=True)

    raw_csv = os.path.join(OUTPUT_DIR, "grasp_raw_results_50.csv")
    results_df.to_csv(raw_csv, index=False)
    print(f"\n[INFO] Raw GRASP results (size 50) saved to: {raw_csv}")

    # 3) Aggregate metrics by (Size, Algorithm, Family)
    agg_df = aggregate_metrics(results_df)

    print("\n===== GRASP – AGGREGATED COMPARISON (all sizes in these CSVs) =====")
    print(agg_df.to_string(index=False))

    agg_csv = os.path.join(OUTPUT_DIR, "grasp_aggregated_all_sizes.csv")
    agg_df.to_csv(agg_csv, index=False)
    print(f"\n[INFO] Aggregated GRASP metrics saved to: {agg_csv}")

    # 4) Focus on size 50 (used in grasp_experiments)
    size_of_interest = "50"
    grasp_50 = agg_df[agg_df["Size"] == size_of_interest].copy()

    if grasp_50.empty:
        print(f"\n[WARN] No GRASP rows for size {size_of_interest} in aggregated data.")
        return

    print(f"\n===== GRASP-ONLY COMPARISON (size {size_of_interest}) =====")
    print(grasp_50.to_string(index=False))

    comp_csv = os.path.join(
        OUTPUT_DIR, f"grasp_comparison_size{size_of_interest}.csv"
    )
    grasp_50.to_csv(comp_csv, index=False)
    print(f"\n[INFO] GRASP comparison for size {size_of_interest} saved to: {comp_csv}")

    # 5) Plots: average objective and runtime for GRASP variants (size 50)
    plot_bar_by_algorithm(
        df=agg_df,
        size_filter=size_of_interest,
        metric="Avg_objective",
        ylabel="Average objective (feasible only)",
        title=f"GRASP variants – average objective (size {size_of_interest})",
        filename=f"grasp_avg_obj_size{size_of_interest}.png",
    )

    plot_bar_by_algorithm(
        df=agg_df,
        size_filter=size_of_interest,
        metric="Avg_runtime_s",
        ylabel="Average runtime (s)",
        title=f"GRASP variants – average runtime (size {size_of_interest})",
        filename=f"grasp_avg_runtime_size{size_of_interest}.png",
    )


if __name__ == "__main__":
    main()
