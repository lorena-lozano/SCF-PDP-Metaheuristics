import os
import pandas as pd

# Use a non-interactive backend (no Tk windows)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np


# ================================================================
# 1. CONFIGURATION
# ================================================================
#
# Map a short algorithm label to:
#   - path: path to its summary CSV (from batch_runner or grasp_experiments)
#   - family: high-level category (for grouping in plots)
#
# Adapt these paths to your real folder structure.
#

ALGORITHMS = {
    # (a) Construction heuristics
    "det_cons": {
        "path": "solutions/det_cons/results_summary.csv",
        "family": "construction",
    },
    "rand_cons": {
        "path": "solutions/rand_cons/results_summary.csv",
        "family": "construction",
    },

    # (b-i) Local search variants (example names; adapt if needed)
    "ls_N1_best": {
        "path": "solutions/ls_N1_best/results_summary.csv",
        "family": "local_search",
    },
    "ls_N1_first": {
        "path": "solutions/ls_N1_first/results_summary.csv",
        "family": "local_search",
    },
    "ls_N2_best": {
        "path": "solutions/ls_N2_best/results_summary.csv",
        "family": "local_search",
    },
    "ls_N2_first": {
        "path": "solutions/ls_N2_first/results_summary.csv",
        "family": "local_search",
    },
    "ls_N3_best": {
        "path": "solutions/ls_N3_best/results_summary.csv",
        "family": "local_search",
    },
    "ls_N3_first": {
        "path": "solutions/ls_N3_first/results_summary.csv",
        "family": "local_search",
    },
    # You can add composite LS variants here if you have them.

    # (b-ii) VND (adapt folder/file name if needed)
    "VND": {
        "path": "solutions/VND/results_summary.csv",
        "family": "VND",
    },

    # (b-iii) SA / TS / GVNS (choose the one you implemented)
    "SA": {
        "path": "solutions/SA/results_summary.csv",
        "family": "SA",
    },
    # You can add "TS", "GVNS", etc. here if implemented.

    # GRASP: for the FINAL global comparison, keep ONLY the best variant.
    # For now you can temporarily put one GRASP here to test.
    # Example (replace N2_best by your chosen best variant):
    "grasp_best": {
        "path": "solutions/grasp_N2_best/results_summary_50.csv",
        "family": "GRASP",
    },
}

OUTPUT_DIR = "solutions/analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================================================================
# 2. LOADING AND CLEANING A SINGLE SUMMARY CSV
# ================================================================

def load_summary(csv_path: str, algo_label: str, family: str) -> pd.DataFrame:
    """
    Load one summary CSV produced by write_summary_csv (batch_runner / grasp_experiments)
    and clean types:
      - Objective: float, 'INF' -> np.inf
      - Runtime:   float
      - Feasible:  bool
    Also adds columns:
      - Algorithm: algo_label
      - Family:    family
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Expected columns: 'Size', 'Instance', 'Objective', 'Runtime', 'Feasible', 'Message'

    # Normalize Objective
    def parse_objective(x):
        if isinstance(x, str):
            s = x.strip()
            if s.upper() == "INF":
                return np.inf
        try:
            return float(x)
        except Exception:
            return np.inf

    df["Objective"] = df["Objective"].apply(parse_objective)

    # Runtime as float
    df["Runtime"] = df["Runtime"].astype(float)

    # Feasible as bool
    df["Feasible_bool"] = (
        df["Feasible"].astype(str).str.strip().str.lower().eq("yes")
    )

    # Ensure Size is treated consistently as string (e.g. "50", "100")
    df["Size"] = df["Size"].astype(str)

    # Add algorithm info
    df["Algorithm"] = algo_label
    df["Family"] = family

    return df


# ================================================================
# 3. AGGREGATION OF METRICS
# ================================================================

def aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute aggregated statistics for each (Size, Algorithm, Family):
      - Instances
      - Feasible
      - Feasibility_rate_%
      - Avg_objective (over feasible only)
      - Std_objective (over feasible only)
      - Avg_runtime_s (over all)
      - Std_runtime_s (over all)
    """
    records = []

    # Group by size and algorithm
    for (size, algo, family), g in df.groupby(["Size", "Algorithm", "Family"]):
        total = len(g)
        feas_mask = g["Feasible_bool"]
        n_feas = feas_mask.sum()

        if n_feas > 0:
            feas_obj = g.loc[feas_mask, "Objective"]
            avg_obj = feas_obj.mean()
            std_obj = feas_obj.std(ddof=0)
        else:
            avg_obj = np.inf
            std_obj = np.nan

        avg_time = g["Runtime"].mean()
        std_time = g["Runtime"].std(ddof=0)

        feas_rate = 100.0 * n_feas / total if total > 0 else 0.0

        records.append({
            "Size": size,
            "Algorithm": algo,
            "Family": family,
            "Instances": total,
            "Feasible": n_feas,
            "Feasibility_rate_%": feas_rate,
            "Avg_objective": avg_obj,
            "Std_objective": std_obj,
            "Avg_runtime_s": avg_time,
            "Std_runtime_s": std_time,
        })

    agg_df = pd.DataFrame(records)

    # For nicer ordering: create numeric size column when possible
    def to_int(x):
        try:
            return int(x)
        except ValueError:
            return 0

    agg_df["Size_int"] = agg_df["Size"].apply(to_int)
    agg_df = agg_df.sort_values(by=["Size_int", "Algorithm"]).reset_index(drop=True)
    agg_df = agg_df.drop(columns=["Size_int"])

    return agg_df


# ================================================================
# 4. PLOTTING HELPERS
# ================================================================

def plot_bar_by_algorithm(df: pd.DataFrame, size_filter: str,
                          metric: str, ylabel: str,
                          title: str, filename: str):
    """
    Bar plot of a given metric per algorithm, filtered by instance size.
    """
    sub = df[df["Size"] == size_filter].copy()
    if sub.empty:
        print(f"[PLOT] No data for size={size_filter} and metric={metric}")
        return

    # Order by metric (if numeric) for nicer plots
    if np.issubdtype(sub[metric].dtype, np.number):
        sub = sub.sort_values(by=metric)
    else:
        sub = sub.sort_values(by="Algorithm")

    plt.figure()
    plt.bar(sub["Algorithm"], sub[metric])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"[PLOT] Saved: {plot_path}")


def plot_family_boxplot(df: pd.DataFrame, size_filter: str):
    """
    Boxplot of objective (feasible only) per "Family" (construction, LS, VND,
    GRASP, SA, etc.) for a given size.
    """
    sub = df[(df["Size"] == size_filter) & (df["Feasible_bool"])].copy()
    if sub.empty:
        print(f"[BOXPLOT] No feasible data for size={size_filter}")
        return

    plt.figure()
    sub.boxplot(column="Objective", by="Family")
    plt.title(f"Objective distribution by family (size {size_filter})")
    plt.suptitle("")  # remove default pandas suptitle
    plt.ylabel("Objective (feasible only)")
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, f"box_family_obj_size{size_filter}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"[BOXPLOT] Saved: {plot_path}")


# ================================================================
# 5. MAIN
# ================================================================

def main():
    all_results = []

    # 5.1 Load all algorithm summaries
    for algo_label, cfg in ALGORITHMS.items():
        csv_path = cfg["path"]
        family = cfg["family"]

        if not os.path.exists(csv_path):
            print(f"[WARNING] File not found for {algo_label}: {csv_path} (skipping)")
            continue

        try:
            print(f"Loading {algo_label} from {csv_path} ...")
            df_algo = load_summary(csv_path, algo_label=algo_label, family=family)
            all_results.append(df_algo)
        except Exception as e:
            print(f"[ERROR] Loading {algo_label}: {e}")

    if not all_results:
        print("No data loaded. Check ALGORITHMS paths.")
        return

    # 5.2 Concatenate all algorithms into one big DF
    results_df = pd.concat(all_results, ignore_index=True)

    # 5.3 Save raw merged results (optional, but useful for debugging)
    merged_csv = os.path.join(OUTPUT_DIR, "all_algorithms_raw_results.csv")
    results_df.to_csv(merged_csv, index=False)
    print(f"\nMerged raw results saved to: {merged_csv}")

    # 5.4 Aggregated stats (per size, algorithm, family)
    agg_df = aggregate_metrics(results_df)

    print("\n===== AGGREGATED COMPARISON (all sizes) =====")
    print(agg_df.to_string(index=False))

    agg_csv = os.path.join(OUTPUT_DIR, "all_algorithms_aggregated.csv")
    agg_df.to_csv(agg_csv, index=False)
    print(f"\nAggregated comparison saved to: {agg_csv}")

    # ============================================================
    # 5.5 Example plots / tables you can mention in the report
    # ============================================================

    # Example sizes – adapt to your instance folders
    for size_of_interest in ["11", "21", "50", "100"]:
        # (1) Compare average objective across all algorithms
        plot_bar_by_algorithm(
            df=agg_df,
            size_filter=size_of_interest,
            metric="Avg_objective",
            ylabel="Average objective (feasible only)",
            title=f"Algorithms – average objective (size {size_of_interest})",
            filename=f"avg_obj_size{size_of_interest}.png",
        )

        # (2) Compare average runtime
        plot_bar_by_algorithm(
            df=agg_df,
            size_filter=size_of_interest,
            metric="Avg_runtime_s",
            ylabel="Average runtime (s)",
            title=f"Algorithms – average runtime (size {size_of_interest})",
            filename=f"avg_runtime_size{size_of_interest}.png",
        )

        # (3) Boxplot by family
        plot_family_boxplot(results_df, size_filter=size_of_interest)


if __name__ == "__main__":
    main()
