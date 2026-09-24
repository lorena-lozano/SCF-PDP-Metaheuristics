"""
Experimental runner for GRASP variants on SCF-PDP.

This script:
  - Runs several GRASP variants (different neighborhoods + step functions)
  - Only on instances of size 50 (folder: instances/50/train)
  - Uses the existing summary functions from batch_runner:
        - write_summary_csv
        - print_summary_statistics
  - Writes one results_summary_50.csv per variant in solutions/<variant_name>/
"""

import os
import time
import glob

from VRP import SCFPDPInstance, write_solution, get_instance_name, Solution

# *** IMPORTANT ***
# Reuse the summary functions already implemented in batch_runner.py
from batch_runner import write_summary_csv, print_summary_statistics

# Import GRASP variants (make sure they exist in GRASP.py)
from GRASP import (
    grasp_N1_best,
    grasp_N1_first,
    grasp_N2_best,
    grasp_N2_first,
    grasp_N3_best,
    grasp_N3_first,
    grasp_N4_best,
    grasp_N4_first,
)


def run_grasp_variant_on_size50(
    algorithm_func,
    variant_name: str,
    instances_root: str = "./instances",
    output_root: str = "./solutions",
    max_instances_50: int = 30,
):
    """
    Run one GRASP variant only on size-50 training instances and
    write a summary CSV, using the summary utilities from batch_runner.py.

    Args:
        algorithm_func: function(inst: SCFPDPInstance) -> Solution
        variant_name:   name of the variant (used for output folder and CSV)
        instances_root: base directory for instances (default: ./instances)
        output_root:    base directory for solutions (default: ./solutions)
        max_instances_50: maximum number of size-50 instances to process
                          (use a smaller number for quick tests)
    """

    # Directory for size 50 training instances
    train_dir = os.path.join(instances_root, "50", "train")
    if not os.path.exists(train_dir):
        print(f"[{variant_name}] WARNING: {train_dir} not found. Skipping.")
        return

    # Find all instance files in 50/train and limit to max_instances_50
    instance_files = sorted(glob.glob(os.path.join(train_dir, "*.txt")))
    if not instance_files:
        print(f"[{variant_name}] WARNING: No instances found in {train_dir}.")
        return

    instance_files = instance_files[:max_instances_50]

    # Output directory: solutions/<variant_name>/50/
    variant_root = os.path.join(output_root, variant_name)
    size_output_dir = os.path.join(variant_root, "50")
    os.makedirs(size_output_dir, exist_ok=True)

    print(f"\n=== Running GRASP variant: {variant_name} on size 50 ===")
    print(f"Instances directory: {train_dir}")
    print(f"Number of instances: {len(instance_files)}")
    print("-" * 80)

    results = []

    for i, instance_file in enumerate(instance_files, 1):
        inst_name = get_instance_name(instance_file)
        print(f"  [{i}/{len(instance_files)}] {inst_name}...", end=" ", flush=True)

        try:
            # Load instance
            inst = SCFPDPInstance(instance_file)

            # Run algorithm with timing
            start_time = time.time()
            solution: Solution = algorithm_func(inst)
            runtime = time.time() - start_time

            # Check feasibility
            feasible, msg = solution.is_feasible()

            # Objective value
            obj_value = solution.objective_value() if feasible else float("inf")

            # Save solution file
            output_file = os.path.join(size_output_dir, f"{inst_name}.txt")
            write_solution(solution, output_file, inst_name)

            # Store result (same keys used by batch_runner.summary)
            results.append({
                "size": "50",
                "instance": inst_name,
                "objective": obj_value,
                "runtime": runtime,
                "feasible": feasible,
                "message": msg,
            })

            # Print short result line
            if feasible:
                print(f"✓ Obj: {obj_value:.2f}, Time: {runtime:.3f}s")
            else:
                print(f"✗ INFEASIBLE: {msg[:50]}")

        except Exception as e:
            import traceback
            err_msg = f"ERROR: {str(e)}"
            print(f"✗ {err_msg}")
            # Store error as infeasible with INF objective
            results.append({
                "size": "50",
                "instance": inst_name,
                "objective": float("inf"),
                "runtime": 0.0,
                "feasible": False,
                "message": err_msg + "\n" + traceback.format_exc(),
            })

    # Summary CSV for this variant (only size 50)
    # We REUSE write_summary_csv from batch_runner
    summary_file = os.path.join(variant_root, "results_summary_50.csv")
    write_summary_csv(results, summary_file)

    print("\n" + "=" * 80)
    print(f"SUMMARY for variant: {variant_name} (size 50)")
    print("=" * 80)
    # And we also reuse print_summary_statistics
    print_summary_statistics(results)
    print(f"\nResults saved to: {variant_root}")
    print(f"Summary CSV: {summary_file}")
    print("=" * 80 + "\n")


def main():
    # Base folders (adapt if necessary)
    instances_root = "./instances"
    output_root = "./solutions"

    # Maximum number of size-50 instances per run.
    #  - Use 30 if you want ALL (since there are 30 instances of size 50)
    #  - Use a smaller number (e.g. 5 or 10) for very quick experiments
    max_instances_50 = 10

    # List of GRASP variants to test.
    # Comment out the ones that are too slow or not needed for a given experiment.
    variants = [
        ("grasp_N1_best",  grasp_N1_best),
        ("grasp_N1_first", grasp_N1_first),
        ("grasp_N2_best",  grasp_N2_best),
        ("grasp_N2_first", grasp_N2_first),
        ("grasp_N3_best",  grasp_N3_best),
        ("grasp_N3_first", grasp_N3_first),
        ("grasp_N4_best",  grasp_N4_best),
        ("grasp_N4_first", grasp_N4_first),
    ]

    print("GRASP experiments: starting...")
    print(f"Instances root: {instances_root}")
    print(f"Output root:    {output_root}")
    print(f"Max instances (size 50) per variant: {max_instances_50}")
    print()

    for variant_name, func in variants:
        run_grasp_variant_on_size50(
            algorithm_func=func,
            variant_name=variant_name,
            instances_root=instances_root,
            output_root=output_root,
            max_instances_50=max_instances_50,
        )

    print("GRASP experiments finished.")


if __name__ == "__main__":
    main()
