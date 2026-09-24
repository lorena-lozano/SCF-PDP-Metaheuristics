"""
Batch runner for SCF-PDP algorithms
Runs algorithm on all instances in a folder structure and saves results
MODIFIED: Can skip existing solutions and run only specific sizes
"""

import os
import sys
import time
import glob
from pathlib import Path
from VRP import (
    SCFPDPInstance, Solution, write_solution, get_instance_name
)


def run_algorithm_on_instance(algorithm_func, instance_file: str,
                              output_dir: str, skip_existing: bool = True):
    """
    Run an algorithm on a single instance and save the solution.
    Returns: (instance_name, objective_value, runtime, feasible, error_msg, skipped)
    """
    try:
        inst_name = get_instance_name(instance_file)
        output_file = os.path.join(output_dir, f"{inst_name}.txt")
        
        # Check if solution already exists
        if skip_existing and os.path.exists(output_file):
            return (inst_name, 0, 0, True, "SKIPPED - already exists", True)

        # Load instance
        inst = SCFPDPInstance(instance_file)

        # Run algorithm with timing
        start_time = time.time()
        solution = algorithm_func(inst)
        runtime = (time.time() - start_time) / 60  # Duration in minutes

        # Check feasibility
        feasible, msg = solution.is_feasible()

        # Get objective value
        obj_value = solution.objective_value() if feasible else float('inf')

        # Save solution
        write_solution(solution, output_file, inst_name)

        return (inst_name, obj_value, runtime, feasible, msg, False)

    except Exception as e:
        import traceback
        error_msg = f"ERROR: {str(e)}\n{traceback.format_exc()}"
        return (
        get_instance_name(instance_file), float('inf'), 0, False, error_msg, False)


def run_batch(algorithm_func, instances_root: str, output_root: str,
              algorithm_name: str, target_sizes: set = None, skip_existing: bool = False):
    """
    Run algorithm on all instances in the folder structure.
    
    Args:
        target_sizes: Set of size strings to process (e.g., {"10000"}). If None, process all.
        skip_existing: If True, skip instances that already have solution files.
    """

    # Create output directories
    algo_output_dir = os.path.join(output_root, algorithm_name)
    os.makedirs(algo_output_dir, exist_ok=True)

    # Find all instance sizes (subdirectories)
    size_dirs = [d for d in os.listdir(instances_root)
                 if os.path.isdir(os.path.join(instances_root, d))]
    size_dirs.sort(key=lambda x: int(x) if x.isdigit() else 0)

    # Filter by target sizes if specified
    if target_sizes is not None:
        size_dirs = [s for s in size_dirs if s in target_sizes]
        print(f"Filtering to target sizes: {target_sizes}")

    print(f"Processing instance sizes: {size_dirs}")
    print(f"Skip existing solutions: {skip_existing}")
    print("=" * 80)

    all_results = []

    for size in size_dirs:
        train_dir = os.path.join(instances_root, size, "train")

        if not os.path.exists(train_dir):
            print(f"WARNING: {train_dir} not found, skipping...")
            continue

        # Find all instance files
        instance_files = sorted(glob.glob(os.path.join(train_dir, "*.txt")))

        if not instance_files:
            print(f"WARNING: No instances found in {train_dir}")
            continue

        print(f"\nProcessing size {size}: {len(instance_files)} instances")
        print("-" * 80)

        # Create output directory for this size
        size_output_dir = os.path.join(algo_output_dir, size)
        os.makedirs(size_output_dir, exist_ok=True)

        # Process each instance
        skipped_count = 0
        for i, instance_file in enumerate(instance_files, 1):
            inst_name = get_instance_name(instance_file)
            print(f"  [{i}/{len(instance_files)}] {inst_name}...", end=" ",
                  flush=True)

            # Run algorithm
            result = run_algorithm_on_instance(algorithm_func, instance_file,
                                               size_output_dir, skip_existing)
            inst_name, obj_value, runtime, feasible, msg, skipped = result

            if skipped:
                skipped_count += 1
                print(f"⊘ SKIPPED (already exists)")
                continue

            # Store result
            all_results.append({
                'size': size,
                'instance': inst_name,
                'objective': obj_value,
                'runtime': runtime,
                'feasible': feasible,
                'message': msg
            })

            # Print result
            if feasible:
                print(f"✓ Obj: {obj_value:.2f}, Time: {runtime:.3f}m")
            else:
                print(f"✗ INFEASIBLE: {msg[:50]}")
        
        if skipped_count > 0:
            print(f"\nSkipped {skipped_count} existing solutions for size {size}")

    # Write summary CSV
    summary_file = os.path.join(algo_output_dir, "results_summary.csv")
    write_summary_csv(all_results, summary_file)

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print_summary_statistics(all_results)
    print(f"\nResults saved to: {algo_output_dir}")
    print(f"Summary CSV: {summary_file}")


def write_summary_csv(results: list, filename: str):
    """Write results summary to CSV file"""
    import csv

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            ['Size', 'Instance', 'Objective', 'Runtime', 'Feasible',
             'Message'])

        for r in results:
            writer.writerow([
                r['size'],
                r['instance'],
                f"{r['objective']:.2f}" if r['objective'] != float(
                    'inf') else 'INF',
                f"{r['runtime']:.3f}",
                'Yes' if r['feasible'] else 'No',
                r['message'] if not r['feasible'] else 'OK'
            ])


def print_summary_statistics(results: list):
    """Print summary statistics by size"""
    from collections import defaultdict

    by_size = defaultdict(list)
    for r in results:
        by_size[r['size']].append(r)

    print(
        f"\n{'Size':<10} {'Total':<8} {'Feasible':<10} {'Avg Obj':<12} {'Avg Time':<12}")
    print("-" * 80)

    total_instances = 0
    total_feasible = 0
    all_objectives = []
    all_times = []

    for size in sorted(by_size.keys(),
                       key=lambda x: int(x) if x.isdigit() else 0):
        size_results = by_size[size]
        total = len(size_results)
        feasible = sum(1 for r in size_results if r['feasible'])

        feasible_results = [r for r in size_results if r['feasible']]
        if feasible_results:
            avg_obj = sum(r['objective'] for r in feasible_results) / len(
                feasible_results)
            avg_time = sum(r['runtime'] for r in feasible_results) / len(
                feasible_results)
            all_objectives.extend(r['objective'] for r in feasible_results)
            all_times.extend(r['runtime'] for r in feasible_results)
        else:
            avg_obj = float('inf')
            avg_time = 0

        print(
            f"{size:<10} {total:<8} {feasible}/{total:<7} {avg_obj:<12.2f} {avg_time:<12.3f}m")

        total_instances += total
        total_feasible += feasible

    print("-" * 80)

    if all_objectives:
        overall_avg_obj = sum(all_objectives) / len(all_objectives)
        overall_avg_time = sum(all_times) / len(all_times)
    else:
        overall_avg_obj = float('inf')
        overall_avg_time = 0


    print(
        f"{'TOTAL':<10} {total_instances:<8} {total_feasible}/{total_instances:<7} "
        f"{overall_avg_obj:<12.2f} {overall_avg_time:<12.3f}m")
    
    # Avoid divide-by-zero when all instances were skipped
    if total_instances == 0:
        print(f"{'TOTAL':<10} 0        0/0       N/A           N/A")
        print("\nFeasibility rate: N/A (no instances processed — all were skipped)")
        return

    print(f"\nFeasibility rate: {100 * total_feasible / total_instances:.1f}%")



def main():
    print("Batch Runner with Size Filter and Skip Existing")
    
    # Default paths
    instances_root = sys.argv[1] if len(sys.argv) > 1 else "./instances"
    output_root = sys.argv[2] if len(sys.argv) > 2 else "./solutions"
    algorithm_name = sys.argv[3] if len(sys.argv) > 3 else "greedy_tabu_best"

    # NEW: Optional size filter (comma-separated list)
    size_filter = sys.argv[4] if len(sys.argv) > 4 else None
    target_sizes = set(size_filter.split(',')) if size_filter else None

    # NEW: Optional skip existing flag
    skip_existing = True

    print(f"Batch Runner Configuration:")
    print(f"  Instances: {instances_root}")
    print(f"  Output: {output_root}")
    print(f"  Algorithm: {algorithm_name}")
    print(f"  Target sizes: {target_sizes if target_sizes else 'ALL'}")
    print(f"  Skip existing: {skip_existing}")
    print()

    if not os.path.exists(instances_root):
        print(f"ERROR: Instances directory not found: {instances_root}")
        print(f"Usage: python batch_runner.py [instances_root] [output_root] [algorithm_name] [sizes] [skip_existing]")
        print(f"Example: python batch_runner.py ./instances ./solutions greedy_tabu_best 10000 true")
        return

    # Import the algorithm based on algorithm_name
    try:
        from tabu_search import (
            greedy_tabu_best, greedy_tabu_first,
        )
        
        algorithms = {
            "greedy_tabu_best": greedy_tabu_best,
            "greedy_tabu_first": greedy_tabu_first,
        }
        
        if algorithm_name not in algorithms:
            print(f"ERROR: Unknown algorithm '{algorithm_name}'")
            print(f"Available algorithms: {list(algorithms.keys())}")
            return
        
        alg_function = algorithms[algorithm_name]
        
    except ImportError as e:
        print(f"ERROR: Could not import algorithms")
        print(f"Error: {e}")
        return

    # Run batch processing
    run_batch(alg_function, instances_root, output_root, algorithm_name, 
              target_sizes=target_sizes, skip_existing=skip_existing)


if __name__ == "__main__":
    main()