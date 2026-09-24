#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Competition Benchmark Runner

Run any algorithm on competition instances (instance61) and save solutions.
Easily test multiple algorithms by changing configuration.

Usage:
    python run_competition_benchmark.py
"""

import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from VRP import (
    SCFPDPInstance, Solution,
    write_solution, get_instance_name
)


# ============================================================================
# COMPETITION INSTANCES CONFIGURATION
# ============================================================================

COMPETITION_INSTANCES = {
    # 50: "/Users/tina/HOT-1/instances/50/competition/instance61_nreq50_nveh2_gamma44.txt",
    100: "/Users/tina/HOT-1/instances/100/competition/instance61_nreq100_nveh2_gamma91.txt",
    # 200: "/Users/tina/HOT-1/instances/200/competition/instance61_nreq200_nveh4_gamma191.txt",
    # 500: "/Users/tina/HOT-1/instances/500/competition/instance61_nreq500_nveh10_gamma430.txt",
    1000: "/Users/tina/HOT-1/instances/1000/competition/instance61_nreq1000_nveh20_gamma879.txt",
    2000: "/Users/tina/HOT-1/instances/2000/competition/instance61_nreq2000_nveh40_gamma1829.txt",
    # 5000: "/Users/tina/HOT-1/instances/5000/competition/instance61_nreq5000_nveh100_gamma4448.txt",
    # 10000: "/Users/tina/HOT-1/instances/10000/competition/instance61_nreq10000_nveh200_gamma8803.txt",
}


# ============================================================================
# ALGORITHM REGISTRY
# ============================================================================

def get_algorithm(algorithm_name: str) -> Callable:
    """
    Import and return algorithm function.
    
    Add your algorithms here!
    """
    import importlib
    
    ALGORITHMS = {
        # Construction Heuristics
        "greedy": ("det_cons_heu_delta", "greedy_construction"),
        "random_greedy": ("rand_cons_heu_delta", "randomized_greedy_construction"),
        
        # Tabu Search Variants
        "tabu_greedy_best": ("tabu_search", "greedy_tabu_best"),
        "tabu_greedy_first": ("tabu_search", "greedy_tabu_first"),
       
        
        # Other Methods
        "beam_search": ("beam_search", "beam_search_construction"),
        
        # Add your algorithms here:
        # "my_algorithm": ("my_module", "my_function"),
    }
    
    if algorithm_name not in ALGORITHMS:
        available = ", ".join(ALGORITHMS.keys())
        raise ValueError(f"Unknown algorithm '{algorithm_name}'. Available: {available}")
    
    module_name, function_name = ALGORITHMS[algorithm_name]
    
    try:
        module = importlib.import_module(module_name)
        return getattr(module, function_name)
    except ImportError as e:
        raise ImportError(f"Could not import {module_name}: {e}")
    except AttributeError as e:
        raise AttributeError(f"Function {function_name} not found in {module_name}: {e}")


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

def run_algorithm_on_instance(algorithm_func: Callable,
                              instance_file: str,
                              algorithm_params: Dict[str, Any] = None) -> tuple:
    """
    Run algorithm on a single instance.
    
    Returns:
        (solution, runtime, objective, feasible, message, convergence_data)
    """
    if algorithm_params is None:
        algorithm_params = {}
    
    print(f"\n{'='*80}")
    print(f"Instance: {get_instance_name(instance_file)}")
    print(f"{'='*80}")
    
    # Load instance
    print("Loading instance...")
    instance = SCFPDPInstance(instance_file)
    print(f"  Requests: {instance.n}")
    print(f"  Vehicles: {instance.n_K}")
    print(f"  Capacity: {instance.C}")
    print(f"  Min requests: {instance.gamma}")
    
    # Run algorithm
    print(f"\nRunning algorithm...")
    start_time = time.time()
    
    try:
        result = algorithm_func(instance, **algorithm_params)
        
        # Handle both old-style (Solution) and new-style (Solution, convergence_data)
        if isinstance(result, tuple) and len(result) == 2:
            solution, convergence_data = result
        else:
            solution = result
            convergence_data = {
                'iterations': 0,
                'objective_history': [],
                'fairness_history': []
            }
        
        runtime = time.time() - start_time
        
        # Validate solution
        feasible, msg = solution.is_feasible()
        objective = solution.objective_value() if feasible else float('inf')
        
        # Print results
        print(f"\n{'='*80}")
        print(f"RESULTS")
        print(f"{'='*80}")
        print(f"Status: {'✓ FEASIBLE' if feasible else '✗ INFEASIBLE'}")
        print(f"Message: {msg}")
        print(f"Objective: {objective:.2f}")
        print(f"Runtime: {runtime:.3f}s")
        if convergence_data.get('iterations', 0) > 0:
            print(f"Iterations: {convergence_data['iterations']}")
        print(f"{'='*80}\n")
        
        return solution, runtime, objective, feasible, msg, convergence_data
        
    except Exception as e:
        print(f"\n✗ ALGORITHM FAILED!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
        runtime = time.time() - start_time
        return None, runtime, float('inf'), False, str(e), {}


def run_competition_benchmark(algorithm_name: str,
                              algorithm_params: Dict[str, Any] = None,
                              output_dir: str = "./competition_results",
                              sizes_to_run: list = None):
    """
    Run algorithm on all competition instances.
    
    Args:
        algorithm_name: Name of algorithm (from registry)
        algorithm_params: Parameters to pass to algorithm
        output_dir: Where to save solutions
        sizes_to_run: List of sizes to test (default: all)
    """
    if algorithm_params is None:
        algorithm_params = {}
    
    if sizes_to_run is None:
        sizes_to_run = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
    
    # Create output directory
    algo_output_dir = os.path.join(output_dir, algorithm_name)
    os.makedirs(algo_output_dir, exist_ok=True)
    
    # Get algorithm function
    print(f"\n{'#'*80}")
    print(f"COMPETITION BENCHMARK")
    print(f"{'#'*80}")
    print(f"Algorithm: {algorithm_name}")
    print(f"Parameters: {algorithm_params}")
    print(f"Output directory: {algo_output_dir}")
    print(f"Sizes to test: {sizes_to_run}")
    print(f"{'#'*80}\n")
    
    algorithm_func = get_algorithm(algorithm_name)
    
    # Results storage
    results_summary = []
    
    # Run on each instance
    for size in sizes_to_run:
        if size not in COMPETITION_INSTANCES:
            print(f"⚠ Warning: No competition instance for size {size}, skipping")
            continue
        
        instance_file = COMPETITION_INSTANCES[size]
        
        # Check if file exists
        if not os.path.exists(instance_file):
            print(f"⚠ Warning: Instance file not found: {instance_file}")
            print(f"   Skipping size {size}")
            continue
        
        # Run algorithm
        solution, runtime, objective, feasible, msg, convergence = run_algorithm_on_instance(
            algorithm_func, instance_file, algorithm_params
        )
        
        # Save solution if feasible
        if solution and feasible:
            instance_name = get_instance_name(instance_file)
            solution_file = os.path.join(algo_output_dir, f"{instance_name}_solution.txt")
            write_solution(solution, solution_file, instance_name)
            print(f"✓ Saved solution: {solution_file}")
        
        # Store results
        results_summary.append({
            'size': size,
            'instance': get_instance_name(instance_file),
            'objective': objective,
            'runtime': runtime,
            'feasible': feasible,
            'iterations': convergence.get('iterations', 0),
            'message': msg
        })
    
    # Save summary CSV
    summary_file = os.path.join(algo_output_dir, f"{algorithm_name}_competition_summary.csv")
    save_summary_csv(results_summary, summary_file)
    print(f"\n✓ Saved summary: {summary_file}")
    
    # Print final summary
    print(f"\n{'='*80}")
    print(f"BENCHMARK COMPLETE")
    print(f"{'='*80}")
    print(f"Algorithm: {algorithm_name}")
    print(f"Instances tested: {len(results_summary)}")
    print(f"Feasible solutions: {sum(1 for r in results_summary if r['feasible'])}")
    print(f"Total runtime: {sum(r['runtime'] for r in results_summary):.2f}s")
    print(f"Solutions saved in: {algo_output_dir}")
    print(f"{'='*80}\n")
    
    return results_summary


def save_summary_csv(results: list, filename: str):
    """Save results summary to CSV"""
    import csv
    
    if not results:
        return
    
    fieldnames = ['size', 'instance', 'objective', 'runtime', 'feasible', 'iterations', 'message']
    
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            formatted = result.copy()
            if isinstance(formatted.get('objective'), float):
                formatted['objective'] = f"{formatted['objective']:.2f}" if formatted['objective'] != float('inf') else 'INF'
            if isinstance(formatted.get('runtime'), float):
                formatted['runtime'] = f"{formatted['runtime']:.3f}"
            writer.writerow(formatted)


# ============================================================================
# MAIN - CONFIGURE YOUR TEST HERE
# ============================================================================

def main():
    """
    Main function - Configure your benchmark here!
    """
    
    # ========================================================================
    # CONFIGURATION - CHANGE THESE VALUES
    # ========================================================================
    
    # Choose algorithm to test
    ALGORITHM = "beam_search"  # ← Change this
    
    # Algorithm parameters (if needed)
    PARAMS = {
        # "tabu_tenure": 7,  # Best parameter from tuning
        # Add more parameters as needed
    }
    
    # Which sizes to test (comment out sizes you don't want)
    SIZES = [
        50,
        100,
        200,
        500,
        1000,
        2000,
        # 5000,    # Uncomment to test
        # 10000,   # Uncomment to test
    ]
    
    # Output directory
    OUTPUT_DIR = "./competition_results"
    
    # ========================================================================
    # RUN BENCHMARK
    # ========================================================================
    
    run_competition_benchmark(
        algorithm_name=ALGORITHM,
        algorithm_params=PARAMS,
        output_dir=OUTPUT_DIR,
        sizes_to_run=SIZES
    )


# ============================================================================
# ALTERNATIVE: Test Multiple Algorithms
# ============================================================================

def test_multiple_algorithms():
    """
    Example: Test multiple algorithms on competition instances
    """
    
    algorithms_to_test = [
        ("greedy", {}),
        ("tabu_greedy_first", {"tabu_tenure": 7}),
        ("tabu_random_first", {"tabu_tenure": 7}),
        # Add more...
    ]
    
    sizes = [50, 100, 200, 500]  # Test on smaller sizes first
    
    all_results = {}
    
    for algo_name, params in algorithms_to_test:
        print(f"\n{'#'*80}")
        print(f"Testing: {algo_name}")
        print(f"{'#'*80}\n")
        
        results = run_competition_benchmark(
            algorithm_name=algo_name,
            algorithm_params=params,
            sizes_to_run=sizes
        )
        
        all_results[algo_name] = results
    
    # Compare results
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Algorithm':<25} {'Avg Objective':<15} {'Avg Runtime':<15} {'Feasible'}")
    print(f"{'-'*80}")
    
    for algo_name, results in all_results.items():
        feasible_results = [r for r in results if r['feasible']]
        if feasible_results:
            avg_obj = sum(r['objective'] for r in feasible_results) / len(feasible_results)
            avg_time = sum(r['runtime'] for r in feasible_results) / len(feasible_results)
            num_feasible = len(feasible_results)
            print(f"{algo_name:<25} {avg_obj:<15.2f} {avg_time:<15.3f} {num_feasible}/{len(results)}")
        else:
            print(f"{algo_name:<25} {'NO FEASIBLE':<15} {'-':<15} 0/{len(results)}")
    
    print(f"{'='*80}\n")


if __name__ == "__main__":
    # Choose one:
    
    # Option 1: Test single algorithm (configured in main())
    main()
    
    # Option 2: Test multiple algorithms
    # test_multiple_algorithms()