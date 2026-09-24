#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Best Beam Search Configuration on Test Instances
Run the best beam width configuration on test data
"""

import os
import sys
import csv
import time

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from VRP import SCFPDPInstance, get_instance_name

# ============================================================================
# CONFIGURATION - Best parameters from tuning experiments
# ============================================================================

ALGORITHM_NAME = "beam_search"
BEST_PARAMS = {
    "beam_width": 5,
    "seed": 42
}

# Paths
INSTANCES_ROOT = "./instances"
OUTPUT_FILE = "./beam_search_test_results.csv"

# Test on same sizes as tuning: 50, 100, 200, 500, 1000
# 3 instances per size (matching tuning setup)
TEST_SIZES = ['50', '100', '200', '500', '1000']
INSTANCES_PER_SIZE = 3

# ============================================================================

def get_algorithm_function(algorithm_name: str):
    """Import and return the algorithm function"""
    import importlib
    
    algorithm_map = {
        "beam_search": ("beam_search", "beam_search_construction"),
        "randomized_greedy": ("rand_cons_heu_delta", "randomized_greedy_construction"),
        "greedy_construction": ("det_cons_heu_delta", "greedy_construction"),
    }
    
    if algorithm_name not in algorithm_map:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    module_name, function_name = algorithm_map[algorithm_name]
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def run_beam_search_test():
    """Run best beam search config on test instances"""
    
    print("=" * 80)
    print("TESTING BEST BEAM SEARCH CONFIGURATION ON TEST INSTANCES")
    print("=" * 80)
    print(f"Algorithm: {ALGORITHM_NAME}")
    print(f"Parameters: {BEST_PARAMS}")
    print(f"Test sizes: {TEST_SIZES}")
    print(f"Instances per size: {INSTANCES_PER_SIZE}")
    print(f"Instances directory: {os.path.abspath(INSTANCES_ROOT)}")
    print("=" * 80)
    print()
    
    # Check if instances directory exists
    if not os.path.exists(INSTANCES_ROOT):
        print(f"ERROR: Instances directory not found: {os.path.abspath(INSTANCES_ROOT)}")
        print("\nPlease update INSTANCES_ROOT in the script to point to your instances folder.")
        print("Current working directory:", os.getcwd())
        return
    
    # Get algorithm function
    algorithm_func = get_algorithm_function(ALGORITHM_NAME)
    
    results = []
    
    for size in TEST_SIZES:
        test_dir = os.path.join(INSTANCES_ROOT, size, "test")
        
        if not os.path.exists(test_dir):
            print(f"⚠ Skipping size {size}: test directory not found ({test_dir})")
            continue
        
        # Get first N test instances
        test_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.txt')])[:INSTANCES_PER_SIZE]
        
        if not test_files:
            print(f"⚠ Skipping size {size}: no test instances found")
            continue
        
        print(f"Size {size}:")
        
        for test_filename in test_files:
            test_file = os.path.join(test_dir, test_filename)
            instance_name = get_instance_name(test_file)
            print(f"  Testing {instance_name}...", end=" ", flush=True)
            
            try:
                # Load instance
                instance = SCFPDPInstance(test_file)
                
                # Run beam search with best parameters
                start_time = time.time()
                
                print(f"Running...", end=" ", flush=True)
                solution = algorithm_func(instance, **BEST_PARAMS)
                runtime = time.time() - start_time
                
                print(f"Done. ", end="", flush=True)
                
                # Evaluate solution
                feasible, msg = solution.is_feasible()
                objective = solution.objective_value() if feasible else float('inf')
                
                # Store result
                results.append({
                    'size': size,
                    'instance': instance_name,
                    'objective': objective,
                    'runtime': runtime,
                    'feasible': 'Yes' if feasible else 'No',
                    'message': 'OK' if feasible else msg,
                    **{f'param_{k}': v for k, v in BEST_PARAMS.items()}
                })
                
                # Print result
                status = "✓" if feasible else "✗"
                print(f"{status} Obj: {objective:.2f}, Time: {runtime:.3f}s")
                
            except KeyboardInterrupt:
                print(f"\n⚠ Interrupted by user. Saving partial results...")
                break
            except Exception as e:
                print(f"✗ Error: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    'size': size,
                    'instance': instance_name,
                    'objective': 'ERROR',
                    'runtime': 0,
                    'feasible': 'No',
                    'message': str(e),
                    **{f'param_{k}': v for k, v in BEST_PARAMS.items()}
                })
        
        print()
    
    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS...")
    print("=" * 80)
    
    if results:
        fieldnames = list(results[0].keys())
        with open(OUTPUT_FILE, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                # Format numbers for CSV
                formatted_row = row.copy()
                if isinstance(formatted_row.get('objective'), float):
                    formatted_row['objective'] = f"{formatted_row['objective']:.2f}"
                if isinstance(formatted_row.get('runtime'), float):
                    formatted_row['runtime'] = f"{formatted_row['runtime']:.3f}"
                writer.writerow(formatted_row)
        
        print("=" * 80)
        print(f"✓ Results saved to: {os.path.abspath(OUTPUT_FILE)}")
        print(f"Total test instances: {len(results)}")
        feasible_count = sum(1 for r in results if r['feasible'] == 'Yes')
        print(f"Feasible solutions: {feasible_count}/{len(results)}")
        
        if feasible_count > 0:
            feasible_objs = [r['objective'] for r in results if r['feasible'] == 'Yes' and isinstance(r['objective'], (int, float))]
            feasible_times = [r['runtime'] for r in results if r['feasible'] == 'Yes' and isinstance(r['runtime'], (int, float))]
            
            if feasible_objs:
                avg_obj = sum(feasible_objs) / len(feasible_objs)
                print(f"Average objective: {avg_obj:.2f}")
            
            if feasible_times:
                avg_time = sum(feasible_times) / len(feasible_times)
                print(f"Average runtime: {avg_time:.3f}s")
        
        # Summary by size
        print("\nSummary by Size:")
        print("-" * 80)
        for size in TEST_SIZES:
            size_results = [r for r in results if r['size'] == size and r['feasible'] == 'Yes']
            if size_results:
                size_objs = [r['objective'] for r in size_results if isinstance(r['objective'], (int, float))]
                size_times = [r['runtime'] for r in size_results if isinstance(r['runtime'], (int, float))]
                
                if size_objs:
                    avg_obj = sum(size_objs) / len(size_objs)
                    avg_time = sum(size_times) / len(size_times) if size_times else 0
                    print(f"Size {size:>5}: Avg Obj = {avg_obj:>10.2f}, Avg Time = {avg_time:>6.3f}s ({len(size_results)} instances)")
        
        print("=" * 80)
    else:
        print("=" * 80)
        print("⚠ No results to save! Check your instances directory structure.")
        print("=" * 80)


if __name__ == "__main__":
    run_beam_search_test()