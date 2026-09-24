# -*- coding: utf-8 -*-
"""
General Parameter Tuning Framework for VRP Algorithms with Convergence Tracking

This script tracks:
1. Number of iterations for each algorithm run
2. Objective value at each step
3. Fairness value at each step

Usage:
    python parameter_tuning_with_convergence.py
"""

import os
import sys
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Callable
import itertools
from collections import defaultdict
import statistics

# Add parent directory to path to import VRP module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from VRP import (
    SCFPDPInstance, Solution, 
    write_solution, read_solution, get_instance_name
)


# ============================================================================
# CONFIGURATION
# ============================================================================

class ParameterConfig:
    """Represents a parameter configuration to test"""
    def __init__(self, config_dict: Dict[str, Any], config_id: str):
        self.params = config_dict
        self.config_id = config_id
    
    def __repr__(self):
        return f"Config({self.config_id}): {self.params}"


def load_parameter_configs(config_data: Dict) -> List[ParameterConfig]:
    """
    Generate parameter configurations from config dictionary.
    """
    param_names = list(config_data['parameters'].keys())
    param_values = list(config_data['parameters'].values())
    
    # Generate all combinations
    configs = []
    for i, combination in enumerate(itertools.product(*param_values), 1):
        config_dict = dict(zip(param_names, combination))
        config_id = f"config_{i:03d}"
        configs.append(ParameterConfig(config_dict, config_id))
    
    return configs


# ============================================================================
# INSTANCE SELECTION
# ============================================================================

def select_test_instances(instances_root: str, instances_per_size: int = 2) -> Dict[str, List[str]]:
    """
    Select a subset of instances for parameter tuning.
    Returns: {size: [instance_file_paths]}
    """
    size_dirs = ['50', '100', '200', '500', '1000', '2000']
    
    selected_instances = {}
    
    for size in size_dirs:
        train_dir = os.path.join(instances_root, size, "train")
        
        if not os.path.exists(train_dir):
            print(f"Warning: {train_dir} not found, skipping size {size}")
            continue
        
        # Get all instance files
        instance_files = sorted([
            os.path.join(train_dir, f) 
            for f in os.listdir(train_dir) 
            if f.endswith('.txt')
        ])
        
        # Select first N instances (deterministic selection)
        selected = instance_files[:instances_per_size]
        selected_instances[size] = selected
        
        print(f"Size {size}: Selected {len(selected)} instances")
    
    return selected_instances


# ============================================================================
# ALGORITHM EXECUTION WITH CONVERGENCE TRACKING
# ============================================================================

def run_algorithm_with_params(algorithm_func: Callable, 
                              instance: SCFPDPInstance,
                              params: Dict[str, Any]) -> tuple:
    """
    Run algorithm with specific parameters and return:
    (solution, runtime, objective, feasible, msg, error, convergence_data)
    
    convergence_data contains:
    - iterations: number of iterations
    - objective_history: list of objective values at each step
    - fairness_history: list of fairness values at each step
    """
    try:
        start_time = time.time()
        
        # Call algorithm with unpacked parameters and capture convergence data
        result = algorithm_func(instance, **params)
        
        # Check if algorithm returns convergence data
        if isinstance(result, tuple) and len(result) == 2:
            solution, convergence_data = result
        else:
            # Old-style algorithm that only returns solution
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
        
        return solution, runtime, objective, feasible, msg, None, convergence_data
        
    except Exception as e:
        import traceback
        error_msg = f"ERROR: {str(e)}\n{traceback.format_exc()}"
        empty_convergence = {
            'iterations': 0,
            'objective_history': [],
            'fairness_history': []
        }
        return None, 0, float('inf'), False, "Error", error_msg, empty_convergence


# ============================================================================
# EXPERIMENT RUNNER WITH CONVERGENCE TRACKING
# ============================================================================

def run_parameter_tuning(algorithm_name: str,
                        algorithm_func: Callable,
                        configs: List[ParameterConfig],
                        instances: Dict[str, List[str]],
                        output_dir: str):
    """
    Run parameter tuning experiments and save results including convergence data.
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    convergence_dir = os.path.join(output_dir, "convergence_data")
    os.makedirs(convergence_dir, exist_ok=True)
    print(f"✓ Created output directory: {output_dir}")
    print(f"✓ Created convergence directory: {convergence_dir}")
    
    # Results storage
    all_results = []
    all_convergence = []  # Store convergence data separately
    
    print("=" * 80)
    print(f"PARAMETER TUNING WITH CONVERGENCE TRACKING: {algorithm_name}")
    print("=" * 80)
    print(f"Number of configurations: {len(configs)}")
    print(f"Total instances: {sum(len(v) for v in instances.values())}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    print()
    
    # Check if we have instances
    if not instances:
        print("ERROR: No instances found!")
        return
    
    total_experiments = len(configs) * sum(len(v) for v in instances.values())
    print(f"Total experiments to run: {total_experiments}")
    print()
    
    # Run experiments
    for config_idx, config in enumerate(configs, 1):
        print(f"\n{'='*80}")
        print(f"Configuration {config_idx}/{len(configs)}: {config.config_id}")
        print(f"Parameters: {config.params}")
        print(f"{'='*80}\n")
        
        config_results = []
        config_convergence = []
        
        for size, instance_files in sorted(instances.items()):
            print(f"Size {size}:")
            
            for inst_file in instance_files:
                inst_name = get_instance_name(inst_file)
                print(f"  Processing {inst_name}...", end=" ", flush=True)
                
                # Load instance
                instance = SCFPDPInstance(inst_file)
                
                # Run algorithm with convergence tracking
                solution, runtime, objective, feasible, msg, error, convergence_data = run_algorithm_with_params(
                    algorithm_func, instance, config.params
                )
                
                # Store main result
                result = {
                    'config_id': config.config_id,
                    'size': size,
                    'instance': inst_name,
                    'objective': objective,
                    'runtime': runtime,
                    'feasible': feasible,
                    'iterations': convergence_data.get('iterations', 0),
                    'message': msg if not error else error,
                    **{f'param_{k}': v for k, v in config.params.items()}
                }
                
                config_results.append(result)
                all_results.append(result)
                
                # Store convergence data
                convergence_record = {
                    'config_id': config.config_id,
                    'size': size,
                    'instance': inst_name,
                    'iterations': convergence_data.get('iterations', 0),
                    'objective_history': convergence_data.get('objective_history', []),
                    'fairness_history': convergence_data.get('fairness_history', []),
                    **{f'param_{k}': v for k, v in config.params.items()}
                }
                config_convergence.append(convergence_record)
                all_convergence.append(convergence_record)
                
                # Print result
                status = "✓" if feasible else "✗"
                print(f"{status} Obj: {objective:.2f}, Time: {runtime:.3f}s, Iters: {convergence_data.get('iterations', 0)}")
        
        # Save intermediate results for this config
        config_csv = os.path.join(output_dir, f"{config.config_id}_results.csv")
        save_results_to_csv(config_results, config_csv)
        
        # Save convergence data for this config
        convergence_csv = os.path.join(convergence_dir, f"{config.config_id}_convergence.csv")
        save_convergence_to_csv(config_convergence, convergence_csv)
        
        print(f"  ✓ Saved results: {config_csv}")
        print(f"  ✓ Saved convergence: {convergence_csv}")
    
    # Save complete results
    complete_csv = os.path.join(output_dir, f"{algorithm_name}_parameter_tuning_complete.csv")
    save_results_to_csv(all_results, complete_csv)
    print(f"\n✓ Saved complete results: {complete_csv}")
    
    # Save complete convergence data
    complete_convergence_csv = os.path.join(convergence_dir, f"{algorithm_name}_convergence_complete.csv")
    save_convergence_to_csv(all_convergence, complete_convergence_csv)
    print(f"✓ Saved complete convergence: {complete_convergence_csv}")
    
    # Generate summary
    summary_csv = os.path.join(output_dir, f"{algorithm_name}_parameter_tuning_summary.csv")
    generate_summary(all_results, configs, summary_csv)
    print(f"✓ Saved summary: {summary_csv}")
    
    # Generate convergence analysis
    convergence_summary_csv = os.path.join(convergence_dir, f"{algorithm_name}_convergence_summary.csv")
    generate_convergence_summary(all_convergence, convergence_summary_csv)
    print(f"✓ Saved convergence summary: {convergence_summary_csv}")
    
    print("\n" + "=" * 80)
    print("PARAMETER TUNING COMPLETE")
    print("=" * 80)
    print(f"Complete results: {complete_csv}")
    print(f"Summary: {summary_csv}")
    print(f"Convergence data: {convergence_dir}")
    print("=" * 80)


def save_results_to_csv(results: List[Dict], filename: str):
    """Save results to CSV file"""
    if not results:
        print(f"  ⚠ Warning: No results to save for {filename}")
        return
    
    fieldnames = list(results[0].keys())
    
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            # Format numbers
            formatted = result.copy()
            if isinstance(formatted.get('objective'), float):
                formatted['objective'] = f"{formatted['objective']:.2f}" if formatted['objective'] != float('inf') else 'INF'
            if isinstance(formatted.get('runtime'), float):
                formatted['runtime'] = f"{formatted['runtime']:.3f}"
            writer.writerow(formatted)


def save_convergence_to_csv(convergence_data: List[Dict], filename: str):
    """Save convergence data to CSV file with one row per iteration"""
    if not convergence_data:
        print(f"  ⚠ Warning: No convergence data to save for {filename}")
        return
    
    rows = []
    for record in convergence_data:
        config_id = record['config_id']
        size = record['size']
        instance = record['instance']
        iterations = record['iterations']
        obj_history = record['objective_history']
        fair_history = record['fairness_history']
        
        # Get parameter columns
        param_cols = {k: v for k, v in record.items() if k.startswith('param_')}
        
        # Create one row per iteration
        for i in range(len(obj_history)):
            row = {
                'config_id': config_id,
                'size': size,
                'instance': instance,
                'iteration': i,
                'objective': obj_history[i] if i < len(obj_history) else None,
                'fairness': fair_history[i] if i < len(fair_history) else None,
                **param_cols
            }
            rows.append(row)
    
    if rows:
        fieldnames = list(rows[0].keys())
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                formatted = row.copy()
                if isinstance(formatted.get('objective'), float):
                    formatted['objective'] = f"{formatted['objective']:.2f}"
                if isinstance(formatted.get('fairness'), float):
                    formatted['fairness'] = f"{formatted['fairness']:.4f}"
                writer.writerow(formatted)


def generate_convergence_summary(all_convergence: List[Dict], summary_file: str):
    """Generate summary of convergence behavior"""
    summary_data = []
    
    # Group by config and instance
    by_config = defaultdict(list)
    for record in all_convergence:
        key = (record['config_id'], record['size'], record['instance'])
        by_config[key].append(record)
    
    for (config_id, size, instance), records in by_config.items():
        if not records:
            continue
        
        record = records[0]  # Should only be one record per key
        obj_history = record['objective_history']
        fair_history = record['fairness_history']
        
        if not obj_history:
            continue
        
        # Calculate convergence metrics
        initial_obj = obj_history[0] if obj_history else None
        final_obj = obj_history[-1] if obj_history else None
        best_obj = min(obj_history) if obj_history else None
        improvement = initial_obj - final_obj if (initial_obj and final_obj) else None
        improvement_pct = (improvement / initial_obj * 100) if (improvement and initial_obj and initial_obj != 0) else None
        
        initial_fair = fair_history[0] if fair_history else None
        final_fair = fair_history[-1] if fair_history else None
        best_fair = min(fair_history) if fair_history else None
        
        # Get parameter columns
        param_cols = {k: v for k, v in record.items() if k.startswith('param_')}
        
        summary_data.append({
            'config_id': config_id,
            'size': size,
            'instance': instance,
            'total_iterations': record['iterations'],
            'initial_objective': initial_obj,
            'final_objective': final_obj,
            'best_objective': best_obj,
            'improvement': improvement,
            'improvement_pct': improvement_pct,
            'initial_fairness': initial_fair,
            'final_fairness': final_fair,
            'best_fairness': best_fair,
            **param_cols
        })
    
    # Save summary
    if summary_data:
        fieldnames = list(summary_data[0].keys())
        with open(summary_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary_data:
                formatted = row.copy()
                for key in ['initial_objective', 'final_objective', 'best_objective', 
                           'improvement', 'initial_fairness', 'final_fairness', 'best_fairness']:
                    if isinstance(formatted.get(key), float):
                        formatted[key] = f"{formatted[key]:.2f}"
                if isinstance(formatted.get('improvement_pct'), float):
                    formatted['improvement_pct'] = f"{formatted['improvement_pct']:.2f}"
                writer.writerow(formatted)


def generate_summary(all_results: List[Dict], 
                     configs: List[ParameterConfig],
                     summary_file: str):
    """
    Generate summary statistics for each configuration.
    """
    # Group results by config
    by_config = defaultdict(list)
    for result in all_results:
        if result['feasible']:
            by_config[result['config_id']].append(result)
    
    summary_data = []
    
    for config in configs:
        config_results = by_config[config.config_id]
        
        if not config_results:
            summary_data.append({
                'config_id': config.config_id,
                'total_instances': 0,
                'feasible': 0,
                'avg_objective': 'N/A',
                'std_objective': 'N/A',
                'avg_runtime': 'N/A',
                'avg_iterations': 'N/A',
                **{f'param_{k}': v for k, v in config.params.items()}
            })
            continue
        
        objectives = [r['objective'] for r in config_results]
        runtimes = [r['runtime'] for r in config_results]
        iterations = [r['iterations'] for r in config_results if r.get('iterations', 0) > 0]
        
        summary_data.append({
            'config_id': config.config_id,
            'total_instances': len(config_results),
            'feasible': len(config_results),
            'avg_objective': statistics.mean(objectives),
            'std_objective': statistics.stdev(objectives) if len(objectives) > 1 else 0,
            'min_objective': min(objectives),
            'max_objective': max(objectives),
            'avg_runtime': statistics.mean(runtimes),
            'avg_iterations': statistics.mean(iterations) if iterations else 0,
            'min_iterations': min(iterations) if iterations else 0,
            'max_iterations': max(iterations) if iterations else 0,
            **{f'param_{k}': v for k, v in config.params.items()}
        })
    
    # Sort by average objective (best first)
    summary_data.sort(key=lambda x: x['avg_objective'] if isinstance(x['avg_objective'], (int, float)) else float('inf'))
    
    # Save summary
    if summary_data:
        fieldnames = list(summary_data[0].keys())
        with open(summary_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary_data:
                formatted = row.copy()
                for key in ['avg_objective', 'std_objective', 'min_objective', 'max_objective', 
                           'avg_runtime', 'avg_iterations']:
                    if isinstance(formatted.get(key), float):
                        formatted[key] = f"{formatted[key]:.2f}"
                for key in ['min_iterations', 'max_iterations']:
                    if isinstance(formatted.get(key), int):
                        formatted[key] = str(formatted[key])
                writer.writerow(formatted)


# ============================================================================
# ALGORITHM REGISTRY
# ============================================================================

def get_algorithm_function(algorithm_name: str) -> Callable:
    """
    Import and return the algorithm function based on name.
    """
    import importlib
    
    algorithm_map = {
        "greedy_construction": ("det_cons_heu_delta", "greedy_construction"),
        "randomized_greedy": ("rand_cons_heu_delta", "randomized_greedy_construction"),
        "greedy_tabu_best": ("tabu_search", "greedy_tabu_best"),
        "greedy_tabu_first": ("tabu_search", "greedy_tabu_first"),
        "random_tabu_best": ("tabu_search", "random_tabu_best"),
        "random_tabu_first": ("tabu_search_2", "random_tabu_first"),
        "beam_search": ("beam_search", "beam_search_construction"),
    }
    
    if algorithm_name not in algorithm_map:
        raise ValueError(f"Unknown algorithm: {algorithm_name}. Available: {list(algorithm_map.keys())}")
    
    module_name, function_name = algorithm_map[algorithm_name]
    
    try:
        module = importlib.import_module(module_name)
        return getattr(module, function_name)
    except ImportError as e:
        raise ImportError(f"Could not import {function_name} from {module_name}: {e}")
    except AttributeError as e:
        raise AttributeError(f"Function {function_name} not found in module {module_name}: {e}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    ALL_CONFIGS = {
        "randomized_greedy": {
            "algorithm": "randomized_greedy",
            "parameters": {
                "alpha": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0],
                "seed": [42]
            }
        },
        
        "beam_search": {
            "algorithm": "beam_search",
            "parameters": {
                "beam_width": [2, 3, 5],
                "seed": [42]
            }
        },
        
        "greedy_construction": {
            "algorithm": "greedy_construction",
            "parameters": {}
        },
        
        "tabu_search_2": {
            "algorithm": "random_tabu_first",
            "parameters": {
                "tabu_tenure": [5, 7, 10, 15],
            }
        },
    }
    
    ACTIVE_CONFIG = "tabu_search_2"
    
    instances_root = "./instances"
    output_dir = "./parameter_tuning_results"
    
    print("=" * 80)
    print("PARAMETER TUNING WITH CONVERGENCE TRACKING")
    print("=" * 80)
    print(f"Active config: {ACTIVE_CONFIG}")
    print(f"Instances: {instances_root}")
    print(f"Output: {output_dir}")
    print("=" * 80)
    print()
    
    if ACTIVE_CONFIG not in ALL_CONFIGS:
        print(f"ERROR: Unknown configuration '{ACTIVE_CONFIG}'")
        print(f"Available configurations: {list(ALL_CONFIGS.keys())}")
        sys.exit(1)
    
    config_data = ALL_CONFIGS[ACTIVE_CONFIG]
    algorithm_name = config_data['algorithm']
    configs = load_parameter_configs(config_data)
    
    print(f"Algorithm: {algorithm_name}")
    print(f"Configurations to test: {len(configs)}")
    
    algorithm_func = get_algorithm_function(algorithm_name)
    instances = select_test_instances(instances_root, instances_per_size=5)
    algo_output_dir = os.path.join(output_dir, algorithm_name)
    
    run_parameter_tuning(
        algorithm_name=algorithm_name,
        algorithm_func=algorithm_func,
        configs=configs,
        instances=instances,
        output_dir=algo_output_dir
    )


if __name__ == "__main__":
    main()