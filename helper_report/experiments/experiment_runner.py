# -*- coding: utf-8 -*-
"""
General Parameter Tuning Framework for VRP Algorithms

This script allows you to:
1. Test different parameter configurations for any algorithm
2. Run experiments on a subset of instances (2 per size)
3. Save results in organized CSV files
4. Compare parameter impact on solution quality

Usage:
    python parameter_tuning.py <algorithm_name> <config_file>
    
Example:
    python parameter_tuning.py greedy_construction greedy_configs.json
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
    
    Config format:
    {
        "algorithm": "randomized_greedy",
        "parameters": {
            "alpha": [0.0, 0.1, 0.3, 0.5],
            "seed": [42]
        }
    }
    
    This will generate all combinations of parameters.
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
    size_dirs = ['50', '100', '200', '500', '1000', '2000', '5000', '10000']
    
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
# ALGORITHM EXECUTION
# ============================================================================

def run_algorithm_with_params(algorithm_func: Callable, 
                              instance: SCFPDPInstance,
                              params: Dict[str, Any]) -> tuple:
    """
    Run algorithm with specific parameters and return (solution, runtime, error)
    """
    try:
        start_time = time.time()
        
        # Call algorithm with unpacked parameters
        solution = algorithm_func(instance, **params)
        
        runtime = time.time() - start_time
        
        # Validate solution
        feasible, msg = solution.is_feasible()
        objective = solution.objective_value() if feasible else float('inf')
        
        return solution, runtime, objective, feasible, msg, None
        
    except Exception as e:
        import traceback
        error_msg = f"ERROR: {str(e)}\n{traceback.format_exc()}"
        return None, 0, float('inf'), False, "Error", error_msg


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_parameter_tuning(algorithm_name: str,
                        algorithm_func: Callable,
                        configs: List[ParameterConfig],
                        instances: Dict[str, List[str]],
                        output_dir: str):
    """
    Run parameter tuning experiments and save results.
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"✓ Created output directory: {output_dir}")
    
    # Results storage
    all_results = []
    
    print("=" * 80)
    print(f"PARAMETER TUNING: {algorithm_name}")
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
        
        for size, instance_files in sorted(instances.items()):
            print(f"Size {size}:")
            
            for inst_file in instance_files:
                inst_name = get_instance_name(inst_file)
                print(f"  Processing {inst_name}...", end=" ", flush=True)
                
                # Load instance
                instance = SCFPDPInstance(inst_file)
                
                # Run algorithm
                solution, runtime, objective, feasible, msg, error = run_algorithm_with_params(
                    algorithm_func, instance, config.params
                )
                
                # Store result
                result = {
                    'config_id': config.config_id,
                    'size': size,
                    'instance': inst_name,
                    'objective': objective,
                    'runtime': runtime,
                    'feasible': feasible,
                    'message': msg if not error else error,
                    **{f'param_{k}': v for k, v in config.params.items()}
                }
                
                config_results.append(result)
                all_results.append(result)
                
                # Print result
                status = "✓" if feasible else "✗"
                print(f"{status} Obj: {objective:.2f}, Time: {runtime:.3f}s")
        
        # Save intermediate results for this config
        config_csv = os.path.join(output_dir, f"{config.config_id}_results.csv")
        save_results_to_csv(config_results, config_csv)
        print(f"  ✓ Saved: {config_csv}")
    
    # Save complete results
    complete_csv = os.path.join(output_dir, f"{algorithm_name}_parameter_tuning_complete.csv")
    save_results_to_csv(all_results, complete_csv)
    print(f"\n✓ Saved complete results: {complete_csv}")
    
    # Generate summary
    summary_csv = os.path.join(output_dir, f"{algorithm_name}_parameter_tuning_summary.csv")
    generate_summary(all_results, configs, summary_csv)
    print(f"✓ Saved summary: {summary_csv}")
    
    print("\n" + "=" * 80)
    print("PARAMETER TUNING COMPLETE")
    print("=" * 80)
    print(f"Complete results: {complete_csv}")
    print(f"Summary: {summary_csv}")
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
    
    print(f"  ✓ Saved {len(results)} results to: {filename}")


def generate_summary(all_results: List[Dict], 
                     configs: List[ParameterConfig],
                     summary_file: str):
    """
    Generate summary statistics for each configuration.
    """
    from collections import defaultdict
    import statistics
    
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
                **{f'param_{k}': v for k, v in config.params.items()}
            })
            continue
        
        objectives = [r['objective'] for r in config_results]
        runtimes = [r['runtime'] for r in config_results]
        
        summary_data.append({
            'config_id': config.config_id,
            'total_instances': len(config_results),
            'feasible': len(config_results),
            'avg_objective': statistics.mean(objectives),
            'std_objective': statistics.stdev(objectives) if len(objectives) > 1 else 0,
            'min_objective': min(objectives),
            'max_objective': max(objectives),
            'avg_runtime': statistics.mean(runtimes),
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
                # Format numbers
                formatted = row.copy()
                for key in ['avg_objective', 'std_objective', 'min_objective', 'max_objective', 'avg_runtime']:
                    if isinstance(formatted.get(key), float):
                        formatted[key] = f"{formatted[key]:.2f}"
                writer.writerow(formatted)
    
    print(f"  ✓ Saved summary with {len(summary_data)} configurations")
    
    # Generate analysis report
    analysis_file = summary_file.replace('_summary.csv', '_analysis.txt')
    generate_analysis_report(summary_data, configs, analysis_file)
    print(f"  ✓ Saved analysis report: {analysis_file}")


def generate_analysis_report(summary_data: List[Dict], 
                             configs: List[ParameterConfig],
                             analysis_file: str):
    """
    Generate a detailed analysis report with tables and interpretation.
    """
    import statistics
    
    with open(analysis_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PARAMETER TUNING ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Extract algorithm name and parameters from first config
        if not summary_data:
            f.write("No data available for analysis.\n")
            return
        
        # Get parameter names (exclude non-parameter fields)
        param_keys = [k for k in summary_data[0].keys() if k.startswith('param_')]
        param_names = [k.replace('param_', '') for k in param_keys]
        
        f.write(f"Algorithm: {configs[0].params if configs else 'Unknown'}\n")
        f.write(f"Parameters tested: {', '.join(param_names)}\n")
        f.write(f"Total configurations: {len(summary_data)}\n")
        f.write(f"Instances per configuration: {summary_data[0].get('total_instances', 'N/A')}\n")
        f.write("\n" + "=" * 80 + "\n\n")
        
        # ==================================================================
        # TABLE 1: SUMMARY STATISTICS FOR ALL CONFIGURATIONS
        # ==================================================================
        f.write("TABLE 1: SUMMARY STATISTICS FOR ALL CONFIGURATIONS\n")
        f.write("-" * 80 + "\n")
        
        # Determine column widths
        config_width = 12
        param_width = 10
        obj_width = 12
        time_width = 10
        
        # Header
        header = f"{'Config':<{config_width}}"
        for param_name in param_names:
            header += f"{param_name:<{param_width}}"
        header += f"{'Avg Obj':<{obj_width}}{'Std Obj':<{obj_width}}{'Avg Time(s)':<{time_width}}"
        f.write(header + "\n")
        f.write("-" * 80 + "\n")
        
        # Data rows
        for row in summary_data:
            line = f"{row['config_id']:<{config_width}}"
            for param_key in param_keys:
                value = row.get(param_key, 'N/A')
                line += f"{str(value):<{param_width}}"
            
            avg_obj = row.get('avg_objective', 'N/A')
            std_obj = row.get('std_objective', 'N/A')
            avg_time = row.get('avg_runtime', 'N/A')
            
            if isinstance(avg_obj, str) and avg_obj != 'N/A':
                avg_obj = float(avg_obj)
            if isinstance(std_obj, str) and std_obj != 'N/A':
                std_obj = float(std_obj)
            if isinstance(avg_time, str) and avg_time != 'N/A':
                avg_time = float(avg_time)
            
            line += f"{avg_obj if isinstance(avg_obj, str) else f'{avg_obj:.2f}':<{obj_width}}"
            line += f"{std_obj if isinstance(std_obj, str) else f'{std_obj:.2f}':<{obj_width}}"
            line += f"{avg_time if isinstance(avg_time, str) else f'{avg_time:.3f}':<{time_width}}"
            f.write(line + "\n")
        
        f.write("\n" + "=" * 80 + "\n\n")
        
        # ==================================================================
        # TABLE 2: TOP 5 CONFIGURATIONS
        # ==================================================================
        f.write("TABLE 2: TOP 5 BEST CONFIGURATIONS (by Average Objective)\n")
        f.write("-" * 80 + "\n")
        
        top_5 = [row for row in summary_data if isinstance(row.get('avg_objective'), (int, float))][:5]
        
        f.write(f"{'Rank':<6}{'Config':<12}{'Avg Objective':<15}{'Parameters':<40}\n")
        f.write("-" * 80 + "\n")
        
        for rank, row in enumerate(top_5, 1):
            params_str = ', '.join([f"{k.replace('param_', '')}={row[k]}" for k in param_keys])
            f.write(f"{rank:<6}{row['config_id']:<12}{row['avg_objective']:<15.2f}{params_str:<40}\n")
        
        f.write("\n" + "=" * 80 + "\n\n")
        
        # ==================================================================
        # TABLE 3: PARAMETER IMPACT ANALYSIS
        # ==================================================================
        f.write("TABLE 3: PARAMETER IMPACT ANALYSIS\n")
        f.write("-" * 80 + "\n")
        
        # Analyze impact of each parameter
        for param_name, param_key in zip(param_names, param_keys):
            f.write(f"\nParameter: {param_name}\n")
            f.write("-" * 40 + "\n")
            
            # Group by parameter value
            by_param_value = defaultdict(list)
            for row in summary_data:
                if isinstance(row.get('avg_objective'), (int, float)):
                    param_value = row[param_key]
                    by_param_value[param_value].append(row['avg_objective'])
            
            # Calculate statistics for each value
            param_stats = []
            for value, objectives in sorted(by_param_value.items()):
                param_stats.append({
                    'value': value,
                    'avg_obj': statistics.mean(objectives),
                    'std_obj': statistics.stdev(objectives) if len(objectives) > 1 else 0,
                    'count': len(objectives)
                })
            
            # Sort by average objective
            param_stats.sort(key=lambda x: x['avg_obj'])
            
            f.write(f"{'Value':<15}{'Avg Objective':<15}{'Std Dev':<15}{'Count':<10}\n")
            f.write("-" * 40 + "\n")
            for stat in param_stats:
                f.write(f"{str(stat['value']):<15}{stat['avg_obj']:<15.2f}{stat['std_obj']:<15.2f}{stat['count']:<10}\n")
            
            # Best value for this parameter
            if param_stats:
                best = param_stats[0]
                f.write(f"\n→ Best {param_name} value: {best['value']} (avg obj: {best['avg_obj']:.2f})\n")
        
        f.write("\n" + "=" * 80 + "\n\n")
        
        # ==================================================================
        # ANALYSIS AND RECOMMENDATIONS
        # ==================================================================
        f.write("ANALYSIS AND RECOMMENDATIONS\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall best configuration
        if top_5:
            best_config = top_5[0]
            f.write("1. BEST OVERALL CONFIGURATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"   Configuration: {best_config['config_id']}\n")
            f.write(f"   Average Objective: {best_config['avg_objective']:.2f}\n")
            f.write(f"   Standard Deviation: {best_config.get('std_objective', 'N/A')}\n")
            f.write(f"   Average Runtime: {best_config.get('avg_runtime', 'N/A')} seconds\n")
            f.write(f"   Parameters:\n")
            for param_key, param_name in zip(param_keys, param_names):
                f.write(f"      - {param_name}: {best_config[param_key]}\n")
            f.write("\n")
        
        # Parameter sensitivity analysis
        f.write("2. PARAMETER SENSITIVITY ANALYSIS\n")
        f.write("-" * 80 + "\n")
        
        for param_name, param_key in zip(param_names, param_keys):
            # Calculate range of objectives for this parameter
            by_param_value = defaultdict(list)
            for row in summary_data:
                if isinstance(row.get('avg_objective'), (int, float)):
                    param_value = row[param_key]
                    by_param_value[param_value].append(row['avg_objective'])
            
            if len(by_param_value) > 1:
                param_avgs = [statistics.mean(objs) for objs in by_param_value.values()]
                obj_range = max(param_avgs) - min(param_avgs)
                overall_avg = statistics.mean([obj for objs in by_param_value.values() for obj in objs])
                sensitivity = (obj_range / overall_avg) * 100 if overall_avg > 0 else 0
                
                f.write(f"   {param_name}:\n")
                f.write(f"      - Objective range: {obj_range:.2f}\n")
                f.write(f"      - Sensitivity: {sensitivity:.2f}% of average objective\n")
                
                if sensitivity > 10:
                    f.write(f"      - Impact: HIGH - This parameter significantly affects solution quality\n")
                elif sensitivity > 5:
                    f.write(f"      - Impact: MEDIUM - This parameter moderately affects solution quality\n")
                else:
                    f.write(f"      - Impact: LOW - This parameter has minimal effect on solution quality\n")
                f.write("\n")
        
        # Performance vs Runtime tradeoff
        f.write("3. PERFORMANCE vs RUNTIME TRADEOFF\n")
        f.write("-" * 80 + "\n")
        
        valid_configs = [row for row in summary_data 
                        if isinstance(row.get('avg_objective'), (int, float)) 
                        and isinstance(row.get('avg_runtime'), (int, float))]
        
        if valid_configs:
            # Find fastest config
            fastest = min(valid_configs, key=lambda x: float(x['avg_runtime']))
            # Find best quality config
            best_quality = min(valid_configs, key=lambda x: float(x['avg_objective']))
            
            # Format parameter strings outside f-strings
            fastest_params = ', '.join([f"{k.replace('param_', '')}={fastest[k]}" for k in param_keys])
            best_params = ', '.join([f"{k.replace('param_', '')}={best_quality[k]}" for k in param_keys])
            
            f.write(f"   Fastest Configuration: {fastest['config_id']}\n")
            f.write(f"      - Runtime: {fastest['avg_runtime']:.3f}s\n")
            f.write(f"      - Objective: {fastest['avg_objective']:.2f}\n")
            f.write(f"      - Parameters: {fastest_params}\n\n")
            
            f.write(f"   Best Quality Configuration: {best_quality['config_id']}\n")
            f.write(f"      - Objective: {best_quality['avg_objective']:.2f}\n")
            f.write(f"      - Runtime: {best_quality['avg_runtime']:.3f}s\n")
            f.write(f"      - Parameters: {best_params}\n\n")
            
            if fastest['config_id'] != best_quality['config_id']:
                time_diff = float(best_quality['avg_runtime']) - float(fastest['avg_runtime'])
                obj_diff = float(fastest['avg_objective']) - float(best_quality['avg_objective'])
                f.write(f"   Trade-off: Spending {time_diff:.3f}s more improves objective by {obj_diff:.2f}\n")
                f.write(f"   Efficiency: {obj_diff/time_diff:.2f} objective improvement per second\n\n")
        
        # Recommendations
        f.write("4. RECOMMENDATIONS\n")
        f.write("-" * 80 + "\n")
        
        if top_5:
            best = top_5[0]
            best_params_str = ', '.join([f"{k.replace('param_', '')}={best[k]}" for k in param_keys])
            f.write(f"   ✓ For best solution quality, use configuration {best['config_id']}\n")
            f.write(f"     Parameters: {best_params_str}\n\n")
        
        if valid_configs and fastest['config_id'] != best_quality['config_id']:
            fastest_params_str = ', '.join([f"{k.replace('param_', '')}={fastest[k]}" for k in param_keys])
            f.write(f"   ✓ For fastest runtime, use configuration {fastest['config_id']}\n")
            f.write(f"     Parameters: {fastest_params_str}\n\n")
        
        # Top 3 balanced configs (good quality and reasonable time)
        if len(valid_configs) >= 3:
            # Score based on normalized objective and runtime
            min_obj = min(float(c['avg_objective']) for c in valid_configs)
            max_obj = max(float(c['avg_objective']) for c in valid_configs)
            min_time = min(float(c['avg_runtime']) for c in valid_configs)
            max_time = max(float(c['avg_runtime']) for c in valid_configs)
            
            for config in valid_configs:
                norm_obj = (float(config['avg_objective']) - min_obj) / (max_obj - min_obj + 1e-6)
                norm_time = (float(config['avg_runtime']) - min_time) / (max_time - min_time + 1e-6)
                config['balance_score'] = norm_obj + 0.3 * norm_time  # Weight quality higher
            
            balanced = sorted(valid_configs, key=lambda x: x['balance_score'])[:3]
            
            f.write(f"   ✓ Top 3 balanced configurations (quality + speed):\n")
            for i, config in enumerate(balanced, 1):
                config_params_str = ', '.join([f"{k.replace('param_', '')}={config[k]}" for k in param_keys])
                f.write(f"     {i}. {config['config_id']}: Obj={config['avg_objective']:.2f}, Time={config['avg_runtime']:.3f}s\n")
                f.write(f"        Parameters: {config_params_str}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n")


# ============================================================================
# ALGORITHM REGISTRY
# ============================================================================

def get_algorithm_function(algorithm_name: str) -> Callable:
    """
    Import and return the algorithm function based on name.
    Uses lazy imports to avoid circular dependencies.
    Add your algorithms here!
    """
    
    # Import inside function to avoid circular imports
    import importlib
    
    # Define module and function mappings
    algorithm_map = {
        "greedy_construction": ("det_cons_heu_delta", "greedy_construction"),
        "randomized_greedy": ("rand_cons_heu_delta", "randomized_greedy_construction"),
        "greedy_tabu_best": ("tabu_search", "greedy_tabu_best"),
        "greedy_tabu_first": ("tabu_search", "greedy_tabu_first"),
        "random_tabu_best": ("tabu_search", "random_tabu_best"),
        "random_tabu_first": ("tabu_search", "random_tabu_first"),
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
    # ========================================================================
    # STATIC CONFIGURATION - All configs defined here, no external files needed
    # ========================================================================
    
    # Define all your parameter configurations here
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
            "parameters": {}  # No parameters - baseline
        },
        
        # Add more configs as needed
        "tabu_search": {
            "algorithm": "greedy_tabu_best",
            "parameters": {
                "tabu_tenure": [5, 7, 10, 15],
            }
        },
    }
    
    # ========================================================================
    # SELECT WHICH CONFIG TO RUN - Just change this variable
    # ========================================================================
    
    ACTIVE_CONFIG = "tabu_search"  # ← Change this to run different experiments
    
    # Other settings
    instances_root = "./instances"
    output_dir = "./parameter_tuning_results"
    
    # ========================================================================
    
    print("=" * 80)
    print("PARAMETER TUNING - STATIC CONFIGURATION")
    print("=" * 80)
    print(f"Active config: {ACTIVE_CONFIG}")
    print(f"Instances: {instances_root}")
    print(f"Output: {output_dir}")
    print("=" * 80)
    print()
    
    # Get the selected configuration
    if ACTIVE_CONFIG not in ALL_CONFIGS:
        print(f"ERROR: Unknown configuration '{ACTIVE_CONFIG}'")
        print(f"Available configurations: {list(ALL_CONFIGS.keys())}")
        sys.exit(1)
    
    config_data = ALL_CONFIGS[ACTIVE_CONFIG]
    
    algorithm_name = config_data['algorithm']
    configs = load_parameter_configs(config_data)
    
    print(f"Algorithm: {algorithm_name}")
    print(f"Configurations to test: {len(configs)}")
    
    # Get algorithm function
    algorithm_func = get_algorithm_function(algorithm_name)
    
    # Select test instances (2 per size)
    instances = select_test_instances(instances_root, instances_per_size=5)
    
    # Create algorithm-specific output directory
    algo_output_dir = os.path.join(output_dir, algorithm_name)
    
    # Run experiments
    run_parameter_tuning(
        algorithm_name=algorithm_name,
        algorithm_func=algorithm_func,
        configs=configs,
        instances=instances,
        output_dir=algo_output_dir
    )


if __name__ == "__main__":
    main()