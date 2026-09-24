from typing import Callable, Iterable, Tuple, Dict
import time
from VRP import Solution, SCFPDPInstance
from det_cons_heu_delta import greedy_construction
from rand_cons_heu_delta import randomized_greedy_construction
from neighborhoods import (
    generate_intra_route_reloc,     # N1
    generate_inter_route_reloc,     # N2
    generate_swap_requests,         # N3
)

NeighborhoodFunc = Callable[[Solution], Iterable[Tuple[Solution, float, Dict]]]


def tabu_search_core(inst: SCFPDPInstance,
                neighborhoods: list,
                construction_func: Callable, 
                max_iterations: int = None,
                tabu_tenure: int = 10,
                step_function: str = "best",
                max_neighbors_per_iter: int = None,
                time_limit: float = 840.0) -> Tuple[Solution, Dict]:
    """
    Tabu search with convergence tracking.
    
    Returns:
        Tuple[Solution, Dict]: (best_solution, convergence_data)
        
    convergence_data contains:
        - iterations: total number of iterations performed
        - objective_history: list of objective values at each iteration
        - fairness_history: list of fairness values at each iteration
    """
    print('\n[TABU] ========== TABU SEARCH STARTING ==========')
    print(f'[TABU] Instance size: n={inst.n}')
    print(f'[TABU] Parameters: tabu_tenure={tabu_tenure}, step_function={step_function}')
    
    construction_start = time.time()
    print('[TABU] Running construction heuristic...')
    current = construction_func(inst)
    best_solution = current.copy()
    construction_time = time.time() - construction_start
    print(f"[TABU] Construction completed in {construction_time:.2f}s")
    
    # Initialize convergence tracking
    convergence_data = {
        'iterations': 0,
        'objective_history': [],
        'fairness_history': []
    }
    
    # Record initial state
    print('[TABU] Calculating initial objective...')
    initial_objective = current.objective_value()
    print(f'[TABU] Initial objective: {initial_objective}')
    
    print('[TABU] Calculating initial fairness...')
    try:
        initial_fairness = current.calculate_fairness()
        print(f'[TABU] Initial fairness: {initial_fairness:.4f}')
    except AttributeError:
        print('[TABU] ERROR: calculate_fairness() method not found in Solution class!')
        print('[TABU] Please add calculate_fairness() method to Solution class')
        raise
    except Exception as e:
        print(f'[TABU] ERROR: calculate_fairness() crashed: {e}')
        raise
    
    convergence_data['objective_history'].append(initial_objective)
    convergence_data['fairness_history'].append(initial_fairness)
    print(f'[TABU] Initial state recorded: obj={initial_objective:.2f}, fairness={initial_fairness:.4f}')
    
    # Reserve time for construction and final operations
    remaining_time = time_limit - construction_time - 10.0
    
    tabu_list: list[Dict] = []
    start_time = time.time()
    
    # DYNAMIC parameter calculation based on instance size
    n = inst.n
    
    print(f'\n[TABU] Setting dynamic parameters based on instance size (n={n})...')
    
    if n >= 8000:
        estimated_time_per_iter = 120.0
        max_neighbors_per_iter = 50
        tabu_tenure = 3
    elif n >= 5000:
        estimated_time_per_iter = 60.0
        max_neighbors_per_iter = 150
        tabu_tenure = 5
    elif n >= 2000:
        estimated_time_per_iter = 20.0
        max_neighbors_per_iter = 300
        tabu_tenure = 7
    elif n >= 1000:
        estimated_time_per_iter = 8.0
        max_neighbors_per_iter = 500
        tabu_tenure = 8
    elif n >= 500:
        estimated_time_per_iter = 3.0
        max_neighbors_per_iter = 1000
        tabu_tenure = 10
    elif n >= 200:
        estimated_time_per_iter = 1.5
        max_neighbors_per_iter = 2000
        tabu_tenure = 10
    else:
        estimated_time_per_iter = 0.5
        max_neighbors_per_iter = 5000
        tabu_tenure = 15
    
    max_iterations = max(5, int(remaining_time / estimated_time_per_iter))
    max_iterations = min(max_iterations, 300)
    
    print(f"[TABU] Dynamic parameters set:")
    print(f"[TABU]   - Estimated time per iteration: {estimated_time_per_iter:.1f}s")
    print(f"[TABU]   - Max iterations: {max_iterations}")
    print(f"[TABU]   - Max neighbors per iteration: {max_neighbors_per_iter}")
    print(f"[TABU]   - Tabu tenure: {tabu_tenure}")
    print(f"[TABU]   - Available time: {remaining_time:.1f}s")
    print(f'\n[TABU] Starting main search loop...\n')
    
    iteration = 0
    iterations_without_improvement = 0
    last_best_value = best_solution.objective_value()
    
    while iteration < max_iterations:
        iteration += 1
        
        # Check time limit
        elapsed = time.time() - start_time
        total_elapsed = construction_time + elapsed
        
        if total_elapsed >= time_limit:
            print(f"[Tabu Search] Time limit reached at iteration {iteration}, elapsed={total_elapsed:.1f}s")
            break
        
        # ADAPTIVE exploration
        time_pressure = total_elapsed / time_limit
        
        if iterations_without_improvement > 10:
            convergence_factor = 0.5
        elif iterations_without_improvement > 5:
            convergence_factor = 0.7
        else:
            convergence_factor = 1.0
        
        if time_pressure > 0.85:
            current_max_neighbors = max(50, int(max_neighbors_per_iter * 0.2 * convergence_factor))
        elif time_pressure > 0.7:
            current_max_neighbors = max(100, int(max_neighbors_per_iter * 0.4 * convergence_factor))
        elif time_pressure > 0.5:
            current_max_neighbors = max(200, int(max_neighbors_per_iter * 0.6 * convergence_factor))
        else:
            current_max_neighbors = int(max_neighbors_per_iter * convergence_factor)
        
        # Time estimation check
        if iteration > 1:
            avg_time_per_iter = elapsed / iteration
            estimated_next_iter_time = total_elapsed + avg_time_per_iter
            
            if estimated_next_iter_time > time_limit - 30:
                current_max_neighbors = min(current_max_neighbors, 100)
                print(f"[Tabu Search] Reducing neighbors due to time pressure (iter {iteration})")
            
            if estimated_next_iter_time > time_limit - 10:
                print(f"[Tabu Search] Breaking early at iteration {iteration} to avoid timeout")
                break
        
        best_delta = float('inf')
        best_neighbor = None
        best_move_info = None
        neighbors_checked = 0
        
        # First-improvement or best-improvement
        if step_function == "first":
            found_improving = False
            for neighborhood in neighborhoods:
                if found_improving:
                    break
                for neighbor, delta, move_info in neighborhood(current):
                    neighbors_checked += 1
                    
                    if _is_tabu(move_info, tabu_list) and neighbor.objective_value() >= best_solution.objective_value():
                        continue
                    
                    if delta < 0:
                        best_neighbor = neighbor
                        best_move_info = move_info
                        found_improving = True
                        break
                    
                    if neighbors_checked >= current_max_neighbors:
                        break
        else:  # "best"
            for neighborhood in neighborhoods:
                if neighbors_checked >= current_max_neighbors:
                    break
                    
                for neighbor, delta, move_info in neighborhood(current):
                    neighbors_checked += 1
                    
                    if _is_tabu(move_info, tabu_list) and neighbor.objective_value() >= best_solution.objective_value():
                        continue
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_neighbor = neighbor
                        best_move_info = move_info
                    
                    if best_delta < -50:
                        break
                    
                    if neighbors_checked >= current_max_neighbors:
                        break
        
        # No valid moves found
        if best_neighbor is None:
            print(f'[TABU] No valid moves found at iteration {iteration}. Stopping.')
            break
        
        # Update current solution
        current = best_neighbor
        
        # Update tabu list
        tabu_list.append(best_move_info)
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)
        
        # Track convergence data
        print(f'[TABU] Iteration {iteration}: Recording convergence data...')
        current_objective = current.objective_value()
        try:
            current_fairness = current.calculate_fairness()
        except Exception as e:
            print(f'[TABU] WARNING: calculate_fairness() failed at iteration {iteration}: {e}')
            current_fairness = 0.0
        
        convergence_data['objective_history'].append(current_objective)
        convergence_data['fairness_history'].append(current_fairness)
        
        # Update best solution
        if current_objective < best_solution.objective_value():
            best_solution = current.copy()
            iterations_without_improvement = 0
            print(f"[TABU] ★ Iteration {iteration}: NEW BEST = {best_solution.objective_value():.2f}, "
                  f"Fairness = {current_fairness:.4f}")
        else:
            iterations_without_improvement += 1
            if iteration % 10 == 0:
                print(f"[TABU]   Iteration {iteration}: Current = {current_objective:.2f}, "
                      f"Best = {best_solution.objective_value():.2f}, "
                      f"No improvement for {iterations_without_improvement} iters")
        
        # Early stopping
        if iterations_without_improvement > 30 and iteration > max_iterations * 0.5:
            print(f"[TABU] Early stopping due to convergence at iteration {iteration}")
            break
    
    # Update final iteration count
    convergence_data['iterations'] = iteration
    
    total_time = construction_time + (time.time() - start_time)
    
    print(f'\n[TABU] ========== TABU SEARCH COMPLETED ==========')
    print(f"[TABU] Total iterations: {iteration}")
    print(f"[TABU] Total time: {total_time:.1f}s (construction: {construction_time:.1f}s, search: {total_time - construction_time:.1f}s)")
    print(f"[TABU] Initial objective: {initial_objective:.2f}")
    print(f"[TABU] Final objective: {best_solution.objective_value():.2f}")
    
    improvement = initial_objective - best_solution.objective_value()
    improvement_pct = (improvement / initial_objective * 100) if initial_objective > 0 else 0
    print(f"[TABU] Improvement: {improvement:.2f} ({improvement_pct:.2f}%)")
    print(f"[TABU] Convergence data points: {len(convergence_data['objective_history'])}")
    print(f'[TABU] =============================================\n')
    
    return best_solution, convergence_data


def _is_tabu(move_info: Dict, tabu_list: list) -> bool:
    """Fast tabu check"""
    if not tabu_list:
        return False
    
    move_type = move_info.get("type")
    
    for tabu_move in tabu_list:
        if tabu_move.get("type") != move_type:
            continue
        
        if move_type == "intra_reloc":
            if (tabu_move.get("route") == move_info.get("route") and 
                tabu_move.get("req") == move_info.get("req")):
                return True
        
        elif move_type == "inter_reloc":
            if (tabu_move.get("req") == move_info.get("req") and
                tabu_move.get("from") == move_info.get("from") and
                tabu_move.get("to") == move_info.get("to")):
                return True
        
        elif move_type == "swap":
            if (tabu_move.get("req1") == move_info.get("req1") and
                tabu_move.get("req2") == move_info.get("req2")):
                return True
    
    return False


# ============== Greedy Construction Variants ==============

def greedy_tabu_best(inst: SCFPDPInstance, **kwargs) -> Tuple[Solution, Dict]:
    """Greedy + tabu with best-improvement - returns (solution, convergence_data)"""
    neighborhoods = [
        generate_intra_route_reloc,
        generate_inter_route_reloc,
        generate_swap_requests,
    ]
    
    # Extract tabu_tenure if provided, otherwise use default
    tabu_tenure = kwargs.get('tabu_tenure', 10)
    
    return tabu_search_core(
        inst=inst,
        neighborhoods=neighborhoods,
        construction_func=greedy_construction,
        step_function="best",
        max_iterations=200,
        tabu_tenure=tabu_tenure,
    )


def greedy_tabu_first(inst: SCFPDPInstance, **kwargs) -> Tuple[Solution, Dict]:
    """Greedy + tabu with first-improvement - returns (solution, convergence_data)"""
    neighborhoods = [
        generate_intra_route_reloc,
        generate_inter_route_reloc,
        generate_swap_requests,
    ]
    
    tabu_tenure = kwargs.get('tabu_tenure', 10)
    
    return tabu_search_core(
        inst=inst,
        neighborhoods=neighborhoods,
        construction_func=greedy_construction,
        step_function="first",
        max_iterations=200,
        tabu_tenure=tabu_tenure,
    )


# ============== Randomized Construction Variants ==============

def random_tabu_best(inst: SCFPDPInstance, **kwargs) -> Tuple[Solution, Dict]:
    """Randomized + tabu with best-improvement - returns (solution, convergence_data)"""
    neighborhoods = [
        generate_intra_route_reloc,
        generate_inter_route_reloc,
        generate_swap_requests,
    ]
    
    tabu_tenure = kwargs.get('tabu_tenure', 10)
    
    return tabu_search_core(
        inst=inst,
        neighborhoods=neighborhoods,
        construction_func=randomized_greedy_construction,
        step_function="best",
        max_iterations=200,
        tabu_tenure=tabu_tenure,
    )


def random_tabu_first(inst: SCFPDPInstance, **kwargs) -> Tuple[Solution, Dict]:
    """Randomized + tabu with first-improvement - returns (solution, convergence_data)"""
    neighborhoods = [
        generate_intra_route_reloc,
        generate_inter_route_reloc,
        generate_swap_requests,
    ]
    
    tabu_tenure = kwargs.get('tabu_tenure', 10)
    
    return tabu_search_core(
        inst=inst,
        neighborhoods=neighborhoods,
        construction_func=randomized_greedy_construction,
        step_function="first",
        max_iterations=200,
        tabu_tenure=tabu_tenure,
    )