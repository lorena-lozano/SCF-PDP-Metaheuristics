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
                max_iterations: int = None,  # Now dynamic by default
                tabu_tenure: int = 10,
                step_function: str = "best",
                max_neighbors_per_iter: int = None,  # Now dynamic by default
                time_limit: float = 840.0) -> Solution:  # 14 min (60s buffer)
    print('tabu search. being run')
    
    construction_start = time.time()
    current = construction_func(inst)
    best_solution = current.copy()
    construction_time = time.time() - construction_start
    print(construction_time)
    
    # Reserve time for construction (already used) and final operations
    remaining_time = time_limit - construction_time - 10.0  # 10s safety buffer
    
    tabu_list: list[Dict] = []
    start_time = time.time()
    
    # DYNAMIC parameter calculation based on instance size and available time
    n = inst.n
    
    # Estimate time per iteration based on instance size (calibrated values)
    # These are rough estimates - adjust based on your hardware
    if n >= 8000:
        estimated_time_per_iter = 120.0  # ~1 minute per iteration for huge instances
        max_neighbors_per_iter = 50
        tabu_tenure = 3
    elif n >= 5000:
        estimated_time_per_iter = 60.0  # ~1 minute per iteration for huge instances
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
    
    # Calculate max iterations based on available time
    max_iterations = max(5, int(remaining_time / estimated_time_per_iter))
    
    # Cap iterations to reasonable bounds
    max_iterations = min(max_iterations, 300)  # Don't exceed 300 even for tiny instances
    
    print(f"[Tabu Search] n={n}, estimated time/iter={estimated_time_per_iter:.1f}s, "
          f"max_iterations={max_iterations}, max_neighbors/iter={max_neighbors_per_iter}, "
          f"available_time={remaining_time:.1f}s")
    
    iteration = 0
    iterations_without_improvement = 0
    last_best_value = best_solution.objective_value()
    
    while iteration < max_iterations:
        iteration += 1
        
        # Check time limit - leave buffer for safety
        elapsed = time.time() - start_time
        total_elapsed = construction_time + elapsed
        
        if total_elapsed >= time_limit:
            print(f"[Tabu Search] Time limit reached at iteration {iteration}, elapsed={total_elapsed:.1f}s")
            break
        
        # ADAPTIVE: Adjust exploration based on both time pressure AND convergence
        time_pressure = total_elapsed / time_limit
        
        # If no improvement for many iterations, reduce exploration to speed up
        if iterations_without_improvement > 10:
            convergence_factor = 0.5
        elif iterations_without_improvement > 5:
            convergence_factor = 0.7
        else:
            convergence_factor = 1.0
        
        # Combine time pressure and convergence
        if time_pressure > 0.85:
            current_max_neighbors = max(50, int(max_neighbors_per_iter * 0.2 * convergence_factor))
        elif time_pressure > 0.7:
            current_max_neighbors = max(100, int(max_neighbors_per_iter * 0.4 * convergence_factor))
        elif time_pressure > 0.5:
            current_max_neighbors = max(200, int(max_neighbors_per_iter * 0.6 * convergence_factor))
        else:
            current_max_neighbors = int(max_neighbors_per_iter * convergence_factor)
        
        # DYNAMIC: Estimate if we have time for another full iteration
        if iteration > 1:
            avg_time_per_iter = elapsed / iteration
            estimated_next_iter_time = total_elapsed + avg_time_per_iter
            
            # If next iteration would exceed time limit, reduce neighbors drastically
            if estimated_next_iter_time > time_limit - 30:  # 30s buffer
                current_max_neighbors = min(current_max_neighbors, 100)
                print(f"[Tabu Search] Reducing neighbors due to time pressure (iter {iteration})")
            
            # If we're really close to time limit, break early
            if estimated_next_iter_time > time_limit - 10:
                print(f"[Tabu Search] Breaking early at iteration {iteration} to avoid timeout")
                break
        
        best_delta = float('inf')
        best_neighbor = None
        best_move_info = None
        neighbors_checked = 0
        
        # CRITICAL: Use first-improvement style - stop early
        if step_function == "first":
            found_improving = False
            for neighborhood in neighborhoods:
                if found_improving:
                    break
                for neighbor, delta, move_info in neighborhood(current):
                    neighbors_checked += 1
                    
                    # Skip if tabu (unless aspiration criterion met)
                    if _is_tabu(move_info, tabu_list) and neighbor.objective_value() >= best_solution.objective_value():
                        continue
                    
                    # Accept first improving move
                    if delta < 0:
                        best_neighbor = neighbor
                        best_move_info = move_info
                        found_improving = True
                        break
                    
                    if neighbors_checked >= current_max_neighbors:
                        break
        
        # Best-improvement: limit neighbors checked
        else:  # "best"
            for neighborhood in neighborhoods:
                if neighbors_checked >= current_max_neighbors:
                    break
                    
                for neighbor, delta, move_info in neighborhood(current):
                    neighbors_checked += 1
                    
                    # Skip if tabu (unless aspiration)
                    if _is_tabu(move_info, tabu_list) and neighbor.objective_value() >= best_solution.objective_value():
                        continue
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_neighbor = neighbor
                        best_move_info = move_info
                    
                    # Early exit if very good move found
                    if best_delta < -50:  # Significant improvement (reduced threshold)
                        break
                    
                    if neighbors_checked >= current_max_neighbors:
                        break
        
        # No valid moves found
        if best_neighbor is None:
            break
        
        # Update current solution
        current = best_neighbor
        
        # Update tabu list (keep it simple and fast)
        tabu_list.append(best_move_info)
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)
        
        # Update best solution and track improvement
        if current.objective_value() < best_solution.objective_value():
            best_solution = current.copy()
            iterations_without_improvement = 0
            print(f"[Tabu Search] New best at iteration {iteration}: {best_solution.objective_value():.2f}")
        else:
            iterations_without_improvement += 1
        
        # Early stopping if stuck for too long (but only if we have few iterations left)
        if iterations_without_improvement > 30 and iteration > max_iterations * 0.5:
            print(f"[Tabu Search] Early stopping due to convergence at iteration {iteration}")
            break
    
    total_time = construction_time + (time.time() - start_time)
    print(f"[Tabu Search] Finished: {iteration} iterations, {total_time:.1f}s total time")
    
    return best_solution


def _is_tabu(move_info: Dict, tabu_list: list) -> bool:
    """Fast tabu check - simplified to avoid expensive comparisons"""
    if not tabu_list:
        return False
    
    move_type = move_info.get("type")
    
    # Simple attribute-based tabu (not full dict comparison)
    for tabu_move in tabu_list:
        if tabu_move.get("type") != move_type:
            continue
        
        # Type-specific checks (fast)
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

def greedy_tabu_best(inst: SCFPDPInstance) -> Solution:
    """Greedy + tabu with best-improvement"""
    neighborhoods = [
        generate_intra_route_reloc,
        generate_inter_route_reloc,
        generate_swap_requests,
    ]
    return tabu_search_core(
        inst=inst,
        neighborhoods=neighborhoods,
        construction_func=greedy_construction,
        step_function="best",
        max_iterations=200,
    )


def greedy_tabu_first(inst: SCFPDPInstance) -> Solution:
    """Greedy + tabu with first-improvement"""
    neighborhoods = [
        generate_intra_route_reloc,
        generate_inter_route_reloc,
        generate_swap_requests,
    ]
    return tabu_search_core(
        inst=inst,
        neighborhoods=neighborhoods,
        construction_func=greedy_construction,
        step_function="first",
        max_iterations=200,
    )


# ============== Randomized Construction Variants ==============

def random_tabu_best(inst: SCFPDPInstance) -> Solution:
    """Randomized + tabu with best-improvement"""
    neighborhoods = [
        generate_intra_route_reloc,
        generate_inter_route_reloc,
        generate_swap_requests,
    ]
    return tabu_search_core(
        inst=inst,
        neighborhoods=neighborhoods,
        construction_func=randomized_greedy_construction,
        step_function="best",
        max_iterations=200,
    )


def random_tabu_first(inst: SCFPDPInstance) -> Solution:
    """Randomized + tabu with first-improvement"""
    neighborhoods = [
        generate_intra_route_reloc,
        generate_inter_route_reloc,
        generate_swap_requests,
    ]
    return tabu_search_core(
        inst=inst,
        neighborhoods=neighborhoods,
        construction_func=randomized_greedy_construction,
        step_function="first",
        max_iterations=200,
    )