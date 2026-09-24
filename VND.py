"""
Variable Neighborhood Descent (VND) for the SCF-PDP
Time-Limited implementation with adaptive parameters based on instance size.
"""

import time
from typing import Callable, Iterable, Tuple, Dict, List

from VRP import Solution, SCFPDPInstance
from det_cons_heu import greedy_construction
from local_search import local_search
from neighborhoods import (
    generate_intra_route_reloc,     # N1
    generate_inter_route_reloc,     # N2
    generate_swap_requests,         # N3
    generate_add_remove_requests,   # N4
)

# Constants
DEFAULT_TIME_LIMIT = 880.0
DEFAULT_MAX_LS_ITERS = 1000

NeighborhoodFunc = Callable[[Solution], Iterable[Tuple[Solution, float, Dict]]]


def vnd_time_limited(inst: SCFPDPInstance,
                     neighborhoods: List[NeighborhoodFunc],
                     step_function: str = "best",
                     time_limit: float = DEFAULT_TIME_LIMIT) -> Solution:
    """
    VND implementation with time control and adaptive parameters
    """
    start_time = time.time()
    n = inst.n

    # Adaptive configuration depending on instance size
    if n >= 5000:
        max_neighbors = 50
    elif n >= 2000:
        max_neighbors = 150
    elif n >= 1000:
        max_neighbors = 400
    elif n >= 500:
        max_neighbors = 800
    else:
        max_neighbors = None

    print(f"[VND] Start. N={n}. Step={step_function}. MaxNeighbors={max_neighbors}")

    # Initial solution (deterministic)
    current = greedy_construction(inst)
    best_obj = current.objective_value()

    k = 0
    k_max = len(neighborhoods)

    # Main loop
    while k < k_max:
        # Global time check
        elapsed = time.time() - start_time
        if elapsed >= time_limit:
            print(f"[VND] STOP: Time limit reached ({elapsed:.1f}s)")
            break

        neighborhood_func = neighborhoods[k]

        # Run Local Search in the current neighborhood
        improved_sol = local_search(
            initial_solution=current,
            neighborhood=neighborhood_func,
            step_function=step_function,
            max_iterations=DEFAULT_MAX_LS_ITERS,
            max_neighbors_per_iter=max_neighbors,
            time_limit=(time_limit - elapsed) # Pass remaining time
        )

        new_obj = improved_sol.objective_value()

        # Acceptance criterion
        if new_obj < best_obj - 1e-4:
            print(f"[VND] Improvement found in k={k}: {best_obj:.2f} -> {new_obj:.2f}")
            current = improved_sol
            best_obj = new_obj
            k = 0  # return to first neighborhood
        else:
            k += 1 # move to next neighborhood

    total_time = time.time() - start_time
    print(f"[VND] Finished. Total time: {total_time:.1f}s. Best: {best_obj:.2f}")

    return current


# WRAPPERS
def vnd_standard_best(inst: SCFPDPInstance) -> Solution:
    """N1 -> N2 -> N3 (Best Improvement)"""
    neighborhoods = [
        generate_intra_route_reloc,  # N1
        generate_inter_route_reloc,  # N2
        generate_swap_requests,      # N3
    ]
    return vnd_time_limited(inst, neighborhoods, step_function="best")

def vnd_standard_first(inst: SCFPDPInstance) -> Solution:
    """N1 -> N2 -> N3 (First Improvement)"""
    neighborhoods = [
        generate_intra_route_reloc,
        generate_inter_route_reloc,
        generate_swap_requests,
    ]
    return vnd_time_limited(inst, neighborhoods, step_function="first")

def vnd_reverse_first(inst: SCFPDPInstance) -> Solution:
    """N3 -> N2 -> N1 (Best Improvement)"""
    neighborhoods = [
        generate_swap_requests,
        generate_inter_route_reloc,
        generate_intra_route_reloc,
    ]
    return vnd_time_limited(inst, neighborhoods, step_function="first")

# Without Swap (N3). To see if N3 adds real value
def vnd_light_first(inst: SCFPDPInstance) -> Solution:
    """N1 -> N2"""
    neighborhoods = [
        generate_intra_route_reloc,
        generate_inter_route_reloc,
    ]
    return vnd_time_limited(inst, neighborhoods, step_function="first")
