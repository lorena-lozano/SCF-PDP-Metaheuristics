"""
Local Search framework for the SCF-PDP
Supports:
  - Different neighborhood structures
  - Step functions: first-improvement / best-improvement
  - Delta evaluation through neighborhoods.py
"""

from typing import Callable, Iterable, Tuple, Dict, Optional
import time

from VRP import Solution, SCFPDPInstance
from det_cons_heu import greedy_construction

# Neighborhoods
from neighborhoods import (
    generate_intra_route_reloc,     # N1
    generate_inter_route_reloc,     # N2
    generate_swap_requests,         # N3
    generate_add_remove_requests,   # N4
)

# Type alias for neighborhood generators
NeighborhoodFunc = Callable[
    [Solution],
    Iterable[Tuple[Solution, float, Dict]]
]


def local_search(initial_solution: Solution, neighborhood: NeighborhoodFunc, step_function: str = "best",
                 max_iterations: int = 200, time_limit: Optional[float] = None,
                 max_neighbors_per_iter: Optional[int] = None) -> Solution:
    """
    Local search for a neighborhood structure.
      - max_iterations: max number of iterations for LS
      - time_limit: restriction for the time (None = without limit)
      - max_neighbors_per_iter: max number of neighbors explored per iteration (None = hole neighborhood)
    """
    current = initial_solution
    iteration = 0
    start_time = time.time()

    while iteration < max_iterations:
        # global time limit
        if time_limit is not None and (time.time() - start_time) >= time_limit:
            break

        iteration += 1
        improved = False

        # first-improvement step function
        if step_function == "first":
            neighbor_count = 0
            for neighbor, delta, move in neighborhood(current):
                neighbor_count += 1

                # limit neighbors per iteration
                if max_neighbors_per_iter is not None and neighbor_count > max_neighbors_per_iter:
                    break

                if delta < 0:
                    current = neighbor
                    improved = True
                    break

        # best-improvement step function
        elif step_function == "best":
            best_delta = 0.0
            best_neighbor = None
            neighbor_count = 0

            for neighbor, delta, move in neighborhood(current):
                neighbor_count += 1

                if max_neighbors_per_iter is not None and neighbor_count > max_neighbors_per_iter:
                    break

                if delta < best_delta:
                    best_delta = delta
                    best_neighbor = neighbor

            if best_neighbor is not None:
                current = best_neighbor
                improved = True

        else:
            raise ValueError("Unknown step function. Use 'first' or 'best'.")

        # No improvement: local optimum
        if not improved:
            break

        # Chcking time after iteration
        if time_limit is not None and (time.time() - start_time) >= time_limit:
            break

    return current


# Helper functions
LS_MAX_ITERS = 400
LS_TIME_LIMIT = 60.0
LS_MAX_NEIGHBORS = 5000


def ls_intra_route(initial_solution, step_function="best",
                   max_iterations: int = LS_MAX_ITERS):
    return local_search(
        initial_solution,
        generate_intra_route_reloc,
        step_function=step_function,
        max_iterations=max_iterations,
        time_limit=LS_TIME_LIMIT,
        max_neighbors_per_iter=LS_MAX_NEIGHBORS,
    )


def ls_inter_route(initial_solution, step_function="best",
                   max_iterations: int = LS_MAX_ITERS):
    return local_search(
        initial_solution,
        generate_inter_route_reloc,
        step_function=step_function,
        max_iterations=max_iterations,
        time_limit=LS_TIME_LIMIT,
        max_neighbors_per_iter=LS_MAX_NEIGHBORS,
    )


def ls_swap(initial_solution, step_function="best",
            max_iterations: int = LS_MAX_ITERS):
    return local_search(
        initial_solution,
        generate_swap_requests,
        step_function=step_function,
        max_iterations=max_iterations,
        time_limit=LS_TIME_LIMIT,
        max_neighbors_per_iter=LS_MAX_NEIGHBORS,
    )


def ls_add_remove(initial_solution, step_function="best",
                  max_iterations: int = LS_MAX_ITERS):
    return local_search(
        initial_solution,
        generate_add_remove_requests,
        step_function=step_function,
        max_iterations=max_iterations,
        time_limit=LS_TIME_LIMIT,
        max_neighbors_per_iter=LS_MAX_NEIGHBORS,
    )


# Wrappers for running the code
def ls_N1_best(inst: SCFPDPInstance) -> Solution:
    initial = greedy_construction(inst)
    return ls_intra_route(initial, step_function="best")


def ls_N1_first(inst: SCFPDPInstance) -> Solution:
    initial = greedy_construction(inst)
    return ls_intra_route(initial, step_function="first")


def ls_N2_best(inst: SCFPDPInstance) -> Solution:
    initial = greedy_construction(inst)
    return ls_inter_route(initial, step_function="best")


def ls_N2_first(inst: SCFPDPInstance) -> Solution:
    initial = greedy_construction(inst)
    return ls_inter_route(initial, step_function="first")


def ls_N3_best(inst: SCFPDPInstance) -> Solution:
    initial = greedy_construction(inst)
    return ls_swap(initial, step_function="best")


def ls_N3_first(inst: SCFPDPInstance) -> Solution:
    initial = greedy_construction(inst)
    return ls_swap(initial, step_function="first")


def ls_N4_best(inst: SCFPDPInstance) -> Solution:
    initial = greedy_construction(inst)
    return ls_add_remove(initial, step_function="best")


def ls_N4_first(inst: SCFPDPInstance) -> Solution:
    initial = greedy_construction(inst)
    return ls_add_remove(initial, step_function="first")
