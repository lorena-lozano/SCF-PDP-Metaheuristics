"""
GRASP metaheuristic for the SCF-PDP
Time-Limited implementation to handle large instances within 15 min.
"""

import time
import random
from math import inf
from typing import Callable, Optional
from VRP import SCFPDPInstance, Solution
from rand_cons_heu_opt import randomized_greedy_construction

from local_search import (
    ls_intra_route,  # N1
    ls_inter_route,  # N2
    ls_swap,  # N3
    ls_add_remove,  # N4
)

DEFAULT_TIME_LIMIT = 870.0  # 14.5 min
DEFAULT_LS_ITERS = 300

ImproveFunc = Callable[[Solution], Solution]

def grasp_time_limited(inst: SCFPDPInstance,
                       improve_func: ImproveFunc,
                       max_iterations: int = 50,
                       alpha: float = 0.3,
                       base_seed: Optional[int] = None,
                       time_limit: float = DEFAULT_TIME_LIMIT) -> Solution:
    """
    GRASP with time limit
    """
    start_time = time.time()
    best_solution: Optional[Solution] = None
    best_obj: float = inf

    # 1. Dynamic Adjustment of Iterations Based on Instance Size
    n = inst.n
    if n >= 5000:
        target_iters = 5
    elif n >= 2000:
        target_iters = 15
    elif n >= 500:
        target_iters = 30
    else:
        target_iters = max_iterations

    for i in range(target_iters):
        # Time control
        current_time = time.time()
        elapsed = current_time - start_time

        if elapsed >= (time_limit - 30):
            print(
                f"[GRASP] STOP: Time limit reached after {i} iterations")
            break

        # If the average per iteration is greater than the remaining time, we stop
        if i > 0:
            avg_time = elapsed / i
            if elapsed + avg_time > time_limit:
                print(
                    f"[GRASP] STOP: No hay tiempo suficiente para otra iteración completa.")
                break

        iter_seed = (base_seed + i) if base_seed is not None else None
        # Iteration 0: Pure greedy (alpha=0), the rest random (alpha=0.3)
        current_alpha = 0.0 if i == 0 else alpha

        sol = randomized_greedy_construction(inst, alpha=current_alpha,
                                             seed=iter_seed)

        # local search
        sol = improve_func(sol)

        # --- ACTUALIZAR MEJOR ---
        obj = sol.objective_value()
        if obj < best_obj:
            best_obj = obj
            best_solution = sol

    # if there wasn’t time for anything return a quick greedy
    if best_solution is None:
        print(
            "[GRASP] ALERTA: Tiempo agotado antes de completar iteración 0. Usando Greedy.")
        best_solution = randomized_greedy_construction(inst, alpha=0.0)

    total_time = time.time() - start_time
    print(
        f"[GRASP] Finished. Total time: {total_time:.1f}s. Best: {best_obj:.2f}")

    return best_solution


# Wrappers for batch runner
# Neighborhood N1: Intra-route Relocate
def grasp_N1_best(inst: SCFPDPInstance) -> Solution:
    return grasp_time_limited(
        inst,
        improve_func=lambda s: ls_intra_route(s, step_function="best",
                                              max_iterations=DEFAULT_LS_ITERS),
        base_seed=1234
    )

def grasp_N1_first(inst: SCFPDPInstance) -> Solution:
    return grasp_time_limited(
        inst,
        improve_func=lambda s: ls_intra_route(s, step_function="first",
                                              max_iterations=DEFAULT_LS_ITERS),
        base_seed=1234
    )


# Neighborhood N2: Inter-route Relocate
def grasp_N2_best(inst: SCFPDPInstance) -> Solution:
    return grasp_time_limited(
        inst,
        improve_func=lambda s: ls_inter_route(s, step_function="best",
                                              max_iterations=DEFAULT_LS_ITERS),
        base_seed=1234
    )

def grasp_N2_first(inst: SCFPDPInstance) -> Solution:
    return grasp_time_limited(
        inst,
        improve_func=lambda s: ls_inter_route(s, step_function="first",
                                              max_iterations=DEFAULT_LS_ITERS),
        base_seed=1234
    )


# Neighborhood N3: Swap Requests
def grasp_N3_best(inst: SCFPDPInstance) -> Solution:
    return grasp_time_limited(
        inst,
        improve_func=lambda s: ls_swap(s, step_function="best",
                                       max_iterations=DEFAULT_LS_ITERS),
        base_seed=1234
    )

def grasp_N3_first(inst: SCFPDPInstance) -> Solution:
    return grasp_time_limited(
        inst,
        improve_func=lambda s: ls_swap(s, step_function="first",
                                       max_iterations=DEFAULT_LS_ITERS),
        base_seed=1234
    )


# Neighborhood N4: Add/Remove Requests
def grasp_N4_best(inst: SCFPDPInstance) -> Solution:
    return grasp_time_limited(
        inst,
        improve_func=lambda s: ls_add_remove(s, step_function="best",
                                             max_iterations=DEFAULT_LS_ITERS),
        base_seed=1234
    )

def grasp_N4_first(inst: SCFPDPInstance) -> Solution:
    return grasp_time_limited(
        inst,
        improve_func=lambda s: ls_add_remove(s, step_function="first",
                                             max_iterations=DEFAULT_LS_ITERS),
        base_seed=1234
    )
