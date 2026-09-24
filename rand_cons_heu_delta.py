import sys
import random
from VRP import (
    SCFPDPInstance, Solution, Route, 
    write_solution, get_instance_name,
    get_unserved_requests, route_has_capacity
)

def randomized_greedy_construction(inst: SCFPDPInstance, alpha: float = 0.3, seed: int = None, **params) -> Solution:
    """
    Randomized greedy construction using delta-based objective evaluation
    (like the second greedy_construction), with probabilistic vehicle selection.
    
    alpha: 0 = greedy, 1 = uniform random
    """
    if seed is not None:
        random.seed(seed)
    
    solution = Solution(inst)
    served = set()

    # Compute priorities as before
    priorities = []
    for req in range(1, inst.n + 1):
        pickup = inst.pickup_idx(req)
        dropoff = inst.dropoff_idx(req)
        dist = inst.dist[0][pickup] + inst.dist[pickup][dropoff] + inst.dist[dropoff][0]
        priorities.append((dist / (inst.demands[req-1] + 1), req))
    priorities.sort()

    K = inst.n_K
    durations = [0.0] * K
    route_max_demand = [0] * K
    sum_d = 0.0
    sum_d2 = 0.0
    m_active = 0
    fairness_old = 1.0
    rho = inst.rho
    current_obj = sum_d + rho * (1.0 - fairness_old)

    for _, req in priorities:
        if len(served) >= inst.gamma:
            break
        
        demand = inst.demands[req - 1]
        pickup = inst.pickup_idx(req)
        dropoff = inst.dropoff_idx(req)
        
        feasible_routes = []

        # Evaluate insertion in each route
        for k, route in enumerate(solution.routes):
            new_max_dem = max(route_max_demand[k], demand)
            if new_max_dem > inst.C:
                continue

            old_d = durations[k]
            if not route.stops:
                new_d = inst.dist[0][pickup] + inst.dist[pickup][dropoff] + inst.dist[dropoff][0]
            else:
                last = route.stops[-1]
                new_d = old_d - inst.dist[last][0] + inst.dist[last][pickup] + inst.dist[pickup][dropoff] + inst.dist[dropoff][0]

            # Delta-based objective evaluation
            sum_d_candidate = sum_d - old_d + new_d
            sum_d2_candidate = sum_d2 - old_d * old_d + new_d * new_d
            m_candidate = m_active
            if old_d == 0.0 and new_d > 0.0:
                m_candidate += 1

            fairness_old_local = 1.0 if m_active == 0 or sum_d2 == 0 else (sum_d * sum_d) / (m_active * sum_d2)
            fairness_new = 1.0 if m_candidate == 0 or sum_d2_candidate == 0 else (sum_d_candidate * sum_d_candidate) / (m_candidate * sum_d2_candidate)

            delta = (sum_d_candidate - sum_d) + rho * (fairness_old_local - fairness_new)
            feasible_routes.append((k, delta, new_d, new_max_dem))

        if feasible_routes:
            # Randomized selection of route based on delta
            selected_idx = weighted_random_selection(feasible_routes, alpha)
            k = feasible_routes[selected_idx][0]
            new_d = feasible_routes[selected_idx][2]
            new_max_dem = feasible_routes[selected_idx][3]

            # Apply insertion
            solution.routes[k].stops.extend([pickup, dropoff])
            served.add(req)

            # Update global variables
            old_d = durations[k]
            durations[k] = new_d
            sum_d = sum_d - old_d + new_d
            sum_d2 = sum_d2 - old_d * old_d + new_d * new_d
            if old_d == 0.0 and new_d > 0.0:
                m_active += 1
            fairness_old = 1.0 if m_active == 0 or sum_d2 == 0 else (sum_d * sum_d) / (m_active * sum_d2)
            route_max_demand[k] = new_max_dem
            current_obj = sum_d + rho * (1.0 - fairness_old)

    return solution


def weighted_random_selection(routes_with_deltas, alpha: float) -> int:
    """
    Select a route probabilistically based on delta values.
    Lower delta = higher chance to be selected.
    """
    if len(routes_with_deltas) == 1:
        return 0

    # Sort ascending by delta
    routes_with_deltas.sort(key=lambda x: x[1])

    weights = []
    for i in range(len(routes_with_deltas)):
        if alpha == 0:
            weight = 1.0 if i == 0 else 0.0
        else:
            weight = (1 - alpha) ** i
        weights.append(weight)

    total_weight = sum(weights)
    if total_weight == 0:
        probabilities = [1.0 / len(weights)] * len(weights)
    else:
        probabilities = [w / total_weight for w in weights]

    selected_idx = random.choices(range(len(routes_with_deltas)), weights=probabilities)[0]
    return selected_idx
