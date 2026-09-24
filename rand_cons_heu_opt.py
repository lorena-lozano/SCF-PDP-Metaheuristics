import sys
import random
from VRP import (
    SCFPDPInstance, Solution, Route,
    write_solution, get_instance_name,
    get_unserved_requests, route_has_capacity
)


def randomized_greedy_construction(inst: SCFPDPInstance,
                                       alpha: float = 0.3,
                                       seed: int = None) -> Solution:
    """
    Builds a solution using a randomized greedy strategy.
    - 'alpha' controls how greedy or exploratory the selection is: Lower
    alpha more greedy; higher alpha more randomness
    We keep incremental values of total durations and squared for delta
    evaluation
    """
    if seed is not None:
        random.seed(seed)

    solution = Solution(inst)
    served = set()

    # Compute a priority score for each request based on distance and demand
    # Lower score served earlier
    priorities = []
    for req in range(1, inst.n + 1):
        pickup = inst.pickup_idx(req)
        dropoff = inst.dropoff_idx(req)

        # Approximate cost of serving this request alone
        dist = inst.dist[0][pickup] + inst.dist[pickup][dropoff] + \
               inst.dist[dropoff][0]

        priorities.append((dist / (inst.demands[req - 1] + 1), req))

    priorities.sort()

    K = inst.n_K
    durations = [0.0] * K              # Total travel duration per route
    route_max_demand = [0] * K         # Max load encountered in each route

    # Global objective components maintained incrementally
    sum_d = 0.0                        # Sum of route durations
    sum_d2 = 0.0                       # Sum of squared route durations
    m_active = 0                       # Number of non-empty routes

    # Sets used to quickly check and evaluate empty vs non-empty routes
    empty_routes = set(range(K))
    non_empty_routes = set()

    # Fairness weight
    rho = inst.rho

    for _, req in priorities:
        # Stop if we already served the required number of requests
        if len(served) >= inst.gamma:
            break

        demand = inst.demands[req - 1]
        pickup = inst.pickup_idx(req)
        dropoff = inst.dropoff_idx(req)

        feasible_routes = []

        # Evaluate insertion into empty routes
        if empty_routes:
            k_empty = next(iter(empty_routes))

            # Basic capacity check for empty route
            if demand <= inst.C:
                old_d = 0.0
                # Serving a request in an empty route is always depot -
                # pickup - dropoff - depot
                new_d = inst.dist[0][pickup] + inst.dist[pickup][dropoff] + \
                        inst.dist[dropoff][0]

                # Update global terms if we activate a new route
                sum_d_candidate = sum_d + new_d
                sum_d2_candidate = sum_d2 + new_d * new_d
                m_candidate = m_active + 1

                # Fairness before and after inserting into an empty route
                fairness_old_local = 1.0 if m_active == 0 or sum_d2 == 0 else (
                    (sum_d * sum_d) / (m_active * sum_d2)
                )
                fairness_new = 1.0 if m_candidate == 0 or sum_d2_candidate == 0 else (
                    (sum_d_candidate * sum_d_candidate) / (m_candidate * sum_d2_candidate)
                )

                delta = (sum_d_candidate - sum_d) + rho * (
                    fairness_old_local - fairness_new
                )

                # Add this option for every empty route so that the
                # weighted selection considers all of them equally
                for k in empty_routes:
                    feasible_routes.append((k, delta, new_d, demand))

        # Evaluate insertion into each non-empty route
        for k in non_empty_routes:
            route = solution.routes[k]

            # Check whether adding this demand would violate capacity
            new_max_dem = max(route_max_demand[k], demand)
            if new_max_dem > inst.C:
                continue

            old_d = durations[k]
            last = route.stops[-1]

            # Replace the final depot return with pickup→dropoff→depot
            new_d = old_d - inst.dist[last][0] + inst.dist[last][pickup] + \
                    inst.dist[pickup][dropoff] + inst.dist[dropoff][0]

            # Incremental update of global metrics
            sum_d_candidate = sum_d - old_d + new_d
            sum_d2_candidate = sum_d2 - old_d * old_d + new_d * new_d
            m_candidate = m_active  # No new route is activated here

            fairness_old_local = 1.0 if m_active == 0 or sum_d2 == 0 else (
                (sum_d * sum_d) / (m_active * sum_d2)
            )
            fairness_new = 1.0 if m_candidate == 0 or sum_d2_candidate == 0 else (
                (sum_d_candidate * sum_d_candidate) / (m_candidate * sum_d2_candidate)
            )

            delta = (sum_d_candidate - sum_d) + rho * (
                fairness_old_local - fairness_new
            )

            feasible_routes.append((k, delta, new_d, new_max_dem))

        # If at least one route is feasible, pick one using the randomized greedy rule
        if feasible_routes:
            selected_idx = weighted_random_selection(feasible_routes, alpha)
            k_sel = feasible_routes[selected_idx][0]
            new_d = feasible_routes[selected_idx][2]
            new_max_dem = feasible_routes[selected_idx][3]

            # Add pickup and dropoff to the selected route
            solution.routes[k_sel].stops.extend([pickup, dropoff])
            served.add(req)

            # Update route sets
            if k_sel in empty_routes:
                empty_routes.remove(k_sel)
                non_empty_routes.add(k_sel)
                m_active += 1

            # Update global objective components
            old_d = durations[k_sel]
            durations[k_sel] = new_d

            sum_d = sum_d - old_d + new_d
            sum_d2 = sum_d2 - old_d * old_d + new_d * new_d

            route_max_demand[k_sel] = new_max_dem

    return solution


def weighted_random_selection(routes_with_deltas, alpha: float) -> int:
    """
    Selects one route index from the candidate list using a biased random rule
    Routes are sorted by their delta value (lower is better)
    The weighting scheme uses (1 - alpha)^i, where i is the rank in the
    sorted list, this gives high probability to better-ranked routes while
    keeping some diversity
    """
    if len(routes_with_deltas) == 1:
        return 0

    routes_with_deltas.sort(key=lambda x: x[1])

    # If alpha is zero, the selection is purely greedy
    if alpha == 0:
        return 0

    # Compute weights based on the geometric decay rule
    weights = [(1 - alpha) ** i for i in range(len(routes_with_deltas))]

    # weighted sampling
    selected_idx = random.choices(
        range(len(routes_with_deltas)), weights=weights, k=1)[0]

    return selected_idx
