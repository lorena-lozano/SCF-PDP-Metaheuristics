import sys
from VRP import (
    SCFPDPInstance, Solution, Route,
    write_solution, get_instance_name,
)


def greedy_construction(inst: SCFPDPInstance) -> Solution:
    """
    Greedy Constructive Heuristic
    Always inserts each request as [pickup, dropoff] at the end of a route.
        * current duration per route (durations[k])
        * sum of durations (sum_d)
        * sum of squares (sum_d2)
        * number of active routes (m = routes with duration > 0)
        * max demand per route (route_max_demand[k])
    Uses delta evaluation for the objective value
    """


    solution = Solution(inst)
    served = set()

    # Priorities
    priorities = []
    for req in range(1, inst.n + 1):
        pickup = inst.pickup_idx(req)
        dropoff = inst.dropoff_idx(req)
        dist = (inst.dist[0][pickup] + inst.dist[pickup][dropoff] + inst.dist[dropoff][0])
        priorities.append((dist / (inst.demands[req - 1] + 1), req))
    priorities.sort()


    K = inst.n_K
    # Current duration of each route
    durations = [0.0] * K

    # Maximum demand used in each route (since we always add [pickup, dropoff] at the end, the route's max load is
    # the maximum demand of the requests assigned to that route)
    route_max_demand = [0] * K

    # Global sums for Jain's fairness
    sum_d = 0.0
    sum_d2 = 0.0
    m_active = 0  # number of routes with duration > 0

    # Initial fairness (all durations 0) = 1.0
    fairness_old = 1.0
    rho = inst.rho

    # Current objective
    current_obj = sum_d + rho * (1.0 - fairness_old)

    for _, req in priorities:
        # if we have reached gamma stop
        if len(served) >= inst.gamma:
            break

        demand = inst.demands[req - 1]
        pickup = inst.pickup_idx(req)
        dropoff = inst.dropoff_idx(req)

        best_k = -1
        best_delta = float("inf")
        best_new_duration_k = 0.0
        best_new_max_demand = 0

        # Iterate through all routes as possible candidates
        for k, route in enumerate(solution.routes):
            # The route always ends with 0 load and we always add [pickup, dropoff] at the end
            # the new max load of the route will be: max(route_max_demand[k], demand) <= C.
            new_max_dem = max(route_max_demand[k], demand)
            if new_max_dem > inst.C:
                continue

            # New duration of route k
            old_d = durations[k]

            if not route.stops:
                # Empty route
                new_d = (inst.dist[0][pickup] + inst.dist[pickup][dropoff] + inst.dist[dropoff][0])
            else:
                # Non-empty route
                last = route.stops[-1]
                new_d = (
                    old_d
                    - inst.dist[last][0]                 # remove last -> depot
                    + inst.dist[last][pickup]            # add last -> pickup
                    + inst.dist[pickup][dropoff]         # pickup -> dropoff
                    + inst.dist[dropoff][0]              # dropoff -> depot
                )

            # objective evaluation
            sum_d_candidate = sum_d - old_d + new_d
            sum_d2_candidate = sum_d2 - old_d * old_d + new_d * new_d
            m_candidate = m_active
            if old_d == 0.0 and new_d > 0.0:
                m_candidate += 1

            if m_active == 0 or sum_d2 == 0.0:
                fairness_old_local = 1.0
            else:
                fairness_old_local = (sum_d * sum_d) / (m_active * sum_d2)
            # New fairness
            if m_candidate == 0 or sum_d2_candidate == 0.0:
                fairness_new = 1.0
            else:
                fairness_new = (
                    (sum_d_candidate * sum_d_candidate)
                    / (m_candidate * sum_d2_candidate)
                )

            # Change in objective:
            delta = (sum_d_candidate - sum_d) + rho * (fairness_old_local - fairness_new)

            if delta < best_delta:
                best_delta = delta
                best_k = k
                best_new_duration_k = new_d
                best_new_max_demand = new_max_dem

        # If a feasible route is found, apply the best move
        if best_k >= 0:
            # Update route
            solution.routes[best_k].stops.extend([pickup, dropoff])
            served.add(req)

            # Update global structures
            old_d = durations[best_k]
            durations[best_k] = best_new_duration_k

            # Recalculate sum_d, sum_d2, m_active and fairness_old for real
            sum_d = sum_d - old_d + best_new_duration_k
            sum_d2 = sum_d2 - old_d * old_d + best_new_duration_k * best_new_duration_k
            if old_d == 0.0 and best_new_duration_k > 0.0:
                m_active += 1

            if m_active == 0 or sum_d2 == 0.0:
                fairness_old = 1.0
            else:
                fairness_old = (sum_d * sum_d) / (m_active * sum_d2)

            # Update max demand on the route
            route_max_demand[best_k] = best_new_max_demand

            # Update objective
            current_obj = sum_d + rho * (1.0 - fairness_old)

    return solution
