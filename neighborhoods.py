"""
Neighborhood structures
Each generator takes a Solution and yields tuples:
    (neighbor_solution, delta_obj, move_info)
- neighbor_solution : modified Solution (a neighbor of the current solution)
- delta_obj         : f(neighbor) - f(current) (< 0: improvement)
- move_info         : dictionary describing the move
"""

from typing import List, Set
from VRP import Solution, Route, SCFPDPInstance


# Helper functions
def _remove_request_from_route(route: Route, inst: SCFPDPInstance, req: int) -> List[int]:
    """
    Return a new list of stops where both the pickup and the dropoff of 'req'
    have been removed from the given route.
    """
    p_idx = inst.pickup_idx(req)
    d_idx = inst.dropoff_idx(req)
    return [s for s in route.stops if s not in (p_idx, d_idx)]


def _insert_request_consecutive(stops: List[int], inst: SCFPDPInstance, req: int, pos: int) -> List[int]:
    """
    Insert the pickup and dropoff of 'req' consecutively into the list of stops
    at position 'pos'. Returns a new list without modifying the original stops.
    """
    p_idx = inst.pickup_idx(req)
    d_idx = inst.dropoff_idx(req)

    new_stops = stops.copy()
    new_stops[pos:pos] = [p_idx, d_idx]
    return new_stops


def _route_feasible_capacity(stops: List[int], inst: SCFPDPInstance) -> bool:
    """
    Fast capacity check for a route defined by a list of stops.
    Assumes the order pickup/dropoff ya respeta la lógica del problema.
    """
    load = 0
    for stop in stops:
        if stop <= inst.n:           # pickup
            req = stop
            load += inst.demands[req - 1]
        else:                        # dropoff
            req = stop - inst.n
            load -= inst.demands[req - 1]

        if load < 0 or load > inst.C:
            return False
    return True


def _route_duration_from_stops(stops: List[int], inst: SCFPDPInstance) -> float:
    """
    Compute route duration for a list of stops (without construir Route).
    """
    if not stops:
        return 0.0
    total = inst.dist[0][stops[0]]  # depot -> first
    for i in range(len(stops) - 1):
        total += inst.dist[stops[i]][stops[i + 1]]
    total += inst.dist[stops[-1]][0]  # last -> depot
    return total


# N1: Intra-route relocate
def generate_intra_route_reloc(solution: Solution):
    """
    Neighborhood N1: relocate a complete request (pickup + dropoff)
    within the same route, reinserting it at another position
    (pickup and dropoff as consecutive stops).
    """
    inst = solution.inst

    # Precompute current route durations and objective value
    current_durations = [r.duration(inst) for r in solution.routes]
    current_obj = solution.objective_value_from_durations(current_durations)

    for k, route in enumerate(solution.routes):
        route_requests = route.get_requests(inst)

        for req in route_requests:
            # Remove the request from route k
            base_stops = _remove_request_from_route(route, inst, req)

            # Try reinserting (pickup, dropoff) at all positions
            for pos in range(len(base_stops) + 1):
                new_stops = _insert_request_consecutive(base_stops, inst, req, pos)

                # Skip if this would reconstruct exactly the same route
                if new_stops == route.stops:
                    continue

                # Fast feasibility: capacity only on route k
                if not _route_feasible_capacity(new_stops, inst):
                    continue

                # Delta-evaluation: only route k has changed
                new_durations = current_durations.copy()
                new_durations[k] = _route_duration_from_stops(new_stops, inst)

                new_obj = solution.objective_value_from_durations(new_durations)
                delta = new_obj - current_obj

                # Build neighbor solution only if feasible
                neighbor = solution.copy()
                neighbor.routes[k].stops = new_stops

                move_info = {
                    "type": "intra_reloc",
                    "route": k,
                    "req": req,
                    "pos": pos,
                }
                yield neighbor, delta, move_info


# N2: Inter-route relocate
def generate_inter_route_reloc(solution: Solution):
    """
    Neighborhood N2: relocate a complete request (pickup + dropoff)
    from one route to another route.
    """
    inst = solution.inst
    current_durations = [r.duration(inst) for r in solution.routes]
    current_obj = solution.objective_value_from_durations(current_durations)
    n_routes = len(solution.routes)

    for k_from in range(n_routes):
        route_from = solution.routes[k_from]
        route_requests = route_from.get_requests(inst)

        for req in route_requests:
            # Route 'from' without this request
            from_without = _remove_request_from_route(route_from, inst, req)

            for k_to in range(n_routes):
                if k_to == k_from:
                    continue

                route_to = solution.routes[k_to]

                # Try inserting into route 'to' at all positions
                for pos in range(len(route_to.stops) + 1):
                    new_to_stops = _insert_request_consecutive(route_to.stops, inst, req, pos)

                    # Feasibility: only capacity in route_to can get worse
                    if not _route_feasible_capacity(new_to_stops, inst):
                        continue

                    # Delta-evaluation: routes k_from and k_to change
                    new_durations = current_durations.copy()
                    new_durations[k_from] = _route_duration_from_stops(from_without, inst)
                    new_durations[k_to] = _route_duration_from_stops(new_to_stops, inst)

                    new_obj = solution.objective_value_from_durations(new_durations)
                    delta = new_obj - current_obj

                    # Build neighbor
                    neighbor = solution.copy()
                    neighbor.routes[k_from].stops = from_without.copy()
                    neighbor.routes[k_to].stops = new_to_stops

                    move_info = {
                        "type": "inter_reloc",
                        "from": k_from,
                        "to": k_to,
                        "req": req,
                        "pos": pos,
                    }
                    yield neighbor, delta, move_info


# N3: Swap between routes
def generate_swap_requests(solution: Solution):
    """
    Neighborhood N3: swap two complete requests between two different routes
      - remove req1 from route1 and req2 from route2
      - insert req2 in route1 at the original pickup position of req1
      - insert req1 in route2 at the original pickup position of req2
    """
    inst = solution.inst
    current_durations = [r.duration(inst) for r in solution.routes]
    current_obj = solution.objective_value_from_durations(current_durations)
    n_routes = len(solution.routes)

    for k1 in range(n_routes):
        route1 = solution.routes[k1]
        reqs1 = list(route1.get_requests(inst))

        # Precompute pickup positions for reqs in route1
        pickup_pos1 = {
            req: route1.stops.index(inst.pickup_idx(req))
            for req in reqs1
        }

        for k2 in range(k1 + 1, n_routes):
            route2 = solution.routes[k2]
            reqs2 = list(route2.get_requests(inst))

            # Precompute pickup positions for reqs in route2
            pickup_pos2 = {
                req: route2.stops.index(inst.pickup_idx(req))
                for req in reqs2
            }

            for req1 in reqs1:
                for req2 in reqs2:
                    # Remove req1 from route1 and req2 from route2
                    r1_without = _remove_request_from_route(route1, inst, req1)
                    r2_without = _remove_request_from_route(route2, inst, req2)

                    # Original pickup positions (in the original routes)
                    pos1 = pickup_pos1[req1]
                    pos2 = pickup_pos2[req2]

                    # New stops after swap
                    r1_new = _insert_request_consecutive(r1_without, inst, req2, pos1)
                    r2_new = _insert_request_consecutive(r2_without, inst, req1, pos2)

                    # Capacity feasibility in both routes
                    if not _route_feasible_capacity(r1_new, inst):
                        continue
                    if not _route_feasible_capacity(r2_new, inst):
                        continue

                    # Delta evaluation: only routes k1 and k2 change
                    new_durations = current_durations.copy()
                    new_durations[k1] = _route_duration_from_stops(r1_new, inst)
                    new_durations[k2] = _route_duration_from_stops(r2_new, inst)

                    new_obj = solution.objective_value_from_durations(new_durations)
                    delta = new_obj - current_obj

                    # Build neighbor
                    neighbor = solution.copy()
                    neighbor.routes[k1].stops = r1_new
                    neighbor.routes[k2].stops = r2_new

                    move_info = {
                        "type": "swap",
                        "route1": k1,
                        "route2": k2,
                        "req1": req1,
                        "req2": req2,
                        "pos1": pos1,
                        "pos2": pos2,
                    }
                    yield neighbor, delta, move_info


# N4: Add / Remove
def generate_add_remove_requests(solution: Solution):
    """
    Neighborhood N4: add/remove requests
      - Remove a served request from its route
      - Insert an unserved request into some route
    """
    inst = solution.inst
    current_durations = [r.duration(inst) for r in solution.routes]
    current_obj = solution.objective_value_from_durations(current_durations)

    # Global info about served requests
    served: Set[int] = solution.get_all_served_requests()
    n_served = len(served)

    # - remove part -
    allow_removal = (n_served - 1) >= inst.gamma

    if allow_removal:
        for k, route in enumerate(solution.routes):
            for req in route.get_requests(inst):
                new_stops = _remove_request_from_route(route, inst, req)

                # Delta-evaluation
                new_durations = current_durations.copy()
                new_durations[k] = _route_duration_from_stops(new_stops, inst)

                new_obj = solution.objective_value_from_durations(new_durations)
                delta = new_obj - current_obj

                # Build neighbor
                neighbor = solution.copy()
                neighbor.routes[k].stops = new_stops

                move_info = {
                    "type": "remove",
                    "route": k,
                    "req": req,
                }
                yield neighbor, delta, move_info

    # - add part -
    all_reqs = set(range(1, inst.n + 1))
    unserved = all_reqs - served

    for req in unserved:
        for k, route in enumerate(solution.routes):
            for pos in range(len(route.stops) + 1):
                new_stops = _insert_request_consecutive(route.stops, inst, req, pos)

                # Capacity check en la ruta k
                if not _route_feasible_capacity(new_stops, inst):
                    continue

                # Delta-evaluation
                new_durations = current_durations.copy()
                new_durations[k] = _route_duration_from_stops(new_stops, inst)

                new_obj = solution.objective_value_from_durations(new_durations)
                delta = new_obj - current_obj

                # Build neighbor
                neighbor = solution.copy()
                neighbor.routes[k].stops = new_stops

                move_info = {
                    "type": "add",
                    "route": k,
                    "req": req,
                    "pos": pos,
                }
                yield neighbor, delta, move_info
