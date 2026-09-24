import sys
import random
import numpy as np
from functools import lru_cache
from VRP import (
    SCFPDPInstance, Solution, Route, 
    write_solution, get_instance_name,
    get_unserved_requests, route_has_capacity
)

def beam_search_construction(inst: SCFPDPInstance, beam_width: int = 5, seed: int = None, **params) -> Solution:

    """
    Optimized beam search with DELTA/LAZY COPYING for massive speedup.
    
    Key optimization: Instead of copying entire solutions, we track only deltas
    (parent + modification). Full solution is built only at the end.
    """
    
    # Adaptive beam width
    if inst.n >= 8000:
        beam_width = 1
    elif inst.n >= 5000:
        beam_width = 2
    elif inst.n >= 2000:
        beam_width = 5
    
    # For high gamma ratios, need wider beams
    gamma_ratio = inst.gamma / inst.n
    if gamma_ratio > 0.85:
        beam_width = max(beam_width, 3)
        print(f"High gamma ratio ({gamma_ratio:.1%}), increasing beam_width to {beam_width}")
    
    print(f"Beam width set to {beam_width} for n={inst.n}, gamma={inst.gamma}")
    
    # Calculate request priorities
    request_order = calculate_request_priorities_fast(inst)
    
    # Ensure we process enough requests to reach gamma
    min_needed = int(inst.gamma * 1.3)
    max_requests_to_consider = inst.n
    
    if inst.n >= 5000 and gamma_ratio < 0.85:
        max_requests_to_consider = max(min_needed, min(inst.gamma + 1000, inst.n))
    
    request_order = request_order[:max_requests_to_consider]
    
    print(f"Processing {len(request_order)} out of {inst.n} requests (target={inst.gamma})")
    
    # Initialize beam with root partial solution (no parent, no modifications)
    beam = [PartialSolution(inst, parent=None, vehicle_idx=None, request=None, 
                            pickup=None, dropoff=None, insertion_cost=0.0)]
    
    early_stopped = False
    
    # Iterate over requests
    for i, req in enumerate(request_order, 1):
        # Progress update every 100 requests
        if i % 100 == 0 or i == len(request_order):
            best_served = max(len(ps.served) for ps in beam)
            print(f"[{i}/{len(request_order)}] Beam size: {len(beam)}, Best served: {best_served}/{inst.gamma}", flush=True)
        
        # Early stopping - all beams at gamma
        if all(len(ps.served) >= inst.gamma for ps in beam):
            if not early_stopped:
                print(f"✓ All beams reached gamma={inst.gamma} at request {i}/{len(request_order)}")
                early_stopped = True
            break
        
        candidates = []
        
        for partial_sol in beam:
            # Skip beams already at capacity
            if len(partial_sol.served) >= inst.gamma:
                candidates.append(partial_sol)
                continue
            
            demand = inst.demands[req - 1]
            pickup = inst.pickup_idx(req)
            dropoff = inst.dropoff_idx(req)
            
            # Pre-filter feasible vehicles
            feasible_vehicles = get_feasible_vehicles_fast(partial_sol, inst, req, demand)
            
            if not feasible_vehicles:
                continue
            
            # Try each feasible vehicle
            for k in feasible_vehicles:
                # Find best insertion position for this request in this vehicle
                best_insertion = find_best_insertion_position(
                    partial_sol, inst, k, req, pickup, dropoff, demand
                )
                
                if best_insertion is None:
                    continue  # No feasible insertion position
                
                insertion_cost = best_insertion
                
                # Create LIGHTWEIGHT child with delta information
                new_partial = PartialSolution(
                    inst,
                    parent=partial_sol,
                    vehicle_idx=k,
                    request=req,
                    pickup=pickup,
                    dropoff=dropoff,
                    insertion_cost=insertion_cost
                )
                
                # Update objective incrementally
                new_partial.objective = partial_sol.objective + insertion_cost
                
                # Track served requests (cheap set copy)
                new_partial.served = partial_sol.served | {req}
                
                candidates.append(new_partial)
        
        if not candidates:
            print(f"⚠ No candidates at request {i}, stopping early")
            break
        
        # Keep only top beam_width candidates
        candidates.sort(key=lambda x: x.objective)
        beam = candidates[:beam_width]
    
    if not beam:
        print("⚠ No beam solutions found, returning empty solution")
        return Solution(inst)
    
    # Select best solution from beam
    beam.sort(key=lambda x: x.objective)
    best_partial = beam[0]
    
    # Build the full solution safely
    final_solution = best_partial.build_full_solution(inst)
    
    # Check if we reached gamma
    if len(best_partial.served) < inst.gamma:
        print(f"⚠ WARNING: Only served {len(best_partial.served)}/{inst.gamma} requests")
    else:
        print(f"✓ Successfully served {len(best_partial.served)}/{inst.gamma} requests")
    
    return final_solution


def calculate_request_priorities_fast(inst: SCFPDPInstance):
    """Fast priority calculation using numpy."""
    n = inst.n
    
    pickups = np.array([inst.pickup_idx(r) for r in range(1, n + 1)])
    dropoffs = np.array([inst.dropoff_idx(r) for r in range(1, n + 1)])
    
    dist = np.array(inst.dist)
    base_costs = dist[0, pickups] + dist[pickups, dropoffs] + dist[dropoffs, 0]
    
    demands = np.array(inst.demands)
    priorities = base_costs / (demands + 1)
    
    sorted_indices = np.argsort(priorities)
    return [r + 1 for r in sorted_indices]


def find_best_insertion_position(partial_sol, inst: SCFPDPInstance, 
                                 vehicle_idx: int, request: int,
                                 pickup: int, dropoff: int, demand: int):
    """Find the BEST position to insert pickup and dropoff in the route."""
    route_stops = get_route_stops_from_deltas(partial_sol, vehicle_idx)
    n_stops = len(route_stops)
    
    if n_stops == 0:
        cost = inst.dist[0][pickup] + inst.dist[pickup][dropoff] + inst.dist[dropoff][0]
        return cost
    
    # For large routes, only try appending at end
    if n_stops > 50:
        return try_append_at_end(partial_sol, inst, vehicle_idx, pickup, dropoff, route_stops, demand)
    
    best_cost = float('inf')
    
    # Try all valid insertion positions
    for i in range(n_stops + 1):
        for j in range(i + 1, n_stops + 2):
            if not is_insertion_feasible(partial_sol, inst, vehicle_idx, pickup, dropoff, demand, i, j, route_stops):
                continue
            cost = calculate_insertion_cost_at_positions(inst, route_stops, pickup, dropoff, i, j)
            if cost < best_cost:
                best_cost = cost
    
    if best_cost == float('inf'):
        return None
    
    return best_cost


def try_append_at_end(partial_sol, inst, vehicle_idx, pickup, dropoff, route_stops, demand):
    """Append at end with full load simulation to prevent capacity violation."""
    # Simulate load along the route with new stops appended
    load = 0
    new_stops = route_stops + [pickup, dropoff]
    
    for stop in new_stops:
        if 1 <= stop <= inst.n:
            load += inst.demands[stop - 1]
        else:
            req = stop - inst.n
            load -= inst.demands[req - 1]
        if load > inst.C:
            return None
    
    # Calculate cost
    if len(route_stops) == 0:
        cost = inst.dist[0][pickup] + inst.dist[pickup][dropoff] + inst.dist[dropoff][0]
    else:
        last_stop = route_stops[-1]
        cost_removed = inst.dist[last_stop][0]
        cost_added = inst.dist[last_stop][pickup] + inst.dist[pickup][dropoff] + inst.dist[dropoff][0]
        cost = cost_added - cost_removed
    
    return cost


def get_route_stops_from_deltas(partial_sol, vehicle_idx: int):
    """Get current list of stops for a vehicle by walking delta chain."""
    route_stops = []
    current = partial_sol
    seen = set()
    while current is not None:
        if current.vehicle_idx == vehicle_idx and current.pickup is not None:
            if (current.pickup, current.dropoff) not in seen:
                route_stops = [current.pickup, current.dropoff] + route_stops
                seen.add((current.pickup, current.dropoff))
        current = current.parent
    return route_stops


def is_insertion_feasible(partial_sol, inst, vehicle_idx, pickup, dropoff, 
                          demand, pickup_pos, dropoff_pos, route_stops):
    """Check if insertion is feasible w.r.t vehicle capacity."""
    # Build the new route with insertion
    new_stops = route_stops[:pickup_pos] + [pickup] + route_stops[pickup_pos:dropoff_pos] + [dropoff] + route_stops[dropoff_pos:]
    
    # Simulate load
    load = 0
    for stop in new_stops:
        if 1 <= stop <= inst.n:
            load += inst.demands[stop - 1]
        else:
            req = stop - inst.n
            load -= inst.demands[req - 1]
        if load > inst.C:
            return False
    
    return True


def calculate_insertion_cost_at_positions(inst, route_stops, pickup, dropoff, pickup_pos, dropoff_pos):
    """Cost delta from inserting at specific positions."""
    new_stops = route_stops[:pickup_pos] + [pickup] + route_stops[pickup_pos:dropoff_pos] + [dropoff] + route_stops[dropoff_pos:]
    
    cost = inst.dist[0][new_stops[0]] if new_stops else 0
    for i in range(len(new_stops) - 1):
        cost += inst.dist[new_stops[i]][new_stops[i+1]]
    if new_stops:
        cost += inst.dist[new_stops[-1]][0]
    
    old_cost = inst.dist[0][route_stops[0]] if route_stops else 0
    for i in range(len(route_stops) - 1):
        old_cost += inst.dist[route_stops[i]][route_stops[i+1]]
    if route_stops:
        old_cost += inst.dist[route_stops[-1]][0]
    
    return cost - old_cost


def get_feasible_vehicles_fast(partial_sol, inst: SCFPDPInstance, request: int, demand: int):
    """Fast pre-filtering: check if vehicle has enough capacity left."""
    feasible = []
    
    if request in partial_sol.served:
        return feasible
    
    for k in range(inst.n_K):
        # Get current max load for this vehicle
        route_stops = get_route_stops_from_deltas(partial_sol, k)
        max_load = calculate_max_load(inst, route_stops)
        
        if max_load + demand <= inst.C:
            feasible.append(k)
    
    return feasible


def calculate_max_load(inst: SCFPDPInstance, route_stops):
    """Calculate maximum load along a route."""
    max_load = 0
    current_load = 0
    
    for stop in route_stops:
        if 1 <= stop <= inst.n:
            current_load += inst.demands[stop - 1]
        else:
            req = stop - inst.n
            current_load -= inst.demands[req - 1]
        max_load = max(max_load, current_load)
    
    return max_load


class PartialSolution:
    """Delta-based partial solution - stores only the modification, not positions."""
    def __init__(self, inst: SCFPDPInstance, parent=None, vehicle_idx=None, 
                 request=None, pickup=None, dropoff=None, insertion_cost=0.0):
        self.inst = inst
        self.parent = parent
        self.vehicle_idx = vehicle_idx
        self.request = request
        self.pickup = pickup
        self.dropoff = dropoff
        self.insertion_cost = insertion_cost
        
        self.objective = 0.0
        self.served = set() if parent is None else parent.served.copy()
        if parent is None:
            self.objective = 0.0
            self.served = set()
    
    def build_full_solution(self, inst: SCFPDPInstance) -> Solution:
        """Build full solution by simply collecting modifications in order - fast O(n) reconstruction."""
        # Collect all modifications in order
        modifications = []
        current = self
        while current is not None:
            if current.vehicle_idx is not None and current.request is not None:
                modifications.append({
                    'vehicle_idx': current.vehicle_idx,
                    'request': current.request,
                    'pickup': current.pickup,
                    'dropoff': current.dropoff
                })
            current = current.parent
        modifications.reverse()
        
        # Group by vehicle - preserving insertion order
        vehicle_requests = [[] for _ in range(inst.n_K)]
        for mod in modifications:
            vehicle_requests[mod['vehicle_idx']].append(mod)
        
        # Build solution by simply appending in order
        solution = Solution(inst)
        
        for k in range(inst.n_K):
            if not vehicle_requests[k]:
                continue
            
            # Just append all pickup-dropoff pairs in the order they were added during beam search
            route_stops = []
            for mod in vehicle_requests[k]:
                route_stops.append(mod['pickup'])
                route_stops.append(mod['dropoff'])
            
            solution.routes[k].stops = route_stops
        
        return solution
    
    def __lt__(self, other):
        return self.objective < other.objective