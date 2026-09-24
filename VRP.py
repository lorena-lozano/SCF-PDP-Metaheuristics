"""
SCF-PDP Framework: Reusable components for any algorithm
Import this module to access instance parsing, solution representation,
validation, evaluation, and I/O utilities.
"""

import math
import os
from typing import List, Tuple, Set, Optional
from dataclasses import dataclass, field
from copy import deepcopy
import time



class SCFPDPInstance:
    """Problem instance with all data and distance calculations"""

    def __init__(self, filename: str):
        """Parse instance file"""
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        # Parse header
        header = lines[0].split()
        self.n = int(header[0])  # number of requests
        self.n_K = int(header[1])  # number of vehicles
        self.C = int(header[2])  # vehicle capacity
        self.gamma = int(header[3])  # minimum requests to serve
        self.rho = float(header[4])  # fairness weight

        # Parse demands
        demands_idx = lines.index('# demands') + 1
        self.demands = list(map(int, lines[demands_idx].split()))

        # Parse locations
        loc_idx = lines.index('# request locations') + 1
        locations = []
        for i in range(loc_idx, len(lines)):
            coords = list(map(float, lines[i].split()))
            locations.extend(
                [(coords[j], coords[j + 1]) for j in range(0, len(coords), 2)])

        self.depot = locations[0]
        self.pickups = locations[1:self.n + 1]
        self.dropoffs = locations[self.n + 1:2 * self.n + 1]

        # Precompute distance matrix
        import torch
    
        all_locs = [self.depot] + self.pickups + self.dropoffs
        self.n_locs = len(all_locs)
        
        print(f"[INSTANCE] Computing distance matrix with PyTorch...")
        start = time.time()
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        locs_tensor = torch.tensor(all_locs, dtype=torch.float32, device=device)
        
        # Compute pairwise distances
        diff = locs_tensor.unsqueeze(1) - locs_tensor.unsqueeze(0)
        distances = torch.sqrt((diff ** 2).sum(dim=2))
        self.dist = torch.ceil(distances).cpu().numpy().astype(int).tolist()
        
        
    def pickup_idx(self, req: int) -> int:
        """Get location index for pickup of request req (1-indexed)"""
        return req

    def dropoff_idx(self, req: int) -> int:
        """Get location index for dropoff of request req (1-indexed)"""
        return self.n + req

    def get_distance(self, from_idx: int, to_idx: int) -> float:
        """Get distance between two location indices (0=depot, 1..n=pickups, n+1..2n=dropoffs)"""
        return self.dist[from_idx][to_idx]

    def request_from_location(self, loc_idx: int) -> Optional[int]:
        """Get request number from location index (returns None for depot)"""
        if loc_idx == 0:
            return None
        elif loc_idx <= self.n:
            return loc_idx
        else:
            return loc_idx - self.n


@dataclass
class Route:
    """Represents a single vehicle route"""
    stops: List[int] = field(
        default_factory=list)  # sequence of location indices

    def copy(self) -> 'Route':
        """Create a deep copy of the route"""
        return Route(stops=self.stops.copy())

    def is_empty(self) -> bool:
        """Check if route has no stops"""
        return len(self.stops) == 0

    def get_requests(self, inst: SCFPDPInstance) -> Set[int]:
        """Get set of all requests served in this route"""
        requests = set()
        for stop in self.stops:
            req = inst.request_from_location(stop)
            if req:
                requests.add(req)
        return requests

    def duration(self, inst: SCFPDPInstance) -> float:
        """Calculate total route duration including depot trips"""
        if not self.stops:
            return 0.0

        total = inst.dist[0][self.stops[0]]  # depot to first stop
        for i in range(len(self.stops) - 1):
            total += inst.dist[self.stops[i]][self.stops[i + 1]]
        total += inst.dist[self.stops[-1]][0]  # last stop to depot
        return total

    def calculate_load_at_each_stop(self, inst: SCFPDPInstance) -> List[int]:
        """Calculate vehicle load after each stop"""
        loads = []
        current_load = 0

        for stop in self.stops:
            if stop <= inst.n:  # pickup
                req = stop
                current_load += inst.demands[req - 1]
            else:  # dropoff
                req = stop - inst.n
                current_load -= inst.demands[req - 1]
            loads.append(current_load)

        return loads


class Solution:
    """Complete solution with all vehicle routes"""

    def __init__(self, inst: SCFPDPInstance,
                 routes: Optional[List[Route]] = None):
        self.inst = inst
        if routes is None:
            self.routes = [Route() for _ in range(inst.n_K)]
        else:
            self.routes = routes

    def copy(self) -> 'Solution':
        """Create a deep copy of the solution"""
        return Solution(self.inst, [r.copy() for r in self.routes])

    def get_all_served_requests(self) -> Set[int]:
        """Get set of all served requests across all routes"""
        served = set()
        for route in self.routes:
            served.update(route.get_requests(self.inst))
        return served

    def num_served_requests(self) -> int:
        """Count total number of served requests"""
        return len(self.get_all_served_requests())

    def is_feasible(self) -> Tuple[bool, str]:
        """
        Check if solution is feasible.
        Returns (is_feasible, error_message)
        """
        served = set()

        for k, route in enumerate(self.routes):
            # Check capacity constraints
            loads = route.calculate_load_at_each_stop(self.inst)
            for i, load in enumerate(loads):
                if load > self.inst.C:
                    return False, f"Route {k} exceeds capacity at stop {i}: load={load}, capacity={self.inst.C}"
                if load < 0:
                    return False, f"Route {k} has negative load at stop {i}: load={load}"

            # Check pickup before dropoff for each request
            route_requests = route.get_requests(self.inst)
            for req in route_requests:
                pickup_idx = self.inst.pickup_idx(req)
                dropoff_idx = self.inst.dropoff_idx(req)

                try:
                    pickup_pos = route.stops.index(pickup_idx)
                    dropoff_pos = route.stops.index(dropoff_idx)

                    if pickup_pos >= dropoff_pos:
                        return False, f"Route {k}: dropoff before pickup for request {req}"
                except ValueError:
                    return False, f"Route {k}: incomplete request {req} (missing pickup or dropoff)"

            # Check for duplicate requests
            for req in route_requests:
                if req in served:
                    return False, f"Request {req} served multiple times"
                served.add(req)

        # Check minimum requests constraint
        if len(served) < self.inst.gamma:
            return False, f"Insufficient requests served: {len(served)} < {self.inst.gamma}"

        return True, "Feasible"

    def objective_value(self) -> float:
        """Calculate objective function value"""
        durations = [r.duration(self.inst) for r in self.routes]
        total_duration = sum(durations)
        fairness = self.jain_fairness(durations)
        return total_duration + self.inst.rho * (1 - fairness)

    def objective_value_from_durations(self, durations: List[float]) -> float:
        """
        Compute the objective value given a list of route durations.
        This allows delta-evaluation
        """
        total_duration = sum(durations)
        fairness = self.jain_fairness(durations)
        return total_duration + self.inst.rho * (1 - fairness)

    @staticmethod
    def jain_fairness(durations: List[float]) -> float:
        """Calculate Jain fairness index"""
        n_k = len(durations)
        if n_k == 0:
            return 1.0  # no routes: trivially "perfect"

        sum_d = sum(durations)
        sum_d2 = sum(d * d for d in durations)

        if sum_d2 == 0:
            # all durations are zero: perfectly equal
            return 1.0

        return (sum_d * sum_d) / (n_k * sum_d2)

    def get_statistics(self) -> dict:
        """Get comprehensive solution statistics"""
        durations = [r.duration(self.inst) for r in self.routes]
        active_routes = sum(1 for d in durations if d > 0)

        return {
            'objective': self.objective_value(),
            'total_duration': sum(durations),
            'fairness': self.jain_fairness(durations),
            'num_served': self.num_served_requests(),
            'num_active_routes': active_routes,
            'min_duration': min(durations) if durations else 0,
            'max_duration': max(durations) if durations else 0,
            'avg_duration': sum(durations) / len(
                durations) if durations else 0,
            'route_durations': durations
        }
    def calculate_fairness(self):
        """Return the same Jain fairness value used in the objective."""
        try:
            durations = [
                r.duration(self.inst)
                for r in self.routes
                if not r.is_empty()
            ]

            if len(durations) == 0:
                return 1.0  # no routes → perfectly fair

            return self.jain_fairness(durations)

        except Exception as e:
            print(f"Fairness calculation error: {e}")
            return 0.0


    def print_statistics(self):
        """Print solution statistics in a readable format"""
        stats = self.get_statistics()
        feasible, msg = self.is_feasible()

        print("=" * 60)
        print("SOLUTION STATISTICS")
        print("=" * 60)
        print(f"Feasible: {feasible} - {msg}")
        print(
            f"Requests served: {stats['num_served']}/{self.inst.n} (required: {self.inst.gamma})")
        print(f"Active routes: {stats['num_active_routes']}/{self.inst.n_K}")
        print(f"\nObjective value: {stats['objective']:.2f}")
        print(f"  Total duration: {stats['total_duration']:.2f}")
        print(f"  Fairness: {stats['fairness']:.4f}")
        print(
            f"  Fairness penalty: {self.inst.rho * (1 - stats['fairness']):.2f}")
        print(f"\nRoute durations:")
        for k, dur in enumerate(stats['route_durations']):
            print(f"  Route {k + 1}: {dur:.2f}")
        print(
            f"  Min: {stats['min_duration']:.2f}, Max: {stats['max_duration']:.2f}, Avg: {stats['avg_duration']:.2f}")
        print("=" * 60)


def read_solution(filename: str, inst: SCFPDPInstance) -> Solution:
    """Read solution from file"""
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    # Skip first line (instance name)
    routes = []
    for line in lines[1:]:
        if line:
            stops = list(map(int, line.split()))
            routes.append(Route(stops=stops))
        else:
            routes.append(Route())

    # Ensure we have exactly n_K routes
    while len(routes) < inst.n_K:
        routes.append(Route())

    return Solution(inst, routes[:inst.n_K])


def write_solution(solution: Solution, filename: str, instance_name: str):
    """Write solution to file"""
    with open(filename, 'w') as f:
        f.write(f"{instance_name}\n")
        for route in solution.routes:
            if route.stops:
                f.write(" ".join(map(str, route.stops)) + "\n")
            else:
                f.write("\n")


def validate_solution_file(instance_file: str, solution_file: str) -> bool:
    """
    Validate a solution file against an instance.
    Returns True if valid, False otherwise.
    Prints detailed error messages.
    """
    try:
        inst = SCFPDPInstance(instance_file)
        sol = read_solution(solution_file, inst)

        feasible, msg = sol.is_feasible()
        if not feasible:
            print(f"INFEASIBLE: {msg}")
            return False

        print("Solution is FEASIBLE")
        sol.print_statistics()
        return True

    except Exception as e:
        print(f"ERROR validating solution: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_instance_name(filepath: str) -> str:
    """Extract instance name from file path (without extension)"""
    return os.path.splitext(os.path.basename(filepath))[0]


# Utility functions for algorithms

def create_empty_solution(inst: SCFPDPInstance) -> Solution:
    """Create an empty solution"""
    return Solution(inst)


def get_unserved_requests(solution: Solution) -> Set[int]:
    """Get set of unserved requests"""
    all_requests = set(range(1, solution.inst.n + 1))
    served = solution.get_all_served_requests()
    return all_requests - served


def route_has_capacity(route: Route, inst: SCFPDPInstance,
                       additional_demand: int) -> bool:
    """Check if route can accommodate additional demand"""
    if route.is_empty():
        return additional_demand <= inst.C

    loads = route.calculate_load_at_each_stop(inst)
    max_load = max(loads) if loads else 0
    return max_load + additional_demand <= inst.C
