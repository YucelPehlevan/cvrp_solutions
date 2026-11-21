import math
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class CVRPInstance:
    name: str
    n: int                          # number of customers (1..n), node 0 is the depot
    coords: List[Tuple[float, float]]
    demands: List[int]              # demands[0] = 0
    capacity: int                   # Q
    max_vehicles: int               # K
    dist_matrix: List[List[float]]  # (n+1) x (n+1), O(1) distance lookup
    rounding: bool = False          # True for CVRPLIB (nearest integer)
    split_x: Optional[float] = None # Bosphorus line: x < split_x is EU, x >= split_x is AS
    penalty: float = 0.0            # Bosphorus crossing penalty P
    bks: Optional[float] = None     # best-known solution (CVRPLIB)


def side(instance: CVRPInstance, i: int) -> str:
    if instance.split_x is None:
        return "EU"
    return "EU" if instance.coords[i][0] < instance.split_x else "AS"


def compute_distance(instance: CVRPInstance, i: int, j: int) -> float:
    """d(i, j) = Euclidean + P * [different sides], rounded for CVRPLIB."""
    (x1, y1), (x2, y2) = instance.coords[i], instance.coords[j]
    d = math.hypot(x1 - x2, y1 - y2)
    if instance.penalty > 0 and side(instance, i) != side(instance, j):
        d += instance.penalty
    if instance.rounding:
        d = float(int(d + 0.5))
    return d


def build_dist_matrix(instance: CVRPInstance) -> List[List[float]]:
    m = instance.n + 1
    D = [[0.0] * m for _ in range(m)]
    for i in range(m):
        for j in range(i + 1, m):
            D[i][j] = D[j][i] = compute_distance(instance, i, j)
    return D


def calculate_cost(instance: CVRPInstance, routes: List[List[int]]) -> float:
    """Total distance; each route is 0 -> customers -> 0. O(n)."""
    total_dist = 0.0
    for route in routes:
        if not route:
            continue
        prev = 0
        for node in route:
            total_dist += instance.dist_matrix[prev][node]
            prev = node
        total_dist += instance.dist_matrix[prev][0]
    return total_dist


def verify_solution(instance: CVRPInstance, routes: List[List[int]]) -> dict:
    """Independent feasibility checker = the NP verifier from Task 1, O(n):
    routes <= K, valid ids, every customer exactly once, load <= Q,
    and the cost is recomputed from the coordinates (not from dist_matrix)."""
    errors = []
    non_empty = [r for r in routes if len(r) > 0]
    if len(non_empty) > instance.max_vehicles:
        errors.append(f"Too many routes: {len(non_empty)} > K = {instance.max_vehicles}")

    count = [0] * (instance.n + 1)
    cost = 0.0
    for idx, route in enumerate(non_empty):
        load, prev, valid = 0, 0, True
        for node in route:
            if not isinstance(node, int) or node < 1 or node > instance.n:
                errors.append(f"Route {idx}: invalid customer id {node!r}")
                valid = False
                continue
            count[node] += 1
            load += instance.demands[node]
            cost += compute_distance(instance, prev, node)
            prev = node
        cost += compute_distance(instance, prev, 0)
        if valid and load > instance.capacity:
            errors.append(f"Route {idx}: capacity exceeded ({load} > {instance.capacity})")

    for c in range(1, instance.n + 1):
        if count[c] == 0:
            errors.append(f"Customer {c} is not visited")
        elif count[c] > 1:
            errors.append(f"Customer {c} is visited {count[c]} times")

    return {"ok": not errors, "errors": errors, "cost": cost, "n_routes": len(non_empty)}


def is_feasible(instance: CVRPInstance, routes: List[List[int]]) -> Tuple[bool, str]:
    res = verify_solution(instance, routes)
    return (True, "OK") if res["ok"] else (False, res["errors"][0])
