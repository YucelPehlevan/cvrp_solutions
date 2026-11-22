"""Task 3: Clarke-Wright savings algorithm (parallel version) for the CVRP.

Idea:
- Start with one route per customer: 0 -> i -> 0.
- Saving of joining i and j in one route: s(i, j) = d(0, i) + d(0, j) - d(i, j).
- Try the pairs in decreasing order of saving and merge the two routes when
    (1) i and j are in different routes,
    (2) i and j are both at an end of their route (next to the depot),
    (3) the total load of the merged route is at most Q.

Output: list of routes without the depot, e.g. [[1, 2], [3, 4]].
Note: the algorithm does not know K, so it can return more than K routes
(e.g. A-n45-k6 gives 7 routes). Tabu Search repairs this with _reduce_to_k.

Time complexity (n = number of customers):
- computing the n(n-1)/2 savings:            O(n^2)
- sorting them:                              O(n^2 log n)
- scanning the sorted list, O(1) checks/pair: O(n^2)
- at most n - 1 merges, each copies a route: O(n) each -> O(n^2)
Total: O(n^2 log n), dominated by the sort. Memory: O(n^2) for the savings list.
"""
from typing import List

from problem import CVRPInstance


def clarke_wright(instance: CVRPInstance) -> List[List[int]]:
    n = instance.n
    D = instance.dist_matrix
    q = instance.demands
    Q = instance.capacity

    # ---- 1) start: one route per customer ----
    routes = {i: [i] for i in range(1, n + 1)}   # route id -> list of customers
    route_of = [0] * (n + 1)                      # customer -> route id
    load = {}                                     # route id -> total demand
    for i in range(1, n + 1):
        route_of[i] = i
        load[i] = q[i]

    # ---- 2) savings for every pair i < j, sorted in decreasing order ----
    savings = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            s = D[0][i] + D[0][j] - D[i][j]
            savings.append((s, i, j))
    savings.sort(key=lambda t: t[0], reverse=True)

    # ---- 3) greedy merging ----
    for s, i, j in savings:
        if s <= 0:          # no gain any more (list is sorted, so stop)
            break

        ri, rj = route_of[i], route_of[j]

        # rule 1: different routes
        if ri == rj:
            continue

        # rule 3: capacity
        if load[ri] + load[rj] > Q:
            continue

        A, B = routes[ri], routes[rj]

        # rule 2: i and j must be at an end of their routes; build the merged route
        # so that i and j become neighbours: ... i, j ...
        if A[-1] == i and B[0] == j:        # A = [.., i], B = [j, ..]
            merged = A + B
        elif A[0] == i and B[-1] == j:      # A = [i, ..], B = [.., j]
            merged = B + A
        elif A[-1] == i and B[-1] == j:     # A = [.., i], B = [.., j]
            merged = A + B[::-1]
        elif A[0] == i and B[0] == j:       # A = [i, ..], B = [j, ..]
            merged = A[::-1] + B
        else:                               # i or j is in the middle of a route
            continue

        # keep the merged route under id ri, delete rj
        routes[ri] = merged
        load[ri] += load[rj]
        del routes[rj]
        del load[rj]
        for c in B:
            route_of[c] = ri

    return list(routes.values())


if __name__ == "__main__":
    from data_loader import parse_cvrplib
    from problem import calculate_cost, verify_solution

    for name in ["A-n32-k5", "A-n45-k6", "A-n60-k9", "A-n80-k10"]:
        inst = parse_cvrplib(f"data/{name}.vrp")
        sol = clarke_wright(inst)
        cost = calculate_cost(inst, sol)
        check = verify_solution(inst, sol)
        gap = 100 * (cost - inst.bks) / inst.bks if inst.bks else float("nan")
        print(f"{name}: cost = {cost:.0f}, routes = {len(sol)} (K = {inst.max_vehicles}), "
              f"gap = {gap:.2f}%, feasible = {check['ok']} {check['errors'][:1]}")
