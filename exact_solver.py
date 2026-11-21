"""Task 2(c): exact optimum for tiny instances (n = 7, 8, 9) by dynamic programming.

1) Held-Karp: tsp[S] = shortest depot tour through customer subset S.   O(2^n * n^2)
2) Set partition: f[k][S] = cheapest cover of S with exactly k routes,
   each route a subset with load <= Q.                                    O(K * 3^n)

Run:  python exact_solver.py  -> data/tiny/*.vrp and results/tiny_optima.csv
"""
import csv
import math
from pathlib import Path

from problem import CVRPInstance, verify_solution


def _held_karp(inst: CVRPInstance):
    n, D = inst.n, inst.dist_matrix
    full = 1 << n
    INF = math.inf
    # dp[mask][j]: shortest path from the depot through mask, ending at customer j+1
    dp = [[INF] * n for _ in range(full)]
    parent = [[-1] * n for _ in range(full)]
    for j in range(n):
        dp[1 << j][j] = D[0][j + 1]
    for mask in range(1, full):
        for j in range(n):
            cur = dp[mask][j]
            if cur == INF:
                continue
            for k in range(n):
                if mask & (1 << k):
                    continue
                nm = mask | (1 << k)
                val = cur + D[j + 1][k + 1]
                if val < dp[nm][k]:
                    dp[nm][k] = val
                    parent[nm][k] = j
    tsp = [INF] * full
    last = [-1] * full
    tsp[0] = 0.0
    for mask in range(1, full):
        for j in range(n):
            val = dp[mask][j] + D[j + 1][0]
            if val < tsp[mask]:
                tsp[mask], last[mask] = val, j

    def tour(mask):
        seq, j = [], last[mask]
        while j != -1:
            seq.append(j + 1)
            mask, j = mask ^ (1 << j), parent[mask][j]
        return seq[::-1]

    return tsp, tour


def solve_exact(inst: CVRPInstance):
    """Returns (optimal cost, routes), or (inf, None) if no feasible solution exists."""
    n, Q, K, q = inst.n, inst.capacity, inst.max_vehicles, inst.demands
    full = 1 << n
    tsp, tour = _held_karp(inst)

    load = [0] * full
    for mask in range(1, full):
        low = mask & -mask
        load[mask] = load[mask ^ low] + q[low.bit_length()]
    route_ok = [load[m] <= Q for m in range(full)]

    INF = math.inf
    f = [[INF] * full for _ in range(K + 1)]
    pick = [[0] * full for _ in range(K + 1)]
    f[0][0] = 0.0
    for k in range(1, K + 1):
        prev, cur, pk = f[k - 1], f[k], pick[k]
        for S in range(1, full):
            low = S & -S  # the route must contain the lowest customer -> no duplicate partitions
            sub = S
            while sub:
                if (sub & low) and route_ok[sub] and prev[S ^ sub] < INF:
                    val = prev[S ^ sub] + tsp[sub]
                    if val < cur[S]:
                        cur[S], pk[S] = val, sub
                sub = (sub - 1) & S
    k_best = min(range(1, K + 1), key=lambda k: f[k][full - 1])
    if f[k_best][full - 1] == INF:
        return INF, None
    routes, S, k = [], full - 1, k_best
    while S:
        sub = pick[k][S]
        routes.append(tour(sub))
        S, k = S ^ sub, k - 1
    return f[k_best][full - 1], routes


if __name__ == "__main__":
    from data_loader import generate_medidrone_instance, write_vrp, team_seed

    Path("results").mkdir(exist_ok=True)
    rows = []
    for n in (7, 8, 9):
        inst = generate_medidrone_instance(seed=team_seed(), n=n)
        inst.name = "MediDrone-tiny-n%d" % n
        write_vrp(inst, "data/tiny/%s.vrp" % inst.name)
        cost, routes = solve_exact(inst)
        check = verify_solution(inst, routes)
        assert check["ok"] and abs(check["cost"] - cost) < 1e-6
        rows.append({"instance": inst.name, "n": n, "Q": inst.capacity, "K": inst.max_vehicles,
                     "total_demand": sum(inst.demands), "optimal_cost": round(cost, 4),
                     "routes": " | ".join(" ".join(map(str, r)) for r in routes)})
        print(rows[-1])
    with open("results/tiny_optima.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
