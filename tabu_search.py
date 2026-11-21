"""Task 5: Tabu Search for the CVRP.

- Moves: relocate, swap, 2-opt, 2-opt*; every move cost is evaluated in O(1).
- Candidate list: only the p nearest customers are tried (granular neighbourhood).
- Tabu list: arcs removed by a move cannot be added back for `tenure` iterations.
- Aspiration: a tabu move is allowed if it gives a new best feasible solution.
- Capacity: f = distance + lambda * overload, lambda is adapted during the search.

Input/output routes: lists of customers without the depot.
Internally every route is [0, c1, ..., ck, 0] and there are exactly K routes ([0, 0] = empty).
"""
import math
import random
import time
from typing import List, Optional

from problem import CVRPInstance, calculate_cost, is_feasible


def sweep_initial(instance: CVRPInstance) -> List[List[int]]:
    """Temporary start solution until Clarke-Wright is ready: sort customers by angle
    around the depot and open a new route when the capacity is full."""
    x0, y0 = instance.coords[0]
    order = sorted(range(1, instance.n + 1),
                   key=lambda c: math.atan2(instance.coords[c][1] - y0, instance.coords[c][0] - x0))
    routes, cur, load = [], [], 0
    for c in order:
        if load + instance.demands[c] > instance.capacity:
            routes.append(cur)
            cur, load = [], 0
        cur.append(c)
        load += instance.demands[c]
    if cur:
        routes.append(cur)
    return routes


def _reduce_to_k(instance: CVRPInstance, routes: List[List[int]]) -> List[List[int]]:
    """If the start has more than K routes, dissolve the lightest ones (cheapest insertion).
    Capacity is ignored here; the penalty term repairs any overload."""
    D, q = instance.dist_matrix, instance.demands
    routes = [list(r) for r in routes if r]
    while len(routes) > instance.max_vehicles:
        lightest = min(range(len(routes)), key=lambda r: sum(q[c] for c in routes[r]))
        moved = routes.pop(lightest)
        for c in moved:
            best = None
            for r, route in enumerate(routes):
                path = [0] + route + [0]
                for k in range(len(path) - 1):
                    u, v = path[k], path[k + 1]
                    inc = D[u][c] + D[c][v] - D[u][v]
                    if best is None or inc < best[0]:
                        best = (inc, r, k)
            _, r, k = best
            routes[r].insert(k, c)
    return routes


def tabu_search(instance: CVRPInstance,
                init_routes: List[List[int]],
                time_limit: float = 10.0,
                max_iter: Optional[int] = None,
                seed: int = 0,
                n_neighbors: int = 15,
                tenure_range=None,
                lambda0: Optional[float] = None,
                lambda_factor: float = 1.5,
                adapt_period: int = 10,
                debug: bool = False) -> dict:
    """Returns {'routes', 'cost', 'feasible', 'iterations', 'history', 'time'}.
    history = [(seconds, best cost)] for the convergence plot."""
    rng = random.Random(seed)
    D, q, Q, K, n = (instance.dist_matrix, instance.demands, instance.capacity,
                     instance.max_vehicles, instance.n)

    # ---- parameters ----
    p = min(n_neighbors, n - 1)
    if tenure_range is None:  # about sqrt(n): [5, 15] for n = 100
        tenure_range = (max(1, int(0.5 * math.sqrt(n))), max(2, int(1.5 * math.sqrt(n))))
    # customers sorted by distance, computed once: O(n^2 log n).
    # Normally only the p nearest are tried; if c's route is overloaded all are tried,
    # because the nearby routes are usually full and free capacity is far away.
    nbrs_all = [[]] + [sorted((e for e in range(1, n + 1) if e != c), key=lambda e: D[c][e])
                       for c in range(1, n + 1)]
    nbrs = [lst[:p] for lst in nbrs_all]

    # ---- solution state ----
    routes = [[0] + r + [0] for r in _reduce_to_k(instance, init_routes)]
    while len(routes) < K:
        routes.append([0, 0])
    route_of = [0] * (n + 1)   # customer -> route index
    pos_of = [0] * (n + 1)     # customer -> position in its route
    load = [0] * K
    pre = [None] * K           # prefix loads: pre[r][k] = load of route[1..k]

    def rebuild(r):  # O(L) update of one changed route
        route = routes[r]
        acc, pr = 0, [0] * len(route)
        for k in range(1, len(route) - 1):
            c = route[k]
            route_of[c], pos_of[c] = r, k
            acc += q[c]
            pr[k] = acc
        pr[-1] = acc
        pre[r], load[r] = pr, acc

    for r in range(K):
        rebuild(r)

    def ex(l):  # overload of one route
        return l - Q if l > Q else 0

    cur_cost = calculate_cost(instance, [r[1:-1] for r in routes])
    cur_excess = sum(ex(l) for l in load)
    lam = lambda0 if lambda0 is not None else max(cur_cost / max(1, sum(q)), 1e-6)
    lam_min, lam_max = lam / 100.0, lam * 100.0

    best_routes, best_cost = None, math.inf
    t0 = time.perf_counter()
    history = []
    if cur_excess == 0:
        best_routes, best_cost = [r[1:-1] for r in routes if len(r) > 2], cur_cost
        history.append((0.0, best_cost))

    tabu = {}  # arc (u, v), u < v -> tabu until this iteration
    it = 0
    feasible_count = 0

    def is_tabu(arcs):
        for u, v in arcs:
            key = (u, v) if u < v else (v, u)
            if tabu.get(key, -1) > it:
                return True
        return False

    while True:
        if max_iter is not None and it >= max_iter:
            break
        if time.perf_counter() - t0 >= time_limit:
            break

        best_val, best_move = math.inf, None

        def consider(val, dc, new_excess, added, removed, move):
            """Keep the move if it is the best so far and not tabu (or aspiration holds)."""
            nonlocal best_val, best_move
            if val >= best_val - 1e-12:
                return
            if is_tabu(added):
                if not (new_excess == 0 and cur_cost + dc < best_cost - 1e-9):  # aspiration
                    return
            best_val = val
            best_move = (move, dc, removed)

        empty_r = next((r for r in range(K) if len(routes[r]) == 2), None)

        for c in range(1, n + 1):
            rc, i = route_of[c], pos_of[c]
            R1 = routes[rc]
            pc, sc = R1[i - 1], R1[i + 1]
            qc = q[c]
            d_rem = D[pc][sc] - D[pc][c] - D[c][sc]   # removing c: pc-c-sc -> pc-sc

            # RELOCATE c into an empty route
            if empty_r is not None and len(R1) > 3:
                dc = d_rem + 2 * D[0][c]
                dex = ex(load[rc] - qc) - ex(load[rc])
                consider(dc + lam * dex, dc, cur_excess + dex,
                         [(pc, sc), (0, c)], [(pc, c), (c, sc)],
                         ('reloc', c, empty_r, 0))

            for e in (nbrs_all[c] if load[rc] > Q else nbrs[c]):
                re, j = route_of[e], pos_of[e]
                R2 = routes[re]
                pe, se = R2[j - 1], R2[j + 1]

                # RELOCATE c between (e, se) or (pe, e):  delta = d_rem + d(u,c) + d(c,v) - d(u,v)
                if re == rc:
                    dex = 0
                else:
                    dex = (ex(load[rc] - qc) + ex(load[re] + qc)
                           - ex(load[rc]) - ex(load[re]))
                for k, u, v in ((j, e, se), (j - 1, pe, e)):
                    if u == c or v == c:
                        continue
                    dc = d_rem + D[u][c] + D[c][v] - D[u][v]
                    consider(dc + lam * dex, dc, cur_excess + dex,
                             [(pc, sc), (u, c), (c, v)], [(pc, c), (c, sc), (u, v)],
                             ('reloc', c, re, k))

                if re != rc:
                    qe = q[e]
                    # SWAP c and e (different routes)
                    dc = (D[pc][e] + D[e][sc] + D[pe][c] + D[c][se]
                          - D[pc][c] - D[c][sc] - D[pe][e] - D[e][se])
                    dex = (ex(load[rc] - qc + qe) + ex(load[re] - qe + qc)
                           - ex(load[rc]) - ex(load[re]))
                    consider(dc + lam * dex, dc, cur_excess + dex,
                             [(pc, e), (e, sc), (pe, c), (c, se)],
                             [(pc, c), (c, sc), (pe, e), (e, se)],
                             ('swap', c, e))

                    # 2-OPT* (straight): R1[..c] + R2[e..]  and  R2[..pe] + R1[sc..]
                    dc = D[c][e] + D[pe][sc] - D[c][sc] - D[pe][e]
                    l1 = pre[rc][i] + load[re] - pre[re][j - 1]   # new loads in O(1)
                    l2 = pre[re][j - 1] + load[rc] - pre[rc][i]
                    dex = ex(l1) + ex(l2) - ex(load[rc]) - ex(load[re])
                    consider(dc + lam * dex, dc, cur_excess + dex,
                             [(c, e), (pe, sc)], [(c, sc), (pe, e)],
                             ('2optS', rc, i, re, j))

                    # 2-OPT* (reversed): R1[..c] + rev(R2[..e])  and  rev(R1[sc..]) + R2[se..]
                    dc = D[c][e] + D[sc][se] - D[c][sc] - D[e][se]
                    l1 = pre[rc][i] + pre[re][j]
                    l2 = load[rc] - pre[rc][i] + load[re] - pre[re][j]
                    dex = ex(l1) + ex(l2) - ex(load[rc]) - ex(load[re])
                    consider(dc + lam * dex, dc, cur_excess + dex,
                             [(c, e), (sc, se)], [(c, sc), (e, se)],
                             ('2optR', rc, i, re, j))
                else:
                    # 2-OPT inside one route: two reversals that create the arc (c, e)
                    a, b = (i, j) if i < j else (j, i)
                    for x, y in ((a, b), (a - 1, b - 1)):
                        if y - x < 2 or x < 0:
                            continue
                        u1, v1, u2, v2 = R1[x], R1[x + 1], R1[y], R1[y + 1]
                        dc = D[u1][u2] + D[v1][v2] - D[u1][v1] - D[u2][v2]
                        consider(dc, dc, cur_excess,
                                 [(u1, u2), (v1, v2)], [(u1, v1), (u2, v2)],
                                 ('2opt', rc, x, y))

        if best_move is None:  # every move is tabu
            it += 1
            continue

        # ---- apply the best move ----
        move, dc, removed = best_move
        kind = move[0]
        if kind == 'reloc':
            _, c, rt, k = move
            rs = route_of[c]
            src = routes[rs]
            i = pos_of[c]
            src.pop(i)
            if rs == rt and k > i:
                k -= 1
            routes[rt].insert(k + 1, c)
            changed = {rs, rt}
        elif kind == 'swap':
            _, c, e = move
            rc, i, re, j = route_of[c], pos_of[c], route_of[e], pos_of[e]
            routes[rc][i], routes[re][j] = e, c
            changed = {rc, re}
        elif kind == '2opt':
            _, r, x, y = move
            routes[r][x + 1:y + 1] = routes[r][x + 1:y + 1][::-1]
            changed = {r}
        elif kind == '2optS':
            _, r1, i, r2, j = move
            A, B = routes[r1], routes[r2]
            routes[r1], routes[r2] = A[:i + 1] + B[j:], B[:j] + A[i + 1:]
            changed = {r1, r2}
        else:  # '2optR'
            _, r1, i, r2, j = move
            A, B = routes[r1], routes[r2]
            routes[r1], routes[r2] = A[:i + 1] + B[j::-1], A[:i:-1] + B[j + 1:]
            changed = {r1, r2}

        for r in changed:
            rebuild(r)
        cur_cost += dc
        cur_excess = sum(ex(l) for l in load)

        tenure = rng.randint(*tenure_range)  # random tenure helps to avoid cycles
        for u, v in removed:
            if u != v:
                tabu[(u, v) if u < v else (v, u)] = it + tenure

        if debug:  # O(1) delta must equal full recomputation
            full = calculate_cost(instance, [r[1:-1] for r in routes])
            assert abs(full - cur_cost) < 1e-6, (kind, full, cur_cost)
            assert all(r[0] == 0 and r[-1] == 0 and 0 not in r[1:-1] for r in routes), kind
            assert sorted(c for r in routes for c in r[1:-1]) == list(range(1, n + 1)), kind

        if cur_excess == 0:
            feasible_count += 1
            if cur_cost < best_cost - 1e-9:
                best_cost = cur_cost
                best_routes = [r[1:-1] for r in routes if len(r) > 2]
                history.append((time.perf_counter() - t0, best_cost))

        # lambda: always feasible -> relax, never feasible -> increase
        it += 1
        if it % adapt_period == 0:
            if feasible_count == adapt_period:
                lam = max(lam_min, lam / lambda_factor)
            elif feasible_count == 0:
                lam = min(lam_max, lam * lambda_factor)
            feasible_count = 0

    elapsed = time.perf_counter() - t0
    if best_routes is None:  # no feasible solution found
        return {'routes': [r[1:-1] for r in routes if len(r) > 2], 'cost': cur_cost,
                'feasible': False, 'iterations': it, 'history': history, 'time': elapsed}
    ok, _ = is_feasible(instance, best_routes)
    return {'routes': best_routes, 'cost': calculate_cost(instance, best_routes),
            'feasible': ok, 'iterations': it, 'history': history, 'time': elapsed}


if __name__ == "__main__":
    from data_loader import parse_cvrplib
    inst = parse_cvrplib("data/A-n32-k5.vrp")
    start = sweep_initial(inst)
    res = tabu_search(inst, start, time_limit=5, seed=1)
    print("start:", calculate_cost(inst, start), "| TS:", res['cost'],
          "| feasible:", res['feasible'], "| iterations:", res['iterations'])
