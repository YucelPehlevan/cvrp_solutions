"""Task 4: Genetic Algorithm (Giant Tour + Split DP) for the CVRP.

Representation & Decoding (Prins, 2004):
- Chromosome: permutation pi = (pi_1, ..., pi_n) of customers without trip delimiters.
- Split DAG: vertices {0, 1, ..., n}; arc (i, j) exists if sum_{m=i+1..j} q[pi_m] <= Q,
  with cost c(i, j) = d(0, pi_{i+1}) + sum_{m=i+1..j-1} d(pi_m, pi_{m+1}) + d(pi_j, 0).
- Recurrence: V[0] = 0, V[j] = min_{0 <= i < j, load(i+1..j) <= Q} (V[i] + c(i, j)).
- Complexity: O(n^2) worst-case, O(n * L) in practice where L is max customers per route
  (inner loop breaks as soon as load > Q).
- Fleet limit repair: if Split produces > K routes due to fragmentation along the tour,
  lightest routes are merged into remaining routes via capacity-feasible cheapest insertion.
"""
import math
import random
import time
from typing import List, Optional, Tuple

from problem import CVRPInstance, calculate_cost, is_feasible


def split_dp(instance: CVRPInstance, tour: List[int]) -> Tuple[float, List[List[int]]]:
    """Exact optimal Split in O(n * L) time."""
    n, D, q, Q = instance.n, instance.dist_matrix, instance.demands, instance.capacity
    V = [math.inf] * (n + 1)
    pred = [-1] * (n + 1)
    V[0] = 0.0

    for i in range(n):
        if V[i] == math.inf:
            continue
        load = 0
        cost = 0.0
        for j in range(i + 1, n + 1):
            curr_c = tour[j - 1]
            load += q[curr_c]
            if load > Q:
                break  # capacity exceeded -> O(L) inner loop instead of O(n)
            if j == i + 1:
                cost = D[0][curr_c]
            else:
                cost += D[tour[j - 2]][curr_c]
            total = V[i] + cost + D[curr_c][0]
            if total < V[j] - 1e-12:
                V[j] = total
                pred[j] = i

    routes: List[List[int]] = []
    j = n
    while j > 0:
        i = pred[j]
        routes.append(tour[i:j])
        j = i
    routes.reverse()
    return V[n], routes


def _repair_fleet_limit(instance: CVRPInstance, routes: List[List[int]]) -> List[List[int]]:
    """If len(routes) > K, dissolve lightest routes via capacity-feasible cheapest insertion."""
    D, q, Q, K = instance.dist_matrix, instance.demands, instance.capacity, instance.max_vehicles
    routes = [list(r) for r in routes if r]
    if len(routes) <= K:
        return routes

    loads = [sum(q[c] for c in r) for r in routes]
    while len(routes) > K:
        lightest = min(range(len(routes)), key=lambda idx: loads[idx])
        moved_customers = routes.pop(lightest)
        loads.pop(lightest)

        for idx, c in enumerate(moved_customers):
            best = None
            for r_idx, r in enumerate(routes):
                if loads[r_idx] + q[c] > Q:
                    continue
                path = [0] + r + [0]
                for pos in range(len(path) - 1):
                    u, v = path[pos], path[pos + 1]
                    inc = D[u][c] + D[c][v] - D[u][v]
                    if best is None or inc < best[0]:
                        best = (inc, r_idx, pos)
            if best is None:
                # Keep all remaining unplaced customers so none are lost
                rem = moved_customers[idx:]
                routes.append(rem)
                loads.append(sum(q[x] for x in rem))
                return routes
            _, r_idx, pos = best
            routes[r_idx].insert(pos, c)
            loads[r_idx] += q[c]

    return routes


def _local_search_2opt(instance: CVRPInstance, routes: List[List[int]]) -> List[List[int]]:
    """Optional Memetic GA step: fast intra-route 2-opt local search."""
    D = instance.dist_matrix
    improved_routes = []
    for r in routes:
        if len(r) < 3:
            improved_routes.append(r)
            continue
        path = [0] + r + [0]
        improved = True
        while improved:
            improved = False
            m = len(path)
            for i in range(m - 3):
                for j in range(i + 2, m - 1):
                    delta = (D[path[i]][path[j]] + D[path[i + 1]][path[j + 1]]
                             - D[path[i]][path[i + 1]] - D[path[j]][path[j + 1]])
                    if delta < -1e-9:
                        path[i + 1:j + 1] = path[i + 1:j + 1][::-1]
                        improved = True
        improved_routes.append(path[1:-1])
    return improved_routes


def order_crossover(p1: List[int], p2: List[int], rng: random.Random) -> List[int]:
    """Order Crossover (OX): preserves relative order and produces valid permutations."""
    n = len(p1)
    a, b = sorted(rng.sample(range(n), 2))
    child = [0] * n
    child[a:b + 1] = p1[a:b + 1]
    in_child = set(p1[a:b + 1])

    p2_order = [gene for gene in (p2[b + 1:] + p2[:b + 1]) if gene not in in_child]
    idx = (b + 1) % n
    for gene in p2_order:
        child[idx] = gene
        idx = (idx + 1) % n
    return child


def mutate(tour: List[int], rng: random.Random) -> List[int]:
    """Permutation mutation: randomly applies segment inversion or customer swap."""
    res = list(tour)
    n = len(res)
    a, b = sorted(rng.sample(range(n), 2))
    if rng.random() < 0.7:
        res[a:b + 1] = res[a:b + 1][::-1]  # inversion
    else:
        res[a], res[b] = res[b], res[a]      # swap
    return res


def _binary_tournament(pop: List[Tuple[float, List[int], List[List[int]]]],
                       rng: random.Random) -> List[int]:
    i, j = rng.sample(range(len(pop)), 2)
    return pop[i][1] if pop[i][0] <= pop[j][0] else pop[j][1]


def genetic_algorithm(instance: CVRPInstance,
                      init_routes: Optional[List[List[int]]] = None,
                      time_limit: float = 10.0,
                      max_gen: Optional[int] = None,
                      seed: int = 0,
                      pop_size: int = 80,
                      crossover_rate: float = 0.85,
                      mutation_rate: float = 0.25,
                      elite_size: int = 4,
                      memetic: bool = False) -> dict:
    """Returns {'routes', 'cost', 'feasible', 'generations', 'history', 'time'}.
    Matches tabu_search return schema for unified experiment scripts."""
    rng = random.Random(seed)
    n, K = instance.n, instance.max_vehicles
    t0 = time.perf_counter()

    def evaluate_tour(tour: List[int]) -> Tuple[float, List[int], List[List[int]]]:
        _, routes = split_dp(instance, tour)
        if len(routes) > K:
            routes = _repair_fleet_limit(instance, routes)
        if memetic:
            routes = _local_search_2opt(instance, routes)
        dist = calculate_cost(instance, routes)
        excess_vehicles = max(0, len(routes) - K)
        fitness = dist + 10000.0 * excess_vehicles
        clean_tour = [c for r in routes for c in r]
        return (fitness, clean_tour, routes)

    # ---- 1) Initial population (with duplicate rejection + CW seed) ----
    seen = set()
    population: List[Tuple[float, List[int], List[List[int]]]] = []

    if init_routes:
        cw_tour = [c for r in init_routes for c in r]
        if len(cw_tour) == n:
            ind = evaluate_tour(cw_tour)
            seen.add(tuple(ind[1]))
            population.append(ind)

    base = list(range(1, n + 1))
    while len(population) < pop_size:
        cand = list(base)
        rng.shuffle(cand)
        key = tuple(cand)
        if key not in seen:
            seen.add(key)
            population.append(evaluate_tour(cand))

    population.sort(key=lambda x: x[0])
    best_fitness, _, best_routes = population[0]
    history = [(0.0, calculate_cost(instance, best_routes))]
    gen = 0

    # ---- 2) Evolutionary loop ----
    while True:
        if max_gen is not None and gen >= max_gen:
            break
        if time.perf_counter() - t0 >= time_limit:
            break

        next_pop: List[Tuple[float, List[int], List[List[int]]]] = []
        next_seen = set()

        # Elitism: preserve top distinct individuals
        for ind in population[:elite_size]:
            key = tuple(ind[1])
            if key not in next_seen:
                next_seen.add(key)
                next_pop.append(ind)

        # Offspring generation with duplicate rejection
        attempts = 0
        max_attempts = pop_size * 5
        while len(next_pop) < pop_size and attempts < max_attempts:
            attempts += 1
            p1 = _binary_tournament(population, rng)
            if rng.random() < crossover_rate:
                p2 = _binary_tournament(population, rng)
                child = order_crossover(p1, p2, rng)
            else:
                child = list(p1)

            if rng.random() < mutation_rate:
                child = mutate(child, rng)

            key = tuple(child)
            if key in next_seen:
                child = mutate(child, rng)
                key = tuple(child)
                if key in next_seen:
                    continue

            next_seen.add(key)
            next_pop.append(evaluate_tour(child))

        while len(next_pop) < pop_size:
            cand = list(base)
            rng.shuffle(cand)
            key = tuple(cand)
            if key not in next_seen:
                next_seen.add(key)
                next_pop.append(evaluate_tour(cand))

        next_pop.sort(key=lambda x: x[0])
        population = next_pop
        gen += 1

        if population[0][0] < best_fitness - 1e-9:
            best_fitness, _, best_routes = population[0]
            history.append((time.perf_counter() - t0, calculate_cost(instance, best_routes)))

    elapsed = time.perf_counter() - t0
    ok, _ = is_feasible(instance, best_routes)
    return {
        'routes': best_routes,
        'cost': calculate_cost(instance, best_routes),
        'feasible': ok,
        'generations': gen,
        'history': history,
        'time': elapsed,
    }


if __name__ == "__main__":
    from clarke_wright import clarke_wright
    from data_loader import generate_medidrone_instance, parse_cvrplib, team_seed
    from problem import verify_solution

    print("--- Standard GA vs Memetic GA Benchmark Test ---")
    for name in ["A-n32-k5", "A-n45-k6", "A-n60-k9", "A-n80-k10"]:
        inst = parse_cvrplib(f"data/{name}.vrp")
        cw_routes = clarke_wright(inst)
        cw_cost = calculate_cost(inst, cw_routes)

        res_ga = genetic_algorithm(inst, init_routes=cw_routes, time_limit=5.0, seed=1, memetic=False)
        res_ma = genetic_algorithm(inst, init_routes=cw_routes, time_limit=5.0, seed=1, memetic=True)

        gap_ga = 100 * (res_ga['cost'] - inst.bks) / inst.bks
        gap_ma = 100 * (res_ma['cost'] - inst.bks) / inst.bks

        check_ga = verify_solution(inst, res_ga['routes'])
        check_ma = verify_solution(inst, res_ma['routes'])

        print(f"{name} (BKS={inst.bks:.0f}): CW={cw_cost:.0f} | "
              f"GA={res_ga['cost']:.0f} (gap={gap_ga:.2f}%, ok={check_ga['ok']}, gen={res_ga['generations']}) | "
              f"Memetic={res_ma['cost']:.0f} (gap={gap_ma:.2f}%, ok={check_ma['ok']}, gen={res_ma['generations']})")

    # Team MediDrone-100 test
    medi = generate_medidrone_instance(seed=team_seed(), n=100)
    cw_medi = clarke_wright(medi)
    res_medi = genetic_algorithm(medi, init_routes=cw_medi, time_limit=5.0, seed=1, memetic=True)
    check_medi = verify_solution(medi, res_medi['routes'])
    print(f"\nMediDrone-100 (K={medi.max_vehicles}): CW={calculate_cost(medi, cw_medi):.2f} -> "
          f"Memetic GA={res_medi['cost']:.2f}, routes={len(res_medi['routes'])}, "
          f"feasible={check_medi['ok']}, gen={res_medi['generations']}")