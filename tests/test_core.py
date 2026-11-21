import itertools
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_loader import generate_medidrone_instance, parse_cvrplib, team_seed, write_vrp
from exact_solver import solve_exact
from problem import calculate_cost, compute_distance, is_feasible, verify_solution

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def test_seed_is_sum_of_last_four_digits():
    assert team_seed() == 6915 + 5071 + 5804 == 17790


def test_medidrone_follows_project_rules():
    inst = generate_medidrone_instance()
    assert inst.n == 100 and inst.coords[0] == (500.0, 500.0)
    assert all(1 <= q <= 30 for q in inst.demands[1:]) and inst.demands[0] == 0
    assert inst.capacity == 100
    assert inst.max_vehicles == math.ceil(sum(inst.demands) / 100) + 1
    sides = {inst.coords[i][0] < 500 for i in range(1, inst.n + 1)}
    assert sides == {True, False}  # customers on both sides
    # different sides -> Euclidean + 150
    eu = next(i for i in range(1, 101) if inst.coords[i][0] < 500)
    (x1, y1), (x2, y2) = inst.coords[0], inst.coords[eu]
    assert abs(inst.dist_matrix[0][eu] - (math.hypot(x1 - x2, y1 - y2) + 150)) < 1e-9


def test_generator_is_deterministic():
    a, b = generate_medidrone_instance(), generate_medidrone_instance()
    assert a.coords == b.coords and a.demands == b.demands


def test_medidrone_triangle_inequality():
    # Task 1, Section 5: the penalised distance is still a metric
    D = generate_medidrone_instance().dist_matrix
    m = len(D)
    for i in range(m):
        for j in range(m):
            for k in range(m):
                assert D[i][j] <= D[i][k] + D[k][j] + 1e-9


def test_cvrplib_parser():
    inst = parse_cvrplib(os.path.join(DATA, "A-n32-k5.vrp"))
    assert (inst.n, inst.capacity, inst.max_vehicles, inst.bks) == (31, 100, 5, 784)
    assert inst.rounding and inst.dist_matrix[0][1] == float(int(inst.dist_matrix[0][1]))
    # published optimal CVRPLIB solution must cost exactly 784
    opt = [[21, 31, 19, 17, 13, 7, 26], [12, 1, 16, 30], [27, 24],
           [29, 18, 8, 9, 22, 15, 10, 25, 5, 20], [14, 28, 11, 4, 23, 3, 2, 6]]
    res = verify_solution(inst, opt)
    assert res["ok"] and res["cost"] == calculate_cost(inst, opt) == 784


def test_write_read_roundtrip(tmp_path):
    inst = generate_medidrone_instance(n=9)
    p = tmp_path / "x.vrp"
    write_vrp(inst, str(p))
    back = parse_cvrplib(str(p))
    assert back.coords == inst.coords and back.demands == inst.demands
    assert back.dist_matrix == inst.dist_matrix and back.max_vehicles == inst.max_vehicles


def test_checker_catches_every_error():
    inst = generate_medidrone_instance(n=9)
    good = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert verify_solution(inst, [[c] for c in range(1, 10)])["ok"] is False  # 9 routes > K
    assert is_feasible(inst, [[1, 2, 3], [4, 5, 6], [7, 8]])[0] is False        # 9 missing
    assert is_feasible(inst, [[1, 2, 3, 9], [4, 5, 6], [7, 8, 9]])[0] is False  # 9 twice
    assert is_feasible(inst, [[1, 2, 3], [4, 5, 0, 6], [7, 8, 9]])[0] is False  # depot inside a route
    assert is_feasible(inst, [[1, 2, 3], [4, 5, 6], [7, 8, 99]])[0] is False    # invalid id
    big = [list(range(1, 10))]
    load = sum(inst.demands)
    assert verify_solution(inst, big)["ok"] == (load <= inst.capacity)        # capacity
    # cost recomputed from coordinates equals the matrix-based cost
    res = verify_solution(inst, good)
    assert abs(res["cost"] - calculate_cost(inst, good)) < 1e-9


def _brute_force(inst):
    """All permutations x best split of each permutation into <= K routes."""
    n, D, q, Q, K = inst.n, inst.dist_matrix, inst.demands, inst.capacity, inst.max_vehicles
    best = math.inf
    for perm in itertools.permutations(range(1, n + 1)):
        # f[k][i] = cheapest way to serve the first i customers with k routes
        f = [[math.inf] * (n + 1) for _ in range(K + 1)]
        f[0][0] = 0.0
        for k in range(1, K + 1):
            for i in range(n):
                if f[k - 1][i] == math.inf:
                    continue
                load, path = 0, 0.0
                for j in range(i, n):
                    load += q[perm[j]]
                    if load > Q:
                        break
                    path += D[0][perm[i]] if j == i else D[perm[j - 1]][perm[j]]
                    f[k][j + 1] = min(f[k][j + 1], f[k - 1][i] + path + D[perm[j]][0])
        best = min(best, min(f[k][n] for k in range(1, K + 1)))
    return best


def test_exact_solver_matches_brute_force():
    for n in (7, 8):
        inst = generate_medidrone_instance(n=n)
        cost, routes = solve_exact(inst)
        res = verify_solution(inst, routes)
        assert res["ok"] and abs(res["cost"] - cost) < 1e-6
        assert abs(cost - _brute_force(inst)) < 1e-6


def test_compute_distance_matches_matrix():
    inst = parse_cvrplib(os.path.join(DATA, "A-n45-k6.vrp"))
    for i in range(inst.n + 1):
        for j in range(inst.n + 1):
            if i != j:
                assert compute_distance(inst, i, j) == inst.dist_matrix[i][j]
