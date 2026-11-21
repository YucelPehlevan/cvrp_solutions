import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_loader import generate_medidrone_instance, parse_cvrplib
from problem import calculate_cost, is_feasible
from tabu_search import sweep_initial, tabu_search

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def load(name):
    return parse_cvrplib(os.path.join(DATA, name + ".vrp"))


def test_delta_matches_full_recomputation():
    # debug=True asserts after every move that the O(1) delta cost equals full recomputation
    for name in ["A-n32-k5", "A-n60-k9"]:
        inst = load(name)
        for seed in range(3):
            tabu_search(inst, sweep_initial(inst), max_iter=300, time_limit=1e9,
                        seed=seed, debug=True)
    inst = generate_medidrone_instance()
    tabu_search(inst, sweep_initial(inst), max_iter=200, time_limit=1e9, seed=0, debug=True)


def test_result_is_feasible_and_not_worse_than_start():
    inst = load("A-n45-k6")
    start = sweep_initial(inst)
    res = tabu_search(inst, start, max_iter=500, time_limit=1e9, seed=0)
    ok, msg = is_feasible(inst, res["routes"])
    assert ok, msg
    assert res["feasible"]
    assert abs(res["cost"] - calculate_cost(inst, res["routes"])) < 1e-6
    assert res["cost"] <= calculate_cost(inst, start)


def test_same_seed_same_result():
    inst = load("A-n32-k5")
    a = tabu_search(inst, sweep_initial(inst), max_iter=200, time_limit=1e9, seed=7)
    b = tabu_search(inst, sweep_initial(inst), max_iter=200, time_limit=1e9, seed=7)
    assert a["routes"] == b["routes"] and a["cost"] == b["cost"]


def test_more_than_k_initial_routes_is_repaired():
    # one route per customer (n > K routes): TS must reduce to K routes and become feasible
    inst = generate_medidrone_instance()
    start = [[c] for c in range(1, inst.n + 1)]
    res = tabu_search(inst, start, max_iter=2000, time_limit=1e9, seed=0)
    assert res["feasible"]
    assert len(res["routes"]) <= inst.max_vehicles


def test_reaches_optimum_on_a_n32_k5():
    inst = load("A-n32-k5")
    res = tabu_search(inst, sweep_initial(inst), max_iter=2000, time_limit=1e9, seed=0)
    assert res["cost"] == 784  # CVRPLIB BKS (optimal)
