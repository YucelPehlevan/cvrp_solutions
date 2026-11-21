# MediDrone Istanbul – CVRP with Genetic Algorithm & Tabu Search

CMP 3005 Analysis of Algorithms, Fall 2026–2027 term project.

**Team:** Batın Arda Ünal (2366915) · Yücel Pehlevan (2365071) · İclal Korkmaz (2365804)
**Instance seed:** 6915 + 5071 + 5804 = **17790** (sum of the last four digits of the student IDs, see `data_loader.team_seed`)

## Requirements

Python 3.10+ and two packages (no VRP / metaheuristic libraries are used):

```
pip install -r requirements.txt
```

## Project structure

| File | Content | Task |
|---|---|---|
| `problem.py` | `CVRPInstance`, distance rule, `calculate_cost`, independent feasibility checker `verify_solution` / `is_feasible` | 1, 2 |
| `data_loader.py` | CVRPLIB parser, BKS table, MediDrone generator, `write_vrp`, `team_seed` | 2(a), 2(b) |
| `exact_solver.py` | Exact optimum for tiny instances (Held-Karp + set-partition DP) | 2(c) |
| `plotting.py` | Instance and route plots (Bosphorus line, EU/AS sides) | M1, 7 |
| `clarke_wright.py` | Clarke–Wright savings baseline | 3 |
| `genetic_algo.py` | Genetic Algorithm (giant tour + Split) | 4 |
| `tabu_search.py` | Tabu Search | 5 |
| `main.py` | Experiment runner | 7 |
| `data/` | CVRPLIB A-n32-k5, A-n45-k6, A-n60-k9, A-n80-k10; `medidrone/`, `tiny/` (generated) | 2 |
| `results/`, `figures/` | Generated tables (CSV) and plots | 7 |
| `tests/` | `pytest` tests | – |

## Reproduce

```
python plotting.py        # data/medidrone/MediDrone-100.vrp + figures/medidrone_instance.png
python exact_solver.py    # data/tiny/*.vrp + results/tiny_optima.csv (n = 7, 8, 9)
python -m pytest -q       # all tests
```

## Data

**Benchmarks (CVRPLIB, Augerat Set A).** Distances are rounded to the nearest integer as in CVRPLIB.
Best-known solutions (all proven optimal): A-n32-k5 = 784, A-n45-k6 = 944, A-n60-k9 = 1354, A-n80-k10 = 1763.
Note: integer rounding can break the triangle inequality by at most 1 unit (we found 66 to 1194
violating triples per instance). None of our algorithms relies on the triangle inequality.

**MediDrone-100 (seed 17790).** Depot at (500, 500); European side x < 500, Asian side x ≥ 500
(the depot is on the Asian side). 5 fixed cluster centres (3 EU, 2 AS), 80 clustered customers
(Gaussian, σ = 60) + 20 uniform customers; qᵢ ~ U{1..30}; Q = 100; K = ⌈Σqᵢ/Q⌉ + 1.
d(i, j) = Euclidean + 150 if i and j are on different sides (no rounding).
Result: 53 EU / 47 AS customers, total demand 1494, K = 16.

**Tiny instances.** Same generator with n = 7, 8, 9 (same seed). Exact optima:

| Instance | n | K | Optimal cost |
|---|---|---|---|
| MediDrone-tiny-n7 | 7 | 3 | 2581.5101 |
| MediDrone-tiny-n8 | 8 | 3 | 2597.1171 |
| MediDrone-tiny-n9 | 9 | 3 | 3216.7790 |

The DP solver is cross-checked against brute force (all permutations × optimal split) in the tests.

## Feasibility checker

`problem.verify_solution` checks every reported solution independently of the solvers:
at most K routes, valid customer ids (depot not inside a route), every customer exactly once,
every route load ≤ Q, and it **recomputes the cost from the coordinates** (not from the distance
matrix the solvers use). It returns all errors, not only the first one.

## Code origin

All code is written by the team. AI assistance (Claude) was used for parts of the code and tests;
details are given in Appendix B of the report.

## References

- Clarke & Wright (1964); Prins (2004); Glover & Laguna (1997); Toth & Vigo (2014).
- CVRPLIB: http://vrp.galgos.inf.puc-rio.br/
