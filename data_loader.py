import math
import random
import re
from pathlib import Path

from problem import CVRPInstance, build_dist_matrix

# CVRPLIB best-known solutions (Augerat Set A, all proven optimal)
BKS = {"A-n32-k5": 784, "A-n45-k6": 944, "A-n60-k9": 1354, "A-n80-k10": 1763}

TEAM_IDS = ["2366915", "2365071", "2365804"]


def team_seed(ids=TEAM_IDS) -> int:
    """Sum of the last four digits: 6915 + 5071 + 5804 = 17790."""
    return sum(int(s[-4:]) for s in ids)


def generate_medidrone_instance(seed: int = 17790, n: int = 100) -> CVRPInstance:
    """Task 2(b) team instance (n = 100); small n gives the Task 2(c) tiny instances."""
    rng = random.Random(seed)
    capacity = 100
    coords = [(500.0, 500.0)]  # depot at the centre

    # 5 clusters (3 EU, 2 AS) with 80% of customers, 20% uniform
    clusters = [(200, 250), (350, 750), (150, 600), (700, 300), (800, 700)]
    n_uniform = n // 5
    for i in range(n - n_uniform):
        cx, cy = clusters[i % len(clusters)]
        x = min(999.0, max(0.0, rng.gauss(cx, 60)))
        y = min(999.0, max(0.0, rng.gauss(cy, 60)))
        coords.append((x, y))

    for _ in range(n_uniform):
        coords.append((rng.uniform(0, 1000), rng.uniform(0, 1000)))

    demands = [0] + [rng.randint(1, 30) for _ in range(n)]
    max_vehicles = math.ceil(sum(demands) / capacity) + 1

    name = "MediDrone-%d" % n if seed == 17790 else "MediDrone-%d-s%d" % (n, seed)
    inst = CVRPInstance(name, n, coords, demands, capacity, max_vehicles, [],
                        rounding=False, split_x=500.0, penalty=150.0)
    inst.dist_matrix = build_dist_matrix(inst)  # Euclidean + 150 across the Bosphorus
    return inst


def parse_cvrplib(filepath: str) -> CVRPInstance:
    """Task 2(a): reads CVRPLIB .vrp files (and our own files written by write_vrp)."""
    header, coords, demands, depots = {}, {}, {}, []
    section = None
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line == "EOF":
                continue
            if line.endswith("_SECTION"):
                section = line
                continue
            if section is None:
                if ":" in line:
                    key, val = line.split(":", 1)
                    header[key.strip().upper()] = val.strip()
                continue
            parts = line.split()
            if section == "NODE_COORD_SECTION":
                coords[int(parts[0])] = (float(parts[1]), float(parts[2]))
            elif section == "DEMAND_SECTION":
                demands[int(parts[0])] = int(parts[1])
            elif section == "DEPOT_SECTION" and int(parts[0]) != -1:
                depots.append(int(parts[0]))

    if header.get("EDGE_WEIGHT_TYPE", "EUC_2D") != "EUC_2D":
        raise ValueError("Only EUC_2D is supported")

    name = header.get("NAME", Path(filepath).stem)
    depot = depots[0] if depots else min(coords)
    order = [depot] + sorted(k for k in coords if k != depot)  # renumber: depot = 0

    if "VEHICLES" in header:
        k = int(header["VEHICLES"])
    elif re.search(r"-k(\d+)$", name):
        k = int(re.search(r"-k(\d+)$", name).group(1))  # A-n32-k5 -> K = 5
    else:
        raise ValueError("Number of vehicles (K) not found: " + name)

    bks = BKS.get(name)
    m = re.search(r"(?:Optimal|Best) value:\s*(\d+(?:\.\d+)?)", header.get("COMMENT", ""))
    if bks is None and m:
        bks = float(m.group(1))

    inst = CVRPInstance(
        name=name,
        n=len(order) - 1,
        coords=[coords[i] for i in order],
        demands=[demands[i] for i in order],
        capacity=int(header["CAPACITY"]),
        max_vehicles=k,
        dist_matrix=[],
        rounding=header.get("DISTANCE_ROUNDING", "nint") == "nint",  # CVRPLIB default
        split_x=float(header["BOSPHORUS_X"]) if "BOSPHORUS_X" in header else None,
        penalty=float(header.get("BOSPHORUS_PENALTY", 0)),
        bks=bks,
    )
    inst.dist_matrix = build_dist_matrix(inst)
    return inst


def write_vrp(inst: CVRPInstance, filepath: str) -> None:
    """CVRPLIB format plus extra keys for rounding and the Bosphorus."""
    lines = ["NAME : " + inst.name, "TYPE : CVRP", "DIMENSION : %d" % (inst.n + 1),
             "EDGE_WEIGHT_TYPE : EUC_2D", "CAPACITY : %d" % inst.capacity,
             "VEHICLES : %d" % inst.max_vehicles,
             "DISTANCE_ROUNDING : " + ("nint" if inst.rounding else "none")]
    if inst.split_x is not None:
        lines += ["BOSPHORUS_X : %g" % inst.split_x, "BOSPHORUS_PENALTY : %g" % inst.penalty]
    lines.append("NODE_COORD_SECTION")
    lines += ["%d %r %r" % (i + 1, x, y) for i, (x, y) in enumerate(inst.coords)]
    lines.append("DEMAND_SECTION")
    lines += ["%d %d" % (i + 1, q) for i, q in enumerate(inst.demands)]
    lines += ["DEPOT_SECTION", "1", "-1", "EOF"]
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    Path(filepath).write_text("\n".join(lines) + "\n", encoding="utf-8")
