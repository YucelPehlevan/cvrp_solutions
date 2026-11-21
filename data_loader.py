import math
import random
from problem import CVRPInstance

def generate_medidrone_instance(seed: int = 17790) -> CVRPInstance:
    """Task 2(b): Ekip seed'i (17790) ile 100 noktalı MediDrone İstanbul haritası üretir."""
    rng = random.Random(seed)
    n = 100
    capacity = 100
    coords = [(500.0, 500.0)]  # 0: Merkez laboratuvar (depo)

    # 5 küme (Avrupa x < 500, Asya x >= 500) + %20 rastgele nokta
    clusters = [(200, 250), (350, 750), (150, 600), (700, 300), (800, 700)]
    for i in range(80):
        cx, cy = clusters[i % len(clusters)]
        x = min(999.0, max(0.0, rng.gauss(cx, 60)))
        y = min(999.0, max(0.0, rng.gauss(cy, 60)))
        coords.append((x, y))

    for _ in range(20):
        coords.append((rng.uniform(0, 1000), rng.uniform(0, 1000)))

    demands = [0] + [rng.randint(1, 30) for _ in range(n)]
    max_vehicles = math.ceil(sum(demands) / capacity) + 1

    # Mesafe matrisi: Öklid + Boğaz geçiş cezası (P = 150)
    dist_matrix = [[0.0] * (n + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            d = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
            if (coords[i][0] < 500) != (coords[j][0] < 500):
                d += 150.0
            dist_matrix[i][j] = dist_matrix[j][i] = d

    return CVRPInstance("MediDrone-100", n, coords, demands, capacity, max_vehicles, dist_matrix)

def parse_cvrplib(filepath: str) -> CVRPInstance:
    """Task 2(a): CVRPLIB (.vrp) dosyalarını okur ve mesafeleri tam sayıya yuvarlar."""
    name = filepath.split("/")[-1].replace(".vrp", "")
    k = int(name.split("-k")[-1]) if "-k" in name else 10
    coords, demands, section = [], [], None
    capacity = 0

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line == "EOF":
                continue
            if line.startswith("CAPACITY"):
                capacity = int(line.split(":")[1])
            elif line in ("NODE_COORD_SECTION", "DEMAND_SECTION", "DEPOT_SECTION"):
                section = line
            elif section == "NODE_COORD_SECTION":
                _, x, y = line.split()
                coords.append((float(x), float(y)))
            elif section == "DEMAND_SECTION":
                _, d = line.split()
                demands.append(int(d))

    n = len(coords) - 1
    dist_matrix = [[0.0] * (n + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            # CVRPLIB standardı: en yakın tam sayıya yuvarlama
            d = round(math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1]))
            dist_matrix[i][j] = dist_matrix[j][i] = d

    return CVRPInstance(name, n, coords, demands, capacity, k, dist_matrix)