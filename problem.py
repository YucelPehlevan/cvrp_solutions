from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class CVRPInstance:
    name: str
    n: int                          # Klinik sayısı (1..n arası, 0 depo)
    coords: List[Tuple[float, float]]
    demands: List[int]              # demands[0] = 0
    capacity: int                   # Drone kapasitesi (Q)
    max_vehicles: int               # Maksimum drone sayısı (K)
    dist_matrix: List[List[float]]  # (n+1) x (n+1) mesafe tablosu

def calculate_cost(instance: CVRPInstance, routes: List[List[int]]) -> float:
    """Rotaların toplam uçuş mesafesini hesaplar (0'dan çıkar, 0'a döner)."""
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

def is_feasible(instance: CVRPInstance, routes: List[List[int]]) -> Tuple[bool, str]:
    """Çözümün tüm CVRP kurallarına uyup uymadığını kontrol eder."""
    non_empty_routes = [r for r in routes if len(r) > 0]
    if len(non_empty_routes) > instance.max_vehicles:
        return False, f"Araç sınırı aşıldı: {len(non_empty_routes)} > {instance.max_vehicles}"

    visited = []
    for idx, route in enumerate(non_empty_routes):
        load = sum(instance.demands[node] for node in route)
        if load > instance.capacity:
            return False, f"Rota {idx} kapasiteyi aştı: {load} > {instance.capacity}"
        visited.extend(route)

    if len(visited) != instance.n or set(visited) != set(range(1, instance.n + 1)):
        return False, "Her kliniğe tam olarak 1 kez uğranmamış!"

    return True, "OK"