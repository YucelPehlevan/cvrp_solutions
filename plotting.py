"""Instance plot (Milestone 1) and route plots (Task 7)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from problem import CVRPInstance, side


def _base(inst: CVRPInstance, ax):
    if inst.split_x is not None:
        ax.axvspan(0, inst.split_x, color="#dbe8f7", alpha=0.6, label="European side")
        ax.axvspan(inst.split_x, 1000, color="#f7e6d6", alpha=0.6, label="Asian side")
        ax.axvline(inst.split_x, color="#1f5fa8", lw=2, ls="--", label="Bosphorus (x = %g)" % inst.split_x)
        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 1000)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def plot_instance(inst: CVRPInstance, path: str, title: str = None):
    fig, ax = plt.subplots(figsize=(8, 8))
    _base(inst, ax)
    xs = [inst.coords[i][0] for i in range(1, inst.n + 1)]
    ys = [inst.coords[i][1] for i in range(1, inst.n + 1)]
    sizes = [10 + 4 * inst.demands[i] for i in range(1, inst.n + 1)]
    ax.scatter(xs, ys, s=sizes, c="#333333", alpha=0.75, label="Clinic (size ~ demand)")
    ax.scatter(*inst.coords[0], s=250, marker="*", c="red", zorder=5, label="Depot (central lab)")
    n_eu = sum(1 for i in range(1, inst.n + 1) if side(inst, i) == "EU")
    ax.set_title(title or "%s: n = %d (%d EU / %d AS), Q = %d, K = %d, total demand = %d"
                 % (inst.name, inst.n, n_eu, inst.n - n_eu, inst.capacity,
                    inst.max_vehicles, sum(inst.demands)), fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_solution(inst: CVRPInstance, routes, path: str, title: str = ""):
    """One colour per route; routes that cross the Bosphorus are dashed."""
    fig, ax = plt.subplots(figsize=(8, 8))
    _base(inst, ax)
    cmap = plt.get_cmap("tab20")
    for k, r in enumerate(routes):
        path_nodes = [0] + list(r) + [0]
        xs = [inst.coords[i][0] for i in path_nodes]
        ys = [inst.coords[i][1] for i in path_nodes]
        crosses = len({side(inst, i) for i in path_nodes}) > 1
        ax.plot(xs, ys, "-" if not crosses else "--", color=cmap(k % 20), lw=1.5, marker="o", ms=3)
    ax.scatter(*inst.coords[0], s=250, marker="*", c="red", zorder=5)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    from pathlib import Path
    from data_loader import generate_medidrone_instance, write_vrp

    Path("figures").mkdir(exist_ok=True)
    inst = generate_medidrone_instance()
    write_vrp(inst, "data/medidrone/MediDrone-100.vrp")
    plot_instance(inst, "figures/medidrone_instance.png")
    print("wrote figures/medidrone_instance.png and data/medidrone/MediDrone-100.vrp")
