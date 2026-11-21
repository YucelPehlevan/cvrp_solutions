# Task 1 – Corrections to the theoretical PDF

The uploaded Task 1 PDF ("CMP3005 Task 1 Theoretical Foundations.pdf") has two errors and needs
two short additions so that it matches the code and the project description. The replacement
text below is ready to paste into the report.

## 1. Section 4 – Bin Packing reduction (ERROR: d = 0 contradicts the problem definition)

**Problem.** Section 1 defines d : E → ℝ⁺ (strictly positive distances), but the reduction sets
d(u, v) = 0 and B = 0, so the constructed instance is not a valid CVRP instance.

**Replacement text:**

> **Reduction (Bin Packing ≤ₚ CVRP feasibility).** A Bin Packing instance gives n items with
> integer sizes s₁, …, sₙ and K bins of capacity C. We build a CVRP instance with one customer per
> item, demand qᵢ = sᵢ, capacity Q = C, fleet size K, unit distances d(u, v) = 1 for all u ≠ v
> (a valid metric: 1 ≤ 1 + 1), and bound B = n + K. The construction takes O(n²) time.
> A route that serves k customers has cost exactly k + 1, so a solution with m routes costs
> n + m. Hence a feasible solution with cost ≤ B exists ⇔ the customers can be split into
> m ≤ K groups of load ≤ Q ⇔ the items fit into K bins. The distance bound is always satisfied
> when the fleet bound is, so the question is purely the packing question.
> Bin Packing is **strongly** NP-complete, so the feasibility question stays NP-complete even
> when all demands are small integers (as in our instances, qᵢ ≤ 30).

## 2. Section 6 – Bonus approximation ratio (ERROR: wrong ratio for our problem)

**Problem.** 1 + (1 − 1/Q)α holds for **unit (or splittable)** demands. Our demands are
unequal and **unsplittable** (each clinic is served by exactly one route), where the known
ITP guarantee is 2 + (1 − 2/Q)α (Altinkemer & Gavish, 1987).

**Replacement text (last part of Section 6):**

> For unit demands, ITP with an α-approximate TSP tour gives a ratio of 1 + (1 − 1/Q)α
> (Haimovich & Rinnooy Kan, 1985; Altinkemer & Gavish, 1990). In our setting demands are unequal
> and may not be split between drones; for this case Altinkemer and Gavish (1987) showed that the
> optimal-partition variant of ITP achieves 2 + (1 − 2/Q)α. With Christofides (α = 1.5) and
> Q = 100 this is 2 + 0.98 · 1.5 ≈ **3.47**, i.e. a worst-case cost up to about 247% above optimal.
> Such a guarantee is far too weak for daily drone operations, which is why practical systems use
> metaheuristics that typically reach solutions within a few percent of optimal in seconds.

## 3. Addition to Section 5 – CVRPLIB rounding (consistency with the code)

The metric proof is correct for our MediDrone distances (the test suite checks all triples:
0 violations). The CVRPLIB benchmarks, however, use distances rounded to the nearest integer,
which can break the triangle inequality. Add after the proof:

> **Remark (benchmarks).** CVRPLIB rounds every distance to the nearest integer. Rounding can
> violate the triangle inequality by at most 1 unit (we found 66–1194 violating triples in the
> four Augerat instances). Our algorithms never rely on the triangle inequality, so this does not
> affect them; it only matters for theoretical guarantees such as the ITP bound above.

## 4. Section 2 – link the verifier to the code (addition)

> Our feasibility checker `problem.verify_solution` implements exactly this verifier: it checks
> the fleet size, valid customer ids, single visits (counter array), capacities and recomputes the
> total distance from the coordinates, all in O(n) time.

## 5. Section 4 – "Implications for metaheuristics" (check with the GA owner)

The text promises "λ · overload in Tabu Search" (implemented in `tabu_search.py`) and
"route-count penalties in the GA Split decoder". The GA must implement such a penalty (or a
Split that limits the number of routes to K); otherwise this sentence must be changed.
