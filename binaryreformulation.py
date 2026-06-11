import itertools
import math
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt

# ============================================================
# Basic probability helpers
# ============================================================

def normal_cdf(z):
    """
    Standard normal CDF Phi(z).
    """
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def all_binary_points(n):
    return list(itertools.product([0, 1], repeat=n))


def make_psd_matrix(n, seed=0):
    """
    Generates a positive definite matrix Q.
    """
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    Q = A.T @ A + 0.5 * np.eye(n)
    return Q


# ============================================================
# Generate a small random test instance
# ============================================================

def generate_test_instance(n=4, seed=1):
    """
    Generates arbitrary data for the two-component Gaussian mixture model.

    We use:
        alpha_1 Phi(z_1) + alpha_2 Phi(z_2) >= theta

    where:
        z_i = (b_i - mu_i^T x) / sqrt(x^T Q_i x).
    """
    rng = np.random.default_rng(seed)

    c = rng.uniform(0.5, 2.0, size=n)

    Q1 = make_psd_matrix(n, seed=seed + 10)
    Q2 = make_psd_matrix(n, seed=seed + 20)

    mu1 = rng.uniform(0.1, 1.0, size=n)
    mu2 = rng.uniform(0.1, 1.0, size=n)

    # Pick b values so that the probability constraint is neither trivial nor impossible.
    b1 = rng.uniform(1.0, 3.0)
    b2 = rng.uniform(1.0, 3.0)

    alpha1 = 0.55
    alpha2 = 0.45

    # Choose theta based on the distribution of true probabilities over binary points.
    probs = []

    for x_tuple in all_binary_points(n):
        x = np.array(x_tuple, dtype=float)

        val1 = x @ Q1 @ x
        val2 = x @ Q2 @ x

        if val1 <= 1e-9 or val2 <= 1e-9:
            continue

        z1 = (b1 - mu1 @ x) / math.sqrt(val1)
        z2 = (b2 - mu2 @ x) / math.sqrt(val2)

        prob = alpha1 * normal_cdf(z1) + alpha2 * normal_cdf(z2)
        probs.append(prob)

    theta = float(np.quantile(probs, 0.75))

    data = {
        "n": n,
        "c": c,
        "Q": [Q1, Q2],
        "mu": [mu1, mu2],
        "b": [b1, b2],
        "alpha": [alpha1, alpha2],
        "theta": theta
    }

    return data


# ============================================================
# Original model solved by enumeration
# ============================================================

def solve_original_by_enumeration(data):
    """
    Solves the original formulation exactly by enumerating x in {0,1}^n.

    Original model:
        min c^T x
        s.t. alpha_1 Phi(z_1) + alpha_2 Phi(z_2) >= theta
             b_i - mu_i^T x = z_i sqrt(x^T Q_i x), i=1,2
             x binary

    For fixed binary x, z_i is determined:
        z_i = (b_i - mu_i^T x) / sqrt(x^T Q_i x).
    """
    n = data["n"]
    c = data["c"]
    Q = data["Q"]
    mu = data["mu"]
    b = data["b"]
    alpha = data["alpha"]
    theta = data["theta"]

    best_x = None
    best_obj = float("inf")
    best_info = None

    feasible_points = []

    for x_tuple in all_binary_points(n):
        x = np.array(x_tuple, dtype=float)

        # Avoid denominator zero.
        denom = []
        valid = True

        for i in range(2):
            val = x @ Q[i] @ x

            if val <= 1e-9:
                valid = False
                break

            denom.append(math.sqrt(val))

        if not valid:
            continue

        z = []
        phi = []

        for i in range(2):
            zi = (b[i] - mu[i] @ x) / denom[i]
            z.append(zi)
            phi.append(normal_cdf(zi))

        mixture_prob = alpha[0] * phi[0] + alpha[1] * phi[1]
        obj = c @ x

        if mixture_prob >= theta - 1e-9:
            feasible_points.append({
                "x": x.copy(),
                "obj": obj,
                "z": z,
                "phi": phi,
                "mixture_prob": mixture_prob
            })

            if obj < best_obj:
                best_obj = obj
                best_x = x.copy()
                best_info = feasible_points[-1]

    return best_x, best_obj, best_info, feasible_points


# ============================================================
# Extended binary-expansion MILP
# ============================================================

def solve_binary_expansion_milp(
    data,
    K=1,
    phi_grid_size=81,
    eq_tol=0.05,
    time_limit=60
):
    """
    Solves the binary-expansion extended formulation.

    Important practical note:
    The equality involving z_i^2 x^T Q_i x is very restrictive when z_i
    is represented on a finite binary grid. Therefore, we allow a small
    tolerance eq_tol:

        |left_i - right_i| <= eq_tol.

    This makes the model usable as a finite approximation.

    z_i is represented as:
        z_i = sum_{k=-K}^{K} 2^k (a_ik - b_ik).

    Phi(z_i) is handled by a piecewise-linear approximation.
    """
    n = data["n"]
    c = data["c"]
    Q = data["Q"]
    mu = data["mu"]
    b = data["b"]
    alpha_mix = data["alpha"]
    theta = data["theta"]

    bit_range = list(range(-K, K + 1))
    pow2 = {k: 2.0 ** k for k in bit_range}

    z_max = sum(pow2[k] for k in bit_range)
    z_min = -z_max

    m = gp.Model("binary_expansion_extended_formulation")
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = time_limit

    # ------------------------------------------------------------
    # Main binary decision variables
    # ------------------------------------------------------------
    x = m.addVars(n, vtype=GRB.BINARY, name="x")

    # y_jl = x_j x_l
    y = {}
    for j in range(n):
        for ell in range(j + 1, n):
            y[j, ell] = m.addVar(vtype=GRB.BINARY, name=f"y_{j}_{ell}")

    # ------------------------------------------------------------
    # Binary expansion variables for z_i
    # ------------------------------------------------------------
    z = m.addVars(2, lb=z_min, ub=z_max, vtype=GRB.CONTINUOUS, name="z")
    p = m.addVars(2, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="Phi")

    s = m.addVars(2, vtype=GRB.BINARY, name="s")

    a = {}
    bbit = {}

    for i in range(2):
        for k in bit_range:
            a[i, k] = m.addVar(vtype=GRB.BINARY, name=f"a_{i}_{k}")
            bbit[i, k] = m.addVar(vtype=GRB.BINARY, name=f"b_{i}_{k}")

    # alpha_ijk = a_ik x_j
    # beta_ijk  = b_ik x_j
    alpha_var = {}
    beta_var = {}

    for i in range(2):
        for j in range(n):
            for k in bit_range:
                alpha_var[i, j, k] = m.addVar(vtype=GRB.BINARY, name=f"alpha_{i}_{j}_{k}")
                beta_var[i, j, k] = m.addVar(vtype=GRB.BINARY, name=f"beta_{i}_{j}_{k}")

    # A_ijellkm = alpha_ijk alpha_iellm
    # D_ijellkm = beta_ijk beta_iellm
    A = {}
    D = {}

    for i in range(2):
        for j in range(n):
            for ell in range(n):
                for k in bit_range:
                    for mm in bit_range:
                        A[i, j, ell, k, mm] = m.addVar(
                            vtype=GRB.BINARY,
                            name=f"A_{i}_{j}_{ell}_{k}_{mm}"
                        )
                        D[i, j, ell, k, mm] = m.addVar(
                            vtype=GRB.BINARY,
                            name=f"D_{i}_{j}_{ell}_{k}_{mm}"
                        )

    # ------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------
    m.setObjective(gp.quicksum(c[j] * x[j] for j in range(n)), GRB.MINIMIZE)

    # ------------------------------------------------------------
    # Piecewise-linear approximation of Phi(z)
    # ------------------------------------------------------------
    z_grid = np.linspace(z_min, z_max, phi_grid_size)
    phi_grid = [normal_cdf(v) for v in z_grid]

    for i in range(2):
        m.addGenConstrPWL(
            z[i],
            p[i],
            z_grid.tolist(),
            phi_grid,
            name=f"pwl_phi_{i}"
        )

    # Mixture chance constraint
    m.addConstr(
        alpha_mix[0] * p[0] + alpha_mix[1] * p[1] >= theta,
        name="mixture_chance_constraint"
    )

    # ------------------------------------------------------------
    # z_i binary expansion and sign logic
    # ------------------------------------------------------------
    for i in range(2):
        m.addConstr(
            z[i] == gp.quicksum(pow2[k] * (a[i, k] - bbit[i, k]) for k in bit_range),
            name=f"z_expansion_{i}"
        )

        for k in bit_range:
            # If s_i = 1, positive bits may turn on and negative bits are off.
            # If s_i = 0, negative bits may turn on and positive bits are off.
            m.addConstr(a[i, k] <= s[i], name=f"a_sign_{i}_{k}")
            m.addConstr(bbit[i, k] <= 1 - s[i], name=f"b_sign_{i}_{k}")

    # ------------------------------------------------------------
    # McCormick constraints for y_jl = x_j x_l
    # ------------------------------------------------------------
    for j in range(n):
        for ell in range(j + 1, n):
            m.addConstr(y[j, ell] <= x[j])
            m.addConstr(y[j, ell] <= x[ell])
            m.addConstr(y[j, ell] >= x[j] + x[ell] - 1)

    # ------------------------------------------------------------
    # McCormick constraints for alpha_ijk = a_ik x_j
    # and beta_ijk = b_ik x_j
    # ------------------------------------------------------------
    for i in range(2):
        for j in range(n):
            for k in bit_range:
                m.addConstr(alpha_var[i, j, k] <= a[i, k])
                m.addConstr(alpha_var[i, j, k] <= x[j])
                m.addConstr(alpha_var[i, j, k] >= a[i, k] + x[j] - 1)

                m.addConstr(beta_var[i, j, k] <= bbit[i, k])
                m.addConstr(beta_var[i, j, k] <= x[j])
                m.addConstr(beta_var[i, j, k] >= bbit[i, k] + x[j] - 1)

    # ------------------------------------------------------------
    # McCormick constraints for A and D variables
    # ------------------------------------------------------------
    for i in range(2):
        for j in range(n):
            for ell in range(n):
                for k in bit_range:
                    for mm in bit_range:
                        m.addConstr(A[i, j, ell, k, mm] <= alpha_var[i, j, k])
                        m.addConstr(A[i, j, ell, k, mm] <= alpha_var[i, ell, mm])
                        m.addConstr(
                            A[i, j, ell, k, mm]
                            >= alpha_var[i, j, k] + alpha_var[i, ell, mm] - 1
                        )

                        m.addConstr(D[i, j, ell, k, mm] <= beta_var[i, j, k])
                        m.addConstr(D[i, j, ell, k, mm] <= beta_var[i, ell, mm])
                        m.addConstr(
                            D[i, j, ell, k, mm]
                            >= beta_var[i, j, k] + beta_var[i, ell, mm] - 1
                        )

    # ------------------------------------------------------------
    # Main squared equality approximation
    # ------------------------------------------------------------
    for i in range(2):
        # Left-hand side:
        # (b_i - mu_i^T x)^2 linearized using x_j^2 = x_j and y_jl = x_j x_l
        lhs = b[i] ** 2
        lhs += gp.quicksum(-2 * b[i] * mu[i][j] * x[j] for j in range(n))
        lhs += gp.quicksum(mu[i][j] ** 2 * x[j] for j in range(n))
        lhs += gp.quicksum(
            2 * mu[i][j] * mu[i][ell] * y[j, ell]
            for j in range(n)
            for ell in range(j + 1, n)
        )

        # Right-hand side:
        # z_i^2 x^T Q_i x using binary-expanded z_i and products.
        rhs = gp.quicksum(
            Q[i][j, ell]
            * pow2[k]
            * pow2[mm]
            * (A[i, j, ell, k, mm] + D[i, j, ell, k, mm])
            for j in range(n)
            for ell in range(n)
            for k in bit_range
            for mm in bit_range
        )

        # Use a tolerance because finite binary expansion may not match equality exactly.
        m.addConstr(lhs - rhs <= eq_tol, name=f"squared_eq_upper_{i}")
        m.addConstr(rhs - lhs <= eq_tol, name=f"squared_eq_lower_{i}")

    # ------------------------------------------------------------
    # Sign consistency constraint:
    # z_i (b_i - mu_i^T x) >= 0
    # ------------------------------------------------------------
    for i in range(2):
        sign_expr = b[i] * gp.quicksum(pow2[k] * (a[i, k] - bbit[i, k]) for k in bit_range)

        sign_expr -= gp.quicksum(
            mu[i][j] * pow2[k] * (alpha_var[i, j, k] - beta_var[i, j, k])
            for j in range(n)
            for k in bit_range
        )

        m.addConstr(sign_expr >= 0, name=f"sign_consistency_{i}")

    # ------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------
    m.optimize()

    if m.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        return None

    x_sol = np.array([x[j].X for j in range(n)])
    z_sol = np.array([z[i].X for i in range(2)])
    p_sol = np.array([p[i].X for i in range(2)])

    obj = m.ObjVal

    return {
        "status": m.Status,
        "objective": obj,
        "x": x_sol,
        "z": z_sol,
        "phi_approx": p_sol,
        "mixture_prob_approx": alpha_mix[0] * p_sol[0] + alpha_mix[1] * p_sol[1],
        "runtime": m.Runtime,
        "gap": m.MIPGap if m.Status == GRB.TIME_LIMIT else 0.0
    }


# ============================================================
# Evaluate original probability for any x
# ============================================================

def evaluate_true_probability(data, x):
    """
    Given an x, compute the true z_i and true mixture probability.
    """
    Q = data["Q"]
    mu = data["mu"]
    b = data["b"]
    alpha = data["alpha"]

    z_true = []
    phi_true = []

    for i in range(2):
        denom_sq = x @ Q[i] @ x

        if denom_sq <= 1e-9:
            return None

        zi = (b[i] - mu[i] @ x) / math.sqrt(denom_sq)
        z_true.append(zi)
        phi_true.append(normal_cdf(zi))

    mixture_prob = alpha[0] * phi_true[0] + alpha[1] * phi_true[1]

    return {
        "z_true": np.array(z_true),
        "phi_true": np.array(phi_true),
        "mixture_prob_true": mixture_prob
    }


# ============================================================
# Run comparison
# ============================================================

def run_comparison(n=4, seed=1, K=1, eq_tol=0.05):
    data = generate_test_instance(n=n, seed=seed)

    print("=" * 80)
    print("TEST INSTANCE")
    print("=" * 80)

    print("n =", data["n"])
    print("c =", np.round(data["c"], 4))
    print("b =", np.round(data["b"], 4))
    print("alpha =", data["alpha"])
    print("theta =", round(data["theta"], 6))
    print("lambda_min(Q1) =", round(np.linalg.eigvalsh(data["Q"][0]).min(), 6))
    print("lambda_min(Q2) =", round(np.linalg.eigvalsh(data["Q"][1]).min(), 6))

    print("\n" + "=" * 80)
    print("SOLVING ORIGINAL MODEL BY ENUMERATION")
    print("=" * 80)

    x_orig, obj_orig, info_orig, feasible_points = solve_original_by_enumeration(data)

    print("number of feasible binary points =", len(feasible_points))
    print("original x* =", x_orig)
    print("original objective =", obj_orig)

    if info_orig is not None:
        print("original z =", np.round(info_orig["z"], 6))
        print("original Phi(z) =", np.round(info_orig["phi"], 6))
        print("original mixture probability =", round(info_orig["mixture_prob"], 6))

    print("\n" + "=" * 80)
    print("SOLVING EXTENDED BINARY-EXPANSION MILP")
    print("=" * 80)

    ext = solve_binary_expansion_milp(
        data,
        K=K,
        phi_grid_size=101,
        eq_tol=eq_tol,
        time_limit=60
    )

    if ext is None:
        print("Extended MILP was infeasible or did not solve.")
        return data, None

    print("extended x =", np.round(ext["x"], 6))
    print("extended objective =", round(ext["objective"], 6))
    print("extended z expansion =", np.round(ext["z"], 6))
    print("extended Phi approximation =", np.round(ext["phi_approx"], 6))
    print("extended mixture probability approximation =", round(ext["mixture_prob_approx"], 6))
    print("runtime =", round(ext["runtime"], 4))

    true_eval = evaluate_true_probability(data, ext["x"])

    print("\nTrue evaluation of extended solution:")
    if true_eval is not None:
        print("true z =", np.round(true_eval["z_true"], 6))
        print("true Phi(z) =", np.round(true_eval["phi_true"], 6))
        print("true mixture probability =", round(true_eval["mixture_prob_true"], 6))
        print("theta =", round(data["theta"], 6))
        print("true feasible for original chance constraint? =",
              true_eval["mixture_prob_true"] >= data["theta"] - 1e-9)

    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    print("original objective =", obj_orig)
    print("extended objective =", ext["objective"])

    if x_orig is not None:
        print("same x solution? =", np.allclose(x_orig, ext["x"], atol=1e-6))

    return data, ext


# ============================================================
# Example run
# ============================================================

data, ext = run_comparison(
    n=4,
    seed=4,
    K=3,
    eq_tol=0.10
)

# ============================================================
# Multi-instance experiment: original vs binary-expansion MILP
# ============================================================

def run_many_comparisons(
    num_instances=100,
    n=4,
    seed_start=100,
    K=3,
    eq_tol=0.10,
    phi_grid_size=101,
    time_limit=30
):
    """
    Runs many random 4D instances and compares:

        1. Original formulation solved by enumeration.
        2. Binary-expansion extended MILP.

    Returns a list of dictionaries with solution information.
    """

    results = []

    for t in range(num_instances):
        seed = seed_start + t
        data = generate_test_instance(n=n, seed=seed)

        print("\n" + "=" * 80)
        print(f"INSTANCE {t + 1}/{num_instances}, seed={seed}")
        print("=" * 80)

        # Original exact solution by enumeration
        x_orig, obj_orig, info_orig, feasible_points = solve_original_by_enumeration(data)

        if x_orig is None:
            print("Original model has no feasible binary point. Skipping.")
            results.append({
                "instance": t + 1,
                "seed": seed,
                "status": "original_infeasible",
                "obj_orig": None,
                "obj_ext": None,
                "gap": None
            })
            continue

        # Extended binary expansion MILP
        ext = solve_binary_expansion_milp(
            data,
            K=K,
            phi_grid_size=phi_grid_size,
            eq_tol=eq_tol,
            time_limit=time_limit
        )

        if ext is None:
            print("Extended MILP infeasible or failed.")
            results.append({
                "instance": t + 1,
                "seed": seed,
                "status": "extended_failed",
                "obj_orig": obj_orig,
                "obj_ext": None,
                "gap": None,
                "x_orig": x_orig,
                "x_ext": None
            })
            continue

        # True evaluation of extended solution
        true_eval_ext = evaluate_true_probability(data, ext["x"])

        obj_ext = ext["objective"]
        abs_gap = obj_ext - obj_orig
        rel_gap = abs_gap / abs(obj_orig) if abs(obj_orig) > 1e-9 else None

        same_x = np.allclose(x_orig, ext["x"], atol=1e-6)

        true_feasible_ext = False
        if true_eval_ext is not None:
            true_feasible_ext = (
                true_eval_ext["mixture_prob_true"] >= data["theta"] - 1e-9
            )

        print("original objective =", round(obj_orig, 6))
        print("extended objective =", round(obj_ext, 6))
        print("absolute gap ext - orig =", round(abs_gap, 6))
        print("same x? =", same_x)
        print("extended true feasible? =", true_feasible_ext)

        results.append({
            "instance": t + 1,
            "seed": seed,
            "status": "solved",
            "obj_orig": obj_orig,
            "obj_ext": obj_ext,
            "abs_gap": abs_gap,
            "rel_gap": rel_gap,
            "x_orig": x_orig,
            "x_ext": ext["x"],
            "same_x": same_x,
            "extended_true_feasible": true_feasible_ext,
            "theta": data["theta"],
            "true_prob_ext": None if true_eval_ext is None else true_eval_ext["mixture_prob_true"],
            "runtime_ext": ext["runtime"]
        })

    return results


def summarize_many_results(results):
    """
    Prints a compact summary of the multi-instance experiment.
    """

    solved = [r for r in results if r["status"] == "solved"]
    failed = [r for r in results if r["status"] != "solved"]

    print("\n" + "=" * 80)
    print("MULTI-INSTANCE SUMMARY")
    print("=" * 80)

    print("total instances =", len(results))
    print("solved instances =", len(solved))
    print("failed/skipped instances =", len(failed))

    if len(solved) == 0:
        return

    gaps = np.array([r["abs_gap"] for r in solved])
    rel_gaps = np.array([
        r["rel_gap"] for r in solved
        if r["rel_gap"] is not None
    ])

    same_x_count = sum(r["same_x"] for r in solved)
    true_feasible_count = sum(r["extended_true_feasible"] for r in solved)

    print("\nObjective gap statistics: extended - original")
    print("min gap =", round(np.min(gaps), 6))
    print("mean gap =", round(np.mean(gaps), 6))
    print("median gap =", round(np.median(gaps), 6))
    print("max gap =", round(np.max(gaps), 6))

    if len(rel_gaps) > 0:
        print("\nRelative gap statistics")
        print("mean relative gap =", round(np.mean(rel_gaps), 6))
        print("median relative gap =", round(np.median(rel_gaps), 6))

    print("\nSolution agreement")
    print("same x count =", same_x_count, "out of", len(solved))
    print("same x percentage =", round(100 * same_x_count / len(solved), 2), "%")

    print("\nFeasibility check of extended solution in original model")
    print("true feasible count =", true_feasible_count, "out of", len(solved))
    print("true feasible percentage =", round(100 * true_feasible_count / len(solved), 2), "%")


def plot_original_vs_extended(results):
    """
    Plots original objective values in red and extended MILP objective values in blue.
    """

    solved = [r for r in results if r["status"] == "solved"]

    if len(solved) == 0:
        print("No solved instances to plot.")
        return

    idx = np.array([r["instance"] for r in solved])
    obj_orig = np.array([r["obj_orig"] for r in solved])
    obj_ext = np.array([r["obj_ext"] for r in solved])

    plt.figure(figsize=(10, 5))

    plt.plot(
        idx,
        obj_orig,
        marker="o",
        linewidth=1.5,
        color="red",
        label="Original exact optimum"
    )

    plt.plot(
        idx,
        obj_ext,
        marker="o",
        linewidth=1.5,
        color="blue",
        label="Binary-expansion MILP optimum"
    )

    plt.xlabel("Instance")
    plt.ylabel("Objective value")
    plt.title("Original formulation vs binary-expansion reformulation")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

# ============================================================
# Run the multi-instance experiment
# ============================================================

def plot_original_vs_extended(results, eq_tol=None, K=None):
    solved = [r for r in results if r["status"] == "solved"]

    if len(solved) == 0:
        print("No solved instances to plot.")
        return

    idx = np.array([r["instance"] for r in solved])
    obj_orig = np.array([r["obj_orig"] for r in solved])
    obj_ext = np.array([r["obj_ext"] for r in solved])

    title = "Original vs Binary-Expansion MILP Objective Values"
    if eq_tol is not None and K is not None:
        title += f" (K={K}, equality tolerance={eq_tol})"

    plt.figure(figsize=(11, 5.5))

    plt.plot(
        idx,
        obj_orig,
        marker="o",
        markersize=4,
        linewidth=1.5,
        color="red",
        label="Original formulation, exact enumeration"
    )

    plt.plot(
        idx,
        obj_ext,
        marker="s",
        markersize=4,
        linewidth=1.5,
        color="blue",
        label="Binary-expansion MILP approximation"
    )

    plt.xlabel("Random instance number")
    plt.ylabel("Optimal objective value")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

for tol in [1.0, 0.25, 0.10]:
    print("\n\n" + "#" * 80)
    print("RUNNING WITH eq_tol =", tol)
    print("#" * 80)

    many_results = run_many_comparisons(
        num_instances=100,
        n=4,
        seed_start=4,
        K=3,
        eq_tol=tol,
        phi_grid_size=101,
        time_limit=30
    )

    summarize_many_results(many_results)

    plot_original_vs_extended(
        many_results,
        eq_tol=tol,
        K=3
    )