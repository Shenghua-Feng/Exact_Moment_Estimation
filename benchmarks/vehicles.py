import time
import sympy as sp

from src.moment import *
from src.latex_helper import *

###############################################################################
# Vehicle platoon
###############################################################################

def example_vehicle_platoon_simple():
    # State variables ordered as (p1, v1, p2, v2)
    p1, v1, p2, v2 = sp.symbols('p1 v1 p2 v2')
    vars_ = (p1, v1, p2, v2)

    # Parameters (instantiated as in the text)
    d_des = 0

    # Spacing error s = (p1 - p2)
    s = p1 - p2

    # Drift b(x):
    #   dp1 = v1 dt
    #   dv1 = (-a1 v1 + u1) dt + sigma1 dW^1, with a1=1, u1=1, sigma1=1
    #   dp2 = v2 dt
    #   dv2 = (-a2 v2 + (v1 - 1)^2) dt + sigma2 dW^2, a2=1, k=1, sigma2=1
    b_vec = [
        v1,                           # dp1
        -v1 + 1,                      # dv1
        v2,                           # dp2
        -v2 + sp.Rational(1, 2) + (v1 - 1)**2 ,     # dv2
        # -v2 + s + (v1 - 1)**2
    ]

    # Diffusion matrix sigma(x) (4×2):
    # noise only in v1 (W^1) and v2 (W^2), both with coefficient 1
    sigma_mat = [
        [0, 0],   # dp1
        [0, 0],   # dv1
        [0, 0],   # dp2
        [0, sp.Rational(1, 10)],   # dv2
    ]

    # Initial state (p1, v1, p2, v2) = (1, 0, 0, 0)
    p1_0, v1_0, p2_0, v2_0 = 1, 0, 0, 0

    # We want moments of p1 and p2:
    targets = [
        # ((1, 0, 0, 0), "x_1"),  # monomial p1
        # ((0, 0, 1, 0), "x_2"),  # monomial p2
        # ((2, 0, 0, 0), "x_1^2"),  # monomial p1^2
        ((0, 0, 2, 0), "x_2^2"),  # monomial p2^2
        # ((1, 0, 1, 0), "x_1x_2"),  # monomial p1p2
    ]

    t = sp.symbols('t', real=True)

    for alpha, label in targets:
        # --------- closure S (Algorithm 1) ----------
        t0 = time.perf_counter()
        S, A_mat, c_vec = moment_closure_algorithm(b_vec, sigma_mat, alpha, vars_)
        t1 = time.perf_counter()

        print(f"\n=== vehicles example ({label}) ===")
        print(f"Target monomial: {label}\n")
        print(f"Size of S: |S| = {len(S)}\n")
        print(f"Time for obtaining closure S: {t1 - t0:.6f} seconds")

        # print("Index set S (multi-indices and corresponding monomials):")
        # for s_str in pretty_print_S(S, vars_):
        #     print("  ", s_str)

        # print("\nA matrix (size {}x{}):".format(*A_mat.shape))
        # sp.pprint(A_mat)
        # print("\nc vector:")
        # sp.pprint(c_vec)

        # Initial moment vector from deterministic initial state
        m0_list = []
        for beta in S:
            val = (p1_0 ** beta[0]) * (v1_0 ** beta[1]) \
                  * (p2_0 ** beta[2]) * (v2_0 ** beta[3])
            m0_list.append(val)
        m0_vec = sp.Matrix(m0_list)

        # --------- solve ODE system ----------
        t2 = time.perf_counter()
        m_t = solve_moment_system(A_mat, c_vec, m0_vec, t)
        t3 = time.perf_counter()
        print(f"\nTime for solving ODE system: {t3 - t2:.6f} seconds")

        # Extract desired moment E[p1(t)] or E[p2(t)]
        idx = S.index(alpha)
        moment_expr = sp.simplify(m_t[idx, 0])

        # print(f"\nSolution for E[{label}(t)]:")
        # sp.pprint(moment_expr)

        # # --------- optional LaTeX output ----------
        # print("\nLaTeX: index set S")
        # print(latex_index_set(S))

        # print("\nLaTeX: ODE system")
        # print(latex_moment_ode_system(S, A_mat, c_vec))

        # print("\nLaTeX: solution for all moments in S")
        # print(latex_moment_solutions(S, m_t))

        print(f"\nSolution m_alpha in LaTeX: E[{label}(t)]:")
        print("$" + latex_single_moment(S, m_t, alpha) + "$\n")


if __name__ == "__main__":
    example_vehicle_platoon_simple()
