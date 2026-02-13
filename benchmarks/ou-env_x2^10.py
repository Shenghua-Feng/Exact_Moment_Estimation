import time
import sympy as sp

from src.moment import *
from src.latex_helper import *

###############################################################################
# OU-env Example
###############################################################################

def example_OUenv():
    # SDE in 2D:
    #   dX_t = -X_t dt + dW_t^(1)
    #   dY_t = (-2 Y_t + X_t + X_t^2) dt + X_t dW_t^(2)
    #
    # State variables: x1 = X, x2 = Y
    x1, x2 = sp.symbols('x1 x2')
    vars_ = (x1, x2)

    b_vec = [
        -x1,
        -sp.Integer(2)*x2 + x1 + x1**2
    ]

    sigma_mat = [
        [sp.Integer(1), 0],
        [0, x1]
    ]

    # target multi-index
    # alpha = (0, 2)
    # alpha = (0, 3)
    # alpha = (0, 4)
    # alpha = (0, 5)
    alpha = (0, 10)

    # ---- Timing: closure (Algorithm 1) ----
    t0 = time.perf_counter()
    S, A_mat, c_vec = moment_closure_algorithm(b_vec, sigma_mat, alpha, vars_)
    t1 = time.perf_counter()
    closure_time = t1 - t0

    print("\n=== OU-env example (moment x2^10) ===")
    print(f"Size of S: |S| = {len(S)}\n")
    print(f"Time for obtaining closure S: {closure_time:.6f} seconds")

    # print("Index set S (multi-indices and corresponding monomials):")
    # for s in pretty_print_S(S, vars_):
    #     print("  ", s)

    # print("\nA matrix (size {}x{}):".format(*A_mat.shape))
    # sp.pprint(A_mat)
    # print("\nc vector:")
    # sp.pprint(c_vec)

    t = sp.symbols('t', real=True)

    # Initial condition (X_0, Y_0) = (0, 0) ⇒ all moments in S are 0 at t=0
    m0_vec = sp.zeros(len(S), 1)

    # ---- Timing: solving ODE system ----
    t2 = time.perf_counter()
    m_t = solve_moment_system(A_mat, c_vec, m0_vec, t)
    t3 = time.perf_counter()
    ode_time = t3 - t2

    print(f"\nTime for solving ODE system: {ode_time:.6f} seconds")

    # target moment
    idx = S.index(alpha)
    m_alpha = sp.simplify(m_t[idx, 0])

    # print("\nSolution m_{alpha}(t) = E[X^alpha]:")
    # sp.pprint(m_alpha)

    # -------- LaTeX output --------
    # print("\nLaTeX: index set S")
    # print(latex_index_set(S))

    # print("\nLaTeX: ODE system for the moments")
    # print(latex_moment_ode_system(S, A_mat, c_vec))

    # print("\nLaTeX: solution for all moments in S")
    # print(latex_moment_solutions(S, m_t))

    print("\nSolution m_alpha in LaTeX: E[X^alpha]")
    print("$" + latex_single_moment(S, m_t, alpha) + "$\n")


if __name__ == "__main__":
    example_OUenv()
