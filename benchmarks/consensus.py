import time
import sympy as sp

from src.moment import *
from src.latex_helper import *

###############################################################################
# Case study 1: Consensus System
###############################################################################

def example_concensus_system():
    # 2D SDE:
    #   dX1_t = ( - 2 X1_t + X2_t) dt +  X1_t dW_t^(1)
    #   dX2_t = ( X1_t - 2 X2_t) dt +  X2_t dW_t^(2)
    #
    # State variables
    x1, x2 = sp.symbols('x1 x2')
    vars_ = (x1, x2)

    # Drift b(x)
    b_vec = [
        - sp.Integer(2)*x1 + x2,
         x1 - sp.Integer(2)*x2
    ]

    # Diffusion sigma(x): diagonal, independent Brownian motions
    sigma_mat = [
        [ x1, sp.Integer(0)],
        [sp.Integer(0), x2]
    ]

    # Target monomial E[x_1x_2] = m_{1,1}(t)
    alpha = (1, 1)

    # ---- Timing: closure (Algorithm 1) ----
    t0 = time.perf_counter()
    S, A_mat, c_vec = moment_closure_algorithm(b_vec, sigma_mat, alpha, vars_)
    t1 = time.perf_counter()
    closure_time = t1 - t0

    print("\n=== concensus example (monomial x1 x2) ===")
    print(f"Size of S: |S| = {len(S)}\n")
    print(f"Time for obtaining closure S: {closure_time:.6f} seconds")

    # print("Index set S (multi-indices and corresponding monomials):")
    # for s in pretty_print_S(S, vars_):
    #     print("  ", s)

    # hide A and c, since they are large and not very informative

    # print("\nA matrix (size {}x{}):".format(*A_mat.shape))
    # sp.pprint(A_mat)
    # print("\nc vector:")
    # sp.pprint(c_vec)

    t = sp.symbols('t', real=True)

    # # Initial state X_0 = (0,0)  ⇒ all moments in S start from 0
    # m0_vec = sp.zeros(len(S), 1)

    # Initial state X_0 = (1,0)
    x10 = sp.Integer(1)
    x20 = sp.Integer(0)
    m0_list = []
    for beta in S:
        # m_beta(0) = 1^{beta1} * 0^{beta2}
        val = (x10 ** beta[0]) * (x20 ** beta[1])
        m0_list.append(val)
    m0_vec = sp.Matrix(m0_list)

    # ---- Timing: solving ODE system ----
    t2 = time.perf_counter()
    m_t = solve_moment_system(A_mat, c_vec, m0_vec, t)
    t3 = time.perf_counter()
    ode_time = t3 - t2

    print(f"\nTime for solving ODE system: {ode_time:.6f} seconds")

    # m_{0,2}(t) corresponds to monomial x2^2
    idx = S.index(alpha)
    m_0_2_t = sp.simplify(m_t[idx, 0])

    # print("\nSolution m_alpha:")
    # sp.pprint(m_0_2_t)

    # # -------- LaTeX output --------
    # print("\nLaTeX: index set S")
    # print(latex_index_set(S))

    # print("\nLaTeX: ODE system for the moments")
    # print(latex_moment_ode_system(S, A_mat, c_vec))

    # print("\nLaTeX: solution for all moments in S")
    # print(latex_moment_solutions(S, m_t))

    print("\nSolution m_alpha in LaTeX: E[X_alpha]:")
    print("$" + latex_single_moment(S, m_t, alpha) + "$\n")


if __name__ == "__main__":
    example_concensus_system()
