import time
import sympy as sp

from src.moment import *
from src.latex_helper import *

###############################################################################
# Gene example
###############################################################################

def example_gene_expression_network():
    # 5D gene regulatory network SDE (from the figure in the paper)
    x1, x2, x3, x4, x5 = sp.symbols('x1 x2 x3 x4 x5')
    vars_ = (x1, x2, x3, x4, x5)

    # Drift b(x)
    b_vec = [
        -x1 + 1,
        sp.Rational(12, 10)* x1 - sp.Rational(8, 10) * x2,
        x2 - sp.Rational(7, 10) * x3 + sp.Rational(2, 10) * x1**2,
        sp.Rational(9, 10) * x3 - sp.Rational(6, 10) * x4 + sp.Rational(1, 10) * x1 * x2,
        sp.Rational(8, 10)* x4 - sp.Rational(5, 10) * x5 + sp.Rational(15, 100) * x3**2 + sp.Rational(5, 100) * x1**3,
    ]

    # Diffusion sigma(x): diagonal, one independent Brownian motion per component
    sigma_mat = [
        [sp.Rational(1, 2), sp.Integer(0), sp.Integer(0), sp.Integer(0), sp.Integer(0)],
        [sp.Integer(0), sp.Rational(3, 10) * x1 +sp.Rational(2, 5), 0, 0, 0],
        [sp.Integer(0),sp.Integer(0), sp.Rational(1, 2) * x2 + sp.Rational(1, 10) * x1**2, sp.Integer(0), sp.Integer(0)],
        [sp.Integer(0), sp.Integer(0), sp.Integer(0), sp.Rational(2, 5) * x3 +sp.Rational(1, 5)* x2**2, sp.Integer(0)],
        [sp.Integer(0), sp.Integer(0),sp.Integer(0), sp.Integer(0), sp.Rational(3, 10) * x4 + sp.Rational(1, 10) * x3**2 + sp.Rational(5, 100) * x1**3],
    ]

    # Target moment
    alpha = (1, 0, 0, 0, 1)
    # alpha = (0, 0, 0, 0, 2)
    # alpha = (1, 0, 0, 0, 2)

    # ---- Timing: closure (Algorithm 1) ----
    t0 = time.perf_counter()
    S, A_mat, c_vec = moment_closure_algorithm(b_vec, sigma_mat, alpha, vars_)
    t1 = time.perf_counter()
    closure_time = t1 - t0

    print("\n=== Gene example (moment x1 x5) ===")
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

    # Initial state X_0 = 0 => all moments in S start from 0
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

    # # -------- LaTeX output --------
    # print("\nLaTeX: index set S")
    # print(latex_index_set(S))

    # print("\nLaTeX: ODE system for the moments")
    # print(latex_moment_ode_system(S, A_mat, c_vec))

    # print("\nLaTeX: solution for all moments in S")
    # print(latex_moment_solutions(S, m_t))

    print("\nSolution m_alpha in LaTeX: E[X^alpha]")
    print("$" + latex_single_moment(S, m_t, alpha) + "$")


if __name__ == "__main__":
    example_gene_expression_network()
