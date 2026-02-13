import time
import sympy as sp

from src.moment import *
from src.latex_helper import *

###############################################################################
# Oscillator
###############################################################################

def example_oscillator():
    # State variables X_t = (x1, x2, x3)
    x1, x2, x3 = sp.symbols('x1 x2 x3')
    vars_ = (x1, x2, x3)

    # Use exact rationals instead of floats:
    # 0.3 = 3/10, 0.8 = 4/5, 0.2 = 1/5, 0.5 = 1/2.
    b_vec = [
        x2,  # dX1_t = X2_t dt
        -sp.Rational(3, 10)*x2 - x1 + sp.Rational(4, 5)*x3**2,  # dX2 drift
        -x3  # dX3 drift
    ]

    # Diffusion matrix sigma(x) (3×2), W^(1) and W^(2) independent
    sigma_mat = [
        [0, 0],                                  # dX1 has no noise
        [sp.Rational(1, 5)*x2, 0],               # 0.2 X2 dW^(1)
        [0, sp.Rational(1, 2)]                   # 0.5 dW^(2)
    ]


    alpha = (0, 1, 2)

    # ---------- closure S (Algorithm 1) ----------
    t0 = time.perf_counter()
    S, A_mat, c_vec = moment_closure_algorithm(b_vec, sigma_mat, alpha, vars_)
    t1 = time.perf_counter()

    print("\n=== oscillator example (moment x2^1 x3^2) ===")
    print(f"Size of S: |S| = {len(S)}\n")
    print(f"Time for obtaining closure S: {t1 - t0:.6f} seconds")

    # print("Index set S (multi-indices and corresponding monomials):")
    # for s_str in pretty_print_S(S, vars_):
    #     print("  ", s_str)

    # print("\nA matrix (size {}x{}):".format(*A_mat.shape))
    # sp.pprint(A_mat)
    # print("\nc vector:")
    # sp.pprint(c_vec)

    t = sp.symbols('t', real=True)

    # Initial condition (X1_0, X2_0, X3_0) = (0, 0, 0)
    x1_0 = sp.Integer(0)
    x2_0 = sp.Integer(0)
    x3_0 = sp.Integer(0)

    # Build initial moment vector m(0)
    m0_list = []
    for beta in S:
        val = (x1_0 ** beta[0]) * (x2_0 ** beta[1]) * (x3_0 ** beta[2])
        m0_list.append(val)
    m0_vec = sp.Matrix(m0_list)

    # ---------- solve ODE system ----------
    t2 = time.perf_counter()
    m_t = solve_moment_system(A_mat, c_vec, m0_vec, t)
    t3 = time.perf_counter()
    print(f"\nTime for solving ODE system: {t3 - t2:.6f} seconds")


    idx = S.index(alpha)
    m_alpha = sp.simplify(m_t[idx, 0])

    # print("\nSolution m_alpha(t) = E[X_alpha]:")
    # sp.pprint(m_alpha)

    # # ---------- LaTeX outputs (optional) ----------
    # print("\nLaTeX: index set S")
    # print(latex_index_set(S))

    # print("\nLaTeX: ODE system")
    # print(latex_moment_ode_system(S, A_mat, c_vec))

    # print("\nLaTeX: solution for all moments in S")
    # print(latex_moment_solutions(S, m_t))

    print("\nSolution m_alpha in LaTeX: E[X_alpha]:")
    print("$" + latex_single_moment(S, m_t, alpha) + "$\n")


if __name__ == "__main__":
    example_oscillator()
