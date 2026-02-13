import time
import sympy as sp

from src.moment import *
from src.latex_helper import *

###############################################################################
# Coupled3d
###############################################################################

def example_coupled_3d_box():
    # SDE:
    #   dX1_t = (-1/2 X1_t - X1_t X2_t - 1/2 X1_t X2_t^2) dt + X1_t (1 + X2_t) dW_t^(1)
    #   dX2_t = (-X2_t + X3_t) dt + 0.3 X3_t dW_t^(2)
    #   dX3_t = (X2_t - X3_t) dt + 0.3 X2_t dW_t^(3)
    #
    # Initial state: (X1_0, X2_0, X3_0) = (0, 0, 0)
    # Target moment: x1^2 x2^2 -> m_{2,2,0}(t)

    x1, x2, x3 = sp.symbols('x1 x2 x3')
    vars_ = (x1, x2, x3)

    # Use exact rationals: 1/2, 0.3 = 3/10
    half = sp.Rational(1, 2)
    three_tenths = sp.Rational(3, 10)

    # Drift b(x)
    b_vec = [
        -half * x1 - x1 * x2 - half * x1 * x2**2,  # dX1 drift
        -x2 + x3,                                  # dX2 drift
        x2 - x3                                    # dX3 drift
    ]

    # Diffusion matrix sigma(x): 3×3, W^(1), W^(2), W^(3) independent
    sigma_mat = [
        [x1 * (1 + x2), 0, 0],                 # dW^(1) in X1
        [0, three_tenths * x3, 0],             # dW^(2) in X2
        [0, 0, three_tenths * x2],             # dW^(3) in X3
    ]

    # Target monomial x1^2 x2^2
    alpha = (2, 2, 0)

    # ---------- closure S (Algorithm 1) ----------
    t0 = time.perf_counter()
    S, A_mat, c_vec = moment_closure_algorithm(b_vec, sigma_mat, alpha, vars_)
    t1 = time.perf_counter()

    print("\n=== coupled3d example (moment x1^2 x2^2) ===")
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

    # Initial condition (0,0,0)
    x1_0 = sp.Integer(0)
    x2_0 = sp.Integer(0)
    x3_0 = sp.Integer(0)

    # Build m(0)
    m0_vec = sp.Matrix([
        (x1_0 ** beta[0]) * (x2_0 ** beta[1]) * (x3_0 ** beta[2])
        for beta in S
    ])

    # ---------- solve ODE system ----------
    t2 = time.perf_counter()
    m_t = solve_moment_system(A_mat, c_vec, m0_vec, t)
    t3 = time.perf_counter()
    print(f"\nTime for solving ODE system: {t3 - t2:.6f} seconds")

    # m_{2,2,0}(t) = E[X1_t^2 X2_t^2]
    idx = S.index(alpha)
    m_2_2_0_t = sp.simplify(m_t[idx, 0])

    # print("\nSolution m_{2,2,0}(t) = E[X_{1,t}^2 X_{2,t}^2]:")
    # sp.pprint(m_2_2_0_t)

    # # ---------- optional LaTeX output ----------
    # print("\nLaTeX: index set S")
    # print(latex_index_set(S))

    # print("\nLaTeX: ODE system")
    # print(latex_moment_ode_system(S, A_mat, c_vec))

    # print("\nLaTeX: solution for all moments in S")
    # print(latex_moment_solutions(S, m_t))

    print("\nSolution m_alpha in LaTeX: E[X_{1,t}^2 X_{2,t}^2]:")
    print("$" + latex_single_moment(S, m_t, alpha) + "$\n")


if __name__ == "__main__":
    example_coupled_3d_box()
