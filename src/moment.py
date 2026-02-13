import sympy as sp

###############################################################################
# Core utilities
###############################################################################

def monomial_from_multi_index(alpha, vars_):
    """
    Given a multi–index alpha = (a1,...,an) and variables vars_ = (x1,...,xn),
    return the monomial x1**a1 * ... * xn**an.
    """
    return sp.prod(v**e for v, e in zip(vars_, alpha))

def generator_for_sde(b_vec, sigma_mat, vars_):
    r"""
    Build the infinitesimal generator A for the polynomial SDE

        dX_t = b(X_t) dt + sigma(X_t) dW_t,

    where b_vec is an n×1 list/matrix of polynomials, sigma_mat is n×m,
    and vars_ = (x1,...,xn) are the state variables.

    For a smooth f, the generator acts as
        A f = sum_i b_i(x) ∂_i f + 1/2 sum_{i,j} (σσ^T)_{ij}(x) ∂_{ij} f.
    """
    vars_ = tuple(vars_)
    n = len(vars_)
    b_vec = sp.Matrix(b_vec)
    sigma_mat = sp.Matrix(sigma_mat)
    assert b_vec.shape == (n, 1)
    assert sigma_mat.shape[0] == n

    Sigma = sigma_mat * sigma_mat.T  # n×n polynomial matrix

    def A(f):
        # Drift part
        res = 0
        for i, x_i in enumerate(vars_):
            res += b_vec[i, 0] * sp.diff(f, x_i)

        # Diffusion part
        for i, x_i in enumerate(vars_):
            for j, x_j in enumerate(vars_):
                if Sigma[i, j] != 0:
                    res += sp.Rational(1, 2) * Sigma[i, j] * sp.diff(sp.diff(f, x_i), x_j)

        return sp.expand(res)

    return A

def moment_closure_algorithm(b_vec, sigma_mat, alpha, vars_):
    r"""
    Implementation of Algorithm 1 (moment-closure) from the paper
    "Exact Moment Estimation of Stochastic Differential Dynamics".

    Input:
        b_vec     – drift vector (list or Matrix of polynomials)
        sigma_mat – diffusion matrix (list of lists or Matrix of polynomials)
        alpha     – initial multi-index (tuple/list of nonnegative ints)
        vars_     – state variables (x1,...,xn) as SymPy symbols

    Output:
        S      – list of multi-indices in the closed set
        A_mat  – coefficient matrix in d/dt m(t) = A_mat * m(t) + c_vec
        c_vec  – constant vector in the moment ODE
    """
    vars_ = tuple(vars_)
    A = generator_for_sde(b_vec, sigma_mat, vars_)

    alpha = tuple(alpha)
    S = [alpha]         # ordered list of multi-indices
    S_set = {alpha}     # for fast membership checks
    P = [alpha]         # queue of unprocessed multi-indices

    coeffs = {}         # beta -> {gamma: a_{beta,gamma}}
    consts = {}         # beta -> c_beta

    while P:
        beta = P.pop(0)
        monom = monomial_from_multi_index(beta, vars_)
        Ax = sp.expand(A(monom))

        # Represent Ax as a polynomial and read off constant + monomial coefficients
        poly = sp.Poly(Ax, *vars_)
        const = poly.TC()   # constant term
        consts[beta] = sp.simplify(const)

        term_coeffs = {}
        for exps, coeff in poly.terms():
            gamma = tuple(exps)
            if all(e == 0 for e in gamma):
                # Constant term already handled
                continue
            term_coeffs[gamma] = term_coeffs.get(gamma, 0) + coeff
            if gamma not in S_set:
                S_set.add(gamma)
                S.append(gamma)
                P.append(gamma)

        coeffs[beta] = term_coeffs

    # Build A_mat and c_vec
    dim = len(S)
    A_mat = sp.zeros(dim, dim)
    c_vec = sp.zeros(dim, 1)

    index_of = {beta: i for i, beta in enumerate(S)}

    for beta, i in index_of.items():
        c_vec[i, 0] = consts.get(beta, 0)
        for gamma, a in coeffs.get(beta, {}).items():
            j = index_of[gamma]
            A_mat[i, j] = a

    return S, A_mat, c_vec

#################################################################

# def matrix_exp_constant(M, t, use_diagonalization=True):
#     """
#     Efficient symbolic matrix exponential for a constant matrix M:
#         exp(M t)

#     Tries diagonalization first (when M is diagonalizable), which is
#     usually much faster and produces simpler expressions; falls back
#     to SymPy's generic matrix exponential otherwise.
#     """
#     n = M.shape[0]

#     if use_diagonalization:
#         try:
#             P, D = M.diagonalize()
#             # exp(D t) is diagonal if D is diagonal
#             expDt = sp.diag(*[sp.exp(D[i, i] * t) for i in range(n)])
#             return sp.simplify(P * expDt * P.inv())
#         except Exception:
#             # not diagonalizable (or diagonalization too hard) → fall back
#             pass

#     # generic fallback
#     return (M * t).exp()


# def solve_moment_system(A_mat, c_vec, m0_vec, t,
#                         use_diagonalization=True,
#                         simplify_result=True):
#     r"""
#     Solve the linear ODE system

#         d/dt m(t) = A_mat * m(t) + c_vec,
#         m(0) = m0_vec,

#     symbolically.

#     Implementation detail:
#       - Build the augmented homogeneous system

#             d/dt [ m(t) ] = [A  c] [ m(t) ]
#                  [   1  ]   [0  0] [   1   ]

#         and compute exp(M t) once. This avoids A^{-1} and also works
#         when A is singular.
#     """
#     dim = A_mat.shape[0]

#     # Build augmented (dim+1)x(dim+1) matrix M
#     M = sp.zeros(dim + 1, dim + 1)
#     M[:dim, :dim] = A_mat
#     M[:dim, dim] = c_vec  # last column is c
#     # last row stays zeros

#     # Initial extended state [m0; 1]
#     ext0 = sp.Matrix(list(m0_vec) + [1])

#     # Matrix exponential
#     B = matrix_exp_constant(M, t, use_diagonalization=use_diagonalization)

#     ext_t = B * ext0
#     if simplify_result:
#         ext_t = sp.simplify(ext_t)

#     # First dim entries are the moments m(t)
#     m_t = ext_t[:dim, 0]
#     return sp.Matrix(m_t)

############################################################################



###############################################################################
# Helpers
###############################################################################

def topological_order_from_A(A_mat):
    """
    Try to compute a topological order of the dependency graph induced by A_mat.

    Node j -> i  if  A[i, j] != 0 and i != j  (i depends on j).

    Returns:
        order (list of indices) if graph is acyclic,
        None otherwise.
    """
    n = A_mat.shape[0]
    succ = [set() for _ in range(n)]
    indeg = [0] * n

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if A_mat[i, j] != 0:
                succ[j].add(i)
                indeg[i] += 1

    # Kahn's algorithm
    queue = [i for i in range(n) if indeg[i] == 0]
    order = []
    while queue:
        u = queue.pop(0)
        order.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)

    if len(order) != n:
        return None  # cycle detected
    return order


def matrix_exp_constant(M, t):
    """
    Generic symbolic matrix exponential exp(M t).
    (kept as a fallback method)
    """
    # Try diagonalization for speed/simplicity when possible
    try:
        P, D = M.diagonalize()
        expDt = sp.diag(*[sp.exp(D[i, i] * t) for i in range(D.shape[0])])
        return sp.simplify(P * expDt * P.inv())
    except Exception:
        # Fallback: generic exp
        return (M * t).exp()


###############################################################################
# Improved solver exploiting sparsity when possible
###############################################################################

def solve_moment_system(A_mat, c_vec, m0_vec, t,
                        simplify_result=True,
                        exploit_sparsity=True):
    r"""
    Solve the linear ODE system

        d/dt m(t) = A_mat * m(t) + c_vec,
        m(0) = m0_vec,

    symbolically.

    If exploit_sparsity=True and the dependency graph of A_mat is acyclic,
    solve the system one scalar ODE at a time using the sparsity pattern.
    Otherwise, fall back to a matrix-exponential based method.
    """
    n = A_mat.shape[0]

    # ---------- Sparse / triangular solver ----------
    if exploit_sparsity:
        order = topological_order_from_A(A_mat)
        if order is not None:
            try:
                m_list = [None] * n
                tau = sp.symbols('tau', real=True)

                for i in order:
                    # a y + f(t)
                    a = A_mat[i, i]
                    f = c_vec[i, 0]

                    # add contributions from already solved moments
                    for j in range(n):
                        if j == i:
                            continue
                        if A_mat[i, j] != 0:
                            if m_list[j] is None:
                                # Should not happen if order is correct
                                raise RuntimeError(
                                    "Dependency not yet solved when expected."
                                )
                            f += A_mat[i, j] * m_list[j]

                    # Solve y' = a y + f(t), y(0) = y0
                    y0 = m0_vec[i, 0]
                    f_tau = f.subs(t, tau)

                    if a == 0:
                        integ = sp.integrate(f_tau, (tau, 0, t))
                        y = y0 + integ
                    else:
                        integ = sp.integrate(sp.exp(-a * tau) * f_tau, (tau, 0, t))
                        y = sp.exp(a * t) * (y0 + integ)

                    m_list[i] = sp.simplify(y) if simplify_result else y

                return sp.Matrix(m_list)

            except Exception:
                # Something went wrong (e.g. integration failed) — fall back.
                pass

    # ---------- Fallback: augmented matrix + exp ----------
    dim = n
    M = sp.zeros(dim + 1, dim + 1)
    M[:dim, :dim] = A_mat
    M[:dim, dim] = c_vec  # last column is c; last row is zeros

    ext0 = sp.Matrix(list(m0_vec) + [1])  # [m0; 1]
    B = matrix_exp_constant(M, t)
    ext_t = B * ext0
    if simplify_result:
        ext_t = sp.simplify(ext_t)

    m_t = ext_t[:dim, 0]
    return sp.Matrix(m_t)



# def solve_moment_system(A_mat, c_vec, m0_vec, t):
#     r"""
#     Solve the linear ODE system

#         d/dt m(t) = A_mat * m(t) + c_vec

#     with initial condition m(0) = m0_vec (column vector).

#     Formula used (when A_mat is invertible):
#         m(t) = exp(A_mat t) m(0) + A_mat^{-1} (exp(A_mat t) - I) c_vec.

#     If A_mat is singular, this function falls back to the homogeneous solution
#     and ignores c_vec (you can replace this by a call to sympy.dsolve if needed).
#     """
#     dim = A_mat.shape[0]
#     I = sp.eye(dim)
#     B = (A_mat * t).exp()   # matrix exponential exp(A t)

#     try:
#         A_inv = A_mat.inv()
#         particular = A_inv * (B - I) * c_vec
#     except Exception:
#         # Fallback: ignore constant term if A is not invertible
#         particular = sp.zeros(dim, 1)

#     m_t = sp.simplify(B * m0_vec + particular)
#     return m_t


def pretty_print_S(S, vars_):
    """Return a list of strings for the monomials in S for nice printing."""
    vars_ = tuple(vars_)
    out = []
    for alpha in S:
        mono = monomial_from_multi_index(alpha, vars_)
        out.append(f"{alpha} -> {sp.simplify(mono)}")
    return out

