import sympy as sp

###############################################################################
# LaTeX helpers
###############################################################################

def latex_moment_symbol(alpha, with_t=False):
    """
    Return LaTeX string for m_alpha, e.g.
        alpha = (2,)   -> 'm_{2}'
        alpha = (0,2)  -> 'm_{0,2}'
    If with_t=True, append '(t)'.
    """
    idx = ",".join(str(a) for a in alpha)
    base = f"m_{{{idx}}}"
    return base + "(t)" if with_t else base

def latex_index_set(S):
    """
    LaTeX for the index set S in terms of moment symbols:
        S = { m_{...}, m_{...}, ... }.
    """
    elems = [latex_moment_symbol(alpha, with_t=False) for alpha in S]
    return r"S = \{" + ", ".join(elems) + r"\}"

def latex_moment_ode_system(S, A_mat, c_vec):
    r"""
    Return a LaTeX aligned environment for the ODE system:

        \dot m_alpha(t) = ...    for all alpha in S.
    """
    dim = len(S)
    # helper symbols to build an expression then rewrite names
    symbs = [sp.Symbol(f"m{i}") for i in range(dim)]

    lines = []
    for i, alpha in enumerate(S):
        # build RHS as symbolic expression in m0,...,m_{dim-1}
        rhs_expr = sum(A_mat[i, j] * symbs[j] for j in range(dim)) + c_vec[i, 0]
        rhs_ltx = sp.latex(sp.simplify(rhs_expr))
        # replace m0, m1, ... by m_{alpha}(t)
        for j, beta in enumerate(S):
            orig = sp.latex(symbs[j])     # e.g. 'm_{0}'
            repl = latex_moment_symbol(beta, with_t=True)
            rhs_ltx = rhs_ltx.replace(orig, repl)
        lhs = r"\dot{" + latex_moment_symbol(alpha, with_t=False) + r"}(t)"
        lines.append(lhs + " = " + rhs_ltx)

    body = r" \\" + "\n"
    body = body.join(lines)
    return r"\begin{aligned}" + "\n" + body + "\n" + r"\end{aligned}"

def latex_moment_solutions(S, m_t):
    """
    Return a LaTeX aligned block with formulas

        m_alpha(t) = ...   for all alpha in S.
    """
    lines = []
    for i, alpha in enumerate(S):
        expr = sp.simplify(m_t[i, 0])
        lhs = latex_moment_symbol(alpha, with_t=True)
        rhs = sp.latex(expr)
        lines.append(lhs + " = " + rhs)
    body = r" \\" + "\n"
    body = body.join(lines)
    return r"\begin{aligned}" + "\n" + body + "\n" + r"\end{aligned}"

def latex_single_moment(S, m_t, alpha):
    """
    Return LaTeX formula for a single moment m_alpha(t).
    """
    idx = S.index(alpha)
    expr = sp.simplify(m_t[idx, 0])
    lhs = latex_moment_symbol(alpha, with_t=True)
    rhs = sp.latex(expr)
    return lhs + " = " + rhs
