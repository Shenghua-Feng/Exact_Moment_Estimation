## Artifact README

This artifact contains the code to reproduce the symbolic moment-closure computations from the paper *Exact Moment Estimation of Stochastic Differential Dynamics* (see `paper-228.pdf`).

> **Virtual machine username: fm2026 password: fm2026**

> **Note on runtime:** Since the code runs in a virtual machine, it will be slower than the times reported in Table 1 of the paper.

---

## 1) Requirements

- **VirtualBox host OS**: macOS (ARM64 Apple Silicon), e.g., MacBook Pro M4.
- **Python**: 3.9+ recommended (already satisfied in the virtual machine).
- **Packages**: `sympy` (already satisfied in the virtual machine).

---

## 2) Setup (step‑by‑step, not needed in the virtual machine)

1. Open a terminal and change to the artifact root, e.g., `~/Desktop/FM_Artifact`.

The code uses only SymPy and the local modules in `src/`.

---

## 3) Smoke testing (“kick‑the‑tire phase”)

Run the provided smoke test script:

- `sh run_smoke.sh`

This runs two lightweight benchmarks:
- `benchmarks/vehicles.py`
- `benchmarks/oscillator.py`

**Expected output:** Each script prints the size of the closure set, timings, and a LaTeX expression for the target moment.

**Estimated time:** typically **seconds to a few minutes** total, depending on the host machine.

---

## 4) Reproduce the paper results (cf. Table~1)

The paper’s benchmarks correspond to the benchmark scripts in `benchmarks/` (note that Table~1 covers examples in the case study):

### Step‑by‑step

1. From the artifact root, run `sh run_all.sh` to reproduce all examples.

2. If you want to reproduce a specific benchmark, run `python -m benchmarks.consensus` from the artifact root.

**Estimated time:** varies widely by example and machine. Each benchmark typically runs in **seconds to minutes**. The full suite can take **several minutes or longer**, but is usually less than 30 minutes.

**If you have limited resources:** run only the smoke test or a single benchmark (e.g., `benchmarks/oscillator.py`) to validate functionality.

---

## 5) What results are reproduced

This artifact reproduces the **symbolic moment-closure construction** and **closed-form moment solutions** for the benchmark SDEs in Table~1. The scripts print:

- The closure index set size.
- Time to compute the closure set and solve the linear ODE system.
- LaTeX expressions for the target moment.

These outputs correspond to the symbolic results reported in the paper’s examples.

---


## 7) Troubleshooting

- **Module import errors** (e.g., `No module named 'src'`): run from the artifact root and use module execution, e.g., `python -m benchmarks.oscillator`.
- **SymPy performance**: very large closures can be slow; try a smaller example first.

---

## 8) Quick command summary

- Smoke test: `sh run_smoke.sh`
- Run all benchmarks: `sh run_all.sh`
- Run a single example: `python -m benchmarks.oscillator`
