# Exact Moment Estimation for SDEs

This repository implements the exact moment estimation algorithm used in our paper, together with utilities for LaTeX output and a collection of benchmark examples.

## Files

- `moment.py`  
  Core implementation of our exact moment estimation algorithm for stochastic differential equations (SDEs). It builds the closed moment system and solves the resulting linear ODEs symbolically.

- `latex_helper.py`  
  Utility functions for formatting index sets, ODE systems, and moment solutions as LaTeX math expressions, suitable for direct inclusion in the paper.

- `examples.ipynb`  
  Jupyter notebook containing the benchmark case studies from the paper. It calls the functions in `moment.py` (and optionally `latex_helper.py`) to compute exact moments and display both symbolic and LaTeX-formatted results.
