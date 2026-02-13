#!/usr/bin/env bash
set -euo pipefail

echo "Running all benchmarks in Table 1 of the paper..."
python -m benchmarks.ou-env_x2^2
python -m benchmarks.ou-env_x2^3
python -m benchmarks.ou-env_x2^4
python -m benchmarks.ou-env_x2^5
python -m benchmarks.ou-env_x2^10
python -m benchmarks.gene_x1x5
python -m benchmarks.gene_x5^2
python -m benchmarks.gene_x1x5^2
python -m benchmarks.consensus
python -m benchmarks.vehicles
python -m benchmarks.oscillator
python -m benchmarks.coupled3d
