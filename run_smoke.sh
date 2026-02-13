#!/usr/bin/env bash
set -euo pipefail

echo "Running smoke tests for part of benchmarks in Table 1 of the paper..."
python -m benchmarks.vehicles
python -m benchmarks.oscillator
