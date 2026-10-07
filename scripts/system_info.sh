#!/usr/bin/env bash
set -euo pipefail
echo "=== GPU ==="; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo "=== CUDA compiler ==="; nvcc --version
echo "=== Host compiler ==="; c++ --version | head -n 1
