#!/bin/bash
# GPU runs: the system CUDA 12.8 on LD_LIBRARY_PATH shadows the pip CUDA libs JAX needs.
exec env -u LD_LIBRARY_PATH XLA_PYTHON_CLIENT_PREALLOCATE=false "$(dirname "$0")/../.venv/bin/python" "$@"
