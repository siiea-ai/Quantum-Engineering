#!/bin/bash
# Quick activation helper
# Usage: source activate.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/.venv/bin/activate"
echo "Quantum Engineering environment activated"
echo "  Python: $(which python3)"
echo "  Run: jupyter lab"
