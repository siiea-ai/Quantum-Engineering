#!/bin/bash
# ==============================================================================
# SIIEA Quantum Engineering - Environment Setup
# ==============================================================================
# Detects hardware (Mac Studio 512GB / MacBook Pro 128GB) and installs
# the appropriate packages in a virtual environment for interactive
# Jupyter notebooks + MLX labs.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# After setup, activate with:
#   source .venv/bin/activate
#   jupyter lab
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo ""
echo -e "${PURPLE}══════════════════════════════════════════════════════════════${NC}"
echo -e "${PURPLE}  SIIEA Quantum Engineering - Environment Setup${NC}"
echo -e "${PURPLE}══════════════════════════════════════════════════════════════${NC}"
echo ""

# ------------------------------------------------------------------------------
# 1. Detect Hardware
# ------------------------------------------------------------------------------
CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown")
MEM_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo "0")
MEM_GB=$((MEM_BYTES / 1073741824))

echo -e "${BLUE}Hardware Detection:${NC}"
echo -e "  Chip:   ${GREEN}${CHIP}${NC}"
echo -e "  Memory: ${GREEN}${MEM_GB} GB${NC}"
echo ""

# Determine machine profile
if [ "$MEM_GB" -ge 256 ]; then
    PROFILE="studio"
    MAX_QUBITS=33
    echo -e "${GREEN}Profile: Mac Studio (512GB) — Full quantum simulation capacity${NC}"
    echo -e "  Max qubits: ~${MAX_QUBITS} (state vector requires 2^n x 16 bytes)"
    echo -e "  MLX: Full large-scale labs enabled"
elif [ "$MEM_GB" -ge 96 ]; then
    PROFILE="pro"
    MAX_QUBITS=30
    echo -e "${GREEN}Profile: MacBook Pro (128GB) — High quantum simulation capacity${NC}"
    echo -e "  Max qubits: ~${MAX_QUBITS}"
    echo -e "  MLX: Most labs enabled"
else
    PROFILE="standard"
    MAX_QUBITS=25
    echo -e "${YELLOW}Profile: Standard — Basic quantum simulation${NC}"
    echo -e "  Max qubits: ~${MAX_QUBITS}"
    echo -e "  MLX: Limited labs"
fi

echo ""

# ------------------------------------------------------------------------------
# 2. Create Virtual Environment
# ------------------------------------------------------------------------------
echo -e "${BLUE}Setting up virtual environment...${NC}"

if [ -d "$VENV_DIR" ]; then
    echo -e "  ${YELLOW}Existing .venv found — reusing${NC}"
else
    python3 -m venv "$VENV_DIR"
    echo -e "  ${GREEN}Created .venv${NC}"
fi

# Activate venv
source "${VENV_DIR}/bin/activate"
echo -e "  ${GREEN}Activated: $(which python3)${NC}"

# Upgrade pip inside venv
python3 -m pip install --upgrade pip --quiet
echo ""

# ------------------------------------------------------------------------------
# 3. Install Core Scientific Stack
# ------------------------------------------------------------------------------
echo -e "${BLUE}[1/5] Installing core scientific packages...${NC}"

pip install --quiet \
    numpy \
    scipy \
    matplotlib \
    sympy \
    ipywidgets \
    plotly \
    pandas

echo -e "  ${GREEN}Done${NC}"

# ------------------------------------------------------------------------------
# 4. Install Jupyter Ecosystem
# ------------------------------------------------------------------------------
echo -e "${BLUE}[2/5] Installing Jupyter ecosystem...${NC}"

pip install --quiet \
    jupyter \
    jupyterlab \
    ipykernel \
    nbformat \
    ipywidgets

# Register venv kernel with Jupyter
python3 -m ipykernel install --user --name quantum-eng --display-name "Quantum Engineering"

echo -e "  ${GREEN}Done (kernel: 'Quantum Engineering')${NC}"

# ------------------------------------------------------------------------------
# 5. Install Quantum Computing Libraries
# ------------------------------------------------------------------------------
echo -e "${BLUE}[3/5] Installing quantum computing libraries...${NC}"

pip install --quiet \
    qiskit \
    qiskit-aer \
    qutip \
    pennylane

echo -e "  ${GREEN}Done (Qiskit, QuTiP, PennyLane)${NC}"

# ------------------------------------------------------------------------------
# 6. Install MLX (Apple Silicon Only)
# ------------------------------------------------------------------------------
echo -e "${BLUE}[4/5] Installing Apple MLX framework...${NC}"

if [[ $(uname -m) == "arm64" ]]; then
    pip install --quiet mlx
    echo -e "  ${GREEN}Done — Apple Silicon acceleration enabled${NC}"
else
    echo -e "  ${YELLOW}Skipped — not on Apple Silicon${NC}"
fi

# ------------------------------------------------------------------------------
# 7. Install Additional Tools
# ------------------------------------------------------------------------------
echo -e "${BLUE}[5/5] Installing additional tools...${NC}"

pip install --quiet \
    tqdm \
    pillow \
    networkx \
    scikit-learn

echo -e "  ${GREEN}Done${NC}"
echo ""

# ------------------------------------------------------------------------------
# 8. Create Hardware Config for Notebooks
# ------------------------------------------------------------------------------
NOTEBOOKS_DIR="${SCRIPT_DIR}/notebooks"
mkdir -p "$NOTEBOOKS_DIR"

cat > "${NOTEBOOKS_DIR}/hardware_config.py" << PYEOF
"""
SIIEA Quantum Engineering - Hardware Configuration
Auto-generated by setup.sh on $(date +%Y-%m-%d)

This file is auto-detected by notebooks to adapt simulations
to your hardware. Re-run setup.sh to regenerate.
"""

HARDWARE = {
    "chip": "${CHIP}",
    "memory_gb": ${MEM_GB},
    "profile": "${PROFILE}",  # "studio" | "pro" | "standard"
    "max_qubits": ${MAX_QUBITS},
}

# Qubit simulation limits based on available memory
# State vector: 2^n complex128 values = 2^n x 16 bytes
QUBIT_LIMITS = {
    "safe": ${MAX_QUBITS} - 3,      # Comfortable with other apps running
    "max": ${MAX_QUBITS},            # Maximum with most memory available
    "demo": min(${MAX_QUBITS}, 20),  # Fast demos and quick tests
}


def get_max_qubits(mode="safe"):
    """Get maximum qubit count for current hardware."""
    return QUBIT_LIMITS.get(mode, QUBIT_LIMITS["safe"])


def print_hardware_info():
    """Print current hardware configuration."""
    print(f"Chip:    {HARDWARE['chip']}")
    print(f"Memory:  {HARDWARE['memory_gb']} GB")
    print(f"Profile: {HARDWARE['profile']}")
    print(f"Qubits:  {QUBIT_LIMITS['safe']} (safe) / {QUBIT_LIMITS['max']} (max)")


if __name__ == "__main__":
    print_hardware_info()
PYEOF

echo -e "${GREEN}Hardware config written to notebooks/hardware_config.py${NC}"
echo ""

# ------------------------------------------------------------------------------
# 9. Verify Installation
# ------------------------------------------------------------------------------
echo -e "${BLUE}Verifying installation...${NC}"
echo ""

python3 << 'VERIFY'
import sys

packages = {
    "numpy": "numpy",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "sympy": "sympy",
    "ipywidgets": "ipywidgets",
    "plotly": "plotly",
    "pandas": "pandas",
    "jupyterlab": "jupyterlab",
    "qiskit": "qiskit",
    "qutip": "qutip",
    "pennylane": "pennylane",
    "sklearn": "scikit-learn",
    "networkx": "networkx",
}

mlx_packages = {
    "mlx": "mlx",
}

print("  Core Packages:")
all_ok = True
for import_name, display_name in packages.items():
    try:
        mod = __import__(import_name)
        ver = getattr(mod, '__version__', 'OK')
        print(f"    \u2705 {display_name:20s} {ver}")
    except ImportError:
        print(f"    \u274c {display_name:20s} MISSING")
        all_ok = False

print("\n  MLX (Apple Silicon):")
for import_name, display_name in mlx_packages.items():
    try:
        mod = __import__(import_name)
        ver = getattr(mod, '__version__', 'OK')
        print(f"    \u2705 {display_name:20s} {ver}")
    except ImportError:
        print(f"    \u26a0\ufe0f  {display_name:20s} not installed")

if all_ok:
    print("\n  \u2705 All core packages verified!")
else:
    print("\n  \u26a0\ufe0f  Some packages missing — check errors above")
    sys.exit(1)
VERIFY

echo ""

# ------------------------------------------------------------------------------
# 10. Create activation helper
# ------------------------------------------------------------------------------
cat > "${SCRIPT_DIR}/activate.sh" << 'ACTIVATE'
#!/bin/bash
# Quick activation helper
# Usage: source activate.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/.venv/bin/activate"
echo "Quantum Engineering environment activated"
echo "  Python: $(which python3)"
echo "  Run: jupyter lab"
ACTIVATE
chmod +x "${SCRIPT_DIR}/activate.sh"

echo -e "${PURPLE}══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Setup complete!${NC}"
echo -e "${PURPLE}══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${BLUE}To activate:${NC}"
echo -e "    source activate.sh"
echo ""
echo -e "  ${BLUE}To launch Jupyter:${NC}"
echo -e "    source activate.sh && jupyter lab"
echo ""
echo -e "  ${BLUE}Hardware:${NC} ${PROFILE} (${MEM_GB} GB, ~${MAX_QUBITS} qubits max)"
echo ""
echo -e "  ${BLUE}Replicate on another machine:${NC}"
echo -e "    git clone <repo> && cd Quantum-Engineering && ./setup.sh"
echo -e "    Hardware is auto-detected — no config changes needed."
echo ""
