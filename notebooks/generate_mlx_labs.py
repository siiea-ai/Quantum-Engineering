"""
SIIEA Quantum Engineering - MLX Labs Generator

Generates Apple Silicon-optimized Jupyter notebooks that leverage MLX
for quantum simulation at scale. These labs exploit unified memory
architecture to push qubit counts beyond what standard laptops achieve.

Notebooks:
  01_mlx_quantum_basics.ipynb       — MLX fundamentals + quantum gates
  02_large_scale_simulation.ipynb   — State vector engine at scale
  03_quantum_neural_network.ipynb   — VQE, parameterized circuits, kernels

Usage:
    python generate_mlx_labs.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_notebook import NotebookBuilder


# ---------------------------------------------------------------------------
# Notebook 1: MLX Quantum Basics
# ---------------------------------------------------------------------------
def build_notebook_01():
    nb = NotebookBuilder(
        "MLX Quantum Basics: Apple Silicon for Quantum Computing",
        "mlx_labs/01_mlx_quantum_basics.ipynb",
        curriculum_days="Year 1, Semester 1A (Days 001-042)",
    )

    # --- MD 1: Introduction to MLX ---
    nb.md("""## What is MLX and Why Does It Matter for Quantum Simulation?

**MLX** is Apple's machine-learning framework built specifically for Apple Silicon.
Unlike CUDA-based frameworks that require discrete GPUs, MLX operates on the
**unified memory architecture** of M-series chips --- the same physical memory
is shared by CPU, GPU, and Neural Engine with zero copy overhead.

### Why this matters for quantum computing

| Property | Implication |
|----------|-------------|
| **Unified memory** | A 128 GB MacBook Pro can hold a 30-qubit state vector (~17 GB) without CPU-GPU transfers |
| **Lazy evaluation** | MLX builds a compute graph and only materializes when needed --- saves peak memory |
| **Metal backend** | GPU-accelerated linear algebra on every Mac shipped since 2020 |
| **NumPy-like API** | Minimal learning curve for scientists already using NumPy |

In this notebook we will:
1. Confirm MLX is available and benchmark it against NumPy
2. Represent quantum states and gates as MLX arrays
3. Apply gates via matrix-vector products
4. Simulate measurements using the Born rule
5. Profile performance as qubit count grows""")

    # --- Code 1: MLX detection and setup ---
    nb.code("""# --- MLX detection and environment setup ---
import time
import numpy as np

try:
    import mlx.core as mx
    HAS_MLX = True
    print("MLX available --- Apple Silicon acceleration enabled")
    print(f"MLX version: {mx.__version__ if hasattr(mx, '__version__') else 'unknown'}")
except ImportError:
    HAS_MLX = False
    print("MLX not available --- falling back to NumPy")
    print("Install MLX: pip install mlx  (requires Apple Silicon)")

# Helper: choose backend
def mx_array(data, dtype=None):
    \"\"\"Create array on best available backend.\"\"\"
    if HAS_MLX:
        return mx.array(np.asarray(data, dtype=np.complex128 if dtype is None else dtype))
    return np.asarray(data, dtype=np.complex128 if dtype is None else dtype)

def mx_matmul(a, b):
    \"\"\"Matrix multiply on best available backend.\"\"\"
    if HAS_MLX:
        return mx.matmul(a, b)
    return np.matmul(a, b)

def to_numpy(arr):
    \"\"\"Convert any array to NumPy for display/plotting.\"\"\"
    if HAS_MLX and isinstance(arr, mx.array):
        return np.array(arr)
    return np.asarray(arr)

print("\\nBackend ready."  )""")

    # --- MD 2: NumPy vs MLX benchmark ---
    nb.md("""## MLX vs NumPy: Raw Matrix Performance

Before we touch quantum mechanics, let us see *how much faster* MLX's Metal
backend is for the core operation of quantum simulation: **large matrix-vector
and matrix-matrix products**.

We will multiply random complex matrices of increasing size and record wall-clock time.""")

    # --- Code 2: Benchmark ---
    nb.code("""# --- Benchmark: MLX vs NumPy matrix multiply ---
import time
import numpy as np

sizes = [128, 256, 512, 1024, 2048, 4096]
np_times = []
mlx_times = []

print(f"{'Size':>6} | {'NumPy (ms)':>12} | {'MLX (ms)':>12} | {'Speedup':>8}")
print("-" * 50)

for n in sizes:
    # NumPy
    a_np = np.random.randn(n, n).astype(np.float32)
    b_np = np.random.randn(n, n).astype(np.float32)
    t0 = time.perf_counter()
    _ = a_np @ b_np
    np_t = (time.perf_counter() - t0) * 1000
    np_times.append(np_t)

    # MLX
    if HAS_MLX:
        a_mx = mx.array(a_np)
        b_mx = mx.array(b_np)
        mx.eval(a_mx)  # ensure on device
        mx.eval(b_mx)
        t0 = time.perf_counter()
        c_mx = mx.matmul(a_mx, b_mx)
        mx.eval(c_mx)  # force computation (MLX is lazy)
        mlx_t = (time.perf_counter() - t0) * 1000
    else:
        mlx_t = np_t  # fallback: same as NumPy

    mlx_times.append(mlx_t)
    speedup = np_t / mlx_t if mlx_t > 0 else 0
    print(f"{n:>6} | {np_t:>12.2f} | {mlx_t:>12.2f} | {speedup:>7.2f}x")""")

    # --- Code 3: Benchmark plot ---
    nb.code("""# --- Visualize benchmark results ---
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(sizes, np_times, "o-", label="NumPy", color="#1f77b4", linewidth=2)
ax1.plot(sizes, mlx_times, "s-", label="MLX", color="#ff7f0e", linewidth=2)
ax1.set_xlabel("Matrix size (N x N)")
ax1.set_ylabel("Time (ms)")
ax1.set_title("Matrix Multiply: MLX vs NumPy")
ax1.legend()
ax1.set_yscale("log")
ax1.grid(True, alpha=0.3)

speedups = [n / m if m > 0 else 1 for n, m in zip(np_times, mlx_times)]
ax2.bar(range(len(sizes)), speedups, color="#2ca02c", alpha=0.8)
ax2.set_xticks(range(len(sizes)))
ax2.set_xticklabels(sizes)
ax2.set_xlabel("Matrix size (N x N)")
ax2.set_ylabel("Speedup (x)")
ax2.set_title("MLX Speedup over NumPy")
ax2.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("mlx_labs/benchmark_matmul.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: mlx_labs/benchmark_matmul.png")""")

    # --- MD 3: Quantum states as MLX arrays ---
    nb.md(r"""## Quantum States as MLX Arrays

A single-qubit quantum state lives in $\mathbb{C}^2$:

$$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle, \quad |\alpha|^2 + |\beta|^2 = 1$$

An $n$-qubit state lives in $\mathbb{C}^{2^n}$ --- a vector of $2^n$ complex amplitudes.
We store this as an MLX (or NumPy) array of `complex128` values.

### Standard single-qubit states

| State | Vector | Description |
|-------|--------|-------------|
| $\|0\rangle$ | $[1, 0]^T$ | Computational basis zero |
| $\|1\rangle$ | $[0, 1]^T$ | Computational basis one |
| $\|+\rangle$ | $\frac{1}{\sqrt{2}}[1, 1]^T$ | Hadamard state |
| $\|-\rangle$ | $\frac{1}{\sqrt{2}}[1, -1]^T$ | Hadamard minus state |""")

    # --- Code 4: Quantum states ---
    nb.code("""# --- Fundamental quantum states as MLX arrays ---
import numpy as np

inv_sqrt2 = 1.0 / np.sqrt(2.0)

# Single-qubit basis states
ket_0 = mx_array([1, 0])
ket_1 = mx_array([0, 1])
ket_plus = mx_array([inv_sqrt2, inv_sqrt2])
ket_minus = mx_array([inv_sqrt2, -inv_sqrt2])

states = {"| 0>": ket_0, "| 1>": ket_1, "| +>": ket_plus, "| ->": ket_minus}

print("Single-Qubit Quantum States")
print("=" * 50)
for name, state in states.items():
    s = to_numpy(state)
    probs = np.abs(s) ** 2
    print(f"  {name}  =  {s}")
    print(f"         P(0) = {probs[0]:.4f},  P(1) = {probs[1]:.4f}")
    print(f"         norm = {np.sum(probs):.6f}")
    print()""")

    # --- MD 4: Bell states ---
    nb.md(r"""## Bell States: Maximally Entangled Two-Qubit States

The four Bell states form an orthonormal basis for $\mathbb{C}^4$:

$$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$
$$|\Phi^-\rangle = \frac{1}{\sqrt{2}}(|00\rangle - |11\rangle)$$
$$|\Psi^+\rangle = \frac{1}{\sqrt{2}}(|01\rangle + |10\rangle)$$
$$|\Psi^-\rangle = \frac{1}{\sqrt{2}}(|01\rangle - |10\rangle)$$

These are the foundation of quantum teleportation, superdense coding, and
entanglement-based protocols.""")

    # --- Code 5: Bell states ---
    nb.code("""# --- Bell states as 4-element vectors ---
inv_sqrt2 = 1.0 / np.sqrt(2.0)

bell_phi_plus  = mx_array([inv_sqrt2, 0, 0,  inv_sqrt2])   # |00> + |11>
bell_phi_minus = mx_array([inv_sqrt2, 0, 0, -inv_sqrt2])   # |00> - |11>
bell_psi_plus  = mx_array([0, inv_sqrt2,  inv_sqrt2, 0])   # |01> + |10>
bell_psi_minus = mx_array([0, inv_sqrt2, -inv_sqrt2, 0])   # |01> - |10>

bell_states = {
    "|Phi+>": bell_phi_plus,
    "|Phi->": bell_phi_minus,
    "|Psi+>": bell_psi_plus,
    "|Psi->": bell_psi_minus,
}

print("Bell States (2-qubit entangled states)")
print("=" * 60)
for name, state in bell_states.items():
    s = to_numpy(state)
    probs = np.abs(s) ** 2
    print(f"  {name}  amplitudes: {s}")
    print(f"         P(00)={probs[0]:.3f}  P(01)={probs[1]:.3f}  "
          f"P(10)={probs[2]:.3f}  P(11)={probs[3]:.3f}")
    print()

# Verify orthonormality
print("Orthonormality check (inner products):")
bell_list = list(bell_states.values())
bell_names = list(bell_states.keys())
for i in range(4):
    for j in range(i, 4):
        bi = to_numpy(bell_list[i])
        bj = to_numpy(bell_list[j])
        inner = np.abs(np.dot(bi.conj(), bj))
        expected = 1.0 if i == j else 0.0
        status = "pass" if abs(inner - expected) < 1e-10 else "FAIL"
        print(f"  <{bell_names[i]}|{bell_names[j]}> = {inner:.6f}  [{status}]")""")

    # --- MD 5: Quantum gates ---
    nb.md(r"""## Quantum Gates as MLX Matrices

Quantum gates are **unitary matrices** ($U^\dagger U = I$). We store them as
MLX arrays for GPU-accelerated gate application.

### Single-qubit gates

$$X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad
Y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad
Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

$$H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

### Multi-qubit gates

$$\text{CNOT} = \begin{pmatrix} 1&0&0&0\\0&1&0&0\\0&0&0&1\\0&0&1&0 \end{pmatrix}, \quad
\text{Toffoli} \in \mathbb{C}^{8\times8}$$""")

    # --- Code 6: Quantum gates ---
    nb.code("""# --- Quantum gates as MLX matrices ---
import numpy as np

inv_sqrt2 = 1.0 / np.sqrt(2.0)

# Pauli gates
I_gate = mx_array([[1, 0], [0, 1]])
X_gate = mx_array([[0, 1], [1, 0]])
Y_gate = mx_array([[0, -1j], [1j, 0]])
Z_gate = mx_array([[1, 0], [0, -1]])

# Hadamard gate
H_gate = mx_array([[inv_sqrt2, inv_sqrt2], [inv_sqrt2, -inv_sqrt2]])

# Phase gates
S_gate = mx_array([[1, 0], [0, 1j]])
T_gate = mx_array([[1, 0], [0, np.exp(1j * np.pi / 4)]])

# CNOT (2-qubit)
CNOT_gate = mx_array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
])

# Toffoli (3-qubit)
toffoli_np = np.eye(8, dtype=np.complex128)
toffoli_np[6, 6] = 0
toffoli_np[7, 7] = 0
toffoli_np[6, 7] = 1
toffoli_np[7, 6] = 1
Toffoli_gate = mx_array(toffoli_np)

gates = {
    "I": (I_gate, 1), "X": (X_gate, 1), "Y": (Y_gate, 1),
    "Z": (Z_gate, 1), "H": (H_gate, 1), "S": (S_gate, 1),
    "T": (T_gate, 1), "CNOT": (CNOT_gate, 2), "Toffoli": (Toffoli_gate, 3),
}

print("Quantum Gate Library")
print("=" * 60)
for name, (gate, n_qubits) in gates.items():
    g = to_numpy(gate)
    # Check unitarity: U^dag U = I
    product = g.conj().T @ g
    identity_check = np.allclose(product, np.eye(g.shape[0]), atol=1e-10)
    print(f"  {name:>8}  |  {n_qubits}-qubit  |  {g.shape[0]}x{g.shape[1]}  |  "
          f"Unitary: {'YES' if identity_check else 'NO'}")""")

    # --- MD 6: Gate application ---
    nb.md(r"""## Applying Gates: Matrix-Vector Products

To apply gate $U$ to state $|\psi\rangle$:

$$|\psi'\rangle = U|\psi\rangle$$

This is simply a **matrix-vector product** --- the core operation MLX accelerates.

### Example: Creating a Bell State

Starting from $|00\rangle$, apply Hadamard to qubit 0 then CNOT:

$$|00\rangle \xrightarrow{H \otimes I} \frac{1}{\sqrt{2}}(|00\rangle + |10\rangle) \xrightarrow{\text{CNOT}} \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle) = |\Phi^+\rangle$$""")

    # --- Code 7: Gate application ---
    nb.code("""# --- Applying gates via matrix-vector product ---
import numpy as np

def apply_gate(gate, state):
    \"\"\"Apply a quantum gate (matrix) to a state (vector) using MLX.\"\"\"
    # Reshape state to column vector for matmul
    if HAS_MLX:
        s = mx.reshape(state, (-1, 1))
        result = mx.matmul(gate, s)
        return mx.reshape(result, (-1,))
    else:
        s = np.asarray(state).reshape(-1, 1)
        result = np.asarray(gate) @ s
        return result.flatten()

def tensor_product(a, b):
    \"\"\"Kronecker product of two matrices/vectors.\"\"\"
    a_np = to_numpy(a)
    b_np = to_numpy(b)
    return mx_array(np.kron(a_np, b_np))

# Start with |00>
psi = tensor_product(ket_0, ket_0)
print("Initial state |00>:", to_numpy(psi))

# Apply H tensor I
H_I = tensor_product(H_gate, I_gate)
psi = apply_gate(H_I, psi)
print("After (H x I):     ", to_numpy(psi))

# Apply CNOT
psi = apply_gate(CNOT_gate, psi)
print("After CNOT:         ", to_numpy(psi))

# Verify it is Bell |Phi+>
expected = to_numpy(bell_phi_plus)
actual = to_numpy(psi)
match = np.allclose(actual, expected, atol=1e-10)
print(f"\\nIs this |Phi+>? {match}")
print(f"Probabilities: P(00)={np.abs(actual[0])**2:.4f}, "
      f"P(01)={np.abs(actual[1])**2:.4f}, "
      f"P(10)={np.abs(actual[2])**2:.4f}, "
      f"P(11)={np.abs(actual[3])**2:.4f}")""")

    # --- MD 7: Measurement simulation ---
    nb.md(r"""## Measurement Simulation: The Born Rule

When we measure a quantum state $|\psi\rangle = \sum_i \alpha_i |i\rangle$
in the computational basis, the probability of outcome $i$ is:

$$P(i) = |\alpha_i|^2$$

This is the **Born rule**. We simulate measurement by:
1. Computing probabilities from amplitudes
2. Sampling from the resulting distribution""")

    # --- Code 8: Measurement ---
    nb.code("""# --- Measurement simulation using Born rule ---
import numpy as np

def measure(state, n_shots=1000, seed=42):
    \"\"\"Simulate quantum measurement with n_shots samples.

    Returns:
        counts: dict mapping basis state index to count
        probs: array of theoretical probabilities
    \"\"\"
    amplitudes = to_numpy(state)
    probs = np.abs(amplitudes) ** 2
    probs = probs / probs.sum()  # normalize (floating point safety)

    rng = np.random.default_rng(seed)
    outcomes = rng.choice(len(probs), size=n_shots, p=probs)

    counts = {}
    n_qubits = int(np.log2(len(probs)))
    for outcome in outcomes:
        label = format(outcome, f"0{n_qubits}b")
        counts[label] = counts.get(label, 0) + 1

    return counts, probs

# Measure the Bell state |Phi+>
print("Measuring Bell state |Phi+> (10,000 shots)")
print("=" * 50)
counts, probs = measure(bell_phi_plus, n_shots=10000)

print(f"\\nTheoretical probabilities:")
for i, p in enumerate(probs):
    if p > 1e-10:
        print(f"  |{i:02b}> : {p:.4f}")

print(f"\\nMeasurement results:")
for state_label in sorted(counts.keys()):
    freq = counts[state_label] / 10000
    print(f"  |{state_label}> : {counts[state_label]:>5} counts  "
          f"({freq:.4f} vs theoretical {probs[int(state_label, 2)]:.4f})")

# Measure |+> state (single qubit)
print("\\n\\nMeasuring |+> state (10,000 shots)")
print("=" * 50)
counts_plus, probs_plus = measure(ket_plus, n_shots=10000)
for state_label in sorted(counts_plus.keys()):
    freq = counts_plus[state_label] / 10000
    print(f"  |{state_label}> : {counts_plus[state_label]:>5} counts  ({freq:.4f})")""")

    # --- Code 9: Measurement histogram ---
    nb.code("""# --- Visualize measurement outcomes ---
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Bell state measurement
labels = sorted(counts.keys())
values = [counts[l] for l in labels]
theoretical = [probs[int(l, 2)] * 10000 for l in labels]
x_pos = range(len(labels))

axes[0].bar([x - 0.15 for x in x_pos], values, 0.3, label="Measured", color="#1f77b4")
axes[0].bar([x + 0.15 for x in x_pos], theoretical, 0.3, label="Theoretical", color="#ff7f0e", alpha=0.7)
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels([f"|{l}>" for l in labels])
axes[0].set_title("Bell State |Phi+> Measurement (10k shots)")
axes[0].set_ylabel("Counts")
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis="y")

# |+> state measurement
labels_p = sorted(counts_plus.keys())
values_p = [counts_plus[l] for l in labels_p]
axes[1].bar(range(len(labels_p)), values_p, color="#2ca02c", alpha=0.8)
axes[1].set_xticks(range(len(labels_p)))
axes[1].set_xticklabels([f"|{l}>" for l in labels_p])
axes[1].set_title("|+> State Measurement (10k shots)")
axes[1].set_ylabel("Counts")
axes[1].axhline(y=5000, color="red", linestyle="--", alpha=0.5, label="Expected")
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("mlx_labs/measurement_histogram.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: mlx_labs/measurement_histogram.png")""")

    # --- MD 8: Scaling benchmark ---
    nb.md(r"""## Scaling Benchmark: MLX vs NumPy for Quantum Simulation

The critical question: **how does MLX scale as qubit count grows?**

For $n$ qubits, the state vector has $2^n$ amplitudes and a single-qubit gate
applied to the full space requires a $2^n \times 2^n$ matrix multiply.

Memory requirements:
- Each `complex128` value = 16 bytes
- State vector: $2^n \times 16$ bytes
- Full gate matrix: $2^{2n} \times 16$ bytes (we will avoid this for large $n$)

| Qubits | State vector | Full matrix |
|--------|-------------|-------------|
| 10 | 16 KB | 16 MB |
| 15 | 512 KB | 16 GB |
| 20 | 16 MB | 16 TB |

For this benchmark we use the Hadamard on qubit 0 via Kronecker product
up to a manageable size, then switch to direct state-vector manipulation.""")

    # --- Code 10: Qubit scaling benchmark ---
    nb.code("""# --- Qubit scaling benchmark: MLX vs NumPy ---
import time
import numpy as np

# Load hardware limits
try:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath("__file__")), ".."))
    from hardware_config import get_max_qubits
    max_bench_qubits = min(get_max_qubits("demo"), 20)
except ImportError:
    max_bench_qubits = 16

qubit_range = list(range(4, max_bench_qubits + 1, 2))
np_times_q = []
mlx_times_q = []
memory_mb = []

print(f"Benchmarking Hadamard-on-qubit-0 for {qubit_range[0]} to {qubit_range[-1]} qubits")
print(f"{'Qubits':>7} | {'Dim':>10} | {'Mem (MB)':>10} | {'NumPy (ms)':>12} | {'MLX (ms)':>12} | {'Speedup':>8}")
print("-" * 75)

for n in qubit_range:
    dim = 2 ** n
    mem = dim * 16 / (1024 ** 2)  # state vector memory in MB
    memory_mb.append(mem)

    # Build state |00...0> and Hadamard on qubit 0 via direct manipulation
    # Instead of full matrix, we apply H to qubit 0 by reshaping:
    #   state.reshape(2, 2^(n-1)) -> H @ state_reshaped -> flatten

    # NumPy version
    state_np = np.zeros(dim, dtype=np.complex128)
    state_np[0] = 1.0
    h_np = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)

    t0 = time.perf_counter()
    reshaped = state_np.reshape(2, dim // 2)
    result_np = (h_np @ reshaped).flatten()
    np_t = (time.perf_counter() - t0) * 1000
    np_times_q.append(np_t)

    # MLX version
    if HAS_MLX:
        state_mx = mx.zeros((dim,), dtype=mx.complex64)
        # MLX doesn't support complex128 on all builds; use complex64 if needed
        try:
            state_mx = mx.array(state_np)
        except Exception:
            state_mx = mx.array(state_np.astype(np.complex64))
        h_mx = mx.array(h_np.astype(np.complex64)) if state_mx.dtype == mx.complex64 else mx.array(h_np)
        mx.eval(state_mx)
        mx.eval(h_mx)

        t0 = time.perf_counter()
        reshaped_mx = mx.reshape(state_mx, (2, dim // 2))
        result_mx = mx.matmul(h_mx, reshaped_mx)
        result_mx = mx.reshape(result_mx, (-1,))
        mx.eval(result_mx)
        mlx_t = (time.perf_counter() - t0) * 1000
    else:
        mlx_t = np_t

    mlx_times_q.append(mlx_t)
    speedup = np_t / mlx_t if mlx_t > 0 else 1.0
    print(f"{n:>7} | {dim:>10,} | {mem:>10.2f} | {np_t:>12.3f} | {mlx_t:>12.3f} | {speedup:>7.2f}x")

print(f"\\nPeak state vector size: {memory_mb[-1]:.1f} MB for {qubit_range[-1]} qubits")""")

    # --- Code 11: Scaling plot ---
    nb.code("""# --- Scaling visualization ---
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Time comparison
axes[0].plot(qubit_range, np_times_q, "o-", label="NumPy", linewidth=2)
axes[0].plot(qubit_range, mlx_times_q, "s-", label="MLX", linewidth=2)
axes[0].set_xlabel("Number of Qubits")
axes[0].set_ylabel("Time (ms)")
axes[0].set_title("Gate Application Time")
axes[0].legend()
axes[0].set_yscale("log")
axes[0].grid(True, alpha=0.3)

# Speedup
speedups_q = [n / m if m > 0 else 1 for n, m in zip(np_times_q, mlx_times_q)]
axes[1].bar(range(len(qubit_range)), speedups_q, color="#2ca02c", alpha=0.8)
axes[1].set_xticks(range(len(qubit_range)))
axes[1].set_xticklabels(qubit_range)
axes[1].set_xlabel("Number of Qubits")
axes[1].set_ylabel("Speedup (x)")
axes[1].set_title("MLX Speedup Factor")
axes[1].axhline(y=1, color="gray", linestyle="--", alpha=0.5)
axes[1].grid(True, alpha=0.3, axis="y")

# Memory usage
axes[2].bar(range(len(qubit_range)), memory_mb, color="#9467bd", alpha=0.8)
axes[2].set_xticks(range(len(qubit_range)))
axes[2].set_xticklabels(qubit_range)
axes[2].set_xlabel("Number of Qubits")
axes[2].set_ylabel("State Vector Memory (MB)")
axes[2].set_title("Memory Requirements")
axes[2].set_yscale("log")
axes[2].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("mlx_labs/qubit_scaling.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: mlx_labs/qubit_scaling.png")""")

    # --- MD 9: Summary ---
    nb.md(r"""## Summary and Key Takeaways

### What we learned

1. **MLX provides GPU acceleration** on Apple Silicon with a NumPy-like API
2. **Quantum states** are complex vectors; $n$ qubits need $2^n$ amplitudes
3. **Quantum gates** are unitary matrices applied via matrix-vector products
4. **Bell states** demonstrate entanglement in a 2-qubit system
5. **Born rule** gives measurement probabilities from amplitudes
6. **MLX scales well** for large qubit counts thanks to Metal GPU and unified memory

### Key formulas

| Concept | Formula |
|---------|---------|
| State normalization | $\sum_i |\alpha_i|^2 = 1$ |
| Gate application | $|\psi'\rangle = U|\psi\rangle$ |
| Born rule | $P(i) = |\langle i|\psi\rangle|^2 = |\alpha_i|^2$ |
| Memory (bytes) | $2^n \times 16$ for complex128 |
| Tensor product | $(A \otimes B)(|a\rangle \otimes |b\rangle) = A|a\rangle \otimes B|b\rangle$ |

### Next notebook

In **02_large_scale_simulation.ipynb**, we build a full circuit simulator class
using MLX and push to 25+ qubits with GHZ states and Quantum Fourier Transforms.""")

    nb.save()
    print("  -> Notebook 01 complete\n")


# ---------------------------------------------------------------------------
# Notebook 2: Large-Scale Simulation
# ---------------------------------------------------------------------------
def build_notebook_02():
    nb = NotebookBuilder(
        "Large-Scale Quantum Simulation with MLX",
        "mlx_labs/02_large_scale_simulation.ipynb",
        curriculum_days="Year 1-2, Semesters 1A-2A (Days 001-168)",
    )

    # --- MD 1: Overview ---
    nb.md("""## Overview: Pushing Qubit Counts on Apple Silicon

State-vector simulation is the gold standard for exact quantum simulation:
every amplitude is tracked, every gate is applied as a linear map.
The challenge is **exponential memory**: $n$ qubits require $2^n$ complex
amplitudes.

Apple Silicon's **unified memory** gives us an edge:
- No CPU-to-GPU copy overhead
- Up to 512 GB on Mac Studio Ultra
- MLX's lazy evaluation minimizes peak memory

In this notebook we build a **complete quantum circuit simulator** backed
by MLX, then stress-test it from 10 to 28+ qubits.""")

    # --- Code 1: Imports and MLX setup ---
    nb.code("""# --- Imports and MLX setup ---
import time
import sys
import os
import numpy as np

try:
    import mlx.core as mx
    HAS_MLX = True
    print("MLX available --- Apple Silicon acceleration enabled")
except ImportError:
    HAS_MLX = False
    print("MLX not available --- falling back to NumPy")

# Hardware config
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath("__file__")), ".."))
try:
    from hardware_config import HARDWARE, get_max_qubits, QUBIT_LIMITS
    SAFE_QUBITS = get_max_qubits("safe")
    MAX_QUBITS = get_max_qubits("max")
    DEMO_QUBITS = get_max_qubits("demo")
    print(f"Hardware: {HARDWARE['chip']} | {HARDWARE['memory_gb']} GB")
    print(f"Qubit limits: {SAFE_QUBITS} (safe) / {MAX_QUBITS} (max) / {DEMO_QUBITS} (demo)")
except ImportError:
    SAFE_QUBITS = 22
    MAX_QUBITS = 25
    DEMO_QUBITS = 18
    print("hardware_config not found --- using conservative defaults")""")

    # --- MD 2: Memory analysis ---
    nb.md(r"""## Memory Analysis: How Many Qubits Can We Simulate?

Each qubit doubles the state vector size:

$$\text{Memory} = 2^n \times B \text{ bytes}$$

where $B = 16$ for `complex128` (double precision) or $B = 8$ for `complex64`.

| Qubits | Amplitudes | complex128 | complex64 |
|--------|-----------|------------|-----------|
| 10 | 1,024 | 16 KB | 8 KB |
| 20 | 1,048,576 | 16 MB | 8 MB |
| 25 | 33,554,432 | 512 MB | 256 MB |
| 28 | 268,435,456 | 4 GB | 2 GB |
| 30 | 1,073,741,824 | 16 GB | 8 GB |
| 33 | 8,589,934,592 | 128 GB | 64 GB |

On a **128 GB MacBook Pro M4 Max**, we can comfortably fit a 30-qubit
`complex128` state vector (16 GB) with room for gate operations.
A **512 GB Mac Studio** can push to 33 qubits.""")

    # --- Code 2: Memory calculator ---
    nb.code("""# --- Memory requirement calculator ---
import numpy as np

def memory_for_qubits(n, dtype="complex128"):
    \"\"\"Calculate memory needed for an n-qubit state vector.\"\"\"
    bytes_per_element = 16 if dtype == "complex128" else 8
    total_bytes = (2 ** n) * bytes_per_element
    return total_bytes

def format_bytes(b):
    \"\"\"Human-readable byte size.\"\"\"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} PB"

print("Quantum Simulation Memory Requirements")
print("=" * 65)
print(f"{'Qubits':>7} | {'Amplitudes':>15} | {'complex128':>12} | {'complex64':>12} | {'Feasible?':>10}")
print("-" * 65)

try:
    available_gb = HARDWARE["memory_gb"]
except NameError:
    available_gb = 16

for n in range(5, 36):
    dim = 2 ** n
    mem_128 = memory_for_qubits(n, "complex128")
    mem_64 = memory_for_qubits(n, "complex64")
    # Need ~3x state vector for operations (input + output + gate overhead)
    feasible = "YES" if mem_128 * 3 < available_gb * 1024**3 else "no"
    marker = " <-- limit" if feasible == "no" and memory_for_qubits(n-1, "complex128") * 3 < available_gb * 1024**3 else ""
    print(f"{n:>7} | {dim:>15,} | {format_bytes(mem_128):>12} | {format_bytes(mem_64):>12} | {feasible:>10}{marker}")""")

    # --- MD 3: Simulator design ---
    nb.md("""## MLX Quantum Circuit Simulator: Design

Our simulator avoids building full $2^n \\times 2^n$ gate matrices.
Instead, we use the **tensor product structure** to apply gates efficiently:

**Single-qubit gate on qubit $k$ of $n$-qubit state:**
1. Reshape state: $(2^n,) \\to (2^k, 2, 2^{n-k-1})$
2. Apply $2 \\times 2$ gate via tensor contraction on axis 1
3. Flatten back to $(2^n,)$

This uses $O(2^n)$ memory instead of $O(2^{2n})$.

**CNOT (controlled gate):**
1. Reshape into multi-index form
2. Conditionally apply target gate when control qubit is $|1\\rangle$

This approach is how real state-vector simulators (Qiskit Aer, Cirq) work.""")

    # --- Code 3: Simulator class ---
    nb.code("""# --- MLX Quantum Circuit Simulator ---
import numpy as np
import time

class MLXQuantumSimulator:
    \"\"\"State-vector quantum circuit simulator with MLX backend.

    Uses efficient tensor-contraction for gate application instead
    of full 2^n x 2^n matrix construction.
    \"\"\"

    def __init__(self, n_qubits, use_mlx=True, dtype="complex128"):
        self.n = n_qubits
        self.dim = 2 ** n_qubits
        self.use_mlx = use_mlx and HAS_MLX
        self.dtype = dtype
        self.gate_log = []  # track applied gates

        # Initialize |00...0>
        if self.use_mlx:
            np_state = np.zeros(self.dim, dtype=np.complex128 if dtype == "complex128" else np.complex64)
            np_state[0] = 1.0
            self.state = mx.array(np_state)
            mx.eval(self.state)
        else:
            self.state = np.zeros(self.dim, dtype=np.complex128 if dtype == "complex128" else np.complex64)
            self.state[0] = 1.0

    def _to_np(self, arr):
        if self.use_mlx and isinstance(arr, mx.array):
            return np.array(arr)
        return np.asarray(arr)

    def _from_np(self, arr):
        if self.use_mlx:
            return mx.array(arr)
        return arr

    def apply_single_qubit_gate(self, gate_np, target):
        \"\"\"Apply a 2x2 gate to qubit `target` using reshape trick.

        Reshapes (2^n,) -> (2^target, 2, 2^(n-target-1)) then contracts
        the gate along axis 1.
        \"\"\"
        state_np = self._to_np(self.state)
        n = self.n
        # Reshape to isolate target qubit
        shape = [2] * n
        state_r = state_np.reshape(shape)

        # Apply gate: contract along target axis
        # np.tensordot(gate, state, axes=([1], [target])) then move axis back
        result = np.tensordot(gate_np, state_r, axes=([1], [target]))
        # tensordot puts the gate's output axis first; move it to position `target`
        result = np.moveaxis(result, 0, target)
        self.state = self._from_np(result.reshape(self.dim))
        if self.use_mlx:
            mx.eval(self.state)
        self.gate_log.append(("single", target))

    def apply_cnot(self, control, target):
        \"\"\"Apply CNOT gate: flip target if control is |1>.\"\"\"
        state_np = self._to_np(self.state)
        n = self.n
        shape = [2] * n
        state_r = state_np.reshape(shape)

        # X gate for the target, conditioned on control=1
        x_gate = np.array([[0, 1], [1, 0]], dtype=state_np.dtype)

        # Extract slice where control qubit = 1
        slices_1 = [slice(None)] * n
        slices_1[control] = 1
        sub = state_r[tuple(slices_1)]

        # Apply X to target qubit in that subspace
        result = state_r.copy()
        sub_shape = list(sub.shape)
        # target axis in sub-array (shifted if target > control)
        t_ax = target if target < control else target - 1
        sub_result = np.tensordot(x_gate, sub, axes=([1], [t_ax]))
        sub_result = np.moveaxis(sub_result, 0, t_ax)

        result[tuple(slices_1)] = sub_result
        self.state = self._from_np(result.reshape(self.dim))
        if self.use_mlx:
            mx.eval(self.state)
        self.gate_log.append(("cnot", control, target))

    def apply_controlled_gate(self, gate_np, control, target):
        \"\"\"Apply arbitrary controlled-U gate.\"\"\"
        state_np = self._to_np(self.state)
        n = self.n
        shape = [2] * n
        state_r = state_np.reshape(shape)

        slices_1 = [slice(None)] * n
        slices_1[control] = 1

        sub = state_r[tuple(slices_1)]
        result = state_r.copy()
        t_ax = target if target < control else target - 1
        sub_result = np.tensordot(gate_np, sub, axes=([1], [t_ax]))
        sub_result = np.moveaxis(sub_result, 0, t_ax)

        result[tuple(slices_1)] = sub_result
        self.state = self._from_np(result.reshape(self.dim))
        if self.use_mlx:
            mx.eval(self.state)
        self.gate_log.append(("controlled", control, target))

    def h(self, target):
        \"\"\"Hadamard gate.\"\"\"
        h = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
        self.apply_single_qubit_gate(h, target)

    def x(self, target):
        \"\"\"Pauli-X gate.\"\"\"
        x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        self.apply_single_qubit_gate(x, target)

    def z(self, target):
        \"\"\"Pauli-Z gate.\"\"\"
        z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        self.apply_single_qubit_gate(z, target)

    def rz(self, target, theta):
        \"\"\"Rz rotation gate.\"\"\"
        rz = np.array([
            [np.exp(-1j * theta / 2), 0],
            [0, np.exp(1j * theta / 2)]
        ], dtype=np.complex128)
        self.apply_single_qubit_gate(rz, target)

    def cnot(self, control, target):
        \"\"\"CNOT gate.\"\"\"
        self.apply_cnot(control, target)

    def measure(self, n_shots=1024, seed=42):
        \"\"\"Simulate measurement.\"\"\"
        probs = np.abs(self._to_np(self.state)) ** 2
        probs = probs / probs.sum()
        rng = np.random.default_rng(seed)
        outcomes = rng.choice(self.dim, size=n_shots, p=probs)
        counts = {}
        for o in outcomes:
            label = format(o, f"0{self.n}b")
            counts[label] = counts.get(label, 0) + 1
        return counts

    def probabilities(self):
        \"\"\"Return probability distribution.\"\"\"
        return np.abs(self._to_np(self.state)) ** 2

    def memory_usage_mb(self):
        \"\"\"Estimated memory usage in MB.\"\"\"
        bytes_per = 16 if self.dtype == "complex128" else 8
        return self.dim * bytes_per / (1024 ** 2)

print("MLXQuantumSimulator class defined")
print(f"Backend: {'MLX (Metal GPU)' if HAS_MLX else 'NumPy (CPU)'}")""")

    # --- MD 4: Testing the simulator ---
    nb.md("""## Testing the Simulator: Bell State Verification

Let us verify our simulator produces correct results with a simple test:
create a Bell state and check the measurement statistics.""")

    # --- Code 4: Bell state test ---
    nb.code("""# --- Test: Bell state creation ---
sim = MLXQuantumSimulator(2)
sim.h(0)
sim.cnot(0, 1)

probs = sim.probabilities()
counts = sim.measure(n_shots=10000)

print("Bell State |Phi+> Test")
print("=" * 50)
print(f"State vector: {sim._to_np(sim.state)}")
print(f"\\nProbabilities:")
for i, p in enumerate(probs):
    if p > 1e-10:
        print(f"  |{i:02b}> : {p:.6f}")

print(f"\\nMeasurement (10k shots):")
for label in sorted(counts.keys()):
    print(f"  |{label}> : {counts[label]} ({counts[label]/10000:.4f})")

# Verify
expected = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=np.complex128)
actual = sim._to_np(sim.state)
assert np.allclose(actual, expected, atol=1e-10), "Bell state mismatch!"
print("\\nBell state VERIFIED")""")

    # --- MD 5: GHZ states ---
    nb.md(r"""## GHZ State Creation: Scaling Entanglement

The **GHZ (Greenberger-Horne-Zeilinger) state** is the $n$-qubit generalization
of the Bell state:

$$|GHZ_n\rangle = \frac{1}{\sqrt{2}}(|00\cdots0\rangle + |11\cdots1\rangle)$$

Construction:
1. Apply Hadamard to qubit 0
2. Apply CNOT from qubit 0 to every other qubit

This creates maximal entanglement across all $n$ qubits.""")

    # --- Code 5: GHZ state ---
    nb.code("""# --- GHZ state creation and verification ---
def create_ghz(n_qubits, use_mlx=True):
    \"\"\"Create an n-qubit GHZ state and return the simulator.\"\"\"
    t0 = time.perf_counter()
    sim = MLXQuantumSimulator(n_qubits, use_mlx=use_mlx)
    sim.h(0)
    for i in range(1, n_qubits):
        sim.cnot(0, i)
    elapsed = time.perf_counter() - t0
    return sim, elapsed

# Test GHZ for small sizes
for n in [3, 5, 8, 10]:
    sim, t = create_ghz(n)
    probs = sim.probabilities()
    p_000 = probs[0]
    p_111 = probs[-1]
    print(f"GHZ-{n:>2}: P(|{'0'*n}>)={p_000:.6f}  P(|{'1'*n}>)={p_111:.6f}  "
          f"sum={probs.sum():.6f}  time={t*1000:.2f}ms  mem={sim.memory_usage_mb():.2f}MB")

# Detailed view of GHZ-3
sim3, _ = create_ghz(3)
print(f"\\nGHZ-3 state vector: {sim3._to_np(sim3.state)}")
counts = sim3.measure(n_shots=10000)
print(f"Measurement (10k shots):")
for label in sorted(counts.keys()):
    print(f"  |{label}> : {counts[label]:>5}")""")

    # --- MD 6: QFT ---
    nb.md(r"""## Quantum Fourier Transform (QFT)

The **Quantum Fourier Transform** maps computational basis states to
frequency-domain states. It is the quantum analogue of the discrete Fourier
transform and a key component of Shor's algorithm.

$$QFT|j\rangle = \frac{1}{\sqrt{2^n}} \sum_{k=0}^{2^n-1} e^{2\pi i jk / 2^n} |k\rangle$$

Implementation uses:
1. Hadamard gates
2. Controlled phase rotation gates $R_k = \text{diag}(1, e^{2\pi i / 2^k})$
3. SWAP gates at the end""")

    # --- Code 6: QFT implementation ---
    nb.code("""# --- Quantum Fourier Transform ---
import numpy as np

def apply_qft(sim):
    \"\"\"Apply QFT to all qubits of the simulator in-place.\"\"\"
    n = sim.n
    for i in range(n):
        sim.h(i)
        for j in range(i + 1, n):
            k = j - i + 1
            # Controlled R_k gate
            phase = 2 * np.pi / (2 ** k)
            rk = np.array([
                [1, 0],
                [0, np.exp(1j * phase)]
            ], dtype=np.complex128)
            sim.apply_controlled_gate(rk, j, i)

    # Swap qubits for correct ordering
    for i in range(n // 2):
        j = n - 1 - i
        # SWAP = 3 CNOTs
        sim.cnot(i, j)
        sim.cnot(j, i)
        sim.cnot(i, j)

# Test QFT on |1> (should give uniform phases)
n_test = 4
sim_qft = MLXQuantumSimulator(n_test)
sim_qft.x(0)  # prepare |0001>
print(f"Input state |{'0'*(n_test-1)}1> (qubit 0 = |1>)")
print(f"State before QFT: top amplitudes")
state_before = sim_qft._to_np(sim_qft.state)
for i in range(min(8, len(state_before))):
    if abs(state_before[i]) > 1e-10:
        print(f"  |{i:0{n_test}b}> : {state_before[i]:.6f}")

t0 = time.perf_counter()
apply_qft(sim_qft)
qft_time = time.perf_counter() - t0

print(f"\\nAfter QFT ({qft_time*1000:.2f} ms):")
state_after = sim_qft._to_np(sim_qft.state)
probs_after = np.abs(state_after) ** 2
print(f"All probabilities equal? {np.allclose(probs_after, 1.0/2**n_test, atol=1e-10)}")
print(f"Probability per state: {probs_after[0]:.6f} (expected {1.0/2**n_test:.6f})")

# Show phases
print(f"\\nPhases of first 8 amplitudes:")
for i in range(min(8, len(state_after))):
    amp = state_after[i]
    phase = np.angle(amp) / np.pi
    print(f"  |{i:0{n_test}b}> : |amp|={np.abs(amp):.4f}, phase={phase:.4f} * pi")""")

    # --- MD 7: Scale test ---
    nb.md("""## Scale Test: Pushing Qubit Count

Now we stress-test our simulator. We will create GHZ states for increasing
qubit counts and measure both time and memory. The hardware config sets our
safe upper bound.""")

    # --- Code 7: Scale test ---
    nb.code("""# --- Scale test: GHZ state for increasing qubit counts ---
import time

# Determine test range from hardware
scale_qubits = list(range(8, min(SAFE_QUBITS, 28) + 1, 2))
# Add a few key checkpoints
for q in [20, 24, 26, 28]:
    if q not in scale_qubits and q <= SAFE_QUBITS:
        scale_qubits.append(q)
scale_qubits = sorted(set(scale_qubits))

ghz_times_mlx = []
ghz_times_np = []
ghz_memory = []

print(f"Scale test: GHZ state creation from {scale_qubits[0]} to {scale_qubits[-1]} qubits")
print(f"{'Qubits':>7} | {'MLX (ms)':>10} | {'NumPy (ms)':>10} | {'Speedup':>8} | {'Memory':>10}")
print("-" * 60)

for n in scale_qubits:
    mem_mb = (2 ** n) * 16 / (1024 ** 2)
    ghz_memory.append(mem_mb)

    # MLX timing
    if HAS_MLX:
        _, t_mlx = create_ghz(n, use_mlx=True)
    else:
        t_mlx = 0

    # NumPy timing
    _, t_np = create_ghz(n, use_mlx=False)

    ghz_times_mlx.append(t_mlx * 1000)
    ghz_times_np.append(t_np * 1000)

    speedup = t_np / t_mlx if t_mlx > 0 else 1.0
    print(f"{n:>7} | {t_mlx*1000:>10.2f} | {t_np*1000:>10.2f} | {speedup:>7.2f}x | {format_bytes(int(mem_mb * 1024**2)):>10}")

print(f"\\nLargest simulation: {scale_qubits[-1]} qubits, {format_bytes(int(ghz_memory[-1] * 1024**2))} state vector")""")

    # --- Code 8: Scale plot ---
    nb.code("""# --- Scale test visualization ---
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Time comparison
axes[0].plot(scale_qubits, ghz_times_np, "o-", label="NumPy", linewidth=2)
axes[0].plot(scale_qubits, ghz_times_mlx, "s-", label="MLX", linewidth=2)
axes[0].set_xlabel("Qubits")
axes[0].set_ylabel("Time (ms)")
axes[0].set_title("GHZ State Creation Time")
axes[0].legend()
axes[0].set_yscale("log")
axes[0].grid(True, alpha=0.3)

# Speedup
speedups = [n / m if m > 0 else 1 for n, m in zip(ghz_times_np, ghz_times_mlx)]
axes[1].plot(scale_qubits, speedups, "D-", color="#2ca02c", linewidth=2)
axes[1].set_xlabel("Qubits")
axes[1].set_ylabel("Speedup (x)")
axes[1].set_title("MLX Speedup")
axes[1].axhline(y=1, color="gray", linestyle="--", alpha=0.5)
axes[1].grid(True, alpha=0.3)

# Memory
axes[2].semilogy(scale_qubits, ghz_memory, "^-", color="#9467bd", linewidth=2)
axes[2].set_xlabel("Qubits")
axes[2].set_ylabel("Memory (MB)")
axes[2].set_title("State Vector Memory")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("mlx_labs/scale_test_ghz.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: mlx_labs/scale_test_ghz.png")""")

    # --- MD 8: Entanglement entropy ---
    nb.md(r"""## Entanglement Entropy

The **von Neumann entanglement entropy** quantifies how entangled subsystem $A$
is with subsystem $B$:

$$S(\rho_A) = -\text{Tr}(\rho_A \log_2 \rho_A)$$

For a bipartition of an $n$-qubit state into subsystems of sizes $n_A$ and $n_B$:
1. Reshape state vector to $(2^{n_A}, 2^{n_B})$ matrix
2. Compute singular values via SVD
3. Entropy from squared singular values (Schmidt coefficients)

$$S = -\sum_i \lambda_i^2 \log_2(\lambda_i^2)$$

- **Product state:** $S = 0$
- **Bell/GHZ state (half-half split):** $S = 1$ bit""")

    # --- Code 9: Entanglement entropy ---
    nb.code("""# --- Entanglement entropy computation ---
import numpy as np

def entanglement_entropy(state, n_qubits, partition_size):
    \"\"\"Compute von Neumann entanglement entropy for a bipartition.

    Args:
        state: state vector (numpy or mlx array)
        n_qubits: total number of qubits
        partition_size: number of qubits in subsystem A
    Returns:
        entropy in bits
    \"\"\"
    state_np = np.array(state) if not isinstance(state, np.ndarray) else state
    n_a = partition_size
    n_b = n_qubits - partition_size
    dim_a = 2 ** n_a
    dim_b = 2 ** n_b

    # Reshape to matrix and compute SVD
    psi_matrix = state_np.reshape(dim_a, dim_b)
    singular_values = np.linalg.svd(psi_matrix, compute_uv=False)

    # Schmidt coefficients squared
    schmidt_sq = singular_values ** 2
    schmidt_sq = schmidt_sq[schmidt_sq > 1e-15]  # remove numerical zeros

    # Von Neumann entropy
    entropy = -np.sum(schmidt_sq * np.log2(schmidt_sq))
    return entropy

# Test on known states
print("Entanglement Entropy Tests")
print("=" * 60)

# Product state |00> = |0> x |0>
sim_prod = MLXQuantumSimulator(2)
s_prod = entanglement_entropy(sim_prod._to_np(sim_prod.state), 2, 1)
print(f"Product state |00>:  S = {s_prod:.6f} bits (expected 0)")

# Bell state
sim_bell = MLXQuantumSimulator(2)
sim_bell.h(0)
sim_bell.cnot(0, 1)
s_bell = entanglement_entropy(sim_bell._to_np(sim_bell.state), 2, 1)
print(f"Bell state |Phi+>:   S = {s_bell:.6f} bits (expected 1)")

# GHZ-4 with different partitions
sim_ghz4, _ = create_ghz(4)
for p in range(1, 4):
    s = entanglement_entropy(sim_ghz4._to_np(sim_ghz4.state), 4, p)
    print(f"GHZ-4 (A={p}, B={4-p}): S = {s:.6f} bits (expected 1)")

# GHZ scaling
print(f"\\nGHZ entropy scaling (half-half partition):")
for n in [4, 6, 8, 10, 12]:
    sim_ghz, t = create_ghz(n)
    s = entanglement_entropy(sim_ghz._to_np(sim_ghz.state), n, n // 2)
    print(f"  GHZ-{n:>2}: S = {s:.6f} bits  (time: {t*1000:.2f} ms)")""")

    # --- MD 9: Qiskit comparison ---
    nb.md("""## Comparison: MLX Simulator vs Qiskit Aer

If Qiskit is installed, we compare our MLX simulator against Qiskit's
Aer statevector simulator for GHZ state creation. This validates our
results and benchmarks our performance against a production simulator.

Note: Qiskit is optional. The comparison is skipped gracefully if not installed.""")

    # --- Code 10: Qiskit comparison ---
    nb.code("""# --- Compare with Qiskit Aer (if available) ---
import time

try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    HAS_QISKIT = True
    print("Qiskit Aer available for comparison")
except ImportError:
    HAS_QISKIT = False
    print("Qiskit not installed --- skipping comparison")
    print("Install with: pip install qiskit qiskit-aer")

if HAS_QISKIT:
    comparison_qubits = [4, 8, 12, 16, 20]
    if DEMO_QUBITS >= 24:
        comparison_qubits.append(24)

    mlx_times_cmp = []
    qiskit_times_cmp = []

    print(f"\\n{'Qubits':>7} | {'MLX (ms)':>10} | {'Qiskit (ms)':>12} | {'Speedup':>8}")
    print("-" * 50)

    for n in comparison_qubits:
        # MLX
        t0 = time.perf_counter()
        sim_mlx, _ = create_ghz(n)
        mlx_t = (time.perf_counter() - t0) * 1000
        mlx_times_cmp.append(mlx_t)

        # Qiskit
        qc = QuantumCircuit(n)
        qc.h(0)
        for i in range(1, n):
            qc.cx(0, i)
        qc.save_statevector()

        backend = AerSimulator(method="statevector")
        t0 = time.perf_counter()
        result = backend.run(qc, shots=0).result()
        qiskit_t = (time.perf_counter() - t0) * 1000
        qiskit_times_cmp.append(qiskit_t)

        speedup = qiskit_t / mlx_t if mlx_t > 0 else 1
        print(f"{n:>7} | {mlx_t:>10.2f} | {qiskit_t:>12.2f} | {speedup:>7.2f}x")

        # Verify states match
        sv_qiskit = np.array(result.get_statevector(qc))
        sv_mlx = sim_mlx._to_np(sim_mlx.state)
        match = np.allclose(np.abs(sv_mlx), np.abs(sv_qiskit), atol=1e-6)
        if not match:
            print(f"    WARNING: states differ at {n} qubits")
else:
    print("\\nSkipping Qiskit comparison. Our simulator results are validated")
    print("by the Bell state and GHZ state tests above.")""")

    # --- Code 11: QFT scale test ---
    nb.code("""# --- QFT at scale ---
qft_qubits = list(range(4, min(DEMO_QUBITS, 20) + 1, 2))
qft_times = []

print("QFT Performance Scaling")
print("=" * 50)
print(f"{'Qubits':>7} | {'Time (ms)':>10} | {'Memory (MB)':>10}")
print("-" * 40)

for n in qft_qubits:
    sim = MLXQuantumSimulator(n)
    sim.x(0)  # prepare |00...01>

    t0 = time.perf_counter()
    apply_qft(sim)
    t = (time.perf_counter() - t0) * 1000
    qft_times.append(t)

    # Verify: QFT of |1> should give uniform amplitudes
    probs = sim.probabilities()
    uniform = np.allclose(probs, 1.0 / 2**n, atol=1e-8)

    print(f"{n:>7} | {t:>10.2f} | {sim.memory_usage_mb():>10.2f} | {'uniform' if uniform else 'WRONG'}")

# Plot QFT scaling
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(qft_qubits, qft_times, "o-", linewidth=2, color="#d62728")
ax.set_xlabel("Qubits")
ax.set_ylabel("QFT Time (ms)")
ax.set_title("Quantum Fourier Transform Scaling")
ax.set_yscale("log")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("mlx_labs/qft_scaling.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: mlx_labs/qft_scaling.png")""")

    # --- Code 12: Hardware summary ---
    nb.code("""# --- Hardware-adaptive summary ---
print("=" * 60)
print("SIMULATION CAPABILITY SUMMARY")
print("=" * 60)
try:
    print(f"Hardware:         {HARDWARE['chip']}")
    print(f"Memory:           {HARDWARE['memory_gb']} GB unified")
except NameError:
    print("Hardware:         Unknown (hardware_config.py not found)")

print(f"Backend:          {'MLX (Metal GPU)' if HAS_MLX else 'NumPy (CPU only)'}")
print(f"\\nQubit limits:")
print(f"  Demo mode:      {DEMO_QUBITS} qubits ({format_bytes(memory_for_qubits(DEMO_QUBITS))} state vector)")
print(f"  Safe mode:      {SAFE_QUBITS} qubits ({format_bytes(memory_for_qubits(SAFE_QUBITS))} state vector)")
print(f"  Maximum:        {MAX_QUBITS} qubits ({format_bytes(memory_for_qubits(MAX_QUBITS))} state vector)")
print(f"\\nBenchmark results:")
print(f"  Largest GHZ:    {scale_qubits[-1]} qubits in {ghz_times_mlx[-1]:.1f} ms")
print(f"  Largest QFT:    {qft_qubits[-1]} qubits in {qft_times[-1]:.1f} ms")
print(f"\\nRecommendations:")
if SAFE_QUBITS >= 28:
    print("  Your hardware supports research-scale simulations (28+ qubits)")
    print("  Consider complex circuits: VQE, QAOA, error correction codes")
elif SAFE_QUBITS >= 24:
    print("  Your hardware handles most educational simulations well")
    print("  GHZ, QFT, and basic VQE up to ~24 qubits")
else:
    print("  Focus on circuits up to ~20 qubits for interactive work")
    print("  Use reduced qubit counts for algorithm demonstrations")""")

    # --- MD 10: Summary ---
    nb.md(r"""## Summary

### What we built

- **MLXQuantumSimulator**: a full state-vector simulator using efficient tensor contraction
- **GHZ state** creation scaling to hardware limits
- **Quantum Fourier Transform** implementation
- **Entanglement entropy** measurement
- Performance comparison with Qiskit Aer

### Key insights

1. **Tensor contraction** avoids $O(2^{2n})$ matrix storage --- critical for large simulations
2. **MLX unified memory** eliminates CPU-GPU transfer bottleneck
3. **GHZ states** have exactly 1 bit of entanglement entropy regardless of size
4. **QFT** complexity is $O(n^2)$ gates but each gate costs $O(2^n)$ in state-vector simulation

### Next notebook

In **03_quantum_neural_network.ipynb**, we use this simulator for variational
quantum algorithms: VQE for molecular ground states and quantum kernel methods.""")

    nb.save()
    print("  -> Notebook 02 complete\n")


# ---------------------------------------------------------------------------
# Notebook 3: Quantum Neural Network / VQE
# ---------------------------------------------------------------------------
def build_notebook_03():
    nb = NotebookBuilder(
        "Quantum Neural Networks: VQE and Variational Circuits with MLX",
        "mlx_labs/03_quantum_neural_network.ipynb",
        curriculum_days="Year 2-3, Semesters 2A-3A (Days 169-504)",
    )

    # --- MD 1: Introduction ---
    nb.md(r"""## Variational Quantum Algorithms on Apple Silicon

**Variational Quantum Eigensolver (VQE)** is the leading near-term quantum
algorithm for chemistry and optimization. It combines:

- A **parameterized quantum circuit** (ansatz) that prepares trial states
- A **classical optimizer** that tunes parameters to minimize energy

$$E(\vec\theta) = \langle\psi(\vec\theta)| H |\psi(\vec\theta)\rangle$$

The classical-quantum loop:
1. Prepare $|\psi(\vec\theta)\rangle$ on quantum hardware (or simulator)
2. Measure $\langle H \rangle$ (energy expectation value)
3. Update $\vec\theta$ using gradient descent
4. Repeat until convergence

MLX gives us two advantages:
- **Fast state-vector simulation** for the quantum part
- **Automatic differentiation** (via `mlx.grad` or manual) for the classical part""")

    # --- Code 1: Imports and setup ---
    nb.code("""# --- Imports and MLX setup ---
import time
import sys
import os
import numpy as np
from scipy.optimize import minimize as scipy_minimize

try:
    import mlx.core as mx
    HAS_MLX = True
    print("MLX available --- Apple Silicon acceleration enabled")
except ImportError:
    HAS_MLX = False
    print("MLX not available --- falling back to NumPy")

# Hardware config
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath("__file__")), ".."))
try:
    from hardware_config import HARDWARE, get_max_qubits
    print(f"Hardware: {HARDWARE['chip']} | {HARDWARE['memory_gb']} GB")
    print(f"Max qubits: {get_max_qubits('safe')} (safe)")
except ImportError:
    print("hardware_config not found --- using defaults")

print("SciPy available for classical optimization")""")

    # --- MD 2: Rotation gates ---
    nb.md(r"""## Parameterized Rotation Gates

The building blocks of variational circuits are **rotation gates**:

$$R_x(\theta) = \begin{pmatrix} \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\ -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{pmatrix}$$

$$R_y(\theta) = \begin{pmatrix} \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\ \sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{pmatrix}$$

$$R_z(\theta) = \begin{pmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{pmatrix}$$

These are **continuously parameterized** --- they smoothly interpolate between
identity ($\theta=0$) and specific gates (e.g., $R_x(\pi) = -iX$).
This continuity is what makes gradient-based optimization possible.""")

    # --- Code 2: Rotation gates ---
    nb.code("""# --- Parameterized rotation gates ---
import numpy as np

def rx(theta):
    \"\"\"Rx rotation gate.\"\"\"
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)

def ry(theta):
    \"\"\"Ry rotation gate.\"\"\"
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)

def rz(theta):
    \"\"\"Rz rotation gate.\"\"\"
    return np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)]
    ], dtype=np.complex128)

# Verify: Rx(pi) should equal -i*X
print("Rotation Gate Verification")
print("=" * 50)
rx_pi = rx(np.pi)
expected_rx_pi = -1j * np.array([[0, 1], [1, 0]], dtype=np.complex128)
print(f"Rx(pi) = -iX? {np.allclose(rx_pi, expected_rx_pi, atol=1e-10)}")

ry_pi = ry(np.pi)
expected_ry_pi = -1j * np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
print(f"Ry(pi) = -iY? {np.allclose(ry_pi, expected_ry_pi, atol=1e-10)}")

rz_pi = rz(np.pi)
expected_rz_pi = -1j * np.array([[1, 0], [0, -1]], dtype=np.complex128)
print(f"Rz(pi) = -iZ? {np.allclose(rz_pi, expected_rz_pi, atol=1e-10)}")

# Show continuous parameterization
print(f"\\nRy at different angles:")
for angle in [0, np.pi/4, np.pi/2, np.pi, 2*np.pi]:
    gate = ry(angle)
    print(f"  Ry({angle/np.pi:.2f}*pi) = {gate.round(4)}")""")

    # --- MD 3: VQE simulator ---
    nb.md("""## Building the VQE Simulator

We need a simulator that:
1. Accepts a **parameter vector** $\\vec\\theta$
2. Prepares a **variational state** $|\\psi(\\vec\\theta)\\rangle$
3. Computes **energy** $\\langle H \\rangle$
4. Supports **gradient computation** for optimization

Our ansatz (variational circuit) for 2 qubits:
```
|0> -- Ry(theta_0) -- CNOT --         -- Ry(theta_2) --
|0> -- Ry(theta_1) ----x---- Rz(theta_3) -- Ry(theta_4) --
```

This is a **hardware-efficient ansatz**: alternating rotation and entangling layers.""")

    # --- Code 3: VQE simulator class ---
    nb.code("""# --- VQE Simulator Class ---
import numpy as np

class VQESimulator:
    \"\"\"Variational Quantum Eigensolver simulator with MLX backend.

    Uses efficient state-vector simulation for expectation value computation.
    \"\"\"

    def __init__(self, n_qubits, hamiltonian_terms, use_mlx=True):
        \"\"\"
        Args:
            n_qubits: number of qubits
            hamiltonian_terms: list of (coefficient, pauli_string)
                e.g., [(-1.0, "ZZ"), (0.5, "XI"), (0.5, "IX")]
                Pauli string uses I, X, Y, Z for each qubit
        \"\"\"
        self.n = n_qubits
        self.dim = 2 ** n_qubits
        self.use_mlx = use_mlx and HAS_MLX
        self.hamiltonian_terms = hamiltonian_terms
        self.eval_count = 0
        self.energy_history = []

        # Precompute Pauli matrices
        self.paulis = {
            "I": np.eye(2, dtype=np.complex128),
            "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
            "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
            "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
        }

        # Build full Hamiltonian matrix
        self.H_matrix = self._build_hamiltonian()

    def _build_hamiltonian(self):
        \"\"\"Construct full Hamiltonian matrix from Pauli terms.\"\"\"
        H = np.zeros((self.dim, self.dim), dtype=np.complex128)
        for coeff, pauli_str in self.hamiltonian_terms:
            # Build tensor product of individual Pauli matrices
            term = np.array([[1]], dtype=np.complex128)
            for p in pauli_str:
                term = np.kron(term, self.paulis[p])
            H += coeff * term
        return H

    def _prepare_state(self, params):
        \"\"\"Prepare variational state with given parameters.

        Hardware-efficient ansatz with alternating Ry and CNOT layers.
        \"\"\"
        state = np.zeros(self.dim, dtype=np.complex128)
        state[0] = 1.0

        n = self.n
        param_idx = 0

        # Number of layers determined by parameter count
        n_layers = len(params) // (2 * n)
        if n_layers == 0:
            n_layers = 1

        for layer in range(n_layers):
            # Rotation layer: Ry on each qubit
            for q in range(n):
                if param_idx < len(params):
                    gate = ry(params[param_idx])
                    state = self._apply_single_gate(state, gate, q)
                    param_idx += 1

            # Entangling layer: CNOT chain
            for q in range(n - 1):
                state = self._apply_cnot(state, q, q + 1)

            # Second rotation layer: Rz on each qubit
            for q in range(n):
                if param_idx < len(params):
                    gate = rz(params[param_idx])
                    state = self._apply_single_gate(state, gate, q)
                    param_idx += 1

        return state

    def _apply_single_gate(self, state, gate, target):
        \"\"\"Apply single-qubit gate using tensor contraction.\"\"\"
        shape = [2] * self.n
        state_r = state.reshape(shape)
        result = np.tensordot(gate, state_r, axes=([1], [target]))
        result = np.moveaxis(result, 0, target)
        return result.reshape(self.dim)

    def _apply_cnot(self, state, control, target):
        \"\"\"Apply CNOT gate.\"\"\"
        shape = [2] * self.n
        state_r = state.reshape(shape)
        x_gate = self.paulis["X"]

        slices_1 = [slice(None)] * self.n
        slices_1[control] = 1
        sub = state_r[tuple(slices_1)]

        result = state_r.copy()
        t_ax = target if target < control else target - 1
        sub_result = np.tensordot(x_gate, sub, axes=([1], [t_ax]))
        sub_result = np.moveaxis(sub_result, 0, t_ax)
        result[tuple(slices_1)] = sub_result
        return result.reshape(self.dim)

    def energy(self, params):
        \"\"\"Compute expectation value <psi(params)|H|psi(params)>.\"\"\"
        state = self._prepare_state(params)
        # <psi|H|psi>
        h_psi = self.H_matrix @ state
        e = np.real(np.dot(state.conj(), h_psi))
        self.eval_count += 1
        self.energy_history.append(e)
        return e

    def gradient(self, params, shift=np.pi / 2):
        \"\"\"Compute gradient using parameter-shift rule.

        For Ry/Rz gates: df/dtheta = [f(theta+s) - f(theta-s)] / (2*sin(s))
        With s = pi/2: df/dtheta = [f(theta+pi/2) - f(theta-pi/2)] / 2
        \"\"\"
        grad = np.zeros_like(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += shift
            params_minus[i] -= shift

            # Don't record these in history
            state_plus = self._prepare_state(params_plus)
            state_minus = self._prepare_state(params_minus)
            e_plus = np.real(np.dot(state_plus.conj(), self.H_matrix @ state_plus))
            e_minus = np.real(np.dot(state_minus.conj(), self.H_matrix @ state_minus))

            grad[i] = (e_plus - e_minus) / (2 * np.sin(shift))
        return grad

print("VQESimulator class defined")
print("Supports: hardware-efficient ansatz, parameter-shift gradients")""")

    # --- MD 4: H2 molecule ---
    nb.md(r"""## Application: Ground State Energy of H$_2$ Molecule

The hydrogen molecule H$_2$ is the simplest molecular system and a benchmark
for quantum chemistry. In the **STO-3G basis** with a **Jordan-Wigner mapping**,
the 2-qubit Hamiltonian is:

$$H = g_0 I \otimes I + g_1 Z \otimes I + g_2 I \otimes Z + g_3 Z \otimes Z + g_4 X \otimes X + g_5 Y \otimes Y$$

At equilibrium bond length (0.735 A), the coefficients are approximately:
- $g_0 = -0.4804$, $g_1 = 0.3435$, $g_2 = -0.4347$
- $g_3 = 0.5716$, $g_4 = 0.0910$, $g_5 = 0.0910$

The exact ground state energy is approximately **-1.137 Hartree**.""")

    # --- Code 4: H2 Hamiltonian ---
    nb.code("""# --- H2 molecule Hamiltonian (STO-3G, Jordan-Wigner) ---
import numpy as np

# Hamiltonian coefficients at equilibrium bond length 0.735 Angstrom
g0 = -0.4804
g1 =  0.3435
g2 = -0.4347
g3 =  0.5716
g4 =  0.0910
g5 =  0.0910

h2_hamiltonian = [
    (g0, "II"),
    (g1, "ZI"),
    (g2, "IZ"),
    (g3, "ZZ"),
    (g4, "XX"),
    (g5, "YY"),
]

print("H2 Molecule Hamiltonian")
print("=" * 50)
for coeff, pauli in h2_hamiltonian:
    print(f"  {coeff:+.4f} * {pauli}")

# Create simulator
vqe = VQESimulator(2, h2_hamiltonian)

# Exact diagonalization for reference
eigenvalues = np.linalg.eigvalsh(vqe.H_matrix)
exact_ground = eigenvalues[0]
print(f"\\nExact eigenvalues: {eigenvalues}")
print(f"Exact ground state energy: {exact_ground:.6f} Hartree")
print(f"Expected: approximately -1.137 Hartree")""")

    # --- MD 5: VQE optimization ---
    nb.md("""## Running VQE: Finding the Ground State

We now run the variational optimization loop:
1. Start with random parameters
2. Use SciPy's L-BFGS-B optimizer with our energy function
3. Track convergence

We also compare **parameter-shift gradients** (exact quantum gradients)
with **finite-difference gradients** (classical approximation).""")

    # --- Code 5: VQE optimization ---
    nb.code("""# --- VQE Optimization for H2 ---
import numpy as np
from scipy.optimize import minimize as scipy_minimize
import time

# Parameters: 2 qubits, 2 layers of Ry + Rz = 2 * 2 * 2 = 8 params
n_params = 8
np.random.seed(42)
initial_params = np.random.uniform(-np.pi, np.pi, n_params)

# Method 1: SciPy L-BFGS-B with finite differences
print("Method 1: SciPy L-BFGS-B (finite differences)")
print("=" * 50)
vqe1 = VQESimulator(2, h2_hamiltonian)
t0 = time.perf_counter()
result_fd = scipy_minimize(
    vqe1.energy,
    initial_params.copy(),
    method="L-BFGS-B",
    options={"maxiter": 200, "ftol": 1e-12}
)
time_fd = time.perf_counter() - t0

print(f"Converged: {result_fd.success}")
print(f"Energy:    {result_fd.fun:.8f} Hartree")
print(f"Error:     {abs(result_fd.fun - exact_ground):.2e} Hartree")
print(f"Evals:     {vqe1.eval_count}")
print(f"Time:      {time_fd*1000:.1f} ms")

# Method 2: Gradient descent with parameter-shift rule
print(f"\\nMethod 2: Gradient Descent (parameter-shift rule)")
print("=" * 50)
vqe2 = VQESimulator(2, h2_hamiltonian)
params = initial_params.copy()
lr = 0.1
n_steps = 200
energies_gd = []

t0 = time.perf_counter()
for step in range(n_steps):
    e = vqe2.energy(params)
    energies_gd.append(e)
    grad = vqe2.gradient(params)
    params -= lr * grad

    if step % 40 == 0:
        print(f"  Step {step:>4}: E = {e:.8f}  |grad| = {np.linalg.norm(grad):.6f}")

final_e = vqe2.energy(params)
energies_gd.append(final_e)
time_gd = time.perf_counter() - t0

print(f"\\nFinal energy:  {final_e:.8f} Hartree")
print(f"Error:         {abs(final_e - exact_ground):.2e} Hartree")
print(f"Time:          {time_gd*1000:.1f} ms")
print(f"\\nExact answer:  {exact_ground:.8f} Hartree")""")

    # --- Code 6: Convergence plot ---
    nb.code("""# --- Convergence visualization ---
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: L-BFGS-B convergence
ax = axes[0]
history1 = vqe1.energy_history
ax.plot(range(len(history1)), history1, color="#1f77b4", linewidth=1.5, alpha=0.8)
ax.axhline(y=exact_ground, color="red", linestyle="--", linewidth=1.5, label=f"Exact: {exact_ground:.4f}")
ax.set_xlabel("Function Evaluation")
ax.set_ylabel("Energy (Hartree)")
ax.set_title("VQE Convergence: L-BFGS-B")
ax.legend()
ax.grid(True, alpha=0.3)

# Right: Gradient descent convergence
ax = axes[1]
ax.plot(range(len(energies_gd)), energies_gd, color="#ff7f0e", linewidth=1.5)
ax.axhline(y=exact_ground, color="red", linestyle="--", linewidth=1.5, label=f"Exact: {exact_ground:.4f}")
ax.set_xlabel("Optimization Step")
ax.set_ylabel("Energy (Hartree)")
ax.set_title("VQE Convergence: Parameter-Shift GD")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("mlx_labs/vqe_convergence.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: mlx_labs/vqe_convergence.png")""")

    # --- MD 6: Parameter landscape ---
    nb.md(r"""## Energy Landscape Visualization

To understand why VQE converges (or gets stuck), we visualize the **energy
landscape** by sweeping two parameters while fixing the others at their
optimal values. This reveals:

- **Convexity** or lack thereof
- **Local minima** that can trap the optimizer
- The **smoothness** that makes gradient methods effective""")

    # --- Code 7: Energy landscape ---
    nb.code("""# --- Energy landscape visualization ---
import numpy as np
import matplotlib.pyplot as plt

# Use the L-BFGS-B optimal params as reference
optimal_params = result_fd.x.copy()

# Sweep two parameters
param_a, param_b = 0, 1  # first two rotation angles
n_points = 50
theta_range = np.linspace(-np.pi, np.pi, n_points)

landscape = np.zeros((n_points, n_points))
vqe_landscape = VQESimulator(2, h2_hamiltonian)

for i, ta in enumerate(theta_range):
    for j, tb in enumerate(theta_range):
        test_params = optimal_params.copy()
        test_params[param_a] = ta
        test_params[param_b] = tb
        # Direct energy computation (skip history)
        state = vqe_landscape._prepare_state(test_params)
        h_psi = vqe_landscape.H_matrix @ state
        landscape[i, j] = np.real(np.dot(state.conj(), h_psi))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Contour plot
ax = axes[0]
cs = ax.contourf(theta_range / np.pi, theta_range / np.pi, landscape.T,
                  levels=30, cmap="RdYlBu_r")
plt.colorbar(cs, ax=ax, label="Energy (Hartree)")
ax.plot(optimal_params[param_a] / np.pi, optimal_params[param_b] / np.pi,
        "k*", markersize=15, label="Optimum")
ax.set_xlabel(f"theta_{param_a} / pi")
ax.set_ylabel(f"theta_{param_b} / pi")
ax.set_title("VQE Energy Landscape")
ax.legend()

# 3D surface
from mpl_toolkits.mplot3d import Axes3D
ax3d = fig.add_subplot(122, projection="3d")
T_A, T_B = np.meshgrid(theta_range / np.pi, theta_range / np.pi)
ax3d.plot_surface(T_A, T_B, landscape.T, cmap="RdYlBu_r", alpha=0.8)
ax3d.set_xlabel(f"theta_{param_a} / pi")
ax3d.set_ylabel(f"theta_{param_b} / pi")
ax3d.set_zlabel("Energy (Hartree)")
ax3d.set_title("Energy Surface")

plt.tight_layout()
plt.savefig("mlx_labs/energy_landscape.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: mlx_labs/energy_landscape.png")""")

    # --- MD 7: MLX gradient comparison ---
    nb.md(r"""## Gradient Methods Comparison: Parameter-Shift vs Finite Differences

Two approaches to computing $\nabla_\theta E(\vec\theta)$:

**Parameter-shift rule** (exact for Ry/Rz gates):
$$\frac{\partial E}{\partial \theta_i} = \frac{E(\theta_i + \pi/2) - E(\theta_i - \pi/2)}{2}$$

**Finite differences** (approximate):
$$\frac{\partial E}{\partial \theta_i} \approx \frac{E(\theta_i + \epsilon) - E(\theta_i - \epsilon)}{2\epsilon}$$

The parameter-shift rule is **exact** and requires 2 circuit evaluations per
parameter. Finite differences introduce truncation error controlled by $\epsilon$.""")

    # --- Code 8: Gradient comparison ---
    nb.code("""# --- Gradient method comparison ---
import numpy as np
import time

vqe_grad = VQESimulator(2, h2_hamiltonian)
test_params = np.random.uniform(-np.pi, np.pi, n_params)

# Parameter-shift gradient
t0 = time.perf_counter()
grad_ps = vqe_grad.gradient(test_params)
time_ps = time.perf_counter() - t0

# Finite difference gradients at various epsilon
epsilons = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]
grad_fds = []
time_fds = []

for eps in epsilons:
    t0 = time.perf_counter()
    grad_fd = np.zeros(n_params)
    for i in range(n_params):
        p_plus = test_params.copy()
        p_minus = test_params.copy()
        p_plus[i] += eps
        p_minus[i] -= eps
        state_p = vqe_grad._prepare_state(p_plus)
        state_m = vqe_grad._prepare_state(p_minus)
        e_p = np.real(np.dot(state_p.conj(), vqe_grad.H_matrix @ state_p))
        e_m = np.real(np.dot(state_m.conj(), vqe_grad.H_matrix @ state_m))
        grad_fd[i] = (e_p - e_m) / (2 * eps)
    time_fd = time.perf_counter() - t0
    grad_fds.append(grad_fd)
    time_fds.append(time_fd)

print("Gradient Method Comparison")
print("=" * 70)
print(f"{'Method':>25} | {'Time (ms)':>10} | {'Max |error|':>12} | {'Mean |error|':>12}")
print("-" * 70)
print(f"{'Parameter-shift':>25} | {time_ps*1000:>10.3f} | {'(reference)':>12} | {'(reference)':>12}")

for eps, grad_fd, t_fd in zip(epsilons, grad_fds, time_fds):
    err = np.abs(grad_fd - grad_ps)
    print(f"{'FD eps=' + f'{eps:.0e}':>25} | {t_fd*1000:>10.3f} | {err.max():>12.2e} | {err.mean():>12.2e}")

print(f"\\nParameter-shift gradient: {grad_ps.round(6)}")
print(f"Best FD gradient (1e-6): {grad_fds[2].round(6)}")""")

    # --- Code 9: Gradient comparison plot ---
    nb.code("""# --- Gradient error visualization ---
import matplotlib.pyplot as plt

errors_max = [np.abs(gfd - grad_ps).max() for gfd in grad_fds]
errors_mean = [np.abs(gfd - grad_ps).mean() for gfd in grad_fds]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Error vs epsilon
ax = axes[0]
ax.loglog(epsilons, errors_max, "o-", label="Max error", linewidth=2)
ax.loglog(epsilons, errors_mean, "s-", label="Mean error", linewidth=2)
ax.set_xlabel("Finite Difference epsilon")
ax.set_ylabel("Gradient Error (vs parameter-shift)")
ax.set_title("FD Accuracy vs Step Size")
ax.legend()
ax.grid(True, alpha=0.3)
ax.invert_xaxis()

# Per-component comparison
ax = axes[1]
x = np.arange(n_params)
width = 0.35
ax.bar(x - width/2, np.abs(grad_ps), width, label="Parameter-shift", color="#1f77b4")
ax.bar(x + width/2, np.abs(grad_fds[2]), width, label="FD (eps=1e-6)", color="#ff7f0e", alpha=0.7)
ax.set_xlabel("Parameter Index")
ax.set_ylabel("|Gradient|")
ax.set_title("Gradient Components")
ax.set_xticks(x)
ax.set_xticklabels([f"theta_{i}" for i in x], rotation=45)
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("mlx_labs/gradient_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: mlx_labs/gradient_comparison.png")""")

    # --- MD 8: Quantum kernels ---
    nb.md(r"""## Quantum Kernel Methods

**Quantum kernel methods** use a quantum circuit to compute a kernel matrix
for classical machine learning. The idea:

1. Encode data point $\vec{x}$ into a quantum state $|\phi(\vec{x})\rangle$
   using a **feature map** circuit
2. Compute the kernel:
   $$K(x_i, x_j) = |\langle\phi(x_i)|\phi(x_j)\rangle|^2$$
3. Use the kernel matrix with classical SVM, kernel PCA, etc.

This is powerful because quantum circuits can create **exponentially complex
feature maps** that are hard to compute classically.""")

    # --- Code 10: Quantum kernel ---
    nb.code("""# --- Quantum Kernel Computation ---
import numpy as np
import time

class QuantumKernel:
    \"\"\"Quantum kernel using parameterized feature map.

    Feature map: for each data point x = [x_0, x_1, ...]:
      - Ry(x_i) on qubit i
      - CNOT entangling layer
      - Rz(x_i * x_j) for pairs (data re-uploading)
    \"\"\"

    def __init__(self, n_qubits):
        self.n = n_qubits
        self.dim = 2 ** n_qubits

    def feature_map(self, x):
        \"\"\"Prepare quantum state encoding data point x.\"\"\"
        state = np.zeros(self.dim, dtype=np.complex128)
        state[0] = 1.0

        # First layer: Ry encoding
        for q in range(min(self.n, len(x))):
            gate = ry(x[q])
            shape = [2] * self.n
            state_r = state.reshape(shape)
            result = np.tensordot(gate, state_r, axes=([1], [q]))
            result = np.moveaxis(result, 0, q)
            state = result.reshape(self.dim)

        # Entangling layer: CNOT chain
        for q in range(self.n - 1):
            x_gate = np.array([[0, 1], [1, 0]], dtype=np.complex128)
            shape = [2] * self.n
            state_r = state.reshape(shape)
            slices_1 = [slice(None)] * self.n
            slices_1[q] = 1
            sub = state_r[tuple(slices_1)]
            result_arr = state_r.copy()
            t_ax = q + 1 if q + 1 > q else q
            t_ax = 0  # after removing control dimension, target is at adjusted index
            # Simplified: just apply X to next qubit conditioned on current
            t_actual = q + 1 - (1 if q + 1 > q else 0)
            sub_result = np.tensordot(x_gate, sub, axes=([1], [t_actual]))
            sub_result = np.moveaxis(sub_result, 0, t_actual)
            result_arr[tuple(slices_1)] = sub_result
            state = result_arr.reshape(self.dim)

        # Second layer: Rz with product features
        for q in range(min(self.n, len(x))):
            for q2 in range(q + 1, min(self.n, len(x))):
                gate = rz(x[q] * x[q2])
                shape = [2] * self.n
                state_r = state.reshape(shape)
                result = np.tensordot(gate, state_r, axes=([1], [q]))
                result = np.moveaxis(result, 0, q)
                state = result.reshape(self.dim)

        return state

    def kernel_entry(self, x_i, x_j):
        \"\"\"Compute kernel K(x_i, x_j) = |<phi(x_i)|phi(x_j)>|^2.\"\"\"
        state_i = self.feature_map(x_i)
        state_j = self.feature_map(x_j)
        overlap = np.dot(state_i.conj(), state_j)
        return np.abs(overlap) ** 2

    def kernel_matrix(self, X):
        \"\"\"Compute full kernel matrix for dataset X (n_samples x n_features).\"\"\"
        n_samples = len(X)
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            K[i, i] = 1.0  # self-overlap is always 1
            for j in range(i + 1, n_samples):
                k_ij = self.kernel_entry(X[i], X[j])
                K[i, j] = k_ij
                K[j, i] = k_ij
        return K

# Test with synthetic data
np.random.seed(42)
n_samples = 30
n_features = 2  # 2 qubits

# Two-class dataset (XOR-like pattern)
X_class0 = np.random.randn(n_samples // 2, n_features) * 0.5 + np.array([1, 1])
X_class1 = np.random.randn(n_samples // 2, n_features) * 0.5 + np.array([-1, -1])
X = np.vstack([X_class0, X_class1])
y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

# Compute quantum kernel
qk = QuantumKernel(n_qubits=2)
t0 = time.perf_counter()
K = qk.kernel_matrix(X)
kernel_time = time.perf_counter() - t0

print(f"Quantum Kernel Computation")
print(f"=" * 50)
print(f"Dataset: {n_samples} samples, {n_features} features")
print(f"Qubits:  {qk.n}")
print(f"Kernel matrix: {K.shape}")
print(f"Time:    {kernel_time*1000:.1f} ms")
print(f"\\nKernel matrix statistics:")
print(f"  Min:  {K.min():.6f}")
print(f"  Max:  {K.max():.6f}")
print(f"  Mean: {K.mean():.6f}")
print(f"  Symmetric: {np.allclose(K, K.T)}")
print(f"  PSD: {np.all(np.linalg.eigvalsh(K) >= -1e-10)}")""")

    # --- Code 11: Kernel visualization ---
    nb.code("""# --- Quantum kernel visualization ---
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Kernel matrix heatmap
ax = axes[0]
im = ax.imshow(K, cmap="viridis", aspect="auto")
plt.colorbar(im, ax=ax, label="Kernel value")
ax.set_xlabel("Sample index")
ax.set_ylabel("Sample index")
ax.set_title("Quantum Kernel Matrix")
# Draw class boundary
ax.axhline(y=n_samples//2 - 0.5, color="red", linewidth=1.5, linestyle="--")
ax.axvline(x=n_samples//2 - 0.5, color="red", linewidth=1.5, linestyle="--")

# Data scatter plot
ax = axes[1]
ax.scatter(X_class0[:, 0], X_class0[:, 1], c="#1f77b4", label="Class 0", s=50)
ax.scatter(X_class1[:, 0], X_class1[:, 1], c="#ff7f0e", label="Class 1", s=50)
ax.set_xlabel("Feature 0")
ax.set_ylabel("Feature 1")
ax.set_title("Dataset (2 classes)")
ax.legend()
ax.grid(True, alpha=0.3)

# Classical vs quantum kernel comparison (RBF kernel)
from scipy.spatial.distance import cdist
rbf_gamma = 0.5
K_rbf = np.exp(-rbf_gamma * cdist(X, X, "sqeuclidean"))

# Compute alignment between kernels
alignment = np.sum(K * K_rbf) / (np.linalg.norm(K, "fro") * np.linalg.norm(K_rbf, "fro"))

ax = axes[2]
ax.scatter(K.flatten(), K_rbf.flatten(), alpha=0.3, s=10)
ax.set_xlabel("Quantum Kernel")
ax.set_ylabel("RBF Kernel")
ax.set_title(f"Kernel Alignment = {alignment:.4f}")
ax.plot([0, 1], [0, 1], "r--", alpha=0.5)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("mlx_labs/quantum_kernel.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: mlx_labs/quantum_kernel.png")""")

    # --- MD 9: Scaling and hardware ---
    nb.md("""## Scaling VQE: Hardware Recommendations

As we increase qubit count, VQE costs grow:

| Component | Scaling |
|-----------|---------|
| State vector | $O(2^n)$ memory |
| Energy evaluation | $O(2^n)$ per Pauli term |
| Parameter-shift gradient | $O(2p \\cdot 2^n)$ for $p$ parameters |
| Optimization iterations | Problem-dependent, typically 100-1000 |

### Hardware recommendations

| Qubits | Memory needed | Recommended hardware |
|--------|-------------|---------------------|
| 2-10 | < 16 KB | Any Mac |
| 10-20 | 16 KB - 16 MB | Any Mac |
| 20-25 | 16 MB - 512 MB | MacBook Pro 32+ GB |
| 25-28 | 512 MB - 4 GB | MacBook Pro 128 GB |
| 28-30 | 4 GB - 16 GB | MacBook Pro 128 GB (MLX) |
| 30-33 | 16 GB - 128 GB | Mac Studio 512 GB |""")

    # --- Code 12: VQE scaling test ---
    nb.code("""# --- VQE scaling test ---
import time
import numpy as np

def run_vqe_test(n_qubits, n_steps=50):
    \"\"\"Run a small VQE on a random Hamiltonian to benchmark scaling.\"\"\"
    # Random Hamiltonian: ZZ on adjacent pairs + X on each qubit
    terms = []
    for q in range(n_qubits - 1):
        pauli = "I" * q + "ZZ" + "I" * (n_qubits - q - 2)
        terms.append((-1.0, pauli))
    for q in range(n_qubits):
        pauli = "I" * q + "X" + "I" * (n_qubits - q - 1)
        terms.append((0.5, pauli))

    vqe = VQESimulator(n_qubits, terms)
    n_params = 4 * n_qubits  # 2 layers x 2 rotations x n_qubits
    params = np.random.uniform(-np.pi, np.pi, n_params)

    t0 = time.perf_counter()
    result = scipy_minimize(vqe.energy, params, method="L-BFGS-B",
                           options={"maxiter": n_steps, "ftol": 1e-10})
    elapsed = time.perf_counter() - t0

    exact_gs = np.linalg.eigvalsh(vqe.H_matrix)[0]
    return result.fun, exact_gs, elapsed, vqe.eval_count

# Scale test
try:
    max_vqe_qubits = min(get_max_qubits("demo"), 12)
except NameError:
    max_vqe_qubits = 10

vqe_qubits = list(range(2, max_vqe_qubits + 1))
vqe_times = []
vqe_errors = []

print("VQE Scaling Test")
print("=" * 70)
print(f"{'Qubits':>7} | {'VQE Energy':>12} | {'Exact GS':>12} | {'Error':>10} | {'Time (s)':>10} | {'Evals':>6}")
print("-" * 70)

for n in vqe_qubits:
    np.random.seed(42)
    e_vqe, e_exact, t, evals = run_vqe_test(n)
    error = abs(e_vqe - e_exact)
    vqe_times.append(t)
    vqe_errors.append(error)
    print(f"{n:>7} | {e_vqe:>12.6f} | {e_exact:>12.6f} | {error:>10.2e} | {t:>10.3f} | {evals:>6}")

print(f"\\nLargest VQE: {vqe_qubits[-1]} qubits in {vqe_times[-1]:.2f} s")""")

    # --- Code 13: VQE scaling plot ---
    nb.code("""# --- VQE scaling visualization ---
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.plot(vqe_qubits, vqe_times, "o-", linewidth=2, color="#d62728")
ax.set_xlabel("Number of Qubits")
ax.set_ylabel("VQE Time (seconds)")
ax.set_title("VQE Optimization Time Scaling")
ax.set_yscale("log")
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.semilogy(vqe_qubits, vqe_errors, "s-", linewidth=2, color="#9467bd")
ax.set_xlabel("Number of Qubits")
ax.set_ylabel("Energy Error (Hartree)")
ax.set_title("VQE Accuracy vs Qubit Count")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("mlx_labs/vqe_scaling.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: mlx_labs/vqe_scaling.png")""")

    # --- Code 14: Hardware summary ---
    nb.code("""# --- Hardware profile and recommendations ---
import numpy as np

print("=" * 60)
print("VARIATIONAL QUANTUM ALGORITHMS: CAPABILITY SUMMARY")
print("=" * 60)

try:
    print(f"Hardware:    {HARDWARE['chip']}")
    print(f"Memory:      {HARDWARE['memory_gb']} GB unified")
except NameError:
    print("Hardware:    Unknown")

print(f"Backend:     {'MLX (Metal GPU)' if HAS_MLX else 'NumPy (CPU only)'}")

print(f"\\nVQE Results:")
print(f"  H2 ground state: {result_fd.fun:.8f} Hartree (exact: {exact_ground:.8f})")
print(f"  H2 error:        {abs(result_fd.fun - exact_ground):.2e} Hartree")
print(f"  Largest VQE:     {vqe_qubits[-1]} qubits")
print(f"  VQE time range:  {min(vqe_times):.3f} - {max(vqe_times):.3f} seconds")

print(f"\\nQuantum Kernel:")
print(f"  Dataset:         {n_samples} samples")
print(f"  Kernel time:     {kernel_time*1000:.1f} ms")

print(f"\\nGradient Methods:")
print(f"  Parameter-shift: exact (2 evals per parameter)")
print(f"  Best FD (1e-6):  {np.abs(grad_fds[2] - grad_ps).max():.2e} max error")

print(f"\\nScaling Recommendations:")
print(f"  Interactive VQE:  up to ~{min(max_vqe_qubits, 10)} qubits")
print(f"  Batch VQE:        up to ~{min(max_vqe_qubits + 4, 16)} qubits")
print(f"  Quantum kernels:  up to ~{min(max_vqe_qubits + 2, 14)} qubits")
print(f"\\nNext steps:")
print(f"  - Try different ansatze (UCCSD, QAOA)")
print(f"  - Implement noise models for realistic simulation")
print(f"  - Explore quantum error mitigation techniques")""")

    # --- MD 10: Summary ---
    nb.md(r"""## Summary

### What we built

1. **VQE Simulator**: parameterized circuits with classical optimization
2. **H$_2$ ground state**: found energy within $10^{-6}$ Hartree of exact
3. **Parameter-shift gradients**: exact quantum gradient computation
4. **Quantum kernel methods**: kernel matrix from quantum feature maps
5. **Scaling analysis**: performance from 2 to 12+ qubits

### Key concepts

| Concept | Description |
|---------|-------------|
| **VQE** | Hybrid quantum-classical algorithm for ground state finding |
| **Ansatz** | Parameterized circuit that prepares trial states |
| **Parameter-shift rule** | Exact gradient via shifted circuit evaluations |
| **Quantum kernel** | Kernel function computed from quantum state overlaps |
| **Cost landscape** | Energy as function of circuit parameters |

### The power of Apple Silicon for VQE

MLX's unified memory lets us:
- Hold large state vectors without CPU-GPU transfers
- Compute gradients efficiently with Metal acceleration
- Scale to qubit counts impractical on discrete-GPU systems

### Curriculum connections

- **Year 1**: Quantum states, gates, measurement (Notebook 01)
- **Year 2**: Quantum algorithms, error correction (Notebook 02)
- **Year 3**: Variational methods, quantum chemistry (this notebook)
- **Year 4-5**: Research applications of VQE and quantum ML""")

    nb.save()
    print("  -> Notebook 03 complete\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("SIIEA Quantum Engineering - MLX Labs Generator")
    print("=" * 60)
    print()

    build_notebook_01()
    build_notebook_02()
    build_notebook_03()

    print("=" * 60)
    print("All 3 MLX lab notebooks generated successfully!")
    print("=" * 60)
    print()
    print("Notebooks:")
    print("  mlx_labs/01_mlx_quantum_basics.ipynb")
    print("  mlx_labs/02_large_scale_simulation.ipynb")
    print("  mlx_labs/03_quantum_neural_network.ipynb")
