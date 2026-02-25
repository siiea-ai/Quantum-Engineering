# Day 961: First-Order Trotter-Suzuki Formula

## Schedule Overview

| Time Block | Duration | Focus |
|------------|----------|-------|
| Morning | 3.5 hours | Lie-Trotter derivation and error analysis |
| Afternoon | 2.5 hours | Problem solving and circuit implementation |
| Evening | 1 hour | Computational lab: Trotter circuits |

## Learning Objectives

By the end of today, you will be able to:

1. Derive the Lie-Trotter formula from the Baker-Campbell-Hausdorff expansion
2. Prove the first-order error bound in terms of commutators
3. Compile Trotter steps into quantum circuits for local Hamiltonians
4. Calculate the optimal number of Trotter steps for a target precision
5. Implement Trotter simulation for the Ising and Heisenberg models
6. Analyze the trade-offs between Trotter error and gate count

## Core Content

### 1. The Lie-Trotter Formula

The fundamental challenge of Hamiltonian simulation is implementing $e^{-iHt}$ when $H = A + B$ and $[A, B] \neq 0$. The Lie-Trotter formula provides the solution:

$$\boxed{e^{-i(A+B)t} = \lim_{n \to \infty} \left(e^{-iAt/n} e^{-iBt/n}\right)^n}$$

This is also called the **product formula** or **Trotter decomposition**.

#### Why Does This Work?

For finite $n$, we have:

$$\left(e^{-iAt/n} e^{-iBt/n}\right)^n \approx e^{-i(A+B)t}$$

The approximation becomes exact as $n \to \infty$ because each step is a small perturbation.

#### Historical Context

- **Sophus Lie** (1875): Developed the theory of continuous transformation groups
- **Hale Trotter** (1959): Proved the formula for operator semigroups
- **Seth Lloyd** (1996): Applied it to quantum simulation, proving universality

---

### 2. Baker-Campbell-Hausdorff Analysis

The error comes from the **Baker-Campbell-Hausdorff (BCH) formula**:

$$e^A e^B = e^{A + B + \frac{1}{2}[A,B] + \frac{1}{12}[A,[A,B]] - \frac{1}{12}[B,[A,B]] + \cdots}$$

For small operators $A = -iA't/n$ and $B = -iB't/n$:

$$e^{-iAt/n} e^{-iBt/n} = e^{-i(A+B)t/n + O(t^2/n^2)}$$

The leading error term is:

$$\boxed{\frac{1}{2}\left[\frac{-iAt}{n}, \frac{-iBt}{n}\right] = -\frac{[A,B]t^2}{2n^2}}$$

#### Accumulated Error Over n Steps

Taking $n$ products, the total error accumulates:

$$\left(e^{-iAt/n} e^{-iBt/n}\right)^n = e^{-i(A+B)t} \cdot e^{O(t^2/n)}$$

The error scales as $O(t^2/n)$, or equivalently:

$$\boxed{\text{Trotter error} = O\left(\frac{\|[A,B]\| t^2}{n}\right)}$$

---

### 3. Rigorous Error Bounds

**Theorem (First-Order Trotter Error):**

For Hermitian operators $A$ and $B$:

$$\left\| e^{-i(A+B)t} - \left(e^{-iAt/n} e^{-iBt/n}\right)^n \right\| \leq \frac{\|[A,B]\| t^2}{2n}$$

**Proof Sketch:**

1. Define $U(s) = e^{-iAs} e^{-iBs}$ for one Trotter step of time $s = t/n$.

2. Expand using Taylor series:
   $$U(s) = I - i(A+B)s - \frac{1}{2}(A^2 + AB + BA + B^2)s^2 + O(s^3)$$

3. The exact evolution is:
   $$e^{-i(A+B)s} = I - i(A+B)s - \frac{1}{2}(A+B)^2 s^2 + O(s^3)$$

4. The difference is:
   $$U(s) - e^{-i(A+B)s} = -\frac{1}{2}(AB + BA - A^2 - 2AB - B^2)s^2 + O(s^3)$$
   $$= -\frac{1}{2}[A,B]s^2 + O(s^3)$$

5. For $n$ steps with $s = t/n$:
   $$\text{Total error} \leq n \cdot \frac{\|[A,B]\| (t/n)^2}{2} = \frac{\|[A,B]\| t^2}{2n}$$

$\square$

---

### 4. Extension to Many Terms

For a Hamiltonian with multiple terms:

$$H = \sum_{j=1}^{L} H_j$$

The first-order Trotter formula becomes:

$$\boxed{e^{-iHt} \approx \left(\prod_{j=1}^{L} e^{-iH_j t/n}\right)^n}$$

#### Error for L Terms

**Theorem:** For $L$ terms with bounded norms and commutators:

$$\left\| e^{-iHt} - \left(\prod_{j=1}^{L} e^{-iH_j t/n}\right)^n \right\| \leq \frac{L^2 \Lambda^2 t^2}{2n}$$

where $\Lambda = \max_j \|H_j\|$.

A tighter bound uses the commutator structure:

$$\text{Error} \leq \frac{t^2}{2n} \sum_{j < k} \|[H_j, H_k]\|$$

---

### 5. Circuit Compilation for Local Hamiltonians

Each Trotter step requires implementing $e^{-iH_j t/n}$ for local terms.

#### Single-Qubit Terms

For $H_j = \alpha P_j$ where $P_j \in \{X, Y, Z\}$:

$$e^{-i\alpha P_j s} = \cos(\alpha s) I - i\sin(\alpha s) P_j$$

**Circuit implementations:**

| Term | Circuit |
|------|---------|
| $e^{-i\theta Z}$ | $R_z(2\theta)$ |
| $e^{-i\theta X}$ | $R_x(2\theta)$ |
| $e^{-i\theta Y}$ | $R_y(2\theta)$ |

#### Two-Qubit ZZ Interaction

For $H_{jk} = \alpha Z_j Z_k$:

$$e^{-i\alpha Z_j Z_k s}$$

**Circuit:**

```
q_j: ──●──────────●──
       │          │
q_k: ──X──Rz(2αs)─X──
```

This uses the identity:
$$\text{CNOT}_{jk} \cdot (I \otimes R_z(\theta)) \cdot \text{CNOT}_{jk} = e^{-i\theta Z_j Z_k / 2}$$

#### Two-Qubit XX Interaction

For $H_{jk} = \alpha X_j X_k$:

**Circuit:**

```
q_j: ──H──●──────────●──H──
          │          │
q_k: ──H──X──Rz(2αs)─X──H──
```

Uses the transformation $H Z H = X$ on each qubit.

#### Two-Qubit YY Interaction

For $H_{jk} = \alpha Y_j Y_k$:

**Circuit:**

```
q_j: ──Sdg──H──●──────────●──H──S──
               │          │
q_k: ──Sdg──H──X──Rz(2αs)─X──H──S──
```

Uses $S^\dagger H Z H S = Y$.

---

### 6. Gate Count Analysis

For a Hamiltonian with $L$ terms, each being $k$-local:

| Component | Gates per Term | Total per Step |
|-----------|----------------|----------------|
| Single-qubit terms | 1 rotation | $O(1)$ |
| Two-qubit terms | 2 CNOTs + 1 rotation | $O(1)$ |
| **Per Trotter step** | | $O(L)$ |
| **$n$ Trotter steps** | | $O(nL)$ |

#### Choosing n for Target Precision

To achieve error $\leq \epsilon$:

$$\frac{C t^2}{n} \leq \epsilon \implies n \geq \frac{C t^2}{\epsilon}$$

where $C = \frac{1}{2}\sum_{j<k}\|[H_j, H_k]\|$.

**Total gate count:**

$$\boxed{N_{\text{gates}} = O\left(\frac{L \cdot C \cdot t^2}{\epsilon}\right) = O\left(\frac{L \cdot \Lambda^2 \cdot t^2}{\epsilon}\right)}$$

This is the **first-order Trotter complexity**: quadratic in time $t$.

---

### 7. Ordering Effects and Optimization

The order of terms in the Trotter step matters:

$$\prod_{j=1}^{L} e^{-iH_j s} \neq \prod_{j=L}^{1} e^{-iH_j s}$$

#### Randomized Ordering (qDRIFT)

**qDRIFT** (Campbell, 2019) randomly samples terms:

1. Choose term $H_j$ with probability $p_j = |h_j| / \lambda$
2. Apply $e^{\pm i \lambda s H_j / |h_j|}$ (sign matching $h_j$)
3. Repeat $N$ times

This achieves error $O(\lambda^2 t^2 / N)$ but with better constant factors.

#### Commuting Groups

If terms can be grouped into sets of mutually commuting terms:

$$H = H_{\text{odd}} + H_{\text{even}}$$

where all terms in $H_{\text{odd}}$ commute with each other, then:

$$e^{-iH_{\text{odd}} t} = \prod_{j \in \text{odd}} e^{-iH_j t}$$

exactly! This reduces the effective number of non-commuting groups.

---

### 8. Connection to Digital vs. Analog Simulation

**Digital (Trotter) simulation:**
- Discretize time into $n$ steps
- Each step uses finite gate set
- Error controlled by $n$

**Analog simulation:**
- Continuous evolution under engineered Hamiltonian
- No Trotter error
- Limited to specific Hamiltonian forms

The Trotter approach is **universal** but incurs discretization error. Hybrid approaches combine analog blocks with digital corrections.

---

## Worked Examples

### Example 1: Trotter Step for Transverse-Field Ising

**Problem:** Derive the Trotter circuit for one step of the 2-qubit transverse-field Ising model:
$$H = -J Z_0 Z_1 - h(X_0 + X_1)$$

**Solution:**

Step 1: Identify terms.
- $H_1 = -J Z_0 Z_1$ (ZZ interaction)
- $H_2 = -h X_0$ (transverse field on qubit 0)
- $H_3 = -h X_1$ (transverse field on qubit 1)

Step 2: Exponentials for each term with time step $s = t/n$.
- $e^{iJ Z_0 Z_1 s}$: Use CNOT-Rz-CNOT with angle $\theta = -2Js$
- $e^{ih X_0 s}$: Single $R_x(2hs)$ gate
- $e^{ih X_1 s}$: Single $R_x(2hs)$ gate

Step 3: Construct circuit.

```
q_0: ──●──────────────●──Rx(2hs)──
       │              │
q_1: ──X──Rz(-2Js)────X──Rx(2hs)──
```

Step 4: Repeat $n$ times for full evolution.

**Gate count per Trotter step:** 2 CNOTs + 3 single-qubit rotations = 5 gates.

**Total for $n$ steps:** $5n$ gates.

$\square$

---

### Example 2: Error Bound Calculation

**Problem:** Calculate the number of Trotter steps needed to simulate $H = X + Z$ for time $t = 10$ with error $\epsilon = 0.01$.

**Solution:**

Step 1: Compute the commutator.
$$[X, Z] = XZ - ZX = 2iY$$
$$\|[X, Z]\| = 2$$

Step 2: Apply the error bound.
$$\text{Error} \leq \frac{\|[X, Z]\| t^2}{2n} = \frac{2 \cdot 100}{2n} = \frac{100}{n}$$

Step 3: Solve for $n$.
$$\frac{100}{n} \leq 0.01$$
$$n \geq 10000$$

Step 4: Verify.
With $n = 10000$ Trotter steps:
- Each step has time $s = 10/10000 = 0.001$
- Error $\leq 100/10000 = 0.01$ ✓

**Answer:** Need at least $n = 10000$ Trotter steps.

Note: This is a pessimistic bound. In practice, the actual error is often smaller.

$\square$

---

### Example 3: Heisenberg XXX Circuit

**Problem:** Write the Trotter step circuit for the nearest-neighbor Heisenberg interaction:
$$H_{12} = J(X_1 X_2 + Y_1 Y_2 + Z_1 Z_2)$$

**Solution:**

Step 1: Recognize the structure.
The three terms $XX$, $YY$, $ZZ$ all commute with each other! (Proven in Day 960)

Therefore:
$$e^{-iH_{12} t} = e^{-iJ X_1 X_2 t} \cdot e^{-iJ Y_1 Y_2 t} \cdot e^{-iJ Z_1 Z_2 t}$$

exactly, with no Trotter error.

Step 2: Circuit for each term (with $\theta = Jt$).

**XX term:**
```
q_1: ──H──●─────────────●──H──
          │             │
q_2: ──H──X──Rz(2θ)─────X──H──
```

**YY term:**
```
q_1: ──Sdg──H──●─────────────●──H──S──
               │             │
q_2: ──Sdg──H──X──Rz(2θ)─────X──H──S──
```

**ZZ term:**
```
q_1: ──●─────────────●──
       │             │
q_2: ──X──Rz(2θ)─────X──
```

Step 3: Total circuit.
Combine all three (order doesn't matter since they commute):

```
q_1: ─H──●────────●──H───Sdg─H──●────────●──H─S───●────────●──
         │        │             │        │        │        │
q_2: ─H──X─Rz(2θ)─X──H───Sdg─H──X─Rz(2θ)─X──H─S───X─Rz(2θ)─X──
```

**Gate count:** 6 CNOTs + 3 Rz + 4 H + 4 S/Sdg = 17 gates per bond.

$\square$

---

## Practice Problems

### Level 1: Direct Application

1. **Basic Trotter:** Compute the first-order Trotter approximation for $e^{-i(X+Y)t}$ with $n=2$ steps. Express as a sequence of rotation gates.

2. **Error calculation:** For $H = 0.5 Z_1 Z_2 + 0.3 X_1$, calculate $\|[H_1, H_2]\|$ and the first-order Trotter error bound for $t=5$, $n=100$.

3. **Gate count:** A 10-qubit 1D Ising chain has 9 ZZ terms and 10 X terms. How many gates are in one Trotter step? How many for $n=1000$ steps?

### Level 2: Intermediate Analysis

4. **Optimized ordering:** For the XY model $H = \sum_i (X_i X_{i+1} + Y_i Y_{i+1})$, identify groups of commuting terms and design an efficient Trotter step.

5. **Tight error bound:** Show that for $H = \sum_{j=1}^L h_j H_j$ with $\|H_j\| = 1$, the tighter error bound is:
   $$\text{Error} \leq \frac{t^2}{2n} \sum_{j<k} |h_j h_k| \|[H_j, H_k]\|$$

6. **Commutator structure:** Prove that for the Heisenberg interaction $H = X_1 X_2 + Y_1 Y_2 + Z_1 Z_2$, all pairwise commutators vanish.

### Level 3: Challenging Problems

7. **Randomized Trotter (qDRIFT):** Prove that random ordering of Trotter terms with probability proportional to $|h_j|$ achieves error $O(\lambda^2 t^2 / N)$ where $N$ is the number of random gates.

8. **Beyond worst case:** For the transverse-field Ising model, compute the actual commutator sum and compare to the naive $L^2$ bound. When is the improvement significant?

9. **Compilation depth:** Design a parallel Trotter step for a 2D square lattice Ising model that minimizes circuit depth by exploiting the checkerboard structure (even/odd sites).

---

## Computational Lab: First-Order Trotter Circuits

### Lab Objective

Implement first-order Trotter circuits for physical Hamiltonians and analyze the error scaling.

```python
"""
Day 961 Lab: First-Order Trotter-Suzuki Simulation
Week 138: Quantum Simulation
"""

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Operator, state_fidelity, Statevector
from typing import List, Tuple

# =============================================================
# Part 1: Basic Trotter Building Blocks
# =============================================================

def rz_gate(theta: float) -> np.ndarray:
    """Rz rotation matrix."""
    return np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)]
    ])

def rx_gate(theta: float) -> np.ndarray:
    """Rx rotation matrix."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([
        [c, -1j * s],
        [-1j * s, c]
    ])

def exp_zz(theta: float) -> np.ndarray:
    """
    Compute exp(-i * theta * Z tensor Z).
    """
    Z = np.array([[1, 0], [0, -1]])
    ZZ = np.kron(Z, Z)
    return expm(-1j * theta * ZZ)

def exp_xx(theta: float) -> np.ndarray:
    """Compute exp(-i * theta * X tensor X)."""
    X = np.array([[0, 1], [1, 0]])
    XX = np.kron(X, X)
    return expm(-1j * theta * XX)

print("=" * 60)
print("Part 1: Trotter Building Blocks")
print("=" * 60)

# Verify ZZ gate decomposition
theta = 0.5
ZZ_exact = exp_zz(theta)

# Circuit decomposition: CNOT - Rz - CNOT
CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]])
Rz_2theta = np.kron(np.eye(2), rz_gate(2 * theta))
ZZ_circuit = CNOT @ Rz_2theta @ CNOT

print(f"\nZZ gate verification (theta = {theta}):")
print(f"  ||exact - circuit|| = {np.linalg.norm(ZZ_exact - ZZ_circuit):.2e}")

# =============================================================
# Part 2: Trotter Circuit for Ising Model
# =============================================================

def trotter_ising_step(n_qubits: int, J: float, h: float, dt: float) -> QuantumCircuit:
    """
    Create one Trotter step for the transverse-field Ising model.

    H = -J * sum_i Z_i Z_{i+1} - h * sum_i X_i
    """
    qc = QuantumCircuit(n_qubits)

    # ZZ interactions: exp(i * J * dt * Z_i Z_{i+1})
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
        qc.rz(2 * J * dt, i + 1)  # Note: +J because H has -J
        qc.cx(i, i + 1)

    # X field: exp(i * h * dt * X_i)
    for i in range(n_qubits):
        qc.rx(2 * h * dt, i)

    return qc

def trotter_simulation(n_qubits: int, J: float, h: float,
                       total_time: float, n_steps: int) -> QuantumCircuit:
    """
    Full Trotter simulation circuit.
    """
    qc = QuantumCircuit(n_qubits)
    dt = total_time / n_steps

    # Apply n_steps Trotter steps
    step = trotter_ising_step(n_qubits, J, h, dt)
    for _ in range(n_steps):
        qc.compose(step, inplace=True)

    return qc

print("\n" + "=" * 60)
print("Part 2: Ising Model Trotter Circuit")
print("=" * 60)

n_qubits = 3
J, h = 1.0, 0.5
total_time = 1.0
n_steps = 10

qc_trotter = trotter_simulation(n_qubits, J, h, total_time, n_steps)
print(f"\nTrotter circuit for {n_qubits}-qubit Ising model:")
print(f"  Total time: {total_time}")
print(f"  Trotter steps: {n_steps}")
print(f"  Gates in circuit: {qc_trotter.size()}")
print(f"  Circuit depth: {qc_trotter.depth()}")

# =============================================================
# Part 3: Error Analysis
# =============================================================

def ising_hamiltonian_matrix(n_qubits: int, J: float, h: float) -> np.ndarray:
    """Build the Ising Hamiltonian matrix."""
    dim = 2**n_qubits
    H = np.zeros((dim, dim), dtype=complex)

    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Z = np.array([[1, 0], [0, -1]])

    # ZZ terms
    for i in range(n_qubits - 1):
        term = np.eye(1)
        for j in range(n_qubits):
            if j == i or j == i + 1:
                term = np.kron(term, Z)
            else:
                term = np.kron(term, I)
        H -= J * term

    # X terms
    for i in range(n_qubits):
        term = np.eye(1)
        for j in range(n_qubits):
            if j == i:
                term = np.kron(term, X)
            else:
                term = np.kron(term, I)
        H -= h * term

    return H

def exact_evolution(H: np.ndarray, t: float) -> np.ndarray:
    """Compute exact time evolution operator."""
    return expm(-1j * H * t)

print("\n" + "=" * 60)
print("Part 3: Error Analysis")
print("=" * 60)

H = ising_hamiltonian_matrix(n_qubits, J, h)
U_exact = exact_evolution(H, total_time)

# Compare Trotter approximations with different step counts
step_counts = [1, 2, 5, 10, 20, 50, 100, 200]
errors = []

for n in step_counts:
    qc = trotter_simulation(n_qubits, J, h, total_time, n)
    U_trotter = Operator(qc).data
    error = np.linalg.norm(U_exact - U_trotter, ord=2)
    errors.append(error)
    print(f"  n = {n:3d}: error = {error:.6f}")

# Plot error scaling
plt.figure(figsize=(10, 6))
plt.loglog(step_counts, errors, 'bo-', linewidth=2, markersize=8, label='Actual error')

# Theoretical 1/n scaling
n_theory = np.array(step_counts)
error_theory = errors[0] * step_counts[0] / n_theory
plt.loglog(step_counts, error_theory, 'r--', linewidth=2, label=r'$O(1/n)$ reference')

plt.xlabel('Number of Trotter Steps', fontsize=12)
plt.ylabel('Operator Norm Error', fontsize=12)
plt.title('First-Order Trotter Error Scaling', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.savefig('day_961_trotter_error.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================
# Part 4: Time Evolution Fidelity
# =============================================================

print("\n" + "=" * 60)
print("Part 4: State Evolution Fidelity")
print("=" * 60)

# Initial state: |000...0>
psi0 = np.zeros(2**n_qubits, dtype=complex)
psi0[0] = 1.0

# Exact final state
psi_exact = U_exact @ psi0

# Trotter final states
fidelities = []
for n in step_counts:
    qc = trotter_simulation(n_qubits, J, h, total_time, n)
    U_trotter = Operator(qc).data
    psi_trotter = U_trotter @ psi0
    fid = np.abs(np.vdot(psi_exact, psi_trotter))**2
    fidelities.append(fid)
    print(f"  n = {n:3d}: fidelity = {fid:.8f}")

# Plot fidelity
plt.figure(figsize=(10, 6))
plt.semilogx(step_counts, fidelities, 'go-', linewidth=2, markersize=8)
plt.axhline(y=0.99, color='r', linestyle='--', label='99% threshold')
plt.axhline(y=0.999, color='orange', linestyle='--', label='99.9% threshold')
plt.xlabel('Number of Trotter Steps', fontsize=12)
plt.ylabel('State Fidelity', fontsize=12)
plt.title('Trotter Simulation Fidelity', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim([0.9, 1.001])
plt.savefig('day_961_trotter_fidelity.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================
# Part 5: Heisenberg Model Trotter Circuit
# =============================================================

def trotter_heisenberg_step(n_qubits: int, J: float, dt: float) -> QuantumCircuit:
    """
    Create one Trotter step for the Heisenberg XXX model.

    H = J * sum_i (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})
    """
    qc = QuantumCircuit(n_qubits)
    theta = J * dt

    for i in range(n_qubits - 1):
        # XX interaction
        qc.h(i)
        qc.h(i + 1)
        qc.cx(i, i + 1)
        qc.rz(2 * theta, i + 1)
        qc.cx(i, i + 1)
        qc.h(i)
        qc.h(i + 1)

        # YY interaction
        qc.sdg(i)
        qc.sdg(i + 1)
        qc.h(i)
        qc.h(i + 1)
        qc.cx(i, i + 1)
        qc.rz(2 * theta, i + 1)
        qc.cx(i, i + 1)
        qc.h(i)
        qc.h(i + 1)
        qc.s(i)
        qc.s(i + 1)

        # ZZ interaction
        qc.cx(i, i + 1)
        qc.rz(2 * theta, i + 1)
        qc.cx(i, i + 1)

    return qc

print("\n" + "=" * 60)
print("Part 5: Heisenberg Model Trotter Circuit")
print("=" * 60)

qc_heisenberg = trotter_heisenberg_step(n_qubits, J=1.0, dt=0.1)
print(f"Heisenberg Trotter step ({n_qubits} qubits):")
print(f"  Gates: {qc_heisenberg.size()}")
print(f"  Depth: {qc_heisenberg.depth()}")

# =============================================================
# Part 6: Commutator Analysis
# =============================================================

print("\n" + "=" * 60)
print("Part 6: Commutator Analysis for Error Bound")
print("=" * 60)

def pauli_matrix(pauli: str, n_qubits: int, position: int) -> np.ndarray:
    """Create Pauli matrix at given position."""
    I = np.eye(2)
    paulis = {'I': I,
              'X': np.array([[0, 1], [1, 0]]),
              'Y': np.array([[0, -1j], [1j, 0]]),
              'Z': np.array([[1, 0], [0, -1]])}

    result = np.eye(1)
    for i in range(n_qubits):
        if i == position:
            result = np.kron(result, paulis[pauli])
        else:
            result = np.kron(result, I)
    return result

def compute_commutator_norm(A: np.ndarray, B: np.ndarray) -> float:
    """Compute ||[A, B]||."""
    comm = A @ B - B @ A
    return np.linalg.norm(comm, ord=2)

# Compute commutator sum for Ising model
n = n_qubits
H_terms = []

# ZZ terms
for i in range(n - 1):
    H_terms.append(-J * pauli_matrix('Z', n, i) @ pauli_matrix('Z', n, i + 1))

# X terms
for i in range(n):
    H_terms.append(-h * pauli_matrix('X', n, i))

print(f"Ising model with {n} qubits:")
print(f"  Number of terms: {len(H_terms)}")

# Compute commutator sum
comm_sum = 0.0
for i, H_i in enumerate(H_terms):
    for j, H_j in enumerate(H_terms):
        if i < j:
            norm = compute_commutator_norm(H_i, H_j)
            if norm > 1e-10:
                comm_sum += norm

print(f"  Sum of ||[H_i, H_j]||: {comm_sum:.4f}")

# Theoretical error bound
t = total_time
n_steps_target = 100
error_bound = comm_sum * t**2 / (2 * n_steps_target)
print(f"\nTheoretical error bound (n={n_steps_target}, t={t}):")
print(f"  Error <= {error_bound:.6f}")

# Compare with actual error
qc = trotter_simulation(n_qubits, J, h, total_time, n_steps_target)
U_trotter = Operator(qc).data
actual_error = np.linalg.norm(U_exact - U_trotter, ord=2)
print(f"  Actual error: {actual_error:.6f}")
print(f"  Bound / Actual: {error_bound / actual_error:.2f}x")

# =============================================================
# Part 7: Visualization of Dynamics
# =============================================================

print("\n" + "=" * 60)
print("Part 7: Time Evolution Dynamics")
print("=" * 60)

# Simulate dynamics with Trotter
times = np.linspace(0, 5, 50)
n_trotter = 50  # Trotter steps per unit time

# Observables
magnetizations_exact = []
magnetizations_trotter = []

psi0 = np.zeros(2**n_qubits, dtype=complex)
psi0[0] = 1.0

# Magnetization operator
M_z = sum(pauli_matrix('Z', n_qubits, i) for i in range(n_qubits)) / n_qubits

for t in times:
    # Exact evolution
    U_exact_t = exact_evolution(H, t)
    psi_exact_t = U_exact_t @ psi0
    mag_exact = np.real(np.vdot(psi_exact_t, M_z @ psi_exact_t))
    magnetizations_exact.append(mag_exact)

    # Trotter evolution
    n_steps_t = max(1, int(n_trotter * t))
    if t > 0:
        qc = trotter_simulation(n_qubits, J, h, t, n_steps_t)
        U_trotter_t = Operator(qc).data
        psi_trotter_t = U_trotter_t @ psi0
        mag_trotter = np.real(np.vdot(psi_trotter_t, M_z @ psi_trotter_t))
    else:
        mag_trotter = mag_exact
    magnetizations_trotter.append(mag_trotter)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(times, magnetizations_exact, 'b-', linewidth=2, label='Exact')
plt.plot(times, magnetizations_trotter, 'r--', linewidth=2, label='Trotter')
plt.xlabel('Time', fontsize=12)
plt.ylabel(r'$\langle M_z \rangle$', fontsize=12)
plt.title('Magnetization Dynamics', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
error_dynamics = np.abs(np.array(magnetizations_exact) - np.array(magnetizations_trotter))
plt.plot(times, error_dynamics, 'g-', linewidth=2)
plt.xlabel('Time', fontsize=12)
plt.ylabel('|Exact - Trotter|', fontsize=12)
plt.title('Observable Error Growth', fontsize=14)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_961_dynamics.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nLab complete!")
print("Figures saved: day_961_trotter_error.png, day_961_trotter_fidelity.png, day_961_dynamics.png")
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Lie-Trotter formula | $e^{-i(A+B)t} = \lim_{n\to\infty} (e^{-iAt/n}e^{-iBt/n})^n$ |
| First-order error | $O(\|[A,B]\| t^2 / n)$ |
| Multi-term Trotter | $({\prod_j e^{-iH_j t/n}})^n$ |
| ZZ circuit | CNOT - Rz$(2\theta)$ - CNOT |
| XX circuit | H$\otimes$H - CNOT - Rz - CNOT - H$\otimes$H |
| Required steps | $n \geq Ct^2/\epsilon$ |
| Total gates | $O(L \cdot n) = O(Lt^2/\epsilon)$ |

### Key Takeaways

1. **Lie-Trotter formula** decomposes non-commuting exponentials into products of simpler terms.

2. **First-order error** scales as $O(t^2/n)$, requiring many steps for long times.

3. **Circuit compilation** maps each Pauli term to a standard gate sequence (CNOT + rotations).

4. **Commutator structure** determines actual error; non-commuting terms contribute.

5. **Trade-off:** More Trotter steps reduce error but increase circuit depth and gate count.

6. **Commuting groups** can be exponentiatedexactly, reducing effective non-commutativity.

---

## Daily Checklist

- [ ] I can derive the Lie-Trotter formula from BCH expansion
- [ ] I understand the first-order error bound and its dependence on commutators
- [ ] I can compile Trotter steps for ZZ, XX, YY interactions
- [ ] I can calculate the required Trotter steps for a target precision
- [ ] I completed the computational lab and verified error scaling
- [ ] I understand the trade-offs in Trotter simulation

---

## Preview of Day 962

Tomorrow we explore **higher-order product formulas** that dramatically improve the error scaling. We will:

- Derive the second-order (symmetric) Suzuki formula with $O(t^3/n^2)$ error
- Construct recursive higher-order formulas with $O((t/n)^{2k+1})$ error
- Analyze the trade-off between order and gate count
- Implement randomized Trotter (qDRIFT) for improved practical performance
- Compare asymptotic complexity of different product formula orders

Higher-order formulas are essential for practical quantum simulation, enabling much longer evolution times with fewer Trotter steps.

---

*"The art of doing mathematics consists in finding that special case which contains all the germs of generality."*
*— David Hilbert*

---

**Next:** [Day_962_Wednesday.md](Day_962_Wednesday.md) - Higher-Order Product Formulas
