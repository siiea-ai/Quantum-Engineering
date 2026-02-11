# Day 603: Eigenvalue Problem in Quantum Computing

## Overview

**Day 603** | Week 87, Day 1 | Month 22 | Quantum Algorithms I

Today we introduce the eigenvalue problem in quantum computing and motivate why extracting eigenvalues from unitary operators is fundamentally important. Understanding eigenvalue extraction is key to Shor's algorithm, quantum simulation, and many quantum machine learning applications.

---

## Learning Objectives

1. Understand eigenvalues and eigenvectors of unitary operators
2. Recognize why eigenvalue extraction is important
3. Connect eigenvalues to physical observables
4. Understand the structure of unitary eigenvalues
5. Preview how phase estimation solves the eigenvalue problem
6. Identify applications requiring eigenvalue extraction

---

## Core Content

### Eigenvalues of Unitary Operators

A unitary operator $U$ satisfies $U^\dagger U = I$, preserving inner products.

**Key Property:** All eigenvalues of unitary operators have magnitude 1.

If $U|\psi\rangle = \lambda|\psi\rangle$, then:
$$\langle\psi|U^\dagger U|\psi\rangle = |\lambda|^2\langle\psi|\psi\rangle = \langle\psi|\psi\rangle$$

Therefore $|\lambda|^2 = 1$, so $|\lambda| = 1$.

**Consequence:** Every eigenvalue can be written as:
$$\boxed{\lambda = e^{2\pi i\phi} \text{ for some } \phi \in [0, 1)}$$

The **phase** $\phi$ completely characterizes the eigenvalue!

### The Phase Estimation Problem

**Problem:** Given:
- A unitary operator $U$ (as a quantum circuit)
- An eigenstate $|\psi\rangle$ with $U|\psi\rangle = e^{2\pi i\phi}|\psi\rangle$

Find the phase $\phi$ to $n$ bits of precision.

**Output:** An n-bit approximation $\tilde{\phi}$ such that:
$$|\phi - \tilde{\phi}| < \frac{1}{2^n}$$

### Why Eigenvalues Matter

**1. Energy Levels (Quantum Simulation)**

For Hamiltonian $H$, evolution operator $U = e^{-iHt/\hbar}$.

If $H|\psi\rangle = E|\psi\rangle$ (energy eigenstate), then:
$$U|\psi\rangle = e^{-iEt/\hbar}|\psi\rangle$$

The phase $\phi = Et/(2\pi\hbar)$ encodes the energy!

**2. Period Finding (Shor's Algorithm)**

For modular exponentiation $U_a|x\rangle = |ax \mod N\rangle$:

The eigenvalues encode the order $r$ of $a$ modulo $N$:
$$\text{eigenvalue} = e^{2\pi is/r}$$

Extracting $s/r$ leads to factoring!

**3. Linear Algebra (HHL Algorithm)**

For solving $Ax = b$:

If $A = \sum_j \lambda_j |v_j\rangle\langle v_j|$, the solution involves $\lambda_j^{-1}$.

Phase estimation finds $\lambda_j$, enabling efficient inversion.

### Spectral Decomposition

Any unitary $U$ has spectral decomposition:
$$U = \sum_{j} e^{2\pi i\phi_j}|\psi_j\rangle\langle\psi_j|$$

where $\{|\psi_j\rangle\}$ are orthonormal eigenvectors.

**Action on arbitrary state:**
$$U|v\rangle = \sum_j e^{2\pi i\phi_j} \langle\psi_j|v\rangle |\psi_j\rangle$$

### The Challenge

**Classical approach:** Diagonalize $U$ to find eigenvalues. For $N \times N$ matrix: $O(N^3)$ operations.

**Quantum approach:** Use phase estimation. For $n$-qubit system ($N = 2^n$): $O(\text{poly}(n))$ operations!

The quantum approach is exponentially faster for large systems.

### Eigenstates in Practice

**Challenge:** We often don't know the eigenstates $|\psi\rangle$!

**Solution:** Use superposition of eigenstates:
$$|v\rangle = \sum_j \alpha_j |\psi_j\rangle$$

Phase estimation on $|v\rangle$ gives:
- Eigenvalue $\phi_j$ with probability $|\alpha_j|^2$

This allows sampling from the spectrum!

### Connection to Phase Kickback

From Week 85: controlled-$U$ on eigenstate causes phase kickback:
$$CU|+\rangle|\psi\rangle = \frac{1}{\sqrt{2}}(|0\rangle + e^{2\pi i\phi}|1\rangle)|\psi\rangle$$

Phase estimation systematically extracts $\phi$ using multiple controlled operations.

---

## Worked Examples

### Example 1: Eigenvalues of Pauli Z

Find eigenvalues and eigenstates of $Z$.

**Solution:**

$Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$

Eigenvalue equation: $Z|\psi\rangle = \lambda|\psi\rangle$

For $|0\rangle$: $Z|0\rangle = |0\rangle$, so $\lambda_0 = 1 = e^{2\pi i \cdot 0}$, $\phi_0 = 0$

For $|1\rangle$: $Z|1\rangle = -|1\rangle$, so $\lambda_1 = -1 = e^{2\pi i \cdot 0.5}$, $\phi_1 = 0.5$

### Example 2: Eigenvalues of Hadamard

Find eigenvalues of $H$.

**Solution:**

$H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$

Characteristic equation: $\det(H - \lambda I) = 0$

$\det\begin{pmatrix} 1/\sqrt{2} - \lambda & 1/\sqrt{2} \\ 1/\sqrt{2} & -1/\sqrt{2} - \lambda \end{pmatrix} = 0$

$(1/\sqrt{2} - \lambda)(-1/\sqrt{2} - \lambda) - 1/2 = 0$

$\lambda^2 - 1 = 0$

$\lambda = \pm 1$

So $\lambda_+ = 1 = e^{0}$ ($\phi_+ = 0$) and $\lambda_- = -1 = e^{i\pi}$ ($\phi_- = 0.5$)

**Eigenvectors:**
- $|+_H\rangle = \cos(\pi/8)|0\rangle + \sin(\pi/8)|1\rangle$ for $\lambda = 1$
- $|-_H\rangle = \sin(\pi/8)|0\rangle - \cos(\pi/8)|1\rangle$ for $\lambda = -1$

### Example 3: Rotation Gate Eigenvalues

For $R_z(\theta) = \begin{pmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{pmatrix}$, find eigenvalues.

**Solution:**

$R_z$ is already diagonal!

Eigenvalues: $\lambda_0 = e^{-i\theta/2}$, $\lambda_1 = e^{i\theta/2}$

For $\theta = \pi/4$:
- $\lambda_0 = e^{-i\pi/8}$, so $\phi_0 = -1/16 \equiv 15/16 \pmod{1}$
- $\lambda_1 = e^{i\pi/8}$, so $\phi_1 = 1/16$

### Example 4: Modular Multiplication

For $U_7|x\rangle = |7x \mod 15\rangle$ on 4 qubits, find an eigenvalue.

**Solution:**

The order of 7 mod 15: $7^1 = 7$, $7^2 = 49 = 4$, $7^3 = 28 = 13$, $7^4 = 91 = 1$

So $r = 4$ (period 4).

Eigenstates have form:
$$|u_s\rangle = \frac{1}{2}\sum_{j=0}^{3} e^{-2\pi ijs/4}|7^j \mod 15\rangle$$

For $s = 1$:
$$|u_1\rangle = \frac{1}{2}(|1\rangle + e^{-i\pi/2}|7\rangle + e^{-i\pi}|4\rangle + e^{-3i\pi/2}|13\rangle)$$
$$= \frac{1}{2}(|1\rangle - i|7\rangle - |4\rangle + i|13\rangle)$$

Eigenvalue: $e^{2\pi i \cdot 1/4} = e^{i\pi/2} = i$

Phase: $\phi = 1/4 = 0.25$

---

## Practice Problems

### Problem 1: CNOT Eigenvalues

Find all eigenvalues of the CNOT gate and express them as $e^{2\pi i\phi}$.

### Problem 2: Two-Qubit Operator

For $U = Z \otimes X$, find the eigenvalues and eigenstates.

### Problem 3: Evolution Operator

A Hamiltonian has eigenvalues $E_0 = 0$ and $E_1 = \hbar\omega$. For $U = e^{-iHt}$, what are the eigenvalue phases at time $t = \pi/(2\omega)$?

### Problem 4: Superposition Input

If phase estimation is run on $|v\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$ with $U = Z$, what outcomes are possible and with what probabilities?

---

## Computational Lab

```python
"""Day 603: Eigenvalue Problem in Quantum Computing"""
import numpy as np
from scipy.linalg import eig

# Define common gates
def pauli_z():
    return np.array([[1, 0], [0, -1]])

def pauli_x():
    return np.array([[0, 1], [1, 0]])

def hadamard():
    return np.array([[1, 1], [1, -1]]) / np.sqrt(2)

def cnot():
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ])

def rz(theta):
    return np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)]
    ])

def extract_phases(U):
    """
    Extract eigenvalues and convert to phases
    Returns: list of (phase, eigenvector) tuples
    """
    eigenvalues, eigenvectors = eig(U)

    results = []
    for i, (val, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
        # Convert eigenvalue to phase
        phase = np.angle(val) / (2 * np.pi)
        if phase < 0:
            phase += 1
        results.append((phase, vec / np.linalg.norm(vec)))

    return results

def verify_eigenvalue(U, eigenvalue, eigenvector):
    """Verify that U|ψ⟩ = λ|ψ⟩"""
    Uv = U @ eigenvector
    expected = eigenvalue * eigenvector
    return np.allclose(Uv, expected)

# Test on various gates
print("=" * 60)
print("EIGENVALUE ANALYSIS OF QUANTUM GATES")
print("=" * 60)

gates = [
    ("Pauli Z", pauli_z()),
    ("Pauli X", pauli_x()),
    ("Hadamard", hadamard()),
    ("R_z(π/4)", rz(np.pi/4)),
    ("CNOT", cnot()),
]

for name, U in gates:
    print(f"\n--- {name} ---")

    # Verify unitary
    is_unitary = np.allclose(U @ U.conj().T, np.eye(len(U)))
    print(f"Unitary: {is_unitary}")

    # Extract phases
    phases = extract_phases(U)
    print("Eigenvalues (as phases):")
    for phase, eigvec in phases:
        eigenvalue = np.exp(2j * np.pi * phase)
        print(f"  φ = {phase:.4f} → λ = e^(2πi·{phase:.4f}) = {eigenvalue:.4f}")

        # Verify
        is_valid = verify_eigenvalue(U, eigenvalue, eigvec)
        print(f"    Verified: {is_valid}")

# Demonstrate eigenstate decomposition
print("\n" + "=" * 60)
print("EIGENSTATE DECOMPOSITION")
print("=" * 60)

U = pauli_z()
phases = extract_phases(U)

# Create a superposition
psi = np.array([0.8, 0.6])  # Not an eigenstate

print(f"\nInput state: {psi}")
print("\nDecomposition into eigenstates:")

for i, (phase, eigvec) in enumerate(phases):
    # Component in this eigenstate
    coeff = np.vdot(eigvec, psi)
    prob = abs(coeff)**2
    print(f"  |ψ_{i}⟩: coefficient = {coeff:.4f}, probability = {prob:.4f}")
    print(f"    Phase if measured: φ = {phase:.4f}")

# Modular exponentiation example
print("\n" + "=" * 60)
print("MODULAR EXPONENTIATION EIGENVALUES")
print("=" * 60)

def mod_exp_matrix(a, N, n_qubits):
    """
    Create matrix for U_a|x⟩ = |ax mod N⟩
    Only acts properly on x < N
    """
    dim = 2**n_qubits
    U = np.zeros((dim, dim))
    for x in range(dim):
        if x < N:
            y = (a * x) % N
            U[y, x] = 1
        else:
            U[x, x] = 1  # Leave states ≥ N unchanged
    return U

# Example: a = 7, N = 15
a, N = 7, 15
n_qubits = 4
U_mod = mod_exp_matrix(a, N, n_qubits)

print(f"\nU_{a}|x⟩ = |{a}x mod {N}⟩")

# Find order
r = 1
while pow(a, r, N) != 1:
    r += 1
print(f"Order of {a} mod {N}: r = {r}")

# Expected eigenvalues: e^{2πis/r} for s = 0, 1, ..., r-1
print(f"\nExpected eigenvalue phases: s/r for s = 0, 1, ..., {r-1}")
for s in range(r):
    print(f"  s = {s}: φ = {s}/{r} = {s/r:.4f}")

# Numerically find eigenvalues
phases = extract_phases(U_mod)

print(f"\nNumerically computed phases (non-trivial):")
for phase, eigvec in sorted(phases, key=lambda x: x[0]):
    # Check if phase is close to s/r for some s
    for s in range(r):
        if abs(phase - s/r) < 0.01 or abs(phase - s/r - 1) < 0.01:
            print(f"  φ = {phase:.4f} ≈ {s}/{r}")
            break

# Phase estimation preview
print("\n" + "=" * 60)
print("PHASE ESTIMATION PREVIEW")
print("=" * 60)

def simple_phase_estimation(U, psi, n_ancilla):
    """
    Simplified phase estimation simulation
    Returns probability distribution over phase estimates
    """
    # Get eigendecomposition
    eigenvalues, eigenvectors = eig(U)
    phases = [np.angle(val) / (2 * np.pi) % 1 for val in eigenvalues]

    # Decompose psi into eigenstates
    coefficients = [np.vdot(eigvec, psi) for eigvec in eigenvectors.T]

    # Phase estimation output distribution
    N_ancilla = 2**n_ancilla
    probs = np.zeros(N_ancilla)

    for phase, coeff in zip(phases, coefficients):
        prob_this_eigenstate = abs(coeff)**2
        if prob_this_eigenstate < 1e-10:
            continue

        # Ideal outcome for this phase
        ideal_k = phase * N_ancilla

        # Distribution around ideal outcome
        for k in range(N_ancilla):
            # Use sinc-like distribution
            diff = k - ideal_k
            if abs(diff) < 1e-10:
                p = 1.0
            else:
                p = (np.sin(np.pi * diff) / (N_ancilla * np.sin(np.pi * diff / N_ancilla)))**2
            probs[k] += prob_this_eigenstate * p

    return probs / probs.sum()

# Demo with Z gate on superposition
U = pauli_z()
psi = np.array([1, 1]) / np.sqrt(2)  # |+⟩

print(f"\nPhase estimation on Z gate with |+⟩ input:")
print(f"Expected: 50% chance of φ=0 (eigenstate |0⟩)")
print(f"          50% chance of φ=0.5 (eigenstate |1⟩)")

for n_ancilla in [2, 3, 4]:
    probs = simple_phase_estimation(U, psi, n_ancilla)
    N = 2**n_ancilla

    print(f"\nn = {n_ancilla} ancilla qubits:")
    for k in range(N):
        if probs[k] > 0.01:
            phase_est = k / N
            print(f"  k = {k:2d}: φ_est = {phase_est:.4f}, P = {probs[k]:.4f}")

# Summary
print("\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)
print("""
1. UNITARY EIGENVALUES are always on the unit circle: λ = e^{2πiφ}
2. THE PHASE φ fully characterizes the eigenvalue
3. PHASE ESTIMATION extracts φ to n bits of precision
4. SUPERPOSITION INPUTS give samples from the eigenvalue spectrum
5. APPLICATIONS include:
   - Quantum simulation (energy levels)
   - Shor's algorithm (period finding)
   - HHL algorithm (matrix inversion)
   - Quantum machine learning
""")
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Unitary eigenvalue | $\lambda = e^{2\pi i\phi}$, $\|\lambda\| = 1$ |
| Eigenvalue equation | $U\|\psi\rangle = e^{2\pi i\phi}\|\psi\rangle$ |
| Spectral decomposition | $U = \sum_j e^{2\pi i\phi_j}\|\psi_j\rangle\langle\psi_j\|$ |
| Superposition result | Measure $\phi_j$ with probability $\|\langle\psi_j\|v\rangle\|^2$ |

### Key Takeaways

1. **Unitary eigenvalues** lie on the unit circle
2. **Phase** completely characterizes eigenvalue
3. **Phase estimation** extracts phase efficiently
4. **Superposition inputs** sample the spectrum
5. **Many algorithms** reduce to eigenvalue extraction

---

## Daily Checklist

- [ ] I understand unitary eigenvalue structure
- [ ] I can compute phases from eigenvalues
- [ ] I know applications of eigenvalue extraction
- [ ] I understand superposition input behavior
- [ ] I see the connection to phase kickback
- [ ] I ran the lab and explored eigenvalue decomposition

---

*Next: Day 604 - QPE Circuit Design*
