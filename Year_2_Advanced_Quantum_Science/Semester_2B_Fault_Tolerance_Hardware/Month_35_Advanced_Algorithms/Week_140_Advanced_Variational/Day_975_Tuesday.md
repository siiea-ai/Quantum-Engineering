# Day 975: Symmetry-Preserving Ansatze

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Symmetry Constraints in VQE |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 975, you will be able to:

1. Identify relevant symmetries in molecular and many-body systems
2. Design ansatze that preserve particle number ($\hat{N}$) symmetry
3. Construct spin-preserving circuits ($\hat{S}^2$, $\hat{S}_z$)
4. Incorporate point group symmetries in molecular calculations
5. Compare hard constraints vs penalty methods for symmetry enforcement
6. Analyze the trainability benefits of symmetry-adapted circuits

---

## Core Content

### 1. Why Symmetry Matters

Physical systems possess symmetries that constrain allowed states. Ignoring these in VQE leads to:

1. **Wasted parameters:** Exploring unphysical states
2. **Optimization difficulties:** Spurious local minima
3. **Incorrect results:** Ground states in wrong symmetry sectors
4. **Inefficiency:** Larger circuits than necessary

**The Principle:**
> If $[H, S] = 0$ and $|\psi_0\rangle$ is in a symmetry sector, the true ground state is also in that sector.

**Symmetry Constraint:**
$$\boxed{[\hat{U}(\boldsymbol{\theta}), \hat{S}] = 0 \quad \forall \boldsymbol{\theta}}$$

This ensures the ansatz never leaves the correct symmetry sector.

---

### 2. Particle Number Conservation

For fermionic systems, particle number $\hat{N} = \sum_i a_i^\dagger a_i$ is conserved.

**Requirement:**
$$[\hat{U}(\boldsymbol{\theta}), \hat{N}] = 0$$

**Generators that Preserve $\hat{N}$:**

*Number-preserving operations:*
- Single excitations: $a_p^\dagger a_q - a_q^\dagger a_p$ (preserves $N$)
- Double excitations: $a_p^\dagger a_q^\dagger a_r a_s - \text{h.c.}$ (preserves $N$)

*Non-preserving operations (avoid these):*
- Creation: $a_p^\dagger$
- Annihilation: $a_p$
- Pair creation: $a_p^\dagger a_q^\dagger$

**Jordan-Wigner Encoding:**

After JW mapping:
$$a_j^\dagger a_i = \frac{1}{4}(X_i X_j + Y_i Y_j)(Z_{i+1} \cdots Z_{j-1}) + \frac{i}{4}(X_i Y_j - Y_i X_j)(Z_{i+1} \cdots Z_{j-1})$$

These preserve particle number when implemented correctly.

**Hardware Implementation:**

Number-preserving gates in the computational basis:

$$\text{iSWAP} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & i & 0 \\ 0 & i & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

$$\text{XY}(\theta) = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & \cos(\theta/2) & i\sin(\theta/2) & 0 \\ 0 & i\sin(\theta/2) & \cos(\theta/2) & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

Both preserve the subspace spanned by $\{|01\rangle, |10\rangle\}$.

---

### 3. Spin Symmetry

For electronic systems, spin operators:
$$\hat{S}^2 = \hat{S}_x^2 + \hat{S}_y^2 + \hat{S}_z^2$$
$$\hat{S}_z = \frac{1}{2}\sum_i (n_{i\uparrow} - n_{i\downarrow})$$

**$\hat{S}_z$ Conservation:**

Easier to enforce. Use operators that change $\alpha$ and $\beta$ spin equally:
- Paired excitations: $a_{p\alpha}^\dagger a_{q\alpha} + a_{p\beta}^\dagger a_{q\beta}$
- Spin-paired doubles: $a_{p\alpha}^\dagger a_{q\beta}^\dagger a_{r\beta} a_{s\alpha}$

**$\hat{S}^2$ Conservation:**

More challenging. Requires using spin-adapted operators:

$$E_{pq} = a_{p\alpha}^\dagger a_{q\alpha} + a_{p\beta}^\dagger a_{q\beta}$$
$$e_{pq,rs} = E_{pq}E_{rs} - \delta_{qr}E_{ps}$$

These generate a spin-complete representation.

**Singlet State Preservation:**

For closed-shell molecules (S=0):
$$\hat{S}^2 |\psi\rangle = 0$$

Use singlet-adapted excitations only.

---

### 4. Point Group Symmetries

Molecules have spatial symmetries (rotation, reflection, inversion).

**Example: H2O with C2v Symmetry**

| Irrep | Orbitals |
|-------|----------|
| A1 | 1s(O), 2s(O), σ bonding |
| A2 | (none in minimal basis) |
| B1 | π⊥ |
| B2 | π∥, σ antibonding |

**Symmetry-Adapted Operators:**

Only mix orbitals within the same irreducible representation:
- A1 → A1 allowed
- A1 → B1 forbidden
- B1 → B1 allowed

This dramatically reduces the operator pool.

**Pool Reduction Example:**

For N orbitals without symmetry: $O(N^4)$ double excitations
With point group symmetry: Often reduced by factor of 4-16

---

### 5. Symmetry-Adapted ADAPT-VQE

Combine ADAPT with symmetry constraints:

```
Symmetry-Adapted ADAPT-VQE:
────────────────────────────────────────
1. Identify symmetries: N, Sz, S², point group
2. Construct symmetry-adapted operator pool
3. Start with symmetry-correct reference
4. Standard ADAPT iteration (guaranteed to stay in sector)
────────────────────────────────────────
```

**Advantages:**
- Smaller operator pool
- Faster convergence
- Correct symmetry automatically
- Better optimization landscape

---

### 6. Hard Constraints vs Penalty Methods

**Hard Constraints:**

Enforce symmetry by construction:
$$[\hat{U}(\boldsymbol{\theta}), \hat{S}] = 0$$

*Pros:*
- Exact enforcement
- No tuning needed
- Reduced parameter space

*Cons:*
- May limit expressibility
- Complex circuit design
- Not all symmetries easy to encode

**Penalty Methods:**

Add symmetry violation to cost function:
$$C(\theta) = \langle H \rangle + \lambda \langle (\hat{S} - s)^2 \rangle$$

*Pros:*
- Easy to implement
- Works for any symmetry
- Flexible

*Cons:*
- Approximate enforcement
- Requires tuning $\lambda$
- Additional measurements

**Comparison:**

| Method | Enforcement | Implementation | Measurements |
|--------|-------------|----------------|--------------|
| Hard constraint | Exact | Complex circuits | Energy only |
| Penalty | Approximate | Simple circuits | Energy + symmetry |
| Projection | Post-process | Extra circuit | Both |

---

### 7. Symmetry-Verified Quantum Eigensolver (SVQE)

A hybrid approach combining VQE with symmetry verification:

1. Run standard VQE
2. Measure symmetry quantum numbers
3. Post-select or correct if violated
4. Iterate with feedback

**Symmetry Measurement:**

For particle number:
$$\langle \hat{N} \rangle = \sum_i \langle n_i \rangle$$

For $S_z$:
$$\langle \hat{S}_z \rangle = \frac{1}{2}\sum_i (\langle n_{i\uparrow} \rangle - \langle n_{i\downarrow} \rangle)$$

**Virtual Distillation for Symmetry:**

Prepare two copies, project onto symmetric subspace:
$$\rho_{\text{proj}} = \frac{P_S \rho P_S}{\text{Tr}(P_S \rho)}$$

---

## Practical Applications

### Application to Molecular Ground States

Consider BeH2 (linear, D∞h symmetry):

**Without Symmetry:**
- 14 spin orbitals
- ~500 double excitations in UCCSD

**With Symmetry (Σg ground state):**
- Restrict to Σg excitations
- ~50 relevant operators (10x reduction!)

**ADAPT-VQE Comparison:**

| Approach | Operators Selected | Final Error |
|----------|-------------------|-------------|
| No symmetry | 25 | 0.5 mHa |
| With symmetry | 12 | 0.3 mHa |

---

## Worked Examples

### Example 1: Number-Preserving Two-Qubit Gate

**Problem:** Design a parameterized gate that preserves particle number for 2 qubits.

**Solution:**

The computational basis states by particle number:
- $N=0$: $|00\rangle$
- $N=1$: $|01\rangle$, $|10\rangle$
- $N=2$: $|11\rangle$

A number-preserving gate must not mix these sectors. The most general form in the $N=1$ subspace:

$$U_{N=1}(\theta, \phi) = \begin{pmatrix} e^{i\phi}\cos(\theta/2) & e^{i\phi}i\sin(\theta/2) \\ e^{-i\phi}i\sin(\theta/2) & e^{-i\phi}\cos(\theta/2) \end{pmatrix}$$

In the full 4x4 space:
$$U_{\text{preserve}}(\theta) = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & \cos\theta & i\sin\theta & 0 \\ 0 & i\sin\theta & \cos\theta & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

This is the $XY(\theta)$ gate or fSWAP variant.

**Circuit Implementation:**
```
     ┌───┐                   ┌───┐
q0: ─┤RZ ├───●───────────────●───┤RZ†├
     └───┘   │               │   └───┘
     ┌───┐ ┌─┴─┐ ┌────────┐ ┌─┴─┐ ┌───┐
q1: ─┤RZ ├─┤ X ├─┤ RY(θ)  ├─┤ X ├─┤RZ†├
     └───┘ └───┘ └────────┘ └───┘ └───┘
```

---

### Example 2: Spin-Adapted Singles

**Problem:** Construct a spin-conserving single excitation from orbital $p$ to $q$.

**Solution:**

For a singlet state, we need paired excitations:

$$\hat{E}_{pq}^{(+)} = a_{p\alpha}^\dagger a_{q\alpha} + a_{p\beta}^\dagger a_{q\beta}$$
$$\hat{E}_{pq}^{(-)} = a_{p\alpha}^\dagger a_{q\alpha} - a_{p\beta}^\dagger a_{q\beta}$$

The spin-singlet generator:
$$\hat{T}_{pq} = \hat{E}_{pq}^{(+)} - \hat{E}_{qp}^{(+)}$$

This conserves both $\hat{N}$ and $\hat{S}_z$, and for closed-shell reference, also $\hat{S}^2$.

**JW Mapping (adjacent orbitals, $p = 2q$, $p+1 = 2q+1$):**
$$\hat{T}_{pq} \rightarrow \frac{1}{2}(X_p Y_q Z_{...} - Y_p X_q Z_{...} + X_{p+1} Y_{q+1} Z_{...} - Y_{p+1} X_{q+1} Z_{...})$$

---

### Example 3: Pool Reduction from Symmetry

**Problem:** For H2O in STO-3G (7 orbitals, 10 electrons), estimate pool reduction from C2v symmetry.

**Solution:**

**Without Symmetry:**
- Occupied: 5 orbitals
- Virtual: 2 orbitals
- Singles: 5 × 2 = 10
- Doubles: $\binom{5}{2} \times \binom{2}{2} = 10$ (occupied pairs to virtual pairs)

Actually for spin orbitals (14 total):
- Singles: ~20
- Doubles: ~200
- Total: ~220 operators

**With C2v Symmetry:**

Irrep decomposition:
- A1: 5 orbitals
- B1: 1 orbital
- B2: 1 orbital

Allowed transitions (same irrep only):
- A1→A1: Most orbitals
- B1→B1: 1→1 (trivial)
- B2→B2: 1→1 (trivial)

Reduced operators: ~50 (4x reduction)

**With Spin Symmetry (Singlet):**
Further reduction to spin-adapted: ~25 operators

**Total Reduction:** ~220 → ~25 (nearly 10x!)

---

## Practice Problems

### Level 1: Direct Application

1. For a 4-spin-orbital system, list all number-preserving single excitation operators.

2. Show that the iSWAP gate preserves particle number by verifying its action on all computational basis states.

3. If a molecule has C2 symmetry with orbitals split as A(3) + B(2), how many symmetry-allowed single excitations exist?

### Level 2: Intermediate

4. Derive the $XY(\theta)$ gate from the generator $\hat{G} = X_0 Y_1 - Y_0 X_1$ using $U = e^{-i\theta \hat{G}/2}$.

5. Design a circuit that prepares a singlet state $\frac{1}{\sqrt{2}}(|01\rangle - |10\rangle)$ starting from $|00\rangle$ using only number-preserving gates.

6. For the penalty method $C = \langle H \rangle + \lambda \langle (N - N_0)^2 \rangle$, derive the gradient with respect to a variational parameter.

### Level 3: Challenging

7. Prove that if $[\hat{U}, \hat{N}] = 0$ and $[\hat{U}, \hat{S}_z] = 0$, then $\hat{U}$ can be written as a product of generalized Givens rotations within each $(N, S_z)$ sector.

8. Design a symmetry-adapted operator pool for N2 (D∞h symmetry, Σg+ ground state) in a minimal basis (10 spatial orbitals).

9. **Research:** How do topological symmetries (like fermion parity) affect ansatz design for superconducting qubit implementations?

---

## Computational Lab

### Objective
Implement symmetry-preserving VQE for a simple system.

```python
"""
Day 975 Computational Lab: Symmetry-Preserving Ansatze
Advanced Variational Methods - Week 140
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# =============================================================================
# Part 1: Number-Preserving Gates
# =============================================================================

print("=" * 70)
print("Part 1: Number-Preserving Gates")
print("=" * 70)

def xy_gate(theta, wires):
    """
    XY gate: preserves particle number.
    Acts non-trivially only on |01⟩, |10⟩ subspace.
    """
    return qml.IsingXY(theta, wires=wires)

# Verify number preservation
dev_test = qml.device('default.qubit', wires=2)

@qml.qnode(dev_test)
def test_particle_number(theta, initial_state):
    """Prepare state and measure particle number."""
    # Prepare initial state
    if initial_state == '01':
        qml.PauliX(1)
    elif initial_state == '10':
        qml.PauliX(0)
    elif initial_state == '11':
        qml.PauliX(0)
        qml.PauliX(1)
    # else '00' - do nothing

    # Apply XY gate
    qml.IsingXY(theta, wires=[0, 1])

    # Measure number operator: N = (I-Z0)/2 + (I-Z1)/2 = I - (Z0+Z1)/2
    return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

print("\nTesting XY gate particle number preservation:")
print("-" * 50)

for state in ['00', '01', '10', '11']:
    for theta in [0, np.pi/4, np.pi/2, np.pi]:
        z0, z1 = test_particle_number(theta, state)
        n0 = (1 - z0) / 2
        n1 = (1 - z1) / 2
        N = n0 + n1
        N_initial = state.count('1')
        print(f"  |{state}⟩, θ={theta:.2f}: N = {N:.3f} (initial: {N_initial})")

# =============================================================================
# Part 2: Symmetry-Preserving Ansatz Construction
# =============================================================================

print("\n" + "=" * 70)
print("Part 2: Symmetry-Preserving Ansatz")
print("=" * 70)

n_qubits = 4

def number_preserving_ansatz(params, n_layers=2):
    """
    Build a particle-number preserving ansatz.
    Uses XY gates which preserve the {|01⟩, |10⟩} subspace on each pair.
    """
    param_idx = 0

    for layer in range(n_layers):
        # Even pairs: (0,1), (2,3), ...
        for i in range(0, n_qubits - 1, 2):
            qml.IsingXY(params[param_idx], wires=[i, i+1])
            param_idx += 1

        # Odd pairs: (1,2), (3,4), ...
        for i in range(1, n_qubits - 1, 2):
            qml.IsingXY(params[param_idx], wires=[i, i+1])
            param_idx += 1

    return param_idx  # Return number of parameters used

# Count parameters
n_layers = 3
n_params_per_layer = (n_qubits // 2) + ((n_qubits - 1) // 2)
n_params = n_layers * n_params_per_layer
print(f"Ansatz configuration:")
print(f"  Qubits: {n_qubits}")
print(f"  Layers: {n_layers}")
print(f"  Parameters per layer: {n_params_per_layer}")
print(f"  Total parameters: {n_params}")

# =============================================================================
# Part 3: VQE with Symmetry Preservation
# =============================================================================

print("\n" + "=" * 70)
print("Part 3: Symmetry-Preserving VQE")
print("=" * 70)

# Simple 4-qubit Hamiltonian (Heisenberg-like)
def create_heisenberg_hamiltonian(J=1.0, h=0.5):
    """Create a Heisenberg XXZ Hamiltonian."""
    coeffs = []
    ops = []

    # XX + YY + ZZ interactions
    for i in range(n_qubits - 1):
        coeffs.extend([J, J, J])
        ops.extend([
            qml.PauliX(i) @ qml.PauliX(i+1),
            qml.PauliY(i) @ qml.PauliY(i+1),
            qml.PauliZ(i) @ qml.PauliZ(i+1)
        ])

    # External field
    for i in range(n_qubits):
        coeffs.append(h)
        ops.append(qml.PauliZ(i))

    return qml.Hamiltonian(coeffs, ops)

H_heisenberg = create_heisenberg_hamiltonian()
print(f"Heisenberg Hamiltonian created")

# Ground state search with fixed particle number
dev = qml.device('default.qubit', wires=n_qubits)

@qml.qnode(dev)
def symmetry_preserving_vqe(params):
    """VQE with number-preserving ansatz."""
    # Initial state: |0011⟩ (N=2 sector)
    qml.PauliX(2)
    qml.PauliX(3)

    # Apply symmetry-preserving ansatz
    number_preserving_ansatz(params, n_layers=n_layers)

    return qml.expval(H_heisenberg)

@qml.qnode(dev)
def measure_particle_number(params):
    """Measure total particle number."""
    qml.PauliX(2)
    qml.PauliX(3)
    number_preserving_ansatz(params, n_layers=n_layers)

    # N = sum of (I - Zi)/2 = n/2 - sum(Zi)/2
    observables = [qml.PauliZ(i) for i in range(n_qubits)]
    return [qml.expval(obs) for obs in observables]

def cost_symmetry(params):
    return float(symmetry_preserving_vqe(pnp.array(params)))

def check_symmetry(params):
    z_vals = measure_particle_number(pnp.array(params))
    n_vals = [(1 - z) / 2 for z in z_vals]
    return sum(n_vals)

# Optimize
print("\nOptimizing with symmetry-preserving ansatz...")
x0 = np.random.uniform(-0.1, 0.1, n_params)
result = minimize(cost_symmetry, x0, method='COBYLA',
                 options={'maxiter': 300, 'rhobeg': 0.5})

print(f"\nResults:")
print(f"  Final energy: {result.fun:.6f}")
print(f"  Particle number: {check_symmetry(result.x):.4f} (should be 2.0)")

# =============================================================================
# Part 4: Comparison - With vs Without Symmetry
# =============================================================================

print("\n" + "=" * 70)
print("Part 4: Comparing With vs Without Symmetry")
print("=" * 70)

@qml.qnode(dev)
def generic_vqe(params):
    """VQE with generic hardware-efficient ansatz (no symmetry)."""
    qml.PauliX(2)
    qml.PauliX(3)

    param_idx = 0
    for layer in range(n_layers):
        for i in range(n_qubits):
            qml.RY(params[param_idx], wires=i)
            param_idx += 1
            qml.RZ(params[param_idx], wires=i)
            param_idx += 1
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i+1])

    return qml.expval(H_heisenberg)

@qml.qnode(dev)
def measure_particle_generic(params):
    """Measure N for generic ansatz."""
    qml.PauliX(2)
    qml.PauliX(3)

    param_idx = 0
    for layer in range(n_layers):
        for i in range(n_qubits):
            qml.RY(params[param_idx], wires=i)
            param_idx += 1
            qml.RZ(params[param_idx], wires=i)
            param_idx += 1
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i+1])

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

n_params_generic = n_layers * n_qubits * 2

def cost_generic(params):
    return float(generic_vqe(pnp.array(params)))

def check_symmetry_generic(params):
    z_vals = measure_particle_generic(pnp.array(params))
    n_vals = [(1 - z) / 2 for z in z_vals]
    return sum(n_vals)

print("Optimizing with generic ansatz (no symmetry)...")
x0_generic = np.random.uniform(-0.1, 0.1, n_params_generic)
result_generic = minimize(cost_generic, x0_generic, method='COBYLA',
                         options={'maxiter': 300, 'rhobeg': 0.5})

print(f"\nGeneric Ansatz Results:")
print(f"  Final energy: {result_generic.fun:.6f}")
print(f"  Particle number: {check_symmetry_generic(result_generic.x):.4f}")
print(f"  Parameters: {n_params_generic}")

print(f"\nSymmetry-Preserving Ansatz Results:")
print(f"  Final energy: {result.fun:.6f}")
print(f"  Particle number: {check_symmetry(result.x):.4f}")
print(f"  Parameters: {n_params}")

print(f"\nComparison:")
print(f"  Parameter ratio: {n_params}/{n_params_generic} = {n_params/n_params_generic:.2f}")
if abs(check_symmetry(result.x) - 2.0) < 0.01:
    print(f"  Symmetry preserved: YES")
else:
    print(f"  Symmetry preserved: NO")

# =============================================================================
# Part 5: Penalty Method Comparison
# =============================================================================

print("\n" + "=" * 70)
print("Part 5: Penalty Method for Symmetry")
print("=" * 70)

def cost_with_penalty(params, lambda_penalty=10.0):
    """Cost function with particle number penalty."""
    energy = cost_generic(params)
    N = check_symmetry_generic(params)
    penalty = lambda_penalty * (N - 2.0)**2
    return energy + penalty

print("Optimizing with penalty method...")
result_penalty = minimize(lambda p: cost_with_penalty(p, lambda_penalty=10.0),
                         x0_generic, method='COBYLA',
                         options={'maxiter': 300, 'rhobeg': 0.5})

N_final_penalty = check_symmetry_generic(result_penalty.x)
E_final_penalty = cost_generic(result_penalty.x)

print(f"\nPenalty Method Results:")
print(f"  Final energy: {E_final_penalty:.6f}")
print(f"  Particle number: {N_final_penalty:.4f}")
print(f"  Symmetry violation: {abs(N_final_penalty - 2.0):.4f}")

# =============================================================================
# Part 6: Visualization
# =============================================================================

print("\n" + "=" * 70)
print("Part 6: Visualization")
print("=" * 70)

# Compare methods
methods = ['Generic', 'Penalty', 'Hard Constraint']
energies = [result_generic.fun, E_final_penalty, result.fun]
particle_numbers = [check_symmetry_generic(result_generic.x),
                   N_final_penalty,
                   check_symmetry(result.x)]
params_count = [n_params_generic, n_params_generic, n_params]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Energy comparison
ax1 = axes[0]
colors = ['salmon', 'lightgreen', 'steelblue']
ax1.bar(methods, energies, color=colors, edgecolor='black')
ax1.set_ylabel('Final Energy')
ax1.set_title('Energy Comparison')
ax1.grid(True, alpha=0.3, axis='y')

# Particle number
ax2 = axes[1]
ax2.bar(methods, particle_numbers, color=colors, edgecolor='black')
ax2.axhline(y=2.0, color='red', linestyle='--', label='Target N=2')
ax2.set_ylabel('Particle Number')
ax2.set_title('Particle Number Conservation')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Parameter count
ax3 = axes[2]
ax3.bar(methods, params_count, color=colors, edgecolor='black')
ax3.set_ylabel('Number of Parameters')
ax3.set_title('Parameter Efficiency')
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('day_975_symmetry_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure saved as 'day_975_symmetry_comparison.png'")

print("\n" + "=" * 70)
print("Lab Complete!")
print("=" * 70)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Symmetry constraint | $[\hat{U}(\boldsymbol{\theta}), \hat{S}] = 0$ |
| Number operator | $\hat{N} = \sum_i a_i^\dagger a_i$ |
| Spin-z operator | $\hat{S}_z = \frac{1}{2}\sum_i (n_{i\uparrow} - n_{i\downarrow})$ |
| XY gate | $U = \exp(-i\theta(X_1X_2 + Y_1Y_2)/4)$ |
| Penalty cost | $C = \langle H \rangle + \lambda\langle(S - s_0)^2\rangle$ |

### Main Takeaways

1. **Symmetries constrain physical states** and should be respected by ansatze
2. **Number-preserving gates** (XY, iSWAP) maintain particle count exactly
3. **Spin symmetry** requires paired excitations or spin-adapted operators
4. **Point group symmetry** dramatically reduces operator pools
5. **Hard constraints** guarantee symmetry but require careful circuit design
6. **Penalty methods** are flexible but only approximate
7. **Symmetry-adapted ADAPT-VQE** combines the best of both approaches

---

## Daily Checklist

- [ ] Understand particle number conservation in circuits
- [ ] Derive spin-conserving excitation operators
- [ ] Work through all three examples
- [ ] Complete Level 1 practice problems
- [ ] Attempt at least one Level 2 problem
- [ ] Run and modify the computational lab
- [ ] Write a comparison table of hard constraints vs penalties

---

## Preview: Day 976

Tomorrow we explore **hardware-efficient ansatze**—circuit designs optimized for native device gates and connectivity rather than physical intuition. We'll analyze the trade-off between expressibility and trainability.

---

*"Symmetry is a key to nature's secrets."*
--- Adapted from Emmy Noether's legacy

---

**Next:** [Day_976_Wednesday.md](Day_976_Wednesday.md) - Hardware-Efficient Ansatze
