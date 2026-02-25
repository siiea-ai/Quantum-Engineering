# Day 974: Adaptive Ansatz Construction (ADAPT-VQE)

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: ADAPT-VQE Algorithm |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 974, you will be able to:

1. Explain the limitations of fixed-structure ansatze in VQE
2. Derive the gradient-based operator selection criterion in ADAPT-VQE
3. Construct fermionic and qubit operator pools
4. Implement the ADAPT-VQE algorithm for molecular systems
5. Analyze convergence properties and circuit depth scaling
6. Compare ADAPT-VQE against UCCSD and hardware-efficient approaches

---

## Core Content

### 1. The Problem with Fixed Ansatze

Traditional VQE approaches use predetermined circuit structures:

**Unitary Coupled Cluster (UCC):**
$$|\psi_{\text{UCC}}\rangle = e^{T - T^\dagger} |\text{HF}\rangle$$

where $T = T_1 + T_2 + \ldots$ includes single, double, and higher excitations.

**Limitations of Fixed Ansatze:**

1. **Over-parameterization:** UCCSD includes all singles and doubles, even those with negligible contribution
2. **Circuit depth:** Full UCCSD requires $O(N^4)$ parameters for $N$ orbitals
3. **Hardware constraints:** Deep circuits accumulate errors
4. **Problem agnostic:** Same structure regardless of molecular specifics

**The Key Question:**
> Can we build ansatze that grow organically based on the problem at hand?

---

### 2. The ADAPT-VQE Algorithm

Introduced by Grimsley et al. (2019), ADAPT-VQE builds the ansatz iteratively by selecting operators from a pool based on their energy gradient magnitude.

**Core Idea:**
$$|\psi_{n+1}\rangle = e^{\theta_{n+1} A_{n+1}} |\psi_n\rangle$$

At each iteration, we choose the operator $A_{n+1}$ that has the largest gradient with respect to the current energy.

**The Selection Criterion:**

The gradient of energy with respect to a new operator $A_k$ at $\theta_k = 0$ is:

$$\boxed{\frac{\partial E}{\partial \theta_k}\bigg|_{\theta_k=0} = \langle \psi_n | [H, A_k] | \psi_n \rangle}$$

This is the expectation value of the commutator $[H, A_k]$!

**Derivation:**

Consider adding operator $A_k$ with parameter $\theta_k$:
$$|\psi(\theta_k)\rangle = e^{\theta_k A_k} |\psi_n\rangle$$

The energy is:
$$E(\theta_k) = \langle \psi_n | e^{-\theta_k A_k} H e^{\theta_k A_k} | \psi_n \rangle$$

Taking the derivative:
$$\frac{\partial E}{\partial \theta_k} = \langle \psi_n | e^{-\theta_k A_k} [H, A_k] e^{\theta_k A_k} | \psi_n \rangle$$

At $\theta_k = 0$:
$$\frac{\partial E}{\partial \theta_k}\bigg|_{\theta_k=0} = \langle \psi_n | [H, A_k] | \psi_n \rangle$$

---

### 3. Algorithm Pseudocode

```
ADAPT-VQE Algorithm:
────────────────────────────────────────
Input: Hamiltonian H, operator pool {A_k}, threshold ε
Output: Optimized state |ψ⟩ and energy E

1. Initialize |ψ_0⟩ = |HF⟩ (Hartree-Fock reference)
2. Set operator list L = []

3. REPEAT:
   a. For each A_k in pool:
      Compute gradient g_k = |⟨ψ_n|[H, A_k]|ψ_n⟩|

   b. Find k* = argmax_k |g_k|

   c. If max|g_k| < ε:
      CONVERGED - break

   d. Append A_{k*} to L with new parameter θ_{n+1}

   e. Optimize ALL parameters {θ_1, ..., θ_{n+1}}:
      E = min_θ ⟨ψ(θ)|H|ψ(θ)⟩

   f. n ← n + 1

4. Return |ψ_n⟩, E
────────────────────────────────────────
```

---

### 4. Operator Pool Design

The operator pool determines what operations are available for selection.

**Fermionic Operator Pool (Original ADAPT):**

For a system with $N$ orbitals and $\eta$ electrons:

*Single excitations:*
$$A_{pq} = a_p^\dagger a_q - a_q^\dagger a_p$$

*Double excitations:*
$$A_{pqrs} = a_p^\dagger a_q^\dagger a_r a_s - a_s^\dagger a_r^\dagger a_q a_p$$

Pool size: $O(N^2)$ singles + $O(N^4)$ doubles

**Qubit-ADAPT Pool:**

After Jordan-Wigner transformation, use individual Pauli strings:

$$\{X_i, Y_i, X_iY_j, Y_iX_j, X_iZ_jY_k, \ldots\}$$

Pool size: $O(\text{poly}(N))$ — typically much smaller!

**Advantages of Qubit-ADAPT:**
- Smaller pool → faster gradient screening
- Direct hardware mapping
- Often comparable accuracy with fewer parameters

---

### 5. Convergence Analysis

**Convergence Criterion:**

ADAPT-VQE converges when:
$$\max_k |\langle \psi_n | [H, A_k] | \psi_n \rangle| < \epsilon$$

This means no operator in the pool can significantly lower the energy.

**Convergence Properties:**

1. **Monotonic energy decrease:** Each iteration can only lower the energy
2. **Compact circuits:** Only necessary operators are included
3. **Problem-adapted:** Circuit structure reflects molecular physics

**Typical Scaling:**

| System | UCCSD Parameters | ADAPT-VQE Parameters | Ratio |
|--------|------------------|----------------------|-------|
| H2 | 3 | 1-2 | 0.5x |
| LiH | 56 | 8-15 | 0.2x |
| BeH2 | 156 | 20-40 | 0.2x |
| H2O | 448 | 50-80 | 0.15x |

---

### 6. Comparison with Other Ansatze

**UCCSD vs ADAPT-VQE:**

| Aspect | UCCSD | ADAPT-VQE |
|--------|-------|-----------|
| Structure | Fixed | Adaptive |
| Parameters | $O(N^4)$ | Problem-dependent |
| Depth | Very deep | Typically shorter |
| Accuracy | Chemical accuracy | Same or better |
| Optimization | Single round | Iterative |

**Hardware-Efficient vs ADAPT-VQE:**

| Aspect | Hardware-Efficient | ADAPT-VQE |
|--------|-------------------|-----------|
| Design | Hardware-matched | Problem-matched |
| Expressibility | High | Constrained |
| Trainability | Barren plateaus | Generally better |
| Physical insight | None | Chemical meaning |

---

### 7. Practical Considerations

**Gradient Measurement Cost:**

Each gradient requires measuring $\langle [H, A_k] \rangle$. For a Hamiltonian with $M$ terms:

$$\text{Measurements per iteration} = O(M \times |\text{pool}|)$$

**Strategies to Reduce Cost:**

1. **Pool pruning:** Remove operators with consistently small gradients
2. **Grouping:** Measure compatible Pauli terms together
3. **Lazy evaluation:** Only re-evaluate gradients for likely candidates
4. **Threshold screening:** Skip operators below a threshold

**Stopping Criteria:**

1. Gradient threshold: $\max |g_k| < \epsilon$ (standard)
2. Energy threshold: $|E_n - E_{n-1}| < \delta$
3. Maximum iterations: Prevent infinite loops
4. Parameter count limit: Hardware constraints

---

## Practical Applications

### Application to Molecular Systems

ADAPT-VQE has been successfully applied to:

1. **Small molecules:** H2, LiH, BeH2, H2O
2. **Excited states:** Generalized to excited state calculations
3. **Periodic systems:** Extension to solid-state problems
4. **Spin systems:** Quantum magnetism applications

**Advantages in Practice:**

- Significantly shallower circuits than UCCSD
- Natural incorporation of chemical intuition
- Systematic improvement path
- Compatible with error mitigation

---

## Worked Examples

### Example 1: ADAPT-VQE for H2

**Problem:** Apply ADAPT-VQE to the hydrogen molecule at equilibrium.

**Solution:**

The H2 Hamiltonian in STO-3G basis (4 spin-orbitals, 2 electrons):

After Jordan-Wigner mapping:
$$H = g_0 I + g_1 Z_0 + g_2 Z_1 + g_3 Z_0Z_1 + g_4 (X_0X_1 + Y_0Y_1) + \ldots$$

**Operator Pool (minimal):**
$$\{A_1 = Y_0X_1X_2X_3, A_2 = X_0Y_1X_2X_3, \ldots\}$$

**Iteration 1:**
- Start with |HF⟩ = |0011⟩
- Compute gradients: $g_1 = |\langle \text{HF}|[H, A_1]|\text{HF}\rangle|$
- Select $A^*$ with maximum gradient
- Optimize: $E_1 = \min_{\theta_1} \langle \text{HF}|e^{-\theta_1 A^*} H e^{\theta_1 A^*}|\text{HF}\rangle$

For H2, typically **one double excitation** suffices:
$$|\psi\rangle = e^{\theta(a_2^\dagger a_3^\dagger a_1 a_0 - \text{h.c.})} |\text{HF}\rangle$$

**Result:** Chemical accuracy with 1-2 operators vs 3 in UCCSD.

---

### Example 2: Gradient Calculation

**Problem:** Compute the gradient $\langle \psi | [H, A] | \psi \rangle$ for a simple case.

**Solution:**

Let $H = Z_0 + X_0X_1$ and $A = Y_0Z_1$.

Compute the commutator:
$$[H, A] = [Z_0, Y_0Z_1] + [X_0X_1, Y_0Z_1]$$

Using $[Z, Y] = 2iX$ and $[X, Y] = 2iZ$:
$$[Z_0, Y_0Z_1] = [Z_0, Y_0]Z_1 = 2iX_0Z_1$$

$$[X_0X_1, Y_0Z_1] = X_0[X_1, Z_1]Y_0 + [X_0, Y_0]X_1Z_1$$
$$= -2iX_0Y_1Y_0 + 2iZ_0X_1Z_1$$

The gradient is:
$$g = \langle \psi | (2iX_0Z_1 - 2iX_0Y_0Y_1 + 2iZ_0X_1Z_1) | \psi \rangle$$

This involves measuring three Pauli strings on the current state.

---

### Example 3: Pool Size Comparison

**Problem:** Compare fermionic vs qubit pool sizes for a 4-orbital system.

**Solution:**

**Fermionic Pool:**
- Singles: $\binom{4}{1} \times \binom{4}{1} = 16$ (minus redundant)
- Doubles: $\binom{4}{2} \times \binom{4}{2} = 36$ (minus redundant)
- After symmetry: ~10 singles + ~20 doubles = ~30 operators

**Qubit-ADAPT Pool:**
- 8 spin-orbitals after JW mapping
- Single-qubit: $3 \times 8 = 24$ (X, Y, Z on each)
- Two-qubit: Selected products
- Typical: ~50-100 operators (but simpler circuits each)

**Trade-off:**
- Fermionic: Fewer operators, deeper circuits per operator
- Qubit: More operators, shallower circuits per operator

---

## Practice Problems

### Level 1: Direct Application

1. Given the operator pool $\{X_0Y_1, Y_0X_1, Z_0Z_1\}$ and Hamiltonian $H = Z_0 + Z_1 + X_0X_1$, compute the commutators $[H, A_k]$ for each pool operator.

2. For a 2-qubit system, write out the Hartree-Fock state for 2 electrons filling the lowest energy orbitals.

3. If ADAPT-VQE converges with 5 operators and each operator has 1 parameter, how many function evaluations are needed for one COBYLA iteration (assuming numerical gradients)?

### Level 2: Intermediate

4. Derive the parameter shift rule for computing $\partial E/\partial \theta$ for an operator $A = iP$ where $P$ is a Pauli string with $P^2 = I$.

5. Given convergence threshold $\epsilon = 10^{-3}$ and gradient measurements with shot noise $\sigma = 10^{-2}$, estimate the number of shots needed to reliably detect convergence.

6. Design a minimal operator pool for LiH that respects particle number conservation.

### Level 3: Challenging

7. Prove that if the operator pool spans the Lie algebra generated by $H$ and the initial state, ADAPT-VQE can in principle reach any state in the reachable set.

8. Analyze the computational complexity of ADAPT-VQE: How does the total number of gradient evaluations scale with system size for typical molecules?

9. **Research:** How can ADAPT-VQE be modified to find excited states? Investigate the "orthogonally constrained VQE" approach.

---

## Computational Lab

### Objective
Implement ADAPT-VQE for the H2 molecule using PennyLane.

```python
"""
Day 974 Computational Lab: ADAPT-VQE Implementation
Advanced Variational Methods - Week 140
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# =============================================================================
# Part 1: Define the H2 Hamiltonian
# =============================================================================

print("=" * 70)
print("Part 1: H2 Hamiltonian Setup")
print("=" * 70)

# H2 Hamiltonian coefficients at equilibrium (R = 0.74 Angstrom)
# Using Jordan-Wigner mapping with STO-3G basis
# Simplified 2-qubit representation after tapering

h2_coeffs = {
    'II': -0.04207897,
    'IZ': 0.17771287,
    'ZI': 0.17771287,
    'ZZ': 0.17059738,
    'XX': 0.04475014,
    'YY': 0.04475014
}

def create_hamiltonian():
    """Create the H2 Hamiltonian as a PennyLane Hamiltonian."""
    coeffs = []
    ops = []

    for pauli_string, coeff in h2_coeffs.items():
        coeffs.append(coeff)
        if pauli_string == 'II':
            ops.append(qml.Identity(0))
        else:
            obs_list = []
            for i, p in enumerate(pauli_string):
                if p == 'I':
                    continue
                elif p == 'X':
                    obs_list.append(qml.PauliX(i))
                elif p == 'Y':
                    obs_list.append(qml.PauliY(i))
                elif p == 'Z':
                    obs_list.append(qml.PauliZ(i))

            if len(obs_list) == 1:
                ops.append(obs_list[0])
            else:
                ops.append(obs_list[0] @ obs_list[1])

    return qml.Hamiltonian(coeffs, ops)

H = create_hamiltonian()
print(f"H2 Hamiltonian created with {len(h2_coeffs)} terms")
print(f"Exact ground state energy: -1.136 Hartree")

# =============================================================================
# Part 2: Define Operator Pool
# =============================================================================

print("\n" + "=" * 70)
print("Part 2: Operator Pool for ADAPT-VQE")
print("=" * 70)

# For 2 qubits, define a pool of generators
# These are Pauli strings that generate rotations

operator_pool = [
    ('Y0', lambda theta: qml.RY(theta, wires=0)),
    ('Y1', lambda theta: qml.RY(theta, wires=1)),
    ('Y0Z1', lambda theta: [qml.CNOT(wires=[1, 0]),
                            qml.RY(theta, wires=0),
                            qml.CNOT(wires=[1, 0])]),
    ('Z0Y1', lambda theta: [qml.CNOT(wires=[0, 1]),
                            qml.RY(theta, wires=1),
                            qml.CNOT(wires=[0, 1])]),
    ('X0X1', lambda theta: [qml.RY(np.pi/2, wires=0),
                            qml.RY(np.pi/2, wires=1),
                            qml.CNOT(wires=[0, 1]),
                            qml.RZ(theta, wires=1),
                            qml.CNOT(wires=[0, 1]),
                            qml.RY(-np.pi/2, wires=0),
                            qml.RY(-np.pi/2, wires=1)]),
    ('Y0Y1', lambda theta: [qml.RX(-np.pi/2, wires=0),
                            qml.RX(-np.pi/2, wires=1),
                            qml.CNOT(wires=[0, 1]),
                            qml.RZ(theta, wires=1),
                            qml.CNOT(wires=[0, 1]),
                            qml.RX(np.pi/2, wires=0),
                            qml.RX(np.pi/2, wires=1)]),
]

print(f"Operator pool size: {len(operator_pool)}")
for name, _ in operator_pool:
    print(f"  - {name}")

# =============================================================================
# Part 3: ADAPT-VQE Implementation
# =============================================================================

print("\n" + "=" * 70)
print("Part 3: ADAPT-VQE Algorithm")
print("=" * 70)

dev = qml.device('default.qubit', wires=2)

class ADAPTVQESolver:
    """ADAPT-VQE solver for molecular systems."""

    def __init__(self, hamiltonian, operator_pool, n_qubits=2):
        self.H = hamiltonian
        self.pool = operator_pool
        self.n_qubits = n_qubits
        self.selected_ops = []
        self.parameters = []
        self.energy_history = []
        self.gradient_history = []

    def compute_gradient(self, op_index, current_params):
        """
        Compute gradient of energy with respect to new operator.
        Uses parameter shift rule.
        """
        # Create circuit with current operators plus new one at epsilon
        @qml.qnode(dev)
        def energy_circuit(params, new_param):
            # Hartree-Fock initial state |00>
            # (for 2 electrons in bonding orbital)

            # Apply current operators
            for i, (name, op_func) in enumerate(self.selected_ops):
                result = op_func(params[i])
                if isinstance(result, list):
                    for gate in result:
                        qml.apply(gate)
                else:
                    qml.apply(result)

            # Apply new operator
            name, op_func = self.pool[op_index]
            result = op_func(new_param)
            if isinstance(result, list):
                for gate in result:
                    qml.apply(gate)
            else:
                qml.apply(result)

            return qml.expval(self.H)

        # Parameter shift rule
        shift = np.pi / 2
        if len(current_params) > 0:
            params = pnp.array(current_params, requires_grad=False)
        else:
            params = pnp.array([], requires_grad=False)

        E_plus = energy_circuit(params, shift)
        E_minus = energy_circuit(params, -shift)

        gradient = (E_plus - E_minus) / 2
        return gradient

    def compute_all_gradients(self, current_params):
        """Compute gradients for all operators in pool."""
        gradients = []
        for i in range(len(self.pool)):
            g = self.compute_gradient(i, current_params)
            gradients.append(abs(g))
        return gradients

    def optimize_parameters(self):
        """Optimize all current parameters."""
        if len(self.selected_ops) == 0:
            return []

        @qml.qnode(dev)
        def cost_circuit(params):
            for i, (name, op_func) in enumerate(self.selected_ops):
                result = op_func(params[i])
                if isinstance(result, list):
                    for gate in result:
                        qml.apply(gate)
                else:
                    qml.apply(result)
            return qml.expval(self.H)

        def cost(params):
            return float(cost_circuit(pnp.array(params)))

        x0 = self.parameters if len(self.parameters) > 0 else [0.0]
        result = minimize(cost, x0, method='COBYLA',
                         options={'maxiter': 100, 'rhobeg': 0.5})
        return list(result.x)

    def run(self, max_iterations=10, gradient_threshold=1e-3):
        """Run ADAPT-VQE algorithm."""
        print("\nStarting ADAPT-VQE...")
        print("-" * 50)

        for iteration in range(max_iterations):
            # Compute all gradients
            gradients = self.compute_all_gradients(self.parameters)
            max_grad = max(gradients)
            max_idx = gradients.index(max_grad)

            self.gradient_history.append(max_grad)

            print(f"\nIteration {iteration + 1}:")
            print(f"  Max gradient: {max_grad:.6f}")
            print(f"  Selected operator: {self.pool[max_idx][0]}")

            # Check convergence
            if max_grad < gradient_threshold:
                print(f"\n  CONVERGED! Max gradient below threshold.")
                break

            # Add operator
            self.selected_ops.append(self.pool[max_idx])
            self.parameters.append(0.0)

            # Optimize
            self.parameters = self.optimize_parameters()

            # Compute energy
            @qml.qnode(dev)
            def final_energy(params):
                for i, (name, op_func) in enumerate(self.selected_ops):
                    result = op_func(params[i])
                    if isinstance(result, list):
                        for gate in result:
                            qml.apply(gate)
                    else:
                        qml.apply(result)
                return qml.expval(self.H)

            energy = float(final_energy(pnp.array(self.parameters)))
            self.energy_history.append(energy)

            print(f"  Energy: {energy:.6f} Ha")
            print(f"  Parameters: {self.parameters}")
            print(f"  Circuit depth: {len(self.selected_ops)} operators")

        return energy, self.parameters

# Run ADAPT-VQE
solver = ADAPTVQESolver(H, operator_pool)
final_energy, final_params = solver.run(max_iterations=5, gradient_threshold=1e-4)

# =============================================================================
# Part 4: Analysis and Visualization
# =============================================================================

print("\n" + "=" * 70)
print("Part 4: Results Analysis")
print("=" * 70)

exact_energy = -1.1361894539
print(f"\nFinal Results:")
print(f"  ADAPT-VQE Energy: {final_energy:.6f} Ha")
print(f"  Exact Energy: {exact_energy:.6f} Ha")
print(f"  Error: {abs(final_energy - exact_energy)*1000:.3f} mHa")
print(f"  Number of operators: {len(solver.selected_ops)}")
print(f"  Selected operators: {[op[0] for op in solver.selected_ops]}")

# Plot convergence
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Energy convergence
ax1.plot(range(1, len(solver.energy_history) + 1),
         solver.energy_history, 'bo-', markersize=8)
ax1.axhline(y=exact_energy, color='r', linestyle='--', label='Exact')
ax1.set_xlabel('ADAPT Iteration')
ax1.set_ylabel('Energy (Hartree)')
ax1.set_title('ADAPT-VQE Energy Convergence')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Gradient convergence
ax2.semilogy(range(1, len(solver.gradient_history) + 1),
             solver.gradient_history, 'go-', markersize=8)
ax2.set_xlabel('ADAPT Iteration')
ax2.set_ylabel('Max Gradient (log scale)')
ax2.set_title('ADAPT-VQE Gradient Convergence')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_974_adapt_vqe.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure saved as 'day_974_adapt_vqe.png'")

# =============================================================================
# Part 5: Comparison with Standard VQE
# =============================================================================

print("\n" + "=" * 70)
print("Part 5: Comparison with Standard VQE")
print("=" * 70)

@qml.qnode(dev)
def standard_vqe_circuit(params):
    """Standard hardware-efficient ansatz."""
    # Layer 1
    qml.RY(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    # Layer 2
    qml.RY(params[2], wires=0)
    qml.RY(params[3], wires=1)
    return qml.expval(H)

def standard_vqe_cost(params):
    return float(standard_vqe_circuit(pnp.array(params)))

# Optimize standard VQE
x0_standard = [0.1, 0.1, 0.1, 0.1]
result_standard = minimize(standard_vqe_cost, x0_standard, method='COBYLA',
                          options={'maxiter': 200})

print(f"\nStandard VQE (4 parameters):")
print(f"  Energy: {result_standard.fun:.6f} Ha")
print(f"  Parameters: {list(result_standard.x)}")

print(f"\nADAPT-VQE ({len(final_params)} parameters):")
print(f"  Energy: {final_energy:.6f} Ha")

print(f"\nADAPT-VQE uses {len(final_params)} vs 4 parameters")
print(f"with comparable or better accuracy!")

print("\n" + "=" * 70)
print("Lab Complete!")
print("=" * 70)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| ADAPT gradient | $g_k = \langle \psi_n \| [H, A_k] \| \psi_n \rangle$ |
| State update | $\|\psi_{n+1}\rangle = e^{\theta_{n+1} A_{n+1}} \|\psi_n\rangle$ |
| Convergence | $\max_k \|g_k\| < \epsilon$ |
| Pool (singles) | $A_{pq} = a_p^\dagger a_q - a_q^\dagger a_p$ |
| Pool (doubles) | $A_{pqrs} = a_p^\dagger a_q^\dagger a_r a_s - \text{h.c.}$ |

### Main Takeaways

1. **ADAPT-VQE builds problem-specific ansatze** by iteratively selecting operators
2. **Gradient-based selection** identifies the most impactful operations
3. **Compact circuits** result from including only necessary operators
4. **Operator pool design** trades circuit depth for selection efficiency
5. **Convergence** is signaled by small gradients across the entire pool
6. **Significant compression** compared to fixed UCCSD ansatze

---

## Daily Checklist

- [ ] Understand limitations of fixed ansatze
- [ ] Derive the ADAPT gradient formula
- [ ] Work through all three examples
- [ ] Complete Level 1 practice problems
- [ ] Attempt at least one Level 2 problem
- [ ] Run and modify the computational lab
- [ ] Write a one-paragraph summary comparing ADAPT-VQE to UCCSD

---

## Preview: Day 975

Tomorrow we explore **symmetry-preserving ansatze**—how to constrain variational forms to respect particle number, spin, and point group symmetries. This ensures physically valid solutions and can dramatically improve optimization landscapes.

---

*"Let the physics guide the circuit, not the hardware."*
--- Philosophy of ADAPT-VQE

---

**Next:** [Day_975_Tuesday.md](Day_975_Tuesday.md) - Symmetry-Preserving Ansatze
