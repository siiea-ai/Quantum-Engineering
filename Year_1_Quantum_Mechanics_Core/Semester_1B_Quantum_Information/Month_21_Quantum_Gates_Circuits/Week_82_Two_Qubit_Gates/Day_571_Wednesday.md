# Day 571: Entangling Power

## Schedule Overview

| Session | Time | Focus |
|---------|------|-------|
| Morning | 3 hours | Theory: Entanglement measures, entangling capacity |
| Afternoon | 2.5 hours | Problem solving: Computing entangling power |
| Evening | 1.5 hours | Computational lab: Comparing gates' entangling abilities |

## Learning Objectives

By the end of today, you will be able to:

1. **Define entangling power** as a measure of gate capability
2. **Create Bell states** using various entangling gates
3. **Compute concurrence** for two-qubit pure states
4. **Distinguish local from non-local gates** mathematically
5. **Classify gates** by their entangling capacity
6. **Understand the role** of entanglement in quantum advantage

---

## Core Content

### 1. What is Entangling Power?

The **entangling power** of a two-qubit gate U quantifies its ability to create entanglement from product states.

**Definition:** A gate U is **entangling** if there exists a product state $|\psi\rangle \otimes |\phi\rangle$ such that $U(|\psi\rangle \otimes |\phi\rangle)$ is entangled.

**Non-entangling (local) gates:** Gates of the form $U = A \otimes B$ cannot create entanglement.

### 2. Concurrence for Pure States

For a two-qubit pure state $|\psi\rangle$, the **concurrence** is:

$$\boxed{C(|\psi\rangle) = 2|ad - bc|}$$

where $|\psi\rangle = a|00\rangle + b|01\rangle + c|10\rangle + d|11\rangle$.

**Properties:**
- $0 \leq C \leq 1$
- $C = 0$: separable (product) state
- $C = 1$: maximally entangled

### 3. Bell States are Maximally Entangled

For the Bell state $|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$:

$$C(|\Phi^+\rangle) = 2 \cdot \left|\frac{1}{\sqrt{2}} \cdot \frac{1}{\sqrt{2}} - 0 \cdot 0\right| = 2 \cdot \frac{1}{2} = 1$$

**All four Bell states have concurrence 1** - they are maximally entangled.

### 4. Entangling Power Definition

The **entangling power** $e_p(U)$ of a gate U is the average entanglement produced when U acts on product states:

$$\boxed{e_p(U) = \overline{E(U|\psi\rangle \otimes |\phi\rangle)}}$$

where the average is over all product states (with Haar measure).

**Normalized entangling power:** Often scaled so maximum is 1.

### 5. Local vs Non-Local Gates

**Local gates:** $U = A \otimes B$
- Cannot create entanglement
- Entangling power = 0

**Non-local gates:** Cannot be written as $A \otimes B$
- Can create entanglement
- Entangling power > 0

**CNOT, CZ, SWAP:** All non-local!

### 6. Maximum Entangling Power

The maximum entangling power is achieved by **perfect entanglers**:

$$\boxed{e_p^{max} = \frac{2}{9}}$$

(using linear entropy as the entanglement measure)

**Perfect entanglers include:**
- CNOT
- CZ
- iSWAP
- √SWAP (for certain definitions)
- DCNOT (double-CNOT)

### 7. Bell State Creation Circuits

Different gates create Bell states differently:

**CNOT:**
$$|\Phi^+\rangle = \text{CNOT}(H \otimes I)|00\rangle$$

**CZ:**
$$|\Phi^+\rangle = (H \otimes H) \cdot \text{CZ} \cdot (H \otimes I)|00\rangle$$

**iSWAP:**
$$|\Phi^+\rangle = (S^\dagger \otimes I) \cdot \text{iSWAP} \cdot (H \otimes H)|01\rangle$$

### 8. Entangling Capacity

The **entangling capacity** $K_E(U)$ is the maximum entanglement U can produce:

$$\boxed{K_E(U) = \max_{|\psi\rangle, |\phi\rangle} E(U|\psi\rangle \otimes |\phi\rangle)}$$

For CNOT and CZ: $K_E = 1$ ebit (they can create maximally entangled states).

### 9. Local Equivalence Classes

Two gates U and V are **locally equivalent** if:
$$V = (A \otimes B) \cdot U \cdot (C \otimes D)$$

for some single-qubit gates A, B, C, D.

**Locally equivalent gates have the same entangling power!**

### 10. The Magic Basis

In the **magic basis**, the entangling properties become clear:

$$|m_1\rangle = \frac{|00\rangle + |11\rangle}{\sqrt{2}}, \quad |m_2\rangle = \frac{i|00\rangle - i|11\rangle}{\sqrt{2}}$$
$$|m_3\rangle = \frac{|01\rangle - |10\rangle}{\sqrt{2}}, \quad |m_4\rangle = \frac{i|01\rangle + i|10\rangle}{\sqrt{2}}$$

In this basis, CNOT becomes diagonal!

### 11. Non-Local Content

The **non-local content** of a gate can be characterized by parameters (c₁, c₂, c₃):

$$U = (A_1 \otimes B_1) \cdot \exp\left[i\sum_j c_j \sigma_j \otimes \sigma_j\right] \cdot (A_2 \otimes B_2)$$

For CNOT: $(c_1, c_2, c_3) = (\pi/4, 0, 0)$

### 12. Quantum Advantage from Entanglement

Entanglement is a key resource for:

| Application | Role of Entanglement |
|-------------|---------------------|
| Quantum teleportation | Bell pair as channel |
| Superdense coding | Doubled communication |
| Quantum key distribution | Security guarantee |
| Quantum algorithms | Speedup source |
| Quantum error correction | Stabilizer states |

---

## Quantum Computing Connection

Understanding entangling power helps:

1. **Gate selection:** Choose gates that efficiently create needed entanglement
2. **Circuit optimization:** Minimize entangling gate count
3. **Error mitigation:** Entangling gates are typically noisier
4. **Resource estimation:** Count entanglement as a computational resource
5. **Hardware benchmarking:** Compare platforms by entangling gate fidelity

---

## Worked Examples

### Example 1: Concurrence of √SWAP Output

**Problem:** Compute the concurrence of $\sqrt{\text{SWAP}}|01\rangle$.

**Solution:**

From Day 570:
$$\sqrt{\text{SWAP}}|01\rangle = \frac{1+i}{2}|01\rangle + \frac{1-i}{2}|10\rangle$$

So $a = 0$, $b = \frac{1+i}{2}$, $c = \frac{1-i}{2}$, $d = 0$.

$$C = 2|ad - bc| = 2\left|0 - \frac{1+i}{2} \cdot \frac{1-i}{2}\right| = 2\left|\frac{(1+i)(1-i)}{4}\right|$$

$$= 2\left|\frac{1 - i^2}{4}\right| = 2 \cdot \frac{2}{4} = 1$$

**Maximally entangled!** √SWAP on |01⟩ produces a maximally entangled state.

### Example 2: CNOT Entangling Power

**Problem:** Show CNOT can create any amount of entanglement from 0 to 1.

**Solution:**

Consider input $|\psi\rangle = \cos(\theta/2)|0\rangle + \sin(\theta/2)|1\rangle$ on control, $|0\rangle$ on target:

$$\text{CNOT}(|\psi\rangle \otimes |0\rangle) = \cos(\theta/2)|00\rangle + \sin(\theta/2)|11\rangle$$

Concurrence:
$$C = 2|\cos(\theta/2) \cdot \sin(\theta/2)| = |\sin\theta|$$

- θ = 0: C = 0 (product state)
- θ = π/2: C = 1 (maximally entangled)
- θ between: any intermediate entanglement

### Example 3: Local Gate Has Zero Entangling Power

**Problem:** Show that $U = H \otimes Z$ cannot create entanglement.

**Solution:**

For any product state $|\psi\rangle \otimes |\phi\rangle$:

$$(H \otimes Z)(|\psi\rangle \otimes |\phi\rangle) = (H|\psi\rangle) \otimes (Z|\phi\rangle)$$

This is still a product state! The Schmidt rank is 1.

Therefore, $e_p(H \otimes Z) = 0$.

---

## Practice Problems

### Direct Application

1. Compute the concurrence of $\frac{1}{2}(|00\rangle + |01\rangle + |10\rangle + |11\rangle)$.

2. Show that $|00\rangle$ is a product state (concurrence = 0).

3. Find the concurrence of $(|00\rangle + |01\rangle + |11\rangle)/\sqrt{3}$.

### Intermediate

4. **Partial entanglement:** For what value of α is the state $\cos\alpha|00\rangle + \sin\alpha|11\rangle$ half-maximally entangled (C = 0.5)?

5. Show that SWAP is not entangling: $\text{SWAP}(|\psi\rangle \otimes |\phi\rangle) = |\phi\rangle \otimes |\psi\rangle$ is still a product state.

6. Compute the entanglement produced by CZ on $|+\rangle \otimes |+\rangle$.

### Challenging

7. **Average entangling power:** Compute the average concurrence when CNOT acts on uniformly random product states $|\psi\rangle \otimes |0\rangle$.

8. Prove that CNOT and CZ have the same entangling power (they are locally equivalent).

9. **Non-local unitary:** Show that $\exp(i\theta Z \otimes Z)$ is local for all θ, but $\exp(i\theta X \otimes X)$ for θ ≠ 0 is non-local.

---

## Computational Lab: Entangling Power Analysis

```python
"""
Day 571: Entangling Power
Measuring and comparing the entangling ability of two-qubit gates
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import unitary_group

# Define basic gates
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

# Two-qubit gates
CNOT = np.array([[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]], dtype=complex)
CZ = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,-1]], dtype=complex)
SWAP = np.array([[1,0,0,0], [0,0,1,0], [0,1,0,0], [0,0,0,1]], dtype=complex)
sqrt_SWAP = np.array([[1,0,0,0], [0,(1+1j)/2,(1-1j)/2,0], [0,(1-1j)/2,(1+1j)/2,0], [0,0,0,1]], dtype=complex)
iSWAP = np.array([[1,0,0,0], [0,0,1j,0], [0,1j,0,0], [0,0,0,1]], dtype=complex)

# Basis states
ket_0 = np.array([[1], [0]], dtype=complex)
ket_1 = np.array([[0], [1]], dtype=complex)

def ket(bitstring):
    state = np.array([[1]], dtype=complex)
    for bit in bitstring:
        state = np.kron(state, ket_0 if bit == '0' else ket_1)
    return state

print("=" * 60)
print("CONCURRENCE AND ENTANGLEMENT")
print("=" * 60)

def concurrence(state):
    """
    Compute concurrence for 2-qubit pure state.
    C = 2|ad - bc| where state = a|00⟩ + b|01⟩ + c|10⟩ + d|11⟩
    """
    state = state.flatten()
    a, b, c, d = state
    return 2 * np.abs(a*d - b*c)

def linear_entropy(state):
    """
    Compute linear entropy of reduced density matrix.
    S_L = 1 - Tr(ρ_A²)
    """
    state = state.flatten()
    # Reduced density matrix by tracing out second qubit
    rho = state.reshape(2, 2)
    rho_A = rho @ rho.conj().T
    return 1 - np.real(np.trace(rho_A @ rho_A))

def von_neumann_entropy(state):
    """
    Compute von Neumann entropy of reduced density matrix.
    """
    state = state.flatten()
    rho = state.reshape(2, 2)
    rho_A = rho @ rho.conj().T
    eigenvalues = np.linalg.eigvalsh(rho_A)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    return -np.sum(eigenvalues * np.log2(eigenvalues))

# Test on Bell states
print("\n1. Entanglement of Bell States:")
bell_states = {
    'Φ⁺': (ket('00') + ket('11')) / np.sqrt(2),
    'Φ⁻': (ket('00') - ket('11')) / np.sqrt(2),
    'Ψ⁺': (ket('01') + ket('10')) / np.sqrt(2),
    'Ψ⁻': (ket('01') - ket('10')) / np.sqrt(2),
}

for name, state in bell_states.items():
    C = concurrence(state)
    S = von_neumann_entropy(state)
    print(f"   |{name}⟩: Concurrence = {C:.4f}, vN Entropy = {S:.4f} ebits")

# Test on product states
print("\n2. Entanglement of Product States:")
product_states = [
    ('|00⟩', ket('00')),
    ('|+⟩|0⟩', np.kron((ket_0 + ket_1)/np.sqrt(2), ket_0)),
    ('|0⟩|+⟩', np.kron(ket_0, (ket_0 + ket_1)/np.sqrt(2))),
]

for name, state in product_states:
    C = concurrence(state)
    print(f"   {name}: Concurrence = {C:.4f} (should be 0)")

# Entangling power analysis
print("\n" + "=" * 60)
print("ENTANGLING POWER OF GATES")
print("=" * 60)

def random_product_state():
    """Generate random product state |ψ⟩⊗|φ⟩."""
    # Random single-qubit states (Haar measure)
    psi = unitary_group.rvs(2)[:, 0].reshape(-1, 1)
    phi = unitary_group.rvs(2)[:, 0].reshape(-1, 1)
    return np.kron(psi, phi)

def estimate_entangling_power(U, n_samples=1000):
    """
    Estimate entangling power by averaging over random product states.
    """
    concurrences = []
    for _ in range(n_samples):
        product_state = random_product_state()
        output_state = U @ product_state
        C = concurrence(output_state)
        concurrences.append(C)
    return np.mean(concurrences), np.max(concurrences)

print("\n3. Estimated Entangling Power (1000 samples):")

gates_to_test = [
    ('CNOT', CNOT),
    ('CZ', CZ),
    ('SWAP', SWAP),
    ('√SWAP', sqrt_SWAP),
    ('iSWAP', iSWAP),
    ('I⊗I', np.eye(4)),
    ('H⊗H', np.kron(H, H)),
]

gate_powers = {}
for name, gate in gates_to_test:
    avg_C, max_C = estimate_entangling_power(gate)
    gate_powers[name] = (avg_C, max_C)
    print(f"   {name:10}: Avg Concurrence = {avg_C:.4f}, Max Concurrence = {max_C:.4f}")

# Bell state creation
print("\n" + "=" * 60)
print("BELL STATE CREATION CIRCUITS")
print("=" * 60)

print("\n4. Creating |Φ⁺⟩ with different gates:")

# CNOT method
state1 = CNOT @ np.kron(H, I) @ ket('00')
print(f"   CNOT(H⊗I)|00⟩ = |Φ⁺⟩: {np.allclose(state1, bell_states['Φ⁺'])}")

# CZ method
state2 = np.kron(I, H) @ CZ @ np.kron(H, I) @ ket('00')
match = any(np.allclose(state2, bs) for bs in bell_states.values())
print(f"   (I⊗H)CZ(H⊗I)|00⟩ is Bell state: {match}")

# √SWAP method
state3 = sqrt_SWAP @ ket('01')
C3 = concurrence(state3)
print(f"   √SWAP|01⟩: Concurrence = {C3:.4f} (maximally entangled)")

# Entanglement vs input state
print("\n" + "=" * 60)
print("ENTANGLEMENT VS INPUT STATE")
print("=" * 60)

print("\n5. CNOT entanglement as function of control qubit state:")
thetas = np.linspace(0, np.pi, 50)
cnot_concurrences = []

for theta in thetas:
    # Input: (cos(θ/2)|0⟩ + sin(θ/2)|1⟩) ⊗ |0⟩
    control = np.cos(theta/2) * ket_0 + np.sin(theta/2) * ket_1
    input_state = np.kron(control, ket_0)
    output_state = CNOT @ input_state
    cnot_concurrences.append(concurrence(output_state))

print(f"   Min concurrence: {min(cnot_concurrences):.4f} (at θ=0 or π)")
print(f"   Max concurrence: {max(cnot_concurrences):.4f} (at θ=π/2)")

# Visualization
print("\n" + "=" * 60)
print("VISUALIZATION")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: CNOT entanglement vs input
ax1 = axes[0, 0]
ax1.plot(thetas / np.pi, cnot_concurrences, 'b-', linewidth=2)
ax1.plot(thetas / np.pi, np.abs(np.sin(thetas)), 'r--', linewidth=2, label='|sin(θ)|')
ax1.set_xlabel('θ/π (control = cos(θ/2)|0⟩ + sin(θ/2)|1⟩)', fontsize=10)
ax1.set_ylabel('Concurrence', fontsize=10)
ax1.set_title('CNOT Entanglement vs Control State', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1.1])

# Plot 2: Comparison of gate entangling powers
ax2 = axes[0, 1]
gate_names = list(gate_powers.keys())
avg_powers = [gate_powers[g][0] for g in gate_names]
max_powers = [gate_powers[g][1] for g in gate_names]

x = np.arange(len(gate_names))
width = 0.35
ax2.bar(x - width/2, avg_powers, width, label='Average', color='steelblue', alpha=0.7)
ax2.bar(x + width/2, max_powers, width, label='Maximum', color='coral', alpha=0.7)
ax2.set_xticks(x)
ax2.set_xticklabels(gate_names, rotation=45, ha='right')
ax2.set_ylabel('Concurrence', fontsize=10)
ax2.set_title('Entangling Power Comparison', fontsize=12)
ax2.legend()
ax2.set_ylim([0, 1.2])

# Plot 3: Distribution of concurrences for CNOT
ax3 = axes[1, 0]
cnot_samples = []
for _ in range(5000):
    product_state = random_product_state()
    output_state = CNOT @ product_state
    cnot_samples.append(concurrence(output_state))

ax3.hist(cnot_samples, bins=50, density=True, color='steelblue', alpha=0.7, edgecolor='black')
ax3.axvline(x=np.mean(cnot_samples), color='red', linestyle='--', linewidth=2, label=f'Mean = {np.mean(cnot_samples):.3f}')
ax3.set_xlabel('Concurrence', fontsize=10)
ax3.set_ylabel('Probability Density', fontsize=10)
ax3.set_title('CNOT: Distribution of Concurrences', fontsize=12)
ax3.legend()

# Plot 4: √SWAP vs SWAP
ax4 = axes[1, 1]

# Track state through partial swaps
swap_powers = [0, 0.25, 0.5, 0.75, 1.0]
swap_labels = ['I', '√⁴SWAP', '√SWAP', '√⁴SWAP³', 'SWAP']

for t in np.linspace(0, 1, 5):
    # Interpolate between I and SWAP
    U_t = (1-t) * np.eye(4) + t * SWAP  # This isn't unitary, just for illustration

# Better approach: parameterize by power of SWAP
def fractional_swap(power):
    """Compute SWAP^power using eigendecomposition."""
    eigenvalues, eigenvectors = np.linalg.eig(SWAP)
    D_power = np.diag(eigenvalues ** power)
    return eigenvectors @ D_power @ np.linalg.inv(eigenvectors)

powers = np.linspace(0, 1, 50)
swap_concurrences = []
for p in powers:
    U_p = fractional_swap(p)
    output = U_p @ ket('01')
    swap_concurrences.append(concurrence(output))

ax4.plot(powers, swap_concurrences, 'b-', linewidth=2)
ax4.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
ax4.set_xlabel('Power p (SWAP^p)', fontsize=10)
ax4.set_ylabel('Concurrence of SWAP^p|01⟩', fontsize=10)
ax4.set_title('Entanglement from Fractional SWAP', fontsize=12)
ax4.grid(True, alpha=0.3)
ax4.set_xlim([0, 1])
ax4.set_ylim([0, 1.1])

# Mark special points
special_powers = [0, 0.5, 1]
special_labels = ['I', '√SWAP', 'SWAP']
for p, label in zip(special_powers, special_labels):
    U_p = fractional_swap(p)
    C = concurrence(U_p @ ket('01'))
    ax4.scatter([p], [C], s=100, zorder=5)
    ax4.annotate(label, (p, C), textcoords="offset points", xytext=(5, 5))

plt.tight_layout()
plt.savefig('entangling_power.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved: entangling_power.png")

# Local equivalence
print("\n" + "=" * 60)
print("LOCAL EQUIVALENCE")
print("=" * 60)

print("\n6. CNOT and CZ are locally equivalent:")
# CZ = (I⊗H) CNOT (I⊗H)
CZ_from_CNOT = np.kron(I, H) @ CNOT @ np.kron(I, H)
print(f"   (I⊗H)CNOT(I⊗H) = CZ: {np.allclose(CZ_from_CNOT, CZ)}")
print(f"   Same entangling power: CNOT avg = {gate_powers['CNOT'][0]:.4f}, CZ avg = {gate_powers['CZ'][0]:.4f}")

# Summary statistics
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print("\n7. Gate Classification by Entangling Power:")
print("-" * 50)
print(f"{'Gate':<12} {'Avg C':<10} {'Max C':<10} {'Classification':<20}")
print("-" * 50)

for name in gate_names:
    avg, mx = gate_powers[name]
    if mx < 0.01:
        classification = "Non-entangling"
    elif mx > 0.99:
        classification = "Maximally entangling"
    else:
        classification = "Partially entangling"
    print(f"{name:<12} {avg:<10.4f} {mx:<10.4f} {classification:<20}")
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Concurrence | $C = 2\|ad - bc\|$ for state $a\|00\rangle + b\|01\rangle + c\|10\rangle + d\|11\rangle$ |
| Separable | $C = 0$ |
| Maximally entangled | $C = 1$ |
| Local gate | $U = A \otimes B$, $e_p = 0$ |
| Entangling power | Average entanglement over product states |
| Bell state | $\|\Phi^+\rangle = \text{CNOT}(H \otimes I)\|00\rangle$, $C = 1$ |

### Main Takeaways

1. **Concurrence quantifies entanglement:** 0 for product states, 1 for maximally entangled
2. **Entangling power:** Average ability to create entanglement from product states
3. **CNOT, CZ, iSWAP are maximally entangling:** Can produce any entanglement level
4. **Local gates don't entangle:** $A \otimes B$ preserves separability
5. **Local equivalence:** Gates differing by local unitaries have same entangling power

---

## Daily Checklist

- [ ] I can compute concurrence for two-qubit pure states
- [ ] I understand the definition of entangling power
- [ ] I can classify gates as entangling vs non-entangling
- [ ] I know that CNOT, CZ are maximally entangling
- [ ] I understand local equivalence of gates
- [ ] I completed the computational lab
- [ ] I solved at least 3 practice problems

---

## Preview of Day 572

Tomorrow we study **gate identities** - algebraic relationships between two-qubit gates that enable circuit simplification. We'll derive key identities like how Hadamards swap control and target in CNOT, and how to commute gates through each other.
