# Day 511: Week 73 Review — Pure vs Mixed States

## Overview

**Day 511** | Week 73, Day 7 | Year 1, Month 19 | Integration and Consolidation

Today we synthesize all concepts from Week 73: density operators, trace properties, expectation values, purity, Bloch representation, and state distinguishability measures.

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Concept review and connections |
| Afternoon | 2:00 PM - 5:00 PM | 3 hrs | Comprehensive problem solving |
| Evening | 7:00 PM - 8:30 PM | 1.5 hrs | Self-assessment |

---

## Week 73 Concept Map

```
                    DENSITY MATRICES
                          │
          ┌───────────────┼───────────────┐
          │               │               │
    DEFINITION      PROPERTIES       BLOCH SPHERE
          │               │               │
    ρ = Σpᵢ|ψᵢ⟩⟨ψᵢ|   Hermitian        Mixed states
          │           Positive         inside ball
          │           Tr(ρ)=1              │
          │               │               │
          └───────┬───────┴───────┬───────┘
                  │               │
           EXPECTATION       PURITY
             VALUES              │
                  │          γ = Tr(ρ²)
           ⟨A⟩ = Tr(ρA)         │
                  │               │
                  └───────┬───────┘
                          │
                 DISTINGUISHABILITY
                          │
              ┌───────────┴───────────┐
              │                       │
        TRACE DISTANCE           FIDELITY
        D = ½Tr|ρ-σ|         F = (Tr√...)²
              │                       │
              └───────────┬───────────┘
                          │
                   APPLICATIONS
                          │
         ┌────────┬───────┼───────┬────────┐
         │        │       │       │        │
      Quantum  Error   State   Quantum  Crypto-
      Computing Analysis  Tomog  Channels graphy
```

---

## Key Concepts Summary

### 1. Density Operator Definition (Day 505)

**Pure state:** Complete quantum information
$$\rho = |\psi\rangle\langle\psi|$$

**Mixed state:** Classical uncertainty over quantum states
$$\rho = \sum_i p_i |\psi_i\rangle\langle\psi_i|, \quad \sum_i p_i = 1$$

**Key distinction:** Superposition has coherences, mixture does not (in mixture basis).

### 2. Properties and Trace (Day 506)

**Three defining properties:**
1. Hermitian: ρ† = ρ
2. Positive semidefinite: ρ ≥ 0
3. Normalized: Tr(ρ) = 1

**Trace formulas:**
- Expectation: ⟨A⟩ = Tr(ρA)
- Probability: p(m) = Tr(Πₘρ)

### 3. Expectation Values (Day 507)

**Complete statistics:**
- Mean: ⟨A⟩ = Tr(ρA)
- Variance: (ΔA)² = Tr(ρA²) - [Tr(ρA)]²
- n-th moment: ⟨Aⁿ⟩ = Tr(ρAⁿ)

**Measurement update (Lüders rule):**
$$\rho \rightarrow \frac{\Pi_m \rho \Pi_m}{\text{Tr}(\Pi_m \rho)}$$

### 4. Purity and Mixedness (Day 508)

**Purity:**
$$\gamma = \text{Tr}(\rho^2), \quad \frac{1}{d} \leq \gamma \leq 1$$

**State classification:**
- γ = 1: Pure state
- γ = 1/d: Maximally mixed
- 1/d < γ < 1: Partially mixed

### 5. Bloch Sphere (Day 509)

**Bloch representation:**
$$\rho = \frac{1}{2}(I + \vec{r} \cdot \vec{\sigma})$$

**Key relationships:**
- |r⃗| = 1: Pure (surface)
- |r⃗| < 1: Mixed (interior)
- |r⃗| = 0: Maximally mixed (center)
- γ = (1 + |r⃗|²)/2

### 6. Distinguishing States (Day 510)

**Trace distance:**
$$D(\rho, \sigma) = \frac{1}{2}\text{Tr}|\rho - \sigma|$$

**Fidelity:**
$$F(\rho, \sigma) = \left(\text{Tr}\sqrt{\sqrt{\rho}\sigma\sqrt{\rho}}\right)^2$$

**Fuchs-van de Graaf:**
$$1 - \sqrt{F} \leq D \leq \sqrt{1-F}$$

---

## Master Formula Sheet

### Density Matrix Basics
| Formula | Description |
|---------|-------------|
| ρ = \|ψ⟩⟨ψ\| | Pure state |
| ρ = Σᵢ pᵢ\|ψᵢ⟩⟨ψᵢ\| | Mixed state |
| Tr(ρ) = 1 | Normalization |
| ρ† = ρ | Hermiticity |
| ρ ≥ 0 | Positive semidefinite |

### Expectation Values
| Formula | Description |
|---------|-------------|
| ⟨A⟩ = Tr(ρA) | Expectation value |
| p(m) = Tr(Πₘρ) | Measurement probability |
| (ΔA)² = ⟨A²⟩ - ⟨A⟩² | Variance |

### Purity and Entropy
| Formula | Description |
|---------|-------------|
| γ = Tr(ρ²) | Purity |
| S_L = 1 - γ | Linear entropy |
| S = -Tr(ρ log ρ) | von Neumann entropy |

### Bloch Representation
| Formula | Description |
|---------|-------------|
| ρ = ½(I + r⃗·σ⃗) | Bloch form |
| rᵢ = Tr(ρσᵢ) | Extract Bloch vector |
| γ = ½(1 + \|r⃗\|²) | Purity from Bloch |
| λ± = ½(1 ± \|r⃗\|) | Eigenvalues |

### Distinguishability
| Formula | Description |
|---------|-------------|
| D = ½Tr\|ρ-σ\| | Trace distance |
| F = (Tr√(√ρσ√ρ))² | Fidelity |
| p_max = ½(1+D) | Max distinguish prob |
| D = ½\|r⃗-s⃗\| | Qubit trace distance |

---

## Comprehensive Problem Set

### Part A: Fundamentals

**A1.** Prove that if ρ is a valid density matrix, then 0 ≤ ⟨ψ|ρ|ψ⟩ ≤ 1 for any normalized |ψ⟩.

**A2.** Show that the set of density matrices is convex: if ρ₁ and ρ₂ are density matrices, so is pρ₁ + (1-p)ρ₂ for 0 ≤ p ≤ 1.

**A3.** Calculate ρ² for a general qubit ρ = ½(I + r⃗·σ⃗) and verify γ = ½(1 + |r⃗|²).

### Part B: Calculations

**B1.** For the ensemble {(½, |0⟩), (¼, |+⟩), (¼, |1⟩)}:
a) Construct the density matrix
b) Find the Bloch vector
c) Calculate the purity

**B2.** A qubit is measured in the X basis. Starting from ρ = ¾|0⟩⟨0| + ¼|1⟩⟨1|:
a) Find p(+) and p(-)
b) Find the post-measurement state for each outcome
c) Find the non-selective post-measurement state

**B3.** Calculate D(ρ, σ) and F(ρ, σ) for:
- ρ = |+⟩⟨+|
- σ = ½|0⟩⟨0| + ½|1⟩⟨1|

### Part C: Proofs

**C1.** Prove that purity is invariant under unitary transformations.

**C2.** Show that non-selective measurement cannot increase purity.

**C3.** Prove the lower bound in Fuchs-van de Graaf: D ≥ 1 - √F.

### Part D: Applications

**D1.** A quantum computer prepares |0⟩ with 99% fidelity (1% error gives |1⟩). Write the density matrix and calculate the purity.

**D2.** After a T₁ = 100μs decay time, a qubit initially in |1⟩ has Bloch vector (0, 0, e^(-t/T₁) - 1 + 2e^(-t/T₁)). Find the purity at t = 50μs.

**D3.** In quantum key distribution, Eve's information is bounded by the trace distance between the actual state and the ideal state. If F = 0.99, what is the maximum trace distance?

---

## Self-Assessment Checklist

### Conceptual Understanding
- [ ] I can explain why density matrices are needed beyond state vectors
- [ ] I understand the difference between superposition and mixture
- [ ] I can interpret purity physically
- [ ] I understand the Bloch ball representation
- [ ] I know what trace distance and fidelity measure

### Computational Skills
- [ ] I can construct density matrices from ensembles
- [ ] I can compute expectation values using Tr(ρA)
- [ ] I can find the Bloch vector from a density matrix
- [ ] I can calculate purity
- [ ] I can compute trace distance and fidelity

### Problem Solving
- [ ] I can verify if a matrix is a valid density matrix
- [ ] I can apply measurement update rules
- [ ] I can relate Bloch vector properties to state properties
- [ ] I can use Fuchs-van de Graaf bounds

---

## Computational Review Lab

```python
"""
Day 511: Week 73 Comprehensive Review
Complete density matrix toolkit
"""

import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === TOOLKIT FUNCTIONS ===

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
paulis = [I, X, Y, Z]

def ket(state):
    """Standard basis states"""
    states = {
        '0': np.array([[1], [0]], dtype=complex),
        '1': np.array([[0], [1]], dtype=complex),
        '+': np.array([[1], [1]], dtype=complex) / np.sqrt(2),
        '-': np.array([[1], [-1]], dtype=complex) / np.sqrt(2),
        '+i': np.array([[1], [1j]], dtype=complex) / np.sqrt(2),
        '-i': np.array([[1], [-1j]], dtype=complex) / np.sqrt(2),
    }
    return states[state]

def density(psi):
    """Create density matrix from state vector"""
    return psi @ psi.conj().T

def mixed_state(states_probs):
    """Create mixed state from [(prob, state), ...]"""
    rho = np.zeros((2, 2), dtype=complex)
    for prob, state in states_probs:
        if isinstance(state, str):
            state = ket(state)
        rho += prob * density(state)
    return rho

def is_valid_density_matrix(rho, tol=1e-10):
    """Check all three properties"""
    hermitian = np.allclose(rho, rho.conj().T, atol=tol)
    trace_one = np.isclose(np.trace(rho), 1, atol=tol)
    eigenvalues = np.linalg.eigvalsh(rho)
    positive = all(eigenvalues >= -tol)
    return hermitian and trace_one and positive

def expectation(rho, A):
    """Compute ⟨A⟩ = Tr(ρA)"""
    return np.trace(rho @ A).real

def variance(rho, A):
    """Compute (ΔA)² = ⟨A²⟩ - ⟨A⟩²"""
    return expectation(rho, A @ A) - expectation(rho, A)**2

def purity(rho):
    """Compute γ = Tr(ρ²)"""
    return np.trace(rho @ rho).real

def von_neumann_entropy(rho):
    """Compute S = -Tr(ρ log₂ ρ)"""
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    return -np.sum(eigenvalues * np.log2(eigenvalues))

def bloch_vector(rho):
    """Extract Bloch vector (rₓ, rᵧ, r_z)"""
    return np.array([expectation(rho, X),
                     expectation(rho, Y),
                     expectation(rho, Z)])

def from_bloch(r):
    """Create density matrix from Bloch vector"""
    return 0.5 * (I + r[0]*X + r[1]*Y + r[2]*Z)

def trace_distance(rho, sigma):
    """Compute D(ρ,σ) = ½Tr|ρ-σ|"""
    diff = rho - sigma
    abs_diff = sqrtm(diff.conj().T @ diff)
    return 0.5 * np.trace(abs_diff).real

def fidelity(rho, sigma):
    """Compute F(ρ,σ) = (Tr√(√ρσ√ρ))²"""
    sqrt_rho = sqrtm(rho)
    inner = sqrt_rho @ sigma @ sqrt_rho
    return (np.trace(sqrtm(inner)).real)**2

def measurement_update(rho, projector, selective=True):
    """Apply measurement update rule"""
    prob = np.trace(projector @ rho).real
    if prob < 1e-10:
        return None, 0
    if selective:
        return projector @ rho @ projector / prob, prob
    else:
        return projector @ rho @ projector, prob

# === COMPREHENSIVE REVIEW ===

print("=" * 70)
print("WEEK 73 COMPREHENSIVE REVIEW: DENSITY MATRICES")
print("=" * 70)

# Problem B1: Ensemble construction
print("\n--- Problem B1: Ensemble Construction ---")
ensemble = [(0.5, '0'), (0.25, '+'), (0.25, '1')]
rho_B1 = mixed_state(ensemble)
r_B1 = bloch_vector(rho_B1)
gamma_B1 = purity(rho_B1)

print(f"Ensemble: ½|0⟩ + ¼|+⟩ + ¼|1⟩")
print(f"ρ = \n{rho_B1}")
print(f"Bloch vector: r = {r_B1}")
print(f"|r| = {np.linalg.norm(r_B1):.4f}")
print(f"Purity γ = {gamma_B1:.4f}")
print(f"Valid density matrix: {is_valid_density_matrix(rho_B1)}")

# Problem B2: X-basis measurement
print("\n--- Problem B2: X-basis Measurement ---")
rho_B2 = mixed_state([(0.75, '0'), (0.25, '1')])
Pi_plus = density(ket('+'))
Pi_minus = density(ket('-'))

p_plus = expectation(rho_B2, Pi_plus)
p_minus = expectation(rho_B2, Pi_minus)

rho_after_plus, _ = measurement_update(rho_B2, Pi_plus)
rho_after_minus, _ = measurement_update(rho_B2, Pi_minus)
rho_nonselective = Pi_plus @ rho_B2 @ Pi_plus + Pi_minus @ rho_B2 @ Pi_minus

print(f"Initial ρ = ¾|0⟩⟨0| + ¼|1⟩⟨1|")
print(f"p(+) = {p_plus:.4f}")
print(f"p(-) = {p_minus:.4f}")
print(f"\nAfter outcome +:")
print(f"ρ₊ = \n{rho_after_plus}")
print(f"\nAfter outcome -:")
print(f"ρ₋ = \n{rho_after_minus}")
print(f"\nNon-selective:")
print(f"ρ' = \n{rho_nonselective}")

# Problem B3: Trace distance and fidelity
print("\n--- Problem B3: Distinguishability ---")
rho_plus = density(ket('+'))
rho_mixed = 0.5 * I

D_B3 = trace_distance(rho_plus, rho_mixed)
F_B3 = fidelity(rho_plus, rho_mixed)

print(f"ρ = |+⟩⟨+|")
print(f"σ = I/2")
print(f"D(ρ,σ) = {D_B3:.4f}")
print(f"F(ρ,σ) = {F_B3:.4f}")
print(f"Max distinguish prob = {0.5*(1+D_B3):.4f}")

# Verify bounds
lower = 1 - np.sqrt(F_B3)
upper = np.sqrt(1 - F_B3)
print(f"\nFuchs-van de Graaf: {lower:.4f} ≤ D ≤ {upper:.4f}")
print(f"D = {D_B3:.4f} ✓")

# Problem D1: Quantum computer state prep
print("\n--- Problem D1: State Preparation Error ---")
rho_D1 = 0.99 * density(ket('0')) + 0.01 * density(ket('1'))
r_D1 = bloch_vector(rho_D1)
gamma_D1 = purity(rho_D1)

print(f"State prep: 99% |0⟩, 1% |1⟩")
print(f"ρ = \n{rho_D1}")
print(f"Bloch vector: r = {r_D1}")
print(f"Purity γ = {gamma_D1:.4f}")
print(f"Fidelity with |0⟩: F = {fidelity(rho_D1, density(ket('0'))):.4f}")

# Visualization
fig = plt.figure(figsize=(15, 10))

# 1. Bloch sphere with all example states
ax1 = fig.add_subplot(221, projection='3d')

# Draw sphere
u = np.linspace(0, 2*np.pi, 30)
v = np.linspace(0, np.pi, 20)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))
ax1.plot_wireframe(x, y, z, alpha=0.1)

# Plot states from problems
states_to_plot = [
    (r_B1, 'Ensemble B1', 'red'),
    (bloch_vector(rho_B2), 'Initial B2', 'blue'),
    (bloch_vector(rho_nonselective), 'After meas B2', 'green'),
    (r_D1, 'State prep D1', 'orange'),
]

for r, name, color in states_to_plot:
    ax1.scatter([r[0]], [r[1]], [r[2]], c=color, s=100, label=name)
ax1.scatter([0], [0], [0], c='black', s=50, marker='x', label='Center')
ax1.set_title('Week 73 States on Bloch Ball')
ax1.legend(fontsize=8)

# 2. Purity vs entropy
ax2 = fig.add_subplot(222)
p_vals = np.linspace(0.001, 0.999, 100)
purity_vals = p_vals**2 + (1-p_vals)**2
entropy_vals = -p_vals*np.log2(p_vals) - (1-p_vals)*np.log2(1-p_vals)

ax2.plot(purity_vals, entropy_vals, 'b-', lw=2)
ax2.scatter([gamma_B1], [von_neumann_entropy(rho_B1)], c='red', s=100, label='B1')
ax2.scatter([gamma_D1], [von_neumann_entropy(rho_D1)], c='orange', s=100, label='D1')
ax2.set_xlabel('Purity γ')
ax2.set_ylabel('von Neumann Entropy S')
ax2.set_title('Entropy vs Purity')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Trace distance visualization
ax3 = fig.add_subplot(223)
theta_vals = np.linspace(0, np.pi, 100)
D_vals = []
F_vals = []

rho_ref = density(ket('0'))
for theta in theta_vals:
    psi = np.cos(theta/2)*ket('0') + np.sin(theta/2)*ket('1')
    rho_test = density(psi)
    D_vals.append(trace_distance(rho_ref, rho_test))
    F_vals.append(fidelity(rho_ref, rho_test))

ax3.plot(theta_vals/np.pi, D_vals, 'b-', lw=2, label='Trace Distance D')
ax3.plot(theta_vals/np.pi, F_vals, 'r-', lw=2, label='Fidelity F')
ax3.set_xlabel('θ/π')
ax3.set_ylabel('Value')
ax3.set_title('Distinguishability from |0⟩')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Summary statistics table
ax4 = fig.add_subplot(224)
ax4.axis('off')

summary_text = """
WEEK 73 SUMMARY

State               │ Purity │ |r|   │ Entropy
────────────────────┼────────┼───────┼────────
Ensemble B1         │ {:.3f}  │ {:.3f} │ {:.3f}
Initial B2          │ {:.3f}  │ {:.3f} │ {:.3f}
After meas B2       │ {:.3f}  │ {:.3f} │ {:.3f}
State prep D1       │ {:.3f}  │ {:.3f} │ {:.3f}
Pure state          │ 1.000  │ 1.000 │ 0.000
Max mixed           │ 0.500  │ 0.000 │ 1.000

KEY FORMULAS:
• ρ = ½(I + r⃗·σ⃗)
• γ = Tr(ρ²) = ½(1 + |r|²)
• ⟨A⟩ = Tr(ρA)
• D(ρ,σ) = ½Tr|ρ-σ|
""".format(
    gamma_B1, np.linalg.norm(r_B1), von_neumann_entropy(rho_B1),
    purity(rho_B2), np.linalg.norm(bloch_vector(rho_B2)), von_neumann_entropy(rho_B2),
    purity(rho_nonselective), np.linalg.norm(bloch_vector(rho_nonselective)), von_neumann_entropy(rho_nonselective),
    gamma_D1, np.linalg.norm(r_D1), von_neumann_entropy(rho_D1),
)

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace')

plt.tight_layout()
plt.savefig('week73_review.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 70)
print("Week 73 Complete! Ready for Week 74: Composite Systems")
print("=" * 70)
```

---

## Looking Ahead: Week 74

Next week we extend density matrices to **composite systems**:

- **Tensor products** of Hilbert spaces
- **Partial trace** operation
- **Reduced density matrices**
- **Schmidt decomposition**
- **Entanglement** detection via mixedness
- **Purification** of mixed states

The partial trace connects to entanglement: when a composite system is entangled, the subsystems are mixed even if the global state is pure!

---

## Key Takeaways from Week 73

1. **Density matrices** generalize state vectors to handle classical uncertainty
2. **Three properties** (Hermitian, positive, trace 1) completely characterize density matrices
3. **Trace formulas** give expectation values and measurement probabilities
4. **Purity** quantifies mixedness: γ = 1 (pure) to γ = 1/d (maximally mixed)
5. **Bloch ball** visualizes qubits: surface = pure, interior = mixed
6. **Trace distance and fidelity** quantify state distinguishability

---

## References

- Nielsen & Chuang, Chapter 2.4
- Preskill Ph219, Chapters 2-3
- Wilde, Quantum Information Theory, Chapter 3

---

**Week 73 Complete!**

You now have a solid foundation in density matrix formalism. Next week, we'll explore how density matrices describe parts of composite quantum systems.

---

*Next: Week 74 — Composite Systems*
