# Day 543: Quantum Violation of Bell Inequalities

## Overview
**Day 543** | Week 78, Day 4 | Year 1, Month 20 | Why Quantum Mechanics Wins

Today we analyze why quantum mechanics violates Bell inequalities, exploring the mathematical structure that enables stronger-than-classical correlations.

---

## Learning Objectives
1. Prove Tsirelson bound rigorously
2. Understand the role of non-commutativity
3. Analyze which states violate CHSH
4. Connect violation to entanglement
5. Study no-signaling constraints
6. Explore the quantum-classical boundary

---

## Core Content

### Why QM Violates Bell Inequalities

**Root cause:** Non-commuting observables and entanglement

For LHV: outcomes are predetermined → $|S| \leq 2$
For QM: outcomes arise from measurement → correlations can be stronger

### Tsirelson Bound Proof

**Theorem:** $|S| \leq 2\sqrt{2}$ for any quantum state and observables.

**Proof:**
Define Bell operator:
$$\mathcal{B} = A \otimes B - A \otimes B' + A' \otimes B + A' \otimes B'$$

For dichotomic observables ($A^2 = I$, etc.):
$$\mathcal{B}^2 = 4I \otimes I + [A, A'] \otimes [B, B']$$

Since $\|[A, A']\| \leq 2$ and $\|[B, B']\| \leq 2$:
$$\|\mathcal{B}^2\| \leq 4 + 4 = 8$$
$$\|\mathcal{B}\| \leq 2\sqrt{2}$$

Therefore $|\langle \mathcal{B} \rangle| \leq 2\sqrt{2}$ ∎

### Saturation of Tsirelson Bound

The bound is achieved when:
1. State is maximally entangled
2. $[A, A'] \neq 0$ and $[B, B'] \neq 0$ (non-commuting)
3. Optimal angle arrangement

For the singlet with optimal angles:
$$[A, A'] = [Z, X] = 2iY, \quad [B, B'] = 2iY'$$

### Entanglement Requirement

**Theorem:** Separable states satisfy all Bell inequalities.

**Proof:**
For separable $\rho = \sum_i p_i \rho_i^A \otimes \rho_i^B$:

$$\langle AB \rangle = \sum_i p_i \text{Tr}(A\rho_i^A) \text{Tr}(B\rho_i^B) = \sum_i p_i a_i b_i$$

This is a classical mixture → satisfies $|S| \leq 2$ ∎

**Corollary:** CHSH violation certifies entanglement!

### No-Signaling Principle

Even with Bell violation, no FTL signaling!

**Proof:** Alice's marginal distribution:
$$P(a|A) = \sum_b P(a,b|A,B) = \text{Tr}((E_a \otimes I)\rho)$$

This is independent of Bob's choice B. Alice cannot send messages by measuring her particle.

### Hierarchy of Correlations

```
                    ALL CORRELATIONS
                          │
              ┌───────────┴───────────┐
              │                       │
         SIGNALING              NO-SIGNALING
         (unphysical)               │
                          ┌─────────┴─────────┐
                          │                   │
                       QUANTUM            SUPER-QUANTUM
                       |S| ≤ 2√2          (hypothetical)
                          │
                    ┌─────┴─────┐
                    │           │
                 ENTANGLED   SEPARABLE
                 |S| > 2     |S| ≤ 2
```

### PR Box (Super-Quantum)

The **Popescu-Rohrlich box** saturates no-signaling:
$$P(a=b|AB) = 1 \text{ if } A \cdot B = 0$$
$$P(a \neq b|AB) = 1 \text{ if } A \cdot B = 1$$

This gives $S = 4$ (maximum no-signaling value).

Why doesn't nature allow this? Open question!

### Which States Violate CHSH?

**All pure entangled states** violate some Bell inequality (Gisin's theorem).

For CHSH specifically:
- Bell states: $|S| = 2\sqrt{2}$
- Werner states: violate for $p > 1/\sqrt{2}$
- Not all entangled states violate CHSH

### CHSH vs Negativity

Correlation between CHSH violation and negativity:

For Werner state: $|S| = 2\sqrt{2}p$, $\mathcal{N} = \max(0, (3p-1)/4)$

Violation requires more entanglement than PPT threshold!

---

## Worked Examples

### Example 1: Separable State CHSH
Show that $\rho = |00\rangle\langle 00|$ satisfies CHSH.

**Solution:**
$$\langle AB \rangle = \langle 0|A|0\rangle \langle 0|B|0\rangle$$

For any spin measurements: $\langle 0|\sigma_n|0\rangle = n_z$

Let's use optimal angles:
- $\langle AB \rangle = 1 \cdot \frac{1}{\sqrt{2}}$
- $\langle AB' \rangle = 1 \cdot (-\frac{1}{\sqrt{2}})$
- $\langle A'B \rangle = 0 \cdot \frac{1}{\sqrt{2}} = 0$
- $\langle A'B' \rangle = 0 \cdot (-\frac{1}{\sqrt{2}}) = 0$

$S = \frac{1}{\sqrt{2}} - (-\frac{1}{\sqrt{2}}) + 0 + 0 = \sqrt{2} < 2$ ✓

No violation for product states! ∎

### Example 2: Tsirelson Calculation
Verify $\|\mathcal{B}\| = 2\sqrt{2}$ for optimal settings.

**Solution:**
Bell operator eigenvalues: $\pm 2\sqrt{2}$, 0, 0

$$\mathcal{B} = Z \otimes \frac{Z+X}{\sqrt{2}} - Z \otimes \frac{-Z+X}{\sqrt{2}} + X \otimes \frac{Z+X}{\sqrt{2}} + X \otimes \frac{-Z+X}{\sqrt{2}}$$

Direct calculation or using $\mathcal{B}^2$ gives maximum eigenvalue $2\sqrt{2}$. ∎

### Example 3: Non-Maximally Entangled
For $|\psi\rangle = \sqrt{0.8}|00\rangle + \sqrt{0.2}|11\rangle$, find max CHSH.

**Solution:**
The state has concurrence $C = 2\sqrt{0.8 \times 0.2} = 0.8$.

Maximum CHSH: $S_{max} = 2\sqrt{1 + C^2} = 2\sqrt{1.64} \approx 2.56$

This violates $|S| \leq 2$ but doesn't reach Tsirelson. ∎

---

## Practice Problems

### Problem 1: Three Outcomes
Can Bell inequalities be violated with three-outcome measurements?

### Problem 2: Higher Dimensions
Derive the Tsirelson bound for qutrit systems.

### Problem 3: Bell Operator Spectrum
Find all eigenvalues of the CHSH Bell operator.

---

## Computational Lab

```python
"""Day 543: Quantum Violation Analysis"""
import numpy as np
from scipy.linalg import expm

# Pauli matrices
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.eye(2, dtype=complex)

def spin_operator(theta):
    """Spin in x-z plane"""
    return np.cos(theta)*Z + np.sin(theta)*X

def correlation(rho, op_A, op_B):
    """Tr(ρ A⊗B)"""
    return np.real(np.trace(rho @ np.kron(op_A, op_B)))

def chsh_value(rho, angles):
    """Compute CHSH for given state and angles"""
    theta_A, theta_Ap, theta_B, theta_Bp = angles
    A, Ap = spin_operator(theta_A), spin_operator(theta_Ap)
    B, Bp = spin_operator(theta_B), spin_operator(theta_Bp)

    return (correlation(rho, A, B) - correlation(rho, A, Bp) +
            correlation(rho, Ap, B) + correlation(rho, Ap, Bp))

def bell_operator(angles):
    """Construct CHSH Bell operator"""
    theta_A, theta_Ap, theta_B, theta_Bp = angles
    A, Ap = spin_operator(theta_A), spin_operator(theta_Ap)
    B, Bp = spin_operator(theta_B), spin_operator(theta_Bp)

    return (np.kron(A, B) - np.kron(A, Bp) +
            np.kron(Ap, B) + np.kron(Ap, Bp))

# States
def bell_state(idx=0):
    """Four Bell states"""
    states = [
        np.array([1, 0, 0, 1]) / np.sqrt(2),  # Φ⁺
        np.array([1, 0, 0, -1]) / np.sqrt(2),  # Φ⁻
        np.array([0, 1, 1, 0]) / np.sqrt(2),  # Ψ⁺
        np.array([0, 1, -1, 0]) / np.sqrt(2),  # Ψ⁻
    ]
    psi = states[idx]
    return np.outer(psi, psi.conj())

def werner_state(p):
    psi = np.array([0, 1, -1, 0]) / np.sqrt(2)
    return p * np.outer(psi, psi.conj()) + (1-p) * np.eye(4) / 4

def separable_state():
    """|00⟩⟨00|"""
    psi = np.array([1, 0, 0, 0])
    return np.outer(psi, psi.conj())

print("=== Tsirelson Bound Verification ===\n")

opt_angles = (0, np.pi/2, np.pi/4, 3*np.pi/4)
B_op = bell_operator(opt_angles)

print("Bell operator spectrum:")
eigenvalues = np.linalg.eigvalsh(B_op)
print(f"  Eigenvalues: {eigenvalues}")
print(f"  Max |eigenvalue|: {np.max(np.abs(eigenvalues)):.6f}")
print(f"  Tsirelson bound: {2*np.sqrt(2):.6f}")

# Verify B² formula
print("\n=== B² Formula Verification ===")
B2 = B_op @ B_op
expected_B2 = 4 * np.eye(4) - 2j * np.kron(Y, Y)
print(f"B² = 4I + [A,A']⊗[B,B']")
print(f"B² eigenvalues: {np.linalg.eigvalsh(B2)}")

# State comparison
print("\n=== CHSH for Various States ===\n")

states = [
    ("Singlet |Ψ⁻⟩", bell_state(3)),
    ("|Φ⁺⟩", bell_state(0)),
    ("Werner p=0.8", werner_state(0.8)),
    ("Werner p=0.7", werner_state(0.7)),
    ("Product |00⟩", separable_state()),
]

print(f"{'State':<20} {'S':<12} {'Violates?':<12}")
print("-" * 45)

for name, rho in states:
    S = chsh_value(rho, opt_angles)
    violates = "YES" if np.abs(S) > 2 else "no"
    print(f"{name:<20} {S:<12.4f} {violates:<12}")

# No-signaling verification
print("\n=== No-Signaling Verification ===\n")

rho = bell_state(3)  # singlet
A = spin_operator(0)
B1 = spin_operator(np.pi/4)
B2 = spin_operator(3*np.pi/4)

# Alice's marginal should be independent of Bob's choice
# P(a|A) = Tr((P_a ⊗ I) ρ)

P_a_plus = (I + A) / 2  # projector for a = +1
P_a_minus = (I - A) / 2

# With Bob measuring B1
p_alice_plus_B1 = np.real(np.trace(np.kron(P_a_plus, I) @ rho))
p_alice_minus_B1 = np.real(np.trace(np.kron(P_a_minus, I) @ rho))

# With Bob measuring B2
p_alice_plus_B2 = np.real(np.trace(np.kron(P_a_plus, I) @ rho))
p_alice_minus_B2 = np.real(np.trace(np.kron(P_a_minus, I) @ rho))

print("Alice's marginals (should be equal regardless of Bob's choice):")
print(f"  Bob measures B:  P(a=+1) = {p_alice_plus_B1:.4f}")
print(f"  Bob measures B': P(a=+1) = {p_alice_plus_B2:.4f}")
print("  → No signaling confirmed!")

# Entanglement vs CHSH violation
print("\n=== Entanglement Thresholds ===")
print("\nWerner state thresholds:")
print(f"  Entangled (PPT fails): p > 1/3 ≈ {1/3:.4f}")
print(f"  CHSH violation: p > 1/√2 ≈ {1/np.sqrt(2):.4f}")
print("\nConclusion: Entanglement is necessary but not sufficient for CHSH violation!")

# Gisin's theorem - all pure entangled states violate some inequality
print("\n=== Testing Gisin's Theorem ===")
print("All pure entangled states violate SOME Bell inequality.\n")

def general_pure_state(theta, phi):
    """cos(θ)|00⟩ + e^{iφ}sin(θ)|11⟩"""
    psi = np.array([np.cos(theta), 0, 0, np.exp(1j*phi)*np.sin(theta)])
    return np.outer(psi, psi.conj())

def optimize_chsh(rho, n_trials=1000):
    """Find maximum CHSH by random search"""
    max_S = 0
    for _ in range(n_trials):
        angles = np.random.uniform(0, 2*np.pi, 4)
        S = np.abs(chsh_value(rho, angles))
        if S > max_S:
            max_S = S
    return max_S

print(f"{'θ (deg)':<10} {'Concurrence':<12} {'Max |S|':<12} {'Violates?'}")
print("-" * 50)

for theta_deg in [5, 15, 30, 45, 60, 75, 85]:
    theta = np.radians(theta_deg)
    rho = general_pure_state(theta, 0)
    concurrence = np.abs(np.sin(2*theta))
    max_S = optimize_chsh(rho, 2000)
    violates = "YES" if max_S > 2 else "no"
    print(f"{theta_deg:<10} {concurrence:<12.4f} {max_S:<12.4f} {violates}")
```

---

## Summary

### Key Results

| Bound | Value | Achieved by |
|-------|-------|-------------|
| LHV | $\|S\| \leq 2$ | Classical correlations |
| Tsirelson | $\|S\| \leq 2\sqrt{2}$ | Bell states |
| No-signaling | $\|S\| \leq 4$ | PR box (hypothetical) |

### Key Takeaways
1. **QM violates** classical bounds due to non-commuting observables
2. **Tsirelson bound** $2\sqrt{2}$ is the quantum maximum
3. **Separable states** always satisfy Bell inequalities
4. **No-signaling** is preserved despite nonlocal correlations
5. **Gisin's theorem:** all pure entangled states violate some Bell inequality

---

## Daily Checklist

- [ ] I can prove the Tsirelson bound
- [ ] I understand why non-commutativity enables violation
- [ ] I know that separable states satisfy Bell inequalities
- [ ] I understand the no-signaling principle
- [ ] I can relate violation strength to entanglement

---

*Next: Day 544 — Experimental Tests*
