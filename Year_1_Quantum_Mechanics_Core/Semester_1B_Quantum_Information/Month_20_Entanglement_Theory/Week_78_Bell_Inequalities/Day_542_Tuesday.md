# Day 542: CHSH Inequality

## Overview
**Day 542** | Week 78, Day 3 | Year 1, Month 20 | The Experimentally Testable Form

Today we study the CHSH inequality—the most important Bell inequality for experiments, requiring only four measurement settings total.

---

## Learning Objectives
1. Derive the CHSH inequality
2. Understand its experimental advantages
3. Calculate the quantum violation
4. Find optimal measurement settings
5. Compute the Tsirelson bound
6. Connect to nonlocality measures

---

## Core Content

### CHSH Setup

**Clauser-Horne-Shimony-Holt (1969)**

Alice chooses between A or A' (results ±1)
Bob chooses between B or B' (results ±1)

The **CHSH parameter:**
$$\boxed{S = \langle AB \rangle - \langle AB' \rangle + \langle A'B \rangle + \langle A'B' \rangle}$$

### Classical Bound

**Theorem:** For any local hidden variable model:
$$\boxed{|S| \leq 2}$$

**Proof:**
For fixed λ, define: $S(\lambda) = AB - AB' + A'B + A'B'$

Factor: $S(\lambda) = A(B - B') + A'(B + B')$

Since $B, B' = \pm 1$:
- If $B = B'$: $S(\lambda) = 0 + A'(\pm 2) = \pm 2$
- If $B = -B'$: $S(\lambda) = A(\pm 2) + 0 = \pm 2$

Therefore $|S(\lambda)| = 2$ for every λ.

Averaging: $|\langle S \rangle| = |\int d\lambda \rho(\lambda) S(\lambda)| \leq 2$ ∎

### Quantum Mechanical Value

For singlet state and spin measurements:
$$\langle AB \rangle = -\cos(\theta_A - \theta_B)$$

**Optimal angles:**
- $\theta_A = 0$
- $\theta_{A'} = \pi/2$
- $\theta_B = \pi/4$
- $\theta_{B'} = 3\pi/4$

Calculate each term:
- $\langle AB \rangle = -\cos(-\pi/4) = -\frac{1}{\sqrt{2}}$
- $\langle AB' \rangle = -\cos(-3\pi/4) = \frac{1}{\sqrt{2}}$
- $\langle A'B \rangle = -\cos(\pi/4) = -\frac{1}{\sqrt{2}}$
- $\langle A'B' \rangle = -\cos(-\pi/4) = -\frac{1}{\sqrt{2}}$

$$S_{QM} = -\frac{1}{\sqrt{2}} - \frac{1}{\sqrt{2}} - \frac{1}{\sqrt{2}} - \frac{1}{\sqrt{2}} = -\frac{4}{\sqrt{2}} = -2\sqrt{2}$$

$$\boxed{|S_{QM}| = 2\sqrt{2} \approx 2.828 > 2}$$

### Tsirelson Bound

**Theorem (Tsirelson, 1980):** For any quantum state and observables:
$$\boxed{|S| \leq 2\sqrt{2}}$$

This is the **maximum quantum violation** of CHSH!

**Proof sketch:** Uses the operator inequality
$$(A \otimes B - A \otimes B' + A' \otimes B + A' \otimes B')^2 \leq 8I$$

### Diagram of Settings

```
                B' (135°)
                   ↑
                   │
         A' (90°)──┼── A (0°)
                   │
                   ↓
                B (45°)
```

Alice and Bob's optimal angles are offset by 45°.

### CHSH as Nonlocality Witness

**S > 2:** Certifies entanglement without trusting devices!

This is the foundation of **device-independent** quantum information.

### Generalized CHSH

For non-maximally entangled states $|\psi\rangle = \cos\phi|00\rangle + \sin\phi|11\rangle$:

$$S_{max} = 2\sqrt{1 + \sin^2(2\phi)}$$

Maximum $2\sqrt{2}$ at $\phi = \pi/4$ (Bell state).

---

## Worked Examples

### Example 1: Verify CHSH for Optimal Settings
Compute S explicitly for the singlet state with optimal angles.

**Solution:**
Singlet: $|\Psi^-\rangle = (|01\rangle - |10\rangle)/\sqrt{2}$

Correlation: $\langle \sigma_{\hat{n}} \otimes \sigma_{\hat{m}} \rangle = -\hat{n} \cdot \hat{m}$

Angles (in x-z plane):
- A at $\theta = 0$ → $\hat{n}_A = \hat{z}$
- A' at $\theta = 90°$ → $\hat{n}_{A'} = \hat{x}$
- B at $\theta = 45°$ → $\hat{n}_B = (\hat{z} + \hat{x})/\sqrt{2}$
- B' at $\theta = 135°$ → $\hat{n}_{B'} = (-\hat{z} + \hat{x})/\sqrt{2}$

$$\langle AB \rangle = -\hat{z} \cdot \frac{\hat{z}+\hat{x}}{\sqrt{2}} = -\frac{1}{\sqrt{2}}$$
$$\langle AB' \rangle = -\hat{z} \cdot \frac{-\hat{z}+\hat{x}}{\sqrt{2}} = \frac{1}{\sqrt{2}}$$
$$\langle A'B \rangle = -\hat{x} \cdot \frac{\hat{z}+\hat{x}}{\sqrt{2}} = -\frac{1}{\sqrt{2}}$$
$$\langle A'B' \rangle = -\hat{x} \cdot \frac{-\hat{z}+\hat{x}}{\sqrt{2}} = -\frac{1}{\sqrt{2}}$$

$$S = -\frac{1}{\sqrt{2}} - \frac{1}{\sqrt{2}} - \frac{1}{\sqrt{2}} - \frac{1}{\sqrt{2}} = -2\sqrt{2}$$ ∎

### Example 2: Non-Optimal Angles
Compute S for settings at 0°, 90° (Alice) and 0°, 90° (Bob).

**Solution:**
$$\langle AB \rangle = -\cos(0°) = -1$$
$$\langle AB' \rangle = -\cos(-90°) = 0$$
$$\langle A'B \rangle = -\cos(90°) = 0$$
$$\langle A'B' \rangle = -\cos(0°) = -1$$

$$S = -1 - 0 + 0 - 1 = -2$$

Just at the classical bound—no violation! ∎

### Example 3: Werner State CHSH
For Werner state $\rho_W = p|\Psi^-\rangle\langle\Psi^-| + (1-p)I/4$, find S.

**Solution:**
Correlations scale linearly:
$$\langle AB \rangle_W = p \cdot (-\frac{1}{\sqrt{2}}) = -\frac{p}{\sqrt{2}}$$

$$S_W = p \cdot (-2\sqrt{2}) = -2\sqrt{2}p$$

Violation requires $|S| > 2$: $2\sqrt{2}p > 2$ → $p > 1/\sqrt{2} \approx 0.707$

Werner state violates CHSH for $p > 1/\sqrt{2}$. ∎

---

## Practice Problems

### Problem 1: Alternative Settings
Find S for Alice at 0°, 45° and Bob at 22.5°, 67.5°.

### Problem 2: Tsirelson Bound Proof
Prove that $(A⊗B - A⊗B' + A'⊗B + A'⊗B')^2 \leq 8I$ for dichotomic observables.

### Problem 3: Mixed State Threshold
For the isotropic state $\rho_F$, find the CHSH violation threshold.

---

## Computational Lab

```python
"""Day 542: CHSH Inequality"""
import numpy as np

# Pauli matrices
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.eye(2, dtype=complex)

def spin_operator(theta, phi=0):
    """Spin measurement along direction (θ, φ) in spherical coords"""
    return np.sin(theta)*np.cos(phi)*X + np.sin(theta)*np.sin(phi)*Y + np.cos(theta)*Z

def spin_operator_2d(angle):
    """Spin in x-z plane at angle from z-axis"""
    return np.cos(angle)*Z + np.sin(angle)*X

def correlation(state, op_A, op_B):
    """Compute ⟨ψ|A⊗B|ψ⟩"""
    joint_op = np.kron(op_A, op_B)
    return np.real(state.conj() @ joint_op @ state)

def chsh_value(state, theta_A, theta_Ap, theta_B, theta_Bp):
    """Compute CHSH parameter S"""
    A = spin_operator_2d(theta_A)
    Ap = spin_operator_2d(theta_Ap)
    B = spin_operator_2d(theta_B)
    Bp = spin_operator_2d(theta_Bp)

    S = (correlation(state, A, B) - correlation(state, A, Bp) +
         correlation(state, Ap, B) + correlation(state, Ap, Bp))
    return S

# Singlet state
psi_minus = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)

print("=== CHSH Inequality Analysis ===\n")

# Optimal settings
theta_A = 0
theta_Ap = np.pi/2
theta_B = np.pi/4
theta_Bp = 3*np.pi/4

S_opt = chsh_value(psi_minus, theta_A, theta_Ap, theta_B, theta_Bp)
print(f"Optimal CHSH value: S = {S_opt:.6f}")
print(f"Expected: -2√2 = {-2*np.sqrt(2):.6f}")
print(f"Classical bound: |S| ≤ 2")
print(f"Violation: {np.abs(S_opt) - 2:.4f} above classical bound")

# Verify each term
print("\n=== Individual Correlations ===")
print(f"{'Term':<10} {'Value':>10} {'Expected':>12}")
print("-" * 35)

A = spin_operator_2d(theta_A)
Ap = spin_operator_2d(theta_Ap)
B = spin_operator_2d(theta_B)
Bp = spin_operator_2d(theta_Bp)

terms = [
    ("⟨AB⟩", correlation(psi_minus, A, B), -1/np.sqrt(2)),
    ("⟨AB'⟩", correlation(psi_minus, A, Bp), 1/np.sqrt(2)),
    ("⟨A'B⟩", correlation(psi_minus, Ap, B), -1/np.sqrt(2)),
    ("⟨A'B'⟩", correlation(psi_minus, Ap, Bp), -1/np.sqrt(2)),
]

for name, val, exp in terms:
    print(f"{name:<10} {val:>10.6f} {exp:>12.6f}")

# Scan over angles to find maximum
print("\n=== Scanning for Maximum Violation ===")

max_S = 0
best_angles = None

for a in np.linspace(0, np.pi, 20):
    for ap in np.linspace(0, np.pi, 20):
        for b in np.linspace(0, np.pi, 20):
            for bp in np.linspace(0, np.pi, 20):
                S = chsh_value(psi_minus, a, ap, b, bp)
                if np.abs(S) > np.abs(max_S):
                    max_S = S
                    best_angles = (a, ap, b, bp)

print(f"Maximum |S| found: {np.abs(max_S):.6f}")
print(f"Tsirelson bound: {2*np.sqrt(2):.6f}")

# Werner state analysis
print("\n=== Werner State CHSH Analysis ===")
print("ρ(p) = p|Ψ⁻⟩⟨Ψ⁻| + (1-p)I/4\n")

def werner_chsh(p):
    """CHSH value for Werner state (scales linearly)"""
    return p * chsh_value(psi_minus, theta_A, theta_Ap, theta_B, theta_Bp)

print(f"{'p':<8} {'S':<12} {'|S| > 2?'}")
print("-" * 30)

for p in [0.5, 0.6, 0.707, 0.71, 0.8, 0.9, 1.0]:
    S = werner_chsh(p)
    violates = "YES" if np.abs(S) > 2 else "no"
    print(f"{p:<8.3f} {S:<12.4f} {violates}")

print(f"\nViolation threshold: p > 1/√2 ≈ {1/np.sqrt(2):.4f}")

# Non-maximally entangled state
print("\n=== Non-Maximally Entangled States ===")
print("|ψ(φ)⟩ = cos(φ)|00⟩ + sin(φ)|11⟩\n")

def non_max_entangled(phi):
    """Create state cos(φ)|00⟩ + sin(φ)|11⟩"""
    return np.array([np.cos(phi), 0, 0, np.sin(phi)], dtype=complex)

print(f"{'φ (deg)':<10} {'Max S':<12} {'Violates?'}")
print("-" * 35)

for phi_deg in [0, 15, 30, 45, 60, 75, 90]:
    phi = np.radians(phi_deg)
    state = non_max_entangled(phi)

    # Find optimal CHSH for this state
    max_S_state = 0
    for a in np.linspace(0, np.pi, 15):
        for ap in np.linspace(0, np.pi, 15):
            for b in np.linspace(0, np.pi, 15):
                for bp in np.linspace(0, np.pi, 15):
                    S = chsh_value(state, a, ap, b, bp)
                    if np.abs(S) > np.abs(max_S_state):
                        max_S_state = S

    violates = "YES" if np.abs(max_S_state) > 2 else "no"
    print(f"{phi_deg:<10} {np.abs(max_S_state):<12.4f} {violates}")

# Tsirelson bound verification
print("\n=== Tsirelson Bound Verification ===")

# Bell operator
def bell_operator(theta_A, theta_Ap, theta_B, theta_Bp):
    """Construct CHSH Bell operator"""
    A = spin_operator_2d(theta_A)
    Ap = spin_operator_2d(theta_Ap)
    B = spin_operator_2d(theta_B)
    Bp = spin_operator_2d(theta_Bp)

    return (np.kron(A, B) - np.kron(A, Bp) +
            np.kron(Ap, B) + np.kron(Ap, Bp))

bell_op = bell_operator(theta_A, theta_Ap, theta_B, theta_Bp)
eigenvalues = np.linalg.eigvalsh(bell_op)
print(f"Bell operator eigenvalues: {eigenvalues}")
print(f"Maximum eigenvalue: {np.max(eigenvalues):.6f}")
print(f"This equals 2√2 = {2*np.sqrt(2):.6f}")
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| CHSH parameter | $S = \langle AB \rangle - \langle AB' \rangle + \langle A'B \rangle + \langle A'B' \rangle$ |
| Classical bound | $\|S\| \leq 2$ |
| Tsirelson bound | $\|S\| \leq 2\sqrt{2}$ |
| QM maximum | $S = 2\sqrt{2}$ for Bell states |
| Optimal angles | A:0°, A':90°, B:45°, B':135° |

### Key Takeaways
1. **CHSH** is the most experimentally practical Bell inequality
2. **Classical bound** is 2, quantum maximum is $2\sqrt{2}$
3. **Optimal settings** are equally spaced at 45° intervals
4. **Tsirelson bound** is the absolute QM maximum
5. **Device-independent** security relies on CHSH violation

---

## Daily Checklist

- [ ] I can derive the CHSH inequality
- [ ] I know the optimal measurement settings
- [ ] I can compute S for given settings
- [ ] I understand the Tsirelson bound
- [ ] I can analyze CHSH for various states

---

*Next: Day 543 — Quantum Violation*
