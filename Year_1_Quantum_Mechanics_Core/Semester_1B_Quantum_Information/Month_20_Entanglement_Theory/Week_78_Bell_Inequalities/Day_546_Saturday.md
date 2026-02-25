# Day 546: Week 78 Review — Bell Inequalities

## Overview
**Day 546** | Week 78, Day 7 | Year 1, Month 20 | Synthesis and Assessment

Today we consolidate all Bell inequality concepts from Week 78 through comprehensive review and historical perspective.

---

## Week 78 Concept Map

```
                     BELL INEQUALITIES
                           │
    ┌──────────────────────┼──────────────────────┐
    │                      │                      │
 EPR PARADOX          INEQUALITIES            APPLICATIONS
    │                      │                      │
• Local realism       • Bell original        • Experiments
• Hidden variables    • CHSH (S≤2)           • Loopholes
• Einstein's         • Tsirelson (2√2)      • DIQKD
  challenge          • QM violation         • Self-testing
    │                      │                      │
    └──────────────────────┴──────────────────────┘
                           │
              QUANTUM NONLOCALITY CONFIRMED
```

---

## Master Formula Reference

### CHSH Inequality
$$S = \langle AB \rangle - \langle AB' \rangle + \langle A'B \rangle + \langle A'B' \rangle$$

| Bound | Value | Meaning |
|-------|-------|---------|
| Classical (LHV) | $\|S\| \leq 2$ | Local realism |
| Quantum (Tsirelson) | $\|S\| \leq 2\sqrt{2}$ | QM maximum |
| No-signaling | $\|S\| \leq 4$ | Theoretical maximum |

### Optimal Quantum Settings
- Alice: $\theta_A = 0$, $\theta_{A'} = 90°$
- Bob: $\theta_B = 45°$, $\theta_{B'} = 135°$
- Achieves: $S = -2\sqrt{2} \approx -2.828$

### Key Thresholds
| Quantity | Value | Significance |
|----------|-------|--------------|
| CHSH violation | $S > 2$ | Certifies entanglement |
| Werner threshold | $p > 1/\sqrt{2}$ | CHSH violates |
| Detection efficiency | $\eta > 82.84\%$ | Closes detection loophole |

---

## Historical Timeline

| Year | Event | Significance |
|------|-------|--------------|
| 1935 | EPR paper | Challenge to QM completeness |
| 1964 | Bell's theorem | LHV theories testable |
| 1969 | CHSH inequality | Experimentally practical form |
| 1972 | Freedman-Clauser | First test, violation seen |
| 1982 | Aspect experiments | Locality loophole addressed |
| 1998 | Weihs et al. | Strict space-like separation |
| 2015 | Loophole-free tests | All loopholes closed |
| 2022 | Nobel Prize | Aspect, Clauser, Zeilinger |

---

## Comprehensive Problem Set

### Problem 1: CHSH Calculation
For the state $|\psi\rangle = \frac{1}{\sqrt{3}}(|00\rangle + \sqrt{2}|11\rangle)$:

a) Find the maximum CHSH value
b) Does it violate the classical bound?

**Solution:**

a) The state has concurrence $C = 2|\alpha\beta| = 2 \cdot \frac{1}{\sqrt{3}} \cdot \frac{\sqrt{2}}{\sqrt{3}} = \frac{2\sqrt{2}}{3}$

Maximum CHSH: $S_{max} = 2\sqrt{1 + C^2} = 2\sqrt{1 + 8/9} = 2\sqrt{17/9} \approx 2.75$

b) Yes, $2.75 > 2$ violates classical bound! ∎

### Problem 2: Detection Efficiency
If your detector efficiency is 75%, can you close the detection loophole with CHSH?

**Solution:**
Critical efficiency for CHSH: $\eta_c = 2/(1 + \sqrt{2}) \approx 82.84\%$

Since $75\% < 82.84\%$, **cannot** close detection loophole with CHSH.

Options:
- Use CH inequality (lower threshold)
- Improve detector efficiency
- Use different inequality with lower threshold ∎

### Problem 3: Local Hidden Variable Model
Construct the best deterministic LHV model for singlet measurements at 0° and 45°.

**Solution:**
LHV must give $a, b = \pm 1$ based on hidden angle λ.

Best strategy:
- Alice: $a = \text{sign}(\cos(\theta_A - \lambda))$
- Bob: $b = -\text{sign}(\cos(\theta_B - \lambda))$ (opposite for anti-correlation)

For θ_A = 0°, θ_B = 45°:
$$\langle AB \rangle_{LHV} = -\frac{2}{\pi}\cos^{-1}(\cos(45°)) = -\frac{2}{\pi} \cdot 45° = -0.5$$

QM predicts: $-\cos(45°) = -0.707$

LHV cannot match QM correlation! ∎

### Problem 4: Self-Testing
If you observe S = 2.75, what can you certify about the quantum state?

**Solution:**
Self-testing bound: distance to ideal Bell state scales as $\sqrt{2\sqrt{2} - S}$

$$\epsilon = 2\sqrt{2} - 2.75 = 2.828 - 2.75 = 0.078$$
$$d \lesssim \sqrt{0.078} \approx 0.28$$

The state is within trace distance ~0.28 of a perfect Bell state. ∎

### Problem 5: DIQKD Security
Calculate the secure key rate for DIQKD with S = 2.5.

**Solution:**
Using the key rate formula:
$$\sqrt{(S/2)^2 - 1} = \sqrt{1.5625 - 1} = 0.75$$
$$p = (1 + 0.75)/2 = 0.875$$
$$h(0.875) \approx 0.544$$
$$r = 1 - 0.544 = 0.456$$

About 0.46 secure bits per round! ∎

---

## Computational Lab: Week Integration

```python
"""Day 546: Week 78 Integration - Bell Inequalities"""
import numpy as np
from scipy.linalg import eigvalsh

# ===== Pauli matrices =====
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.eye(2, dtype=complex)

def spin_op(theta):
    """Spin operator in x-z plane"""
    return np.cos(theta)*Z + np.sin(theta)*X

def correlation(state, opA, opB):
    """⟨ψ|A⊗B|ψ⟩"""
    return np.real(state.conj() @ np.kron(opA, opB) @ state)

def chsh(state, angles):
    """Compute CHSH parameter"""
    a, ap, b, bp = [spin_op(θ) for θ in angles]
    return (correlation(state, a, b) - correlation(state, a, bp) +
            correlation(state, ap, b) + correlation(state, ap, bp))

# ===== Bell states =====
singlet = np.array([0, 1, -1, 0]) / np.sqrt(2)
phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)

# ===== Week 78 Summary =====
print("=" * 60)
print("WEEK 78: BELL INEQUALITIES - COMPREHENSIVE REVIEW")
print("=" * 60)

# 1. CHSH bounds
print("\n1. CHSH BOUNDS")
print("-" * 40)
opt = (0, np.pi/2, np.pi/4, 3*np.pi/4)
S_qm = chsh(singlet, opt)
print(f"Classical bound:  |S| ≤ 2")
print(f"Quantum maximum:  |S| ≤ 2√2 ≈ {2*np.sqrt(2):.4f}")
print(f"Singlet (optimal): S = {S_qm:.4f}")

# 2. Various states
print("\n2. CHSH FOR VARIOUS STATES")
print("-" * 40)

def make_state(theta):
    """cos(θ)|00⟩ + sin(θ)|11⟩"""
    return np.array([np.cos(theta), 0, 0, np.sin(theta)])

states = [
    ("Singlet |Ψ⁻⟩", singlet),
    ("Bell |Φ⁺⟩", phi_plus),
    ("Product |00⟩", np.array([1,0,0,0])),
    ("Partial (θ=30°)", make_state(np.pi/6)),
    ("Partial (θ=60°)", make_state(np.pi/3)),
]

print(f"{'State':<20} {'Max |S|':<12} {'Violates?'}")
print("-" * 40)

for name, state in states:
    # Optimize over angles
    max_S = 0
    for a in np.linspace(0, np.pi, 15):
        for ap in np.linspace(0, np.pi, 15):
            for b in np.linspace(0, np.pi, 15):
                for bp in np.linspace(0, np.pi, 15):
                    S = np.abs(chsh(state, (a, ap, b, bp)))
                    max_S = max(max_S, S)

    violates = "YES" if max_S > 2 else "no"
    print(f"{name:<20} {max_S:<12.4f} {violates}")

# 3. LHV simulation
print("\n3. LOCAL HIDDEN VARIABLE SIMULATION")
print("-" * 40)

def lhv_correlation(theta_a, theta_b, n=10000):
    """Simulate LHV model"""
    correlations = []
    for _ in range(n):
        lambda_angle = np.random.uniform(0, 2*np.pi)
        a = np.sign(np.cos(theta_a - lambda_angle))
        b = -np.sign(np.cos(theta_b - lambda_angle))  # anti-corr
        correlations.append(a * b)
    return np.mean(correlations)

print(f"{'θ_A':<8} {'θ_B':<8} {'QM':<12} {'LHV':<12}")
print("-" * 40)

for ta, tb in [(0, 0), (0, np.pi/4), (0, np.pi/2), (np.pi/4, 3*np.pi/4)]:
    qm = correlation(singlet, spin_op(ta), spin_op(tb))
    lhv = lhv_correlation(ta, tb)
    print(f"{np.degrees(ta):<8.0f} {np.degrees(tb):<8.0f} {qm:<12.4f} {lhv:<12.4f}")

# 4. Detection efficiency
print("\n4. DETECTION LOOPHOLE")
print("-" * 40)

eta_crit = 2 / (1 + np.sqrt(2))
print(f"Critical efficiency for CHSH: η > {eta_crit:.4f} = {eta_crit*100:.2f}%")

def effective_S(S_true, eta):
    p = eta**2
    return p * S_true + (1-p) * 2  # worst case

print(f"\n{'Efficiency':<12} {'Effective S':<12} {'Violates?'}")
print("-" * 40)
for eta in [0.70, 0.80, 0.82, 0.83, 0.85, 0.90, 1.00]:
    S_eff = effective_S(2*np.sqrt(2), eta)
    violates = "YES" if S_eff > 2 else "no"
    print(f"{eta*100:<12.1f}% {S_eff:<12.4f} {violates}")

# 5. Historical milestones
print("\n5. EXPERIMENTAL MILESTONES")
print("-" * 40)

experiments = [
    (1972, "Freedman-Clauser", 2.85, "First test"),
    (1982, "Aspect", 2.70, "Fast switching"),
    (1998, "Weihs", 2.73, "Strict locality"),
    (2015, "Delft", 2.42, "Loophole-free"),
]

for year, team, S, note in experiments:
    sigma = (np.abs(S) - 2) * 10  # rough significance
    print(f"{year}: {team:<20} S={S:.2f} ({sigma:.1f}σ) - {note}")

# 6. Key takeaways
print("\n" + "=" * 60)
print("WEEK 78 KEY TAKEAWAYS")
print("=" * 60)
print("""
1. EPR challenged QM completeness with local realism

2. Bell proved: LHV theories → |S| ≤ 2 (testable!)

3. QM predicts: |S| ≤ 2√2 ≈ 2.828 (Tsirelson bound)

4. Experiments confirm QM, ruling out local realism

5. Loopholes (locality, detection, freedom) now all closed

6. Applications: Device-independent QKD, randomness certification

7. 2022 Nobel Prize: Aspect, Clauser, Zeilinger
""")
```

---

## Self-Assessment Checklist

### Conceptual Understanding
- [ ] I can explain the EPR paradox
- [ ] I understand Bell's theorem and its implications
- [ ] I know the CHSH inequality and its bounds
- [ ] I can explain experimental loopholes

### Computational Skills
- [ ] I can compute CHSH for given states and settings
- [ ] I can find optimal measurement angles
- [ ] I can analyze detection efficiency requirements
- [ ] I can simulate LHV models

### Historical Knowledge
- [ ] I know the key experiments from 1972-2015
- [ ] I understand why 2015 was significant
- [ ] I know about device-independent applications

---

## Looking Ahead: Week 79

Next week covers **Entanglement Measures**:
- Von Neumann entropy
- Entropy of entanglement
- Concurrence
- Negativity
- Operational measures

---

*Next: Day 547 — Von Neumann Entropy*
