# Day 541: Bell's Theorem

## Overview
**Day 541** | Week 78, Day 2 | Year 1, Month 20 | The Most Profound Discovery

Today we prove Bell's theorem—the mathematical result showing that no local hidden variable theory can reproduce all quantum mechanical predictions.

---

## Learning Objectives
1. State Bell's theorem precisely
2. Derive the original Bell inequality
3. Understand the assumptions required
4. Show quantum mechanical violation
5. Appreciate the philosophical implications
6. Connect to experimental tests

---

## Core Content

### Bell's Theorem (1964)

**Theorem:** No local hidden variable theory can reproduce all predictions of quantum mechanics.

**Proof strategy:** Show that LHV theories satisfy certain inequalities that QM violates.

### Setting Up the Inequality

Consider measurements on an entangled pair:
- Alice measures observable A (result a = ±1) or A' (result a' = ±1)
- Bob measures observable B (result b = ±1) or B' (result b' = ±1)

### Local Hidden Variable Assumption

Outcomes are determined by hidden variable λ:
$$a = A(\hat{a}, \lambda), \quad b = B(\hat{b}, \lambda)$$

**Locality:** Alice's result depends only on her setting and λ
**Realism:** Results are predetermined by λ

### Original Bell Inequality

For three measurement directions $\hat{a}, \hat{b}, \hat{c}$:

$$\boxed{|P(\hat{a}, \hat{b}) - P(\hat{a}, \hat{c})| \leq 1 + P(\hat{b}, \hat{c})}$$

where $P(\hat{x}, \hat{y}) = \langle A(\hat{x}) B(\hat{y}) \rangle$ is the correlation.

### Derivation

**Step 1:** For any trial with hidden variable λ:
$$A(\hat{a},\lambda)B(\hat{b},\lambda) - A(\hat{a},\lambda)B(\hat{c},\lambda) = A(\hat{a},\lambda)B(\hat{b},\lambda)[1 - B(\hat{b},\lambda)B(\hat{c},\lambda)]$$

Using $B^2 = 1$:
$$= A(\hat{a},\lambda)B(\hat{b},\lambda)[1 \mp 1]$$

**Step 2:** Average over λ:
$$P(\hat{a},\hat{b}) - P(\hat{a},\hat{c}) = \int d\lambda \rho(\lambda) A(\hat{a})B(\hat{b})[1 - B(\hat{b})B(\hat{c})]$$

**Step 3:** Take absolute value and use $|AB| \leq 1$:
$$|P(\hat{a},\hat{b}) - P(\hat{a},\hat{c})| \leq \int d\lambda \rho(\lambda)[1 - B(\hat{b})B(\hat{c})]$$
$$= 1 + P(\hat{b},\hat{c})$$

### Quantum Mechanical Prediction

For singlet state $|\Psi^-\rangle$:
$$P_{QM}(\hat{a}, \hat{b}) = -\hat{a} \cdot \hat{b} = -\cos\theta_{ab}$$

### Violation Example

Choose angles: $\theta_{ab} = \theta_{ac} = 60°$, $\theta_{bc} = 120°$

LHV bound:
$$|P(\hat{a},\hat{b}) - P(\hat{a},\hat{c})| \leq 1 + P(\hat{b},\hat{c})$$
$$|(-\cos 60°) - (-\cos 60°)| \leq 1 + (-\cos 120°)$$
$$0 \leq 1 + 0.5 = 1.5$$ ✓ (satisfied)

Better choice: $\theta_{ab} = \theta_{bc} = 60°$, $\theta_{ac} = 120°$

QM values: $P(\hat{a},\hat{b}) = -0.5$, $P(\hat{b},\hat{c}) = -0.5$, $P(\hat{a},\hat{c}) = 0.5$

$$|-0.5 - 0.5| = 1 \leq 1 + (-0.5) = 0.5$$ ✗ **Violated!**

### Implications

1. **At least one** LHV assumption is wrong
2. **Locality or realism** (or both) must be abandoned
3. **Nature is nonlocal** in some sense
4. **Quantum correlations** are stronger than classical

### Fine Print: Assumptions

Bell's theorem requires:
1. **Locality:** No FTL signaling of settings
2. **Realism:** Outcomes exist before measurement
3. **Freedom:** Measurement choices are independent of λ
4. **Fair sampling:** Detected pairs represent all pairs

Each assumption corresponds to a "loophole."

---

## Worked Examples

### Example 1: Verify Bell Inequality for LHV
Show that predetermined values satisfy Bell's inequality.

**Solution:**
Let $A = \pm 1$, $A' = \pm 1$, $B = \pm 1$, $B' = \pm 1$ be fixed values.

Consider: $AB - AB' + A'B + A'B' = A(B - B') + A'(B + B')$

Since $B, B' = \pm 1$:
- Either $B = B'$: then $B - B' = 0$, $B + B' = \pm 2$, sum $= \pm 2A' = \pm 2$
- Or $B \neq B'$: then $B - B' = \pm 2$, $B + B' = 0$, sum $= \pm 2A = \pm 2$

Averaged: $|\langle AB \rangle - \langle AB' \rangle + \langle A'B \rangle + \langle A'B' \rangle| \leq 2$ ∎

### Example 2: Maximum QM Violation
For which angles is the Bell inequality maximally violated?

**Solution:**
For the original Bell inequality with three settings, maximum violation occurs at specific angles.

For CHSH (next day), the optimal angles are:
- $A$ at 0°, $A'$ at 90°
- $B$ at 45°, $B'$ at 135°

This gives $S = 2\sqrt{2} \approx 2.83 > 2$. ∎

### Example 3: Classical Limit
If Alice and Bob share classical randomness only, what's the maximum correlation?

**Solution:**
Shared randomness = shared λ, but still LHV.
Maximum correlation: obey Bell inequality.
For any pair of settings: $|\langle AB \rangle| \leq 1$.
Combined (CHSH): $|S| \leq 2$.

Classical correlations cannot exceed the Bell bound! ∎

---

## Practice Problems

### Problem 1: Three-Setting Inequality
Derive the Bell inequality for settings at 0°, 120°, 240°.

### Problem 2: Product States
Show that product states (not entangled) satisfy all Bell inequalities.

### Problem 3: Deterministic Model
Construct the best deterministic LHV model for the singlet. What's its maximum S?

---

## Computational Lab

```python
"""Day 541: Bell's Theorem"""
import numpy as np
from itertools import product

def correlation_qm(theta_a, theta_b):
    """QM correlation for singlet: -cos(θ_a - θ_b)"""
    return -np.cos(theta_a - theta_b)

def bell_inequality_check(theta_a, theta_b, theta_c):
    """
    Check Bell inequality:
    |P(a,b) - P(a,c)| ≤ 1 + P(b,c)
    """
    P_ab = correlation_qm(theta_a, theta_b)
    P_ac = correlation_qm(theta_a, theta_c)
    P_bc = correlation_qm(theta_b, theta_c)

    lhs = np.abs(P_ab - P_ac)
    rhs = 1 + P_bc

    return lhs, rhs, lhs <= rhs + 1e-10

print("=== Bell's Original Inequality ===\n")
print("|P(a,b) - P(a,c)| ≤ 1 + P(b,c)\n")

# Test various angle configurations
test_configs = [
    ("0°, 60°, 120°", 0, np.pi/3, 2*np.pi/3),
    ("0°, 45°, 90°", 0, np.pi/4, np.pi/2),
    ("0°, 30°, 60°", 0, np.pi/6, np.pi/3),
    ("0°, 22.5°, 67.5°", 0, np.pi/8, 3*np.pi/8),
]

print(f"{'Config':<20} {'LHS':>8} {'RHS':>8} {'Satisfied?'}")
print("-" * 45)

for name, a, b, c in test_configs:
    lhs, rhs, satisfied = bell_inequality_check(a, b, c)
    status = "✓" if satisfied else "✗ VIOLATED"
    print(f"{name:<20} {lhs:>8.4f} {rhs:>8.4f} {status}")

# Find maximum violation
print("\n=== Searching for Maximum Violation ===\n")

max_violation = 0
best_angles = None

for a in np.linspace(0, np.pi, 50):
    for b in np.linspace(0, np.pi, 50):
        for c in np.linspace(0, np.pi, 50):
            lhs, rhs, satisfied = bell_inequality_check(a, b, c)
            violation = lhs - rhs
            if violation > max_violation:
                max_violation = violation
                best_angles = (a, b, c)

print(f"Maximum violation: {max_violation:.4f}")
print(f"Best angles (radians): {best_angles}")
print(f"Best angles (degrees): {tuple(np.degrees(a) for a in best_angles)}")

# Deterministic LHV simulation
print("\n=== Deterministic LHV Model ===\n")

def lhv_deterministic(theta_a, theta_b, n_samples=10000):
    """
    Simulate deterministic LHV: each λ assigns ±1 to each angle.
    """
    correlations = []

    for _ in range(n_samples):
        # Hidden variable: a random direction
        lambda_angle = np.random.uniform(0, 2*np.pi)

        # Deterministic rule: +1 if measurement angle within 90° of λ
        a = 1 if np.abs(np.cos(theta_a - lambda_angle)) > 0.5 else -1
        b = 1 if np.abs(np.cos(theta_b - lambda_angle)) > 0.5 else -1

        # Actually use the sign
        a = np.sign(np.cos(theta_a - lambda_angle))
        b = -np.sign(np.cos(theta_b - lambda_angle))  # opposite for anti-correlation

        correlations.append(a * b)

    return np.mean(correlations)

print("Comparing QM and LHV correlations:")
print(f"{'θ_a':<8} {'θ_b':<8} {'QM':>10} {'LHV':>10}")
print("-" * 40)

for theta_a_deg, theta_b_deg in [(0, 0), (0, 45), (0, 90), (0, 180)]:
    theta_a = np.radians(theta_a_deg)
    theta_b = np.radians(theta_b_deg)
    qm = correlation_qm(theta_a, theta_b)
    lhv = lhv_deterministic(theta_a, theta_b)
    print(f"{theta_a_deg}°{'':<5} {theta_b_deg}°{'':<5} {qm:>10.4f} {lhv:>10.4f}")

# Visualize correlation function
print("\n=== QM vs LHV Correlation Function ===\n")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

angles = np.linspace(0, np.pi, 100)
qm_corr = [correlation_qm(0, a) for a in angles]
lhv_corr = [lhv_deterministic(0, a, 5000) for a in angles]

plt.figure(figsize=(10, 6))
plt.plot(np.degrees(angles), qm_corr, 'b-', linewidth=2, label='Quantum Mechanics')
plt.plot(np.degrees(angles), lhv_corr, 'r--', linewidth=2, label='LHV Model')
plt.xlabel('Angle difference (degrees)')
plt.ylabel('Correlation P(a,b)')
plt.title("Bell's Theorem: QM vs Local Hidden Variables")
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.savefig('bell_correlation.png', dpi=150, bbox_inches='tight')
print("Plot saved to bell_correlation.png")
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Bell inequality | $\|P(a,b) - P(a,c)\| \leq 1 + P(b,c)$ |
| QM correlation | $P_{QM}(\hat{a}, \hat{b}) = -\cos\theta$ |
| LHV model | $P(a,b) = \int d\lambda\, \rho(\lambda) A(\lambda) B(\lambda)$ |

### Key Takeaways
1. **Bell's theorem** rules out local hidden variables
2. **QM violates** Bell inequalities for entangled states
3. **Assumptions:** locality, realism, freedom, fair sampling
4. **Implication:** Nature is "nonlocal" in correlations
5. **Not signaling:** Still no FTL communication!

---

## Daily Checklist

- [ ] I can state Bell's theorem precisely
- [ ] I understand the derivation of Bell's inequality
- [ ] I can show quantum mechanical violation
- [ ] I understand what assumptions Bell requires
- [ ] I appreciate the philosophical implications

---

*Next: Day 542 — CHSH Inequality*
