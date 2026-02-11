# Day 626: Fixed-Point Amplitude Amplification

## Overview
**Day 626** | Week 90, Day 3 | Year 1, Month 23 | Amplitude Amplification

Today we study fixed-point amplitude amplification, which avoids the overshooting problem of standard Grover by ensuring monotonic convergence to the target state.

---

## Learning Objectives

1. Understand the overshooting problem in standard Grover
2. Define fixed-point amplification criteria
3. Derive fixed-point operators using phase adjustments
4. Analyze convergence properties
5. Implement fixed-point protocols
6. Compare to standard amplitude amplification

---

## Core Content

### The Overshooting Problem

In standard Grover/amplitude amplification:
- Success probability oscillates: $P(k) = \sin^2((2k+1)\theta)$
- If we don't know $\theta$, we might overshoot
- Too many iterations reduce success probability!

**Fixed-point amplification** solves this by ensuring:
$$P(k+1) \geq P(k) \text{ for all } k$$

### Fixed-Point Criterion

A fixed-point algorithm satisfies:
1. **Monotonicity:** Success probability never decreases
2. **Convergence:** $\lim_{k\to\infty} P(k) = 1$
3. **Target invariance:** If $P = 1$, it stays at 1

### The Phase Rotation Approach

**Key Idea:** Instead of using phases $\pm 1$ for reflections, use variable phases.

**Modified operators:**
$$S_\chi(\phi) = I - (1 - e^{i\phi})|good\rangle\langle good|$$
$$S_0(\phi) = I - (1 - e^{i\phi})|0\rangle\langle 0|$$

Standard Grover uses $\phi = \pi$, giving $e^{i\pi} = -1$.

### Grover-Long Algorithm

**Grover-Long (2001)** showed that using:
$$\phi = 2\arcsin\left(\frac{\sin\theta}{\sin(3\theta)}\right)$$

gives exact amplitude amplification in one iteration.

However, this requires knowing $\theta$!

### Fixed-Point via Composite Pulses

**Berry et al. (2014)** developed fixed-point sequences using ideas from NMR:

**The fixed-point operator:**
$$Q_{FP} = \prod_{j=1}^{L} Q(\phi_j)$$

where $Q(\phi) = -AS_0(\phi)A^{-1}S_\chi(\phi)$ with carefully chosen phases $\{\phi_j\}$.

### Yoder-Low-Chuang Algorithm

**YLC (2014)** provides an explicit fixed-point construction:

For target success probability $1 - \delta$:

$$\phi_j = 2\arctan\left(\frac{1}{\tan(\pi j/(2L+1))}\right)$$

where $L = O(\log(1/\delta)/\sqrt{a})$ composite iterations.

**Key result:** Success probability is:
$$P_L = 1 - (1-a)^{2L+1} \cdot h(L, a)$$

where $h$ is bounded, ensuring monotonic approach to 1.

### Comparison of Methods

| Method | Knows $\theta$? | Overshoots? | Query Complexity |
|--------|----------------|-------------|------------------|
| Standard Grover | Yes | Yes if wrong k | $O(1/\sqrt{a})$ |
| Fixed-point | No | No | $O(\log(1/\delta)/\sqrt{a})$ |
| Randomized | No | N/A | $O(1/\sqrt{a})$ |

**Note:** Fixed-point has a logarithmic overhead but guarantees convergence.

### Mathematical Analysis

For the YLC algorithm with $L$ composite operations:

**Theorem:** If initial success probability is $a$, then after $L$ composite iterations:
$$P_L \geq 1 - \delta$$

where $\delta = O\left(\frac{1}{(aL^2)^{L}}\right)$.

**Inverse:** To achieve $1 - \delta$ success:
$$L = O\left(\frac{\log(1/\delta)}{\sqrt{a}}\right)$$

### Circuit Structure

```
|0⟩ ─[A]─[S_χ(φ₁)]─[A†S₀(φ₁)A]─[S_χ(φ₂)]─[A†S₀(φ₂)A]─ ... ─[Measure]
```

Each composite iteration requires:
- 1 call to $S_\chi(\phi_j)$
- 1 call to $A^\dagger$
- 1 call to $S_0(\phi_j)$
- 1 call to $A$

---

## Worked Examples

### Example 1: Overshooting Demonstration
For $a = 0.1$, show the overshooting in standard Grover.

**Solution:**
$\theta = \arcsin(\sqrt{0.1}) = 0.3217$ rad

Optimal iterations: $k_{opt} = \lfloor\pi/(4\theta)\rfloor = \lfloor 2.44 \rfloor = 2$

Success probabilities:
- $k=0$: $\sin^2(0.3217) = 0.1$
- $k=1$: $\sin^2(0.9651) = 0.673$
- $k=2$: $\sin^2(1.608) = 0.999$
- $k=3$: $\sin^2(2.252) = 0.631$ (overshoot!)
- $k=4$: $\sin^2(2.896) = 0.054$ (back down!)

### Example 2: Fixed-Point Phase Calculation
Calculate the YLC phases for $L = 2$.

**Solution:**
$$\phi_j = 2\arctan\left(\frac{1}{\tan(\pi j/(2L+1))}\right)$$

For $L=2$: $2L+1 = 5$

$\phi_1 = 2\arctan\left(\frac{1}{\tan(\pi/5)}\right) = 2\arctan(1.376) = 1.88$ rad

$\phi_2 = 2\arctan\left(\frac{1}{\tan(2\pi/5)}\right) = 2\arctan(0.325) = 0.63$ rad

### Example 3: Convergence Comparison
Compare iterations needed for standard vs fixed-point to reach $P \geq 0.99$.

**Solution:**
Initial $a = 0.01$, $\theta = 0.1002$

**Standard:**
$k$ such that $\sin^2((2k+1)\theta) \geq 0.99$
$(2k+1)\theta \geq \arcsin(0.995) = 1.471$
$k \geq (1.471/0.1002 - 1)/2 = 6.8$
$k = 7$ iterations

**Fixed-point (YLC):**
Need $L$ such that $P_L \geq 0.99$
$L \approx \frac{\log(100)}{\sqrt{0.01}} = \frac{4.6}{0.1} = 46$ composite iterations

Fixed-point requires more iterations but never overshoots!

---

## Practice Problems

### Problem 1: Overshooting Analysis
For $a = 0.04$, find:
a) The iteration number where maximum overshoot occurs
b) The success probability at that iteration

### Problem 2: YLC Phase Sequence
Compute the complete phase sequence $\{\phi_1, ..., \phi_L\}$ for $L = 3$.

### Problem 3: Query Complexity
Derive the total number of oracle queries for fixed-point amplification achieving $1-\delta$ success probability.

---

## Computational Lab

```python
"""Day 626: Fixed-Point Amplitude Amplification"""
import numpy as np
import matplotlib.pyplot as plt

def standard_amplification_prob(k, theta):
    """Success probability after k standard Grover iterations."""
    return np.sin((2*k + 1) * theta)**2

def ylc_phases(L):
    """Calculate YLC phase sequence."""
    phases = []
    for j in range(1, L + 1):
        arg = np.pi * j / (2*L + 1)
        phi = 2 * np.arctan(1 / np.tan(arg))
        phases.append(phi)
    return phases

def modified_grover_operator(A, good_states, phi):
    """Grover operator with modified phase."""
    N = A.shape[0]
    A_inv = A.conj().T

    # Modified S_chi
    S_chi = np.eye(N, dtype=complex)
    for g in good_states:
        S_chi[g, g] = np.exp(1j * phi)

    # Modified S_0
    S_0 = np.eye(N, dtype=complex)
    S_0[0, 0] = np.exp(1j * phi)

    return -A @ S_0 @ A_inv @ S_chi

def fixed_point_amplification(A, good_states, L):
    """
    Apply L composite fixed-point iterations.

    Returns success probability at each step.
    """
    N = A.shape[0]
    phases = ylc_phases(L)

    # Initial state
    zero = np.zeros(N)
    zero[0] = 1
    psi = A @ zero

    probs = [sum(abs(psi[g])**2 for g in good_states)]

    for phi in phases:
        Q = modified_grover_operator(A, good_states, phi)
        psi = Q @ psi
        prob = sum(abs(psi[g])**2 for g in good_states)
        probs.append(prob)

    return probs

def compare_standard_vs_fixed_point(A, good_states, max_iter=20):
    """Compare standard and fixed-point amplification."""
    N = A.shape[0]

    # Initial state and parameters
    zero = np.zeros(N)
    zero[0] = 1
    psi = A @ zero
    a = sum(abs(psi[g])**2 for g in good_states)
    theta = np.arcsin(np.sqrt(a))

    # Standard Grover
    standard_probs = [standard_amplification_prob(k, theta)
                      for k in range(max_iter + 1)]

    # Fixed-point
    fixed_probs = fixed_point_amplification(A, good_states, max_iter)

    return standard_probs, fixed_probs, a, theta

def visualize_comparison(A, good_states, max_iter=30):
    """Visualize standard vs fixed-point amplification."""
    standard, fixed, a, theta = compare_standard_vs_fixed_point(
        A, good_states, max_iter
    )

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(len(standard)), standard, 'b-o', label='Standard Grover',
             markersize=4)
    plt.plot(range(len(fixed)), fixed, 'r-s', label='Fixed-Point',
             markersize=4)
    plt.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=a, color='green', linestyle=':', alpha=0.5,
                label=f'Initial a={a:.4f}')

    # Mark optimal for standard
    k_opt = int(np.round(np.pi / (4*theta) - 0.5))
    plt.axvline(x=k_opt, color='blue', linestyle=':', alpha=0.5)

    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Success Probability', fontsize=12)
    plt.title('Standard vs Fixed-Point Amplification', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    # Show the monotonicity of fixed-point
    differences = np.diff(fixed)
    plt.bar(range(len(differences)), differences, color='green', alpha=0.7)
    plt.axhline(y=0, color='red', linestyle='-', linewidth=1)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('ΔP (change in probability)', fontsize=12)
    plt.title('Fixed-Point: Probability Change per Iteration', fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fixed_point_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def analyze_ylc_phases():
    """Analyze the YLC phase sequence."""
    L_values = [2, 4, 6, 8, 10]

    plt.figure(figsize=(10, 6))

    for L in L_values:
        phases = ylc_phases(L)
        plt.plot(range(1, L+1), phases, 'o-', label=f'L={L}')

    plt.xlabel('Composite Iteration j', fontsize=12)
    plt.ylabel('Phase φⱼ (radians)', fontsize=12)
    plt.title('YLC Phase Sequences', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ylc_phases.png', dpi=150, bbox_inches='tight')
    plt.show()

def convergence_analysis(a_values, target_delta=0.01):
    """Analyze convergence for different initial amplitudes."""
    print(f"\nConvergence Analysis (target δ = {target_delta}):")
    print("-" * 60)
    print(f"{'Initial a':>12} | {'θ (rad)':>10} | {'Standard k':>12} | {'Fixed L':>10}")
    print("-" * 60)

    for a in a_values:
        theta = np.arcsin(np.sqrt(a))

        # Standard: need (2k+1)θ ≈ π/2
        k_standard = int(np.round(np.pi / (4*theta) - 0.5))

        # Fixed-point: L ≈ log(1/δ) / sqrt(a)
        L_fixed = int(np.ceil(np.log(1/target_delta) / np.sqrt(a)))

        print(f"{a:>12.4f} | {theta:>10.4f} | {k_standard:>12} | {L_fixed:>10}")

# Main execution
print("="*60)
print("Fixed-Point Amplitude Amplification")
print("="*60)

# Create test preparation
n = 4
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
A = H
for _ in range(n - 1):
    A = np.kron(A, H)

good_states = [3]  # One marked state

# Compare methods
print("\n1. COMPARISON: Standard vs Fixed-Point")
print("-"*50)
visualize_comparison(A, good_states, max_iter=25)

# Analyze YLC phases
print("\n2. YLC PHASE ANALYSIS")
print("-"*50)
analyze_ylc_phases()

# Print specific phases
for L in [2, 3, 4]:
    phases = ylc_phases(L)
    print(f"\nL={L}: φ = {[f'{p:.4f}' for p in phases]}")

# Convergence analysis
print("\n3. CONVERGENCE ANALYSIS")
a_values = [0.01, 0.04, 0.1, 0.25]
convergence_analysis(a_values)

# Detailed trace for small example
print("\n4. DETAILED TRACE")
print("-"*50)

n_small = 2
A_small = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
A_small = np.kron(A_small, A_small)

good_small = [0]  # Mark |00⟩

# Initial
zero = np.zeros(4)
zero[0] = 1
psi = A_small @ zero
a_init = abs(psi[0])**2
print(f"Initial amplitude: a = {a_init:.4f}")

# Standard Grover trace
theta = np.arcsin(np.sqrt(a_init))
print(f"\nStandard Grover (θ = {theta:.4f} rad):")
for k in range(8):
    p = standard_amplification_prob(k, theta)
    marker = " <- optimal" if k == int(np.round(np.pi/(4*theta) - 0.5)) else ""
    marker += " OVERSHOOT!" if k > 0 and p < standard_amplification_prob(k-1, theta) else ""
    print(f"  k={k}: P = {p:.4f}{marker}")

# Fixed-point trace
print(f"\nFixed-point (L=5):")
fixed_probs = fixed_point_amplification(A_small, good_small, 5)
for i, p in enumerate(fixed_probs):
    increase = "↑" if i > 0 and p > fixed_probs[i-1] else ""
    print(f"  iter {i}: P = {p:.4f} {increase}")
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Modified reflection | $S(\phi) = I - (1-e^{i\phi})\|target\rangle\langle target\|$ |
| YLC phases | $\phi_j = 2\arctan(1/\tan(\pi j/(2L+1)))$ |
| Fixed-point complexity | $O(\log(1/\delta)/\sqrt{a})$ |
| Standard complexity | $O(1/\sqrt{a})$ |

### Key Takeaways

1. **Standard Grover overshoots** if iteration count is wrong
2. **Fixed-point methods** guarantee monotonic convergence
3. **Phase modification** transforms oscillatory to convergent behavior
4. **YLC algorithm** provides explicit phase sequence
5. **Logarithmic overhead** but no knowledge of $\theta$ needed
6. **Useful when** success probability is unknown

---

## Daily Checklist

- [ ] I understand why standard Grover overshoots
- [ ] I can define fixed-point criteria
- [ ] I know how phase modification enables fixed-point
- [ ] I can compute YLC phase sequences
- [ ] I understand the complexity tradeoff
- [ ] I ran the computational lab and compared methods

---

*Next: Day 627 — Oblivious Amplitude Amplification*
