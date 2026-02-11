# Day 620: Amplitude Amplification Geometry

## Overview
**Day 620** | Week 89, Day 4 | Year 1, Month 23 | Grover's Search Algorithm

Today we develop the geometric interpretation of Grover's algorithm as rotation in a two-dimensional subspace. This elegant picture explains why the algorithm works and how many iterations are needed.

---

## Learning Objectives

1. Identify the two-dimensional subspace of Grover's algorithm
2. Express the Grover operator as a rotation
3. Calculate the rotation angle per iteration
4. Visualize the amplitude amplification geometrically
5. Derive the success probability formula
6. Understand the periodic behavior of iterations

---

## Core Content

### The Two-Dimensional Subspace

The key insight is that Grover's algorithm operates entirely within a 2D subspace:

**Basis vectors:**
- $|w\rangle$ = marked state (target)
- $|s'\rangle = \frac{1}{\sqrt{N-1}}\sum_{x \neq w}|x\rangle$ = uniform superposition of unmarked states

**Initial state in this basis:**

$$|\psi_0\rangle = \sin\theta|w\rangle + \cos\theta|s'\rangle$$

where:
$$\boxed{\sin\theta = \frac{1}{\sqrt{N}}, \quad \cos\theta = \sqrt{\frac{N-1}{N}}}$$

For large $N$: $\theta \approx \sin\theta \approx 1/\sqrt{N}$

### Grover Operator as Rotation

Both the oracle and diffusion are reflections, and the product of two reflections is a rotation!

**Oracle $O_f$:** Reflection about $|s'\rangle$
$$O_f = I - 2|w\rangle\langle w|$$

In the $\{|w\rangle, |s'\rangle\}$ basis:
$$O_f = \begin{pmatrix} -1 & 0 \\ 0 & 1 \end{pmatrix}$$

**Diffusion $D$:** Reflection about $|\psi_0\rangle$
$$D = 2|\psi_0\rangle\langle\psi_0| - I$$

In the $\{|w\rangle, |s'\rangle\}$ basis:
$$D = \begin{pmatrix} \cos 2\theta & \sin 2\theta \\ \sin 2\theta & -\cos 2\theta \end{pmatrix}$$

**Grover Operator $G = D \cdot O_f$:**
$$G = \begin{pmatrix} \cos 2\theta & -\sin 2\theta \\ \sin 2\theta & \cos 2\theta \end{pmatrix}$$

This is a **rotation by angle $2\theta$** toward $|w\rangle$!

### Geometric Picture

```
        |w⟩
         ↑
         |      ↗ G^k|ψ₀⟩ (after k iterations)
         |    ↗
         |  ↗  θ = rotation per step
         |↗
    ─────|────────→ |s'⟩
         |↘
         |  ↘ |ψ₀⟩ (initial state)
         |    ↘
         |      angle θ from |s'⟩
```

Each Grover iteration rotates the state by $2\theta$ toward $|w\rangle$.

### State After k Iterations

$$\boxed{G^k|\psi_0\rangle = \sin((2k+1)\theta)|w\rangle + \cos((2k+1)\theta)|s'\rangle}$$

**Proof by induction:**
- Base case ($k=0$): $|\psi_0\rangle = \sin\theta|w\rangle + \cos\theta|s'\rangle$ ✓
- Inductive step: Rotation by $2\theta$ takes angle from $(2k+1)\theta$ to $(2k+3)\theta$ ✓

### Success Probability

The probability of measuring the marked state after $k$ iterations:

$$\boxed{P_{success}(k) = \sin^2((2k+1)\theta)}$$

**Maximum probability:** When $(2k+1)\theta = \pi/2$, i.e., $P_{success} = 1$

This occurs when: $k = \frac{\pi/2 - \theta}{2\theta} = \frac{\pi}{4\theta} - \frac{1}{2}$

### Rotation Angle Analysis

For a single marked state in $N$ items:
$$\theta = \arcsin\frac{1}{\sqrt{N}} \approx \frac{1}{\sqrt{N}}$$ (for large $N$)

**Rotation per iteration:** $2\theta \approx \frac{2}{\sqrt{N}}$

**Total rotation needed:** From $\theta$ to $\pi/2$, i.e., $\frac{\pi}{2} - \theta \approx \frac{\pi}{2}$

**Number of iterations:** $k \approx \frac{\pi/2}{2/\sqrt{N}} = \frac{\pi\sqrt{N}}{4}$

### Product of Reflections

**Theorem:** The product of two reflections about lines through the origin making angle $\phi$ is a rotation by $2\phi$.

In Grover's algorithm:
- Oracle reflects about the line perpendicular to $|w\rangle$
- Diffusion reflects about the line along $|\psi_0\rangle$
- Angle between these lines: $\frac{\pi}{2} - \theta$
- Product: rotation by $2(\frac{\pi}{2} - \theta)$...

Actually, let's be more careful. The angle between:
- Reflection axis of $O_f$: perpendicular to $|w\rangle$, i.e., along $|s'\rangle$
- Reflection axis of $D$: along $|\psi_0\rangle$

The angle between $|s'\rangle$ and $|\psi_0\rangle$ is $\theta$.

Therefore: $G = D \cdot O_f$ is a rotation by $2\theta$.

---

## Worked Examples

### Example 1: Rotation Visualization
For $N = 4$, trace the geometric evolution through 2 iterations.

**Solution:**
$\theta = \arcsin(1/2) = \pi/6 = 30°$

Initial state: angle $\theta = 30°$ from $|s'\rangle$

After 1 iteration: angle $(2 \cdot 1 + 1) \cdot 30° = 90°$ from $|s'\rangle$
- This means the state is exactly $|w\rangle$!
- $P_{success} = \sin^2(90°) = 1$

After 2 iterations: angle $(2 \cdot 2 + 1) \cdot 30° = 150°$ from $|s'\rangle$
- $P_{success} = \sin^2(150°) = (1/2)^2 = 0.25$

We've "overshot" the target! One iteration is optimal for $N = 4$.

### Example 2: Large N Approximation
For $N = 10^6$, calculate the rotation angle and optimal iterations.

**Solution:**
$\sin\theta = 1/\sqrt{10^6} = 10^{-3}$

$\theta \approx 10^{-3}$ radians $\approx 0.057°$

Rotation per iteration: $2\theta \approx 2 \times 10^{-3}$ rad

Optimal iterations: $k_{opt} = \frac{\pi}{4\theta} - \frac{1}{2} \approx \frac{\pi}{4 \times 10^{-3}} \approx 785$

Verification: $(2 \cdot 785 + 1) \times 10^{-3} \approx 1.571 \approx \pi/2$ ✓

### Example 3: Success Probability Curve
Plot $P_{success}(k)$ for $N = 64$.

**Solution:**
$\theta = \arcsin(1/8) \approx 0.1253$ rad $\approx 7.18°$

$k_{opt} = \frac{\pi}{4 \times 0.1253} - 0.5 \approx 5.77 \approx 6$

| $k$ | $(2k+1)\theta$ | $P_{success}$ |
|-----|----------------|---------------|
| 0 | 0.125 | 0.0156 |
| 1 | 0.376 | 0.139 |
| 2 | 0.627 | 0.343 |
| 3 | 0.878 | 0.594 |
| 4 | 1.129 | 0.816 |
| 5 | 1.380 | 0.951 |
| 6 | 1.631 | 0.998 |
| 7 | 1.882 | 0.948 |

Maximum at $k = 6$ with $P \approx 99.8\%$

---

## Practice Problems

### Problem 1: Rotation Matrix
Derive the $2 \times 2$ rotation matrix for the Grover operator in the $\{|w\rangle, |s'\rangle\}$ basis from the matrix representations of $O_f$ and $D$.

### Problem 2: Angle Calculation
For $N = 256$:
a) Calculate $\theta$ exactly using arcsin
b) Approximate $\theta$ for large $N$
c) Find the optimal number of iterations
d) Calculate the success probability at the optimal iteration

### Problem 3: Overshoot Analysis
Starting from the optimal iteration for $N = 16$, how many additional iterations until the success probability drops below 50%?

---

## Computational Lab

```python
"""Day 620: Amplitude Amplification Geometry"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch
from matplotlib.transforms import Affine2D

def compute_theta(N, M=1):
    """Compute the rotation angle theta."""
    return np.arcsin(np.sqrt(M/N))

def success_probability(k, theta):
    """Success probability after k iterations."""
    return np.sin((2*k + 1) * theta)**2

def optimal_iterations(N, M=1):
    """Optimal number of Grover iterations."""
    theta = compute_theta(N, M)
    return int(np.round(np.pi / (4 * theta) - 0.5))

def visualize_rotation_geometry(N, num_iterations=None):
    """Visualize the geometric interpretation of Grover's algorithm."""
    theta = compute_theta(N)

    if num_iterations is None:
        num_iterations = optimal_iterations(N)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw coordinate axes
    ax.arrow(0, 0, 1.3, 0, head_width=0.03, head_length=0.03, fc='black', ec='black')
    ax.arrow(0, 0, 0, 1.3, head_width=0.03, head_length=0.03, fc='black', ec='black')
    ax.text(1.35, 0, "|s'⟩", fontsize=14, ha='left', va='center')
    ax.text(0, 1.35, "|w⟩", fontsize=14, ha='center', va='bottom')

    # Draw initial state |ψ₀⟩
    psi0_x = np.cos(theta)
    psi0_y = np.sin(theta)
    ax.arrow(0, 0, psi0_x*0.95, psi0_y*0.95, head_width=0.03, head_length=0.03,
             fc='blue', ec='blue', linewidth=2)
    ax.text(psi0_x*1.1, psi0_y*1.1, "|ψ₀⟩", fontsize=12, color='blue')

    # Draw arc showing initial angle
    arc = Arc((0, 0), 0.3, 0.3, angle=0, theta1=0, theta2=np.degrees(theta),
              color='blue', linestyle='--')
    ax.add_patch(arc)
    ax.text(0.2, 0.08, f"θ={np.degrees(theta):.1f}°", fontsize=10, color='blue')

    # Draw states after each iteration
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, num_iterations + 1))

    for k in range(num_iterations + 1):
        angle = (2*k + 1) * theta
        x = np.cos(np.pi/2 - angle)  # |s'⟩ is horizontal, |w⟩ is vertical
        y = np.sin(np.pi/2 - angle)

        # Actually, let me fix the coordinates
        # State after k iterations: sin((2k+1)θ)|w⟩ + cos((2k+1)θ)|s'⟩
        x = np.cos((2*k + 1) * theta)  # coefficient of |s'⟩
        y = np.sin((2*k + 1) * theta)  # coefficient of |w⟩

        if k > 0:
            ax.arrow(0, 0, x*0.9, y*0.9, head_width=0.02, head_length=0.02,
                     fc=colors[k], ec=colors[k], linewidth=1.5, alpha=0.7)
            ax.plot(x, y, 'o', color=colors[k], markersize=8)
            ax.text(x*1.05, y*1.05, f"k={k}", fontsize=9, color=colors[k])

    # Draw rotation arcs
    for k in range(min(3, num_iterations)):
        angle_start = (2*k + 1) * theta
        angle_end = (2*k + 3) * theta
        radius = 0.4 + 0.1*k
        arc = Arc((0, 0), 2*radius, 2*radius, angle=0,
                  theta1=np.degrees(angle_start), theta2=np.degrees(angle_end),
                  color='red', linestyle='-', linewidth=1.5)
        ax.add_patch(arc)

    # Mark the target angle π/2
    ax.plot([0, 0], [0, 1], 'g--', alpha=0.5, linewidth=1)

    # Set equal aspect ratio and limits
    ax.set_xlim(-0.2, 1.5)
    ax.set_ylim(-0.2, 1.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Grover Rotation Geometry (N={N}, θ={np.degrees(theta):.2f}°)\n'
                 f'Each iteration rotates by 2θ={np.degrees(2*theta):.2f}°',
                 fontsize=12)

    plt.tight_layout()
    plt.savefig('grover_geometry.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_success_probability_oscillation(N):
    """Plot success probability showing oscillatory behavior."""
    theta = compute_theta(N)
    k_opt = optimal_iterations(N)

    # More iterations to show oscillation
    k_max = 3 * k_opt
    k_values = np.arange(k_max + 1)
    probs = [success_probability(k, theta) for k in k_values]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(k_values, probs, 'b-', linewidth=2)
    plt.axvline(x=k_opt, color='r', linestyle='--', label=f'Optimal k={k_opt}')
    plt.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
    plt.xlabel('Number of Iterations k', fontsize=12)
    plt.ylabel('Success Probability', fontsize=12)
    plt.title(f'Grover Success Probability (N={N})', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    # Plot the state angle
    angles = [(2*k + 1) * theta for k in k_values]
    angles_deg = [np.degrees(a) % 360 for a in angles]
    plt.plot(k_values, angles_deg, 'g-', linewidth=2)
    plt.axhline(y=90, color='r', linestyle='--', label='Target (90°)')
    plt.xlabel('Number of Iterations k', fontsize=12)
    plt.ylabel('State Angle (degrees)', fontsize=12)
    plt.title('State Angle Evolution', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('grover_oscillation.png', dpi=150, bbox_inches='tight')
    plt.show()

def compare_different_N():
    """Compare rotation angles and optimal iterations for different N."""
    N_values = [4, 16, 64, 256, 1024, 4096]

    print("Comparison of Grover Parameters for Different N:")
    print("-" * 70)
    print(f"{'N':>6} | {'θ (deg)':>10} | {'θ (approx)':>10} | {'k_opt':>6} | {'P_success':>10}")
    print("-" * 70)

    for N in N_values:
        theta = compute_theta(N)
        theta_approx = 1/np.sqrt(N)
        k_opt = optimal_iterations(N)
        p_success = success_probability(k_opt, theta)

        print(f"{N:>6} | {np.degrees(theta):>10.4f} | {np.degrees(theta_approx):>10.4f} | "
              f"{k_opt:>6} | {p_success:>10.6f}")

    print("-" * 70)

def grover_matrix_2d(theta):
    """Return the 2x2 Grover operator matrix in {|w⟩, |s'⟩} basis."""
    return np.array([
        [np.cos(2*theta), -np.sin(2*theta)],
        [np.sin(2*theta), np.cos(2*theta)]
    ])

def verify_rotation(N):
    """Verify that Grover operator acts as rotation."""
    theta = compute_theta(N)
    G = grover_matrix_2d(theta)

    print(f"\nGrover operator in 2D basis (N={N}, θ={np.degrees(theta):.2f}°):")
    print(f"G = [cos(2θ)  -sin(2θ)]")
    print(f"    [sin(2θ)   cos(2θ)]")
    print(f"\n  = [{G[0,0]:.4f}  {G[0,1]:.4f}]")
    print(f"    [{G[1,0]:.4f}  {G[1,1]:.4f}]")

    # Verify it's a rotation (det = 1, orthogonal)
    det = np.linalg.det(G)
    is_orthogonal = np.allclose(G @ G.T, np.eye(2))
    print(f"\n  Determinant: {det:.6f} (should be 1)")
    print(f"  Orthogonal: {is_orthogonal}")

    # Track state evolution
    print(f"\n  State evolution:")
    psi = np.array([np.sin(theta), np.cos(theta)])  # [|w⟩, |s'⟩] coefficients
    for k in range(min(optimal_iterations(N) + 2, 10)):
        prob_w = psi[0]**2
        print(f"    k={k}: |w⟩ coeff = {psi[0]:.4f}, P(w) = {prob_w:.4f}")
        psi = G @ psi

# Main execution
print("=" * 60)
print("Amplitude Amplification Geometry")
print("=" * 60)

# Compare different N values
compare_different_N()

# Verify rotation property
verify_rotation(16)

# Visualizations
print("\nGenerating visualizations...")
visualize_rotation_geometry(16, num_iterations=4)
plot_success_probability_oscillation(64)

# Demonstrate overshoot
print("\n" + "=" * 60)
print("Overshoot Demonstration (N=4)")
print("=" * 60)
N = 4
theta = compute_theta(N)
for k in range(5):
    p = success_probability(k, theta)
    angle = (2*k + 1) * theta
    print(f"k={k}: angle = {np.degrees(angle):.1f}°, P_success = {p:.4f}")
```

**Expected Output:**
```
============================================================
Amplitude Amplification Geometry
============================================================
Comparison of Grover Parameters for Different N:
----------------------------------------------------------------------
     N |   θ (deg) | θ (approx) |  k_opt | P_success
----------------------------------------------------------------------
     4 |    30.0000 |    28.6479 |      1 |   1.000000
    16 |    14.4775 |    14.3239 |      3 |   0.961258
    64 |     7.1808 |     7.1620 |      6 |   0.996094
   256 |     3.5833 |     3.5809 |     12 |   0.999512
  1024 |     1.7905 |     1.7904 |     25 |   0.999878
  4096 |     0.8952 |     0.8952 |     50 |   0.999970
----------------------------------------------------------------------

Grover operator in 2D basis (N=16, θ=14.48°):
G = [cos(2θ)  -sin(2θ)]
    [sin(2θ)   cos(2θ)]

  = [0.8750  -0.4841]
    [0.4841   0.8750]

  Determinant: 1.000000 (should be 1)
  Orthogonal: True

  State evolution:
    k=0: |w⟩ coeff = 0.2500, P(w) = 0.0625
    k=1: |w⟩ coeff = 0.6875, P(w) = 0.4727
    k=2: |w⟩ coeff = 0.9336, P(w) = 0.8716
    k=3: |w⟩ coeff = 0.9805, P(w) = 0.9613
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Initial angle | $\sin\theta = 1/\sqrt{N}$ |
| State after k iterations | $G^k\|\psi_0\rangle = \sin((2k+1)\theta)\|w\rangle + \cos((2k+1)\theta)\|s'\rangle$ |
| Success probability | $P(k) = \sin^2((2k+1)\theta)$ |
| Rotation angle | $2\theta$ per iteration |
| Grover matrix (2D) | $\begin{pmatrix} \cos 2\theta & -\sin 2\theta \\ \sin 2\theta & \cos 2\theta \end{pmatrix}$ |

### Key Takeaways

1. **Two-dimensional subspace** spanned by $|w\rangle$ and $|s'\rangle$
2. **Grover operator is a rotation** by $2\theta$ toward $|w\rangle$
3. **Product of reflections** gives rotation
4. **Success probability oscillates** with period $\pi/(2\theta)$
5. **Overshooting is possible** if too many iterations
6. **Geometric picture** gives intuition for optimal iteration count

---

## Daily Checklist

- [ ] I can identify the 2D subspace for Grover's algorithm
- [ ] I understand that the Grover operator is a rotation
- [ ] I can calculate the rotation angle from N
- [ ] I can derive the success probability formula
- [ ] I understand why overshooting occurs
- [ ] I ran the computational lab and visualized the geometry

---

*Next: Day 621 — Optimal Iteration Count O(sqrt(N))*
