# Day 74: Phase Portraits and Stability

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Phase Portrait Theory |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Stability Analysis |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Practice |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Sketch phase portraits for 2×2 linear systems
2. Classify equilibrium points (nodes, saddles, spirals, centers)
3. Determine stability from eigenvalues
4. Understand the relationship between eigenvalues and phase portrait geometry
5. Apply stability analysis to physical systems

---

## 📚 Required Reading

### Primary Text: Boyce & DiPrima (11th Edition)
- **Section 7.5**: Phase Portraits (pp. 419-425)
- **Section 7.6**: Stability discussion (pp. 436-438)

---

## 📖 Phase Portraits

### 1. The Phase Plane

For a 2D system $\mathbf{x}' = A\mathbf{x}$, the **phase plane** plots $(x_1, x_2)$ trajectories.

Each point represents a state; arrows show direction of motion.

### 2. Equilibrium Points

An **equilibrium point** (critical point) satisfies $\mathbf{x}' = \mathbf{0}$.

For linear systems, the origin is always an equilibrium.

### 3. Classification by Eigenvalues

| Eigenvalues | Portrait Type | Stability |
|-------------|---------------|-----------|
| $\lambda_1 < \lambda_2 < 0$ | **Stable node** | Asymptotically stable |
| $\lambda_1 > \lambda_2 > 0$ | **Unstable node** | Unstable |
| $\lambda_1 < 0 < \lambda_2$ | **Saddle** | Unstable |
| $\alpha \pm i\beta$, $\alpha < 0$ | **Stable spiral** | Asymptotically stable |
| $\alpha \pm i\beta$, $\alpha > 0$ | **Unstable spiral** | Unstable |
| $\pm i\beta$ (pure imaginary) | **Center** | Stable (not asymptotic) |
| $\lambda < 0$ (repeated, complete) | **Stable star** | Asymptotically stable |
| $\lambda > 0$ (repeated, complete) | **Unstable star** | Unstable |
| $\lambda$ (repeated, deficient) | **Degenerate node** | Depends on sign |

---

## 📖 Stability Definitions

### 4. Types of Stability

> **Stable:** Solutions starting near equilibrium stay near forever.

> **Asymptotically stable:** Stable, and solutions approach equilibrium as $t \to \infty$.

> **Unstable:** Some solutions move away from equilibrium.

### 5. Stability Criterion

For $\mathbf{x}' = A\mathbf{x}$:

| All eigenvalues have Re(λ) < 0 | Asymptotically stable |
| All eigenvalues have Re(λ) ≤ 0, and pure imaginary ones are simple | Stable |
| Any eigenvalue has Re(λ) > 0 | Unstable |

---

## ✏️ Classification Examples

### Example 1: Stable Node
$A = \begin{pmatrix} -2 & 0 \\ 0 & -1 \end{pmatrix}$

Eigenvalues: $\lambda_1 = -2$, $\lambda_2 = -1$ (both negative)

All trajectories approach origin along eigenvector directions.
**Faster decay** along $\mathbf{v}_1$ (more negative eigenvalue).

---

### Example 2: Saddle Point
$A = \begin{pmatrix} 1 & 0 \\ 0 & -2 \end{pmatrix}$

Eigenvalues: $\lambda_1 = 1 > 0$, $\lambda_2 = -2 < 0$

Trajectories approach along $\mathbf{v}_2$, escape along $\mathbf{v}_1$.
**Unstable** (hyperbolic trajectories).

---

### Example 3: Center
$A = \begin{pmatrix} 0 & 1 \\ -4 & 0 \end{pmatrix}$

Eigenvalues: $\lambda = \pm 2i$ (pure imaginary)

Closed elliptical orbits around origin.
**Stable but not asymptotically stable.**

---

### Example 4: Spiral
$A = \begin{pmatrix} -1 & 2 \\ -2 & -1 \end{pmatrix}$

Eigenvalues: $\lambda = -1 \pm 2i$

Since Re(λ) = -1 < 0: **stable spiral**.
Trajectories spiral into origin.

---

## 📊 Phase Portrait Gallery

### Nodes (Real Eigenvalues, Same Sign)

```
Stable Node (λ₁ < λ₂ < 0)    Unstable Node (0 < λ₁ < λ₂)
        ↘ ↓ ↙                      ↗ ↑ ↖
         \|/                        \|/
      ←───●───→                  ←───●───→
         /|\                        /|\
        ↗ ↑ ↖                      ↘ ↓ ↙
   All trajectories           All trajectories
   approach origin            leave origin
```

### Saddle (Real Eigenvalues, Opposite Signs)

```
Saddle Point (λ₁ < 0 < λ₂)
         ↑
    ↖    |    ↗
      \  |  /
   ←───────────→ (unstable)
      /  |  \
    ↙    |    ↘
         ↓
    (stable direction)
```

### Spirals (Complex Eigenvalues with α ≠ 0)

```
Stable Spiral (α < 0)        Unstable Spiral (α > 0)
      ╭──→──╮                     ╭──←──╮
     ↗      ↘                    ↗      ↘
    ↑   ●    ↓                  ↑   ●    ↓
     ↖      ↙                    ↖      ↙
      ╰──←──╯                     ╰──→──╯
   Spirals inward              Spirals outward
```

### Center (Pure Imaginary Eigenvalues)

```
Center (λ = ±iβ)
      ╭──→──╮
     ↗      ↘
    │   ●    │
     ↖      ↙
      ╰──←──╯
   Closed orbits
```

---

## 📖 Quick Classification Algorithm

Given $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$:

1. **Trace:** $\tau = a + d = \lambda_1 + \lambda_2$
2. **Determinant:** $\Delta = ad - bc = \lambda_1 \lambda_2$
3. **Discriminant:** $\tau^2 - 4\Delta$

| Condition | Classification |
|-----------|----------------|
| $\Delta < 0$ | **Saddle** (unstable) |
| $\Delta > 0$, $\tau^2 > 4\Delta$, $\tau < 0$ | **Stable node** |
| $\Delta > 0$, $\tau^2 > 4\Delta$, $\tau > 0$ | **Unstable node** |
| $\Delta > 0$, $\tau^2 < 4\Delta$, $\tau < 0$ | **Stable spiral** |
| $\Delta > 0$, $\tau^2 < 4\Delta$, $\tau > 0$ | **Unstable spiral** |
| $\Delta > 0$, $\tau = 0$ | **Center** |
| $\tau^2 = 4\Delta$ | **Degenerate** (repeated eigenvalue) |

---

## 📝 Practice Problems

### Level 1-3: Classification
1. Classify: $A = \begin{pmatrix} -3 & 0 \\ 0 & -2 \end{pmatrix}$
2. Classify: $A = \begin{pmatrix} 2 & -5 \\ 1 & -2 \end{pmatrix}$
3. Classify: $A = \begin{pmatrix} 1 & -2 \\ 3 & -4 \end{pmatrix}$

### Level 4-6: From Trace and Determinant
4. If $\tau = -4$ and $\Delta = 5$, classify the equilibrium.
5. If $\tau = 0$ and $\Delta = 9$, classify the equilibrium.
6. If $\tau = 2$ and $\Delta = -8$, classify the equilibrium.

### Level 7-9: Sketch Phase Portraits
7. Sketch the phase portrait for $\mathbf{x}' = \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}\mathbf{x}$
8. Sketch for $\mathbf{x}' = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}\mathbf{x}$
9. Sketch for $\mathbf{x}' = \begin{pmatrix} -1 & 1 \\ 0 & -1 \end{pmatrix}\mathbf{x}$

### Level 10: Applications
10. A mass-spring system has $m = 1$, $c = 2$, $k = 5$. Classify the motion and sketch the phase portrait.

---

## 📊 Answers

1. Stable node ($\lambda = -3, -2$)
2. Center ($\lambda = \pm i$)
3. Saddle ($\lambda = -1, -2$... wait, check: $\tau = -3$, $\Delta = -4+6 = 2$, so stable node)
4. Stable spiral ($\tau^2 = 16 < 20 = 4\Delta$, $\tau < 0$)
5. Center (pure imaginary, $\lambda = \pm 3i$)
6. Saddle ($\Delta < 0$)
7. Center (closed ellipses)
8. Saddle (hyperbolas)
9. Stable degenerate node
10. Underdamped (stable spiral)

---

## 🔬 Quantum Mechanics Connection

### Bloch Sphere Dynamics

A spin-1/2 particle's dynamics can be visualized on the Bloch sphere.
- Stable spirals → relaxation to ground state
- Centers → persistent Rabi oscillations
- Saddles → unstable dynamics in open systems

---

## ✅ Daily Checklist

- [ ] Understand all phase portrait types
- [ ] Classify using eigenvalues or trace-determinant
- [ ] Sketch phase portraits
- [ ] Apply to physical systems

---

## 🔜 Preview: Tomorrow

**Day 75: Applications of Systems**
- Coupled oscillators
- Predator-prey models
- Compartmental models

---

*"Phase portraits reveal the geometry of dynamics—every trajectory tells a story of how states evolve."*
