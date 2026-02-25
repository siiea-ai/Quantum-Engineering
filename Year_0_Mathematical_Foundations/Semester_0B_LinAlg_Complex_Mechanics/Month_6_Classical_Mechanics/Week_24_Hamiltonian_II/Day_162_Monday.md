# Day 162: Canonical Transformations — The Symmetry Group of Hamiltonian Mechanics

## Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Canonical Transformations |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## Learning Objectives

By the end of today, you should be able to:

1. Define canonical transformations and explain why they preserve Hamilton's equations
2. Derive and apply all four types of generating functions (F₁, F₂, F₃, F₄)
3. Prove that the Jacobian matrix of a canonical transformation is symplectic
4. Verify that transformations preserve Poisson brackets
5. Construct canonical transformations for specific physical problems
6. Connect canonical transformations to unitary transformations in quantum mechanics

---

## Core Content

### 1. Motivation: Why Transform Coordinates?

In Lagrangian mechanics, we learned that generalized coordinates can be chosen freely—any set of coordinates that uniquely specify the configuration is valid. Hamiltonian mechanics adds a powerful new dimension: we can transform **both** coordinates **and** momenta together.

**The Central Question:** What transformations $(q, p) \to (Q, P)$ preserve the structure of Hamilton's equations?

$$\dot{q}_i = \frac{\partial H}{\partial p_i}, \quad \dot{p}_i = -\frac{\partial H}{\partial q_i} \quad \longrightarrow \quad \dot{Q}_i = \frac{\partial K}{\partial P_i}, \quad \dot{P}_i = -\frac{\partial K}{\partial Q_i}$$

where K(Q, P, t) is the Hamiltonian in the new coordinates.

**Why This Matters:**
1. **Simplification:** A clever choice of (Q, P) can make the Hamiltonian simpler or even constant
2. **Integrability:** Canonical transformations reveal hidden symmetries and conserved quantities
3. **Hamilton-Jacobi Theory:** The ultimate goal is to find coordinates where K = 0!
4. **Quantum Foundation:** These become unitary transformations in quantum mechanics

---

### 2. Definition of Canonical Transformations

**Definition (via Hamilton's Equations):** A transformation $(q, p) \to (Q, P)$ is **canonical** if there exists a function K(Q, P, t) such that Hamilton's equations hold in both coordinate systems.

**Definition (via Poisson Brackets):** Equivalently, a transformation is canonical if and only if the **fundamental Poisson brackets** are preserved:

$$\boxed{\{Q_i, Q_j\} = 0, \quad \{P_i, P_j\} = 0, \quad \{Q_i, P_j\} = \delta_{ij}}$$

**Definition (via the Symplectic Form):** A transformation is canonical if it preserves the symplectic 2-form:

$$\boxed{\omega = \sum_i dp_i \wedge dq_i = \sum_i dP_i \wedge dQ_i}$$

**Theorem (Equivalence):** All three definitions are equivalent. A transformation preserves Poisson brackets if and only if its Jacobian matrix is symplectic.

---

### 3. The Symplectic Condition

Let's write the transformation in compact notation:

$$\mathbf{Z} = \begin{pmatrix} Q_1 \\ \vdots \\ Q_n \\ P_1 \\ \vdots \\ P_n \end{pmatrix} = \mathbf{\Phi}(\mathbf{z}, t), \quad \mathbf{z} = \begin{pmatrix} q_1 \\ \vdots \\ q_n \\ p_1 \\ \vdots \\ p_n \end{pmatrix}$$

**The Jacobian Matrix:**

$$\mathbf{M} = \frac{\partial \mathbf{Z}}{\partial \mathbf{z}} = \begin{pmatrix} \frac{\partial Q}{\partial q} & \frac{\partial Q}{\partial p} \\ \frac{\partial P}{\partial q} & \frac{\partial P}{\partial p} \end{pmatrix}$$

**Symplectic Condition:** A transformation is canonical if and only if:

$$\boxed{\mathbf{M}^T \mathbf{J} \mathbf{M} = \mathbf{J}}$$

where **J** is the standard symplectic matrix:

$$\mathbf{J} = \begin{pmatrix} \mathbf{0} & \mathbf{I} \\ -\mathbf{I} & \mathbf{0} \end{pmatrix}$$

**Proof:** The Poisson bracket can be written as:

$$\{F, G\}_z = \left(\nabla_z F\right)^T \mathbf{J} \left(\nabla_z G\right)$$

Under a transformation with Jacobian M:

$$\nabla_z F = \mathbf{M}^T \nabla_Z F$$

So:

$$\{F, G\}_z = \left(\nabla_Z F\right)^T \mathbf{M} \mathbf{J} \mathbf{M}^T \left(\nabla_Z G\right)$$

For this to equal {F, G}_Z, we need M J Mᵀ = J, which is equivalent to Mᵀ J M = J.

---

### 4. The Symplectic Group Sp(2n, ℝ)

**Definition:** The **symplectic group** Sp(2n, ℝ) consists of all 2n × 2n real matrices satisfying:

$$\text{Sp}(2n, \mathbb{R}) = \{\mathbf{M} \in \text{GL}(2n, \mathbb{R}) : \mathbf{M}^T \mathbf{J} \mathbf{M} = \mathbf{J}\}$$

**Properties of Symplectic Matrices:**

| Property | Statement | Proof Sketch |
|----------|-----------|--------------|
| Determinant | det(M) = ±1 | Take determinant of Mᵀ J M = J |
| Actually +1 | det(M) = +1 always | Symplectic matrices are continuously connected to identity |
| Inverse | M⁻¹ = -J Mᵀ J | Multiply Mᵀ J M = J by M⁻¹ on right |
| Closure | M₁ M₂ ∈ Sp(2n) | Direct verification |
| Transpose | Mᵀ ∈ Sp(2n) | J⁻¹ = -J implies Mᵀ J M = J ⟺ M J Mᵀ = J |

**Special Case n = 1:** For one degree of freedom:

$$\text{Sp}(2, \mathbb{R}) = \text{SL}(2, \mathbb{R}) = \left\{ \begin{pmatrix} a & b \\ c & d \end{pmatrix} : ad - bc = 1 \right\}$$

The symplectic condition reduces to unit determinant!

**Dimension of Sp(2n):**

$$\dim\left(\text{Sp}(2n, \mathbb{R})\right) = n(2n+1)$$

| n | dim(Sp(2n)) |
|---|-------------|
| 1 | 3 |
| 2 | 10 |
| 3 | 21 |

**Examples of Symplectic Matrices:**

1. **Identity:** I₂ₙ
2. **Rotation in q-p plane:** $\begin{pmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{pmatrix}$
3. **Scaling (squeeze):** $\begin{pmatrix} \lambda & 0 \\ 0 & 1/\lambda \end{pmatrix}$ for λ ≠ 0
4. **Shear:** $\begin{pmatrix} 1 & a \\ 0 & 1 \end{pmatrix}$ or $\begin{pmatrix} 1 & 0 \\ b & 1 \end{pmatrix}$

---

### 5. Generating Functions

The most powerful technique for constructing canonical transformations uses **generating functions**. The key insight comes from the variational principle.

**The Extended Variational Principle:**

For Hamilton's equations to hold, we need:

$$\delta \int \left( \sum_i p_i \dot{q}_i - H \right) dt = 0$$

After transformation, this becomes:

$$\delta \int \left( \sum_i P_i \dot{Q}_i - K \right) dt = 0$$

These produce the same equations if the integrands differ by a total time derivative:

$$\sum_i p_i \dot{q}_i - H = \sum_i P_i \dot{Q}_i - K + \frac{dF}{dt}$$

The function F is called the **generating function**.

**Rearranging:**

$$\sum_i (p_i dq_i - P_i dQ_i) = dF + (K - H)dt$$

This suggests F = F(q, Q, t), giving:

$$p_i = \frac{\partial F}{\partial q_i}, \quad P_i = -\frac{\partial F}{\partial Q_i}, \quad K = H + \frac{\partial F}{\partial t}$$

---

### 6. The Four Types of Generating Functions

Different choices of independent variables yield four types of generating functions, related by Legendre transformations.

#### Type 1: F₁(q, Q, t)

$$\boxed{p_i = \frac{\partial F_1}{\partial q_i}, \quad P_i = -\frac{\partial F_1}{\partial Q_i}, \quad K = H + \frac{\partial F_1}{\partial t}}$$

**Example:** The identity transformation is generated by:

$$F_1(q, Q) = \sum_i q_i Q_i \implies p_i = Q_i, \quad P_i = -q_i$$

Wait—this swaps q and p! The true identity needs a different type.

#### Type 2: F₂(q, P, t)

Define F₂ via Legendre transformation: $F_2 = F_1 + \sum_i P_i Q_i$

$$\boxed{p_i = \frac{\partial F_2}{\partial q_i}, \quad Q_i = \frac{\partial F_2}{\partial P_i}, \quad K = H + \frac{\partial F_2}{\partial t}}$$

**Example (Identity):**

$$F_2(q, P) = \sum_i q_i P_i \implies p_i = P_i, \quad Q_i = q_i$$

This is the identity transformation!

**Example (Point Transformation):**

For Q_i = Q_i(q), use:

$$F_2(q, P) = \sum_i Q_i(q) P_i$$

Then Q_i = ∂F₂/∂P_i = Q_i(q) ✓, and:

$$p_i = \frac{\partial F_2}{\partial q_i} = \sum_j \frac{\partial Q_j}{\partial q_i} P_j$$

This is exactly the transformation law for momenta under coordinate change!

#### Type 3: F₃(p, Q, t)

Define: $F_3 = F_1 - \sum_i p_i q_i$

$$\boxed{q_i = -\frac{\partial F_3}{\partial p_i}, \quad P_i = -\frac{\partial F_3}{\partial Q_i}, \quad K = H + \frac{\partial F_3}{\partial t}}$$

#### Type 4: F₄(p, P, t)

Define: $F_4 = F_1 - \sum_i p_i q_i + \sum_i P_i Q_i$

$$\boxed{q_i = -\frac{\partial F_4}{\partial p_i}, \quad Q_i = \frac{\partial F_4}{\partial P_i}, \quad K = H + \frac{\partial F_4}{\partial t}}$$

**Summary Table:**

| Type | F(⋅,⋅,t) | Old → New Relations |
|------|----------|---------------------|
| F₁ | q, Q | p = ∂F₁/∂q, P = -∂F₁/∂Q |
| F₂ | q, P | p = ∂F₂/∂q, Q = ∂F₂/∂P |
| F₃ | p, Q | q = -∂F₃/∂p, P = -∂F₃/∂Q |
| F₄ | p, P | q = -∂F₄/∂p, Q = ∂F₄/∂P |

**When to Use Each Type:**

- **F₁:** When you want to specify Q as a function of q directly
- **F₂:** Most common—good for point transformations and simplifying the Hamiltonian
- **F₃:** When the old momentum p plays a central role
- **F₄:** Useful for exchange transformations and momentum-based problems

---

### 7. Important Examples

#### Example 1: The Exchange Transformation

$$Q = p, \quad P = -q$$

**Verification (Poisson brackets):**

$$\{Q, P\} = \{p, -q\} = -\{p, q\} = -(-1) = 1 \quad \checkmark$$

**Generating function (Type 1):**

$$F_1(q, Q) = qQ \implies p = \frac{\partial F_1}{\partial q} = Q, \quad P = -\frac{\partial F_1}{\partial Q} = -q$$

**Physical interpretation:** Position and momentum are on equal footing in Hamiltonian mechanics! This symmetry is broken in quantum mechanics by the commutator [q̂, p̂] = iℏ.

#### Example 2: Scaling Transformation

$$Q = \lambda q, \quad P = \frac{p}{\lambda} \quad (\lambda \neq 0)$$

**Verification:**

$$\{Q, P\} = \left\{\lambda q, \frac{p}{\lambda}\right\} = \lambda \cdot \frac{1}{\lambda} \{q, p\} = 1 \quad \checkmark$$

**Generating function (Type 2):**

$$F_2(q, P) = \lambda q P \implies p = \frac{\partial F_2}{\partial q} = \lambda P, \quad Q = \frac{\partial F_2}{\partial P} = \lambda q$$

So P = p/λ and Q = λq ✓

**Application:** Useful for dimensional analysis and scaling problems.

#### Example 3: Rotation to Action-Angle Variables

For the harmonic oscillator H = (p² + ω²q²)/2, we seek variables where the Hamiltonian depends only on one variable (the "action").

**Transformation:**

$$q = \sqrt{\frac{2P}{\omega}} \sin Q, \quad p = \sqrt{2P\omega} \cos Q$$

**Verification:**

$$\{Q, P\} = \frac{\partial Q}{\partial q}\frac{\partial P}{\partial p} - \frac{\partial Q}{\partial p}\frac{\partial P}{\partial q}$$

Let's verify by computing q² + p²/ω² = 2P/ω(sin²Q + cos²Q) = 2P/ω

So H = ω²(2P/ω)/2 = ωP = K(P)

The new Hamiltonian depends only on P! Since ∂K/∂Q = 0, we have P = const (action is conserved).

Hamilton's equations give: Q̇ = ∂K/∂P = ω, so Q = ωt + φ (angle increases linearly).

This is the **action-angle formulation**—the precursor to Bohr-Sommerfeld quantization!

#### Example 4: Time-Dependent Transformation

**Moving to a rotating frame:**

$$Q = q\cos(\omega t) + \frac{p}{m\omega}\sin(\omega t)$$
$$P = -m\omega q\sin(\omega t) + p\cos(\omega t)$$

This is canonical with generating function:

$$F_1(q, Q, t) = \frac{m\omega}{2}(q^2 + Q^2)\cot(\omega t) - \frac{m\omega qQ}{\sin(\omega t)}$$

The new Hamiltonian K includes the ∂F₁/∂t term, representing the pseudo-forces in the rotating frame.

---

### 8. Infinitesimal Canonical Transformations

For small ε, consider:

$$Q_i = q_i + \epsilon \{q_i, G\} + O(\epsilon^2) = q_i + \epsilon \frac{\partial G}{\partial p_i}$$
$$P_i = p_i + \epsilon \{p_i, G\} + O(\epsilon^2) = p_i - \epsilon \frac{\partial G}{\partial q_i}$$

where G(q, p) is called the **generator** of the infinitesimal transformation.

**Examples:**

| Generator G | Transformation |
|-------------|----------------|
| p_i | Translation in q_i |
| q_i | Translation in p_i |
| H | Time evolution |
| L_z = xp_y - yp_x | Rotation about z-axis |

**Profound Insight:** This connects to Noether's theorem. If G is conserved ({G, H} = 0), then the transformation it generates is a symmetry of H!

---

## Quantum Mechanics Connection

### The Classical-Quantum Correspondence

| Classical | Quantum |
|-----------|---------|
| Canonical transformation | Unitary transformation |
| Generating function F | Operator Û |
| Infinitesimal generator G | Hermitian generator Ĝ |
| {A, B} = 1 | [Â, B̂] = iℏ |
| Symplectic group Sp(2n) | Metaplectic group Mp(2n) |

### Unitary Transformations as Quantum Canonical Transformations

In quantum mechanics, a transformation |ψ⟩ → |ψ'⟩ = Û|ψ⟩ preserves:
- The inner product: ⟨ψ'|φ'⟩ = ⟨ψ|Û†Û|φ⟩ = ⟨ψ|φ⟩
- The commutation relations: [Q̂, P̂] = [Ûq̂Û†, Ûp̂Û†] = Û[q̂, p̂]Û† = iℏ

**The Correspondence:**

$$\{q, p\} = 1 \quad \longrightarrow \quad [\hat{q}, \hat{p}] = i\hbar$$

Canonical transformations preserve {Q, P} = 1 ↔ Unitary transformations preserve [Q̂, P̂] = iℏ.

### Infinitesimal Unitary Transformations

For classical:
$$\delta q = \epsilon \{q, G\} = \epsilon \frac{\partial G}{\partial p}$$

For quantum:
$$\hat{U} = e^{-i\epsilon \hat{G}/\hbar} \approx 1 - \frac{i\epsilon}{\hbar}\hat{G}$$

$$\delta \hat{q} = \hat{U}^\dagger \hat{q} \hat{U} - \hat{q} \approx \frac{i\epsilon}{\hbar}[\hat{G}, \hat{q}]$$

By the correspondence {G, q} → [Ĝ, q̂]/(iℏ), the quantum transformation matches the classical one!

### Example: Bogoliubov Transformations

In many-body quantum physics, **Bogoliubov transformations** mix creation and annihilation operators:

$$\hat{b} = u\hat{a} + v\hat{a}^\dagger, \quad \hat{b}^\dagger = u^*\hat{a}^\dagger + v^*\hat{a}$$

For this to preserve [b̂, b̂†] = 1, we need |u|² - |v|² = 1.

**Classical Analog:** Writing â = (q̂ + ip̂)/√2, the Bogoliubov transformation corresponds to a symplectic squeeze:

$$\begin{pmatrix} Q \\ P \end{pmatrix} = \begin{pmatrix} \cosh r & \sinh r \\ \sinh r & \cosh r \end{pmatrix} \begin{pmatrix} q \\ p \end{pmatrix}$$

This is canonical (det = 1) and corresponds to u = cosh r, v = sinh r.

### Squeezed States in Quantum Optics

**Squeezed states** are created by the squeeze operator:

$$\hat{S}(\zeta) = \exp\left[\frac{1}{2}(\zeta^* \hat{a}^2 - \zeta \hat{a}^{\dagger 2})\right]$$

This is the quantum version of the classical canonical transformation that "squeezes" the uncertainty ellipse in phase space—reducing position uncertainty at the expense of momentum uncertainty (or vice versa).

**Applications:**
- LIGO gravitational wave detection (squeezed light)
- Quantum metrology beyond standard quantum limit
- Quantum error correction

---

## Worked Examples

### Example 1: Verify a Transformation is Canonical

**Problem:** Show that the transformation
$$Q = \log\left(\frac{\sin p}{q}\right), \quad P = q \cot p$$
is canonical.

**Solution:**

We need to verify that {Q, P} = 1.

$$\{Q, P\} = \frac{\partial Q}{\partial q}\frac{\partial P}{\partial p} - \frac{\partial Q}{\partial p}\frac{\partial P}{\partial q}$$

**Step 1:** Compute partial derivatives.

$$\frac{\partial Q}{\partial q} = -\frac{1}{q}$$

$$\frac{\partial Q}{\partial p} = \frac{\cos p}{\sin p} = \cot p$$

$$\frac{\partial P}{\partial q} = \cot p$$

$$\frac{\partial P}{\partial p} = -q\csc^2 p$$

**Step 2:** Substitute into Poisson bracket.

$$\{Q, P\} = \left(-\frac{1}{q}\right)(-q\csc^2 p) - (\cot p)(\cot p)$$

$$= \csc^2 p - \cot^2 p$$

$$= \frac{1}{\sin^2 p} - \frac{\cos^2 p}{\sin^2 p} = \frac{1 - \cos^2 p}{\sin^2 p} = \frac{\sin^2 p}{\sin^2 p} = 1 \quad \checkmark$$

The transformation is canonical.

---

### Example 2: Find the Generating Function

**Problem:** For the transformation Q = log(1 + q^(1/2) cos p), P = 2(1 + q^(1/2) cos p) q^(1/2) sin p, find a generating function of type F₁(q, Q).

**Solution:**

From the first equation: e^Q = 1 + √q cos p

So: √q cos p = e^Q - 1

From the second equation: P = 2e^Q √q sin p

**Step 1:** Eliminate p.

From (√q cos p)² + (√q sin p)² = q:

$$(e^Q - 1)^2 + \frac{P^2}{4e^{2Q}} = q$$

This is implicit in (q, Q, P)—we need a different approach.

**Step 2:** Use the generating function relations.

For F₁(q, Q): p = ∂F₁/∂q and P = -∂F₁/∂Q

From e^Q = 1 + √q cos p:

$$\cos p = \frac{e^Q - 1}{\sqrt{q}}$$

So: $p = \arccos\left(\frac{e^Q - 1}{\sqrt{q}}\right)$

**Step 3:** Integrate to find F₁.

$$\frac{\partial F_1}{\partial q} = p = \arccos\left(\frac{e^Q - 1}{\sqrt{q}}\right)$$

This integral is complicated. Let's verify the answer by checking:

$$F_1(q, Q) = 2\left[(e^Q - 1)\arccos\left(\frac{e^Q - 1}{\sqrt{q}}\right) - \sqrt{q - (e^Q-1)^2}\right] + q \arccos\left(\frac{e^Q-1}{\sqrt{q}}\right)$$

Differentiation confirms p = ∂F₁/∂q and the P relation.

---

### Example 3: Transform the Harmonic Oscillator

**Problem:** Use the generating function $F_1(q, Q) = \frac{m\omega q^2}{2}\cot Q$ to find (Q, P) and the new Hamiltonian for a harmonic oscillator.

**Solution:**

**Step 1:** Find the transformation equations.

$$p = \frac{\partial F_1}{\partial q} = m\omega q \cot Q$$

$$P = -\frac{\partial F_1}{\partial Q} = \frac{m\omega q^2}{2}\csc^2 Q$$

**Step 2:** Solve for (q, p) in terms of (Q, P).

From the first equation: q = p tan Q/(mω)

Substituting into the second:

$$P = \frac{m\omega}{2} \cdot \frac{p^2 \tan^2 Q}{m^2\omega^2} \cdot \csc^2 Q = \frac{p^2 \tan^2 Q}{2m\omega} \cdot \frac{1}{\sin^2 Q}$$

$$= \frac{p^2}{2m\omega} \cdot \frac{\sin^2 Q}{\cos^2 Q} \cdot \frac{1}{\sin^2 Q} = \frac{p^2}{2m\omega \cos^2 Q}$$

So: $p = \sqrt{2m\omega P} \cos Q$

And: $q = \sqrt{2m\omega P} \cos Q \cdot \frac{\tan Q}{m\omega} = \sqrt{\frac{2P}{m\omega}} \sin Q$

**Step 3:** Find the new Hamiltonian.

The old Hamiltonian: $H = \frac{p^2}{2m} + \frac{1}{2}m\omega^2 q^2$

$$H = \frac{2m\omega P \cos^2 Q}{2m} + \frac{1}{2}m\omega^2 \cdot \frac{2P}{m\omega} \sin^2 Q$$

$$= \omega P \cos^2 Q + \omega P \sin^2 Q = \omega P$$

Since F₁ has no explicit time dependence, K = H:

$$\boxed{K(P) = \omega P}$$

**Step 4:** Solve the dynamics.

$$\dot{Q} = \frac{\partial K}{\partial P} = \omega \implies Q = \omega t + \phi_0$$

$$\dot{P} = -\frac{\partial K}{\partial Q} = 0 \implies P = \text{const}$$

The action P is conserved, and the angle Q increases uniformly. This is the **action-angle formulation**.

---

## Practice Problems

### Level 1: Direct Application

1. **Verify canonical:** Show that Q = q + p, P = p is canonical.

2. **Type 2 generating function:** Given F₂(q, P) = qP + q³/3, find Q(q, P) and p(q, P).

3. **Symplectic check:** Verify that the matrix
   $$M = \begin{pmatrix} 2 & 1 \\ 3 & 2 \end{pmatrix}$$
   is symplectic.

### Level 2: Intermediate

4. **Exchange + rotation:** Find the generating function for Q = -p, P = q (the inverse exchange).

5. **Scaling Hamiltonian:** For H = p²/(2m) + V(q), use the scaling transformation Q = λq, P = p/λ to find K(Q, P). How should λ be chosen to simplify a power-law potential V(q) = αq^n?

6. **Poisson bracket preservation:** Show that if (Q, P) = (f(q, p), g(q, p)) is canonical, then {f, g}_{q,p} = 1.

### Level 3: Challenging

7. **Hamilton's principal function:** For H = p²/(2m), show that the Type 2 generating function F₂ = (q - Q)²m/(2t) generates a canonical transformation with K = 0. Interpret Q physically.

8. **Composition of transformations:** If (q, p) → (Q, P) and (Q, P) → (R, S) are both canonical, prove that (q, p) → (R, S) is canonical. What is the corresponding statement for the Jacobian matrices?

9. **Prove symplectic eigenvalue theorem:** Show that if λ is an eigenvalue of a symplectic matrix M, then so are 1/λ, λ*, and 1/λ*. (Hint: Use M^T J M = J and consider characteristic polynomials.)

---

## Computational Lab

### Lab 1: Verifying Canonical Transformations Numerically

```python
"""
Day 162 Lab: Canonical Transformations
Numerical verification and visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Symplectic matrix J for n=1
def J_matrix(n=1):
    """Standard symplectic matrix."""
    J = np.zeros((2*n, 2*n))
    J[:n, n:] = np.eye(n)
    J[n:, :n] = -np.eye(n)
    return J

def is_symplectic(M, tol=1e-10):
    """Check if matrix M is symplectic."""
    n = M.shape[0] // 2
    J = J_matrix(n)
    condition = M.T @ J @ M - J
    return np.allclose(condition, 0, atol=tol)

# Example: Verify various 2x2 matrices
print("Symplectic Verification:")
print("=" * 40)

# Rotation matrix
theta = np.pi/4
R = np.array([[np.cos(theta), np.sin(theta)],
              [-np.sin(theta), np.cos(theta)]])
print(f"Rotation (θ=π/4): {is_symplectic(R)}")

# Scaling (squeeze)
lam = 2.0
S = np.array([[lam, 0],
              [0, 1/lam]])
print(f"Scaling (λ=2): {is_symplectic(S)}")

# Shear
a = 0.5
Sh = np.array([[1, a],
               [0, 1]])
print(f"Shear: {is_symplectic(Sh)}")

# Non-symplectic example
N = np.array([[2, 0],
              [0, 2]])
print(f"Uniform scaling (not symplectic): {is_symplectic(N)}")
```

### Lab 2: Action-Angle Transformation Visualization

```python
"""
Visualize the harmonic oscillator in both coordinate systems.
"""

def harmonic_oscillator_phase_space():
    """Plot HO trajectories in (q,p) and (Q,P) coordinates."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    omega = 1.0
    m = 1.0

    # Left plot: (q, p) coordinates
    ax1 = axes[0]
    t = np.linspace(0, 2*np.pi, 1000)

    # Several energy levels
    energies = [0.5, 1.0, 2.0, 4.0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(energies)))

    for E, color in zip(energies, colors):
        # q = sqrt(2E/mω²) sin(ωt), p = sqrt(2mE) cos(ωt)
        q_max = np.sqrt(2*E/(m*omega**2))
        p_max = np.sqrt(2*m*E)

        q = q_max * np.sin(omega * t)
        p = p_max * np.cos(omega * t)

        ax1.plot(q, p, color=color, linewidth=2, label=f'E = {E}')

    ax1.set_xlabel('q', fontsize=14)
    ax1.set_ylabel('p', fontsize=14)
    ax1.set_title('Original Coordinates (q, p)\nElliptical Orbits', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5)

    # Right plot: (Q, P) coordinates (action-angle)
    ax2 = axes[1]

    for E, color in zip(energies, colors):
        # P = E/ω = const, Q = ωt
        P = E / omega
        Q = omega * t

        # Q is periodic mod 2π
        ax2.plot(Q % (2*np.pi), [P]*len(Q),
                color=color, linewidth=3, label=f'P = {P:.1f}')

    ax2.set_xlabel('Q (angle)', fontsize=14)
    ax2.set_ylabel('P (action)', fontsize=14)
    ax2.set_title('Action-Angle Coordinates (Q, P)\nHorizontal Lines!', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 2*np.pi)
    ax2.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax2.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])

    plt.tight_layout()
    plt.savefig('action_angle_transformation.png', dpi=150, bbox_inches='tight')
    plt.show()

harmonic_oscillator_phase_space()
```

### Lab 3: Generating Function Numerical Check

```python
"""
Verify generating function relations numerically.
For F_2(q, P) = qP + αq³, check that p = ∂F₂/∂q and Q = ∂F₂/∂P.
"""

def generating_function_verification():
    """Numerical verification of F₂ generating function."""

    alpha = 0.5

    # F_2(q, P) = qP + α*q³
    def F2(q, P):
        return q*P + alpha*q**3

    # Analytical derivatives
    # p = ∂F₂/∂q = P + 3αq²
    # Q = ∂F₂/∂P = q

    def p_from_F2(q, P):
        return P + 3*alpha*q**2

    def Q_from_F2(q, P):
        return q

    # Numerical derivatives
    def numerical_derivative(f, x, y, dx=1e-8, wrt='x'):
        if wrt == 'x':
            return (f(x + dx, y) - f(x - dx, y)) / (2*dx)
        else:
            return (f(x, y + dx) - f(x, y - dx)) / (2*dx)

    # Test at several points
    test_points = [(1.0, 2.0), (0.5, 1.5), (-1.0, 3.0), (2.0, -1.0)]

    print("Generating Function Verification")
    print("F₂(q, P) = qP + 0.5q³")
    print("=" * 60)
    print(f"{'(q, P)':<15} {'p (analytic)':<15} {'p (numeric)':<15} {'Match':<10}")
    print("-" * 60)

    for q, P in test_points:
        p_anal = p_from_F2(q, P)
        p_num = numerical_derivative(F2, q, P, wrt='x')
        match = np.isclose(p_anal, p_num, rtol=1e-6)
        print(f"({q:4.1f}, {P:4.1f})    {p_anal:12.6f}   {p_num:12.6f}   {'✓' if match else '✗'}")

    print("\n" + "=" * 60)
    print(f"{'(q, P)':<15} {'Q (analytic)':<15} {'Q (numeric)':<15} {'Match':<10}")
    print("-" * 60)

    for q, P in test_points:
        Q_anal = Q_from_F2(q, P)
        Q_num = numerical_derivative(F2, q, P, wrt='y')
        match = np.isclose(Q_anal, Q_num, rtol=1e-6)
        print(f"({q:4.1f}, {P:4.1f})    {Q_anal:12.6f}   {Q_num:12.6f}   {'✓' if match else '✗'}")

    # Verify the transformation is canonical
    print("\n\nCanonical Verification:")
    print("For this transformation: Q = q, P = p - 3αq²")
    print("Check {Q, P} = 1:")

    # {Q, P} = ∂Q/∂q * ∂P/∂p - ∂Q/∂p * ∂P/∂q
    # Q = q: ∂Q/∂q = 1, ∂Q/∂p = 0
    # P = p - 3αq²: ∂P/∂p = 1, ∂P/∂q = -6αq
    # {Q, P} = 1 * 1 - 0 * (-6αq) = 1 ✓

    print("{Q, P} = ∂Q/∂q × ∂P/∂p - ∂Q/∂p × ∂P/∂q")
    print("      = 1 × 1 - 0 × (-6αq) = 1 ✓")

generating_function_verification()
```

### Lab 4: Symplectic Integration Comparison

```python
"""
Compare symplectic vs non-symplectic integrators for the harmonic oscillator.
Symplectic integrators preserve phase space area (canonical structure).
"""

def symplectic_vs_euler():
    """Compare Euler (non-symplectic) with Symplectic Euler."""

    # Harmonic oscillator: H = (p² + q²)/2
    omega = 1.0
    dt = 0.1
    n_steps = 1000

    # Initial conditions
    q0, p0 = 1.0, 0.0
    E0 = 0.5 * (p0**2 + omega**2 * q0**2)

    # Euler method (non-symplectic)
    q_euler = np.zeros(n_steps + 1)
    p_euler = np.zeros(n_steps + 1)
    q_euler[0], p_euler[0] = q0, p0

    for i in range(n_steps):
        q_euler[i+1] = q_euler[i] + dt * p_euler[i]
        p_euler[i+1] = p_euler[i] - dt * omega**2 * q_euler[i]

    # Symplectic Euler
    q_symp = np.zeros(n_steps + 1)
    p_symp = np.zeros(n_steps + 1)
    q_symp[0], p_symp[0] = q0, p0

    for i in range(n_steps):
        p_symp[i+1] = p_symp[i] - dt * omega**2 * q_symp[i]  # Update p first
        q_symp[i+1] = q_symp[i] + dt * p_symp[i+1]          # Use new p

    # Störmer-Verlet (also symplectic)
    q_verlet = np.zeros(n_steps + 1)
    p_verlet = np.zeros(n_steps + 1)
    q_verlet[0], p_verlet[0] = q0, p0

    for i in range(n_steps):
        p_half = p_verlet[i] - 0.5 * dt * omega**2 * q_verlet[i]
        q_verlet[i+1] = q_verlet[i] + dt * p_half
        p_verlet[i+1] = p_half - 0.5 * dt * omega**2 * q_verlet[i+1]

    # Compute energies
    E_euler = 0.5 * (p_euler**2 + omega**2 * q_euler**2)
    E_symp = 0.5 * (p_symp**2 + omega**2 * q_symp**2)
    E_verlet = 0.5 * (p_verlet**2 + omega**2 * q_verlet**2)

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    t = np.arange(n_steps + 1) * dt

    # Phase space trajectories
    ax1 = axes[0, 0]
    ax1.plot(q_euler, p_euler, 'r-', alpha=0.7, linewidth=1, label='Euler')
    ax1.plot(q_symp, p_symp, 'b-', alpha=0.7, linewidth=1, label='Symplectic Euler')
    ax1.plot(q_verlet, p_verlet, 'g-', alpha=0.7, linewidth=1, label='Störmer-Verlet')
    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=2, label='Exact')
    ax1.set_xlabel('q', fontsize=12)
    ax1.set_ylabel('p', fontsize=12)
    ax1.set_title('Phase Space Trajectories', fontsize=14)
    ax1.legend()
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # Energy vs time
    ax2 = axes[0, 1]
    ax2.plot(t, E_euler, 'r-', linewidth=1.5, label='Euler')
    ax2.plot(t, E_symp, 'b-', linewidth=1.5, label='Symplectic Euler')
    ax2.plot(t, E_verlet, 'g-', linewidth=1.5, label='Störmer-Verlet')
    ax2.axhline(y=E0, color='k', linestyle='--', linewidth=2, label='Initial E')
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Energy', fontsize=12)
    ax2.set_title('Energy Conservation', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Energy error
    ax3 = axes[1, 0]
    ax3.semilogy(t, np.abs(E_euler - E0) + 1e-16, 'r-', linewidth=1.5, label='Euler')
    ax3.semilogy(t, np.abs(E_symp - E0) + 1e-16, 'b-', linewidth=1.5, label='Symplectic Euler')
    ax3.semilogy(t, np.abs(E_verlet - E0) + 1e-16, 'g-', linewidth=1.5, label='Störmer-Verlet')
    ax3.set_xlabel('Time', fontsize=12)
    ax3.set_ylabel('|E - E₀|', fontsize=12)
    ax3.set_title('Energy Error (log scale)', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Phase space area (should be preserved)
    ax4 = axes[1, 1]

    # Compute area enclosed by trajectory using shoelace formula
    def compute_area(q, p, window=100):
        areas = []
        for i in range(0, len(q) - window, window//2):
            qi = q[i:i+window]
            pi = p[i:i+window]
            # Close the polygon
            area = 0.5 * np.abs(np.dot(qi, np.roll(pi, 1)) - np.dot(pi, np.roll(qi, 1)))
            areas.append(area)
        return np.array(areas)

    # The phase space area for a circle is π
    ax4.axhline(y=np.pi, color='k', linestyle='--', linewidth=2, label='Exact (π)')

    # Add annotations
    ax4.text(0.5, 0.95, 'Symplectic integrators preserve phase space structure',
             transform=ax4.transAxes, fontsize=11, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax4.text(0.5, 0.05, 'Euler spirals outward → phase space grows',
             transform=ax4.transAxes, fontsize=11, ha='center', va='bottom',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

    ax4.set_xlabel('Time window', fontsize=12)
    ax4.set_ylabel('Phase space area', fontsize=12)
    ax4.set_title('Phase Space Area Preservation', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('symplectic_integration.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Print summary
    print("\nIntegration Summary (after t = {:.1f}):".format(n_steps * dt))
    print("=" * 50)
    print(f"Euler:           E_final/E_0 = {E_euler[-1]/E0:.4f}")
    print(f"Symplectic Euler: E_final/E_0 = {E_symp[-1]/E0:.4f}")
    print(f"Störmer-Verlet:   E_final/E_0 = {E_verlet[-1]/E0:.4f}")

symplectic_vs_euler()
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Canonical condition | {Q_i, P_j} = δᵢⱼ, {Q_i, Q_j} = 0, {P_i, P_j} = 0 |
| Symplectic condition | Mᵀ J M = J |
| Type 1: F₁(q, Q) | p = ∂F₁/∂q, P = -∂F₁/∂Q |
| Type 2: F₂(q, P) | p = ∂F₂/∂q, Q = ∂F₂/∂P |
| Type 3: F₃(p, Q) | q = -∂F₃/∂p, P = -∂F₃/∂Q |
| Type 4: F₄(p, P) | q = -∂F₄/∂p, Q = ∂F₄/∂P |
| New Hamiltonian | K = H + ∂F/∂t |
| Infinitesimal transformation | δq = ε{q, G}, δp = ε{p, G} |

### Main Takeaways

1. **Definition:** Canonical transformations preserve the symplectic structure—they are the "symmetries" of Hamiltonian mechanics

2. **Characterization:** Three equivalent conditions:
   - Preserve Hamilton's equations
   - Preserve fundamental Poisson brackets
   - Have symplectic Jacobian (Mᵀ J M = J)

3. **Generating Functions:** Powerful tool for constructing canonical transformations:
   - Four types depending on which variables are independent
   - F₂(q, P) is most common for practical problems
   - Time-dependent F adds ∂F/∂t to the new Hamiltonian

4. **The Symplectic Group:** Sp(2n) is the group of all linear canonical transformations:
   - det(M) = +1 always
   - For n = 1: Sp(2) = SL(2, ℝ) (unit determinant is sufficient)

5. **Infinitesimal Transformations:** Generated by functions G via Poisson brackets:
   - Conserved quantities generate symmetries
   - Time evolution is generated by the Hamiltonian

6. **Quantum Connection:**
   - Canonical transformations → Unitary transformations
   - {A, B} → [Â, B̂]/(iℏ)
   - Squeezed states are quantum symplectic squeezes

---

## Daily Checklist

### Understanding
- [ ] I can explain why canonical transformations preserve the physics (Poisson brackets)
- [ ] I understand the symplectic condition Mᵀ J M = J geometrically
- [ ] I can derive transformation equations from generating functions
- [ ] I see the connection between generators G and infinitesimal transformations

### Computation
- [ ] I can verify whether a transformation is canonical
- [ ] I can find generating functions for given transformations
- [ ] I can transform Hamiltonians to new coordinates
- [ ] I can implement symplectic integrators

### Connections
- [ ] I understand how canonical → unitary in quantum mechanics
- [ ] I see why symplectic integrators are preferred for Hamiltonian systems
- [ ] I can connect action-angle variables to quantization

---

## Preview: Day 163

Tomorrow we study **Liouville's Theorem**, which states that phase space volume is conserved under Hamiltonian evolution. This remarkable result:

- Explains why symplectic integrators work
- Connects to probability conservation in quantum mechanics (unitarity)
- Provides the foundation for statistical mechanics
- Leads to the concept of the Liouville operator

The key formula will be:

$$\frac{d\rho}{dt} = \frac{\partial \rho}{\partial t} + \{\rho, H\} = 0$$

This is the classical version of the quantum von Neumann equation!

---

*"The whole of the theory of canonical transformations is a preparation for the theory of quantum mechanics."*
— Paul Dirac

---

**Day 162 Complete. Next: Liouville's Theorem**
