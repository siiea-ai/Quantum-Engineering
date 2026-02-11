# Day 158: Poisson Brackets — The Algebraic Heart of Mechanics

## Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Poisson Bracket Formalism |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## Learning Objectives

By the end of today, you should be able to:

1. Define the Poisson bracket and compute it for arbitrary functions
2. Prove and apply the fundamental Poisson brackets {qᵢ, pⱼ} = δᵢⱼ
3. Verify the algebraic properties: antisymmetry, bilinearity, Leibniz rule, and Jacobi identity
4. Express Hamilton's equations and time evolution in terms of Poisson brackets
5. Apply Poisson's theorem to generate new constants of motion
6. Explain the quantum correspondence {A, B} → [Â, B̂]/(iℏ) and derive the canonical commutation relations

---

## Core Content

### 1. Historical Context

The Poisson bracket was introduced by **Siméon Denis Poisson** in 1809 during his studies of perturbation theory for planetary orbits. **Carl Gustav Jacob Jacobi** (1842-43) recognized its profound utility and reformulated all of Hamiltonian mechanics using this elegant structure.

The most revolutionary development came in **1925** when **Paul Dirac**, a 23-year-old graduate student at Cambridge, recognized that Poisson brackets provide the classical analog of quantum mechanical commutators. This insight established a rigorous bridge between classical and quantum mechanics, replacing the ad-hoc quantization rules of Bohr's old quantum theory.

---

### 2. Definition of the Poisson Bracket

**Definition:** For a system with n degrees of freedom and canonical coordinates (q₁, ..., qₙ, p₁, ..., pₙ), the **Poisson bracket** of two functions f(q, p, t) and g(q, p, t) is:

$$\boxed{\{f, g\} = \sum_{i=1}^{n} \left( \frac{\partial f}{\partial q_i} \frac{\partial g}{\partial p_i} - \frac{\partial f}{\partial p_i} \frac{\partial g}{\partial q_i} \right)}$$

**Compact Matrix Form:** Using the symplectic matrix J:

$$\{f, g\} = (\nabla f)^T \mathbf{J} (\nabla g)$$

where:
$$\mathbf{J} = \begin{pmatrix} \mathbf{0} & \mathbf{I} \\ -\mathbf{I} & \mathbf{0} \end{pmatrix}$$

**Physical Interpretation:** The Poisson bracket measures the "incompatibility" of two observables — how much knowledge of one limits knowledge of the other. This interpretation becomes exact in quantum mechanics.

---

### 3. Fundamental Poisson Brackets

The canonical coordinates themselves satisfy:

$$\boxed{\{q_i, q_j\} = 0, \quad \{p_i, p_j\} = 0, \quad \{q_i, p_j\} = \delta_{ij}}$$

**Proof of {qᵢ, pⱼ} = δᵢⱼ:**

$$\{q_i, p_j\} = \sum_k \left( \frac{\partial q_i}{\partial q_k} \frac{\partial p_j}{\partial p_k} - \frac{\partial q_i}{\partial p_k} \frac{\partial p_j}{\partial q_k} \right)$$

Since ∂qᵢ/∂qₖ = δᵢₖ, ∂pⱼ/∂pₖ = δⱼₖ, and ∂qᵢ/∂pₖ = ∂pⱼ/∂qₖ = 0:

$$\{q_i, p_j\} = \sum_k \delta_{ik}\delta_{jk} - 0 = \delta_{ij}$$

**The fundamental brackets encode the entire canonical structure of phase space!**

---

### 4. Algebraic Properties

The Poisson bracket satisfies four fundamental properties, forming a **Lie algebra**:

| Property | Statement | Formula |
|----------|-----------|---------|
| **Antisymmetry** | Order matters with sign flip | {f, g} = −{g, f} |
| **Bilinearity** | Linear in both arguments | {αf + βg, h} = α{f, h} + β{g, h} |
| **Leibniz Rule** | Product rule | {fg, h} = f{g, h} + {f, h}g |
| **Jacobi Identity** | Triple bracket cycles | {f, {g, h}} + {g, {h, f}} + {h, {f, g}} = 0 |

**Immediate Consequence:** From antisymmetry: {f, f} = 0 for any function f.

---

### 5. The Jacobi Identity

**Statement:** For any three functions f, g, h on phase space:

$$\boxed{\{f, \{g, h\}\} + \{g, \{h, f\}\} + \{h, \{f, g\}\} = 0}$$

**Proof Strategy:** Direct computation shows that each term generates third-order partial derivatives like ∂³f/(∂qᵢ∂pⱼ∂qₖ), which cancel pairwise due to the symmetry of mixed partial derivatives.

**Significance:** The Jacobi identity ensures that:
1. The space of observables forms a proper **Lie algebra**
2. Canonical transformations are consistent
3. The correspondence with quantum commutators is exact (commutators also satisfy Jacobi)

---

### 6. Time Evolution via Poisson Brackets

**Fundamental Evolution Equation:**

For any observable f(q, p, t), its total time derivative is:

$$\boxed{\frac{df}{dt} = \{f, H\} + \frac{\partial f}{\partial t}}$$

**Proof:**
$$\frac{df}{dt} = \sum_i \left( \frac{\partial f}{\partial q_i}\dot{q}_i + \frac{\partial f}{\partial p_i}\dot{p}_i \right) + \frac{\partial f}{\partial t}$$

Using Hamilton's equations ∂H/∂pᵢ = q̇ᵢ and -∂H/∂qᵢ = ṗᵢ:

$$\frac{df}{dt} = \sum_i \left( \frac{\partial f}{\partial q_i}\frac{\partial H}{\partial p_i} - \frac{\partial f}{\partial p_i}\frac{\partial H}{\partial q_i} \right) + \frac{\partial f}{\partial t} = \{f, H\} + \frac{\partial f}{\partial t}$$

---

### 7. Hamilton's Equations as Poisson Brackets

Hamilton's equations emerge as special cases:

$$\dot{q}_i = \{q_i, H\} = \frac{\partial H}{\partial p_i}$$

$$\dot{p}_i = \{p_i, H\} = -\frac{\partial H}{\partial q_i}$$

**Key Insight:** The Hamiltonian generates time translation via the Poisson bracket!

More generally: any conserved quantity G generates a symmetry transformation via δf = ε{f, G}.

---

### 8. Constants of Motion

**Definition:** A function f(q, p, t) is a **constant of motion** if df/dt = 0 along trajectories.

**Conservation Criterion:** For f with no explicit time dependence:

$$\boxed{\{f, H\} = 0 \quad \Leftrightarrow \quad f \text{ is conserved}}$$

This provides a systematic method to test and discover conservation laws.

**Examples:**
- Energy: {H, H} = 0 (automatic — H is always conserved if ∂H/∂t = 0)
- Angular momentum: {L, H} = 0 for central forces
- Linear momentum: {p, H} = 0 for translation-invariant systems

---

### 9. Poisson's Theorem

**Theorem:** If f and g are both constants of motion (with no explicit time dependence), then {f, g} is also a constant of motion.

$$\boxed{\{f, H\} = 0 \text{ and } \{g, H\} = 0 \quad \Rightarrow \quad \{\{f, g\}, H\} = 0}$$

**Proof:** Using the Jacobi identity with h = H:

$$\{f, \{g, H\}\} + \{g, \{H, f\}\} + \{H, \{f, g\}\} = 0$$

Since {g, H} = 0 and {H, f} = -{f, H} = 0:

$$0 + 0 + \{H, \{f, g\}\} = 0$$

By antisymmetry: {{f, g}, H} = 0, so {f, g} is conserved. ∎

**Significance:** Poisson's theorem provides a method to generate **new** constants of motion from known ones!

**Limitation:** The result may be trivial (a constant) or dependent on existing integrals.

---

### 10. Angular Momentum Algebra

For a particle with position **r** = (x, y, z) and momentum **p** = (pₓ, pᵧ, pᵤ):

$$L_x = yp_z - zp_y, \quad L_y = zp_x - xp_z, \quad L_z = xp_y - yp_x$$

**The Poisson brackets form the so(3) Lie algebra:**

$$\boxed{\{L_i, L_j\} = \varepsilon_{ijk} L_k}$$

Explicitly:
$$\{L_x, L_y\} = L_z, \quad \{L_y, L_z\} = L_x, \quad \{L_z, L_x\} = L_y$$

**Proof of {Lₓ, Lᵧ} = Lᵤ:**

$$\{L_x, L_y\} = \{yp_z - zp_y, zp_x - xp_z\}$$

Using bilinearity:
$$= \{yp_z, zp_x\} - \{yp_z, xp_z\} - \{zp_y, zp_x\} + \{zp_y, xp_z\}$$

Using the Leibniz rule on {yp_z, zp_x}:
$$\{yp_z, zp_x\} = y\{p_z, zp_x\} + \{y, zp_x\}p_z = y(p_x\{p_z, z\}) + z\{y, p_x\}p_z$$
$$= y \cdot p_x \cdot (-1) + 0 = -yp_x$$

Similarly computing all terms and combining: {Lₓ, Lᵧ} = xpᵧ - ypₓ = Lᵤ ✓

**Total Angular Momentum:** L² = Lₓ² + Lᵧ² + Lᵤ² commutes with all components:

$$\{L^2, L_i\} = 0 \quad \text{for all } i$$

---

## Quantum Mechanics Connection

### The Dirac Correspondence

In 1925, Dirac recognized the profound correspondence between classical Poisson brackets and quantum commutators:

$$\boxed{\{A, B\}_{\text{classical}} \quad \longleftrightarrow \quad \frac{1}{i\hbar}[\hat{A}, \hat{B}]_{\text{quantum}}}$$

Or equivalently:

$$\boxed{[\hat{A}, \hat{B}] = i\hbar \widehat{\{A, B\}}}$$

---

### Canonical Commutation Relations

The fundamental Poisson brackets become the **canonical commutation relations**:

| Classical | → | Quantum |
|-----------|---|---------|
| {qᵢ, pⱼ} = δᵢⱼ | → | [q̂ᵢ, p̂ⱼ] = iℏδᵢⱼ |
| {qᵢ, qⱼ} = 0 | → | [q̂ᵢ, q̂ⱼ] = 0 |
| {pᵢ, pⱼ} = 0 | → | [p̂ᵢ, p̂ⱼ] = 0 |

These relations are the foundation of all quantum mechanics!

---

### Heisenberg Equation of Motion

The classical evolution equation transforms into the **Heisenberg equation**:

**Classical:**
$$\frac{df}{dt} = \{f, H\} + \frac{\partial f}{\partial t}$$

**Quantum:**
$$\boxed{\frac{d\hat{A}}{dt} = \frac{1}{i\hbar}[\hat{A}, \hat{H}] + \frac{\partial \hat{A}}{\partial t}}$$

The structural similarity is **exact** — only the Poisson bracket is replaced by (1/iℏ) × commutator.

---

### Why Does the Correspondence Work?

Both structures are **Lie algebras**:
- Classical observables with Poisson bracket
- Quantum operators with (1/iℏ) × commutator

Both satisfy:
1. Antisymmetry
2. Bilinearity
3. Leibniz rule (product rule)
4. Jacobi identity

Quantization is essentially a **Lie algebra homomorphism** from classical to quantum observables.

---

### Ehrenfest's Theorem

Taking expectation values of the Heisenberg equation:

$$\frac{d\langle \hat{x} \rangle}{dt} = \frac{\langle \hat{p} \rangle}{m}, \quad \frac{d\langle \hat{p} \rangle}{dt} = -\left\langle \frac{\partial V}{\partial x} \right\rangle$$

**Expectation values follow approximately classical trajectories** when quantum corrections are small — the correspondence principle in action!

---

### Ordering Ambiguity

The correspondence has subtleties:

Classical: qp = pq (numbers commute)
Quantum: q̂p̂ ≠ p̂q̂ (operators don't commute)

The quantization of qp is ambiguous:
- Could be q̂p̂
- Could be p̂q̂
- Could be ½(q̂p̂ + p̂q̂) (symmetric ordering)

The **Groenewold-van Hove theorem** proves there's no consistent quantization preserving all Poisson brackets for polynomials beyond degree 2.

---

### Angular Momentum Quantization

The classical angular momentum algebra:
$$\{L_i, L_j\} = \varepsilon_{ijk} L_k$$

becomes the quantum **su(2) Lie algebra**:
$$[\hat{L}_i, \hat{L}_j] = i\hbar \varepsilon_{ijk} \hat{L}_k$$

This structure governs:
- Orbital angular momentum
- Spin (even with no classical analog!)
- Isospin in nuclear physics
- SU(2) gauge symmetry

---

## Worked Examples

### Example 1: Compute {x², p²}

**Problem:** Calculate the Poisson bracket of x² and p² for a 1D system.

**Solution:**

Using the Leibniz rule:
$$\{x^2, p^2\} = x\{x, p^2\} + \{x, p^2\}x = 2x\{x, p^2\}$$

Now compute {x, p²}:
$$\{x, p^2\} = p\{x, p\} + \{x, p\}p = p \cdot 1 + 1 \cdot p = 2p$$

Therefore:
$$\{x^2, p^2\} = 2x \cdot 2p = 4xp$$

**Verification via definition:**
$$\{x^2, p^2\} = \frac{\partial(x^2)}{\partial x}\frac{\partial(p^2)}{\partial p} - \frac{\partial(x^2)}{\partial p}\frac{\partial(p^2)}{\partial x} = 2x \cdot 2p - 0 = 4xp \quad ✓$$

---

### Example 2: Verify {L², Lᵤ} = 0

**Problem:** Show that total angular momentum commutes with any component.

**Solution:**

$$\{L^2, L_z\} = \{L_x^2 + L_y^2 + L_z^2, L_z\}$$

Using bilinearity and Leibniz rule:
$$= 2L_x\{L_x, L_z\} + 2L_y\{L_y, L_z\} + 2L_z\{L_z, L_z\}$$

Using {Lₓ, Lᵤ} = -Lᵧ, {Lᵧ, Lᵤ} = Lₓ, {Lᵤ, Lᵤ} = 0:
$$= 2L_x(-L_y) + 2L_y(L_x) + 0 = -2L_xL_y + 2L_yL_x$$

In classical mechanics, LₓLᵧ = LᵧLₓ (they're just numbers), so:
$$\{L^2, L_z\} = 0 \quad ✓$$

---

### Example 3: Lᵤ Generates Rotations

**Problem:** Show that Lᵤ generates infinitesimal rotations about the z-axis.

**Solution:**

An infinitesimal canonical transformation generated by G = εLᵤ gives:

$$\delta x = \varepsilon\{x, L_z\} = \varepsilon\{x, xp_y - yp_x\}$$

Computing:
$$\{x, xp_y - yp_x\} = \{x, xp_y\} - \{x, yp_x\} = 0 - y\{x, p_x\} = -y$$

Similarly:
$$\delta y = \varepsilon\{y, L_z\} = +\varepsilon x$$
$$\delta z = \varepsilon\{z, L_z\} = 0$$

This is exactly an **infinitesimal rotation** by angle ε about the z-axis!

$$\begin{pmatrix} x' \\ y' \end{pmatrix} = \begin{pmatrix} x - \varepsilon y \\ y + \varepsilon x \end{pmatrix} = \begin{pmatrix} \cos\varepsilon & -\sin\varepsilon \\ \sin\varepsilon & \cos\varepsilon \end{pmatrix}\begin{pmatrix} x \\ y \end{pmatrix} + O(\varepsilon^2)$$

---

### Example 4: Harmonic Oscillator Constants of Motion

**Problem:** For H = (p² + ω²q²)/2, find all independent polynomial constants of motion.

**Solution:**

**Energy H itself** is conserved (since {H, H} = 0).

**Try quadratic f = aq² + bqp + cp²:**

$$\{f, H\} = \left\{aq^2 + bqp + cp^2, \frac{p^2 + \omega^2 q^2}{2}\right\}$$

Computing term by term:
- {q², p²/2} = q{q, p²/2} + {q, p²/2}q = q·p + p·q = 2qp
- {q², ω²q²/2} = 0
- {qp, p²/2} = q{p, p²}/2 + {q, p²/2}p = 0 + p·p = p²
- {qp, ω²q²/2} = q{p, ω²q²/2} + {q, ω²q²/2}p = -ω²q² + 0 = -ω²q²
- {p², p²/2} = 0
- {p², ω²q²/2} = -2ω²qp

Combining:
$$\{f, H\} = a(2qp) + b(p^2 - \omega^2 q^2) + c(-2\omega^2 qp) = 0$$

Grouping by terms:
- Coefficient of qp: 2a - 2cω² = 0 → a = cω²
- Coefficient of p²: b
- Coefficient of q²: -bω²

For {f, H} = 0: b = 0 (no qp term allowed unless ω = 1).

With b = 0 and a = cω²:
$$f = c\omega^2 q^2 + cp^2 = c(\omega^2 q^2 + p^2) = 2cH$$

**Conclusion:** For the harmonic oscillator, the only polynomial constants of motion are functions of H itself. The system is **maximally non-degenerate**.

---

## Practice Problems

### Level 1: Direct Application

1. **Basic Computation:** Calculate {x³, p²} using the Leibniz rule.

2. **Fundamental Brackets:** Verify that {q², p} = 2q directly from the definition.

3. **Time Evolution:** For H = p²/(2m) + V(x), compute dx/dt and dp/dt using Poisson brackets.

### Level 2: Intermediate

4. **Angular Momentum:** Calculate {Lₓ, Lᵧ} explicitly using the definition (not just quoting the result).

5. **Poisson's Theorem:** Given that energy H and angular momentum Lᵤ are conserved for a central force, what can you conclude about {H, Lᵤ}?

6. **Generator:** Show that linear momentum pₓ generates translations in x, i.e., {x, pₓ} = 1 and {pₓ, pₓ} = 0.

### Level 3: Challenging

7. **Jacobi Identity:** Verify the Jacobi identity for f = x, g = p, h = x² explicitly.

8. **2D Oscillator:** For H = (pₓ² + pᵧ²)/2 + (ω₁²x² + ω₂²y²)/2 with ω₁ ≠ ω₂, find all quadratic constants of motion. Compare with the case ω₁ = ω₂.

9. **Quantum Correspondence:** Starting from {Lᵢ, Lⱼ} = εᵢⱼₖLₖ, write the quantum commutation relations. If L² commutes classically with all Lᵢ, what does this imply about [L̂², L̂ᵢ]?

---

## Computational Lab

```python
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, simplify, sin, cos, sqrt, Matrix
from sympy import init_printing
init_printing()

def poisson_bracket_demo():
    """
    Demonstrate Poisson bracket calculations symbolically and numerically.
    """

    print("=" * 70)
    print("POISSON BRACKETS: THE ALGEBRAIC HEART OF MECHANICS")
    print("=" * 70)

    # Define symbolic variables
    q, p, t = symbols('q p t', real=True)
    x, y, z = symbols('x y z', real=True)
    px, py, pz = symbols('p_x p_y p_z', real=True)
    m, omega, k = symbols('m omega k', positive=True)

    def poisson_bracket_1d(f, g, q_var, p_var):
        """Compute Poisson bracket {f, g} for 1D system."""
        return diff(f, q_var) * diff(g, p_var) - diff(f, p_var) * diff(g, q_var)

    def poisson_bracket_3d(f, g):
        """Compute Poisson bracket for 3D Cartesian coordinates."""
        coords = [(x, px), (y, py), (z, pz)]
        result = 0
        for q_i, p_i in coords:
            result += diff(f, q_i) * diff(g, p_i) - diff(f, p_i) * diff(g, q_i)
        return simplify(result)

    # =========================================
    # Part 1: Fundamental Brackets
    # =========================================
    print("\n" + "=" * 50)
    print("PART 1: FUNDAMENTAL POISSON BRACKETS")
    print("=" * 50)

    # {q, p} = 1
    bracket_qp = poisson_bracket_1d(q, p, q, p)
    print(f"\n{{q, p}} = {bracket_qp}")

    # {q, q} = 0
    bracket_qq = poisson_bracket_1d(q, q, q, p)
    print(f"{{q, q}} = {bracket_qq}")

    # {p, p} = 0
    bracket_pp = poisson_bracket_1d(p, p, q, p)
    print(f"{{p, p}} = {bracket_pp}")

    print("\n✓ Fundamental brackets verified: {q,p}=1, {q,q}={p,p}=0")

    # =========================================
    # Part 2: Hamilton's Equations via Poisson Brackets
    # =========================================
    print("\n" + "=" * 50)
    print("PART 2: HAMILTON'S EQUATIONS AS POISSON BRACKETS")
    print("=" * 50)

    # Simple Harmonic Oscillator
    H_sho = p**2/(2*m) + m*omega**2*q**2/2
    print(f"\nHarmonic Oscillator: H = {H_sho}")

    q_dot = poisson_bracket_1d(q, H_sho, q, p)
    p_dot = poisson_bracket_1d(p, H_sho, q, p)

    print(f"\nq̇ = {{q, H}} = {simplify(q_dot)}")
    print(f"ṗ = {{p, H}} = {simplify(p_dot)}")
    print("\n✓ Hamilton's equations recovered!")

    # =========================================
    # Part 3: Angular Momentum Algebra
    # =========================================
    print("\n" + "=" * 50)
    print("PART 3: ANGULAR MOMENTUM POISSON BRACKETS (so(3) algebra)")
    print("=" * 50)

    # Define angular momentum components
    Lx = y*pz - z*py
    Ly = z*px - x*pz
    Lz = x*py - y*px

    print(f"\nL_x = {Lx}")
    print(f"L_y = {Ly}")
    print(f"L_z = {Lz}")

    # Compute brackets
    Lx_Ly = poisson_bracket_3d(Lx, Ly)
    Ly_Lz = poisson_bracket_3d(Ly, Lz)
    Lz_Lx = poisson_bracket_3d(Lz, Lx)

    print(f"\n{{L_x, L_y}} = {Lx_Ly}")
    print(f"{{L_y, L_z}} = {Ly_Lz}")
    print(f"{{L_z, L_x}} = {Lz_Lx}")

    # Verify
    print(f"\n✓ Verification:")
    print(f"  {{L_x, L_y}} = L_z? {simplify(Lx_Ly - Lz) == 0}")
    print(f"  {{L_y, L_z}} = L_x? {simplify(Ly_Lz - Lx) == 0}")
    print(f"  {{L_z, L_x}} = L_y? {simplify(Lz_Lx - Ly) == 0}")

    # L² commutes with all components
    L_squared = Lx**2 + Ly**2 + Lz**2
    L2_Lz = poisson_bracket_3d(L_squared, Lz)
    print(f"\n{{L², L_z}} = {simplify(L2_Lz)}")
    print("✓ Total angular momentum commutes with components!")

    # =========================================
    # Part 4: Verify Jacobi Identity
    # =========================================
    print("\n" + "=" * 50)
    print("PART 4: JACOBI IDENTITY VERIFICATION")
    print("=" * 50)

    # Test with angular momentum components (non-trivial test!)
    f, g, h = Lx, Ly, Lz

    term1 = poisson_bracket_3d(f, poisson_bracket_3d(g, h))
    term2 = poisson_bracket_3d(g, poisson_bracket_3d(h, f))
    term3 = poisson_bracket_3d(h, poisson_bracket_3d(f, g))

    jacobi_sum = simplify(term1 + term2 + term3)

    print(f"\n{{{f}, {{{g}, {h}}}}} + {{{g}, {{{h}, {f}}}}} + {{{h}, {{{f}, {g}}}}} = {jacobi_sum}")
    print(f"\n✓ Jacobi identity verified: {jacobi_sum == 0}")

    # =========================================
    # Part 5: Poisson's Theorem Example
    # =========================================
    print("\n" + "=" * 50)
    print("PART 5: POISSON'S THEOREM")
    print("=" * 50)

    # For central force, both H and L_z are conserved
    # Their bracket {H, L_z} must also be conserved (trivially, it's zero)

    r = sqrt(x**2 + y**2 + z**2)
    H_central = (px**2 + py**2 + pz**2)/(2*m) - k/r

    print(f"\nCentral force Hamiltonian: H = p²/(2m) - k/r")

    H_Lz = poisson_bracket_3d(H_central, Lz)
    print(f"{{H, L_z}} = {simplify(H_Lz)}")
    print("\n✓ Angular momentum is conserved for central forces!")
    print("  Poisson's theorem: {{H, L_z}, H} = 0 (automatically satisfied)")

    return


def numerical_poisson_evolution():
    """
    Numerically demonstrate time evolution via Poisson brackets.
    """

    print("\n" + "=" * 70)
    print("NUMERICAL EVOLUTION VIA POISSON BRACKETS")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # =========================================
    # System: Simple Harmonic Oscillator
    # =========================================

    def poisson_bracket_numerical(df_dq, df_dp, dg_dq, dg_dp):
        """Compute Poisson bracket {f, g} numerically."""
        return df_dq * dg_dp - df_dp * dg_dq

    # SHO parameters
    m, omega = 1.0, 1.0

    # Hamilton's equations via Poisson brackets
    # dq/dt = {q, H} = p/m
    # dp/dt = {p, H} = -m*omega^2*q

    def derivatives_sho(state, t):
        q, p = state
        # Using Poisson bracket formulation
        dq_dt = p / m  # {q, H}
        dp_dt = -m * omega**2 * q  # {p, H}
        return [dq_dt, dp_dt]

    # Solve
    from scipy.integrate import odeint

    t = np.linspace(0, 4*np.pi, 500)
    state0 = [1.0, 0.0]
    solution = odeint(derivatives_sho, state0, t)
    q_t, p_t = solution[:, 0], solution[:, 1]

    # Plot phase space
    ax = axes[0, 0]
    ax.plot(q_t, p_t, 'b-', lw=2)
    ax.plot(state0[0], state0[1], 'go', markersize=10, label='Start')
    ax.set_xlabel('q (position)', fontsize=12)
    ax.set_ylabel('p (momentum)', fontsize=12)
    ax.set_title('SHO: Phase Space Trajectory\nEvolved via {q,H} and {p,H}', fontsize=12)
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot time evolution
    ax = axes[0, 1]
    ax.plot(t, q_t, 'b-', lw=2, label='q(t)')
    ax.plot(t, p_t, 'r-', lw=2, label='p(t)')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('q, p', fontsize=12)
    ax.set_title('SHO: Time Evolution\ndq/dt = {q, H}, dp/dt = {p, H}', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Verify energy conservation (H should commute with itself)
    H_t = p_t**2/(2*m) + m*omega**2*q_t**2/2
    ax = axes[1, 0]
    ax.plot(t, H_t, 'g-', lw=2)
    ax.axhline(H_t[0], color='k', linestyle='--', alpha=0.5, label=f'H₀ = {H_t[0]:.4f}')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('H (Energy)', fontsize=12)
    ax.set_title(f'Energy Conservation: {{H, H}} = 0\nΔH/H = {(H_t.max()-H_t.min())/H_t[0]:.2e}', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # =========================================
    # System: 2D Central Force (Angular Momentum)
    # =========================================

    def derivatives_central(state, t, k=1.0, m=1.0):
        r, theta, p_r, L = state
        # Hamilton's equations via Poisson brackets
        # H = p_r²/(2m) + L²/(2mr²) - k/r
        dr_dt = p_r / m  # {r, H}
        dtheta_dt = L / (m * r**2)  # {θ, H}
        dp_r_dt = L**2 / (m * r**3) - k / r**2  # {p_r, H}
        dL_dt = 0  # {L, H} = 0 (θ is cyclic!)
        return [dr_dt, dtheta_dt, dp_r_dt, dL_dt]

    # Initial conditions for elliptical orbit
    r0, theta0, p_r0, L0 = 1.0, 0.0, 0.0, 0.8
    state0_central = [r0, theta0, p_r0, L0]

    t_central = np.linspace(0, 30, 2000)
    sol_central = odeint(derivatives_central, state0_central, t_central)

    r_t, theta_t, p_r_t, L_t = sol_central[:, 0], sol_central[:, 1], sol_central[:, 2], sol_central[:, 3]
    x_t = r_t * np.cos(theta_t)
    y_t = r_t * np.sin(theta_t)

    ax = axes[1, 1]
    ax.plot(x_t, y_t, 'b-', lw=1)
    ax.scatter([0], [0], c='orange', s=200, marker='*', zorder=5, label='Force center')
    ax.plot(x_t[0], y_t[0], 'go', markersize=8, label='Start')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(f'Central Force: {{L, H}} = 0 → L conserved\nL(t) = {L_t[0]:.4f} (constant)', fontsize=12)
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('poisson_bracket_evolution.png', dpi=150)
    plt.show()

    print("\nKey observations:")
    print(f"  • SHO energy variation: ΔH/H = {(H_t.max()-H_t.min())/H_t[0]:.2e}")
    print(f"  • Angular momentum variation: ΔL/L = {(L_t.max()-L_t.min())/L_t[0]:.2e}")
    print("  • Both conserved quantities satisfy {f, H} = 0")


def quantum_correspondence_demo():
    """
    Demonstrate the classical-quantum Poisson bracket correspondence.
    """

    print("\n" + "=" * 70)
    print("CLASSICAL → QUANTUM CORRESPONDENCE")
    print("{A, B} → [Â, B̂]/(iℏ)")
    print("=" * 70)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # =========================================
    # Demonstration 1: Ehrenfest's Theorem
    # =========================================

    # Compare classical trajectory with quantum expectation value
    # For SHO, they match exactly (linear system)

    m, omega, hbar = 1.0, 1.0, 1.0

    # Classical solution
    t = np.linspace(0, 4*np.pi, 500)
    q0, p0 = 3.0, 0.0
    q_classical = q0 * np.cos(omega * t)
    p_classical = -m * omega * q0 * np.sin(omega * t)

    # Quantum expectation (same for SHO - Ehrenfest exact)
    q_quantum = q0 * np.cos(omega * t)
    p_quantum = -m * omega * q0 * np.sin(omega * t)

    ax = axes[0]
    ax.plot(t, q_classical, 'b-', lw=2, label='Classical q(t)')
    ax.plot(t, q_quantum, 'r--', lw=2, label='Quantum ⟨q̂⟩(t)')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Position', fontsize=12)
    ax.set_title("Ehrenfest's Theorem: SHO\nClassical = Quantum (exactly!)", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # =========================================
    # Demonstration 2: Angular Momentum Algebra
    # =========================================

    # Classical: {L_i, L_j} = ε_ijk L_k
    # Quantum: [L̂_i, L̂_j] = iℏ ε_ijk L̂_k

    ax = axes[1]

    # Draw the algebra structure
    ax.text(0.5, 0.9, "ANGULAR MOMENTUM ALGEBRA", fontsize=14,
            ha='center', transform=ax.transAxes, fontweight='bold')

    ax.text(0.5, 0.7, "Classical (Poisson):", fontsize=12,
            ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.6, r"{$L_x$, $L_y$} = $L_z$", fontsize=14,
            ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.5, r"{$L_y$, $L_z$} = $L_x$", fontsize=14,
            ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.4, r"{$L_z$, $L_x$} = $L_y$", fontsize=14,
            ha='center', transform=ax.transAxes)

    ax.text(0.5, 0.25, "Quantum (Commutator):", fontsize=12,
            ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.15, r"[$\hat{L}_x$, $\hat{L}_y$] = $i\hbar\hat{L}_z$", fontsize=14,
            ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.05, "Same structure × iℏ!", fontsize=12,
            ha='center', transform=ax.transAxes, style='italic', color='red')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('so(3) Lie Algebra\n{·,·} → [·,·]/(iℏ)', fontsize=12)

    # =========================================
    # Demonstration 3: Canonical Commutation Relations
    # =========================================

    ax = axes[2]

    ax.text(0.5, 0.9, "CANONICAL RELATIONS", fontsize=14,
            ha='center', transform=ax.transAxes, fontweight='bold')

    ax.text(0.5, 0.7, "Classical:", fontsize=12,
            ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.6, "{q, p} = 1", fontsize=16,
            ha='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightblue'))

    ax.text(0.5, 0.4, "Quantum:", fontsize=12,
            ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.3, r"[$\hat{q}$, $\hat{p}$] = $i\hbar$", fontsize=16,
            ha='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightgreen'))

    ax.text(0.5, 0.12, "Heisenberg Uncertainty:", fontsize=12,
            ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.02, r"$\Delta q \cdot \Delta p \geq \hbar/2$", fontsize=16,
            ha='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow'))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Foundation of QM\nfrom Poisson Brackets!', fontsize=12)

    plt.tight_layout()
    plt.savefig('quantum_correspondence.png', dpi=150)
    plt.show()

    print("\nThe Dirac Correspondence:")
    print("  Classical: {A, B} = C")
    print("  Quantum:   [Â, B̂] = iℏĈ")
    print("\nThis single insight bridges classical and quantum mechanics!")


def leibniz_jacobi_visual():
    """
    Visualize the Leibniz rule and Jacobi identity.
    """

    print("\n" + "=" * 70)
    print("ALGEBRAIC PROPERTIES: LEIBNIZ RULE & JACOBI IDENTITY")
    print("=" * 70)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # =========================================
    # Leibniz Rule Visualization
    # =========================================

    ax = axes[0]
    ax.text(0.5, 0.95, "LEIBNIZ RULE (Product Rule)", fontsize=14,
            ha='center', transform=ax.transAxes, fontweight='bold')

    ax.text(0.5, 0.8, "{fg, h} = f{g, h} + {f, h}g", fontsize=18,
            ha='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightcyan', edgecolor='blue'))

    ax.text(0.5, 0.6, "Example: {q²p, H}", fontsize=14,
            ha='center', transform=ax.transAxes)

    ax.text(0.5, 0.45, "= q²{p, H} + {q², H}p", fontsize=14,
            ha='center', transform=ax.transAxes)

    ax.text(0.5, 0.3, "= q²{p, H} + 2q{q, H}p", fontsize=14,
            ha='center', transform=ax.transAxes)

    ax.text(0.5, 0.1, "Allows systematic computation\nof complex brackets!", fontsize=12,
            ha='center', transform=ax.transAxes, style='italic', color='green')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # =========================================
    # Jacobi Identity Visualization
    # =========================================

    ax = axes[1]
    ax.text(0.5, 0.95, "JACOBI IDENTITY", fontsize=14,
            ha='center', transform=ax.transAxes, fontweight='bold')

    ax.text(0.5, 0.75, "{f, {g, h}} + {g, {h, f}} + {h, {f, g}} = 0", fontsize=14,
            ha='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange'))

    # Draw cyclic diagram
    import matplotlib.patches as patches

    # Triangle vertices
    r = 0.15
    center = (0.5, 0.4)
    angles = [90, 210, 330]  # degrees

    for i, (angle, label) in enumerate(zip(angles, ['f', 'g', 'h'])):
        x = center[0] + r * np.cos(np.radians(angle))
        y = center[1] + r * np.sin(np.radians(angle))
        circle = plt.Circle((x, y), 0.05, color='lightblue', ec='blue')
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=14, fontweight='bold')

    # Draw arrows (cyclic)
    for i in range(3):
        angle1 = angles[i]
        angle2 = angles[(i+1) % 3]
        x1 = center[0] + r * np.cos(np.radians(angle1))
        y1 = center[1] + r * np.sin(np.radians(angle1))
        x2 = center[0] + r * np.cos(np.radians(angle2))
        y2 = center[1] + r * np.sin(np.radians(angle2))
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2))

    ax.text(0.5, 0.15, "Cyclic sum = 0", fontsize=14,
            ha='center', transform=ax.transAxes)

    ax.text(0.5, 0.02, "Ensures Lie algebra structure\n→ Same as quantum commutators!",
            fontsize=11, ha='center', transform=ax.transAxes, style='italic', color='purple')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('algebraic_properties.png', dpi=150)
    plt.show()


# Run all demonstrations
if __name__ == "__main__":
    poisson_bracket_demo()
    numerical_poisson_evolution()
    quantum_correspondence_demo()
    leibniz_jacobi_visual()

    print("\n" + "=" * 70)
    print("POISSON BRACKET LAB COMPLETE")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. {f, g} = Σᵢ(∂f/∂qᵢ ∂g/∂pᵢ - ∂f/∂pᵢ ∂g/∂qᵢ)")
    print("  2. Fundamental: {qᵢ, pⱼ} = δᵢⱼ")
    print("  3. Time evolution: df/dt = {f, H} + ∂f/∂t")
    print("  4. Conservation: {f, H} = 0 ↔ f conserved")
    print("  5. Quantum: {A, B} → [Â, B̂]/(iℏ)")
    print("  6. Angular momentum: {Lᵢ, Lⱼ} = εᵢⱼₖLₖ (so(3) algebra)")
```

---

## Summary

### The Poisson Bracket

$$\boxed{\{f, g\} = \sum_{i=1}^{n} \left( \frac{\partial f}{\partial q_i} \frac{\partial g}{\partial p_i} - \frac{\partial f}{\partial p_i} \frac{\partial g}{\partial q_i} \right)}$$

### Fundamental Relations

| Bracket | Value | Significance |
|---------|-------|--------------|
| {qᵢ, pⱼ} | δᵢⱼ | Canonical structure |
| {qᵢ, qⱼ} | 0 | Positions commute |
| {pᵢ, pⱼ} | 0 | Momenta commute |

### Key Equations

$$\boxed{\frac{df}{dt} = \{f, H\} + \frac{\partial f}{\partial t}}$$

$$\boxed{\{f, H\} = 0 \quad \Leftrightarrow \quad f \text{ is conserved}}$$

### The Classical-Quantum Bridge

$$\boxed{\{A, B\}_{\text{classical}} \quad \longleftrightarrow \quad \frac{1}{i\hbar}[\hat{A}, \hat{B}]_{\text{quantum}}}$$

### Angular Momentum Algebra

$$\boxed{\{L_i, L_j\} = \varepsilon_{ijk} L_k \quad \longrightarrow \quad [\hat{L}_i, \hat{L}_j] = i\hbar\varepsilon_{ijk}\hat{L}_k}$$

---

## Daily Checklist

- [ ] Define Poisson bracket and compute basic examples
- [ ] Verify fundamental brackets {q, p} = 1
- [ ] Prove and use the Leibniz rule
- [ ] State and understand the Jacobi identity
- [ ] Express Hamilton's equations via Poisson brackets
- [ ] Apply conservation criterion {f, H} = 0
- [ ] Compute angular momentum Poisson brackets
- [ ] Explain the Dirac correspondence to quantum mechanics
- [ ] Run computational lab and verify results

---

## Preview: Day 159

Tomorrow we study **Constants of Motion and Integrable Systems** — how the criterion {f, H} = 0 leads to complete solvability of mechanical systems and the deep connection between symmetries and conservation laws through Noether's theorem!
