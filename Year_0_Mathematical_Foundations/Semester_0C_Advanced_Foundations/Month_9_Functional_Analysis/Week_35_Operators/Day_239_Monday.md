# Day 239: Bounded Linear Operators

## Schedule Overview (8 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 3 hours | Theory: Linear operators, boundedness, continuity |
| Afternoon | 3 hours | Problems: Examples, verification, computations |
| Evening | 2 hours | Computational lab: Matrix operators and visualizations |

## Learning Objectives

By the end of today, you will be able to:

1. **Define** linear operators between normed spaces and verify linearity
2. **Explain** the concept of bounded operators and why boundedness matters
3. **Prove** the fundamental equivalence: bounded ⟺ continuous for linear operators
4. **Compute** bounds and verify boundedness for specific operators
5. **Identify** examples of bounded and unbounded operators
6. **Connect** bounded operators to quantum mechanical transformations

---

## 1. Core Content: Linear Operators

### 1.1 Definition of Linear Operators

**Definition**: Let $X$ and $Y$ be vector spaces over $\mathbb{F}$ (where $\mathbb{F} = \mathbb{R}$ or $\mathbb{C}$). A **linear operator** (or **linear map**) is a function $A: X \to Y$ satisfying:

$$\boxed{\begin{aligned}
&\text{(L1) Additivity:} && A(x + y) = Ax + Ay \quad \forall x, y \in X \\
&\text{(L2) Homogeneity:} && A(\alpha x) = \alpha Ax \quad \forall \alpha \in \mathbb{F}, x \in X
\end{aligned}}$$

Equivalently, $A$ is linear if and only if:
$$A(\alpha x + \beta y) = \alpha Ax + \beta Ay \quad \forall \alpha, \beta \in \mathbb{F}, \; x, y \in X$$

**Notation**:
- We write $Ax$ instead of $A(x)$
- The space of all linear operators from $X$ to $Y$ is denoted $\mathcal{L}(X, Y)$
- When $X = Y$, we write $\mathcal{L}(X) = \mathcal{L}(X, X)$

### 1.2 Fundamental Examples

**Example 1: Matrix Operators**

For $A \in \mathbb{C}^{m \times n}$, the map $T_A: \mathbb{C}^n \to \mathbb{C}^m$ defined by $T_A(x) = Ax$ (matrix-vector multiplication) is linear.

**Example 2: Differentiation Operator**

On $C^1[a,b]$, the derivative $D: C^1[a,b] \to C[a,b]$ defined by $Df = f'$ is linear:
$$D(\alpha f + \beta g) = (\alpha f + \beta g)' = \alpha f' + \beta g' = \alpha Df + \beta Dg$$

**Example 3: Integration Operator**

On $C[a,b]$, the integral operator $I: C[a,b] \to C[a,b]$ defined by:
$$(If)(x) = \int_a^x f(t) \, dt$$
is linear.

**Example 4: Shift Operators on $\ell^2$**

The **right shift** $S_R: \ell^2 \to \ell^2$:
$$S_R(x_1, x_2, x_3, \ldots) = (0, x_1, x_2, x_3, \ldots)$$

The **left shift** $S_L: \ell^2 \to \ell^2$:
$$S_L(x_1, x_2, x_3, \ldots) = (x_2, x_3, x_4, \ldots)$$

**Example 5: Multiplication Operators**

On $L^2[a,b]$, given $\phi \in L^\infty[a,b]$, define $M_\phi: L^2 \to L^2$ by:
$$(M_\phi f)(x) = \phi(x) f(x)$$

This is linear: $M_\phi(\alpha f + \beta g) = \phi(\alpha f + \beta g) = \alpha \phi f + \beta \phi g$.

---

## 2. Bounded Operators

### 2.1 Definition and Motivation

**Definition**: A linear operator $A: X \to Y$ between normed spaces is **bounded** if there exists $M \geq 0$ such that:

$$\boxed{\|Ax\|_Y \leq M \|x\|_X \quad \forall x \in X}$$

The infimum of all such $M$ is the **operator norm** (studied in detail tomorrow):
$$\|A\| = \inf\{M : \|Ax\| \leq M\|x\| \text{ for all } x\}$$

**Why "bounded"?** The terminology comes from the fact that $A$ maps the unit ball $\{x : \|x\| \leq 1\}$ to a bounded set in $Y$.

### 2.2 Equivalent Characterizations

**Theorem**: For a linear operator $A: X \to Y$, the following are equivalent:

1. $A$ is bounded
2. $A$ maps bounded sets to bounded sets
3. $A$ maps the unit ball to a bounded set
4. $\sup_{\|x\| = 1} \|Ax\| < \infty$
5. $\sup_{x \neq 0} \frac{\|Ax\|}{\|x\|} < \infty$

**Proof**: (1) $\Rightarrow$ (4): If $\|Ax\| \leq M\|x\|$ and $\|x\| = 1$, then $\|Ax\| \leq M$.

(4) $\Rightarrow$ (5): For $x \neq 0$, $\frac{\|Ax\|}{\|x\|} = \|A(x/\|x\|)\| \leq \sup_{\|y\|=1}\|Ay\|$.

(5) $\Rightarrow$ (1): Let $M = \sup_{x \neq 0} \frac{\|Ax\|}{\|x\|}$. Then $\|Ax\| \leq M\|x\|$.

(1) $\Rightarrow$ (2): If $S$ is bounded, $\|x\| \leq R$ for all $x \in S$. Then $\|Ax\| \leq MR$.

(2) $\Rightarrow$ (3): The unit ball is bounded. $\square$

### 2.3 The BLT Theorem: Bounded ⟺ Continuous

**Theorem (BLT - Bounded Linear Transformation)**: For a linear operator $A: X \to Y$ between normed spaces, the following are equivalent:

1. $A$ is bounded
2. $A$ is continuous at $0$
3. $A$ is continuous at every point
4. $A$ is uniformly continuous

This is one of the most important theorems in functional analysis!

**Proof**:

**(1) $\Rightarrow$ (4)**: Let $\varepsilon > 0$. Choose $\delta = \varepsilon / M$ where $\|Ax\| \leq M\|x\|$.

If $\|x - y\| < \delta$, then:
$$\|Ax - Ay\| = \|A(x-y)\| \leq M\|x-y\| < M \cdot \frac{\varepsilon}{M} = \varepsilon$$

So $A$ is uniformly continuous.

**(4) $\Rightarrow$ (3) $\Rightarrow$ (2)**: Trivial implications.

**(2) $\Rightarrow$ (1)**: Assume $A$ is continuous at $0$. Taking $\varepsilon = 1$, there exists $\delta > 0$ such that:
$$\|x\| < \delta \Rightarrow \|Ax\| < 1$$

For any $x \neq 0$, let $y = \frac{\delta x}{2\|x\|}$. Then $\|y\| = \delta/2 < \delta$, so $\|Ay\| < 1$.

Since $Ay = \frac{\delta}{2\|x\|} Ax$, we have:
$$\|Ax\| = \frac{2\|x\|}{\delta} \|Ay\| < \frac{2\|x\|}{\delta} \cdot 1 = \frac{2}{\delta} \|x\|$$

So $A$ is bounded with $M = 2/\delta$. $\square$

### 2.4 Unbounded Operators

**Important**: Not all linear operators are bounded!

**Example: Differentiation on $C^1[0,1]$**

Consider $D: C^1[0,1] \to C[0,1]$ with the supremum norm on both spaces.

Let $f_n(x) = \sin(nx)$. Then:
- $\|f_n\|_\infty = 1$
- $Df_n = f_n' = n\cos(nx)$, so $\|Df_n\|_\infty = n$

Thus $\frac{\|Df_n\|}{\|f_n\|} = n \to \infty$, so $D$ is **unbounded**.

**Physical Significance**: In quantum mechanics, the momentum operator $\hat{p} = -i\hbar\frac{d}{dx}$ is unbounded. This reflects the fact that momentum is not bounded for arbitrary quantum states.

---

## 3. Quantum Mechanics Connection

### 3.1 Operators as Physical Transformations

In quantum mechanics, every physical operation corresponds to an operator:

| Physical Operation | Operator Type |
|-------------------|---------------|
| Observable measurement | Self-adjoint operator |
| Symmetry transformation | Unitary operator |
| Time evolution | Unitary operator $U(t) = e^{-iHt/\hbar}$ |
| State preparation | Projection operator |
| General transformation | Linear operator |

### 3.2 Bounded vs. Unbounded in QM

| Property | Bounded Operators | Unbounded Operators |
|----------|------------------|---------------------|
| Domain | All of $\mathcal{H}$ | Dense subspace |
| Continuity | Always continuous | Often discontinuous |
| Examples | Spin, angular momentum components, bounded observables | Position $\hat{x}$, momentum $\hat{p}$, Hamiltonian $\hat{H}$ |
| Spectrum | Always bounded | Can be unbounded |

### 3.3 Why Boundedness Matters

For bounded operators:
1. **Well-defined on all states**: $A|\psi\rangle$ exists for every $|\psi\rangle$
2. **Continuous physics**: Small changes in state → small changes in output
3. **Algebraic operations work**: Can freely multiply, add bounded operators
4. **Exponentials converge**: $e^A = \sum_{n=0}^\infty A^n/n!$ converges

For unbounded operators (like $\hat{p}$):
1. **Domain restrictions**: Only acts on "nice enough" functions
2. **Careful with composition**: $\hat{x}\hat{p}$ may not equal $\hat{p}\hat{x}$ even on their common domain
3. **Essential for physics**: Position, momentum, energy are unbounded

### 3.4 The Bounded Approximation

Even when dealing with unbounded operators, we often:
1. Restrict to finite-dimensional subspaces (where everything is bounded)
2. Consider bounded functions of unbounded operators
3. Use spectral projections (bounded) from unbounded operators

---

## 4. Worked Examples

### Example 1: Verifying Boundedness of the Right Shift

**Problem**: Show that the right shift $S_R: \ell^2 \to \ell^2$ defined by $S_R(x_1, x_2, \ldots) = (0, x_1, x_2, \ldots)$ is bounded, and find $\|S_R\|$.

**Solution**:

**Step 1: Verify linearity.**
$$S_R(\alpha x + \beta y) = (0, \alpha x_1 + \beta y_1, \alpha x_2 + \beta y_2, \ldots) = \alpha S_R x + \beta S_R y \checkmark$$

**Step 2: Compute the norm.**
$$\|S_R x\|^2 = |0|^2 + |x_1|^2 + |x_2|^2 + \cdots = \sum_{n=1}^\infty |x_n|^2 = \|x\|^2$$

So $\|S_R x\| = \|x\|$ for all $x$, meaning $S_R$ is an **isometry**.

**Step 3: Find the operator norm.**
$$\|S_R\| = \sup_{\|x\|=1} \|S_R x\| = \sup_{\|x\|=1} \|x\| = 1$$

$$\boxed{\|S_R\| = 1}$$

---

### Example 2: The Multiplication Operator

**Problem**: Let $\phi \in L^\infty[0,1]$ and define $M_\phi: L^2[0,1] \to L^2[0,1]$ by $(M_\phi f)(x) = \phi(x)f(x)$. Prove $M_\phi$ is bounded and find $\|M_\phi\|$.

**Solution**:

**Step 1: Boundedness.**
$$\|M_\phi f\|_2^2 = \int_0^1 |\phi(x) f(x)|^2 \, dx \leq \|\phi\|_\infty^2 \int_0^1 |f(x)|^2 \, dx = \|\phi\|_\infty^2 \|f\|_2^2$$

So $\|M_\phi f\|_2 \leq \|\phi\|_\infty \|f\|_2$.

**Step 2: The bound is achieved.**

For any $\varepsilon > 0$, there exists a set $E \subseteq [0,1]$ with positive measure where $|\phi(x)| > \|\phi\|_\infty - \varepsilon$.

Let $f = \chi_E / \sqrt{|E|}$ (the normalized indicator function of $E$). Then $\|f\|_2 = 1$ and:
$$\|M_\phi f\|_2^2 = \frac{1}{|E|} \int_E |\phi(x)|^2 \, dx > (\|\phi\|_\infty - \varepsilon)^2$$

Since $\varepsilon$ is arbitrary:
$$\boxed{\|M_\phi\| = \|\phi\|_\infty = \text{ess sup}|\phi|}$$

---

### Example 3: An Integral Operator

**Problem**: Consider the integral operator $K: L^2[0,1] \to L^2[0,1]$ defined by:
$$(Kf)(x) = \int_0^1 k(x,y) f(y) \, dy$$
where $k(x,y) = xy$. Prove $K$ is bounded.

**Solution**:

**Step 1: Apply Cauchy-Schwarz.**
$$|(Kf)(x)| = \left|\int_0^1 xy \cdot f(y) \, dy\right| \leq \int_0^1 |xy| \cdot |f(y)| \, dy$$

By Cauchy-Schwarz:
$$|(Kf)(x)| \leq \left(\int_0^1 |xy|^2 \, dy\right)^{1/2} \left(\int_0^1 |f(y)|^2 \, dy\right)^{1/2}$$

**Step 2: Compute the inner integral.**
$$\int_0^1 |xy|^2 \, dy = x^2 \int_0^1 y^2 \, dy = \frac{x^2}{3}$$

So $|(Kf)(x)| \leq \frac{|x|}{\sqrt{3}} \|f\|_2$.

**Step 3: Compute $\|Kf\|_2$.**
$$\|Kf\|_2^2 = \int_0^1 |(Kf)(x)|^2 \, dx \leq \frac{\|f\|_2^2}{3} \int_0^1 x^2 \, dx = \frac{\|f\|_2^2}{9}$$

Therefore:
$$\|Kf\|_2 \leq \frac{1}{3} \|f\|_2$$

So $K$ is bounded with $\|K\| \leq 1/3$. $\square$

---

## 5. Practice Problems

### Level 1: Direct Application

1. **Verify linearity** of the left shift $S_L: \ell^2 \to \ell^2$ defined by $S_L(x_1, x_2, x_3, \ldots) = (x_2, x_3, x_4, \ldots)$.

2. Prove that the left shift is bounded and compute $\|S_L\|$.

3. For the operator $A: \mathbb{C}^2 \to \mathbb{C}^2$ given by the matrix $A = \begin{pmatrix} 1 & 2 \\ 0 & 3 \end{pmatrix}$, verify boundedness and estimate $\|A\|$.

### Level 2: Intermediate

4. **Prove**: If $A$ and $B$ are bounded linear operators, then so is $A + B$ and $\|A + B\| \leq \|A\| + \|B\|$.

5. Let $A: \ell^2 \to \ell^2$ be defined by $A(x_n) = (x_n / n)$. Prove $A$ is bounded and compute $\|A\|$.

6. **Quantum Connection**: The number operator $N$ on Fock space acts on number states as $N|n\rangle = n|n\rangle$. Is $N$ bounded? Justify your answer.

### Level 3: Challenging

7. **Prove or disprove**: Every linear operator $A: \ell^2 \to \ell^2$ that is bounded when restricted to each finite-dimensional subspace $\text{span}\{e_1, \ldots, e_n\}$ must be bounded on all of $\ell^2$.

8. Let $A: L^2[0,1] \to L^2[0,1]$ be defined by $(Af)(x) = xf(x)$.
   - Prove $A$ is bounded.
   - Find $\|A\|$ exactly.
   - Compute $A^2$ and $\|A^2\|$.

9. **Research problem**: Define the Volterra operator $V: L^2[0,1] \to L^2[0,1]$ by $(Vf)(x) = \int_0^x f(t) \, dt$. Prove $V$ is bounded with $\|V\| \leq 1$. (Hint: This is actually a compact operator, studied on Day 244.)

---

## 6. Computational Lab: Bounded Operators

```python
"""
Day 239 Computational Lab: Bounded Linear Operators
Visualizing operators on finite-dimensional and sequence spaces
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm as matrix_norm
from mpl_toolkits.mplot3d import Axes3D

# ============================================================
# Part 1: Matrix Operators and Unit Ball Images
# ============================================================

def visualize_operator_action():
    """
    Visualize how a 2x2 matrix transforms the unit circle.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Different matrices to visualize
    matrices = [
        (np.array([[2, 0], [0, 1]]), "Stretch: diag(2,1)"),
        (np.array([[1, 1], [0, 1]]), "Shear"),
        (np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)],
                   [np.sin(np.pi/4), np.cos(np.pi/4)]]), "Rotation 45°"),
        (np.array([[0.5, 0], [0, 0.5]]), "Contraction"),
        (np.array([[1, 2], [0, 3]]), "General"),
        (np.array([[0, 1], [1, 0]]), "Reflection/Permutation")
    ]

    # Unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    unit_circle = np.array([np.cos(theta), np.sin(theta)])

    for ax, (A, title) in zip(axes.flat, matrices):
        # Transform unit circle
        transformed = A @ unit_circle

        # Plot
        ax.plot(unit_circle[0], unit_circle[1], 'b-',
                linewidth=2, label='Unit circle', alpha=0.5)
        ax.plot(transformed[0], transformed[1], 'r-',
                linewidth=2, label='Image')

        # Mark some points
        for t in [0, np.pi/2, np.pi, 3*np.pi/2]:
            pt = np.array([np.cos(t), np.sin(t)])
            img = A @ pt
            ax.plot([pt[0], img[0]], [pt[1], img[1]], 'g--', alpha=0.5)
            ax.plot(pt[0], pt[1], 'bo', markersize=5)
            ax.plot(img[0], img[1], 'ro', markersize=5)

        # Compute operator norm (spectral norm = largest singular value)
        op_norm = np.linalg.norm(A, ord=2)

        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{title}\n||A|| = {op_norm:.3f}')
        ax.legend(fontsize=8)

    plt.suptitle('Linear Operators Transform the Unit Ball', fontsize=14)
    plt.tight_layout()
    plt.savefig('operator_action.png', dpi=150)
    plt.show()

# ============================================================
# Part 2: Shift Operators on Sequences
# ============================================================

def demonstrate_shift_operators():
    """
    Visualize right and left shift operators on ell^2.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Original sequence (truncated to N terms)
    N = 20
    n = np.arange(N)
    x = np.exp(-n/5) * np.cos(n)  # Decaying oscillation

    # Right shift
    x_right = np.concatenate([[0], x[:-1]])

    # Left shift
    x_left = np.concatenate([x[1:], [0]])

    # Plot original
    axes[0, 0].stem(n, x, basefmt='k-', linefmt='b-', markerfmt='bo')
    axes[0, 0].set_title('Original sequence $x = (x_1, x_2, ...)$')
    axes[0, 0].set_xlabel('n')
    axes[0, 0].set_ylabel('$x_n$')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot right shift
    axes[0, 1].stem(n, x_right, basefmt='k-', linefmt='r-', markerfmt='ro')
    axes[0, 1].set_title('Right shift $S_R x = (0, x_1, x_2, ...)$')
    axes[0, 1].set_xlabel('n')
    axes[0, 1].set_ylabel('$(S_R x)_n$')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot left shift
    axes[1, 0].stem(n, x_left, basefmt='k-', linefmt='g-', markerfmt='go')
    axes[1, 0].set_title('Left shift $S_L x = (x_2, x_3, ...)$')
    axes[1, 0].set_xlabel('n')
    axes[1, 0].set_ylabel('$(S_L x)_n$')
    axes[1, 0].grid(True, alpha=0.3)

    # Norms comparison
    norm_x = np.linalg.norm(x)
    norm_right = np.linalg.norm(x_right)
    norm_left = np.linalg.norm(x_left)

    # Note: Left shift loses x_1, so norm decreases if x_1 != 0
    bars = axes[1, 1].bar(['||x||', '||S_R x||', '||S_L x||'],
                          [norm_x, norm_right, norm_left],
                          color=['blue', 'red', 'green'], alpha=0.7)
    axes[1, 1].set_title('Norm Comparison')
    axes[1, 1].set_ylabel('Norm')

    # Annotate
    for bar, val in zip(bars, [norm_x, norm_right, norm_left]):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                       f'{val:.4f}', ha='center', fontsize=10)

    axes[1, 1].axhline(y=norm_x, color='b', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('shift_operators.png', dpi=150)
    plt.show()

    print("Shift Operator Properties:")
    print(f"  ||x|| = {norm_x:.6f}")
    print(f"  ||S_R x|| = {norm_right:.6f} (isometry: ||S_R x|| = ||x||)")
    print(f"  ||S_L x|| = {norm_left:.6f} (contraction: ||S_L x|| ≤ ||x||)")
    print(f"  Note: ||S_L x|| < ||x|| because x_1 = {x[0]:.4f} is lost")

# ============================================================
# Part 3: Multiplication Operator
# ============================================================

def multiplication_operator():
    """
    Visualize the multiplication operator M_phi on L^2[0,1].
    """
    x = np.linspace(0, 1, 1000)

    # Different multiplier functions
    phis = [
        (lambda x: x, '$\\phi(x) = x$'),
        (lambda x: x**2, '$\\phi(x) = x^2$'),
        (lambda x: np.sin(2*np.pi*x), '$\\phi(x) = \\sin(2\\pi x)$'),
        (lambda x: np.exp(-x), '$\\phi(x) = e^{-x}$')
    ]

    # Test function
    f = lambda x: np.sin(np.pi * x)  # Simple function in L^2

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, (phi, phi_name) in zip(axes.flat, phis):
        phi_vals = phi(x)
        f_vals = f(x)
        Mf_vals = phi_vals * f_vals

        ax.plot(x, f_vals, 'b-', linewidth=2, label='$f(x) = \\sin(\\pi x)$')
        ax.plot(x, phi_vals, 'g--', linewidth=2, label=phi_name)
        ax.plot(x, Mf_vals, 'r-', linewidth=2, label='$(M_\\phi f)(x)$')

        # Compute norms
        norm_f = np.sqrt(np.trapz(f_vals**2, x))
        norm_Mf = np.sqrt(np.trapz(Mf_vals**2, x))
        norm_phi_inf = np.max(np.abs(phi_vals))

        ax.set_xlabel('$x$')
        ax.set_title(f'{phi_name}\n$||M_\\phi|| = ||\\phi||_\\infty = {norm_phi_inf:.3f}$')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Add norm info
        textstr = f'$||f||_2 = {norm_f:.3f}$\n$||M_\\phi f||_2 = {norm_Mf:.3f}$'
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Multiplication Operators $M_\\phi: f \\mapsto \\phi \\cdot f$', fontsize=14)
    plt.tight_layout()
    plt.savefig('multiplication_operator.png', dpi=150)
    plt.show()

# ============================================================
# Part 4: Bounded vs Unbounded - Differentiation
# ============================================================

def bounded_vs_unbounded():
    """
    Demonstrate why differentiation is unbounded.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.linspace(0, 1, 1000)

    # Functions f_n(x) = sin(n*pi*x) / sqrt(2)
    # ||f_n||_2 = 1 for all n
    # f_n'(x) = n*pi*cos(n*pi*x) / sqrt(2)
    # ||f_n'||_2 = n*pi (approximately)

    n_values = [1, 2, 5, 10, 20]
    colors = plt.cm.viridis(np.linspace(0, 1, len(n_values)))

    # Plot functions and derivatives
    for n, color in zip(n_values, colors):
        f_n = np.sin(n * np.pi * x) / np.sqrt(0.5)  # Normalized
        # Note: ||sin(n*pi*x)||_2 = 1/sqrt(2) on [0,1], so divide by sqrt(0.5)

        norm_fn = np.sqrt(np.trapz(f_n**2, x))

        axes[0].plot(x, f_n, color=color, linewidth=1.5,
                    label=f'$f_{{{n}}}$, $||f_{{{n}}}||_2 = {norm_fn:.2f}$')

    axes[0].set_xlabel('$x$')
    axes[0].set_ylabel('$f_n(x)$')
    axes[0].set_title('Functions $f_n(x) = \\sin(n\\pi x)$ (normalized)')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Ratio ||f_n'|| / ||f_n|| as n increases
    n_range = np.arange(1, 51)
    ratios = []

    for n in n_range:
        f_n = np.sin(n * np.pi * x)
        f_n_prime = n * np.pi * np.cos(n * np.pi * x)

        norm_fn = np.sqrt(np.trapz(f_n**2, x))
        norm_fn_prime = np.sqrt(np.trapz(f_n_prime**2, x))

        ratios.append(norm_fn_prime / norm_fn if norm_fn > 0 else 0)

    axes[1].plot(n_range, ratios, 'b-o', markersize=3)
    axes[1].plot(n_range, n_range * np.pi, 'r--', label='$n\\pi$ (theoretical)')
    axes[1].set_xlabel('$n$')
    axes[1].set_ylabel('$||Df_n|| / ||f_n||$')
    axes[1].set_title('Differentiation is Unbounded: $||D|| = \\infty$')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('unbounded_differentiation.png', dpi=150)
    plt.show()

    print("Differentiation Operator Analysis:")
    print("  f_n(x) = sin(nπx) with ||f_n||_2 = √(1/2)")
    print("  f_n'(x) = nπ cos(nπx) with ||f_n'||_2 = nπ√(1/2)")
    print("  Ratio ||Df_n||/||f_n|| = nπ → ∞ as n → ∞")
    print("  Conclusion: Differentiation D is UNBOUNDED")

# ============================================================
# Part 5: Quantum Application - Creation/Annihilation Operators
# ============================================================

def quantum_ladder_operators():
    """
    Visualize creation and annihilation operators (finite truncation).
    """
    N = 10  # Truncate Fock space to N dimensions

    # Creation operator a†: |n⟩ → √(n+1)|n+1⟩
    # Annihilation operator a: |n⟩ → √n|n-1⟩

    a_dag = np.zeros((N, N))  # a†
    a = np.zeros((N, N))      # a

    for n in range(N-1):
        a_dag[n+1, n] = np.sqrt(n + 1)
        a[n, n+1] = np.sqrt(n + 1)

    # Number operator N = a†a
    number_op = a_dag @ a

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot a†
    im1 = axes[0].imshow(np.abs(a_dag), cmap='Blues')
    axes[0].set_title('Creation Operator $a^\\dagger$\n$a^\\dagger|n\\rangle = \\sqrt{n+1}|n+1\\rangle$')
    axes[0].set_xlabel('Input state $|n\\rangle$')
    axes[0].set_ylabel('Output state $|m\\rangle$')
    plt.colorbar(im1, ax=axes[0])

    # Plot a
    im2 = axes[1].imshow(np.abs(a), cmap='Reds')
    axes[1].set_title('Annihilation Operator $a$\n$a|n\\rangle = \\sqrt{n}|n-1\\rangle$')
    axes[1].set_xlabel('Input state $|n\\rangle$')
    axes[1].set_ylabel('Output state $|m\\rangle$')
    plt.colorbar(im2, ax=axes[1])

    # Plot N
    im3 = axes[2].imshow(np.abs(number_op), cmap='Greens')
    axes[2].set_title('Number Operator $N = a^\\dagger a$\n$N|n\\rangle = n|n\\rangle$')
    axes[2].set_xlabel('Input state $|n\\rangle$')
    axes[2].set_ylabel('Output state $|m\\rangle$')
    plt.colorbar(im3, ax=axes[2])

    plt.suptitle('Quantum Harmonic Oscillator Operators (truncated to 10 states)', fontsize=14)
    plt.tight_layout()
    plt.savefig('ladder_operators.png', dpi=150)
    plt.show()

    # Compute operator norms
    norm_a_dag = np.linalg.norm(a_dag, ord=2)
    norm_a = np.linalg.norm(a, ord=2)
    norm_N = np.linalg.norm(number_op, ord=2)

    print(f"\nOperator Norms (truncated to {N} dimensions):")
    print(f"  ||a†|| = {norm_a_dag:.4f} ≈ √{N-1} = {np.sqrt(N-1):.4f}")
    print(f"  ||a|| = {norm_a:.4f} ≈ √{N-1} = {np.sqrt(N-1):.4f}")
    print(f"  ||N|| = {norm_N:.4f} ≈ {N-1}")
    print(f"\nNote: In infinite dimensions, a, a†, and N are all UNBOUNDED!")

# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Day 239: Bounded Linear Operators - Computational Lab")
    print("=" * 60)

    print("\n1. Visualizing operator action on unit ball...")
    visualize_operator_action()

    print("\n2. Demonstrating shift operators...")
    demonstrate_shift_operators()

    print("\n3. Multiplication operator examples...")
    multiplication_operator()

    print("\n4. Bounded vs unbounded: differentiation...")
    bounded_vs_unbounded()

    print("\n5. Quantum ladder operators...")
    quantum_ladder_operators()

    print("\n" + "=" * 60)
    print("Lab complete! Key takeaway:")
    print("  Bounded operators map bounded sets to bounded sets.")
    print("  Differentiation is the classic example of an unbounded operator.")
    print("=" * 60)
```

---

## 7. Summary

### Key Definitions

| Concept | Definition |
|---------|------------|
| **Linear Operator** | $A(\alpha x + \beta y) = \alpha Ax + \beta Ay$ |
| **Bounded Operator** | $\exists M: \|Ax\| \leq M\|x\|$ for all $x$ |
| **Operator Norm** | $\|A\| = \sup_{\|x\|=1} \|Ax\|$ |

### Key Formulas

$$\boxed{\begin{aligned}
&\text{Boundedness:} && \|Ax\| \leq M\|x\| \\
&\text{Operator Norm:} && \|A\| = \sup_{\|x\|=1} \|Ax\| = \sup_{x \neq 0} \frac{\|Ax\|}{\|x\|} \\
&\text{BLT Theorem:} && A \text{ bounded} \Leftrightarrow A \text{ continuous}
\end{aligned}}$$

### Key Examples

| Operator | Space | Bounded? | Norm |
|----------|-------|----------|------|
| Right shift $S_R$ | $\ell^2$ | Yes | $\|S_R\| = 1$ |
| Left shift $S_L$ | $\ell^2$ | Yes | $\|S_L\| = 1$ |
| Multiplication $M_\phi$ | $L^2$ | Yes | $\|\phi\|_\infty$ |
| Differentiation $D$ | $C^1$ | **No** | — |

### Key Insights

1. **Bounded = Continuous** for linear operators (BLT Theorem)
2. **Matrix operators** are always bounded on finite-dimensional spaces
3. **Differentiation** is the prototype of an unbounded operator
4. **In quantum mechanics**: observables like position and momentum are unbounded
5. **Bounded operators** form a nice algebraic structure (tomorrow's topic)

---

## 8. Daily Checklist

- [ ] I can define a linear operator and verify linearity
- [ ] I can define a bounded operator and explain why boundedness matters
- [ ] I can prove the BLT theorem (bounded ⟺ continuous)
- [ ] I can verify boundedness and compute bounds for specific operators
- [ ] I can give examples of bounded and unbounded operators
- [ ] I understand why differentiation is unbounded
- [ ] I can connect bounded operators to quantum mechanics
- [ ] I completed the computational lab exercises

---

## 9. Preview: Day 240

Tomorrow we study the **operator norm** in detail and prove that $\mathcal{B}(\mathcal{H})$, the space of all bounded linear operators on a Hilbert space, is a **Banach algebra**. This means:
- The operator norm makes $\mathcal{B}(\mathcal{H})$ a complete normed space
- Multiplication satisfies $\|AB\| \leq \|A\|\|B\|$
- We can define functions of operators like $e^A$

This algebraic structure is essential for quantum mechanics, where operator products represent sequential measurements and exponentials describe time evolution.

---

*"The notion of a bounded linear operator is fundamental to functional analysis. It captures exactly those linear transformations that behave 'nicely' with respect to the topology—the continuous ones."* — Walter Rudin
