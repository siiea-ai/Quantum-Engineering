# Day 184: Classification of Singularities

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Theory: Singularity Classification |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Theorems & Applications |
| Evening | 7:00 PM - 9:00 PM | 2 hours | Computational Lab |

**Total Study Time: 8 hours**

---

## Learning Objectives

By the end of Day 184, you will be able to:

1. Classify isolated singularities as removable, poles, or essential
2. Apply Riemann's theorem on removable singularities
3. Determine the order of a pole
4. Apply the Casorati-Weierstrass theorem for essential singularities
5. Connect singularity classification to Laurent series structure
6. Relate poles to resonances in quantum scattering theory

---

## Core Content

### 1. Isolated Singularities: Definition and Overview

An **isolated singularity** of $f(z)$ at $z_0$ is a point where $f$ is not analytic, but $f$ is analytic in some **punctured disk** $0 < |z - z_0| < R$.

**Examples:**
- $f(z) = \frac{1}{z}$ has isolated singularity at $z = 0$
- $f(z) = \frac{1}{\sin z}$ has isolated singularities at $z = n\pi$
- $f(z) = \frac{1}{\sin(1/z)}$ has non-isolated singularity at $z = 0$

### 2. The Three Types of Isolated Singularities

The classification is based on the **principal part** of the Laurent series:

$$f(z) = \underbrace{\sum_{n=1}^{\infty} a_{-n}(z-z_0)^{-n}}_{\text{Principal Part}} + \underbrace{\sum_{n=0}^{\infty} a_n(z-z_0)^n}_{\text{Analytic Part}}$$

| Type | Principal Part | Condition |
|------|----------------|-----------|
| **Removable** | No terms | $a_{-n} = 0$ for all $n \geq 1$ |
| **Pole of order $m$** | Finitely many terms | $a_{-m} \neq 0$, $a_{-n} = 0$ for $n > m$ |
| **Essential** | Infinitely many terms | Infinitely many $a_{-n} \neq 0$ |

### 3. Removable Singularities

**Definition:** $z_0$ is a **removable singularity** of $f$ if $\lim_{z \to z_0} f(z)$ exists and is finite.

**Example:** $f(z) = \frac{\sin z}{z}$ has removable singularity at $z = 0$

Laurent series:
$$\frac{\sin z}{z} = \frac{1}{z}\left(z - \frac{z^3}{3!} + \frac{z^5}{5!} - \cdots\right) = 1 - \frac{z^2}{3!} + \frac{z^4}{5!} - \cdots$$

No negative powers! The singularity is removable by defining $f(0) = 1$.

**Riemann's Theorem on Removable Singularities:**

$$\boxed{\text{If } f \text{ is bounded near } z_0, \text{ then } z_0 \text{ is removable.}}$$

**Proof Sketch:**
1. If $|f(z)| \leq M$ for $0 < |z - z_0| < r$, consider the Laurent coefficients
2. Using $a_{-n} = \frac{1}{2\pi i}\oint_{|z-z_0|=\rho} \frac{f(z)}{(z-z_0)^{-n+1}} dz$
3. By ML inequality: $|a_{-n}| \leq M\rho^{n-1}$
4. As $\rho \to 0$: $a_{-n} \to 0$ for all $n \geq 1$

**Corollary:** If $\lim_{z \to z_0} (z - z_0)f(z) = 0$, then $z_0$ is removable.

### 4. Poles

**Definition:** $z_0$ is a **pole of order $m$** if the Laurent series has the form:

$$\boxed{f(z) = \frac{a_{-m}}{(z-z_0)^m} + \frac{a_{-m+1}}{(z-z_0)^{m-1}} + \cdots + \frac{a_{-1}}{z-z_0} + a_0 + a_1(z-z_0) + \cdots}$$

with $a_{-m} \neq 0$.

**Equivalent Characterizations:**

1. $\lim_{z \to z_0} |f(z)| = \infty$

2. $(z - z_0)^m f(z)$ has a removable singularity at $z_0$

3. $f(z) = \frac{g(z)}{(z-z_0)^m}$ where $g$ is analytic at $z_0$ with $g(z_0) \neq 0$

4. $\frac{1}{f(z)}$ has a zero of order $m$ at $z_0$

**Example 1:** $f(z) = \frac{1}{z^3}$ has pole of order 3 at $z = 0$

**Example 2:** $f(z) = \frac{1}{\sin z}$ has simple pole at $z = 0$

Since $\sin z = z - \frac{z^3}{6} + \cdots = z(1 - \frac{z^2}{6} + \cdots)$:
$$\frac{1}{\sin z} = \frac{1}{z} \cdot \frac{1}{1 - z^2/6 + \cdots} = \frac{1}{z} + \frac{z}{6} + \cdots$$

**Example 3:** $f(z) = \frac{z^2 + 1}{(z-1)^2(z+2)}$

- Double pole (order 2) at $z = 1$
- Simple pole at $z = -2$

### 5. Essential Singularities

**Definition:** $z_0$ is an **essential singularity** if it is neither removable nor a pole.

**Key feature:** The Laurent series has infinitely many negative power terms.

**Example 1:** $f(z) = e^{1/z}$ at $z = 0$

$$e^{1/z} = 1 + \frac{1}{z} + \frac{1}{2!z^2} + \frac{1}{3!z^3} + \cdots = \sum_{n=0}^{\infty} \frac{1}{n! z^n}$$

Infinitely many negative powers $\Rightarrow$ essential singularity.

**Example 2:** $f(z) = \sin(1/z)$ at $z = 0$

$$\sin(1/z) = \frac{1}{z} - \frac{1}{3!z^3} + \frac{1}{5!z^5} - \cdots$$

**Example 3:** $f(z) = e^{1/z^2}\cos(1/z)$ at $z = 0$

### 6. The Casorati-Weierstrass Theorem

**Theorem:** If $z_0$ is an essential singularity of $f$, then for any $w \in \mathbb{C}$ and any $\varepsilon > 0$, there exists $z$ arbitrarily close to $z_0$ with $|f(z) - w| < \varepsilon$.

$$\boxed{\text{Near an essential singularity, } f \text{ comes arbitrarily close to every complex value.}}$$

**Proof:**
Suppose the theorem fails for some $w_0$. Then $|f(z) - w_0| \geq \delta > 0$ in some punctured neighborhood.

Define $g(z) = \frac{1}{f(z) - w_0}$. Then $g$ is bounded, so $z_0$ is removable or a pole for $g$.

Case 1: $g(z_0) \neq 0$ after removal $\Rightarrow$ $f(z) \to w_0 + 1/g(z_0)$, removable for $f$.

Case 2: $g$ has zero at $z_0$ $\Rightarrow$ $f$ has pole at $z_0$.

Either contradicts $z_0$ being essential. $\blacksquare$

**Picard's Great Theorem (Stronger):** Near an essential singularity, $f$ takes every value infinitely often, with at most one exception.

**Example:** $e^{1/z}$ takes every value except 0 infinitely often near $z = 0$.

### 7. Summary Table

| Property | Removable | Pole (order $m$) | Essential |
|----------|-----------|------------------|-----------|
| $\lim_{z \to z_0} f(z)$ | Exists, finite | $\infty$ | Does not exist |
| $\lim_{z \to z_0} (z-z_0)^m f(z)$ | 0 for $m \geq 1$ | Finite nonzero | Never settles |
| Laurent principal part | None | $m$ terms | $\infty$ terms |
| Bounded near $z_0$? | Yes | No | No |
| Range near $z_0$ | Limited | Misses $\infty$ | Dense in $\mathbb{C}$ |

---

## Quantum Mechanics Connection

### Poles in Scattering Amplitudes

In quantum scattering theory, the scattering amplitude $f(k)$ and S-matrix $S(k)$ are functions of the complex momentum $k$.

**Bound States as Poles:**

For a square well potential, bound state energies appear as **simple poles** of the S-matrix on the positive imaginary axis:

$$S(k) = e^{2i\delta(k)} = \frac{k - i\kappa_n}{k + i\kappa_n}$$

The pole at $k = i\kappa_n$ corresponds to binding energy $E_n = -\hbar^2\kappa_n^2/2m$.

**Residue = Wave Function Normalization:**
The residue at a bound state pole is related to the asymptotic normalization of the bound state wave function.

### Resonances as Complex Poles

**Resonances** appear as poles in the **lower half of the complex $k$-plane**:

$$k_{\text{res}} = k_R - i\frac{\Gamma}{2\hbar v}$$

where:
- $k_R$ = resonance momentum (energy $E_R = \hbar^2 k_R^2/2m$)
- $\Gamma$ = resonance width (inverse lifetime)

Near a resonance, the scattering amplitude has a **Breit-Wigner form**:

$$\boxed{f(E) \approx \frac{\Gamma/2}{E - E_R + i\Gamma/2}}$$

This is a **simple pole** in the complex energy plane!

### Green's Function Poles

The resolvent operator has poles at eigenvalues:

$$G(E) = \frac{1}{E - H} = \sum_n \frac{|n\rangle\langle n|}{E - E_n}$$

Each **simple pole** corresponds to an eigenstate. The residue is the **projection operator**.

### Essential Singularities in QM

The partition function $Z(\beta)$ can have essential singularities at phase transitions. In field theory, instantons and non-perturbative effects create essential singularities in coupling constant $g$.

---

## Worked Examples

### Example 1: Classify All Singularities

**Problem:** Classify all singularities of $f(z) = \frac{z^2 - 1}{z^2(z-1)}$.

**Solution:**

Factor: $f(z) = \frac{(z-1)(z+1)}{z^2(z-1)} = \frac{z+1}{z^2}$ for $z \neq 1$

**At $z = 0$:** $f(z) = \frac{z+1}{z^2} = \frac{1}{z^2} + \frac{1}{z}$

This is a **pole of order 2**.

**At $z = 1$:** The factor $(z-1)$ cancels!

$\lim_{z \to 1} f(z) = \lim_{z \to 1} \frac{z+1}{z^2} = \frac{2}{1} = 2$

This is a **removable singularity**.

### Example 2: Essential Singularity Behavior

**Problem:** Show that $e^{1/z}$ comes arbitrarily close to any nonzero complex number near $z = 0$.

**Solution:**

Want to find $z$ near 0 with $|e^{1/z} - w| < \varepsilon$ for any $w \neq 0$.

Let $w = |w|e^{i\phi}$. We need $e^{1/z} \approx w$, so $\frac{1}{z} \approx \ln|w| + i(\phi + 2\pi n)$.

For large $n$:
$$z_n = \frac{1}{\ln|w| + i(\phi + 2\pi n)}$$

As $n \to \infty$: $|z_n| \approx \frac{1}{2\pi n} \to 0$

And $e^{1/z_n} = e^{\ln|w| + i(\phi + 2\pi n)} = |w|e^{i\phi} = w$.

So $f(z)$ equals $w$ at infinitely many points accumulating at $z = 0$! $\checkmark$

### Example 3: Determining Pole Order

**Problem:** Find the order of the pole of $f(z) = \frac{1 - \cos z}{z^4}$ at $z = 0$.

**Solution:**

Expand $\cos z$:
$$1 - \cos z = 1 - \left(1 - \frac{z^2}{2!} + \frac{z^4}{4!} - \cdots\right) = \frac{z^2}{2} - \frac{z^4}{24} + \cdots$$

$$f(z) = \frac{1}{z^4}\left(\frac{z^2}{2} - \frac{z^4}{24} + \cdots\right) = \frac{1}{2z^2} - \frac{1}{24} + \cdots$$

The lowest power is $z^{-2}$, so $z = 0$ is a **pole of order 2**.

### Example 4: Singularity at Infinity

**Problem:** Classify the singularity of $f(z) = z^3 + z$ at $z = \infty$.

**Solution:**

To study $z = \infty$, substitute $w = 1/z$ and examine $w = 0$:

$$g(w) = f(1/w) = \frac{1}{w^3} + \frac{1}{w}$$

At $w = 0$: this has a **pole of order 3**.

Therefore, $f(z)$ has a **pole of order 3 at infinity**.

---

## Practice Problems

### Problem Set A: Classification

**A1.** Classify the singularity of each function at the given point:
(a) $\frac{z}{e^z - 1}$ at $z = 0$
(b) $\frac{z^3}{(z-1)^2}$ at $z = 1$
(c) $z^2 e^{1/z}$ at $z = 0$
(d) $\frac{\sin z - z}{z^3}$ at $z = 0$

**A2.** Find all singularities and classify each:
(a) $\frac{1}{z^4 + 1}$
(b) $\tan z$
(c) $\frac{e^z}{z(z-1)^3}$

**A3.** Show that $\frac{1}{e^z - 1}$ has simple poles at $z = 2\pi in$ for all integers $n$.

### Problem Set B: Pole Orders

**B1.** Find the order of the pole at the indicated point:
(a) $\frac{\sin z}{z^5}$ at $z = 0$
(b) $\frac{1}{(z^2+1)^2}$ at $z = i$
(c) $\cot z$ at $z = 0$

**B2.** If $f$ has a zero of order $n$ at $z_0$ and $g$ has a zero of order $m$ at $z_0$, what is the singularity of $f/g$ at $z_0$?

### Problem Set C: Essential Singularities

**C1.** Show that $\cos(1/z)$ has essential singularity at $z = 0$ by writing its Laurent series.

**C2.** Using Casorati-Weierstrass, explain why $\sin(1/z)$ takes real values arbitrarily close to $z = 0$.

**C3.** Show that $e^z + e^{1/z}$ has essential singularities at both $z = 0$ and $z = \infty$.

---

## Computational Lab

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def classify_singularity(f, z0, epsilon=1e-6, n_samples=1000):
    """
    Numerically investigate singularity type at z0.

    Returns evidence for classification based on behavior.
    """
    # Sample points approaching z0 from various directions
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    radii = np.logspace(-6, -2, n_samples)

    max_vals = []
    min_vals = []

    for r in radii:
        vals = []
        for theta in angles:
            z = z0 + r * np.exp(1j * theta)
            try:
                fz = f(z)
                if np.isfinite(fz):
                    vals.append(np.abs(fz))
            except:
                pass

        if vals:
            max_vals.append(max(vals))
            min_vals.append(min(vals))

    max_vals = np.array(max_vals)
    min_vals = np.array(min_vals)

    # Analysis
    results = {
        'max_values': max_vals,
        'min_values': min_vals,
        'radii': radii[:len(max_vals)]
    }

    # Classification heuristics
    if max_vals[-1] < 1e6:  # Bounded
        results['type'] = 'Removable'
    elif min_vals[-1] > 1e3:  # All large
        results['type'] = 'Pole'
        # Estimate order
        log_r = np.log(radii[:len(max_vals)])
        log_f = np.log(max_vals)
        slope = np.polyfit(log_r[-100:], log_f[-100:], 1)[0]
        results['estimated_order'] = int(round(-slope))
    else:  # Oscillating
        results['type'] = 'Essential'

    return results


def visualize_singularity(f, z0, r_max=1, n_points=500, title=""):
    """
    Visualize |f(z)| near a singularity.
    """
    # Create grid around z0
    x = np.linspace(z0.real - r_max, z0.real + r_max, n_points)
    y = np.linspace(z0.imag - r_max, z0.imag + r_max, n_points)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    # Compute |f(z)|, avoiding the singularity
    F = np.zeros_like(Z, dtype=float)
    for i in range(n_points):
        for j in range(n_points):
            z = Z[i, j]
            if abs(z - z0) > 1e-6:
                try:
                    F[i, j] = min(np.abs(f(z)), 10)  # Cap for visualization
                except:
                    F[i, j] = np.nan
            else:
                F[i, j] = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Contour plot
    levels = np.linspace(0, 10, 50)
    cs = axes[0].contourf(X, Y, F, levels=levels, cmap='viridis')
    axes[0].set_xlabel('Re(z)')
    axes[0].set_ylabel('Im(z)')
    axes[0].set_title(f'{title}\n|f(z)| contour plot')
    plt.colorbar(cs, ax=axes[0], label='|f(z)|')
    axes[0].plot(z0.real, z0.imag, 'r*', markersize=15, label='Singularity')
    axes[0].legend()

    # Phase plot
    phase = np.angle(np.array([[f(z) if abs(z-z0) > 1e-6 else np.nan
                                 for z in row] for row in Z]))
    axes[1].contourf(X, Y, phase, levels=50, cmap='hsv')
    axes[1].set_xlabel('Re(z)')
    axes[1].set_ylabel('Im(z)')
    axes[1].set_title(f'{title}\nPhase of f(z)')
    axes[1].plot(z0.real, z0.imag, 'w*', markersize=15)

    plt.tight_layout()
    return fig


def demonstrate_casorati_weierstrass(f, z0, target_values, n_points=10000):
    """
    Demonstrate Casorati-Weierstrass by finding z near z0 with f(z) close to targets.
    """
    print("Casorati-Weierstrass Demonstration")
    print("-" * 50)
    print(f"Finding z near {z0} such that f(z) approximates target values:")

    # Sample many points near z0
    radii = np.logspace(-8, -1, 100)
    angles = np.linspace(0, 2*np.pi, n_points)

    for target in target_values:
        best_z = None
        best_dist = np.inf

        for r in radii:
            for theta in angles:
                z = z0 + r * np.exp(1j * theta)
                try:
                    fz = f(z)
                    dist = abs(fz - target)
                    if dist < best_dist:
                        best_dist = dist
                        best_z = z
                except:
                    pass

        print(f"\n  Target w = {target}")
        print(f"  Found z = {best_z:.8f}")
        print(f"  Distance |z - z0| = {abs(best_z - z0):.2e}")
        print(f"  f(z) = {f(best_z):.6f}")
        print(f"  |f(z) - w| = {best_dist:.2e}")


# Example demonstrations
if __name__ == "__main__":

    print("=" * 60)
    print("SINGULARITY CLASSIFICATION DEMONSTRATIONS")
    print("=" * 60)

    # Example 1: Removable singularity
    print("\n1. REMOVABLE SINGULARITY: sin(z)/z at z=0")
    f1 = lambda z: np.sin(z)/z if abs(z) > 1e-10 else 1
    result1 = classify_singularity(f1, 0)
    print(f"   Classification: {result1['type']}")

    # Example 2: Simple pole
    print("\n2. SIMPLE POLE: 1/z at z=0")
    f2 = lambda z: 1/z
    result2 = classify_singularity(f2, 0)
    print(f"   Classification: {result2['type']}")
    if 'estimated_order' in result2:
        print(f"   Estimated order: {result2['estimated_order']}")

    # Example 3: Double pole
    print("\n3. DOUBLE POLE: 1/z^2 at z=0")
    f3 = lambda z: 1/z**2
    result3 = classify_singularity(f3, 0)
    print(f"   Classification: {result3['type']}")
    if 'estimated_order' in result3:
        print(f"   Estimated order: {result3['estimated_order']}")

    # Example 4: Essential singularity
    print("\n4. ESSENTIAL SINGULARITY: exp(1/z) at z=0")
    f4 = lambda z: np.exp(1/z)
    result4 = classify_singularity(f4, 0)
    print(f"   Classification: {result4['type']}")

    # Visualizations
    fig1 = visualize_singularity(f2, 0, r_max=2, title="Simple Pole: 1/z")
    plt.savefig('singularity_pole.png', dpi=150, bbox_inches='tight')

    fig2 = visualize_singularity(f4, 0, r_max=0.5, title="Essential: exp(1/z)")
    plt.savefig('singularity_essential.png', dpi=150, bbox_inches='tight')

    # Casorati-Weierstrass demonstration
    print("\n" + "=" * 60)
    demonstrate_casorati_weierstrass(
        f4, 0,
        target_values=[1+0j, -1+0j, 1j, 2+3j, 100+0j]
    )

    plt.show()


# Additional visualization: comparison of singularity types
def compare_singularities():
    """Compare behavior near different singularity types."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    r_vals = np.linspace(0.01, 1, 100)
    theta = np.pi/4  # Approach from one direction

    # Functions and their singularities at z=0
    functions = [
        (lambda z: np.sin(z)/z, "sin(z)/z\n(Removable)"),
        (lambda z: 1/z, "1/z\n(Simple Pole)"),
        (lambda z: 1/z**3, "1/z³\n(Pole order 3)"),
        (lambda z: np.exp(1/z), "exp(1/z)\n(Essential)"),
        (lambda z: np.sin(1/z), "sin(1/z)\n(Essential)"),
        (lambda z: z**2 * np.exp(1/z), "z²exp(1/z)\n(Essential)")
    ]

    for idx, (f, title) in enumerate(functions):
        ax = axes[idx // 3, idx % 3]

        # Compute |f| along approach path
        z_vals = r_vals * np.exp(1j * theta)
        f_vals = np.array([abs(f(z)) for z in z_vals])

        ax.semilogy(r_vals, f_vals, 'b-', linewidth=2)
        ax.set_xlabel('|z|')
        ax.set_ylabel('|f(z)|')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)

    plt.suptitle('Behavior of |f(z)| approaching z=0', fontsize=14)
    plt.tight_layout()
    plt.savefig('singularity_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


compare_singularities()
```

---

## Summary

### Key Formulas

| Formula | Description |
|---------|-------------|
| Removable: $\lim_{z \to z_0} f(z)$ exists | No principal part |
| Pole order $m$: $\lim_{z \to z_0} (z-z_0)^m f(z) \neq 0$ | $m$ negative powers |
| Essential: infinitely many $a_{-n} \neq 0$ | Infinite principal part |
| Riemann: bounded $\Rightarrow$ removable | Characterization theorem |
| Casorati-Weierstrass: dense range | Essential singularity property |

### Main Takeaways

1. **Singularity type** is determined by the Laurent series principal part.

2. **Removable singularities** can be "filled in" — the function extends analytically.

3. **Poles** have controlled blow-up; $(z-z_0)^m f(z)$ becomes analytic.

4. **Essential singularities** have wild behavior — Casorati-Weierstrass and Picard's theorems.

5. **In QM**, poles in the S-matrix correspond to bound states and resonances.

6. **Residues** (tomorrow's topic) are computed differently for each singularity type.

---

## Daily Checklist

- [ ] I can classify isolated singularities as removable, pole, or essential
- [ ] I understand Riemann's theorem on removable singularities
- [ ] I can determine the order of a pole
- [ ] I understand the Casorati-Weierstrass theorem
- [ ] I can connect singularity types to Laurent series structure
- [ ] I see how poles appear in quantum scattering theory

---

## Preview: Day 185

Tomorrow we develop **residue computation techniques**:
- Simple pole formula: $\text{Res} = \lim_{z \to z_0} (z - z_0)f(z)$
- Higher order poles: derivative formula
- L'Hopital technique for rational functions
- Residue at infinity

These techniques are essential for applying the residue theorem to evaluate integrals.

---

*"The theory of functions of a complex variable is one of the most beautiful branches of mathematics."*
— Henri Poincare
