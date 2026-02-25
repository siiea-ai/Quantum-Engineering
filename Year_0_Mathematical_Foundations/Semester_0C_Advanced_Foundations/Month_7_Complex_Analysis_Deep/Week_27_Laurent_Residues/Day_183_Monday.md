# Day 183: Laurent Series

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Theory: Laurent Series Expansion |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Computation & Applications |
| Evening | 7:00 PM - 9:00 PM | 2 hours | Computational Lab |

**Total Study Time: 8 hours**

---

## Learning Objectives

By the end of Day 183, you will be able to:

1. Define and construct Laurent series expansions
2. Understand the annulus of convergence
3. Compute Laurent series using various techniques
4. Identify the principal part and analytic part
5. Relate Laurent series to singularity classification
6. Connect to quantum mechanical expansions

---

## Core Content

### 1. Motivation: Beyond Taylor Series

Taylor series work for analytic functions:
$$f(z) = \sum_{n=0}^{\infty} a_n (z - z_0)^n$$

But what about functions with singularities like $\frac{1}{z}$ or $\frac{e^z}{z^2}$?

We need **negative powers** — this gives the **Laurent series**.

### 2. Laurent Series Definition

**Theorem (Laurent Series):**
Let $f$ be analytic in an annulus $r < |z - z_0| < R$ (where $0 \leq r < R \leq \infty$). Then $f$ has a unique expansion:

$$\boxed{f(z) = \sum_{n=-\infty}^{\infty} a_n (z - z_0)^n = \sum_{n=0}^{\infty} a_n (z-z_0)^n + \sum_{n=1}^{\infty} a_{-n} (z-z_0)^{-n}}$$

with coefficients:
$$\boxed{a_n = \frac{1}{2\pi i} \oint_C \frac{f(z)}{(z - z_0)^{n+1}} \, dz}$$

where $C$ is any circle $|z - z_0| = \rho$ with $r < \rho < R$.

### 3. Parts of the Laurent Series

**Analytic Part (Regular Part):**
$$\sum_{n=0}^{\infty} a_n (z - z_0)^n$$

This converges for $|z - z_0| < R$.

**Principal Part:**
$$\sum_{n=1}^{\infty} a_{-n} (z - z_0)^{-n} = \frac{a_{-1}}{z-z_0} + \frac{a_{-2}}{(z-z_0)^2} + \cdots$$

This converges for $|z - z_0| > r$.

**Key Observation:** The coefficient $a_{-1}$ is the **residue** of $f$ at $z_0$.

### 4. Computing Laurent Series

#### Method 1: Direct Coefficient Formula

Use $a_n = \frac{1}{2\pi i}\oint \frac{f(z)}{(z-z_0)^{n+1}} dz$.

This is often impractical but theoretically important.

#### Method 2: Algebraic Manipulation

**Example:** Find Laurent series of $\frac{1}{z(z-1)}$ around $z_0 = 0$.

**For $0 < |z| < 1$:**
$$\frac{1}{z(z-1)} = \frac{-1}{z} \cdot \frac{1}{1-z} = \frac{-1}{z}(1 + z + z^2 + \cdots)$$
$$= -\frac{1}{z} - 1 - z - z^2 - \cdots$$

**For $|z| > 1$:**
$$\frac{1}{z-1} = \frac{1}{z}\cdot\frac{1}{1-1/z} = \frac{1}{z}(1 + \frac{1}{z} + \frac{1}{z^2} + \cdots)$$
$$\frac{1}{z(z-1)} = \frac{1}{z^2} + \frac{1}{z^3} + \frac{1}{z^4} + \cdots$$

Different Laurent series in different annuli!

#### Method 3: Known Expansions

**Example:** Laurent series of $\frac{e^z}{z^2}$ around $z = 0$.

Since $e^z = 1 + z + \frac{z^2}{2!} + \frac{z^3}{3!} + \cdots$:
$$\frac{e^z}{z^2} = \frac{1}{z^2} + \frac{1}{z} + \frac{1}{2!} + \frac{z}{3!} + \frac{z^2}{4!} + \cdots$$

The residue (coefficient of $1/z$) is $a_{-1} = 1$.

#### Method 4: Partial Fractions

**Example:** $f(z) = \frac{1}{(z-1)(z-2)}$

$$= \frac{-1}{z-1} + \frac{1}{z-2}$$

**For $|z| < 1$:** Expand both around $z = 0$:
$$\frac{-1}{z-1} = \frac{1}{1-z} = 1 + z + z^2 + \cdots$$
$$\frac{1}{z-2} = \frac{-1/2}{1 - z/2} = -\frac{1}{2}(1 + \frac{z}{2} + \frac{z^2}{4} + \cdots)$$

Add them to get the Laurent series (actually just Taylor, no negative powers here).

### 5. Uniqueness of Laurent Series

**Theorem:** The Laurent series of $f$ in a given annulus is unique.

**Proof:** If $\sum a_n(z-z_0)^n = \sum b_n(z-z_0)^n$, multiply both sides by $(z-z_0)^{-m-1}$ and integrate around a circle. The orthogonality of powers gives $a_m = b_m$.

### 6. Annulus of Convergence

For a Laurent series $\sum_{n=-\infty}^{\infty} a_n(z-z_0)^n$:

- The positive powers converge for $|z - z_0| < R$ (usual Taylor radius)
- The negative powers converge for $|z - z_0| > r$
- The full series converges in the annulus $r < |z - z_0| < R$

**Example:** $\sum_{n=-\infty}^{\infty} \frac{z^n}{2^{|n|}}$

Positive: $\sum_{n=0}^{\infty} \frac{z^n}{2^n}$ converges for $|z| < 2$
Negative: $\sum_{n=1}^{\infty} \frac{2^n}{z^n}$ converges for $|z| > 1/2$

Full series converges in $1/2 < |z| < 2$.

---

## Quantum Mechanics Connection

### Perturbation Theory

In quantum mechanics, we often expand observables in powers of a parameter:
$$E(\lambda) = E_0 + \lambda E_1 + \lambda^2 E_2 + \cdots$$

But perturbation series can be **asymptotic** (divergent yet useful) — related to essential singularities at $\lambda = 0$.

### Green's Function Expansion

The resolvent/Green's function:
$$G(E) = \frac{1}{E - H}$$

Near an eigenvalue $E_n$ has a **Laurent expansion**:
$$G(E) = \frac{|n\rangle\langle n|}{E - E_n} + \text{(analytic part)}$$

The residue is the **projection operator** onto the eigenstate!

### S-Matrix Poles

In scattering theory, the S-matrix $S(k)$ has poles in the complex $k$-plane:
- Bound states: poles on positive imaginary axis
- Resonances: poles in lower half-plane

The Laurent expansion near a pole gives the resonance width and position.

---

## Worked Examples

### Example 1: Complete Laurent Series

**Problem:** Find all Laurent series of $f(z) = \frac{z}{(z-1)(z-2)}$ around $z_0 = 0$.

**Solution:**

Partial fractions: $\frac{z}{(z-1)(z-2)} = \frac{-1}{z-1} + \frac{2}{z-2}$

**Region I: $|z| < 1$** (Taylor series)
$$\frac{-1}{z-1} = \frac{1}{1-z} = \sum_{n=0}^{\infty} z^n$$
$$\frac{2}{z-2} = \frac{-1}{1-z/2} = -\sum_{n=0}^{\infty} \frac{z^n}{2^n}$$

$$f(z) = \sum_{n=0}^{\infty}\left(1 - \frac{1}{2^n}\right)z^n$$

**Region II: $1 < |z| < 2$** (True Laurent series)
$$\frac{-1}{z-1} = \frac{-1}{z}\cdot\frac{1}{1-1/z} = -\sum_{n=0}^{\infty} \frac{1}{z^{n+1}} = -\sum_{n=1}^{\infty} z^{-n}$$
$$\frac{2}{z-2} = -\sum_{n=0}^{\infty} \frac{z^n}{2^n}$$

$$f(z) = -\sum_{n=1}^{\infty} z^{-n} - \sum_{n=0}^{\infty} \frac{z^n}{2^n}$$

**Region III: $|z| > 2$**
$$\frac{2}{z-2} = \frac{2}{z}\cdot\frac{1}{1-2/z} = \sum_{n=0}^{\infty} \frac{2^{n+1}}{z^{n+1}} = \sum_{n=1}^{\infty} \frac{2^n}{z^n}$$

$$f(z) = \sum_{n=1}^{\infty} \frac{2^n - 1}{z^n}$$

### Example 2: Essential Singularity

**Problem:** Find the Laurent series of $f(z) = e^{1/z}$ around $z = 0$.

**Solution:**

Using $e^w = \sum_{n=0}^{\infty} \frac{w^n}{n!}$ with $w = 1/z$:

$$e^{1/z} = \sum_{n=0}^{\infty} \frac{1}{n! z^n} = 1 + \frac{1}{z} + \frac{1}{2!z^2} + \frac{1}{3!z^3} + \cdots$$

This has **infinitely many negative powers** — signature of an **essential singularity**.

The residue is $a_{-1} = 1$.

### Example 3: Finding the Residue

**Problem:** Find the residue of $f(z) = \frac{\sin z}{z^4}$ at $z = 0$.

**Solution:**

$$\sin z = z - \frac{z^3}{3!} + \frac{z^5}{5!} - \cdots$$

$$\frac{\sin z}{z^4} = \frac{1}{z^3} - \frac{1}{6z} + \frac{z}{120} - \cdots$$

The residue is the coefficient of $1/z$: $\boxed{a_{-1} = -\frac{1}{6}}$

---

## Practice Problems

### Problem Set A: Basic Expansions

**A1.** Find the Laurent series of $\frac{1}{z^2(z-3)}$ valid for:
(a) $0 < |z| < 3$
(b) $|z| > 3$

**A2.** Expand $\frac{e^z - 1}{z^3}$ in Laurent series around $z = 0$.

**A3.** Find the Laurent series of $\cot z = \frac{\cos z}{\sin z}$ around $z = 0$, up to the $z^3$ term.

### Problem Set B: Multiple Annuli

**B1.** Find all Laurent series of $\frac{1}{z(z-1)(z-2)}$ centered at $z = 0$.

**B2.** For $f(z) = \frac{1}{z^2 - 1}$, find Laurent series valid in $|z| < 1$ and $|z| > 1$.

### Problem Set C: Residue Identification

**C1.** Find the residue of $\frac{z^2 + 1}{z(z-1)^2}$ at each singularity.

**C2.** Find the residue of $\frac{e^{1/z}}{z-1}$ at $z = 0$ and $z = 1$.

---

## Computational Lab

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial

def laurent_coefficients(f, z0, n_terms=10, radius=1.0, n_points=10000):
    """
    Compute Laurent series coefficients using contour integration.

    a_n = (1/2πi) ∮ f(z)/(z-z0)^{n+1} dz
    """
    theta = np.linspace(0, 2*np.pi, n_points)
    z = z0 + radius * np.exp(1j * theta)
    dz = 1j * radius * np.exp(1j * theta)

    coeffs = {}
    for n in range(-n_terms, n_terms + 1):
        integrand = f(z) / (z - z0)**(n + 1) * dz
        coeffs[n] = np.trapz(integrand, theta) / (2 * np.pi * 1j)

    return coeffs

def reconstruct_from_laurent(coeffs, z, z0):
    """Reconstruct function from Laurent coefficients."""
    result = 0
    for n, a_n in coeffs.items():
        result += a_n * (z - z0)**n
    return result

# Example: f(z) = e^z / z^2
def f(z):
    return np.exp(z) / z**2

# Compute coefficients
coeffs = laurent_coefficients(f, 0, n_terms=5, radius=0.5)

print("Laurent coefficients of e^z/z² around z=0:")
print("-" * 40)
for n in sorted(coeffs.keys()):
    if abs(coeffs[n]) > 1e-10:
        # Compare with theoretical: a_n = 1/(n+2)! for n >= -2
        if n >= -2:
            theoretical = 1 / factorial(n + 2)
        else:
            theoretical = 0
        print(f"a_{n:2d} = {coeffs[n].real:10.6f} (theory: {theoretical:10.6f})")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot original function and Laurent approximation
theta = np.linspace(0, 2*np.pi, 100)
r_values = [0.3, 0.5, 0.7]

for r in r_values:
    z = r * np.exp(1j * theta)
    f_exact = f(z)
    f_laurent = reconstruct_from_laurent(coeffs, z, 0)

    axes[0].plot(theta * 180/np.pi, np.abs(f_exact), '-',
                 label=f'Exact |f| at r={r}')
    axes[0].plot(theta * 180/np.pi, np.abs(f_laurent), '--',
                 label=f'Laurent at r={r}')

axes[0].set_xlabel('Angle (degrees)')
axes[0].set_ylabel('|f(z)|')
axes[0].set_title('Laurent Series Approximation')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Coefficient magnitude
n_vals = sorted(coeffs.keys())
coeff_mags = [np.abs(coeffs[n]) for n in n_vals]
axes[1].bar(n_vals, coeff_mags)
axes[1].set_xlabel('n')
axes[1].set_ylabel('|a_n|')
axes[1].set_title('Laurent Coefficient Magnitudes')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('laurent_series.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## Summary

### Key Formulas

| Formula | Description |
|---------|-------------|
| $f(z) = \sum_{n=-\infty}^{\infty} a_n(z-z_0)^n$ | Laurent series |
| $a_n = \frac{1}{2\pi i}\oint \frac{f(z)}{(z-z_0)^{n+1}} dz$ | Coefficient formula |
| $a_{-1} = \text{Res}_{z=z_0} f(z)$ | Residue identification |

### Main Takeaways

1. **Laurent series** extend Taylor series to include negative powers.

2. **Different annuli** require different Laurent expansions of the same function.

3. **The principal part** (negative powers) encodes singularity information.

4. **The residue** $a_{-1}$ is crucial for contour integration.

5. **In QM**, Laurent expansions appear in perturbation theory and Green's functions.

---

## Daily Checklist

- [ ] I can define and construct Laurent series
- [ ] I understand the annulus of convergence
- [ ] I can compute Laurent coefficients by various methods
- [ ] I can identify the residue from a Laurent series
- [ ] I see the connection to singularity classification

---

## Preview: Day 184

Tomorrow we classify singularities systematically:
- **Removable singularities:** No principal part
- **Poles:** Finite principal part
- **Essential singularities:** Infinite principal part

This classification is key to understanding function behavior and computing residues.

---

*"The language of mathematics reveals its unreasonable effectiveness in the natural sciences."*
— Eugene Wigner
