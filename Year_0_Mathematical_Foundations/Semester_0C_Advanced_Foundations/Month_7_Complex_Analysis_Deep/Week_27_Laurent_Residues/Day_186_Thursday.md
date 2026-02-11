# Day 186: The Residue Theorem

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Theory: Residue Theorem Proof |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Argument Principle & Rouche's Theorem |
| Evening | 7:00 PM - 9:00 PM | 2 hours | Computational Lab |

**Total Study Time: 8 hours**

---

## Learning Objectives

By the end of Day 186, you will be able to:

1. State and prove the Residue Theorem
2. Apply the residue theorem to compute contour integrals
3. Understand the relationship to Cauchy's integral theorem
4. Apply the Argument Principle to count zeros and poles
5. Use Rouche's theorem to locate zeros
6. Connect to counting bound states in quantum mechanics

---

## Core Content

### 1. The Residue Theorem: Statement

**Theorem (Cauchy's Residue Theorem):**

Let $f$ be analytic inside and on a simple closed contour $C$, except for isolated singularities $z_1, z_2, \ldots, z_n$ inside $C$. Then:

$$\boxed{\oint_C f(z)\, dz = 2\pi i \sum_{k=1}^{n} \text{Res}_{z=z_k} f(z)}$$

**Interpretation:** The contour integral equals $2\pi i$ times the sum of all enclosed residues.

### 2. Proof of the Residue Theorem

**Proof:**

**Step 1:** Surround each singularity $z_k$ with a small circle $C_k$ of radius $\varepsilon_k$, oriented counterclockwise.

**Step 2:** The region between $C$ and the small circles is simply connected with $f$ analytic there.

**Step 3:** By Cauchy's theorem for multiply connected regions:

$$\oint_C f(z)\, dz = \sum_{k=1}^{n} \oint_{C_k} f(z)\, dz$$

**Step 4:** For each small circle around $z_k$, write the Laurent series:

$$f(z) = \sum_{n=-\infty}^{\infty} a_n^{(k)}(z - z_k)^n$$

**Step 5:** Integrate term by term. Only the $n = -1$ term contributes:

$$\oint_{C_k} (z - z_k)^n \, dz = \begin{cases} 2\pi i & \text{if } n = -1 \\ 0 & \text{if } n \neq -1 \end{cases}$$

**Step 6:** Therefore:

$$\oint_{C_k} f(z)\, dz = 2\pi i \cdot a_{-1}^{(k)} = 2\pi i \cdot \text{Res}_{z=z_k} f(z)$$

**Step 7:** Summing over all singularities:

$$\oint_C f(z)\, dz = 2\pi i \sum_{k=1}^{n} \text{Res}_{z=z_k} f(z) \quad \blacksquare$$

### 3. Connection to Cauchy's Theorem

**Special Case:** If $f$ has no singularities inside $C$, the sum of residues is zero:

$$\oint_C f(z)\, dz = 2\pi i \cdot 0 = 0$$

This recovers **Cauchy's Theorem**!

**Special Case:** If $f$ has one simple pole at $z_0$ with $f(z) = \frac{g(z)}{z - z_0}$ where $g$ is analytic:

$$\oint_C f(z)\, dz = 2\pi i \cdot g(z_0)$$

This is **Cauchy's Integral Formula**!

The Residue Theorem **unifies** all our previous results.

### 4. Examples Using the Residue Theorem

**Example 1:** Evaluate $\oint_{|z|=2} \frac{e^z}{z(z-1)}\, dz$

**Singularities inside:** $z = 0$ and $z = 1$

**Residue at $z = 0$:**
$$\text{Res}_{z=0} = \lim_{z \to 0} z \cdot \frac{e^z}{z(z-1)} = \frac{e^0}{0-1} = -1$$

**Residue at $z = 1$:**
$$\text{Res}_{z=1} = \lim_{z \to 1}(z-1) \cdot \frac{e^z}{z(z-1)} = \frac{e}{1} = e$$

**By Residue Theorem:**
$$\oint_{|z|=2} \frac{e^z}{z(z-1)}\, dz = 2\pi i(-1 + e) = 2\pi i(e - 1)$$

**Example 2:** Evaluate $\oint_{|z|=3} \frac{z^2}{(z^2+1)(z^2+4)}\, dz$

**Singularities:** $z = \pm i$ (inside), $z = \pm 2i$ (inside)

**Residue at $z = i$:**
$$\frac{i^2}{(2i)(i^2+4)} = \frac{-1}{2i \cdot 3} = \frac{-1}{6i} = \frac{i}{6}$$

**Residue at $z = -i$:**
$$\frac{(-i)^2}{(-2i)((-i)^2+4)} = \frac{-1}{-2i \cdot 3} = \frac{-1}{-6i} = \frac{-i}{6}$$

**Residue at $z = 2i$:**
$$\frac{(2i)^2}{((2i)^2+1)(4i)} = \frac{-4}{(-3)(4i)} = \frac{-4}{-12i} = \frac{-i}{3}$$

**Residue at $z = -2i$:**
$$\frac{(-2i)^2}{((-2i)^2+1)(-4i)} = \frac{-4}{(-3)(-4i)} = \frac{-4}{12i} = \frac{i}{3}$$

**Sum of residues:** $\frac{i}{6} - \frac{i}{6} - \frac{i}{3} + \frac{i}{3} = 0$

**Result:** $\oint = 2\pi i \cdot 0 = 0$

### 5. The Argument Principle

**Theorem (Argument Principle):**

Let $f$ be meromorphic inside and on $C$ with zeros $z_1, \ldots, z_m$ (counted with multiplicity) and poles $p_1, \ldots, p_n$ (counted with multiplicity). Then:

$$\boxed{\frac{1}{2\pi i}\oint_C \frac{f'(z)}{f(z)}\, dz = N - P}$$

where $N = $ number of zeros, $P = $ number of poles inside $C$.

**Proof:**

Near a zero of order $m$: $f(z) = (z - z_0)^m g(z)$ with $g(z_0) \neq 0$

$$\frac{f'(z)}{f(z)} = \frac{m(z-z_0)^{m-1}g(z) + (z-z_0)^m g'(z)}{(z-z_0)^m g(z)} = \frac{m}{z-z_0} + \frac{g'(z)}{g(z)}$$

The residue at $z_0$ is $m$ (the multiplicity).

Similarly, near a pole of order $n$: residue is $-n$.

By the Residue Theorem:
$$\frac{1}{2\pi i}\oint_C \frac{f'(z)}{f(z)}\, dz = \sum(\text{zero multiplicities}) - \sum(\text{pole multiplicities}) = N - P \quad \blacksquare$$

**Geometric Interpretation:**

$$\frac{1}{2\pi i}\oint_C \frac{f'(z)}{f(z)}\, dz = \frac{1}{2\pi i}\oint_C d(\ln f) = \frac{\Delta \arg f(z)}{2\pi}$$

This counts how many times $f(z)$ winds around the origin as $z$ traverses $C$.

### 6. Rouche's Theorem

**Theorem (Rouche):**

If $f$ and $g$ are analytic inside and on $C$, and $|g(z)| < |f(z)|$ on $C$, then $f$ and $f + g$ have the same number of zeros inside $C$.

$$\boxed{|g(z)| < |f(z)| \text{ on } C \implies N_f = N_{f+g}}$$

**Proof:**

Consider $h(t) = f + tg$ for $t \in [0, 1]$. On $C$:

$$|h(t)| = |f + tg| \geq |f| - t|g| > |f| - |f| = 0$$

So $h(t)$ has no zeros on $C$ for any $t$.

The number of zeros $N(t) = \frac{1}{2\pi i}\oint_C \frac{h'(t)}{h(t)}dz$ is continuous in $t$.

Since $N(t)$ is integer-valued and continuous, it must be constant.

Therefore $N(0) = N_f = N(1) = N_{f+g}$ $\blacksquare$

### 7. Applications of Rouche's Theorem

**Example 3:** Show that $z^4 + 6z + 3 = 0$ has exactly one root in $|z| < 1$.

**In $|z| < 1$:** Let $f(z) = 6z$ and $g(z) = z^4 + 3$.

On $|z| = 1$: $|f(z)| = 6$ and $|g(z)| \leq |z|^4 + 3 = 1 + 3 = 4$

Since $|g| = 4 < 6 = |f|$ on $|z| = 1$, by Rouche:

$z^4 + 6z + 3$ has same number of zeros as $6z$ in $|z| < 1$.

$6z$ has exactly **one zero** (at $z = 0$).

Therefore $z^4 + 6z + 3$ has exactly **one root** in $|z| < 1$. $\checkmark$

**Example 4:** Show that all roots of $z^n + az + b = 0$ lie in $|z| \leq 1 + |a| + |b|$ for $n \geq 2$.

Let $R = 1 + |a| + |b|$. On $|z| = R$:

$|z^n| = R^n > R^2 = (1 + |a| + |b|)^2 > |a|R + |b| \geq |az + b|$

By Rouche with $f = z^n$, $g = az + b$: all $n$ roots are inside $|z| < R$.

### 8. Counting Zeros with the Argument Principle

**Example 5:** How many roots does $e^z = 3z$ have in $|z| < 1$?

Rewrite: $f(z) = e^z - 3z$. We want zeros of $f$ in $|z| < 1$.

$$N = \frac{1}{2\pi i}\oint_{|z|=1} \frac{e^z - 3}{e^z - 3z}\, dz$$

This integral is tricky. Use Rouche instead:

On $|z| = 1$: $|e^z| \leq e^1 \approx 2.718$ and $|3z| = 3$

Actually $|e^z| \leq e$ and $|3z| = 3$, so $|e^z| < |3z|$ on $|z| = 1$.

By Rouche: $e^z - 3z$ has same zeros as $-3z$, which has **one zero**.

---

## Quantum Mechanics Connection

### Counting Bound States

In quantum mechanics, the number of bound states for a potential can be found using the Argument Principle applied to the Jost function or S-matrix.

**Levinson's Theorem:**

For a spherically symmetric potential, the phase shift $\delta(k)$ satisfies:

$$\boxed{\delta(0) - \delta(\infty) = n_b \pi}$$

where $n_b$ is the number of bound states.

**Proof via Argument Principle:**

The S-matrix $S(k) = e^{2i\delta(k)}$ is meromorphic in the upper half $k$-plane.
- **Poles** on positive imaginary axis = bound states
- **Zeros** on negative imaginary axis = anti-bound states

Applying the argument principle to $S(k)$ around a large semicircle:

$$\frac{1}{2\pi i}\oint \frac{S'(k)}{S(k)}dk = N_{\text{zeros}} - N_{\text{poles}} = -n_b$$

This gives Levinson's theorem!

### Resonance Counting

Similarly, resonances (quasi-bound states) can be counted using:

$$\frac{1}{2\pi i}\oint \frac{S'(k)}{S(k)}dk = N_{\text{resonances in region}}$$

The contour encloses the region of the complex $k$-plane where resonances are located.

### Stability Analysis

In quantum field theory, the stability of a system is analyzed by counting poles of propagators. The number of unstable modes equals the number of poles in certain regions of the complex energy plane.

---

## Worked Examples

### Example 1: Complete Residue Calculation

**Problem:** Evaluate $\oint_{|z|=2} \frac{z^2 + 1}{z(z-1)^2}\, dz$.

**Solution:**

**Singularities:** $z = 0$ (simple pole), $z = 1$ (double pole)

**Residue at $z = 0$:**
$$\text{Res}_{z=0} = \lim_{z \to 0} z \cdot \frac{z^2+1}{z(z-1)^2} = \frac{0+1}{(0-1)^2} = 1$$

**Residue at $z = 1$ (order 2):**

Let $g(z) = (z-1)^2 \cdot \frac{z^2+1}{z(z-1)^2} = \frac{z^2+1}{z}$

$$\text{Res}_{z=1} = g'(1) = \frac{d}{dz}\left(\frac{z^2+1}{z}\right)\bigg|_{z=1}$$

$$= \frac{2z \cdot z - (z^2+1)}{z^2}\bigg|_{z=1} = \frac{2-2}{1} = 0$$

**By Residue Theorem:**
$$\oint_{|z|=2} \frac{z^2+1}{z(z-1)^2}\, dz = 2\pi i(1 + 0) = 2\pi i$$

### Example 2: Argument Principle Application

**Problem:** Find the number of zeros of $f(z) = z^3 - 2z + 4$ in the right half-plane $\text{Re}(z) > 0$.

**Solution:**

Use a semicircular contour: imaginary axis from $-iR$ to $iR$, plus semicircle in right half-plane.

**On imaginary axis** $z = iy$:
$f(iy) = -iy^3 - 2iy + 4 = 4 - i(y^3 + 2y)$

As $y$ goes from $-\infty$ to $\infty$, the imaginary part goes from $+\infty$ to $-\infty$, crossing zero once (at $y = 0$).

At $y = 0$: $f(0) = 4 > 0$

The argument changes by $-\pi$ (from $+\varepsilon$ to $-\varepsilon$ in imaginary part).

**On large semicircle** $z = Re^{i\theta}$, $\theta \in [-\pi/2, \pi/2]$:

$f(z) \approx z^3 = R^3 e^{3i\theta}$

Argument change: $3 \cdot \pi = 3\pi$

**Total argument change:** $-\pi + 3\pi = 2\pi$

**Number of zeros:** $\frac{2\pi}{2\pi} = 1$

There is **one zero** in the right half-plane.

### Example 3: Rouche's Theorem

**Problem:** Show that $p(z) = z^7 - 5z^3 + 12$ has exactly 3 zeros in $|z| < 1$.

**Solution:**

On $|z| = 1$, let's try $f(z) = 12$ and $g(z) = z^7 - 5z^3$:

$|f| = 12$ and $|g| \leq |z|^7 + 5|z|^3 = 1 + 5 = 6 < 12$

By Rouche, $p(z)$ has same number of zeros as $12$ in $|z| < 1$.

But $f(z) = 12$ has **no zeros**. This gives 0 zeros, which contradicts the problem!

**Try differently:** Let $f(z) = -5z^3 + 12$ and $g(z) = z^7$:

$|g| = |z|^7 = 1$

$|f| = |-5z^3 + 12| \geq |12| - |5z^3| = 12 - 5 = 7 > 1$

By Rouche, $p(z)$ has same zeros as $-5z^3 + 12$ in $|z| < 1$.

Zeros of $-5z^3 + 12 = 0$: $z^3 = 12/5$, giving $z = \sqrt[3]{12/5} \approx 1.34 > 1$.

So $-5z^3 + 12$ has **no zeros** in $|z| < 1$.

Wait, this also gives 0. Let me reconsider...

Actually, $|-5z^3 + 12|$ on $|z| = 1$: minimum is when $z^3$ is positive real.
At $z = 1$: $|-5 + 12| = 7$. At $z = e^{i\pi/3}$: $|-5e^{i\pi} + 12| = |5 + 12| = 17$.

The minimum is 7, not uniform. Let's check: on $|z| = 1$, $-5z^3$ traces a circle of radius 5 centered at 12.

Distance from origin: ranges from $12 - 5 = 7$ to $12 + 5 = 17$.

So indeed $|f| \geq 7 > 1 = |g|$, and $-5z^3 + 12$ has no zeros in $|z| \leq 1$.

**Conclusion:** $p(z)$ has **0 zeros** in $|z| < 1$.

---

## Practice Problems

### Problem Set A: Residue Theorem

**A1.** Evaluate:
(a) $\oint_{|z|=2} \frac{z}{(z-1)(z+3)}\, dz$
(b) $\oint_{|z|=1} \frac{e^z}{z^2(z-2)}\, dz$
(c) $\oint_{|z|=4} \frac{\sin z}{z^2 - \pi^2}\, dz$

**A2.** Show that $\oint_{|z|=R} \frac{dz}{z^n} = 0$ for $n \neq 1$ and $= 2\pi i$ for $n = 1$.

**A3.** Compute $\oint_{|z|=2} \frac{z^3}{(z-1)^2(z+1)}\, dz$.

### Problem Set B: Argument Principle

**B1.** Find the number of zeros of $z^4 - 5z + 1$ in $|z| < 1$.

**B2.** How many roots does $z^5 + z^3 + 1 = 0$ have in the first quadrant?

**B3.** Show that $e^z = z$ has exactly one solution in $|z| < 1$.

### Problem Set C: Rouche's Theorem

**C1.** Use Rouche's theorem to prove the Fundamental Theorem of Algebra.

**C2.** Show that $z^n - 5z + 1 = 0$ has exactly one root in $|z| < 1/2$ for $n \geq 2$.

**C3.** Find the number of roots of $z^8 + 3z^7 + 6z^2 + 1 = 0$ in $1 < |z| < 2$.

---

## Computational Lab

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def residue_theorem_verification(f, poles_residues, R=5, n_points=10000):
    """
    Verify the residue theorem numerically.

    Parameters:
    -----------
    f : function of z
    poles_residues : list of (pole, residue) tuples
    R : radius of contour
    """
    # Compute contour integral
    theta = np.linspace(0, 2*np.pi, n_points)
    z = R * np.exp(1j * theta)
    dz = 1j * R * np.exp(1j * theta)

    integrand = f(z) * dz
    integral = np.trapz(integrand, theta)

    # Sum of residues
    residue_sum = sum(res for pole, res in poles_residues if abs(pole) < R)

    print(f"Contour integral:     {integral:.6f}")
    print(f"2πi × Σ Residues:     {2*np.pi*1j * residue_sum:.6f}")
    print(f"Match: {np.isclose(integral, 2*np.pi*1j * residue_sum)}")

    return integral, 2*np.pi*1j * residue_sum


def argument_principle_count(f, f_prime, R=2, n_points=50000):
    """
    Count zeros minus poles using the Argument Principle.
    """
    theta = np.linspace(0, 2*np.pi, n_points)
    z = R * np.exp(1j * theta)
    dz = 1j * R * np.exp(1j * theta)

    integrand = f_prime(z) / f(z) * dz
    integral = np.trapz(integrand, theta)

    N_minus_P = integral / (2 * np.pi * 1j)

    return N_minus_P


def visualize_argument_principle(f, R=2, n_points=1000):
    """
    Visualize the argument principle by tracking f(z) as z traverses C.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    theta = np.linspace(0, 2*np.pi, n_points)
    z = R * np.exp(1j * theta)
    w = f(z)

    # Plot 1: Contour in z-plane
    axes[0].plot(z.real, z.imag, 'b-', linewidth=2)
    axes[0].plot([R], [0], 'go', markersize=10, label='Start')
    axes[0].set_xlabel('Re(z)')
    axes[0].set_ylabel('Im(z)')
    axes[0].set_title('Contour C in z-plane')
    axes[0].axis('equal')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Plot 2: Image f(C) in w-plane
    axes[1].plot(w.real, w.imag, 'r-', linewidth=2)
    axes[1].plot([0], [0], 'k*', markersize=15, label='Origin')
    axes[1].plot([w[0].real], [w[0].imag], 'go', markersize=10, label='Start')
    axes[1].set_xlabel('Re(w)')
    axes[1].set_ylabel('Im(w)')
    axes[1].set_title('Image f(C) in w-plane')
    axes[1].axis('equal')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Count windings
    angles = np.unwrap(np.angle(w))
    winding_number = (angles[-1] - angles[0]) / (2 * np.pi)
    plt.suptitle(f'Winding number around origin = {winding_number:.2f}', fontsize=14)

    plt.tight_layout()
    return fig, winding_number


def rouche_visualization(f, g, R=1, n_points=1000):
    """
    Visualize Rouche's theorem by comparing |f| and |g| on contour.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    theta = np.linspace(0, 2*np.pi, n_points)
    z = R * np.exp(1j * theta)

    f_vals = f(z)
    g_vals = g(z)
    fg_vals = f_vals + g_vals

    # Plot 1: |f| and |g| on contour
    axes[0].plot(theta * 180/np.pi, np.abs(f_vals), 'b-', linewidth=2, label='|f(z)|')
    axes[0].plot(theta * 180/np.pi, np.abs(g_vals), 'r-', linewidth=2, label='|g(z)|')
    axes[0].fill_between(theta * 180/np.pi, np.abs(f_vals), np.abs(g_vals),
                         where=np.abs(f_vals) > np.abs(g_vals),
                         alpha=0.3, color='green', label='|f| > |g|')
    axes[0].set_xlabel('Angle (degrees)')
    axes[0].set_ylabel('Magnitude')
    axes[0].set_title('Comparison of |f| and |g| on contour')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Check Rouche condition
    rouche_satisfied = np.all(np.abs(f_vals) > np.abs(g_vals))
    axes[0].text(0.5, 0.95, f'Rouche condition satisfied: {rouche_satisfied}',
                 transform=axes[0].transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='center')

    # Plot 2: Images of contour
    axes[1].plot(f_vals.real, f_vals.imag, 'b-', linewidth=2, label='f(C)')
    axes[1].plot(fg_vals.real, fg_vals.imag, 'g-', linewidth=2, label='(f+g)(C)')
    axes[1].plot([0], [0], 'k*', markersize=15, label='Origin')
    axes[1].set_xlabel('Re(w)')
    axes[1].set_ylabel('Im(w)')
    axes[1].set_title('Images in w-plane')
    axes[1].legend()
    axes[1].axis('equal')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def find_zeros_newton(f, f_prime, initial_guesses, tol=1e-10, max_iter=100):
    """
    Find zeros of f using Newton's method.
    """
    zeros = []
    for z0 in initial_guesses:
        z = z0
        for _ in range(max_iter):
            fz = f(z)
            fpz = f_prime(z)
            if abs(fpz) < 1e-15:
                break
            z_new = z - fz / fpz
            if abs(z_new - z) < tol:
                zeros.append(z_new)
                break
            z = z_new
    return zeros


# Demonstrations
if __name__ == "__main__":
    print("=" * 60)
    print("RESIDUE THEOREM DEMONSTRATIONS")
    print("=" * 60)

    # Example 1: Verify residue theorem
    print("\n1. RESIDUE THEOREM VERIFICATION")
    print("   f(z) = e^z / (z(z-1))")

    f1 = lambda z: np.exp(z) / (z * (z - 1))

    # Poles at z=0 (res=-1) and z=1 (res=e)
    poles_residues = [(0, -1), (1, np.e)]

    residue_theorem_verification(f1, poles_residues, R=2)

    # Example 2: Argument principle
    print("\n" + "=" * 60)
    print("2. ARGUMENT PRINCIPLE")
    print("   f(z) = z^3 - 1 (3 zeros at cube roots of unity)")

    f2 = lambda z: z**3 - 1
    f2_prime = lambda z: 3*z**2

    N_minus_P = argument_principle_count(f2, f2_prime, R=2)
    print(f"   N - P = {N_minus_P:.4f}")
    print(f"   Expected: 3 (three zeros, no poles)")

    fig2, winding = visualize_argument_principle(f2, R=2)
    plt.savefig('argument_principle.png', dpi=150, bbox_inches='tight')

    # Example 3: Rouche's theorem
    print("\n" + "=" * 60)
    print("3. ROUCHE'S THEOREM")
    print("   Showing z^5 + 3z + 1 has same zeros as 3z in |z| < 1")

    f3 = lambda z: 3*z
    g3 = lambda z: z**5 + 1

    fig3 = rouche_visualization(f3, g3, R=1)
    plt.savefig('rouche_theorem.png', dpi=150, bbox_inches='tight')

    # Example 4: Zero finding
    print("\n" + "=" * 60)
    print("4. ZERO FINDING: z^4 + 6z + 3 = 0")

    f4 = lambda z: z**4 + 6*z + 3
    f4_prime = lambda z: 4*z**3 + 6

    # Initial guesses in grid
    guesses = [x + 1j*y for x in np.linspace(-2, 2, 10)
                        for y in np.linspace(-2, 2, 10)]

    zeros = find_zeros_newton(f4, f4_prime, guesses)

    # Remove duplicates
    unique_zeros = []
    for z in zeros:
        is_duplicate = False
        for uz in unique_zeros:
            if abs(z - uz) < 1e-6:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_zeros.append(z)

    print("   Zeros found:")
    for z in unique_zeros:
        print(f"      z = {z:.6f}, |z| = {abs(z):.6f}, f(z) = {f4(z):.2e}")

    zeros_in_unit_disk = sum(1 for z in unique_zeros if abs(z) < 1)
    print(f"\n   Zeros in |z| < 1: {zeros_in_unit_disk}")

    # Visualize zeros
    fig4, ax4 = plt.subplots(figsize=(8, 8))

    theta = np.linspace(0, 2*np.pi, 100)
    ax4.plot(np.cos(theta), np.sin(theta), 'b--', linewidth=2, label='|z| = 1')

    for z in unique_zeros:
        ax4.plot(z.real, z.imag, 'ro', markersize=10)

    ax4.set_xlabel('Re(z)')
    ax4.set_ylabel('Im(z)')
    ax4.set_title('Zeros of $z^4 + 6z + 3 = 0$')
    ax4.axis('equal')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.savefig('zeros_polynomial.png', dpi=150, bbox_inches='tight')
    plt.show()


# Additional: Levinson's theorem visualization
def levinson_theorem_demo():
    """
    Demonstrate Levinson's theorem for a simple potential.
    """
    print("\n" + "=" * 60)
    print("5. LEVINSON'S THEOREM DEMONSTRATION")
    print("   Square well potential: phase shift at k=0 vs k=∞")

    # For a square well, the phase shift satisfies tan(δ) = ...
    # Simplified model: δ(k) = arctan(1/k) for demonstration

    k_values = np.linspace(0.01, 10, 1000)

    # Model phase shift with 2 bound states
    delta = np.arctan(1/k_values) + np.arctan(0.5/k_values)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(k_values, delta * 180 / np.pi, 'b-', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.axhline(y=180, color='r', linestyle='--', alpha=0.5, label='n_b × 180°')

    ax.set_xlabel('k (momentum)')
    ax.set_ylabel('Phase shift δ(k) (degrees)')
    ax.set_title("Levinson's Theorem: δ(0) - δ(∞) = n_b × π")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Annotate
    delta_0 = delta[0] * 180 / np.pi
    delta_inf = delta[-1] * 180 / np.pi
    n_bound = (delta_0 - delta_inf) / 180

    ax.annotate(f'δ(0) ≈ {delta_0:.1f}°', xy=(0.1, delta_0),
                fontsize=12, color='blue')
    ax.annotate(f'δ(∞) ≈ {delta_inf:.1f}°', xy=(8, delta_inf + 5),
                fontsize=12, color='blue')

    print(f"   δ(0) - δ(∞) = {delta_0 - delta_inf:.1f}° = {n_bound:.1f} × 180°")
    print(f"   Number of bound states: {int(round(n_bound))}")

    plt.savefig('levinson_theorem.png', dpi=150, bbox_inches='tight')
    plt.show()


levinson_theorem_demo()
```

---

## Summary

### Key Formulas

| Theorem | Formula |
|---------|---------|
| Residue Theorem | $\oint_C f(z)\,dz = 2\pi i \sum_k \text{Res}_{z=z_k} f(z)$ |
| Argument Principle | $\frac{1}{2\pi i}\oint_C \frac{f'}{f}dz = N - P$ |
| Rouche's Theorem | $|g| < |f|$ on $C \Rightarrow N_f = N_{f+g}$ |

### Main Takeaways

1. **The Residue Theorem** unifies Cauchy's theorem and integral formula.

2. **Computing contour integrals** reduces to summing residues.

3. **The Argument Principle** counts zeros minus poles by winding number.

4. **Rouche's Theorem** lets us count zeros by comparing functions.

5. **In QM**, these theorems count bound states and resonances.

---

## Daily Checklist

- [ ] I can state and prove the Residue Theorem
- [ ] I can apply the residue theorem to evaluate contour integrals
- [ ] I understand how Cauchy's theorems are special cases
- [ ] I can apply the Argument Principle
- [ ] I can use Rouche's theorem to count zeros
- [ ] I see the connection to counting bound states in QM

---

## Preview: Day 187

Tomorrow we explore **advanced applications of residues**:
- Summation of infinite series
- Mittag-Leffler expansion
- Review of definite integral techniques
- Casimir effect calculations in QFT

---

*"God made the integers, all else is the work of man."*
— Leopold Kronecker
