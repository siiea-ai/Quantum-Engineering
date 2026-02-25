# Day 236: Orthonormal Bases and Parseval's Identity

## Schedule Overview (8 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 3 hours | Theory: Orthonormal bases, completeness, Parseval |
| Afternoon | 3 hours | Problems: Fourier series convergence, completeness proofs |
| Evening | 2 hours | Computational lab: Parseval and energy conservation |

## Learning Objectives

By the end of today, you will be able to:

1. **Define** orthonormal bases in Hilbert spaces (complete orthonormal systems)
2. **State and prove** Parseval's identity
3. **Characterize** orthonormal bases through multiple equivalent conditions
4. **Apply** Parseval's identity to Fourier series
5. **Distinguish** between Hamel bases and orthonormal bases
6. **Connect** Parseval's identity to probability conservation in quantum mechanics

---

## 1. Core Content: Orthonormal Bases

### 1.1 Definition of Orthonormal Basis

**Definition**: An orthonormal set $$\{e_n\}_{n \in I}$$ in a Hilbert space $$\mathcal{H}$$ is an **orthonormal basis** (or **complete orthonormal system**) if it satisfies any of the equivalent conditions:

1. **Spanning**: The closed linear span equals $$\mathcal{H}$$: $$\overline{\text{span}\{e_n\}} = \mathcal{H}$$
2. **Completeness**: If $$\langle x, e_n\rangle = 0$$ for all $$n$$, then $$x = 0$$
3. **Expansion**: Every $$x \in \mathcal{H}$$ can be written as $$x = \sum_n \langle e_n, x\rangle e_n$$
4. **Parseval**: $$\|x\|^2 = \sum_n |\langle e_n, x\rangle|^2$$ for all $$x \in \mathcal{H}$$

**Important Warning**: "Orthonormal basis" in Hilbert space theory is NOT the same as "Hamel basis" from linear algebra!

### 1.2 Hamel Basis vs. Orthonormal Basis

| Hamel Basis | Orthonormal Basis |
|-------------|-------------------|
| Every vector is a **finite** linear combination | Vectors are **infinite** series |
| Exists for all vector spaces (by Zorn's lemma) | Requires Hilbert space structure |
| Uncountable for infinite-dimensional Hilbert spaces | Often countable (separable spaces) |
| Not useful for analysis | Central to analysis and physics |

**Example**: In $$\ell^2$$, the standard basis $$\{e_n\}$$ where $$e_n = (0, \ldots, 0, 1, 0, \ldots)$$ (1 in position $$n$$) is an orthonormal basis:
$$x = (x_1, x_2, x_3, \ldots) = \sum_{n=1}^\infty x_n e_n$$

This is an infinite sum, not a finite linear combination.

---

## 2. Characterization Theorem

### 2.1 Equivalence of Conditions

**Theorem**: For an orthonormal set $$\{e_n\}$$ in a Hilbert space $$\mathcal{H}$$, the following are equivalent:

1. $$\{e_n\}$$ is an orthonormal basis ($$\overline{\text{span}} = \mathcal{H}$$)
2. $$x \perp e_n$$ for all $$n$$ implies $$x = 0$$ (no non-zero vector orthogonal to all $$e_n$$)
3. $$x = \sum_n \langle e_n, x\rangle e_n$$ for all $$x \in \mathcal{H}$$ (Fourier expansion)
4. $$\|x\|^2 = \sum_n |\langle e_n, x\rangle|^2$$ for all $$x \in \mathcal{H}$$ (Parseval)
5. $$\langle x, y\rangle = \sum_n \langle e_n, x\rangle \overline{\langle e_n, y\rangle}$$ for all $$x, y$$ (Generalized Parseval)

### 2.2 Proof Outline

**(1) ⇒ (2)**: If $$x \perp e_n$$ for all $$n$$, then $$x \perp \text{span}\{e_n\}$$. Since $$\overline{\text{span}} = \mathcal{H}$$, we have $$x \perp \mathcal{H}$$, so $$x \perp x$$, giving $$\|x\|^2 = 0$$, hence $$x = 0$$.

**(2) ⇒ (3)**: Let $$y = x - \sum_n \langle e_n, x\rangle e_n$$ (this sum converges by Bessel). Then:
$$\langle e_k, y\rangle = \langle e_k, x\rangle - \sum_n \langle e_n, x\rangle \langle e_k, e_n\rangle = \langle e_k, x\rangle - \langle e_k, x\rangle = 0$$
By (2), $$y = 0$$, so $$x = \sum_n \langle e_n, x\rangle e_n$$.

**(3) ⇒ (4)**: Using continuity of the inner product:
$$\|x\|^2 = \langle x, x\rangle = \left\langle \sum_n c_n e_n, \sum_m c_m e_m \right\rangle = \sum_n \sum_m c_n \bar{c}_m \delta_{nm} = \sum_n |c_n|^2$$

**(4) ⇒ (1)**: If $$\overline{\text{span}} \neq \mathcal{H}$$, take $$x \notin \overline{\text{span}}$$. The projection of $$x$$ onto $$\overline{\text{span}}$$ is $$P x = \sum_n \langle e_n, x\rangle e_n$$. Then $$\|P x\|^2 = \sum_n |c_n|^2 < \|x\|^2$$ (strict inequality since $$x \neq Px$$), contradicting (4). $$\square$$

---

## 3. Parseval's Identity

### 3.1 Statement

**Theorem (Parseval's Identity)**: If $$\{e_n\}$$ is an orthonormal basis for $$\mathcal{H}$$, then for all $$x \in \mathcal{H}$$:

$$\boxed{\|x\|^2 = \sum_{n=1}^\infty |\langle e_n, x\rangle|^2}$$

More generally (Generalized Parseval):
$$\boxed{\langle x, y\rangle = \sum_{n=1}^\infty \langle e_n, x\rangle \overline{\langle e_n, y\rangle}}$$

### 3.2 Relation to Bessel's Inequality

- **Bessel**: $$\sum_n |c_n|^2 \leq \|x\|^2$$ (holds for ANY orthonormal set)
- **Parseval**: $$\sum_n |c_n|^2 = \|x\|^2$$ (holds for orthonormal BASES)

Parseval's identity is "Bessel with equality"—it characterizes when an orthonormal set is complete.

### 3.3 Energy Interpretation

Think of $$\|x\|^2$$ as the "total energy" of $$x$$, and $$|c_n|^2$$ as the "energy in mode $$n$$".

- **Bessel**: Sum of energies in each mode ≤ total energy
- **Parseval**: Sum of energies in each mode = total energy (no energy "leaks out")

---

## 4. Classical Examples

### 4.1 Fourier Series in L²[0, 2π]

The set $$\left\{\frac{e^{inx}}{\sqrt{2\pi}}\right\}_{n \in \mathbb{Z}}$$ is an orthonormal basis for $$L^2[0, 2\pi]$$.

**Parseval for Fourier Series**: For $$f \in L^2[0, 2\pi]$$ with Fourier coefficients $$c_n = \frac{1}{2\pi}\int_0^{2\pi} f(x) e^{-inx} dx$$:

$$\boxed{\frac{1}{2\pi}\int_0^{2\pi} |f(x)|^2 \, dx = \sum_{n=-\infty}^\infty |c_n|^2}$$

### 4.2 Real Fourier Series

Using $$\{1/\sqrt{2\pi}, \cos(nx)/\sqrt{\pi}, \sin(nx)/\sqrt{\pi}\}_{n=1}^\infty$$:

If $$f(x) = \frac{a_0}{2} + \sum_{n=1}^\infty (a_n \cos nx + b_n \sin nx)$$, then:

$$\boxed{\frac{1}{\pi}\int_0^{2\pi} |f(x)|^2 \, dx = \frac{a_0^2}{2} + \sum_{n=1}^\infty (a_n^2 + b_n^2)}$$

### 4.3 Application: A Beautiful Identity

**Example**: For $$f(x) = x$$ on $$[0, 2\pi]$$:

The Fourier series is $$x \sim \pi - 2\sum_{n=1}^\infty \frac{\sin(nx)}{n}$$.

Applying Parseval:
$$\frac{1}{\pi}\int_0^{2\pi} x^2 \, dx = \pi^2 + 2\sum_{n=1}^\infty \frac{1}{n^2}$$

$$\frac{8\pi^2}{3} = \pi^2 + 2\sum_{n=1}^\infty \frac{1}{n^2}$$

Therefore:
$$\boxed{\sum_{n=1}^\infty \frac{1}{n^2} = \frac{\pi^2}{6}}$$

This is the famous **Basel problem**!

---

## 5. Quantum Mechanics Connection

### 5.1 State Expansion and Probability

In quantum mechanics, if $$\{|n\rangle\}$$ is a complete orthonormal basis (eigenstates of some observable), then any state $$|\psi\rangle$$ can be expanded:

$$|\psi\rangle = \sum_n c_n |n\rangle, \quad c_n = \langle n|\psi\rangle$$

### 5.2 Parseval = Probability Conservation

Parseval's identity states:
$$\langle\psi|\psi\rangle = \sum_n |c_n|^2$$

For a normalized state ($$\langle\psi|\psi\rangle = 1$$):
$$\boxed{\sum_n |c_n|^2 = 1}$$

This is **probability conservation**: the probabilities of all possible outcomes sum to 1!

### 5.3 Completeness Relation

The statement "$$\{|n\rangle\}$$ is an orthonormal basis" can be written as:

$$\boxed{\sum_n |n\rangle\langle n| = \hat{I}}$$

This is the **completeness relation** (or resolution of the identity). It means:

$$|\psi\rangle = \hat{I}|\psi\rangle = \sum_n |n\rangle\langle n|\psi\rangle = \sum_n c_n |n\rangle$$

### 5.4 Continuous Spectrum (Preview)

For observables with continuous spectrum (like position), the sum becomes an integral:

$$\int_{-\infty}^\infty |x\rangle\langle x| \, dx = \hat{I}$$

$$|\psi\rangle = \int_{-\infty}^\infty \psi(x)|x\rangle \, dx$$

$$\langle\psi|\psi\rangle = \int_{-\infty}^\infty |\psi(x)|^2 \, dx = 1$$

This is Parseval for the "continuous basis" $$\{|x\rangle\}$$.

---

## 6. Existence of Orthonormal Bases

### 6.1 Existence Theorem

**Theorem**: Every Hilbert space has an orthonormal basis.

**Proof (using Zorn's Lemma)**:
Let $$\mathcal{F}$$ be the collection of all orthonormal sets in $$\mathcal{H}$$, partially ordered by inclusion. Every chain has an upper bound (the union). By Zorn's lemma, there exists a maximal orthonormal set $$\{e_\alpha\}$$.

Claim: This maximal set is a basis.

If not, there exists $$x \neq 0$$ with $$x \perp e_\alpha$$ for all $$\alpha$$. Then $$\{e_\alpha\} \cup \{x/\|x\|\}$$ is a larger orthonormal set, contradicting maximality. $$\square$$

### 6.2 Non-Constructive Nature

The existence proof uses Zorn's lemma and is non-constructive. For specific Hilbert spaces, we construct bases explicitly:

- $$\ell^2$$: Standard basis $$\{e_n\}$$
- $$L^2[0, 2\pi]$$: Fourier basis $$\{e^{inx}/\sqrt{2\pi}\}$$
- $$L^2(\mathbb{R})$$: Hermite functions (harmonic oscillator eigenstates)

---

## 7. Worked Examples

### Example 1: Proving Completeness

**Problem**: Prove that $$\{e^{inx}/\sqrt{2\pi}\}_{n \in \mathbb{Z}}$$ is complete in $$L^2[0, 2\pi]$$ by showing that if $$f \perp e^{inx}$$ for all $$n$$, then $$f = 0$$.

**Solution**:

Suppose $$\langle f, e^{inx}\rangle = 0$$ for all $$n \in \mathbb{Z}$$, i.e.,
$$\int_0^{2\pi} f(x) e^{-inx} \, dx = 0 \quad \forall n \in \mathbb{Z}$$

This means all Fourier coefficients of $$f$$ are zero.

**Key fact (Stone-Weierstrass)**: Trigonometric polynomials are dense in $$C[0, 2\pi]$$ (continuous functions), which is dense in $$L^2[0, 2\pi]$$.

For any trigonometric polynomial $$p(x) = \sum_{|k| \leq N} a_k e^{ikx}$$:
$$\langle f, p\rangle = \sum_{|k| \leq N} \bar{a}_k \langle f, e^{ikx}\rangle = 0$$

Since trig polynomials are dense and $$\langle f, p\rangle = 0$$ for all of them:
$$\langle f, f\rangle = \lim \langle f, p_n\rangle = 0$$

where $$p_n \to f$$ in $$L^2$$. Therefore $$f = 0$$. $$\square$$

---

### Example 2: Using Parseval to Compute a Sum

**Problem**: Use Parseval's identity to evaluate $$\sum_{n=1}^\infty \frac{1}{n^4}$$.

**Solution**:

Consider $$f(x) = x^2$$ on $$[-\pi, \pi]$$.

The Fourier series of $$f(x) = x^2$$ is:
$$x^2 = \frac{\pi^2}{3} + \sum_{n=1}^\infty \frac{4(-1)^n}{n^2} \cos(nx)$$

Parseval's identity (for the interval $$[-\pi, \pi]$$):
$$\frac{1}{\pi}\int_{-\pi}^{\pi} x^4 \, dx = \frac{2\pi^4}{9} + 2\sum_{n=1}^\infty \frac{16}{n^4}$$

Left side:
$$\frac{1}{\pi}\int_{-\pi}^{\pi} x^4 \, dx = \frac{1}{\pi} \cdot \frac{2\pi^5}{5} = \frac{2\pi^4}{5}$$

So:
$$\frac{2\pi^4}{5} = \frac{2\pi^4}{9} + 32\sum_{n=1}^\infty \frac{1}{n^4}$$

$$32\sum_{n=1}^\infty \frac{1}{n^4} = \frac{2\pi^4}{5} - \frac{2\pi^4}{9} = 2\pi^4 \left(\frac{1}{5} - \frac{1}{9}\right) = 2\pi^4 \cdot \frac{4}{45} = \frac{8\pi^4}{45}$$

$$\boxed{\sum_{n=1}^\infty \frac{1}{n^4} = \frac{\pi^4}{90}}$$

This is $$\zeta(4) = \pi^4/90$$! $$\square$$

---

### Example 3: Quantum Harmonic Oscillator

**Problem**: A quantum harmonic oscillator is in the state $$|\psi\rangle = \frac{1}{\sqrt{3}}(|0\rangle + |1\rangle + |2\rangle)$$. Verify that Parseval's identity holds and interpret physically.

**Solution**:

The state has Fourier coefficients:
$$c_0 = c_1 = c_2 = \frac{1}{\sqrt{3}}, \quad c_n = 0 \text{ for } n \geq 3$$

**Parseval**:
$$\|\psi\|^2 = \sum_{n=0}^\infty |c_n|^2 = |c_0|^2 + |c_1|^2 + |c_2|^2 = \frac{1}{3} + \frac{1}{3} + \frac{1}{3} = 1$$ $$\checkmark$$

**Physical Interpretation**:
- Probability of measuring energy $$E_0 = \frac{1}{2}\hbar\omega$$: $$P_0 = |c_0|^2 = 1/3$$
- Probability of measuring energy $$E_1 = \frac{3}{2}\hbar\omega$$: $$P_1 = |c_1|^2 = 1/3$$
- Probability of measuring energy $$E_2 = \frac{5}{2}\hbar\omega$$: $$P_2 = |c_2|^2 = 1/3$$
- Total probability: $$P_0 + P_1 + P_2 = 1$$ $$\checkmark$$

Parseval guarantees probability conservation! $$\square$$

---

## 8. Practice Problems

### Level 1: Direct Application

1. Verify Parseval's identity for $$f(x) = \sin(x)$$ in $$L^2[0, 2\pi]$$.

2. If $$\{e_n\}$$ is an orthonormal basis and $$x = \sum_n c_n e_n$$, $$y = \sum_n d_n e_n$$, express $$\langle x, y\rangle$$ in terms of $$c_n$$ and $$d_n$$.

3. Prove that $$\{(1, 0, 0), (0, 1, 0), (0, 0, 1)\}$$ is an orthonormal basis for $$\mathbb{R}^3$$ by verifying the completeness condition.

### Level 2: Intermediate

4. Use Parseval's identity to evaluate $$\sum_{n=1}^\infty \frac{1}{(2n-1)^2}$$.
   (Hint: Consider the Fourier series of a suitable odd function.)

5. Prove that if $$\{e_n\}$$ is an orthonormal basis and $$U: \mathcal{H} \to \mathcal{H}$$ is unitary, then $$\{Ue_n\}$$ is also an orthonormal basis.

6. **Quantum Problem**: A spin-1 particle has basis states $$|1\rangle, |0\rangle, |-1\rangle$$. If the state is $$|\psi\rangle = \frac{1}{2}|1\rangle + \frac{1}{\sqrt{2}}|0\rangle + \frac{1}{2}|-1\rangle$$:
   - Verify normalization using Parseval
   - What is the probability of measuring $$S_z = 0$$?

### Level 3: Challenging

7. Prove that if $$f \in L^2[0, 2\pi]$$ has Fourier coefficients $$c_n$$ with $$\sum_n n^2 |c_n|^2 < \infty$$, then $$f$$ has a derivative $$f' \in L^2$$.

8. Show that the completeness relation $$\sum_n |e_n\rangle\langle e_n| = I$$ is equivalent to Parseval's identity.

9. **(Müntz-Szász Theorem direction)**: Prove that $$\{1, x, x^2, \ldots\}$$ spans $$L^2[0, 1]$$ (i.e., polynomials are dense). Use this to show $$\{x^{n_k}\}$$ is complete if $$\sum 1/n_k = \infty$$.

---

## 9. Computational Lab: Parseval and Energy Conservation

```python
"""
Day 236 Computational Lab: Parseval's Identity and Energy Conservation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import hermite

# ============================================================
# Part 1: Verifying Parseval for Fourier Series
# ============================================================

def verify_parseval_fourier():
    """
    Verify Parseval's identity for various functions using Fourier series.
    """
    x = np.linspace(0, 2*np.pi, 1000)

    # Test functions
    functions = {
        'f(x) = x': lambda x: x,
        'f(x) = x^2': lambda x: x**2,
        'f(x) = sin(x)': lambda x: np.sin(x),
        'f(x) = |x - π|': lambda x: np.abs(x - np.pi)
    }

    print("Parseval's Identity Verification")
    print("=" * 60)
    print("For f ∈ L²[0, 2π], Σ|c_n|² = (1/2π)∫|f(x)|²dx")
    print()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (name, f) in enumerate(functions.items()):
        f_vals = f(x)

        # Compute ||f||² directly
        norm_sq_direct = np.trapz(np.abs(f_vals)**2, x) / (2*np.pi)

        # Compute Fourier coefficients
        N_max = 100
        coeffs_sq_sum = []
        n_range = range(-N_max, N_max + 1)

        partial_sum = 0
        for n in n_range:
            # c_n = (1/2π) ∫ f(x) e^{-inx} dx
            integrand_real = f_vals * np.cos(n * x)
            integrand_imag = f_vals * np.sin(n * x)
            c_n_real = np.trapz(integrand_real, x) / (2*np.pi)
            c_n_imag = -np.trapz(integrand_imag, x) / (2*np.pi)
            c_n = c_n_real + 1j * c_n_imag
            partial_sum += np.abs(c_n)**2
            if n >= -N_max:
                coeffs_sq_sum.append(partial_sum)

        print(f"{name}:")
        print(f"  ||f||²/(2π) (direct) = {norm_sq_direct:.6f}")
        print(f"  Σ|c_n|² (N={N_max})  = {partial_sum:.6f}")
        print(f"  Difference           = {abs(norm_sq_direct - partial_sum):.2e}")
        print()

        # Plot convergence
        ax = axes[idx]
        ax.plot(range(1, len(coeffs_sq_sum)+1), coeffs_sq_sum, 'b-', linewidth=2)
        ax.axhline(y=norm_sq_direct, color='r', linestyle='--',
                   label=f'$\\|f\\|^2/(2\\pi) = {norm_sq_direct:.4f}$')
        ax.set_xlabel('Number of terms (2N+1)')
        ax.set_ylabel('$\\sum |c_n|^2$')
        ax.set_title(f'Parseval Convergence: {name}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('parseval_verification.png', dpi=150)
    plt.show()

# ============================================================
# Part 2: Computing Famous Sums Using Parseval
# ============================================================

def compute_famous_sums():
    """
    Use Parseval's identity to compute famous infinite series.
    """
    print("\nComputing Famous Sums via Parseval")
    print("=" * 60)

    # Basel problem: Σ 1/n² = π²/6
    # Use f(x) = x on [-π, π]
    # Fourier: x = 2Σ (-1)^{n+1}/n * sin(nx)

    x = np.linspace(-np.pi, np.pi, 10000)
    f = x

    # ||f||²
    norm_sq = np.trapz(f**2, x) / np.pi  # Factor of 1/π for [-π,π]

    # Fourier coefficients: b_n = 2*(-1)^{n+1}/n
    # Σ b_n² = 4 * Σ 1/n² = ||f||²/π... let me recalculate

    # Actually: ||f||² = ∫_{-π}^{π} x² dx = 2π³/3
    # Parseval: ||f||² = 2π * Σ |c_n|² where c_n are complex Fourier coeffs
    # For x: c_0 = 0, c_n = i*(-1)^n/n for n ≠ 0
    # So |c_n|² = 1/n²
    # 2π * 2*Σ_{n=1}^∞ 1/n² = 2π³/3
    # Σ 1/n² = π²/6

    sum_inv_n2 = np.trapz(f**2, x) / (4 * np.pi)
    theoretical = np.pi**2 / 6

    print(f"Basel Problem: Σ 1/n²")
    print(f"  Numerical (via Parseval): {sum_inv_n2:.10f}")
    print(f"  Theoretical (π²/6):       {theoretical:.10f}")
    print()

    # Σ 1/n⁴ = π⁴/90
    # Use f(x) = x² on [-π, π]
    f2 = x**2

    # ||x²||² = ∫_{-π}^{π} x⁴ dx = 2π⁵/5
    # Fourier: x² = π²/3 + 4Σ(-1)^n/n² cos(nx)
    # a_0 = 2π²/3, a_n = 4*(-1)^n/n²
    # Parseval: a_0²/2 + Σ a_n² = ||f||²/π
    # (2π²/3)²/2 + 16*Σ 1/n⁴ = 2π⁴/5
    # 2π⁴/9 + 16*Σ 1/n⁴ = 2π⁴/5
    # Σ 1/n⁴ = (2π⁴/5 - 2π⁴/9)/16 = π⁴/90

    # Numerical verification
    a0_sq_half = (2*np.pi**2/3)**2 / 2
    norm_sq_f2 = np.trapz(f2**2, x) / np.pi
    sum_inv_n4 = (norm_sq_f2 - a0_sq_half) / 16
    theoretical_n4 = np.pi**4 / 90

    print(f"Σ 1/n⁴:")
    print(f"  Numerical (via Parseval): {sum_inv_n4:.10f}")
    print(f"  Theoretical (π⁴/90):      {theoretical_n4:.10f}")

# ============================================================
# Part 3: Quantum Probability Conservation
# ============================================================

def quantum_probability_conservation():
    """
    Demonstrate Parseval as probability conservation in QM.
    """
    x = np.linspace(-10, 10, 1000)

    # Harmonic oscillator eigenstates
    def psi_n(x, n):
        Hn = hermite(n)
        norm = 1 / np.sqrt(2**n * np.math.factorial(n) * np.sqrt(np.pi))
        return norm * Hn(x) * np.exp(-x**2 / 2)

    # Create a general state
    np.random.seed(42)
    n_states = 10
    raw_coeffs = np.random.randn(n_states) + 1j * np.random.randn(n_states)
    coeffs = raw_coeffs / np.linalg.norm(raw_coeffs)  # Normalize

    # Construct the wave function
    psi = np.zeros_like(x, dtype=complex)
    for n, c in enumerate(coeffs):
        psi += c * psi_n(x, n)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot wave function
    ax = axes[0, 0]
    ax.plot(x, np.real(psi), 'b-', linewidth=2, label=r'Re($\psi$)')
    ax.plot(x, np.imag(psi), 'r--', linewidth=2, label=r'Im($\psi$)')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$\\psi(x)$')
    ax.set_title('Quantum State (Superposition of Eigenstates)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Probability density
    ax = axes[0, 1]
    prob_density = np.abs(psi)**2
    ax.fill_between(x, prob_density, alpha=0.5, color='purple')
    ax.plot(x, prob_density, 'purple', linewidth=2)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|\\psi(x)|^2$')
    ax.set_title('Probability Density')
    ax.grid(True, alpha=0.3)

    # Verify normalization
    norm_position = np.trapz(prob_density, x)
    norm_coeffs = np.sum(np.abs(coeffs)**2)

    print("\nQuantum Probability Conservation (Parseval)")
    print("=" * 60)
    print(f"||ψ||² from position space: ∫|ψ(x)|²dx = {norm_position:.10f}")
    print(f"||ψ||² from coefficients:   Σ|c_n|²    = {norm_coeffs:.10f}")
    print(f"Both should equal 1 (normalized state)")

    # Bar chart of probabilities
    ax = axes[1, 0]
    probs = np.abs(coeffs)**2
    ax.bar(range(n_states), probs, color='steelblue', alpha=0.7)
    ax.set_xlabel('Energy level $n$')
    ax.set_ylabel('$|c_n|^2$')
    ax.set_title(f'Probability Distribution (Sum = {np.sum(probs):.6f})')
    ax.grid(True, alpha=0.3)

    # Time evolution (phases rotate, but |c_n|² stays constant)
    ax = axes[1, 1]
    omega = 1.0  # Fundamental frequency
    times = np.linspace(0, 4*np.pi, 100)
    total_probs = []

    for t in times:
        # Time evolution: c_n(t) = c_n(0) * exp(-i E_n t / ℏ) = c_n(0) * exp(-i (n+1/2) ω t)
        phases = np.exp(-1j * (np.arange(n_states) + 0.5) * omega * t)
        evolved_coeffs = coeffs * phases
        total_probs.append(np.sum(np.abs(evolved_coeffs)**2))

    ax.plot(times, total_probs, 'b-', linewidth=2)
    ax.axhline(y=1, color='r', linestyle='--', label='Conservation')
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('$\\sum |c_n(t)|^2$')
    ax.set_title('Probability Conservation Under Time Evolution')
    ax.legend()
    ax.set_ylim(0.99, 1.01)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('quantum_parseval.png', dpi=150)
    plt.show()

# ============================================================
# Part 4: Completeness Relation
# ============================================================

def demonstrate_completeness_relation():
    """
    Demonstrate the completeness relation Σ|n><n| = I.
    """
    x = np.linspace(-8, 8, 500)

    def psi_n(x, n):
        Hn = hermite(n)
        norm = 1 / np.sqrt(2**n * np.math.factorial(n) * np.sqrt(np.pi))
        return norm * Hn(x) * np.exp(-x**2 / 2)

    # Construct Σ |n><n| for n = 0, 1, ..., N-1
    # This is a kernel: K(x, x') = Σ_n ψ_n(x) ψ_n(x')
    # For complete basis, K(x, x') → δ(x - x')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    N_values = [3, 10, 30, 50]
    for idx, N in enumerate(N_values):
        ax = axes[idx // 2, idx % 2]

        # Compute partial completeness kernel
        # For visualization, we look at K(x, 0) which should approach δ(x)
        K_x_0 = np.zeros_like(x)
        for n in range(N):
            K_x_0 += psi_n(x, n) * psi_n(0, n)

        ax.plot(x, K_x_0, 'b-', linewidth=2)
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('$x$')
        ax.set_ylabel(f'$K_N(x, 0)$')
        ax.set_title(f'Partial Completeness Kernel ($N = {N}$)')
        ax.set_xlim(-4, 4)
        ax.grid(True, alpha=0.3)

    plt.suptitle(r'$K_N(x, x^\prime) = \sum_{n=0}^{N-1} \psi_n(x)\psi_n(x^\prime) \rightarrow \delta(x - x^\prime)$',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('completeness_relation.png', dpi=150)
    plt.show()

    print("\nCompleteness Relation Demonstration")
    print("=" * 60)
    print("As N → ∞, the kernel K_N(x, x') approaches δ(x - x')")
    print("This is the position-space form of Σ|n><n| = I")

# ============================================================
# Part 5: Energy in Fourier Modes
# ============================================================

def energy_distribution():
    """
    Visualize how 'energy' distributes across Fourier modes.
    """
    x = np.linspace(0, 2*np.pi, 1000)

    # Signal with multiple frequencies
    def signal(x):
        return np.sin(x) + 0.5*np.sin(3*x) + 0.3*np.sin(5*x) + 0.1*np.sin(7*x)

    f = signal(x)

    # Compute energy contributions
    total_energy = np.trapz(f**2, x) / np.pi  # ||f||²/π

    # Fourier coefficients (for sin series since f is odd-like)
    n_max = 20
    energies = []
    for n in range(1, n_max + 1):
        b_n = (2/np.pi) * np.trapz(f * np.sin(n*x), x)
        energies.append(b_n**2 / 2)  # Energy in mode n

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Signal
    ax = axes[0]
    ax.plot(x, f, 'b-', linewidth=2)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$f(x)$')
    ax.set_title('Signal: $\\sin x + 0.5\\sin 3x + 0.3\\sin 5x + 0.1\\sin 7x$')
    ax.grid(True, alpha=0.3)

    # Energy spectrum
    ax = axes[1]
    bars = ax.bar(range(1, n_max+1), energies, color='steelblue', alpha=0.7)
    # Highlight expected peaks
    for n in [1, 3, 5, 7]:
        if n <= n_max:
            bars[n-1].set_color('red')
    ax.set_xlabel('Mode $n$')
    ax.set_ylabel('Energy $|b_n|^2/2$')
    ax.set_title('Energy Spectrum (Parseval Components)')
    ax.grid(True, alpha=0.3)

    # Cumulative energy
    ax = axes[2]
    cumsum = np.cumsum(energies)
    ax.plot(range(1, n_max+1), cumsum, 'b-o', markersize=4)
    ax.axhline(y=total_energy, color='r', linestyle='--',
               label=f'Total Energy = {total_energy:.4f}')
    ax.set_xlabel('Mode $n$')
    ax.set_ylabel('Cumulative Energy')
    ax.set_title('Energy Accumulation (Parseval Convergence)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    print("\nEnergy Distribution Analysis")
    print("=" * 60)
    print(f"Total energy ||f||²/π = {total_energy:.6f}")
    print(f"Sum of Fourier energies = {sum(energies):.6f}")
    print("Energy by mode:")
    for n in [1, 3, 5, 7]:
        if n <= len(energies):
            print(f"  Mode {n}: {energies[n-1]:.6f}")

    plt.tight_layout()
    plt.savefig('energy_distribution.png', dpi=150)
    plt.show()

# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Day 236: Parseval's Identity - Computational Lab")
    print("=" * 60)

    print("\n1. Verifying Parseval for Fourier series...")
    verify_parseval_fourier()

    print("\n2. Computing famous sums using Parseval...")
    compute_famous_sums()

    print("\n3. Quantum probability conservation...")
    quantum_probability_conservation()

    print("\n4. Demonstrating completeness relation...")
    demonstrate_completeness_relation()

    print("\n5. Energy distribution in Fourier modes...")
    energy_distribution()

    print("\n" + "=" * 60)
    print("Lab complete!")
    print("=" * 60)
```

---

## 10. Summary

### Key Definitions

| Concept | Definition |
|---------|------------|
| **Orthonormal Basis** | Complete orthonormal set ($$\overline{\text{span}} = \mathcal{H}$$) |
| **Completeness** | $$x \perp e_n \; \forall n \Rightarrow x = 0$$ |
| **Fourier Expansion** | $$x = \sum_n \langle e_n, x\rangle e_n$$ |

### Key Formulas

$$\boxed{\begin{aligned}
&\text{Parseval's Identity:} && \|x\|^2 = \sum_n |\langle e_n, x\rangle|^2 \\[5pt]
&\text{Generalized Parseval:} && \langle x, y\rangle = \sum_n \langle e_n, x\rangle \overline{\langle e_n, y\rangle} \\[5pt]
&\text{Completeness Relation:} && \sum_n |e_n\rangle\langle e_n| = I
\end{aligned}}$$

### Key Insights

1. **Orthonormal bases are complete** — every vector has a convergent Fourier expansion
2. **Parseval = Bessel with equality** — characterizes complete bases
3. **Parseval gives famous sums** — $$\zeta(2) = \pi^2/6$$, $$\zeta(4) = \pi^4/90$$
4. **Quantum probability conservation** — $$\sum |c_n|^2 = 1$$
5. **Completeness relation** — $$\sum |n\rangle\langle n| = I$$ is Parseval in operator form

---

## 11. Daily Checklist

- [ ] I can define an orthonormal basis and distinguish it from a Hamel basis
- [ ] I can state and prove Parseval's identity
- [ ] I understand the equivalence of completeness conditions
- [ ] I can use Parseval to compute infinite series
- [ ] I understand Parseval as probability conservation in quantum mechanics
- [ ] I completed the computational lab exercises

---

## 12. Preview: Day 237

Tomorrow we study **separable Hilbert spaces**—those with countable orthonormal bases. We'll prove the remarkable classification theorem: all infinite-dimensional separable Hilbert spaces are isomorphic to $$\ell^2$$! This means $$L^2$$, the space of quantum states, is "just" $$\ell^2$$ in disguise.

---

*"Parseval's identity is not merely a mathematical theorem—it is the mathematical expression of probability conservation, the cornerstone of quantum mechanics."*
