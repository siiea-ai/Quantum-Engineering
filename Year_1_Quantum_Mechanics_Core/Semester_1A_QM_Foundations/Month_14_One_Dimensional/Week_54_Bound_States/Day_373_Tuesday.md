# Day 373: Infinite Square Well - Eigenfunctions, Orthonormality, and Completeness

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Eigenfunctions and orthonormality |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem solving: Completeness and expansions |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational lab: Eigenfunction visualization |

**Total Study Time: 7 hours**

---

## Learning Objectives

By the end of today, you will be able to:

1. Derive the normalization constant for ISW eigenfunctions
2. Write normalized eigenfunctions in standard form
3. Prove orthonormality of eigenfunctions using explicit integration
4. Explain and apply the completeness relation
5. Expand arbitrary wave functions in the energy eigenbasis
6. Analyze parity properties of the eigenfunctions

---

## Core Content

### 1. Normalized Eigenfunctions

From Day 372, we found that the energy eigenfunctions have the form:

$$\psi_n(x) = A\sin\left(\frac{n\pi x}{L}\right)$$

To find the normalization constant $A$, we require:

$$\int_0^L |\psi_n(x)|^2 dx = 1$$

#### Normalization Integral

$$|A|^2 \int_0^L \sin^2\left(\frac{n\pi x}{L}\right) dx = 1$$

Using the identity $\sin^2(\theta) = \frac{1}{2}(1 - \cos(2\theta))$:

$$|A|^2 \int_0^L \frac{1}{2}\left[1 - \cos\left(\frac{2n\pi x}{L}\right)\right] dx = 1$$

$$|A|^2 \cdot \frac{1}{2}\left[x - \frac{L}{2n\pi}\sin\left(\frac{2n\pi x}{L}\right)\right]_0^L = 1$$

$$|A|^2 \cdot \frac{1}{2}\left[L - 0\right] = 1$$

$$|A|^2 = \frac{2}{L} \implies A = \sqrt{\frac{2}{L}}$$

(We choose $A$ real and positive by convention.)

#### The Normalized Eigenfunctions

$$\boxed{\psi_n(x) = \sqrt{\frac{2}{L}}\sin\left(\frac{n\pi x}{L}\right), \quad n = 1, 2, 3, \ldots}$$

This is valid for $0 \leq x \leq L$, with $\psi_n(x) = 0$ outside this region.

### 2. Visualizing the Eigenfunctions

| State | Function | Nodes | Parity about $x=L/2$ |
|-------|----------|-------|----------------------|
| $n=1$ | $\sqrt{2/L}\sin(\pi x/L)$ | 0 | Even |
| $n=2$ | $\sqrt{2/L}\sin(2\pi x/L)$ | 1 | Odd |
| $n=3$ | $\sqrt{2/L}\sin(3\pi x/L)$ | 2 | Even |
| $n=4$ | $\sqrt{2/L}\sin(4\pi x/L)$ | 3 | Odd |

**General pattern:**
- $n$ odd: symmetric about $x = L/2$ (even parity)
- $n$ even: antisymmetric about $x = L/2$ (odd parity)

```
ψ₁(x):  /\        (no nodes, one "bump")
       /  \

ψ₂(x):  /\  /\    (one node at L/2)
       /  \/  \

ψ₃(x): /\ /\ /\   (two nodes)
      /  X  X  \
```

### 3. Orthonormality

The eigenfunctions of a Hermitian operator form an **orthonormal set**. For the ISW:

$$\boxed{\langle\psi_m|\psi_n\rangle = \int_0^L \psi_m^*(x)\psi_n(x)\,dx = \delta_{mn}}$$

where $\delta_{mn}$ is the Kronecker delta:

$$\delta_{mn} = \begin{cases} 1 & \text{if } m = n \\ 0 & \text{if } m \neq n \end{cases}$$

#### Proof of Orthonormality

**Case 1: $m = n$ (Normalization)**

Already shown above: $\int_0^L |\psi_n|^2 dx = 1$.

**Case 2: $m \neq n$ (Orthogonality)**

$$\langle\psi_m|\psi_n\rangle = \frac{2}{L}\int_0^L \sin\left(\frac{m\pi x}{L}\right)\sin\left(\frac{n\pi x}{L}\right)dx$$

Using the product-to-sum formula:

$$\sin A \sin B = \frac{1}{2}[\cos(A-B) - \cos(A+B)]$$

With $A = m\pi x/L$ and $B = n\pi x/L$:

$$\langle\psi_m|\psi_n\rangle = \frac{1}{L}\int_0^L \left[\cos\left(\frac{(m-n)\pi x}{L}\right) - \cos\left(\frac{(m+n)\pi x}{L}\right)\right]dx$$

$$= \frac{1}{L}\left[\frac{L}{(m-n)\pi}\sin\left(\frac{(m-n)\pi x}{L}\right) - \frac{L}{(m+n)\pi}\sin\left(\frac{(m+n)\pi x}{L}\right)\right]_0^L$$

For integer $m \neq n$, both $(m-n)$ and $(m+n)$ are integers, so:

$$\sin((m-n)\pi) = 0 \quad \text{and} \quad \sin((m+n)\pi) = 0$$

Therefore:

$$\langle\psi_m|\psi_n\rangle = 0 \quad \text{for } m \neq n \quad \checkmark$$

### 4. Completeness

The energy eigenfunctions form a **complete basis** for the Hilbert space of square-integrable functions on $[0, L]$ that vanish at the boundaries.

#### The Completeness Relation

$$\boxed{\sum_{n=1}^{\infty} |\psi_n\rangle\langle\psi_n| = \hat{\mathbb{1}}}$$

In position representation:

$$\sum_{n=1}^{\infty} \psi_n(x)\psi_n^*(x') = \delta(x - x')$$

This means: **any function $f(x)$ satisfying the boundary conditions can be expanded as:**

$$f(x) = \sum_{n=1}^{\infty} c_n \psi_n(x)$$

where the expansion coefficients are:

$$\boxed{c_n = \langle\psi_n|f\rangle = \int_0^L \psi_n^*(x)f(x)\,dx}$$

#### Physical Interpretation

- The $\{|\psi_n\rangle\}$ are like unit vectors in an infinite-dimensional space
- Any state can be decomposed into energy eigenstates
- The coefficient $|c_n|^2$ is the probability of measuring energy $E_n$

### 5. Fourier Series Connection

The expansion $f(x) = \sum_n c_n \psi_n(x)$ is precisely a **Fourier sine series**!

Writing explicitly:

$$f(x) = \sum_{n=1}^{\infty} c_n \sqrt{\frac{2}{L}}\sin\left(\frac{n\pi x}{L}\right)$$

With Fourier coefficients:

$$c_n = \sqrt{\frac{2}{L}}\int_0^L f(x)\sin\left(\frac{n\pi x}{L}\right)dx$$

This connection between quantum mechanics and Fourier analysis is deep and pervasive:
- Position and momentum representations are Fourier transform pairs
- Energy eigenstates are "frequency" components of the wave function
- Quantum dynamics decomposes into independent oscillations

### 6. Parity Symmetry

#### Shifted Well: $-L/2 < x < L/2$

If we center the well at the origin with boundaries at $x = \pm L/2$, the Hamiltonian commutes with the parity operator $\hat{P}$:

$$\hat{P}f(x) = f(-x)$$

Since $[\hat{H}, \hat{P}] = 0$, we can choose energy eigenfunctions that are also parity eigenstates.

**Even parity ($\hat{P}\psi = +\psi$):**
$$\psi_n^{(+)}(x) = \sqrt{\frac{2}{L}}\cos\left(\frac{n\pi x}{L}\right), \quad n = 1, 3, 5, \ldots$$

**Odd parity ($\hat{P}\psi = -\psi$):**
$$\psi_n^{(-)}(x) = \sqrt{\frac{2}{L}}\sin\left(\frac{n\pi x}{L}\right), \quad n = 2, 4, 6, \ldots$$

The energies are still $E_n = n^2\pi^2\hbar^2/2mL^2$, and the parity alternates with $n$.

### 7. Expansion Coefficient Calculations

#### Finding $c_n$ for Specific Initial States

**Example: Uniform initial state**

$$f(x) = \begin{cases} \sqrt{1/L} & 0 < x < L \\ 0 & \text{otherwise} \end{cases}$$

$$c_n = \int_0^L \sqrt{\frac{2}{L}}\sin\left(\frac{n\pi x}{L}\right) \cdot \sqrt{\frac{1}{L}} dx$$

$$c_n = \sqrt{\frac{2}{L^2}}\left[-\frac{L}{n\pi}\cos\left(\frac{n\pi x}{L}\right)\right]_0^L$$

$$c_n = \sqrt{\frac{2}{L^2}} \cdot \frac{L}{n\pi}[1 - \cos(n\pi)] = \sqrt{\frac{2}{L^2}} \cdot \frac{L}{n\pi}[1 - (-1)^n]$$

$$c_n = \begin{cases} \frac{2\sqrt{2}}{\sqrt{L}n\pi} & n \text{ odd} \\ 0 & n \text{ even} \end{cases}$$

Only odd-$n$ states contribute because the uniform function is symmetric about $x = L/2$!

### 8. Parseval's Theorem

The normalization of $f(x)$ is preserved in the expansion:

$$\int_0^L |f(x)|^2 dx = \sum_{n=1}^{\infty} |c_n|^2$$

This is the quantum mechanical statement that **probabilities sum to 1**:

$$\sum_{n=1}^{\infty} |c_n|^2 = 1$$

---

## Physical Interpretation

### The Eigenfunction as a Standing Wave

Each $\psi_n(x)$ represents a **standing wave** with:
- Wavelength $\lambda_n = 2L/n$
- $n$ antinodes (points of maximum amplitude)
- $(n-1)$ nodes (points where $\psi = 0$, excluding boundaries)

The energy eigenstates are stationary - their probability density $|\psi_n(x)|^2$ is time-independent.

### Superposition and Interference

A general state $\Psi(x) = \sum_n c_n \psi_n(x)$ exhibits quantum interference:

$$|\Psi|^2 = \sum_{m,n} c_m^* c_n \psi_m^* \psi_n$$

The cross terms $c_m^* c_n \psi_m^* \psi_n$ for $m \neq n$ create interference patterns that evolve in time.

### Probability Interpretation of Coefficients

If we measure the energy:
- Probability of finding $E_n$: $P(E_n) = |c_n|^2$
- Expected energy: $\langle E \rangle = \sum_n |c_n|^2 E_n$
- Energy uncertainty: $(\Delta E)^2 = \langle E^2 \rangle - \langle E \rangle^2$

---

## Quantum Computing Connection

### Qubit State Representation

A qubit uses the two lowest energy levels of a quantum system:

$$|0\rangle \equiv |\psi_1\rangle, \quad |1\rangle \equiv |\psi_2\rangle$$

A general qubit state:

$$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle = \alpha\psi_1(x) + \beta\psi_2(x)$$

with $|\alpha|^2 + |\beta|^2 = 1$.

### Orthonormality in Quantum Computing

The orthonormality $\langle 0|1\rangle = 0$ is essential for:
- **Distinguishability**: $|0\rangle$ and $|1\rangle$ can be perfectly distinguished by measurement
- **Unitary gates**: Quantum operations preserve orthonormality
- **No cloning**: Orthonormal states cannot be copied

### Completeness and Quantum Error Correction

Completeness $\sum_n |n\rangle\langle n| = \mathbb{1}$ underlies:
- **Syndrome measurement**: Project onto error subspaces
- **Recovery operations**: Restore state using completeness
- **Stabilizer codes**: Logical qubits live in subspaces spanned by physical eigenstates

---

## Worked Examples

### Example 1: Normalization Verification

**Problem:** Verify that $\psi_3(x) = \sqrt{2/L}\sin(3\pi x/L)$ is normalized.

**Solution:**

$$\int_0^L |\psi_3(x)|^2 dx = \frac{2}{L}\int_0^L \sin^2\left(\frac{3\pi x}{L}\right)dx$$

Let $u = 3\pi x/L$, so $du = 3\pi dx/L$ and $dx = L\,du/(3\pi)$:

$$= \frac{2}{L} \cdot \frac{L}{3\pi}\int_0^{3\pi} \sin^2(u)\,du$$

$$= \frac{2}{3\pi}\left[\frac{u}{2} - \frac{\sin(2u)}{4}\right]_0^{3\pi}$$

$$= \frac{2}{3\pi}\left[\frac{3\pi}{2} - 0\right] = 1 \quad \checkmark$$

---

### Example 2: Orthogonality Check

**Problem:** Show that $\langle\psi_1|\psi_2\rangle = 0$ by direct integration.

**Solution:**

$$\langle\psi_1|\psi_2\rangle = \frac{2}{L}\int_0^L \sin\left(\frac{\pi x}{L}\right)\sin\left(\frac{2\pi x}{L}\right)dx$$

Using $\sin A \sin B = \frac{1}{2}[\cos(A-B) - \cos(A+B)]$:

$$= \frac{1}{L}\int_0^L \left[\cos\left(\frac{-\pi x}{L}\right) - \cos\left(\frac{3\pi x}{L}\right)\right]dx$$

$$= \frac{1}{L}\int_0^L \left[\cos\left(\frac{\pi x}{L}\right) - \cos\left(\frac{3\pi x}{L}\right)\right]dx$$

$$= \frac{1}{L}\left[\frac{L}{\pi}\sin\left(\frac{\pi x}{L}\right) - \frac{L}{3\pi}\sin\left(\frac{3\pi x}{L}\right)\right]_0^L$$

$$= \frac{1}{L}\left[\frac{L}{\pi}(\sin\pi - \sin 0) - \frac{L}{3\pi}(\sin 3\pi - \sin 0)\right]$$

$$= \frac{1}{L}\left[0 - 0\right] = 0 \quad \checkmark$$

---

### Example 3: Expansion of a Triangular Wave

**Problem:** A particle is prepared in the triangular state:

$$f(x) = \begin{cases} Ax & 0 \leq x \leq L/2 \\ A(L - x) & L/2 \leq x \leq L \end{cases}$$

Find (a) the normalization constant $A$, and (b) the first three non-zero expansion coefficients $c_n$.

**Solution:**

(a) **Normalization:**

$$1 = \int_0^L |f|^2 dx = A^2\left[\int_0^{L/2} x^2 dx + \int_{L/2}^L (L-x)^2 dx\right]$$

$$= A^2\left[\frac{x^3}{3}\Big|_0^{L/2} + \frac{-(L-x)^3}{3}\Big|_{L/2}^L\right]$$

$$= A^2\left[\frac{L^3}{24} + \frac{L^3}{24}\right] = A^2 \frac{L^3}{12}$$

$$A = \sqrt{\frac{12}{L^3}} = \frac{2\sqrt{3}}{L^{3/2}}$$

(b) **Expansion coefficients:**

$$c_n = \sqrt{\frac{2}{L}}\int_0^L f(x)\sin\left(\frac{n\pi x}{L}\right)dx$$

By symmetry, $f(x)$ is symmetric about $x = L/2$, so only odd-$n$ coefficients survive.

For $n = 1$:

$$c_1 = \sqrt{\frac{2}{L}} \cdot A \left[\int_0^{L/2} x\sin\left(\frac{\pi x}{L}\right)dx + \int_{L/2}^L (L-x)\sin\left(\frac{\pi x}{L}\right)dx\right]$$

Using integration by parts and symmetry:

$$c_1 = \sqrt{\frac{2}{L}} \cdot \frac{2\sqrt{3}}{L^{3/2}} \cdot \frac{2L^2}{\pi^2} = \frac{8\sqrt{6}}{\pi^2 L} \cdot L = \frac{8\sqrt{6}}{\pi^2}$$

Wait, let me recalculate more carefully. The integral:

$$\int_0^{L/2} x\sin\left(\frac{\pi x}{L}\right)dx = \left[-\frac{xL}{\pi}\cos\left(\frac{\pi x}{L}\right)\right]_0^{L/2} + \frac{L}{\pi}\int_0^{L/2}\cos\left(\frac{\pi x}{L}\right)dx$$

$$= -\frac{L^2}{2\pi}\cos\left(\frac{\pi}{2}\right) + \frac{L^2}{\pi^2}\sin\left(\frac{\pi x}{L}\right)\Big|_0^{L/2} = 0 + \frac{L^2}{\pi^2}$$

By symmetry, the second integral gives the same, so:

$$c_1 = \sqrt{\frac{2}{L}} \cdot \frac{2\sqrt{3}}{L^{3/2}} \cdot \frac{2L^2}{\pi^2} = \frac{8\sqrt{6}}{\pi^2}$$

$$\boxed{c_1 \approx 0.988}$$

Similarly, $c_2 = 0$, $c_3 = 8\sqrt{6}/(9\pi^2) \approx 0.110$, $c_5 = 8\sqrt{6}/(25\pi^2) \approx 0.040$.

---

## Practice Problems

### Level 1: Direct Application

1. **Normalization check:** Show that $\int_0^L |\psi_5(x)|^2 dx = 1$ by explicit integration.

2. **Orthogonality:** Compute $\langle\psi_2|\psi_4\rangle$ and verify it equals zero.

3. **Coefficient calculation:** If $\Psi(x) = 0.6\psi_1(x) + 0.8\psi_2(x)$, find the probabilities of measuring $E_1$ and $E_2$.

4. **Node counting:** Draw $\psi_4(x)$ and mark all nodes.

### Level 2: Intermediate

5. **Expansion coefficients:** Find $c_1$, $c_2$, and $c_3$ for the initial state $\Psi(x,0) = \psi_1(x)$ (trivial but instructive!).

6. **Delta function approximation:** A particle is initially localized near $x = L/4$: $\Psi(x,0) \approx \sqrt{L}\delta(x - L/4)$. Find $c_n$.

7. **Parity analysis:** For the symmetric well $-L/2 < x < L/2$, write the first four eigenfunctions and classify their parity.

8. **Parseval's theorem:** For the expansion $\Psi = c_1\psi_1 + c_2\psi_2$ with $|c_1|^2 = 0.36$ and normalization, find $|c_2|^2$.

### Level 3: Challenging

9. **Half-well initial state:** A particle starts in:
   $$\Psi(x,0) = \begin{cases} \sqrt{2/L} & 0 < x < L/2 \\ 0 & L/2 < x < L \end{cases}$$
   Calculate $c_n$ and verify $\sum_n |c_n|^2 = 1$ (numerically to 3-4 terms).

10. **Completeness proof:** Using the identity $\sum_{n=1}^{\infty} \frac{\sin(na)\sin(nb)}{n^2}$, prove the completeness relation for the ISW.

11. **Ground state dominance:** Show that for a slowly-varying initial state, $|c_1|^2$ is typically the largest coefficient.

12. **Truncation error:** If you approximate $\Psi \approx \sum_{n=1}^{N} c_n \psi_n$, estimate the error in $\langle E \rangle$ for the triangular wave with $N = 5$.

---

## Computational Lab

### Exercise 1: Visualizing Eigenfunctions

```python
"""
Day 373 Computational Lab: ISW Eigenfunctions
Visualize normalized eigenfunctions and verify orthonormality
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Well parameters
L = 1.0  # Work in units where L = 1

def psi_n(x, n, L=1.0):
    """Normalized eigenfunction for ISW"""
    return np.sqrt(2/L) * np.sin(n * np.pi * x / L)

# Create figure with eigenfunctions
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

x = np.linspace(0, L, 500)

# Plot first 4 eigenfunctions
for n in range(1, 5):
    ax = axes[(n-1)//2, (n-1)%2]
    psi = psi_n(x, n, L)
    ax.plot(x, psi, 'b-', linewidth=2, label=f'$\\psi_{n}(x)$')
    ax.fill_between(x, 0, psi, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)

    # Mark nodes
    nodes = [k*L/n for k in range(1, n)]
    for node in nodes:
        ax.axvline(x=node, color='red', linestyle='--', alpha=0.5)
        ax.plot(node, 0, 'ro', markersize=8)

    ax.set_xlabel('x/L', fontsize=11)
    ax.set_ylabel(f'$\\psi_{n}(x)$', fontsize=11)
    ax.set_title(f'n = {n}: {n-1} nodes, {"even" if n%2==1 else "odd"} parity', fontsize=12)
    ax.legend(loc='upper right')
    ax.set_xlim([0, L])
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('isw_eigenfunctions.png', dpi=150, bbox_inches='tight')
plt.show()

# Verify orthonormality
print("\nOrthonormality Verification:")
print("="*50)

for m in range(1, 5):
    for n in range(1, 5):
        def integrand(x):
            return psi_n(x, m, L) * psi_n(x, n, L)

        result, error = quad(integrand, 0, L)
        expected = 1 if m == n else 0
        print(f"<ψ_{m}|ψ_{n}> = {result:8.5f}  (expected: {expected})")
    print()
```

### Exercise 2: Probability Density and Nodes

```python
"""
Compare probability densities for different eigenstates
"""

import numpy as np
import matplotlib.pyplot as plt

L = 1.0
x = np.linspace(0, L, 1000)

fig, ax = plt.subplots(figsize=(12, 6))

colors = plt.cm.viridis(np.linspace(0, 0.8, 6))

for n in range(1, 7):
    psi = np.sqrt(2/L) * np.sin(n * np.pi * x / L)
    prob_density = psi**2

    # Offset for visualization
    offset = (n - 1) * 2
    ax.plot(x, prob_density + offset, color=colors[n-1], linewidth=2,
            label=f'n={n}')
    ax.fill_between(x, offset, prob_density + offset, color=colors[n-1], alpha=0.3)
    ax.axhline(y=offset, color='gray', linestyle='-', alpha=0.3)
    ax.text(-0.05, offset + 1, f'n={n}', fontsize=10, va='center')

ax.set_xlabel('x/L', fontsize=12)
ax.set_ylabel('$|\\psi_n(x)|^2$ (offset for clarity)', fontsize=12)
ax.set_title('Probability Densities for ISW Eigenstates', fontsize=14)
ax.set_xlim([-0.1, 1.1])
ax.set_ylim([-0.5, 13])

# Classical comparison: uniform distribution
ax.text(0.5, 12.5, 'Classical: uniform $|\\psi|^2 = 1/L$', fontsize=10,
        ha='center', color='gray')

plt.tight_layout()
plt.savefig('isw_probability_densities.png', dpi=150, bbox_inches='tight')
plt.show()

# Average probability density analysis
print("\nProbability at x = L/2 for each eigenstate:")
print("-"*40)
for n in range(1, 7):
    prob_at_center = 2/L * (np.sin(n * np.pi * 0.5))**2
    print(f"n = {n}: |ψ_{n}(L/2)|² = {prob_at_center:.4f}")
```

### Exercise 3: Fourier Expansion

```python
"""
Expand an arbitrary function in the ISW eigenbasis
Demonstrate completeness
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

L = 1.0

def psi_n(x, n):
    """Normalized eigenfunction"""
    return np.sqrt(2/L) * np.sin(n * np.pi * x / L)

def compute_cn(f, n):
    """Compute expansion coefficient c_n = <ψ_n|f>"""
    def integrand(x):
        return psi_n(x, n) * f(x)
    result, _ = quad(integrand, 0, L)
    return result

def reconstruct(x, coefficients):
    """Reconstruct function from expansion coefficients"""
    result = np.zeros_like(x)
    for n, cn in enumerate(coefficients, start=1):
        result += cn * psi_n(x, n)
    return result

# Define test function: triangular
def triangular(x):
    A = np.sqrt(12/L**3)  # normalization
    return np.where(x <= L/2, A*x, A*(L-x))

# Compute expansion coefficients
n_max = 20
coefficients = [compute_cn(triangular, n) for n in range(1, n_max + 1)]

print("Expansion coefficients for triangular wave:")
print("="*50)
for n, cn in enumerate(coefficients[:10], start=1):
    prob = cn**2
    print(f"c_{n:2d} = {cn:+8.5f},  |c_{n}|² = {prob:.5f}")

# Check normalization
total_prob = sum(c**2 for c in coefficients)
print(f"\nΣ|cₙ|² = {total_prob:.6f}  (should be 1)")

# Visualize reconstruction
x = np.linspace(0, L, 500)
f_original = triangular(x)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: reconstructions with different N
ax1 = axes[0]
ax1.plot(x, f_original, 'k-', linewidth=2, label='Original $f(x)$')

for N in [1, 3, 5, 15]:
    f_approx = reconstruct(x, coefficients[:N])
    ax1.plot(x, f_approx, '--', linewidth=1.5, label=f'N = {N} terms')

ax1.set_xlabel('x/L', fontsize=12)
ax1.set_ylabel('f(x)', fontsize=12)
ax1.set_title('Fourier-Sine Series Reconstruction', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right: coefficient magnitudes
ax2 = axes[1]
n_vals = np.arange(1, n_max + 1)
ax2.bar(n_vals, [abs(c) for c in coefficients], color='steelblue', alpha=0.7)
ax2.set_xlabel('n', fontsize=12)
ax2.set_ylabel('$|c_n|$', fontsize=12)
ax2.set_title('Expansion Coefficient Magnitudes', fontsize=12)
ax2.set_xticks(n_vals[::2])

# Note odd-n dominance
ax2.annotate('Only odd n\ncontribute\n(parity)', xy=(3, coefficients[2]),
             xytext=(8, 0.6), fontsize=10,
             arrowprops=dict(arrowstyle='->', color='red'))

plt.tight_layout()
plt.savefig('isw_fourier_expansion.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## Summary

### Key Formulas

| Quantity | Formula |
|----------|---------|
| Normalized eigenfunctions | $\psi_n(x) = \sqrt{2/L}\sin(n\pi x/L)$ |
| Orthonormality | $\langle\psi_m|\psi_n\rangle = \delta_{mn}$ |
| Completeness | $\sum_n |\psi_n\rangle\langle\psi_n| = \mathbb{1}$ |
| Expansion coefficients | $c_n = \langle\psi_n|f\rangle = \int_0^L \psi_n^* f\,dx$ |
| Parseval's theorem | $\int|f|^2 dx = \sum_n |c_n|^2$ |
| Measurement probability | $P(E_n) = |c_n|^2$ |

### Main Takeaways

1. **Normalization factor $\sqrt{2/L}$** ensures $\int|\psi_n|^2 dx = 1$

2. **Orthonormality** $\langle\psi_m|\psi_n\rangle = \delta_{mn}$ follows from the trigonometric product formula

3. **Completeness** means any boundary-respecting function can be expanded as $f = \sum c_n \psi_n$

4. **Coefficients $c_n$** give probabilities $|c_n|^2$ for energy measurements

5. **Parity** (about the well center) determines which coefficients are non-zero

6. The ISW eigenfunctions form a **Fourier sine series** basis

---

## Daily Checklist

- [ ] I can derive the normalization constant $\sqrt{2/L}$
- [ ] I can prove orthonormality by explicit integration
- [ ] I understand the completeness relation and its physical meaning
- [ ] I can compute expansion coefficients for given initial states
- [ ] I know how parity affects which $c_n$ vanish
- [ ] I can relate $|c_n|^2$ to measurement probabilities
- [ ] I understand the connection to Fourier series
- [ ] I completed the computational lab exercises

---

## Preview: Day 374

Tomorrow we explore **time evolution** in the infinite square well. Starting from an initial state:

$$\Psi(x, 0) = \sum_n c_n \psi_n(x)$$

The time evolution is:

$$\Psi(x, t) = \sum_n c_n \psi_n(x) e^{-iE_n t/\hbar}$$

Key phenomena we'll discover:
- **Quantum revivals**: The wave function periodically returns to its initial shape
- **Revival time**: $T_{\text{rev}} = 4mL^2/(\pi\hbar)$
- **Fractional revivals**: Partial reconstructions at rational fractions of $T_{\text{rev}}$

We'll also visualize the time-dependent probability density and expectation values.

---

*Day 373 of QSE Self-Study Curriculum*
*Week 54: Bound States - Infinite and Finite Wells*
*Month 14: One-Dimensional Systems*
