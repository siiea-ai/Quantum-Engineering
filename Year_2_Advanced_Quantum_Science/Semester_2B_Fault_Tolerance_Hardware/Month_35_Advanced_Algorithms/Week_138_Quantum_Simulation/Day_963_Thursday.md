# Day 963: Quantum Signal Processing

## Schedule Overview

| Time Block | Duration | Focus |
|------------|----------|-------|
| Morning | 3.5 hours | QSP fundamentals and polynomial transformations |
| Afternoon | 2.5 hours | Problem solving and phase angle computation |
| Evening | 1 hour | Computational lab: QSP implementation |

## Learning Objectives

By the end of today, you will be able to:

1. Define the quantum signal processing framework and its key components
2. Explain how QSP encodes polynomial transformations of matrix elements
3. Derive the QSP theorem connecting phase angles to polynomial functions
4. Apply QSP to implement specific polynomial transformations
5. Connect QSP to Quantum Singular Value Transformation (QSVT)
6. Understand how QSP achieves optimal Hamiltonian simulation complexity

## Core Content

### 1. The QSP Revolution

**Quantum Signal Processing (QSP)** represents a paradigm shift in quantum algorithms. Rather than building circuits through intuition, QSP provides a **systematic framework** for implementing any polynomial transformation of matrix elements.

**Key insight:** Many quantum algorithms can be understood as polynomial transformations of eigenvalues or singular values.

| Algorithm | Polynomial Transformation |
|-----------|--------------------------|
| Phase estimation | $\text{sinc}(x)$ for amplitude estimation |
| Hamiltonian simulation | $e^{ix} \approx \sum_k c_k T_k(x)$ (Chebyshev) |
| Matrix inversion (HHL) | $1/x$ (inverse function) |
| Ground state preparation | Sign function: $\text{sign}(x - \lambda_0)$ |

---

### 2. The Signal Rotation Operator

QSP works with a **signal rotation operator** that encodes a real parameter $a \in [-1, 1]$:

$$\boxed{W(a) = \begin{pmatrix} a & i\sqrt{1-a^2} \\ i\sqrt{1-a^2} & a \end{pmatrix}}$$

This can be written as:

$$W(a) = e^{i \arccos(a) X} = \begin{pmatrix} \cos\theta & i\sin\theta \\ i\sin\theta & \cos\theta \end{pmatrix}$$

where $a = \cos\theta$.

**Properties of $W(a)$:**
- Unitary for all $a \in [-1, 1]$
- Eigenvalues: $e^{\pm i\arccos(a)}$
- Encodes the "signal" $a$ in its matrix structure

---

### 3. The QSP Sequence

A **QSP sequence** interleaves signal operators with Z-rotations:

$$\boxed{U_{\vec{\phi}}(a) = e^{i\phi_0 Z} \prod_{j=1}^{d} W(a) \cdot e^{i\phi_j Z}}$$

Expanded:

$$U_{\vec{\phi}}(a) = e^{i\phi_0 Z} W(a) e^{i\phi_1 Z} W(a) e^{i\phi_2 Z} \cdots W(a) e^{i\phi_d Z}$$

Here:
- $\vec{\phi} = (\phi_0, \phi_1, \ldots, \phi_d)$ are the **phase angles**
- $d$ is the **degree** of the polynomial
- The result is a $2 \times 2$ unitary depending on $a$

---

### 4. The QSP Theorem

**Theorem (Low, Chuang, Yoder, 2016):**

For any polynomial $P(x)$ of degree $d$ satisfying:
1. $|P(x)| \leq 1$ for all $x \in [-1, 1]$
2. $P$ has definite parity (even or odd)
3. For even $P$: $|P(x)|^2 + (1-x^2)|Q(x)|^2 = 1$ for some polynomial $Q$
4. For odd $P$: $|P(x)|^2 + (1-x^2)|Q(x)|^2 \leq 1$

There exist phase angles $\vec{\phi}$ such that:

$$\boxed{U_{\vec{\phi}}(a) = \begin{pmatrix} P(a) & \cdot \\ \cdot & \cdot \end{pmatrix}}$$

(The $\cdot$ entries depend on the complementary polynomial.)

**Converse:** Any QSP sequence produces a polynomial in the top-left entry.

---

### 5. Polynomial Families for Simulation

For Hamiltonian simulation $e^{-iHt}$, we need the polynomial:

$$P(x) \approx e^{-ixt}$$

on the eigenvalue range.

#### Jacobi-Anger Expansion

$$e^{-it\cos\theta} = \sum_{k=-\infty}^{\infty} (-i)^k J_k(t) e^{ik\theta}$$

where $J_k$ are Bessel functions.

#### Chebyshev Approximation

For $x \in [-1, 1]$:

$$e^{-ixt} = \sum_{k=0}^{\infty} c_k(t) T_k(x)$$

where $T_k$ are Chebyshev polynomials and:

$$c_k(t) = \frac{2-\delta_{k0}}{\pi} \int_{-1}^{1} \frac{e^{-ixt} T_k(x)}{\sqrt{1-x^2}} dx$$

**Truncation:** A degree-$d$ polynomial with $d = O(t + \log(1/\epsilon))$ achieves error $\epsilon$.

---

### 6. Computing Phase Angles

Given a target polynomial $P(x)$, finding the phase angles $\vec{\phi}$ is a **non-trivial computational problem**.

#### The Phase Finding Algorithm

1. **Input:** Polynomial $P(x)$ of degree $d$
2. **Factorization:** Write $P(x) = \prod_j (x - r_j)$ over roots
3. **Recursive construction:** Build phases from lower degree
4. **Output:** Phase vector $\vec{\phi} = (\phi_0, \ldots, \phi_d)$

**Complexity:** The algorithm runs in $O(d \log d)$ classical time.

#### Numerical Methods

For general polynomials, optimization-based methods work:

$$\min_{\vec{\phi}} \|P_{\text{QSP}}(\vec{\phi}, x) - P_{\text{target}}(x)\|$$

Libraries like `pyqsp` and `QSVT` provide implementations.

---

### 7. From QSP to QSVT

**Quantum Singular Value Transformation (QSVT)** extends QSP from scalars to matrices.

Given a block-encoded matrix $A$ (see Day 964), QSVT applies a polynomial $P$ to singular values:

$$\boxed{A = \sum_j \sigma_j |u_j\rangle\langle v_j| \quad \xrightarrow{\text{QSVT}} \quad \sum_j P(\sigma_j) |u_j\rangle\langle v_j|}$$

**Key result (Gilyen et al., 2019):** QSVT unifies:
- Hamiltonian simulation
- Amplitude estimation
- Matrix inversion
- Quantum walks
- Many other quantum algorithms

---

### 8. QSP for Hamiltonian Simulation

The connection to simulation works as follows:

1. **Block-encode** the Hamiltonian $H$ with normalization $\alpha$:
   $$\langle 0|^{\otimes a} U_H |0\rangle^{\otimes a} = H/\alpha$$

2. **Apply QSVT** with polynomial approximating $e^{-ix\alpha t}$:
   $$P(x) \approx e^{-ix\alpha t}, \quad x \in [-1, 1]$$

3. **Resulting operator:**
   $$U_{\text{sim}} \approx e^{-iHt}$$

**Complexity:**

$$\boxed{O\left(\alpha t + \frac{\log(1/\epsilon)}{\log\log(1/\epsilon)}\right)}$$

This is **optimal** up to sub-logarithmic factors!

---

### 9. Comparison with Product Formulas

| Aspect | Product Formulas | QSP/QSVT |
|--------|------------------|----------|
| Query complexity | $O((\alpha t)^{1+o(1)})$ | $O(\alpha t + \log(1/\epsilon))$ |
| Precision dependence | Polynomial in $1/\epsilon$ | Logarithmic in $1/\epsilon$ |
| Circuit structure | Repeated layers | Single sequence |
| Classical preprocessing | Minimal | Phase angle computation |
| Ancilla requirement | None | Block encoding overhead |
| Implementation complexity | Simple | Complex |

**Trade-off:** QSP achieves better asymptotic scaling but requires more sophisticated compilation.

---

## Worked Examples

### Example 1: Linear Function in QSP

**Problem:** Find QSP phases for the polynomial $P(x) = x$ (identity).

**Solution:**

Step 1: Recognize the structure.
We need $U_{\vec{\phi}}(a)$ to have top-left entry $a$.

Step 2: Try the simplest sequence.
For $d = 1$, we have:
$$U_{\vec{\phi}} = e^{i\phi_0 Z} W(a) e^{i\phi_1 Z}$$

Step 3: Compute the matrix product.

$$e^{i\phi Z} = \begin{pmatrix} e^{i\phi} & 0 \\ 0 & e^{-i\phi} \end{pmatrix}$$

$$W(a) = \begin{pmatrix} a & i\sqrt{1-a^2} \\ i\sqrt{1-a^2} & a \end{pmatrix}$$

$$U_{\vec{\phi}} = \begin{pmatrix} e^{i\phi_0} & 0 \\ 0 & e^{-i\phi_0} \end{pmatrix} \begin{pmatrix} a & i\sqrt{1-a^2} \\ i\sqrt{1-a^2} & a \end{pmatrix} \begin{pmatrix} e^{i\phi_1} & 0 \\ 0 & e^{-i\phi_1} \end{pmatrix}$$

The (1,1) entry:
$$[U]_{11} = e^{i\phi_0}(a \cdot e^{i\phi_1} + i\sqrt{1-a^2} \cdot 0 \cdot e^{-i\phi_1}) = a \cdot e^{i(\phi_0 + \phi_1)}$$

Wait, this gives $ae^{i(\phi_0 + \phi_1)}$, not $a$.

Step 4: Actually, $P(x) = x$ needs even phases.
Set $\phi_0 = 0, \phi_1 = 0$:

$$[U]_{11} = a \cdot e^{i \cdot 0} = a \checkmark$$

**Answer:** $\vec{\phi} = (0, 0)$ gives $P(x) = x$.

Note: The full matrix also has off-diagonal terms involving $\sqrt{1-a^2}$.

$\square$

---

### Example 2: Chebyshev Polynomial $T_2$

**Problem:** Find QSP phases for $P(x) = T_2(x) = 2x^2 - 1$.

**Solution:**

Step 1: Note that $T_2$ is even and degree 2, so we need $d = 2$.

The QSP sequence:
$$U = e^{i\phi_0 Z} W e^{i\phi_1 Z} W e^{i\phi_2 Z}$$

Step 2: Use the fact that $T_2(a) = 2a^2 - 1 = \cos(2\arccos(a))$.

Since $W(a) = e^{i\arccos(a) X}$, we have:
$$W^2 = e^{2i\arccos(a) X}$$

Step 3: The (1,1) entry of $W^2$ is:
$$[W^2]_{11} = \cos(2\arccos(a)) = T_2(a)$$

So actually $W^2$ already gives $T_2$!

Step 4: Relate to QSP form.
$$U = e^{i\phi_0 Z} W e^{i\phi_1 Z} W e^{i\phi_2 Z}$$

If we set $\phi_0 = \phi_1 = \phi_2 = 0$:
$$U = W \cdot W = W^2$$

And $[W^2]_{11} = T_2(a)$ ✓

**Answer:** $\vec{\phi} = (0, 0, 0)$ gives $P(x) = T_2(x)$.

This generalizes: $\vec{\phi} = (0, 0, \ldots, 0)$ with $d+1$ zeros gives $T_d(x)$.

$\square$

---

### Example 3: Approximating $e^{-ix}$ for Small $x$

**Problem:** Find QSP phases to approximate $e^{-ix}$ to first order.

**Solution:**

Step 1: Taylor expansion.
$$e^{-ix} = 1 - ix + O(x^2)$$

To first order: $P(x) \approx 1 - ix = 1 - ix$

But wait, QSP produces real polynomials in the (1,1) entry! For complex polynomials, we need a different approach.

Step 2: Use real and imaginary parts separately.
$$e^{-ix} = \cos(x) - i\sin(x)$$

Apply QSP twice:
- $P_R(x) = \cos(x) \approx 1 - x^2/2 + O(x^4)$ (even)
- $P_I(x) = \sin(x) \approx x - x^3/6 + O(x^5)$ (odd)

Step 3: For the real part $\cos(x)$ to second order:
$$P_R(x) = T_0(x) - \frac{1}{2}(T_0(x) - T_2(x))/2 = \frac{1}{2}(1 + T_2(x))$$

This can be achieved with QSP phases (computed numerically).

Step 4: Combine using linear combination of unitaries (LCU).

**Answer:** Complex time evolution requires combining QSP for real and imaginary parts using LCU techniques.

$\square$

---

## Practice Problems

### Level 1: Direct Application

1. **Signal operator:** Verify that $W(a) = e^{i\arccos(a) X}$ by computing the matrix exponential.

2. **QSP structure:** For the sequence $e^{i\phi_0 Z} W e^{i\phi_1 Z}$, compute the full $2 \times 2$ matrix as a function of $a$, $\phi_0$, and $\phi_1$.

3. **Chebyshev basics:** Express $T_3(x) = 4x^3 - 3x$ as a QSP sequence (determine $d$ and verify $\vec{\phi} = \vec{0}$).

### Level 2: Intermediate Analysis

4. **Polynomial constraints:** Verify that $P(x) = (1+x)/2$ satisfies the QSP constraints (bounded, parity, completeness).

5. **Error scaling:** For approximating $e^{-ixt}$ on $[-1, 1]$ with Chebyshev polynomials, derive the degree $d$ needed for error $\epsilon$.

6. **Phase symmetry:** Show that for real even polynomials, the QSP phases satisfy $\phi_j = \phi_{d-j}$ (palindromic symmetry).

### Level 3: Challenging Problems

7. **QSVT simulation complexity:** Prove that QSVT achieves Hamiltonian simulation with query complexity $O(\alpha t + \log(1/\epsilon))$ assuming ideal block encoding.

8. **Phase computation:** Implement the recursive phase-finding algorithm for polynomials expressible as products of $(x - r_j)$ factors.

9. **Beyond real polynomials:** Show how to implement complex polynomial transformations $P(x) + iQ(x)$ using two QSP sequences and LCU.

---

## Computational Lab: Quantum Signal Processing

### Lab Objective

Implement QSP sequences and verify polynomial transformations.

```python
"""
Day 963 Lab: Quantum Signal Processing
Week 138: Quantum Simulation
"""

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from typing import List, Tuple
from numpy.polynomial import chebyshev as cheb

# =============================================================
# Part 1: Signal Rotation Operator
# =============================================================

def signal_operator(a: float) -> np.ndarray:
    """
    Construct the signal rotation operator W(a).

    W(a) = [[a, i*sqrt(1-a^2)],
            [i*sqrt(1-a^2), a]]
    """
    sqrt_term = 1j * np.sqrt(1 - a**2 + 0j)  # +0j to handle a=±1
    return np.array([
        [a, sqrt_term],
        [sqrt_term, a]
    ], dtype=complex)

def z_rotation(phi: float) -> np.ndarray:
    """Construct e^{i*phi*Z}."""
    return np.array([
        [np.exp(1j * phi), 0],
        [0, np.exp(-1j * phi)]
    ], dtype=complex)

print("=" * 60)
print("Part 1: Signal Rotation Operator")
print("=" * 60)

# Verify W(a) properties
test_values = [0, 0.5, -0.5, 0.9, -0.9, 1.0, -1.0]

print("\nVerifying W(a) is unitary:")
for a in test_values:
    W = signal_operator(a)
    should_be_identity = W @ W.conj().T
    is_unitary = np.allclose(should_be_identity, np.eye(2))
    print(f"  a = {a:5.2f}: W W† = I? {is_unitary}")

# Verify W(a) = exp(i*arccos(a)*X)
X = np.array([[0, 1], [1, 0]])
print("\nVerifying W(a) = exp(i*arccos(a)*X):")
for a in test_values[:-2]:  # Skip ±1 (edge cases)
    W = signal_operator(a)
    theta = np.arccos(a)
    W_from_exp = expm(1j * theta * X)
    is_equal = np.allclose(W, W_from_exp)
    print(f"  a = {a:5.2f}: match? {is_equal}")

# =============================================================
# Part 2: QSP Sequence
# =============================================================

def qsp_sequence(a: float, phases: List[float]) -> np.ndarray:
    """
    Construct QSP unitary U_phi(a).

    U = e^{i*phi_0*Z} * W(a) * e^{i*phi_1*Z} * W(a) * ... * e^{i*phi_d*Z}
    """
    d = len(phases) - 1
    W = signal_operator(a)

    U = z_rotation(phases[0])
    for j in range(d):
        U = U @ W @ z_rotation(phases[j + 1])

    return U

def extract_polynomial(phases: List[float], a_values: np.ndarray) -> np.ndarray:
    """Extract the (0,0) entry of QSP sequence for multiple a values."""
    poly_values = []
    for a in a_values:
        U = qsp_sequence(a, phases)
        poly_values.append(U[0, 0])
    return np.array(poly_values)

print("\n" + "=" * 60)
print("Part 2: QSP Sequence Examples")
print("=" * 60)

# Example 1: Identity polynomial P(x) = x with phases [0, 0]
phases_linear = [0.0, 0.0]
a_test = np.linspace(-0.99, 0.99, 100)
P_linear = extract_polynomial(phases_linear, a_test)

print("\nPhases [0, 0] -> P(x):")
print(f"  P(0.5) = {extract_polynomial(phases_linear, [0.5])[0]:.6f}")
print(f"  Expected: 0.5")

# Example 2: T_2(x) = 2x^2 - 1 with phases [0, 0, 0]
phases_T2 = [0.0, 0.0, 0.0]
P_T2 = extract_polynomial(phases_T2, a_test)
T2_exact = 2 * a_test**2 - 1

print("\nPhases [0, 0, 0] -> P(x):")
print(f"  P(0.5) = {extract_polynomial(phases_T2, [0.5])[0].real:.6f}")
print(f"  T_2(0.5) = {2*0.5**2 - 1:.6f}")
print(f"  Max error: {np.max(np.abs(P_T2.real - T2_exact)):.2e}")

# Plot comparison
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(a_test, P_linear.real, 'b-', linewidth=2, label='QSP [0,0]')
plt.plot(a_test, a_test, 'r--', linewidth=2, label='x (expected)')
plt.xlabel('a', fontsize=12)
plt.ylabel('P(a)', fontsize=12)
plt.title('QSP Identity Polynomial', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(a_test, P_T2.real, 'b-', linewidth=2, label='QSP [0,0,0]')
plt.plot(a_test, T2_exact, 'r--', linewidth=2, label='$T_2(x) = 2x^2-1$')
plt.xlabel('a', fontsize=12)
plt.ylabel('P(a)', fontsize=12)
plt.title('QSP Chebyshev $T_2$', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_963_qsp_basic.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================
# Part 3: Chebyshev Polynomials via QSP
# =============================================================

print("\n" + "=" * 60)
print("Part 3: Chebyshev Polynomials via QSP")
print("=" * 60)

def chebyshev_T(n: int, x: np.ndarray) -> np.ndarray:
    """Compute Chebyshev polynomial T_n(x)."""
    return np.cos(n * np.arccos(x))

# QSP with all-zero phases gives Chebyshev polynomials
for d in [1, 2, 3, 4, 5]:
    phases = [0.0] * (d + 1)
    P_qsp = extract_polynomial(phases, a_test)
    T_d = chebyshev_T(d, a_test)
    error = np.max(np.abs(P_qsp.real - T_d))
    print(f"  d = {d}: max|QSP - T_{d}| = {error:.2e}")

# Plot Chebyshev polynomials from QSP
plt.figure(figsize=(10, 6))
for d in [1, 2, 3, 4, 5]:
    phases = [0.0] * (d + 1)
    P_qsp = extract_polynomial(phases, a_test)
    plt.plot(a_test, P_qsp.real, linewidth=2, label=f'$T_{d}$')

plt.xlabel('x', fontsize=12)
plt.ylabel('$T_n(x)$', fontsize=12)
plt.title('Chebyshev Polynomials from QSP (all-zero phases)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('day_963_chebyshev.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================
# Part 4: Non-Trivial QSP Phases
# =============================================================

print("\n" + "=" * 60)
print("Part 4: Non-Trivial QSP Phases")
print("=" * 60)

def find_qsp_phases_numerical(target_poly: callable, degree: int,
                              n_points: int = 50) -> List[float]:
    """
    Find QSP phases numerically using optimization.

    This is a simplified approach; real implementations use
    more sophisticated algorithms.
    """
    from scipy.optimize import minimize

    a_sample = np.linspace(-0.95, 0.95, n_points)
    target_values = np.array([target_poly(a) for a in a_sample])

    def objective(phases):
        qsp_values = extract_polynomial(list(phases), a_sample)
        return np.sum(np.abs(qsp_values.real - target_values)**2)

    # Initialize with small random phases
    x0 = np.random.randn(degree + 1) * 0.1
    result = minimize(objective, x0, method='BFGS', options={'maxiter': 1000})

    return list(result.x), result.fun

# Try to find phases for P(x) = (1 + x) / 2
def P_half_plus(x):
    return (1 + x) / 2

print("\nFinding phases for P(x) = (1+x)/2 (degree 1):")
phases_half, residual = find_qsp_phases_numerical(P_half_plus, 1)
print(f"  Phases: {[f'{p:.4f}' for p in phases_half]}")
print(f"  Residual: {residual:.2e}")

P_qsp_half = extract_polynomial(phases_half, a_test)
plt.figure(figsize=(10, 6))
plt.plot(a_test, P_qsp_half.real, 'b-', linewidth=2, label='QSP result')
plt.plot(a_test, (1 + a_test) / 2, 'r--', linewidth=2, label='(1+x)/2')
plt.xlabel('x', fontsize=12)
plt.ylabel('P(x)', fontsize=12)
plt.title('QSP for P(x) = (1+x)/2', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('day_963_custom_poly.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================
# Part 5: Approximating cos(x) for Simulation
# =============================================================

print("\n" + "=" * 60)
print("Part 5: Approximating cos(x*t) for Hamiltonian Simulation")
print("=" * 60)

def chebyshev_expansion_cos(t: float, max_degree: int) -> List[float]:
    """
    Compute Chebyshev expansion coefficients for cos(x*t).

    cos(x*t) = sum_{k=0}^{d} c_k T_k(x)

    Uses the relation: cos(t*cos(theta)) = sum_k (-1)^k J_{2k}(t) * T_{2k}(cos(theta))
    where J_k are Bessel functions.
    """
    from scipy.special import jv

    # cos(xt) is even, so only even Chebyshev terms
    coeffs = []
    for k in range(max_degree + 1):
        if k == 0:
            c_k = jv(0, t)
        elif k % 2 == 0:
            # For even k: coefficient involves Bessel function
            c_k = 2 * (-1)**(k//2) * jv(k, t)
        else:
            c_k = 0.0
        coeffs.append(c_k)

    return coeffs

t_sim = 2.0  # Simulation time parameter
max_d = 20

coeffs = chebyshev_expansion_cos(t_sim, max_d)

# Reconstruct the function
x_test = np.linspace(-1, 1, 200)
cos_exact = np.cos(t_sim * x_test)

cos_approx = np.zeros_like(x_test)
for k, c in enumerate(coeffs):
    cos_approx += c * chebyshev_T(k, x_test)

error = np.max(np.abs(cos_exact - cos_approx))
print(f"  t = {t_sim}, max_degree = {max_d}")
print(f"  Max approximation error: {error:.2e}")

# Plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x_test, cos_exact, 'b-', linewidth=2, label=f'$\\cos({t_sim}x)$ (exact)')
plt.plot(x_test, cos_approx, 'r--', linewidth=2, label=f'Chebyshev (d={max_d})')
plt.xlabel('x', fontsize=12)
plt.ylabel('$\\cos(tx)$', fontsize=12)
plt.title('Chebyshev Approximation for Hamiltonian Simulation', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
relevant_coeffs = [c for c in coeffs if abs(c) > 1e-10]
plt.bar(range(len(relevant_coeffs)), np.abs(relevant_coeffs))
plt.xlabel('Coefficient Index', fontsize=12)
plt.ylabel('|Coefficient|', fontsize=12)
plt.title('Chebyshev Expansion Coefficients', fontsize=14)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_963_simulation_approx.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================
# Part 6: Degree vs Precision for Simulation
# =============================================================

print("\n" + "=" * 60)
print("Part 6: Degree Required for Target Precision")
print("=" * 60)

def degree_for_precision(t: float, epsilon: float, max_search: int = 200) -> int:
    """Find minimum degree for approximating e^{-ixt} with error epsilon."""
    x_test = np.linspace(-1, 1, 500)
    target_real = np.cos(t * x_test)
    target_imag = -np.sin(t * x_test)

    for d in range(1, max_search):
        coeffs_cos = chebyshev_expansion_cos(t, d)
        approx = np.zeros_like(x_test)
        for k, c in enumerate(coeffs_cos):
            approx += c * chebyshev_T(k, x_test)

        error = np.max(np.abs(target_real - approx))
        if error < epsilon:
            return d

    return max_search

# Table of required degrees
print("\nMinimum degree for various t and epsilon:")
print("-" * 50)
print(f"{'t':>6} {'ε=1e-3':>10} {'ε=1e-6':>10} {'ε=1e-9':>10}")
print("-" * 50)

t_values = [1.0, 2.0, 5.0, 10.0, 20.0]
for t in t_values:
    d1 = degree_for_precision(t, 1e-3)
    d2 = degree_for_precision(t, 1e-6)
    d3 = degree_for_precision(t, 1e-9)
    print(f"{t:>6.1f} {d1:>10} {d2:>10} {d3:>10}")

# Observe scaling: d ~ O(t + log(1/epsilon))
print("\nObservation: d scales roughly as O(t + log(1/ε))")

# =============================================================
# Part 7: QSP vs Trotter Comparison
# =============================================================

print("\n" + "=" * 60)
print("Part 7: Asymptotic Comparison")
print("=" * 60)

def trotter_queries(t: float, epsilon: float, order: int) -> float:
    """Estimate Trotter queries (proportional to gate count)."""
    if order == 1:
        return t**2 / epsilon
    elif order == 2:
        return (t / epsilon**0.5)**1.5
    elif order == 4:
        return (t / epsilon**0.25)**1.25
    else:
        p = order
        return (t / epsilon**(1.0/p))**(1 + 1.0/p)

def qsp_queries(t: float, epsilon: float) -> float:
    """Estimate QSP/QSVT queries."""
    return t + np.log(1.0 / epsilon) / np.log(np.log(1.0 / epsilon) + 2)

t = 10.0
epsilons = np.logspace(-2, -10, 50)

plt.figure(figsize=(10, 6))

for order in [1, 2, 4]:
    queries = [trotter_queries(t, eps, order) for eps in epsilons]
    plt.loglog(epsilons, queries, linewidth=2, label=f'Trotter order {order}')

qsp_q = [qsp_queries(t, eps) for eps in epsilons]
plt.loglog(epsilons, qsp_q, 'k-', linewidth=3, label='QSP/QSVT')

plt.xlabel(r'Precision $\epsilon$', fontsize=12)
plt.ylabel('Query Complexity', fontsize=12)
plt.title(f'Query Complexity Comparison (t = {t})', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.gca().invert_xaxis()
plt.savefig('day_963_complexity.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nKey observation:")
print("  - Trotter: polynomial dependence on 1/ε")
print("  - QSP/QSVT: logarithmic dependence on 1/ε")
print("  - For high precision (small ε), QSP wins dramatically")

print("\nLab complete!")
print("Figures saved: day_963_qsp_basic.png, day_963_chebyshev.png,")
print("               day_963_custom_poly.png, day_963_simulation_approx.png,")
print("               day_963_complexity.png")
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Signal operator | $W(a) = e^{i\arccos(a)X}$ |
| QSP sequence | $U_{\vec{\phi}}(a) = e^{i\phi_0 Z} \prod_j W(a) e^{i\phi_j Z}$ |
| QSP output | $[U_{\vec{\phi}}]_{11} = P(a)$ polynomial of degree $d$ |
| Chebyshev (zero phases) | $\vec{\phi} = \vec{0}$ gives $T_d(x)$ |
| Simulation polynomial | $P(x) \approx e^{-ixt}$ requires degree $O(t + \log(1/\epsilon))$ |
| QSVT complexity | $O(\alpha t + \log(1/\epsilon)/\log\log(1/\epsilon))$ |

### Key Takeaways

1. **QSP is a systematic framework** for implementing polynomial transformations of matrix elements.

2. **Phase angles encode polynomials** through a structured product of rotations.

3. **Zero phases give Chebyshev polynomials**, which are optimal for approximating smooth functions.

4. **QSVT extends QSP to matrices**, enabling polynomial transformations of singular values.

5. **Optimal simulation complexity** is achieved through QSP/QSVT, with logarithmic precision dependence.

6. **Classical preprocessing** (finding phases) is the main practical challenge.

---

## Daily Checklist

- [ ] I can define the signal operator $W(a)$ and its key properties
- [ ] I understand the structure of QSP sequences
- [ ] I know why zero phases give Chebyshev polynomials
- [ ] I can explain how QSP achieves polynomial transformations
- [ ] I understand the connection to Hamiltonian simulation
- [ ] I completed the computational lab and verified QSP behavior

---

## Preview of Day 964

Tomorrow we dive into **Qubitization and Block Encoding**, the other half of optimal Hamiltonian simulation. We will:

- Define block encoding: how to embed a matrix in a larger unitary
- Learn the qubitization technique for constructing quantum walks
- Connect block encoding to QSP/QSVT for complete simulation algorithms
- Analyze the overhead of block encoding for local Hamiltonians
- Implement block-encoded matrix operations

Together with QSP, block encoding completes the picture of optimal quantum simulation.

---

*"The purpose of computing is insight, not numbers."*
*— Richard Hamming*

---

**Next:** [Day_964_Friday.md](Day_964_Friday.md) - Qubitization and Block Encoding
