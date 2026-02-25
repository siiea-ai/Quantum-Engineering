# Day 20: L'Hôpital's Rule and Newton's Method Lab

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | L'Hôpital's Rule |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Newton's Method |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Python Lab |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Identify indeterminate forms
2. Apply L'Hôpital's Rule correctly
3. Understand Newton's Method for root finding
4. Implement Newton's Method in Python
5. Analyze convergence of Newton's Method

---

## 📚 Required Reading

### Primary Text: Stewart's Calculus (8th Edition)
- **Section 4.4**: Indeterminate Forms and L'Hôpital's Rule (pp. 307-316)
- **Section 4.8**: Newton's Method (pp. 341-346)

---

## 📖 Part I: L'Hôpital's Rule

### 1. Indeterminate Forms

When evaluating limits, we may encounter **indeterminate forms**:
- Type 0/0: lim f(x)/g(x) where both → 0
- Type ∞/∞: lim f(x)/g(x) where both → ∞
- Other forms: 0·∞, ∞-∞, 0⁰, 1^∞, ∞⁰

### 2. L'Hôpital's Rule

> **Theorem (L'Hôpital's Rule):** Suppose f and g are differentiable near a (except possibly at a), and g'(x) ≠ 0 near a. If
> $$\lim_{x \to a} f(x) = \lim_{x \to a} g(x) = 0 \quad \text{or} \quad \pm\infty$$
> then
> $$\lim_{x \to a} \frac{f(x)}{g(x)} = \lim_{x \to a} \frac{f'(x)}{g'(x)}$$
> provided the limit on the right exists (or is ±∞).

**Key Points:**
- Only applies to 0/0 or ∞/∞ forms
- Differentiate numerator AND denominator SEPARATELY (not quotient rule!)
- May need to apply multiple times
- Works for x → a, x → ∞, x → -∞, and one-sided limits

### 3. Examples

**Example 1:** $\lim_{x \to 0} \frac{\sin x}{x}$ (Type 0/0)

$$= \lim_{x \to 0} \frac{\cos x}{1} = \cos(0) = 1$$

**Example 2:** $\lim_{x \to \infty} \frac{e^x}{x^2}$ (Type ∞/∞)

$$= \lim_{x \to \infty} \frac{e^x}{2x} = \lim_{x \to \infty} \frac{e^x}{2} = \infty$$

**Example 3:** $\lim_{x \to 0} \frac{1 - \cos x}{x^2}$ (Type 0/0)

$$= \lim_{x \to 0} \frac{\sin x}{2x} = \lim_{x \to 0} \frac{\cos x}{2} = \frac{1}{2}$$

### 4. Converting Other Indeterminate Forms

**Type 0·∞:** Convert to 0/0 or ∞/∞
$$\lim_{x \to 0^+} x \ln x = \lim_{x \to 0^+} \frac{\ln x}{1/x} = \lim_{x \to 0^+} \frac{1/x}{-1/x^2} = \lim_{x \to 0^+} (-x) = 0$$

**Type ∞-∞:** Combine fractions
$$\lim_{x \to 0} \left(\frac{1}{x} - \frac{1}{\sin x}\right) = \lim_{x \to 0} \frac{\sin x - x}{x \sin x}$$

**Type 1^∞, 0⁰, ∞⁰:** Take logarithm
$$y = f(x)^{g(x)} \implies \ln y = g(x) \ln f(x)$$

**Example 4:** $\lim_{x \to 0^+} x^x$ (Type 0⁰)

Let y = xˣ, then ln y = x ln x

From above: lim(x→0⁺) x ln x = 0

So lim(x→0⁺) ln y = 0, thus lim(x→0⁺) y = e⁰ = 1

---

## 📖 Part II: Newton's Method

### 5. The Idea

Newton's Method finds roots of f(x) = 0 by iteratively improving an estimate using tangent lines.

Starting with x₀, generate:
$$x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}$$

### 6. Geometric Interpretation

1. Start at point (xₙ, f(xₙ)) on the curve
2. Draw tangent line at this point
3. Find where tangent crosses x-axis → this is xₙ₊₁
4. Repeat

### 7. Derivation

Tangent line at (xₙ, f(xₙ)):
$$y - f(x_n) = f'(x_n)(x - x_n)$$

Set y = 0:
$$-f(x_n) = f'(x_n)(x - x_n)$$
$$x = x_n - \frac{f(x_n)}{f'(x_n)}$$

### 8. Example: Finding √2

Find root of f(x) = x² - 2.

f'(x) = 2x

Newton's formula: $x_{n+1} = x_n - \frac{x_n^2 - 2}{2x_n} = \frac{x_n^2 + 2}{2x_n} = \frac{x_n + 2/x_n}{2}$

Starting with x₀ = 1:
- x₁ = (1 + 2)/2 = 1.5
- x₂ = (1.5 + 2/1.5)/2 = 1.4167
- x₃ = 1.4142157
- x₄ = 1.4142136 (√2 to 7 decimal places!)

### 9. Convergence

Newton's Method typically converges **quadratically** (doubles correct digits each iteration) when:
- f'(x) ≠ 0 at the root
- Initial guess is "close enough"

**Failures:**
- f'(xₙ) = 0 (horizontal tangent)
- Poor initial guess → divergence or wrong root
- Cycling behavior

---

## 🖥️ Python Lab: Newton's Method

### Lab 1: Basic Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

def newton_method(f, f_prime, x0, tol=1e-10, max_iter=100):
    """
    Newton's Method for finding roots.
    
    Parameters:
    - f: function to find root of
    - f_prime: derivative of f
    - x0: initial guess
    - tol: tolerance for convergence
    - max_iter: maximum iterations
    
    Returns:
    - root estimate, list of iterations
    """
    x = x0
    iterations = [x0]
    
    for i in range(max_iter):
        fx = f(x)
        fpx = f_prime(x)
        
        if abs(fpx) < 1e-15:
            print(f"Derivative too small at iteration {i}")
            return x, iterations
        
        x_new = x - fx / fpx
        iterations.append(x_new)
        
        if abs(x_new - x) < tol:
            print(f"Converged in {i+1} iterations")
            return x_new, iterations
        
        x = x_new
    
    print("Max iterations reached")
    return x, iterations

# Example: Find √2 (root of x² - 2)
f = lambda x: x**2 - 2
f_prime = lambda x: 2*x

root, iters = newton_method(f, f_prime, 1.0)
print(f"Root: {root}")
print(f"Actual √2: {np.sqrt(2)}")
print(f"Iterations: {iters}")
```

### Lab 2: Visualization

```python
def visualize_newton(f, f_prime, x0, x_range, n_iters=5):
    """
    Visualize Newton's Method iterations.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot function
    x = np.linspace(x_range[0], x_range[1], 1000)
    ax.plot(x, f(x), 'b-', linewidth=2, label='f(x)')
    ax.axhline(y=0, color='k', linewidth=0.5)
    
    # Perform iterations
    xn = x0
    colors = plt.cm.rainbow(np.linspace(0, 1, n_iters))
    
    for i in range(n_iters):
        fxn = f(xn)
        fpxn = f_prime(xn)
        
        # Plot point on curve
        ax.plot(xn, fxn, 'o', color=colors[i], markersize=10)
        
        # Plot tangent line
        x_tangent = np.linspace(xn - 1, xn + 1, 100)
        y_tangent = fxn + fpxn * (x_tangent - xn)
        ax.plot(x_tangent, y_tangent, '--', color=colors[i], 
                alpha=0.7, label=f'Iter {i}: x={xn:.4f}')
        
        # Update xn
        xn_new = xn - fxn / fpxn
        
        # Draw vertical line to x-axis, then horizontal to curve
        ax.plot([xn, xn], [fxn, 0], ':', color=colors[i], alpha=0.5)
        ax.plot([xn, xn_new], [0, 0], ':', color=colors[i], alpha=0.5)
        
        xn = xn_new
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title("Newton's Method Visualization", fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 10)
    
    plt.tight_layout()
    plt.savefig('newton_method_visualization.png', dpi=150)
    plt.show()

# Visualize for f(x) = x³ - 2x - 5
f = lambda x: x**3 - 2*x - 5
f_prime = lambda x: 3*x**2 - 2
visualize_newton(f, f_prime, 3, (0, 4), n_iters=4)
```

### Lab 3: L'Hôpital's Rule with SymPy

```python
from sympy import *

x = Symbol('x')

# Example limits using L'Hôpital's Rule
print("L'Hôpital's Rule Examples")
print("=" * 50)

# 0/0 form
expr1 = sin(x) / x
print(f"\nlim(x→0) sin(x)/x = {limit(expr1, x, 0)}")

# ∞/∞ form
expr2 = exp(x) / x**3
print(f"lim(x→∞) e^x/x³ = {limit(expr2, x, oo)}")

# 0·∞ form (converted)
expr3 = x * ln(x)
print(f"lim(x→0⁺) x·ln(x) = {limit(expr3, x, 0, '+')}")

# 1^∞ form
expr4 = (1 + 1/x)**x
print(f"lim(x→∞) (1 + 1/x)^x = {limit(expr4, x, oo)}")

# Complex limit
expr5 = (1 - cos(x)) / x**2
print(f"lim(x→0) (1-cos(x))/x² = {limit(expr5, x, 0)}")
```

---

## 📝 Practice Problems

### L'Hôpital's Rule
1. $\lim_{x \to 0} \frac{e^x - 1}{x}$
2. $\lim_{x \to 0} \frac{\tan x - x}{x^3}$
3. $\lim_{x \to \infty} \frac{\ln x}{\sqrt{x}}$
4. $\lim_{x \to 0^+} x^{\sin x}$
5. $\lim_{x \to 1} \frac{x^3 - 1}{x - 1}$ (also solvable by factoring)

### Newton's Method
6. Use Newton's Method to find √5 starting from x₀ = 2 (3 iterations)
7. Find a root of x³ - 2x - 5 = 0 using Newton's Method
8. Why might Newton's Method fail for f(x) = x^(1/3) starting at x₀ = 1?

---

## 📊 Answers

1. 1
2. 1/3
3. 0
4. 1
5. 3
6. x₁ = 2.25, x₂ = 2.2361, x₃ = 2.2360679...
7. x ≈ 2.0946
8. f'(x) = (1/3)x^(-2/3) → Newton's formula gives x_{n+1} = -2x_n, which diverges

---

## ✅ Daily Checklist

- [ ] Read Stewart 4.4 and 4.8
- [ ] Master L'Hôpital's Rule conditions
- [ ] Understand all indeterminate form conversions
- [ ] Derive Newton's Method formula
- [ ] Implement Newton's Method in Python
- [ ] Visualize convergence behavior
- [ ] Complete practice problems

---

## 🔜 Tomorrow: Rest and Review

Day 21 completes Week 3 with review and preparation for Integration.

---

*"Newton's method is the most powerful general method for root-finding."*
