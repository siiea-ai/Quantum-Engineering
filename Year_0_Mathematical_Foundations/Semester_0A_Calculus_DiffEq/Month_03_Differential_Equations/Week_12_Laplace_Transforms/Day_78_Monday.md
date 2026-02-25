# Day 78: Introduction to Laplace Transforms

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Definition & Basic Transforms |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Properties |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Practice |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Understand the definition and motivation for Laplace transforms
2. Compute Laplace transforms from the definition
3. Use the table of standard transforms
4. Apply linearity and shifting properties
5. Connect transforms to solving differential equations

---

## 📚 Required Reading

### Primary Text: Boyce & DiPrima (11th Edition)
- **Section 6.1**: Definition of the Laplace Transform (pp. 293-303)
- **Section 6.2**: Solution of Initial Value Problems (pp. 304-315)

---

## 📖 Core Content: What is the Laplace Transform?

### 1. Motivation

**Problem:** Solving ODEs with:
- Discontinuous forcing (switches, impulses)
- Complicated initial conditions
- Systems where algebraic methods would be easier

**Solution:** Transform the ODE into an algebraic equation!

### 2. Definition

> **Definition:** The Laplace transform of $f(t)$ is:
> $$\mathcal{L}\{f(t)\} = F(s) = \int_0^\infty f(t)e^{-st} dt$$
> for all $s$ where the integral converges.

**Key idea:** Transform from **time domain** $f(t)$ to **frequency domain** $F(s)$.

### 3. Existence

The Laplace transform exists if:
1. $f(t)$ is piecewise continuous on $[0, \infty)$
2. $f(t)$ is of **exponential order**: $|f(t)| \leq Me^{ct}$ for large $t$

---

## ✏️ Computing Basic Transforms

### Example 1: $\mathcal{L}\{1\}$

$$\mathcal{L}\{1\} = \int_0^\infty e^{-st} dt = \left[-\frac{1}{s}e^{-st}\right]_0^\infty = \frac{1}{s}$$

**Result:** $\mathcal{L}\{1\} = \frac{1}{s}$, for $s > 0$

---

### Example 2: $\mathcal{L}\{e^{at}\}$

$$\mathcal{L}\{e^{at}\} = \int_0^\infty e^{at}e^{-st} dt = \int_0^\infty e^{-(s-a)t} dt = \frac{1}{s-a}$$

**Result:** $\mathcal{L}\{e^{at}\} = \frac{1}{s-a}$, for $s > a$

---

### Example 3: $\mathcal{L}\{t\}$

$$\mathcal{L}\{t\} = \int_0^\infty te^{-st} dt$$

Using integration by parts ($u = t$, $dv = e^{-st}dt$):

$$= \left[-\frac{t}{s}e^{-st}\right]_0^\infty + \frac{1}{s}\int_0^\infty e^{-st} dt = 0 + \frac{1}{s} \cdot \frac{1}{s} = \frac{1}{s^2}$$

**Result:** $\mathcal{L}\{t\} = \frac{1}{s^2}$

---

### Example 4: $\mathcal{L}\{t^n\}$ (General)

By repeated integration by parts:

$$\mathcal{L}\{t^n\} = \frac{n!}{s^{n+1}}$$

---

### Example 5: $\mathcal{L}\{\sin(\omega t)\}$

$$\mathcal{L}\{\sin(\omega t)\} = \int_0^\infty \sin(\omega t)e^{-st} dt$$

Using $\sin(\omega t) = \frac{e^{i\omega t} - e^{-i\omega t}}{2i}$ or integration by parts twice:

$$\mathcal{L}\{\sin(\omega t)\} = \frac{\omega}{s^2 + \omega^2}$$

Similarly: $\mathcal{L}\{\cos(\omega t)\} = \frac{s}{s^2 + \omega^2}$

---

## 📋 Table of Standard Laplace Transforms

| $f(t)$ | $F(s) = \mathcal{L}\{f(t)\}$ | Condition |
|--------|------------------------------|-----------|
| $1$ | $\frac{1}{s}$ | $s > 0$ |
| $t$ | $\frac{1}{s^2}$ | $s > 0$ |
| $t^n$ | $\frac{n!}{s^{n+1}}$ | $s > 0$ |
| $e^{at}$ | $\frac{1}{s-a}$ | $s > a$ |
| $\sin(\omega t)$ | $\frac{\omega}{s^2+\omega^2}$ | $s > 0$ |
| $\cos(\omega t)$ | $\frac{s}{s^2+\omega^2}$ | $s > 0$ |
| $\sinh(at)$ | $\frac{a}{s^2-a^2}$ | $s > |a|$ |
| $\cosh(at)$ | $\frac{s}{s^2-a^2}$ | $s > |a|$ |
| $t^n e^{at}$ | $\frac{n!}{(s-a)^{n+1}}$ | $s > a$ |
| $e^{at}\sin(\omega t)$ | $\frac{\omega}{(s-a)^2+\omega^2}$ | $s > a$ |
| $e^{at}\cos(\omega t)$ | $\frac{s-a}{(s-a)^2+\omega^2}$ | $s > a$ |

---

## 📖 Properties of Laplace Transforms

### Property 1: Linearity

$$\mathcal{L}\{af(t) + bg(t)\} = aF(s) + bG(s)$$

**Example:** $\mathcal{L}\{3t^2 - 2e^{5t}\} = 3 \cdot \frac{2!}{s^3} - 2 \cdot \frac{1}{s-5} = \frac{6}{s^3} - \frac{2}{s-5}$

### Property 2: First Shifting Theorem (s-shifting)

$$\mathcal{L}\{e^{at}f(t)\} = F(s-a)$$

**Example:** $\mathcal{L}\{e^{3t}\sin(2t)\} = \frac{2}{(s-3)^2+4}$

### Property 3: Transform of Derivatives

$$\mathcal{L}\{f'(t)\} = sF(s) - f(0)$$
$$\mathcal{L}\{f''(t)\} = s^2F(s) - sf(0) - f'(0)$$

**General:**
$$\mathcal{L}\{f^{(n)}(t)\} = s^nF(s) - s^{n-1}f(0) - s^{n-2}f'(0) - \cdots - f^{(n-1)}(0)$$

### Property 4: Transform of Integrals

$$\mathcal{L}\left\{\int_0^t f(\tau)d\tau\right\} = \frac{F(s)}{s}$$

---

## 📝 Practice Problems

### Level 1: Direct Computation
1. $\mathcal{L}\{5\}$
2. $\mathcal{L}\{3t + 2\}$
3. $\mathcal{L}\{t^3\}$
4. $\mathcal{L}\{e^{-2t}\}$
5. $\mathcal{L}\{\cos(3t)\}$

### Level 2: Using Properties
6. $\mathcal{L}\{e^{2t}t^2\}$
7. $\mathcal{L}\{e^{-t}\cos(4t)\}$
8. $\mathcal{L}\{t\sin(t)\}$ (Hint: use differentiation property)
9. $\mathcal{L}\{(t-1)^2\}$
10. $\mathcal{L}\{e^{3t}(2\cos t - 3\sin t)\}$

### Level 3: From Definition
11. Compute $\mathcal{L}\{t^2\}$ from the definition
12. Show $\mathcal{L}\{\cosh(at)\} = \frac{s}{s^2-a^2}$

### Level 4: Transforms of Derivatives
13. If $f(0) = 2$ and $\mathcal{L}\{f(t)\} = F(s)$, find $\mathcal{L}\{f'(t)\}$
14. If $f(0) = 1$, $f'(0) = -3$, find $\mathcal{L}\{f''(t)\}$ in terms of $F(s)$
15. Find $\mathcal{L}\{f'(t)\}$ if $f(t) = t^2$ and verify using the derivative property

---

## 📊 Answers

1. $5/s$
2. $3/s^2 + 2/s$
3. $6/s^4$
4. $1/(s+2)$
5. $s/(s^2+9)$
6. $2/(s-2)^3$
7. $(s+1)/[(s+1)^2+16]$
8. $2s/(s^2+1)^2$
9. $2/s^3 - 2/s^2 + 1/s$
10. $\frac{2(s-3)}{(s-3)^2+1} - \frac{3}{(s-3)^2+1}$
11. Direct calculation gives $2/s^3$
12. Use $\cosh(at) = (e^{at}+e^{-at})/2$
13. $sF(s) - 2$
14. $s^2F(s) - s + 3$
15. $2/s^2$ both ways

---

## 🔬 Quantum Mechanics Connection

### The Resolvent Operator

In quantum mechanics, the **resolvent** of a Hamiltonian $H$:
$$G(E) = (E - H)^{-1}$$

is closely related to the Laplace transform of the time evolution operator:
$$\mathcal{L}\{e^{-iHt/\hbar}\} = \frac{i\hbar}{E - H}$$

### Green's Functions

The Green's function (propagator) can be found via Laplace transform:
$$G(x, x'; E) = \mathcal{L}\{K(x, x'; t)\}$$

where $K$ is the quantum propagator.

---

## ✅ Daily Checklist

- [ ] Read Boyce & DiPrima Sections 6.1-6.2
- [ ] Memorize basic transform table
- [ ] Practice computing transforms from definition
- [ ] Apply linearity and shifting properties
- [ ] Complete practice problems

---

## 🔜 Preview: Tomorrow

**Day 79: Inverse Laplace Transforms**
- Going from $F(s)$ back to $f(t)$
- Partial fraction decomposition
- Completing the square

---

*"The Laplace transform turns calculus into algebra—differentiation becomes multiplication by s."*
