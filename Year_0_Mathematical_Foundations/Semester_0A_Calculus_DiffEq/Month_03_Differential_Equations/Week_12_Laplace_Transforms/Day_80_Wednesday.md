# Day 80: Solving ODEs with Laplace Transforms

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | The Solution Method |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Examples & Applications |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Practice |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Transform ODEs into algebraic equations
2. Solve for $Y(s)$ and invert to find $y(t)$
3. Handle any constant-coefficient linear ODE
4. Solve systems of ODEs using Laplace transforms
5. Apply to RLC circuits and mechanical systems

---

## 📚 Required Reading

### Primary Text: Boyce & DiPrima (11th Edition)
- **Section 6.2**: Solution of Initial Value Problems (pp. 304-315)
- **Section 6.3**: Step Functions (pp. 316-328)

---

## 📖 Core Method: Solving IVPs

### The Three-Step Process

**Given:** $ay'' + by' + cy = f(t)$, $y(0) = y_0$, $y'(0) = y_0'$

**Step 1: Transform**
Apply $\mathcal{L}$ to both sides using:
- $\mathcal{L}\{y'\} = sY(s) - y(0)$
- $\mathcal{L}\{y''\} = s^2Y(s) - sy(0) - y'(0)$

**Step 2: Solve for Y(s)**
Rearrange the algebraic equation to isolate $Y(s)$.

**Step 3: Invert**
Find $y(t) = \mathcal{L}^{-1}\{Y(s)\}$ using partial fractions.

---

## ✏️ Worked Examples

### Example 1: First-Order ODE
Solve $y' + 2y = e^{-t}$, $y(0) = 1$

**Step 1: Transform**
$$sY - y(0) + 2Y = \frac{1}{s+1}$$
$$sY - 1 + 2Y = \frac{1}{s+1}$$

**Step 2: Solve for Y**
$$(s + 2)Y = 1 + \frac{1}{s+1} = \frac{s+1+1}{s+1} = \frac{s+2}{s+1}$$
$$Y(s) = \frac{s+2}{(s+1)(s+2)} = \frac{1}{s+1}$$

**Step 3: Invert**
$$y(t) = e^{-t}$$

**Verify:** $y' + 2y = -e^{-t} + 2e^{-t} = e^{-t}$ ✓, $y(0) = 1$ ✓

---

### Example 2: Second-Order ODE (Real Roots)
Solve $y'' - 3y' + 2y = 0$, $y(0) = 1$, $y'(0) = 0$

**Step 1: Transform**
$$s^2Y - sy(0) - y'(0) - 3(sY - y(0)) + 2Y = 0$$
$$s^2Y - s - 3sY + 3 + 2Y = 0$$

**Step 2: Solve for Y**
$$(s^2 - 3s + 2)Y = s - 3$$
$$Y(s) = \frac{s-3}{s^2-3s+2} = \frac{s-3}{(s-1)(s-2)}$$

**Partial fractions:**
$$\frac{s-3}{(s-1)(s-2)} = \frac{A}{s-1} + \frac{B}{s-2}$$

Cover-up: $A = \frac{1-3}{1-2} = 2$, $B = \frac{2-3}{2-1} = -1$

**Step 3: Invert**
$$y(t) = 2e^t - e^{2t}$$

---

### Example 3: Second-Order with Oscillation
Solve $y'' + 4y = 0$, $y(0) = 2$, $y'(0) = 0$

**Step 1: Transform**
$$s^2Y - 2s - 0 + 4Y = 0$$

**Step 2: Solve for Y**
$$(s^2 + 4)Y = 2s$$
$$Y(s) = \frac{2s}{s^2+4}$$

**Step 3: Invert**
$$y(t) = 2\cos(2t)$$

---

### Example 4: Forced Oscillation
Solve $y'' + y = \sin(2t)$, $y(0) = 0$, $y'(0) = 0$

**Step 1: Transform**
$$s^2Y + Y = \frac{2}{s^2+4}$$

**Step 2: Solve for Y**
$$Y(s) = \frac{2}{(s^2+1)(s^2+4)}$$

**Partial fractions:**
$$\frac{2}{(s^2+1)(s^2+4)} = \frac{As+B}{s^2+1} + \frac{Cs+D}{s^2+4}$$

Multiply out and compare:
$$2 = (As+B)(s^2+4) + (Cs+D)(s^2+1)$$

Setting $s = i$: $2 = (Ai+B)(3) \Rightarrow B = 2/3, A = 0$
Setting $s = 2i$: $2 = (2Ci+D)(-3) \Rightarrow D = -2/3, C = 0$

$$Y(s) = \frac{2/3}{s^2+1} - \frac{2/3}{s^2+4}$$

**Step 3: Invert**
$$y(t) = \frac{2}{3}\sin(t) - \frac{1}{3}\sin(2t)$$

---

### Example 5: Damped Oscillation
Solve $y'' + 2y' + 5y = 0$, $y(0) = 1$, $y'(0) = -1$

**Step 1: Transform**
$$s^2Y - s + 1 + 2(sY - 1) + 5Y = 0$$
$$s^2Y - s + 1 + 2sY - 2 + 5Y = 0$$

**Step 2: Solve**
$$(s^2 + 2s + 5)Y = s + 1$$
$$Y(s) = \frac{s+1}{s^2+2s+5}$$

Complete the square: $s^2 + 2s + 5 = (s+1)^2 + 4$

$$Y(s) = \frac{s+1}{(s+1)^2+4}$$

**Step 3: Invert**
$$y(t) = e^{-t}\cos(2t)$$

---

## 📖 Systems of ODEs

### Example 6: Coupled System
Solve:
$$x' = 3x - y, \quad x(0) = 1$$
$$y' = x + y, \quad y(0) = 0$$

**Transform both:**
$$sX - 1 = 3X - Y$$
$$sY - 0 = X + Y$$

**Rearrange:**
$$(s-3)X + Y = 1$$
$$-X + (s-1)Y = 0$$

From equation 2: $X = (s-1)Y$

Substitute into equation 1:
$$(s-3)(s-1)Y + Y = 1$$
$$[(s-3)(s-1) + 1]Y = 1$$
$$[s^2 - 4s + 4]Y = 1$$
$$Y = \frac{1}{(s-2)^2}$$

Then: $X = \frac{s-1}{(s-2)^2} = \frac{1}{s-2} + \frac{1}{(s-2)^2}$

**Invert:**
$$x(t) = e^{2t} + te^{2t}, \quad y(t) = te^{2t}$$

---

## 📋 Summary: Transform Method

| ODE Term | Transform |
|----------|-----------|
| $y$ | $Y(s)$ |
| $y'$ | $sY - y(0)$ |
| $y''$ | $s^2Y - sy(0) - y'(0)$ |
| $y'''$ | $s^3Y - s^2y(0) - sy'(0) - y''(0)$ |

**Advantage:** Initial conditions are automatically incorporated!

---

## 📝 Practice Problems

### Level 1: First-Order
1. $y' + 3y = 0$, $y(0) = 2$
2. $y' - y = e^{2t}$, $y(0) = 1$
3. $y' + 2y = 4$, $y(0) = 0$

### Level 2: Second-Order (Real Roots)
4. $y'' - 4y = 0$, $y(0) = 1$, $y'(0) = 2$
5. $y'' + 5y' + 6y = 0$, $y(0) = 2$, $y'(0) = -5$
6. $y'' - y = e^t$, $y(0) = 0$, $y'(0) = 0$

### Level 3: Oscillations
7. $y'' + 9y = 0$, $y(0) = 0$, $y'(0) = 3$
8. $y'' + 4y' + 8y = 0$, $y(0) = 1$, $y'(0) = 0$
9. $y'' + y = \cos(t)$, $y(0) = 0$, $y'(0) = 0$

### Level 4: Systems
10. $x' = x + 2y$, $y' = 3x + 2y$, $x(0) = 1$, $y(0) = 0$

---

## 📊 Answers

1. $y = 2e^{-3t}$
2. $y = e^{2t} + e^t - e^t = e^{2t}$... (recalculate)
3. $y = 2(1 - e^{-2t})$
4. $y = e^{2t}$
5. $y = e^{-2t} + e^{-3t}$
6. $y = \frac{1}{2}(e^t - e^{-t}) - \frac{t}{2}e^t$
7. $y = \sin(3t)$
8. $y = e^{-2t}\cos(2t) + e^{-2t}\sin(2t)$
9. $y = \frac{t}{2}\sin(t)$ (resonance!)
10. Solve using method above

---

## 🔬 Quantum Application: Driven Two-Level System

The Schrödinger equation for a driven qubit:
$$i\hbar\dot{c}_1 = E_1 c_1 + V(t)c_2$$
$$i\hbar\dot{c}_2 = V(t)c_1 + E_2 c_2$$

Laplace transforms convert this to:
$$(is\hbar - E_1)C_1(s) - VC_2(s) = i\hbar c_1(0)$$

Solving gives the **dressed state** dynamics!

---

## ✅ Daily Checklist

- [ ] Master the three-step process
- [ ] Practice with various IC combinations
- [ ] Solve at least one system
- [ ] Complete practice problems

---

## 🔜 Preview: Tomorrow

**Day 81: Step Functions and Impulses**
- Heaviside step function
- Dirac delta function
- Discontinuous forcing

---

*"Laplace transforms turn differential equations into high school algebra—solve for the unknown and invert."*
