# Day 64: Homogeneous Second-Order Linear ODEs

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Theory & Characteristic Equation |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Three Cases |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Practice |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Recognize second-order linear ODEs
2. Understand the principle of superposition
3. Solve homogeneous equations using the characteristic equation
4. Handle all three cases: distinct real, complex, repeated roots
5. Find general solutions and apply initial conditions

---

## 📚 Required Reading

### Primary Text: Boyce & DiPrima (11th Edition)
- **Section 3.1**: Homogeneous Equations with Constant Coefficients (pp. 139-151)
- **Section 3.3**: Complex Roots of the Characteristic Equation (pp. 163-172)
- **Section 3.4**: Repeated Roots; Reduction of Order (pp. 173-180)

---

## 🎬 Video Resources

### 3Blue1Brown
**Differential Equations: Second Order** - Visual intuition

### MIT OpenCourseWare 18.03
**Lectures on Second-Order ODEs**

### Professor Leonard
**Second Order Differential Equations Playlist**

---

## 📖 Core Content: Second-Order Linear ODEs

### 1. General Form

A **second-order linear ODE** has the form:
$$a(x)y'' + b(x)y' + c(x)y = f(x)$$

When $a$, $b$, $c$ are constants:
$$ay'' + by' + cy = f(x)$$

### 2. Homogeneous vs. Nonhomogeneous

- **Homogeneous:** $f(x) = 0$ → $ay'' + by' + cy = 0$
- **Nonhomogeneous:** $f(x) \neq 0$ → $ay'' + by' + cy = f(x)$

Today we focus on homogeneous equations.

### 3. Principle of Superposition

> **Theorem:** If $y_1$ and $y_2$ are solutions to a homogeneous linear ODE, then so is any linear combination:
> $$y = c_1 y_1 + c_2 y_2$$

This is why we need **two** independent solutions for the general solution.

### 4. Linear Independence

Two solutions $y_1$, $y_2$ are **linearly independent** if neither is a constant multiple of the other.

**Wronskian test:**
$$W(y_1, y_2) = \begin{vmatrix} y_1 & y_2 \\ y_1' & y_2' \end{vmatrix} = y_1 y_2' - y_2 y_1' \neq 0$$

---

## 📖 The Characteristic Equation Method

### 5. The Key Idea

Try a solution of the form $y = e^{rx}$.

Substituting into $ay'' + by' + cy = 0$:
$$ar^2 e^{rx} + br e^{rx} + c e^{rx} = 0$$
$$e^{rx}(ar^2 + br + c) = 0$$

Since $e^{rx} \neq 0$, we need:
$$ar^2 + br + c = 0$$

This is the **characteristic equation**!

### 6. The Three Cases

The roots of $ar^2 + br + c = 0$ determine the solution:

| Case | Discriminant | Roots | General Solution |
|------|-------------|-------|------------------|
| 1 | $b^2 - 4ac > 0$ | Two real: $r_1 \neq r_2$ | $y = c_1 e^{r_1 x} + c_2 e^{r_2 x}$ |
| 2 | $b^2 - 4ac < 0$ | Complex: $r = \alpha \pm i\beta$ | $y = e^{\alpha x}(c_1 \cos\beta x + c_2 \sin\beta x)$ |
| 3 | $b^2 - 4ac = 0$ | Repeated: $r_1 = r_2 = r$ | $y = c_1 e^{rx} + c_2 x e^{rx}$ |

---

## 📖 Case 1: Distinct Real Roots

### Example 1: Two Distinct Real Roots
Solve $y'' - 5y' + 6y = 0$.

**Characteristic equation:**
$$r^2 - 5r + 6 = 0$$
$$(r - 2)(r - 3) = 0$$
$$r_1 = 2, \quad r_2 = 3$$

**General solution:**
$$y = c_1 e^{2x} + c_2 e^{3x}$$

---

### Example 2: With Initial Conditions
Solve $y'' - y = 0$, $y(0) = 2$, $y'(0) = -1$.

**Characteristic equation:**
$$r^2 - 1 = 0 \Rightarrow r = \pm 1$$

**General solution:**
$$y = c_1 e^x + c_2 e^{-x}$$

**Apply ICs:**
$$y(0) = c_1 + c_2 = 2$$
$$y'(0) = c_1 - c_2 = -1$$

Solving: $c_1 = 1/2$, $c_2 = 3/2$

**Particular solution:**
$$y = \frac{1}{2}e^x + \frac{3}{2}e^{-x}$$

---

## 📖 Case 2: Complex Roots

### 7. Euler's Formula

When roots are complex: $r = \alpha \pm i\beta$

Using Euler's formula: $e^{i\theta} = \cos\theta + i\sin\theta$

The complex solutions $e^{(\alpha + i\beta)x}$ and $e^{(\alpha - i\beta)x}$ combine to give real solutions:
$$y = e^{\alpha x}(c_1 \cos\beta x + c_2 \sin\beta x)$$

### Example 3: Complex Roots (Oscillation)
Solve $y'' + 4y = 0$.

**Characteristic equation:**
$$r^2 + 4 = 0 \Rightarrow r = \pm 2i$$

Here $\alpha = 0$, $\beta = 2$.

**General solution:**
$$y = c_1 \cos(2x) + c_2 \sin(2x)$$

This describes **simple harmonic motion**!

---

### Example 4: Damped Oscillation
Solve $y'' + 2y' + 5y = 0$.

**Characteristic equation:**
$$r^2 + 2r + 5 = 0$$
$$r = \frac{-2 \pm \sqrt{4 - 20}}{2} = \frac{-2 \pm 4i}{2} = -1 \pm 2i$$

Here $\alpha = -1$, $\beta = 2$.

**General solution:**
$$y = e^{-x}(c_1 \cos 2x + c_2 \sin 2x)$$

This describes **damped oscillation**—oscillations that decay exponentially!

---

### Example 5: Complex Roots with ICs
Solve $y'' + 9y = 0$, $y(0) = 1$, $y'(0) = 6$.

**Solution form:** $r = \pm 3i$, so $y = c_1 \cos 3x + c_2 \sin 3x$

**Apply ICs:**
$$y(0) = c_1 = 1$$
$$y' = -3c_1 \sin 3x + 3c_2 \cos 3x$$
$$y'(0) = 3c_2 = 6 \Rightarrow c_2 = 2$$

**Particular solution:**
$$y = \cos 3x + 2\sin 3x$$

---

## 📖 Case 3: Repeated Roots

### 8. The Problem with Repeated Roots

If $r_1 = r_2 = r$, then $e^{rx}$ is one solution, but we need two!

### 9. Reduction of Order

If $y_1 = e^{rx}$ is one solution, try $y_2 = v(x)e^{rx}$.

After substitution and simplification, this gives $y_2 = xe^{rx}$.

### Example 6: Repeated Root
Solve $y'' - 4y' + 4y = 0$.

**Characteristic equation:**
$$r^2 - 4r + 4 = 0$$
$$(r - 2)^2 = 0$$
$$r = 2$$ (repeated)

**General solution:**
$$y = c_1 e^{2x} + c_2 x e^{2x} = (c_1 + c_2 x)e^{2x}$$

---

### Example 7: Repeated Root with ICs
Solve $y'' + 6y' + 9y = 0$, $y(0) = 2$, $y'(0) = 1$.

**Characteristic equation:**
$$r^2 + 6r + 9 = (r + 3)^2 = 0$$
$$r = -3$$ (repeated)

**General solution:**
$$y = (c_1 + c_2 x)e^{-3x}$$

**Apply ICs:**
$$y(0) = c_1 = 2$$
$$y' = c_2 e^{-3x} - 3(c_1 + c_2 x)e^{-3x}$$
$$y'(0) = c_2 - 3c_1 = c_2 - 6 = 1 \Rightarrow c_2 = 7$$

**Particular solution:**
$$y = (2 + 7x)e^{-3x}$$

---

## 📋 Summary Table

| Discriminant | Roots | General Solution | Physical Meaning |
|-------------|-------|------------------|------------------|
| $b^2 > 4ac$ | Real distinct: $r_1, r_2$ | $c_1 e^{r_1 x} + c_2 e^{r_2 x}$ | Overdamped |
| $b^2 < 4ac$ | Complex: $\alpha \pm i\beta$ | $e^{\alpha x}(c_1 \cos\beta x + c_2 \sin\beta x)$ | Underdamped (oscillation) |
| $b^2 = 4ac$ | Repeated: $r$ | $(c_1 + c_2 x)e^{rx}$ | Critically damped |

---

## 📝 Practice Problems

### Level 1: Distinct Real Roots
1. Solve $y'' - 3y' + 2y = 0$
2. Solve $y'' + y' - 6y = 0$
3. Solve $y'' - 4y = 0$, $y(0) = 1$, $y'(0) = 2$

### Level 2: Complex Roots
4. Solve $y'' + 16y = 0$
5. Solve $y'' + 2y' + 10y = 0$
6. Solve $y'' + 4y' + 13y = 0$, $y(0) = 1$, $y'(0) = 0$

### Level 3: Repeated Roots
7. Solve $y'' - 6y' + 9y = 0$
8. Solve $y'' + 10y' + 25y = 0$
9. Solve $4y'' + 4y' + y = 0$, $y(0) = 1$, $y'(0) = -1$

### Level 4: Mixed
10. Solve $y'' - 2y' - 3y = 0$
11. Solve $y'' + 6y' + 9y = 0$, $y(0) = 0$, $y'(0) = 1$
12. Solve $y'' - 2y' + 2y = 0$

### Level 5: Applications
13. A spring-mass system satisfies $y'' + 4y = 0$, $y(0) = 3$, $y'(0) = 0$. Find position y(t).
14. A damped oscillator: $y'' + 2y' + 5y = 0$, $y(0) = 1$, $y'(0) = 0$. Find y(t).
15. Find the general solution to $y'' + \omega^2 y = 0$ and interpret physically.

---

## 📊 Answers

1. $y = c_1 e^x + c_2 e^{2x}$
2. $y = c_1 e^{2x} + c_2 e^{-3x}$
3. $y = \frac{3}{4}e^{2x} + \frac{1}{4}e^{-2x}$
4. $y = c_1 \cos 4x + c_2 \sin 4x$
5. $y = e^{-x}(c_1 \cos 3x + c_2 \sin 3x)$
6. $y = e^{-2x}(\cos 3x + \frac{2}{3}\sin 3x)$
7. $y = (c_1 + c_2 x)e^{3x}$
8. $y = (c_1 + c_2 x)e^{-5x}$
9. $y = (1 - \frac{1}{2}x)e^{-x/2}$
10. $y = c_1 e^{3x} + c_2 e^{-x}$
11. $y = xe^{-3x}$
12. $y = e^x(c_1 \cos x + c_2 \sin x)$
13. $y = 3\cos 2t$ (simple harmonic motion)
14. $y = e^{-t}(\cos 2t + \frac{1}{2}\sin 2t)$
15. $y = A\cos(\omega t) + B\sin(\omega t)$ = simple harmonic oscillation

---

## 🔬 Quantum Mechanics Connection

### The Schrödinger Equation

The time-independent Schrödinger equation for a free particle:
$$-\frac{\hbar^2}{2m}\frac{d^2\psi}{dx^2} = E\psi$$
$$\psi'' + \frac{2mE}{\hbar^2}\psi = 0$$

This is our $y'' + \omega^2 y = 0$! Solutions are:
$$\psi = A e^{ikx} + B e^{-ikx}$$

where $k = \sqrt{2mE}/\hbar$ is the wave vector.

### Harmonic Oscillator

The quantum harmonic oscillator:
$$-\frac{\hbar^2}{2m}\psi'' + \frac{1}{2}m\omega^2 x^2 \psi = E\psi$$

This second-order ODE gives the famous energy levels $E_n = \hbar\omega(n + 1/2)$!

---

## ✅ Daily Checklist

- [ ] Read Boyce & DiPrima 3.1, 3.3, 3.4
- [ ] Understand the characteristic equation method
- [ ] Master all three cases
- [ ] Apply initial conditions correctly
- [ ] Complete practice problems

---

## 🔜 Preview: Tomorrow

**Day 65: Nonhomogeneous Equations**
- Method of undetermined coefficients
- Finding particular solutions
- General solution = homogeneous + particular

---

*"Second-order equations describe the dance of springs and pendulums—the rhythm of the physical world."*
