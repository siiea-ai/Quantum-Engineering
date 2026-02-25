# Day 66: Variation of Parameters

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Theory & Derivation |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Examples |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Practice |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Understand why variation of parameters is needed
2. Derive the variation of parameters formulas
3. Apply the method to any second-order linear ODE
4. Compare with undetermined coefficients
5. Handle forcing functions that undetermined coefficients can't

---

## 📚 Required Reading

### Primary Text: Boyce & DiPrima (11th Edition)
- **Section 3.6**: Variation of Parameters (pp. 195-202)

---

## 📖 Core Content: Why Variation of Parameters?

### 1. Limitations of Undetermined Coefficients

Undetermined coefficients only works when $f(x)$ is:
- Polynomials
- Exponentials
- Sines and cosines
- Products of the above

What about $f(x) = \tan x$, $\sec x$, $\ln x$, $1/x$, etc.?

**Variation of parameters works for ANY continuous $f(x)$!**

### 2. The Setup

Given: $y'' + p(x)y' + q(x)y = f(x)$

Let $y_1$ and $y_2$ be the fundamental solutions to the homogeneous equation.

Homogeneous solution: $y_h = c_1 y_1 + c_2 y_2$

### 3. The Key Idea

Replace constants with functions:
$$y_p = u_1(x) y_1(x) + u_2(x) y_2(x)$$

Find $u_1$ and $u_2$ such that this works!

---

## 📖 Derivation

### 4. Setting Up the System

From $y_p = u_1 y_1 + u_2 y_2$:

$$y_p' = u_1' y_1 + u_1 y_1' + u_2' y_2 + u_2 y_2'$$

**First condition:** To simplify, require:
$$u_1' y_1 + u_2' y_2 = 0$$

Then: $y_p' = u_1 y_1' + u_2 y_2'$

$$y_p'' = u_1' y_1' + u_1 y_1'' + u_2' y_2' + u_2 y_2''$$

### 5. Substituting into the ODE

After substitution and using that $y_1$, $y_2$ satisfy the homogeneous equation:
$$u_1' y_1' + u_2' y_2' = f(x)$$

### 6. The System of Equations

We have two equations for $u_1'$ and $u_2'$:

$$\begin{cases}
u_1' y_1 + u_2' y_2 = 0 \\
u_1' y_1' + u_2' y_2' = f(x)
\end{cases}$$

### 7. Solving using Cramer's Rule

$$u_1' = \frac{-y_2 f(x)}{W(y_1, y_2)}, \quad u_2' = \frac{y_1 f(x)}{W(y_1, y_2)}$$

where $W = y_1 y_2' - y_2 y_1'$ is the Wronskian.

### 8. Final Formulas

$$u_1 = -\int \frac{y_2 f}{W} dx, \quad u_2 = \int \frac{y_1 f}{W} dx$$

> **Particular Solution:**
> $$y_p = -y_1 \int \frac{y_2 f}{W} dx + y_2 \int \frac{y_1 f}{W} dx$$

---

## ✏️ Worked Examples

### Example 1: Basic Application
Solve $y'' + y = \sec x$.

**Step 1: Solve homogeneous**
$$r^2 + 1 = 0 \Rightarrow r = \pm i$$
$$y_h = c_1 \cos x + c_2 \sin x$$
$$y_1 = \cos x, \quad y_2 = \sin x$$

**Step 2: Compute Wronskian**
$$W = \begin{vmatrix} \cos x & \sin x \\ -\sin x & \cos x \end{vmatrix} = \cos^2 x + \sin^2 x = 1$$

**Step 3: Find $u_1$ and $u_2$**
$$u_1' = -\frac{y_2 f}{W} = -\frac{\sin x \cdot \sec x}{1} = -\tan x$$
$$u_1 = -\int \tan x \, dx = \ln|\cos x|$$

$$u_2' = \frac{y_1 f}{W} = \frac{\cos x \cdot \sec x}{1} = 1$$
$$u_2 = \int 1 \, dx = x$$

**Step 4: Form particular solution**
$$y_p = \cos x \cdot \ln|\cos x| + x \sin x$$

**General solution:**
$$y = c_1 \cos x + c_2 \sin x + \cos x \ln|\cos x| + x \sin x$$

---

### Example 2: Another Trig Forcing
Solve $y'' + y = \csc x$.

**Same homogeneous solution:** $y_1 = \cos x$, $y_2 = \sin x$, $W = 1$

$$u_1' = -\sin x \cdot \csc x = -1$$
$$u_1 = -x$$

$$u_2' = \cos x \cdot \csc x = \cot x$$
$$u_2 = \ln|\sin x|$$

**Particular solution:**
$$y_p = -x\cos x + \sin x \ln|\sin x|$$

---

### Example 3: Exponential with Factor
Solve $y'' - 2y' + y = \frac{e^x}{x}$.

**Step 1: Solve homogeneous**
$$r^2 - 2r + 1 = (r-1)^2 = 0 \Rightarrow r = 1 \text{ (repeated)}$$
$$y_h = (c_1 + c_2 x)e^x$$
$$y_1 = e^x, \quad y_2 = xe^x$$

**Step 2: Wronskian**
$$W = \begin{vmatrix} e^x & xe^x \\ e^x & e^x + xe^x \end{vmatrix} = e^x(e^x + xe^x) - xe^x \cdot e^x = e^{2x}$$

**Step 3: Find $u_1$, $u_2$**
$$u_1' = -\frac{xe^x \cdot e^x/x}{e^{2x}} = -1$$
$$u_1 = -x$$

$$u_2' = \frac{e^x \cdot e^x/x}{e^{2x}} = \frac{1}{x}$$
$$u_2 = \ln|x|$$

**Step 4: Particular solution**
$$y_p = -xe^x + xe^x \ln|x| = xe^x(\ln|x| - 1)$$

**General solution:**
$$y = (c_1 + c_2 x)e^x + xe^x \ln|x|$$
(absorbing the $-xe^x$ into $y_h$)

---

### Example 4: Comparison with Undetermined Coefficients
Solve $y'' - y = e^x$ using variation of parameters.

**Homogeneous:** $y_1 = e^x$, $y_2 = e^{-x}$, $W = -2$

$$u_1' = -\frac{e^{-x} \cdot e^x}{-2} = \frac{1}{2}$$
$$u_1 = \frac{x}{2}$$

$$u_2' = \frac{e^x \cdot e^x}{-2} = -\frac{e^{2x}}{2}$$
$$u_2 = -\frac{e^{2x}}{4}$$

$$y_p = \frac{x}{2}e^x - \frac{e^{2x}}{4}e^{-x} = \frac{x}{2}e^x - \frac{e^x}{4}$$

**General solution:**
$$y = c_1 e^x + c_2 e^{-x} + \frac{x}{2}e^x$$
(absorbing the $-e^x/4$ term)

Compare with undetermined coefficients (overlap case): same answer!

---

## 📋 Comparison of Methods

| Aspect | Undetermined Coefficients | Variation of Parameters |
|--------|--------------------------|------------------------|
| Forcing functions | Limited (polynomials, exp, trig) | ANY continuous $f(x)$ |
| Coefficient type | Constants only | Can handle variable coefficients |
| Ease of use | Often simpler | More systematic |
| Computation | Algebraic | Involves integration |
| Best for | Standard forcing | Unusual forcing |

---

## 📝 Practice Problems

### Level 1: Basic Application
1. Solve $y'' + y = \tan x$
2. Solve $y'' + y = \sec^2 x$
3. Solve $y'' - y = \frac{1}{e^x + 1}$

### Level 2: Repeated Roots
4. Solve $y'' - 2y' + y = \frac{e^x}{x^2}$
5. Solve $y'' + 4y' + 4y = \frac{e^{-2x}}{x}$

### Level 3: Variable Coefficients
6. Solve $y'' + y = \sec x \tan x$
7. Solve $y'' - y = \sinh x$ using variation of parameters

### Level 4: With Initial Conditions
8. Solve $y'' + y = \sec x$, $y(0) = 1$, $y'(0) = 0$
9. Solve $y'' + 4y = \sec 2x$, $y(0) = 0$, $y'(0) = 1$

### Level 5: Challenging
10. Derive the variation of parameters formula for first-order equations
11. Solve $x^2 y'' - 2xy' + 2y = x^3$ (Cauchy-Euler type)
12. Show that variation of parameters gives the same answer as undetermined coefficients for $y'' + y = \cos x$

---

## 📊 Answers

1. $y = c_1 \cos x + c_2 \sin x - \cos x \ln|\sec x + \tan x|$
2. $y = c_1 \cos x + c_2 \sin x + \tan x$
3. $y = c_1 e^x + c_2 e^{-x} + \frac{1}{2}(e^x - e^{-x})\ln(1 + e^{-x})$ (simplifies)
4. $y = (c_1 + c_2 x)e^x - \frac{e^x}{x}$
5. $y = (c_1 + c_2 x)e^{-2x} + e^{-2x}\ln|x|$
6. $y = c_1 \cos x + c_2 \sin x + \sec x$
7. $y = c_1 e^x + c_2 e^{-x} + \frac{x}{2}(e^x - e^{-x})$
8. $y = \cos x + \cos x \ln|\cos x| + x\sin x$
9. After applying ICs
10. $y_p = y_1 \int \frac{f}{y_1^2 \cdot (d/dx)(1/y_1)} dx$
11. $y = c_1 x + c_2 x^2 + x^3$
12. Direct comparison

---

## 🔬 Quantum Mechanics Connection

### Green's Functions

Variation of parameters is related to Green's function methods:
$$y_p(x) = \int G(x, x') f(x') \, dx'$$

In quantum mechanics, the Green's function (propagator) gives:
$$\psi(x, t) = \int G(x, t; x', t') \psi(x', t') \, dx'$$

### Perturbation Theory

Time-dependent perturbation theory uses similar "variation of constants":
$$c_n(t) = c_n^{(0)} + c_n^{(1)}(t) + \cdots$$

where the coefficients evolve according to the perturbation.

---

## ✅ Daily Checklist

- [ ] Read Boyce & DiPrima Section 3.6
- [ ] Understand the derivation of variation of parameters
- [ ] Apply the formulas correctly
- [ ] Compare with undetermined coefficients
- [ ] Complete practice problems

---

## 🔜 Preview: Tomorrow

**Day 67: Mechanical and Electrical Oscillations**
- Simple harmonic motion
- Damped oscillations
- Forced oscillations and resonance
- RLC circuits

---

*"Variation of parameters reveals the particular solution through integration—a universal method for any forcing."*
