# Day 57: Introduction to Differential Equations

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | ODE Fundamentals |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Separable Equations |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Practice |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Define differential equations and their classifications
2. Understand initial value problems (IVPs)
3. Verify solutions to differential equations
4. Solve separable first-order ODEs
5. Apply separation of variables to physical problems

---

## 📚 Required Reading

### Primary Text: Boyce & DiPrima (11th Edition)
- **Chapter 1**: Introduction (pp. 1-25)
- **Section 2.2**: Separable Equations (pp. 42-51)

### Alternative: Zill's Differential Equations
- Chapter 1: Introduction to Differential Equations
- Section 2.2: Separable Variables

---

## 🎬 Video Resources

### 3Blue1Brown
**Differential Equations** - Beautiful visual introduction

### MIT OpenCourseWare 18.03
**Lecture 1: Introduction to ODEs**

### Professor Leonard
**Differential Equations: Complete Playlist**

---

## 📖 Core Content: What is a Differential Equation?

### 1. Definition

> **Definition:** A **differential equation** (DE) is an equation that contains derivatives of an unknown function.

**Examples:**
- $\frac{dy}{dx} = 2x$ (first-order)
- $\frac{d^2y}{dx^2} + 3\frac{dy}{dx} + 2y = 0$ (second-order)
- $\frac{\partial u}{\partial t} = k\frac{\partial^2 u}{\partial x^2}$ (partial DE)

### 2. Classification

**By Type:**
- **Ordinary DE (ODE):** Contains derivatives with respect to ONE independent variable
- **Partial DE (PDE):** Contains partial derivatives (multiple independent variables)

**By Order:**
The **order** is the highest derivative that appears.
- $y' + y = x$ (first-order)
- $y'' + y = 0$ (second-order)

**By Linearity:**
A DE is **linear** if it can be written as:
$$a_n(x)y^{(n)} + a_{n-1}(x)y^{(n-1)} + \cdots + a_1(x)y' + a_0(x)y = g(x)$$

Otherwise, it's **nonlinear**.

### 3. Solutions

> **Definition:** A **solution** to a DE is a function y(x) that satisfies the equation when substituted.

**General Solution:** Contains arbitrary constants (one per order)
**Particular Solution:** Specific values of constants determined by conditions

### 4. Initial Value Problems (IVPs)

> **Definition:** An **initial value problem** consists of:
> - A differential equation
> - Initial conditions specifying y (and derivatives) at a specific point

**Example:**
$$\frac{dy}{dx} = 2x, \quad y(0) = 3$$

---

## ✏️ Verification Examples

### Example 1: Verify a Solution
Verify that $y = e^{2x}$ is a solution to $y' - 2y = 0$.

**Check:** 
$$y' = 2e^{2x}$$
$$y' - 2y = 2e^{2x} - 2e^{2x} = 0$$ ✓

---

### Example 2: Verify with Parameter
Verify that $y = Ce^{-x}$ is the general solution to $y' + y = 0$.

**Check:**
$$y' = -Ce^{-x}$$
$$y' + y = -Ce^{-x} + Ce^{-x} = 0$$ ✓

The constant C can be any value, making this the general solution.

---

### Example 3: Find Particular Solution
Given $y = Ce^{-x}$ solves $y' + y = 0$, find the particular solution with $y(0) = 5$.

$$y(0) = Ce^0 = C = 5$$

**Particular solution:** $y = 5e^{-x}$

---

## 📖 Separable Equations

### 5. Definition

> **Definition:** A first-order ODE is **separable** if it can be written as:
> $$\frac{dy}{dx} = f(x)g(y)$$
> or equivalently:
> $$\frac{dy}{dx} = \frac{f(x)}{h(y)}$$

### 6. Solution Method

**Step 1:** Separate variables (get all y's on one side, all x's on the other)
$$\frac{dy}{g(y)} = f(x) \, dx$$

**Step 2:** Integrate both sides
$$\int \frac{dy}{g(y)} = \int f(x) \, dx$$

**Step 3:** Solve for y (if possible)

**Step 4:** Apply initial conditions (if given)

---

## ✏️ Separable Equation Examples

### Example 4: Basic Separable
Solve $\frac{dy}{dx} = xy$.

**Separate:**
$$\frac{dy}{y} = x \, dx$$

**Integrate:**
$$\int \frac{dy}{y} = \int x \, dx$$
$$\ln|y| = \frac{x^2}{2} + C$$

**Solve for y:**
$$|y| = e^{x^2/2 + C} = e^C \cdot e^{x^2/2}$$
$$y = Ae^{x^2/2}$$ where $A = \pm e^C$

---

### Example 5: With Initial Condition
Solve $\frac{dy}{dx} = \frac{x}{y}$, $y(0) = 2$.

**Separate:**
$$y \, dy = x \, dx$$

**Integrate:**
$$\frac{y^2}{2} = \frac{x^2}{2} + C$$
$$y^2 = x^2 + 2C$$

**Apply IC:** $y(0) = 2 \Rightarrow 4 = 0 + 2C \Rightarrow C = 2$

**Solution:** $y^2 = x^2 + 4$, so $y = \sqrt{x^2 + 4}$ (taking positive root)

---

### Example 6: Exponential Growth/Decay
Solve $\frac{dy}{dt} = ky$, $y(0) = y_0$.

**Separate:**
$$\frac{dy}{y} = k \, dt$$

**Integrate:**
$$\ln|y| = kt + C$$
$$y = Ae^{kt}$$

**Apply IC:** $y(0) = A = y_0$

**Solution:** $y = y_0 e^{kt}$

This is the **exponential growth/decay** model!
- k > 0: exponential growth
- k < 0: exponential decay

---

### Example 7: Logistic Growth
Solve $\frac{dP}{dt} = rP\left(1 - \frac{P}{K}\right)$.

**Separate:**
$$\frac{dP}{P(1 - P/K)} = r \, dt$$

**Partial fractions:**
$$\frac{1}{P(1 - P/K)} = \frac{1}{P} + \frac{1/K}{1 - P/K}$$

**Integrate:**
$$\ln|P| - \ln|1 - P/K| = rt + C$$
$$\ln\left|\frac{P}{1 - P/K}\right| = rt + C$$

**Solve:** (after algebra)
$$P(t) = \frac{K}{1 + Ae^{-rt}}$$

This is the **logistic growth** model with carrying capacity K.

---

## 📋 Common Separable Forms

| Equation | Solution |
|----------|----------|
| $y' = ky$ | $y = Ce^{kx}$ |
| $y' = k(y - A)$ | $y = A + Ce^{kx}$ |
| $y' = \frac{y}{x}$ | $y = Cx$ |
| $y' = -\frac{x}{y}$ | $x^2 + y^2 = C$ |
| $y' = y^2$ | $y = -\frac{1}{x + C}$ |

---

## 📝 Practice Problems

### Level 1: Verification
1. Verify $y = x^2 + 1$ solves $y' = 2x$
2. Verify $y = \sin(x)$ solves $y'' + y = 0$
3. Verify $y = Ce^{3x}$ solves $y' = 3y$

### Level 2: Basic Separable
4. Solve $\frac{dy}{dx} = 3y$
5. Solve $\frac{dy}{dx} = \frac{y}{x}$
6. Solve $\frac{dy}{dx} = x^2 y^2$

### Level 3: With Initial Conditions
7. Solve $y' = 2xy$, $y(0) = 1$
8. Solve $y' = \frac{1+y}{1+x}$, $y(0) = 0$
9. Solve $y' = y^2 \sin(x)$, $y(0) = 1$

### Level 4: Applications
10. A population grows at a rate proportional to its size. If P(0) = 100 and P(2) = 200, find P(t).
11. A radioactive substance decays at a rate proportional to its mass. If half remains after 10 years, find the decay constant k.
12. Newton's Law of Cooling: $\frac{dT}{dt} = k(T - T_s)$. If $T(0) = 100°$, $T_s = 20°$, and $T(10) = 60°$, find T(t).

### Level 5: Challenging
13. Solve $\frac{dy}{dx} = \frac{x + y}{x - y}$ (Hint: substitute $v = y/x$)
14. Solve $y' = e^{x+y}$
15. Find the orthogonal trajectories to $y = Cx^2$ (curves that cross at right angles)

---

## 📊 Answers

1. Direct verification
2. $y' = \cos x$, $y'' = -\sin x$, so $y'' + y = 0$ ✓
3. $y' = 3Ce^{3x} = 3y$ ✓
4. $y = Ce^{3x}$
5. $y = Cx$
6. $y = \frac{-1}{x^3/3 + C}$
7. $y = e^{x^2}$
8. $y = -1 + 2(1 + x)^{1/2}$ or $y = (1+x) - 1$
9. $y = \frac{1}{2 + \cos x - 1}$ (check)
10. $P(t) = 100 \cdot 2^{t/2}$
11. $k = -\frac{\ln 2}{10}$
12. $T(t) = 20 + 80e^{kt}$ where $k = \frac{\ln(0.5)}{10}$
13. $x^2 + y^2 = Ce^{2\arctan(y/x)}$
14. $e^{-y} = -e^x + C$
15. $y = \frac{C}{x}$ (hyperbolas)

---

## 🔬 Quantum Mechanics Connection

### Time Evolution

The time-dependent Schrödinger equation:
$$i\hbar\frac{\partial \Psi}{\partial t} = \hat{H}\Psi$$

For time-independent Hamiltonians, we separate variables:
$$\Psi(x,t) = \psi(x)T(t)$$

The time part satisfies:
$$i\hbar\frac{dT}{dt} = ET$$

This is a separable first-order ODE with solution:
$$T(t) = e^{-iEt/\hbar}$$

### Radioactive Decay

Quantum tunneling leads to exponential decay:
$$N(t) = N_0 e^{-\lambda t}$$

The half-life is:
$$t_{1/2} = \frac{\ln 2}{\lambda}$$

---

## ✅ Daily Checklist

- [ ] Read Boyce & DiPrima Chapter 1, Section 2.2
- [ ] Understand ODE classification
- [ ] Verify solutions by substitution
- [ ] Master separation of variables technique
- [ ] Solve IVPs with separable equations
- [ ] Complete practice problems

---

## 🔜 Preview: Tomorrow

**Day 58: Linear First-Order Equations**
- Standard form: $y' + P(x)y = Q(x)$
- Integrating factor method
- Applications to mixing problems and circuits

---

*"Differential equations are the poetry of physics—they express the laws of nature in the language of change."*
