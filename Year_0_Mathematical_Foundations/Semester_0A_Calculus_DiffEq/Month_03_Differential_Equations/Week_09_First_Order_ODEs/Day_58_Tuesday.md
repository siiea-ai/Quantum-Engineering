# Day 58: Linear First-Order Equations

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Integrating Factor Method |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Applications |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Practice |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Identify first-order linear ODEs
2. Apply the integrating factor method
3. Solve mixing/tank problems
4. Solve RC circuit problems
5. Handle equations with variable coefficients

---

## 📚 Required Reading

### Primary Text: Boyce & DiPrima (11th Edition)
- **Section 2.1**: Linear Equations; Method of Integrating Factors (pp. 30-42)

---

## 📖 Core Content: First-Order Linear ODEs

### 1. Standard Form

> **Definition:** A **first-order linear ODE** has the form:
> $$\frac{dy}{dx} + P(x)y = Q(x)$$

**Key features:**
- y and y' appear to first power only
- No products like y·y'
- P(x) and Q(x) can be any functions of x

### 2. The Integrating Factor Method

**Goal:** Multiply by a function μ(x) that makes the left side a perfect derivative.

We want:
$$\mu(x)[y' + P(x)y] = \frac{d}{dx}[\mu(x)y]$$

Expanding the right side:
$$\frac{d}{dx}[\mu y] = \mu y' + \mu' y$$

Comparing: we need $\mu' = \mu P(x)$

### 3. Finding the Integrating Factor

From $\frac{d\mu}{\mu} = P(x)dx$:
$$\ln|\mu| = \int P(x) \, dx$$
$$\mu(x) = e^{\int P(x) dx}$$

### 4. Solution Formula

Multiply the equation by μ:
$$\mu y' + \mu P y = \mu Q$$
$$\frac{d}{dx}[\mu y] = \mu Q$$

Integrate:
$$\mu y = \int \mu Q \, dx + C$$

> **Solution:**
> $$y = \frac{1}{\mu(x)}\left[\int \mu(x) Q(x) \, dx + C\right]$$
> where $\mu(x) = e^{\int P(x) dx}$

---

## ✏️ Worked Examples

### Example 1: Constant Coefficient
Solve $y' + 2y = 6$.

**Standard form:** $P(x) = 2$, $Q(x) = 6$

**Integrating factor:**
$$\mu = e^{\int 2 dx} = e^{2x}$$

**Multiply and integrate:**
$$e^{2x}y' + 2e^{2x}y = 6e^{2x}$$
$$\frac{d}{dx}[e^{2x}y] = 6e^{2x}$$
$$e^{2x}y = 3e^{2x} + C$$

**Solution:** $y = 3 + Ce^{-2x}$

---

### Example 2: Variable Coefficient
Solve $y' + \frac{1}{x}y = x^2$.

**Integrating factor:**
$$\mu = e^{\int (1/x) dx} = e^{\ln|x|} = x$$

**Multiply:**
$$xy' + y = x^3$$
$$\frac{d}{dx}[xy] = x^3$$

**Integrate:**
$$xy = \frac{x^4}{4} + C$$

**Solution:** $y = \frac{x^3}{4} + \frac{C}{x}$

---

### Example 3: With Initial Condition
Solve $y' - 3y = e^{2x}$, $y(0) = 1$.

**Integrating factor:**
$$\mu = e^{\int -3 dx} = e^{-3x}$$

**Multiply:**
$$e^{-3x}y' - 3e^{-3x}y = e^{-x}$$
$$\frac{d}{dx}[e^{-3x}y] = e^{-x}$$

**Integrate:**
$$e^{-3x}y = -e^{-x} + C$$
$$y = -e^{2x} + Ce^{3x}$$

**Apply IC:** $y(0) = -1 + C = 1 \Rightarrow C = 2$

**Solution:** $y = -e^{2x} + 2e^{3x}$

---

### Example 4: Trigonometric Coefficient
Solve $y' + y\tan(x) = \sec(x)$.

**Integrating factor:**
$$\mu = e^{\int \tan(x) dx} = e^{-\ln|\cos x|} = \frac{1}{\cos x} = \sec x$$

**Multiply:**
$$\sec(x)y' + \sec(x)\tan(x)y = \sec^2(x)$$
$$\frac{d}{dx}[\sec(x)y] = \sec^2(x)$$

**Integrate:**
$$\sec(x)y = \tan(x) + C$$

**Solution:** $y = \sin(x) + C\cos(x)$

---

## 📖 Applications

### 5. Mixing Problems

**Setup:** A tank contains a solution. Liquid flows in and out at specified rates.

**Variables:**
- V(t) = volume of liquid in tank
- x(t) = amount of substance in tank
- c(t) = x(t)/V(t) = concentration

**General equation:**
$$\frac{dx}{dt} = \text{(rate in)} - \text{(rate out)}$$

### Example 5: Mixing Problem
A 100-gallon tank initially contains 50 lbs of salt dissolved in water. Brine containing 2 lbs/gal flows in at 3 gal/min, and the well-mixed solution flows out at 3 gal/min. Find the amount of salt x(t).

**Rate in:** $2 \text{ lb/gal} \times 3 \text{ gal/min} = 6$ lb/min

**Rate out:** $\frac{x}{100} \text{ lb/gal} \times 3 \text{ gal/min} = \frac{3x}{100}$ lb/min

**ODE:**
$$\frac{dx}{dt} = 6 - \frac{3x}{100}$$
$$x' + \frac{3}{100}x = 6$$

**Integrating factor:** $\mu = e^{3t/100}$

**Solution:** $x = 200 + Ce^{-3t/100}$

**IC:** $x(0) = 50 \Rightarrow C = -150$

**Answer:** $x(t) = 200 - 150e^{-3t/100}$

As $t \to \infty$, $x \to 200$ lbs (equilibrium).

---

### 6. RC Circuits

**Kirchhoff's voltage law:**
$$L\frac{dI}{dt} + RI = E(t)$$ (RL circuit)
$$R\frac{dQ}{dt} + \frac{Q}{C} = E(t)$$ (RC circuit)

### Example 6: RC Circuit
A series RC circuit has R = 10 Ω, C = 0.01 F, and is connected to a 12 V battery at t = 0. Find the charge Q(t) and current I(t).

**ODE:**
$$10\frac{dQ}{dt} + \frac{Q}{0.01} = 12$$
$$\frac{dQ}{dt} + 10Q = 1.2$$

**Integrating factor:** $\mu = e^{10t}$

**Solution:** $Q = 0.12 + Ce^{-10t}$

**IC:** $Q(0) = 0 \Rightarrow C = -0.12$

**Charge:** $Q(t) = 0.12(1 - e^{-10t})$ coulombs

**Current:** $I(t) = \frac{dQ}{dt} = 1.2e^{-10t}$ amperes

Time constant: $\tau = RC = 0.1$ s

---

## 📋 Summary: Integrating Factor Method

| Step | Action |
|------|--------|
| 1 | Write in standard form: $y' + P(x)y = Q(x)$ |
| 2 | Find integrating factor: $\mu = e^{\int P dx}$ |
| 3 | Multiply both sides by μ |
| 4 | Recognize left side as $\frac{d}{dx}[\mu y]$ |
| 5 | Integrate: $\mu y = \int \mu Q \, dx + C$ |
| 6 | Solve for y and apply initial conditions |

---

## 📝 Practice Problems

### Level 1: Basic Integrating Factor
1. Solve $y' + y = e^x$
2. Solve $y' - 2y = 4$
3. Solve $y' + 3y = 6$

### Level 2: Variable Coefficients
4. Solve $y' + \frac{2}{x}y = x^3$
5. Solve $xy' + y = x^2$
6. Solve $y' + y\cot(x) = \csc(x)$

### Level 3: With Initial Conditions
7. Solve $y' + 2y = e^{-x}$, $y(0) = 2$
8. Solve $xy' - 2y = x^3$, $y(1) = 0$
9. Solve $y' + y = \sin(x)$, $y(0) = 1$

### Level 4: Applications
10. A tank contains 200 gallons of water with 10 lbs of salt. Pure water flows in at 5 gal/min and the mixture flows out at 5 gal/min. Find the amount of salt after 20 minutes.
11. An RC circuit has R = 5 Ω, C = 0.1 F, and E = 20 V. If Q(0) = 0, find Q(t) and I(t).
12. Newton's Law of Cooling: $\frac{dT}{dt} = -k(T - 25)$. If T(0) = 100 and T(5) = 75, find T(t).

### Level 5: Challenging
13. Solve $y' + y = f(x)$ where $f(x) = \begin{cases} 1 & 0 \leq x < 1 \\ 0 & x \geq 1 \end{cases}$, $y(0) = 0$
14. Find all solutions to $y' + y = y^2$ (Hint: Bernoulli equation)
15. Solve $(1-x^2)y' - xy = 1$

---

## 📊 Answers

1. $y = \frac{1}{2}e^x + Ce^{-x}$
2. $y = -2 + Ce^{2x}$
3. $y = 2 + Ce^{-3x}$
4. $y = \frac{x^4}{6} + \frac{C}{x^2}$
5. $y = \frac{x^2}{3} + \frac{C}{x}$
6. $y = -\cos(x) + C\sin(x)$
7. $y = e^{-x} + e^{-2x}$
8. $y = x^3 - x^2$
9. $y = \frac{1}{2}(\sin x - \cos x) + \frac{3}{2}e^{-x}$
10. $x(20) = 10e^{-0.5} \approx 6.07$ lbs
11. $Q(t) = 2(1-e^{-2t})$, $I(t) = 4e^{-2t}$
12. $T(t) = 25 + 75e^{-kt}$ where $k = \frac{\ln(75/50)}{5}$
13. Piecewise solution
14. $y = \frac{1}{1 + Ce^x}$
15. $y = \frac{x + C\sqrt{1-x^2}}{1-x^2}$

---

## 🔬 Quantum Mechanics Connection

### Probability Decay

When measuring a quantum system, the probability of remaining in the initial state often decays:
$$\frac{dP}{dt} = -\Gamma P$$

This is a first-order linear ODE with solution:
$$P(t) = P_0 e^{-\Gamma t}$$

where Γ is the decay rate.

### Master Equations

Open quantum systems are described by master equations of the form:
$$\frac{d\rho}{dt} = -\frac{i}{\hbar}[H, \rho] + \mathcal{L}[\rho]$$

The Lindblad term $\mathcal{L}$ often leads to exponential decay of coherence.

---

## ✅ Daily Checklist

- [ ] Read Boyce & DiPrima Section 2.1
- [ ] Derive the integrating factor formula
- [ ] Practice the integrating factor method
- [ ] Solve mixing problems
- [ ] Solve RC circuit problems
- [ ] Complete practice problems

---

## 🔜 Preview: Tomorrow

**Day 59: Exact Equations and Special Methods**
- Exact differential equations
- Test for exactness
- Finding the potential function
- Bernoulli equations

---

*"The integrating factor is the key that unlocks linear equations—transforming complexity into simplicity."*
