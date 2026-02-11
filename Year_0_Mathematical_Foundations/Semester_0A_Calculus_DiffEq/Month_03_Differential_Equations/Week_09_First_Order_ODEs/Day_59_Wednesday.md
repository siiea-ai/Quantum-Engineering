# Day 59: Exact Equations and Special Methods

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Exact Equations |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Bernoulli & Substitution Methods |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Practice |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Recognize exact differential equations
2. Test for exactness using partial derivatives
3. Solve exact equations by finding the potential function
4. Apply integrating factors to make equations exact
5. Solve Bernoulli equations via substitution

---

## 📚 Required Reading

### Primary Text: Boyce & DiPrima (11th Edition)
- **Section 2.6**: Exact Equations and Integrating Factors (pp. 96-106)
- **Section 2.4**: Differences Between Linear and Nonlinear Equations (Bernoulli)

---

## 📖 Core Content: Exact Equations

### 1. Differential Form

A first-order ODE can be written in differential form:
$$M(x,y) \, dx + N(x,y) \, dy = 0$$

### 2. Definition of Exactness

> **Definition:** The equation $M \, dx + N \, dy = 0$ is **exact** if there exists a function F(x,y) such that:
> $$\frac{\partial F}{\partial x} = M \quad \text{and} \quad \frac{\partial F}{\partial y} = N$$

If exact, the solution is given implicitly by:
$$F(x,y) = C$$

### 3. Test for Exactness

> **Theorem:** If M and N have continuous first partial derivatives, then $M \, dx + N \, dy = 0$ is exact if and only if:
> $$\frac{\partial M}{\partial y} = \frac{\partial N}{\partial x}$$

**Why?** If F exists, then:
$$\frac{\partial M}{\partial y} = \frac{\partial^2 F}{\partial y \partial x} = \frac{\partial^2 F}{\partial x \partial y} = \frac{\partial N}{\partial x}$$

### 4. Solving Exact Equations

**Step 1:** Verify exactness: $\frac{\partial M}{\partial y} = \frac{\partial N}{\partial x}$

**Step 2:** Find F by integrating M with respect to x:
$$F(x,y) = \int M(x,y) \, dx + g(y)$$

**Step 3:** Determine g(y) using $\frac{\partial F}{\partial y} = N$

**Step 4:** Solution is $F(x,y) = C$

---

## ✏️ Exact Equation Examples

### Example 1: Basic Exact Equation
Solve $(2xy + 3) \, dx + (x^2 + 4y) \, dy = 0$.

**Test exactness:**
$$M = 2xy + 3, \quad N = x^2 + 4y$$
$$\frac{\partial M}{\partial y} = 2x, \quad \frac{\partial N}{\partial x} = 2x$$ ✓ Exact!

**Find F:**
$$F = \int (2xy + 3) \, dx = x^2y + 3x + g(y)$$

**Determine g(y):**
$$\frac{\partial F}{\partial y} = x^2 + g'(y) = N = x^2 + 4y$$
$$g'(y) = 4y \Rightarrow g(y) = 2y^2$$

**Solution:** $x^2y + 3x + 2y^2 = C$

---

### Example 2: With Initial Condition
Solve $(ye^{xy} + 2x) \, dx + (xe^{xy} + 2y) \, dy = 0$, $y(0) = 1$.

**Test exactness:**
$$\frac{\partial M}{\partial y} = e^{xy} + xye^{xy}$$
$$\frac{\partial N}{\partial x} = e^{xy} + xye^{xy}$$ ✓ Exact!

**Find F:**
$$F = \int (ye^{xy} + 2x) \, dx = e^{xy} + x^2 + g(y)$$

**Determine g(y):**
$$\frac{\partial F}{\partial y} = xe^{xy} + g'(y) = xe^{xy} + 2y$$
$$g'(y) = 2y \Rightarrow g(y) = y^2$$

**Solution:** $e^{xy} + x^2 + y^2 = C$

**Apply IC:** $e^0 + 0 + 1 = 2 = C$

**Answer:** $e^{xy} + x^2 + y^2 = 2$

---

### Example 3: Not Exact - Finding Integrating Factor
Solve $(y + 1) \, dx - x \, dy = 0$.

**Test:**
$$\frac{\partial M}{\partial y} = 1, \quad \frac{\partial N}{\partial x} = -1$$

Not exact! Try an integrating factor μ.

**If μ = μ(x):**
$$\frac{1}{\mu}\frac{d\mu}{dx} = \frac{M_y - N_x}{N} = \frac{1-(-1)}{-x} = -\frac{2}{x}$$
$$\mu = x^{-2}$$

**Multiply:**
$$\frac{y+1}{x^2} \, dx - \frac{1}{x} \, dy = 0$$

Now $M_y = 1/x^2$ and $N_x = 1/x^2$ ✓

**Solve:**
$$F = \int \frac{y+1}{x^2} \, dx = -\frac{y+1}{x} + g(y)$$
$$\frac{\partial F}{\partial y} = -\frac{1}{x} + g'(y) = -\frac{1}{x}$$
$$g'(y) = 0 \Rightarrow g(y) = 0$$

**Solution:** $-\frac{y+1}{x} = C$ or $y = -Cx - 1$

---

## 📖 Bernoulli Equations

### 5. Definition

> **Definition:** A **Bernoulli equation** has the form:
> $$y' + P(x)y = Q(x)y^n$$
> where n ≠ 0, 1.

### 6. Solution Method

**Substitution:** Let $v = y^{1-n}$

Then $v' = (1-n)y^{-n}y'$

The equation transforms to:
$$v' + (1-n)P(x)v = (1-n)Q(x)$$

This is **linear** in v! Solve, then back-substitute.

---

## ✏️ Bernoulli Examples

### Example 4: Bernoulli Equation
Solve $y' + y = y^2$.

**Here:** P = 1, Q = 1, n = 2

**Substitution:** $v = y^{1-2} = y^{-1} = 1/y$

$v' = -y^{-2}y'$, so $y' = -y^2 v' = -v'/v^2$

**Transform:**
$$-\frac{v'}{v^2} + \frac{1}{v} = \frac{1}{v^2}$$

Multiply by $-v^2$:
$$v' - v = -1$$

**Solve linear equation:**
$$\mu = e^{-x}$$
$$v = 1 + Ce^x$$

**Back-substitute:**
$$y = \frac{1}{v} = \frac{1}{1 + Ce^x}$$

---

### Example 5: Another Bernoulli
Solve $y' - \frac{2}{x}y = -x^2y^2$.

**Here:** P = -2/x, Q = -x², n = 2

**Substitution:** $v = y^{-1}$

**Transformed equation:**
$$v' + \frac{2}{x}v = x^2$$

**Integrating factor:** $\mu = x^2$

$$x^2 v' + 2xv = x^4$$
$$\frac{d}{dx}[x^2 v] = x^4$$
$$x^2 v = \frac{x^5}{5} + C$$
$$v = \frac{x^3}{5} + \frac{C}{x^2}$$

**Solution:** $y = \frac{1}{v} = \frac{x^2}{x^5/5 + C} = \frac{5x^2}{x^5 + 5C}$

---

## 📖 Homogeneous Equations

### 7. Definition

> **Definition:** A first-order ODE is **homogeneous** (in the sense of substitution) if it can be written as:
> $$\frac{dy}{dx} = F\left(\frac{y}{x}\right)$$

### 8. Solution Method

**Substitution:** Let $v = y/x$, so $y = vx$ and $y' = v + xv'$

The equation becomes:
$$v + xv' = F(v)$$
$$xv' = F(v) - v$$

This is separable!

### Example 6: Homogeneous Equation
Solve $y' = \frac{x + y}{x}$.

**Rewrite:** $y' = 1 + y/x = F(y/x)$

**Substitute:** $v = y/x$
$$v + xv' = 1 + v$$
$$xv' = 1$$
$$v' = 1/x$$
$$v = \ln|x| + C$$

**Solution:** $y = x(\ln|x| + C)$

---

## 📋 Summary: First-Order Methods

| Type | Form | Method |
|------|------|--------|
| Separable | $y' = f(x)g(y)$ | Separate and integrate |
| Linear | $y' + P(x)y = Q(x)$ | Integrating factor $e^{\int P dx}$ |
| Exact | $M dx + N dy = 0$, $M_y = N_x$ | Find potential F |
| Bernoulli | $y' + Py = Qy^n$ | Substitute $v = y^{1-n}$ |
| Homogeneous | $y' = F(y/x)$ | Substitute $v = y/x$ |

---

## 📝 Practice Problems

### Level 1: Exactness Test
1. Is $(2x + y) dx + (x + 2y) dy = 0$ exact?
2. Is $(y^2) dx + (2xy) dy = 0$ exact?
3. Is $(e^y) dx + (xe^y + 1) dy = 0$ exact?

### Level 2: Solve Exact Equations
4. Solve $(3x^2 + y) dx + (x + 2y) dy = 0$
5. Solve $(y\cos x + 2xe^y) dx + (\sin x + x^2e^y) dy = 0$
6. Solve $(2xy - 3) dx + (x^2 + 4y) dy = 0$

### Level 3: Bernoulli Equations
7. Solve $y' + y = xy^3$
8. Solve $y' - y/x = y^2/x$
9. Solve $y' + 2y = y^2 e^x$

### Level 4: Homogeneous Equations
10. Solve $y' = \frac{y^2 + xy}{x^2}$
11. Solve $(x^2 + y^2) dx - xy \, dy = 0$
12. Solve $y' = \frac{x + y}{x - y}$

### Level 5: Mixed Problems
13. Classify and solve: $(x + y^2) dx + 2xy \, dy = 0$
14. Find an integrating factor for $y \, dx - x \, dy = 0$ and solve
15. Solve $xy' + y = y^2 \ln x$

---

## 📊 Answers

1. Yes ($M_y = N_x = 1$)
2. Yes ($M_y = N_x = 2y$)
3. Yes ($M_y = e^y$, $N_x = e^y$)
4. $x^3 + xy + y^2 = C$
5. $y\sin x + x^2 e^y = C$
6. $x^2y - 3x + 2y^2 = C$
7. $y^{-2} = x - 1 + Ce^{2x}$
8. $y = \frac{x}{1 - Cx}$
9. $y^{-1} = e^x + Ce^{2x}$
10. $y = \frac{x}{C - \ln|x|}$
11. $x^2 + y^2 = Cx$
12. $\arctan(y/x) = \frac{1}{2}\ln(x^2 + y^2) + C$
13. Exact: $x^2/2 + xy^2 = C$
14. $\mu = 1/y^2$; $x/y = C$
15. $y^{-1} = \ln x + 1 + C/x$

---

## 🔬 Quantum Mechanics Connection

### Hamilton-Jacobi Equation

In classical mechanics, the Hamilton-Jacobi equation:
$$\frac{\partial S}{\partial t} + H\left(q, \frac{\partial S}{\partial q}\right) = 0$$

is a first-order PDE that connects to quantum mechanics through:
$$\psi = e^{iS/\hbar}$$

### Phase Space

Exact equations relate to conservative systems where:
$$H(p, q) = E$$

The solution curves are level sets of the Hamiltonian!

---

## ✅ Daily Checklist

- [ ] Read Boyce & DiPrima Section 2.6
- [ ] Test equations for exactness
- [ ] Solve exact equations systematically
- [ ] Master Bernoulli substitution
- [ ] Practice homogeneous equation substitution
- [ ] Complete practice problems

---

## 🔜 Preview: Tomorrow

**Day 60: Applications and Modeling**
- Population dynamics
- Radioactive decay and carbon dating
- Newton's Law of Cooling
- Mixture problems revisited
- Orthogonal trajectories

---

*"Exact equations reveal hidden conservation laws—when the curl vanishes, a potential exists."*
