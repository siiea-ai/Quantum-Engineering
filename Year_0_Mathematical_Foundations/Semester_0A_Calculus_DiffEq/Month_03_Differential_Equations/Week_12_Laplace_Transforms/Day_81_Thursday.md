# Day 81: Step Functions and Impulses

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Heaviside Step Function |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Dirac Delta & Applications |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Practice |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Define and use the Heaviside step function
2. Apply the second shifting theorem (t-shifting)
3. Transform piecewise-defined functions
4. Understand the Dirac delta function
5. Solve ODEs with discontinuous or impulsive forcing

---

## 📚 Required Reading

### Primary Text: Boyce & DiPrima (11th Edition)
- **Section 6.3**: Step Functions (pp. 316-328)
- **Section 6.5**: Impulse Functions (pp. 340-349)

---

## 📖 Part I: The Heaviside Step Function

### 1. Definition

> **Heaviside Step Function:**
> $$u_c(t) = u(t-c) = \begin{cases} 0, & t < c \\ 1, & t \geq c \end{cases}$$

Also written as $H(t-c)$ or $\theta(t-c)$.

### 2. Laplace Transform of Step Function

$$\mathcal{L}\{u(t-c)\} = \frac{e^{-cs}}{s}$$

**Proof:**
$$\mathcal{L}\{u(t-c)\} = \int_c^\infty e^{-st} dt = \frac{e^{-cs}}{s}$$

### 3. Second Shifting Theorem (t-Shifting)

> **Theorem:** If $\mathcal{L}\{f(t)\} = F(s)$, then:
> $$\mathcal{L}\{u(t-c) \cdot f(t-c)\} = e^{-cs}F(s)$$

**Inverse form:**
$$\mathcal{L}^{-1}\{e^{-cs}F(s)\} = u(t-c) \cdot f(t-c)$$

---

## ✏️ Step Function Examples

### Example 1: Basic Step
Find $\mathcal{L}\{u(t-3)\}$

$$\mathcal{L}\{u(t-3)\} = \frac{e^{-3s}}{s}$$

---

### Example 2: Shifted Function
Find $\mathcal{L}\{u(t-2)(t-2)^2\}$

Let $f(t) = t^2$, so $F(s) = \frac{2}{s^3}$

Using second shifting theorem:
$$\mathcal{L}\{u(t-2)(t-2)^2\} = e^{-2s} \cdot \frac{2}{s^3}$$

---

### Example 3: Piecewise Function
Find the Laplace transform of:
$$f(t) = \begin{cases} 0, & 0 \leq t < 2 \\ t-2, & t \geq 2 \end{cases}$$

Write as: $f(t) = u(t-2)(t-2)$

$$\mathcal{L}\{f(t)\} = e^{-2s} \cdot \frac{1}{s^2}$$

---

### Example 4: Rectangular Pulse
Find $\mathcal{L}$ of a pulse of height 1 from $t = 1$ to $t = 3$:
$$f(t) = u(t-1) - u(t-3)$$

$$\mathcal{L}\{f(t)\} = \frac{e^{-s}}{s} - \frac{e^{-3s}}{s} = \frac{e^{-s} - e^{-3s}}{s}$$

---

### Example 5: Writing Functions with Steps
Express in terms of step functions:
$$g(t) = \begin{cases} t, & 0 \leq t < 1 \\ 2, & 1 \leq t < 3 \\ 0, & t \geq 3 \end{cases}$$

**Solution:**
$$g(t) = t - u(t-1)(t-2) - u(t-3) \cdot 2$$

Or more carefully:
$$g(t) = t[1 - u(t-1)] + 2[u(t-1) - u(t-3)]$$
$$= t - tu(t-1) + 2u(t-1) - 2u(t-3)$$

---

## 📖 Part II: The Dirac Delta Function

### 4. Definition (Informal)

> **Dirac Delta:** $\delta(t-c)$ is a "function" satisfying:
> 1. $\delta(t-c) = 0$ for $t \neq c$
> 2. $\int_{-\infty}^{\infty} \delta(t-c) dt = 1$
> 3. $\int_{-\infty}^{\infty} f(t)\delta(t-c) dt = f(c)$ (sifting property)

Physically: An **instantaneous impulse** at time $c$.

### 5. Laplace Transform of Delta

$$\mathcal{L}\{\delta(t-c)\} = e^{-cs}$$

Special case: $\mathcal{L}\{\delta(t)\} = 1$

### 6. Relation to Step Function

$$\delta(t) = \frac{d}{dt}u(t)$$

or equivalently: $u(t) = \int_{-\infty}^t \delta(\tau) d\tau$

---

## ✏️ Delta Function Examples

### Example 6: Impulse Response
Solve $y'' + 4y = \delta(t)$, $y(0) = 0$, $y'(0) = 0$

**Transform:**
$$s^2Y + 4Y = 1$$
$$Y(s) = \frac{1}{s^2+4}$$

**Invert:**
$$y(t) = \frac{1}{2}\sin(2t)$$

This is the **impulse response** of the system!

---

### Example 7: Delayed Impulse
Solve $y'' + y = \delta(t-\pi)$, $y(0) = 0$, $y'(0) = 1$

**Transform:**
$$s^2Y - 1 + Y = e^{-\pi s}$$
$$(s^2+1)Y = 1 + e^{-\pi s}$$
$$Y(s) = \frac{1}{s^2+1} + \frac{e^{-\pi s}}{s^2+1}$$

**Invert:**
$$y(t) = \sin(t) + u(t-\pi)\sin(t-\pi)$$

Since $\sin(t-\pi) = -\sin(t)$:
$$y(t) = \begin{cases} \sin(t), & 0 \leq t < \pi \\ 0, & t \geq \pi \end{cases}$$

---

### Example 8: Switched Forcing
Solve $y' + y = u(t-2)$, $y(0) = 0$

**Transform:**
$$sY + Y = \frac{e^{-2s}}{s}$$
$$Y(s) = \frac{e^{-2s}}{s(s+1)}$$

**Partial fractions:** $\frac{1}{s(s+1)} = \frac{1}{s} - \frac{1}{s+1}$

So: $Y(s) = e^{-2s}\left(\frac{1}{s} - \frac{1}{s+1}\right)$

**Invert:**
$$y(t) = u(t-2)[1 - e^{-(t-2)}]$$

---

## 📋 Summary: Shifting Theorems

| Theorem | Formula |
|---------|---------|
| **First Shifting (s-shift)** | $\mathcal{L}\{e^{at}f(t)\} = F(s-a)$ |
| **Second Shifting (t-shift)** | $\mathcal{L}\{u(t-c)f(t-c)\} = e^{-cs}F(s)$ |

| Transform | $F(s)$ |
|-----------|--------|
| $u(t-c)$ | $e^{-cs}/s$ |
| $\delta(t-c)$ | $e^{-cs}$ |
| $\delta(t)$ | $1$ |

---

## 📝 Practice Problems

### Level 1: Step Functions
1. $\mathcal{L}\{u(t-5)\}$
2. $\mathcal{L}\{u(t-1)(t-1)\}$
3. $\mathcal{L}\{u(t-2)e^{-(t-2)}\}$

### Level 2: Piecewise Functions
4. Find $\mathcal{L}\{f(t)\}$ where $f(t) = \begin{cases} 1, & 0 \leq t < 2 \\ 0, & t \geq 2 \end{cases}$
5. Express and transform: $f(t) = \begin{cases} t, & 0 \leq t < 1 \\ 1, & t \geq 1 \end{cases}$

### Level 3: Delta Functions
6. $\mathcal{L}\{\delta(t-3)\}$
7. Solve: $y' + 2y = \delta(t)$, $y(0) = 0$
8. Solve: $y'' + y = \delta(t-\pi/2)$, $y(0) = 1$, $y'(0) = 0$

### Level 4: Applications
9. Solve: $y' + y = u(t-1) - u(t-2)$, $y(0) = 0$
10. A mass-spring system receives a hammer blow at $t = 1$: $y'' + 4y = 3\delta(t-1)$, $y(0) = 0$, $y'(0) = 0$

---

## 📊 Answers

1. $e^{-5s}/s$
2. $e^{-s}/s^2$
3. $e^{-2s}/(s+1)$
4. $(1-e^{-2s})/s$
5. $1/s^2 - e^{-s}/s^2 + e^{-s}/s$
6. $e^{-3s}$
7. $y = e^{-2t}u(t)$
8. $y = \cos t + u(t-\pi/2)\sin(t-\pi/2)$
9. $y = u(t-1)(1-e^{-(t-1)}) - u(t-2)(1-e^{-(t-2)})$
10. $y = \frac{3}{2}u(t-1)\sin(2(t-1))$

---

## 🔬 Quantum Mechanics Connection

### Sudden Perturbation

When a quantum system receives a sudden "kick":
$$H(t) = H_0 + V\delta(t)$$

The delta function models **instantaneous** perturbations. The Laplace transform approach gives the response function.

### Fermi's Golden Rule

Transition rates in quantum mechanics involve integrals similar to:
$$\int_{-\infty}^{\infty} e^{i\omega t}\delta(E_f - E_i - \hbar\omega) d\omega$$

---

## ✅ Daily Checklist

- [ ] Understand step and delta functions
- [ ] Apply second shifting theorem
- [ ] Convert piecewise functions to step notation
- [ ] Solve ODEs with discontinuous forcing
- [ ] Complete practice problems

---

## 🔜 Preview: Tomorrow

**Day 82: Week 12 Problem Set**
- Comprehensive assessment of Laplace methods

---

*"Step functions model switches; delta functions model impulses—together they handle any real-world forcing."*
