# Day 63: Rest and Review

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 10:00 AM - 11:30 AM | 1.5 hours | Week 9 Review |
| Afternoon | 2:00 PM - 3:00 PM | 1 hour | Self-Assessment |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Week 10 Preview |

**Total Study Time: 3.5 hours (REST DAY)**

---

## 🎉 Week 9 Complete!

You've mastered first-order ordinary differential equations—the foundation of all differential equations work!

---

## 📝 Week 9 Summary Sheet

### Classification of First-Order ODEs

| Type | Form | Recognition |
|------|------|-------------|
| Separable | $\frac{dy}{dx} = f(x)g(y)$ | Variables can be separated |
| Linear | $y' + P(x)y = Q(x)$ | Linear in y and y' |
| Exact | $M dx + N dy = 0$, $M_y = N_x$ | Test partial derivatives |
| Bernoulli | $y' + P(x)y = Q(x)y^n$ | Power of y on right |
| Homogeneous | $y' = F(y/x)$ | Same degree in x and y |

### Solution Methods

**Separable:**
$$\int \frac{dy}{g(y)} = \int f(x) \, dx + C$$

**Linear (Integrating Factor):**
$$\mu = e^{\int P(x) dx}, \quad y = \frac{1}{\mu}\left[\int \mu Q \, dx + C\right]$$

**Exact:**
Find F where $F_x = M$, $F_y = N$; solution is $F(x, y) = C$

**Bernoulli:**
Substitute $v = y^{1-n}$ to get linear equation

**Homogeneous:**
Substitute $v = y/x$ to get separable equation

---

## 📊 Key Applications

### Exponential Models
$$\frac{dy}{dt} = ky \quad \Rightarrow \quad y = y_0 e^{kt}$$

- Half-life: $t_{1/2} = \frac{\ln 2}{|k|}$
- Doubling time: $t_d = \frac{\ln 2}{k}$

### Newton's Law of Cooling
$$\frac{dT}{dt} = -k(T - T_s) \quad \Rightarrow \quad T = T_s + (T_0 - T_s)e^{-kt}$$

### Logistic Growth
$$\frac{dP}{dt} = rP\left(1 - \frac{P}{K}\right) \quad \Rightarrow \quad P = \frac{K}{1 + Ae^{-rt}}$$

### Mixing Problems
$$\frac{dx}{dt} = (\text{rate in}) - (\text{rate out})$$

---

## 🔄 Self-Assessment Quiz

### Quick Checks (2 minutes each)

1. What type of ODE is $y' = xy^2$?
2. What is the integrating factor for $y' + 3y = e^x$?
3. Is $(2xy)dx + (x^2 + 1)dy = 0$ exact?
4. What substitution transforms $y' + y = y^3$ into a linear equation?
5. What is the half-life if k = 0.1/day?

### Answers
1. Separable (and Bernoulli with n=2)
2. $\mu = e^{3x}$
3. Yes: $M_y = 2x = N_x$
4. $v = y^{-2}$
5. $t_{1/2} = \ln 2/0.1 \approx 6.93$ days

---

## 📈 Skills Checklist

Rate yourself 1-5:

| Skill | Rating |
|-------|--------|
| Solving separable equations | /5 |
| Using integrating factors | /5 |
| Testing for exactness | /5 |
| Solving exact equations | /5 |
| Bernoulli substitution | /5 |
| Homogeneous substitution | /5 |
| Exponential growth/decay | /5 |
| Newton's Law of Cooling | /5 |
| Logistic growth | /5 |
| Mixing problems | /5 |
| Numerical solutions (Python) | /5 |
| Direction fields | /5 |

**Target:** 4+ on each before proceeding

---

## 🔜 Week 10 Preview: Second-Order Linear ODEs

### The Big Picture

Second-order ODEs describe systems with **acceleration**—springs, circuits, and quantum mechanics!

$$a\frac{d^2y}{dx^2} + b\frac{dy}{dx} + cy = f(x)$$

### Topics Coming Up

**Day 64:** Homogeneous equations with constant coefficients
- Characteristic equation
- Real distinct roots
- Complex roots (oscillations!)
- Repeated roots

**Day 65:** Nonhomogeneous equations
- Method of undetermined coefficients
- Particular solutions

**Day 66:** Variation of parameters
- General method for any forcing function

**Day 67:** Applications
- Harmonic oscillator
- Damped oscillations
- Forced oscillations and resonance
- RLC circuits

### Why This Matters for Quantum

The time-independent Schrödinger equation is a second-order ODE:
$$-\frac{\hbar^2}{2m}\frac{d^2\psi}{dx^2} + V(x)\psi = E\psi$$

Solutions give wave functions and energy levels!

### Key Equation Types

**Harmonic Oscillator:**
$$\frac{d^2x}{dt^2} + \omega^2 x = 0$$

**Damped Oscillator:**
$$\frac{d^2x}{dt^2} + 2\gamma\frac{dx}{dt} + \omega_0^2 x = 0$$

**Forced Oscillator:**
$$\frac{d^2x}{dt^2} + 2\gamma\frac{dx}{dt} + \omega_0^2 x = F_0\cos(\omega t)$$

---

## 📚 Preparation for Week 10

### Review These Topics
- Quadratic formula
- Complex numbers: $e^{i\theta} = \cos\theta + i\sin\theta$
- Basic trigonometric identities

### Preview Reading
- Boyce & DiPrima Sections 3.1-3.2
- Complex exponentials

---

## 💡 Concept Map: First-Order ODEs

```
                    First-Order ODEs
                          |
        ┌─────────────────┼─────────────────┐
        |                 |                 |
    Separable         Linear            Special
        |                 |                 |
   f(x)g(y)        y' + Py = Q         ┌───┴───┐
        |                 |             |       |
  Separate &      Integrating      Exact   Bernoulli
   Integrate        Factor              |       |
                      |              M_y=N_x   v=y^{1-n}
                    μ = e^∫P               |       |
                                    Find F    Linear
```

---

## 📓 Reflection Questions

1. Which first-order method do you find most intuitive? Most challenging?
2. How do exact equations relate to conservative vector fields from Week 8?
3. What real-world phenomena follow exponential vs. logistic growth?
4. How might first-order methods extend to higher-order equations?

---

## ✅ Checklist Before Week 10

- [ ] All Week 9 practice problems completed
- [ ] Problem set scored 140+/200
- [ ] Computational lab finished
- [ ] Can identify ODE types quickly
- [ ] Confident with each solution method
- [ ] Ready for second-order equations!

---

## 🧘 Rest Day Activities

**Suggested:**
- Light review of challenging topics
- Watch 3Blue1Brown's differential equations video
- Take a walk and think about exponential growth in nature
- Review complex numbers if needed

---

**Week 9 Complete! 🎉**

You can now solve any first-order ODE. Next: the fascinating world of oscillations and second-order equations!

*"Every expert was once a beginner. You've completed the first step in mastering differential equations."*
