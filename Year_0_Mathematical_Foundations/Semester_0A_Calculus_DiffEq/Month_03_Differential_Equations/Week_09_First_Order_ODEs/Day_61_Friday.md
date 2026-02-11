# Day 61: Week 9 Problem Set — First-Order ODEs Mastery

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Parts I-II |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Parts III-IV |
| Evening | 7:00 PM - 8:30 PM | 1.5 hours | Review & Self-Assessment |

**Total Study Time: 7.5 hours**

---

## 📋 Instructions

This problem set covers all first-order ODE techniques. Show all work clearly.

**Time:** 4 hours recommended for problems
**Total Points:** 200

---

# 📝 PART I: SEPARABLE EQUATIONS (40 points)

**Problem 1** (8 pts): Solve and find the general solution:
$$\frac{dy}{dx} = \frac{x^2}{y}$$

**Problem 2** (8 pts): Solve the IVP:
$$\frac{dy}{dx} = y^2 \sin x, \quad y(0) = 1$$

**Problem 3** (8 pts): Solve:
$$\frac{dy}{dx} = \frac{1 + y^2}{1 + x^2}$$

**Problem 4** (8 pts): Solve the IVP:
$$y' = e^{x-y}, \quad y(0) = 0$$

**Problem 5** (8 pts): Solve:
$$\frac{dy}{dx} = \frac{y \ln y}{x}$$

---

# 📝 PART II: LINEAR EQUATIONS (50 points)

**Problem 6** (10 pts): Solve using the integrating factor method:
$$y' + 2y = 4e^{-x}$$

**Problem 7** (10 pts): Solve the IVP:
$$y' - \frac{y}{x} = x^2, \quad y(1) = 2$$

**Problem 8** (10 pts): Solve:
$$y' + y \tan x = \sec x$$

**Problem 9** (10 pts): Solve the IVP:
$$(1 + x^2)y' + 2xy = \frac{1}{1 + x^2}, \quad y(0) = 1$$

**Problem 10** (10 pts): Solve:
$$xy' + (1 + x)y = e^{-x}$$

---

# 📝 PART III: EXACT & SPECIAL EQUATIONS (50 points)

**Problem 11** (10 pts): Test for exactness and solve:
$$(2xy + 1)dx + (x^2 + 4y)dy = 0$$

**Problem 12** (10 pts): Test for exactness and solve:
$$(ye^{xy} + 2x)dx + (xe^{xy} - 2y)dy = 0$$

**Problem 13** (10 pts): Solve the Bernoulli equation:
$$y' + \frac{y}{x} = x^2 y^3$$

**Problem 14** (10 pts): Solve the homogeneous equation:
$$\frac{dy}{dx} = \frac{x^2 + y^2}{xy}$$

**Problem 15** (10 pts): Find an integrating factor and solve:
$$y \, dx - x \, dy = 0$$

---

# 📝 PART IV: APPLICATIONS (60 points)

**Problem 16** (12 pts): **Population Growth**
A bacteria culture grows at a rate proportional to its size. Initially there are 500 bacteria, and after 2 hours there are 2000.
(a) Find the population P(t).
(b) When will the population reach 10,000?
(c) What is the doubling time?

**Problem 17** (12 pts): **Radioactive Decay**
A radioactive isotope has a half-life of 8 days.
(a) Find the decay constant k.
(b) How long until only 5% remains?
(c) If a sample initially has 100 grams, how much remains after 20 days?

**Problem 18** (12 pts): **Newton's Law of Cooling**
A cup of coffee at 95°C is placed in a room at 22°C. After 5 minutes, the temperature is 70°C.
(a) Find the temperature function T(t).
(b) When will the coffee reach 40°C?
(c) What is the temperature after 20 minutes?

**Problem 19** (12 pts): **Mixing Problem**
A 100-gallon tank initially contains 50 gallons of pure water. Brine containing 2 lbs/gal flows in at 3 gal/min, and the well-mixed solution flows out at 2 gal/min.
(a) Set up the differential equation for x(t), the amount of salt.
(b) Solve the equation.
(c) Find the amount of salt when the tank is full.

**Problem 20** (12 pts): **Logistic Growth**
A population follows the logistic equation:
$$\frac{dP}{dt} = 0.1P\left(1 - \frac{P}{5000}\right), \quad P(0) = 500$$
(a) Find the general solution P(t).
(b) Find P(10) and P(50).
(c) When will the population reach 4000?

---

# ✅ ANSWER KEY

## Part I: Separable Equations

**1.** $y^2/2 = x^3/3 + C$, or $y = \pm\sqrt{2x^3/3 + C}$

**2.** $-1/y = -\cos x + C$; with IC: $-1 = -1 + C$, so $C = 0$
$$y = \frac{1}{\cos x} = \sec x$$

**3.** $\arctan y = \arctan x + C$, or $y = \tan(\arctan x + C)$

**4.** $e^y = e^x + C$; with IC: $1 = 1 + C$, so $C = 0$
$$e^y = e^x \Rightarrow y = x$$

**5.** $\ln|\ln y| = \ln|x| + C$, or $\ln y = Ax$, giving $y = e^{Ax}$

## Part II: Linear Equations

**6.** $\mu = e^{2x}$; $y = 4e^{-x} + Ce^{-2x}$

**7.** $\mu = 1/x$; $y = x^3/2 + Cx$; with IC: $y = x^3/2 + 3x/2$

**8.** $\mu = \sec x$; $y = \sin x + C\cos x$

**9.** $\mu = 1 + x^2$; $y = (\arctan x + C)/(1 + x^2)$; with IC: $y = (\arctan x + 1)/(1 + x^2)$

**10.** $\mu = xe^x$; $y = (1 + C/x)e^{-x}$

## Part III: Exact & Special

**11.** $M_y = 2x = N_x$ ✓ Exact; $F = x^2y + x + 2y^2 = C$

**12.** $M_y = e^{xy}(1 + xy)$, $N_x = e^{xy}(1 + xy)$ ✓ Exact; $F = e^{xy} + x^2 - y^2 = C$

**13.** Bernoulli with n = 3; substitute $v = y^{-2}$; $y^{-2} = 2x^2 + Cx^2 = x^2(2 + C/x^4)$

**14.** Substitute $v = y/x$; $y^2 = x^2(2\ln|x| + C)$

**15.** $\mu = 1/y^2$ or $\mu = 1/x^2$; solution $y = Cx$

## Part IV: Applications

**16.** 
(a) $P(t) = 500e^{kt}$ where $k = \ln 4/2$; $P(t) = 500 \cdot 4^{t/2}$
(b) $10000 = 500 \cdot 4^{t/2}$; $t = \ln 20/\ln 2 \approx 4.32$ hours
(c) $t_d = \ln 2/k = 1$ hour

**17.**
(a) $k = \ln 2/8 \approx 0.0866$/day
(b) $0.05 = e^{-kt}$; $t = \ln 20/k \approx 34.6$ days
(c) $N(20) = 100e^{-20k} \approx 17.7$ grams

**18.**
(a) $70 = 22 + 73e^{-5k}$; $k = \ln(73/48)/5$; $T(t) = 22 + 73e^{-kt}$
(b) $40 = 22 + 73e^{-kt}$; $t \approx 17.4$ minutes
(c) $T(20) \approx 30.5°C$

**19.**
(a) $V = 50 + t$; $\frac{dx}{dt} = 6 - \frac{2x}{50+t}$
(b) $x = \frac{2(50+t)^2 - 5000}{50+t} = 2(50+t) - \frac{5000}{50+t}$
(c) At $t = 50$: $x(50) = 200 - 50 = 150$ lbs

**20.**
(a) $P(t) = \frac{5000}{1 + 9e^{-0.1t}}$
(b) $P(10) \approx 1218$; $P(50) \approx 4866$
(c) $4000 = 5000/(1 + 9e^{-0.1t})$; $t = 10\ln 36 \approx 35.8$

---

## 📊 Scoring Guide

| Part | Points | Your Score |
|------|--------|------------|
| I: Separable (5 × 8) | 40 | |
| II: Linear (5 × 10) | 50 | |
| III: Exact/Special (5 × 10) | 50 | |
| IV: Applications (5 × 12) | 60 | |
| **TOTAL** | **200** | |

### Grade Scale
- 180-200: Excellent (A)
- 160-179: Good (B)
- 140-159: Satisfactory (C)
- Below 140: Review needed

---

## 🔜 Tomorrow: Computational Lab

---

*"First-order ODEs are the foundation—master them, and higher-order equations follow naturally."*
