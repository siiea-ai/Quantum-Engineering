# Day 82: Week 12 Problem Set — Laplace Transform Mastery

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Parts I-II |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Parts III-IV |
| Evening | 7:00 PM - 8:30 PM | 1.5 hours | Review |

**Total Study Time: 7.5 hours**

---

## 📋 Instructions

This comprehensive problem set covers Laplace transforms. Show all work clearly.

**Total Points:** 200

---

# 📝 PART I: BASIC TRANSFORMS (40 points)

**Problem 1** (8 pts): Compute from definition:
$$\mathcal{L}\{t e^{-2t}\}$$

**Problem 2** (8 pts): Find using table and properties:
(a) $\mathcal{L}\{3t^2 - 2\cos(4t)\}$
(b) $\mathcal{L}\{e^{3t}\sin(2t)\}$

**Problem 3** (8 pts): Find inverse transforms:
(a) $\mathcal{L}^{-1}\left\{\frac{3}{s^4}\right\}$
(b) $\mathcal{L}^{-1}\left\{\frac{s+1}{s^2+9}\right\}$

**Problem 4** (8 pts): Use partial fractions to find:
$$\mathcal{L}^{-1}\left\{\frac{5s-2}{s^2-4}\right\}$$

**Problem 5** (8 pts): Find using completing the square:
$$\mathcal{L}^{-1}\left\{\frac{s+3}{s^2+6s+13}\right\}$$

---

# 📝 PART II: SOLVING ODEs (50 points)

**Problem 6** (10 pts): Solve using Laplace transforms:
$$y' + 5y = e^{-2t}, \quad y(0) = 3$$

**Problem 7** (10 pts): Solve:
$$y'' - 4y = 0, \quad y(0) = 2, \quad y'(0) = 0$$

**Problem 8** (10 pts): Solve:
$$y'' + 2y' + 5y = 0, \quad y(0) = 1, \quad y'(0) = -1$$

**Problem 9** (10 pts): Solve:
$$y'' + 4y = \sin(2t), \quad y(0) = 0, \quad y'(0) = 0$$

**Problem 10** (10 pts): Solve:
$$y'' - 2y' + y = e^t, \quad y(0) = 0, \quad y'(0) = 1$$

---

# 📝 PART III: STEP FUNCTIONS & IMPULSES (50 points)

**Problem 11** (10 pts): Find the Laplace transform:
$$f(t) = \begin{cases} 0, & 0 \leq t < 3 \\ t-3, & t \geq 3 \end{cases}$$

**Problem 12** (10 pts): Find the inverse transform:
$$\mathcal{L}^{-1}\left\{\frac{e^{-2s}(1-e^{-s})}{s}\right\}$$

**Problem 13** (10 pts): Solve:
$$y' + y = u(t-2), \quad y(0) = 1$$

**Problem 14** (10 pts): Solve:
$$y'' + y = \delta(t-\pi), \quad y(0) = 0, \quad y'(0) = 1$$

**Problem 15** (10 pts): A spring-mass system at rest receives an impulse:
$$y'' + 4y = 5\delta(t-1), \quad y(0) = 0, \quad y'(0) = 0$$
Find the position $y(t)$.

---

# 📝 PART IV: APPLICATIONS (60 points)

**Problem 16** (15 pts): **RLC Circuit**

An RLC circuit with $L = 1$ H, $R = 4$ Ω, $C = 1/5$ F has initial charge $Q(0) = 2$ C and current $I(0) = 0$ A. The voltage source is $E(t) = 10$ V.

The governing equation is: $LQ'' + RQ' + Q/C = E(t)$

(a) Write the ODE with given values
(b) Solve for $Q(t)$ using Laplace transforms
(c) Find the steady-state charge

**Problem 17** (15 pts): **Switched Input**

A system satisfies:
$$y'' + 3y' + 2y = f(t), \quad y(0) = 0, \quad y'(0) = 0$$

where $f(t) = \begin{cases} 0, & 0 \leq t < 1 \\ 1, & t \geq 1 \end{cases}$

(a) Write $f(t)$ using step functions
(b) Find $Y(s)$
(c) Find $y(t)$

**Problem 18** (15 pts): **Resonance**

Solve the resonance problem:
$$y'' + \omega_0^2 y = \cos(\omega_0 t), \quad y(0) = 0, \quad y'(0) = 0$$

(a) Take the Laplace transform
(b) Find $Y(s)$
(c) Use partial fractions (note the repeated factor)
(d) Show $y(t) = \frac{t}{2\omega_0}\sin(\omega_0 t)$

**Problem 19** (15 pts): **System of ODEs**

Solve using Laplace transforms:
$$x' = -x + y, \quad x(0) = 1$$
$$y' = -y, \quad y(0) = 1$$

(a) Transform both equations
(b) Solve for $X(s)$ and $Y(s)$
(c) Invert to find $x(t)$ and $y(t)$

---

# ✅ ANSWER KEY

## Part I: Basic Transforms

**1.** Using first shifting: $\mathcal{L}\{te^{-2t}\} = \frac{1}{(s+2)^2}$

**2.** 
(a) $\frac{6}{s^3} - \frac{2s}{s^2+16}$
(b) $\frac{2}{(s-3)^2+4}$

**3.**
(a) $\frac{t^3}{2}$
(b) $\cos(3t) + \frac{1}{3}\sin(3t)$

**4.** $\frac{5s-2}{(s-2)(s+2)} = \frac{2}{s-2} + \frac{3}{s+2}$ → $y = 2e^{2t} + 3e^{-2t}$

**5.** $s^2+6s+13 = (s+3)^2+4$, so answer is $e^{-3t}\cos(2t)$

## Part II: Solving ODEs

**6.** $Y = \frac{3}{s+5} + \frac{1}{(s+2)(s+5)} = \frac{3}{s+5} + \frac{1/3}{s+2} - \frac{1/3}{s+5}$
$$y = \frac{8}{3}e^{-5t} + \frac{1}{3}e^{-2t}$$

**7.** $Y = \frac{2s}{s^2-4}$ → $y = e^{2t} + e^{-2t} = 2\cosh(2t)$

**8.** $Y = \frac{s+1}{(s+1)^2+4}$ → $y = e^{-t}\cos(2t)$

**9.** $Y = \frac{2}{(s^2+4)^2}$ → $y = \frac{1}{4}(\sin 2t - 2t\cos 2t)$ (resonance effect)

**10.** $Y = \frac{1}{(s-1)^3} + \frac{1}{(s-1)^2}$ → $y = \frac{t^2}{2}e^t + te^t$

## Part III: Step Functions & Impulses

**11.** $f(t) = u(t-3)(t-3)$ → $F(s) = \frac{e^{-3s}}{s^2}$

**12.** $f(t) = u(t-2) - u(t-3)$ (pulse from $t=2$ to $t=3$)

**13.** $Y = \frac{1}{s+1} + \frac{e^{-2s}}{s(s+1)}$
$$y = e^{-t} + u(t-2)(1-e^{-(t-2)})$$

**14.** $Y = \frac{1}{s^2+1} + \frac{e^{-\pi s}}{s^2+1}$
$$y = \sin t + u(t-\pi)\sin(t-\pi) = \sin t - u(t-\pi)\sin t$$

**15.** $Y = \frac{5e^{-s}}{s^2+4}$ → $y = \frac{5}{2}u(t-1)\sin(2(t-1))$

## Part IV: Applications

**16.**
(a) $Q'' + 4Q' + 5Q = 10$
(b) $Q = 2 + e^{-2t}(c_1\cos t + c_2\sin t)$... (complete with ICs)
(c) Steady-state: $Q_{ss} = 2$ C

**17.**
(a) $f(t) = u(t-1)$
(b) $Y = \frac{e^{-s}}{s(s+1)(s+2)}$
(c) $y = u(t-1)\left[\frac{1}{2} - e^{-(t-1)} + \frac{1}{2}e^{-2(t-1)}\right]$

**18.**
(a) $(s^2+\omega_0^2)Y = \frac{s}{s^2+\omega_0^2}$
(b) $Y = \frac{s}{(s^2+\omega_0^2)^2}$
(c) Use $\mathcal{L}^{-1}\{s/(s^2+a^2)^2\} = \frac{t}{2a}\sin(at)$
(d) $y = \frac{t}{2\omega_0}\sin(\omega_0 t)$ ✓

**19.**
(a) $(s+1)X - Y = 1$, $(s+1)Y = 1$
(b) $Y = \frac{1}{s+1}$, $X = \frac{1}{s+1} + \frac{1}{(s+1)^2}$
(c) $y = e^{-t}$, $x = e^{-t} + te^{-t}$

---

## 📊 Scoring Guide

| Part | Points | Your Score |
|------|--------|------------|
| I: Basic Transforms (5 × 8) | 40 | |
| II: Solving ODEs (5 × 10) | 50 | |
| III: Steps & Impulses (5 × 10) | 50 | |
| IV: Applications (4 × 15) | 60 | |
| **TOTAL** | **200** | |

---

## 🔜 Tomorrow: Computational Lab

---

*"Laplace transforms: where differential equations become algebra problems."*
