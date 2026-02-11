# Day 68: Week 10 Problem Set — Second-Order ODEs Mastery

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Parts I-II |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Parts III-IV |
| Evening | 7:00 PM - 8:30 PM | 1.5 hours | Review |

**Total Study Time: 7.5 hours**

---

## 📋 Instructions

This problem set tests mastery of second-order linear ODEs. Show all work.

**Total Points:** 200

---

# 📝 PART I: HOMOGENEOUS EQUATIONS (50 points)

## Section A: Constant Coefficients (5 pts each)

**A1.** Solve $y'' + 5y' + 6y = 0$

**A2.** Solve $y'' - 2y' + 5y = 0$

**A3.** Solve $y'' + 8y' + 16y = 0$

**A4.** Solve $y'' + 9y = 0$

**A5.** Solve $y'' - y' - 6y = 0$, $y(0) = 1$, $y'(0) = 0$

## Section B: Classification (5 pts each)

**B1.** Classify and solve: $y'' + 2y' + y = 0$

**B2.** Classify and solve: $y'' + 3y' + y = 0$

**B3.** For $y'' + 2\gamma y' + \omega_0^2 y = 0$, what value of γ gives critical damping when $\omega_0 = 5$?

**B4.** Find all values of k for which $y'' + ky' + 4y = 0$ has oscillatory solutions.

**B5.** Verify that $y_1 = e^x$ and $y_2 = e^{2x}$ are independent solutions to $y'' - 3y' + 2y = 0$ using the Wronskian.

---

# 📝 PART II: NONHOMOGENEOUS EQUATIONS (50 points)

## Section C: Undetermined Coefficients (6 pts each)

**C1.** Solve $y'' - 4y = e^{3x}$

**C2.** Solve $y'' + y = 2x + 1$

**C3.** Solve $y'' + 4y = \cos 2x$ (overlap case!)

**C4.** Solve $y'' - 2y' + y = e^x$ (overlap case!)

**C5.** Solve $y'' + y' - 2y = e^x + x$

## Section D: Variation of Parameters (7 pts each)

**D1.** Solve $y'' + y = \sec x$

**D2.** Solve $y'' + y = \tan x$

**D3.** Solve $y'' - y = \frac{1}{e^x + e^{-x}}$

---

# 📝 PART III: APPLICATIONS (50 points)

## Section E: Mechanical Oscillations (10 pts each)

**E1.** A 1 kg mass on a spring (k = 4 N/m) with damping (c = 2 N·s/m) is released from x = 0.5 m at rest. Find x(t) and classify the motion.

**E2.** An undamped spring-mass system has natural frequency ω₀ = 3 rad/s. If driven by F(t) = 2cos(3t), find the position x(t) with x(0) = 0, x'(0) = 0.

**E3.** A damped oscillator satisfies $x'' + 0.4x' + 4x = 0$. Find:
(a) The damped frequency ωd
(b) The time for amplitude to decay to 1/e of initial value
(c) The number of oscillations for amplitude to decay to 10%

## Section F: RLC Circuits (10 pts each)

**F1.** An RLC circuit has L = 0.5 H, R = 2 Ω, C = 0.125 F. Find the charge Q(t) if Q(0) = 2 C and I(0) = 0.

**F2.** Design an LC circuit (no resistance) with resonant frequency 1000 Hz if L = 0.01 H.

---

# 📝 PART IV: COMPREHENSIVE (50 points)

**G1.** (10 pts) Solve the IVP:
$$y'' + 4y' + 5y = 10e^{-2x}\cos x, \quad y(0) = 0, \quad y'(0) = 1$$

**G2.** (10 pts) A mass-spring system satisfies:
$$x'' + 2x' + 5x = 10\cos t$$
Find the steady-state solution and its amplitude.

**G3.** (10 pts) Use variation of parameters to solve:
$$y'' + 4y = \csc(2x)$$

**G4.** (10 pts) Show that the general solution to $y'' + y = \sec^3 x$ is:
$$y = c_1\cos x + c_2\sin x + \frac{1}{2}\sec x + \frac{1}{2}\sin x \tan x$$

**G5.** (10 pts) A quantum particle in a 1D box satisfies:
$$\psi'' + \frac{2mE}{\hbar^2}\psi = 0$$
with boundary conditions $\psi(0) = \psi(L) = 0$. Find the allowed energies.

---

# ✅ ANSWER KEY

## Part I: Homogeneous

**A1.** $r = -2, -3$; $y = c_1 e^{-2x} + c_2 e^{-3x}$

**A2.** $r = 1 \pm 2i$; $y = e^x(c_1\cos 2x + c_2\sin 2x)$

**A3.** $r = -4$ (repeated); $y = (c_1 + c_2 x)e^{-4x}$

**A4.** $r = \pm 3i$; $y = c_1\cos 3x + c_2\sin 3x$

**A5.** $r = 3, -2$; $y = c_1 e^{3x} + c_2 e^{-2x}$; ICs give $y = \frac{2}{5}e^{3x} + \frac{3}{5}e^{-2x}$

**B1.** Critically damped; $y = (c_1 + c_2 x)e^{-x}$

**B2.** Overdamped; $r = \frac{-3 \pm \sqrt{5}}{2}$

**B3.** $\gamma = 5$

**B4.** $-4 < k < 4$

**B5.** $W = e^{3x} \neq 0$ ✓

## Part II: Nonhomogeneous

**C1.** $y = c_1 e^{2x} + c_2 e^{-2x} + \frac{1}{5}e^{3x}$

**C2.** $y = c_1\cos x + c_2\sin x + 2x + 1$

**C3.** $y = c_1\cos 2x + c_2\sin 2x + \frac{x}{4}\sin 2x$

**C4.** $y = (c_1 + c_2 x)e^x + \frac{x^2}{2}e^x$

**C5.** $y = c_1 e^x + c_2 e^{-2x} + \frac{x}{3}e^x - \frac{x}{2} - \frac{1}{4}$

**D1.** $y = c_1\cos x + c_2\sin x + \cos x \ln|\cos x| + x\sin x$

**D2.** $y = c_1\cos x + c_2\sin x - \cos x \ln|\sec x + \tan x|$

**D3.** $y = c_1 e^x + c_2 e^{-x} + \frac{1}{2}(e^x + e^{-x})\arctan(e^x)$

## Part III: Applications

**E1.** $\gamma = 1$, $\omega_0 = 2$; underdamped; $\omega_d = \sqrt{3}$
$$x = e^{-t}(0.5\cos\sqrt{3}t + \frac{0.5}{\sqrt{3}}\sin\sqrt{3}t)$$

**E2.** Resonance! $x = \frac{t}{3}\sin 3t$ (amplitude grows)

**E3.** (a) $\omega_d = \sqrt{3.96} \approx 1.99$ rad/s
(b) $\tau = 1/0.2 = 5$ s
(c) About 3.7 oscillations

**F1.** $r = -2 \pm 2i$; $Q = e^{-2t}(2\cos 2t + 2\sin 2t)$

**F2.** $C = 1/(4\pi^2 \cdot 10^6 \cdot 0.01) = 2.53$ μF

## Part IV: Comprehensive

**G1.** $y_h = e^{-2x}(c_1\cos x + c_2\sin x)$
Trial overlaps! Use $y_p = xe^{-2x}(A\cos x + B\sin x)$
After work: $y = e^{-2x}(\sin x + 5x\cos x)$

**G2.** $x_p = \frac{10}{(5-1)^2 + 4}\cos(t - \delta) = \frac{5}{4}\cos t + \frac{5}{2}\sin t$
Amplitude = $\frac{5\sqrt{5}}{4}$

**G3.** $y = c_1\cos 2x + c_2\sin 2x - \frac{1}{4}\cos 2x \ln|\csc 2x + \cot 2x|$

**G4.** Verification by substitution

**G5.** $\psi = A\sin(n\pi x/L)$, $E_n = \frac{n^2\pi^2\hbar^2}{2mL^2}$, $n = 1, 2, 3, ...$

---

## 📊 Scoring Guide

| Part | Points | Your Score |
|------|--------|------------|
| I: Homogeneous (10 × 5) | 50 | |
| II: Nonhomogeneous (5×6 + 3×7) | 51 | |
| III: Applications (5 × 10) | 50 | |
| IV: Comprehensive (5 × 10) | 50 | |
| **TOTAL** | **201** | |

### Grade Scale
- 180-201: Excellent (A)
- 160-179: Good (B)
- 140-159: Satisfactory (C)
- Below 140: Review needed

---

## 🔜 Tomorrow: Computational Lab

---

*"Second-order ODEs reveal the dance between acceleration and position—the heart of classical and quantum mechanics."*
