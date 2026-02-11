# Day 67: Mechanical and Electrical Oscillations

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Mechanical Vibrations |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Electrical Circuits |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Practice |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Model spring-mass systems with second-order ODEs
2. Analyze free, damped, and forced oscillations
3. Understand resonance and its consequences
4. Model RLC circuits with differential equations
5. See the analogy between mechanical and electrical systems

---

## 📚 Required Reading

### Primary Text: Boyce & DiPrima (11th Edition)
- **Section 3.7**: Mechanical and Electrical Vibrations (pp. 203-219)

---

## 📖 Part I: Mechanical Vibrations

### 1. The Spring-Mass System

A mass m attached to a spring with constant k, subject to damping c:

**Newton's Second Law:**
$$m\frac{d^2x}{dt^2} = -kx - c\frac{dx}{dt} + F(t)$$

**Standard form:**
$$m x'' + c x' + k x = F(t)$$

or dividing by m:
$$x'' + 2\gamma x' + \omega_0^2 x = \frac{F(t)}{m}$$

where:
- $\omega_0 = \sqrt{k/m}$ = natural frequency
- $\gamma = c/(2m)$ = damping coefficient

---

## 📖 Case 1: Undamped Free Oscillation (SHM)

### 2. Simple Harmonic Motion

When $c = 0$ and $F = 0$:
$$x'' + \omega_0^2 x = 0$$

**Characteristic equation:** $r^2 + \omega_0^2 = 0 \Rightarrow r = \pm i\omega_0$

**Solution:**
$$x(t) = A\cos(\omega_0 t) + B\sin(\omega_0 t)$$

Or in amplitude-phase form:
$$x(t) = R\cos(\omega_0 t - \phi)$$

where $R = \sqrt{A^2 + B^2}$ and $\tan\phi = B/A$

### Key Features:
- **Period:** $T = 2\pi/\omega_0 = 2\pi\sqrt{m/k}$
- **Frequency:** $f = 1/T = \frac{1}{2\pi}\sqrt{k/m}$
- Oscillation continues forever (no energy loss)

### Example 1
A 2 kg mass on a spring with k = 8 N/m is displaced 0.5 m and released. Find x(t).

$$\omega_0 = \sqrt{8/2} = 2 \text{ rad/s}$$
$$x(t) = A\cos 2t + B\sin 2t$$

ICs: $x(0) = 0.5$, $x'(0) = 0$
- $A = 0.5$
- $x'(0) = 2B = 0 \Rightarrow B = 0$

**Solution:** $x(t) = 0.5\cos 2t$ meters

---

## 📖 Case 2: Damped Free Oscillation

### 3. Adding Damping

$$x'' + 2\gamma x' + \omega_0^2 x = 0$$

**Characteristic equation:**
$$r^2 + 2\gamma r + \omega_0^2 = 0$$
$$r = -\gamma \pm \sqrt{\gamma^2 - \omega_0^2}$$

### 4. Three Damping Regimes

**Underdamped** ($\gamma < \omega_0$): Oscillations that decay
$$x(t) = e^{-\gamma t}(A\cos\omega_d t + B\sin\omega_d t)$$
where $\omega_d = \sqrt{\omega_0^2 - \gamma^2}$ (damped frequency)

**Critically damped** ($\gamma = \omega_0$): Fastest return without oscillation
$$x(t) = (A + Bt)e^{-\gamma t}$$

**Overdamped** ($\gamma > \omega_0$): Slow exponential return
$$x(t) = c_1 e^{r_1 t} + c_2 e^{r_2 t}$$
where $r_1, r_2 < 0$

### Example 2: Underdamped
Solve $x'' + 2x' + 5x = 0$, $x(0) = 1$, $x'(0) = 0$.

Here $\gamma = 1$, $\omega_0^2 = 5$, so $\gamma < \omega_0$ (underdamped)

$$r = -1 \pm \sqrt{1-5} = -1 \pm 2i$$
$$\omega_d = 2$$

$$x(t) = e^{-t}(A\cos 2t + B\sin 2t)$$

ICs: $A = 1$, $B = 1/2$

**Solution:** $x(t) = e^{-t}(\cos 2t + \frac{1}{2}\sin 2t)$

---

## 📖 Case 3: Forced Oscillations

### 5. External Forcing

$$x'' + 2\gamma x' + \omega_0^2 x = F_0\cos(\omega t)$$

The particular solution has the form:
$$x_p(t) = A(\omega)\cos(\omega t - \delta)$$

**Amplitude:**
$$A(\omega) = \frac{F_0/m}{\sqrt{(\omega_0^2 - \omega^2)^2 + (2\gamma\omega)^2}}$$

**Phase lag:**
$$\tan\delta = \frac{2\gamma\omega}{\omega_0^2 - \omega^2}$$

### 6. Resonance

When $\gamma = 0$ (no damping) and $\omega = \omega_0$:
**The amplitude becomes infinite!**

This is **resonance**—the driving frequency matches the natural frequency.

With light damping, maximum amplitude occurs near $\omega = \omega_0$.

### Example 3: Resonance
Solve $x'' + 4x = 3\cos 2t$ (no damping, $\omega = \omega_0 = 2$).

Standard trial fails (overlap). Modified trial: $x_p = t(A\cos 2t + B\sin 2t)$

After substitution: $A = 0$, $B = 3/4$

**Solution:** $x = c_1\cos 2t + c_2\sin 2t + \frac{3t}{4}\sin 2t$

The amplitude grows linearly without bound!

---

## 📖 Part II: RLC Circuits

### 7. Circuit Equation

For a series RLC circuit with voltage source E(t):

**Kirchhoff's Voltage Law:**
$$L\frac{d^2Q}{dt^2} + R\frac{dQ}{dt} + \frac{Q}{C} = E(t)$$

Or in terms of current $I = dQ/dt$:
$$L\frac{dI}{dt} + RI + \frac{1}{C}\int I \, dt = E(t)$$

### 8. Analogy Table

| Mechanical | Electrical |
|------------|------------|
| Mass m | Inductance L |
| Damping c | Resistance R |
| Spring constant k | 1/Capacitance (1/C) |
| Position x | Charge Q |
| Velocity v | Current I |
| Applied force F | Voltage E |

The equations are **mathematically identical**!

### 9. Natural Frequency and Damping

$$\omega_0 = \frac{1}{\sqrt{LC}} \quad \text{(resonant frequency)}$$
$$\gamma = \frac{R}{2L} \quad \text{(damping factor)}$$

### Example 4: RLC Circuit
An RLC circuit has L = 1 H, R = 4 Ω, C = 0.05 F, E = 0. Find Q(t) if Q(0) = 1 C and I(0) = 0.

$$Q'' + 4Q' + 20Q = 0$$

Characteristic: $r = -2 \pm 4i$

$$Q(t) = e^{-2t}(A\cos 4t + B\sin 4t)$$

ICs: $A = 1$, $Q'(0) = -2A + 4B = 0 \Rightarrow B = 1/2$

**Solution:** $Q(t) = e^{-2t}(\cos 4t + \frac{1}{2}\sin 4t)$ coulombs

---

## 📋 Summary: Oscillation Types

| Type | Condition | Behavior |
|------|-----------|----------|
| Simple harmonic | $\gamma = 0$, $F = 0$ | Eternal oscillation |
| Underdamped | $\gamma < \omega_0$ | Decaying oscillation |
| Critically damped | $\gamma = \omega_0$ | Fast decay, no oscillation |
| Overdamped | $\gamma > \omega_0$ | Slow exponential decay |
| Resonance | $\omega = \omega_0$, $\gamma = 0$ | Unbounded amplitude |
| Forced (damped) | $\gamma > 0$, $F \neq 0$ | Steady state + transient |

---

## 📝 Practice Problems

### Level 1: Simple Harmonic Motion
1. A 0.5 kg mass on a spring (k = 2 N/m) is released from x = 0.1 m. Find x(t) and the period.
2. A pendulum has period 2 s. Find the length (use $T = 2\pi\sqrt{L/g}$, g = 9.8 m/s²).
3. Find the frequency of an LC circuit with L = 0.1 H and C = 10 μF.

### Level 2: Damped Oscillations
4. Classify: $x'' + 4x' + 3x = 0$ (under/over/critical?)
5. Classify: $x'' + 6x' + 9x = 0$
6. Solve $x'' + 4x' + 5x = 0$, $x(0) = 2$, $x'(0) = 0$

### Level 3: Forced Oscillations
7. Solve $x'' + 4x = 5\cos t$ (find steady state)
8. Solve $x'' + x = \cos t$ (resonance case!)
9. Find the amplitude of steady-state oscillation for $x'' + 0.2x' + 4x = 3\cos 2t$

### Level 4: RLC Circuits
10. An RLC circuit has L = 2 H, R = 0, C = 0.5 F. Find the resonant frequency.
11. Solve $Q'' + 2Q' + 5Q = 10\cos t$, $Q(0) = 0$, $Q'(0) = 0$
12. Design an RLC circuit (choose L, R, C) that is critically damped with $\omega_0 = 10$ rad/s

### Level 5: Advanced
13. Show that for the forced damped oscillator, maximum amplitude occurs at $\omega = \sqrt{\omega_0^2 - 2\gamma^2}$
14. A building sways with period 2 s. An earthquake oscillates at 0.5 Hz. Is there danger of resonance?
15. Derive the energy dissipated per cycle in a damped harmonic oscillator.

---

## 📊 Answers

1. $x = 0.1\cos 2t$, T = π s
2. L ≈ 0.99 m
3. f = 159 Hz
4. Overdamped (r = -1, -3)
5. Critically damped (r = -3 repeated)
6. $x = e^{-2t}(2\cos t + 4\sin t)$
7. $x_p = \frac{5}{3}\cos t$
8. $x_p = \frac{t}{2}\sin t$
9. A = 3/0.4 = 7.5
10. $\omega_0 = 1$ rad/s
11. $Q = e^{-t}(-2\cos 2t + \frac{1}{2}\sin 2t) + 2\cos t + \sin t$
12. L = 0.01 H, C = 0.01 F, R = 2 Ω (example)
13. Differentiate $A(\omega)$ and set to zero
14. Yes! Building's natural freq ≈ 0.5 Hz matches earthquake
15. $\Delta E = \pi c \omega A^2$

---

## 🔬 Quantum Mechanics Connection

### Quantum Harmonic Oscillator

The quantum harmonic oscillator Hamiltonian:
$$\hat{H} = \frac{\hat{p}^2}{2m} + \frac{1}{2}m\omega^2\hat{x}^2$$

gives the Schrödinger equation:
$$-\frac{\hbar^2}{2m}\psi'' + \frac{1}{2}m\omega^2 x^2 \psi = E\psi$$

**Energy levels:**
$$E_n = \hbar\omega\left(n + \frac{1}{2}\right), \quad n = 0, 1, 2, \ldots$$

**Zero-point energy:** Even at n = 0, $E_0 = \frac{1}{2}\hbar\omega \neq 0$!

### Driven Quantum Systems

When a quantum oscillator is driven at its resonant frequency, transitions between energy levels occur—this is the basis of spectroscopy!

---

## ✅ Daily Checklist

- [ ] Read Boyce & DiPrima Section 3.7
- [ ] Model spring-mass systems
- [ ] Understand the three damping regimes
- [ ] Analyze forced oscillations and resonance
- [ ] Draw the mechanical-electrical analogy
- [ ] Complete practice problems

---

## 🔜 Preview: Tomorrow

**Day 68: Week 10 Problem Set**
- Comprehensive review of second-order ODEs

---

*"Every swing of a pendulum, every note of a guitar, every beat of your heart follows the mathematics of oscillation."*
