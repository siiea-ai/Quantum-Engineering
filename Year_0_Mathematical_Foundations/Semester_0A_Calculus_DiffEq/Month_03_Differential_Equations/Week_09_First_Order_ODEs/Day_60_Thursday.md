# Day 60: Applications and Modeling

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Growth and Decay Models |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Physical Applications |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Practice |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Model exponential growth and decay
2. Apply radioactive decay to carbon dating
3. Solve Newton's Law of Cooling problems
4. Model population dynamics with logistic growth
5. Find orthogonal trajectories

---

## 📚 Required Reading

### Primary Text: Boyce & DiPrima (11th Edition)
- **Section 2.3**: Modeling with First Order Equations (pp. 52-65)
- **Section 2.5**: Autonomous Equations and Population Dynamics (pp. 78-95)

---

## 📖 Exponential Growth and Decay

### 1. The Basic Model

Many quantities change at a rate proportional to their current value:
$$\frac{dP}{dt} = kP$$

**Solution:** $P(t) = P_0 e^{kt}$

| k > 0 | k < 0 |
|-------|-------|
| Exponential growth | Exponential decay |
| Population growth | Radioactive decay |
| Compound interest | Drug elimination |

### 2. Half-Life and Doubling Time

**Half-life** (for decay, k < 0):
$$t_{1/2} = \frac{\ln 2}{|k|}$$

**Doubling time** (for growth, k > 0):
$$t_d = \frac{\ln 2}{k}$$

---

## ✏️ Growth/Decay Examples

### Example 1: Population Growth
A bacteria population doubles every 3 hours. If P(0) = 1000, find P(t) and P(10).

**Find k:** Doubling time = 3, so:
$$k = \frac{\ln 2}{3}$$

**Model:** $P(t) = 1000 e^{t \ln 2/3} = 1000 \cdot 2^{t/3}$

**P(10):** $P(10) = 1000 \cdot 2^{10/3} \approx 10,079$ bacteria

---

### Example 2: Radioactive Decay
Carbon-14 has a half-life of 5730 years. A fossil has 20% of its original C-14. How old is it?

**Find k:**
$$k = -\frac{\ln 2}{5730}$$

**Model:** $N(t) = N_0 e^{-t \ln 2/5730}$

**Solve for t when N = 0.2N₀:**
$$0.2 = e^{-t \ln 2/5730}$$
$$\ln(0.2) = -\frac{t \ln 2}{5730}$$
$$t = -\frac{5730 \ln(0.2)}{\ln 2} = \frac{5730 \ln 5}{\ln 2} \approx 13,305 \text{ years}$$

---

### Example 3: Compound Interest
\$1000 is invested at 5% annual interest, compounded continuously. Find the balance after 10 years.

**Model:** $\frac{dA}{dt} = 0.05A$

**Solution:** $A(t) = 1000 e^{0.05t}$

**A(10):** $A(10) = 1000 e^{0.5} \approx \$1,648.72$

---

## 📖 Newton's Law of Cooling

### 3. The Model

The rate of change of temperature is proportional to the difference from ambient:
$$\frac{dT}{dt} = -k(T - T_s)$$

where $T_s$ is the surrounding (ambient) temperature.

**Solution:** $T(t) = T_s + (T_0 - T_s)e^{-kt}$

---

### Example 4: Cooling Coffee
Coffee at 95°C is placed in a room at 20°C. After 5 minutes, it's 80°C. When will it reach 50°C?

**Find k:**
$$80 = 20 + 75e^{-5k}$$
$$60 = 75e^{-5k}$$
$$e^{-5k} = 0.8$$
$$k = -\frac{\ln(0.8)}{5} \approx 0.0446$$

**Find t when T = 50:**
$$50 = 20 + 75e^{-kt}$$
$$30 = 75e^{-kt}$$
$$e^{-kt} = 0.4$$
$$t = -\frac{\ln(0.4)}{k} \approx 20.5 \text{ minutes}$$

---

### Example 5: Time of Death (Forensics)
A body is found at 10 PM with temperature 85°F. Room temperature is 70°F. At 11 PM, the body is 80°F. Normal body temperature is 98.6°F. Estimate the time of death.

**Find k:**
$$80 = 70 + (85-70)e^{-k}$$
$$10 = 15e^{-k}$$
$$k = \ln(1.5) \approx 0.405 \text{ per hour}$$

**Find t when T = 98.6:**
Going backward from 10 PM:
$$85 = 70 + (98.6-70)e^{-k \cdot t}$$
$$15 = 28.6 e^{-0.405t}$$
$$t = \frac{\ln(28.6/15)}{0.405} \approx 1.6 \text{ hours before 10 PM}$$

**Time of death:** Approximately 8:24 PM

---

## 📖 Logistic Growth

### 4. The Model

When growth is limited by resources:
$$\frac{dP}{dt} = rP\left(1 - \frac{P}{K}\right)$$

- r = intrinsic growth rate
- K = carrying capacity

**Solution:**
$$P(t) = \frac{K}{1 + \left(\frac{K - P_0}{P_0}\right)e^{-rt}}$$

### 5. Key Features

- P < K: population grows
- P > K: population decreases
- P → K as t → ∞ (equilibrium)
- Maximum growth rate at P = K/2

---

### Example 6: Logistic Population
A lake can support 10,000 fish. Initially there are 1,000 fish, and the population grows logistically with r = 0.5/year. Find P(t) and when P = 5,000.

**Parameters:** K = 10,000, P₀ = 1,000, r = 0.5

$$P(t) = \frac{10000}{1 + 9e^{-0.5t}}$$

**When P = 5000:**
$$5000 = \frac{10000}{1 + 9e^{-0.5t}}$$
$$1 + 9e^{-0.5t} = 2$$
$$e^{-0.5t} = 1/9$$
$$t = \frac{\ln 9}{0.5} \approx 4.4 \text{ years}$$

---

## 📖 Mixture Problems

### 6. General Setup

**Variables:**
- V(t) = volume in tank
- x(t) = amount of substance
- c(t) = x(t)/V(t) = concentration

**General ODE:**
$$\frac{dx}{dt} = (\text{rate in}) - (\text{rate out})$$
$$= (c_{in} \cdot r_{in}) - \left(\frac{x}{V}\right) \cdot r_{out}$$

### Example 7: Variable Volume Tank
A 100-gallon tank initially contains 50 gallons of brine with 10 lbs of salt. Brine with 1 lb/gal enters at 4 gal/min, and the mixture leaves at 2 gal/min. Find the amount of salt when the tank is full.

**Volume:** $V(t) = 50 + 2t$ (full when t = 25)

**ODE:**
$$\frac{dx}{dt} = 4 - \frac{2x}{50 + 2t} = 4 - \frac{x}{25 + t}$$

**Rearrange:** $x' + \frac{x}{25+t} = 4$

**Integrating factor:** $\mu = 25 + t$

$$(25+t)x' + x = 4(25+t)$$
$$\frac{d}{dt}[(25+t)x] = 100 + 4t$$
$$(25+t)x = 100t + 2t^2 + C$$

**IC:** $x(0) = 10 \Rightarrow 25(10) = C \Rightarrow C = 250$

$$x = \frac{100t + 2t^2 + 250}{25 + t}$$

**At t = 25:** $x(25) = \frac{2500 + 1250 + 250}{50} = 80$ lbs

---

## 📖 Orthogonal Trajectories

### 7. Definition

**Orthogonal trajectories** are curves that intersect a given family of curves at right angles.

### 8. Method

1. Given family F(x, y, C) = 0, find the DE satisfied by the family
2. Replace y' with -1/y' (perpendicular slopes)
3. Solve the new DE

### Example 8: Find Orthogonal Trajectories
Find the orthogonal trajectories of $y = Cx^2$.

**Step 1:** Eliminate C
$$C = y/x^2$$
Differentiate: $y' = 2Cx = 2y/x$

**Step 2:** Perpendicular slopes
$$y' = -\frac{x}{2y}$$

**Step 3:** Solve
$$2y \, dy = -x \, dx$$
$$y^2 = -\frac{x^2}{2} + K$$
$$x^2 + 2y^2 = C'$$

**Orthogonal family:** Ellipses $x^2 + 2y^2 = C$

---

## 📋 Summary of Models

| Model | Equation | Solution |
|-------|----------|----------|
| Exponential growth | $y' = ky$ | $y = y_0 e^{kt}$ |
| Newton's cooling | $T' = -k(T - T_s)$ | $T = T_s + (T_0 - T_s)e^{-kt}$ |
| Logistic | $P' = rP(1-P/K)$ | $P = K/(1 + Ae^{-rt})$ |
| Mixing | $x' = r_{in}c_{in} - r_{out}x/V$ | Depends on rates |

---

## 📝 Practice Problems

### Level 1: Exponential Models
1. A population grows from 500 to 800 in 10 years. Find P(t) and P(25).
2. A radioactive substance has a half-life of 20 days. How long until 10% remains?
3. \$5000 invested at 4% continuous interest. Find the balance after 15 years.

### Level 2: Newton's Cooling
4. An object at 150°F is placed in a room at 70°F. After 30 min, it's 100°F. When is it 80°F?
5. A pie at 350°F is placed in a room at 75°F. After 15 min, it's 150°F. Find T(t).
6. A thermometer reading 70°F is placed in a freezer at 0°F. After 3 min, it reads 40°F. When will it read 10°F?

### Level 3: Logistic Growth
7. A population follows $P' = 0.1P(1 - P/1000)$, $P(0) = 100$. Find P(t) and $\lim_{t\to\infty} P(t)$.
8. A rumor spreads logistically in a town of 5000. If 100 know initially and 500 know after 2 days, find the model.
9. Find the time at which the logistic population in Problem 7 is growing fastest.

### Level 4: Mixing Problems
10. A 50-gallon tank with 20 lbs of salt receives pure water at 3 gal/min. Find the salt after 10 min.
11. A 200-gallon tank initially has pure water. Brine (0.5 lb/gal) enters at 4 gal/min; mixture leaves at 4 gal/min. Find x(t).
12. Repeat Problem 11 if the outflow rate is 2 gal/min.

### Level 5: Orthogonal Trajectories
13. Find orthogonal trajectories of $y = Ce^x$
14. Find orthogonal trajectories of $x^2 + y^2 = C$
15. Find orthogonal trajectories of $xy = C$

---

## 📊 Answers

1. $P(t) = 500e^{t\ln(1.6)/10}$; P(25) ≈ 1280
2. $t = 20\ln(10)/\ln(2) \approx 66.4$ days
3. \$9,110.59
4. About 73.5 minutes
5. $T(t) = 75 + 275e^{-kt}$ where $k = \ln(11/3)/15$
6. About 8.5 minutes
7. $P = 1000/(1 + 9e^{-0.1t})$; limit = 1000
8. $P = 5000/(1 + 49e^{-rt})$ where $r = \ln(49/9)/2$
9. When P = 500 (half of carrying capacity)
10. 20e^{-0.6} ≈ 10.98 lbs
11. $x = 100(1 - e^{-t/50})$
12. $x = \frac{100t + t^2/2}{200 + 2t}$ (variable volume)
13. $y = -x + C$ (lines)
14. $y = Cx$ (lines through origin)
15. $y^2 - x^2 = C$ (hyperbolas)

---

## 🔬 Quantum Mechanics Connection

### Radioactive Decay and Quantum Tunneling

Radioactive decay is fundamentally quantum mechanical—particles tunnel through potential barriers with probability:
$$P \propto e^{-2\gamma L}$$

The decay constant λ depends on the tunneling probability!

### State Decay

Excited quantum states decay exponentially:
$$|c_n(t)|^2 = |c_n(0)|^2 e^{-\Gamma t}$$

This is Fermi's Golden Rule in action.

---

## ✅ Daily Checklist

- [ ] Read Boyce & DiPrima Sections 2.3, 2.5
- [ ] Master exponential growth/decay
- [ ] Apply Newton's Law of Cooling
- [ ] Solve logistic growth problems
- [ ] Work through mixing problems
- [ ] Find orthogonal trajectories
- [ ] Complete practice problems

---

## 🔜 Preview: Tomorrow

**Day 61: Week 9 Problem Set**
- Comprehensive assessment of first-order ODEs

---

*"Differential equations are the bridge between mathematics and the physical world—they describe how things change."*
