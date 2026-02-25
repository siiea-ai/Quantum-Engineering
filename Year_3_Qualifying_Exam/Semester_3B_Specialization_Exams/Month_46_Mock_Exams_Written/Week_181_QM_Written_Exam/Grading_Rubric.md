# Quantum Mechanics Written Exam - Grading Rubric

## Overview

**Total Points:** 200
**Number of Problems:** 8
**Points per Problem:** 25 each
**Passing Score:** 160 points (80%)

---

## General Grading Principles

### Partial Credit Philosophy

1. **Process over answer**: A correct approach with computational errors earns substantial partial credit
2. **Show your work**: Credit cannot be given for steps not shown
3. **Physical reasoning**: Correct physical intuition is rewarded even if execution fails
4. **Units and limits**: Checking dimensions and limiting cases demonstrates understanding

### Common Deductions

| Error Type | Typical Deduction |
|------------|-------------------|
| Sign error (not affecting method) | 1-2 points |
| Numerical factor error | 1-3 points |
| Missing units in final answer | 1 point |
| Incomplete justification | 2-3 points |
| Conceptual error | 5-10 points |
| Wrong approach entirely | Most points for that part |

---

## Problem 1: Operator Algebra and Uncertainty (25 points)

### Part (a): Proving Uncertainty Relation (8 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Define shifted operators or equivalent setup | 2 | $\hat{A}' = \hat{A} - \langle A \rangle$ |
| Construct auxiliary state | 2 | $|\phi\rangle = (\hat{A}' + i\lambda\hat{B}')|\psi\rangle$ |
| Use non-negativity of norm | 2 | $\langle\phi|\phi\rangle \geq 0$ |
| Complete derivation to final result | 2 | Correct final inequality |

**Common errors:**
- Forgetting the factor of 1/2 (-1 point)
- Not taking absolute value of $\langle C \rangle$ (-1 point)
- Incorrect expansion of quadratic (-2 points)

### Part (b): Explicit Calculation (7 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Correct $\langle x \rangle$ calculation | 2 | Uses ladder operators correctly |
| Correct $\langle x^2 \rangle$ calculation | 2 | Includes all contributing terms |
| Correct $\Delta x$ | 1 | Proper variance formula |
| Correct $\Delta p$ | 1 | Can use symmetry argument |
| Verify inequality satisfied | 1 | Shows $\Delta x \Delta p \geq \hbar/2$ |

**Common errors:**
- Missing cross terms in $\langle x^2 \rangle$ (-2 points)
- Incorrect action of ladder operators (-1 point each)

### Part (c): Minimum Uncertainty States (10 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Calculate $[\hat{A}, \hat{B}]$ correctly | 4 | Get $2i$ |
| Identify $\hat{C} = 2$ | 2 | Recognize constant |
| State uncertainty bound $\Delta A \Delta B \geq 1$ | 2 | Correct application |
| Identify coherent states | 2 | Name and/or define them |

**Full credit alternative:** Any equivalent description of minimum uncertainty states.

---

## Problem 2: Finite Square Well (25 points)

### Part (a): Setting Up Equations (8 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Correct TISE in each region | 3 | Three correct equations |
| Define $k$ and $\kappa$ correctly | 2 | Correct expressions |
| Correct boundary conditions at $\pm\infty$ | 2 | Exponential decay |
| Write general solutions | 1 | Correct form in each region |

### Part (b): Transcendental Equation (10 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Apply continuity of $\psi$ at $x = a$ | 3 | Correct equation |
| Apply continuity of $\psi'$ at $x = a$ | 3 | Correct equation |
| Correctly eliminate coefficients | 2 | Divide equations |
| Obtain $\kappa = k\tan(ka)$ | 2 | Final correct form |

**Common errors:**
- Using $x = 0$ instead of $x = a$ (-3 points)
- Sign errors in derivative (-2 points)

### Part (c): Limits (7 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Explain $V_0 \to \infty$ limit correctly | 3 | $\tan(ka) \to 0$ argument |
| Recover infinite well energies | 2 | $E_n \propto n^2$ |
| Numerical/graphical estimate for finite $V_0$ | 2 | Reasonable approximation |

---

## Problem 3: Spin-1/2 Dynamics (25 points)

### Part (a): Hamiltonian Matrix (8 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Express $\vec{S}$ in terms of $\sigma$ matrices | 2 | $\vec{S} = \frac{\hbar}{2}\vec{\sigma}$ |
| Correct matrix for $\sigma_z$ component | 2 | With $\cos(\omega t)$ |
| Correct matrix for $\sigma_x$ component | 2 | With $\sin(\omega t)$ |
| Correct overall factor | 2 | $-\gamma\hbar B_0/2$ |

### Part (b): Rotating Frame (10 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Write $\hat{U}(t)$ as explicit matrix | 2 | Correct exponential |
| Calculate $\hat{U}^\dagger\hat{H}\hat{U}$ | 3 | Correct transformation |
| Calculate $-i\hbar\hat{U}^\dagger\partial_t\hat{U}$ | 3 | Get $\frac{\hbar\omega}{2}\sigma_z$ |
| Correct final $\hat{H}_{eff}$ | 2 | Combine terms correctly |

**Common errors:**
- Missing the time-derivative term (-3 points)
- Sign error in rotating wave approximation (-2 points)

### Part (c): Resonance Probability (7 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Recognize resonance condition simplification | 2 | $\omega = \omega_0$ |
| Solve for time evolution | 3 | Get $\sin^2(\omega_0 t/2)$ |
| Physical interpretation | 2 | Rabi oscillations, spin flip |

---

## Problem 4: Angular Momentum Addition (25 points)

### Part (a): $|J, M\rangle$ Basis (8 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| List possible $J$ values | 2 | $J = 0, 1, 2$ |
| Identify $M = 0$ | 2 | From component states |
| Recognize antisymmetry implies $J = 1$ | 2 | Symmetry argument |
| Express in $|J, M\rangle$ basis | 2 | $|1, 0\rangle$ |

### Part (b): Expectation Values (8 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| $\langle S_{1z} \rangle = 0$ | 3 | Correct calculation |
| $\langle S_{2z} \rangle = 0$ | 2 | By symmetry or calculation |
| $\langle S_{1z}S_{2z} \rangle = -\hbar^2$ | 3 | Correct calculation |

### Part (c): $\hat{J}^2$ Measurement (9 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Identify state is $J^2$ eigenstate | 3 | Recognize it's pure $|J=1\rangle$ |
| Give outcome $J^2 = 2\hbar^2$ | 3 | Correct eigenvalue |
| State probability = 1 | 1 | Deterministic |
| State after measurement | 2 | Unchanged |

---

## Problem 5: Time-Independent Perturbation Theory (25 points)

### Part (a): First-Order Correction (10 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Write perturbation expectation value | 2 | $E^{(1)} = \langle V \rangle$ |
| Set up $\langle r^2 \rangle$ integral | 3 | Correct integrand |
| Evaluate integral correctly | 3 | Get $3a_0^2$ |
| Final answer | 2 | $E^{(1)} = 3\lambda e^2/a_0$ |

**Common errors:**
- Factor errors in Gaussian integral (-2 points)
- Missing $4\pi r^2$ in volume element (-3 points)

### Part (b): Degeneracy Analysis (8 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Identify $n = 2$ degeneracy | 2 | 4-fold |
| Explain scalar perturbation selection rules | 3 | $\Delta\ell = 0$, $\Delta m = 0$ |
| Conclude $\ell = 0, 1$ degeneracy lifted | 2 | Different $\langle r^2 \rangle$ |
| Note $m$ states remain degenerate | 1 | Same $\langle r^2 \rangle$ for different $m$ |

### Part (c): Second-Order Correction (7 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Set up second-order formula | 2 | Correct sum or closure |
| Calculate $\langle r^4 \rangle$ | 2 | Correct integral |
| Apply closure approximation | 2 | Reasonable $\bar{E}$ |
| Final estimate | 1 | Correct sign (negative) |

---

## Problem 6: Time-Dependent Perturbation Theory (25 points)

### Part (a): Perturbation and Selection Rules (8 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Write $\hat{V} = -eE_0 z \Theta(t)$ | 3 | Correct form |
| State selection rules | 3 | $\Delta\ell = \pm 1$, $\Delta m = 0$ |
| Identify $|2,1,0\rangle$ as allowed | 2 | Correct application |

### Part (b): Transition Probability (10 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Write first-order amplitude formula | 2 | Correct integral |
| Evaluate time integral | 3 | Get $(e^{i\omega t} - 1)/i\omega$ |
| Use given matrix element | 2 | $0.745 a_0$ |
| Final probability expression | 3 | $\sin^2(\omega_{21}t/2)$ form |

### Part (c): Long-Time Behavior (7 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Note oscillatory behavior | 2 | Does not grow unbounded |
| Explain bounded oscillation | 2 | Discrete final state |
| When perturbation theory fails | 3 | Strong field, back-transitions |

---

## Problem 7: Identical Particles (25 points)

### Part (a): Ground State (8 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Both fermions in $n = 0$ | 2 | Lowest energy |
| Symmetric spatial wavefunction | 2 | $\psi_0(x_1)\psi_0(x_2)$ |
| Antisymmetric spin (singlet) | 2 | Correct form |
| Ground state energy $\hbar\omega$ | 2 | Sum of zero-point energies |

### Part (b): Contact Interaction (9 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Set up perturbation integral | 2 | With delta function |
| Evaluate using delta function | 3 | $\int |\psi_0(x)|^4 dx$ |
| Compute Gaussian integral | 2 | Correct result |
| Final energy shift | 2 | Positive, correct form |

### Part (c): First Excited State (8 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Identify spatial configurations | 2 | Symmetric and antisymmetric |
| Note energy $2\hbar\omega$ | 1 | One particle excited |
| Identify 4-fold degeneracy | 2 | 1 singlet + 3 triplet |
| Triplet unaffected by contact | 3 | Antisymmetric spatial vanishes at $x_1 = x_2$ |

---

## Problem 8: Scattering Theory (25 points)

### Part (a): s-Wave Dominance (8 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Explain centrifugal barrier | 3 | $\ell(\ell+1)/r^2$ suppresses higher $\ell$ |
| Write radial equation inside | 2 | With $\kappa$ |
| Write radial equation outside | 2 | With $k$ |
| Use $u = rR$ substitution | 1 | Correct form |

### Part (b): Phase Shift (10 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Write solutions in each region | 2 | $\sinh$, $\sin$ forms |
| Apply continuity at $r = a$ | 2 | Two equations |
| Logarithmic derivative matching | 3 | Correct procedure |
| Final phase shift expression | 3 | Correct formula |

### Part (c): Hard Sphere Limit (7 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Take $V_0 \to \infty$ limit correctly | 2 | $\delta_0 \to -ka$ |
| Calculate $\sigma = 4\pi a^2$ | 3 | Use $\sin^2\delta_0 \approx (ka)^2$ |
| Explain factor of 4 | 2 | Diffraction/shadow scattering |

---

## Score Calculation

### Individual Problem Scores

| Problem | Topic | Points Earned | Max Points |
|---------|-------|---------------|------------|
| 1 | Uncertainty | ___ | 25 |
| 2 | Finite Well | ___ | 25 |
| 3 | Spin Dynamics | ___ | 25 |
| 4 | Angular Momentum | ___ | 25 |
| 5 | TI Perturbation | ___ | 25 |
| 6 | TD Perturbation | ___ | 25 |
| 7 | Identical Particles | ___ | 25 |
| 8 | Scattering | ___ | 25 |
| **Total** | | ___ | **200** |

### Grade Determination

| Score Range | Percentage | Grade |
|-------------|------------|-------|
| 180-200 | 90-100% | Pass with Distinction |
| 160-179 | 80-89% | Pass |
| 140-159 | 70-79% | Conditional Pass |
| 120-139 | 60-69% | Marginal Fail |
| <120 | <60% | Fail |

---

## Performance Analysis

### Topic Mastery Assessment

After grading, categorize your performance:

| Topic | Strong (>80%) | Adequate (60-80%) | Needs Work (<60%) |
|-------|---------------|-------------------|-------------------|
| Operator algebra | | | |
| Bound states | | | |
| Spin dynamics | | | |
| Angular momentum | | | |
| Perturbation theory | | | |
| Identical particles | | | |
| Scattering | | | |

### Error Pattern Analysis

Count your errors in each category:

| Error Type | Count | Priority |
|------------|-------|----------|
| Conceptual misunderstanding | ___ | High |
| Setup/approach error | ___ | High |
| Algebraic/arithmetic error | ___ | Medium |
| Missing steps/justification | ___ | Medium |
| Careless mistakes | ___ | Low |

---

## Remediation Recommendations

Based on your score:

### 180-200 points (Pass with Distinction)
- Focus on speed and efficiency
- Review any minor conceptual gaps
- Begin oral exam preparation

### 160-179 points (Pass)
- Review topics where points were lost
- Practice 3-5 more problems in weak areas
- Strengthen physical intuition

### 140-159 points (Conditional Pass)
- Intensive review of weak topics
- Work through textbook sections again
- Complete 10+ additional problems per weak area
- Consider retaking exam in 1-2 weeks

### Below 140 points (Fail)
- Comprehensive review needed
- Return to primary texts for weak areas
- Work with study group or tutor
- Complete full problem sets before retaking
- Schedule retake in 2-3 weeks minimum

---

*This rubric is modeled after common grading practices for physics qualifying examinations.*
