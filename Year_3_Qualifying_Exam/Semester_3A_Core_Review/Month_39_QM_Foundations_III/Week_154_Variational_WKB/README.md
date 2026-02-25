# Week 154: Variational and WKB Methods

## Overview

**Days:** 1072-1078
**Theme:** Approximation methods for quantum systems without exact solutions
**Prerequisites:** Perturbation theory, classical mechanics, wave mechanics

---

## Learning Objectives

By the end of this week, you will be able to:

1. **Apply** the variational principle to estimate ground state energies
2. **Construct** effective trial wavefunctions with appropriate symmetries
3. **Optimize** variational parameters using calculus
4. **Derive** the WKB approximation from the Schrodinger equation
5. **Use** connection formulas at classical turning points
6. **Calculate** tunneling rates through potential barriers
7. **Apply** the Born-Oppenheimer approximation to molecular systems

---

## Daily Schedule

| Day | Focus | Key Concepts |
|-----|-------|--------------|
| **1072** | Variational Principle | Upper bound theorem, Rayleigh quotient |
| **1073** | Applications I | Hydrogen, helium ground states |
| **1074** | Applications II | Molecular ions, Born-Oppenheimer |
| **1075** | WKB Approximation | Semiclassical limit, validity conditions |
| **1076** | Connection Formulas | Turning points, Airy functions |
| **1077** | Tunneling & Decay | Barrier penetration, alpha decay |
| **1078** | Review & Oral Practice | Problem synthesis, oral Q&A |

---

## Core Concepts

### 1. The Variational Principle

**Statement:** For any normalized trial wavefunction $|\tilde{\psi}\rangle$:

$$\boxed{E_0 \leq \langle\tilde{\psi}|H|\tilde{\psi}\rangle \equiv E[\tilde{\psi}]}$$

The expectation value of the Hamiltonian in any state is an upper bound on the ground state energy.

**Proof:**
Expand $|\tilde{\psi}\rangle$ in energy eigenstates: $|\tilde{\psi}\rangle = \sum_n c_n|n\rangle$

$$E[\tilde{\psi}] = \sum_n |c_n|^2 E_n \geq E_0 \sum_n |c_n|^2 = E_0$$

since $E_n \geq E_0$ for all $n$.

### 2. Variational Method Procedure

1. Choose a trial wavefunction $\psi(\mathbf{r}; \alpha_1, \alpha_2, \ldots)$ with adjustable parameters
2. Ensure proper normalization and boundary conditions
3. Calculate $E(\alpha_1, \alpha_2, \ldots) = \langle\psi|H|\psi\rangle$
4. Minimize: $\frac{\partial E}{\partial \alpha_i} = 0$ for all $i$
5. Solve for optimal parameters
6. Result is upper bound on true $E_0$

### 3. The WKB Approximation

The WKB (Wentzel-Kramers-Brillouin) approximation is a semiclassical method valid when:
$$\left|\frac{d\lambda}{dx}\right| \ll 1$$

where $\lambda = h/p$ is the de Broglie wavelength.

**In classically allowed regions** ($E > V(x)$):
$$\psi(x) \approx \frac{C}{\sqrt{p(x)}}\exp\left(\pm\frac{i}{\hbar}\int^x p(x')dx'\right)$$

**In classically forbidden regions** ($E < V(x)$):
$$\psi(x) \approx \frac{C}{\sqrt{\kappa(x)}}\exp\left(\pm\frac{1}{\hbar}\int^x \kappa(x')dx'\right)$$

where:
- $p(x) = \sqrt{2m(E-V(x))}$ (classical momentum)
- $\kappa(x) = \sqrt{2m(V(x)-E)}$ (decay constant)

### 4. Connection Formulas

At classical turning points ($E = V(x_0)$), WKB breaks down. The connection formulas bridge across:

**From allowed to forbidden (moving right):**
$$\frac{1}{\sqrt{p}}\cos\left(\int_{x_0}^x p\,dx'/\hbar - \pi/4\right) \leftrightarrow \frac{1}{2\sqrt{\kappa}}e^{-\int_x^{x_0}\kappa\,dx'/\hbar}$$

### 5. Born-Oppenheimer Approximation

For molecules, nuclei are much heavier than electrons:
$$\frac{m_e}{M_{\text{nucleus}}} \ll 1$$

**Procedure:**
1. Fix nuclear positions $\mathbf{R}$
2. Solve electronic Schrodinger equation → $E_{\text{elec}}(\mathbf{R})$
3. Use $E_{\text{elec}}(\mathbf{R})$ as effective potential for nuclear motion
4. Solve nuclear Schrodinger equation

---

## Key Equations

### Variational Method

| Equation | Description |
|----------|-------------|
| $$E[\psi] = \frac{\langle\psi|H|\psi\rangle}{\langle\psi|\psi\rangle}$$ | Rayleigh quotient |
| $$\frac{\partial E}{\partial \alpha} = 0$$ | Optimization condition |
| $$E_0^{\text{var}} \geq E_0^{\text{exact}}$$ | Upper bound property |

### WKB Approximation

| Equation | Description |
|----------|-------------|
| $$\psi \sim \frac{1}{\sqrt{p}}\exp\left(\pm\frac{i}{\hbar}\int p\,dx\right)$$ | WKB wavefunction |
| $$\oint p\,dx = \left(n + \frac{1}{2}\right)h$$ | Bohr-Sommerfeld quantization |
| $$T \approx e^{-2\int_{x_1}^{x_2}\kappa\,dx/\hbar}$$ | Tunneling probability |

### Tunneling

| Equation | Description |
|----------|-------------|
| $$\Gamma = \nu \cdot T$$ | Decay rate ($\nu$ = attempt frequency) |
| $$t_{1/2} = \frac{\ln 2}{\Gamma}$$ | Half-life |

---

## Study Resources

### Primary Texts
- Shankar, Chapter 16 (Variational Methods)
- Griffiths, Chapter 8 (WKB Approximation)
- Sakurai, Chapter 5 (Approximation Methods)

### Supplementary
- Landau & Lifshitz, *Quantum Mechanics* (WKB in detail)
- [MIT 8.06 Lecture Notes on WKB](https://ocw.mit.edu/courses/physics/)

---

## Qualifying Exam Relevance

### Typical Problem Types
1. Estimate ground state energy using simple trial function
2. Optimize multi-parameter trial wavefunction
3. Apply WKB to bound states (quantization condition)
4. Calculate tunneling rate through given barrier
5. Estimate alpha decay lifetimes

### Common Pitfalls
- Forgetting normalization in variational calculation
- Wrong boundary conditions at turning points
- Sign errors in WKB phase integrals
- Applying WKB where it's not valid

---

## Week Checklist

- [ ] Derive variational principle from completeness
- [ ] Apply to hydrogen and helium
- [ ] Derive WKB approximation
- [ ] Master connection formulas
- [ ] Calculate tunneling through square barrier
- [ ] Apply to alpha decay
- [ ] Complete all problem sets
- [ ] Practice oral explanations

---

**Created:** February 2026
**Status:** NOT STARTED
