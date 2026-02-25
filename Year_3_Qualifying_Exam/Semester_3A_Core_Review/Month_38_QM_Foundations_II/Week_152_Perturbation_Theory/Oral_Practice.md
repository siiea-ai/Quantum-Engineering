# Week 152: Perturbation Theory - Oral Exam Practice

## Introduction

Perturbation theory questions are guaranteed on any PhD qualifying exam. You must be able to derive formulas, apply them correctly, and explain when each method is appropriate.

---

## Question 1: Non-Degenerate Perturbation Theory

### Initial Question
"Derive the first-order energy correction in time-independent perturbation theory."

### Suggested Response

**Setup:**
"We write $H = H_0 + \lambda H'$ where $H_0$ has known eigenstates $|n^{(0)}\rangle$ with energies $E_n^{(0)}$. We expand the perturbed energy and state in powers of $\lambda$."

**Derivation:**
"The Schrödinger equation gives:
$$(H_0 + \lambda H')(|n^{(0)}\rangle + \lambda|n^{(1)}\rangle + \cdots) = (E_n^{(0)} + \lambda E_n^{(1)} + \cdots)(|n^{(0)}\rangle + \lambda|n^{(1)}\rangle + \cdots)$$

At order $\lambda^1$:
$$H_0|n^{(1)}\rangle + H'|n^{(0)}\rangle = E_n^{(0)}|n^{(1)}\rangle + E_n^{(1)}|n^{(0)}\rangle$$

Taking the inner product with $\langle n^{(0)}|$:
$$\langle n^{(0)}|H_0|n^{(1)}\rangle + \langle n^{(0)}|H'|n^{(0)}\rangle = E_n^{(0)}\langle n^{(0)}|n^{(1)}\rangle + E_n^{(1)}$$

The first and third terms cancel, giving:
$$E_n^{(1)} = \langle n^{(0)}|H'|n^{(0)}\rangle$$"

### Follow-up Questions

**Q: "When does this break down?"**

A: "When there's degeneracy. If $E_n^{(0)} = E_m^{(0)}$ for $n \neq m$, the first-order state correction has a zero denominator. We must use degenerate perturbation theory."

**Q: "What about second order?"**

A: "Second-order energy:
$$E_n^{(2)} = \sum_{k\neq n}\frac{|\langle k^{(0)}|H'|n^{(0)}\rangle|^2}{E_n^{(0)} - E_k^{(0)}}$$
For the ground state, this is always negative since all denominators are negative."

---

## Question 2: Degenerate Perturbation Theory

### Initial Question
"Explain how to handle perturbation theory when there's degeneracy."

### Suggested Response

**The Problem:**
"If states $|n^{(0)}\rangle$ and $|m^{(0)}\rangle$ have the same energy, non-degenerate theory fails because the state correction contains $1/(E_n - E_m) = 1/0$."

**The Solution:**
"We must find the 'good' linear combinations that diagonalize $H'$ in the degenerate subspace before applying perturbation theory.

1. Identify all degenerate states spanning a subspace of dimension $g$
2. Construct the $g \times g$ matrix $W_{ij} = \langle i|H'|j\rangle$
3. Diagonalize $W$: eigenvalues are $E^{(1)}$, eigenvectors are the good states
4. The good states have definite first-order energy corrections"

**Example:**
"For the linear Stark effect in hydrogen $n=2$, we diagonalize the $4 \times 4$ matrix of $H' = eE_0z$. Only $|2,0,0\rangle$ and $|2,1,0\rangle$ mix, giving eigenvalues $\pm 3eE_0a_0$."

### Follow-up Questions

**Q: "Why is there no first-order Stark effect for the ground state?"**

A: "The ground state is non-degenerate and has even parity. Since $z$ is odd, $\langle 1s|z|1s\rangle = 0$ by parity."

**Q: "What determines the 'good' quantum numbers?"**

A: "The good quantum numbers label eigenstates of operators that commute with both $H_0$ and $H'$. For the Stark effect, these are $n$, $m$, and the combination $l \pm 1$ that diagonalizes $H'$."

---

## Question 3: Time-Dependent Perturbation

### Initial Question
"Derive the formula for transition probability in time-dependent perturbation theory."

### Suggested Response

**Setup:**
"We have $H = H_0 + H'(t)$ with $H'$ turned on at $t=0$. The system starts in eigenstate $|i\rangle$ of $H_0$."

**Interaction Picture:**
"We expand $|\psi(t)\rangle = \sum_n c_n(t)e^{-iE_nt/\hbar}|n\rangle$. Substituting into Schrödinger's equation:

$$i\hbar\dot{c}_f = \sum_n c_n\langle f|H'|n\rangle e^{i\omega_{fn}t}$$"

**First Order:**
"With $c_i(0) = 1$, $c_{n\neq i}(0) = 0$:

$$c_f^{(1)}(t) = -\frac{i}{\hbar}\int_0^t \langle f|H'(t')|i\rangle e^{i\omega_{fi}t'}dt'$$

Transition probability: $P_{i\to f}(t) = |c_f(t)|^2$"

### Follow-up Questions

**Q: "What happens at resonance for a sinusoidal perturbation?"**

A: "For $H' = V\cos\omega t$, near resonance ($\omega \approx \omega_{fi}$), the transition probability shows a sharp peak proportional to $t^2$ at exact resonance. For off-resonance, it oscillates and averages to a constant."

**Q: "When is this approximation valid?"**

A: "When $|c_f| \ll 1$, i.e., when transitions are rare. Also, $H'$ must be small enough that higher-order terms are negligible."

---

## Question 4: Fermi's Golden Rule

### Initial Question
"State and derive Fermi's golden rule."

### Suggested Response

**Statement:**
"For transitions to a continuum of final states, the transition rate is:
$$\Gamma = \frac{2\pi}{\hbar}|\langle f|H'|i\rangle|^2\rho(E_f)$$
where $\rho(E_f)$ is the density of final states at the energy determined by conservation."

**Key Assumptions:**
1. Long time limit (steady-state rate)
2. Continuous spectrum of final states
3. Matrix element approximately constant over relevant energy range

**Derivation Outline:**
"From time-dependent perturbation theory:
$$P_{i\to f}(t) \approx \frac{|V_{fi}|^2}{\hbar^2}\frac{\sin^2(\omega_{fi}t/2)}{(\omega_{fi}/2)^2}$$

Summing over final states with density $\rho$:
$$P = \int |V_{fi}|^2 \rho(E_f)\frac{\sin^2[(E_f-E_i)t/(2\hbar)]}{[(E_f-E_i)/(2\hbar)]^2}\frac{dE_f}{\hbar}$$

For long times, $\frac{\sin^2(xt/2)}{(x/2)^2} \to 2\pi t\delta(x)$

This gives $P = \frac{2\pi}{\hbar}|V_{fi}|^2\rho(E_i)t$, so $\Gamma = P/t$ is constant."

### Follow-up Questions

**Q: "Give an example application."**

A: "Spontaneous emission rate of atoms. The density of photon states is $\rho(\omega) \propto \omega^2$. Combined with the dipole matrix element, this gives the famous $\omega^3$ dependence of the emission rate."

**Q: "Why is it called 'golden'?"**

A: "Fermi himself called it 'Golden Rule #2' in his notes. It's golden because of its widespread applicability - it appears throughout atomic, nuclear, and condensed matter physics."

---

## Question 5: Adiabatic Theorem and Berry Phase

### Initial Question
"State the adiabatic theorem and explain Berry phase."

### Suggested Response

**Adiabatic Theorem:**
"If a Hamiltonian $H(t)$ varies slowly enough, a system starting in an instantaneous eigenstate $|n(0)\rangle$ remains in the instantaneous eigenstate $|n(t)\rangle$. The condition is that changes occur slowly compared to $\hbar/\Delta E$ where $\Delta E$ is the energy gap to other levels."

**Berry Phase:**
"During adiabatic evolution around a closed loop in parameter space, the state acquires a geometric phase in addition to the dynamical phase:

$$\gamma_n = i\oint\langle n(\mathbf{R})|\nabla_{\mathbf{R}}|n(\mathbf{R})\rangle \cdot d\mathbf{R}$$

This Berry phase depends only on the path geometry, not on how fast it's traversed."

**Example:**
"For a spin-1/2 in a magnetic field that traces a closed loop, the Berry phase equals minus half the solid angle enclosed: $\gamma = -\Omega/2$."

### Follow-up Questions

**Q: "Why is Berry phase observable?"**

A: "It's gauge-invariant and appears in interference experiments. For example, it explains the Aharonov-Bohm effect and appears in molecular dynamics at conical intersections."

**Q: "What happens if the gap closes?"**

A: "The adiabatic approximation breaks down. Transitions to other levels become likely. At a 'diabolic point' where two levels cross, non-adiabatic effects dominate."

---

## Quick-Fire Questions

1. **First-order energy formula?**
   - $E_n^{(1)} = \langle n^{(0)}|H'|n^{(0)}\rangle$

2. **Is second-order ground state energy positive or negative?**
   - Negative (ground state is always lowered)

3. **When do you use degenerate perturbation theory?**
   - When unperturbed states have equal energies

4. **What's the resonance condition in time-dependent PT?**
   - $\omega = \omega_{fi} = (E_f - E_i)/\hbar$

5. **What determines transition rate in Fermi's golden rule?**
   - Matrix element squared times density of states

6. **Berry phase for spin-1/2 around full sphere?**
   - $-2\pi$ (solid angle $4\pi$, phase $= -\Omega/2$)

---

## Whiteboard Problems

1. **Derive** $E_n^{(1)} = \langle H'\rangle$ from scratch

2. **Calculate** the Stark effect for hydrogen $n=2$

3. **Show** that second-order ground state energy is negative

4. **Derive** Fermi's golden rule starting from time-dependent PT

5. **Calculate** Berry phase for spin on a cone

---

**Preparation Time:** 3-4 hours
**Key Skill:** Derive formulas fluently on a whiteboard
