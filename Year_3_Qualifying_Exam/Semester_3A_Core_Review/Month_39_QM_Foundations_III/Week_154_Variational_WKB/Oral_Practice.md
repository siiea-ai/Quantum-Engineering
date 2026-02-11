# Week 154: Variational and WKB Methods - Oral Exam Practice

## Overview

This document contains common oral exam questions on approximation methods. Practice explaining these concepts clearly and completely, as if presenting to a faculty committee.

---

## Variational Method Questions

### Q1: "State and prove the variational principle."

**Statement (30 seconds):**
"For any normalized trial wavefunction, the expectation value of the Hamiltonian is greater than or equal to the ground state energy:
$$E_0 \leq \langle\psi|H|\psi\rangle$$
with equality if and only if the trial function is the exact ground state."

**Proof (2 minutes):**
1. Expand trial function in energy eigenbasis: $|\psi\rangle = \sum_n c_n|n\rangle$
2. Normalization: $\sum_n|c_n|^2 = 1$
3. Calculate: $\langle H\rangle = \sum_n|c_n|^2 E_n$
4. Since $E_n \geq E_0$ for all $n$: $\sum_n|c_n|^2 E_n \geq E_0\sum_n|c_n|^2 = E_0$

**Physical intuition:**
"The variational principle says nature finds the lowest energy state. Any approximation we make will have higher energy than the true ground state."

---

### Q2: "How do you choose a good trial wavefunction?"

**Key criteria:**

1. **Correct symmetry:** Match the Hamiltonian's symmetry
   - Spherical for central potentials
   - Even/odd parity if Hamiltonian is symmetric

2. **Proper boundary conditions:**
   - Vanish at infinity for bound states
   - Satisfy any hard wall conditions

3. **Physical features:**
   - Cusp at Coulomb singularities: $d\psi/dr|_{r=0} = -Z\psi(0)/a_0$
   - Correct nodal structure for excited states

4. **Computational tractability:**
   - Must be able to evaluate integrals analytically or efficiently

**Example:** For hydrogen, $e^{-\alpha r}$ is better than $e^{-\alpha r^2}$ because it captures the cusp.

---

### Q3: "Calculate the helium ground state energy variationally."

**Setup:**
Trial function: $\psi = (Z_{\text{eff}}^3/\pi a_0^3)e^{-Z_{\text{eff}}(r_1+r_2)/a_0}$

**Energy components:**
- Kinetic: $\langle T\rangle = 2 Z_{\text{eff}}^2 \times 13.6$ eV
- Nucleus-electron: $\langle V_{ne}\rangle = -4 Z_{\text{eff}} \times 27.2$ eV (for Z=2)
- Electron-electron: $\langle V_{ee}\rangle = (5Z_{\text{eff}}/8) \times 27.2$ eV

**Optimization:**
$$\frac{dE}{dZ_{\text{eff}}} = 0 \Rightarrow Z_{\text{eff}} = Z - \frac{5}{16} = \frac{27}{16} \approx 1.69$$

**Result:** $E_0 \approx -77.5$ eV (experimental: $-78.98$ eV)

**Physical interpretation:**
"$Z_{\text{eff}} < 2$ because each electron screens the nuclear charge from the other. This screening reduces the effective attraction."

---

### Q4: "Can you use the variational method for excited states?"

**Answer:**
"Yes, with constraints."

**Method 1 - Orthogonality:**
If we ensure $\langle\psi|\psi_0\rangle = 0$, then $\langle\psi|H|\psi\rangle \geq E_1$.

**Method 2 - Linear variational method:**
Use $|\psi\rangle = \sum_i c_i|\phi_i\rangle$ and solve generalized eigenvalue problem.
The N eigenvalues are upper bounds on the first N energy levels.

**Method 3 - Symmetry:**
If the excited state has different symmetry than the ground state, use a trial function with that symmetry.

---

## WKB Questions

### Q5: "Derive the WKB approximation."

**Starting point:**
Schrodinger equation: $-\frac{\hbar^2}{2m}\psi'' + V\psi = E\psi$

**Ansatz:** $\psi = A(x)e^{iS(x)/\hbar}$

**Substitute and expand in powers of $\hbar$:**

Order $\hbar^0$: $(S')^2 = 2m(E-V) = p^2$
- This is the Hamilton-Jacobi equation
- Solution: $S = \pm\int p\,dx$

Order $\hbar^1$: $2S'A' + S''A = 0$
- Solution: $A = C/\sqrt{|p|}$

**Result:**
$$\psi_{\text{WKB}} = \frac{C}{\sqrt{p}}\exp\left(\pm\frac{i}{\hbar}\int p\,dx\right)$$

**Validity condition:** $|d\lambda/dx| \ll 1$ or $|\hbar p'/p^2| \ll 1$

---

### Q6: "State the Bohr-Sommerfeld quantization condition."

**Statement:**
$$\oint p\,dx = \left(n + \frac{1}{2}\right)h$$

**Origin of the 1/2:**
Each classical turning point contributes a phase shift of $\pi/4$.
For a bound state with two turning points: total phase = $\pi/2$ = half a wavelength.

**Application to harmonic oscillator:**
$$\oint p\,dx = \frac{2\pi E}{\omega} = (n+1/2)h$$
$$E_n = (n+1/2)\hbar\omega$$

This is exact!

---

### Q7: "Explain the WKB connection formulas."

**The Problem:**
At turning points, $p \to 0$, so WKB amplitude $1/\sqrt{p} \to \infty$. We need to match solutions across.

**The Solution:**
Near a turning point, the potential is approximately linear. The exact solution involves Airy functions.

**Matching (left turning point, forbidden on left):**

Forbidden region: $\frac{1}{\sqrt{\kappa}}e^{-\int\kappa dx/\hbar}$

Allowed region: $\frac{2}{\sqrt{p}}\sin\left(\int p\,dx/\hbar + \pi/4\right)$

**Key features:**
- Factor of 2 in amplitude
- Phase shift of $\pi/4$
- Only one-way: must be careful about direction

---

### Q8: "Calculate the tunneling probability through a barrier using WKB."

**Formula:**
$$T \approx \exp\left(-\frac{2}{\hbar}\int_{x_1}^{x_2}\sqrt{2m(V-E)}\,dx\right)$$

**Derivation sketch:**
1. In forbidden region, WKB gives exponential decay
2. Match at both turning points
3. Ratio of amplitudes gives transmission

**Example - square barrier:**
$V = V_0$ for $0 < x < a$
$$T = e^{-2\kappa a}$$ where $\kappa = \sqrt{2m(V_0-E)}/\hbar$

**Limitations:**
- Prefactor is approximate (typically factor of 2-10)
- Fails for thin barriers or near-threshold energies

---

### Q9: "Explain alpha decay using WKB."

**Physical picture:**
- Alpha particle trapped inside nucleus (potential well)
- Coulomb barrier outside (repulsion)
- Particle tunnels through barrier

**Gamow's calculation:**
$$T = e^{-2\gamma}$$ where $\gamma = \frac{1}{\hbar}\int_R^{r_0}\sqrt{2m_\alpha(V_C - E)}\,dr$

**Decay rate:**
$$\Gamma = \nu T$$
where $\nu \sim v/R \sim 10^{21}$ s$^{-1}$ is the collision frequency with the barrier.

**Geiger-Nuttall law:**
$$\log t_{1/2} \propto Z/\sqrt{E}$$

"Small changes in energy lead to enormous changes in lifetime because the exponential is very sensitive to the Gamow factor."

---

## Combined/Advanced Questions

### Q10: "When does the variational method fail?"

**It doesn't fail in principle** - it always gives an upper bound.

**But it can give poor results when:**

1. **Bad trial function:** Missing important physics
   - Wrong symmetry
   - Missing nodes
   - Wrong asymptotic behavior

2. **Excited states:** Need orthogonality constraints

3. **Highly correlated systems:** Single-particle ansatz inadequate

**How to improve:**
- Add more variational parameters
- Use better functional forms
- Configuration interaction (multiple determinants)

---

### Q11: "When does WKB break down?"

**WKB fails when:**

1. **Classical turning points:** $p \to 0$, amplitude diverges
   - Solution: Connection formulas

2. **Rapidly varying potential:** $|d\lambda/dx| \not\ll 1$
   - Near singularities
   - Sharp steps

3. **Low quantum numbers:** Semiclassical approximation needs many wavelengths

4. **Above-barrier reflection:** WKB predicts zero reflection for $E > V_{\max}$, but quantum mechanics allows some reflection

**Rule of thumb:** WKB works well when $n \gg 1$ (high quantum numbers).

---

### Q12: "Compare variational and WKB methods."

| Aspect | Variational | WKB |
|--------|-------------|-----|
| Basis | Upper bound theorem | Semiclassical limit |
| When applicable | Always (ground state) | Slowly varying potential |
| Accuracy | Depends on trial function | Better for excited states |
| Information | Single energy | Full spectrum |
| Tunneling | Not directly | Natural framework |

**Complementary uses:**
- Variational: ground state, electronic structure
- WKB: excited states, tunneling, semiclassical physics

---

### Q13: "What is the Born-Oppenheimer approximation?"

**Physical basis:**
- Nuclei are $\sim 2000\times$ heavier than electrons
- Electrons move much faster
- To electrons, nuclei appear stationary
- To nuclei, electrons appear as an average potential

**Mathematical procedure:**
1. Fix nuclear positions $\mathbf{R}$
2. Solve electronic Schrodinger equation → $E_{\text{elec}}(\mathbf{R})$
3. This becomes the potential for nuclear motion
4. Solve nuclear Schrodinger equation

**Result:** Potential energy surfaces for molecular dynamics

**When it fails:**
- Near electronic degeneracies (conical intersections)
- Fast nuclear motion
- Nonadiabatic processes

---

## Quick-Fire Questions

Be prepared to answer these in 30-60 seconds:

1. "What is the variational principle in one sentence?"
   > Any trial function gives an upper bound on the ground state energy.

2. "What does WKB stand for and what is it?"
   > Wentzel-Kramers-Brillouin; a semiclassical approximation where wavelength varies slowly.

3. "Why is $Z_{\text{eff}} < Z$ for helium?"
   > Electron screening - each electron partially shields the nuclear charge from the other.

4. "What's the physical meaning of the Gamow factor?"
   > The logarithm of the tunneling probability through a barrier.

5. "Why does the variational method always work?"
   > Because the exact ground state minimizes the energy, any other state must have higher energy.

---

## Practice Schedule

| Day | Focus | Time |
|-----|-------|------|
| 1 | Q1-Q3 with derivations | 45 min |
| 2 | Q4-Q6 solo practice | 45 min |
| 3 | Q7-Q9 with partner | 45 min |
| 4 | Q10-Q13 and quick-fire | 60 min |
| 5 | Full mock oral | 90 min |

---

## Tips for Success

1. **Start with the big picture** - state the principle before diving into math
2. **Draw pictures** - energy diagrams, turning points, barriers
3. **Know the key numbers** - helium $Z_{\text{eff}} = 27/16$, hydrogen $E_0 = -13.6$ eV
4. **Explain physical meaning** - why does this result make sense?
5. **Know limitations** - when does each method fail?

---

**Remember:** The committee wants to see you think. It's okay to pause and organize your thoughts before answering.
