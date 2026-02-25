# Week 153: Identical Particles - Oral Exam Practice

## Overview

This document contains common oral exam questions on identical particles, along with frameworks for answering them effectively. Practice explaining these concepts out loud, as if presenting to a faculty committee.

---

## Question Framework

For each question, structure your response as:
1. **Statement** - State the key principle or result (30 seconds)
2. **Physical Motivation** - Why this matters physically (1 minute)
3. **Mathematical Derivation** - Key steps (2-3 minutes)
4. **Examples/Applications** - Concrete instances (1 minute)
5. **Connections** - Links to other physics (30 seconds)

---

## Core Concept Questions

### Q1: "Explain why identical particles in quantum mechanics are truly indistinguishable."

**Key Points:**
- In classical mechanics, we can track particles by their trajectories
- In QM, we can only make probabilistic predictions
- No measurement can determine "which particle is which"
- This is not a practical limitation but a fundamental principle

**Physical Consequence:**
- The wavefunction must have definite symmetry under particle exchange
- Only two possibilities: symmetric (bosons) or antisymmetric (fermions)
- This leads to dramatically different behavior (BEC vs. Fermi pressure)

**Example:**
"If I prepare two electrons in a box and later detect one at position A and one at position B, I cannot say which electron is at which position. The question itself is meaningless."

---

### Q2: "State and explain the spin-statistics theorem."

**Statement:**
"Particles with integer spin (0, 1, 2, ...) are bosons with symmetric wavefunctions. Particles with half-integer spin (1/2, 3/2, ...) must be fermions with antisymmetric wavefunctions."

**Why This Is Deep:**
- Connects two seemingly unrelated concepts: spin and exchange statistics
- Requires relativistic quantum field theory to prove rigorously
- Relies on: Lorentz invariance, locality (causality), vacuum stability

**Consequences:**
- Pauli exclusion principle for fermions
- Bose-Einstein condensation for bosons
- Structure of matter (atoms, nuclei, stars)

**What Happens If Violated:**
- Integer spin + antisymmetric → negative norm states (probabilities)
- Half-integer spin + symmetric → causality violation

---

### Q3: "What is the Pauli exclusion principle and why does it follow from antisymmetry?"

**Statement:**
"No two identical fermions can occupy the same quantum state."

**Proof (be ready to write):**
Consider two fermions in the same state $|\phi\rangle$:
$$|\Psi\rangle = |\phi\rangle_1 \otimes |\phi\rangle_2$$

Antisymmetric combination:
$$|\Psi_A\rangle = \frac{1}{\sqrt{2}}(|\phi\rangle_1|\phi\rangle_2 - |\phi\rangle_2|\phi\rangle_1) = 0$$

The state is identically zero - it doesn't exist!

**Physical Consequences:**
- Atomic shell structure and periodic table
- Electron degeneracy pressure (white dwarfs)
- Neutron degeneracy pressure (neutron stars)
- Stability of matter

---

### Q4: "Explain Slater determinants and why they're useful."

**Definition:**
"A Slater determinant is a way to construct an antisymmetric N-fermion wavefunction from N single-particle states."

**Construction:**
$$\Psi = \frac{1}{\sqrt{N!}}\begin{vmatrix}
\phi_1(1) & \phi_1(2) & \cdots \\
\phi_2(1) & \phi_2(2) & \cdots \\
\vdots & & \ddots
\end{vmatrix}$$

**Why Determinants?**
- Exchanging two columns → multiply by -1 (antisymmetry)
- Two identical rows → determinant = 0 (Pauli exclusion)
- Automatically implements all required symmetry properties

**Limitations:**
- Single Slater determinant is only approximate for interacting systems
- Configuration interaction: linear combination of Slater determinants
- True ground state has "correlation" beyond single-determinant description

---

### Q5: "What is second quantization and why is it useful?"

**Core Idea:**
"Instead of tracking N particles in fixed positions, we specify how many particles are in each quantum state."

**Key Objects:**
- Fock space: Hilbert space including all particle numbers
- Creation operator $a^\dagger$: adds one particle
- Annihilation operator $a$: removes one particle
- Number operator $\hat{n} = a^\dagger a$: counts particles

**Advantages:**
1. Variable particle number (field theory, statistical mechanics)
2. Symmetry automatically built in through commutation relations
3. Systematic treatment of many-body interactions
4. Natural language for perturbation theory (Feynman diagrams)

**Commutation vs. Anticommutation:**
- Bosons: $[a_i, a_j^\dagger] = \delta_{ij}$
- Fermions: $\{c_i, c_j^\dagger\} = \delta_{ij}$

---

## Calculation Questions

### Q6: "Calculate the ground state energy of helium to first order in perturbation theory."

**Setup:**
Hamiltonian: $H = H_0 + V_{ee}$ where $H_0$ is two independent hydrogen-like atoms.

**Zeroth Order:**
Each electron sees $Z=2$ nucleus: $E^{(0)}_1 = -Z^2 \times 13.6$ eV $= -54.4$ eV
Total: $E^{(0)} = 2 \times (-54.4) = -108.8$ eV

**First Order Correction:**
$$E^{(1)} = \langle\psi_0|\frac{e^2}{4\pi\epsilon_0 r_{12}}|\psi_0\rangle = \frac{5}{4}\frac{Ze^2}{4\pi\epsilon_0 a_0} = \frac{5}{4}Z \times 27.2 \text{ eV} = 34 \text{ eV}$$

**Result:**
$$E_0 \approx -108.8 + 34 = -74.8 \text{ eV}$$

Experimental: $-78.98$ eV (error ~5%)

**Follow-up:** "How would you improve this?" → Variational method with $Z_{eff}$.

---

### Q7: "Show that $[a, a^\dagger] = 1$ for bosons."

**From ladder operator definition:**
$$a|n\rangle = \sqrt{n}|n-1\rangle, \quad a^\dagger|n\rangle = \sqrt{n+1}|n+1\rangle$$

**Calculate each term:**
$$aa^\dagger|n\rangle = a\sqrt{n+1}|n+1\rangle = \sqrt{n+1}\sqrt{n+1}|n\rangle = (n+1)|n\rangle$$
$$a^\dagger a|n\rangle = n|n\rangle$$

**Commutator:**
$$[a, a^\dagger]|n\rangle = (n+1-n)|n\rangle = |n\rangle$$

Since this holds for all $|n\rangle$: $[a, a^\dagger] = 1$ $\square$

---

### Q8: "Derive the exchange energy splitting for the 1s2s configuration of helium."

**Setup:**
Spatial wavefunctions:
- Symmetric: $\psi_S = \frac{1}{\sqrt{2}}[\phi_{1s}(1)\phi_{2s}(2) + \phi_{1s}(2)\phi_{2s}(1)]$
- Antisymmetric: $\psi_A = \frac{1}{\sqrt{2}}[\phi_{1s}(1)\phi_{2s}(2) - \phi_{1s}(2)\phi_{2s}(1)]$

**Energy expectation:**
$$\langle\psi_S|V_{ee}|\psi_S\rangle = J + K$$
$$\langle\psi_A|V_{ee}|\psi_A\rangle = J - K$$

where:
- Direct integral: $J = \int|\phi_{1s}(1)|^2|\phi_{2s}(2)|^2 V_{ee} \, d^3r_1 d^3r_2$
- Exchange integral: $K = \int\phi_{1s}^*(1)\phi_{2s}^*(2)\phi_{2s}(1)\phi_{1s}(2) V_{ee} \, d^3r_1 d^3r_2$

**Result:**
- Singlet (symmetric spatial, antisymmetric spin): $E_S = E_0 + J + K$
- Triplet (antisymmetric spatial, symmetric spin): $E_T = E_0 + J - K$

Since $K > 0$: **Triplet is lower** → Hund's rule!

---

## Conceptual Deep-Dive Questions

### Q9: "What is the exchange hole and why does it matter?"

**Definition:**
The exchange hole is the region around a fermion where the probability of finding another fermion with the same spin is suppressed.

**Origin:**
Antisymmetry of the wavefunction → $|\psi(r,r)|^2 = 0$ for same-spin particles.

**Physical Size:**
Roughly the Fermi wavelength: $\lambda_F = 2\pi/k_F$

**Importance:**
1. Reduces Coulomb energy in metals (exchange energy is negative)
2. Explains why Hartree-Fock gives good results
3. Basis for density functional theory (DFT)

---

### Q10: "Explain the difference between Hartree and Hartree-Fock approximations."

**Hartree:**
- Mean-field approximation
- Each electron feels average potential from others
- Does NOT include exchange
- Overestimates electron-electron repulsion

**Hartree-Fock:**
- Includes exchange effects through antisymmetry
- Uses Slater determinant ansatz
- Exchange lowers energy (same-spin electrons avoid each other)
- Still misses "correlation" (opposite-spin avoidance)

**Energy Ordering:**
$$E_{\text{exact}} < E_{\text{HF}} < E_{\text{Hartree}}$$

---

### Q11: "Can you have statistics other than Bose or Fermi?"

**In 3D:**
No - exchanging twice must give identity, so eigenvalue of exchange is $\pm 1$.

**In 2D (Anyons):**
Yes! The exchange path cannot be continuously deformed to identity.
Exchange phase can be any $e^{i\theta}$.

**Physical Realizations:**
- Fractional quantum Hall effect: quasiparticles are anyons
- Proposed for topological quantum computing
- Experimentally verified in FQHE systems

---

## Advanced Questions

### Q12: "Explain how second quantization handles the indistinguishability problem."

**Key Insight:**
Instead of labeling particles, we label states and count occupations.

**Example:**
"Two electrons in states $\phi_a$ and $\phi_b$" becomes:
$$|1_a, 1_b\rangle = c_a^\dagger c_b^\dagger |0\rangle$$

No particle labels appear! The antisymmetry is encoded in:
$$c_a^\dagger c_b^\dagger = -c_b^\dagger c_a^\dagger$$

**Advantage:**
The formalism naturally prevents us from writing down unphysical states.

---

### Q13: "Derive the second-quantized form of the Coulomb interaction."

**First quantization:**
$$V = \frac{1}{2}\sum_{i\neq j}\frac{e^2}{4\pi\epsilon_0|\mathbf{r}_i - \mathbf{r}_j|}$$

**Field operator expansion:**
$$\hat{\psi}(\mathbf{r}) = \sum_\alpha \phi_\alpha(\mathbf{r})c_\alpha$$

**Second quantization:**
$$V = \frac{1}{2}\int\int \hat{\psi}^\dagger(\mathbf{r})\hat{\psi}^\dagger(\mathbf{r}')v(\mathbf{r}-\mathbf{r}')\hat{\psi}(\mathbf{r}')\hat{\psi}(\mathbf{r})d^3r\,d^3r'$$

**In momentum space (for uniform systems):**
$$V = \frac{1}{2V}\sum_{\mathbf{k},\mathbf{k}',\mathbf{q}}v_\mathbf{q}c_{\mathbf{k}+\mathbf{q}}^\dagger c_{\mathbf{k}'-\mathbf{q}}^\dagger c_{\mathbf{k}'}c_{\mathbf{k}}$$

Note: The ordering matters for fermions!

---

## Tips for Oral Exam Success

### Do:
- Start with a clear statement of the main result
- Draw pictures when helpful (energy levels, exchange diagrams)
- Mention physical examples and applications
- Acknowledge when you're unsure and reason through it
- Ask clarifying questions if the question is ambiguous

### Don't:
- Rush through derivations - clarity beats speed
- Pretend to know something you don't
- Give one-word answers - explain your thinking
- Forget to check units and limits

### Common Follow-Up Questions:
- "What approximations did you make?"
- "When does this break down?"
- "How would you measure this experimentally?"
- "What's the connection to [related topic]?"

---

## Practice Schedule

| Day | Focus | Time |
|-----|-------|------|
| 1 | Q1-Q3 with partner | 45 min |
| 2 | Q4-Q6 solo, record yourself | 45 min |
| 3 | Q7-Q9 with partner | 45 min |
| 4 | Q10-Q13 comprehensive | 60 min |
| 5 | Full mock oral (all questions) | 90 min |

---

**Remember:** The committee wants to see you *think* like a physicist. It's okay to pause, consider, and work through problems out loud.
