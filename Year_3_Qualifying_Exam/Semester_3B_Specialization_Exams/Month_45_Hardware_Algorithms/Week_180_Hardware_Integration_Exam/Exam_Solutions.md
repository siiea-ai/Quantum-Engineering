# Month 45: Hardware & Algorithms Practice Exam - Solutions

---

# Part A: Short Answer Solutions (20 points)

## Solution A1 (2 points)
The anharmonicity in the transmon arises from the **cosine nonlinearity of the Josephson junction potential**. Expanding $$-E_J\cos\phi$$ beyond the quadratic term gives a quartic correction that makes the energy levels unequally spaced. This is essential because it allows us to address only the $$|0\rangle \leftrightarrow |1\rangle$$ transition without exciting higher levels, enabling qubit operation rather than harmonic oscillator behavior.

---

## Solution A2 (2 points)
The **Rydberg blockade** occurs when two atoms are close enough that their Rydberg-Rydberg interaction energy $$V = C_6/R^6$$ exceeds the laser driving linewidth $$\hbar\Omega$$. When this happens, the doubly-excited state $$|rr\rangle$$ is shifted off-resonance and cannot be populated, preventing simultaneous Rydberg excitation of both atoms. The characteristic distance where this occurs is the blockade radius $$R_b = (C_6/\hbar\Omega)^{1/6}$$.

---

## Solution A3 (2 points)
In the **Cooper pair box regime** ($$E_J/E_C \sim 1$$), the qubit frequency is highly sensitive to offset charge fluctuations, leading to severe dephasing. By operating at $$E_J/E_C \gg 1$$ (transmon regime), charge sensitivity is **exponentially suppressed** as $$\propto e^{-\sqrt{8E_J/E_C}}$$. The trade-off is reduced anharmonicity ($$\alpha \approx -E_C$$), but this remains sufficient for qubit operation (~200-300 MHz).

---

## Solution A4 (2 points)
The **Lamb-Dicke parameter** $$\eta = k\sqrt{\hbar/(2m\omega)}$$ is the ratio of the zero-point motion of a trapped ion to the laser wavelength. It characterizes the coupling between internal (spin) and external (motional) degrees of freedom. The **Lamb-Dicke regime** is defined by $$\eta\sqrt{n+1} \ll 1$$, where $$n$$ is the motional quantum number, meaning the ion's motion is small compared to the laser wavelength.

---

## Solution A5 (2 points)
**GKP codes** correct small displacement errors in position and momentum by encoding in grid states; they treat position and momentum errors symmetrically. **Cat codes** exploit biased noise—bit-flip errors are exponentially suppressed ($$\propto e^{-2|\alpha|^2}$$) while phase-flip errors occur at normal rates. This allows cat codes to use simpler repetition codes for the remaining Z errors, while GKP requires measurement of both quadratures.

---

## Solution A6 (2 points)
A **barren plateau** is a region in the variational parameter landscape where gradients vanish exponentially with system size: $$\text{Var}[\partial_\theta E] \sim O(1/2^n)$$. This makes optimization essentially impossible because an exponential number of shots would be needed to detect the gradient direction. Barren plateaus occur in highly expressive ansatze, deep random circuits, and with global cost functions.

---

## Solution A7 (2 points)
The MS gate uses a **bichromatic laser field** that creates spin-dependent displacements in phase space. After the gate time $$\tau = 2\pi/\delta$$, the motional mode returns to its initial position (closed loop), and the accumulated **geometric phase depends only on the enclosed area**, not on the initial motional state. Therefore, the entangling phase is independent of the thermal phonon number $$n$$, making the gate robust to motional heating.

---

## Solution A8 (2 points)
**Zero-noise extrapolation** measures observables at multiple noise levels (by artificially amplifying noise via gate folding) and extrapolates to the zero-noise limit. It **fails when**: (1) the noise is not a simple function of the scaling parameter, (2) the total error is too large (>50%) for reliable extrapolation, (3) non-depolarizing noise models don't scale as assumed, or (4) the extrapolation model (linear, exponential) is incorrect.

---

## Solution A9 (2 points)
**Superconducting qubits** typically have **nearest-neighbor connectivity** determined by the chip layout (grid, heavy-hex), requiring SWAP gates for non-adjacent operations. **Trapped ions** have **all-to-all connectivity** because any two ions can be coupled via their shared motional modes, or ions can be physically shuttled to interact (QCCD). This gives trapped ions a significant advantage for algorithms requiring long-range interactions.

---

## Solution A10 (2 points)
The **approximation ratio** is the ratio of the QAOA objective value to the optimal (maximum) value: $$r = \langle H_C\rangle_{QAOA} / C_{max}$$. For MaxCut on **3-regular graphs** at $$p=1$$, QAOA achieves an approximation ratio of at least **0.6924**, meaning it finds solutions worth at least 69.24% of the maximum cut, better than random (50%) but below the classical Goemans-Williamson algorithm (87.8%).

---

# Part B: Derivation Solutions (30 points)

## Solution B1 (10 points)

**(a) Classical Lagrangian (3 points):**

For a Josephson junction with shunt capacitance $$C$$:

$$\mathcal{L} = T - U = \frac{1}{2}C\dot{\Phi}^2 + E_J\cos\left(\frac{2\pi\Phi}{\Phi_0}\right)$$

where $$\Phi$$ is the node flux, $$E_J = I_c\Phi_0/(2\pi)$$ is the Josephson energy, and $$\Phi_0 = h/(2e)$$ is the flux quantum.

**(b) Conjugate Momentum and Hamiltonian (3 points):**

The conjugate momentum (charge) is:

$$Q = \frac{\partial\mathcal{L}}{\partial\dot{\Phi}} = C\dot{\Phi}$$

The Hamiltonian via Legendre transformation:

$$H = Q\dot{\Phi} - \mathcal{L} = \frac{Q^2}{2C} - E_J\cos\left(\frac{2\pi\Phi}{\Phi_0}\right)$$

**(c) Quantization (4 points):**

Promote to operators with $$[\hat{\Phi}, \hat{Q}] = i\hbar$$.

Define dimensionless phase $$\hat{\phi} = 2\pi\hat{\Phi}/\Phi_0$$ and number operator $$\hat{n} = \hat{Q}/(2e)$$.

The charging energy is $$E_C = e^2/(2C)$$.

The quantum Hamiltonian becomes:

$$\boxed{\hat{H} = 4E_C\hat{n}^2 - E_J\cos\hat{\phi}}$$

---

## Solution B2 (10 points)

**(a) Blockade Radius Derivation (4 points):**

The blockade condition is that the interaction energy exceeds the laser linewidth:

$$V(R_b) = \hbar\Omega$$

$$\frac{C_6}{R_b^6} = \hbar\Omega$$

Solving for $$R_b$$:

$$\boxed{R_b = \left(\frac{C_6}{\hbar\Omega}\right)^{1/6}}$$

**(b) Numerical Calculation (3 points):**

$$R_b = \left(\frac{500 \times 10^9 \text{ Hz}}{2 \times 10^6 \text{ Hz}}\right)^{1/6} \text{ μm}$$

$$R_b = (2.5 \times 10^5)^{1/6} \text{ μm} = (250000)^{1/6} \text{ μm}$$

$$R_b \approx 7.9 \text{ μm}$$

$$\boxed{R_b \approx 8 \text{ μm}}$$

**(c) Atoms in Blockade Sphere (3 points):**

Volume of blockade sphere: $$V_b = \frac{4}{3}\pi R_b^3 = \frac{4}{3}\pi (8)^3 \approx 2145 \text{ μm}^3$$

Volume per atom (cubic lattice, 4 μm spacing): $$V_{atom} = (4)^3 = 64 \text{ μm}^3$$

Number of atoms: $$N = V_b/V_{atom} = 2145/64 \approx 33$$

$$\boxed{N \approx 33 \text{ atoms}}$$

---

## Solution B3 (10 points)

**(a) Cost Hamiltonian (3 points):**

For edges (1,2), (2,3), (1,3):

$$\hat{H}_C = \frac{1}{2}(1 - Z_1Z_2) + \frac{1}{2}(1 - Z_2Z_3) + \frac{1}{2}(1 - Z_1Z_3)$$

$$\boxed{\hat{H}_C = \frac{3}{2} - \frac{1}{2}(Z_1Z_2 + Z_2Z_3 + Z_1Z_3)}$$

**(b) QAOA Ansatz for p=1 (3 points):**

$$|\gamma, \beta\rangle = e^{-i\beta H_M} e^{-i\gamma H_C} |+\rangle^{\otimes 3}$$

where:
- $$H_M = X_1 + X_2 + X_3$$
- $$|+\rangle^{\otimes 3} = H^{\otimes 3}|000\rangle$$

$$\boxed{|\gamma,\beta\rangle = e^{-i\beta(X_1+X_2+X_3)}e^{-i\gamma H_C}|+++\rangle}$$

**(c) Approximation Ratio (2 points):**

$$r = \frac{\langle H_C\rangle_{QAOA}}{C_{max}} = \frac{1.5}{2} = 0.75$$

$$\boxed{r = 0.75 = 75\%}$$

**(d) Mixer Interpretation (2 points):**

The mixer Hamiltonian $$H_M = \sum_i X_i$$ generates **superpositions between computational basis states**. It enables quantum tunneling between different bitstring solutions, allowing the algorithm to explore the solution space beyond local changes. This is the "quantum" part of QAOA that classical local search cannot replicate.

---

# Part C: Problem Solving Solutions (30 points)

## Solution C1 (10 points)

**(a) Dispersive Regime Check (2 points):**

$$\Delta = \omega_q - \omega_r = 5.0 - 7.0 = -2.0 \text{ GHz}$$

$$\frac{g}{|\Delta|} = \frac{100}{2000} = 0.05 \ll 1$$

$$\boxed{\text{Yes, dispersive regime (}g/|\Delta| = 0.05\text{)}}$$

**(b) Dispersive Shift (3 points):**

$$\chi = \frac{g^2}{\Delta}\frac{\alpha}{\Delta + \alpha} = \frac{(100)^2}{-2000}\frac{-250}{-2000 + (-250)} \text{ MHz}$$

$$\chi = \frac{10000}{-2000}\frac{-250}{-2250} = (-5)(0.111) = -0.56 \text{ MHz}$$

$$\boxed{\chi/2\pi \approx -0.6 \text{ MHz}}$$

**(c) Readout Quality (2 points):**

$$\chi/\kappa = 0.6/2 = 0.3$$

This is reasonable but on the low side. Optimal is $$\chi/\kappa \sim 1$$.

$$\boxed{\chi/\kappa = 0.3 \text{ (acceptable, not optimal)}}$$

**(d) Purcell Limit (3 points):**

$$T_1^{Purcell} = \frac{\Delta^2}{\kappa g^2} = \frac{(2000 \times 10^6)^2}{(2 \times 10^6)(100 \times 10^6)^2} \text{ s}$$

$$= \frac{4 \times 10^{18}}{2 \times 10^{22}} = 2 \times 10^{-4} \text{ s} = 200 \text{ μs}$$

$$\boxed{T_1^{Purcell} = 200 \text{ μs}}$$

---

## Solution C2 (10 points)

**(a) Total Gate Error (2 points):**

$$\epsilon_{total} = 15 \times 0.001 + 10 \times 0.01 = 0.015 + 0.1 = 0.115$$

$$\boxed{\epsilon_{gate} \approx 11.5\%}$$

**(b) Dephasing Error (2 points):**

$$\epsilon_{dephasing} = 1 - e^{-t/T_2} \approx t/T_2 = 2/50 = 0.04$$

$$\boxed{\epsilon_{dephasing} \approx 4\%}$$

**(c) ZNE Extrapolation (3 points):**

Linear fit through points (1, -1.05), (2, -0.95), (3, -0.85):

Slope: $$m = \frac{-0.85 - (-1.05)}{3-1} = \frac{0.2}{2} = 0.1$$ Ha per noise unit

Extrapolate to $$\lambda = 0$$:

$$E(0) = E(1) - 1 \times m = -1.05 - 0.1 = -1.15 \text{ Ha}$$

$$\boxed{E(0) \approx -1.15 \text{ Ha}}$$

**(d) Remaining Error Analysis (3 points):**

Remaining error: $$|-1.15 - (-1.20)| = 0.05$$ Ha = 50 mHa

Possible reasons:
- Nonlinear noise dependence (linear model insufficient)
- Ansatz expressibility error (not reaching true ground state)
- Systematic calibration errors not captured by ZNE
- Readout errors not mitigated

$$\boxed{\text{Error} = 50 \text{ mHa; likely due to ansatz limitations or nonlinear noise}}$$

---

## Solution C3 (10 points)

**(a) Platform A Depth with SWAPs (3 points):**

Average SWAP distance: 5 qubits

Each CNOT between non-adjacent qubits needs ~5 SWAPs = 15 CNOTs

For 200 two-qubit gates: $$200 \times (1 + 15)/\bar{d} \approx 200 \times 3 = 600$$ effective gates

(More careful estimate: ~40% need SWAPs, each adds 3 SWAPs → 600 total)

$$\boxed{\text{Effective depth} \approx 600 \text{ gates}}$$

**(b) Fidelity Calculations (4 points):**

**Platform A:**
$$F_A = (1 - 0.005)^{600} = (0.995)^{600} \approx e^{-3} \approx 0.05$$

**Platform B:**
Only 32 qubits (insufficient for 50-qubit problem)
$$F_B = \text{N/A (not enough qubits)}$$

**Platform C:**
$$F_C = (1 - 0.02)^{200} = (0.98)^{200} \approx e^{-4} \approx 0.02$$

$$\boxed{F_A \approx 5\%, \quad F_B = \text{N/A}, \quad F_C \approx 2\%}$$

**(c) Platform Choice and Mitigation (3 points):**

**Choice: Platform A (superconducting)**

Reasons:
- Platform B doesn't have enough qubits
- Platform A has higher fidelity than C despite SWAP overhead
- Platform A's 5% fidelity can potentially be improved with error mitigation

**Error mitigation strategy:**
- ZNE (3 noise levels) could improve by factor ~2-3
- Symmetry verification if applicable
- May achieve ~10-15% effective fidelity, still below target

**Note:** None of the platforms meet the 50% fidelity target. Would need:
- Shallower algorithm reformulation
- Error correction
- Hardware improvements

$$\boxed{\text{Platform A with ZNE; target still not met without circuit optimization}}$$

---

# Part D: Essay Solution (20 points)

## Model Answer

**Quantum Computing for Transition Metal Catalysis: A Hardware and Algorithm Analysis**

**Introduction and Qubit Requirements**

Studying a transition metal catalyst with 30 electrons in the active site presents a significant challenge for both classical and quantum computation. The full orbital space would require hundreds of qubits, which is impractical for current devices. Instead, we must select an **active space** that captures the essential chemistry.

For transition metal complexes, the d-orbitals of the metal center and the immediately bonding ligand orbitals are most important. A reasonable active space might include 10-14 electrons in 10-14 orbitals, requiring **10-14 qubits** after Jordan-Wigner mapping. This can be further reduced to ~8-12 qubits using symmetry (spin, point group).

**Hardware Recommendation: Trapped Ion System**

For this chemistry problem, I recommend the **30-qubit Quantinuum trapped ion system** over the 100-qubit IBM superconducting device for several reasons:

1. **Gate fidelity**: Trapped ions achieve 99.9% two-qubit gate fidelity versus ~99.5% for superconducting. For chemistry requiring "chemical accuracy" (1 kcal/mol ≈ 1.6 mHa), every error matters.

2. **All-to-all connectivity**: Molecular Hamiltonians have long-range interactions. On superconducting hardware, simulating these requires extensive SWAP networks, inflating circuit depth 3-5×. Trapped ions can implement any two-qubit gate directly.

3. **Coherence times**: Ion qubits have T2 times of seconds to minutes, compared to ~100 μs for superconducting qubits. This provides more margin for the deep circuits chemistry requires.

4. **Qubit count is sufficient**: With 30 qubits available and ~12 needed after active space selection, we have overhead for potential ancilla-assisted error mitigation.

**Ansatz Selection**

For this strongly correlated system, I recommend **ADAPT-VQE** rather than fixed-structure ansatze:

1. UCCSD would require O(N⁴) gates for 12 orbitals, resulting in thousands of gates—likely too deep even for trapped ions.

2. Hardware-efficient ansatze risk barren plateaus and may not capture the multi-reference character of transition metal d-orbitals.

3. ADAPT-VQE iteratively builds a problem-specific ansatz, typically achieving UCCSD-quality results with 30-50% of the circuit depth. For transition metals, the operator pool should include generalized singles and doubles allowing orbital relaxation.

**Error Sources and Mitigation**

The main error sources are:

1. **Gate errors** (~0.1% per two-qubit gate × ~100 gates = ~10% total)
2. **Dephasing** (circuit time ~10 ms vs T2 ~1 s → ~1%)
3. **Measurement errors** (~1% per qubit)
4. **Ansatz expressibility** (systematic, hardware-independent)

Mitigation strategy:
- **Symmetry verification**: Post-select on correct electron number and spin
- **Zero-noise extrapolation**: Scale motional heating by varying laser parameters
- **Purification**: Use redundant measurements for error detection

**Expected Accuracy**

Realistically, with current technology we might achieve:
- Raw circuit fidelity: ~80-90%
- After mitigation: ~95%
- Remaining systematic error: ~5-10 mHa

This represents an energy error of ~3-6 kcal/mol—useful for qualitative trends but not quantitative thermochemistry. Classical methods like CCSD(T) can achieve 1 kcal/mol for smaller systems, but they fail for strongly correlated transition metals where quantum computers could offer genuine advantage.

**Conclusion**

For this transition metal catalyst study, trapped ions offer the best current path forward due to their superior fidelity and connectivity. With careful active space selection, ADAPT-VQE ansatz, and comprehensive error mitigation, we can obtain chemically meaningful results that complement—though not yet replace—classical multireference methods.

---

## Scoring Rubric for Part D

| Criterion | Points | Description |
|-----------|--------|-------------|
| Active space analysis | 4 | Correct qubit estimate, justification |
| Hardware comparison | 5 | Clear reasoning, multiple factors |
| Ansatz selection | 4 | Appropriate choice with justification |
| Error analysis | 4 | Comprehensive, quantitative where possible |
| Accuracy assessment | 3 | Realistic, compared to classical |

**Total: 20 points**
