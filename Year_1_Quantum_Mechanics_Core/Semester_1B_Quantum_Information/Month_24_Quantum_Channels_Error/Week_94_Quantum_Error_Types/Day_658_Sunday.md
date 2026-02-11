# Day 658: Week 94 Review - Quantum Error Types

## Week 94: Quantum Error Types | Month 24: Quantum Channels & Error Introduction

---

## Week Summary

This week we studied the fundamental quantum error types that affect real quantum systems:

| Day | Topic | Key Insight |
|-----|-------|-------------|
| 652 | Bit-Flip (X) | Classical-like error, correctable by repetition |
| 653 | Phase-Flip (Z) | Uniquely quantum, invisible to Z-basis measurement |
| 654 | General Pauli | Unified framework, twirling produces Pauli channels |
| 655 | Depolarizing | Symmetric "worst-case" noise, universal error model |
| 656 | Amplitude Damping | Physical energy decay, non-unital, T1 process |
| 657 | Practical Errors | T1/T2 parameters, gate errors, NISQ noise |

---

## Error Channel Comparison

| Channel | Kraus Rank | Unital? | Fixed Points | Physical Origin |
|---------|-----------|---------|--------------|-----------------|
| Bit-flip | 2 | Yes | $\|+\rangle,\|-\rangle$ line | Transverse fluctuations |
| Phase-flip | 2 | Yes | $\|0\rangle,\|1\rangle$ line | Longitudinal fluctuations |
| Depolarizing | 4 | Yes | $I/2$ only | Isotropic noise |
| Amplitude damping | 2 | **No** | $\|0\rangle$ only | Energy relaxation |
| Phase damping | 2 | Yes | Z-axis | Pure dephasing |

---

## Key Formulas

### Pauli Channels
$$\mathcal{E}_X(\rho) = (1-p)\rho + pX\rho X$$
$$\mathcal{E}_Z(\rho) = (1-p)\rho + pZ\rho Z$$
$$\mathcal{E}_{\text{dep}}(\rho) = (1-p)\rho + p\frac{I}{2}$$

### Amplitude Damping
$$K_0 = \begin{pmatrix}1 & 0\\0 & \sqrt{1-\gamma}\end{pmatrix}, \quad K_1 = \begin{pmatrix}0 & \sqrt{\gamma}\\0 & 0\end{pmatrix}$$

### Coherence Decay
- Bit-flip: $\rho_{01} \mapsto (1-2p)\rho_{01}$
- Phase-flip: $\rho_{01} \mapsto (1-2p)\rho_{01}$
- Amplitude damping: $\rho_{01} \mapsto \sqrt{1-\gamma}\rho_{01}$

### T1/T2 Relations
$$P_1(t) = P_1(0)e^{-t/T_1}, \quad \rho_{01}(t) = \rho_{01}(0)e^{-t/T_2}$$
$$T_2 \leq 2T_1$$

---

## Comprehensive Problems

### Problem 1: Channel Identification
A channel transforms the Bloch sphere as follows: $(r_x, r_y, r_z) \mapsto (0.8r_x, 0.8r_y, r_z)$.
a) Is this a Pauli channel?
b) Find the Kraus operators.
c) What physical process might cause this?

### Problem 2: Error Comparison
Starting from $|+\rangle$, apply each error channel with the same parameter ($p = \gamma = 0.1$). Rank them by:
a) Output purity
b) Fidelity with original state
c) Coherence magnitude

### Problem 3: Practical Circuit
A 50-gate circuit runs on a device with:
- T1 = 100μs, T2 = 80μs
- Gate time = 50ns
- Gate error = 0.1%

Estimate the total circuit fidelity.

### Problem 4: Channel Composition
Show that bit-flip followed by phase-flip gives a Y-error component:
$$\mathcal{E}_Z \circ \mathcal{E}_X(\rho) = ?$$

### Problem 5: Fixed Point Analysis
For the generalized amplitude damping channel at finite temperature, find the fixed point (thermal equilibrium state).

---

## Solutions Outline

**Problem 1:**
a) Yes, it's phase damping (contracts only x-y plane)
b) $K_0 = \begin{pmatrix}1 & 0\\0 & 0.8\end{pmatrix}$, $K_1 = \begin{pmatrix}0 & 0\\0 & 0.6\end{pmatrix}$ (approx)
c) Pure dephasing from low-frequency noise

**Problem 3:**
- Idle error: $50 \times 50\text{ns}/80\mu\text{s} \approx 0.03$
- Gate error: $50 \times 0.001 = 0.05$
- Total: $F \approx (1-0.08) \approx 0.92$

---

## Self-Assessment

### Conceptual Understanding
- [ ] Can I explain why phase-flip is "uniquely quantum"?
- [ ] Do I understand why amplitude damping is non-unital?
- [ ] Can I relate T1/T2 to specific channel types?

### Computational Skills
- [ ] Can I implement all error channels in code?
- [ ] Can I simulate realistic noise models?
- [ ] Can I estimate circuit fidelity from device parameters?

### Problem Solving
- [ ] Can I identify channel type from Bloch sphere behavior?
- [ ] Can I compose different error channels?
- [ ] Can I design detection strategies for different errors?

---

## Looking Ahead: Week 95

Next week we begin **Quantum Error Detection and Correction**:

| Day | Topic |
|-----|-------|
| 659 | Classical Error Correction Review |
| 660 | Quantum Error Correction Conditions |
| 661 | Three-Qubit Bit-Flip Code |
| 662 | Three-Qubit Phase-Flip Code |
| 663 | Nine-Qubit Shor Code |
| 664 | Stabilizer Formalism Preview |
| 665 | Week Review |

---

## Key Takeaways

1. **Different errors require different corrections**: Bit-flip vs phase-flip need different code types
2. **Pauli errors are "nice"**: Easy to analyze, twirling makes any channel Pauli
3. **Amplitude damping is "physical"**: Reflects real energy decay, non-unital
4. **T1/T2 characterize real devices**: Essential parameters for noise modeling
5. **Error rates compound**: Circuit fidelity decays exponentially with depth
6. **Error correction is essential**: Current devices are at/near threshold

---

**Week 94 Complete!**

You now understand the fundamental error types in quantum computing. Week 95 will show how to protect quantum information against these errors through quantum error correction.
