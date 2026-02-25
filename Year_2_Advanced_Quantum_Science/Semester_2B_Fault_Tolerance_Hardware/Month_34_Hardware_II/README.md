# Month 34: Quantum Hardware Platforms II

## Overview

**Days:** 925-952 (28 days)
**Weeks:** 133-136
**Semester:** 2B (Fault Tolerance & Hardware)
**Focus:** Alternative platforms (photonic, topological), error mitigation techniques, and NISQ algorithm design

---

## Status: ✅ COMPLETE

| Week | Days | Topic | Status |
|------|------|-------|--------|
| 133 | 925-931 | Photonic Quantum Computing | ✅ Complete |
| 134 | 932-938 | Topological & Majorana Qubits | ✅ Complete |
| 135 | 939-945 | Error Mitigation Techniques | ✅ Complete |
| 136 | 946-952 | NISQ Algorithm Design | ✅ Complete |

**Progress:** 28/28 days (100%)

---

## Learning Objectives

By the end of this month, you should be able to:

1. **Explain** linear optical quantum computing and measurement-based approaches
2. **Analyze** GKP and cat states for bosonic error correction
3. **Describe** topological quantum computing with Majorana fermions
4. **Compare** error mitigation vs. error correction strategies
5. **Implement** zero-noise extrapolation and probabilistic error cancellation
6. **Design** NISQ algorithms accounting for hardware limitations
7. **Evaluate** variational quantum eigensolver (VQE) and QAOA performance
8. **Assess** near-term quantum advantage opportunities

---

## Weekly Breakdown

### Week 133: Photonic Quantum Computing (Days 925-931)

Light-based quantum computing offers room-temperature operation and natural connectivity.

**Core Topics:**
- Linear optical quantum computing (LOQC)
- KLM protocol and non-deterministic gates
- Boson sampling and Gaussian boson sampling
- Continuous-variable quantum computing
- GKP (Gottesman-Kitaev-Preskill) encoding
- Cat states and bosonic codes
- Integrated photonics and silicon photonics

**Key Equations:**
$$|\text{GKP}\rangle_0 = \sum_{n=-\infty}^{\infty} |2n\sqrt{\pi}\rangle_q$$
$$|\text{cat}_\pm\rangle = \mathcal{N}_\pm(|\alpha\rangle \pm |-\alpha\rangle)$$

### Week 134: Topological & Majorana Qubits (Days 932-938)

Intrinsically protected qubits based on topological order and non-Abelian anyons.

**Core Topics:**
- Topological quantum computing principles
- Majorana fermions and zero modes
- Topological superconductors
- Braiding operations and non-Abelian anyons
- Microsoft's topological approach
- Current experimental status
- Challenges and timeline

**Key Equations:**
$$\gamma = \gamma^\dagger, \quad \gamma^2 = 1$$
$$\{γ_i, γ_j\} = 2δ_{ij}$$

### Week 135: Error Mitigation Techniques (Days 939-945)

Practical techniques to reduce errors on NISQ devices without full error correction.

**Core Topics:**
- Error mitigation vs. error correction
- Zero-noise extrapolation (ZNE)
- Probabilistic error cancellation (PEC)
- Symmetry verification
- Measurement error mitigation
- Dynamical decoupling
- Virtual distillation

**Key Equations:**
$$\langle O \rangle_0 = \lim_{\lambda \to 0} \langle O \rangle_\lambda$$
$$\langle O \rangle_{\text{mitigated}} = \sum_i c_i \langle O_i \rangle$$

### Week 136: NISQ Algorithm Design (Days 946-952)

Designing algorithms that work within the constraints of current quantum hardware.

**Core Topics:**
- NISQ era characteristics and limitations
- Variational algorithms (VQE, QAOA)
- Ansatz design and trainability
- Barren plateaus and expressibility
- Noise-aware compilation
- Hybrid classical-quantum workflows
- Month synthesis and Year 2 progress

**Key Equations:**
$$E(\theta) = \langle \psi(\theta) | H | \psi(\theta) \rangle$$
$$C(\gamma, \beta) = \langle \gamma, \beta | C | \gamma, \beta \rangle$$

---

## Key Concepts

### Photonic Approaches

| Approach | Encoding | Gates | Advantages |
|----------|----------|-------|------------|
| Dual-rail | 1 photon in 2 modes | Linear optics | Simple encoding |
| GKP | Grid states | Gaussian + magic | Error correction |
| Cat | Coherent superposition | Kerr + dissipation | Bias-preserving |
| Cluster | Graph states | Measurements | MBQC |

### Error Mitigation Comparison

| Technique | Overhead | Error Reduction | Applicability |
|-----------|----------|-----------------|---------------|
| ZNE | 2-5× circuits | 10-100× | Local errors |
| PEC | Exponential samples | Exact (principle) | Known noise |
| Symmetry | Post-selection | 10× | Conserved quantities |
| Twirling | Per-gate cost | Tailoring | Coherent errors |

### NISQ Algorithm Landscape

| Algorithm | Application | Qubits Needed | Current Status |
|-----------|-------------|---------------|----------------|
| VQE | Chemistry | 10-100 | Demonstrated |
| QAOA | Optimization | 50-1000 | Promising |
| QML | Machine learning | 10-1000 | Research |
| Simulation | Materials | 50-500 | Active development |

---

## Prerequisites

### From Month 33 (Hardware I)
- Superconducting qubit physics
- Trapped ion operations
- Neutral atom arrays
- Platform trade-offs

### From Semester 2A (Error Correction)
- Stabilizer codes
- Threshold theorem
- Noise models

### Physics Background
- Quantum optics basics
- Topological physics introduction

---

## Resources

### Primary References
- Kok et al., "Linear optical quantum computing with photonic qubits" (2007)
- Nayak et al., "Non-Abelian anyons and topological quantum computation" (2008)
- Temme et al., "Error mitigation for short-depth quantum circuits" (2017)
- Cerezo et al., "Variational quantum algorithms" (2021)

### Key Papers
- Gottesman, Kitaev, Preskill, "Encoding a qubit in an oscillator" (2001)
- Kandala et al., "Error mitigation extends quantum computing" (2019)
- Arute et al., "Quantum supremacy using a programmable superconducting processor" (2019)

### Company Resources
- [Xanadu](https://xanadu.ai/) - Photonic QC
- [PsiQuantum](https://psiquantum.com/) - Photonic QC
- [Microsoft Azure Quantum](https://azure.microsoft.com/quantum/) - Topological

---

## Connections

### From Month 33
- Hardware platforms → Alternative approaches
- Platform comparison → NISQ limitations
- Gate fidelities → Error mitigation need

### To Future Months
- Month 35: Algorithms with error mitigation
- Month 36: QLDPC and future directions
- Year 3: Research-level implementations

---

## Summary

Month 34 completes the hardware survey with photonic and topological platforms, then pivots to practical NISQ-era techniques. Photonic computing offers unique advantages like room-temperature operation and natural connectivity, while topological qubits promise intrinsic error protection. Since fault-tolerant quantum computing remains years away, error mitigation techniques provide essential tools for extracting useful results from noisy devices. Understanding NISQ algorithm design—including variational methods, trainability challenges, and noise-aware compilation—is crucial for practical quantum computing in the current era.

---

*"In the NISQ era, we must be clever about extracting value from imperfect quantum computers."*
— John Preskill

---

**Last Updated:** February 7, 2026
**Status:** ✅ COMPLETE — 28/28 days complete (100%)
