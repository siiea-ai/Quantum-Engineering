# Semester 2A: Quantum Error Correction

## Overview

**Duration:** 6 months (168 days, Days 673-840)
**Months:** 25-30
**Weeks:** 97-120
**Year:** 2 (Advanced Quantum Science)
**Focus:** Complete quantum error correction theory from classical foundations to practical surface code implementations

---

## Status: ✅ COMPLETE

| Month | Weeks | Days | Topic | Status | Progress |
|-------|-------|------|-------|--------|----------|
| **25** | 97-100 | 673-700 | QEC Fundamentals I | ✅ Complete | 28/28 |
| **26** | 101-104 | 701-728 | QEC Fundamentals II | ✅ Complete | 28/28 |
| **27** | 105-108 | 729-756 | Stabilizer Formalism | ✅ Complete | 28/28 |
| **28** | 109-112 | 757-784 | Advanced Stabilizer Applications | ✅ Complete | 28/28 |
| **29** | 113-116 | 785-812 | Topological Codes | ✅ Complete | 28/28 |
| **30** | 117-120 | 813-840 | Surface Codes Deep | ✅ Complete | 28/28 |

**Total Progress:** 168/168 days (100%)

---

## Learning Arc

### Foundation → Application → Implementation

```
Month 25-26: QEC Fundamentals
├── Classical error correction review
├── Quantum error models
├── 3-qubit and 9-qubit codes
└── Knill-Laflamme conditions

Month 27-28: Stabilizer Theory
├── CSS codes and Steane code
├── Stabilizer formalism
├── Fault-tolerant operations
└── Threshold theorems

Month 29-30: Topological Codes
├── Toric code and anyons
├── Surface codes with boundaries
├── Lattice surgery
└── Experimental implementations
```

---

## Month Summaries

### Month 25: QEC Fundamentals I (Days 673-700)

**Focus:** Building intuition from classical to quantum error correction

- **Week 97:** Classical codes (repetition, Hamming, Reed-Solomon)
- **Week 98:** Quantum error models (bit-flip, phase-flip, depolarizing)
- **Week 99:** Three-qubit codes (bit-flip, phase-flip, Shor 9-qubit)
- **Week 100:** Knill-Laflamme conditions, error detection vs correction

**Key Result:** Quantum errors can be discretized and corrected despite continuous nature

### Month 26: QEC Fundamentals II (Days 701-728)

**Focus:** Stabilizer foundations and advanced code theory

- **Week 101:** Stabilizer formalism introduction
- **Week 102:** Gottesman-Knill theorem and Clifford simulation
- **Week 103:** Subsystem codes and gauge qubits
- **Week 104:** Code capacity and quantum channel theory

**Key Result:** Stabilizer codes enable efficient classical simulation and analysis

### Month 27: Stabilizer Formalism (Days 729-756)

**Focus:** Deep dive into CSS codes and graph states

- **Week 105:** CSS code construction from classical codes
- **Week 106:** Stabilizer tableaux and efficient representation
- **Week 107:** Graph states and measurement-based QC
- **Week 108:** Clifford operations and their classification

**Key Result:** Complete toolkit for designing and analyzing stabilizer codes

### Month 28: Advanced Stabilizer Applications (Days 757-784)

**Focus:** Fault tolerance and practical error correction

- **Week 109:** Fault-tolerant operations and gadgets
- **Week 110:** Threshold theorems and concatenation
- **Week 111:** Decoding algorithms (MWPM, Union-Find)
- **Week 112:** Practical QEC system design

**Key Result:** Below threshold, arbitrarily reliable computation is possible with polynomial overhead

### Month 29: Topological Codes (Days 785-812)

**Focus:** Toric codes and topological protection

- **Week 113:** Toric code fundamentals (Kitaev 1997)
- **Week 114:** Anyonic excitations and topological order
- **Week 115:** Surface codes with boundaries
- **Week 116:** Error chains and logical operations

**Key Result:** Topological codes provide natural protection through global properties

### Month 30: Surface Codes Deep (Days 813-840)

**Focus:** Practical surface code implementation

- **Week 117:** Advanced surface code architecture
- **Week 118:** Lattice surgery and logical gates
- **Week 119:** Real-time decoding algorithms
- **Week 120:** Google/IBM experimental implementations

**Key Result:** Surface codes achieve below-threshold operation on real hardware (Google Willow 2024)

---

## Key Concepts Covered

### Error Correction Theory
- Quantum no-cloning and error discretization
- Knill-Laflamme conditions for correctability
- Distance, encoding rate, and code parameters [[n,k,d]]
- Threshold theorem and fault-tolerant computation

### Stabilizer Formalism
- Pauli group and stabilizer groups
- CSS construction from classical codes
- Gottesman-Knill theorem for efficient simulation
- Tableau representation and Clifford operations

### Topological Codes
- Toric code on periodic boundaries [[2L²,2,L]]
- Anyonic excitations: e-particles, m-particles, fermions
- Surface codes with boundaries [[d²,1,d]]
- Homology classes and logical operators

### Fault Tolerance
- Fault-tolerant gadgets and error propagation
- Concatenation and threshold theorems
- Magic state injection and distillation
- Eastin-Knill theorem and universality

### Implementation
- Decoding: MWPM, Union-Find, neural decoders
- Lattice surgery for logical gates
- Real-time decoding constraints
- Hardware: superconducting, trapped-ion, neutral atom

---

## Key Equations

### Code Parameters
$$[[n, k, d]] \text{ encodes } k \text{ logical qubits in } n \text{ physical, distance } d$$

### Knill-Laflamme
$$P_i E_a^\dagger E_b P_j = C_{ab} \delta_{ij}$$

### Threshold Theorem
$$p < p_{th} \Rightarrow p_L^{(k)} \leq p_{th} \left(\frac{p}{p_{th}}\right)^{2^k}$$

### Toric Code Stabilizers
$$A_v = \prod_{e \ni v} X_e, \quad B_p = \prod_{e \in \partial p} Z_e$$

### Surface Code Parameters
$$[[d^2, 1, d]] \text{ (rotated surface code)}$$

### Error Suppression (Google Willow)
$$\Lambda = \frac{\epsilon_L(d)}{\epsilon_L(d+2)} = 2.14 \pm 0.02$$

---

## Computational Tools

```python
# Semester 2A computational stack
import numpy as np
from scipy import linalg, sparse
import matplotlib.pyplot as plt

# Stabilizer simulation
import stim                    # Fast Clifford simulation

# Decoding
import pymatching              # MWPM decoder

# Qiskit for circuits
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, StabilizerState
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
```

---

## Primary References

### Textbooks
- Nielsen & Chuang, Ch. 10 (Quantum Error Correction)
- Preskill Lecture Notes, Ph219 Ch. 7-8
- Lidar & Brun, "Quantum Error Correction" (advanced)
- Gottesman PhD Thesis (foundational)

### Seminal Papers
- Shor 1995: First quantum error-correcting code
- Steane 1996: CSS codes
- Kitaev 1997: Toric code
- Fowler et al. 2012: Surface code review

### Recent Breakthroughs
- Google 2024: Below-threshold surface code (Nature)
- IBM 2024: Heavy-hex error correction
- Various 2024-2025: Dynamic surface codes

---

## Connections

### From Year 1
| Year 1 Topic | Semester 2A Application |
|--------------|------------------------|
| Density matrices | Mixed state error models |
| Quantum channels | Noise characterization |
| Pauli operators | Stabilizer formalism |
| Entanglement | Error detection via ancillas |

### To Semester 2B
| Semester 2A Foundation | Semester 2B Application |
|------------------------|------------------------|
| Fault-tolerant gadgets | Universal FT computation |
| Magic states | T-gate distillation factories |
| Threshold theorems | Hardware error budgets |
| Surface codes | Platform implementations |

---

## What's Next: Semester 2B

**Fault Tolerance & Hardware Platforms** (Months 31-36, Days 841-1008)

- Month 31-32: Complete fault-tolerant quantum computation
- Month 33-34: Hardware platforms (superconducting, trapped-ion, neutral atom)
- Month 35: Advanced quantum algorithms
- Month 36: Year 2 capstone and research frontiers

---

*"The threshold theorem is the most important result in quantum computing."*
— John Preskill

---

**Last Updated:** February 6, 2026
**Status:** ✅ COMPLETE — 168/168 days complete (100%)

