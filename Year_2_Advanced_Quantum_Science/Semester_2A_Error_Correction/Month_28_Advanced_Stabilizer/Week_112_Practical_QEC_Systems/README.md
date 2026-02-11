# Week 112: Practical QEC Systems

## Year 2, Semester 2A: Error Correction | Month 28: Advanced Stabilizer Codes

---

## Week Overview

Week 112 bridges theoretical quantum error correction with practical implementation, addressing the engineering challenges that determine whether fault-tolerant quantum computing is feasible. We analyze resource overheads, explore hardware-efficient encoding strategies, examine cutting-edge experiments, and understand the full quantum computing stack from physical qubits to logical algorithms.

**Prerequisites:** Weeks 109-111 (Stabilizer formalism, surface codes, magic state distillation)

**Week Theme:** *From Theory to Reality: Engineering Fault-Tolerant Quantum Computers*

---

## Daily Schedule

| Day | Date | Topic | Status | Key Concepts |
|-----|------|-------|--------|--------------|
| 778 | Monday | Resource Overhead Analysis | ✅ Complete | Qubit overhead, gate overhead, space-time volume |
| 779 | Tuesday | Lattice Surgery Operations | ✅ Complete | Logical CNOT, merge/split, twist defects |
| 780 | Wednesday | Code Switching & Gauge Fixing | ✅ Complete | Transversal gate switching, subsystem codes |
| 781 | Thursday | Hardware-Efficient Codes | ✅ Complete | Cat codes, GKP states, biased-noise codes |
| 782 | Friday | Near-Term QEC Experiments | ✅ Complete | Google, IBM, IonQ demonstrations |
| 783 | Saturday | Quantum Computer Architecture | ✅ Complete | Full stack, control systems, cryogenics |
| 784 | Sunday | Month 28 Synthesis | ✅ Complete | Comprehensive review, concept integration |

---

## Learning Objectives

By the end of Week 112, you will be able to:

1. **Calculate Resource Overheads** - Quantify physical qubit counts, gate costs, and space-time volumes for fault-tolerant quantum algorithms
2. **Design Lattice Surgery Circuits** - Implement logical multi-qubit gates through merge and split operations on surface code patches
3. **Navigate Code Switching** - Transition between quantum codes to access different transversal gate sets
4. **Evaluate Hardware-Efficient Codes** - Analyze bosonic codes (cat, GKP) and bias-exploiting codes for specific hardware
5. **Interpret Experimental Results** - Critically assess current QEC experiments and their implications
6. **Understand Full-Stack Architecture** - Describe all layers from physical qubits to compiled algorithms

---

## Core Concepts

### 1. Resource Overhead Formulas

**Physical Qubit Count (Surface Code):**
$$\boxed{N_{\text{phys}} = 2d^2 - 1 \approx 2d^2}$$

**T-Gate Factory Overhead:**
$$\boxed{N_{\text{factory}} \approx 15d^2 \cdot k \cdot \lceil \log_{15}(1/\epsilon_T) \rceil}$$

where $k$ is the number of parallel distillation rounds.

**Space-Time Volume:**
$$\boxed{V_{ST} = N_{\text{qubits}} \times T_{\text{cycles}}}$$

### 2. Lattice Surgery

Logical operations via patch manipulation:
- **Merge:** Join two surface code patches to perform joint measurement
- **Split:** Separate a patch to create independent logical qubits
- **Twist defects:** Alternative topological approach using code deformations

**Lattice Surgery CNOT Time:**
$$\boxed{T_{\text{CNOT}} = d \text{ syndrome cycles}}$$

### 3. Hardware-Efficient Encodings

**Cat Code (Kerr-cat):**
$$\boxed{|\mathcal{C}_\alpha^\pm\rangle = \mathcal{N}_\pm(|\alpha\rangle \pm |-\alpha\rangle)}$$

with engineered dissipation $\kappa_1 \ll \kappa_2$.

**GKP (Gottesman-Kitaev-Preskill) Code:**
$$\boxed{|\bar{0}\rangle_{\text{GKP}} \propto \sum_{s=-\infty}^{\infty} |2s\sqrt{\pi}\rangle_q}$$

Grid states in phase space providing protection against small displacements.

### 4. Experimental Milestones

| Platform | Achievement | Year |
|----------|------------|------|
| Google Sycamore | Break-even QEC (distance 5 > distance 3) | 2023 |
| IBM | Heavy-hex lattice, distance-3 surface code | 2022-2024 |
| IonQ | Shuttling-based logical operations | 2024 |
| QuEra | Neutral atom surface codes | 2023-2024 |

---

## Key Formulas Summary

| Concept | Formula |
|---------|---------|
| Surface code qubits | $N = 2d^2$ |
| Logical error rate | $p_L \sim (p/p_{\text{th}})^{\lfloor(d+1)/2\rfloor}$ |
| Lattice surgery time | $T = O(d)$ code cycles |
| T-factory output rate | $\sim 1$ T-state per $O(d)$ cycles |
| GKP squeezing threshold | $\Delta < 0.5$ (9.5 dB) |
| Cat code bit-flip suppression | $\propto e^{-2|\alpha|^2}$ |

---

## Computational Tools

This week emphasizes simulation of practical QEC systems:

```python
# Core libraries for Week 112
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.special import factorial
import networkx as nx

# Key simulations:
# - Resource counting for fault-tolerant algorithms
# - Lattice surgery scheduling optimization
# - Bosonic code state visualization
# - Experimental data analysis
```

---

## Reading List

### Primary Sources
1. Fowler et al., "Surface codes: Towards practical large-scale quantum computation" (2012)
2. Horsman et al., "Surface code quantum computing by lattice surgery" (2012)
3. Litinski, "A Game of Surface Codes" (2019)
4. Chamberland et al., "Building a Fault-Tolerant Quantum Computer Using Concatenated Cat Codes" (2022)

### Experimental Papers
5. Google Quantum AI, "Suppressing quantum errors by scaling a surface code logical qubit" (2023)
6. IBM, "Evidence for the utility of quantum computing before fault tolerance" (2023)

### Architecture
7. Van Meter & Horsman, "A Blueprint for Building a Quantum Computer" (2013)

---

## Assessment Criteria

- [ ] Can calculate physical qubit requirements for a target algorithm
- [ ] Can design lattice surgery circuits for multi-qubit Clifford operations
- [ ] Understands trade-offs between different code families
- [ ] Can critically analyze experimental QEC results
- [ ] Can describe the full quantum computing stack
- [ ] Prepared for Month 29: Topological Error Correction

---

## Connection to Research Frontier

Week 112 directly addresses the central challenge of quantum computing: **scaling**. Current NISQ devices (~100-1000 physical qubits) must grow to millions of physical qubits for useful fault-tolerant computation. Understanding the practical aspects covered this week is essential for:

- Quantum algorithm resource estimation
- Hardware architecture design
- Identifying the most promising paths to fault tolerance
- Interpreting experimental progress claims

---

*Week 112 of 312 | Year 2, Month 28, Week 4*
*Quantum Engineering PhD Curriculum*
