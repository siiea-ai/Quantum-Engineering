# Month 36: Year 2 Capstone

## Overview

**Days:** 981-1008 (28 days)
**Weeks:** 141-144
**Semester:** 2B (Fault Tolerance & Hardware)
**Focus:** QLDPC codes, research frontiers, comprehensive Year 2 review, and Year 3 preparation

---

## Status: ✅ COMPLETE

| Week | Days | Topic | Status |
|------|------|-------|--------|
| 141 | 981-987 | QLDPC Codes & Constant-Overhead QEC | ✅ Complete |
| 142 | 988-994 | Research Frontiers (2025-2026) | ✅ Complete |
| 143 | 995-1001 | Year 2 Comprehensive Review | ✅ Complete |
| 144 | 1002-1008 | Year 3 Preview & Research Preparation | ✅ Complete |

**Progress:** 28/28 days (100%)

---

## Learning Objectives

By the end of this month, you should be able to:

1. **Explain** quantum LDPC codes and their asymptotic advantages
2. **Analyze** constant-overhead fault tolerance and its implications
3. **Describe** the latest research breakthroughs in quantum computing
4. **Synthesize** all Year 2 knowledge into a coherent framework
5. **Evaluate** open problems and research directions
6. **Design** a research proposal leveraging Year 2 expertise
7. **Prepare** for Year 3 qualifying exam topics
8. **Demonstrate** research-level quantum computing competency

---

## Weekly Breakdown

### Week 141: QLDPC Codes & Constant-Overhead QEC (Days 981-987)

The future of quantum error correction: asymptotically optimal codes.

**Core Topics:**
- Classical LDPC codes and belief propagation
- Quantum LDPC (qLDPC) code construction
- Good qLDPC codes: constant rate + distance
- Panteleev-Kalachev breakthrough (2022)
- Constant-overhead fault tolerance
- Comparison with surface codes
- Implementation challenges

**Key Equations:**
$$[[n, k, d]] \text{ with } k = \Theta(n), d = \Theta(n)$$
$$\text{Overhead: } O(1) \text{ vs. } O(d^2) \text{ for surface codes}$$

### Week 142: Research Frontiers (2025-2026) (Days 988-994)

Current state-of-the-art and emerging directions in quantum science.

**Core Topics:**
- Recent experimental milestones
- Logical qubit demonstrations
- Quantum advantage claims and verification
- New algorithmic discoveries
- Hardware scaling progress
- Industry vs. academic developments
- Open problems in the field

**Key Topics:**
- Google's logical qubit improvements
- IBM's 1000+ qubit roadmap
- Neutral atom breakthroughs
- Error correction demonstrations
- Quantum simulation achievements

### Week 143: Year 2 Comprehensive Review (Days 995-1001)

Complete synthesis of Year 2: Advanced Quantum Science.

**Core Topics:**
- Semester 2A: Error Correction review
  - Classical → quantum codes
  - Stabilizer formalism
  - Topological codes
  - Surface codes and lattice surgery
- Semester 2B: Fault Tolerance & Hardware review
  - Magic states and distillation
  - Hardware platforms
  - Advanced algorithms
- Integration and connections
- Practice qualifying exam questions

**Review Framework:**
```
QEC Fundamentals → Stabilizers → Topological → Surface Codes
         ↓
FT Protocols → Hardware → Algorithms → Research Frontiers
```

### Week 144: Year 3 Preview & Research Preparation (Days 1002-1008)

Transition to research phase and qualifying exam preparation.

**Core Topics:**
- Year 3 curriculum overview
- Qualifying exam format and expectations
- Research proposal development
- Literature review methodology
- Identifying research directions
- Building on Year 2 foundations
- Final capstone project

**Key Deliverables:**
- Research interest statement
- Literature review outline
- Mock qualifying exam performance
- Year 2 portfolio compilation

---

## Key Concepts

### QLDPC Code Comparison

| Property | Surface Code | Good QLDPC |
|----------|--------------|------------|
| Rate k/n | O(1/d²) | Θ(1) |
| Distance | O(√n) | Θ(n) |
| Overhead | O(d²) | O(1) |
| Locality | 2D local | Non-local |
| Decoding | Efficient | More complex |

### Year 2 Knowledge Map

| Semester | Core Skills | Research Applications |
|----------|-------------|----------------------|
| 2A | Error correction theory | Code design |
| 2A | Stabilizer formalism | Code analysis |
| 2A | Topological codes | Anyonic QC |
| 2B | Fault tolerance | System design |
| 2B | Hardware knowledge | Experimental proposals |
| 2B | Algorithm design | Applications research |

### Research Frontier Areas

| Area | Key Challenge | Opportunity |
|------|---------------|-------------|
| QLDPC | Practical implementation | Overhead reduction |
| Logical qubits | Scaling | FT demonstrations |
| NISQ algorithms | Advantage proof | Near-term applications |
| Hardware | Coherence + scale | New platforms |

---

## Prerequisites

### From Months 31-35
- Complete fault-tolerant framework
- Hardware platform knowledge
- Advanced algorithm understanding
- Error mitigation techniques

### From Semester 2A
- Stabilizer codes mastery
- Surface code architecture
- Decoding algorithms

### Research Skills
- Literature reading proficiency
- Critical analysis
- Scientific writing

---

## Resources

### Primary References
- Panteleev & Kalachev, "Asymptotically Good QLDPC Codes" (2022)
- Leverrier & Zémor, "Quantum Tanner Codes" (2022)
- Annual Review of Quantum Computing (2025-2026)
- arXiv quantum-ph recent papers

### Key Papers
- Gottesman, "Fault-Tolerant Quantum Computation with Constant Overhead" (2014)
- Breuckmann & Eberhardt, "Quantum LDPC Codes" (2021)
- Various 2024-2026 experimental papers

### Online Resources
- [arXiv quantum-ph](https://arxiv.org/list/quant-ph/recent)
- [Quantum Information Processing Conference Proceedings](https://qipconference.org/)
- [Nature Physics Quantum Collection](https://www.nature.com/subjects/quantum-physics)

---

## Computational Tools

```python
# Month 36 capstone computational stack
import numpy as np
from scipy import linalg, sparse, optimize
import matplotlib.pyplot as plt
import networkx as nx

# Qiskit full suite
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.quantum_info import Operator, Statevector, StabilizerState
from qiskit_algorithms import VQE, QAOA

# Error correction tools
import stim
import pymatching
import ldpc  # For classical LDPC

# Research tools
import pandas as pd
from collections import defaultdict
```

---

## Capstone Project Options

### Option A: QLDPC Code Implementation
Implement a simple qLDPC code and compare with surface code.

### Option B: Algorithm Benchmarking
Compare VQE/QAOA performance across simulated hardware platforms.

### Option C: Literature Review
Comprehensive review of 2025-2026 quantum computing advances.

### Option D: Research Proposal
Develop a detailed research proposal for Year 3 project.

---

## Connections

### From Month 35
- Algorithm foundations → Research applications
- Variational methods → Capstone implementation
- QML → Future research directions

### To Year 3
- Comprehensive review → Qualifying exam
- Research proposal → Year 3 project
- QLDPC knowledge → Advanced research
- Portfolio → PhD candidacy preparation

---

## Summary

Month 36 brings Year 2 to a culmination with cutting-edge QLDPC codes, current research frontiers, comprehensive review, and Year 3 preparation. QLDPC codes represent the future of quantum error correction with their asymptotically optimal properties, offering constant overhead compared to the polynomial overhead of surface codes. The research frontiers survey ensures currency with the rapidly evolving field. The comprehensive review synthesizes all Year 2 knowledge, preparing for qualifying exams. The final week transitions to the research phase, establishing research interests and methodology for Year 3.

---

## Year 2 Completion Requirements

Upon completing Month 36, verify:

- [ ] All 336 days of Year 2 completed
- [ ] Semester 2A (QEC) mastery demonstrated
- [ ] Semester 2B (FT & Hardware) mastery demonstrated
- [ ] Capstone project completed
- [ ] Research proposal drafted
- [ ] Ready for Year 3 qualifying exam preparation

---

*"The development of practical fault-tolerant quantum computers is the central challenge of the field."*
— John Preskill

---

**Last Updated:** February 7, 2026
**Status:** 🟡 IN PROGRESS — 0/28 days complete (0%)
