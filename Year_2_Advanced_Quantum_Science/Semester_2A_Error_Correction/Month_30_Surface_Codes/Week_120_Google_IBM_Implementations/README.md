# Week 120: Google/IBM Implementations

## Semester 2A: Quantum Error Correction | Month 30: Surface Codes | Week 120

### Week Overview

This week marks the culmination of Semester 2A, focusing on real-world implementations of surface codes by leading quantum hardware companies. We examine Google's groundbreaking Willow processor (2024) achieving below-threshold operation, IBM's heavy-hex architecture, and implementations on trapped-ion and neutral atom platforms. The week concludes with a comprehensive capstone project and semester synthesis.

### Learning Objectives for the Week

By the end of Week 120, you will be able to:

1. **Analyze Google Willow Architecture** - Understand the 105-qubit Sycamore-based design achieving below-threshold operation
2. **Interpret Below-Threshold Results** - Quantify the λ = 2.14 error suppression factor and its implications
3. **Compare IBM Heavy-Hex Design** - Evaluate 3-connectivity architectures with flag-based fault tolerance
4. **Assess Alternative Platforms** - Analyze trapped-ion and neutral atom surface code implementations
5. **Project Scaling Requirements** - Calculate resources needed for 1000+ logical qubits
6. **Design Complete Systems** - Integrate all Semester 2A knowledge into practical quantum computer designs

### Week Schedule

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| **Day 834 (Mon)** | Google Willow Architecture | 105 qubits, Sycamore layout, transmon design, coupler engineering |
| **Day 835 (Tue)** | Below-Threshold Operation Analysis | λ = 2.14 suppression, 0.143% error/cycle, 2.4× lifetime improvement |
| **Day 836 (Wed)** | IBM Heavy-Hex Surface Codes | 3-connectivity, flag qubits, Heron processor implementation |
| **Day 837 (Thu)** | Trapped-Ion & Neutral Atom | IonQ, Quantinuum, QuEra approaches to surface codes |
| **Day 838 (Fri)** | Scaling Roadmaps | 1000+ logical qubits, million qubit systems, timeline projections |
| **Day 839 (Sat)** | Semester 2A Capstone Project | Design complete surface code quantum computer |
| **Day 840 (Sun)** | Month 30 & Semester 2A Synthesis | Comprehensive review of Months 25-30 |

### Prerequisites

- **Week 117**: Surface Code Foundations (stabilizers, logical operators)
- **Week 118**: Decoding Algorithms (MWPM, Union-Find, neural decoders)
- **Week 119**: Threshold Theory and Analysis
- **Months 25-29**: Complete error correction theory foundation

### Key Results to Master

#### Google Willow (2024) Breakthrough

$$\boxed{\lambda = \frac{p_L(d)}{p_L(d+2)} = 2.14 \pm 0.02}$$

- **Distance 3**: $p_L = 3.028\% \pm 0.023\%$
- **Distance 5**: $p_L = 0.306\% \pm 0.005\%$
- **Distance 7**: $p_L = 0.143\% \pm 0.003\%$

Logical lifetime exceeds best physical qubit: $T_{\text{logical}} = 2.4 \times T_{\text{best physical}}$

#### Scaling Projections

For practical quantum advantage:
- **Near-term (2025-2027)**: 100-1000 physical qubits, 1-10 logical qubits
- **Medium-term (2027-2030)**: 10,000+ physical qubits, 100+ logical qubits
- **Long-term (2030+)**: Million+ physical qubits, 1000+ logical qubits

### Resources and References

#### Primary Sources
1. Google Quantum AI, "Quantum error correction below the surface code threshold," *Nature* (2024)
2. IBM Quantum, "Heavy-hex lattice documentation," Qiskit (2024)
3. Acharya et al., "Suppressing quantum errors by scaling a surface code logical qubit," *Nature* (2023)

#### Technical Documentation
- Google Quantum AI Blog: ai.google/quantum
- IBM Quantum Documentation: quantum-computing.ibm.com
- IonQ Technical Papers: ionq.com/resources
- QuEra Computing Publications: quera.com/papers

### Mathematical Prerequisites

Ensure familiarity with:
- Stabilizer formalism and syndrome extraction
- Threshold theorem: $p_L \sim (p/p_{\text{th}})^{\lfloor(d+1)/2\rfloor}$
- Decoding algorithms (MWPM, Union-Find)
- Fault-tolerant gate implementations

### Computational Tools

This week uses:
```python
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy.stats import linregress
# Optional: stim, pymatching for advanced simulations
```

### Assessment Components

| Component | Weight | Description |
|-----------|--------|-------------|
| Daily Problems | 30% | Conceptual and computational exercises |
| Lab Implementations | 30% | Python simulations and analysis |
| Capstone Project | 40% | Complete quantum computer design |

### Week 120 Roadmap

```
Day 834: Google Willow Architecture
    ├── Transmon qubit design
    ├── Tunable coupler engineering
    └── Surface code layout

Day 835: Below-Threshold Analysis
    ├── Error suppression measurement
    ├── Statistical analysis methods
    └── Implications for scaling

Day 836: IBM Heavy-Hex
    ├── 3-connectivity constraints
    ├── Flag qubit protocols
    └── Comparison with Google

Day 837: Alternative Platforms
    ├── Trapped-ion implementations
    ├── Neutral atom arrays
    └── Platform comparison

Day 838: Scaling Roadmaps
    ├── Resource estimation
    ├── Technology roadmaps
    └── Timeline projections

Day 839: Capstone Project
    ├── Algorithm selection
    ├── Full system design
    └── Error budget analysis

Day 840: Semester Synthesis
    ├── Months 25-30 review
    ├── Integration exercises
    └── Forward outlook
```

---

*This week represents the culmination of six months of quantum error correction study, bridging theoretical foundations with state-of-the-art experimental implementations.*
