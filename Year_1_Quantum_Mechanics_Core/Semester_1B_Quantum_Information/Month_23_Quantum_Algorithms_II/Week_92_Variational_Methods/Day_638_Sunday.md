# Day 638: NISQ Algorithms Introduction

## Overview
**Day 638** | Week 92, Day 1 | Year 1, Month 23 | Variational Methods

Today we introduce NISQ (Noisy Intermediate-Scale Quantum) algorithms, which are designed to work on near-term quantum hardware with limited qubits and imperfect gates.

---

## Learning Objectives

1. Understand the NISQ era and its constraints
2. Define hybrid classical-quantum algorithms
3. Compare NISQ vs fault-tolerant approaches
4. Identify suitable NISQ applications
5. Understand the variational paradigm
6. Appreciate practical limitations

---

## Core Content

### The NISQ Era

**NISQ (Preskill 2018):** Noisy Intermediate-Scale Quantum
- 50-1000 qubits
- Limited coherence times
- No error correction
- Gate errors ~0.1-1%

### Fault-Tolerant vs NISQ

| Aspect | Fault-Tolerant | NISQ |
|--------|---------------|------|
| Qubits | Millions (logical) | 50-1000 (physical) |
| Error rates | $< 10^{-10}$ | $10^{-2} - 10^{-3}$ |
| Circuit depth | Arbitrary | ~100-1000 gates |
| Algorithms | Shor, Grover full | VQE, QAOA |
| Timeline | 10+ years? | Now |

### Hybrid Classical-Quantum Computing

**Concept:** Use quantum computer for tasks it does well (state preparation, measurement), classical for optimization.

```
Classical Computer ←→ Quantum Computer
    (Optimizer)         (State prep)
         ↓                    ↓
    New params θ      Measure ⟨H⟩_θ
         ↑                    ↓
    ←←←←←←←←←←←←←←←←←←←←←←←←←
           Energy E(θ)
```

### Variational Quantum Algorithms

**Key idea:** Parameterized quantum circuit $U(\theta)$ optimized classically.

$$|\psi(\theta)\rangle = U(\theta)|0\rangle^{\otimes n}$$

**Optimization problem:**
$$\theta^* = \arg\min_\theta C(\theta)$$

where $C(\theta)$ is a cost function computed on quantum device.

### Why Variational?

1. **Shallow circuits:** Reduce decoherence effects
2. **Flexibility:** Ansatz adapted to hardware
3. **Classical optimization:** Mature algorithms available
4. **Noise resilience:** Some tolerance to errors

### Applications of NISQ Algorithms

| Application | Algorithm | Status |
|-------------|-----------|--------|
| Chemistry | VQE | Demonstrated |
| Optimization | QAOA | Active research |
| Machine Learning | VQC | Promising |
| Simulation | VQS | Emerging |

### Challenges

1. **Barren plateaus:** Vanishing gradients
2. **Local minima:** Optimization landscape
3. **Noise:** Error accumulation
4. **Classical simulation:** Can outperform small NISQ

---

## Worked Examples

### Example 1: NISQ Constraints
For a device with T1 = 100μs and gate time 50ns, maximum circuit depth?

**Solution:**
Coherence-limited depth: $100\mu s / 50ns = 2000$ gates max

With errors at 0.5%/gate, for 90% fidelity:
$(0.995)^d = 0.9 \Rightarrow d \approx 21$ gates

### Example 2: Hybrid Loop
Describe one iteration of VQE.

**Solution:**
1. Prepare $|\psi(\theta)\rangle$ on quantum computer
2. Measure $\langle H \rangle$ via Pauli decomposition
3. Return energy to classical computer
4. Classical optimizer updates $\theta$
5. Repeat until converged

---

## Computational Lab

```python
"""Day 638: NISQ Algorithms Introduction"""
import numpy as np
import matplotlib.pyplot as plt

def nisq_fidelity(n_gates, error_per_gate=0.005):
    """Expected circuit fidelity."""
    return (1 - error_per_gate) ** n_gates

def plot_nisq_constraints():
    """Visualize NISQ limitations."""
    gates = np.arange(1, 1001)
    errors = [0.001, 0.005, 0.01, 0.02]

    plt.figure(figsize=(10, 6))
    for err in errors:
        fidelity = nisq_fidelity(gates, err)
        plt.semilogy(gates, fidelity, label=f'Error = {err*100:.1f}%')

    plt.axhline(y=0.5, color='red', linestyle='--', label='50% fidelity')
    plt.xlabel('Number of Gates', fontsize=12)
    plt.ylabel('Expected Fidelity', fontsize=12)
    plt.title('NISQ Circuit Fidelity vs Depth', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('nisq_fidelity.png', dpi=150)
    plt.show()

plot_nisq_constraints()

# NISQ algorithm summary
print("\nNISQ Algorithm Landscape:")
print("="*50)
algorithms = [
    ("VQE", "Chemistry", "Ground state energies"),
    ("QAOA", "Optimization", "MaxCut, SAT"),
    ("VQC", "Machine Learning", "Classification"),
    ("VQS", "Simulation", "Dynamics"),
]
for name, domain, application in algorithms:
    print(f"{name:>10}: {domain:<15} - {application}")
```

---

## Summary

### Key Concepts

| Term | Definition |
|------|------------|
| NISQ | Noisy Intermediate-Scale Quantum |
| Hybrid | Classical + Quantum computing |
| Variational | Parameterized circuit optimization |
| Ansatz | Parameterized circuit structure |

### Key Takeaways

1. **NISQ devices** have limited qubits and high error rates
2. **Hybrid algorithms** combine classical and quantum
3. **Variational methods** use parameterized circuits
4. **Shallow circuits** are essential for NISQ
5. **Applications** in chemistry, optimization, ML
6. **Challenges** include noise and trainability

---

## Daily Checklist

- [ ] I understand NISQ constraints
- [ ] I can explain hybrid computing
- [ ] I know the variational paradigm
- [ ] I understand current limitations
- [ ] I ran the computational lab

---

*Next: Day 639 — VQE Basics*
