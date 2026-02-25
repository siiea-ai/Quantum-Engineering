# Month 23: Quantum Algorithms II

## Overview
**Days 617-644** | Weeks 89-92 | Year 1, Semester 1B

This month builds on the algorithmic foundations from Month 22, covering Grover's search algorithm, amplitude amplification techniques, quantum walks, and variational quantum algorithms for NISQ devices.

---

## Learning Objectives

By the end of this month, you will be able to:

1. **Implement Grover's algorithm** and analyze its quadratic speedup
2. **Generalize amplitude amplification** to arbitrary initial states
3. **Design quantum walk algorithms** on graphs
4. **Construct variational circuits** for optimization problems
5. **Analyze NISQ algorithm limitations** including barren plateaus
6. **Apply quantum algorithms** to practical problems

---

## Month Structure

### Week 89: Grover's Search (Days 617-623)
| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 617 | Unstructured Search Problem | Classical vs quantum bounds, oracle model |
| 618 | Grover Oracle | Phase oracle construction, marking states |
| 619 | Diffusion Operator | Reflection about mean, Grover operator |
| 620 | Amplitude Amplification Geometry | Geometric interpretation, rotation angle |
| 621 | Optimal Iteration Count | O(sqrt(N)) complexity, overshooting |
| 622 | Multiple Solutions Case | k solutions, modified iteration count |
| 623 | Week Review | Problem solving, synthesis |

### Week 90: Amplitude Amplification (Days 624-630)
| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 624 | Generalized Amplitude Amplification | Arbitrary initial states, amplification theorem |
| 625 | Amplitude Estimation | Phase estimation connection, precision |
| 626 | Fixed-Point Amplification | Avoiding overshooting, oblivious operators |
| 627 | Oblivious Amplification | Unknown target amplitude, robust methods |
| 628 | Quantum Counting | Estimating solution count, precision analysis |
| 629 | Applications | Search applications, SAT solving |
| 630 | Week Review | Integration and assessment |

### Week 91: Quantum Walks (Days 631-637)
| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 631 | Classical Random Walks Review | Markov chains, hitting times, mixing |
| 632 | Discrete-Time Quantum Walks | Coined walks, superposition of directions |
| 633 | Coin and Shift Operators | Hadamard coin, conditional shift |
| 634 | Continuous-Time Quantum Walks | Hamiltonian formulation, adjacency matrix |
| 635 | Quantum Walk Search | Spatial search, Childs algorithm |
| 636 | Graph Problems | Element distinctness, triangle finding |
| 637 | Week Review | Synthesis and applications |

### Week 92: Variational Methods (Days 638-644)
| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 638 | NISQ Algorithms Introduction | Near-term devices, hybrid algorithms |
| 639 | VQE Basics | Variational principle, ansatz design |
| 640 | QAOA Formulation | Combinatorial optimization, mixer/cost |
| 641 | Parameterized Circuits | Hardware-efficient ansatze, expressibility |
| 642 | Optimization Landscapes | Gradient computation, parameter shift rule |
| 643 | Barren Plateaus | Vanishing gradients, trainability issues |
| 644 | Month Review | Comprehensive assessment |

---

## Key Formulas

### Grover's Algorithm
$$G = (2|\psi\rangle\langle\psi| - I) \cdot O_f$$
$$\text{Iterations: } k \approx \frac{\pi}{4}\sqrt{\frac{N}{M}}$$
$$P_{success} = \sin^2\left((2k+1)\theta\right), \quad \sin\theta = \sqrt{M/N}$$

### Amplitude Amplification
$$Q = A S_0 A^{-1} S_\chi$$
$$|s\rangle = \sin\theta|good\rangle + \cos\theta|bad\rangle$$

### Quantum Walks
$$U = S \cdot (C \otimes I_p) \quad \text{(discrete-time)}$$
$$U(t) = e^{-iHt} \quad \text{(continuous-time)}$$

### Variational Methods
$$E(\vec{\theta}) = \langle\psi(\vec{\theta})|H|\psi(\vec{\theta})\rangle$$
$$\partial_\theta E = \frac{1}{2}\left[E(\theta + \pi/2) - E(\theta - \pi/2)\right]$$

---

## Primary References

### Textbooks
- Nielsen & Chuang, Chapters 5-6 (quantum algorithms)
- Preskill Lecture Notes, Chapter 6 (Grover, quantum walks)
- Childs, "Lecture Notes on Quantum Algorithms"

### Key Papers
- Grover (1996): "A Fast Quantum Mechanical Algorithm for Database Search"
- Brassard et al. (2002): "Quantum Amplitude Amplification and Estimation"
- Childs et al. (2003): "Exponential Algorithmic Speedup by Quantum Walk"
- Peruzzo et al. (2014): "A variational eigenvalue solver on a photonic quantum processor" (VQE)
- Farhi et al. (2014): "A Quantum Approximate Optimization Algorithm" (QAOA)
- McClean et al. (2018): "Barren Plateaus in Quantum Neural Network Training Landscapes"

---

## Computational Tools

```python
# Core imports for Month 23
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Grover oracle for marked state
def grover_oracle(n, marked_states):
    """Create phase oracle marking specified states"""
    N = 2**n
    oracle = np.eye(N)
    for m in marked_states:
        oracle[m, m] = -1
    return oracle

# Diffusion operator
def diffusion_operator(n):
    """Create Grover diffusion operator"""
    N = 2**n
    psi = np.ones(N) / np.sqrt(N)
    return 2 * np.outer(psi, psi) - np.eye(N)

# Grover iteration
def grover_iteration(n, marked_states):
    """Single Grover iteration G = D @ O"""
    O = grover_oracle(n, marked_states)
    D = diffusion_operator(n)
    return D @ O

# VQE energy evaluation
def vqe_energy(params, hamiltonian, ansatz_func):
    """Evaluate variational energy"""
    state = ansatz_func(params)
    return np.real(state.conj() @ hamiltonian @ state)

# Parameter shift rule gradient
def parameter_shift_gradient(params, hamiltonian, ansatz_func, idx):
    """Compute gradient using parameter shift rule"""
    params_plus = params.copy()
    params_minus = params.copy()
    params_plus[idx] += np.pi/2
    params_minus[idx] -= np.pi/2
    return 0.5 * (vqe_energy(params_plus, hamiltonian, ansatz_func)
                - vqe_energy(params_minus, hamiltonian, ansatz_func))
```

---

## Progress Tracking

| Week | Status | Days Complete |
|------|--------|---------------|
| Week 89: Grover's Search | ✅ Complete | 7/7 |
| Week 90: Amplitude Amplification | ✅ Complete | 7/7 |
| Week 91: Quantum Walks | ✅ Complete | 7/7 |
| Week 92: Variational Methods | ✅ Complete | 7/7 |

**Month Progress:** 28/28 days (100%) ✅

---

## Prerequisites

From Month 22:
- Quantum Fourier Transform
- Phase estimation algorithm
- Oracle model of computation
- Basic circuit construction

---

*Next: Month 24 — Quantum Channels and Error Introduction*
