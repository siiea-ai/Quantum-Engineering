# Week 71: Many-Body Systems

## Overview

**Week 71** | Days 491-497 | Helium and Multi-Electron Atoms

This week we apply the formalism of identical particles to real atomic systems. Starting with the helium atom—the simplest multi-electron system—we develop perturbation and variational methods, then extend to multi-electron atoms and the Hartree-Fock approximation.

---

## Daily Schedule

| Day | Topic | Focus |
|-----|-------|-------|
| **491** | Helium Atom Setup | Two-electron Hamiltonian, electron-electron repulsion |
| **492** | Perturbation Approach | First-order energy correction, comparison to experiment |
| **493** | Variational Method for Helium | Trial wave function, Z_eff optimization |
| **494** | Exchange and Spin States | Singlet/triplet, parahelium vs orthohelium |
| **495** | Multi-Electron Atoms | Central field, aufbau principle, periodic table |
| **496** | Hartree-Fock Introduction | Self-consistent field, exchange interaction |
| **497** | Week Review | Comprehensive summary and problem set |

---

## Key Concepts

### The Helium Atom Challenge

The helium atom (Z = 2, 2 electrons) is the simplest system where:
- Electron-electron repulsion cannot be ignored
- Exchange effects manifest physically
- Exact analytical solution is impossible

This makes it the ideal testing ground for approximation methods.

### Why Helium Matters

1. **Benchmark system** for approximation methods
2. **Demonstrates exchange splitting** between singlet and triplet states
3. **Foundation for understanding** the periodic table
4. **Gateway to** computational quantum chemistry
5. **Key target** for quantum computing (VQE algorithms)

### Methods Covered

| Method | Accuracy | Complexity |
|--------|----------|------------|
| Independent particle | ~5 eV error | Simple |
| First-order perturbation | ~1.6 eV error | Moderate |
| Variational (Z_eff) | ~0.3 eV error | Moderate |
| Hartree-Fock | ~0.04 eV error | Complex |
| Configuration interaction | ~0.001 eV | Very complex |

---

## Key Formulas

### Helium Hamiltonian
$$\hat{H} = -\frac{\hbar^2}{2m}(\nabla_1^2 + \nabla_2^2) - \frac{Ze^2}{r_1} - \frac{Ze^2}{r_2} + \frac{e^2}{r_{12}}$$

### Independent Particle Approximation
$$E_0^{(0)} = 2 \times \left(-\frac{Z^2 e^2}{2a_0}\right) = -8 \times 13.6 \text{ eV} = -108.8 \text{ eV}$$

### First-Order Correction
$$E^{(1)} = \left\langle \frac{e^2}{r_{12}} \right\rangle = \frac{5}{4}Z E_1 = +34 \text{ eV}$$

### Variational with Effective Z
$$E(Z_{\text{eff}}) = \left(Z_{\text{eff}}^2 - 2Z \cdot Z_{\text{eff}} + \frac{5}{8}Z_{\text{eff}}\right)E_1$$
$$Z_{\text{eff}}^{\text{opt}} = Z - \frac{5}{16} = 1.6875$$

### Exchange Integral
$$J = \int d^3r_1 d^3r_2 \, \phi_a^*(r_1)\phi_b^*(r_2)\frac{e^2}{r_{12}}\phi_a(r_2)\phi_b(r_1)$$

---

## Quantum Computing Connection

### VQE for Helium

The Variational Quantum Eigensolver (VQE) is the quantum analog of variational methods:

1. **Encode** the helium Hamiltonian in qubits (Jordan-Wigner)
2. **Prepare** a parameterized trial state (ansatz)
3. **Measure** energy expectation value
4. **Optimize** parameters classically
5. **Iterate** to convergence

Helium with minimal basis: 4 spin-orbitals = 4 qubits

### Why VQE for Chemistry?

- Chemical accuracy requires ~10^-3 Hartree precision
- Systematic improvement possible with larger ansatze
- Near-term quantum advantage target
- Industrial interest (drug discovery, materials)

---

## Prerequisites

From previous weeks:
- Identical particles and exchange symmetry (Week 69)
- Second quantization formalism (Week 70)
- Perturbation theory (Week 65)
- Variational method (Week 66)
- Hydrogen atom solutions (Week 59-60)

---

## References

- Griffiths & Schroeter, Ch. 7.2-7.3
- Sakurai & Napolitano, Ch. 8
- Szabo & Ostlund, *Modern Quantum Chemistry*, Ch. 2-3
- McWeeny, *Methods of Molecular Quantum Mechanics*
