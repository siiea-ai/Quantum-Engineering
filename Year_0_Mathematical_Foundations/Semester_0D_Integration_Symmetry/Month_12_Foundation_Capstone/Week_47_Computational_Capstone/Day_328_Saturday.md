# Day 328: Capstone Project — Documentation

## Overview

**Month 12, Week 47, Day 6 — Saturday**

Today you create comprehensive documentation: README, API documentation, theory summary, and user guide.

## Documentation Template

### README.md

```markdown
# Quantum Harmonic Oscillator Simulator

A comprehensive Python implementation of the quantum harmonic oscillator,
including analytical solutions, numerical methods, time evolution, and
phase space representations.

## Features

- Analytical eigenfunctions using Hermite polynomials
- Numerical eigenvalue solver using finite differences
- Time evolution via spectral decomposition
- Coherent state construction
- Wigner quasi-probability distribution

## Installation

```bash
pip install numpy scipy matplotlib
```

## Quick Start

```python
from qho import QuantumHarmonicOscillator

qho = QuantumHarmonicOscillator(mass=1, omega=1, hbar=1)

# Get eigenfunction
psi_0 = qho.eigenfunction(n=0, x=np.linspace(-5, 5, 100))

# Get energy
E_0 = qho.energy(n=0)  # Returns 0.5 (in natural units)
```

## Mathematical Background

### Hamiltonian
$$\hat{H} = \frac{\hat{p}^2}{2m} + \frac{1}{2}m\omega^2\hat{x}^2$$

### Energy Eigenvalues
$$E_n = \hbar\omega\left(n + \frac{1}{2}\right), \quad n = 0, 1, 2, ...$$

### Eigenfunctions
$$\psi_n(x) = \left(\frac{m\omega}{\pi\hbar}\right)^{1/4} \frac{1}{\sqrt{2^n n!}} H_n(\xi) e^{-\xi^2/2}$$

where $\xi = \sqrt{m\omega/\hbar} \cdot x$ and $H_n$ are Hermite polynomials.

## API Reference

### `QuantumHarmonicOscillator`

**Constructor**
- `mass`: Particle mass (default: 1)
- `omega`: Angular frequency (default: 1)
- `hbar`: Reduced Planck constant (default: 1)

**Methods**
- `energy(n)`: Return nth energy eigenvalue
- `eigenfunction(n, x)`: Compute nth eigenfunction at positions x
- `time_evolve(psi_0, x, t)`: Evolve initial state to time t
- `coherent_state(alpha, x)`: Construct coherent state |α⟩
- `wigner_function(psi, x, p)`: Compute Wigner function

## Examples

See `examples/` directory for:
- Basic eigenfunction plotting
- Time evolution animation
- Coherent state dynamics
- Wigner function visualization

## Testing

```bash
python -m pytest tests/
```

## License

MIT License

## Author

Year 0 Capstone Project
```

---

## Today's Checklist

- [ ] README complete
- [ ] API documented
- [ ] Theory summary written
- [ ] Examples provided
- [ ] Installation instructions clear

---

## Preview: Day 329

Tomorrow: **Project Presentation** — final presentation of your capstone.
