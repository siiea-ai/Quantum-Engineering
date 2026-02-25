# Week 56: Tunneling & Barriers

## Month 14, Week 4 | Days 386-392

### Overview

This week explores one of the most remarkable phenomena in quantum mechanics: **quantum tunneling**. We study how particles can penetrate classically forbidden regions, leading to exponentially small but non-zero transmission probabilities. This purely quantum effect has profound applications ranging from nuclear physics (alpha decay) to modern technology (scanning tunneling microscopy, tunnel diodes, and superconducting qubits).

### Weekly Schedule

| Day | Topic | Focus Areas |
|-----|-------|-------------|
| **386** | Step Potential | Partial reflection/transmission, evanescent waves, current conservation |
| **387** | Rectangular Barrier | Transfer matrix method, transmission/reflection coefficients |
| **388** | Tunneling Probability | WKB approximation, Gamow factor, decay rates |
| **389** | Alpha Decay (Gamow Model) | Nuclear tunneling, half-life predictions, Geiger-Nuttall law |
| **390** | Scanning Tunneling Microscope | Exponential sensitivity, atomic resolution, spectroscopy |
| **391** | Tunnel Diodes & Josephson Effect | Negative resistance, superconducting qubits, quantum computing |
| **392** | Month 14 Capstone | Comprehensive review, integration, assessment |

### Learning Objectives

By the end of this week, you will be able to:

1. **Solve** the Schrodinger equation for step potentials and rectangular barriers
2. **Derive** transmission and reflection coefficients using boundary conditions
3. **Apply** the transfer matrix method for multi-region potentials
4. **Calculate** tunneling probabilities for arbitrary barrier shapes using WKB
5. **Explain** alpha decay using the Gamow model and predict half-lives
6. **Understand** the operating principles of STM and tunnel diodes
7. **Connect** tunneling phenomena to superconducting qubit design

### Key Formulas

#### Step Potential (E < V_0)
$$k_1 = \frac{\sqrt{2mE}}{\hbar}, \quad \kappa = \frac{\sqrt{2m(V_0 - E)}}{\hbar}$$

$$R = \left|\frac{k_1 - i\kappa}{k_1 + i\kappa}\right|^2 = 1 \quad \text{(total reflection)}$$

#### Rectangular Barrier Transmission
$$\boxed{T = \frac{1}{1 + \frac{V_0^2 \sinh^2(\kappa L)}{4E(V_0 - E)}}}$$

For thick barriers ($\kappa L \gg 1$):
$$T \approx 16\frac{E(V_0-E)}{V_0^2}e^{-2\kappa L}$$

#### WKB Tunneling Probability
$$\boxed{T \approx e^{-2\gamma}, \quad \gamma = \frac{1}{\hbar}\int_{x_1}^{x_2}\sqrt{2m(V(x)-E)}\,dx}$$

#### Gamow Factor (Alpha Decay)
$$\lambda = f \cdot e^{-2G}$$

$$G = \frac{1}{\hbar}\int_R^{r_c}\sqrt{2m_\alpha\left(\frac{Z_1Z_2e^2}{4\pi\epsilon_0 r} - E\right)}dr$$

#### STM Tunneling Current
$$\boxed{I \propto e^{-2\kappa d}, \quad \kappa = \frac{\sqrt{2m\phi}}{\hbar}}$$

where $\phi$ is the work function and $d$ is tip-sample distance.

### Prerequisites

- Schrodinger equation in 1D (Week 50)
- Probability current density (Week 51)
- Bound state problems: wells and oscillator (Weeks 54-55)
- Complex exponentials and hyperbolic functions

### Textbook References

| Topic | Shankar | Griffiths | Cohen-Tannoudji |
|-------|---------|-----------|-----------------|
| Step Potential | Ch. 5.3 | Ch. 2.5 | Ch. I.D |
| Rectangular Barrier | Ch. 5.4 | Ch. 2.5 | Ch. I.D |
| WKB Approximation | Ch. 16 | Ch. 8.1-8.2 | Complement MIII |
| Alpha Decay | Ch. 5.4 | Ch. 8.2 | - |

### Mathematical Tools

1. **Hyperbolic functions**: $\sinh(x) = (e^x - e^{-x})/2$, $\cosh(x) = (e^x + e^{-x})/2$
2. **Transfer matrices**: 2x2 matrices relating wave amplitudes across regions
3. **Contour integration**: For evaluating Gamow integrals
4. **Asymptotic expansions**: Large $\kappa L$ approximations

### Quantum Computing Connections

- **Superconducting qubits**: Josephson junctions exploit Cooper pair tunneling
- **Quantum annealing**: Tunneling enables escape from local minima
- **Readout mechanisms**: Qubit state detection via tunneling currents
- **Coherent control**: Tunnel coupling between quantum dots

### Computational Skills

This week emphasizes:
- Numerical solution of transcendental equations
- Transfer matrix implementation
- WKB integral evaluation
- Visualization of probability densities in forbidden regions

### Weekly Project

**Quantum Tunneling Simulator**: Build a comprehensive tool that:
1. Solves arbitrary 1D barrier problems numerically
2. Implements transfer matrix method
3. Compares exact results with WKB approximation
4. Simulates time-dependent wave packet tunneling

### Assessment Criteria

- [ ] Correctly match boundary conditions across potential discontinuities
- [ ] Derive and apply transmission coefficient formulas
- [ ] Explain physical meaning of evanescent waves
- [ ] Calculate alpha decay half-lives within order of magnitude
- [ ] Describe STM operation and resolution limits
- [ ] Connect Josephson tunneling to qubit applications

### Study Tips

1. **Visualize**: Always sketch the potential and wave function
2. **Check limits**: Verify T → 0 as L → ∞ and T → 1 as V_0 → 0
3. **Compare classical**: Contrast quantum behavior with classical expectations
4. **Practice matching**: Boundary conditions are the heart of barrier problems

### Historical Context

- **1928**: Gamow, Condon & Gurney independently explain alpha decay
- **1957**: Esaki discovers tunnel diode (Nobel Prize 1973)
- **1962**: Josephson predicts superconducting tunneling (Nobel Prize 1973)
- **1981**: Binnig & Rohrer invent STM (Nobel Prize 1986)

---

*Week 56 completes Month 14: One-Dimensional Quantum Mechanics. Next week begins Month 15: Angular Momentum and Spin.*
