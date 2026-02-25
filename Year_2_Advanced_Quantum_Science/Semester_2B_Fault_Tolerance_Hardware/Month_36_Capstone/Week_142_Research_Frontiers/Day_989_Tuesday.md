# Day 989: Quantum Advantage & Verification

## Month 36, Week 142, Day 2 | Research Frontiers

### Schedule Overview (7 hours)

| Block | Time | Focus |
|-------|------|-------|
| Morning | 2.5 hrs | Theory: Quantum Advantage Definitions & Claims |
| Afternoon | 2.5 hrs | Critical Analysis: Classical Competition |
| Evening | 2 hrs | Lab: Verification Methods & Simulation |

---

## Learning Objectives

By the end of this day, you should be able to:

1. **Distinguish** between quantum supremacy, advantage, and utility
2. **Critically evaluate** computational speedup claims (including the $10^{25}$ years claim)
3. **Analyze** the ongoing classical algorithm improvements
4. **Apply** verification methods for quantum computational claims
5. **Assess** what constitutes "useful" quantum advantage
6. **Identify** the most promising near-term applications

---

## Core Content

### 1. Taxonomy of Quantum Advantage

#### Definitions and Distinctions

The field has evolved through several concepts, each with different implications:

**Quantum Supremacy (Preskill, 2012)**
$$\exists \text{ task } T: \text{QC}(T) = O(f(n)), \text{CC}(T) = \omega(g(n)) \text{ where } f \ll g$$

A quantum computer performs *some* task faster than any classical computer. This task need not be useful.

**Quantum Advantage**
$$\exists \text{ useful task } T: \text{QC}(T) \ll \text{CC}(T)$$

A quantum computer provides practical speedup for a task with real applications.

**Quantum Utility (IBM terminology, 2023)**
$$\exists \text{ task } T: \text{QC}(T) \text{ provides value beyond classical simulation cost}$$

The quantum computer produces results that would be too expensive to obtain classically, even if not strictly impossible.

| Concept | Computational Claim | Practical Value | Examples |
|---------|---------------------|-----------------|----------|
| Supremacy | Exponential speedup | Not required | Random circuit sampling |
| Advantage | Significant speedup | Required | Factoring, optimization |
| Utility | Cost-effective | Required | Materials simulation |

### 2. Random Circuit Sampling: The Supremacy Battleground

#### Google's Claims (2019 and 2024)

**Sycamore (2019):**
- Task: Sample from random quantum circuit output distribution
- Quantum time: 200 seconds
- Classical estimate: 10,000 years (initially)

**Willow (2024):**
- Task: Random circuit sampling with 105 qubits
- Quantum time: < 5 minutes
- Classical estimate: $10^{25}$ years (Google's claim)

#### The Sampling Problem

For a random quantum circuit $U$, the output distribution is:

$$p(x) = |\langle x | U | 0^n \rangle|^2$$

Sampling from this distribution is believed to be classically hard, related to the conjecture:

$$\boxed{\text{Approximate sampling from } p(x) \notin \text{BPP}}$$

This is connected to the **anticoncentration** property and **Porter-Thomas** distribution.

### 3. Classical Competition: The Moving Target

#### Historical Pattern

Every quantum supremacy claim has been followed by classical algorithm improvements:

| Year | Quantum Claim | Classical Response | Final Gap |
|------|---------------|-------------------|-----------|
| 2019 | Google: 10,000 years | IBM: 2.5 days (with storage) | ~1000× |
| 2019 | Google: 10,000 years | Alibaba, others: hours-days | ~100× |
| 2021 | USTC photonics | GPU clusters: minutes | ~10× |
| 2024 | Google: $10^{25}$ years | TBD (ongoing challenge) | ??? |

#### Classical Simulation Methods

**1. Tensor Network Contraction**

Random circuits can be simulated by contracting tensor networks:

$$\langle x | U | 0^n \rangle = \sum_{\text{paths}} \prod_{\text{gates}} T_{\text{gate}}$$

The complexity depends on the **treewidth** of the contraction:

$$\text{Time} \sim \exp(O(\text{treewidth}))$$

**2. Spoofing Attacks**

Rather than exact simulation, can we produce samples that pass statistical tests?

The **linear cross-entropy benchmark (XEB)** measures:

$$\boxed{F_{\text{XEB}} = 2^n \langle p(x) \rangle_{\text{samples}} - 1}$$

where $F_{\text{XEB}} = 1$ for perfect quantum sampling and $F_{\text{XEB}} = 0$ for uniform random.

**Spoofing strategies:**
- Partial simulation of high-probability outcomes
- Noise-aware truncated simulation
- Hybrid classical-quantum approximations

**3. GPU and TPU Clusters**

Modern classical computing advances:
- NVIDIA H100 clusters: ~1000 PFLOPS aggregate
- Google TPU pods: Optimized for tensor operations
- Distributed memory techniques

#### The $10^{25}$ Years Claim: Critical Analysis

Google's Willow estimate assumptions:
- 105 qubits, depth ~30 cycles
- No known efficient classical algorithm
- Extrapolation from smaller simulations

**Critical questions:**

1. **Is the extrapolation valid?** Classical complexity may have phase transitions
2. **Are there unknown algorithms?** Tensor network methods continue improving
3. **What about approximate methods?** Lower fidelity classical samples might suffice
4. **Is this the right metric?** XEB may not capture computational hardness

**Historical lesson:** Classical estimates have been reduced by factors of $10^6$ within years.

### 4. Beyond Random Circuits: Useful Advantage

#### The Utility Gap

Random circuit sampling demonstrates computational separation but lacks applications:

$$\text{Supremacy} \neq \text{Utility}$$

The challenge: finding tasks that are both:
1. Exponentially faster on quantum computers
2. Useful for science or industry

#### Promising Near-Term Applications

**1. Quantum Chemistry (VQE and Beyond)**

Ground state energy estimation:

$$E_0 = \min_{\theta} \langle \psi(\theta) | H | \psi(\theta) \rangle$$

Current status:
- Molecules up to ~20 orbitals demonstrated
- Classical methods (DMRG, CCSD(T)) remain competitive
- Advantage requires error correction (est. 2028+)

**2. Optimization (QAOA)**

For combinatorial problems:

$$|\gamma, \beta\rangle = e^{-i\beta_p H_M} e^{-i\gamma_p H_C} \cdots e^{-i\beta_1 H_M} e^{-i\gamma_1 H_C} |+\rangle^n$$

Current status:
- No proven speedup over classical heuristics
- QAOA often matches but doesn't beat simulated annealing
- Theoretical advantages for specific problem classes

**3. Machine Learning (QML)**

Quantum kernels and variational classifiers:

$$k(x, x') = |\langle \phi(x) | \phi(x') \rangle|^2$$

Current status:
- Limited evidence of advantage for classical data
- Potential for quantum data (sensing, simulation)
- Dequantization results limit some claims

**4. Simulation of Quantum Systems**

Hamiltonian simulation for physics:

$$|\psi(t)\rangle = e^{-iHt} |\psi(0)\rangle$$

Current status:
- Most promising path to useful advantage
- IBM's 2023 utility demonstration (Ising model)
- Advantage claims disputed by classical methods

### 5. Verification: How Do We Know It Works?

#### The Verification Challenge

For problems where quantum computers claim exponential speedup:

$$\text{If classical computers can't solve it, how do they verify solutions?}$$

#### Verification Methods

**1. Cross-Entropy Benchmarking (XEB)**

Statistical test that samples are from the correct distribution:

$$F_{\text{XEB}} = \frac{\langle p(x) \rangle_{\text{samples}} - 1/2^n}{1/2^n}$$

Limitations:
- Requires knowing the ideal distribution (for small systems)
- Susceptible to spoofing attacks
- May not distinguish genuine quantum from clever classical

**2. Heavy Output Generation (HOG)**

Test whether samples favor high-probability outcomes:

$$\Pr[p(x) > \text{median}] > 0.5$$

For random circuits, quantum should achieve ~66% vs 50% uniform.

**3. Interactive Proofs**

Client-server protocols where verifier is classical:

$$\boxed{\text{QPIP} = \text{BQP}}$$

The Mahadev protocol (2018) enables classical verification of BQP:
- Uses cryptographic assumptions
- Polynomial overhead
- Not yet practical

**4. Trap-Based Verification**

Embed known computational traps:
- Include subcircuits with known outputs
- Verify these while measuring full computation
- Statistical confidence from trap success rate

### 6. The Road to Useful Quantum Advantage

#### Requirements for Practical Advantage

| Requirement | Current Status | Timeline |
|-------------|----------------|----------|
| Error rates < $10^{-10}$ | $10^{-3}$ achieved | 2030+ |
| 1000+ logical qubits | 1 logical qubit | 2028+ |
| Fast logical gates | ms timescale | 2026+ |
| Problem-specific algorithms | Active research | Ongoing |

#### Application-Specific Milestones

**Quantum Chemistry:**
- *Threshold:* Simulate FeMoCo (nitrogenase active site)
- *Requirement:* ~5000 logical qubits, $10^{-10}$ error rate
- *Classical competition:* DMRG, tensor network methods

**Cryptography:**
- *Threshold:* Factor 2048-bit RSA
- *Requirement:* ~4000 logical qubits, $10^{-12}$ error rate
- *Timeline:* 2035+ (optimistic)

**Optimization:**
- *Threshold:* Outperform best classical heuristics
- *Requirement:* Unknown (may not exist for general problems)
- *Status:* Theoretically uncertain

---

## Worked Examples

### Example 1: Evaluating XEB Claims

**Problem:** Google reports $F_{\text{XEB}} = 0.002$ for a 105-qubit, 30-cycle circuit. Analyze what this means.

**Solution:**

The XEB fidelity is defined as:
$$F_{\text{XEB}} = 2^n \langle p(x) \rangle - 1$$

For $F_{\text{XEB}} = 0.002$:
$$\langle p(x) \rangle = \frac{F_{\text{XEB}} + 1}{2^n} = \frac{1.002}{2^{105}}$$

This is only slightly above the uniform average of $1/2^{105}$.

**Interpretation:**
1. The samples are "slightly better" than random
2. The quantum computer is operating with ~0.2% fidelity
3. This is still exponentially hard to spoof classically (believed)

**Critical analysis:**
- Low fidelity makes verification harder
- Classical spoofing at this level might be possible
- The claim relies on extrapolation of classical hardness

**Signal significance:**
Expected samples needed to distinguish from uniform:
$$N \sim \frac{1}{F_{\text{XEB}}^2} = \frac{1}{(0.002)^2} = 250,000 \text{ samples}$$

### Example 2: Comparing Classical Simulation Costs

**Problem:** A tensor network simulation of a 100-qubit circuit costs $10^{20}$ FLOPS. Compare to quantum runtime.

**Solution:**

**Classical cost:**
- FLOPS required: $10^{20}$
- Best supercomputer (Frontier): $10^{18}$ FLOPS
- Time: $10^{20} / 10^{18} = 100$ seconds

**Quantum cost:**
- Circuit depth: 30 cycles
- Gate time: 50 ns
- Total time: $30 \times 50 \text{ ns} = 1.5$ μs

**Speedup:**
$$\frac{100 \text{ s}}{1.5 \text{ μs}} = 6.7 \times 10^7$$

**But consider:**
- Classical simulation gives exact amplitudes
- Quantum gives noisy samples
- Quantum requires many shots for statistics
- With 1 million shots: $1.5$ s quantum vs $100$ s classical

**Conclusion:** The "advantage" depends heavily on what you're measuring and the noise level.

### Example 3: Verification Protocol Analysis

**Problem:** Design a verification protocol for a 50-qubit random circuit sampling claim.

**Solution:**

**Protocol:**

1. **Classical pre-computation phase:**
   - Classically simulate the first 25 qubits (tractable)
   - Store partial amplitude information

2. **Trap insertion:**
   - In 10% of circuits, insert a known-output subcircuit
   - These "traps" have classically verifiable outcomes

3. **Sampling phase:**
   - Request 100,000 samples from the quantum computer
   - Include both real and trap circuits (randomly ordered)

4. **Verification:**
   - Check trap circuits: success rate should be >95%
   - Compute XEB for real circuits using partial simulation
   - Use extrapolation to estimate full-circuit XEB

5. **Statistical analysis:**
   - Confidence interval for XEB
   - Hypothesis test: $H_0$: samples are uniform; $H_1$: genuine quantum

**Expected outcomes:**
- Trap success: 97% (indicates functional quantum computer)
- XEB estimate: $0.002 \pm 0.0005$ (95% CI)
- p-value for quantum origin: < $10^{-10}$

---

## Practice Problems

### Problem 1: XEB Calculation (Direct Application)

A 20-qubit quantum circuit produces samples with measured probabilities. Given:
- 1000 samples collected
- Sum of ideal probabilities for sampled outputs: $\sum_i p(x_i) = 1.5$

Calculate:
a) The average ideal probability $\langle p(x) \rangle$
b) The XEB fidelity $F_{\text{XEB}}$
c) The expected value for uniform random sampling

### Problem 2: Classical Competition Analysis (Intermediate)

A 2024 paper claims to simulate Google's Sycamore circuit in 300 seconds using tensor networks.

a) What is the remaining "quantum speedup" compared to the original 200 second quantum time?
b) If classical algorithms improve by 10× per year, when will they match quantum?
c) How should Google respond to maintain a supremacy claim?

### Problem 3: Useful Advantage Criteria (Challenging)

Propose a framework for evaluating "useful quantum advantage" claims that includes:
a) Computational metrics (runtime, accuracy)
b) Practical value metrics (cost, accessibility)
c) Reproducibility requirements
d) Comparison methodology to classical baselines

Apply your framework to IBM's 2023 utility demonstration for Ising model simulation.

---

## Computational Lab: Verification and Analysis Tools

```python
"""
Day 989 Lab: Quantum Advantage Verification Tools
Analyzing claims and simulating verification protocols
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import factorial
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.figsize'] = (12, 8)

# ============================================================
# 1. XEB (Cross-Entropy Benchmarking) Simulation
# ============================================================

def simulate_xeb_experiment(n_qubits, n_samples, true_fidelity, noise_model='depolarizing'):
    """
    Simulate an XEB experiment with given parameters.

    For a random circuit, output probabilities follow Porter-Thomas distribution.
    Noisy quantum computer produces biased samples toward this distribution.
    """
    # Porter-Thomas: p(x) ~ Exponential(2^n)
    # Mean of PT distribution is 1/2^n
    hilbert_dim = 2**n_qubits

    # Generate "ideal" probabilities (Porter-Thomas distributed)
    ideal_probs = np.random.exponential(scale=1/hilbert_dim, size=hilbert_dim)
    ideal_probs /= ideal_probs.sum()  # Normalize

    # Simulate noisy sampling
    # With fidelity F, sample from: F * ideal_probs + (1-F) * uniform
    noisy_probs = true_fidelity * ideal_probs + (1 - true_fidelity) / hilbert_dim
    noisy_probs /= noisy_probs.sum()

    # Sample from noisy distribution
    samples = np.random.choice(hilbert_dim, size=n_samples, p=noisy_probs)

    # Compute XEB
    sampled_ideal_probs = ideal_probs[samples]
    avg_prob = np.mean(sampled_ideal_probs)
    xeb = hilbert_dim * avg_prob - 1

    return xeb, samples, ideal_probs

def xeb_statistical_test(xeb_measured, n_samples, n_qubits):
    """
    Perform statistical test for XEB significance.

    Under null hypothesis (uniform sampling), XEB ~ Normal(0, sigma)
    where sigma^2 = 2^n / n_samples for Porter-Thomas.
    """
    hilbert_dim = 2**n_qubits

    # Variance of XEB under null (uniform sampling)
    # For Porter-Thomas, var(p) = 1/2^(2n), so var(XEB) = 2^(2n) * var(p)/n = 2^n/n
    sigma = np.sqrt(hilbert_dim / n_samples)

    # Z-score
    z_score = xeb_measured / sigma

    # P-value (one-tailed, testing XEB > 0)
    p_value = 1 - stats.norm.cdf(z_score)

    return z_score, p_value, sigma

# Run XEB simulation
print("="*60)
print("XEB SIMULATION EXPERIMENT")
print("="*60)

n_qubits = 20
n_samples = 10000
true_fidelity = 0.01  # 1% fidelity (typical for large circuits)

xeb, samples, ideal_probs = simulate_xeb_experiment(n_qubits, n_samples, true_fidelity)
z_score, p_value, sigma = xeb_statistical_test(xeb, n_samples, n_qubits)

print(f"\nSimulation parameters:")
print(f"  Qubits: {n_qubits}")
print(f"  Samples: {n_samples}")
print(f"  True fidelity: {true_fidelity}")
print(f"\nResults:")
print(f"  Measured XEB: {xeb:.6f}")
print(f"  Expected XEB: {true_fidelity:.6f}")
print(f"  Z-score: {z_score:.2f}")
print(f"  P-value: {p_value:.2e}")
print(f"  Significant at 0.01 level: {p_value < 0.01}")

# ============================================================
# Figure 1: XEB Distribution Analysis
# ============================================================

fig1, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Porter-Thomas distribution
ax1 = axes[0]
x = np.linspace(0, 5/2**n_qubits, 1000)
pt_pdf = 2**n_qubits * np.exp(-2**n_qubits * x)
ax1.plot(x * 2**n_qubits, pt_pdf / 2**n_qubits, 'b-', linewidth=2)
ax1.hist(ideal_probs * 2**n_qubits, bins=50, density=True, alpha=0.7, color='orange')
ax1.set_xlabel('Probability × 2^n', fontsize=11)
ax1.set_ylabel('Density', fontsize=11)
ax1.set_title('Porter-Thomas Distribution', fontsize=12)
ax1.legend(['Theoretical', 'Simulated'], fontsize=10)

# Plot 2: Sampled probabilities histogram
ax2 = axes[1]
sampled_probs = ideal_probs[samples]
uniform_probs = np.random.exponential(scale=1/2**n_qubits, size=n_samples)
ax2.hist(sampled_probs * 2**n_qubits, bins=50, density=True, alpha=0.7,
         color='blue', label='Quantum samples')
ax2.hist(uniform_probs * 2**n_qubits, bins=50, density=True, alpha=0.5,
         color='red', label='Uniform samples')
ax2.set_xlabel('Ideal probability × 2^n', fontsize=11)
ax2.set_ylabel('Density', fontsize=11)
ax2.set_title('Sampled Output Probabilities', fontsize=12)
ax2.legend(fontsize=10)

# Plot 3: XEB vs number of samples
ax3 = axes[2]
sample_sizes = [100, 500, 1000, 2000, 5000, 10000, 20000]
xeb_values = []
xeb_errors = []
for ns in sample_sizes:
    xebs = []
    for _ in range(20):  # 20 trials
        x, _, _ = simulate_xeb_experiment(n_qubits, ns, true_fidelity)
        xebs.append(x)
    xeb_values.append(np.mean(xebs))
    xeb_errors.append(np.std(xebs))

ax3.errorbar(sample_sizes, xeb_values, yerr=xeb_errors, fmt='o-',
             color='blue', markersize=8, capsize=5)
ax3.axhline(y=true_fidelity, color='green', linestyle='--',
            label=f'True fidelity = {true_fidelity}')
ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Null (uniform)')
ax3.set_xlabel('Number of Samples', fontsize=11)
ax3.set_ylabel('Measured XEB', fontsize=11)
ax3.set_title('XEB Convergence', fontsize=12)
ax3.set_xscale('log')
ax3.legend(fontsize=10)

plt.tight_layout()
plt.savefig('xeb_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 2. Classical vs Quantum Runtime Comparison
# ============================================================

def classical_simulation_cost(n_qubits, depth, method='tensor_network'):
    """
    Estimate classical simulation cost in FLOPS.

    Simplified models for different methods.
    """
    if method == 'tensor_network':
        # Exponential in treewidth, which scales with min(n, depth) for 2D circuits
        treewidth = min(n_qubits, depth)
        return 2**(1.5 * treewidth) * n_qubits * depth
    elif method == 'statevector':
        # Full state vector simulation
        return 2**n_qubits * depth * 100  # ~100 ops per gate
    elif method == 'mps':
        # Matrix product state (limited entanglement)
        bond_dim = min(2**(n_qubits//2), 2**20)  # Cap bond dimension
        return bond_dim**2 * n_qubits * depth * 100
    else:
        raise ValueError(f"Unknown method: {method}")

def quantum_runtime(n_qubits, depth, gate_time_ns=50, shots=1000):
    """
    Estimate quantum computer runtime.
    """
    circuit_time = depth * gate_time_ns * 1e-9  # seconds per shot
    total_time = circuit_time * shots
    return total_time

# Compare costs
print("\n" + "="*60)
print("CLASSICAL VS QUANTUM RUNTIME COMPARISON")
print("="*60)

qubit_range = np.arange(20, 110, 10)
depth = 30

classical_times = []
quantum_times = []
frontier_flops = 1e18  # Frontier supercomputer: 1 exaFLOPS

for n in qubit_range:
    c_cost = classical_simulation_cost(n, depth, 'tensor_network')
    c_time = c_cost / frontier_flops
    classical_times.append(c_time)

    q_time = quantum_runtime(n, depth, shots=1000000)
    quantum_times.append(q_time)

# Figure 2: Runtime comparison
fig2, ax = plt.subplots(figsize=(12, 6))

ax.semilogy(qubit_range, classical_times, 's-', color='red',
            markersize=10, linewidth=2, label='Classical (tensor network)')
ax.semilogy(qubit_range, quantum_times, 'o-', color='blue',
            markersize=10, linewidth=2, label='Quantum (1M shots)')

# Reference lines
ax.axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='1 second')
ax.axhline(y=3600, color='gray', linestyle=':', alpha=0.7, label='1 hour')
ax.axhline(y=86400*365, color='gray', linestyle='-.', alpha=0.7, label='1 year')
ax.axhline(y=86400*365*1e10, color='purple', linestyle='--', alpha=0.7,
           label='Age of universe')

ax.fill_between(qubit_range, quantum_times, classical_times,
                where=np.array(classical_times) > np.array(quantum_times),
                alpha=0.3, color='green', label='Quantum advantage region')

ax.set_xlabel('Number of Qubits', fontsize=12)
ax.set_ylabel('Runtime (seconds, log scale)', fontsize=12)
ax.set_title('Classical vs Quantum Runtime for Random Circuit Sampling', fontsize=14)
ax.legend(loc='upper left', fontsize=10)
ax.set_ylim([1e-3, 1e30])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('runtime_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 3. Classical Algorithm Improvement Timeline
# ============================================================

# Historical data: classical simulation time for Sycamore circuit
years = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
classical_estimates = [10000*365*24*3600,  # 10,000 years (Google original)
                       2.5*24*3600,          # 2.5 days (IBM)
                       15*24*3600,           # 15 days (various)
                       5*24*3600,            # 5 days (improved TN)
                       3*24*3600,            # 3 days
                       300,                   # 300 seconds (latest)
                       100]                   # Projected

quantum_time = 200  # seconds

fig3, ax = plt.subplots(figsize=(12, 6))

ax.semilogy(years, classical_estimates, 's-', color='red',
            markersize=12, linewidth=2.5, label='Classical simulation time')
ax.axhline(y=quantum_time, color='blue', linestyle='--', linewidth=2,
           label='Quantum time (200 s)')

# Trend line
z = np.polyfit(years, np.log10(classical_estimates), 1)
trend = 10**(z[0]*np.array(years) + z[1])
ax.semilogy(years, trend, '--', color='orange', alpha=0.7,
            label=f'Trend: 10^{z[0]:.1f}× per year')

# Projected crossover
crossover_year = years[0] + (np.log10(quantum_time) - z[1]) / z[0]
ax.axvline(x=crossover_year, color='green', linestyle=':', alpha=0.7)
ax.annotate(f'Crossover ~{crossover_year:.0f}', xy=(crossover_year, 1e4),
            fontsize=11, color='green')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Simulation Time (seconds)', fontsize=12)
ax.set_title('Classical Algorithm Improvement for Sycamore Circuit', fontsize=14)
ax.legend(loc='upper right', fontsize=11)
ax.set_ylim([10, 1e15])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('classical_improvement.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 4. Verification Protocol Simulation
# ============================================================

def run_verification_protocol(n_qubits, n_real_circuits, n_trap_circuits,
                               true_fidelity, trap_fidelity=0.98):
    """
    Simulate a verification protocol with trap circuits.
    """
    results = {
        'trap_successes': 0,
        'trap_total': n_trap_circuits,
        'xeb_estimates': [],
        'confidence_interval': None
    }

    # Trap circuits: known outcomes
    trap_outcomes = np.random.binomial(1, trap_fidelity, n_trap_circuits)
    results['trap_successes'] = trap_outcomes.sum()
    results['trap_success_rate'] = results['trap_successes'] / n_trap_circuits

    # Real circuits: XEB estimation
    for _ in range(n_real_circuits):
        xeb, _, _ = simulate_xeb_experiment(n_qubits, 1000, true_fidelity)
        results['xeb_estimates'].append(xeb)

    xeb_mean = np.mean(results['xeb_estimates'])
    xeb_std = np.std(results['xeb_estimates'])
    results['xeb_mean'] = xeb_mean
    results['confidence_interval'] = (xeb_mean - 2*xeb_std, xeb_mean + 2*xeb_std)

    return results

print("\n" + "="*60)
print("VERIFICATION PROTOCOL SIMULATION")
print("="*60)

results = run_verification_protocol(
    n_qubits=20,
    n_real_circuits=50,
    n_trap_circuits=20,
    true_fidelity=0.01,
    trap_fidelity=0.95
)

print(f"\nTrap circuit results:")
print(f"  Successes: {results['trap_successes']}/{results['trap_total']}")
print(f"  Success rate: {results['trap_success_rate']:.1%}")

print(f"\nXEB estimation:")
print(f"  Mean XEB: {results['xeb_mean']:.4f}")
print(f"  95% CI: ({results['confidence_interval'][0]:.4f}, "
      f"{results['confidence_interval'][1]:.4f})")

# Figure 4: Verification results
fig4, axes = plt.subplots(1, 2, figsize=(12, 5))

# Trap success histogram
ax1 = axes[0]
trap_simulations = [run_verification_protocol(20, 50, 20, 0.01, 0.95)['trap_success_rate']
                    for _ in range(100)]
ax1.hist(trap_simulations, bins=20, color='green', alpha=0.7, edgecolor='black')
ax1.axvline(x=0.95, color='red', linestyle='--', linewidth=2, label='True trap fidelity')
ax1.set_xlabel('Trap Success Rate', fontsize=11)
ax1.set_ylabel('Frequency', fontsize=11)
ax1.set_title('Trap Circuit Success Rate Distribution', fontsize=12)
ax1.legend(fontsize=10)

# XEB distribution
ax2 = axes[1]
xeb_simulations = [run_verification_protocol(20, 50, 20, 0.01, 0.95)['xeb_mean']
                   for _ in range(100)]
ax2.hist(xeb_simulations, bins=20, color='blue', alpha=0.7, edgecolor='black')
ax2.axvline(x=0.01, color='red', linestyle='--', linewidth=2, label='True fidelity')
ax2.axvline(x=0, color='gray', linestyle=':', linewidth=2, label='Null (uniform)')
ax2.set_xlabel('Mean XEB', fontsize=11)
ax2.set_ylabel('Frequency', fontsize=11)
ax2.set_title('XEB Estimate Distribution', fontsize=12)
ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig('verification_protocol.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 5. Advantage Timeline Projection
# ============================================================

print("\n" + "="*60)
print("QUANTUM ADVANTAGE TIMELINE PROJECTION")
print("="*60)

applications = {
    'Random circuit sampling': {
        'current_status': 'Demonstrated',
        'useful_advantage': 'None (no application)',
        'estimated_date': 2019
    },
    'Quantum chemistry (small)': {
        'current_status': 'Competitive with classical',
        'useful_advantage': '10-100× speedup',
        'estimated_date': 2028
    },
    'Quantum chemistry (industrial)': {
        'current_status': 'Far from classical',
        'useful_advantage': 'Intractable classically',
        'estimated_date': 2032
    },
    'Optimization (QAOA)': {
        'current_status': 'No advantage shown',
        'useful_advantage': 'Unknown if possible',
        'estimated_date': None
    },
    'Machine learning': {
        'current_status': 'Limited evidence',
        'useful_advantage': 'Specialized problems only',
        'estimated_date': 2030
    },
    'Cryptography (Shor)': {
        'current_status': 'Toy demonstrations',
        'useful_advantage': 'Breaking RSA-2048',
        'estimated_date': 2035
    }
}

for app, info in applications.items():
    print(f"\n{app}:")
    print(f"  Current: {info['current_status']}")
    print(f"  Target: {info['useful_advantage']}")
    if info['estimated_date']:
        print(f"  Estimated: ~{info['estimated_date']}")
    else:
        print(f"  Estimated: Uncertain")

print("\n" + "="*60)
```

---

## Summary

### Key Concepts

| Concept | Definition | Status (2025) |
|---------|------------|---------------|
| Quantum Supremacy | Any task faster on QC | Demonstrated (2019) |
| Quantum Advantage | Useful task faster on QC | Not yet demonstrated |
| Quantum Utility | Cost-effective QC results | Emerging claims |

### Critical Analysis Framework

$$\boxed{\text{Claim Validity} = \text{Evidence Quality} \times \text{Verification} \times \text{Classical Competition}}$$

### Key Formulas

| Metric | Formula |
|--------|---------|
| XEB Fidelity | $$F_{\text{XEB}} = 2^n \langle p(x) \rangle - 1$$ |
| Statistical Significance | $$Z = F_{\text{XEB}} \sqrt{N/2^n}$$ |
| Speedup | $$S = T_{\text{classical}} / T_{\text{quantum}}$$ |

### Main Takeaways

1. **Supremacy ≠ Advantage** - Random circuit sampling has no applications
2. **Classical algorithms improve** - Every claim faces classical competition
3. **Verification is hard** - We cannot classically verify large quantum computations
4. **Useful advantage is years away** - Requires fault-tolerant quantum computers
5. **Claims require scrutiny** - Extraordinary claims need extraordinary evidence

---

## Daily Checklist

- [ ] I can distinguish supremacy, advantage, and utility
- [ ] I can calculate and interpret XEB fidelity
- [ ] I can analyze classical simulation cost estimates
- [ ] I understand verification challenges and methods
- [ ] I can critically evaluate quantum speedup claims
- [ ] I can identify the most promising applications
- [ ] I ran the verification simulation and understand the results

---

## Preview: Day 990

Tomorrow we examine **Error Correction Demonstrations** - analyzing the latest experimental results in quantum error correction, including surface code implementations, LDPC codes, and real-time decoding. We'll compare different approaches and evaluate progress toward fault-tolerant computation.

---

*"The history of quantum advantage claims is one of constantly moving goalposts. Each quantum milestone is followed by classical algorithm improvements, teaching us that the true advantage may only come with fault-tolerant quantum computers running algorithms with proven exponential speedups."*
