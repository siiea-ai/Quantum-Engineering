# Extended Investigation Guide: Parameter Space Exploration

## Introduction

This guide provides a comprehensive methodology for conducting extended parameter space investigations in quantum engineering research. Unlike initial exploratory work, extended investigation requires systematic approaches that generate reproducible, publication-quality results.

## Part 1: Foundations of Parameter Space Exploration

### 1.1 What is Parameter Space?

Parameter space is the multi-dimensional space defined by all variables that affect your experimental or computational system. Effective exploration requires understanding this space's structure.

**Mathematical Definition:**

For a system with $n$ parameters $\theta_1, \theta_2, \ldots, \theta_n$, the parameter space is:

$$\Theta = \{\boldsymbol{\theta} = (\theta_1, \theta_2, \ldots, \theta_n) : \theta_i \in [\theta_i^{min}, \theta_i^{max}]\}$$

The observable output $y$ is some function of these parameters:

$$y = f(\boldsymbol{\theta}) + \epsilon$$

where $\epsilon$ represents measurement noise and uncontrolled variations.

### 1.2 Parameter Classification System

Before exploring, classify each parameter:

#### Tier 1: Primary Control Parameters
- Directly varied in experiments
- Expected to have large effects
- Well-characterized control capability

#### Tier 2: Secondary Control Parameters
- May be varied for optimization
- Moderate expected effects
- Reasonable control capability

#### Tier 3: Nuisance Parameters
- Cannot or should not be varied
- May still affect results
- Require monitoring and/or averaging

#### Tier 4: Hidden Parameters
- Unknown initially
- Discovered through anomalies
- Often most scientifically interesting

### 1.3 Dimensional Analysis

Before exploring, use dimensional analysis to identify:

1. **Dimensionless combinations** that govern behavior
2. **Natural scales** that set characteristic values
3. **Scaling exponents** that relate different regimes

**Example: Qubit Coherence Study**

Relevant parameters: temperature $T$, magnetic field $B$, drive power $P$, qubit frequency $\omega_q$

Dimensionless combinations:
- $k_B T / \hbar \omega_q$ (thermal-to-quantum ratio)
- $\mu_B B / \hbar \omega_q$ (Zeeman-to-qubit ratio)
- $\Omega_R / \omega_q$ (Rabi-to-qubit ratio, where $\Omega_R \propto \sqrt{P}$)

## Part 2: Experimental Design Strategies

### 2.1 Full Factorial Designs

**When to Use:** Few parameters, strong interactions expected, resources available

**Method:**
For $k$ parameters each at $L$ levels, measure all $L^k$ combinations.

**Example: 2-Parameter Study**

| Run | Param A | Param B | Response |
|-----|---------|---------|----------|
| 1   | Low     | Low     | $y_{11}$ |
| 2   | Low     | High    | $y_{12}$ |
| 3   | High    | Low     | $y_{21}$ |
| 4   | High    | High    | $y_{22}$ |

**Analysis:**
- Main effect of A: $\frac{1}{2}[(y_{21} + y_{22}) - (y_{11} + y_{12})]$
- Main effect of B: $\frac{1}{2}[(y_{12} + y_{22}) - (y_{11} + y_{21})]$
- Interaction AB: $\frac{1}{2}[(y_{11} + y_{22}) - (y_{12} + y_{21})]$

### 2.2 Fractional Factorial Designs

**When to Use:** Many parameters, screening phase, limited resources

**Method:**
Use carefully chosen subset that aliases higher-order interactions.

**$2^{k-p}$ Designs:**
- $k$ parameters, $2^{k-p}$ runs
- Resolution determines aliasing structure
- Resolution III: Main effects aliased with 2-way interactions
- Resolution IV: Main effects clear of 2-way interactions
- Resolution V: 2-way interactions clear of each other

**Example: $2^{5-2}$ Design (5 parameters, 8 runs)**

Generators: D = AB, E = AC

| Run | A | B | C | D=AB | E=AC |
|-----|---|---|---|------|------|
| 1   | - | - | - |   +  |   +  |
| 2   | + | - | - |   -  |   -  |
| 3   | - | + | - |   -  |   +  |
| 4   | + | + | - |   +  |   -  |
| 5   | - | - | + |   +  |   -  |
| 6   | + | - | + |   -  |   +  |
| 7   | - | + | + |   -  |   -  |
| 8   | + | + | + |   +  |   +  |

### 2.3 Response Surface Methodology

**When to Use:** Optimization, continuous parameters, smooth response

**Central Composite Design (CCD):**

Combines factorial points, axial points, and center points.

For 2 parameters:
- 4 factorial points: $(\pm 1, \pm 1)$
- 4 axial points: $(\pm \alpha, 0)$, $(0, \pm \alpha)$
- $n_c$ center points: $(0, 0)$

**Second-Order Model:**
$$y = \beta_0 + \sum_{i=1}^k \beta_i x_i + \sum_{i=1}^k \beta_{ii} x_i^2 + \sum_{i<j} \beta_{ij} x_i x_j + \epsilon$$

**Rotatability:** Choose $\alpha = (n_f)^{1/4}$ where $n_f$ is the number of factorial points.

### 2.4 Latin Hypercube Sampling

**When to Use:** Many parameters, simulation studies, want space-filling

**Method:**
1. Divide each parameter range into $n$ equal intervals
2. Sample once from each interval for each parameter
3. Permute to maximize space-filling properties

**Optimal LHS:**
Minimize correlation between parameters and maximize minimum distance between points.

**Python Implementation:**
```python
from scipy.stats.qmc import LatinHypercube
import numpy as np

def generate_parameter_study(n_parameters, n_samples, bounds):
    """
    Generate Latin Hypercube sample for parameter study.

    Parameters:
    -----------
    n_parameters : int
        Number of parameters
    n_samples : int
        Number of sample points
    bounds : list of tuples
        [(min1, max1), (min2, max2), ...] for each parameter

    Returns:
    --------
    samples : ndarray
        Array of shape (n_samples, n_parameters) with parameter values
    """
    sampler = LatinHypercube(d=n_parameters, optimization="random-cd")
    unit_samples = sampler.random(n=n_samples)

    # Scale to actual bounds
    samples = np.zeros_like(unit_samples)
    for i, (low, high) in enumerate(bounds):
        samples[:, i] = unit_samples[:, i] * (high - low) + low

    return samples
```

### 2.5 Adaptive Sampling

**When to Use:** Expensive measurements, nonlinear responses, need to find optima

**Bayesian Optimization Approach:**

1. Fit Gaussian Process (GP) to current data
2. Define acquisition function (Expected Improvement, UCB, etc.)
3. Sample where acquisition is maximized
4. Repeat until convergence

**Acquisition Functions:**

Expected Improvement:
$$EI(\mathbf{x}) = \mathbb{E}[\max(0, f(\mathbf{x}) - f^*)]$$

Upper Confidence Bound:
$$UCB(\mathbf{x}) = \mu(\mathbf{x}) + \kappa \sigma(\mathbf{x})$$

## Part 3: Boundary Condition Testing

### 3.1 Types of Boundaries

**Hard Boundaries:** Physical limits that cannot be exceeded
- Maximum field strength before breakdown
- Minimum temperature of cryostat
- Power limits for amplifiers

**Soft Boundaries:** Performance degrades but system functions
- Coherence drops but still measurable
- Signal-to-noise decreases but detectable
- Gate errors increase but computation possible

**Stability Boundaries:** System becomes unstable
- Feedback loops oscillate
- Thermal runaway begins
- Systematic drift dominates

### 3.2 Systematic Boundary Characterization

**Protocol:**
1. Start from known good operating point
2. Increment one parameter toward expected boundary
3. Monitor stability indicators and primary metrics
4. Record warning signs before failure
5. Document failure mode
6. Return to baseline, repeat for next boundary

**Safety Considerations:**
- Never approach hard boundaries without safety systems
- Have abort protocols ready
- Log all boundary tests separately
- Consider hysteresis effects

### 3.3 Boundary Documentation Template

```markdown
## Boundary: [Name]

**Parameter:** [Which parameter is being bounded]
**Boundary Type:** [Hard/Soft/Stability]
**Direction:** [Upper/Lower]

### Nominal Operating Point
- Value: ___
- Performance: ___

### Warning Region
- Range: [start] to [boundary]
- Indicators: ___
- Recommended response: ___

### Boundary Condition
- Value: ___
- Failure mode: ___
- Recovery procedure: ___

### Safety Margin
- Recommended limit: ___
- Rationale: ___
```

## Part 4: Scaling Analysis

### 4.1 Identifying Scaling Behavior

**Power Law Scaling:**
$$y = A x^{\alpha}$$

Taking logarithms: $\log y = \log A + \alpha \log x$

Linear regression on log-log plot gives exponent $\alpha$.

**Exponential Scaling:**
$$y = A e^{-x/\xi}$$

Taking logarithms: $\log y = \log A - x/\xi$

Linear regression on semi-log plot gives characteristic scale $\xi$.

### 4.2 Finite-Size Scaling

For systems with characteristic size $L$, near a critical point:

$$y(L) = L^{\beta/\nu} \tilde{y}(L^{1/\nu} (p - p_c))$$

**Data Collapse Protocol:**
1. Measure $y$ vs. parameter $p$ for multiple system sizes $L$
2. Guess exponents $\beta/\nu$ and $1/\nu$
3. Plot $y L^{-\beta/\nu}$ vs. $(p - p_c) L^{1/\nu}$
4. Optimize exponents for best collapse

### 4.3 Quantum-Specific Scaling

**Entanglement Entropy Scaling:**

Area law (gapped systems): $S(L) \sim L^{d-1}$
Volume law (thermal): $S(L) \sim L^d$
Critical: $S(L) \sim \log L$ (1D CFT)

**Quantum Error Scaling:**

For depolarizing noise with error rate $p$:
$$\epsilon(n, p) = 1 - (1 - p)^n \approx n p$$

For correlated errors, different scaling may emerge.

## Part 5: Data Analysis and Synthesis

### 5.1 Systematic Data Processing

**Workflow:**
```
Raw Data → Calibration → Filtering → Statistical Analysis → Model Fitting
    ↓           ↓            ↓              ↓                  ↓
 Archive    Document     Document      Uncertainty         Validate
                                         Quantify           Model
```

### 5.2 Pattern Recognition

Look for:
- **Universal behavior:** Same response across different conditions
- **Scaling collapse:** Data from different conditions falling on single curve
- **Phase transitions:** Abrupt changes in behavior
- **Hysteresis:** Path-dependent effects
- **Anomalies:** Points that don't fit patterns

### 5.3 Preliminary Model Development

Based on extended investigation, formulate:

1. **Empirical models:** Fit functional forms to data
2. **Physical interpretations:** Connect parameters to mechanisms
3. **Predictions:** What would new conditions yield?
4. **Validation tests:** How to check the model?

## Part 6: Documentation Standards

### 6.1 Lab Notebook Requirements

Each investigation day should include:

1. **Date and objectives**
2. **Equipment configuration** (photos if changed)
3. **Parameter settings** (exact values)
4. **Procedure followed** (step by step)
5. **Raw data location** (file paths)
6. **Observations** (including unexpected ones)
7. **Preliminary conclusions**
8. **Next steps**

### 6.2 Data Organization

```
project_root/
├── data/
│   └── week_221/
│       ├── raw/
│       │   └── YYYYMMDD_experiment_description/
│       ├── processed/
│       │   └── YYYYMMDD_analysis_description/
│       └── metadata/
│           └── parameter_log.csv
├── analysis/
│   └── scripts/
│       └── parameter_study_analysis.py
└── docs/
    └── parameter_study_report.md
```

### 6.3 Version Control for Research

```bash
# Daily commit protocol
git add data/metadata/parameter_log.csv
git add analysis/scripts/
git commit -m "Week 221 Day X: [brief description of work]"

# Tag major milestones
git tag -a "parameter-study-complete" -m "Completed Week 221 parameter study"
```

## Part 7: Common Pitfalls and Solutions

### 7.1 Over-fitting

**Symptom:** Model fits training data well but fails on new data

**Solution:**
- Use cross-validation
- Apply Occam's razor
- Check residual structure
- Reserve validation data

### 7.2 Under-sampling

**Symptom:** Miss important features of parameter space

**Solution:**
- Use space-filling designs
- Adaptive refinement in interesting regions
- Replicate key measurements

### 7.3 Systematic Errors

**Symptom:** Consistent bias across measurements

**Solution:**
- Regular calibration
- Reference measurements
- Randomize run order
- Check for equipment drift

### 7.4 Confirmation Bias

**Symptom:** Only see what you expect to see

**Solution:**
- Blind analysis when possible
- Check anomalies seriously
- Have others review data
- Test alternative hypotheses

## Part 8: Quantum Engineering Examples

### 8.1 Example: Qubit Gate Optimization

**Parameters:**
- Pulse amplitude: $\Omega \in [0.1, 10]$ MHz
- Pulse duration: $t_g \in [10, 1000]$ ns
- DRAG coefficient: $\alpha \in [-1, 1]$

**Observable:** Gate fidelity $F$

**Design:** Central Composite Design
- Factorial points: 8 runs at corners
- Axial points: 6 runs at face centers
- Center points: 5 replicated runs

**Analysis:** Fit second-order response surface, find optimum

### 8.2 Example: Quantum Error Threshold

**Parameters:**
- Physical error rate: $p \in [10^{-4}, 10^{-1}]$
- Code distance: $d \in \{3, 5, 7, 9, 11\}$

**Observable:** Logical error rate $p_L$

**Design:** Full factorial with replication

**Analysis:** Fit threshold model
$$p_L = A \left(\frac{p}{p_{th}}\right)^{(d+1)/2}$$

Extract $p_{th}$ and scaling coefficients.

## Conclusion

Extended parameter space investigation transforms initial research findings into robust, quantified results. The key principles are:

1. **Systematic design** before data collection
2. **Efficient exploration** using appropriate designs
3. **Rigorous documentation** at every step
4. **Pattern recognition** in results
5. **Preliminary modeling** for validation planning

The Parameter Study Template in this folder provides a structured format for documenting your extended investigation.

---

**Template:** [Parameter Study Template](./Templates/Parameter_Study.md)

**Return to:** [Week 221 README](./README.md)
