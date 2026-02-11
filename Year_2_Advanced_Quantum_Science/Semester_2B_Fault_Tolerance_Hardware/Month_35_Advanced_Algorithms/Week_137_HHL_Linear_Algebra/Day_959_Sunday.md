# Day 959: Applications & Week Synthesis

## Week 137, Day 7 | Month 35: Advanced Quantum Algorithms

---

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 2.5 hours | Applications: ML, PDEs, optimization |
| Afternoon | 2.5 hours | Week synthesis and integration |
| Evening | 2 hours | Comprehensive review and Week 138 preview |

---

## Learning Objectives

By the end of this day, you will be able to:

1. **Apply HHL to quantum machine learning** problems with proper complexity analysis
2. **Solve differential equations** using HHL-based approaches
3. **Evaluate portfolio optimization** and finance applications
4. **Synthesize the complete HHL complexity landscape**
5. **Make informed decisions** about when HHL provides genuine advantage
6. **Connect HHL to broader quantum algorithm ecosystem**

---

## Core Content

### 1. Quantum Machine Learning Applications

#### Linear Regression

**Classical problem:** Given data matrix $X \in \mathbb{R}^{m \times n}$ and labels $y \in \mathbb{R}^m$, find:
$$\hat{\beta} = (X^T X)^{-1} X^T y$$

**HHL approach:**
1. Prepare quantum state $|y\rangle$
2. Apply HHL with $A = X^T X$
3. Output: quantum state $|\hat{\beta}\rangle$

**Complexity:**
- Classical: $O(mn^2 + n^3)$
- HHL: $O(\text{poly}(\log(n), \kappa, 1/\epsilon))$ + state prep/readout

**When advantage exists:**
- $m, n$ very large
- $\kappa$ small (well-conditioned design matrix)
- Only need predictions $\langle\hat{\beta}|x_{new}\rangle$, not full $\hat{\beta}$

#### Support Vector Machines

**Classical problem:** Find optimal hyperplane maximizing margin.

**Quantum approach (Rebentrost et al.):**
1. Kernel matrix $K_{ij} = \langle\phi(x_i)|\phi(x_j)\rangle$
2. Solve $K\alpha = y$ using HHL
3. Classify: $\text{sign}(\sum_i \alpha_i K(x_i, x_{new}))$

**Current status:** Dequantized by Chia et al. (2019)—classical algorithms can match for many kernel choices.

#### Principal Component Analysis

**Classical problem:** Find top eigenvectors of covariance matrix.

**Quantum approach:**
1. Prepare density matrix $\rho$ from data
2. Use QPE to extract eigenvalues
3. Sample from principal components

**Status:** Partially dequantized—classical randomized methods work for low-rank data.

### 2. Solving Differential Equations

#### Finite Element Method

Many PDEs discretize to linear systems:
$$A u = f$$

where $A$ encodes the differential operator and $f$ the source terms.

**Example: Poisson Equation**

$$\nabla^2 u = f \text{ on domain } \Omega$$

Discretized on $N$ grid points:
$$L u = f$$

where $L$ is the discrete Laplacian.

**HHL complexity:**
$$T_{HHL} = O(\log(N) \cdot s^2 \cdot \kappa^2 / \epsilon)$$

**Key insight:** $\kappa$ scales as $O(N^{2/d})$ for $d$-dimensional problems.

For $d = 3$ (3D problems):
$$\kappa = O(N^{2/3}) \implies T_{HHL} = O(\log N \cdot N^{4/3} / \epsilon)$$

Compare to classical: $T_{CG} = O(N \cdot N^{1/3}) = O(N^{4/3})$

**Result:** Similar scaling! No exponential advantage for standard PDEs.

#### When Quantum Helps

| PDE Type | Classical | HHL | Advantage? |
|----------|-----------|-----|------------|
| Elliptic (Poisson) | $O(N^{4/3})$ | $O(N^{4/3})$ | No |
| Parabolic (Heat) | Similar | Similar | No |
| High-dimensional | Exponential | Polynomial | **Yes** |

**High-dimensional PDEs** (e.g., Boltzmann equation, quantum many-body) offer genuine advantage!

### 3. Financial Applications

#### Portfolio Optimization

**Problem:** Minimize portfolio variance subject to return constraints:
$$\min_w w^T \Sigma w \quad \text{s.t.} \quad \mu^T w = r, \quad \mathbf{1}^T w = 1$$

This reduces to solving:
$$\begin{pmatrix} \Sigma & A^T \\ A & 0 \end{pmatrix} \begin{pmatrix} w \\ \lambda \end{pmatrix} = \begin{pmatrix} 0 \\ b \end{pmatrix}$$

**HHL analysis:**
- Matrix size: $N \times N$ (N assets)
- Condition number: depends on $\Sigma$ eigenvalues
- Output: portfolio weights (need full vector!)

**Problem:** Extracting full $w$ destroys advantage. Only useful if computing risk metrics.

#### Risk Analysis

**Value at Risk (VaR):** Estimate quantiles of portfolio distribution.

**Quantum approach:**
1. Encode return distribution as quantum state
2. Use amplitude estimation for tail probabilities
3. Quantum speedup: $O(1/\epsilon)$ vs $O(1/\epsilon^2)$ classical

This is a **genuine advantage** scenario!

### 4. Complete Complexity Comparison

#### The HHL Decision Tree

```
Does the problem involve Ax = b?
├── No → HHL not applicable
└── Yes
    ├── Is N > 10^6?
    │   ├── No → Use classical methods
    │   └── Yes
    │       ├── Is κ > O(log N)?
    │       │   ├── Yes → Classical often better
    │       │   └── No
    │       │       ├── Is input already quantum?
    │       │       │   ├── Yes → HHL advantageous!
    │       │       │   └── No
    │       │       │       ├── Is data low-rank?
    │       │       │       │   ├── Yes → Tang's algorithm
    │       │       │       │   └── No
    │       │       │       │       └── Need full output?
    │       │       │       │           ├── Yes → No advantage
    │       │       │       │           └── No → Possible advantage
```

#### Summary Table

| Method | Time | Space | Output | Requirements |
|--------|------|-------|--------|--------------|
| Gaussian | $O(N^3)$ | $O(N^2)$ | Full $x$ | General |
| CG | $O(Ns\sqrt{\kappa})$ | $O(Ns)$ | Full $x$ | SPD, sparse |
| HHL | $O(\log N \cdot s^2\kappa^2/\epsilon)$ | $O(\log N)$ | $\|x\rangle$ | Quantum I/O |
| Tang | $O(\|A\|_F^6 \kappa^6/\epsilon)$ | Poly | Samples | SQ access |

### 5. The Honest Assessment

#### Where HHL Truly Shines

1. **Quantum simulation subroutines:** Input from another quantum algorithm
2. **High-dimensional PDEs:** Where classical is exponential
3. **Quantum data analysis:** Data generated by quantum processes
4. **Specific structured problems:** With provable quantum advantage

#### Where HHL Falls Short

1. **General ML on classical data:** Dequantization matches
2. **Standard PDEs:** No exponential advantage
3. **Full solution extraction:** Negates speedup
4. **Ill-conditioned systems:** $\kappa^2$ penalty too severe

#### The Nuanced View

HHL is not a universal speedup—it's a tool with specific applications. Its value:
- Sparked quantum ML research
- Drove understanding of quantum advantage
- Motivated quantum-inspired classical algorithms
- Remains powerful for quantum-native problems

### 6. Connections to Other Algorithms

#### HHL in the Algorithm Ecosystem

```
                    ┌─────────────────┐
                    │  Shor's (Factoring)  │
                    └────────┬────────┘
                             │
┌──────────────┐    ┌───────▼────────┐    ┌─────────────┐
│   Grover's   │    │      QPE       │    │  QFT-based  │
│  (Search)    │    │ (Eigenvalues)  │    │ (Transforms)│
└──────────────┘    └───────┬────────┘    └─────────────┘
                             │
                    ┌───────▼────────┐
                    │      HHL       │
                    │ (Linear Sys)   │
                    └───────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼───────┐   ┌───────▼───────┐   ┌───────▼───────┐
│  Quantum ML   │   │  Quantum PDE  │   │ Quantum Opt.  │
└───────────────┘   └───────────────┘   └───────────────┘
```

#### Algorithm Building Blocks

HHL combines:
- **QPE:** Eigenvalue extraction
- **Hamiltonian simulation:** Matrix exponential
- **Amplitude amplification:** Boost success probability
- **Controlled operations:** Conditional transformations

Understanding HHL means understanding these components.

### 7. Week Synthesis

#### Key Takeaways

| Day | Topic | Key Insight |
|-----|-------|-------------|
| 953 | Classical Methods | $O(N^3)$ direct, $O(N\sqrt{\kappa})$ iterative |
| 954 | QPE Review | Eigenvalue extraction in $O(\log(1/\epsilon))$ |
| 955 | HHL Derivation | Controlled rotation inverts eigenvalues |
| 956 | Circuit Implementation | Trotter + controlled rotation complexity |
| 957 | State Prep/Readout | I/O often dominates; $O(N)$ loading kills speedup |
| 958 | Dequantization | Tang matches HHL for many applications |
| 959 | Applications | Genuine advantage requires specific structure |

#### The Complete HHL Formula

$$\boxed{T_{total} = T_{prep}(N) + O\left(\frac{\log N \cdot s^2 \cdot \kappa^2}{\epsilon}\right) + T_{read}(\text{output type})}$$

**Quantum advantage if and only if:**
1. $T_{prep} = O(\text{poly}(\log N))$
2. $T_{read} = O(\text{poly}(\log N))$
3. $\kappa = O(\text{poly}(\log N))$
4. Classical alternatives are slower

---

## Worked Examples

### Example 1: ML Application Assessment

**Problem:** Evaluate HHL for logistic regression on $10^6$ data points with 1000 features.

**Solution:**

Step 1: Problem formulation
Logistic regression uses Newton's method, each iteration solves:
$$H\delta = -g$$
where $H$ is the Hessian ($1000 \times 1000$).

Step 2: Complexity analysis
- Classical (Newton): $O(\text{iterations} \times N \times n^2)$ where N = data, n = features
- Per iteration: $O(10^6 \times 10^6) = O(10^{12})$ for Hessian computation

Step 3: HHL analysis
- Matrix size: $1000 \times 1000$
- Need full $\delta$ vector for parameter update
- **Readout kills advantage**

Step 4: Alternative quantum approach
Use stochastic gradient descent with quantum speedup for gradient estimation.

$$\boxed{\text{HHL not suitable—use QSGD or classical methods}}$$

---

### Example 2: High-Dimensional PDE

**Problem:** Solve 10-dimensional Poisson equation on $N = 10^{10}$ grid points.

**Solution:**

Step 1: Classical complexity
For $d = 10$ dimensions, mesh has $n^{10}$ points where $n = 10$.
$$N = 10^{10}$$

Classical iterative: $O(N \cdot \kappa)$ where $\kappa \sim N^{2/10} = N^{0.2} \approx 250$

$$T_{classical} = O(10^{10} \times 250) = O(2.5 \times 10^{12})$$

Step 2: HHL complexity
$$T_{HHL} = O(\log(10^{10}) \cdot s^2 \cdot 250^2) = O(33 \cdot s^2 \cdot 62500)$$

With sparsity $s \approx 21$ (10D Laplacian):
$$T_{HHL} = O(33 \cdot 441 \cdot 62500) = O(9 \times 10^8)$$

Step 3: Speedup
$$\frac{T_{classical}}{T_{HHL}} = \frac{2.5 \times 10^{12}}{9 \times 10^8} \approx 2800$$

$$\boxed{\text{HHL provides } \sim 2800\times \text{ speedup for 10D PDE}}$$

**Caveat:** Need efficient state prep and readout for this advantage.

---

### Example 3: Complete Decision Process

**Problem:** Should we use HHL for quantum chemistry (ground state energy)?

**Solution:**

Step 1: Problem structure
Computing ground state energy involves:
1. Prepare trial state (VQE or related)
2. Measure energy $\langle H \rangle$
3. Update parameters
4. Repeat

Linear system aspect: Computing overlaps, response properties.

Step 2: Check conditions
- **Quantum input:** Yes—trial states are quantum
- **Quantum output:** Yes—energy is expectation value
- **Condition number:** Depends on energy gaps
- **Problem size:** Molecular orbitals, $N \sim 100-1000$

Step 3: Assessment
For response properties $\chi = \sum_n \frac{|⟨0|V|n⟩|^2}{E_0 - E_n}$:

This involves $(H - E)^{-1}$, a natural HHL application!

$$\boxed{\text{HHL is suitable for response properties in quantum chemistry}}$$

---

## Practice Problems

### Level 1: Application Assessment

**Problem 1.1:** Can HHL help with image classification on 1 million images? Justify.

**Problem 1.2:** A 3D electromagnetic simulation has $N = 10^9$ unknowns. Estimate $\kappa$ and assess HHL viability.

**Problem 1.3:** List three applications where HHL's quantum output is naturally useful.

### Level 2: Quantitative Analysis

**Problem 2.1:** For the heat equation discretized to $N$ points with time-stepping:
- How many linear systems are solved?
- What is total classical vs HHL cost?
- When does HHL help?

**Problem 2.2:** Design a hybrid classical-quantum algorithm that uses HHL only when advantageous. What's the decision criterion?

**Problem 2.3:** Derive the condition number of the 2D Laplacian on an $n \times n$ grid. How does this affect HHL advantage?

### Level 3: Research-Level

**Problem 3.1:** **Novel Application**

Propose a new application for HHL that:
- Has genuinely low condition number
- Naturally provides quantum input
- Needs only expectation value output
Analyze its complexity.

**Problem 3.2:** **Improved HHL**

Research question: Can variable-time amplitude amplification improve HHL's $\kappa^2$ dependence?

**Problem 3.3:** **Beyond Dequantization**

Identify a class of problems that:
- Involves linear algebra
- Cannot be dequantized
- Provides quantum advantage
Prove why classical algorithms fail.

---

## Computational Lab

### Week Integration and Applications

```python
"""
Day 959: HHL Applications and Week Synthesis
Comprehensive implementation and decision tool.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class ProblemSpec:
    """Specification of a linear system problem."""
    name: str
    N: int  # Problem size
    sparsity: int  # Non-zeros per row
    condition_number: float
    input_type: str  # 'classical', 'quantum', 'structured'
    output_type: str  # 'full', 'expectation', 'sample'
    effective_rank: Optional[int] = None


class HHLDecisionFramework:
    """Framework for deciding when to use HHL."""

    def __init__(self, epsilon: float = 0.01):
        """Initialize with target precision."""
        self.epsilon = epsilon

    def analyze_problem(self, problem: ProblemSpec) -> Dict:
        """
        Analyze whether HHL is advantageous for a given problem.

        Returns comprehensive analysis with recommendations.
        """
        # Compute complexities
        complexities = self._compute_complexities(problem)

        # Check conditions
        conditions = self._check_conditions(problem)

        # Make recommendation
        recommendation = self._make_recommendation(problem, complexities, conditions)

        return {
            'problem': problem,
            'complexities': complexities,
            'conditions': conditions,
            'recommendation': recommendation
        }

    def _compute_complexities(self, p: ProblemSpec) -> Dict:
        """Compute complexity for various methods."""
        N, s, kappa = p.N, p.sparsity, p.condition_number

        complexities = {
            'gaussian': N**3,
            'lu': N**3 if s == N else N * s**2,
            'cg': N * s * np.sqrt(kappa),
            'hhl_core': np.log2(N) * s**2 * kappa**2 / self.epsilon,
        }

        # Add prep and readout costs
        if p.input_type == 'classical':
            complexities['hhl_prep'] = N
        elif p.input_type == 'structured':
            complexities['hhl_prep'] = np.log2(N)**2
        else:  # quantum
            complexities['hhl_prep'] = 1

        if p.output_type == 'full':
            complexities['hhl_read'] = N
        elif p.output_type == 'expectation':
            complexities['hhl_read'] = 1 / self.epsilon**2
        else:  # sample
            complexities['hhl_read'] = 1

        complexities['hhl_total'] = (complexities['hhl_core'] *
                                     complexities['hhl_prep'] *
                                     max(1, np.log2(N)))  # Amplification

        if p.output_type == 'full':
            complexities['hhl_total'] *= N  # Tomography

        # Tang's algorithm (if applicable)
        if p.effective_rank is not None and p.output_type != 'full':
            complexities['tang'] = p.effective_rank**6 * kappa**6 / self.epsilon
        else:
            complexities['tang'] = float('inf')

        return complexities

    def _check_conditions(self, p: ProblemSpec) -> Dict:
        """Check conditions for various algorithms."""
        return {
            'large_N': p.N > 1e6,
            'low_kappa': p.condition_number < np.log2(p.N),
            'quantum_input': p.input_type == 'quantum',
            'limited_output': p.output_type != 'full',
            'low_rank': p.effective_rank is not None and p.effective_rank < np.sqrt(p.N),
            'sparse': p.sparsity < np.log2(p.N)
        }

    def _make_recommendation(self, p: ProblemSpec,
                              comp: Dict, cond: Dict) -> str:
        """Generate recommendation based on analysis."""
        recommendations = []

        # Find best classical method
        classical_best = min(comp['gaussian'], comp['lu'], comp['cg'])

        # Check HHL viability
        if cond['quantum_input'] and cond['limited_output'] and cond['low_kappa']:
            if comp['hhl_core'] < classical_best:
                recommendations.append("HHL: Strongly recommended (quantum I/O, low kappa)")
        elif cond['large_N'] and cond['limited_output']:
            if comp['hhl_core'] < classical_best:
                recommendations.append("HHL: Potentially advantageous (large N)")

        # Check Tang viability
        if cond['low_rank'] and cond['limited_output'] and comp['tang'] < classical_best:
            recommendations.append("Tang's algorithm: Recommended (low-rank structure)")

        # Check classical viability
        if comp['cg'] <= comp['hhl_total']:
            recommendations.append("Conjugate Gradient: Efficient classical baseline")

        if not recommendations:
            recommendations.append("Classical methods preferred (no quantum advantage)")

        return " | ".join(recommendations)


def application_comparison():
    """Compare HHL across different applications."""
    applications = [
        ProblemSpec("Quantum Chemistry", N=10000, sparsity=100,
                   condition_number=50, input_type='quantum',
                   output_type='expectation'),
        ProblemSpec("ML on Classical Data", N=1000000, sparsity=1000,
                   condition_number=100, input_type='classical',
                   output_type='full', effective_rank=50),
        ProblemSpec("3D PDE", N=10**9, sparsity=7,
                   condition_number=10**3, input_type='classical',
                   output_type='full'),
        ProblemSpec("High-dim PDE (10D)", N=10**10, sparsity=21,
                   condition_number=250, input_type='structured',
                   output_type='expectation'),
        ProblemSpec("Portfolio Risk", N=5000, sparsity=5000,
                   condition_number=1000, input_type='classical',
                   output_type='sample', effective_rank=20),
    ]

    framework = HHLDecisionFramework(epsilon=0.01)

    print("HHL Application Comparison")
    print("=" * 80)

    for app in applications:
        analysis = framework.analyze_problem(app)

        print(f"\n{app.name}")
        print("-" * 40)
        print(f"  N = {app.N:.2e}, s = {app.sparsity}, κ = {app.condition_number}")
        print(f"  Input: {app.input_type}, Output: {app.output_type}")

        comp = analysis['complexities']
        print(f"\n  Complexities:")
        print(f"    CG:       {comp['cg']:.2e}")
        print(f"    HHL core: {comp['hhl_core']:.2e}")
        print(f"    HHL total:{comp['hhl_total']:.2e}")
        if comp['tang'] < float('inf'):
            print(f"    Tang:     {comp['tang']:.2e}")

        print(f"\n  Recommendation: {analysis['recommendation']}")


def visualize_advantage_regions():
    """Visualize where HHL provides advantage."""
    N_values = np.logspace(3, 12, 50)
    kappa_values = [1, 10, 100, 1000]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Complexity vs N for different kappa
    ax = axes[0]
    for kappa in kappa_values:
        T_cg = N_values * 10 * np.sqrt(kappa)  # Assuming s=10
        T_hhl = np.log2(N_values) * 100 * kappa**2 / 0.01

        ax.loglog(N_values, T_cg, '--', label=f'CG κ={kappa}')
        ax.loglog(N_values, T_hhl, '-', label=f'HHL κ={kappa}')

    ax.set_xlabel('Problem Size N', fontsize=12)
    ax.set_ylabel('Complexity', fontsize=12)
    ax.set_title('HHL vs CG Complexity', fontsize=14)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    # Plot 2: Advantage region (HHL/CG ratio)
    ax = axes[1]
    N_grid, kappa_grid = np.meshgrid(
        np.logspace(3, 12, 100),
        np.logspace(0, 4, 100)
    )

    T_cg = N_grid * 10 * np.sqrt(kappa_grid)
    T_hhl = np.log2(N_grid) * 100 * kappa_grid**2 / 0.01

    ratio = T_cg / T_hhl

    contour = ax.contourf(np.log10(N_grid), np.log10(kappa_grid),
                          np.log10(ratio), levels=20, cmap='RdYlGn')
    ax.contour(np.log10(N_grid), np.log10(kappa_grid),
               np.log10(ratio), levels=[0], colors='black', linewidths=2)

    plt.colorbar(contour, ax=ax, label='log₁₀(CG/HHL)')
    ax.set_xlabel('log₁₀(N)', fontsize=12)
    ax.set_ylabel('log₁₀(κ)', fontsize=12)
    ax.set_title('HHL Advantage Region (green = HHL wins)', fontsize=14)

    plt.tight_layout()
    plt.savefig('hhl_advantage_regions.png', dpi=150, bbox_inches='tight')
    plt.show()


def pde_application():
    """Demonstrate HHL for PDE solving."""
    print("\nPDE Application Analysis")
    print("=" * 60)

    dimensions = [1, 2, 3, 5, 10, 15]

    for d in dimensions:
        # Grid points per dimension
        n_per_dim = 10

        # Total unknowns
        N = n_per_dim ** d

        # Sparsity (2d+1 for d-dimensional Laplacian)
        s = 2 * d + 1

        # Condition number scales as N^(2/d) for d-dim Laplacian
        kappa = N ** (2/d)

        # Classical (CG)
        T_cg = N * s * np.sqrt(kappa)

        # HHL
        T_hhl = np.log2(max(N, 2)) * s**2 * kappa**2 / 0.01

        ratio = T_cg / T_hhl

        print(f"\n{d}D Laplacian:")
        print(f"  N = {N:.2e}, s = {s}, κ = {kappa:.2e}")
        print(f"  T_CG = {T_cg:.2e}, T_HHL = {T_hhl:.2e}")
        print(f"  Ratio (CG/HHL) = {ratio:.2e}")
        print(f"  Winner: {'HHL' if ratio > 1 else 'CG'}")


def quantum_ml_assessment():
    """Assess HHL for various ML tasks."""
    print("\nQuantum Machine Learning Assessment")
    print("=" * 60)

    tasks = [
        {"name": "Linear Regression", "N": 10000, "features": 100,
         "output": "full", "dequantized": True},
        {"name": "Kernel SVM", "N": 10000, "features": 100,
         "output": "sample", "dequantized": True},
        {"name": "PCA", "N": 10000, "features": 1000,
         "output": "sample", "dequantized": True},
        {"name": "Quantum Kernel", "N": 10000, "features": None,
         "output": "sample", "dequantized": False},
        {"name": "Quantum Data Classification", "N": 1000, "features": None,
         "output": "sample", "dequantized": False},
    ]

    for task in tasks:
        print(f"\n{task['name']}:")
        print(f"  N = {task['N']}")

        if task['dequantized']:
            print("  Status: DEQUANTIZED - classical algorithms exist")
            print("  Recommendation: Use classical methods")
        else:
            print("  Status: Potential quantum advantage")
            if task['output'] == 'sample':
                print("  Recommendation: HHL may provide speedup")
            else:
                print("  Recommendation: Analyze specific problem structure")


def week_summary_visualization():
    """Create visual summary of the week."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Complexity comparison
    ax = axes[0, 0]
    methods = ['Gaussian\nO(N³)', 'CG\nO(N√κ)', 'HHL\nO(log N κ²)', 'Tang\nO(κ⁶)']
    # For N=10^6, κ=100
    complexities = [1e18, 1e8, 1e9, 1e18]
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']

    ax.bar(methods, complexities, color=colors)
    ax.set_yscale('log')
    ax.set_ylabel('Complexity (N=10⁶, κ=100)', fontsize=12)
    ax.set_title('Algorithm Complexity Comparison', fontsize=14)

    # 2. HHL circuit components
    ax = axes[0, 1]
    components = ['State\nPrep', 'QPE', 'Ctrl\nRotation', 'QPE†', 'Readout']
    costs = [100, 40, 20, 40, 30]  # Relative costs

    ax.pie(costs, labels=components, autopct='%1.0f%%', colors=plt.cm.Blues(np.linspace(0.3, 0.8, 5)))
    ax.set_title('HHL Circuit Cost Distribution', fontsize=14)

    # 3. Advantage conditions
    ax = axes[1, 0]
    conditions = ['Quantum\nInput', 'Low κ', 'Limited\nOutput', 'High\nRank', 'Large N']
    importance = [5, 4, 5, 3, 4]
    required = ['Required', 'Important', 'Required', 'Helpful', 'Required']
    colors = ['#ff6b6b' if r == 'Required' else '#4ecdc4' for r in required]

    bars = ax.barh(conditions, importance, color=colors)
    ax.set_xlabel('Importance for Quantum Advantage', fontsize=12)
    ax.set_title('HHL Advantage Conditions', fontsize=14)
    ax.set_xlim([0, 6])

    # 4. Application suitability
    ax = axes[1, 1]
    apps = ['Quantum\nChemistry', 'Classical\nML', 'Low-dim\nPDE', 'High-dim\nPDE', 'Finance']
    suitability = [0.9, 0.2, 0.3, 0.8, 0.5]
    colors = ['green' if s > 0.5 else 'red' if s < 0.4 else 'orange' for s in suitability]

    ax.bar(apps, suitability, color=colors)
    ax.set_ylabel('HHL Suitability Score', fontsize=12)
    ax.set_title('Application Suitability', fontsize=14)
    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig('week_summary.png', dpi=150, bbox_inches='tight')
    plt.show()


# Main execution
if __name__ == "__main__":
    print("Day 959: Applications & Week Synthesis")
    print("=" * 60)

    # Application comparison
    application_comparison()

    # Advantage regions
    print("\n" + "=" * 60)
    visualize_advantage_regions()

    # PDE analysis
    pde_application()

    # ML assessment
    quantum_ml_assessment()

    # Week summary
    print("\n" + "=" * 60)
    print("Generating week summary visualization...")
    week_summary_visualization()

    # Final summary
    print("\n" + "=" * 60)
    print("WEEK 137 COMPLETE: HHL Algorithm & Quantum Linear Algebra")
    print("=" * 60)
    print("""
Key Takeaways:
1. HHL provides exponential speedup in N—but with caveats
2. Condition number κ² dependence can erase advantage
3. State preparation and readout are critical bottlenecks
4. Dequantization matches HHL for many applications
5. Genuine advantage exists for quantum-native problems

The HHL algorithm represents both the promise and complexity of
quantum advantage—powerful in the right circumstances, but
requiring careful analysis of the complete problem structure.
""")
```

---

## Summary

### Complete HHL Complexity Picture

$$\boxed{T_{HHL} = T_{prep} + O\left(\frac{\log N \cdot s^2 \cdot \kappa^2}{\epsilon}\right) + T_{read}}$$

### Decision Framework

| Question | Answer for Advantage |
|----------|---------------------|
| Is N very large ($> 10^6$)? | Yes |
| Is κ small ($\lesssim \log N$)? | Yes |
| Is input quantum/structured? | Yes |
| Is output limited (expectation/sample)? | Yes |
| Is matrix high-rank? | Yes |

### Key Applications

| Application | Advantage | Status |
|-------------|-----------|--------|
| Quantum chemistry | Yes | Genuine |
| High-dimensional PDEs | Yes | Genuine |
| Classical ML | No | Dequantized |
| Standard PDEs | No | Classical competitive |
| Quantum-to-quantum | Yes | Natural fit |

---

## Week 137 Checklist

- [ ] Understand classical linear system complexity
- [ ] Master QPE for eigenvalue extraction
- [ ] Derive complete HHL algorithm
- [ ] Implement HHL circuits in Qiskit
- [ ] Recognize state prep/readout bottlenecks
- [ ] Understand dequantization limits
- [ ] Apply HHL decision framework
- [ ] Identify genuine advantage scenarios

---

## Preview: Week 138

Next week: **Quantum Simulation Algorithms**

- Hamiltonian simulation techniques
- Product formulas and their optimization
- Qubitization and quantum signal processing
- Applications to quantum chemistry
- Fault-tolerant resource estimation

We'll explore how quantum computers simulate quantum systems—arguably the most natural application of quantum computing.

---

*Day 959 of 2184 | Week 137 of 312 | Month 35 of 72*

*"Understanding when a quantum algorithm provides advantage is as important as understanding the algorithm itself."*
