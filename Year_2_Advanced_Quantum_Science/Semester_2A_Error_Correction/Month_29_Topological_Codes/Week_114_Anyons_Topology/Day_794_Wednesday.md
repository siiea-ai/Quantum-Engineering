# Day 794: Magnetic Fluxes (m-particles)

## Year 2, Semester 2A: Quantum Error Correction
### Month 29: Topological Codes | Week 114: Anyons & Topological Order

---

## Schedule Overview

| Time Block | Duration | Focus |
|------------|----------|-------|
| Morning | 3 hours | Plaquette operator violations and X-string theory |
| Afternoon | 2.5 hours | Problem solving: m-particle manipulation |
| Evening | 1.5 hours | Computational lab: m-particle dynamics and duality |

---

## Learning Objectives

By the end of today, you will be able to:

1. Identify m-particles as violations of plaquette operators ($B_p = -1$)
2. Construct X-string operators that create m-particle pairs
3. Prove the bosonic self-statistics of m-particles
4. Understand the electric-magnetic duality of the toric code
5. Analyze m-particle behavior on the torus topology
6. Connect m-particles to phase-flip error syndromes

---

## Core Content

### 1. Review: The Plaquette Operator

The plaquette operator $B_p$ acts on all edges bounding plaquette $p$:

$$\boxed{B_p = \prod_{e \in \partial p} X_e}$$

On a square lattice, each plaquette has 4 boundary edges:
$$B_p = X_{\text{top}} X_{\text{bottom}} X_{\text{left}} X_{\text{right}}$$

**Properties**:
- $B_p^2 = I$ (involutory)
- $B_p$ has eigenvalues $\pm 1$
- $[B_p, B_{p'}] = 0$ for all plaquettes
- $[A_v, B_p] = 0$ for all vertices and plaquettes

### 2. Defining m-Particles

A **magnetic flux** or **m-particle** is localized at a plaquette $p$ where:

$$\boxed{B_p |\psi\rangle = -|\psi\rangle}$$

The name "magnetic flux" comes from the $\mathbb{Z}_2$ gauge theory analogy, where these excitations correspond to vortices or magnetic monopoles in 2D.

#### Physical Interpretation

In the gauge theory picture:
- Edges carry $\mathbb{Z}_2$ gauge field configurations
- Plaquettes measure the gauge flux through them
- An m-particle represents a $\pi$ flux through the plaquette

#### Energy Cost

From the Hamiltonian $H = -\sum_v A_v - \sum_p B_p$:
- Ground state: $B_p = +1$ contributes $-1$ to energy
- Excited state: $B_p = -1$ contributes $+1$ to energy
- **Energy cost per m-particle**: $\Delta E = 2$

### 3. Creating m-Particles with X-Strings

#### The Fundamental Creation Operator

An **X-string operator** is a product of $X$ operators along a path $\gamma^*$ on the **dual lattice**:

$$\boxed{S_X(\gamma^*) = \prod_{e \perp \gamma^*} X_e}$$

Here $e \perp \gamma^*$ means edges of the original lattice that cross the dual path.

**Theorem**: $S_X(\gamma^*)$ anticommutes with plaquette operators at the endpoints of $\gamma^*$ and commutes with all other plaquette operators.

**Proof**:

Consider a plaquette $p$ and its operator $B_p = \prod_{e \in \partial p} X_e$.

For any edge $e'$:
$$X_{e'} B_p = \begin{cases} -B_p X_{e'} & \text{if } e' \in \partial p \\ +B_p X_{e'} & \text{if } e' \notin \partial p \end{cases}$$

Wait—this is wrong! X operators commute with each other. The anticommutation comes from the Z operators in $A_v$.

**Corrected Analysis**:

$S_X(\gamma^*)$ commutes with all $B_p$ operators (products of X's commute).

But $S_X(\gamma^*)$ anticommutes with $A_v$ at vertices on the path!

**The key insight**: X-strings create m-particles, which are detected by **measuring in the dual basis**.

Actually, let me reconsider the operator structure more carefully.

#### Correct Framework: Dual Lattice Perspective

The m-particles live on the **dual lattice**:
- Original lattice vertices → Dual lattice plaquettes
- Original lattice plaquettes → Dual lattice vertices
- Original lattice edges → Dual lattice edges (perpendicular)

On the dual lattice:
- **Dual star operator** at dual vertex (= original plaquette $p$): $B_p = \prod_{e^* \ni p} X_{e}$
- This measures the m-particle number at that dual vertex

An X-string along dual path $\gamma^*$ creates m-particles at the endpoints (which are dual vertices = original plaquettes).

#### Creating a Pair of m-Particles

Starting from the ground state $|\Omega\rangle$:

$$|\psi_{m,m}\rangle = S_X(\gamma^*) |\Omega\rangle$$

At the endpoint plaquettes $p_1, p_2$ of the dual path:
$$B_{p_1} |\psi_{m,m}\rangle = -|\psi_{m,m}\rangle$$
$$B_{p_2} |\psi_{m,m}\rangle = -|\psi_{m,m}\rangle$$

The X-string crosses an odd number of edges bordering each endpoint plaquette.

### 4. Electric-Magnetic Duality

The toric code exhibits a beautiful **self-duality** under the interchange:

| Electric (e) | Magnetic (m) |
|--------------|--------------|
| Vertex $v$ | Plaquette $p$ (dual vertex) |
| Star $A_v$ | Plaquette $B_p$ (dual star) |
| Z-string | X-string |
| e-particle | m-particle |
| Z error | X error |

This duality is an exact symmetry of the toric code Hamiltonian:
$$H = -\sum_v A_v - \sum_p B_p$$

Swapping Z ↔ X and v ↔ p leaves $H$ invariant.

#### Consequences of Duality

1. **Symmetric energy**: e and m particles have the same energy cost
2. **Symmetric statistics**: Both are self-bosonic
3. **Symmetric creation**: Z-strings ↔ X-strings
4. **Asymmetric mutual statistics**: e and m are mutual semions (tomorrow!)

### 5. Pair Creation Constraint for m-Particles

**Theorem**: m-particles always appear in pairs.

**Proof** (parallel to e-particles):

Consider the global constraint on the torus:
$$\prod_{p} B_p = I$$

Each edge belongs to exactly 2 plaquettes, so each $X_e$ appears twice:
$$\prod_p B_p = \prod_e X_e^2 = I$$

If $n_m$ plaquettes have $B_p = -1$:
$$\prod_p B_p |\psi\rangle = (-1)^{n_m} |\psi\rangle = |\psi\rangle$$

Therefore $n_m$ is even.

$\square$

### 6. Self-Statistics of m-Particles

#### Exchange of Two m-Particles

The analysis parallels that of e-particles:

1. Two m-particles are created by an X-string
2. Exchanging them deforms the X-string
3. The deformation doesn't enclose any e-particles
4. No phase is acquired

**Result**:
$$\theta_{mm} = 0$$

**m-particles are bosons** with respect to self-exchange.

#### Formal Ribbon Operator Argument

The m-particle ribbon operator is:
$$F_m(R) = S_X(\gamma^*_R)$$

A half-twist of this ribbon gives phase +1 because X operators commute.

### 7. m-Particles on the Torus

#### Topological Sectors

On a torus, X-strings can wrap around non-contractible cycles without creating any m-particles:

**Logical X operators**:
$$\bar{X}_1 = \prod_{e \in \gamma^*_{\text{vertical cycle}}} X_e$$
$$\bar{X}_2 = \prod_{e \in \gamma^*_{\text{horizontal cycle}}} X_e$$

These operators:
- Commute with all $B_p$ (no m-particles created)
- Anticommute with $\bar{Z}_1$ or $\bar{Z}_2$ respectively
- Represent logical qubit operations

#### Flux Confinement

Unlike in pure gauge theory, the toric code **confines** m-particles:
- Separating m-particles costs energy proportional to separation (string energy)
- In the ground state, all m-particles are bound in vacuum

But in the topological code, we don't include string tension—all string states are degenerate.

### 8. m-Particles and Error Correction

#### Connection to Phase-Flip Errors

In the error correction picture:
- **X error on edge $e$**: $X_e$ acting on a codeword
- **Syndrome**: The plaquettes where $B_p = -1$

An X-error string creates m-particles at its endpoints.

#### Error Equivalence Classes

Different X-error strings with the same endpoints are equivalent up to stabilizers:
$$X_{\gamma_1} \sim X_{\gamma_2} \iff X_{\gamma_1} X_{\gamma_2} \in \langle A_v \rangle$$

#### Logical X Errors

An X-string wrapping around a non-trivial cycle creates no m-particles but implements a logical X operation—an undetectable error.

---

## Quantum Computing Connection

### Complete Error Model

With both e and m particles understood:

| Error Type | Syndrome | Particle Created |
|------------|----------|------------------|
| Z error | $A_v = -1$ | e-particle at vertex |
| X error | $B_p = -1$ | m-particle at plaquette |
| Y error | Both | Both e and m |

#### Decoding Strategy

1. Measure all $A_v$ → Identify e-particle locations
2. Measure all $B_p$ → Identify m-particle locations
3. Match e-particles with Z-corrections
4. Match m-particles with X-corrections
5. Apply combined correction

The decoding problems for e and m are independent!

### Threshold Theorem

The toric code has a finite error threshold:
- Below threshold: Errors can be corrected with high probability
- Above threshold: Errors accumulate and corrupt logical information

Threshold ≈ 11% for independent depolarizing noise (2D toric code).

---

## Worked Examples

### Example 1: Computing an X-String

**Problem**: On a 4×4 toric code, write out the explicit X-string operator that creates m-particles at plaquettes $(0,0)$ and $(2,1)$.

**Solution**:

The dual lattice has vertices at plaquette centers. We need a path on the dual lattice from center of $(0,0)$ to center of $(2,1)$.

One dual path: $(0.5, 0.5) \to (1.5, 0.5) \to (2.5, 0.5) \to (2.5, 1.5)$

This path crosses the following original edges:
- Vertical edge between $(1, 0)$ and $(1, 1)$: call it $e_1$
- Vertical edge between $(2, 0)$ and $(2, 1)$: call it $e_2$
- Horizontal edge between $(2, 1)$ and $(3, 1)$: call it $e_3$

Wait, let me reconsider the geometry more carefully.

The dual path from plaquette $(0,0)$ to plaquette $(2,1)$:
- Plaquette $(0,0)$ has corners at vertices $(0,0), (1,0), (0,1), (1,1)$
- Plaquette $(2,1)$ has corners at vertices $(2,1), (3,1), (2,2), (3,2)$

Dual path: move right twice, then up once.
- Cross right edge of plaquette $(0,0)$: this is the vertical edge at $x=1$, between $(1,0)-(1,1)$
- Cross right edge of plaquette $(1,0)$: vertical edge at $x=2$, between $(2,0)-(2,1)$
- Cross top edge of plaquette $(2,0)$: horizontal edge at $y=1$, between $(2,1)-(3,1)$

$$\boxed{S_X = X_{(1,0,v)} X_{(2,0,v)} X_{(2,1,h)}}$$

where $(x,y,v)$ denotes the vertical edge at $(x,y)$ and $(x,y,h)$ denotes the horizontal edge.

### Example 2: Verifying m-Particle Creation

**Problem**: Show that the X-string from Example 1 creates $B_p = -1$ at the endpoint plaquettes.

**Solution**:

Consider plaquette $(0,0)$ with boundary edges:
- Bottom: $(0,0,h)$
- Top: $(0,1,h)$
- Left: $(0,0,v)$
- Right: $(1,0,v)$ ← this is in our X-string!

$$B_{(0,0)} = X_{(0,0,h)} X_{(0,1,h)} X_{(0,0,v)} X_{(1,0,v)}$$

Our X-string includes $X_{(1,0,v)}$. Thus:
$$S_X B_{(0,0)} = X_{(1,0,v)} X_{(2,0,v)} X_{(2,1,h)} \cdot X_{(0,0,h)} X_{(0,1,h)} X_{(0,0,v)} X_{(1,0,v)}$$
$$= X_{(0,0,h)} X_{(0,1,h)} X_{(0,0,v)} \cdot X_{(1,0,v)}^2 \cdot X_{(2,0,v)} X_{(2,1,h)}$$
$$= B_{(0,0)} S_X$$

Hmm, they commute because $X^2 = I$. Let me reconsider.

The issue is that X operators always commute. The m-particle syndrome comes from applying the X-string to a state and then measuring $B_p$.

**Correct approach**:

Starting from ground state $|\Omega\rangle$ where $B_p|\Omega\rangle = +|\Omega\rangle$:

After applying $S_X$:
$$B_{(0,0)} S_X |\Omega\rangle = S_X B_{(0,0)} |\Omega\rangle \cdot (\text{some sign})$$

The sign comes from whether the X-string crosses an odd or even number of edges of plaquette $(0,0)$.

Our X-string crosses exactly 1 edge of plaquette $(0,0)$: the edge $(1,0,v)$.

So the state $S_X|\Omega\rangle$ is not an eigenstate of $B_{(0,0)}$ in the naive sense. Instead:

The ground state is the +1 eigenstate of $B_p$ for all $p$. After applying $X_e$, the eigenvalue of any $B_p$ containing $e$ in its boundary flips sign.

Edge $(1,0,v)$ is in the boundary of plaquettes $(0,0)$ and $(1,0)$.
Edge $(2,0,v)$ is in the boundary of plaquettes $(1,0)$ and $(2,0)$.
Edge $(2,1,h)$ is in the boundary of plaquettes $(2,0)$ and $(2,1)$.

Plaquette $(0,0)$: crossed by 1 edge → $B_{(0,0)} = -1$ ✓
Plaquette $(1,0)$: crossed by 2 edges → $B_{(1,0)} = +1$
Plaquette $(2,0)$: crossed by 2 edges → $B_{(2,0)} = +1$
Plaquette $(2,1)$: crossed by 1 edge → $B_{(2,1)} = -1$ ✓

$$\boxed{\text{m-particles at } (0,0) \text{ and } (2,1) \quad \checkmark}$$

### Example 3: Dual Lattice Geometry

**Problem**: For a 3×3 toric code, describe the dual lattice and count its vertices, edges, and plaquettes.

**Solution**:

Original lattice (3×3 torus):
- Vertices: $3 \times 3 = 9$
- Edges: $2 \times 3 \times 3 = 18$ (horizontal and vertical)
- Plaquettes: $3 \times 3 = 9$

Dual lattice:
- Dual vertices = Original plaquettes: 9
- Dual edges = Original edges (rotated 90°): 18
- Dual plaquettes = Original vertices: 9

The dual lattice is also a $3 \times 3$ torus!

$$\boxed{\text{The toric code is self-dual}}$$

This is why the duality $e \leftrightarrow m$ is an exact symmetry.

---

## Practice Problems

### Direct Application

**Problem 1**: X-String Length
On an $L \times L$ toric code, what is the minimum number of X operators needed to create m-particles at plaquettes $(0,0)$ and $(L/2, L/2)$? Assume $L$ is even.

**Problem 2**: Syndrome from Y Error
A Y error ($Y = iXZ$) occurs on a single edge. What syndrome does this produce? Describe in terms of e and m particles.

**Problem 3**: Logical X Operator
Write out the explicit logical $\bar{X}_1$ operator for a 4×4 toric code as a product of Pauli X's. Verify it commutes with all $B_p$.

### Intermediate

**Problem 4**: X-String Commutation
Let $\gamma^*$ be an X-string creating m-particles at $p_1, p_2$. Let $\gamma$ be a Z-string creating e-particles at $v_1, v_2$.
(a) Under what conditions do $S_X(\gamma^*)$ and $S_Z(\gamma)$ commute?
(b) What is their commutator when they don't commute?

**Problem 5**: Dual Stabilizers
Express the constraint $\prod_p B_p = I$ in terms of the dual lattice. What is its geometric interpretation?

**Problem 6**: Mixed Syndrome Decoding
Given syndrome: $A_{(1,1)} = -1$, $A_{(3,2)} = -1$, $B_{(0,0)} = -1$, $B_{(2,2)} = -1$. Find a minimal error pattern consistent with this syndrome.

### Challenging

**Problem 7**: Non-Abelian Generalization
In the quantum double model $D(G)$ for a non-Abelian group $G$, the m-particles are labeled by conjugacy classes. How many distinct m-particle types exist for $G = S_3$ (symmetric group on 3 elements)?

**Problem 8**: Thermal m-Particle Density
At temperature $T$, estimate the density of m-particles on an $L \times L$ toric code. How does this affect the code's error correction capability?

---

## Computational Lab: m-Particle Dynamics and Duality

```python
"""
Day 794 Computational Lab: Magnetic Fluxes (m-Particles)
Simulating X-string operators and the electric-magnetic duality
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from matplotlib.collections import PatchCollection

class ToricCodeMParticles:
    """
    Toric code simulator focused on m-particle (plaquette violation) dynamics
    Extends the e-particle simulator with X-string capabilities
    """

    def __init__(self, L):
        """Initialize L x L toric code"""
        self.L = L
        self.n_edges = 2 * L * L

        # Edge indexing
        self.edge_list = []
        for x in range(L):
            for y in range(L):
                self.edge_list.append((x, y, 'h'))
                self.edge_list.append((x, y, 'v'))
        self.edge_to_idx = {e: i for i, e in enumerate(self.edge_list)}

        # Track both X and Z errors separately for visualization
        self.reset_state()

    def reset_state(self):
        """Reset to ground state"""
        self.x_errors = np.zeros(self.n_edges, dtype=int)  # X error configuration
        self.z_errors = np.zeros(self.n_edges, dtype=int)  # Z error configuration
        self.e_particles = set()
        self.m_particles = set()

    def get_plaquette_edges(self, plaquette):
        """Get edges bounding plaquette (x, y)"""
        x, y = plaquette
        L = self.L
        edges = [
            (x, y, 'h'),                    # bottom
            (x, (y+1) % L, 'h'),            # top
            (x, y, 'v'),                    # left
            ((x+1) % L, y, 'v'),            # right
        ]
        return edges

    def get_star_edges(self, vertex):
        """Get edges incident to vertex (x, y)"""
        x, y = vertex
        L = self.L
        edges = [
            (x, y, 'h'),                    # right
            ((x-1) % L, y, 'h'),            # left
            (x, y, 'v'),                    # up
            (x, (y-1) % L, 'v'),            # down
        ]
        return edges

    def compute_plaquette_syndrome(self, plaquette):
        """Compute B_p eigenvalue (affected by X errors)"""
        edges = self.get_plaquette_edges(plaquette)
        parity = 0
        for e in edges:
            idx = self.edge_to_idx[e]
            parity += self.x_errors[idx]
        return (-1) ** (parity % 2)

    def compute_star_syndrome(self, vertex):
        """Compute A_v eigenvalue (affected by Z errors)"""
        edges = self.get_star_edges(vertex)
        parity = 0
        for e in edges:
            idx = self.edge_to_idx[e]
            parity += self.z_errors[idx]
        return (-1) ** (parity % 2)

    def apply_x_error(self, edge):
        """Apply X operator to edge"""
        idx = self.edge_to_idx[edge]
        self.x_errors[idx] = 1 - self.x_errors[idx]
        self._update_particles()

    def apply_z_error(self, edge):
        """Apply Z operator to edge"""
        idx = self.edge_to_idx[edge]
        self.z_errors[idx] = 1 - self.z_errors[idx]
        self._update_particles()

    def apply_x_string(self, plaquette_path):
        """
        Apply X-string along dual lattice path
        plaquette_path: list of plaquettes the dual path passes through
        """
        if len(plaquette_path) < 2:
            return

        for i in range(len(plaquette_path) - 1):
            p1, p2 = plaquette_path[i], plaquette_path[i+1]
            edge = self._get_edge_between_plaquettes(p1, p2)
            if edge:
                self.apply_x_error(edge)

    def apply_z_string(self, vertex_path):
        """Apply Z-string along vertex path"""
        if len(vertex_path) < 2:
            return

        for i in range(len(vertex_path) - 1):
            v1, v2 = vertex_path[i], vertex_path[i+1]
            edge = self._get_edge_between_vertices(v1, v2)
            if edge:
                self.apply_z_error(edge)

    def _get_edge_between_plaquettes(self, p1, p2):
        """Get edge between adjacent plaquettes"""
        x1, y1 = p1
        x2, y2 = p2
        L = self.L

        # Horizontal adjacency
        if y1 == y2:
            if (x1 + 1) % L == x2:
                return ((x1 + 1) % L, y1, 'v')
            elif (x2 + 1) % L == x1:
                return ((x2 + 1) % L, y2, 'v')

        # Vertical adjacency
        if x1 == x2:
            if (y1 + 1) % L == y2:
                return (x1, (y1 + 1) % L, 'h')
            elif (y2 + 1) % L == y1:
                return (x2, (y2 + 1) % L, 'h')

        return None

    def _get_edge_between_vertices(self, v1, v2):
        """Get edge between adjacent vertices"""
        x1, y1 = v1
        x2, y2 = v2
        L = self.L

        if y1 == y2:
            if (x1 + 1) % L == x2:
                return (x1, y1, 'h')
            elif (x2 + 1) % L == x1:
                return (x2, y2, 'h')

        if x1 == x2:
            if (y1 + 1) % L == y2:
                return (x1, y1, 'v')
            elif (y2 + 1) % L == y1:
                return (x2, y2, 'v')

        return None

    def _update_particles(self):
        """Update e and m particle locations"""
        self.e_particles = set()
        self.m_particles = set()

        for x in range(self.L):
            for y in range(self.L):
                if self.compute_star_syndrome((x, y)) == -1:
                    self.e_particles.add((x, y))
                if self.compute_plaquette_syndrome((x, y)) == -1:
                    self.m_particles.add((x, y))

    def get_energy(self):
        """Total energy relative to ground state"""
        return 2 * (len(self.e_particles) + len(self.m_particles))

    def visualize(self, title="Toric Code State", ax=None):
        """Visualize state with both e and m particles"""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        L = self.L

        # Draw edges with error coloring
        for e in self.edge_list:
            x, y, d = e
            idx = self.edge_to_idx[e]

            has_x = self.x_errors[idx] == 1
            has_z = self.z_errors[idx] == 1

            if has_x and has_z:
                color = 'purple'  # Y error
                lw = 3
            elif has_x:
                color = 'red'  # X error
                lw = 3
            elif has_z:
                color = 'blue'  # Z error
                lw = 3
            else:
                color = 'lightgray'
                lw = 1

            if d == 'h':
                ax.plot([x, x+1], [y, y], color=color, linewidth=lw)
            else:
                ax.plot([x, x], [y, y+1], color=color, linewidth=lw)

        # Draw vertices
        for x in range(L):
            for y in range(L):
                if (x, y) in self.e_particles:
                    circle = Circle((x, y), 0.15, color='blue', zorder=5)
                    ax.add_patch(circle)
                    ax.annotate('e', (x, y), ha='center', va='center',
                               fontsize=10, color='white', fontweight='bold', zorder=6)
                else:
                    circle = Circle((x, y), 0.06, color='black', zorder=5)
                    ax.add_patch(circle)

        # Draw plaquettes with m-particles
        for x in range(L):
            for y in range(L):
                if (x, y) in self.m_particles:
                    rect = Rectangle((x+0.2, y+0.2), 0.6, 0.6,
                                     color='red', alpha=0.7, zorder=4)
                    ax.add_patch(rect)
                    ax.annotate('m', (x+0.5, y+0.5), ha='center', va='center',
                               fontsize=10, color='white', fontweight='bold', zorder=6)

        ax.set_xlim(-0.5, L + 0.5)
        ax.set_ylim(-0.5, L + 0.5)
        ax.set_aspect('equal')
        ax.set_title(f"{title}\nEnergy = {self.get_energy()}, "
                    f"e = {len(self.e_particles)}, m = {len(self.m_particles)}")

        return ax


def demonstrate_m_particle_creation():
    """Show X-string creating m-particle pairs"""
    print("=" * 60)
    print("Demonstration: m-Particle Creation with X-Strings")
    print("=" * 60)

    tc = ToricCodeMParticles(L=5)

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    # State 1: Ground state
    ax = axes[0, 0]
    tc.reset_state()
    tc.visualize("Ground State", ax)

    # State 2: Single X error
    ax = axes[0, 1]
    tc.reset_state()
    tc.apply_x_error((2, 2, 'h'))
    tc.visualize("Single X Error", ax)
    print(f"\nAfter X on edge (2,2,'h'):")
    print(f"  m-particles at plaquettes: {tc.m_particles}")

    # State 3: X-string
    ax = axes[1, 0]
    tc.reset_state()
    plaquette_path = [(0, 2), (1, 2), (2, 2), (3, 2)]
    tc.apply_x_string(plaquette_path)
    tc.visualize("X-String (Horizontal)", ax)
    print(f"\nAfter X-string from plaquette (0,2) to (3,2):")
    print(f"  m-particles at: {tc.m_particles}")

    # State 4: Two pairs
    ax = axes[1, 1]
    tc.reset_state()
    tc.apply_x_string([(1, 1), (1, 2), (1, 3)])
    tc.apply_x_string([(3, 0), (3, 1)])
    tc.visualize("Two m-Particle Pairs", ax)
    print(f"\nAfter two X-strings:")
    print(f"  m-particles at: {tc.m_particles}")

    plt.tight_layout()
    plt.savefig('m_particle_creation.png', dpi=150, bbox_inches='tight')
    plt.show()


def demonstrate_duality():
    """Demonstrate electric-magnetic duality"""
    print("\n" + "=" * 60)
    print("Demonstration: Electric-Magnetic Duality")
    print("=" * 60)

    tc = ToricCodeMParticles(L=5)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Electric configuration
    ax = axes[0]
    tc.reset_state()
    tc.apply_z_string([(1, 2), (2, 2), (3, 2), (4, 2)])
    tc.visualize("Z-String → e-Particles", ax)
    print(f"\nZ-string creates e-particles at: {tc.e_particles}")

    # Magnetic configuration (dual)
    ax = axes[1]
    tc.reset_state()
    tc.apply_x_string([(1, 2), (2, 2), (3, 2), (4, 2)])
    tc.visualize("X-String → m-Particles", ax)
    print(f"X-string creates m-particles at: {tc.m_particles}")

    fig.suptitle("Electric-Magnetic Duality: Z↔X, e↔m", fontsize=16)
    plt.tight_layout()
    plt.savefig('em_duality.png', dpi=150, bbox_inches='tight')
    plt.show()


def demonstrate_combined_errors():
    """Show combined X and Z errors (Y errors)"""
    print("\n" + "=" * 60)
    print("Demonstration: Combined Errors (Y = iXZ)")
    print("=" * 60)

    tc = ToricCodeMParticles(L=5)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # X error only
    ax = axes[0]
    tc.reset_state()
    tc.apply_x_error((2, 2, 'h'))
    tc.visualize("X Error Only", ax)

    # Z error only
    ax = axes[1]
    tc.reset_state()
    tc.apply_z_error((2, 2, 'h'))
    tc.visualize("Z Error Only", ax)

    # Y error (both X and Z)
    ax = axes[2]
    tc.reset_state()
    tc.apply_x_error((2, 2, 'h'))
    tc.apply_z_error((2, 2, 'h'))
    tc.visualize("Y Error (X + Z)", ax)

    print(f"\nY error creates:")
    print(f"  e-particles at: {tc.e_particles}")
    print(f"  m-particles at: {tc.m_particles}")
    print(f"  This is an ε-particle (fermion) - covered on Day 796!")

    plt.tight_layout()
    plt.savefig('combined_errors.png', dpi=150, bbox_inches='tight')
    plt.show()


def demonstrate_logical_operators():
    """Show logical X operators on the torus"""
    print("\n" + "=" * 60)
    print("Demonstration: Logical X Operators (Non-contractible Loops)")
    print("=" * 60)

    tc = ToricCodeMParticles(L=4)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Logical X_1 (horizontal loop)
    ax = axes[0]
    tc.reset_state()
    # Apply X on vertical edges crossing the horizontal line
    for x in range(tc.L):
        tc.apply_x_error((x, 2, 'v'))
    tc.visualize("Logical X̄₁ (horizontal cycle)", ax)
    print(f"\nLogical X̄₁: m-particles = {tc.m_particles}")
    print(f"  (No m-particles! This is a logical operator)")

    # Logical X_2 (vertical loop)
    ax = axes[1]
    tc.reset_state()
    for y in range(tc.L):
        tc.apply_x_error((2, y, 'h'))
    tc.visualize("Logical X̄₂ (vertical cycle)", ax)
    print(f"Logical X̄₂: m-particles = {tc.m_particles}")

    # Contractible loop (stabilizer)
    ax = axes[2]
    tc.reset_state()
    # Apply X around a single plaquette
    edges = tc.get_plaquette_edges((2, 2))
    for e in edges:
        tc.apply_x_error(e)
    tc.visualize("Stabilizer B_p (contractible loop)", ax)
    print(f"Stabilizer: m-particles = {tc.m_particles}")
    print(f"  (No m-particles! This is a stabilizer)")

    plt.tight_layout()
    plt.savefig('logical_x_operators.png', dpi=150, bbox_inches='tight')
    plt.show()


def demonstrate_syndrome_separation():
    """Show independent syndrome measurements for e and m"""
    print("\n" + "=" * 60)
    print("Demonstration: Independent Syndrome Measurements")
    print("=" * 60)

    tc = ToricCodeMParticles(L=6)

    # Apply random X and Z errors
    np.random.seed(123)
    n_x_errors = 3
    n_z_errors = 2

    x_error_edges = np.random.choice(len(tc.edge_list), n_x_errors, replace=False)
    z_error_edges = np.random.choice(len(tc.edge_list), n_z_errors, replace=False)

    for idx in x_error_edges:
        tc.apply_x_error(tc.edge_list[idx])
    for idx in z_error_edges:
        tc.apply_z_error(tc.edge_list[idx])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Full state
    ax = axes[0]
    tc.visualize("Full Error Pattern", ax)

    # e-syndrome only
    ax = axes[1]
    ax.set_xlim(-0.5, tc.L + 0.5)
    ax.set_ylim(-0.5, tc.L + 0.5)
    ax.set_aspect('equal')
    ax.set_title("Z-Syndrome (e-particles)", fontsize=14)

    for x in range(tc.L):
        for y in range(tc.L):
            if (x, y) in tc.e_particles:
                circle = Circle((x, y), 0.2, color='blue', zorder=5)
                ax.add_patch(circle)
            else:
                circle = Circle((x, y), 0.05, color='gray', zorder=5)
                ax.add_patch(circle)

    # m-syndrome only
    ax = axes[2]
    ax.set_xlim(-0.5, tc.L + 0.5)
    ax.set_ylim(-0.5, tc.L + 0.5)
    ax.set_aspect('equal')
    ax.set_title("X-Syndrome (m-particles)", fontsize=14)

    for x in range(tc.L):
        for y in range(tc.L):
            # Draw plaquette center
            if (x, y) in tc.m_particles:
                rect = Rectangle((x+0.2, y+0.2), 0.6, 0.6, color='red', zorder=5)
                ax.add_patch(rect)
            else:
                circle = Circle((x+0.5, y+0.5), 0.05, color='gray', zorder=5)
                ax.add_patch(circle)

    print(f"\ne-particles (Z-syndrome): {tc.e_particles}")
    print(f"m-particles (X-syndrome): {tc.m_particles}")
    print("\nSyndromes are independent! Can decode separately.")

    plt.tight_layout()
    plt.savefig('syndrome_separation.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Run all demonstrations"""
    print("=" * 60)
    print("Day 794: Magnetic Fluxes (m-Particles) - Computational Lab")
    print("=" * 60)

    demonstrate_m_particle_creation()
    demonstrate_duality()
    demonstrate_combined_errors()
    demonstrate_logical_operators()
    demonstrate_syndrome_separation()

    print("\n" + "=" * 60)
    print("Key Insights from Lab:")
    print("=" * 60)
    print("1. X-strings create m-particle pairs at dual endpoints")
    print("2. Electric-magnetic duality: Z↔X, e↔m, vertex↔plaquette")
    print("3. Y errors create both e and m particles (tomorrow: ε)")
    print("4. Logical X operators are non-contractible X-loops")
    print("5. e and m syndromes are independent → parallel decoding")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Plaquette operator | $B_p = \prod_{e \in \partial p} X_e$ |
| m-particle definition | $B_p \|\psi\rangle = -\|\psi\rangle$ |
| X-string operator | $S_X(\gamma^*) = \prod_{e \perp \gamma^*} X_e$ |
| Energy per m-particle | $\Delta E = 2$ |
| Parity constraint | $\prod_p B_p = I \Rightarrow n_m \in 2\mathbb{Z}$ |
| Self-statistics | $\theta_{mm} = 0$ (bosonic) |

### Main Takeaways

1. **m-particles are plaquette violations**: Plaquettes where $B_p = -1$ host m-particles
2. **Created by X-strings**: The operator $S_X(\gamma^*)$ creates m-particles at dual path endpoints
3. **Electric-magnetic duality**: The toric code is self-dual under e ↔ m, Z ↔ X
4. **Bosonic self-statistics**: Exchanging two m-particles gives phase +1
5. **Independent syndromes**: e and m syndromes can be decoded separately
6. **Logical X operators**: Non-contractible X-loops on the torus

---

## Daily Checklist

### Morning Theory (3 hours)
- [ ] Master the plaquette operator $B_p$ and its eigenvalues
- [ ] Derive how X-strings create m-particle pairs
- [ ] Understand the electric-magnetic duality
- [ ] Prove m-particles have bosonic self-statistics

### Afternoon Problems (2.5 hours)
- [ ] Complete all Direct Application problems
- [ ] Work through at least 2 Intermediate problems
- [ ] Attempt at least 1 Challenging problem

### Evening Lab (1.5 hours)
- [ ] Run all simulation code
- [ ] Experiment with different X-string configurations
- [ ] Compare e and m particle behavior

### Self-Assessment Questions
1. What is the relationship between X-strings and the dual lattice?
2. Why is the toric code called "self-dual"?
3. Can you create a single m-particle? Why or why not?

---

## Preview: Day 795

Tomorrow we explore the most fascinating aspect of toric code anyons: **mutual statistics**. When an e-particle circles an m-particle, the wave function acquires a phase of $e^{i\pi} = -1$—semionic mutual statistics! This is the key property that makes anyons fundamentally different from ordinary bosons and fermions, and it's central to their use in topological quantum computing.

---

*Day 794 of 2184 | Year 2, Month 29, Week 114 | Quantum Engineering PhD Curriculum*
