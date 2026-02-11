# Day 222: Covariant Formulation of Electromagnetism

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Theory: The Electromagnetic Field Tensor |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Covariant Maxwell Equations |
| Evening | 7:00 PM - 9:00 PM | 2 hours | Computational Lab |

**Total Study Time: 8 hours**

---

## Learning Objectives

By the end of Day 222, you will be able to:

1. Construct the electromagnetic field tensor $F^{\mu\nu}$ from $\mathbf{E}$ and $\mathbf{B}$
2. Write Maxwell's equations in covariant tensor form
3. Define and use the electromagnetic 4-potential $A^{\mu}$
4. Understand gauge transformations in relativistic notation
5. Derive the Lorentz force from the field tensor
6. Connect the covariant formulation to quantum electrodynamics

---

## Core Content

### 1. The Electromagnetic 4-Potential

**From potentials to fields:**
Recall that $\mathbf{E}$ and $\mathbf{B}$ can be derived from potentials:
$$\mathbf{E} = -\nabla\phi - \frac{\partial\mathbf{A}}{\partial t}, \quad \mathbf{B} = \nabla \times \mathbf{A}$$

**Define the 4-potential:**
$$\boxed{A^{\mu} = \left(\frac{\phi}{c}, \mathbf{A}\right) = \left(\frac{\phi}{c}, A_x, A_y, A_z\right)}$$

Or with lowered index (using the Minkowski metric):
$$A_{\mu} = \left(-\frac{\phi}{c}, A_x, A_y, A_z\right)$$

### 2. The Electromagnetic Field Tensor

The **field strength tensor** (or Faraday tensor) is defined as:

$$\boxed{F^{\mu\nu} = \partial^{\mu}A^{\nu} - \partial^{\nu}A^{\mu}}$$

where $\partial^{\mu} = \eta^{\mu\nu}\partial_{\nu} = \left(-\frac{1}{c}\frac{\partial}{\partial t}, \nabla\right)$

**Explicit components:**

$$F^{\mu\nu} = \begin{pmatrix}
0 & -E_x/c & -E_y/c & -E_z/c \\
E_x/c & 0 & -B_z & B_y \\
E_y/c & B_z & 0 & -B_x \\
E_z/c & -B_y & B_x & 0
\end{pmatrix}$$

**Properties:**
- **Antisymmetric:** $F^{\mu\nu} = -F^{\nu\mu}$
- **6 independent components:** 3 from $\mathbf{E}$, 3 from $\mathbf{B}$
- **Second-rank tensor:** Transforms as $F'^{\mu\nu} = \Lambda^{\mu}_{\ \alpha}\Lambda^{\nu}_{\ \beta}F^{\alpha\beta}$

**Lowered indices:**
$$F_{\mu\nu} = \eta_{\mu\alpha}\eta_{\nu\beta}F^{\alpha\beta} = \begin{pmatrix}
0 & E_x/c & E_y/c & E_z/c \\
-E_x/c & 0 & -B_z & B_y \\
-E_y/c & B_z & 0 & -B_x \\
-E_z/c & -B_y & B_x & 0
\end{pmatrix}$$

### 3. The Dual Field Tensor

The **dual tensor** (Hodge dual) is:

$$\tilde{F}^{\mu\nu} = \frac{1}{2}\epsilon^{\mu\nu\alpha\beta}F_{\alpha\beta}$$

where $\epsilon^{\mu\nu\alpha\beta}$ is the Levi-Civita symbol.

$$\tilde{F}^{\mu\nu} = \begin{pmatrix}
0 & -B_x & -B_y & -B_z \\
B_x & 0 & E_z/c & -E_y/c \\
B_y & -E_z/c & 0 & E_x/c \\
B_z & E_y/c & -E_x/c & 0
\end{pmatrix}$$

The dual is obtained by swapping $\mathbf{E}/c \leftrightarrow \mathbf{B}$ and changing signs appropriately.

### 4. Covariant Maxwell Equations

Maxwell's equations become remarkably simple in tensor form:

**Inhomogeneous equations** (with sources):
$$\boxed{\partial_{\mu}F^{\mu\nu} = \mu_0 J^{\nu}}$$

This single equation contains Gauss's law and Ampère-Maxwell law:
- $\nu = 0$: $\nabla \cdot \mathbf{E} = \rho/\epsilon_0$ (Gauss's law)
- $\nu = 1,2,3$: $\nabla \times \mathbf{B} - \frac{1}{c^2}\frac{\partial\mathbf{E}}{\partial t} = \mu_0\mathbf{J}$ (Ampère-Maxwell)

**Homogeneous equations** (no sources):
$$\boxed{\partial_{\mu}\tilde{F}^{\mu\nu} = 0}$$

Or equivalently, using the Bianchi identity:
$$\boxed{\partial_{\alpha}F_{\beta\gamma} + \partial_{\beta}F_{\gamma\alpha} + \partial_{\gamma}F_{\alpha\beta} = 0}$$

This contains:
- $\nabla \cdot \mathbf{B} = 0$ (no magnetic monopoles)
- $\nabla \times \mathbf{E} + \frac{\partial\mathbf{B}}{\partial t} = 0$ (Faraday's law)

### 5. Gauge Transformations

The 4-potential is not unique. The **gauge transformation**:
$$\boxed{A^{\mu} \to A'^{\mu} = A^{\mu} + \partial^{\mu}\chi}$$

leaves the field tensor unchanged:
$$F'^{\mu\nu} = \partial^{\mu}A'^{\nu} - \partial^{\nu}A'^{\mu} = F^{\mu\nu}$$

**Common gauge choices:**

| Gauge | Condition | Use |
|-------|-----------|-----|
| Lorenz gauge | $\partial_{\mu}A^{\mu} = 0$ | Covariant, wave equations |
| Coulomb gauge | $\nabla \cdot \mathbf{A} = 0$ | Non-relativistic QED |
| Temporal gauge | $A^0 = 0$ | Quantization |

### 6. The Wave Equation for Potentials

In the **Lorenz gauge** ($\partial_{\mu}A^{\mu} = 0$), Maxwell's equations become:

$$\boxed{\Box A^{\mu} = \mu_0 J^{\mu}}$$

where $\Box = \partial_{\mu}\partial^{\mu} = -\frac{1}{c^2}\frac{\partial^2}{\partial t^2} + \nabla^2$ is the d'Alembertian.

**Component form:**
$$\Box\phi = \frac{\rho}{\epsilon_0}, \quad \Box\mathbf{A} = \mu_0\mathbf{J}$$

### 7. Lorentz Force in Covariant Form

The 4-force on a charged particle:

$$\boxed{f^{\mu} = qF^{\mu\nu}u_{\nu}}$$

where $u_{\nu}$ is the 4-velocity.

**Verification:**
$$f^0 = qF^{0\nu}u_{\nu} = q(-E_x/c \cdot \gamma v_x - E_y/c \cdot \gamma v_y - E_z/c \cdot \gamma v_z) = -\frac{q\gamma}{c}\mathbf{E} \cdot \mathbf{v}$$

This is $\gamma/c$ times the power delivered to the particle.

$$f^i = qF^{i\nu}u_{\nu} = \gamma q(E^i + (\mathbf{v} \times \mathbf{B})^i)$$

This recovers the Lorentz force!

### 8. Lorentz Invariants from the Field Tensor

The two Lorentz invariants can be written as:

**First invariant:**
$$\boxed{F_{\mu\nu}F^{\mu\nu} = 2\left(\frac{E^2}{c^2} - B^2\right) = 2(B^2 - E^2/c^2)}$$

Note: Sign depends on convention. With our metric $(-,+,+,+)$:
$$F_{\mu\nu}F^{\mu\nu} = -\frac{2E^2}{c^2} + 2B^2$$

**Second invariant:**
$$\boxed{F_{\mu\nu}\tilde{F}^{\mu\nu} = -\frac{4}{c}\mathbf{E} \cdot \mathbf{B}}$$

Or equivalently: $\epsilon^{\mu\nu\alpha\beta}F_{\mu\nu}F_{\alpha\beta} = -8\mathbf{E}\cdot\mathbf{B}/c$

### 9. The Stress-Energy Tensor

The **electromagnetic stress-energy tensor**:

$$\boxed{T^{\mu\nu} = \frac{1}{\mu_0}\left(F^{\mu\alpha}F^{\nu}_{\ \alpha} - \frac{1}{4}\eta^{\mu\nu}F_{\alpha\beta}F^{\alpha\beta}\right)}$$

**Components:**
- $T^{00} = u = \frac{1}{2}\left(\epsilon_0 E^2 + \frac{B^2}{\mu_0}\right)$ (energy density)
- $T^{0i} = S^i/c$ where $\mathbf{S} = \mathbf{E} \times \mathbf{B}/\mu_0$ (Poynting vector)
- $T^{ij}$ = Maxwell stress tensor

**Conservation:**
$$\partial_{\mu}T^{\mu\nu} = -F^{\nu\alpha}J_{\alpha}$$

The right side is the 4-force density on charges.

---

## Quantum Mechanics Connection

### QED: The Quantum Field Theory of Light

In quantum electrodynamics (QED), the electromagnetic field is quantized:
- The 4-potential $A^{\mu}$ becomes an **operator**
- **Photons** are the quanta of the electromagnetic field
- **Gauge invariance** is essential for renormalizability

### The QED Lagrangian

The Lagrangian density for QED:

$$\mathcal{L}_{QED} = \bar{\psi}(i\hbar c\gamma^{\mu}D_{\mu} - mc^2)\psi - \frac{1}{4\mu_0}F_{\mu\nu}F^{\mu\nu}$$

where $D_{\mu} = \partial_{\mu} + \frac{iq}{\hbar}A_{\mu}$ is the **covariant derivative**.

### Gauge Symmetry and Charge Conservation

The gauge transformation in QED:
$$\psi \to e^{iq\chi/\hbar}\psi, \quad A_{\mu} \to A_{\mu} + \partial_{\mu}\chi$$

This **local U(1) symmetry** leads to:
1. Conservation of electric charge (Noether's theorem)
2. The photon being massless
3. The specific form of electromagnetic interactions

### From Classical to Quantum

| Classical | Quantum |
|-----------|---------|
| $A^{\mu}$ (field) | $\hat{A}^{\mu}$ (operator) |
| $F^{\mu\nu}$ | Observable field strengths |
| Gauge transformation | Phase transformation of $\psi$ |
| $\partial_{\mu}A^{\mu} = 0$ | Gauge fixing for quantization |
| Radiation | Photon emission/absorption |

### The Photon Propagator

In momentum space, the photon propagator (in Feynman gauge):

$$D^{\mu\nu}(k) = \frac{-i\eta^{\mu\nu}}{k^2 + i\epsilon}$$

This appears in every Feynman diagram with a photon line.

---

## Worked Examples

### Example 1: Verify the Homogeneous Maxwell Equation

**Problem:** Show that $\partial_{\mu}\tilde{F}^{\mu 0} = 0$ gives $\nabla \cdot \mathbf{B} = 0$.

**Solution:**

From the dual tensor:
$$\tilde{F}^{\mu 0} = (\tilde{F}^{00}, \tilde{F}^{10}, \tilde{F}^{20}, \tilde{F}^{30}) = (0, -B_x, -B_y, -B_z)$$

Therefore:
$$\partial_{\mu}\tilde{F}^{\mu 0} = \partial_0\tilde{F}^{00} + \partial_1\tilde{F}^{10} + \partial_2\tilde{F}^{20} + \partial_3\tilde{F}^{30}$$
$$= 0 + \frac{\partial(-B_x)}{\partial x} + \frac{\partial(-B_y)}{\partial y} + \frac{\partial(-B_z)}{\partial z}$$
$$= -\nabla \cdot \mathbf{B} = 0$$

$$\boxed{\nabla \cdot \mathbf{B} = 0 \checkmark}$$

### Example 2: Lorentz Transformation of Field Tensor

**Problem:** Given $F^{\mu\nu}$ in frame $S$ with $\mathbf{E} = E_0\hat{\mathbf{y}}$ and $\mathbf{B} = 0$, find $F'^{\mu\nu}$ in frame $S'$ moving at velocity $v$ along $x$.

**Solution:**

Original field tensor (with only $E_y \neq 0$):
$$F^{\mu\nu} = \begin{pmatrix}
0 & 0 & -E_0/c & 0 \\
0 & 0 & 0 & 0 \\
E_0/c & 0 & 0 & 0 \\
0 & 0 & 0 & 0
\end{pmatrix}$$

Transform: $F'^{\mu\nu} = \Lambda^{\mu}_{\ \alpha}\Lambda^{\nu}_{\ \beta}F^{\alpha\beta}$

For a boost in $x$:
$$\Lambda^{\mu}_{\ \nu} = \begin{pmatrix} \gamma & -\beta\gamma & 0 & 0 \\ -\beta\gamma & \gamma & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

Computing $F'^{02}$:
$$F'^{02} = \Lambda^{0}_{\ \alpha}\Lambda^{2}_{\ \beta}F^{\alpha\beta} = \Lambda^{0}_{\ 2}\Lambda^{2}_{\ 0}F^{20} + \Lambda^{0}_{\ 0}\Lambda^{2}_{\ 2}F^{02}$$
$$= 0 \cdot 1 \cdot (E_0/c) + \gamma \cdot 1 \cdot (-E_0/c) = -\gamma E_0/c$$

So $E'_y = \gamma E_0$.

Computing $F'^{12}$ (relates to $B_z$):
$$F'^{12} = \Lambda^{1}_{\ \alpha}\Lambda^{2}_{\ \beta}F^{\alpha\beta} = \Lambda^{1}_{\ 0}\Lambda^{2}_{\ 2}F^{02} + \Lambda^{1}_{\ 2}\Lambda^{2}_{\ 0}F^{20}$$
$$= (-\beta\gamma)(1)(-E_0/c) + 0 = \beta\gamma E_0/c = \gamma v E_0/c^2$$

So $-B'_z = \gamma v E_0/c^2$, giving $B'_z = -\gamma v E_0/c^2$.

$$\boxed{\mathbf{E}' = \gamma E_0\hat{\mathbf{y}}, \quad \mathbf{B}' = -\frac{\gamma v E_0}{c^2}\hat{\mathbf{z}}}$$

This matches the field transformation formulas!

### Example 3: Wave Equation in Lorenz Gauge

**Problem:** Derive the wave equation $\Box A^{\mu} = \mu_0 J^{\mu}$ from $\partial_{\nu}F^{\mu\nu} = \mu_0 J^{\mu}$ using the Lorenz gauge condition.

**Solution:**

Start with:
$$\partial_{\nu}F^{\mu\nu} = \partial_{\nu}(\partial^{\mu}A^{\nu} - \partial^{\nu}A^{\mu}) = \partial_{\nu}\partial^{\mu}A^{\nu} - \partial_{\nu}\partial^{\nu}A^{\mu}$$

The first term: $\partial_{\nu}\partial^{\mu}A^{\nu} = \partial^{\mu}(\partial_{\nu}A^{\nu})$

In Lorenz gauge: $\partial_{\nu}A^{\nu} = 0$

Therefore:
$$\partial_{\nu}F^{\mu\nu} = 0 - \Box A^{\mu} = -\Box A^{\mu}$$

So:
$$-\Box A^{\mu} = \mu_0 J^{\mu}$$

$$\boxed{\Box A^{\mu} = -\mu_0 J^{\mu}}$$

(Note: Sign convention may vary with metric signature choice.)

---

## Practice Problems

### Problem 1: Direct Application
Write out explicitly the components of the equation $\partial_{\mu}F^{\mu 1} = \mu_0 J^{1}$ and show it gives the $x$-component of Ampère-Maxwell law.

### Problem 2: Intermediate
Given that $F_{\mu\nu}F^{\mu\nu} = 2(B^2 - E^2/c^2)$, calculate this invariant for a plane electromagnetic wave where $|\mathbf{E}| = c|\mathbf{B}|$ and $\mathbf{E} \perp \mathbf{B}$.

**Answer:** $F_{\mu\nu}F^{\mu\nu} = 0$ (electromagnetic waves have null field tensor invariants)

### Problem 3: Challenging
Show that the stress-energy tensor satisfies $T^{\mu}_{\ \mu} = 0$ (the electromagnetic field is "traceless"). What does this imply about photons?

**Hint:** This is related to conformal invariance and the masslessness of photons.

---

## Computational Lab

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Physical constants
c = 3e8  # m/s
epsilon_0 = 8.854e-12
mu_0 = 4 * np.pi * 1e-7

def create_field_tensor(E, B):
    """
    Create the electromagnetic field tensor F^{mu nu} from E and B vectors.
    E = [Ex, Ey, Ez] in V/m
    B = [Bx, By, Bz] in T
    """
    Ex, Ey, Ez = E
    Bx, By, Bz = B

    F = np.array([
        [0, -Ex/c, -Ey/c, -Ez/c],
        [Ex/c, 0, -Bz, By],
        [Ey/c, Bz, 0, -Bx],
        [Ez/c, -By, Bx, 0]
    ])
    return F

def create_dual_tensor(F):
    """Create the dual field tensor from F^{mu nu}"""
    # Extract E and B from F
    Ex, Ey, Ez = -c * F[0, 1], -c * F[0, 2], -c * F[0, 3]
    Bx, By, Bz = F[2, 3], F[3, 1], F[1, 2]

    # Dual swaps E/c <-> B
    F_dual = np.array([
        [0, -Bx, -By, -Bz],
        [Bx, 0, Ez/c, -Ey/c],
        [By, -Ez/c, 0, Ex/c],
        [Bz, Ey/c, -Ex/c, 0]
    ])
    return F_dual

def lorentz_boost_matrix(beta):
    """4x4 Lorentz boost matrix for velocity beta*c in x-direction"""
    gamma = 1 / np.sqrt(1 - beta**2)
    return np.array([
        [gamma, -beta*gamma, 0, 0],
        [-beta*gamma, gamma, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def transform_field_tensor(F, beta):
    """Transform field tensor under Lorentz boost"""
    Lambda = lorentz_boost_matrix(beta)
    # F'[mu,nu] = Lambda[mu,alpha] * Lambda[nu,beta] * F[alpha,beta]
    return np.einsum('ma,nb,ab->mn', Lambda, Lambda, F)

def extract_fields(F):
    """Extract E and B vectors from field tensor"""
    E = np.array([-c * F[0, 1], -c * F[0, 2], -c * F[0, 3]])
    B = np.array([F[2, 3], F[3, 1], F[1, 2]])
    return E, B

def compute_invariants(F):
    """Compute the two Lorentz invariants of the field tensor"""
    # Minkowski metric
    eta = np.diag([-1, 1, 1, 1])

    # Lower indices: F_{mu,nu} = eta_{mu,alpha} eta_{nu,beta} F^{alpha,beta}
    F_lower = np.einsum('ma,nb,ab->mn', eta, eta, F)

    # First invariant: F_{mu,nu} F^{mu,nu}
    inv1 = np.einsum('mn,mn->', F_lower, F)

    # For second invariant, use E·B
    E, B = extract_fields(F)
    inv2 = -4 * np.dot(E, B) / c

    return inv1, inv2

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# ========== Plot 1: Field Tensor Components ==========
ax1 = axes[0, 0]

# Example: E in y-direction, no B
E0 = np.array([0, 1e6, 0])  # 1 MV/m
B0 = np.array([0, 0, 0])

F = create_field_tensor(E0, B0)

# Plot as heatmap
im = ax1.imshow(F, cmap='RdBu', aspect='equal')
ax1.set_xticks([0, 1, 2, 3])
ax1.set_xticklabels(['0 (ct)', '1 (x)', '2 (y)', '3 (z)'])
ax1.set_yticks([0, 1, 2, 3])
ax1.set_yticklabels(['0 (ct)', '1 (x)', '2 (y)', '3 (z)'])
ax1.set_title('Field Tensor $F^{\\mu\\nu}$\n($\\mathbf{E} = E_0\\hat{y}$, $\\mathbf{B} = 0$)', fontsize=14)

# Add values as text
for i in range(4):
    for j in range(4):
        val = F[i, j]
        if abs(val) > 1e-10:
            text = f'{val:.2e}'
        else:
            text = '0'
        ax1.text(j, i, text, ha='center', va='center', fontsize=9,
                color='white' if abs(val) > 1e3 else 'black')

plt.colorbar(im, ax=ax1, label='Field components')

# ========== Plot 2: Transformation of Field Tensor ==========
ax2 = axes[0, 1]

# Track how components change with boost velocity
betas = np.linspace(0, 0.95, 50)

E_y_vals = []
B_z_vals = []
inv1_vals = []

for beta in betas:
    F_prime = transform_field_tensor(F, beta)
    E_prime, B_prime = extract_fields(F_prime)
    E_y_vals.append(E_prime[1] / 1e6)  # MV/m
    B_z_vals.append(B_prime[2] * 1e3)  # mT

    inv1, _ = compute_invariants(F_prime)
    inv1_vals.append(inv1)

ax2.plot(betas, E_y_vals, 'b-', linewidth=2, label="$E'_y$ (MV/m)")
ax2.plot(betas, B_z_vals, 'r-', linewidth=2, label="$B'_z$ (mT)")

ax2_twin = ax2.twinx()
ax2_twin.plot(betas, inv1_vals, 'g--', linewidth=2, label='$F_{\\mu\\nu}F^{\\mu\\nu}$')
ax2_twin.set_ylabel('Invariant', fontsize=12, color='green')
ax2_twin.tick_params(axis='y', labelcolor='green')

ax2.set_xlabel('Boost velocity β = v/c', fontsize=12)
ax2.set_ylabel('Field magnitude', fontsize=12)
ax2.set_title('Field Tensor Transformation\n(verifying Lorentz invariance)', fontsize=14)
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.3)

# ========== Plot 3: Dual Tensor ==========
ax3 = axes[1, 0]

# Example with both E and B
E1 = np.array([0, 1e6, 0])
B1 = np.array([0, 0, 2e-3])

F1 = create_field_tensor(E1, B1)
F1_dual = create_dual_tensor(F1)

# Show both tensors side by side
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))

im3a = ax3a.imshow(F1, cmap='RdBu', aspect='equal')
ax3a.set_xticks([0, 1, 2, 3])
ax3a.set_xticklabels(['0', '1', '2', '3'])
ax3a.set_yticks([0, 1, 2, 3])
ax3a.set_yticklabels(['0', '1', '2', '3'])
ax3a.set_title('$F^{\\mu\\nu}$')
plt.colorbar(im3a, ax=ax3a)

im3b = ax3b.imshow(F1_dual, cmap='RdBu', aspect='equal')
ax3b.set_xticks([0, 1, 2, 3])
ax3b.set_xticklabels(['0', '1', '2', '3'])
ax3b.set_yticks([0, 1, 2, 3])
ax3b.set_yticklabels(['0', '1', '2', '3'])
ax3b.set_title('$\\tilde{F}^{\\mu\\nu}$ (Dual)')
plt.colorbar(im3b, ax=ax3b)

plt.suptitle('Field Tensor and its Dual\n($\\mathbf{E} = E_0\\hat{y}$, $\\mathbf{B} = B_0\\hat{z}$)', fontsize=14)
plt.tight_layout()
plt.savefig('day_222_dual_tensor.png', dpi=150, bbox_inches='tight')
plt.close(fig3)

# Back to main figure - Plot invariants for different field configurations
ax3.clear()

# Different field configurations
configs = [
    ('Pure E', [0, 1, 0], [0, 0, 0]),
    ('Pure B', [0, 0, 0], [0, 0, 1]),
    ('E ⊥ B', [0, 1, 0], [0, 0, 1]),
    ('E ∥ B', [0, 1, 0], [0, 1, 0]),
    ('EM wave (E=cB)', [0, 1, 0], [0, 0, 1/c]),
]

betas = np.linspace(0, 0.9, 30)

for label, E_dir, B_dir in configs:
    E = np.array(E_dir) * 1e6  # 1 MV/m magnitude
    B = np.array(B_dir) * 1e-3  # 1 mT magnitude (or special for EM wave)
    F = create_field_tensor(E, B)

    inv1_list = []
    for beta in betas:
        F_prime = transform_field_tensor(F, beta)
        inv1, _ = compute_invariants(F_prime)
        inv1_list.append(inv1)

    # Normalize to initial value for comparison
    if abs(inv1_list[0]) > 1e-20:
        inv1_list = np.array(inv1_list) / inv1_list[0]
    ax3.plot(betas, inv1_list, linewidth=2, label=label)

ax3.axhline(y=1, color='k', linestyle='--', alpha=0.3)
ax3.set_xlabel('Boost velocity β = v/c', fontsize=12)
ax3.set_ylabel('$F_{\\mu\\nu}F^{\\mu\\nu}$ (normalized)', fontsize=12)
ax3.set_title('Lorentz Invariant for Different Configurations', fontsize=14)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_ylim(-0.5, 2)

# ========== Plot 4: Stress-Energy Tensor components ==========
ax4 = axes[1, 1]

# Compute stress-energy for plane wave
# For a wave propagating in z with E in x, B in y
omega = 2 * np.pi * 1e9  # 1 GHz
k = omega / c
E0_wave = 1e3  # V/m

z = np.linspace(0, 3*c/1e9, 200)  # 3 wavelengths

# Electric field amplitude profile (at t=0)
E_x = E0_wave * np.cos(k * z)
B_y = E0_wave / c * np.cos(k * z)

# Energy density
u = 0.5 * (epsilon_0 * E_x**2 + B_y**2 / mu_0)

# Poynting flux (in z direction)
S_z = E_x * B_y / mu_0

ax4.plot(z * 1e9, u * 1e9, 'b-', linewidth=2, label='Energy density u (nJ/m³)')
ax4.plot(z * 1e9, S_z / c * 1e9, 'r-', linewidth=2, label='Momentum density S/c² (nJ/m³)')

ax4.set_xlabel('z (nm)', fontsize=12)
ax4.set_ylabel('Density (nJ/m³)', fontsize=12)
ax4.set_title('Stress-Energy Tensor Components\nfor Electromagnetic Wave', fontsize=14)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_222_covariant_em.png', dpi=150, bbox_inches='tight')
plt.show()

# ========== Numerical Examples ==========
print("=" * 60)
print("COVARIANT ELECTROMAGNETISM")
print("=" * 60)

# Example 1: Field tensor from E and B
print("\n--- Example 1: Field Tensor Construction ---")
E = np.array([0, 1e6, 0])  # 1 MV/m in y
B = np.array([0, 0, 2e-3])  # 2 mT in z

F = create_field_tensor(E, B)
print(f"E = {E} V/m")
print(f"B = {B} T")
print(f"\nField tensor F^μν:")
print(F)

inv1, inv2 = compute_invariants(F)
print(f"\nInvariants:")
print(f"  F_μν F^μν = {inv1:.6e}")
print(f"  F_μν F̃^μν = {inv2:.6e}")
print(f"  Interpretation: E²-c²B² = {(E[1]**2 - c**2*B[2]**2):.6e}")
print(f"                  E·B = {np.dot(E, B):.6e}")

# Example 2: Transformation verification
print("\n" + "=" * 60)
print("--- Example 2: Tensor Transformation ---")

beta = 0.6
F_prime = transform_field_tensor(F, beta)
E_prime, B_prime = extract_fields(F_prime)

print(f"\nBoost velocity: β = {beta}")
print(f"Original: E_y = {E[1]:.2e} V/m, B_z = {B[2]:.2e} T")
print(f"Transformed: E'_y = {E_prime[1]:.2e} V/m, B'_z = {B_prime[2]:.2e} T")

gamma = 1 / np.sqrt(1 - beta**2)
print(f"\nVerification:")
print(f"  γ = {gamma:.4f}")
print(f"  Expected E'_y = γ(E_y - vB_z) = {gamma*(E[1] - beta*c*B[2]):.2e}")
print(f"  Expected B'_z = γ(B_z - vE_y/c²) = {gamma*(B[2] - beta*E[1]/c):.2e}")

inv1_prime, inv2_prime = compute_invariants(F_prime)
print(f"\nInvariant check:")
print(f"  Original F_μν F^μν = {inv1:.6e}")
print(f"  Transformed F_μν F^μν = {inv1_prime:.6e}")
print(f"  Relative difference: {abs(inv1-inv1_prime)/abs(inv1)*100:.4f}%")

# Example 3: Electromagnetic wave
print("\n" + "=" * 60)
print("--- Example 3: Electromagnetic Wave ---")

E_wave = np.array([1e3, 0, 0])  # E in x
B_wave = np.array([0, 1e3/c, 0])  # B in y, |B| = |E|/c

F_wave = create_field_tensor(E_wave, B_wave)
inv1_wave, inv2_wave = compute_invariants(F_wave)

print(f"E = {E_wave[0]:.0f} x̂ V/m")
print(f"B = {B_wave[1]*1e6:.4f} ŷ μT")
print(f"|E|/c = {np.linalg.norm(E_wave)/c:.4e} T")
print(f"|B| = {np.linalg.norm(B_wave):.4e} T")

print(f"\nInvariants (should be zero for EM wave):")
print(f"  F_μν F^μν = {inv1_wave:.6e} (E² - c²B² = 0)")
print(f"  E·B = {np.dot(E_wave, B_wave):.6e} (E ⊥ B)")

# The field tensor of an EM wave is "null" - both invariants vanish

print("\n" + "=" * 60)
print("Day 222: Covariant Formulation Complete")
print("=" * 60)
```

---

## Summary

### Key Formulas

| Formula | Description |
|---------|-------------|
| $A^{\mu} = (\phi/c, \mathbf{A})$ | Electromagnetic 4-potential |
| $F^{\mu\nu} = \partial^{\mu}A^{\nu} - \partial^{\nu}A^{\mu}$ | Field tensor definition |
| $\partial_{\mu}F^{\mu\nu} = \mu_0 J^{\nu}$ | Covariant inhomogeneous Maxwell equations |
| $\partial_{\alpha}F_{\beta\gamma} + \partial_{\beta}F_{\gamma\alpha} + \partial_{\gamma}F_{\alpha\beta} = 0$ | Bianchi identity (homogeneous Maxwell) |
| $\Box A^{\mu} = \mu_0 J^{\mu}$ | Wave equation (Lorenz gauge) |
| $f^{\mu} = qF^{\mu\nu}u_{\nu}$ | Covariant Lorentz force |

### Main Takeaways

1. The **field tensor** $F^{\mu\nu}$ unifies $\mathbf{E}$ and $\mathbf{B}$ into a single object
2. Maxwell's equations reduce to **two tensor equations**: $\partial_{\mu}F^{\mu\nu} = \mu_0 J^{\nu}$ and $\partial_{\mu}\tilde{F}^{\mu\nu} = 0$
3. The 4-potential $A^{\mu}$ is not unique - **gauge freedom** exists
4. Lorentz invariance is **manifest** in tensor notation
5. The **stress-energy tensor** describes energy and momentum flow
6. This formulation is the foundation for **QED**

---

## Daily Checklist

- [ ] I can construct the field tensor from $\mathbf{E}$ and $\mathbf{B}$
- [ ] I can write Maxwell's equations in covariant form
- [ ] I understand gauge transformations and the Lorenz gauge
- [ ] I can derive the Lorentz force from the field tensor
- [ ] I can calculate the Lorentz invariants of the electromagnetic field
- [ ] I understand how this connects to QED

---

## Preview: Day 223

Tomorrow we apply relativistic electrodynamics to **moving charges**: the **Liénard-Wiechert potentials** and **radiation from accelerating charges**. These results are essential for understanding synchrotron radiation and particle accelerators.

---

*"The only laws of matter are those that our minds must fabricate and the only laws of mind are fabricated for it by matter."*
— James Clerk Maxwell

---

**Next:** Day 223 — Relativistic Electrodynamics (Liénard-Wiechert, Radiation)
