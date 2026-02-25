# Week 150: Spin and Magnetic Interactions - Comprehensive Review Guide

## Introduction

Spin is an intrinsic form of angular momentum carried by elementary particles. Unlike orbital angular momentum, spin has no classical analog - it cannot be explained as rotation of a particle about its own axis. This review covers spin-1/2 systems, which form the foundation for understanding qubits, magnetic resonance, and atomic structure.

---

## Section 1: The Discovery and Nature of Spin

### Historical Context

In 1922, Stern and Gerlach passed silver atoms through an inhomogeneous magnetic field and observed two discrete spots on a detector screen, rather than a continuous distribution. This demonstrated:
1. Angular momentum is quantized
2. The splitting into exactly two components suggests $s = 1/2$

The electron's spin was proposed by Uhlenbeck and Goudsmit in 1925 to explain the anomalous Zeeman effect.

### Spin as Intrinsic Angular Momentum

Spin is an intrinsic property of particles, like mass or charge. Key properties:

- **Spin quantum number $s$:** Fixed for each particle type (electron: $s = 1/2$, photon: $s = 1$)
- **Spin operators** $\hat{S}_x$, $\hat{S}_y$, $\hat{S}_z$ satisfy the same commutation relations as orbital angular momentum:

$$[S_i, S_j] = i\hbar\epsilon_{ijk}S_k$$

- **Eigenvalues:**

$$S^2|s,m_s\rangle = \hbar^2 s(s+1)|s,m_s\rangle$$

$$S_z|s,m_s\rangle = \hbar m_s|s,m_s\rangle$$

where $m_s = -s, -s+1, \ldots, s-1, s$.

---

## Section 2: Spin-1/2 States and Operators

### The Two-Dimensional Hilbert Space

For spin-1/2, the Hilbert space is two-dimensional. We denote the basis states as:

$$|+\rangle = |\uparrow\rangle = |s=\frac{1}{2}, m_s=+\frac{1}{2}\rangle = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$$

$$|-\rangle = |\downarrow\rangle = |s=\frac{1}{2}, m_s=-\frac{1}{2}\rangle = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$$

A general spin-1/2 state is:

$$|\psi\rangle = \alpha|\uparrow\rangle + \beta|\downarrow\rangle = \begin{pmatrix} \alpha \\ \beta \end{pmatrix}$$

with $|\alpha|^2 + |\beta|^2 = 1$.

### Spin Operators

The spin operators for $s = 1/2$ are:

$$\boxed{\mathbf{S} = \frac{\hbar}{2}\boldsymbol{\sigma}}$$

where $\boldsymbol{\sigma} = (\sigma_x, \sigma_y, \sigma_z)$ are the Pauli matrices.

The eigenvalues of $S_z$ are $\pm\hbar/2$, and:

$$S^2 = S_x^2 + S_y^2 + S_z^2 = \frac{3\hbar^2}{4}\mathbf{I}$$

This is consistent with $s(s+1)\hbar^2 = \frac{1}{2}\cdot\frac{3}{2}\hbar^2 = \frac{3}{4}\hbar^2$.

---

## Section 3: Pauli Matrices

### Definition and Properties

The Pauli matrices are:

$$\sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad \sigma_y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad \sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

**Fundamental Properties:**

1. **Hermitian:** $\sigma_i^{\dagger} = \sigma_i$
2. **Unitary:** $\sigma_i^{\dagger}\sigma_i = I$
3. **Traceless:** $\text{Tr}(\sigma_i) = 0$
4. **Square to identity:** $\sigma_i^2 = I$
5. **Determinant:** $\det(\sigma_i) = -1$

### Algebraic Relations

**Commutation relations:**
$$[\sigma_i, \sigma_j] = 2i\epsilon_{ijk}\sigma_k$$

**Anticommutation relations:**
$$\{\sigma_i, \sigma_j\} = 2\delta_{ij}I$$

**Product formula:**
$$\boxed{\sigma_i\sigma_j = \delta_{ij}I + i\epsilon_{ijk}\sigma_k}$$

**Vector dot product identity:**
$$(\boldsymbol{\sigma}\cdot\mathbf{a})(\boldsymbol{\sigma}\cdot\mathbf{b}) = (\mathbf{a}\cdot\mathbf{b})I + i\boldsymbol{\sigma}\cdot(\mathbf{a}\times\mathbf{b})$$

### Eigenvalues and Eigenvectors

Each Pauli matrix has eigenvalues $\pm 1$:

**$\sigma_z$:**
$$|+z\rangle = \begin{pmatrix} 1 \\ 0 \end{pmatrix}, \quad |-z\rangle = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$$

**$\sigma_x$:**
$$|+x\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix}, \quad |-x\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ -1 \end{pmatrix}$$

**$\sigma_y$:**
$$|+y\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ i \end{pmatrix}, \quad |-y\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ -i \end{pmatrix}$$

---

## Section 4: The Bloch Sphere

### General Spin-1/2 State

Any pure state of a spin-1/2 system can be written as:

$$|\psi\rangle = \cos\frac{\theta}{2}|\uparrow\rangle + e^{i\phi}\sin\frac{\theta}{2}|\downarrow\rangle$$

where $\theta \in [0,\pi]$ and $\phi \in [0, 2\pi)$. This represents a point on the unit sphere (Bloch sphere) with:
- North pole: $|\uparrow\rangle$ ($\theta = 0$)
- South pole: $|\downarrow\rangle$ ($\theta = \pi$)
- Equator: Superpositions like $|+x\rangle$, $|+y\rangle$

### Expectation Values

For the state $|\psi\rangle$ with Bloch angles $(\theta, \phi)$:

$$\langle S_x\rangle = \frac{\hbar}{2}\sin\theta\cos\phi$$

$$\langle S_y\rangle = \frac{\hbar}{2}\sin\theta\sin\phi$$

$$\langle S_z\rangle = \frac{\hbar}{2}\cos\theta$$

The expectation value vector $\langle\mathbf{S}\rangle$ points from the origin to the state's position on the Bloch sphere.

### Spin Along Arbitrary Direction

For a spin operator along unit vector $\hat{n} = (\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)$:

$$\boldsymbol{\sigma}\cdot\hat{n} = \begin{pmatrix} \cos\theta & \sin\theta e^{-i\phi} \\ \sin\theta e^{i\phi} & -\cos\theta \end{pmatrix}$$

Eigenvalues: $\pm 1$

Eigenvector for $+1$ eigenvalue:
$$|+\hat{n}\rangle = \cos\frac{\theta}{2}|\uparrow\rangle + e^{i\phi}\sin\frac{\theta}{2}|\downarrow\rangle$$

---

## Section 5: Stern-Gerlach Experiment

### Basic Setup

The Stern-Gerlach apparatus uses an inhomogeneous magnetic field $\mathbf{B} = B_0 + z\frac{\partial B}{\partial z}$ (approximately) to exert a force on magnetic dipoles:

$$\mathbf{F} = \nabla(\boldsymbol{\mu}\cdot\mathbf{B}) \approx \mu_z\frac{\partial B}{\partial z}\hat{z}$$

For spin-1/2 particles, $\mu_z$ takes two values corresponding to $S_z = \pm\hbar/2$, so beams split into two paths.

### Measurement Interpretation

The Stern-Gerlach apparatus performs a projective measurement of $S_z$ (for a z-oriented field gradient):

1. Input state: $|\psi\rangle = \alpha|\uparrow\rangle + \beta|\downarrow\rangle$
2. Upper beam: $|\uparrow\rangle$ with probability $|\alpha|^2$
3. Lower beam: $|\downarrow\rangle$ with probability $|\beta|^2$

### Sequential Measurements

**Example: SGz followed by SGx followed by SGz**

1. Initial: unpolarized beam
2. After SGz: Select $|\uparrow_z\rangle$ beam
3. After SGx: $|\uparrow_z\rangle = \frac{1}{\sqrt{2}}(|\uparrow_x\rangle + |\downarrow_x\rangle)$
   - Select $|\uparrow_x\rangle$ beam (50% intensity)
4. After final SGz: $|\uparrow_x\rangle = \frac{1}{\sqrt{2}}(|\uparrow_z\rangle + |\downarrow_z\rangle)$
   - 50% in upper beam, 50% in lower beam

**Key insight:** The SGx measurement "destroyed" the information about the original $S_z$ value.

---

## Section 6: Spin Dynamics in Magnetic Fields

### Hamiltonian

A spin in a magnetic field $\mathbf{B}$ has Hamiltonian:

$$H = -\boldsymbol{\mu}\cdot\mathbf{B} = -\gamma\mathbf{S}\cdot\mathbf{B}$$

where $\gamma = g_s e/(2m)$ is the gyromagnetic ratio. For electrons:

$$\gamma_e = \frac{g_s e}{2m_e} \approx 1.76 \times 10^{11} \text{ rad/(s·T)}$$

with $g_s \approx 2.002$.

For a field along z: $\mathbf{B} = B_0\hat{z}$

$$H = -\gamma B_0 S_z = -\frac{\gamma\hbar B_0}{2}\sigma_z = \frac{\hbar\omega_L}{2}\sigma_z$$

where $\omega_L = -\gamma B_0$ is the Larmor frequency.

### Spin Precession

For initial state $|\psi(0)\rangle = \alpha|\uparrow\rangle + \beta|\downarrow\rangle$:

$$|\psi(t)\rangle = e^{-iHt/\hbar}|\psi(0)\rangle = \alpha e^{-i\omega_L t/2}|\uparrow\rangle + \beta e^{i\omega_L t/2}|\downarrow\rangle$$

The expectation values evolve as:
$$\langle S_x\rangle(t) = \langle S_x\rangle(0)\cos(\omega_L t) + \langle S_y\rangle(0)\sin(\omega_L t)$$
$$\langle S_y\rangle(t) = -\langle S_x\rangle(0)\sin(\omega_L t) + \langle S_y\rangle(0)\cos(\omega_L t)$$
$$\langle S_z\rangle(t) = \langle S_z\rangle(0)$$

This describes precession of $\langle\mathbf{S}\rangle$ about the z-axis at frequency $\omega_L$.

### Rabi Oscillations

For a rotating magnetic field $\mathbf{B}(t) = B_0\hat{z} + B_1(\cos\omega t\,\hat{x} + \sin\omega t\,\hat{y})$:

In the rotating frame (at frequency $\omega$), the effective field is:
$$\mathbf{B}_{\text{eff}} = \left(B_0 - \frac{\omega}{\gamma}\right)\hat{z} + B_1\hat{x}$$

At resonance ($\omega = \gamma B_0$), the effective field is purely transverse, causing spin flips.

The transition probability oscillates:
$$P_{\uparrow\to\downarrow}(t) = \frac{\Omega_1^2}{\Omega^2}\sin^2\left(\frac{\Omega t}{2}\right)$$

where $\Omega_1 = \gamma B_1$ is the Rabi frequency and $\Omega = \sqrt{\Omega_1^2 + \delta^2}$ with detuning $\delta = \omega - \omega_L$.

---

## Section 7: Magnetic Moment and NMR/ESR

### Magnetic Dipole Moment

The magnetic moment of a spin is:

$$\boldsymbol{\mu} = \gamma\mathbf{S} = \frac{g_s e}{2m}\mathbf{S}$$

For the electron: $\mu_B = e\hbar/(2m_e) = 9.274 \times 10^{-24}$ J/T (Bohr magneton)

For protons: $\mu_N = e\hbar/(2m_p) = 5.051 \times 10^{-27}$ J/T (nuclear magneton)

### Energy Splitting

In a magnetic field $B_0$, the energy difference between spin states is:

$$\Delta E = \hbar\omega_L = \gamma\hbar B_0$$

For electrons in $B = 1$ T: $\Delta E \approx 0.12$ meV, $\nu \approx 28$ GHz
For protons in $B = 1$ T: $\Delta E \approx 0.18$ μeV, $\nu \approx 42.6$ MHz

### Resonance Condition

Electromagnetic radiation at frequency $\nu = \omega_L/(2\pi)$ induces transitions between spin states. This is the basis of:
- **ESR** (Electron Spin Resonance): GHz frequencies (microwave)
- **NMR** (Nuclear Magnetic Resonance): MHz frequencies (radio)
- **MRI** (Magnetic Resonance Imaging): NMR applied to imaging

---

## Section 8: Spinor Rotations

### Rotation Operator

A rotation by angle $\phi$ about axis $\hat{n}$ is generated by:

$$R_{\hat{n}}(\phi) = e^{-i\phi\hat{n}\cdot\mathbf{S}/\hbar} = e^{-i\phi\hat{n}\cdot\boldsymbol{\sigma}/2}$$

Using the Pauli matrix exponential:

$$e^{i\theta\hat{n}\cdot\boldsymbol{\sigma}} = \cos\theta\,I + i\sin\theta\,(\hat{n}\cdot\boldsymbol{\sigma})$$

Therefore:
$$R_{\hat{n}}(\phi) = \cos\frac{\phi}{2}I - i\sin\frac{\phi}{2}(\hat{n}\cdot\boldsymbol{\sigma})$$

### Rotation Matrices

**Rotation about z-axis:**
$$R_z(\phi) = e^{-i\phi\sigma_z/2} = \begin{pmatrix} e^{-i\phi/2} & 0 \\ 0 & e^{i\phi/2} \end{pmatrix}$$

**Rotation about x-axis:**
$$R_x(\phi) = e^{-i\phi\sigma_x/2} = \begin{pmatrix} \cos(\phi/2) & -i\sin(\phi/2) \\ -i\sin(\phi/2) & \cos(\phi/2) \end{pmatrix}$$

**Rotation about y-axis:**
$$R_y(\phi) = e^{-i\phi\sigma_y/2} = \begin{pmatrix} \cos(\phi/2) & -\sin(\phi/2) \\ \sin(\phi/2) & \cos(\phi/2) \end{pmatrix}$$

### $4\pi$ Rotation Property

For spinors, a rotation by $2\pi$ gives:
$$R_{\hat{n}}(2\pi) = -I$$

A spinor must be rotated by $4\pi$ to return to its original state. This distinguishes spinors from ordinary vectors.

---

## Summary of Key Results

| Property | Expression |
|----------|------------|
| Spin operators | $\mathbf{S} = \frac{\hbar}{2}\boldsymbol{\sigma}$ |
| $S^2$ eigenvalue | $\frac{3}{4}\hbar^2$ |
| $S_z$ eigenvalues | $\pm\frac{\hbar}{2}$ |
| Pauli product | $\sigma_i\sigma_j = \delta_{ij}I + i\epsilon_{ijk}\sigma_k$ |
| Magnetic moment | $\boldsymbol{\mu} = \gamma\mathbf{S}$ |
| Larmor frequency | $\omega_L = \gamma B_0$ |
| Rotation operator | $R_{\hat{n}}(\phi) = \cos\frac{\phi}{2}I - i\sin\frac{\phi}{2}(\hat{n}\cdot\boldsymbol{\sigma})$ |

---

## References

1. Sakurai, J.J. *Modern Quantum Mechanics*, Chapter 1, 3
2. Shankar, R. *Principles of Quantum Mechanics*, Chapter 14
3. Griffiths, D.J. *Introduction to Quantum Mechanics*, Section 4.4
4. Feynman, R.P. *Feynman Lectures on Physics*, Vol. III, Chapters 5-6

---

**Word Count:** ~2700
**Last Updated:** February 9, 2026
