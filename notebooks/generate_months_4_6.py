"""
SIIEA Quantum Engineering - Notebook Generator: Months 4-6
==========================================================
Generates 3 Jupyter notebooks for Year 0, Months 4-6:
  - Month 04: Eigenvalues and Transformations (Linear Algebra I)
  - Month 05: Complex Analysis Foundations (Linear Algebra II / Complex)
  - Month 06: Lagrangian & Hamiltonian Mechanics (Classical Mechanics)

Run with:
    .venv/bin/python3 notebooks/generate_months_4_6.py
"""

import sys
import os

# Ensure we can import from the notebooks directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_notebook import NotebookBuilder


# ============================================================================
# NOTEBOOK 1: Month 04 — Eigenvalues and Transformations
# ============================================================================
def build_month_04():
    nb = NotebookBuilder(
        "Eigenvalues and Linear Transformations",
        "year_0/month_04_linear_algebra_I/04_eigenvalues_and_transformations.ipynb",
        "Days 85-112",
    )

    # ---- Imports & Setup ----
    nb.code("""\
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from numpy.linalg import eig, svd, det, inv, norm, qr
%matplotlib inline

# Publication-quality plot defaults
plt.rcParams.update({
    'figure.figsize': (8, 6),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'lines.linewidth': 2,
    'figure.dpi': 100,
})
print("Imports loaded. Ready for linear algebra explorations.")""")

    # ---- MD: Why Linear Algebra ----
    nb.md("""\
## Why Linear Algebra IS the Language of Quantum Mechanics

Every quantum state is a **vector** in a complex Hilbert space. Every observable
is a **Hermitian matrix**. Every time-evolution is a **unitary transformation**.
Understanding eigenvalues, eigenvectors, and matrix decompositions is not merely
preparation for quantum mechanics --- it *is* quantum mechanics in its most
fundamental mathematical form.

**Key correspondences:**

| Linear Algebra Concept | Quantum Mechanics Meaning |
|------------------------|--------------------------|
| Vector $\\|v\\rangle$ | Quantum state |
| Matrix $A$ | Observable or operator |
| Eigenvalue $\\lambda$ | Measurement outcome |
| Eigenvector $\\|\\lambda\\rangle$ | State after measurement |
| Unitary matrix $U$ | Time evolution |
| Inner product $\\langle u \\| v \\rangle$ | Probability amplitude |

In this notebook we build fluency with all of these concepts computationally.""")

    # ---- MD: Matrix Operations ----
    nb.md("""\
## 1. Matrix Operations Review

A matrix $A \\in \\mathbb{R}^{m \\times n}$ is a rectangular array of numbers.
Key operations:

- **Multiplication:** $(AB)_{ij} = \\sum_k A_{ik} B_{kj}$
- **Transpose:** $(A^T)_{ij} = A_{ji}$
- **Determinant** (square matrices): $\\det(A)$ encodes volume scaling
- **Inverse:** $A^{-1}$ exists iff $\\det(A) \\neq 0$

$$\\det\\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix} = ad - bc$$""")

    # ---- Code: Matrix Operations ----
    nb.code("""\
# --- Matrix Operations: multiplication, transpose, determinant, inverse ---

A = np.array([[2, 1],
              [1, 3]])
B = np.array([[1, -1],
              [0,  2]])

print("Matrix A:")
print(A)
print("\\nMatrix B:")
print(B)

# Matrix multiplication (two equivalent ways)
AB = A @ B
print("\\nA @ B (matrix product):")
print(AB)

# Transpose
print("\\nA^T (transpose):")
print(A.T)

# Determinant
det_A = det(A)
print(f"\\ndet(A) = {det_A:.4f}")
print(f"det(B) = {det(B):.4f}")
print(f"det(AB) = {det(AB):.4f}")
print(f"det(A) * det(B) = {det_A * det(B):.4f}  (should match det(AB))")

# Inverse
A_inv = inv(A)
print("\\nA^{-1}:")
print(A_inv)
print("\\nVerification A @ A^{-1} = I:")
print(np.round(A @ A_inv, 10))""")

    # ---- MD: Eigenvalue Theory ----
    nb.md("""\
## 2. Eigenvalue Decomposition

An eigenvector $\\mathbf{v}$ of matrix $A$ satisfies:

$$A\\mathbf{v} = \\lambda \\mathbf{v}$$

where $\\lambda$ is the corresponding **eigenvalue**. Geometrically, $A$ stretches
$\\mathbf{v}$ by factor $\\lambda$ without changing its direction.

For a diagonalizable matrix: $A = P \\Lambda P^{-1}$ where $\\Lambda$ is diagonal
with eigenvalues and $P$ contains eigenvectors as columns.

**In QM:** measuring an observable $\\hat{A}$ on eigenstate $|\\lambda\\rangle$ always
yields the eigenvalue $\\lambda$. The spectral decomposition
$\\hat{A} = \\sum_i \\lambda_i |\\lambda_i\\rangle\\langle\\lambda_i|$ is foundational.""")

    # ---- Code: Eigenvalue Decomposition ----
    nb.code("""\
# --- Eigenvalue decomposition with verification ---

# Symmetric matrix (guaranteed real eigenvalues)
A = np.array([[4, 1],
              [1, 3]])
print("Matrix A:")
print(A)

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = eig(A)
print(f"\\nEigenvalues: {eigenvalues}")
print(f"\\nEigenvectors (as columns):")
print(eigenvectors)

# Verification: A @ v = lambda * v for each eigenpair
print("\\n--- Verification: A v = lambda v ---")
for i in range(len(eigenvalues)):
    lam = eigenvalues[i]
    v = eigenvectors[:, i]
    Av = A @ v
    lam_v = lam * v
    residual = norm(Av - lam_v)
    print(f"  lambda_{i} = {lam:.4f}")
    print(f"    A @ v_{i}     = {Av}")
    print(f"    lambda * v_{i} = {lam_v}")
    print(f"    ||Av - lv||   = {residual:.2e} {'PASS' if residual < 1e-10 else 'FAIL'}")

# Reconstruction: A = P Lambda P^{-1}
P = eigenvectors
Lambda = np.diag(eigenvalues)
A_reconstructed = P @ Lambda @ inv(P)
print(f"\\nReconstruction error ||A - P Lambda P^{{-1}}|| = {norm(A - A_reconstructed):.2e}")""")

    # ---- MD: Linear Transformations ----
    nb.md("""\
## 3. Visualizing 2D Linear Transformations

A $2 \\times 2$ matrix $A$ maps every point $(x, y)$ to a new point $(x', y')$.
The unit circle maps to an **ellipse** whose semi-axes are the **singular values**
of $A$, and whose principal directions are given by the singular vectors.

The eigenvectors of $A$ are the directions that remain unchanged (only scaled)
under the transformation.""")

    # ---- Code: Transformation Visualization ----
    nb.code("""\
# --- Visualize how a matrix transforms the unit circle ---

A = np.array([[2.0, 1.0],
              [0.5, 1.5]])

# Generate unit circle
theta = np.linspace(0, 2 * np.pi, 300)
circle = np.array([np.cos(theta), np.sin(theta)])  # shape (2, 300)

# Apply transformation
transformed = A @ circle  # shape (2, 300)

# Get eigenvalues/vectors for annotation
vals, vecs = eig(A)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Original unit circle
axes[0].plot(circle[0], circle[1], 'b-', linewidth=2, label='Unit circle')
axes[0].set_xlim(-3, 3)
axes[0].set_ylim(-3, 3)
axes[0].set_aspect('equal')
axes[0].grid(True, alpha=0.3)
axes[0].axhline(0, color='k', linewidth=0.5)
axes[0].axvline(0, color='k', linewidth=0.5)
axes[0].set_title('Original: Unit Circle')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')

# Transformed ellipse with eigenvectors
axes[1].plot(transformed[0], transformed[1], 'r-', linewidth=2, label='Transformed')
for i in range(2):
    v = vecs[:, i].real
    lam = vals[i].real
    # Plot eigenvector scaled by eigenvalue
    axes[1].annotate('', xy=lam * v, xytext=[0, 0],
                     arrowprops=dict(arrowstyle='->', color=f'C{i+2}', lw=2.5))
    axes[1].annotate(f'$\\lambda_{i+1}={lam:.2f}$',
                     xy=lam * v * 1.15, fontsize=11, color=f'C{i+2}', weight='bold')

axes[1].set_xlim(-4, 4)
axes[1].set_ylim(-4, 4)
axes[1].set_aspect('equal')
axes[1].grid(True, alpha=0.3)
axes[1].axhline(0, color='k', linewidth=0.5)
axes[1].axvline(0, color='k', linewidth=0.5)
axes[1].set_title(f'After transformation by A (det={det(A):.2f})')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')

plt.suptitle('Linear Transformation: Unit Circle to Ellipse', fontsize=15, y=1.02)
plt.tight_layout()
plt.show()

print(f"Eigenvalues: {vals}")
print(f"Area scaling factor (det A): {det(A):.4f}")""")

    # ---- MD: SVD Theory ----
    nb.md("""\
## 4. Singular Value Decomposition (SVD)

Every matrix $A \\in \\mathbb{R}^{m \\times n}$ admits a decomposition:

$$A = U \\Sigma V^T$$

where:
- $U \\in \\mathbb{R}^{m \\times m}$ is orthogonal (left singular vectors)
- $\\Sigma \\in \\mathbb{R}^{m \\times n}$ is diagonal with singular values $\\sigma_1 \\geq \\sigma_2 \\geq \\cdots \\geq 0$
- $V \\in \\mathbb{R}^{n \\times n}$ is orthogonal (right singular vectors)

**Applications:**
- Low-rank approximation (image compression)
- Principal Component Analysis
- Pseudoinverse computation

**QM Connection:** Schmidt decomposition of bipartite quantum states is the SVD
applied to the coefficient matrix, revealing entanglement structure.""")

    # ---- Code: SVD Image Compression ----
    nb.code("""\
# --- SVD Image Compression Demo ---
# Create a synthetic "image" (grayscale gradient + shapes)

np.random.seed(42)
N = 128

# Create a test image with geometric features
img = np.zeros((N, N))
# Gradient background
for i in range(N):
    for j in range(N):
        img[i, j] = 0.3 * (i + j) / (2 * N)
# Rectangle
img[20:50, 30:90] = 0.9
# Circle
yy, xx = np.ogrid[:N, :N]
circle_mask = (xx - 80)**2 + (yy - 80)**2 < 20**2
img[circle_mask] = 0.7
# Diagonal stripe
for i in range(N):
    j_start = max(0, i - 5)
    j_end = min(N, i + 5)
    img[i, j_start:j_end] = np.maximum(img[i, j_start:j_end], 0.5)

# Perform SVD
U, sigma, Vt = svd(img, full_matrices=False)
print(f"Image shape: {img.shape}")
print(f"Number of singular values: {len(sigma)}")
print(f"Top 10 singular values: {sigma[:10].round(2)}")

# Reconstruct with different ranks
ranks = [1, 5, 10, 20, 50, 128]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for ax, k in zip(axes.flat, ranks):
    # Rank-k approximation
    img_k = U[:, :k] @ np.diag(sigma[:k]) @ Vt[:k, :]
    # Compression ratio: original = N*N, compressed = k*(N+N+1)
    compression = (k * (N + N + 1)) / (N * N) * 100
    error = norm(img - img_k) / norm(img) * 100

    ax.imshow(img_k, cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'Rank {k}\\n{compression:.1f}% of data, {error:.1f}% error')
    ax.axis('off')

plt.suptitle('SVD Image Compression: Rank-k Approximations', fontsize=15)
plt.tight_layout()
plt.show()

# Plot singular value spectrum
fig, ax = plt.subplots(figsize=(10, 4))
ax.semilogy(sigma, 'o-', markersize=3)
ax.set_xlabel('Index $k$')
ax.set_ylabel('Singular value $\\sigma_k$')
ax.set_title('Singular Value Spectrum (log scale)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
print(f"\\nSigma_1 / Sigma_N ratio (condition number): {sigma[0]/sigma[-1]:.2f}")""")

    # ---- MD: Quantum Gates ----
    nb.md("""\
## 5. Quantum Gates as Matrices

Quantum gates are **unitary matrices** acting on qubit state vectors.
The most fundamental single-qubit gates are the **Pauli matrices** and
the **Hadamard gate**:

$$X = \\begin{pmatrix} 0 & 1 \\\\ 1 & 0 \\end{pmatrix}, \\quad
Y = \\begin{pmatrix} 0 & -i \\\\ i & 0 \\end{pmatrix}, \\quad
Z = \\begin{pmatrix} 1 & 0 \\\\ 0 & -1 \\end{pmatrix}$$

$$H = \\frac{1}{\\sqrt{2}}\\begin{pmatrix} 1 & 1 \\\\ 1 & -1 \\end{pmatrix}$$

**Properties:**
- All Pauli matrices are **Hermitian** ($X = X^\\dagger$) and **unitary** ($XX^\\dagger = I$)
- Eigenvalues of Pauli matrices are $\\pm 1$ (measurement outcomes)
- $H$ creates superposition: $H|0\\rangle = \\frac{|0\\rangle + |1\\rangle}{\\sqrt{2}}$""")

    # ---- Code: Quantum Gates ----
    nb.code("""\
# --- Quantum Gates: Pauli matrices and Hadamard ---

# Define the standard qubit basis
ket_0 = np.array([1, 0], dtype=complex)  # |0>
ket_1 = np.array([0, 1], dtype=complex)  # |1>

# Pauli matrices
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)    # Bit flip
Y = np.array([[0, -1j], [1j, 0]], dtype=complex) # Bit + phase flip
Z = np.array([[1, 0], [0, -1]], dtype=complex)    # Phase flip
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)  # Hadamard

gates = {'I': I2, 'X': X, 'Y': Y, 'Z': Z, 'H': H}

print("=== Quantum Gate Properties ===\\n")
for name, G in gates.items():
    vals, vecs = eig(G)
    is_hermitian = np.allclose(G, G.conj().T)
    is_unitary = np.allclose(G @ G.conj().T, I2)

    print(f"--- {name} Gate ---")
    print(f"  Matrix:\\n{G}")
    print(f"  Eigenvalues:  {np.round(vals, 4)}")
    print(f"  Hermitian:    {is_hermitian}")
    print(f"  Unitary:      {is_unitary}")
    print(f"  det = {det(G):.4f}")
    print()

# Demonstrate Hadamard creating superposition
print("=== Hadamard on |0> ===")
psi = H @ ket_0
print(f"H|0> = {psi}  =  (|0> + |1>)/sqrt(2)")
print(f"Probabilities: |<0|psi>|^2 = {abs(psi[0])**2:.4f}, |<1|psi>|^2 = {abs(psi[1])**2:.4f}")

# Show Pauli algebra: XY = iZ, etc.
print("\\n=== Pauli Algebra ===")
print(f"XY = iZ?  {np.allclose(X @ Y, 1j * Z)}")
print(f"YZ = iX?  {np.allclose(Y @ Z, 1j * X)}")
print(f"ZX = iY?  {np.allclose(Z @ X, 1j * Y)}")
print(f"X^2 = I?  {np.allclose(X @ X, I2)}")""")

    # ---- MD: Gram-Schmidt ----
    nb.md("""\
## 6. Gram-Schmidt Orthogonalization

Given a set of linearly independent vectors $\\{\\mathbf{v}_1, \\dots, \\mathbf{v}_n\\}$,
the Gram-Schmidt process produces an **orthonormal** set $\\{\\mathbf{e}_1, \\dots, \\mathbf{e}_n\\}$:

$$\\mathbf{u}_k = \\mathbf{v}_k - \\sum_{j=1}^{k-1} \\frac{\\langle \\mathbf{v}_k, \\mathbf{e}_j \\rangle}{\\langle \\mathbf{e}_j, \\mathbf{e}_j \\rangle} \\mathbf{e}_j, \\qquad \\mathbf{e}_k = \\frac{\\mathbf{u}_k}{\\|\\mathbf{u}_k\\|}$$

**QM Connection:** Orthonormal bases are fundamental. Measurement postulate requires
orthogonal projectors. Every quantum computation begins with choosing an orthonormal basis.""")

    # ---- Code: Gram-Schmidt ----
    nb.code("""\
# --- Gram-Schmidt Orthogonalization from scratch ---

def gram_schmidt(V):
    \"\"\"
    Gram-Schmidt orthogonalization.

    Parameters
    ----------
    V : ndarray, shape (n, k)
        k column vectors of dimension n (must be linearly independent).

    Returns
    -------
    Q : ndarray, shape (n, k)
        Orthonormal columns spanning the same space as V.
    \"\"\"
    n, k = V.shape
    Q = np.zeros_like(V, dtype=float)

    for j in range(k):
        # Start with the original vector
        u = V[:, j].astype(float).copy()

        # Subtract projections onto all previous orthonormal vectors
        for i in range(j):
            u -= np.dot(Q[:, i], V[:, j]) * Q[:, i]

        # Normalize
        u_norm = norm(u)
        if u_norm < 1e-12:
            raise ValueError(f"Vectors are linearly dependent at index {j}")
        Q[:, j] = u / u_norm

    return Q


# Test with 3D vectors
V = np.array([
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1]
], dtype=float).T  # Each column is a vector

print("Original vectors (as columns):")
print(V)

Q = gram_schmidt(V)
print("\\nOrthonormalized vectors (as columns):")
print(np.round(Q, 6))

# Verify orthonormality: Q^T Q should be identity
QTQ = Q.T @ Q
print("\\nQ^T Q (should be identity):")
print(np.round(QTQ, 10))
print(f"\\nMax deviation from identity: {np.max(np.abs(QTQ - np.eye(3))):.2e}")

# Compare with NumPy's QR decomposition (which uses Gram-Schmidt internally)
Q_np, R_np = qr(V)
# qr may flip signs; compare absolute values of columns
print("\\n--- Comparison with np.linalg.qr ---")
for i in range(3):
    # Columns may differ by a sign
    match = np.allclose(Q[:, i], Q_np[:, i]) or np.allclose(Q[:, i], -Q_np[:, i])
    print(f"  Column {i} matches QR: {match}")""")

    # ---- Code: Eigenvalue Visualization ----
    nb.code("""\
# --- Visualize eigenvalue spectrum of a random symmetric matrix ---

np.random.seed(0)
N_dim = 50
# Create random symmetric matrix (guaranteed real eigenvalues)
M = np.random.randn(N_dim, N_dim)
M_sym = (M + M.T) / 2  # Symmetrize

eigenvalues_sym = np.sort(np.linalg.eigvalsh(M_sym))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram of eigenvalues (approaches Wigner semicircle for large N)
axes[0].hist(eigenvalues_sym, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].set_xlabel('Eigenvalue $\\lambda$')
axes[0].set_ylabel('Count')
axes[0].set_title(f'Eigenvalue Distribution ({N_dim}x{N_dim} Random Symmetric)')
axes[0].axvline(0, color='red', linestyle='--', alpha=0.5)
axes[0].grid(True, alpha=0.3)

# Eigenvalue staircase (cumulative distribution)
axes[1].step(eigenvalues_sym, np.arange(1, N_dim + 1) / N_dim, where='post',
             linewidth=2, color='darkgreen')
axes[1].set_xlabel('Eigenvalue $\\lambda$')
axes[1].set_ylabel('Cumulative fraction')
axes[1].set_title('Eigenvalue Cumulative Distribution')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Eigenvalue range: [{eigenvalues_sym[0]:.3f}, {eigenvalues_sym[-1]:.3f}]")
print(f"Trace = sum of eigenvalues = {np.trace(M_sym):.4f} vs {np.sum(eigenvalues_sym):.4f}")""")

    # ---- MD: Summary ----
    nb.md("""\
## Summary

| Concept | Key Formula | NumPy Function |
|---------|-------------|----------------|
| Eigendecomposition | $Av = \\lambda v$ | `np.linalg.eig(A)` |
| SVD | $A = U\\Sigma V^T$ | `np.linalg.svd(A)` |
| Determinant | $\\det(A)$ | `np.linalg.det(A)` |
| Gram-Schmidt | $e_k = u_k / \\|u_k\\|$ | `np.linalg.qr(A)` |
| Matrix exponential | $e^{A}$ | `scipy.linalg.expm(A)` |

**Key Takeaways:**
1. Eigenvalues determine how a matrix scales along special directions
2. SVD generalizes eigendecomposition to non-square matrices
3. Quantum gates are unitary matrices; their eigenvalues have unit modulus
4. Pauli matrices form a basis for all 2x2 Hermitian matrices
5. Gram-Schmidt converts any basis to an orthonormal one

---
*Next: Month 05 explores complex vector spaces, Hermitian/unitary matrices, and the spectral theorem.*""")

    nb.save()
    print("  [OK] Month 04 notebook generated.\n")


# ============================================================================
# NOTEBOOK 2: Month 05 — Complex Analysis Foundations
# ============================================================================
def build_month_05():
    nb = NotebookBuilder(
        "Complex Analysis Foundations",
        "year_0/month_05_linear_algebra_II_complex/05_complex_analysis_foundations.ipynb",
        "Days 113-140",
    )

    # ---- Imports & Setup ----
    nb.code("""\
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import Circle
from numpy.linalg import eig, norm, det, inv
import cmath
%matplotlib inline

# Publication-quality plot defaults
plt.rcParams.update({
    'figure.figsize': (8, 6),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'lines.linewidth': 2,
    'figure.dpi': 100,
})
print("Imports loaded. Ready for complex analysis explorations.")""")

    # ---- MD: Complex Numbers in QM ----
    nb.md("""\
## Why Complex Numbers are Essential for Quantum Mechanics

Quantum mechanics is **inherently complex-valued**. The Schrodinger equation

$$i\\hbar \\frac{\\partial}{\\partial t}|\\psi\\rangle = \\hat{H}|\\psi\\rangle$$

contains $i = \\sqrt{-1}$ explicitly. This is not a mathematical convenience ---
recent experiments (2021-2022) have confirmed that **real-valued quantum theory
cannot reproduce all quantum predictions**.

**Complex numbers provide:**
- **Phase information** --- interference depends on relative phases
- **Unitary evolution** --- $U = e^{-iHt/\\hbar}$ requires complex exponentials
- **Probability amplitudes** --- $\\psi(x) \\in \\mathbb{C}$, with $|\\psi(x)|^2$ giving probability

This notebook builds fluency with complex arithmetic, Euler's formula,
complex function mappings, and the linear algebra structures (Hermitian, unitary
matrices) that form the mathematical backbone of quantum theory.""")

    # ---- MD: Complex Number Basics ----
    nb.md("""\
## 1. Complex Numbers on the Argand Plane

A complex number $z = a + bi$ can be represented as a point $(a, b)$ in the
**Argand plane** (complex plane). Equivalently, in polar form:

$$z = r e^{i\\theta} = r(\\cos\\theta + i\\sin\\theta)$$

where $r = |z| = \\sqrt{a^2 + b^2}$ is the **modulus** and
$\\theta = \\arg(z) = \\text{atan2}(b, a)$ is the **argument**.""")

    # ---- Code: Argand Plane ----
    nb.code("""\
# --- Complex number visualization on the Argand plane ---

# Define several complex numbers
z_list = [
    (2 + 3j,   'z_1 = 2+3i'),
    (-1 + 2j,  'z_2 = -1+2i'),
    (3 - 1j,   'z_3 = 3-i'),
    (-2 - 2j,  'z_4 = -2-2i'),
]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: individual complex numbers
ax = axes[0]
for z, label in z_list:
    ax.annotate('', xy=(z.real, z.imag), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', lw=2, color='steelblue'))
    ax.plot(z.real, z.imag, 'o', markersize=8, color='darkred')
    ax.annotate(label, xy=(z.real, z.imag), xytext=(5, 5),
                textcoords='offset points', fontsize=10)

ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.axhline(0, color='k', linewidth=0.5)
ax.axvline(0, color='k', linewidth=0.5)
ax.set_xlabel('Re(z)')
ax.set_ylabel('Im(z)')
ax.set_title('Complex Numbers on the Argand Plane')

# Right: polar form visualization
ax = axes[1]
z = 3 + 2j
r = abs(z)
theta = cmath.phase(z)

# Draw the vector
ax.annotate('', xy=(z.real, z.imag), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='darkblue'))
ax.plot(z.real, z.imag, 'o', markersize=10, color='red')

# Draw the angle arc
arc_theta = np.linspace(0, theta, 50)
arc_r = 0.8
ax.plot(arc_r * np.cos(arc_theta), arc_r * np.sin(arc_theta), 'g-', linewidth=2)
ax.annotate(f'$\\\\theta = {np.degrees(theta):.1f}°$',
            xy=(0.9, 0.3), fontsize=12, color='green')

# Draw projections
ax.plot([z.real, z.real], [0, z.imag], 'k--', alpha=0.4)
ax.plot([0, z.real], [z.imag, z.imag], 'k--', alpha=0.4)

ax.annotate(f'$z = {z.real:.0f}+{z.imag:.0f}i$\\n$r = |z| = {r:.3f}$',
            xy=(z.real + 0.1, z.imag + 0.2), fontsize=11)

ax.set_xlim(-1, 5)
ax.set_ylim(-1, 4)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.axhline(0, color='k', linewidth=0.5)
ax.axvline(0, color='k', linewidth=0.5)
ax.set_xlabel('Re(z)')
ax.set_ylabel('Im(z)')
ax.set_title('Polar Form: $z = re^{i\\\\theta}$')

plt.tight_layout()
plt.show()

# Print properties
print("Complex number properties:")
for z, label in z_list:
    print(f"  {label}: |z| = {abs(z):.3f}, arg(z) = {np.degrees(cmath.phase(z)):.1f} deg, "
          f"conjugate = {z.conjugate()}")""")

    # ---- MD: Euler's Formula ----
    nb.md("""\
## 2. Euler's Formula: $e^{i\\theta} = \\cos\\theta + i\\sin\\theta$

This is arguably the most important formula connecting algebra, geometry,
and analysis. It shows that:

- **Complex exponentials trace the unit circle** as $\\theta$ varies
- **Multiplication by $e^{i\\theta}$** is a rotation by angle $\\theta$
- Setting $\\theta = \\pi$: $e^{i\\pi} + 1 = 0$ (Euler's identity)

**QM Connection:** Time evolution of an energy eigenstate:
$$|\\psi(t)\\rangle = e^{-iEt/\\hbar}|E\\rangle$$
The state rotates in the complex plane with angular frequency $\\omega = E/\\hbar$.""")

    # ---- Code: Euler's Formula ----
    nb.code("""\
# --- Euler's formula visualization: e^{i theta} on the unit circle ---

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: unit circle traced by e^{i*theta}
ax = axes[0]
theta = np.linspace(0, 2 * np.pi, 500)
z_circle = np.exp(1j * theta)
ax.plot(z_circle.real, z_circle.imag, 'b-', linewidth=2, label='$e^{i\\\\theta}$')

# Mark special angles
special = {
    0: '$1$', np.pi/6: '$e^{i\\\\pi/6}$', np.pi/4: '$e^{i\\\\pi/4}$',
    np.pi/3: '$e^{i\\\\pi/3}$', np.pi/2: '$i$',
    np.pi: '$-1$', 3*np.pi/2: '$-i$'
}
for ang, label in special.items():
    z = np.exp(1j * ang)
    ax.plot(z.real, z.imag, 'ro', markersize=8)
    offset = (10, 10) if ang < np.pi else (-30, -15)
    ax.annotate(label, xy=(z.real, z.imag), xytext=offset,
                textcoords='offset points', fontsize=10)

ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.axhline(0, color='k', linewidth=0.5)
ax.axvline(0, color='k', linewidth=0.5)
ax.set_xlabel('Re')
ax.set_ylabel('Im')
ax.set_title("Euler's Formula: $e^{i\\\\theta}$ traces the unit circle")

# Right: cos and sin as real/imaginary parts
ax = axes[1]
theta_range = np.linspace(0, 4 * np.pi, 500)
z_evolving = np.exp(1j * theta_range)

ax.plot(theta_range, z_evolving.real, 'b-', linewidth=2, label='$\\cos(\\\\theta) = \\mathrm{Re}(e^{i\\\\theta})$')
ax.plot(theta_range, z_evolving.imag, 'r-', linewidth=2, label='$\\sin(\\\\theta) = \\mathrm{Im}(e^{i\\\\theta})$')
ax.set_xlabel('$\\\\theta$ (radians)')
ax.set_ylabel('Value')
ax.set_title('Real and Imaginary parts of $e^{i\\\\theta}$')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 4 * np.pi)

plt.tight_layout()
plt.show()

# Verify Euler's identity
euler_identity = np.exp(1j * np.pi) + 1
print(f"Euler's identity: e^(i*pi) + 1 = {euler_identity:.2e}  (should be ~0)")
print(f"Verification: e^(i*pi/4) = {np.exp(1j*np.pi/4):.6f}")
print(f"Expected:     cos(pi/4) + i*sin(pi/4) = {np.cos(np.pi/4):.6f} + {np.sin(np.pi/4):.6f}i")""")

    # ---- MD: Complex Function Mappings ----
    nb.md("""\
## 3. Complex Function Mappings with Domain Coloring

A function $f: \\mathbb{C} \\to \\mathbb{C}$ maps every point in one complex plane
to another. **Domain coloring** visualizes this by coloring the input plane:
- **Hue** encodes the argument (phase) of $f(z)$
- **Brightness** encodes the modulus $|f(z)|$

We explore three fundamental mappings:
- $f(z) = z^2$ --- doubles angles, squares distances
- $f(z) = 1/z$ --- inversion (maps inside/outside the unit circle)
- $f(z) = e^z$ --- maps vertical lines to circles, horizontal lines to rays""")

    # ---- Code: Domain Coloring ----
    nb.code("""\
# --- Domain coloring for complex functions ---

def domain_coloring(f, extent=(-2, 2, -2, 2), N=500, title="f(z)"):
    \"\"\"
    Domain coloring visualization of a complex function f.
    Hue = arg(f(z)), Brightness = |f(z)| (compressed with log).
    \"\"\"
    x = np.linspace(extent[0], extent[1], N)
    y = np.linspace(extent[2], extent[3], N)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    # Evaluate function (handle division by zero gracefully)
    with np.errstate(divide='ignore', invalid='ignore'):
        W = f(Z)

    # Hue from argument (0 to 1)
    H = (np.angle(W) + np.pi) / (2 * np.pi)

    # Saturation: constant
    S = np.ones_like(H) * 0.9

    # Value (brightness) from modulus, compressed with log
    modulus = np.abs(W)
    V = 1 - 1 / (1 + modulus**0.3)  # Smooth compression

    # Handle NaN/inf
    H = np.nan_to_num(H, nan=0.0)
    V = np.nan_to_num(V, nan=0.5)

    # Convert HSV to RGB
    HSV = np.stack([H, S, V], axis=-1)
    RGB = hsv_to_rgb(HSV)

    return RGB, extent


# Create domain coloring for three functions
functions = [
    (lambda z: z**2,     "$f(z) = z^2$",     (-2, 2, -2, 2)),
    (lambda z: 1/z,      "$f(z) = 1/z$",     (-2, 2, -2, 2)),
    (lambda z: np.exp(z), "$f(z) = e^z$",     (-3, 3, -3, 3)),
]

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

for ax, (f, title, extent) in zip(axes, functions):
    RGB, ext = domain_coloring(f, extent=extent, title=title)
    ax.imshow(RGB, extent=ext, origin='lower', aspect='equal')
    ax.set_xlabel('Re(z)')
    ax.set_ylabel('Im(z)')
    ax.set_title(title, fontsize=14)

    # Add unit circle for reference
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'w--', alpha=0.5, linewidth=1)

plt.suptitle('Domain Coloring of Complex Functions', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()
print("Hue = phase of f(z), Brightness = |f(z)| (compressed)")
print("White dashed circle = unit circle |z| = 1")""")

    # ---- MD: Inner Products ----
    nb.md("""\
## 4. Inner Products in Complex Vector Spaces

In a complex vector space, the inner product is **sesquilinear**:

$$\\langle u, v \\rangle = \\sum_i \\overline{u_i} \\, v_i = u^\\dagger v$$

Note the **conjugation** on the first argument (physics convention). Key properties:
- $\\langle u, v \\rangle = \\overline{\\langle v, u \\rangle}$ (conjugate symmetry)
- $\\langle u, u \\rangle \\geq 0$ with equality iff $u = 0$ (positive definiteness)
- $\\langle u, \\alpha v + \\beta w \\rangle = \\alpha\\langle u, v\\rangle + \\beta\\langle u, w\\rangle$ (linearity in second argument)

**QM Connection:** The probability amplitude for transitioning from state $|\\psi\\rangle$
to $|\\phi\\rangle$ is $\\langle\\phi|\\psi\\rangle$, and the probability is $|\\langle\\phi|\\psi\\rangle|^2$.""")

    # ---- Code: Complex Inner Products ----
    nb.code("""\
# --- Complex inner products and orthogonality ---

# Define complex vectors (qubit states)
# |+> and |-> states (Hadamard basis)
ket_plus  = np.array([1, 1], dtype=complex) / np.sqrt(2)
ket_minus = np.array([1, -1], dtype=complex) / np.sqrt(2)
# |+i> and |-i> states (Y basis)
ket_plus_i  = np.array([1, 1j], dtype=complex) / np.sqrt(2)
ket_minus_i = np.array([1, -1j], dtype=complex) / np.sqrt(2)

states = {
    '|+>': ket_plus,
    '|->': ket_minus,
    '|+i>': ket_plus_i,
    '|-i>': ket_minus_i,
}

print("=== Complex Inner Products <phi|psi> ===\\n")
print(f"{'':8s}", end='')
for name in states:
    print(f"{name:>12s}", end='')
print()

for name_i, psi in states.items():
    print(f"{name_i:8s}", end='')
    for name_j, phi in states.items():
        # Inner product: <phi|psi> = phi^dagger @ psi
        ip = np.vdot(phi, psi)  # np.vdot conjugates the first argument
        print(f"{ip.real:+.3f}{ip.imag:+.3f}i", end='  ')
    print()

print("\\n=== Orthogonality checks ===")
print(f"<+|-> = {np.vdot(ket_plus, ket_minus):.6f}  (orthogonal: should be 0)")
print(f"<+i|-i> = {np.vdot(ket_plus_i, ket_minus_i):.6f}  (orthogonal: should be 0)")
print(f"<+|+i> = {np.vdot(ket_plus, ket_plus_i):.6f}  (NOT orthogonal)")

print("\\n=== Norm (should all be 1 for normalized states) ===")
for name, v in states.items():
    print(f"  ||{name}|| = {norm(v):.6f}")""")

    # ---- MD: Hermitian and Unitary ----
    nb.md("""\
## 5. Hermitian and Unitary Matrices

Two classes of matrices are central to quantum mechanics:

**Hermitian matrices** ($A = A^\\dagger$):
- Eigenvalues are **real** (measurement outcomes)
- Eigenvectors for distinct eigenvalues are **orthogonal**
- Represent **observables** in QM

**Unitary matrices** ($UU^\\dagger = U^\\dagger U = I$):
- Eigenvalues have **unit modulus** ($|\\lambda| = 1$)
- Preserve inner products: $\\langle U\\psi | U\\phi \\rangle = \\langle \\psi | \\phi \\rangle$
- Represent **time evolution** and **quantum gates**

**Spectral Theorem:** Every Hermitian matrix can be diagonalized by a unitary:
$$A = U \\Lambda U^\\dagger, \\qquad \\Lambda = \\text{diag}(\\lambda_1, \\dots, \\lambda_n)$$""")

    # ---- Code: Hermitian & Unitary ----
    nb.code("""\
# --- Hermitian and Unitary matrix properties ---

# Construct a Hermitian matrix
H = np.array([
    [2,    1+1j, 0   ],
    [1-1j, 3,    2-1j],
    [0,    2+1j, 1   ]
], dtype=complex)

print("=== Hermitian Matrix H ===")
print(H)
print(f"\\nIs Hermitian (H = H^dag)? {np.allclose(H, H.conj().T)}")

# Eigendecomposition
eigenvalues, eigenvectors = eig(H)
print(f"\\nEigenvalues: {np.round(eigenvalues.real, 6)}")
print(f"Are eigenvalues real? Max imaginary part: {np.max(np.abs(eigenvalues.imag)):.2e}")

# Check orthogonality of eigenvectors
print("\\nEigenvector orthogonality (should be ~identity):")
overlap = eigenvectors.conj().T @ eigenvectors
print(np.round(overlap, 6))

# Construct a Unitary matrix from a Hermitian (U = e^{iH})
# For small matrices, use eigendecomposition
U = eigenvectors @ np.diag(np.exp(1j * eigenvalues.real)) @ eigenvectors.conj().T
print("\\n=== Unitary Matrix U = e^{iH} ===")
print(np.round(U, 4))
print(f"\\nIs Unitary (UU^dag = I)? {np.allclose(U @ U.conj().T, np.eye(3))}")

# Eigenvalues of unitary should have |lambda| = 1
u_vals = eig(U)[0]
print(f"\\nEigenvalues of U: {np.round(u_vals, 4)}")
print(f"|lambda| values:  {np.round(np.abs(u_vals), 6)}")
print(f"All unit modulus?  {np.allclose(np.abs(u_vals), 1.0)}")""")

    # ---- Code: Spectral Theorem ----
    nb.code("""\
# --- Spectral Theorem Demonstration ---
# For Hermitian A: A = sum_i lambda_i |v_i><v_i|

A = np.array([
    [5, 2-1j],
    [2+1j, 3]
], dtype=complex)

print("Hermitian matrix A:")
print(A)
print(f"Is Hermitian? {np.allclose(A, A.conj().T)}")

# Eigendecomposition
eigenvalues, eigenvectors = eig(A)
print(f"\\nEigenvalues: {np.round(eigenvalues.real, 6)}")

# Spectral decomposition: A = sum lambda_i * |v_i><v_i|
A_reconstructed = np.zeros_like(A)
print("\\n--- Spectral Decomposition ---")
for i in range(len(eigenvalues)):
    lam = eigenvalues[i].real
    v = eigenvectors[:, i:i+1]  # Column vector
    projector = v @ v.conj().T  # |v><v| (outer product)
    A_reconstructed += lam * projector
    print(f"\\n  lambda_{i} = {lam:.4f}")
    print(f"  |v_{i}> = {v.flatten().round(4)}")
    print(f"  |v_{i}><v_{i}| =")
    print(f"  {np.round(projector, 4)}")

print(f"\\nReconstruction: A = sum lambda_i |v_i><v_i|")
print(f"Reconstruction error: {norm(A - A_reconstructed):.2e}")

# Projectors are idempotent and complete
P0 = eigenvectors[:, 0:1] @ eigenvectors[:, 0:1].conj().T
P1 = eigenvectors[:, 1:2] @ eigenvectors[:, 1:2].conj().T
print(f"\\nP0^2 = P0? {np.allclose(P0 @ P0, P0)}")
print(f"P1^2 = P1? {np.allclose(P1 @ P1, P1)}")
print(f"P0 + P1 = I? {np.allclose(P0 + P1, np.eye(2))}")
print(f"P0 P1 = 0? {np.allclose(P0 @ P1, np.zeros((2,2)))}")""")

    # ---- MD: Summary ----
    nb.md("""\
## Summary

| Concept | Key Property | QM Significance |
|---------|-------------|-----------------|
| Complex numbers | $z = re^{i\\theta}$ | Probability amplitudes |
| Euler's formula | $e^{i\\theta} = \\cos\\theta + i\\sin\\theta$ | Time evolution phases |
| Inner product | $\\langle u,v\\rangle = u^\\dagger v$ | Transition amplitudes |
| Hermitian matrix | $A = A^\\dagger$, real eigenvalues | Observables |
| Unitary matrix | $UU^\\dagger = I$, $|\\lambda|=1$ | Quantum gates, evolution |
| Spectral theorem | $A = \\sum \\lambda_i |v_i\\rangle\\langle v_i|$ | Measurement theory |

**Key Takeaways:**
1. Complex numbers are not optional in QM --- they are structurally necessary
2. Euler's formula unifies exponentials, trigonometry, and rotation
3. Domain coloring reveals the rich geometry of complex functions
4. Hermitian matrices guarantee real measurement outcomes
5. Unitary matrices preserve probability (norm) during evolution
6. The spectral theorem is the mathematical core of quantum measurement

---
*Next: Month 06 bridges classical and quantum mechanics via Lagrangian and Hamiltonian formulations.*""")

    nb.save()
    print("  [OK] Month 05 notebook generated.\n")


# ============================================================================
# NOTEBOOK 3: Month 06 — Lagrangian & Hamiltonian Mechanics
# ============================================================================
def build_month_06():
    nb = NotebookBuilder(
        "Lagrangian and Hamiltonian Mechanics",
        "year_0/month_06_classical_mechanics/06_lagrangian_hamiltonian.ipynb",
        "Days 141-168",
    )

    # ---- Imports & Setup ----
    nb.code("""\
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import sympy as sp
from sympy import symbols, Function, cos, sin, diff, simplify, latex, Matrix
from sympy import solve, Rational, pi, sqrt, Eq, pprint
from scipy.integrate import solve_ivp
%matplotlib inline

# Publication-quality plot defaults
plt.rcParams.update({
    'figure.figsize': (8, 6),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'lines.linewidth': 2,
    'figure.dpi': 100,
})
print("Imports loaded. Ready for classical mechanics explorations.")""")

    # ---- MD: Classical to Quantum Bridge ----
    nb.md("""\
## From Classical to Quantum: Why Hamiltonian Mechanics Matters

Classical mechanics has three equivalent formulations:
1. **Newtonian:** $\\mathbf{F} = m\\mathbf{a}$ (forces and accelerations)
2. **Lagrangian:** $\\mathcal{L} = T - V$ (energy-based, generalized coordinates)
3. **Hamiltonian:** $H = T + V$ (phase space, canonical coordinates)

The **Hamiltonian formulation** is the direct bridge to quantum mechanics:

| Classical (Hamiltonian) | Quantum Mechanics |
|-------------------------|-------------------|
| $H(q, p)$ | $\\hat{H}$ (Hamiltonian operator) |
| $\\{f, g\\}_{\\text{Poisson}}$ | $\\frac{1}{i\\hbar}[\\hat{f}, \\hat{g}]$ (commutator) |
| Phase space $(q, p)$ | Hilbert space $|\\psi\\rangle$ |
| Hamilton's equations | Heisenberg equations of motion |
| $\\frac{df}{dt} = \\{f, H\\}$ | $\\frac{d\\hat{f}}{dt} = \\frac{1}{i\\hbar}[\\hat{f}, \\hat{H}]$ |

This notebook develops all three formulations, culminating in the Poisson bracket --
commutator correspondence that is the classical-quantum bridge.""")

    # ---- MD: Newton's Laws ----
    nb.md("""\
## 1. Newton's Laws: Projectile with Air Resistance

The simplest formulation: $\\mathbf{F} = m\\mathbf{a}$.

For a projectile with quadratic air resistance (drag):
$$m\\ddot{x} = -b\\dot{x}|\\mathbf{v}|, \\qquad m\\ddot{y} = -mg - b\\dot{y}|\\mathbf{v}|$$

where $|\\mathbf{v}| = \\sqrt{\\dot{x}^2 + \\dot{y}^2}$ and $b$ is the drag coefficient.

This ODE system has no closed-form solution, so we solve it numerically.""")

    # ---- Code: Projectile ----
    nb.code("""\
# --- Projectile motion with and without air resistance ---

def projectile_ode(t, state, g=9.81, b_over_m=0.0):
    \"\"\"
    ODE for 2D projectile with quadratic drag.
    state = [x, y, vx, vy]
    b_over_m = drag coefficient / mass
    \"\"\"
    x, y, vx, vy = state
    v = np.sqrt(vx**2 + vy**2)
    drag_x = -b_over_m * vx * v
    drag_y = -b_over_m * vy * v
    return [vx, vy, drag_x, -g + drag_y]

# Initial conditions: 45-degree launch at 30 m/s
v0 = 30.0
angle = np.radians(45)
y0 = [0, 0, v0 * np.cos(angle), v0 * np.sin(angle)]

# Solve for different drag coefficients
t_span = (0, 8)
t_eval = np.linspace(0, 8, 1000)

# Event to detect ground contact (y = 0 after launch)
def hit_ground(t, state, g=9.81, b_over_m=0.0):
    return state[1]  # y coordinate
hit_ground.terminal = True
hit_ground.direction = -1

drag_values = [0.0, 0.01, 0.03, 0.05]
colors = ['blue', 'green', 'orange', 'red']

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

for b, color in zip(drag_values, colors):
    # Create a closure for the event function with the right drag value
    def event_func(t, state, g=9.81, b_over_m=b):
        return state[1]
    event_func.terminal = True
    event_func.direction = -1

    sol = solve_ivp(
        lambda t, y: projectile_ode(t, y, b_over_m=b),
        t_span, y0, t_eval=t_eval, events=event_func,
        max_step=0.01, dense_output=True
    )

    # Mask out underground portions
    mask = sol.y[1] >= -0.1
    label = f"b/m = {b}" if b > 0 else "No drag"

    axes[0].plot(sol.y[0][mask], sol.y[1][mask], '-', color=color, label=label)

    # Speed vs time
    speed = np.sqrt(sol.y[2]**2 + sol.y[3]**2)
    axes[1].plot(sol.t[mask], speed[mask], '-', color=color, label=label)

# Trajectory plot
axes[0].set_xlabel('Horizontal distance (m)')
axes[0].set_ylabel('Height (m)')
axes[0].set_title('Projectile Trajectories')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(bottom=-1)

# Speed plot
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Speed (m/s)')
axes[1].set_title('Speed vs Time')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Compare ranges
print("=== Range comparison ===")
for b in drag_values:
    sol = solve_ivp(
        lambda t, y, b=b: projectile_ode(t, y, b_over_m=b),
        t_span, y0, events=hit_ground, max_step=0.01
    )
    x_range = sol.y[0, -1] if len(sol.y[0]) > 0 else 0
    t_flight = sol.t[-1]
    label = f"b/m={b}" if b > 0 else "no drag"
    print(f"  {label:>10s}: range = {x_range:.2f} m, flight time = {t_flight:.2f} s")""")

    # ---- MD: Lagrangian Mechanics ----
    nb.md("""\
## 2. Lagrangian Mechanics

The Lagrangian $\\mathcal{L} = T - V$ encodes the dynamics of a system.
The **Euler-Lagrange equation** yields the equations of motion:

$$\\frac{d}{dt}\\frac{\\partial \\mathcal{L}}{\\partial \\dot{q}_i} - \\frac{\\partial \\mathcal{L}}{\\partial q_i} = 0$$

**Advantages over Newton:**
- Works in *any* coordinate system (generalized coordinates)
- Constraints are handled naturally
- Symmetries directly imply conservation laws (Noether's theorem)

**Example:** Simple pendulum with generalized coordinate $\\theta$:
$$\\mathcal{L} = \\frac{1}{2}ml^2\\dot{\\theta}^2 - mgl(1 - \\cos\\theta)$$""")

    # ---- Code: Lagrangian with SymPy ----
    nb.code("""\
# --- Lagrangian mechanics: derive equations of motion symbolically ---

# Define symbols
t = symbols('t')
m, l, g_sym = symbols('m l g', positive=True)

# theta is a function of time
theta = Function('theta')(t)
theta_dot = diff(theta, t)
theta_ddot = diff(theta, t, 2)

# Simple pendulum Lagrangian
T = Rational(1, 2) * m * l**2 * theta_dot**2      # Kinetic energy
V = m * g_sym * l * (1 - cos(theta))               # Potential energy
L = T - V

print("=== Simple Pendulum ===")
print(f"T (kinetic energy)  = {T}")
print(f"V (potential energy) = {V}")
print(f"L = T - V = {simplify(L)}")

# Euler-Lagrange equation: d/dt(dL/d(theta_dot)) - dL/d(theta) = 0
dL_dtheta_dot = diff(L, theta_dot)
dL_dtheta = diff(L, theta)
EL = diff(dL_dtheta_dot, t) - dL_dtheta

print(f"\\nd/dt(dL/d(theta_dot)) = {simplify(diff(dL_dtheta_dot, t))}")
print(f"dL/d(theta) = {simplify(dL_dtheta)}")

EL_simplified = simplify(EL)
print(f"\\nEuler-Lagrange equation:")
print(f"  {EL_simplified} = 0")

# Solve for theta_ddot
eq_of_motion = solve(EL_simplified, theta_ddot)
print(f"\\nEquation of motion:")
print(f"  theta_ddot = {eq_of_motion[0]}")
print(f"\\nSmall-angle approximation (sin(theta) ~ theta):")
print(f"  theta_ddot = -(g/l) * theta  -->  SHO with omega = sqrt(g/l)")

# ---- Now do the double pendulum ----
print("\\n\\n=== Double Pendulum (symbolic derivation) ===")
m1, m2, l1, l2 = symbols('m_1 m_2 l_1 l_2', positive=True)
theta1 = Function('theta_1')(t)
theta2 = Function('theta_2')(t)
th1_dot = diff(theta1, t)
th2_dot = diff(theta2, t)

# Positions of masses
x1 = l1 * sin(theta1)
y1 = -l1 * cos(theta1)
x2 = x1 + l2 * sin(theta2)
y2 = y1 - l2 * cos(theta2)

# Velocities (squared)
v1_sq = simplify(diff(x1, t)**2 + diff(y1, t)**2)
v2_sq = simplify(diff(x2, t)**2 + diff(y2, t)**2)

# Lagrangian
T_double = Rational(1, 2) * m1 * v1_sq + Rational(1, 2) * m2 * v2_sq
V_double = m1 * g_sym * y1 + m2 * g_sym * y2
L_double = T_double - V_double

print(f"T = {simplify(T_double)}")
print(f"\\n(Full Lagrangian is complex - showing structure)")
print(f"Number of terms in L: ~{len(str(simplify(L_double)).split('+'))} terms")
print("\\nThe double pendulum has coupled, nonlinear equations of motion.")
print("We will simulate it numerically next.")""")

    # ---- MD: Double Pendulum ----
    nb.md("""\
## 3. Double Pendulum: Chaos in Classical Mechanics

The double pendulum is one of the simplest systems exhibiting **deterministic chaos**.
Tiny differences in initial conditions lead to vastly different trajectories.

The equations of motion (derived from the Lagrangian above) are coupled nonlinear
ODEs that must be solved numerically. We use `scipy.integrate.solve_ivp` with the
RK45 method.""")

    # ---- Code: Double Pendulum ----
    nb.code("""\
# --- Double Pendulum Simulation ---

def double_pendulum_ode(t, state, m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81):
    \"\"\"
    Equations of motion for the double pendulum.
    state = [theta1, theta2, omega1, omega2]
    \"\"\"
    th1, th2, w1, w2 = state
    delta = th1 - th2
    den = m1 + m2 * np.sin(delta)**2

    # Angular accelerations (from Lagrangian mechanics)
    alpha1 = (-m2 * l1 * w1**2 * np.sin(delta) * np.cos(delta)
              - m2 * l2 * w2**2 * np.sin(delta)
              - (m1 + m2) * g * np.sin(th1)
              + m2 * g * np.sin(th2) * np.cos(delta)) / (l1 * den)

    alpha2 = ((m1 + m2) * l1 * w1**2 * np.sin(delta)
              + (m1 + m2) * g * np.sin(th1) * np.cos(delta)
              - (m1 + m2) * l2 * w2**2 * np.sin(delta) * np.cos(delta)
              - (m1 + m2) * g * np.sin(th2)) / (l2 * den)

    return [w1, w2, alpha1, alpha2]

# Simulate two nearby initial conditions (to show chaos)
l1, l2 = 1.0, 1.0
t_span = (0, 20)
t_eval = np.linspace(0, 20, 5000)

# Initial conditions: slightly different starting angles
ic1 = [np.pi/2, np.pi/2, 0, 0]          # theta1=90, theta2=90
ic2 = [np.pi/2 + 0.001, np.pi/2, 0, 0]  # theta1=90.06deg, theta2=90

sol1 = solve_ivp(double_pendulum_ode, t_span, ic1, t_eval=t_eval, max_step=0.005)
sol2 = solve_ivp(double_pendulum_ode, t_span, ic2, t_eval=t_eval, max_step=0.005)

# Convert to Cartesian for plotting
def pendulum_xy(sol, l1=1.0, l2=1.0):
    th1, th2 = sol.y[0], sol.y[1]
    x1 = l1 * np.sin(th1)
    y1 = -l1 * np.cos(th1)
    x2 = x1 + l2 * np.sin(th2)
    y2 = y1 - l2 * np.cos(th2)
    return x1, y1, x2, y2

x1_a, y1_a, x2_a, y2_a = pendulum_xy(sol1)
x1_b, y1_b, x2_b, y2_b = pendulum_xy(sol2)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# Trace of second mass
axes[0].plot(x2_a, y2_a, 'b-', alpha=0.4, linewidth=0.5, label='IC 1')
axes[0].plot(x2_b, y2_b, 'r-', alpha=0.4, linewidth=0.5, label='IC 2 (+0.001 rad)')
axes[0].set_xlabel('x (m)')
axes[0].set_ylabel('y (m)')
axes[0].set_title('Trace of Second Mass')
axes[0].set_aspect('equal')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Angle difference divergence
angle_diff = np.abs(sol1.y[0] - sol2.y[0])
axes[1].semilogy(sol1.t, angle_diff, 'k-', linewidth=1)
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('$|\\Delta\\theta_1|$ (rad)')
axes[1].set_title('Sensitive Dependence on Initial Conditions')
axes[1].grid(True, alpha=0.3)
axes[1].axhline(0.001, color='red', linestyle='--', alpha=0.5, label='Initial difference')
axes[1].legend()

# theta1 vs time comparison
axes[2].plot(sol1.t, sol1.y[0], 'b-', alpha=0.7, linewidth=1, label='IC 1')
axes[2].plot(sol2.t, sol2.y[0], 'r-', alpha=0.7, linewidth=1, label='IC 2')
axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('$\\theta_1$ (rad)')
axes[2].set_title('$\\theta_1$ Divergence Over Time')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.suptitle('Double Pendulum: Deterministic Chaos', fontsize=15, y=1.02)
plt.tight_layout()
plt.show()

print(f"Initial angle difference: {0.001:.4f} rad = {np.degrees(0.001):.4f} deg")
print(f"Final angle difference:   {angle_diff[-1]:.4f} rad = {np.degrees(angle_diff[-1]):.2f} deg")""")

    # ---- MD: Hamiltonian Mechanics ----
    nb.md("""\
## 4. Hamiltonian Mechanics and Phase Space

The **Hamiltonian** is obtained from the Lagrangian by a Legendre transform:

$$p_i = \\frac{\\partial \\mathcal{L}}{\\partial \\dot{q}_i}, \\qquad H(q, p) = \\sum_i p_i \\dot{q}_i - \\mathcal{L}$$

**Hamilton's equations** (first-order ODEs):

$$\\dot{q}_i = \\frac{\\partial H}{\\partial p_i}, \\qquad \\dot{p}_i = -\\frac{\\partial H}{\\partial q_i}$$

The system evolves in **phase space** $(q, p)$, and Liouville's theorem guarantees
that phase-space volume is preserved (incompressible flow).

For the simple harmonic oscillator: $H = \\frac{p^2}{2m} + \\frac{1}{2}kq^2$

Phase-space trajectories are **ellipses** (constant energy curves).""")

    # ---- Code: Hamiltonian Phase Space ----
    nb.code("""\
# --- Hamiltonian formulation: phase space of the harmonic oscillator ---

# Symbolic Hamiltonian for SHO
q, p_sym = symbols('q p')
m_val, k_val = symbols('m k', positive=True)

H_sho = p_sym**2 / (2 * m_val) + Rational(1, 2) * k_val * q**2
print("=== Simple Harmonic Oscillator ===")
print(f"H(q, p) = {H_sho}")

# Hamilton's equations
q_dot = diff(H_sho, p_sym)
p_dot = -diff(H_sho, q)
print(f"dq/dt = dH/dp = {q_dot}")
print(f"dp/dt = -dH/dq = {p_dot}")

# Numerical phase space visualization
m_num, k_num = 1.0, 4.0  # omega = sqrt(k/m) = 2
omega = np.sqrt(k_num / m_num)

def sho_hamiltonian(q, p, m=1.0, k=4.0):
    return p**2 / (2*m) + 0.5 * k * q**2

# Phase portrait: vector field + trajectories
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Left: phase portrait with trajectories
ax = axes[0]
q_range = np.linspace(-3, 3, 20)
p_range = np.linspace(-6, 6, 20)
Q, P = np.meshgrid(q_range, p_range)

# Vector field from Hamilton's equations
dQ = P / m_num       # dq/dt = p/m
dP = -k_num * Q      # dp/dt = -kq

# Normalize arrows for visualization
speed = np.sqrt(dQ**2 + dP**2)
speed[speed == 0] = 1
ax.quiver(Q, P, dQ/speed, dP/speed, speed, cmap='coolwarm', alpha=0.6)

# Plot several trajectories (different energies)
t_traj = np.linspace(0, 2 * np.pi / omega, 500)
for E in [0.5, 2.0, 4.5, 8.0]:
    # For SHO: q(t) = A*cos(wt), p(t) = -m*w*A*sin(wt)
    A = np.sqrt(2 * E / k_num)
    q_traj = A * np.cos(omega * t_traj)
    p_traj = -m_num * omega * A * np.sin(omega * t_traj)
    ax.plot(q_traj, p_traj, '-', linewidth=2, label=f'E = {E}')

ax.set_xlabel('Position $q$')
ax.set_ylabel('Momentum $p$')
ax.set_title('Phase Space: Harmonic Oscillator')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_aspect('auto')

# Right: energy contour plot
ax = axes[1]
q_fine = np.linspace(-3, 3, 200)
p_fine = np.linspace(-6, 6, 200)
Q_fine, P_fine = np.meshgrid(q_fine, p_fine)
H_vals = sho_hamiltonian(Q_fine, P_fine, m_num, k_num)

contour = ax.contourf(Q_fine, P_fine, H_vals, levels=20, cmap='viridis')
plt.colorbar(contour, ax=ax, label='Energy $H(q, p)$')
ax.contour(Q_fine, P_fine, H_vals, levels=10, colors='white', linewidths=0.5, alpha=0.5)
ax.set_xlabel('Position $q$')
ax.set_ylabel('Momentum $p$')
ax.set_title('Energy Contours in Phase Space')
ax.set_aspect('auto')

plt.tight_layout()
plt.show()

print(f"omega = sqrt(k/m) = {omega:.4f} rad/s")
print(f"Period = 2*pi/omega = {2*np.pi/omega:.4f} s")
print("Phase space orbits are CLOSED (energy is conserved).")
print("Area is preserved (Liouville's theorem).")""")

    # ---- MD: Conservation Laws ----
    nb.md("""\
## 5. Conservation Laws Verified Numerically

**Noether's Theorem** connects symmetries to conservation laws:

| Symmetry | Conserved Quantity |
|----------|-------------------|
| Time translation | Energy |
| Space translation | Linear momentum |
| Rotational | Angular momentum |

For the simple pendulum, the Hamiltonian (total energy) is conserved:
$$H = \\frac{p_\\theta^2}{2ml^2} + mgl(1 - \\cos\\theta)$$""")

    # ---- Code: Conservation Laws ----
    nb.code("""\
# --- Verify conservation of energy in simple pendulum simulation ---

def pendulum_ode(t, state, g=9.81, l=1.0):
    \"\"\"Simple pendulum ODE: state = [theta, omega]\"\"\"
    theta, omega = state
    return [omega, -(g/l) * np.sin(theta)]

def pendulum_energy(theta, omega, m=1.0, g=9.81, l=1.0):
    \"\"\"Total energy: T + V\"\"\"
    T = 0.5 * m * l**2 * omega**2
    V = m * g * l * (1 - np.cos(theta))
    return T, V, T + V

# Simulate pendulum with large initial angle (nonlinear regime)
g_val, l_val, m_val = 9.81, 1.0, 1.0
theta0 = np.pi * 0.9  # Nearly vertical (162 degrees)
ic = [theta0, 0.0]

sol = solve_ivp(pendulum_ode, (0, 10), ic, t_eval=np.linspace(0, 10, 2000),
                max_step=0.005, args=(g_val, l_val))

# Compute energies
T, V, E_total = pendulum_energy(sol.y[0], sol.y[1], m_val, g_val, l_val)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Angle vs time
axes[0, 0].plot(sol.t, np.degrees(sol.y[0]), 'b-')
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('$\\theta$ (degrees)')
axes[0, 0].set_title(f'Pendulum Angle ($\\theta_0 = {np.degrees(theta0):.0f}°$)')
axes[0, 0].grid(True, alpha=0.3)

# Phase portrait
axes[0, 1].plot(sol.y[0], sol.y[1], 'r-', linewidth=1)
axes[0, 1].set_xlabel('$\\theta$ (rad)')
axes[0, 1].set_ylabel('$\\omega$ (rad/s)')
axes[0, 1].set_title('Phase Portrait')
axes[0, 1].grid(True, alpha=0.3)

# Energy components
axes[1, 0].plot(sol.t, T, 'r-', label='Kinetic $T$', alpha=0.7)
axes[1, 0].plot(sol.t, V, 'b-', label='Potential $V$', alpha=0.7)
axes[1, 0].plot(sol.t, E_total, 'k--', linewidth=2, label='Total $E = T + V$')
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Energy (J)')
axes[1, 0].set_title('Energy Conservation')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Energy error (deviation from initial)
E_error = np.abs(E_total - E_total[0]) / E_total[0]
axes[1, 1].semilogy(sol.t, E_error + 1e-16, 'k-', linewidth=1)
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_ylabel('Relative Energy Error')
axes[1, 1].set_title('Numerical Energy Conservation')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Initial energy:     E(0) = {E_total[0]:.6f} J")
print(f"Final energy:       E(T) = {E_total[-1]:.6f} J")
print(f"Max relative error: {np.max(E_error):.2e}")
print(f"Energy is conserved to ~{np.max(E_error):.1e} (limited by numerical integrator)")""")

    # ---- MD: Poisson Brackets ----
    nb.md("""\
## 6. Poisson Brackets: The Classical-Quantum Bridge

The **Poisson bracket** of two phase-space functions $f(q, p)$ and $g(q, p)$ is:

$$\\{f, g\\} = \\sum_i \\left( \\frac{\\partial f}{\\partial q_i}\\frac{\\partial g}{\\partial p_i} - \\frac{\\partial f}{\\partial p_i}\\frac{\\partial g}{\\partial q_i} \\right)$$

**Fundamental Poisson brackets:**
$$\\{q_i, q_j\\} = 0, \\qquad \\{p_i, p_j\\} = 0, \\qquad \\{q_i, p_j\\} = \\delta_{ij}$$

**The bridge to quantum mechanics (Dirac's prescription):**

$$\\{f, g\\}_{\\text{Poisson}} \\longrightarrow \\frac{1}{i\\hbar}[\\hat{f}, \\hat{g}]$$

This means $\\{q, p\\} = 1$ becomes $[\\hat{q}, \\hat{p}] = i\\hbar$, the
**canonical commutation relation** --- the foundation of all quantum mechanics.""")

    # ---- Code: Poisson Brackets ----
    nb.code("""\
# --- Poisson Brackets with SymPy ---

q1, q2, p1, p2 = symbols('q_1 q_2 p_1 p_2')

def poisson_bracket(f, g, q_vars, p_vars):
    \"\"\"
    Compute the Poisson bracket {f, g} = sum (df/dqi * dg/dpi - df/dpi * dg/dqi).

    Parameters
    ----------
    f, g : sympy expressions
    q_vars : list of position symbols
    p_vars : list of momentum symbols

    Returns
    -------
    sympy expression for {f, g}
    \"\"\"
    result = 0
    for qi, pi in zip(q_vars, p_vars):
        result += diff(f, qi) * diff(g, pi) - diff(f, pi) * diff(g, qi)
    return simplify(result)


# One degree of freedom
q_vars_1d = [q1]
p_vars_1d = [p1]

print("=== Fundamental Poisson Brackets (1D) ===")
print(f"  {{q, p}} = {poisson_bracket(q1, p1, q_vars_1d, p_vars_1d)}")
print(f"  {{q, q}} = {poisson_bracket(q1, q1, q_vars_1d, p_vars_1d)}")
print(f"  {{p, p}} = {poisson_bracket(p1, p1, q_vars_1d, p_vars_1d)}")

# SHO Hamiltonian
m_sym, k_sym = symbols('m k', positive=True)
H = p1**2 / (2 * m_sym) + Rational(1, 2) * k_sym * q1**2
print(f"\\n=== SHO Hamiltonian: H = {H} ===")
print(f"  {{q, H}} = {poisson_bracket(q1, H, q_vars_1d, p_vars_1d)}  (= dq/dt = p/m)")
print(f"  {{p, H}} = {poisson_bracket(p1, H, q_vars_1d, p_vars_1d)}  (= dp/dt = -kq)")

# Angular momentum example (2D)
q_vars_2d = [q1, q2]
p_vars_2d = [p1, p2]
L_z = q1 * p2 - q2 * p1  # Angular momentum

print(f"\\n=== Angular Momentum L_z = {L_z} ===")
print(f"  {{L_z, q1}} = {poisson_bracket(L_z, q1, q_vars_2d, p_vars_2d)}")
print(f"  {{L_z, q2}} = {poisson_bracket(L_z, q2, q_vars_2d, p_vars_2d)}")
print(f"  {{L_z, p1}} = {poisson_bracket(L_z, p1, q_vars_2d, p_vars_2d)}")
print(f"  {{L_z, p2}} = {poisson_bracket(L_z, p2, q_vars_2d, p_vars_2d)}")

# 2D isotropic oscillator
H_2d = (p1**2 + p2**2) / (2 * m_sym) + Rational(1, 2) * k_sym * (q1**2 + q2**2)
print(f"\\n=== 2D Isotropic Oscillator ===")
print(f"  H = {H_2d}")
print(f"  {{L_z, H}} = {poisson_bracket(L_z, H_2d, q_vars_2d, p_vars_2d)}  (L_z conserved!)")

print("\\n=== Classical-Quantum Correspondence ===")
print("  {q, p} = 1          -->  [q_hat, p_hat] = i*hbar")
print("  {f, H} = df/dt      -->  [f_hat, H_hat] / (i*hbar) = df_hat/dt")
print("  {L_z, H} = 0        -->  [L_z_hat, H_hat] = 0 (simultaneous eigenstates)")""")

    # ---- MD: Summary ----
    nb.md("""\
## Summary

| Classical Concept | Formula | Quantum Analog |
|-------------------|---------|----------------|
| Lagrangian | $\\mathcal{L} = T - V$ | Path integral $\\int e^{iS/\\hbar} \\mathcal{D}[q]$ |
| Euler-Lagrange | $\\frac{d}{dt}\\frac{\\partial L}{\\partial \\dot{q}} = \\frac{\\partial L}{\\partial q}$ | Heisenberg equation |
| Hamiltonian | $H = T + V$ | $\\hat{H}$ operator |
| Hamilton's eqs | $\\dot{q} = \\partial H/\\partial p$ | $i\\hbar \\partial_t|\\psi\\rangle = \\hat{H}|\\psi\\rangle$ |
| Poisson bracket | $\\{q, p\\} = 1$ | $[\\hat{q}, \\hat{p}] = i\\hbar$ |
| Phase space | $(q, p)$ | Hilbert space $|\\psi\\rangle$ |

**Key Takeaways:**
1. Lagrangian mechanics replaces forces with energy --- more elegant and general
2. The double pendulum demonstrates that classical mechanics can be chaotic
3. Hamiltonian mechanics reveals phase-space structure preserved by evolution
4. Conservation laws follow from symmetries (Noether's theorem)
5. Poisson brackets map directly to quantum commutators (Dirac's prescription)
6. The Hamiltonian formulation is **the** starting point for quantization

---
*This completes the classical foundations. Next: Year 0 continues with complex analysis,
electromagnetism, and functional analysis --- all building toward quantum mechanics in Year 1.*""")

    nb.save()
    print("  [OK] Month 06 notebook generated.\n")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("SIIEA Quantum Engineering - Generating Months 4-6 Notebooks")
    print("=" * 70)
    print()

    build_month_04()
    build_month_05()
    build_month_06()

    print("=" * 70)
    print("All 3 notebooks generated successfully.")
    print("=" * 70)
