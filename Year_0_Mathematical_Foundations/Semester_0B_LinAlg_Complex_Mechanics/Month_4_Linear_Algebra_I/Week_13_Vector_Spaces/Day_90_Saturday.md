# Day 90: Comprehensive Computational Lab — Vector Spaces in Python

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Core Implementations |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Applications & Visualization |
| Evening | 7:00 PM - 8:00 PM | 1 hour | QM Simulations |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Implement vector space operations using NumPy
2. Write functions to test vector space axioms
3. Create tools for subspace verification
4. Build algorithms for testing linear independence
5. Implement basis-finding algorithms
6. Simulate quantum states in various bases
7. Visualize vector space concepts in 2D and 3D

---

## 💻 Lab 1: Vector Space Foundations (Morning, Part 1)

### 1.1 Setup and Environment

```python
"""
Week 13 Computational Lab: Vector Spaces
QSE Self-Study Preparation — Year 0, Semester 0B

This notebook implements the core concepts from Week 13:
- Vector spaces and axioms
- Subspaces
- Span and linear combinations
- Linear independence
- Bases and dimension
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import null_space, svd
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Display settings
np.set_printoptions(precision=4, suppress=True)

print("Vector Spaces Lab - Week 13")
print("=" * 50)
```

### 1.2 Vector Space Axiom Verification

```python
class VectorSpaceVerifier:
    """
    Tool to verify vector space axioms numerically.
    Works for vector spaces represented as arrays.
    """
    
    def __init__(self, tol: float = 1e-10):
        self.tol = tol
        self.results = {}
    
    def verify_all_axioms(self, V_sample: List[np.ndarray], 
                          scalars_sample: List[float] = None,
                          num_tests: int = 100) -> dict:
        """
        Verify all 8 vector space axioms using random samples.
        
        Parameters:
        -----------
        V_sample : list of vectors from the space
        scalars_sample : list of scalars (default: random reals)
        num_tests : number of random tests per axiom
        """
        if scalars_sample is None:
            scalars_sample = np.random.randn(num_tests * 3).tolist()
        
        results = {
            'commutativity': self._test_commutativity(V_sample, num_tests),
            'associativity_add': self._test_associativity_add(V_sample, num_tests),
            'zero_vector': self._test_zero_vector(V_sample),
            'additive_inverse': self._test_additive_inverse(V_sample, num_tests),
            'scalar_identity': self._test_scalar_identity(V_sample, num_tests),
            'scalar_compatibility': self._test_scalar_compatibility(V_sample, scalars_sample, num_tests),
            'distributivity_vectors': self._test_distributivity_vectors(V_sample, scalars_sample, num_tests),
            'distributivity_scalars': self._test_distributivity_scalars(V_sample, scalars_sample, num_tests),
        }
        
        self.results = results
        return results
    
    def _test_commutativity(self, V, n):
        """Axiom 1: u + v = v + u"""
        for _ in range(n):
            u, v = np.random.choice(len(V), 2, replace=True)
            if not np.allclose(V[u] + V[v], V[v] + V[u], atol=self.tol):
                return False, "Commutativity failed"
        return True, "Passed"
    
    def _test_associativity_add(self, V, n):
        """Axiom 2: (u + v) + w = u + (v + w)"""
        for _ in range(n):
            u, v, w = np.random.choice(len(V), 3, replace=True)
            if not np.allclose((V[u] + V[v]) + V[w], V[u] + (V[v] + V[w]), atol=self.tol):
                return False, "Associativity failed"
        return True, "Passed"
    
    def _test_zero_vector(self, V):
        """Axiom 3: Existence of zero vector"""
        zero = np.zeros_like(V[0])
        for v in V:
            if not np.allclose(v + zero, v, atol=self.tol):
                return False, "Zero vector property failed"
        return True, "Passed"
    
    def _test_additive_inverse(self, V, n):
        """Axiom 4: v + (-v) = 0"""
        for _ in range(n):
            i = np.random.choice(len(V))
            v = V[i]
            if not np.allclose(v + (-v), np.zeros_like(v), atol=self.tol):
                return False, "Additive inverse failed"
        return True, "Passed"
    
    def _test_scalar_identity(self, V, n):
        """Axiom 5: 1 * v = v"""
        for _ in range(n):
            i = np.random.choice(len(V))
            if not np.allclose(1.0 * V[i], V[i], atol=self.tol):
                return False, "Scalar identity failed"
        return True, "Passed"
    
    def _test_scalar_compatibility(self, V, scalars, n):
        """Axiom 6: a(bv) = (ab)v"""
        for _ in range(n):
            i = np.random.choice(len(V))
            a, b = np.random.choice(scalars, 2)
            if not np.allclose(a * (b * V[i]), (a * b) * V[i], atol=self.tol):
                return False, "Scalar compatibility failed"
        return True, "Passed"
    
    def _test_distributivity_vectors(self, V, scalars, n):
        """Axiom 7: a(u + v) = au + av"""
        for _ in range(n):
            u, v = np.random.choice(len(V), 2, replace=True)
            a = np.random.choice(scalars)
            if not np.allclose(a * (V[u] + V[v]), a * V[u] + a * V[v], atol=self.tol):
                return False, "Distributivity over vectors failed"
        return True, "Passed"
    
    def _test_distributivity_scalars(self, V, scalars, n):
        """Axiom 8: (a + b)v = av + bv"""
        for _ in range(n):
            i = np.random.choice(len(V))
            a, b = np.random.choice(scalars, 2)
            if not np.allclose((a + b) * V[i], a * V[i] + b * V[i], atol=self.tol):
                return False, "Distributivity over scalars failed"
        return True, "Passed"
    
    def print_report(self):
        """Print formatted verification report."""
        print("\nVector Space Axiom Verification Report")
        print("=" * 50)
        all_passed = True
        for axiom, (passed, msg) in self.results.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {axiom}: {msg}")
            if not passed:
                all_passed = False
        print("=" * 50)
        print(f"Overall: {'VERIFIED' if all_passed else 'FAILED'}")
        return all_passed


# Test on ℝ³
print("\n" + "="*60)
print("Testing ℝ³ as a vector space")
print("="*60)

R3_sample = [np.random.randn(3) for _ in range(20)]
verifier = VectorSpaceVerifier()
verifier.verify_all_axioms(R3_sample)
verifier.print_report()

# Test on ℂ²
print("\n" + "="*60)
print("Testing ℂ² as a vector space")
print("="*60)

C2_sample = [np.random.randn(2) + 1j * np.random.randn(2) for _ in range(20)]
verifier_complex = VectorSpaceVerifier()
verifier_complex.verify_all_axioms(C2_sample, 
                                    scalars_sample=(np.random.randn(50) + 1j * np.random.randn(50)).tolist())
verifier_complex.print_report()
```

---

## 💻 Lab 2: Subspace Operations (Morning, Part 2)

```python
class SubspaceTools:
    """
    Tools for working with subspaces of ℝⁿ or ℂⁿ.
    """
    
    @staticmethod
    def is_subspace(constraint_func, ambient_dim: int, 
                    num_tests: int = 1000, tol: float = 1e-10) -> Tuple[bool, str]:
        """
        Test if a constraint defines a subspace.
        
        Parameters:
        -----------
        constraint_func : function(vector) -> bool
        ambient_dim : dimension of ambient space
        num_tests : number of random tests
        """
        # Test 1: Zero vector
        zero = np.zeros(ambient_dim)
        if not constraint_func(zero):
            return False, "Zero vector not in subset"
        
        # Generate random vectors in the subset
        vectors_in_subset = []
        attempts = 0
        while len(vectors_in_subset) < min(num_tests, 100) and attempts < num_tests * 10:
            v = np.random.randn(ambient_dim)
            if constraint_func(v):
                vectors_in_subset.append(v)
            attempts += 1
        
        if len(vectors_in_subset) < 2:
            return True, "Passed (limited samples)"
        
        # Test 2 & 3: Closure under linear combinations
        for _ in range(num_tests):
            i, j = np.random.choice(len(vectors_in_subset), 2)
            v, w = vectors_in_subset[i], vectors_in_subset[j]
            
            # Test closure under addition
            if not constraint_func(v + w):
                return False, f"Not closed under addition"
            
            # Test closure under scalar multiplication
            c = np.random.randn()
            if not constraint_func(c * v):
                return False, f"Not closed under scalar multiplication"
        
        return True, "Passed"
    
    @staticmethod
    def subspace_intersection(U_basis: List[np.ndarray], 
                              W_basis: List[np.ndarray]) -> List[np.ndarray]:
        """
        Find basis for intersection of two subspaces.
        
        Method: U ∩ W = null(stack of (U | -W))
        """
        if len(U_basis) == 0 or len(W_basis) == 0:
            return []
        
        U = np.column_stack(U_basis)
        W = np.column_stack(W_basis)
        
        # Combine: [U | -W], find null space
        combined = np.hstack([U, -W])
        ns = null_space(combined)
        
        if ns.shape[1] == 0:
            return []
        
        # Extract coefficients for U
        u_dim = U.shape[1]
        intersection_basis = []
        for i in range(ns.shape[1]):
            coeffs_U = ns[:u_dim, i]
            v = U @ coeffs_U
            if np.linalg.norm(v) > 1e-10:
                intersection_basis.append(v / np.linalg.norm(v))
        
        return intersection_basis
    
    @staticmethod
    def subspace_sum(U_basis: List[np.ndarray], 
                     W_basis: List[np.ndarray]) -> List[np.ndarray]:
        """
        Find basis for sum of two subspaces: U + W.
        """
        all_vectors = U_basis + W_basis
        if len(all_vectors) == 0:
            return []
        
        A = np.column_stack(all_vectors)
        
        # Find basis using SVD
        U, S, Vt = svd(A)
        rank = np.sum(S > 1e-10)
        
        return [U[:, i] for i in range(rank)]


# Test subspace verification
print("\n" + "="*60)
print("Subspace Verification Tests")
print("="*60)

tools = SubspaceTools()

# Test 1: Plane x + y + z = 0 (should be subspace)
def plane_constraint(v):
    return np.abs(v[0] + v[1] + v[2]) < 1e-10

result, msg = tools.is_subspace(plane_constraint, 3)
print(f"\n1. Plane x + y + z = 0: {result} ({msg})")

# Test 2: x² + y² ≤ 1 (unit ball, NOT a subspace)
def ball_constraint(v):
    return v[0]**2 + v[1]**2 <= 1

result, msg = tools.is_subspace(ball_constraint, 3)
print(f"2. Unit ball x² + y² ≤ 1: {result} ({msg})")

# Test 3: x + y + z = 1 (NOT a subspace)
def affine_constraint(v):
    return np.abs(v[0] + v[1] + v[2] - 1) < 1e-10

result, msg = tools.is_subspace(affine_constraint, 3)
print(f"3. Plane x + y + z = 1: {result} ({msg})")

# Test intersection
print("\n" + "="*60)
print("Subspace Intersection Test")
print("="*60)

U_basis = [np.array([1, 0, -1]), np.array([0, 1, -1])]  # x + y + z = 0
W_basis = [np.array([1, -1, 0]), np.array([0, 1, -1])]  # y + z = 0 (in different param)

intersection = tools.subspace_intersection(U_basis, W_basis)
print(f"U basis: {[v.round(4) for v in U_basis]}")
print(f"W basis: {[v.round(4) for v in W_basis]}")
print(f"U ∩ W dimension: {len(intersection)}")
if intersection:
    print(f"U ∩ W basis: {[v.round(4) for v in intersection]}")
```

---

## 💻 Lab 3: Linear Independence and Bases (Afternoon, Part 1)

```python
class LinearAlgebraTools:
    """
    Core linear algebra operations: independence, span, bases.
    """
    
    def __init__(self, tol: float = 1e-10):
        self.tol = tol
    
    def is_linearly_independent(self, vectors: List[np.ndarray]) -> Tuple[bool, int, Optional[np.ndarray]]:
        """
        Test if vectors are linearly independent.
        
        Returns:
        --------
        (independent: bool, rank: int, null_space: array or None)
        """
        if len(vectors) == 0:
            return True, 0, None
        
        A = np.column_stack(vectors)
        rank = np.linalg.matrix_rank(A, tol=self.tol)
        
        if rank < len(vectors):
            ns = null_space(A)
            return False, rank, ns
        else:
            return True, rank, None
    
    def is_in_span(self, target: np.ndarray, 
                   vectors: List[np.ndarray]) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Check if target is in span of vectors.
        
        Returns:
        --------
        (in_span: bool, coefficients: array or None)
        """
        if len(vectors) == 0:
            return np.allclose(target, 0, atol=self.tol), None
        
        A = np.column_stack(vectors)
        
        # Solve least squares
        coeffs, residuals, rank, s = np.linalg.lstsq(A, target, rcond=None)
        
        if np.allclose(A @ coeffs, target, atol=self.tol):
            return True, coeffs
        else:
            return False, None
    
    def find_basis(self, vectors: List[np.ndarray]) -> List[np.ndarray]:
        """
        Find a basis for the span of given vectors.
        Returns subset of original vectors.
        """
        if len(vectors) == 0:
            return []
        
        A = np.column_stack(vectors)
        
        # Use column pivoting QR decomposition
        Q, R, P = np.linalg.qr(A, mode='reduced')
        
        # Find rank from R diagonal
        diag = np.abs(np.diag(R))
        rank = np.sum(diag > self.tol)
        
        # P gives column permutation; first 'rank' columns in original
        # Actually, let's use a different method: greedy selection
        
        basis = []
        for v in vectors:
            test_set = basis + [v]
            ind, _, _ = self.is_linearly_independent(test_set)
            if ind:
                basis.append(v)
        
        return basis
    
    def extend_to_basis(self, vectors: List[np.ndarray], 
                        ambient_dim: int) -> List[np.ndarray]:
        """
        Extend linearly independent vectors to a basis of ℝⁿ (or ℂⁿ).
        """
        basis = list(vectors)
        
        # Try adding standard basis vectors
        for i in range(ambient_dim):
            e_i = np.zeros(ambient_dim, dtype=vectors[0].dtype if vectors else float)
            e_i[i] = 1
            
            test_set = basis + [e_i]
            ind, _, _ = self.is_linearly_independent(test_set)
            if ind:
                basis.append(e_i)
            
            if len(basis) == ambient_dim:
                break
        
        return basis
    
    def dimension(self, vectors: List[np.ndarray]) -> int:
        """Compute dimension of span."""
        if len(vectors) == 0:
            return 0
        A = np.column_stack(vectors)
        return np.linalg.matrix_rank(A, tol=self.tol)
    
    def coordinate_vector(self, v: np.ndarray, 
                          basis: List[np.ndarray]) -> np.ndarray:
        """Find coordinates of v with respect to basis."""
        B = np.column_stack(basis)
        return np.linalg.solve(B, v)
    
    def change_of_basis_matrix(self, old_basis: List[np.ndarray], 
                               new_basis: List[np.ndarray]) -> np.ndarray:
        """
        Compute matrix P such that [v]_new = P @ [v]_old.
        """
        n = len(old_basis)
        P = np.zeros((n, n), dtype=old_basis[0].dtype)
        for i, v in enumerate(old_basis):
            P[:, i] = self.coordinate_vector(v, new_basis)
        return P


# Comprehensive tests
print("\n" + "="*60)
print("Linear Algebra Tools Tests")
print("="*60)

la = LinearAlgebraTools()

# Test 1: Linear independence
v1 = np.array([1, 0, 1])
v2 = np.array([0, 1, 1])
v3 = np.array([1, 1, 2])

ind, rank, ns = la.is_linearly_independent([v1, v2, v3])
print(f"\n1. Vectors (1,0,1), (0,1,1), (1,1,2):")
print(f"   Independent: {ind}, Rank: {rank}")
if not ind and ns is not None:
    print(f"   Dependency relation: {ns[:, 0].round(4)}")
    print(f"   Verification: {(ns[0,0]*v1 + ns[1,0]*v2 + ns[2,0]*v3).round(10)}")

# Test 2: Independent vectors
w1 = np.array([1, 0, 0])
w2 = np.array([1, 1, 0])
w3 = np.array([1, 1, 1])

ind, rank, ns = la.is_linearly_independent([w1, w2, w3])
print(f"\n2. Vectors (1,0,0), (1,1,0), (1,1,1):")
print(f"   Independent: {ind}, Rank: {rank}")

# Test 3: Span membership
target = np.array([3, 5, 8])
in_span, coeffs = la.is_in_span(target, [v1, v2])
print(f"\n3. Is (3,5,8) in span{{(1,0,1), (0,1,1)}}?")
print(f"   In span: {in_span}")
if in_span:
    print(f"   Coefficients: {coeffs.round(4)}")
    print(f"   Verification: {(coeffs[0]*v1 + coeffs[1]*v2).round(10)}")

# Test 4: Find basis
vectors = [np.array([1, 2, 3]), np.array([4, 5, 6]), 
           np.array([7, 8, 9]), np.array([2, 4, 6])]
basis = la.find_basis(vectors)
print(f"\n4. Basis for span of (1,2,3), (4,5,6), (7,8,9), (2,4,6):")
print(f"   Dimension: {len(basis)}")
for i, v in enumerate(basis):
    print(f"   v{i+1}: {v}")

# Test 5: Extend to basis
partial = [np.array([1, 1, 0]), np.array([0, 1, 1])]
full_basis = la.extend_to_basis(partial, 3)
print(f"\n5. Extending {{(1,1,0), (0,1,1)}} to basis of ℝ³:")
for i, v in enumerate(full_basis):
    print(f"   e{i+1}: {v}")

# Test 6: Coordinate vector
std_basis = [np.array([1, 0]), np.array([0, 1])]
new_basis = [np.array([1, 1]), np.array([1, -1])]
v = np.array([3, 5])

coords_std = la.coordinate_vector(v, std_basis)
coords_new = la.coordinate_vector(v, new_basis)

print(f"\n6. Coordinate representations of (3, 5):")
print(f"   Standard basis: {coords_std}")
print(f"   Basis {{(1,1), (1,-1)}}: {coords_new}")
```

---

## 💻 Lab 4: Visualizations (Afternoon, Part 2)

```python
def visualize_span_and_independence():
    """Create comprehensive visualizations of span and independence."""
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Span in ℝ² - two independent vectors
    ax1 = fig.add_subplot(231)
    v1, v2 = np.array([1, 0.5]), np.array([0.3, 1])
    ax1.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, 
               color='red', width=0.02, label='v₁')
    ax1.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, 
               color='blue', width=0.02, label='v₂')
    
    # Grid points
    for a in np.linspace(-2, 2, 9):
        for b in np.linspace(-2, 2, 9):
            p = a*v1 + b*v2
            ax1.plot(p[0], p[1], 'ko', markersize=2, alpha=0.3)
    
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-3, 3)
    ax1.set_aspect('equal')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Independent vectors:\nspan = ℝ²')
    
    # 2. Dependent vectors in ℝ²
    ax2 = fig.add_subplot(232)
    v1, v2 = np.array([1, 2]), np.array([2, 4])
    ax2.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, 
               color='red', width=0.02, label='v₁')
    ax2.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, 
               color='blue', width=0.02, label='v₂ = 2v₁')
    
    # Line
    t = np.linspace(-1.5, 1.5, 100)
    line = np.outer(t, v1)
    ax2.plot(line[:, 0], line[:, 1], 'g-', linewidth=2, alpha=0.5, label='span')
    
    ax2.set_xlim(-3, 4)
    ax2.set_ylim(-3, 5)
    ax2.set_aspect('equal')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Dependent vectors:\nspan = line')
    
    # 3. Subspace vs Not-subspace
    ax3 = fig.add_subplot(233)
    theta = np.linspace(0, 2*np.pi, 100)
    
    # Subspace: line y = x
    t = np.linspace(-2, 2, 50)
    ax3.plot(t, t, 'b-', linewidth=2, label='y = x (subspace)')
    ax3.plot(0, 0, 'bo', markersize=10)
    
    # Not subspace: line y = x + 1
    ax3.plot(t, t + 1, 'r--', linewidth=2, label='y = x + 1 (NOT subspace)')
    ax3.plot(0, 0, 'go', markersize=10, label='Origin')
    
    ax3.set_xlim(-2, 2)
    ax3.set_ylim(-2, 3)
    ax3.set_aspect('equal')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Subspace must contain origin')
    
    # 4. 3D: Plane as subspace
    ax4 = fig.add_subplot(234, projection='3d')
    v1 = np.array([1, 0, -1])
    v2 = np.array([0, 1, -1])
    
    s = np.linspace(-1.5, 1.5, 10)
    t = np.linspace(-1.5, 1.5, 10)
    S, T = np.meshgrid(s, t)
    X = S * v1[0] + T * v2[0]
    Y = S * v1[1] + T * v2[1]
    Z = S * v1[2] + T * v2[2]
    
    ax4.plot_surface(X, Y, Z, alpha=0.5, color='cyan')
    ax4.quiver(0, 0, 0, v1[0], v1[1], v1[2], color='red', arrow_length_ratio=0.1)
    ax4.quiver(0, 0, 0, v2[0], v2[1], v2[2], color='blue', arrow_length_ratio=0.1)
    
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    ax4.set_title('2D subspace of ℝ³\nspan{(1,0,-1), (0,1,-1)}')
    
    # 5. Basis comparison
    ax5 = fig.add_subplot(235)
    
    # Standard basis grid
    for i in range(-3, 4):
        ax5.axhline(y=i, color='lightblue', linewidth=0.5)
        ax5.axvline(x=i, color='lightblue', linewidth=0.5)
    
    # New basis grid
    v1, v2 = np.array([1, 0.5]), np.array([0.5, 1])
    for a in range(-3, 4):
        p1 = a * v1 + (-3) * v2
        p2 = a * v1 + 3 * v2
        ax5.plot([p1[0], p2[0]], [p1[1], p2[1]], 'orange', linewidth=0.5, alpha=0.7)
        
        p1 = (-3) * v1 + a * v2
        p2 = 3 * v1 + a * v2
        ax5.plot([p1[0], p2[0]], [p1[1], p2[1]], 'orange', linewidth=0.5, alpha=0.7)
    
    ax5.quiver(0, 0, 1, 0, angles='xy', scale_units='xy', scale=1, 
               color='blue', width=0.02, label='Standard')
    ax5.quiver(0, 0, 0, 1, angles='xy', scale_units='xy', scale=1, 
               color='blue', width=0.02)
    ax5.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, 
               color='orange', width=0.02, label='New basis')
    ax5.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, 
               color='orange', width=0.02)
    
    ax5.set_xlim(-3, 3)
    ax5.set_ylim(-3, 3)
    ax5.set_aspect('equal')
    ax5.legend()
    ax5.set_title('Different bases, same space')
    
    # 6. Independence test via determinant
    ax6 = fig.add_subplot(236)
    
    thetas = np.linspace(0, np.pi, 100)
    v1 = np.array([1, 0])
    dets = []
    
    for theta in thetas:
        v2 = np.array([np.cos(theta), np.sin(theta)])
        A = np.column_stack([v1, v2])
        dets.append(np.linalg.det(A))
    
    ax6.plot(thetas * 180 / np.pi, dets, 'b-', linewidth=2)
    ax6.axhline(y=0, color='r', linestyle='--', label='det=0 → dependent')
    ax6.fill_between(thetas * 180 / np.pi, dets, alpha=0.3)
    
    ax6.scatter([0, 180], [0, 0], color='red', s=100, zorder=5)
    ax6.set_xlabel('Angle between vectors (degrees)')
    ax6.set_ylabel('Determinant')
    ax6.set_title('Independence: |det| > 0')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('week13_visualizations.png', dpi=150, bbox_inches='tight')
    plt.show()


visualize_span_and_independence()
```

---

## 💻 Lab 5: Quantum Mechanics Applications (Evening)

```python
class QuantumStateTools:
    """
    Tools for quantum state manipulation using linear algebra.
    Simulates finite-dimensional quantum systems.
    """
    
    def __init__(self, dim: int = 2):
        self.dim = dim
        self.computational_basis = [self._basis_ket(i) for i in range(dim)]
    
    def _basis_ket(self, i: int) -> np.ndarray:
        """Create |i⟩ basis state."""
        ket = np.zeros(self.dim, dtype=complex)
        ket[i] = 1
        return ket
    
    def create_state(self, amplitudes: List[complex]) -> np.ndarray:
        """Create a quantum state from amplitudes (normalized)."""
        state = np.array(amplitudes, dtype=complex)
        return state / np.linalg.norm(state)
    
    def inner_product(self, psi: np.ndarray, phi: np.ndarray) -> complex:
        """Compute ⟨psi|phi⟩."""
        return np.vdot(psi, phi)
    
    def probability(self, state: np.ndarray, 
                    outcome: np.ndarray) -> float:
        """Compute |⟨outcome|state⟩|²."""
        return np.abs(self.inner_product(outcome, state))**2
    
    def is_orthonormal_basis(self, basis: List[np.ndarray], 
                             tol: float = 1e-10) -> bool:
        """Check if basis is orthonormal."""
        n = len(basis)
        for i in range(n):
            for j in range(n):
                ip = self.inner_product(basis[i], basis[j])
                expected = 1 if i == j else 0
                if np.abs(ip - expected) > tol:
                    return False
        return True
    
    def expand_in_basis(self, state: np.ndarray, 
                        basis: List[np.ndarray]) -> List[complex]:
        """Find coefficients of state in given basis."""
        return [self.inner_product(b, state) for b in basis]
    
    def change_basis(self, state: np.ndarray, 
                     old_basis: List[np.ndarray],
                     new_basis: List[np.ndarray]) -> np.ndarray:
        """
        Transform state coordinates from old basis to new basis.
        """
        # Get old coordinates
        old_coords = self.expand_in_basis(state, old_basis)
        
        # Compute change of basis matrix
        n = len(old_basis)
        U = np.zeros((n, n), dtype=complex)
        for i, new_b in enumerate(new_basis):
            for j, old_b in enumerate(old_basis):
                U[i, j] = self.inner_product(new_b, old_b)
        
        new_coords = U @ np.array(old_coords)
        return new_coords
    
    def tensor_product(self, psi: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """Compute |psi⟩ ⊗ |phi⟩."""
        return np.kron(psi, phi)
    
    def apply_operator(self, operator: np.ndarray, 
                       state: np.ndarray) -> np.ndarray:
        """Apply operator to state."""
        return operator @ state
    
    def expectation_value(self, operator: np.ndarray, 
                          state: np.ndarray) -> complex:
        """Compute ⟨state|operator|state⟩."""
        return np.vdot(state, operator @ state)


# Demonstration
print("\n" + "="*60)
print("Quantum State Space Demonstration")
print("="*60)

qt = QuantumStateTools(dim=2)

# Standard computational basis
print("\n1. Computational Basis:")
ket_0 = qt.computational_basis[0]
ket_1 = qt.computational_basis[1]
print(f"   |0⟩ = {ket_0}")
print(f"   |1⟩ = {ket_1}")
print(f"   Orthonormal: {qt.is_orthonormal_basis([ket_0, ket_1])}")

# Hadamard basis
print("\n2. Hadamard Basis:")
ket_plus = qt.create_state([1, 1])
ket_minus = qt.create_state([1, -1])
print(f"   |+⟩ = {ket_plus}")
print(f"   |-⟩ = {ket_minus}")
print(f"   Orthonormal: {qt.is_orthonormal_basis([ket_plus, ket_minus])}")

# State in different bases
print("\n3. State in Different Bases:")
psi = qt.create_state([0.6, 0.8])
print(f"   |ψ⟩ = {psi}")

coords_comp = qt.expand_in_basis(psi, [ket_0, ket_1])
coords_had = qt.expand_in_basis(psi, [ket_plus, ket_minus])

print(f"   In computational basis: {[f'{c:.4f}' for c in coords_comp]}")
print(f"   In Hadamard basis: {[f'{c:.4f}' for c in coords_had]}")

# Probabilities
print("\n4. Measurement Probabilities:")
print(f"   P(|0⟩) = {qt.probability(psi, ket_0):.4f}")
print(f"   P(|1⟩) = {qt.probability(psi, ket_1):.4f}")
print(f"   P(|+⟩) = {qt.probability(psi, ket_plus):.4f}")
print(f"   P(|-⟩) = {qt.probability(psi, ket_minus):.4f}")

# Two-qubit system
print("\n5. Two-Qubit System:")
qt2 = QuantumStateTools(dim=4)

# |00⟩ and |11⟩
ket_00 = qt.tensor_product(ket_0, ket_0)
ket_11 = qt.tensor_product(ket_1, ket_1)

# Bell state
bell_state = qt2.create_state([1, 0, 0, 1])  # (|00⟩ + |11⟩)/√2
print(f"   Bell state |Φ+⟩ = {bell_state}")
print(f"   Dimension: {len(bell_state)} (2² = 4)")

# Pauli matrices
print("\n6. Pauli Matrices (Operators):")
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

print(f"   σx|0⟩ = {qt.apply_operator(sigma_x, ket_0)} = |1⟩")
print(f"   σz|+⟩ = {qt.apply_operator(sigma_z, ket_plus)}")

# Expectation values
print("\n7. Expectation Values for |ψ⟩:")
print(f"   ⟨ψ|σx|ψ⟩ = {qt.expectation_value(sigma_x, psi).real:.4f}")
print(f"   ⟨ψ|σy|ψ⟩ = {qt.expectation_value(sigma_y, psi).real:.4f}")
print(f"   ⟨ψ|σz|ψ⟩ = {qt.expectation_value(sigma_z, psi).real:.4f}")
```

---

## 📝 Lab Report Assignment

Create a Jupyter notebook that:

1. **Vector Space Verification** (20%)
   - Test a polynomial space for vector space axioms
   - Create a "broken" space (modified operations) and show which axiom fails

2. **Subspace Analysis** (20%)
   - Find bases for three different subspaces of ℝ⁴
   - Compute their dimensions
   - Find the intersection of two of them

3. **Independence and Basis** (30%)
   - Given 5 vectors in ℝ⁴, find which subsets are linearly independent
   - Extract a basis from the given vectors
   - Extend the basis to all of ℝ⁴

4. **Quantum Application** (30%)
   - Implement a 3-level quantum system (qutrit)
   - Create three different orthonormal bases
   - Transform a state between bases
   - Compute measurement probabilities in each basis

Save your notebook as `Week13_Lab_Report.ipynb` in the Week_13 directory.

---

## ✅ Lab Completion Checklist

- [ ] Completed Lab 1: Vector space axiom verification
- [ ] Completed Lab 2: Subspace tools
- [ ] Completed Lab 3: Linear algebra tools
- [ ] Completed Lab 4: Visualizations saved
- [ ] Completed Lab 5: Quantum state tools
- [ ] Lab report notebook started
- [ ] All code tested and commented
- [ ] Key outputs saved/screenshotted

---

## 🔜 Tomorrow: Review and Assessment

Day 91 (Sunday) will cover:
- Week summary and concept review
- Spaced repetition practice
- Self-assessment questions
- Preparation for Week 14 (Linear Transformations)

---

*"The computer is incredibly fast, accurate, and stupid. Man is incredibly slow, inaccurate, and brilliant. The marriage of the two is a force beyond calculation."*
— Leo Cherne
