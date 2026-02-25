# Day 327: Capstone Project — Testing and Validation

## Overview

**Month 12, Week 47, Day 5 — Friday**

Today you systematically test and validate your implementation against known analytical results, ensuring correctness before documentation.

## Learning Objectives

1. Design comprehensive test suite
2. Validate against analytical results
3. Test edge cases
4. Document any limitations

---

## Testing Framework

```python
"""
Day 327: Testing and Validation
"""

import numpy as np
import unittest


class TestQuantumHarmonicOscillator(unittest.TestCase):
    """Comprehensive test suite for QHO implementation."""

    def setUp(self):
        """Initialize test objects."""
        from day_324 import QuantumHarmonicOscillator
        self.qho = QuantumHarmonicOscillator()
        self.x = np.linspace(-10, 10, 1000)

    def test_energy_eigenvalues(self):
        """Test E_n = ℏω(n + 1/2)."""
        for n in range(10):
            E = self.qho.energy(n)
            expected = n + 0.5  # In natural units
            self.assertAlmostEqual(E, expected, places=10)

    def test_normalization(self):
        """Test ∫|ψ_n|² dx = 1."""
        for n in range(10):
            psi = self.qho.eigenfunction(n, self.x)
            norm = np.trapz(np.abs(psi)**2, self.x)
            self.assertAlmostEqual(norm, 1.0, places=2)

    def test_orthogonality(self):
        """Test ⟨ψ_n|ψ_m⟩ = δ_nm."""
        for n in range(5):
            for m in range(5):
                psi_n = self.qho.eigenfunction(n, self.x)
                psi_m = self.qho.eigenfunction(m, self.x)
                overlap = np.trapz(psi_n * psi_m, self.x)
                expected = 1.0 if n == m else 0.0
                self.assertAlmostEqual(overlap, expected, places=2)

    def test_parity(self):
        """Test ψ_n(-x) = (-1)^n ψ_n(x)."""
        for n in range(5):
            psi_pos = self.qho.eigenfunction(n, self.x)
            psi_neg = self.qho.eigenfunction(n, -self.x)
            parity = (-1)**n
            np.testing.assert_array_almost_equal(psi_neg, parity * psi_pos, decimal=5)

    def test_ground_state_minimum_uncertainty(self):
        """Test Δx·Δp = ℏ/2 for ground state."""
        psi = self.qho.eigenfunction(0, self.x)

        # Position uncertainty
        x_avg = np.trapz(self.x * np.abs(psi)**2, self.x)
        x2_avg = np.trapz(self.x**2 * np.abs(psi)**2, self.x)
        delta_x = np.sqrt(x2_avg - x_avg**2)

        # For ground state: Δx = √(ℏ/2mω), Δp = √(mωℏ/2)
        # In natural units: Δx·Δp = 1/2
        # Theory: delta_x = 1/√2 ≈ 0.707
        self.assertAlmostEqual(delta_x, 1/np.sqrt(2), places=2)

    def test_numerical_vs_analytical(self):
        """Compare numerical and analytical eigenvalues."""
        x = np.linspace(-15, 15, 500)
        eigenvalues, _ = self.qho.solve_numerically(x, n_states=10)

        for n in range(5):
            E_numerical = eigenvalues[n]
            E_analytical = self.qho.energy(n)
            error = abs(E_numerical - E_analytical) / E_analytical
            self.assertLess(error, 0.01)  # Less than 1% error


def run_validation():
    """Run all validation tests."""
    print("=" * 60)
    print("VALIDATION TEST SUITE")
    print("=" * 60)

    # Run unittest
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestQuantumHarmonicOscillator)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✓ ALL TESTS PASSED")
    else:
        print("\n✗ SOME TESTS FAILED")

    return result.wasSuccessful()


if __name__ == "__main__":
    run_validation()
```

---

## Today's Checklist

- [ ] Test suite created
- [ ] All tests pass
- [ ] Edge cases tested
- [ ] Limitations documented
- [ ] Code ready for documentation

---

## Preview: Day 328

Tomorrow: **Documentation** — create comprehensive project documentation.
