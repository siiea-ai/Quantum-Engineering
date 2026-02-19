#!/usr/bin/env python3
"""
SIIEA Quantum Engineering - Notebook Execution Test Suite

Parses all 15 Jupyter notebooks, extracts code cells, and executes them
sequentially in a shared namespace per notebook (simulating Jupyter kernel
behavior). Reports pass/fail per cell and per notebook.

Usage:
    .venv/bin/python3 test_notebooks.py
"""

import json
import os
import sys
import io
import traceback
import time
import warnings
import re
from pathlib import Path

# ─── Configuration ──────────────────────────────────────────────────────────
BASE_DIR = Path("/Users/ali_personal/Projects/Quantum Engineering")
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

NOTEBOOK_PATHS = [
    "notebooks/year_0/month_01_calculus_I/01_limits_and_derivatives.ipynb",
    "notebooks/year_0/month_02_calculus_II/02_integration_and_series.ipynb",
    "notebooks/year_0/month_03_multivariable_ode/03_multivariable_and_odes.ipynb",
    "notebooks/year_0/month_04_linear_algebra_I/04_eigenvalues_and_transformations.ipynb",
    "notebooks/year_0/month_05_linear_algebra_II_complex/05_complex_analysis_foundations.ipynb",
    "notebooks/year_0/month_06_classical_mechanics/06_lagrangian_hamiltonian.ipynb",
    "notebooks/year_0/month_07_complex_analysis/07_contour_integration.ipynb",
    "notebooks/year_0/month_08_electromagnetism/08_maxwell_equations.ipynb",
    "notebooks/year_0/month_09_functional_analysis/09_hilbert_spaces.ipynb",
    "notebooks/year_0/month_10_scientific_computing/10_numerical_methods.ipynb",
    "notebooks/year_0/month_11_group_theory/11_symmetries_and_representations.ipynb",
    "notebooks/year_0/month_12_capstone/12_foundation_capstone.ipynb",
    "notebooks/mlx_labs/01_mlx_quantum_basics.ipynb",
    "notebooks/mlx_labs/02_large_scale_simulation.ipynb",
    "notebooks/mlx_labs/03_quantum_neural_network.ipynb",
]

# ─── Helpers ────────────────────────────────────────────────────────────────

def strip_magics(source: str) -> str:
    """Remove Jupyter magic lines (%) and shell commands (!) from source."""
    lines = source.split('\n')
    cleaned = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith('%') or stripped.startswith('!'):
            # Replace with a comment so line numbers stay aligned
            cleaned.append('# [SKIPPED MAGIC/SHELL] ' + line)
        else:
            cleaned.append(line)
    return '\n'.join(cleaned)


def extract_code_cells(notebook_path: str) -> list:
    """Parse a .ipynb file and return a list of (cell_index, source) tuples
    for code cells only."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = []
    for idx, cell in enumerate(nb.get('cells', [])):
        if cell.get('cell_type') == 'code':
            source = ''.join(cell.get('source', []))
            if source.strip():  # skip empty cells
                cells.append((idx, source))
    return cells


class CellResult:
    """Result of executing a single cell."""
    def __init__(self, notebook: str, cell_index: int, code_cell_num: int):
        self.notebook = notebook
        self.cell_index = cell_index  # index in the full notebook (including markdown)
        self.code_cell_num = code_cell_num  # 0-based code cell number
        self.passed = False
        self.error_type = None
        self.error_message = None
        self.traceback_str = None
        self.stdout = ""
        self.stderr = ""
        self.warnings_list = []
        self.execution_time = 0.0

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"Cell {self.code_cell_num} (nb_idx={self.cell_index}): {status}"


def execute_notebook(notebook_rel_path: str) -> list:
    """Execute all code cells in a notebook sequentially in a shared namespace.

    Returns a list of CellResult objects.
    """
    full_path = BASE_DIR / notebook_rel_path
    notebook_name = Path(notebook_rel_path).name
    code_cells = extract_code_cells(str(full_path))

    if not code_cells:
        print(f"  [WARN] No code cells found in {notebook_name}")
        return []

    # Fresh namespace for this notebook
    namespace = {}

    # Pre-inject matplotlib Agg backend to avoid display issues
    pre_inject = """
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
"""
    try:
        exec(compile(pre_inject, '<pre-inject>', 'exec'), namespace)
    except Exception:
        pass  # matplotlib might not be needed for all notebooks

    # Ensure hardware_config can be imported
    notebooks_dir = str(NOTEBOOKS_DIR)
    if notebooks_dir not in sys.path:
        sys.path.insert(0, notebooks_dir)

    # Also set the working directory context for relative paths in notebooks
    # (some notebooks save files relative to their location)
    notebook_dir = str(full_path.parent)
    original_cwd = os.getcwd()

    results = []
    for code_cell_num, (cell_index, source) in enumerate(code_cells):
        result = CellResult(notebook_name, cell_index, code_cell_num)

        # Clean the source
        cleaned_source = strip_magics(source)

        # Capture stdout/stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        captured_stdout = io.StringIO()
        captured_stderr = io.StringIO()
        sys.stdout = captured_stdout
        sys.stderr = captured_stderr

        try:
            os.chdir(notebook_dir)
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")

                t0 = time.perf_counter()
                # Compile and execute
                compiled = compile(cleaned_source, f'{notebook_name}:cell[{cell_index}]', 'exec')
                exec(compiled, namespace)
                result.execution_time = time.perf_counter() - t0

                result.passed = True
                result.warnings_list = [
                    f"{w.category.__name__}: {w.message}"
                    for w in caught_warnings
                ]

        except Exception as e:
            result.execution_time = time.perf_counter() - t0
            result.passed = False
            result.error_type = type(e).__name__
            result.error_message = str(e)
            result.traceback_str = traceback.format_exc()

        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            result.stdout = captured_stdout.getvalue()
            result.stderr = captured_stderr.getvalue()
            os.chdir(original_cwd)

            # Close any open matplotlib figures to prevent memory leaks
            try:
                import matplotlib.pyplot as _plt
                _plt.close('all')
            except Exception:
                pass

        results.append(result)

    return results


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 78)
    print("  SIIEA Quantum Engineering - Notebook Execution Test Suite")
    print("=" * 78)
    print(f"  Base directory: {BASE_DIR}")
    print(f"  Notebooks to test: {len(NOTEBOOK_PATHS)}")
    print(f"  Python: {sys.version}")
    print()

    # Verify all notebooks exist
    missing = []
    for nb in NOTEBOOK_PATHS:
        if not (BASE_DIR / nb).exists():
            missing.append(nb)
    if missing:
        print("ERROR: Missing notebooks:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)

    all_results = {}  # notebook -> list of CellResult
    total_cells = 0
    total_passed = 0
    total_failed = 0
    total_warnings = 0
    notebooks_passed = 0
    notebooks_failed = 0

    overall_start = time.perf_counter()

    for nb_idx, nb_path in enumerate(NOTEBOOK_PATHS, 1):
        nb_name = Path(nb_path).name
        print(f"[{nb_idx:>2}/{len(NOTEBOOK_PATHS)}] {nb_name}")
        print(f"       Path: {nb_path}")

        nb_start = time.perf_counter()
        results = execute_notebook(nb_path)
        nb_elapsed = time.perf_counter() - nb_start

        all_results[nb_path] = results

        nb_passed = sum(1 for r in results if r.passed)
        nb_failed = sum(1 for r in results if not r.passed)
        nb_warnings = sum(len(r.warnings_list) for r in results)

        total_cells += len(results)
        total_passed += nb_passed
        total_failed += nb_failed
        total_warnings += nb_warnings

        if nb_failed == 0:
            status = "PASS"
            notebooks_passed += 1
        else:
            status = "FAIL"
            notebooks_failed += 1

        print(f"       Status: {status}  |  Cells: {len(results)}  |  "
              f"Passed: {nb_passed}  |  Failed: {nb_failed}  |  "
              f"Warnings: {nb_warnings}  |  Time: {nb_elapsed:.2f}s")

        # Show failures inline
        for r in results:
            if not r.passed:
                print(f"         FAIL cell #{r.code_cell_num} (nb_idx={r.cell_index}): "
                      f"{r.error_type}: {r.error_message}")
        print()

    overall_elapsed = time.perf_counter() - overall_start

    # ─── Summary ────────────────────────────────────────────────────────
    print()
    print("=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    print(f"  Notebooks:  {len(NOTEBOOK_PATHS)} total  |  "
          f"{notebooks_passed} passed  |  {notebooks_failed} failed")
    print(f"  Cells:      {total_cells} total  |  "
          f"{total_passed} passed  |  {total_failed} failed  |  "
          f"{total_warnings} warnings")
    print(f"  Total time: {overall_elapsed:.2f}s")
    print()

    # ─── Detailed Failure Report ────────────────────────────────────────
    if total_failed > 0:
        print("=" * 78)
        print("  FAILURE DETAILS")
        print("=" * 78)

        for nb_path, results in all_results.items():
            failures = [r for r in results if not r.passed]
            if not failures:
                continue

            print(f"\n  Notebook: {nb_path}")
            print(f"  {'─' * 70}")
            for r in failures:
                print(f"\n  Cell #{r.code_cell_num} (notebook index {r.cell_index})")
                print(f"  Error type: {r.error_type}")
                print(f"  Error message: {r.error_message}")
                if r.traceback_str:
                    # Show last few lines of traceback
                    tb_lines = r.traceback_str.strip().split('\n')
                    # Show up to last 15 lines
                    show_lines = tb_lines[-15:]
                    for line in show_lines:
                        print(f"    {line}")
                print()

    # ─── Warning Summary ────────────────────────────────────────────────
    if total_warnings > 0:
        print("=" * 78)
        print("  WARNING SUMMARY")
        print("=" * 78)
        for nb_path, results in all_results.items():
            nb_warnings = []
            for r in results:
                for w in r.warnings_list:
                    nb_warnings.append((r.code_cell_num, w))
            if nb_warnings:
                print(f"\n  {Path(nb_path).name}:")
                # Deduplicate
                seen = set()
                for cell_num, warning in nb_warnings:
                    key = warning
                    if key not in seen:
                        seen.add(key)
                        print(f"    Cell #{cell_num}: {warning}")

    print()
    print("=" * 78)
    if total_failed == 0:
        print("  ALL NOTEBOOKS PASSED")
    else:
        print(f"  {total_failed} CELL(S) FAILED across {notebooks_failed} NOTEBOOK(S)")
    print("=" * 78)

    return 0 if total_failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
