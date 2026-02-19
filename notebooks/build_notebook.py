"""
SIIEA Quantum Engineering - Notebook Builder Utility

Creates properly formatted .ipynb files from Python definitions.
Used by generation scripts to produce curriculum notebooks.

Usage:
    from build_notebook import NotebookBuilder
    nb = NotebookBuilder("Title", "month_01_calculus_I/01_limits.ipynb")
    nb.md("# Theory")
    nb.code("import numpy as np")
    nb.save()
"""

import json
import os
from pathlib import Path


class NotebookBuilder:
    """Build a Jupyter notebook programmatically."""

    def __init__(self, title: str, output_path: str, curriculum_days: str = ""):
        self.title = title
        self.output_path = output_path
        self.curriculum_days = curriculum_days
        self.cells = []
        self._cell_id = 0

        # Add standard header
        self.md(f"""# {title}

**SIIEA Quantum Engineering Curriculum**
{"- **Curriculum Days:** " + curriculum_days if curriculum_days else ""}
- **License:** CC BY-NC-SA 4.0 | Siiea Innovations, LLC

---""")

        # Add hardware detection cell
        self.code("""# Hardware detection — adapts simulations to your machine
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath("__file__")), ".."))
try:
    from hardware_config import HARDWARE, get_max_qubits
    print(f"Hardware: {HARDWARE['chip']} | {HARDWARE['memory_gb']} GB | Profile: {HARDWARE['profile']}")
    print(f"Max qubits: {get_max_qubits('safe')} (safe) / {get_max_qubits('max')} (max)")
except ImportError:
    print("hardware_config.py not found — using defaults")
    print("Run setup.sh from the repo root to generate it")""")

    def md(self, source: str):
        """Add a markdown cell."""
        self._cell_id += 1
        self.cells.append({
            "cell_type": "markdown",
            "id": f"cell-{self._cell_id:04d}",
            "metadata": {},
            "source": source.split("\n")
            if "\n" not in source
            else [line + "\n" for line in source.split("\n")[:-1]] + [source.split("\n")[-1]],
        })

    def code(self, source: str):
        """Add a code cell."""
        self._cell_id += 1
        lines = source.split("\n")
        source_lines = [line + "\n" for line in lines[:-1]] + [lines[-1]]
        self.cells.append({
            "cell_type": "code",
            "execution_count": None,
            "id": f"cell-{self._cell_id:04d}",
            "metadata": {},
            "outputs": [],
            "source": source_lines,
        })

    def save(self):
        """Write the notebook to disk."""
        notebook = {
            "cells": self.cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Quantum Engineering",
                    "language": "python",
                    "name": "quantum-eng",
                },
                "language_info": {
                    "codemirror_mode": {"name": "ipython", "version": 3},
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "nbconvert_exporter": "python",
                    "pygments_lexer": "ipython3",
                    "version": "3.13.0",
                },
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }

        # Resolve path relative to notebooks/ directory
        base_dir = Path(__file__).parent
        full_path = base_dir / self.output_path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        with open(full_path, "w") as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)

        print(f"Created: {full_path}")
        return str(full_path)
