#!/usr/bin/env python3
"""
Comprehensive Jupyter Notebook Validator for SIIEA Quantum Engineering Curriculum
=================================================================================

Audit 1: Structural Validation
  - nbformat compliance
  - Cell ID uniqueness
  - Cell ordering
  - Metadata correctness
  - Source formatting (line endings)
  - Empty cells
  - Output artifacts
  - Cell count sanity

Audit 2: UI/UX Quality
  - Header consistency
  - Section structure
  - Markdown quality (LaTeX, formatting)
  - Code cell quality (comments, imports, plt)
  - Narrative flow
  - Curriculum day references
  - QM connections
  - Summary/review section

Severity levels:
  CRITICAL  - Broken notebook, won't open in Jupyter
  HIGH      - Missing required sections, broken formatting
  MEDIUM    - UX issues that affect learning experience
  LOW       - Style inconsistencies
  INFO      - Suggestions for improvement
"""

import json
import re
import os
import sys
from pathlib import Path
from collections import Counter, defaultdict

# ─── Configuration ────────────────────────────────────────────────────
NOTEBOOK_BASE = Path(__file__).parent
EXPECTED_KERNELSPEC_NAME = "quantum-eng"
EXPECTED_KERNELSPEC_LANGUAGE = "python"
MIN_CELLS = 16
MAX_CELLS = 26
NBFORMAT = 4
NBFORMAT_MINOR = 5

# ─── Issue tracking ──────────────────────────────────────────────────
class Issue:
    def __init__(self, severity, category, message, cell_idx=None, cell_id=None):
        self.severity = severity
        self.category = category
        self.message = message
        self.cell_idx = cell_idx
        self.cell_id = cell_id

    def __repr__(self):
        loc = ""
        if self.cell_idx is not None:
            loc = f" [cell {self.cell_idx}"
            if self.cell_id:
                loc += f" ({self.cell_id})"
            loc += "]"
        return f"  [{self.severity:8s}] ({self.category}) {self.message}{loc}"

SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}


def sort_issues(issues):
    return sorted(issues, key=lambda i: SEVERITY_ORDER.get(i.severity, 5))


# ═══════════════════════════════════════════════════════════════════════
# AUDIT 1: STRUCTURAL VALIDATION
# ═══════════════════════════════════════════════════════════════════════

def audit_structure(nb_path, nb_data):
    """Run all structural validation checks on a notebook."""
    issues = []

    # 1. nbformat compliance
    issues += check_nbformat(nb_data)

    # 2. Valid JSON is implicit (we parsed it)

    # 3. Cell IDs unique
    issues += check_cell_ids(nb_data)

    # 4. Cell ordering
    issues += check_cell_ordering(nb_data)

    # 5. Metadata
    issues += check_metadata(nb_data)

    # 6. Source formatting
    issues += check_source_formatting(nb_data)

    # 7. Empty cells
    issues += check_empty_cells(nb_data)

    # 8. Output artifacts
    issues += check_output_artifacts(nb_data)

    # 9. Cell count
    issues += check_cell_count(nb_data)

    return issues


def check_nbformat(nb):
    issues = []
    if nb.get("nbformat") != NBFORMAT:
        issues.append(Issue("CRITICAL", "nbformat",
                            f"nbformat={nb.get('nbformat')}, expected {NBFORMAT}"))
    if nb.get("nbformat_minor") != NBFORMAT_MINOR:
        issues.append(Issue("LOW", "nbformat",
                            f"nbformat_minor={nb.get('nbformat_minor')}, expected {NBFORMAT_MINOR}"))
    return issues


def check_cell_ids(nb):
    issues = []
    cells = nb.get("cells", [])
    ids = [c.get("id") for c in cells]
    missing = [i for i, cid in enumerate(ids) if cid is None]
    for i in missing:
        issues.append(Issue("HIGH", "cell_ids", "Cell has no 'id' field", cell_idx=i))

    # Check uniqueness
    id_counts = Counter(cid for cid in ids if cid is not None)
    for cid, count in id_counts.items():
        if count > 1:
            indices = [i for i, c in enumerate(cells) if c.get("id") == cid]
            issues.append(Issue("CRITICAL", "cell_ids",
                                f"Duplicate cell ID '{cid}' appears {count} times at indices {indices}"))
    return issues


def check_cell_ordering(nb):
    issues = []
    cells = nb.get("cells", [])
    if not cells:
        issues.append(Issue("CRITICAL", "cell_ordering", "Notebook has no cells"))
        return issues

    # First cell must be markdown header
    if cells[0].get("cell_type") != "markdown":
        issues.append(Issue("HIGH", "cell_ordering",
                            f"First cell is '{cells[0].get('cell_type')}', expected 'markdown' (header)",
                            cell_idx=0, cell_id=cells[0].get("id")))

    # Second cell should be code (hardware detection)
    if len(cells) > 1 and cells[1].get("cell_type") != "code":
        issues.append(Issue("MEDIUM", "cell_ordering",
                            f"Second cell is '{cells[1].get('cell_type')}', expected 'code' (hardware detection)",
                            cell_idx=1, cell_id=cells[1].get("id")))
    elif len(cells) > 1:
        src = "".join(cells[1].get("source", []))
        if "hardware" not in src.lower() and "Hardware" not in src:
            issues.append(Issue("MEDIUM", "cell_ordering",
                                "Second cell (code) does not appear to be hardware detection",
                                cell_idx=1, cell_id=cells[1].get("id")))

    return issues


def check_metadata(nb):
    issues = []
    meta = nb.get("metadata", {})

    # kernelspec
    ks = meta.get("kernelspec", {})
    if not ks:
        issues.append(Issue("HIGH", "metadata", "No kernelspec in metadata"))
    else:
        if ks.get("name") != EXPECTED_KERNELSPEC_NAME:
            issues.append(Issue("HIGH", "metadata",
                                f"kernelspec.name='{ks.get('name')}', expected '{EXPECTED_KERNELSPEC_NAME}'"))
        if ks.get("language") != EXPECTED_KERNELSPEC_LANGUAGE:
            issues.append(Issue("HIGH", "metadata",
                                f"kernelspec.language='{ks.get('language')}', expected '{EXPECTED_KERNELSPEC_LANGUAGE}'"))

    # language_info
    li = meta.get("language_info", {})
    if not li:
        issues.append(Issue("MEDIUM", "metadata", "No language_info in metadata"))
    elif li.get("name") != "python":
        issues.append(Issue("MEDIUM", "metadata",
                            f"language_info.name='{li.get('name')}', expected 'python'"))

    return issues


def check_source_formatting(nb):
    issues = []
    cells = nb.get("cells", [])

    for i, cell in enumerate(cells):
        src = cell.get("source", [])
        if not src:
            continue  # handled by empty cell check

        # Check that all lines except the last end with \n
        for j, line in enumerate(src):
            if j < len(src) - 1:
                if not line.endswith("\n"):
                    issues.append(Issue("LOW", "source_format",
                                        f"Line {j} does not end with \\n (non-last line)",
                                        cell_idx=i, cell_id=cell.get("id")))
            else:
                # Last line should NOT end with \n (standard nbformat convention)
                if line.endswith("\n"):
                    # This is allowed but unusual; only flag as INFO
                    pass  # Many generators include trailing \n; not flagging

    return issues


def check_empty_cells(nb):
    issues = []
    cells = nb.get("cells", [])

    for i, cell in enumerate(cells):
        src = cell.get("source", [])
        joined = "".join(src).strip()
        if not joined:
            issues.append(Issue("MEDIUM", "empty_cell",
                                f"Empty {cell.get('cell_type', 'unknown')} cell",
                                cell_idx=i, cell_id=cell.get("id")))
    return issues


def check_output_artifacts(nb):
    issues = []
    cells = nb.get("cells", [])

    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue

        # execution_count should be null
        ec = cell.get("execution_count")
        if ec is not None:
            issues.append(Issue("MEDIUM", "output_artifacts",
                                f"execution_count={ec}, expected null (clean notebook)",
                                cell_idx=i, cell_id=cell.get("id")))

        # outputs should be empty
        outputs = cell.get("outputs", [])
        if outputs:
            issues.append(Issue("MEDIUM", "output_artifacts",
                                f"Cell has {len(outputs)} output(s), expected empty (clean notebook)",
                                cell_idx=i, cell_id=cell.get("id")))

    return issues


def check_cell_count(nb):
    issues = []
    n = len(nb.get("cells", []))
    if n < MIN_CELLS:
        issues.append(Issue("MEDIUM", "cell_count",
                            f"Only {n} cells, expected at least {MIN_CELLS}"))
    elif n > MAX_CELLS:
        issues.append(Issue("LOW", "cell_count",
                            f"{n} cells, expected at most {MAX_CELLS} (may be fine for capstone)"))
    else:
        issues.append(Issue("INFO", "cell_count", f"{n} cells (within {MIN_CELLS}-{MAX_CELLS} range)"))
    return issues


# ═══════════════════════════════════════════════════════════════════════
# AUDIT 2: UI/UX QUALITY
# ═══════════════════════════════════════════════════════════════════════

def audit_ux(nb_path, nb_data):
    """Run all UI/UX quality checks on a notebook."""
    issues = []

    issues += check_header_consistency(nb_data)
    issues += check_section_structure(nb_data)
    issues += check_markdown_quality(nb_data)
    issues += check_code_quality(nb_data)
    issues += check_narrative_flow(nb_data)
    issues += check_curriculum_days(nb_data, nb_path)
    issues += check_qm_connections(nb_data)
    issues += check_summary_section(nb_data)

    return issues


def get_cell_source(cell):
    """Get joined source text from a cell."""
    return "".join(cell.get("source", []))


def check_header_consistency(nb):
    issues = []
    cells = nb.get("cells", [])
    if not cells:
        return issues

    first_src = get_cell_source(cells[0])

    # Must start with "# " (H1)
    if not first_src.strip().startswith("# "):
        issues.append(Issue("HIGH", "header",
                            "First cell does not start with '# Title'",
                            cell_idx=0, cell_id=cells[0].get("id")))

    # Must contain SIIEA reference
    if "SIIEA" not in first_src and "Siiea" not in first_src:
        issues.append(Issue("HIGH", "header",
                            "First cell missing 'SIIEA Quantum Engineering Curriculum'",
                            cell_idx=0, cell_id=cells[0].get("id")))

    # Must contain license reference
    if "CC BY-NC-SA 4.0" not in first_src:
        issues.append(Issue("HIGH", "header",
                            "First cell missing 'CC BY-NC-SA 4.0' license reference",
                            cell_idx=0, cell_id=cells[0].get("id")))

    return issues


def check_section_structure(nb):
    issues = []
    cells = nb.get("cells", [])

    # Count markdown cells with section headers (## or ###)
    section_headers = []
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "markdown":
            continue
        src = get_cell_source(cell)
        # Find lines starting with ## (but not ###) or ###
        for line in src.split("\n"):
            stripped = line.strip()
            if stripped.startswith("## "):
                section_headers.append((i, stripped, cell.get("id")))
            elif stripped.startswith("### "):
                pass  # subsections are fine

    if len(section_headers) < 3:
        issues.append(Issue("MEDIUM", "section_structure",
                            f"Only {len(section_headers)} section headers (## level), "
                            f"expected at least 3 for clear structure"))

    return issues


def check_markdown_quality(nb):
    issues = []
    cells = nb.get("cells", [])

    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "markdown":
            continue
        src = get_cell_source(cell)
        cid = cell.get("id")

        # Check for LaTeX math blocks (at least one notebook-wide)
        # We'll aggregate this after the loop

        # Check for broken markdown: unclosed bold
        # Count ** pairs - should be even
        bold_count = src.count("**")
        if bold_count % 2 != 0:
            issues.append(Issue("HIGH", "markdown_quality",
                                f"Odd number of '**' ({bold_count}): possible unclosed bold",
                                cell_idx=i, cell_id=cid))

        # Check for unclosed backticks (single)
        # Count ` that are not part of ``` blocks
        src_no_codeblock = re.sub(r'```[\s\S]*?```', '', src)
        backtick_count = src_no_codeblock.count('`')
        if backtick_count % 2 != 0:
            issues.append(Issue("MEDIUM", "markdown_quality",
                                f"Odd number of backticks ({backtick_count}): possible unclosed inline code",
                                cell_idx=i, cell_id=cid))

        # Check for raw HTML that should be markdown
        html_tags = re.findall(r'<(?!/?(?:br|hr|sub|sup|img|!--))[a-z]+[\s>]', src, re.I)
        # Filter out common acceptable HTML in notebooks
        bad_html = [t for t in html_tags if not any(
            ok in t.lower() for ok in ['<br', '<hr', '<sub', '<sup', '<img', '<!--']
        )]
        if bad_html:
            issues.append(Issue("LOW", "markdown_quality",
                                f"Raw HTML tags found: {bad_html[:3]} - consider using markdown",
                                cell_idx=i, cell_id=cid))

        # Check tables - look for | header | header | pattern
        table_lines = [l for l in src.split("\n") if l.strip().startswith("|")]
        for tl in table_lines:
            # Basic check: should have at least 2 |
            pipe_count = tl.count("|")
            if pipe_count < 2:
                issues.append(Issue("LOW", "markdown_quality",
                                    f"Possible malformed table row: '{tl.strip()[:60]}'",
                                    cell_idx=i, cell_id=cid))

    # Check LaTeX presence across entire notebook
    all_md = " ".join(get_cell_source(c) for c in cells if c.get("cell_type") == "markdown")
    if "$$" not in all_md and "$" not in all_md:
        issues.append(Issue("MEDIUM", "markdown_quality",
                            "No LaTeX math blocks found in any markdown cell"))
    elif "$$" not in all_md:
        issues.append(Issue("LOW", "markdown_quality",
                            "No display math ($$...$$) found; only inline math ($...$)"))

    return issues


def check_code_quality(nb):
    issues = []
    cells = nb.get("cells", [])

    code_cells = [(i, c) for i, c in enumerate(cells) if c.get("cell_type") == "code"]

    if not code_cells:
        issues.append(Issue("HIGH", "code_quality", "No code cells in notebook"))
        return issues

    # Check each code cell has a comment explaining what it does
    for i, cell in code_cells:
        src = get_cell_source(cell)
        lines = src.strip().split("\n")
        if not lines:
            continue

        # First non-empty line should be a comment or the cell should have comments
        has_comment = any(line.strip().startswith("#") for line in lines[:5])
        if not has_comment:
            issues.append(Issue("MEDIUM", "code_quality",
                                "Code cell lacks explanatory comment in first 5 lines",
                                cell_idx=i, cell_id=cell.get("id")))

    # Check imports: should be concentrated at top (cells 1-3)
    # Find cells with import statements
    import_cells = []
    for i, cell in code_cells:
        src = get_cell_source(cell)
        if re.search(r'^\s*(?:import |from .+ import )', src, re.MULTILINE):
            import_cells.append(i)

    # Hardware detection cell (typically cell index 1) and main import cell (index 2)
    # are expected. Imports beyond index 4 or 5 are scattered.
    late_imports = [ci for ci in import_cells if ci > 5]
    if late_imports:
        # Check if they're re-imports or new imports
        for ci in late_imports:
            src = get_cell_source(cells[ci])
            import_lines = re.findall(r'^\s*(?:import |from .+ import ).+', src, re.MULTILINE)
            # Filter out "from IPython" and similar notebook-specific imports
            real_imports = [l for l in import_lines if not any(
                ok in l for ok in ['IPython', 'display', 'HTML']
            )]
            if real_imports:
                issues.append(Issue("LOW", "code_quality",
                                    f"Import statements found late in notebook: {real_imports[0].strip()[:60]}",
                                    cell_idx=ci, cell_id=cells[ci].get("id")))

    # Check for plt.show() after plot creation
    for i, cell in code_cells:
        src = get_cell_source(cell)
        has_plot = ("plt.plot" in src or "plt.bar" in src or "plt.contourf" in src or
                    "plt.fill" in src or "ax.plot" in src or "ax.bar" in src or
                    "ax.contourf" in src or "plot_surface" in src or
                    "plt.subplots" in src or "plt.figure" in src)
        if has_plot:
            if "plt.show()" not in src:
                issues.append(Issue("MEDIUM", "code_quality",
                                    "Plot created but plt.show() not called",
                                    cell_idx=i, cell_id=cell.get("id")))

    # Check for plt.tight_layout() in multi-panel figures
    for i, cell in code_cells:
        src = get_cell_source(cell)
        if "plt.subplots" in src and "subplots(1, 1" not in src:
            if "tight_layout" not in src:
                issues.append(Issue("LOW", "code_quality",
                                    "Multi-panel figure without plt.tight_layout()",
                                    cell_idx=i, cell_id=cell.get("id")))

    # Check for explicit figsize
    for i, cell in code_cells:
        src = get_cell_source(cell)
        if ("plt.figure(" in src or "plt.subplots(" in src):
            if "figsize" not in src:
                issues.append(Issue("LOW", "code_quality",
                                    "Figure created without explicit figsize",
                                    cell_idx=i, cell_id=cell.get("id")))

    return issues


def check_narrative_flow(nb):
    issues = []
    cells = nb.get("cells", [])

    # Check for theory -> code -> visualization pattern
    # At minimum: markdown cells should precede code cells in most cases
    consecutive_code = 0
    max_consecutive_code = 0
    for cell in cells:
        if cell.get("cell_type") == "code":
            consecutive_code += 1
            max_consecutive_code = max(max_consecutive_code, consecutive_code)
        else:
            consecutive_code = 0

    if max_consecutive_code > 3:
        issues.append(Issue("MEDIUM", "narrative_flow",
                            f"Up to {max_consecutive_code} consecutive code cells without markdown explanation"))

    # Check that the notebook doesn't start with many code cells without explanation
    code_before_second_md = 0
    found_first_md = False
    for cell in cells:
        if cell.get("cell_type") == "markdown":
            if found_first_md:
                break
            found_first_md = True
        elif found_first_md:
            code_before_second_md += 1

    # The hardware + imports cells (2 code cells) after first markdown are expected
    if code_before_second_md > 3:
        issues.append(Issue("LOW", "narrative_flow",
                            f"{code_before_second_md} code cells before second markdown explanation"))

    return issues


def check_curriculum_days(nb, nb_path):
    issues = []
    cells = nb.get("cells", [])
    if not cells:
        return issues

    first_src = get_cell_source(cells[0])

    # Check for day range mention
    has_days = bool(re.search(r'(?:Days?|Curriculum Days)', first_src, re.I))
    if not has_days:
        issues.append(Issue("MEDIUM", "curriculum_days",
                            "Header does not mention curriculum day range",
                            cell_idx=0, cell_id=cells[0].get("id")))

    # Validate day range against notebook path
    path_str = str(nb_path)

    # Map month numbers to expected day ranges
    month_day_map = {
        "month_01": (1, 28),
        "month_02": (29, 56),
        "month_03": (57, 84),
        "month_04": (85, 112),
        "month_05": (113, 140),
        "month_06": (141, 168),
        "month_07": (169, 196),
        "month_08": (197, 224),
        "month_09": (225, 252),
        "month_10": (253, 280),
        "month_11": (281, 308),
        "month_12": (309, 336),
    }

    for month_key, (start, end) in month_day_map.items():
        if month_key in path_str:
            # Check if the day range is mentioned
            day_pattern = re.search(r'(\d+)\s*[-–]\s*(\d+)', first_src)
            if day_pattern:
                found_start = int(day_pattern.group(1))
                found_end = int(day_pattern.group(2))
                if found_start != start or found_end != end:
                    issues.append(Issue("MEDIUM", "curriculum_days",
                                        f"Day range {found_start}-{found_end} does not match "
                                        f"expected {start}-{end} for {month_key}",
                                        cell_idx=0, cell_id=cells[0].get("id")))
            break

    # MLX labs are a special case - they reference Year 1 or specific day ranges
    if "mlx_labs" in path_str:
        # These just need some curriculum reference
        if not has_days and "Year" not in first_src and "Semester" not in first_src:
            issues.append(Issue("LOW", "curriculum_days",
                                "MLX lab header has no curriculum timing reference"))

    return issues


def check_qm_connections(nb):
    issues = []
    cells = nb.get("cells", [])

    all_text = " ".join(get_cell_source(c) for c in cells
                        if c.get("cell_type") == "markdown")

    # Look for quantum mechanics connections
    qm_keywords = [
        "quantum", "qm", "qubit", "wavefunction", "schrodinger", "schrödinger",
        "hamiltonian", "hilbert", "operator", "eigenvalue", "eigenstate",
        "superposition", "entangle", "born rule", "measurement",
        "unitary", "hermitian", "spin", "angular momentum", "planck",
    ]

    found = [kw for kw in qm_keywords if kw.lower() in all_text.lower()]

    if not found:
        issues.append(Issue("MEDIUM", "qm_connection",
                            "No quantum mechanics connection found in any markdown cell"))
    elif len(found) < 3:
        issues.append(Issue("LOW", "qm_connection",
                            f"Minimal QM connection: only found keywords {found}"))

    # Check for explicit QM connection section header
    has_qm_section = bool(re.search(
        r'##.*(?:quantum|QM|connection|bridge)', all_text, re.I))
    if not has_qm_section:
        issues.append(Issue("INFO", "qm_connection",
                            "No dedicated QM connection section header found (## level)"))

    return issues


def check_summary_section(nb):
    issues = []
    cells = nb.get("cells", [])

    if not cells:
        return issues

    # Check last 3 cells for summary content
    last_cells = cells[-3:]
    last_md = [get_cell_source(c) for c in last_cells if c.get("cell_type") == "markdown"]

    has_summary = False
    for src in last_md:
        if any(kw in src.lower() for kw in ["summary", "review", "takeaway",
                                              "key formula", "what we learned",
                                              "recap", "conclusion"]):
            has_summary = True
            break

    if not has_summary:
        # Check all markdown cells (sometimes summary is not the very last)
        all_md = [get_cell_source(c) for c in cells if c.get("cell_type") == "markdown"]
        for src in all_md:
            if re.search(r'##.*(?:summary|review|takeaway|conclusion)', src, re.I):
                has_summary = True
                break

    if not has_summary:
        issues.append(Issue("MEDIUM", "summary",
                            "No summary/review section found at end of notebook"))

    # Check if notebook ends with markdown (nice conclusion)
    if cells[-1].get("cell_type") != "markdown":
        issues.append(Issue("LOW", "summary",
                            f"Notebook ends with {cells[-1].get('cell_type')} cell, "
                            f"not markdown (summary/conclusion)"))

    return issues


# ═══════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ═══════════════════════════════════════════════════════════════════════

def find_notebooks(base_path):
    """Find all .ipynb files recursively."""
    notebooks = sorted(base_path.rglob("*.ipynb"))
    # Exclude checkpoint files
    notebooks = [nb for nb in notebooks if ".ipynb_checkpoints" not in str(nb)]
    return notebooks


def validate_notebook(nb_path):
    """Validate a single notebook file."""
    issues = []
    rel_path = nb_path.relative_to(NOTEBOOK_BASE)

    # Try to load as JSON
    try:
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb_data = json.load(f)
    except json.JSONDecodeError as e:
        issues.append(Issue("CRITICAL", "json", f"Invalid JSON: {e}"))
        return issues
    except Exception as e:
        issues.append(Issue("CRITICAL", "file", f"Cannot read file: {e}"))
        return issues

    # Run Audit 1: Structural Validation
    issues += audit_structure(nb_path, nb_data)

    # Run Audit 2: UI/UX Quality
    issues += audit_ux(nb_path, nb_data)

    return issues


def print_report(all_results):
    """Print formatted report of all issues."""
    severity_counts = Counter()
    total_issues = 0
    notebooks_with_issues = 0
    clean_notebooks = 0

    print("\n")
    print("=" * 80)
    print("  JUPYTER NOTEBOOK VALIDATION REPORT")
    print("  SIIEA Quantum Engineering Curriculum")
    print("=" * 80)
    print(f"\n  Notebooks scanned: {len(all_results)}")
    print()

    for nb_path, issues in all_results:
        rel_path = nb_path.relative_to(NOTEBOOK_BASE)

        # Filter out INFO for issue counting
        real_issues = [i for i in issues if i.severity != "INFO"]

        if real_issues:
            notebooks_with_issues += 1
        else:
            clean_notebooks += 1

        # Print notebook header
        status_icon = "PASS" if not real_issues else "ISSUES"
        print(f"\n{'─' * 80}")
        print(f"  [{status_icon}] {rel_path}")
        print(f"{'─' * 80}")

        if not issues:
            print("  No issues found.")
            continue

        sorted_issues = sort_issues(issues)
        for issue in sorted_issues:
            severity_counts[issue.severity] += 1
            if issue.severity != "INFO":
                total_issues += 1
            print(issue)

    # Summary
    print(f"\n\n{'=' * 80}")
    print("  SUMMARY")
    print(f"{'=' * 80}")
    print(f"\n  Total notebooks:           {len(all_results)}")
    print(f"  Notebooks with issues:     {notebooks_with_issues}")
    print(f"  Clean notebooks:           {clean_notebooks}")
    print(f"\n  Issue breakdown:")
    for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]:
        count = severity_counts.get(severity, 0)
        marker = "***" if severity == "CRITICAL" and count > 0 else ""
        print(f"    {severity:8s}: {count:4d}  {marker}")
    print(f"    {'TOTAL':8s}: {total_issues:4d}  (excluding INFO)")

    # Category breakdown
    print(f"\n  Issues by category:")
    cat_counts = Counter()
    for _, issues in all_results:
        for i in issues:
            if i.severity != "INFO":
                cat_counts[i.category] += 1
    for cat, cnt in cat_counts.most_common():
        print(f"    {cat:25s}: {cnt:4d}")

    print(f"\n{'=' * 80}")

    return severity_counts


def main():
    print("Scanning for notebooks in:", NOTEBOOK_BASE)
    notebooks = find_notebooks(NOTEBOOK_BASE)
    print(f"Found {len(notebooks)} notebook(s)")

    if not notebooks:
        print("ERROR: No notebooks found!")
        sys.exit(1)

    all_results = []
    for nb_path in notebooks:
        issues = validate_notebook(nb_path)
        all_results.append((nb_path, issues))

    severity_counts = print_report(all_results)

    # Exit with error code if CRITICAL issues found
    if severity_counts.get("CRITICAL", 0) > 0:
        sys.exit(2)
    return 0


if __name__ == "__main__":
    main()
