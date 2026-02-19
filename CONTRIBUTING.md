# Contributing to Quantum Engineering Curriculum

Thank you for your interest in contributing! This curriculum represents thousands of hours of work, and community contributions help make it better for everyone.

## How to Contribute

### Reporting Issues

Found an error in the math, a broken code example, or a concept that could be explained better? Open an issue:

1. Go to [Issues](https://github.com/siiea-ai/Quantum-Engineering/issues)
2. Use the appropriate issue template
3. Be specific — include the day file path (e.g., `Year_0/.../Day_042_Monday.md`) and line numbers

### Types of Contributions We Welcome

- **Error corrections** — Typos, mathematical errors, code bugs
- **Clarity improvements** — Better explanations for difficult concepts
- **Code improvements** — More efficient or clearer Python implementations
- **Additional practice problems** — New problems with solutions
- **Resource suggestions** — Links to helpful papers, videos, or tools
- **Translations** — Help make the curriculum accessible in other languages

### What We Don't Accept

- Changes that alter the curriculum structure or sequencing
- Content that requires paid resources or proprietary software
- Promotional content or links to commercial products
- Content that hasn't been verified for correctness

## Making Changes

### Setup

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/Quantum-Engineering.git
cd Quantum-Engineering

# Create a branch
git checkout -b fix/day-042-matrix-typo
```

### Branch Naming

- `fix/` — Error corrections (e.g., `fix/day-042-matrix-typo`)
- `improve/` — Clarity improvements (e.g., `improve/day-100-fourier-explanation`)
- `add/` — New content (e.g., `add/day-150-extra-problems`)

### Day File Standards

If editing a day file, maintain the existing structure:

1. **Schedule Overview** — 7-hour daily plan
2. **Learning Objectives** — 5-6 measurable objectives
3. **Core Content** — Theory with LaTeX (`$$...$$`)
4. **Quantum Mechanics Connection** — How it relates to QM
5. **Worked Examples** — Step-by-step solutions
6. **Practice Problems** — Three difficulty levels
7. **Computational Lab** — Runnable Python code
8. **Summary** — Key formulas and takeaways
9. **Daily Checklist** — Self-assessment
10. **Preview** — Next day teaser

### LaTeX Standards

- Display equations: `$$E = mc^2$$`
- Boxed key formulas: `$$\boxed{H\psi = E\psi}$$`
- Inline math: `$\psi$`

### Python Code Standards

- All code must be runnable with `numpy`, `scipy`, `matplotlib`, `sympy`
- Include comments explaining each step
- Use clear variable names that match the mathematical notation

### Submitting

1. Commit with a clear message:
   ```bash
   git commit -m "Fix matrix multiplication error in Day 042"
   ```
2. Push to your fork:
   ```bash
   git push origin fix/day-042-matrix-typo
   ```
3. Open a Pull Request using the PR template

## Review Process

1. All PRs are reviewed for mathematical correctness
2. Code examples are tested for execution
3. Changes must maintain the curriculum's quality standards
4. Reviews typically happen within 1-2 weeks

## License

By contributing, you agree that your contributions will be licensed under the same [CC BY-NC-SA 4.0](LICENSE) license that covers the project.

## Questions?

- Open a [Discussion](https://github.com/siiea-ai/Quantum-Engineering/discussions)
- Email: community@siiea.ai
