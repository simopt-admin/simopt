---
name: New Solver
about: Add a new solver to the codebase.
title: "New Solver: <short description>"
labels: "enhancement"
---

## Linked Issue (Optional)
Closes #

## Solver Overview
Provide a brief overview of the new solver being added. Include its purpose, key features, and any relevant background information.

## Gap Analysis
Describe the gaps in existing solvers that this new solver addresses. Explain why this solver is necessary and how it improves upon or complements existing solutions.

## Checklist
- [ ] `ruff check` passes on my code
- [ ] `ty check` passes on my code
- [ ] I have included solver documentation in `docs/source/solvers/` based on the `solver_template.rst` file and confirmed the documentation builds without warnings/errors
- [ ] I have included results generated via the `scripts/generate_experiment_results.py` script