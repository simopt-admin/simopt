---
name: New Model/Problem
about: Add a new model/problem to the codebase.
title: "New Model/Problem: <short description>"
labels: "enhancement"
---

## Linked Issue (Optional)
Closes #

## Model/Problem Overview
Provide a brief overview of the new model/problem being added. Include its purpose, key features, and any relevant background information.

## Gap Analysis
Describe the gaps in existing models/problems that this new model/problem addresses. Explain why this model/problem is necessary and how it improves upon or complements existing solutions.

## Checklist
- [ ] `ruff check` passes on my code
- [ ] `ty check` passes on my code
- [ ] I have included model/problem documentation in `docs/source/models/` based on the `model_template.rst` file and confirmed the documentation builds without warnings/errors
- [ ] I have included results generated via the `scripts/generate_experiment_results.py` script