---
name: Release / Internal Merge
about: Merge `dev` -> `master` (Release), `hotfix` -> `master` (Hotfix), or `master` -> `dev` (Sync)
title: "[<RELEASE TYPE>] <Version # or Description>"
labels: "internal"
---

**INTERNAL USE ONLY**

## Release Type
- [ ] **Release:** Merging `dev` into `master`
- [ ] **Hotfix:** Merging `hotfix` into `master`
- [ ] **Sync:** Back-merging `master` into `dev`

## Summary
A brief description of the changes made in this PR.

## Versioning & Metadata
- [ ] Bumped version in `pyproject.toml`
- [ ] Bumped version in `docs/source/conf.py`
- [ ] Bumped version in `CITATION.cff`

## Post-Merge Reminder
- [ ] After merging, create a new GitHub Release / Tag matching the version number.