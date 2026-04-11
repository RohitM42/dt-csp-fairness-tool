# csp-fairness-tool

A CSP + Decision Tree hybrid approach to AI fairness testing, developed as part of the Intelligent Software Engineering (ISE) module.

## Overview

This tool finds **Individual Discriminatory Instances (IDIs)** in pre-trained neural network models — pairs of inputs identical except for a sensitive feature (e.g., gender, race) where the model predicts a different outcome.

The method combines:
- **Phase 1:** Random seed sampling + Decision Tree to learn which feature regions are likely to contain discrimination
- **Phase 2:** CP-SAT (Google OR-Tools) to systematically and diversely generate inputs in those regions

## Datasets

Targets: KDD, Adult, COMPAS, German, Dutch

## Usage

See `docs/manual.md` (export to `manual.pdf` for submission).

## Replication

See `docs/replication.md` (export to `replication.pdf` for submission).

## Dependencies

See `requirements.txt` and `docs/requirements.md` (export to `requirements.pdf` for submission).
