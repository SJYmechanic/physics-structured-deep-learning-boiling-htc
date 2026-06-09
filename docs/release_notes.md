# Reviewer-facing release notes

Included in this cleaned code release:

- Core Distributed-DNN training code (`code/TwoPhaseFlow.py`)
- Monolithic ANN baseline training script (`code/train_monolithic.py`)
- Data-schema documentation (`data/README.md`)
- Usage documentation (`docs/code_usage.md`)

Excluded from the reviewer-facing release:

- Plotting scripts and manuscript figure-generation code
- Full reconstructed database
- Exhaustive architecture-scan and freeze-mask-screening artifacts
- Local path-dependent run folders and intermediate search outputs

The uploaded scan scripts were not included in the public-ready tree because the manuscript-code-availability target is A2: monolithic ANN + final Distributed-DNN training/evaluation, not the full internal search history.
