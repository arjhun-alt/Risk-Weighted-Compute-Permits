# Risk-Weighted Compute Permits

Reproduction code for the two computational experiments reported in the
submitted paper:

1. **Implementation Verification**
2. **Behavioural Extension**

The repository contains no external dataset. The included results are
generated entirely from the formal model.

## Contents

- `experiments.py` - simulation and plotting code
- `reproduce.ipynb` - reruns both experiments and displays both figures inline
- `results/implementation_verification.csv` - results for Experiment 1
- `results/behavioural_extension.csv` - results for Experiment 2
- `results/summary.json` - headline reproduction summary

No figure files are included in the repository.

## Setup

Python 3.11-3.14 is supported. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[notebook]"
```

## Reproduce

To regenerate the included result files:

```bash
python experiments.py
```

To rerun the experiments and display both figures:

```bash
jupyter lab reproduce.ipynb
```

Run the two notebook cells in order. The first displays the implementation
verification plot; the second displays the behavioural extension plot.
