# DMM-WcycleGAN Project Summary

## Overview

This folder contains a local engineering reproduction of the paper-aligned
`DMM-WcycleGAN` workflow built around the public Neural Latents Benchmark
`MC_Maze_Small` dataset.

The current project covers:

- public NLB dataset download and local storage
- raw NWB to official NLB H5 preprocessing
- official H5 to GAN-ready `npz` export
- paper-aligned `DMM-WcycleGAN` model definition
- end-to-end training, validation, test evaluation, and inference on unlabeled eval data
- notebook-based workflow with in-notebook parameter definitions

This is an engineering implementation for reproducible local experiments.
Some steps still use public-data bridges rather than the paper's original
closed-source feature extraction pipeline.

## Current Structure

```text
GAN/DMM-WcycleGAN/
├── Chen ... .pdf.md
├── DMM-WcycleGAN.py
├── dataset_loader.py
├── trainAndEval.py
├── pipeline_entry.ipynb
├── datasets/
│   └── mc_maze_small/
├── processed/
│   └── mc_maze_small/
│       ├── official_h5/
│       └── gan_ready/
└── results/
    └── mc_maze_small_demo/
```

## File Responsibilities

### `DMM-WcycleGAN.py`

Core model definition and training primitives.

Contains:

- `DMMWcycleGANConfig`
- dual generators `G1` and `G2`
- dual WGAN critics `D1` and `D2`
- online CNN classifier `Du`
- three-stage trainer skeleton:
  - meta-initialization
  - fine-tuning
  - online classifier training

This file is the model backbone used by `trainAndEval.py`.

### `dataset_loader.py`

Data import and preprocessing pipeline for the public NLB dataset.

Main responsibilities:

- locate the training NWB file inside downloaded DANDI folders
- handle compatibility fixes for `pynwb`, `nlb_tools`, and time index sorting
- export official NLB split tensors:
  - `*_train.h5`
  - `*_val.h5`
  - `*_full.h5`
- convert official H5 tensors to GAN-ready `npz` archives

The current GAN-ready export includes arrays such as:

- `encoder_input`
- `heldout_target`
- `full_target`
- `forward_target`
- `extra__behavior`

### `trainAndEval.py`

End-to-end train/validate/evaluate entrypoint.

Main responsibilities:

- load `gan_ready/train.npz` and `gan_ready/eval.npz`
- derive 3-class labels from behavior direction
- adapt spike-sequence tensors into the model input shape `(80, 8)`
- build deterministic train/val/test splits
- run:
  - meta-initialization
  - domain fine-tuning
  - online classifier training
- save:
  - checkpoint
  - histories
  - test predictions
  - eval predictions
  - summary metrics
- export optional plot images during training runs

Important note:

- label construction is currently based on coarse behavior direction sectors
- feature construction is currently an engineering adapter from NLB spike tensors

These two parts are the main gap between the current implementation and a
strict paper-faithful closed-source preprocessing pipeline.

### `pipeline_entry.ipynb`

Notebook entry for the full workflow.

Current notebook behavior:

- defines parameters directly in notebook cells instead of shell commands
- downloads the dataset if needed
- runs preprocessing/export through imported Python modules
- visualizes raw spike tensors and behavior signals
- launches training and evaluation through imported Python modules
- reads back metrics and generated artifacts

The notebook is intended to be the most direct maintenance entrypoint.

### `Chen ... .pdf.md`

Markdown conversion of the target paper.

Used as the local textual reference for:

- architecture interpretation
- training-stage interpretation
- high-level preprocessing constraints

## Data and Artifact Directories

### `datasets/mc_maze_small`

Downloaded public NLB dataset from DANDI.

Current content:

- `000140/dandiset.yaml`
- train NWB file
- test NWB file

This is the raw source data directory.

### `processed/mc_maze_small`

Intermediate and model-ready exported data.

Current subdirectories:

- `official_h5/`
  - `mc_maze_small_train.h5`
  - `mc_maze_small_val.h5`
  - `mc_maze_small_full.h5`
- `gan_ready/`
  - `train.npz`
  - `eval.npz`
  - `summary.json`

This is the main reusable processed-data directory.

### `results/mc_maze_small_demo`

One retained reference run of the current pipeline.

Current artifacts:

- `checkpoint.pt`
- `histories.json`
- `summary.json`
- `split_indices.npz`
- `test_predictions.npz`
- `eval_predictions.npz`

This directory is the retained baseline run for comparison and inspection.

## Current End-to-End Workflow

1. Download `MC_Maze_Small` into `datasets/mc_maze_small`
2. Use `dataset_loader.py` to convert NWB to official H5
3. Export official H5 to GAN-ready `npz`
4. Use `trainAndEval.py` to:
   - derive labels
   - adapt features
   - split data
   - train model
   - evaluate metrics
   - save outputs
5. Use `pipeline_entry.ipynb` to inspect raw data and results visually

## Current Outputs and Metrics

The retained run in `results/mc_maze_small_demo` is currently the main
reference output directory. Its exact metrics are stored in:

- `results/mc_maze_small_demo/summary.json`

Typical saved summary fields include:

- config
- label metadata
- adapter metadata
- history summary
- checkpoint selection
- train / val / test metrics
- eval prediction distribution

## Engineering Constraints and Known Gaps

### Already solved

- DANDI dataset download integrated into the workflow
- `pynwb` and `nlb_tools` compatibility handling
- automatic training-file selection in multi-NWB directories
- non-monotonic NWB time index sorting before tensor extraction
- notebook-side visualization improved for raw data and results

### Still approximate

- labels are not the paper's original task labels
- `(80, 8)` input features are adapted from public NLB spike tensors
- the implementation is paper-aligned, not author-supplement exact

### Maintenance implication

If paper-faithful reproduction becomes the next goal, the first modules to
replace or revise are:

1. label generation in `trainAndEval.py`
2. feature adapter in `trainAndEval.py`
3. any data transformation assumptions in `dataset_loader.py`

## Recommended Maintenance Rule

When extending this folder, prefer keeping only:

- one raw dataset directory per benchmark
- one canonical processed directory per dataset
- one canonical result directory per representative experiment

Delete temporary smoke/debug outputs after verification so the folder remains
readable and the notebook continues to point at a single obvious baseline.
