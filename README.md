# deep-ensamble-hamiltonian

## Overview

This repository contains the code used in the paper *[XXX]* (DOI: XXX), designed to help reproduce the research. The project implements a Deep Ensemble method based on the [DeepH-E3](https://github.com/Xiaoxun-Gong/DeepH-E3.git) framework. The ensemble mechanism is directly written into the original DeepH-E3 code without any modifications to the graph neural network architecture.

> **Note**: The modified sections in `kernel.py` and `parse_configs.py` are marked with comments `# DE-DeepH: ...` for easy identification.

## Based On

This codebase implements Deep Ensemble methods for Hamiltonian prediction, using [DeepH-E3](https://github.com/Xiaoxun-Gong/DeepH-E3.git) as an experimental framework for implementation reference.

## Training Dataset

The ensemble models in this work were trained using [dataset1](https://doi.org/10.5281/zenodo.7553640) from the DeepH-E3 dataset.

The train/validation/test split follows the standard DeepH-E3 data-loading procedure and is determined by the saved training configuration and random seed. No separate manually curated split file was used. The saved `train.ini`, dataset metadata, and training logs included with each model document the split settings used in the reported runs.

## Project Structure

The repository includes the following files and directories:

- `kernel.py`: Core ensemble module (modified from original DeepH-E3)
- `parse_configs.py`: Configuration file parser (modified from original DeepH-E3)
- `Bilayer_graphene_eval_ensemble.ini`: Reference evaluation configuration
- `tools/`: Utility scripts for generating OpenMX input files, running/post-processing Hamiltonian calculations, evaluating prediction errors, and plotting analysis results
- `models/`: Trained ensemble model artifacts and associated run metadata
- `models/config/`: The four saved training configurations, their model-to-seed mapping, and source-file checksums
- `doc/`: Three workflow notebooks, real example data, dependency requirements, and training-geometry analysis instructions

### Core File Descriptions

- **kernel.py**: Implements the core logic for ensemble evaluation, directly embedded into the DeepH-E3 inference workflow. All modifications are annotated with `# DE-DeepH:` comments.
- **parse_configs.py**: Extended configuration parser supporting ensemble-related parameter settings. All modifications are annotated with `# DE-DeepH:` comments.
- **Bilayer_graphene_eval_ensemble.ini**: An example evaluation configuration file for bilayer graphene, ready to use as a template.
- **tools/**: Contains utility scripts used in the data-generation and analysis workflow. In particular, `build_carbon.py`, `graphene_builder.py`, and `graphene_builder.sh` generate carbon-structure/OpenMX input files; `sparse_calc.py`, `sparse_calc.jl`, and `band_calc.sh` support Hamiltonian and band-structure post-processing; `error_analysis.py` and `run_error_analysis.sh` calculate MSE/MAE statistics from prediction results; and `plot_error_analysis.py`, `plot_structure_error.py`, `band_plotter.py`, and related shell scripts generate the corresponding analysis and band-structure figures.
- **models/**: Contains the four trained models used in this work, including model checkpoints, training configurations, target definitions, training logs, TensorBoard records, and train/test reports needed to inspect and reproduce the reported training runs.

## Installation and Usage

### Requirements

This project is built upon [DeepH-E3](https://github.com/Xiaoxun-Gong/DeepH-E3.git). Please ensure it is correctly installed.

### Integration Steps

1. **Replace Core Files**
   Copy `kernel.py` and `parse_configs.py` from this project to your DeepH-E3 installation directory, overwriting the original files:

       cp kernel.py path/to/your/DeepH-E3/deephe3/kernel.py
       cp parse_configs.py path/to/your/DeepH-E3/deephe3/parse_configs.py

2. **Configure Evaluation Parameters**
   Refer to the provided `Bilayer_graphene_eval_ensemble.ini` file and adjust the evaluation parameters according to your specific needs.

3. **Run Evaluation**
   Start the evaluation task following the standard DeepH-E3 workflow.

## Important Note

**Complex tensor operations are currently not supported.** Scenarios involving complex tensors, such as Hamiltonian matrices with spin-orbit coupling (SOC) effects, are beyond the scope of this work.

## Reproducible Notebooks

The [doc directory](doc/README.md) provides three notebooks. Run them in the following order:

| Notebook | Contents |
| --- | --- |
| [01: Model training and evaluation](doc/01_training_and_evaluation.ipynb) | Prepare local configurations and invoke the existing DeepH-E3 training and ensemble-evaluation entry points. |
| [02: Data analysis and band calculations](doc/02_data_analysis_and_bands.ipynb) | Calculate MAE, MSE and ensemble disagreement, compare the results with archived statistics, and run `tools/sparse_calc.py` for DFT-reference and ensemble-mean Hamiltonians. |
| [03: Result visualization](doc/03_result_visualization.ipynb) | Generate six structural-sweep figures and one band comparison with the existing plotting scripts. Figures are saved individually, without composite-figure assembly. |

From the repository root, install the notebook dependencies and launch JupyterLab:

```bash
python -m pip install -r doc/requirements.txt
python -m jupyterlab doc
```

The error-analysis, band-calculation and plotting examples run on a CPU using the included real archived data. These inputs include predictions and DFT labels for three structures, their ensemble standard-deviation files, six structural-sweep tables, and matching Hamiltonians, overlap matrices and structural metadata for a band-calculation example. Sources, units and checksums are documented in [doc/data](doc/data/README.md).

The notebooks have been executed with their default settings. The recomputed error statistics agree with the archived results, and the band example generates 16 bands at 45 k-points for each of the DFT-reference and ensemble-mean Hamiltonians. Full model training and inference require the DeepH-E3 environment and preprocessed dataset; their execution switches are disabled by default. Set the local paths and switches in notebook 01 to run those steps.

Generated configurations, analysis tables, band files, logs and figures are written to `doc/output/`, which is ignored by Git. For model checkpoints, use Git LFS and run `git lfs pull` after cloning.

## Saved Training Configurations

The original saved training configurations have been collected as [model-1.ini](models/config/model-1.ini), [model-2.ini](models/config/model-2.ini), [model-3.ini](models/config/model-3.ini), and [model-4.ini](models/config/model-4.ini). They are unchanged copies of the files stored with the four model runs.

The [configuration README](models/config/README.md) describes the initialization seeds, input ordering and shared settings. The [manifest](models/config/manifest.json) records the original file locations and SHA-256 checksums. Notebook 01 prepares working copies with local paths for reproduction.

## Training-Set Geometry Analysis

[tools/analyze_training_geometry.py](tools/analyze_training_geometry.py) reproduces the C–C bond-length and local perpendicular interlayer-spacing analysis. Its plotting functions are provided in [tools/training_geometry_style.py](tools/training_geometry_style.py).

After installing the notebook dependencies above, run the following command from the repository root:

```bash
python tools/analyze_training_geometry.py
```

The script uses the included coordinate/lattice extract and seed-42 manifest in `doc/data/training_geometry/` to analyze the 300 bilayer-graphene structures and their 180/60/60 training/validation/test split. This manifest records the frozen split used for the geometry analysis. Outputs include per-structure statistics, distribution summaries and joint-domain figures under `doc/output/training_geometry/`, with figures in its `figures/` subdirectory. See the [geometry-analysis instructions](doc/training_geometry.md) for input definitions, optional arguments and output descriptions.
