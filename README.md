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

The [doc directory](doc/README.md) contains Jupyter notebooks for model training and evaluation, data analysis including band calculations, and individual result figures using real archived data. The four saved training configurations are also collected in [models/config](models/config/README.md).
