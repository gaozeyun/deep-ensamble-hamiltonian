# Reproducible notebooks

Run the notebooks in the following order:

1. [Model training and evaluation](01_training_and_evaluation.ipynb): prepare local configurations and invoke the existing DeepH-E3 training and ensemble-evaluation entry points.
2. [Data analysis and band calculations](02_data_analysis_and_bands.ipynb): calculate Hamiltonian errors and ensemble disagreement from real archived outputs, verify them against the archived table, and run the existing band solver for DFT-reference and ensemble-mean Hamiltonians.
3. [Result visualization](03_result_visualization.ipynb): use existing plotting scripts to save six structural-sweep figures and one band-comparison figure individually, without assembling composite manuscript figures.

Install the notebook dependencies with `python -m pip install -r doc/requirements.txt`, then open the notebooks with `python -m jupyterlab doc`. Select a Python kernel with those packages installed. Full training and inference additionally require the upstream DeepH-E3 environment and preprocessed data; these two expensive operations are disabled by default. The included error-analysis, band-calculation and plotting examples execute on a CPU from the supplied real data. Stored outputs demonstrate those examples, not a new full training or inference run.

The training dataset is the bilayer-graphene subset of [DeepH-E3 Dataset1](https://doi.org/10.5281/zenodo.7553640). Download `Bilayer_graphene_dataset.zip` and follow its README. The four saved configurations, original-path mapping and checksums are in [models/config](../models/config). The model files use Git LFS; run `git lfs pull` after cloning if checkpoints are still pointer files.

Generated configurations, analysis tables, band files, logs and figures are saved under `doc/output`, which is ignored by Git. Example-data origins, units and scope are recorded in [data/README.md](data/README.md). The notebooks call the existing scientific scripts.

## Additional script documentation

[Training-set geometry analysis](training_geometry.md) provides instructions for the standalone script that reconstructs C–C bond lengths and local perpendicular interlayer spacings from the included coordinate extract and seed-42 manifest. This analysis remains separate from the three notebook workflows.
