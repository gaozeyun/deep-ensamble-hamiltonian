# Example-data provenance

The three structures in `strain_example/test_result.h5` are copied from the archived untwisted bilayer-graphene strain evaluation. They contain the real `H_pred`, `label`, and `mask` arrays required by `tools/error_analysis.py`. The matching `hamiltonians_std.h5` files are unchanged copies. No model predictions or DFT labels have been synthesized.

The six CSV files contain the full archived bilayer/twisted-bilayer sweeps for bond length, interlayer spacing and strain. They retain their original headers and values. The plotting notebook adapts the first-column header to the English name expected by the plotting script. It does not alter numerical values. These archives contain denser sweeps than some displayed manuscript panels; the example reproduces the analysis workflow rather than claiming pixel-identical manuscript figures.

Hamiltonian MAE and standard deviation are in eV in the stored tables; MSE is in eV squared. `plot_error_analysis.py` converts these to meV and meV squared. The three-structure sample is only an executable workflow example, not a new validation study or calibration set. These are evaluation outputs, not the training dataset. The source training data are available from [DeepH-E3 Dataset1](https://doi.org/10.5281/zenodo.7553640).

`provenance.json` records archive-relative source paths, selected structure identifiers and checksums.

## Band-calculation example

`band_example` contains the matched DFT and ensemble-mean Hamiltonians, overlap matrices and structural metadata for `t-4-0` in the archived bilayer strain sweep. The existing archived band configuration supplies 16 bands along a 45-point Gamma-M-K-Gamma path; its Fermi energy equals the value in this structure's `info.json`. `band_example/provenance.json` records each source file and checksum. Notebook 02 calculates both sets of bands with the existing `tools/sparse_calc.py`; notebook 03 plots them with `tools/band_plotter.py`. This is a small reproducibility example, not a new DFT calculation or a substitute for a full band-accuracy benchmark.
