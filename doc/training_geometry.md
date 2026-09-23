# Training-set bond lengths and perpendicular layer spacings

The reproducible entry point is [tools/analyze_training_geometry.py](../tools/analyze_training_geometry.py), with [tools/training_geometry_style.py](../tools/training_geometry_style.py) providing the original plotting functions. These are copies of the established analysis used to generate the training-geometry figures; only filenames, imports and default filesystem paths have been adapted for this repository. The scientific functions are unchanged.

From the repository root, run:

```bash
python -m pip install -r doc/requirements.txt
python tools/analyze_training_geometry.py
```

The defaults use the included coordinate/lattice extract and seed-42 test manifest in `doc/data/training_geometry`. The compressed NPZ contains 300 structures with 64 atoms each, their lattices, structure identifiers and frozen dataset indices. It contains geometry only, without learned predictions or Hamiltonian matrices. The underlying structures are from the bilayer-graphene source dataset documented in [DeepH-E3 Dataset1](https://doi.org/10.5281/zenodo.7553640). The manifest preserves the archived graph hash, test indices and structure identifiers used by the original analysis. Checksums and source-script identifiers are recorded in the accompanying provenance file.

The script reconstructs the 180/60/60 train/validation/test partition with seed 42 and checks the test membership against the manifest. It identifies intralayer bonds independently in each structure using periodic distances and the separation between the third and fourth same-layer neighbors. Local perpendicular interlayer spacing is evaluated between periodic layer surfaces at matching in-plane positions using Delaunay interpolation. It is not the three-dimensional distance to the nearest opposite-layer atom.

The output directory is `doc/output/training_geometry`, including per-structure statistics, min/max and quantile tables, a split manifest, metadata and numerical checks. The `figures` subdirectory contains the geometric envelope, split-coverage and joint-domain figures in PNG and SVG. In particular, `figure_training_geometry_joint_domain.png` plots each structure's minimum against maximum bond length and local perpendicular spacing, with a convex hull based on the training subset. The hull is a geometric description, not a guarantee of accurate Hamiltonian predictions.

Optional arguments are `--input`, `--manifest`, `--output-dir` and `--figure-dir`. This script reproduces the specific frozen 300-structure, seed-42 analysis; its shape and manifest checks deliberately prevent silently applying those labels to an unrelated dataset. To analyze a different dataset, the dataset assumptions and split validation must be reconsidered explicitly.

The original validation routines check rigid-rotation invariance, perpendicular spacing under lateral shifts, and independent reconstruction of three-coordinated carbon bonds. No model training or DFT calculation is required. The automatically generated method report retains the original Chinese explanatory text; this document describes the workflow in English.
