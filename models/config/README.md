# Saved training configurations

These four files are byte-for-byte copies of the `src/train.ini` files stored with the released model runs. The original copies remain alongside the checkpoints. `manifest.json` records each source path and SHA-256 checksum.

| Configuration | Initialization seed | Input ordering |
| --- | ---: | --- |
| [model-1.ini](model-1.ini) | 42 | Original |
| [model-2.ini](model-2.ini) | 2026 | Original |
| [model-3.ini](model-3.ini) | 42 | Reversed |
| [model-4.ini](model-4.ini) | 2026 | Reversed |

All four runs use the bilayer-graphene source dataset released with DeepH-E3 ([Dataset1](https://doi.org/10.5281/zenodo.7553640)). The network and optimization settings match across the four saved configurations. In addition to the seed, the differing fields are the input-data path and the run-specific output/cache paths. The second input path represents the reversed input ordering described in the manuscript; the INI files do not themselves encode or construct that ordering.

Shared saved values include `cutoff_radius = 7.2`, `spherical_harmonics_lmax = 5`, three message-passing blocks, Adam with learning rate 0.003 and betas (0.9, 0.999), batch size 1, and `min_lr = 1e-4`. The saved split sizes are 180/60/60 and the epoch limit is 1895. These are the values in the released training artifacts.

The data loader shuffles dataset indices with the configured random seed before selecting training, validation and test subsets. A common source dataset and common split sizes therefore do not imply identical split membership across seeds or input orderings. Once trained, the same four model checkpoints are used across the structural evaluation tasks.

The absolute filesystem paths in the original INI files record the training environment. Adjust paths in a working copy when reproducing a run; do not treat these paths as portable defaults. See [the training and evaluation notebook](../../doc/01_training_and_evaluation.ipynb).
