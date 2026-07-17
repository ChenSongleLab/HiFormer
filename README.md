# HiFormer

Official implementation of **HiFormer: Hierarchical Transformer with
Box-packed Positional Encoding for 3D Part Assembly**.

S. Chen, L. Dong, Y. Zhou, S. Chen and K. Xu, "HiFormer: Hierarchical
Transformer With Box-Packed Positional Encoding for 3D Part Assembly,"
in *IEEE Transactions on Visualization and Computer Graphics*,
vol. 32, no. 7, pp. 5128–5143, July 2026.
[DOI](https://doi.org/10.1109/TVCG.2026.3662816)

HiFormer predicts the 6-DoF pose of each unlabeled part from its point cloud.
It combines multi-task 3D Swin Transformer features, hierarchical part-group
reasoning, and box-packed positional encoding. This release contains the
deterministic Chair training and evaluation pipeline.

## Architecture

<img width="4800" height="2780" alt="655a7cae65b1ded8e0dea53e11d35cd4" src="https://github.com/user-attachments/assets/dd8547f5-ec3b-453b-a763-1952ca998496" />


Using the provided 3D point clouds of parts without semantic information, HiFormer infers part relationships purely
from geometry and predicts the 6-DoF of each part to generate a perceptually consistent shape. The lower panel illustrates
the coarse-to-fine pipeline through six HiFormer layers, while the upper panel presents the Layer architecture with BPE and
hierarchical feature fusion, which processes refined part poses from Layer 1

## Project Structure

```text
.
|-- assets/                        # figures used by the documentation
|-- exps/
|   |-- TransAssembly/
|   |   |-- run.py                 # command-line entry point
|   |   |-- train.py               # one training epoch
|   |   |-- eval.py                # evaluation loop
|   |   |-- datasets/partnet.py    # processed PartNet loader
|   |   |-- models/                # assembly model, losses, inference
|   |   |-- scripts/               # runtime and checkpoint utilities
|   |   `-- third_party/pointnet2/ # retained development extension
|   `-- utils/
|       |-- cd/                    # Chamfer Distance extension
|       `-- quaternion.py
|-- lib/pointops2/                 # PointOps2 CUDA extension
|-- model/swin3d_transformer.py    # Swin point-cloud backbone
|-- prep_data/                     # PartNet preprocessing scripts
`-- pretrained_models/             # place released checkpoints here
```

Visualization code is not included in this release.

## Environment

The code was developed with Python 3.8, PyTorch 1.8.1, and CUDA 11.1 on
Linux. Set up the environment with:

```bash
conda env create -f environment.yml
conda activate transassembly

cd exps/utils/cd
python setup.py build develop
cd ../../..

cd lib/pointops2
python setup.py build_ext --inplace
cd ../..
```

## Dataset

Following a preprocessing setup similar to
[3DHPA](https://github.com/pkudba/3DHPA), we use the processed PartNet data
organized as:

```text
prepare_data/
|-- Chair.train.npy
|-- Chair.val.npy
|-- Chair.test.npy
|-- shape_data/
|   `-- <shape_id>_level3.npy
`-- contact_points/
    `-- pairs_with_contact_points_<shape_id>_level3.npy
```

The dataset is not included in this repository. See
[prep_data/README.md](prep_data/README.md) to prepare the data from PartNet.

## Quick Start

Run the following commands from the repository root.

### Training

```bash
python exps/TransAssembly/run.py \
  --data_dir /path/to/prepare_data \
  --output-dir outputs/chair
```

### Evaluation

```bash
python exps/TransAssembly/run.py \
  --eval-only 1 \
  --data_dir /path/to/prepare_data \
  --val_data_fn Chair.test.npy \
  --checkpoint pretrained_models/checkpoint_best.pth.tar \
  --output-dir outputs/chair_eval
```
## Results

### Diversity Modeling

The table reports results with various sampling times *N* for diversity
modeling on the PartNet dataset. Higher *N* consistently improves all
metrics as the model generates more candidate assemblies per shape.

| *N* | SCD ↓ | | | PA ↑ | | | CA ↑ | | | model |
|-----|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:---:|
| | Chair | Table | Lamp | Chair | Table | Lamp | Chair | Table | Lamp | |
| 10  | 0.0048 | 0.0032 | 0.0085 | 66.32 | 66.55 | 43.42 | 54.18 | 64.38 | 73.01 | [model](pretrained_models/diversity%20modeling/checkpoint_best.pth.tar) |
| 20  | 0.0047 | 0.0031 | 0.0082 | 66.66 | 66.74 | 44.21 | 54.23 | 64.77 | 74.53 | - |
| 30  | 0.0046 | 0.0030 | 0.0080 | 66.97 | 66.96 | 44.43 | 54.34 | 65.02 | 75.58 | - |
| 40  | 0.0045 | 0.0029 | 0.0079 | 67.22 | 67.01 | 44.81 | 54.39 | 65.10 | 76.34 | - |
| 50  | 0.0045 | 0.0029 | 0.0076 | **68.67** | **67.11** | **46.09** | **54.44** | **65.18** | **78.72** | - |

### Deterministic Modeling

Result of the deterministic baseline (single prediction, no noise).

| SCD ↓ | | | PA ↑ | | | CA ↑ | | | model |
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:---:|
| Chair | Table | Lamp | Chair | Table | Lamp | Chair | Table | Lamp | |
| 0.0056 | 0.0037 | 0.0108 | 61.36 | 62.35 | 35.31 | 45.88 | 61.10 | 56.89 | [model](pretrained_models/deterministic%20modeling/checkpoint_best.pth.tar) |

## Citation

If you find this project useful, please cite:

```bibtex
@article{chen2026hiformer,
  title   = {HiFormer: Hierarchical Transformer with Box-packed Positional Encoding for 3D Part Assembly},
  author  = {Chen, Songle and Dong, Lulu and Zhou, Yijiao and Chen, Siguang and Xu, Kai},
  journal = {IEEE Transactions on Visualization and Computer Graphics},
  year    = {2026},
  doi     = {10.1109/TVCG.2026.3662816}
}
```
