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
## Quick Start

Run the following commands from the repository root.

### Deterministic Training

Single-prediction baseline without noise injection.

```bash
python exps/TransAssembly/run.py \
  --data_dir /path/to/prepare_data \
  --backbone swin_transformer \
  --output-dir outputs/chair_det
```

### Deterministic Evaluation

```bash
python exps/TransAssembly/run.py \
  --eval-only 1 \
  --data_dir /path/to/prepare_data \
  --val_data_fn Chair.test.npy \
  --checkpoint outputs/chair_det/checkpoint_best.pth.tar \
  --output-dir outputs/chair_det_eval
```

### Diversity Training

Resume from a deterministic checkpoint with noise injection and
Min-of-N (MoN) sampling.

```bash
python exps/TransAssembly/run_sw.py \
  --data_dir /path/to/prepare_data \
  --resume outputs/chair_det/checkpoint_best.pth.tar \
  --output-dir outputs/chair_div
```

### Diversity Evaluation

```bash
python exps/TransAssembly/run_sw.py \
  --eval-only 1 \
  --data_dir /path/to/prepare_data \
  --val_data_fn Chair.test.npy \
  --checkpoint outputs/chair_div/checkpoint_best.pth.tar \
  --output-dir outputs/chair_div_eval
```
## Results

### Diversity Modeling

The table reports results with various sampling times *N* for diversity
modeling on the PartNet dataset. Higher *N* consistently improves all
metrics as the model generates more candidate assemblies per shape.

<table>
  <tr>
    <th rowspan="2"><em>N</em></th>
    <th colspan="3">SCD ↓</th>
    <th colspan="3">PA ↑</th>
    <th colspan="3">CA ↑</th>
    <th rowspan="2">model</th>
  </tr>
  <tr>
    <th>Chair</th><th>Table</th><th>Lamp</th>
    <th>Chair</th><th>Table</th><th>Lamp</th>
    <th>Chair</th><th>Table</th><th>Lamp</th>
  </tr>
  <tr align="center">
    <td>10</td>
    <td>0.0048</td><td>0.0032</td><td>0.0085</td>
    <td>66.32</td><td>66.55</td><td>43.42</td>
    <td>54.18</td><td>64.38</td><td>73.01</td>
    <td><a href="https://github.com/ChenSongleLab/HiFormer/releases/download/v1.0/diversity_checkpoint_best.pth.tar">model</a></td>
  </tr>
  <tr align="center">
    <td>20</td>
    <td>0.0047</td><td>0.0031</td><td>0.0082</td>
    <td>66.66</td><td>66.74</td><td>44.21</td>
    <td>54.23</td><td>64.77</td><td>74.53</td>
    <td>-</td>
  </tr>
  <tr align="center">
    <td>30</td>
    <td>0.0046</td><td>0.0030</td><td>0.0080</td>
    <td>66.97</td><td>66.96</td><td>44.43</td>
    <td>54.34</td><td>65.02</td><td>75.58</td>
    <td>-</td>
  </tr>
  <tr align="center">
    <td>40</td>
    <td>0.0045</td><td>0.0029</td><td>0.0079</td>
    <td>67.22</td><td>67.01</td><td>44.81</td>
    <td>54.39</td><td>65.10</td><td>76.34</td>
    <td>-</td>
  </tr>
  <tr align="center">
    <td>50</td>
    <td>0.0045</td><td>0.0029</td><td>0.0076</td>
    <td><b>68.67</b></td><td><b>67.11</b></td><td><b>46.09</b></td>
    <td><b>54.44</b></td><td><b>65.18</b></td><td><b>78.72</b></td>
    <td>-</td>
  </tr>
</table>

### Deterministic Modeling

Result of the deterministic baseline (single prediction, no noise).

<table>
  <tr>
    <th colspan="3">SCD ↓</th>
    <th colspan="3">PA ↑</th>
    <th colspan="3">CA ↑</th>
    <th>model</th>
  </tr>
  <tr>
    <th>Chair</th><th>Table</th><th>Lamp</th>
    <th>Chair</th><th>Table</th><th>Lamp</th>
    <th>Chair</th><th>Table</th><th>Lamp</th>
    <th></th>
  </tr>
  <tr align="center">
    <td>0.0056</td><td>0.0037</td><td>0.0108</td>
    <td>61.36</td><td>62.35</td><td>35.31</td>
    <td>45.88</td><td>61.10</td><td>56.89</td>
    <td><a href="https://github.com/ChenSongleLab/HiFormer/releases/download/v1.0/checkpoint_best.pth.tar">model</a></td>
  </tr>
</table>


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
