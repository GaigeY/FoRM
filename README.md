# FoRM: Foot-driven Reconstruction of Human Motion Using Dual-Modal Plantar Pressure and Inertial Sensing

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

## Introduction

This repository is the official PyTorch implementation for IMWUT (UbiComp) 2025 paper:
[**FoRM: Foot-driven Reconstruction of Human Motion Using Dual-Modal Plantar Pressure and Inertial Sensing**](https://doi.org/10.1145/3749551)

![Overview](./assets/over_picture.jpg)

## Installation

### 1. Python Environment Setup
Install the required Python packages using pip:
```bash
pip install -r requirements.txt
```

Key dependencies include:
- `PyTorch (>=1.10.0)` for model training/inference
- `joblib` for data loading utilities

### 2. SMPL Model Setup
Download the SMPL neutral model from [here](https://smpl.is.tue.mpg.de/download.php) (select "Download version 1.1.0 for Python 2.7 (female/male/neutral, 300 shape PCs)" and use the file `basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl`). Place it in the following directory:
```bash
$workspace/assets/smplx_models/smpl/SMPL_NEUTRAL.pkl
```

### 3. Model Checkpoint Setup
Download the pre-trained model checkpoint (FoRM.pth) and place it in the following directory:
```bash
$workspace/checkpoint/FoRM.pth
```
> Note: The download link for the model checkpoint will be provided soon.

## HAMPI Dataset

The HAMPI dataset consists of synchronized plantar pressure and inertial measurements with corresponding motion capture data.

Dataset specifications:

- Contains data from 18 subjects (aged 21-69) performing 18 types of daily activities
- Each sample includes:
  - Dual-modal smart shoes measurement with plantar pressure and dual IMUs per foot (60Hz)
  - SMPL pose parameters (60Hz)
  - Noitom motion capture refined output (60Hz)

Please see LICENSE for usage terms and citation requirements.

> We have prepared the dataset ready. We would release the dataset as soon as the paper is published.

<!-- ## Training -->

## Evaluation

To reproduce paper results:

```
python eval.py
```

Note: Reported metrics may vary slightly (±0.1%) due to:

- Hardware differences in floating-point computation
- Random seed initialization in probabilistic components
- Minor version differences in dependency libraries

## TODO

- [ ] HAMPI Dataset release
- [x] Evaluation scripts release
- [ ] Checkpoints and assests release
- [ ] Training scripts release

## Citation

Please cite the following paper if you use this code directly or indirectly in your research/projects:

```
@article{ying2025form,
  title={FoRM: Foot-driven Reconstruction of Human Motion Using Dual-Modal Plantar Pressure and Inertial Sensing},
  author={Qijun Ying, Zehua Cao, Ziyu Wu, Wenwu Deng, Yuchen Zhong, Yukun Diao, Xiaohui Cai},
  journal={Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (IMWUT)},
  year={2025}
}
```

## Acknowledgments
Partial code is adapted from [PIP](https://github.com/Xinyu-Yi/PIP), [WHAM](https://github.com/yohanshin/WHAM), and [GroundLink](https://github.com/hanxingjian/GroundLink).

## License

This repository is released under the MIT License.

## Contact

Please contact [Qijun Ying](yqj@mail.ustc.edu.cn) for any questions related to this work.
