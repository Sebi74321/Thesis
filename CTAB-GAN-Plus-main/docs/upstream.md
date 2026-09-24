# Upstream CTAB-GAN+

The `model/` implementation is adapted from CTAB-GAN+ by Zilong Zhao,
Aditya Kunar, Robert Birke, Hiek Van Der Scheer, and Lydia Y. Chen.

- [Paper](https://arxiv.org/abs/2204.00401)
- [Upstream repository](https://github.com/Team-TUD/CTAB-GAN-Plus)
- [Upstream DP implementation](https://github.com/Team-TUD/CTAB-GAN-Plus-DP)

The thesis code has its own environment and runners. The notes below are
historical upstream documentation, not installation instructions for this fork.
Use `setup_env.sh` and the root requirements files for current experiments.

## Prerequisite

The required package version
```
numpy==1.21.0
torch==1.9.1
pandas==1.2.4
sklearn==0.24.1
dython==0.6.4.post1
scipy==1.4.1
```
The sklean package in newer version has updated its function for `sklearn.mixture.BayesianGaussianMixture`. Therefore, user should use this proposed sklearn version to successfully run the code!

## Example
`Experiment_Script_Adult.ipynb`  `Experiment_Script_king.ipynb` are two example notebooks for training CTAB-GAN+ with Adult (classification) and king (regression) datasets. The datasets are alread under `Real_Datasets` folder.
The evaluation code is also provided.

## Problem type

You can either indicate your dataset problem type as Classification, Regression. If there is no problem type, you can leave the problem type as None as follows:
```
problem_type= {None: None}
```

## For large dataset

If your dataset has large number of column, you may encounter the problem that our currnet code cannot encode all of your data since CTAB-GAN+ will wrap the encoded data into an image-like format. What you can do is changing the line 378 and 385 in `model/synthesizer/ctabgan_synthesizer.py`. The number in the `slide` list
```
sides = [4, 8, 16, 24, 32]
```
is the side size of image. You can enlarge the list to [4, 8, 16, 24, 32, 64] or [4, 8, 16, 24, 32, 64, 128] for accepting larger dataset.


## Bibtex

To cite this paper, you could use this bibtex

```
@article{zhao2023ctab,
  title={Ctab-gan+: Enhancing tabular data synthesis},
  author={Zhao, Zilong and Kunar, Aditya and Birke, Robert and Van der Scheer, Hiek and Chen, Lydia Y},
  journal={Frontiers in big Data},
  volume={6},
  year={2023},
  publisher={Frontiers Media SA}
}
```
