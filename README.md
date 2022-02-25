# Posterior Meta-Replay for Continual Learning

In this study we propose posterior-replay continual learning, a framework for continually learning task-specific posterior approximations within a single shared meta-model. Across a range of experiments, we compare this approach to prior-focused CL, for which a single trade-off solution across all tasks is recursively obtained. Please see [our paper](https://arxiv.org/abs/2103.01133) for more details. You can also checkout our [poster presentation](https://neurips.cc/virtual/2021/poster/27724) or [this longer talk](https://www.youtube.com/watch?v=IF5twAVZ4Ns) in which we present the project.

If you are interested in working with **hypernetworks in PyTorch**, check out the package [hypnettorch](https://github.com/chrhenning/hypnettorch).

### 1D Regression Experiments

You can find instructions on how to reproduce our 1D Regression experiments and on how to use the corresponding code in the subfolder [probabilistic/regression](probabilistic/regression).

### 2D Mode Classification Experiments

You can find instructions on how to reproduce our 2D Mode Classification experiments and on how to use the corresponding code in the subfolder [probabilistic/prob_gmm](probabilistic/prob_gmm).

### Split and Permuted MNIST Experiments

You can find instructions on how to reproduce our Split and Permuted MNIST experiments and on how to use the corresponding code in the subfolder [probabilistic/prob_mnist](probabilistic/prob_mnist).

### SplitCIFAR-10/100 Experiments

You can find instructions on how to reproduce our SplitCIFAR-10 and SplitCIFAR-100 experiments and on how to use the corresponding code in the subfolder [probabilistic/prob_cifar](probabilistic/prob_cifar).

## Documentation

Please refer to the [README](docs/README.md) in the subfolder [docs](docs) for instructions on how to compile and open the documentation.

## Citation

Please cite our corresponding paper if you use this code in your research project.

```
@inproceedings{posterior:replay:2021:henning:cervera,
title={Posterior Meta-Replay for Continual Learning}, 
      author={Christian Henning and Maria R. Cervera and Francesco D'Angelo and Johannes von Oswald and Regina Traber and Benjamin Ehret and Seijin Kobayashi and João Sacramento and Benjamin F. Grewe},
booktitle={Conference on Neural Information Processing Systems},
year={2021},
url={https://arxiv.org/abs/2103.01133}
}
```
