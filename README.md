# DeepSoftLog

DeepSoftLog is a neuro-symbolic framework which adds learnable embeddings to probabilistic logic programming.
For more information, see our paper: 

[Jaron Maene and Luc De Raedt. "Soft-Unification in Deep Probabilistic Logic." Advances in neural information processing systems 37 (2023).](https://openreview.net/pdf?id=s86M8naPSv)

## Install

DeepSoftLog requires Python 3.11 or higher. To install run:
```shell
python setup.py build_ext --inplace
```

Note that DeepSoftLog has only been tested on MacOS and Linux. The exact inference requires [PySDD](https://github.com/wannesm/PySDD), which does not support Windows.

## Experiments

All experiments can be found in `src/experiments`. The hyperparameters for each experiment can be found in their respective `config.yaml` files. 
To run an experiment, use the following command:
```shell
python run_experiments.py <experiment_name> <config_path>
```
For example, to run the MNIST addition experiment:
```shell
python run_experiments.py mnist_addition deepsoftlog/experiments/mnist_addition/config.yaml
```

By default, training metrics and results are logged to wandb.

## Paper

If you use DeepSoftLog in your work, consider citing our paper:

```
@inproceedings{
maene2023softunification,
title={Soft-Unification in Deep Probabilistic Logic},
author={Jaron Maene and Luc De Raedt},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=s86M8naPSv}
}
```