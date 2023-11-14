# DeepSoftLog

[[paper](https://openreview.net/pdf?id=s86M8naPSv)] [[video](https://youtu.be/3yQbcer-suA)] [[slides](https://neurips.cc/media/neurips-2023/Slides/70284.pdf)]

DeepSoftLog is a neuro-symbolic framework which adds embeddings and neural networks to probabilistic logic programming using soft-unification.


## Install

DeepSoftLog requires Python 3.10 or higher. To install run:
```shell
pip install cython==0.29.36
python setup.py build_ext --inplace
pip install -r requirements.txt
```

DeepSoftLog has only been tested on MacOS and Linux. The exact inference requires [PySDD](https://github.com/wannesm/PySDD), which does not support Windows.

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
@inproceedings{maene2023softunification,
  title={Soft-Unification in Deep Probabilistic Logic},
  author={Maene, Jaron and De Raedt, Luc},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```