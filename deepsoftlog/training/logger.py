from abc import ABC
from collections import defaultdict

import numpy as np
import wandb


class Logger(ABC):
    def log(self, values):
        raise NotImplementedError()

    def log_eval(self, values):
        raise NotImplementedError

    def print(self):
        pass


class PrintLogger(Logger):
    def __init__(self):
        self.storage = defaultdict(list)

    def log(self, values):
        for name, value in values.items():
            self.storage[name].append(value)

    def print(self):
        values = [f"{k}: {np.nanmean(v):.4f} ± {np.nanstd(v):.2f}" for k, v in self.storage.items()]
        print(" | ".join(values))
        self.storage = defaultdict(list)

    def log_eval(self, values, **kwargs):
        # TODO
        print("EVAL:", values)


class WandbLogger(Logger):
    def __init__(self, config):
        self.config = config

    def log(self, values):
        self._init_wandb()
        values = {"train/" + k: v for k, v in values.items()}
        wandb.log(values)

    def log_eval(self, values, name="test"):
        self._init_wandb()
        values = {f"{name}/" + k: v for k, v in values.items()}
        wandb.log(values)

    def _init_wandb(self):
        if wandb.run is None:
            name = self.config['name']
            project = self.config['project']
            wandb.init(name=name, project=project, config=self.config)

