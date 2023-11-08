from pathlib import Path
import torch

from deepsoftlog.data.dataloader import DataLoader
from deepsoftlog.embeddings.initialize_vector import SPECIAL_MODELS
from deepsoftlog.embeddings.nn_models import AdditionFunctor, CarryFunctor
from deepsoftlog.experiments.mnist_addition.dataset import mnist_addition_dataset, AdditionDataset, MnistQueryDataset
from deepsoftlog.training import load_program, load_config
from deepsoftlog.training.logger import WandbLogger
from deepsoftlog.training.loss import nll_loss, get_optimizer
from deepsoftlog.training.trainer import Trainer

_EXP_ROOT = Path(__file__).parent


def get_pretrain_dataloader(cfg):
    pretrain_dataset = AdditionDataset(1, 2500).randomly_mutate_output()
    return DataLoader(pretrain_dataset, batch_size=cfg['batch_size'])


def get_train_dataloader(cfg):
    train_dataset = mnist_addition_dataset(cfg['digits'], "training")
    train_dataset = train_dataset.random_subset(cfg['data_subset'])
    dataloader = DataLoader(train_dataset, batch_size=cfg['batch_size'])
    return dataloader


def get_test_dataloader(cfg):
    eval_dataset = mnist_addition_dataset(cfg['digits'], "test")
    return DataLoader(eval_dataset, batch_size=1, shuffle=False)


def get_val_dataloader(cfg):
    eval_dataset = mnist_addition_dataset(cfg['digits'], "val").randomly_mutate_output()
    return DataLoader(eval_dataset, batch_size=cfg['batch_size'], shuffle=False)


def train(config_name):
    cfg = load_config(config_name)
    if cfg['add_symb']:
        SPECIAL_MODELS[('mod_ten_add', 3)] = AdditionFunctor
        SPECIAL_MODELS[('carry', 3)] = CarryFunctor

    # Training
    val_dataloader = get_val_dataloader(cfg)
    program = load_program(cfg, val_dataloader)
    optimizer = get_optimizer(program.get_store(), cfg)
    logger = WandbLogger(cfg)
    trainer = Trainer(program, get_train_dataloader, nll_loss, optimizer, logger=logger)
    trainer.train(cfg, nb_workers=cfg['nb_workers'])
    trainer.eval(get_test_dataloader(cfg))


def eval(folder: str, digits=None):
    cfg = load_config(f"results/{folder}/config.yaml")
    if digits is not None:
        cfg['digits'] = digits
    if cfg['add_symb']:
        SPECIAL_MODELS[('mod_ten_add', 3)] = AdditionFunctor
        SPECIAL_MODELS[('carry', 3)] = CarryFunctor
    print("EVALING", cfg['name'], cfg['digits'])

    test_dataloader = get_test_dataloader(cfg)
    program = load_program(cfg, test_dataloader)
    state_dict = torch.load(f"results/{folder}/store.pt")
    program.store.load_state_dict(state_dict, strict=False)
    trainer = Trainer(program, None, None, None)

    print("Digit accuracy:")
    digit_dataset = MnistQueryDataset("test")
    trainer.eval(DataLoader(digit_dataset, batch_size=4, shuffle=False))
    print("Predicate accuracy:")
    trainer.eval(test_dataloader)


if __name__ == "__main__":
    # eval("1d run", digits=1)
    train("deepsoftlog/experiments/mnist_addition/config.yaml")
