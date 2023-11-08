from deepsoftlog.data.dataloader import DataLoader
from deepsoftlog.experiments.mnist_addition.dataset import MnistQueryDataset
from deepsoftlog.training import load_program, ConfigDict
from deepsoftlog.training.trainer import create_trainer


mnist_config = {
        "project": "mnist",
        "name": None,
        "program": "deepsoftlog/experiments/mnist_addition/programs/mnist.pl",
        "seed": 1337,
        "verbose": True,
        "device": "cpu",

        # optimization
        "optimizer": "AdamW",
        "functor_learning_rate": 0.0003,
        "embedding_learning_rate": 0,
        "functor_weight_decay": 0.00001,
        "nb_epochs": 10,
        "batch_size": 4,
        "grad_clip": None,

        # embeddings
        "semantics": "sdd2",
        "embedding_dimensions": 10,
        "embedding_metric": 'dot',
        "embedding_initialization": 'sphere',
        "data_subset": 1000,
    }


def get_train_dataloader(cfg: dict):
    train_dataset = MnistQueryDataset("training").subset(cfg['data_subset'])
    return DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, seed=cfg['seed'])


def get_eval_dataloader():
    eval_dataset = MnistQueryDataset("test")
    return DataLoader(eval_dataset, batch_size=1, shuffle=False)


def main(cfg):
    cfg = ConfigDict(cfg)
    eval_dataloader = get_eval_dataloader()
    program = load_program(cfg, eval_dataloader)
    trainer = create_trainer(program, get_train_dataloader, cfg)
    trainer.train(cfg)
    trainer.eval(eval_dataloader)
    return program


if __name__ == "__main__":
    main(mnist_config)
