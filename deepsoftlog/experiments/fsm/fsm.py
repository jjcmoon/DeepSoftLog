import wandb
from deepsoftlog.algebraic_prover.terms.expression import Constant

from deepsoftlog.data import to_prolog_image
from deepsoftlog.data.dataloader import DataLoader
from deepsoftlog.experiments.fsm.generate_data import get_train_dataloader, get_eval_dataloader
from deepsoftlog.experiments.mnist_addition.dataset import MnistQueryDataset
from deepsoftlog.experiments.mnist_addition.mnist import mnist_config
from deepsoftlog.training import load_program, load_config, ConfigDict
from deepsoftlog.training.trainer import create_trainer


def train(cfg):
    cfg = load_config(cfg)
    eval_dataloader = get_eval_dataloader(cfg)
    program = load_program(cfg, eval_dataloader)

    if cfg['pretrain']:
        program = pretrain_digits(program, cfg)

    trainer = create_trainer(program, get_train_dataloader, cfg)
    trainer.train(cfg)
    trainer.search_args['max_proofs'] = 1
    trainer.eval(eval_dataloader)

    # for debugging
    for c in ("0", "1", "ns1", "ns2", "halts", "start", "w1", "w2", "w3", "w4"):
        try:
            v = program.store(Constant(c)).abs().detach()
        except AttributeError:
            continue
        v /= v.sum()
        print(c, v.round(decimals=3))

    for label in (0, 1):
        vs = []
        for img in DIGIT_IMAGES["training"][label][:20]:
            # print(img, label)
            prolog_img = to_prolog_image(img).arguments[0]
            v = program.store(prolog_img).abs().detach()
            v /= v.sum()
            vs.append(v)
        v = sum(vs) / len(vs)
        print('img', label, v.round(decimals=3))


def get_pretrain_dataloader(cfg, data_subset=20):
    train_dataset = MnistQueryDataset("training", allowed_labels=[0, 1]).subset(data_subset)
    return DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)


def pretrain_digits(program, program_cfg):
    cfg = ConfigDict(mnist_config)
    cfg['embedding_dimensions'] = program_cfg['embedding_dimensions']
    cfg['name'] = "fsm pretraining"
    cfg['nb_epochs'] = 50
    mnist_program = load_program(cfg, get_pretrain_dataloader(cfg))
    trainer = create_trainer(mnist_program, get_pretrain_dataloader, cfg)
    trainer.train(cfg)
    wandb.finish()

    program.store.functor_embeddings.update(mnist_program.store.functor_embeddings)
    return program


if __name__ == "__main__":
    main("deepsoftlog/experiments/fsm/config.yaml")
