from time import time
import os
import shutil
from pathlib import Path
from typing import Iterable, Callable

import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist

from ..data.dataloader import DataLoader
from ..data.query import Query
from ..logic.spl_module import SoftProofModule
from .logger import PrintLogger, WandbLogger
from .loss import nll_loss, get_optimizer
from .metrics import get_metrics, aggregate_metrics
from . import set_seed, ConfigDict


def ddp_setup(rank, world_size):
    print(f"Starting worker {rank + 1}/{world_size}")
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    set_seed(1532 + rank)

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def _trainp(rank, world_size, trainer, cfg):
    ddp_setup(rank, world_size)
    trainer.program.store = DDP(trainer.program.store, find_unused_parameters=True)
    trainer._train(cfg, master=rank == 0)
    dist.destroy_process_group()


class Trainer:
    def __init__(
            self,
            program: SoftProofModule,
            load_train_dataset: Callable[[dict], DataLoader],
            criterion,
            optimizer: Optimizer,
            logger=PrintLogger(),
            **search_args
    ):
        self.program = program
        self.program.mask_query = True
        self.logger = logger
        self.load_train_dataset = load_train_dataset
        self.train_dataset = None
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = None
        self.grad_clip = None
        self.search_args = search_args

    def _train(self, cfg: dict, master=True, do_eval=True):
        nb_epochs = cfg['nb_epochs']
        self.grad_clip = cfg['grad_clip']
        self.program.store.to(cfg['device'])
        self.program.store.train()
        self.train_dataset = self.load_train_dataset(cfg)
        self.scheduler = CosineAnnealingLR(self.optimizer, nb_epochs + 1)
        for epoch in range(nb_epochs):
            last_lr = self.scheduler.get_last_lr()[0]
            print(f"### EPOCH {epoch} (lr={last_lr:.2g}) ###")
            self.train_epoch(verbose=cfg['verbose'] and master)
            self.scheduler.step()
            if master:
                self.save(cfg)
            if do_eval and master and hasattr(self, 'val_dataloader'):
                self.eval(self.val_dataloader, name='val')

    def train(self, cfg: dict, nb_workers: int = 1):
        if nb_workers == 1:
            return self._train(cfg, True)
        self.program.algebra = None
        self.train_dataset = None
        mp.spawn(_trainp,
                 args=(nb_workers, self, cfg),
                 nprocs=nb_workers,
                 join=True)

    def train_profile(self, *args, **kwargs):
        from pyinstrument import Profiler

        profiler = Profiler()
        profiler.start()
        self.train(*args, **kwargs)
        profiler.stop()
        profiler.open_in_browser()

    def train_epoch(self, verbose: bool):
        for queries in tqdm(self.train_dataset, leave=False, smoothing=0, disable=not verbose):
            current_time = time()
            loss, diff, proof_steps, nb_proofs = self.get_loss(queries)
            grad_norm = 0.
            if loss is not None:
                grad_norm = self.step_optimizer()
            if verbose:
                self.logger.log({
                    'grad_norm': grad_norm,
                    'loss': loss,
                    'diff': diff,
                    "step_time": time() - current_time,
                    "proof_steps": proof_steps,
                    "nb_proofs": nb_proofs,
                })
        if verbose:
            self.logger.print()

    def eval(self, dataloader: DataLoader, name='test'):
        self.program.store.eval()
        metrics = []
        for queries in tqdm(dataloader, leave=False, smoothing=0):
            results = zip(queries, self._eval_queries(queries))
            new_metrics = [get_metrics(query, result, dataloader.dataset) for query, result in results]
            metrics += new_metrics
        self.logger.log_eval(aggregate_metrics(metrics), name=name)

    def _query(self, queries: Iterable[Query]):
        for query in queries:
            result, proof_steps, nb_proofs = self.program(
                query.query, **self.search_args
            )
            if not isinstance(result, torch.Tensor):
                result = torch.tensor(result)
            yield result, proof_steps, nb_proofs

    def _query_result(self, queries: Iterable[Query]):
        for query in queries:
            yield self.program.query(query.query, **self.search_args)

    def _eval_queries(self, queries: Iterable[Query]):
        with torch.no_grad():
            for query, results in zip(queries, self._query_result(queries)):
                if len(results) > 0:
                    results = {k: v.exp().item() for k, v in results.items() if v != 0.}
                    yield results
                else:
                    print(f"WARNING: empty result for {query}")
                    yield {}

    def get_loss(self, queries: Iterable[Query]) -> tuple[float, float, float, float]:
        results, proof_steps, nb_proofs = tuple(zip(*self._query(queries)))
        losses = [self.criterion(result, query.p) for result, query in zip(results, queries)]
        loss = torch.stack(losses).mean()
        errors = [query.error_with(result) for result, query in zip(results, queries)]
        if loss.requires_grad:
            loss.backward()
        proof_steps, nb_proofs = float(np.mean(proof_steps)), float(np.mean(nb_proofs))
        return float(loss), float(np.mean(errors)), proof_steps, nb_proofs

    def step_optimizer(self):
        with torch.no_grad():
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.program.parameters(), max_norm=self.grad_clip)
            grad_norm = self.program.grad_norm()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.get_store().clear_cache()
        return float(grad_norm)

    def save(self, config: ConfigDict):
        save_folder = f"results/{config['name']}"
        save_folder = Path(save_folder)
        if save_folder.exists():
            shutil.rmtree(save_folder, ignore_errors=True)
        save_folder.mkdir(parents=True)

        config.save(save_folder / "config.yaml")
        torch.save(self.get_store().state_dict(), save_folder / "store.pt")

    def get_store(self):
        return self.program.get_store()


def create_trainer(program, load_train_dataset, cfg):
    trainer_args = {
        "program": program,
        "criterion": nll_loss,
        "load_train_dataset": load_train_dataset,
        "optimizer": get_optimizer(program.get_store(), cfg),
        "logger": WandbLogger(cfg),
        "max_proofs": cfg.get("max_proofs", None),
        "max_depth": cfg.get("max_depth", None),
        "max_branching": cfg.get("max_branching", None),
    }
    return Trainer(**trainer_args)
