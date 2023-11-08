from torch.optim import AdamW

from deepsoftlog.algebraic_prover.algebras import safe_log_negate


def get_optimizer(store, config: dict):
    optimizer_name = config['optimizer']
    assert optimizer_name == "AdamW", f"Unknown optimiser `{optimizer_name}`"
    constant_group = {
        'params': store.constant_embeddings.parameters(),
        'lr': config['embedding_learning_rate'],
    }
    functor_group = {
        'params': store.functor_embeddings.parameters(),
        'lr': config['functor_learning_rate'],
        'weight_decay': config.get('functor_weight_decay', 0.)
    }

    optimizer = AdamW([constant_group, functor_group])
    return optimizer


def nll_loss(log_pred, target, gamma=0.):
    """ Negative log-likelihood loss """
    assert target in [0., 1.]
    if target == 0.:
        log_pred = safe_log_negate(log_pred)
    if gamma > 0.:  # focal loss
        log_pred = (1 - log_pred.exp()) ** gamma * log_pred

    return -log_pred
