import numpy as np
from sklearn.metrics import average_precision_score

from ..data import Query


def get_metrics(query: Query, results, dataset) -> dict[str, float]:
    metrics = boolean_metrics(results, query)
    if not query.query.is_ground():
        assert query.p == 1
        metrics.update(rank_metrics(results, dataset))
    return metrics


def boolean_metrics(results, query) -> dict[str, float]:
    pred = max(results.values(), default=0)
    diff = abs(query.p - pred)
    return {
        "diff": diff,
        "target": query.p,
        "pred": pred,
        "threshold_accuracy": 1 if diff <= 0.5 else 0,
    }


def rank_metrics(results, dataset) -> dict[str, float]:
    results = sorted(results.items(), key=lambda x: -x[1])
    for i, (result, _) in enumerate(results):
        if result in dataset:
            rank = i + 1
            return {
                "mrr": 1 / rank,
                "hits@1": 1 if rank <= 1 else 0,
                "hits@3": 1 if rank <= 3 else 0,
                "hits@10": 1 if rank <= 10 else 0,
            }

    return {"mrr": 0, "hits@1": 0, "hits@3": 0, "hits@10": 0}


def aggregate_metrics(metrics_list: list) -> dict:
    result = {}
    metric_names = metrics_list[0].keys()
    for metric_name in metric_names:
        result[metric_name] = np.nanmean([x[metric_name] for x in metrics_list])

    targets = np.array([x['target'] for x in metrics_list])
    preds = np.array([x['pred'] for x in metrics_list])
    result['auc'] = average_precision_score(targets, preds)
    return result
