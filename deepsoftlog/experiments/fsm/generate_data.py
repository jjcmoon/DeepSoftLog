import copy
import random

from deepsoftlog.algebraic_prover.terms.expression import Expr

from deepsoftlog.algebraic_prover.terms.list_term import to_prolog_list
from deepsoftlog.data import Query, to_prolog_image
from deepsoftlog.data.dataloader import DataLoader
from deepsoftlog.data.dataset import StaticDataset
from deepsoftlog.experiments.mnist_addition.dataset import MNIST_DATA

DIGIT_IMAGES = None

def prepare_digits():
    print("preparing digits...")
    digits = dict()
    for split, data in copy.deepcopy(MNIST_DATA).items():
        digits[split] = {i: [] for i in range(10)}
        for image, label in data:
            digits[split][label].append(image)
    return digits


def get_digit(label: int, train=True):
    global DIGIT_IMAGES
    if DIGIT_IMAGES is None:
        DIGIT_IMAGES = prepare_digits()

    """Returns a unique image with the given label."""
    split = 'training' if train else 'test'
    if train:
        img = DIGIT_IMAGES[split][int(label)].pop()
    else:
        img = random.choice(DIGIT_IMAGES[split][int(label)])
    return to_prolog_image(img)


def mine_negatives(positives: set, nb_negatives: int, lengths):
    max_tries = nb_negatives * 100
    negatives = set()
    for _ in range(max_tries):
        # generate a random string with a random length
        length = random.choice(lengths)
        t = "".join(str(random.randint(0, 1)) for _ in range(length))
        if t not in positives:
            negatives.add(t)
        if len(negatives) >= nb_negatives:
            break

    return negatives


def strs_to_prolog_image_lists(data, train: bool):
    digit_lambda = lambda d: get_digit(d, train=train)
    # digit_lambda = lambda d: SoftTerm(Constant(d))
    return [to_prolog_list(d, digit_lambda) for d in data]


def get_language(language: str, ns: list[int], train: bool, verbose: bool = True):
    language_func = {
        "(10)*": language_10star,
        "even": language_even,
        "one_one": language_one_one
    }[language]
    positives = language_func(max(ns))
    positives = {p for p in positives if len(p) in ns}
    negatives = mine_negatives(positives, 100 * len(positives), ns)

    if verbose:
        print(f"Generated {'train' if train else 'test'} language with lengths {ns}")
        print("  positives:", positives)
        print("  negatives:", negatives)

    negatives = list(negatives) * (1 if train else 3)
    positives = list(positives) * (len(negatives) // len(positives))
    positives = strs_to_prolog_image_lists(positives, train)
    negatives = strs_to_prolog_image_lists(negatives, train)
    print(f"  #pos: {len(positives)}, #neg: {len(negatives)}")
    return positives, negatives


def language_10star(n):
    """ Language for (10)* """
    digits = 2
    data = [""]
    for _ in range(n):
        s = data[-1]
        for pi in range(0, digits):
            s = str(pi) + s
        data.append(s)
    return data


def language_even(n):
    """ Language (0|10*10*)* (which has an even number of 1s) """
    even_strings = {1: {"0"}, 2: {"00", "11"}}
    odd_strings = {1: {"1"}, 2: {"01", "10"}}

    for i in range(3, n+1):
        even_strings[i] = {s + "0" for s in even_strings[i-1]} | {s + "1" for s in odd_strings[i-1]}
        odd_strings[i] = {s + "0" for s in odd_strings[i-1]} | {s + "1" for s in even_strings[i-1]}
    return set.union(*even_strings.values())


def language_one_one(n):
    """ Language for 0*10* """
    data = {"0"*i + "1" + "0"*j for i in range(n) for j in range(n)}
    return data


def language_0n1n(n):
    return ["0"*i + "1"*i for i in range(n)]


def language_paren(n: int):
    def dfs(left, right, s):
        if len(s) == n * 2:
            res.append(s)
            return
        if left < n:
            dfs(left + 1, right, s + '0')
        if right < left:
            dfs(left, right + 1, s + '1')

    res = []
    dfs(0, 0, '')
    return [[str(x) for x in r] for r in res]


def get_queries(ns, train):
    positives, negatives = get_language("(10)*", ns, train)
    positive_queries = [Query(Expr("accepts", t), output_ind=()) for t in positives]
    negative_queries = [Query(Expr("accepts", t), output_ind=(), p=0) for t in negatives]
    return positive_queries + negative_queries


class FsmDataset(StaticDataset):
    def __init__(self, ns, train=True):
        data = get_queries(ns, train)
        super().__init__(data)


def get_train_dataloader(cfg, max_length=4):
    dataset = FsmDataset(list(range(1, max_length+1)), train=True)
    return DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True)


def get_eval_dataloader(cfg, max_length=4):
    dataset = FsmDataset(list(range(max_length+4, max_length+5)), train=False)
    return DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=False)


if __name__ == "__main__":
    ds = get_train_dataloader(dict(batch_size=1))
    for x in ds:
        print(x)

