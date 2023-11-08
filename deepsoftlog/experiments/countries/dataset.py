from pathlib import Path

from deepsoftlog.data import load_tsv_file, data_to_prolog
from deepsoftlog.algebraic_prover.terms.expression import Constant
from deepsoftlog.data.dataloader import DataLoader
from deepsoftlog.data.dataset import StaticDataset
from deepsoftlog.logic.soft_term import SoftTerm


def get_train_dataloader(cfg: dict):
    train_dataset = CountriesDataset("train").subset(cfg['data_subset'])
    train_dataset = train_dataset.mutate_all_output()
    return DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, seed=cfg['seed'])


def get_test_dataloader():
    regions = ["africa", "americas", "asia", "europe", "oceania"]
    domain = {-1: [SoftTerm(Constant(r)) for r in regions]}
    eval_dataset = CountriesDataset("test").mutate_all_output(domain)
    return DataLoader(eval_dataset, batch_size=1, shuffle=False)


def get_val_dataloader():
    regions = ["africa", "americas", "asia", "europe", "oceania"]
    domain = {-1: [SoftTerm(Constant(r)) for r in regions]}
    eval_dataset = CountriesDataset("val").mutate_all_output(domain)
    return DataLoader(eval_dataset, batch_size=1, shuffle=False)


class CountriesDataset(StaticDataset):
    def __init__(self, split_name: str = "val"):
        base_path = Path(__file__).parent / 'data' / 'raw'
        data = load_tsv_file(base_path / f"{split_name}.tsv")
        data = data_to_prolog(data, name="countries")
        super().__init__(tuple(data))


def generate_prolog_files():
    base_path = Path(__file__).parent / 'data'
    (base_path / 'tmp').mkdir(exist_ok=True)
    for problem in (f'S{i}' for i in range(4)):
        data = load_tsv_file(base_path / f"raw/countries_{problem}.tsv")
        data = data_to_prolog(data, name="countries")
        file_str = [f"{query.query}." for query in data]
        with open(base_path / f"tmp/countries_{problem}.pl", "w+") as f:
            f.write("\n".join(file_str))

        # add template stuff
        with open(base_path / f"templates/countries_{problem}_templates.pl", "r") as f:
            templates = f.read()
        with open(base_path / f"tmp/countries_{problem}.pl", "a+") as f:
            f.write("\n" + templates)


if __name__ == "__main__":
    d = CountriesDataset()
    print(d)
    generate_prolog_files()
