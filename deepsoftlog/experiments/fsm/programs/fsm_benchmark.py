import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import opt_einsum
import torch

from deepsoftlog.algebraic_prover.algebras.tensor_algebra import TensorPredicate, TensorNetwork


def plot_bench():
    problog_ddnnf = {1: 0, 2: 0.012569427490234375, 3: 0.016071724891662587, 4: 0.024320626258850092, 5: 0.06618607044219971, 6: 0.3188768863677979, 7: 2.6613048553466796}
    problog_sdd = {1: 8.51631164550809e-05, 2: 0.0013299465179443387, 3: 0.005064487457275391, 4: 0.025610399246215818, 5: 0.8044854164123536, 6: 100}
    problog_fsdd = {1: 0.0006206750869751088, 2: 0.002415776252746582, 3: 0.00870552062988282, 4: 0.014897298812866222, 5: 0.04273121356964113, 6: 0.20963270664215086, 7: 9.30973823070526}
    problog_bdd = {1: 0.0013319015502929604, 2: 0.0025802612304687417, 3: 0.023893547058105466, 4: 5.809967803955078}
    problog_ground = {1: 0.0009116888046264704, 2: 0.002194786071777352, 3: 0.0047747373580932645, 4: 0.007128262519836434, 5: 0.01081128120422363, 6: 0.016934680938720706, 7: 0.02241065502166749, 8: 0.029755425453186032, 9: 0.03924832344055175, 10: 0.05047943592071534}

    tensor_nets = {2: 0.0002443054889110809, 3: 0.00033546508626734956, 4: 0.0004399685149497174, 5: 0.0005491616878103702, 6: 0.0006615126386601874, 7: 0.0007578565719279837, 8: 0.0008542867417031146, 9: 0.001003767581696206, 10: 0.0011362344660657518, 11: 0.0012358173410943214, 12: 0.0013405363610450257, 13: 0.0014603569152507377, 14: 0.0015686496775201026, 15: 0.0016324393292690845, 16: 0.0017764669783571934, 17: 0.0019711103845149913, 18: 0.002072534662611941, 19: 0.002224427588442539, 20: 0.002370999214497018}

    plt.plot(*zip(*sorted(problog_ddnnf.items())), label="d-DNNF")
    plt.plot(*zip(*sorted(problog_sdd.items())), label="SDD")
    plt.plot(*zip(*sorted(problog_fsdd.items())), label="forward SDD")
    plt.plot(*zip(*sorted(problog_bdd.items())), label="BDD")
    plt.plot(*zip(*sorted(problog_ground.items())), label="problog grounding")
    plt.plot(*zip(*sorted(tensor_nets.items())), label="tensor contraction")
    plt.yscale("log")
    plt.ylabel("time (s)")
    plt.xlim(2, 10)
    plt.ylim(1e-4, 2)
    plt.xlabel("number of states")
    plt.title("Inference on random probabilistic finite state machines")
    plt.legend()
    plt.show()


def run_problog_bench(max_n=10, nb_repeats=10):
    results = {}
    for n in range(max_n + 1):
        timings = []
        for i in range(nb_repeats):
            write_program(n)
            t1 = time.time()
            os.system("problog ground dummy_fsm.pl >/dev/null")
            t2 = time.time()
            print(n, t2 - t1)
            timings.append(t2 - t1)
        results[n] = np.mean(timings)
    results = {k: max(v - results[0], 0) for k, v in results.items()}
    del results[0]
    print(results)


def write_program(n: int):
    string = ", ".join(random.choice(["0", "1"]) for i in range(n))
    program = f"""
start_state(0).
halt_state(0).

run(X, []) :- start_state(X).
run(State, [Symbol|OldString]) :-
    run(OldState, OldString),
    transition(OldState, State, Symbol).

accepts(X) :- halt_state(H), run(H, X).
query(accepts([{string}])).

"""
    for i in range(n):
        trans_probs = np.random.rand(n * 2)
        trans_probs /= trans_probs.sum()
        prob_facts = [f"{x}::transition({i}, {j//2}, {j%2})" for j, x in enumerate(trans_probs)]
        program += "; ".join(prob_facts) + ".\n"
    with open("dummy_fsm.pl", "w+") as f:
        f.write(program)


COUNTER = 0


def get_name():
    global COUNTER
    COUNTER += 1
    return opt_einsum.get_symbol(COUNTER)


def tpd(data, edges):
    return TensorPredicate(data, edges, {edge: {'disjoint': True} for edge in edges})


def tensor_impl(n: int):
    start_state = torch.tensor([0] * n)
    end_state = torch.tensor([0] * n)
    start_state[0] = 1
    end_state[0] = 1

    transition = np.random.rand(n, n, 2)  # shape: [START_STATE, END_STATE, SYMBOL]
    transition /= transition.sum(axis=(1, 2), keepdims=True)
    transition = torch.tensor(transition)

    name = get_name()
    tensor_network = [tpd(start_state, (name,))]
    for i in range(n):
        symbol_tensor = torch.zeros(2)
        symbol_tensor[random.choice([0, 1])] = 1
        symbol_name = get_name()
        tensor_network.append(tpd(symbol_tensor, (symbol_name,)))
        new_name = get_name()
        tensor_network.append(tpd(transition, (name, new_name, symbol_name)))
        name = new_name
    tensor_network.append(tpd(end_state, (name,)))

    tensor_network = TensorNetwork({t.axes_order: t for t in tensor_network})
    return tensor_network.contract()


def run_tensor_bench(max_n: int, nb_repeats: int):
    results = dict()
    for n in range(2, max_n + 1):
        timings = []
        for i in range(nb_repeats):
            t1 = time.time()
            tensor_impl(n)
            if i > 5: # warm-up
                timings.append(time.time() - t1)
        results[n] = np.mean(timings)

    print(results)


if __name__ == "__main__":
    # run_tensor_bench(20, 100)
    # write_program(0)
    plot_bench()
    # run_problog_bench()
