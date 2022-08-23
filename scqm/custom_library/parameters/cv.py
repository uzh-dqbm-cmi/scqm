import sys
import itertools
import random
import pickle

cvparams = {
    "size_out_scale": [15, 10, 5],
    "num_layers_enc": [2, 5],
    "hidden_enc": [100, 300],
    "num_layers": [1, 3],
    "num_layers_pred": [2, 4, 6],
    "hidden_pred": [100, 200],
    "dropout": [0.0, 0.1, 0.2],
    "size_history": [30, 50, 100],
    "lr": [1e-3],
}


def get_parameters(seed, num_combi):
    combinations = list(itertools.product(*cvparams.values()))
    random.seed(seed)
    combinations = random.sample(combinations, num_combi)
    return cvparams, combinations


if __name__ == "__main__":
    print("picking cv parameters")
    combinations = list(itertools.product(*cvparams.values()))
    if int(sys.argv[1]) > -1:
        combinations = random.sample(combinations, int(sys.argv[1]))

    with open("/cluster/work/medinfmk/scqm/tmp/params.pickle", "wb") as f:
        # with open("scqm/test_bed/dummy_data/params", "wb") as f:
        pickle.dump((cvparams, combinations), f)
    print("parameters created")
