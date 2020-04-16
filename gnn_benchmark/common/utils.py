
import torch
from sklearn.model_selection import StratifiedKFold
from pandas import json_normalize
import pandas as pd
from typing import List
from gnn_benchmark.common.definitions import RunEntry
import itertools
import copy
import numpy as np
from functools import lru_cache
import scipy.stats as st


def num_graphs(data):
    return len(torch.unique(data.batch))


def num_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def k_fold(dataset, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(torch.from_numpy(idx))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero().view(-1))

    return train_indices, test_indices, val_indices


# def confidence_bound(conf=0.95):
#     def conf_bound(a):
#         lower, upper = st.t.interval(conf, len(a)-1, loc=np.mean(a), scale=st.sem(a))
#         return np.mean(a) - lower
#     return conf_bound

def confidence_interval_95(a):
    return 1.96 * st.sem(a)

def nested_update(dic: dict, keys: list, value) -> dict:
    cur = dic
    for key in keys[:-1]:
        try:
            # RunDefinition
            cur = getattr(cur, key)
        except AttributeError:
            # dict
            try:
                cur = cur[key]
            except KeyError:
                cur[key] = {}
                cur = cur[key]

    try:
        cur = setattr(cur, keys[-1], value)
    except AttributeError:
        cur[keys[-1]] = value
    return dic


def create_parameter_combinations(base_config, variable_parameters, max_runs=None, shuffle_seed=0):
    config_values = list(itertools.product(*variable_parameters.values()))
    config_values = [dict(zip(variable_parameters.keys(), l)) for l in config_values]
    if max_runs is not None:
        random_state = np.random.RandomState(shuffle_seed)
        random_state.shuffle(config_values)
        config_values = config_values[:max_runs]
    parameter_combinations = []
    for c in config_values:
        config = copy.deepcopy(base_config)
        for val_name, val in c.items():
            config = nested_update(config, val_name.split("."), val)
        parameter_combinations.append(config)
    return parameter_combinations


def create_fold_configs(parameter_combinations, n_folds):
    folded_configs = []
    for config in parameter_combinations:
        for f in range(n_folds):
            fold_config = copy.deepcopy(config)
            fold_config["fold_idx"] = f
            fold_config["folds"] = 10
            folded_configs.append(fold_config)
    return folded_configs


@lru_cache(maxsize=16)
def run_entries_to_df(run_entries: List[RunEntry], replace_none=None):
    run_dicts = []
    for r in run_entries:
        run_def = r.run_definition
        d = {
            "id": r.id,
            "run_definition": run_def,
            "results": {
                "train_metrics": r.results.train_metrics,
                "val_metrics": r.results.val_metrics,
                "test_metrics": r.results.test_metrics,
                "duration": r.results.duration,
                "trainable_parameters": r.results.trainable_parameters,
                "gpu_mem_usage": r.results.gpu_mem_usage
            }
        }
        run_dicts.append(d)
    df = pd.DataFrame(json_normalize(run_dicts))
    if replace_none is not None:
        df = df.fillna(value=replace_none)
    return df


def choose_gpus(overwrite_device=None):
    if overwrite_device is None:
        gpus = -1 if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 0
    elif overwrite_device is "cpu":
        gpus = 0
    elif overwrite_device is "cuda":
        gpus = -1
    else:
        raise AttributeError(f"{overwrite_device} is unknown device type")
    if gpus == 0:
        print("WARNING: No GPUs are used.")
        device = "cuda"
    else:
        device = "cpu"
    return device, gpus
