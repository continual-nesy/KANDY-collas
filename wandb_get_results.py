import wandb
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import json
import os
import shutil
import os.path as path
from utils import save_dict, load_dict


# Configuration ********************************************************************************************

# easy_100x20_1.0-1.0
sweep_ids = ["melac64/easy_100x20_1.0-1.0/occjodl2",  # (many configs)
             "melac64/easy_100x20_1.0-1.0/vpdvdfas"]  # (... task_epochs 1; balance False; cont. online w/o buffer)

# hard_1000x18_1.0-1.0
'''
sweep_ids = ["melac64/hard_1000x18_1.0-1.0/6ckyp0eu",  # (seed 1234; batch 16; task_epochs 10; many configs)
             "melac64/hard_1000x18_1.0-1.0/srlbl03q",  # (... task_epochs 1; cont. online with buffer)
             "melac64/hard_1000x18_1.0-1.0/aa4dedaj",  # (... task_epochs 1; balance False; cont. online w/o buffer)
             "melac64/hard_1000x18_1.0-1.0/xw3frlbm",  # (seed 5678/9101; batch 16; task_epochs 10; many configs)
             "melac64/hard_1000x18_1.0-1.0/3wpv984d",  # (... task_epochs 1; cont. online with buffer)
             "melac64/hard_1000x18_1.0-1.0/a7z1a2lt",  # (... task_epochs 1; balance False; cont. online w/o buffer)
             "melac64/hard_1000x18_1.0-1.0/ras3lqfq",  # (batch 1; task_epochs 10; many configs)
             "melac64/hard_1000x18_1.0-1.0/e1j131qi",  # (batch 1; task_epochs 1; many configs)
             "melac64/hard_1000x18_1.0-1.0/zp3spv11"]  # (..., task_epochs 1; balance False; cont. online w/o buffer)
'''
# hard_100x18-1.0-1.0
'''
sweep_ids = ["melac64/hard_100x18-1.0-1.0/jau9njmd",  # (many configs)
             "melac64/hard_100x18-1.0-1.0/c6vegccf"]  # (... task_epochs 1; balance False; cont. online w/o buffer)
'''
load_final_results_from_local_json = True
model2id = {'mlp': 0, 'cnn': 1, 'resnet50': 2, 'resnet50_head_only': 3, 'vit_head_only': 4}
train2id = {'joint': 0, 'independent': 1, 'continual_task': 2, 'continual_online': 3}
metrics_of_interest = \
    ['avg_accuracy-train', 'avg_accuracy-val', 'avg_accuracy-test',
     'avg_forgetting-train', 'avg_forgetting-val', 'avg_forgetting-test',
     'backward_transfer-train', 'backward_transfer-val', 'backward_transfer-test',
     'forward_transfer-train', 'forward_transfer-val', 'forward_transfer-test',
     'acc_matrix-train', 'acc_matrix-val', 'acc_matrix-test']
metrics_of_interest_with_temporal_evolution = ['avg_accuracy-test', 'avg_forgetting-test',
                                               'backward_transfer-test', 'forward_transfer-test']
remap_model_names = {'mlp': 'MLP', 'cnn': 'CNN', 'resnet50': 'ResNet-50',
                     'resnet50_head_only': 'ResNet-50 (H)', 'vit_head_only': 'ViT (H)'}
remap_train_names = {'joint': 'Joint', 'independent': 'Independent',
                     'continual_task': 'Task Incr.', 'continual_online': 'Cont. Online'}
remap_param_names = {'lr': "$\\mathtt{learning\\_rate}$", 'optim': '$\\mathtt{optim}$',
                     'batch': '$\\mathtt{batch\\_size}$',
                     'task_epochs': '$\\mathtt{task\\_epochs}$',
                     'replay_buffer': '$\\mathtt{replay\\_buffer\\_size}$'}
# **********************************************************************************************************


def allocate_table() -> list[list[list]]:
    """Allocate an empy table, where each cell is a an empty list."""

    _table = []
    for _m in range(0, len(model2id)):
        _table.append([])
        for _t in range(0, len(train2id)):
            _table[_m].append([])
    return _table


def print_table(_table_mean: list[list[list[float]]],
                _table_std: list[list[list[float]]],
                _table_seed_counts: list[list[list[int]]],
                _name: str):
    """Print the result table of a certain metric (given its name too)."""

    id2model = {}
    for k, v in model2id.items():
        id2model[v] = k
    id2train = {}
    for k, v in train2id.items():
        id2train[v] = k

    s = ''
    s += _name.ljust(23) + ' '
    for _t in range(0, len(train2id)):
        s += id2train[_t].ljust(16) + ' '
    s += '\n'
    for _m in range(0, len(model2id)):
        s += id2model[_m].ljust(23) + ' '
        for _t in range(0, len(train2id)):
            if _table_seed_counts[_m][_t][0] > 0:
                s += ("{0:.2f} ({1:.2f}) [{2:d}]".format(_table_mean[_m][_t][0], _table_std[_m][_t][0],
                      _table_seed_counts[_m][_t][0])).ljust(16) + ' '
            else:
                s += "n.a. (n.a.) [0]".ljust(16) + ' '
        s += '\n'
    print(s)


def print_latex(_tables):
    """Print result tables in latex format."""

    id2model = {}
    for k, v in model2id.items():
        id2model[v] = remap_model_names[k]
    id2train = {}
    for k, v in train2id.items():
        id2train[v] = remap_train_names[k]
    id2param = {}
    _i = 0
    for k, v in remap_param_names.items():
        id2param[_i] = k
        _i += 1

    _means = _tables['avg_accuracy-test']['mean']
    _stds = _tables['avg_accuracy-test']['std']
    _runs = _tables['avg_accuracy-test']['seed_count']

    s = ''
    tab = '    '
    s += '\\begin{table}[t]\n'
    s += tab + '\\centering\n'
    s += tab + '\\caption{TODO (average accuracy on test data).}\n'
    s += tab + '\\label{tab:acc_TODO}\n'
    s += tab + '\\begin{tabular}{l|c|c|c|c}\n'
    s += tab + tab + '\\toprule\n'
    s += tab + tab + ''
    for _t in range(0, len(train2id)):
        s += "& \\footnotesize \\textsc{" + id2train[_t] + "} "
    s += '\\\\'
    s += '\n'
    s += tab + tab + "\\midrule\n"
    _max = [0.] * len(train2id)
    for _t in range(0, len(train2id)):
        for _m in range(0, len(model2id)):
            _max[_t] = max(_max[_t], _means[_m][_t][0])
    for _m in range(0, len(model2id)):
        s += tab + tab + "\\footnotesize \\textsc{" + id2model[_m] + "} "
        for _t in range(0, len(train2id)):
            s += "& "
            if _runs[_m][_t][0] > 0:
                if abs(math.floor(_max[_t] * 100.) - math.floor(_means[_m][_t][0] * 100.)) > 0:
                    s += "${0:.2f}$ {{\\tiny ($\\pm {1:.2f}$)}}".format(_means[_m][_t][0], _stds[_m][_t][0]) + ' '
                else:
                    s += "$\\textbf{{{0:.2f}}}$ {{\\tiny ($\\pm {1:.2f}$)}}".format(_means[_m][_t][0], _stds[_m][_t][0]) + ' '
            else:
                s += "n.a. (n.a.)" + ' '
            if _t == len(train2id) - 1:
                s += "\\\\"
        s += '\n'
    s += tab + tab + "\\bottomrule\n"
    s += tab + "\\end{tabular}\n"
    s += "\\end{table}"
    print(s)

    print('')

    _means1 = _tables['avg_forgetting-test']['mean']
    _stds1 = _tables['avg_forgetting-test']['std']
    _means2 = _tables['backward_transfer-test']['mean']
    _stds2 = _tables['backward_transfer-test']['std']
    _means3 = _tables['forward_transfer-test']['mean']
    _stds3 = _tables['forward_transfer-test']['std']

    s = ''
    tab = '    '
    s += '\\begin{table}[t]\n'
    s += tab + '\\centering\n'
    s += tab + '\\caption{TODO (forgetting, backward, forward transfer on test data, in the continual settings).}\n'
    s += tab + '\\label{tab:cont_TODO}\n'
    s += tab + '\\begin{tabular}{l|ccc|ccc}\n'
    s += tab + tab + '\\toprule\n'
    s += tab + tab + ''
    for _t in range(0, len(train2id)):
        if _t < 2:
            continue
        s += "& \\multicolumn{3}{|c}{\\footnotesize \\textsc{" + id2train[_t] + "}} "
    s += '\\\\'
    s += '\n'
    s += tab + tab
    for _t in range(0, len(train2id)):
        if _t < 2:
            continue
        s += "& \\tiny Forgetting $\\downarrow$ & \\tiny Backward $\\uparrow$ & \\tiny Forward $\\uparrow$ "
    s += '\\\\'
    s += '\n'
    s += tab + tab + "\\midrule\n"
    for _m in range(0, len(model2id)):
        s += tab + tab + "\\footnotesize \\textsc{" + id2model[_m] + "} "
        for _t in range(0, len(train2id)):
            if _t < 2:
                continue
            if _runs[_m][_t][0] > 0:
                s += "& ${0:.2f}$ {{\\tiny ($\\pm {1:.2f}$)}}".format(_means1[_m][_t][0], _stds1[_m][_t][0]) + ' '
                s += "& ${0:.2f}$ {{\\tiny ($\\pm {1:.2f}$)}}".format(_means2[_m][_t][0], _stds2[_m][_t][0]) + ' '
                s += "& ${0:.2f}$ {{\\tiny ($\\pm {1:.2f}$)}}".format(_means3[_m][_t][0], _stds3[_m][_t][0]) + ' '
            else:
                s += "& n.a. (n.a.)" + ' '
                s += "& n.a. (n.a.)" + ' '
                s += "& n.a. (n.a.)" + ' '
            if _t == len(train2id) - 1:
                s += "\\\\"
        s += '\n'
    s += tab + tab + "\\bottomrule\n"
    s += tab + "\\end{tabular}\n"
    s += "\\end{table}"
    print(s)

    print('')

    # hack - optimizer/learning-rate
    _configs = _tables['config']
    for _t in range(0, len(train2id)):
        for _m in range(0, len(model2id)):
            _configs[_m][_t][1]['optim'] = 'SGD' if _configs[_m][_t][1]['lr'] >= 0 else 'Adam'
            if _configs[_m][_t][1]['lr'] < 0:
                _configs[_m][_t][1]['lr'] = -_configs[_m][_t][1]['lr']

    s = ''
    tab = '    '
    s += '\\begin{table}[t]\n'
    s += tab + '\\centering\n'
    s += tab + '\\caption{TODO (best values for the cross-validated hyper-parameters).}\n'
    s += tab + '\\label{tab:best_hyper_TODO}\n'
    s += tab + '\\begin{tabular}{l|'
    for _p in remap_param_names:
        s += "c"
    s += "}\n"
    s += tab + tab + '\\toprule\n'
    for _t in range(0, len(train2id)):
        s += tab + tab + ''
        s += "& \\multicolumn{" + str(len(remap_param_names)) + \
             "}{|c}{\\footnotesize \\textsc{" + id2train[_t] + "}} "
        s += '\\\\'
        s += '\n'
        s += tab + tab
        for _p in range(0, len(id2param)):
            s += '& \\footnotesize ' + remap_param_names[id2param[_p]] + ' '
        s += '\\\\'
        s += '\n'
        s += tab + tab + "\\midrule\n"

        for _m in range(0, len(model2id)):
            s += tab + tab + "\\footnotesize \\textsc{" + id2model[_m] + "} "
            if len(_configs[_m][_t]) > 0:
                for _p in range(0, len(id2param)):
                    if id2param[_p] in _configs[_m][_t][1]:
                        s += "& " + str(_configs[_m][_t][1][id2param[_p]]) + " "
            else:
                s += "n.a." + ' '
            s += "\\\\"
            s += '\n'
            if _m == len(model2id) - 1:
                if _t < len(id2train) - 1:
                    s += tab + tab + "\\midrule\n"

    s += tab + tab + "\\bottomrule\n"
    s += tab + "\\end{tabular}\n"
    s += "\\end{table}"
    print(s)


def print_string_table(_table: list[list[list[str]]], name: str):
    """Print the table with strings of configurations."""

    _max = 0
    id2model = {}
    for k, v in model2id.items():
        id2model[v] = k
    id2train = {}
    for k, v in train2id.items():
        id2train[v] = k
        _max = max(_max, len(k))
    for _s in range(0, len(_table)):
        for _t in range(0, len(_table[_s])):
            _max = max(_max, len(_table[_s][_t][0]))

    s = ''
    s += name.ljust(23) + ' '
    for _t in range(0, len(train2id)):
        s += id2train[_t].ljust(_max) + ' '
    s += '\n'
    for _m in range(0, len(model2id)):
        s += id2model[_m].ljust(23) + ' '
        for _t in range(0, len(train2id)):
            s += _table[_m][_t][0].ljust(_max) + ' '
        s += '\n'
    print(s)


def plot_scores_over_tasks(_table_mean: list[list[list[float]]],
                           _table_std: list[list[list[float]]],
                           _name: str,
                           _sub_folder: str | None = None):

    id2model = {}
    for k, v in model2id.items():
        id2model[v] = k
    id2train = {}
    for k, v in train2id.items():
        id2train[v] = k

    # colors = ['#FF7F50', '#1E90FF', '#98FB98', '#EE82EE', '#FFD700', '#D2B48C']
    colors = ['#FF7F50', '#1E90FF', '#00C957', '#C79FEF', '#FFC125', '#D2B48C']
    bar_enlarging_factor = 2  # integer!
    bar_width = bar_enlarging_factor * (1.0 / float(len(model2id) + 2))
    plt.rcParams.update({'font.size': 9})

    _num_tasks = len(_table_mean[0][0])
    for _t in range(0, len(train2id)):
        plt.figure()
        r = None

        for _m in range(0, len(model2id)):
            _model = remap_model_names[id2model[_m]]
            if r is None:
                r = bar_enlarging_factor * np.arange(_num_tasks)
            else:
                r = [x + bar_width for x in r]
            _mean = _table_mean[_m][_t] if len(_table_mean[_m][_t]) > 1 else ([0.] * _num_tasks)
            _std = _table_std[_m][_t] if len(_table_std[_m][_t]) > 1 else ([0.] * _num_tasks)
            plt.bar(r, _mean, color=colors[_m], width=0.7*bar_width, edgecolor=colors[_m], label=_model)
            plt.errorbar(r, _mean, yerr=_std, fmt='none', color="k", elinewidth=0.3)

        plt.ylabel('Accuracy' if 'acc_matrix' in _name else _name)
        plt.xlabel('Tasks')
        plt.xticks([r + bar_width * (((len(model2id) + 2) // 2) - 1)
                    for r in range(bar_enlarging_factor * _num_tasks)[0:-1:bar_enlarging_factor]],
                   ['T' + str(x) for x in list(range(0, _num_tasks))])
        plt.xlim(0-0.7*bar_width, r[-1]+0.7*bar_width)
        plt.ylim(0.4, 1.0)
        plt.legend(bbox_to_anchor=(0.1, 1.02, 0.8, 0.2), loc="lower left", mode='expand', ncols=len(model2id),
                   fontsize="6")

        # aspect ratio of the plot
        ax = plt.gca()
        x_left, x_right = ax.get_xlim()
        y_low, y_high = ax.get_ylim()
        ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * (1. / 5.0))
        plt.tight_layout()

        _path = 'figs'
        if not path.exists(_path):
            os.makedirs(_path)
        if _sub_folder is not None:
            _path = path.join(_path, _sub_folder)
            if not path.exists(_path):
                os.makedirs(_path)
        plt.savefig(path.join(_path, id2train[_t] + "-" + _name + ".pdf"), format="pdf", bbox_inches="tight")


def plot_scores_over_time(_table_mean: list[list[list[list[float]]]],
                          _table_std: list[list[list[list[float]]]],
                          _name: str,
                          _sub_folder: str | None = None,
                          _train_to_skip: list[str] | None = None):

    id2model = {}
    for k, v in model2id.items():
        id2model[v] = k
    id2train = {}
    for k, v in train2id.items():
        id2train[v] = k

    # colors = ['#FF7F50', '#1E90FF', '#98FB98', '#EE82EE', '#FFD700', '#D2B48C']
    colors = ['#FF7F50', '#1E90FF', '#00C957', '#C79FEF', '#FFC125', '#D2B48C']
    line_styles = ['-', '--', '-.', '-', ':', '--']
    markers = ['o', 'v', '^', '*', 'X', 'd']
    plt.rcParams.update({'font.size': 14})

    _num_tasks = len(_table_mean[0][0][0])
    _max_mean = -100.
    _min_mean = +100.
    for _t in range(0, len(train2id)):
        if _train_to_skip is not None and id2train[_t] in _train_to_skip:
            continue
        for _m in range(0, len(model2id)):
            _mean = _table_mean[_m][_t][0] if len(_table_mean[_m][_t][0]) > 1 else ([0.] * _num_tasks)
            _max_mean = max(_max_mean, max(_mean))
            _min_mean = min(_min_mean, min(_mean))

    for _t in range(0, len(train2id)):
        if _train_to_skip is not None and id2train[_t] in _train_to_skip:
            continue
        plt.figure()
        r = np.arange(_num_tasks)

        for _m in range(0, len(model2id)):
            _model = remap_model_names[id2model[_m]]
            _mean = _table_mean[_m][_t][0] if len(_table_mean[_m][_t][0]) > 1 else ([0.] * _num_tasks)
            _std = _table_std[_m][_t][0] if len(_table_std[_m][_t][0]) > 1 else ([0.] * _num_tasks)
            plt.plot(r, _mean, color=colors[_m], label=_model, marker=markers[_m],
                     linestyle=line_styles[_m], linewidth=2, markersize=6 if markers[_m] != '*' else 8)

        _y_label = 'Avg Accuracy' if 'accuracy' in _name else 'Avg Forgetting' if 'forgetting' in _name else \
            'Forward Transfer' if 'forward' in _name else 'Backward Transfer' if 'backward' in _name else _name
        plt.ylabel(_y_label)
        plt.xlabel('Time (@Task)')
        plt.xticks(r, [str(x) for x in list(range(0, _num_tasks))])
        _range = _max_mean - _min_mean
        plt.ylim(_min_mean - 0.08*_range, _max_mean + 0.08*_range)
        plt.legend(fontsize="12")

        # aspect ratio of the plot
        ax = plt.gca()
        x_left, x_right = ax.get_xlim()
        y_low, y_high = ax.get_ylim()
        ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * (1. / 1.25))
        plt.tight_layout()

        _path = 'figs'
        if not path.exists(_path):
            os.makedirs(_path)
        if _sub_folder is not None:
            _path = path.join(_path, _sub_folder)
            if not path.exists(_path):
                os.makedirs(_path)
        plt.savefig(path.join(_path, id2train[_t] + "-" + _name + ".pdf"), format="pdf", bbox_inches="tight")


def aggregate_by_seed(_metric: str, _reference_run: wandb.run, _list_of_runs: list[wandb.run]) \
        -> tuple[list[float], list[float], list[int]]:
    """Aggregate results of multiple runs w.r.t. different seeds."""

    _results = []
    _found_seeds = {}
    for _my_run in _list_of_runs:
        matched = True
        for k, v in _reference_run.config.items():
            if k != 'seed' and k != 'exp_name' and k != 'command_line' and k != 'output_folder' and \
                    k != 'device' and 'save_' not in k and k != 'data_path' and 'wandb_' not in k:
                if _my_run.config[k] != v:
                    matched = False
                    break

        if matched:
            assert _my_run.config['seed'] not in _found_seeds, "The same seed was found in multiple equivalent" \
                                                               " configurations!"
            _found_seeds[_my_run.config['seed']] = True

            # some runs could be badly logged due to network issues
            if len(_my_run.summary.keys()) == 0:
                _history = _my_run.scan_history()
                _step = -1
                _val = None
                print("Deeply scanning a badly logged run, due to missing summary data: " + str(_my_run))
                for _log in _history:
                    if _metric in log and log[_metric] is not None:
                        print("metric found")
                        if log['_step'] > _step:
                            _step = log['_step']
                            _val = log[_metric]
                if _val is not None:
                    _results.append(_val)
                else:
                    print("Skipping such a run, cannot get any valid results logged for the requested metric.")
                    continue  # skipping a badly logged run
            else:
                if 'matrix' not in _metric and '-time' not in _metric:

                    # most common case, scalar metric
                    _results.append(float(_my_run.summary[_metric]))
                elif '-time' not in _metric:

                    # downloading all artifacts and moving them in the cache folder 'matrices/sweep_id'
                    matrix_path = path.join("matrices", sweep_id_short, _my_run.id + "." + metric + ".json")
                    if not path.exists(matrix_path):
                        for _l in _my_run.logged_artifacts():
                            artifact = _my_run.use_artifact(_l)
                            datadir = artifact.download()
                            for file in os.listdir(datadir):
                                if file.endswith(".json"):

                                    # metric name (.json) guesses from the name of the downloaded file
                                    _file = file.split('.')[0] + "." + file.split('.')[-1]

                                    # moving
                                    if not path.exists("matrices"):
                                        os.makedirs("matrices")
                                    if not path.exists(path.join("matrices", sweep_id_short)):
                                        os.makedirs(path.join("matrices", sweep_id_short))
                                    shutil.move(path.join(datadir, file),
                                                path.join("matrices", sweep_id_short, _my_run.id + "." + _file))
                                    break

                        # purging
                        shutil.rmtree("artifacts")

                    with open(matrix_path) as file:
                        _results.append(json.load(file)['data'])
                else:

                    # this is about data that was logged "over time" (after each task, in principle) to compose plots
                    _history = _my_run.scan_history()
                    _yy = [None] * num_tasks
                    _my_metric = _metric[0:(len(_metric)-len("-time"))]
                    for _log in _history:
                        if _my_metric in _log and _log[_my_metric] is not None:
                            _yy[_log['_step']] = _log[_my_metric]
                    _skip = False
                    for _y in _yy:
                        if _y is None:
                            _skip = True
                            print("Cannot find " + _my_metric + " for some of the time steps! (skipping)")
                            break  # skipping a badly logged run

                    if not _skip:
                        _results.append(_yy)

    if 'matrix' not in _metric and '-time' not in _metric:

        # scalar results
        _results = torch.tensor(_results)
        return [torch.mean(_results).item()], \
            [torch.std(_results).item() if _results.numel() > 1 else 0.], \
            [_results.numel()]
    elif '-time' not in _metric:

        # matrices of accuracies
        _per_task_scores = []
        for _res in _results:
            _per_task_scores.append(torch.tensor(_res[-1][1:]))
        _results = torch.stack(_per_task_scores, dim=1)
        return torch.mean(_results, dim=-1).numpy().tolist(), \
            torch.std(_results, dim=-1).numpy().tolist() if _results.shape[-1] > 1 else ([0.] * _results.shape[0]), \
            [_results.shape[-1]]
    else:

        # time-related plots
        _results = torch.tensor(_results, dtype=torch.float)
        return [torch.mean(_results, dim=0).numpy().tolist()], \
            [torch.std(_results, dim=0).numpy().tolist() if _results.shape[0] > 1 else [0.] * _results.shape[1]], \
            [_results.shape[0]]


def stringify_config(_run):
    s = ""
    z = 0
    for k, v in _run.config.items():
        if k != 'model' and k != 'seed' and k != 'exp_name' and k != 'command_line' and k != 'output_folder' and \
                k != 'print_every' and k != 'train' and k != 'device' and 'save_' not in k and k != 'data_path' and \
                'wandb_' not in k:
            kk = k.split("_")
            if z > 0:
                s += "_"
            for ikk in kk:
                s += ikk[0]
            vv = str(v)
            if vv == 'True' or vv == 'False':
                vv = vv[0]
            s += str(vv)
            z += 1
    return s


sweep_id_short = ""
for i, single_sweep_id in enumerate(sweep_ids):
    if i > 0:
        sweep_id_short += "-"
    sweep_id_short += single_sweep_id.split("/")[-1]

for metric in metrics_of_interest_with_temporal_evolution:
    metrics_of_interest.append(metric + "-time")

if load_final_results_from_local_json is False:
    api = wandb.Api()
    runs = []
    num_tasks = None

    for single_sweep_id in sweep_ids:
        sweep = api.sweep(single_sweep_id)  # retrieving sweep
        sweep_runs = sweep.runs  # listing all runs within the sweep
        print("Found " + str(len(sweep_runs)) + " total runs in sweep: " + str(single_sweep_id))

        # checking
        show_crashed = False
        for i, run in enumerate(sweep_runs):
            if show_crashed and '(crashed)' in str(run):
                print("Crashed run: " + str(run))
                print("Its configuration:")
                print(run.config)

            if '(finished)' in str(run):
                runs.append(run)

                if num_tasks is None:
                    history = run.scan_history()
                    for log in history:
                        if 'acc_matrix-val' in log and log['acc_matrix-val'] is not None:
                            num_tasks = log['acc_matrix-val']['nrows']

    print("Found " + str(len(runs)) + " finished/valid runs.")
    if num_tasks is None:
        print("Cannot guess the number of tasks, what is going on?")
    print("Guessed " + str(num_tasks) + " tasks.")

    # tables of results (rows: model; cols: train)
    tables = {}
    for metric in metrics_of_interest:
        tables[metric] = {'mean': allocate_table(), 'std': allocate_table(), 'seed_count': allocate_table()}
    tables['config'] = allocate_table()
    tables['run_ids'] = allocate_table()

    run_table = allocate_table()
    avg_accuracy_val_table = allocate_table()

    # for each valid run...
    filter_duplicated = {}
    for run in runs:
        if run.__str__() in filter_duplicated:
            print("Found duplicated run, W&B bug? " + str(run.__str__()))
            continue
        filter_duplicated[run.__str__()] = True

        # getting results from the last training task (last-logged-results are what is reported in the run summary)
        m = model2id[run.config['model']]
        t = train2id[run.config['train']]
        run_table[m][t].append(run)

    # finding the best run and storing it
    aggregate_by_seed_in_validation = True
    for m in range(0, len(model2id)):
        for t in range(0, len(train2id)):

            # collecting validation accuracies of the different configurations (aggregating by seed or not)
            for _run in run_table[m][t]:
                if aggregate_by_seed_in_validation:
                    acc_mean, _, _ = aggregate_by_seed('avg_accuracy-val', _run, run_table[m][t])
                    avg_accuracy_val_table[m][t].append(acc_mean)
                else:
                    avg_accuracy_val_table[m][t].append(_run.summary['avg_accuracy-val'])

            # finding the best (val) run
            if len(avg_accuracy_val_table[m][t]) > 0:
                best_acc, best_run_id = torch.max(torch.tensor(avg_accuracy_val_table[m][t]), dim=0)
                best_run = run_table[m][t][best_run_id]

                # aggregating results, for each metric, across different seeds
                for metric in metrics_of_interest:
                    tables[metric]['mean'][m][t], tables[metric]['std'][m][t], tables[metric]['seed_count'][m][t] = \
                        aggregate_by_seed(metric, best_run, run_table[m][t])

                # saving best run info (to see the values of the hyperparameters)
                tables['config'][m][t].append(stringify_config(best_run))
                tables['config'][m][t].append(dict(best_run.config))
                run_name = best_run.__str__().split(' ')[1]
                tables['run_ids'][m][t].append(run_name[run_name.rfind('/') + 1:])

                # fix: updating forward transfer to consider the accuracy (average) of a randomly trained model
                for metric in metrics_of_interest:
                    if metric.startswith('forward_transfer'):
                        for i in range(0, len(tables[metric]['mean'][m][t])):
                            if isinstance(tables[metric]['mean'][m][t][i], list):
                                for j in range(0, len(tables[metric]['mean'][m][t][i])):
                                    tables[metric]['mean'][m][t][i][j] = \
                                        max(0., tables[metric]['mean'][m][t][i][j] - 0.5)
                            else:
                                tables[metric]['mean'][m][t][i] = max(0., tables[metric]['mean'][m][t][i] - 0.5)
            else:
                for metric in metrics_of_interest:
                    tables[metric]['mean'][m][t] = [0.]
                    tables[metric]['std'][m][t] = [0.]
                    tables[metric]['seed_count'][m][t] = [0]
                tables['config'][m][t].append('n.a.')
                tables['run_ids'][m][t].append('n.a.')

    # saving results to local file
    if not path.exists("final_results"):
        os.makedirs("final_results")
    save_dict(path.join("final_results", sweep_id_short + ".json"), tables)
else:
    tables = load_dict(path.join("final_results", sweep_id_short + ".json"))

# table (model, train) with the aggregated results best run in each cell
print("")
for metric in metrics_of_interest:
    if 'matrix' not in metric and '-time' not in metric:
        print_table(tables[metric]['mean'], tables[metric]['std'], tables[metric]['seed_count'], metric)
    elif '-time' not in metric:
        try:
            plot_scores_over_tasks(tables[metric]['mean'], tables[metric]['std'], metric, sweep_id_short)
            pass
        except ValueError:
            print("Error while trying to plot data (scores over tasks).")
    else:
        try:
            plot_scores_over_time(tables[metric]['mean'], tables[metric]['std'], metric, sweep_id_short,
                                  _train_to_skip = ['joint', 'independent'])
        except ValueError:
            print("Error while trying to plot data (scores over time).")
print_string_table(tables['config'], 'config')
print_string_table(tables['run_ids'], 'run_ids')
print("From sweep(s): " + str(sweep_ids))
print("Short sweep(s) name: " + sweep_id_short)
print("Generating LaTeX tables...")
print_latex(tables)
