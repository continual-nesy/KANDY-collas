"""This is just a bare runner, with a list of commands to execute"""
import os
import sys

import argparse
from utils import ArgNumber

assert __name__ == "__main__", "Invalid usage! Run this script from command line, do not import it!"

arg_parser = argparse.ArgumentParser()

constant_params = {
    "weight_decay": 0.,
    "print_every": 32,
    "augment": "false",
    "seed": 9101,
    "cem_emb_size": 12,
    "concept_lambda": 0.,
    #"use_mask": "no",
    "concept_polarization_lambda": 0.,
    "mask_polarization_lambda": 0.,
    "min_pos_concepts": 0,
    "n_concepts": 30,
    "output_folder": "exp",
    "balance": True,
    "store_fuzzy": False,
    "cls_lambda": 1.0,
    "device": "cuda:0",
    "wandb_project": "kandy-cem-experiments",
    "compute_training_metrics": False,
    "correlate_each_task": True,
    "share_embeddings": True,
    "lr": -0.001,
    "batch": 32,
    "task_epochs": 10,
    "share_embeddings": True,
    "cem_emb_size": 12,
    "save_net": False,
    "save_options": False
}

const_str = " ".join(["--{} {}".format(k, v) for k, v in constant_params.items()])

models = [{"model": "cnn"}, {"model": "resnet50_head_only"}, {"model": "vit_head_only"}]

datasets = [{"data_path": "./data/cem_200x26_1.0-1.0/samples/sets", "use_global_concepts": "true"},
            {"data_path": "./data/easy_100x20_1.0-1.0/samples/sets", "use_global_concepts": "false"}]

triplet_params = [{"triplet_lambda": 0., "hamming_margin": 1}, {"triplet_lambda": 1., "hamming_margin": 1},
                  {"triplet_lambda": 1., "hamming_margin": 4}, {"triplet_lambda": 10., "hamming_margin": 1}]

replay_params = [{"replay_lambda": 1., "replay_buffer": 200},
                 {"replay_lambda": 10., "replay_buffer": 200}]

decorrelation_params = [{"decorrelate_concepts": False, "decorrelation_groups": 1},
                        {"decorrelate_concepts": True, "decorrelation_groups": 1},
                        {"decorrelate_concepts": True, "decorrelation_groups": 3}]

mask_params = [{"use_mask": "crisp"}, {"use_mask": "fuzzy"}]

commands = []
# Joint:
for m in models:
    for d in datasets:
        for t in triplet_params:
            for d2 in decorrelation_params:
                for m2 in mask_params:
                    cmd = "python main.py " + const_str + " --train joint "
                    var_params = ["--{} {}".format(k, v) for k, v in m.items()]
                    var_params += ["--{} {}".format(k, v) for k, v in d.items()]
                    var_params += ["--{} {}".format(k, v) for k, v in t.items()]
                    var_params += ["--{} {}".format(k, v) for k, v in d2.items()]
                    var_params += ["--{} {}".format(k, v) for k, v in m2.items()] # A questo punto fare fuzzy anche joint?

                    cmd += " ".join(var_params)
                    commands.append(cmd)

mask_params = [{"use_mask": "no"}, {"use_mask": "crisp"}]

# Continual task-incremental:
for m in models:
    for d in datasets:
        for t in triplet_params:
            for r in replay_params:
                for d2 in decorrelation_params:
                    for m2 in mask_params:
                        cmd = "python main.py " + const_str + " --train continual_task "
                        var_params = ["--{} {}".format(k, v) for k, v in m.items()]
                        var_params += ["--{} {}".format(k, v) for k, v in d.items()]
                        var_params += ["--{} {}".format(k, v) for k, v in t.items()]
                        var_params += ["--{} {}".format(k, v) for k, v in r.items()]
                        var_params += ["--{} {}".format(k, v) for k, v in d2.items()]
                        var_params += ["--{} {}".format(k, v) for k, v in m2.items()]

                        cmd += " ".join(var_params)
  #                      commands.append(cmd)

if len(sys.argv) == 1:
    print("Not enough arguments were provided.\nRun with -h to get the list of supported arguments.")
    print(list(range(len(commands)))) # For convenience, output the list of allowed values for experiment_id.

    sys.exit(0)


arg_parser.add_argument("--experiment_id",
                        help="ID of the experiment to run.",
                        type=ArgNumber(int, min_val=0, max_val=len(commands)), default=1)

opts = vars(arg_parser.parse_args())

cmd = commands[opts['experiment_id']] + " " + const_str

print("Executing the following command:")
print("\t" + cmd)
os.system(cmd)
