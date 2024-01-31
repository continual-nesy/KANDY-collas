"""This is just a bare runner, with a list of commands to execute"""
import os
import sys

import argparse
from utils import ArgNumber

assert __name__ == "__main__", "Invalid usage! Run this script from command line, do not import it!"
if len(sys.argv) == 1:
    print("Not enough arguments were provided.\nRun with -h to get the list of supported arguments.")
    sys.exit(0)

arg_parser = argparse.ArgumentParser()

constant_params = {
    "weight_decay": 0.,
    "print_every": 32,
    "augment": "false",
    "seed": 9101,
    "cem_emb_size": 12,
    #"concept_lambda": 0.,
    #"use_mask": "no",
    #"concept_polarization_lambda": 0.,
    "mask_polarization_lambda": 0.,
    #"min_pos_concepts": 0,
    "n_concepts": 30,
    "output_folder": "exp",
    "balance": "true",
    "store_fuzzy": "no",
    "cls_lambda": 1.0,
    "device": "cuda:0",
    "wandb_project": "kandy-cem2",
    "compute_training_metrics": "false",
    "correlate_each_task": "true",
    "share_embeddings": "true",
}

const_str = " ".join(["--{} {}".format(k, v) for k, v in constant_params.items()])



commands = [
    "python main.py --data_path ./data/cem_200x26_1.0-1.0/samples/sets --use_global_concepts True " # Dataset params.
    "--batch 32 --task_epochs 10 --train continual_task  --model resnet50_head_only --lr -0.001 " # Training params.
    "--hamming_margin 1 --triplet_lambda 1.0 --replay_buffer 200 --replay_lambda 1.0 " # Loss params.
    "--concept_lambda 1.0 --use_mask fuzzy --concept_polarization_lambda 1.0 --min_pos_concepts 3 "
    "--decorrelate_concepts True --decorrelation_groups 6", # Decorrelation Batch Norm params.

    "python main.py --data_path ./data/cem_200x26_1.0-1.0/samples/sets --use_global_concepts True " # Dataset params.
    "--batch 32 --task_epochs 10 --train continual_task  --model resnet50_head_only --lr -0.001 " # Training params.
    "--hamming_margin 1 --triplet_lambda 1.0 --replay_buffer 200 --replay_lambda 1.0 " # Loss params.
    "--concept_lambda 1.0 --use_mask crisp --concept_polarization_lambda 1.0 --min_pos_concepts 3 "
    "--decorrelate_concepts True --decorrelation_groups 6", # Decorrelation Batch Norm params.

    "python main.py --data_path ./data/cem_200x26_1.0-1.0/samples/sets --use_global_concepts True " # Dataset params.
    "--batch 32 --task_epochs 10 --train continual_task  --model resnet50_head_only --lr -0.001 " # Training params.
    "--hamming_margin 1 --triplet_lambda 1.0 --replay_buffer 200 --replay_lambda 1.0 " # Loss params.
    "--concept_lambda 1.0 --use_mask no --concept_polarization_lambda 1.0 --min_pos_concepts 3 "
    "--decorrelate_concepts True --decorrelation_groups 6", # Decorrelation Batch Norm params.

    "python main.py --data_path ./data/cem_200x26_1.0-1.0/samples/sets --use_global_concepts True " # Dataset params.
    "--batch 32 --task_epochs 10 --train continual_task  --model resnet50_head_only --lr -0.001 " # Training params.
    "--hamming_margin 1 --triplet_lambda 1.0 --replay_buffer 200 --replay_lambda 1.0 " # Loss params.
    "--concept_lambda 0.0 --use_mask fuzzy --concept_polarization_lambda 1.0 --min_pos_concepts 0 "
    "--decorrelate_concepts True --decorrelation_groups 6", # Decorrelation Batch Norm params.

    "python main.py --data_path ./data/cem_200x26_1.0-1.0/samples/sets --use_global_concepts True " # Dataset params.
    "--batch 32 --task_epochs 10 --train continual_task  --model resnet50_head_only --lr -0.001 " # Training params.
    "--hamming_margin 1 --triplet_lambda 1.0 --replay_buffer 200 --replay_lambda 1.0 " # Loss params.
    "--concept_lambda 1.0 --use_mask fuzzy --concept_polarization_lambda 0.0 --min_pos_concepts 3 "
    "--decorrelate_concepts True --decorrelation_groups 6", # Decorrelation Batch Norm params.

    "python main.py --data_path ./data/cem_200x26_1.0-1.0/samples/sets --use_global_concepts True " # Dataset params.
    "--batch 32 --task_epochs 10 --train continual_task  --model resnet50_head_only --lr -0.001 " # Training params.
    "--hamming_margin 1 --triplet_lambda 1.0 --replay_buffer 200 --replay_lambda 1.0 " # Loss params.
    "--concept_lambda 0.0 --use_mask no --concept_polarization_lambda 0.0 --min_pos_concepts 0 "
    "--decorrelate_concepts True --decorrelation_groups 6", # Decorrelation Batch Norm params.
]

arg_parser.add_argument("--experiment_id",
                        help="ID of the experiment to run.",
                        type=ArgNumber(int, min_val=0, max_val=len(commands)), default=1)

opts = vars(arg_parser.parse_args())

cmd = commands[opts['experiment_id']] + " " + const_str

print("Executing the following command:")
print("\t" + cmd)
os.system(cmd)
