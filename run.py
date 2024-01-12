"""This is just a bare runner, with a list of commands to execute"""
import os

commands = [
    "python main.py --data_path ./data/easy_100x20_1.0-1.0/samples/sets --weight_decay 0. "
    "--print_every 1 --augment false --batch 16 --task_epochs 1 --train continual_task --seed 9101 "
    "--cem_emb_size 12 --hamming_margin 8 --triplet_lambda 1.0 --concept_lambda 1.0 "
    "--model cnn --output_folder exp --balance true --replay_buffer 0 --replay_lambda 0. --lr -0.001  --device cuda:0",
]

for command in commands:
    print("Executing the following command:")
    print("\t" + command)
    os.system(command)
