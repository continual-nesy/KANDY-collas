"""This is just a bare runner, with a list of commands to execute"""
import os

commands = [
    "python main.py --data_path ./data/micro_cem_200x3_1.0-1.0/samples/sets --weight_decay 0. "
    "--print_every 8 --augment false --batch 32 --task_epochs 100 --train continual_task --seed 9101 "
    "--cem_emb_size 12 --hamming_margin 2 --triplet_lambda 0.1 --concept_lambda 0.1 --use_mask fuzzy "
    "--concept_polarization_lambda 0.1 --mask_polarization_lambda 0.1 --min_pos_concepts 2 --n_concepts 25 "
    "--model vit_head_only --output_folder exp --balance true --replay_buffer 50 --replay_lambda 0.1 --lr -0.001 "
    "--store_fuzzy no --cls_lambda 1.0 --use_global_concepts True --device cuda:0 --wandb_project kandy-cem",
]

for command in commands:
    print("Executing the following command:")
    print("\t" + command)
    os.system(command)
