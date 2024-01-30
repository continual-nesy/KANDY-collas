"""This is just a bare runner, with a list of commands to execute"""
import os

commands = [
    "python main.py --data_path ./data/cem_200x26_1.0-1.0/samples/sets --weight_decay 0. "
    "--print_every 8 --augment false --batch 16 --task_epochs 10 --train continual_task --seed 9101 "
    "--cem_emb_size 12 --hamming_margin 1 --triplet_lambda 0.0 --concept_lambda 0.0 --use_mask no "
    "--concept_polarization_lambda 0.0 --mask_polarization_lambda 0.0 --min_pos_concepts 0 --n_concepts 30 "
    "--model vit_head_only --output_folder exp --balance true --replay_buffer 200 --replay_lambda 1.0 --lr -0.001 "
    "--store_fuzzy no --cls_lambda 1.0 --use_global_concepts True --device cuda:0 --wandb_project kandy-cem "
    "--compute_training_metrics False --correlate_each_task True --share_embeddings False --decorrelate_concepts True "
    "--decorrelation_groups 6"
]

for command in commands:
    print("Executing the following command:")
    print("\t" + command)
    os.system(command)
