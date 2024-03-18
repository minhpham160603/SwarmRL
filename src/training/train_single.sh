#!/bin/bash

# Store the seed provided as an argument
max_seed=1

# Loop for five different seeds
for i in `seq $max_seed`;
do
    echo "Running training with seed ${i}"

    # Execute your Python script with the specified seed as an argument
    python train_single_env.py --seed $i --total_steps 200000 --use_wandb --use_record 

    echo "Training with seed ${i} completed"
done
