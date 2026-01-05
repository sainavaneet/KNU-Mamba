#!/bin/bash

# To test a trained model, uncomment and set the checkpoint_path:
# checkpoint_path=/path/to/checkpoint/directory

# Set number of training epochs (default: 500, set lower for quick testing)
# trainer.training.epoch=10

python run.py \
    dataset=libero_object \
    dataset.demos_per_task=1 \
    trainer.training.eval_every_n_epochs=5 \
    trainer.training.save_every_n_epochs=10 \
    trainer.training.epoch=500 \
    seed=42 \
    device=cuda \
    simulation.render_image=false \
    simulation.n_cores=16 \
    simulation.use_multiprocessing=false \
    simulation.save_video=true \
    ${checkpoint_path:+checkpoint_path=$checkpoint_path}