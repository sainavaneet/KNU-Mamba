#!/bin/bash

python run.py \
    dataset=libero_spatial \
    dataset.demos_per_task=10 \
    trainer.training.eval_every_n_epochs=1 \
    trainer.training.save_every_n_epochs=10 \
    trainer.training.epoch=10 \
    seed=0 \
    device=cuda \
    simulation.render_image=false \
    simulation.n_cores=2 \
    simulation.use_multiprocessing=false \
    simulation.save_video=true

