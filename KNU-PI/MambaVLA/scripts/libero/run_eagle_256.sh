#!/bin/bash

# Training script with Eagle backbone (2048 dimensions) using DistributedDataParallel
# Usage: NUM_GPUS=4 ./scripts/libero/run_eagle_256.sh

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate test2

# Configuration
NUM_GPUS=2

# Run training with DistributedDataParallel
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    run.py \
    wandb.project=eagle_backbone \
    wandb.entity=sainavaneet \
    group=libero_object \
    +use_multi_gpu=true \
    +num_gpus=$NUM_GPUS \
    +multi_gpu_mode=ddp \
    dataset=libero_object \
    image_encoders=eagle \
    latent_dim=2048 \
    len_embd=2048 \
    model_cfg.latent_dim=2048 \
    model_cfg.model.backbones.latent_dim=2048 \
    model_cfg.model.backbones.embed_dim=2048 \
    model_cfg.model.backbones.encoder.d_model=2048 \
    model_cfg.model.backbones.encoder.d_intermediate=2048 \
    model_cfg.model.backbones.encoder.n_layer=5 \
    model_cfg.model.backbones.encoder.ssm_cfg.layer=Mamba1 \
    model_cfg.model.backbones.encoder.ssm_cfg.d_state=64 \
    model_cfg.model.backbones.encoder.ssm_cfg.d_conv=4 \
    model_cfg.model.backbones.encoder.ssm_cfg.expand=2 \
    model_cfg.use_lr_scheduler=true \
    model_cfg.optimizer.learning_rate=0.00005 \
    image_encoders.latent_dim=2048 \
    image_encoders.tune_llm=false \
    image_encoders.tune_visual=true \
    trainer.ema.if_use_ema=true \
    dataset.demos_per_task=50 \
    trainer.training.epoch=500 \
    trainer.training.eval_every_n_epochs=100 \
    trainer.training.save_every_n_epochs=100 \
    trainer.data_loading.train_batch_size=8 \
    trainer.data_loading.val_batch_size=8 \
    trainer.data_loading.num_workers=4 \
    seed=42 \
    device=cuda \
    simulation.render_image=false \
    simulation.n_cores=16 \
    simulation.use_multiprocessing=false \
    simulation.save_video=true
