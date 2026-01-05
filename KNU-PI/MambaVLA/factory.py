import torch
from typing import Any
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from pathlib import Path
from typing import cast
from MambaVLA.benchmark.libero.libero_dataset import LiberoDataset
from MambaVLA.main import Trainer



def create_model(cfg: DictConfig) -> Any:
    """Create a model from the Hydra configuration."""
    model_config = cfg.model_cfg
    
    # Create the encoder first
    encoder = instantiate(model_config.model.backbones.encoder)
    
    # Create the backbone with the encoder
    backbone = instantiate(
        model_config.model.backbones,
        encoder=encoder
    )
    
    # Create the model with the backbone
    model = instantiate(
        model_config.model,
        backbones=backbone
    )
    

    # Instantiate the observation encoder from config
    # This supports both resnet and eagle backbones via config
    obs_encoder = instantiate(cfg.image_encoders)
    lang_encoder = instantiate(cfg.language_encoders)
    

    lr_scheduler_config = DictConfig({'lr_scheduler': OmegaConf.to_container(model_config.lr_scheduler, resolve=True)})
    
    model = instantiate(
        model_config,
        model=model,
        obs_encoders=obs_encoder,
        language_encoders=lang_encoder,
        optimizer=model_config.optimizer,  
        lr_scheduler=lr_scheduler_config,  
        action_dim=cfg.action_dim,
        perception_seq_len=cfg.perception_seq_len,
        action_seq_len=cfg.action_seq_len,
        cam_names=cfg.camera_names,
        device=cfg.device,
        state_dim=cfg.state_dim,
        latent_dim=cfg.latent_dim,
        sampling_steps=cfg.num_sampling_steps
    )
    
    return model


def create_trainer(cfg: DictConfig, use_multi_gpu: bool = False, multi_gpu_mode: str = 'none', rank: int = 0, world_size: int = 1) -> Any:
    """Create a trainer from the Hydra configuration."""
    trainer_config = cfg.trainer
  
    dataset = LiberoDataset(
        data_directory=Path(cfg.dataset.dataset_path),
        device=cfg.device,
        obs_dim=cfg.obs_dim,
        action_dim=cfg.action_dim,
        state_dim=cfg.state_dim,
        max_len_data=cfg.max_len_data,
        chunck_size=cfg.chunck_size,
        demos_per_task=cfg.dataset.demos_per_task
    )
    

    trainer = Trainer(
        training_dataset=dataset,
        validation_dataset=dataset,  
        training_batch_size=trainer_config.data_loading.train_batch_size,
        validation_batch_size=trainer_config.data_loading.val_batch_size,
        dataloader_workers=trainer_config.data_loading.num_workers,
        device=trainer_config.device,
        total_epochs=trainer_config.training.epoch,
        enable_data_scaling=trainer_config.data_scaling.scale_data,
        data_scaler_type=trainer_config.data_scaling.scaling_type,
        evaluation_frequency=trainer_config.training.eval_every_n_epochs,
        observation_sequence_length=trainer_config.training.perception_seq_len,
        ema_decay_rate=trainer_config.ema.decay_ema,
        enable_ema=trainer_config.ema.if_use_ema,
        checkpoint_frequency=trainer_config.training.save_every_n_epochs,
        use_multi_gpu=use_multi_gpu,
        multi_gpu_mode=multi_gpu_mode,
        rank=rank,
        world_size=world_size,
        use_mixed_precision=trainer_config.mixed_precision.use_mixed_precision,
        gradient_clip_val=trainer_config.mixed_precision.gradient_clip_val
    )
    
    return trainer


def create_simulation(cfg: DictConfig) -> Any:
    """Create a simulation from the Hydra configuration."""
    sim_config = cfg.simulation
    
    # Create the simulation with all required parameters
    simulation = instantiate(
        sim_config,
        rollouts=sim_config.rollouts,
        max_step_per_episode=sim_config.max_step_per_episode,
        # Always source the task suite from the dataset to avoid stale defaults
        benchmark_type=cfg.dataset.benchmark_type,
        use_eye_in_hand=sim_config.use_eye_in_hand,
        seed=cfg.seed,
        device=cfg.device,
        render_image=sim_config.render_image,
        n_cores=sim_config.n_cores,
        use_multiprocessing=sim_config.use_multiprocessing
    )
    
    return simulation
