"""Main training script using Hydra configuration."""

import os
import sys
import logging
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import warnings
import multiprocessing as mp
import datetime
from tqdm import tqdm

# Suppress robosuite warnings
warnings.filterwarnings("ignore", message="No private macro file found!")
warnings.filterwarnings("ignore", message="It is recommended to use a private macro file")
warnings.filterwarnings("ignore", message="To setup, run:.*setup_macros.py")

# Setup
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

os.environ['NUMEXPR_MAX_THREADS'] = '64'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from factory import create_model, create_trainer, create_simulation

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
logging.getLogger('robosuite_logs').setLevel(logging.ERROR)


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    checkpoint_path = cfg.get('checkpoint_path', None)
    
    # Multi-GPU setup
    use_multi_gpu = cfg.get('use_multi_gpu', False)
    num_gpus = cfg.get('num_gpus', 1)
    multi_gpu_mode = cfg.get('multi_gpu_mode', 'none')  # 'none', 'dp', or 'ddp'
    
    # Initialize distributed training if using DDP
    if multi_gpu_mode == 'ddp':
        # torchrun sets these environment variables automatically
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ and 'LOCAL_RANK' in os.environ:
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            local_rank = int(os.environ['LOCAL_RANK'])
            
            # Initialize process group if not already initialized
            if not dist.is_initialized():
                dist.init_process_group(backend='nccl')
        else:
            # Fallback: initialize manually (for testing or non-torchrun usage)
            if not dist.is_initialized():
                dist.init_process_group(backend='nccl')
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            local_rank = rank % torch.cuda.device_count()
        
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        cfg.device = f'cuda:{local_rank}'
        
        # Only initialize wandb on rank 0
        if rank == 0:
            wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, group=cfg.group)
            # Log config to wandb
            wandb.config.update(OmegaConf.to_container(cfg, resolve=True))
        else:
            os.environ['WANDB_MODE'] = 'disabled'
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device(cfg.device if 'device' in cfg else 'cuda')
        if use_multi_gpu and multi_gpu_mode == 'dp' and num_gpus > 1:
            # DataParallel mode - use all available GPUs
            if not torch.cuda.is_available():
                log.warning("CUDA not available, falling back to CPU")
                cfg.device = 'cpu'
                device = torch.device('cpu')
                use_multi_gpu = False
            else:
                available_gpus = torch.cuda.device_count()
                num_gpus = min(num_gpus, available_gpus)
                log.info(f"Using DataParallel with {num_gpus} GPUs")
                device = torch.device('cuda:0')
        
        # Initialize wandb
        wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, group=cfg.group)
        # Log config to wandb
        wandb.config.update(OmegaConf.to_container(cfg, resolve=True))
    
    set_seed(cfg.seed)
    
    # Create output directory
    now = datetime.datetime.now()
    run_output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "runs",
        cfg.dataset.benchmark_type,
        now.strftime("%Y-%m-%d"),
        now.strftime("%H-%M-%S")
    )
    os.makedirs(run_output_dir, exist_ok=True)

    # Create model, trainer, and simulation
    model = create_model(cfg)
    model.working_dir = run_output_dir
    
    # Wrap model for multi-GPU training
    if multi_gpu_mode == 'ddp':
        model = model.to(device)
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True  # Set to True to handle unused parameters
        )
        log.info(f"Model wrapped with DistributedDataParallel (rank {rank}/{world_size-1})")
    elif use_multi_gpu and multi_gpu_mode == 'dp' and num_gpus > 1:
        model = model.to('cuda:0')
        model = DataParallel(model, device_ids=list(range(num_gpus)))
        log.info(f"Model wrapped with DataParallel on {num_gpus} GPUs")
    else:
        # Single GPU or CPU
        model = model.to(cfg.device)
    
    # Get the actual model (unwrap DDP/DP if needed) for operations that need the base model
    actual_model = model.module if hasattr(model, 'module') else model
    
    trainer = create_trainer(cfg, use_multi_gpu=use_multi_gpu, multi_gpu_mode=multi_gpu_mode, rank=rank, world_size=world_size)
    trainer.working_dir = run_output_dir
    
    # Only print params on rank 0
    if rank == 0:
        if hasattr(actual_model, 'get_params') and callable(getattr(actual_model, 'get_params', None)):
            actual_model.get_params()  # type: ignore

    # Setup simulation for periodic evaluation (only on rank 0)
    if rank == 0:
        env_sim = create_simulation(cfg)
        if cfg.simulation.save_video:
            env_sim.save_video_dir = os.path.join(run_output_dir, "videos")
        env_sim.get_task_embs(trainer.trainset.tasks)
    else:
        env_sim = None

    # Load checkpoint if provided
    if checkpoint_path:
        if hasattr(actual_model, 'set_scaler') and callable(getattr(actual_model, 'set_scaler', None)):
            actual_model.set_scaler(trainer.scaler)  # type: ignore
        
        if os.path.isfile(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=cfg.device, weights_only=True)
            actual_model.load_state_dict(state_dict)
        else:
            # Try to find checkpoint in directory
            for name in ["final_model.pth", "model_state_dict.pth"]:
                path = os.path.join(checkpoint_path, name)
                if os.path.isfile(path):
                    state_dict = torch.load(path, map_location=cfg.device, weights_only=True)
                    actual_model.load_state_dict(state_dict)
                    break
            else:
                raise FileNotFoundError(f"No checkpoint found in {checkpoint_path}")
        
        if rank == 0 and env_sim is not None:
            env_sim.test_model(actual_model, cfg.model_cfg, epoch=cfg.epoch)
    else:
        # Train with periodic evaluation
        trainer._setup_training_components(actual_model)
        
        current_epoch = 0
        total_epochs = trainer.epoch
        eval_every = trainer.eval_every_n_epochs
        
        # Only show progress bar on rank 0
        if rank == 0:
            pbar = tqdm(total=total_epochs, desc="Training", dynamic_ncols=True)
        else:
            pbar = None
        
        while current_epoch < total_epochs:
            # Train for eval_every_n_epochs
            epochs_to_train = min(eval_every, total_epochs - current_epoch)
            for offset in range(epochs_to_train):
                epoch = current_epoch + offset
                loss = trainer._train_single_epoch(model, epoch, pbar)
                if rank == 0:
                    trainer._log_epoch_results(epoch, loss)
                    trainer._save_checkpoint_if_needed(actual_model, epoch)
                    if pbar is not None:
                        pbar.update(1)
            
            current_epoch += epochs_to_train
            
            # Evaluate periodically (only on rank 0)
            if current_epoch % eval_every == 0 and current_epoch < total_epochs:
                if rank == 0 and env_sim is not None:
                    if pbar is not None:
                        pbar.set_description(f"Evaluating at epoch {current_epoch}")
                    training_device = next(actual_model.parameters()).device
                    env_sim.test_model(actual_model, cfg.model_cfg, epoch=current_epoch)
                    if pbar is not None:
                        pbar.set_description("Training")
            
            # Synchronize all processes after each evaluation
            if multi_gpu_mode == 'ddp':
                dist.barrier()
        
        if pbar is not None:
            pbar.close()
        
        trainer._finalize_training(actual_model)
        
        if rank == 0 and env_sim is not None:
            env_sim.test_model(actual_model, cfg.model_cfg, epoch=total_epochs)
    
    if rank == 0:
        log.info(f"Training done. Checkpoints saved in {actual_model.working_dir if 'actual_model' in locals() else model.working_dir}")
        wandb.finish()
    
    # Clean up distributed training
    if multi_gpu_mode == 'ddp':
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
