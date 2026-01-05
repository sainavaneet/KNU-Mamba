"""Shared utilities for all main scripts."""

import os
import pickle
import random
import logging
import wandb
from typing import Any
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, default_collate
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.amp import autocast, GradScaler
import multiprocessing as mp

from MambaVLA.utils.scaler import Scaler, ActionScaler, MinMaxScaler
from MambaVLA.utils.ema import ExponentialMovingAverage

# Set multiprocessing start method to 'spawn' to avoid CUDA issues
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    # Already set, ignore
    pass

log = logging.getLogger(__name__)


class Trainer:
    """Basic train/test class to be inherited."""

    def __init__(
            self,
            training_dataset: Any,
            validation_dataset: Any,
            training_batch_size: int = 512,
            validation_batch_size: int = 512,
            dataloader_workers: int = 8,
            device: str = 'cpu',
            total_epochs: int = 100,
            enable_data_scaling: bool = True,
            data_scaler_type: str = "minmax",
            evaluation_frequency: int = 50,
            observation_sequence_length: int = 1,
            ema_decay_rate: float = 0.999,
            enable_ema: bool = False,
            checkpoint_frequency: int = 10,
            use_multi_gpu: bool = False,
            multi_gpu_mode: str = 'none',
            rank: int = 0,
            world_size: int = 1,
            use_mixed_precision: bool = True,
            gradient_clip_val: float = 1.0
    ):
        """Initialize."""
        
        # Dataset and data loading configuration
        self.trainset = training_dataset
        self.valset = validation_dataset
        self.train_batch_size = training_batch_size
        self.val_batch_size = validation_batch_size
        self.num_workers = dataloader_workers
        
        # Training configuration
        self.epoch = total_epochs
        self.perception_seq_len = observation_sequence_length
        self.eval_every_n_epochs = evaluation_frequency
        self.save_every_n_epochs = checkpoint_frequency
        
        # Device and environment configuration
        self.device = device
        self.working_dir = os.getcwd()
        
        # Multi-GPU configuration
        self.use_multi_gpu = use_multi_gpu
        self.multi_gpu_mode = multi_gpu_mode
        self.rank = rank
        self.world_size = world_size
        
        # Data scaling configuration
        self.scale_data = enable_data_scaling
        self.scaling_type = data_scaler_type
        
        # EMA configuration
        self.decay_ema = ema_decay_rate
        self.if_use_ema = enable_ema
        
        # Mixed precision and gradient clipping
        self.use_mixed_precision = use_mixed_precision and (device != 'cpu')
        self.gradient_clip_val = gradient_clip_val
        # GradScaler only needed for FP16, not BF16
        self.amp_scaler = GradScaler('cuda') if self.use_mixed_precision else None
        
        # Initialize data loaders
        self._setup_data_loaders()
        
        # Initialize scaler
        self._setup_scaler()

        log.info("Number of training samples: {}".format(len(self.trainset)))

    def _setup_data_loaders(self):
        """Setup training and validation data loaders."""
        # Use DistributedSampler for DDP mode
        if self.multi_gpu_mode == 'ddp' and self.world_size > 1:
            train_sampler = DistributedSampler(
                self.trainset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=True
            )
            shuffle = False  # DistributedSampler handles shuffling
        else:
            train_sampler = None
            shuffle = True
        
        self.train_dataloader = DataLoader(
            self.trainset,
            batch_size=self.train_batch_size,
            shuffle=shuffle,
            sampler=train_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=4 if self.num_workers > 0 else None
        )
        
        # Store sampler for epoch updates in DDP mode
        self.train_sampler = train_sampler

        # self.test_dataloader = DataLoader(
        #     self.valset,
        #     batch_size=self.val_batch_size,
        #     shuffle=False,
        #     num_workers=0,
        #     pin_memory=True,
        #     drop_last=False
        # )

    def _setup_scaler(self):
        """Setup data scaler based on configuration."""
        if self.scaling_type == 'minmax':
            self.scaler = MinMaxScaler(self.trainset.get_all_actions(), self.scale_data, self.device)
        else:
            self.scaler = ActionScaler(self.trainset.get_all_actions(), self.scale_data, self.device)

    def main(self, model):
        """Run main training/testing pipeline."""
        self._setup_training_components(model)
        self._run_training_loop(model)
        self._finalize_training(model)

    def _setup_training_components(self, model):
        """Setup scaler, EMA, and optimizer for training."""
        # assign scaler to model calss
        model.set_scaler(self.scaler)

        if self.if_use_ema:
            self.ema_helper = ExponentialMovingAverage(model.parameters(), self.decay_ema, self.device)

        # define optimizer
        if model.use_lr_scheduler:
            self.optimizer, self.scheduler = model.configure_optimizers()
        else:
            self.optimizer = model.configure_optimizers()

    def _run_training_loop(self, model):
        """Execute the main training loop over all epochs."""
        for num_epoch in tqdm(range(self.epoch), desc="Epochs", dynamic_ncols=True):
            # Set epoch for DistributedSampler
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(num_epoch)
            
            epoch_loss = self._train_single_epoch(model, num_epoch)
            self._log_epoch_results(num_epoch, epoch_loss)
            self._save_checkpoint_if_needed(model, num_epoch)

    def _train_single_epoch(self, model, num_epoch, pbar=None):
        """Train for a single epoch and return the average loss."""
        epoch_loss = torch.tensor(0.0).to(self.device)
        
        # Only show progress bar on rank 0
        show_progress = (self.rank == 0) and (pbar is not None)

        for data in tqdm(self.train_dataloader, desc=f"Epoch {num_epoch+1}", leave=False, dynamic_ncols=True, disable=not show_progress):
            obs_dict, action, mask = data
            obs_dict, action = self._prepare_batch_data(obs_dict, action)
            batch_loss = self.train_one_step(model, obs_dict, action)
            epoch_loss += batch_loss

        # Average loss across all GPUs in DDP mode
        if self.multi_gpu_mode == 'ddp' and self.world_size > 1:
            dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
            epoch_loss = epoch_loss / self.world_size
            num_batches = len(self.train_dataloader)
        else:
            num_batches = len(self.train_dataloader)

        avg_loss = epoch_loss / num_batches
        # Update main progress bar if provided
        if pbar is not None and self.rank == 0:
            pbar.set_postfix({"loss": f"{avg_loss.item():.4f}"})

        return avg_loss

    def _prepare_batch_data(self, obs_dict, action):
        """Prepare observation and action data for training."""
        # put data on cuda
        for camera in obs_dict.keys():
            if camera == 'lang':
                continue
            
            obs_dict[camera] = obs_dict[camera].to(self.device)

            if 'rgb' not in camera and 'image' not in camera:
                continue
            obs_dict[camera] = obs_dict[camera][:, :self.perception_seq_len].contiguous()

        action = self.scaler.scale_output(action)
        action = action[:, self.perception_seq_len - 1:, :].contiguous()

        return obs_dict, action

    def _log_epoch_results(self, num_epoch, epoch_loss):
        """Log epoch results to wandb and console."""
        wandb.log({"train_loss": epoch_loss.item()})
        log.info("Epoch {}: Mean train loss is {}".format(num_epoch, epoch_loss.item()))

    def _save_checkpoint_if_needed(self, model, num_epoch):
        """Save model checkpoint if it's time to do so."""
        if (num_epoch + 1) % self.save_every_n_epochs == 0:
            try:
                model.store_model_weights(self.working_dir, sv_name=f"epoch_{num_epoch + 1:05d}")
            except Exception as e:
                log.warning(f"Failed to save checkpoint at epoch {num_epoch + 1}: {e}")

    def _finalize_training(self, model):
        """Finalize training by applying EMA and saving final model."""
        log.info("training done")

        if self.if_use_ema:
            self.ema_helper.store(model.parameters())
            self.ema_helper.copy_to(model.parameters())

        model.store_model_weights(model.working_dir, sv_name='final_model')
        # or send weight out of the class

    def train_one_step(self, model, obs_dict, action):
        """Run a single training step with mixed precision and gradient clipping."""
        # Handle both wrapped (DDP/DP) and unwrapped models
        if hasattr(model, 'module'):
            model.train()
            actual_model = model.module
            model_params = model.module.parameters()
        else:
            model.train()
            actual_model = model
            model_params = model.parameters()

        # Forward pass with mixed precision
        # Note: Eagle backbone uses BF16 internally, so we use autocast without GradScaler
        if self.use_mixed_precision:
            with autocast(device_type='cuda', dtype=torch.float16):
                loss = model(obs_dict, action)
        else:
            loss = model(obs_dict, action)

        self.optimizer.zero_grad(set_to_none=True)
        
        # Backward pass
        # Note: Don't use GradScaler with Eagle (BF16) - use autocast only
        if self.use_mixed_precision:
            # Use autocast for backward too, but without GradScaler
            with autocast(device_type='cuda', dtype=torch.float16):
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model_params, max_norm=self.gradient_clip_val)
            self.optimizer.step()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_params, max_norm=self.gradient_clip_val)
            self.optimizer.step()

        if actual_model.use_lr_scheduler:
            self.scheduler.step()

        if self.if_use_ema:
            # Update EMA on actual model parameters
            if hasattr(model, 'module'):
                self.ema_helper.update(model.module.parameters())
            else:
                self.ema_helper.update(model.parameters())

        return loss

    @torch.no_grad()
    def evaluate_nsteps(self, model, criterion, loader, step_id, val_iters,
                        split='val'):
        """Run a given number of evaluation steps."""
        return None

