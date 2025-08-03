#adapted from: https://github.com/Vchitect/Latte

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import io
import os
from os.path import join
import math
import logging
import argparse
from pytorch_lightning.callbacks import Callback, EarlyStopping
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from autoencoders import AutoencoderKLAD
from copy import deepcopy
from synthdiff.models import get_models
from synthdiff.datasets import get_datamodule
from synthdiff.diffusion import create_diffusion
from diffusers.optimization import get_scheduler
from omegaconf import OmegaConf
from conddiff.conddiff_utils import (clip_grad_norm_, update_ema, 
                   requires_grad, cleanup, 
                   get_experiment_dir)


        
class BestCheckpointCallback(Callback):
    def __init__(self, checkpoint_dir):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.best_train_loss = float('inf')
        self.best_val_loss = float('inf')

        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def on_train_epoch_end(self, trainer, pl_module):
        checkpoint_path = os.path.join(self.checkpoint_dir, "checkpoint.pt")
        self._save_checkpoint(trainer, pl_module, checkpoint_path)
        current_train_loss = trainer.callback_metrics.get("train_loss", None)
        if current_train_loss is not None and current_train_loss < self.best_train_loss:
            self.best_train_loss = current_train_loss


    def on_validation_epoch_end(self, trainer, pl_module):
        current_val_loss = trainer.callback_metrics.get("val_loss", None)
        if current_val_loss is not None and current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            checkpoint_path = os.path.join(self.checkpoint_dir, "val_checkpoint.pt")
            self._save_checkpoint(trainer, pl_module, checkpoint_path)
            print(f"Updated best validation checkpoint at {checkpoint_path} with loss: {current_val_loss:.4f}")

    def _save_checkpoint(self, trainer, pl_module, path):
        checkpoint = trainer._checkpoint_connector.dump_checkpoint()

        # Add EMA to the checkpoint
        checkpoint["ema"] = pl_module.ema.state_dict()

        # Save the modified checkpoint
        torch.save(checkpoint, path)


class CondDiffTrainingModule(LightningModule):
    def __init__(self, args, device, logger: logging.Logger):
        super(CondDiffTrainingModule, self).__init__()
        self.args = args
        self.logging = logger
        self.model = get_models(args).to(device)
        self.ema = deepcopy(self.model).to(device)  # Create an EMA of the model for use after training
        requires_grad(self.ema, False)

        # Load pretrained model if specified
        if args.pretrained:
            # Load old checkpoint, only load EMA
            self._load_pretrained_parameters(args)
        self.logging.info(f"Model Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        self.diffusion = create_diffusion(timestep_respacing="")
        input_dims = [(160, 160, 160)]

        if not os.path.exists(args.vae_path):
            raise ValueError(f"Model not found at {args.vae_path}")
        self.vae = AutoencoderKLAD.load_from_checkpoint(args.vae_path, cfg="./example_configs/aekl_ad_3d_jp.yaml", input_dim=input_dims)
        self.vae.eval() 


        self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=0)
        self.lr_scheduler = None

        # Freeze VAE
        self.vae.requires_grad_(False)

        update_ema(self.ema, self.model, decay=0)  # Ensure EMA is initialized with synced weights
        self.model.train()  # important! This enables embedding dropout for classifier-free guidance
        self.ema.eval()

    def _load_pretrained_parameters(self, args):
        checkpoint = torch.load(args.pretrained, map_location=lambda storage, loc: storage)
        if "ema" in checkpoint:  
            self.logging.info("Using ema ckpt!")
            checkpoint = checkpoint["ema"]

        model_dict = self.model.state_dict()

        pretrained_dict = {}
        for k, v in checkpoint.items():
            if k in model_dict:
                pretrained_dict[k] = v
            else:
                self.logging.info("Ignoring: {}".format(k))
        self.logging.info(f"Successfully Load {len(pretrained_dict) / len(checkpoint.items()) * 100}% original pretrained model weights ")


        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        self.logging.info(f"Successfully load model at {args.pretrained}!")

    def training_step(self, batch, batch_idx):
        loss = self.step_(batch, batch_idx)
        for loss_n, loss_val in loss.items():
            self.log(
                f"train_{loss_n}", loss_val, on_epoch=True, prog_bar=True, logger=True
            )
        return loss["loss"]

    def validation_step(self, batch, batch_idx):
        loss = self.step_(batch, batch_idx)
        for loss_n, loss_val in loss.items():
            self.log(
                f"val_{loss_n}", loss_val, on_epoch=True, prog_bar=True, logger=True, sync_dist=True,
            )
        return loss["loss"]
    
    def step_(self, batch, batch_idx):
        sample_im = batch['input_healthy']

        x = sample_im.to(self.device, non_blocking=True)
        if self.args.labels:
            label = batch['label'].to(self.device, non_blocking=True)

        with torch.no_grad():

            x = x.to(self.device)
            this_device = x.device
            self.vae = self.vae.to(this_device)
            x = self.vae.encode([x])[0]._sample().mul_(0.18215)
            x = x.to(this_device)

            
        if self.args.labels:
            model_kwargs = dict(y=label)
        else:
            model_kwargs = dict(y=None)

        t = torch.randint(0, self.diffusion.num_timesteps, (x.shape[0],), device=self.device)

        loss_dict = self.diffusion.training_losses(self.model, x, t, model_kwargs)
        loss = loss_dict["loss"].mean()

        if self.global_step < self.args.start_clip_iter:
            gradient_norm = clip_grad_norm_(self.model.parameters(), self.args.clip_max_norm, clip_grad=False)
        else:
            gradient_norm = clip_grad_norm_(self.model.parameters(), self.args.clip_max_norm, clip_grad=True)


        if (self.global_step+1) % self.args.log_every == 0:
            self.logging.info(
                f"(step={self.global_step+1:07d}/epoch={self.current_epoch:04d}) Train Loss: {loss:.4f}, Gradient Norm: {gradient_norm:.4f}"
            )
        loss = {
            "loss": loss,
        }
        return loss
            

    def on_train_batch_end(self, *args, **kwargs):
        update_ema(self.ema, self.model)

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint_dir = self.trainer.checkpoint_callback.dirpath
        epoch = self.trainer.current_epoch
        step = self.trainer.global_step
        checkpoint = {
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
        }
        torch.save(checkpoint, f"{checkpoint_dir}/epoch{epoch}-step{step}.ckpt")

    def configure_optimizers(self):
        self.lr_scheduler = get_scheduler(
            name="constant",
            optimizer=self.opt,
            num_warmup_steps=self.args.lr_warmup_steps * self.args.gradient_accumulation_steps,
            num_training_steps=self.args.max_train_steps * self.args.gradient_accumulation_steps,
        )
        return [self.opt], [self.lr_scheduler]


def create_logger(logging_dir):
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger

def create_experiment_directory(args):

    os.makedirs(args.results_dir, exist_ok=True) 
    folders = [f for f in os.listdir(args.results_dir) if os.path.isdir(os.path.join(args.results_dir, f))]
    #order the folders by name
    folders = sorted(folders)

    if len(folders) > 0:
        experiment_index = int(folders[-1].split("-")[0]) + 1
    else:
        experiment_index = 1 
    num_frame_string = 'F' + str(args.num_frames) + 'S' + str(args.frame_interval)
    if args.resume_from_checkpoint is True:
        experiment_dir = args.experiment_dir
    else:
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{num_frame_string}-{args.dataset}"  

        if os.path.exists(experiment_dir):
            raise ValueError(f"Experiment directory {experiment_dir} already exists")
        experiment_dir = get_experiment_dir(experiment_dir, args)
        
    checkpoint_dir = f"{experiment_dir}/checkpoints"  
    os.makedirs(checkpoint_dir, exist_ok=True)

    return experiment_dir, checkpoint_dir
    
def main(args):
    seed = args.global_seed
    torch.manual_seed(seed)

    is_main_process = (int(os.environ.get("LOCAL_RANK", 0)) == 0)

    if is_main_process:
        experiment_dir, checkpoint_dir = create_experiment_directory(args)
        logger = create_logger(experiment_dir)
        OmegaConf.save(args, os.path.join(experiment_dir, "config.yaml"))
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        experiment_dir = os.getenv("EXPERIMENT_DIR", "default_path")
        checkpoint_dir = os.getenv("CHECKPOINT_DIR", "default_path")
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    tb_logger = TensorBoardLogger(experiment_dir)


    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    print(f"Local rank: {local_rank}")
    device = torch.device("cuda", local_rank)
    args.training_ = True
    args.device = device
    datamodule = get_datamodule(args)
    len_ = len(datamodule.train_dataloader())

    if is_main_process:
        logger.info(f"Dataset contains {len_} videos ({args.data_path})")

    sample_size = args.image_size // 8
    args.latent_size = sample_size

    num_update_steps_per_epoch = math.ceil(len_)

    if args.max_epochs is not None:
        num_train_epochs = args.max_epochs
        args.max_train_steps = num_train_epochs * num_update_steps_per_epoch
    else:
        num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        args.max_train_steps = args.max_train_steps

    if is_main_process:
        logger.info(f"One epoch iteration {num_update_steps_per_epoch} steps")
        logger.info(f"Num train epochs: {num_train_epochs}")

    pl_module = CondDiffTrainingModule(args, device, logger)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch}-{step}-{train_loss:.2f}-{gradient_norm:.2f}",
        save_top_k=-1,
        every_n_train_steps=args.ckpt_every,
        save_on_train_epoch_end=True,  
    )

    best_checkpoint_callback = BestCheckpointCallback(checkpoint_dir=checkpoint_dir)

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.001,
        patience=args.patience,
        verbose=True,
        mode="min",
    )


    trainer = Trainer(
        accelerator="gpu",
        devices=args.devices, 
        strategy="ddp_find_unused_parameters_true",
        max_epochs=num_train_epochs,
        logger=tb_logger,
        callbacks=[checkpoint_callback, best_checkpoint_callback, LearningRateMonitor(), early_stopping_callback],
        log_every_n_steps=args.log_every,
    )

    trainer.fit(pl_module, datamodule, ckpt_path=args.resume_from_checkpoint if 
                args.resume_from_checkpoint else None)

    pl_module.model.eval()
    cleanup()
    if is_main_process:
        logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train.yaml")
    args = parser.parse_args()
    main(OmegaConf.load(args.config))
