from typing import Optional
import os
import sys
import argparse
import yaml
import numpy as np
import copy
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm

import pytorch_lightning as pl
from easydict import EasyDict
import random
from pytorch_lightning import callbacks
from pytorch_lightning.accelerators import accelerator
from pytorch_lightning.core.hooks import CheckpointHooks
from pytorch_lightning.callbacks import ModelCheckpoint,DeviceStatsMonitor,EarlyStopping,LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import Namespace

from torch.utils.data import DataLoader
import pipeline

from torchvision.utils import save_image
from torchvision.models import vgg16
import matplotlib.pyplot as plt
import math
from pretraining.dcg import DCG as AuxCls
from model import *
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description='DiffMIC-v2 Diffusion Training')
    parser.add_argument('--config', type=str, default='configs/aptos.yml', help='Path to config file')
    parser.add_argument('--device', type=int, default=0, help='GPU device id')
    parser.add_argument('--seed', type=int, default=2000, help='Random seed')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--dcg_ckpt', type=str, default=None, help='Path to pretrained DCG checkpoint')
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name override')
    parser.add_argument('--output_dir', type=str, default='logs', help='Output directory')
    args = parser.parse_args()
    return args


class DiffMICv2System(pl.LightningModule):
    
    def __init__(self, hparams, diff_opt):
        super(DiffMICv2System, self).__init__()

        self.params = hparams
        self.epochs = self.params.training.n_epochs
        self.initlr = self.params.optim.lr
        self.diff_opt = diff_opt

        self.model = ConditionalModel(self.params, guidance=self.params.diffusion.include_guidance)
        self.aux_model = AuxCls(self.params)
        
        # Load pretrained DCG if path is provided
        dcg_ckpt = self.params.get('dcg_ckpt', None)
        if dcg_ckpt and os.path.exists(dcg_ckpt):
            self.init_weight(ckpt_path=dcg_ckpt)
        self.aux_model.eval()
        self.aux_model.requires_grad_(False)  # paper: DCG is frozen during diffusion training

        self.save_hyperparameters()
        
        self.gts = []
        self.preds = []

        self.DiffSampler = pipeline.SR3Sampler(
            model=self.model,
            scheduler = pipeline.create_SR3scheduler(self.diff_opt['scheduler'], 'train'),
        )
        self.DiffSampler.scheduler.set_timesteps(self.diff_opt['scheduler']['num_test_timesteps'])
        self.DiffSampler.scheduler.diff_chns = self.params.data.num_classes

    def configure_optimizers(self):
        optimizer = get_optimizer(self.params.optim, filter(lambda p: p.requires_grad, self.model.parameters()))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=self.initlr * 0.01)
        return [optimizer], [scheduler]

    def init_weight(self, ckpt_path=None):
        if ckpt_path and os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint[0] if isinstance(checkpoint, list) else checkpoint
            
            model_state = self.aux_model.state_dict()
            filtered_state = {k: v for k, v in state_dict.items() if k in model_state.keys()}
            model_state.update(filtered_state)
            self.aux_model.load_state_dict(model_state)
            print(f"Loaded DCG checkpoint from {ckpt_path}")
        elif ckpt_path:
            print(f"Warning: DCG checkpoint not found at {ckpt_path}. Using random initialization.")

    def guided_prob_map(self, y0_g, y0_l, bz, nc, num_crops):
        distance_to_diag = torch.abs(torch.arange(num_crops, device=self.device).unsqueeze(0) - torch.arange(num_crops, device=self.device).unsqueeze(1))
        weight_g = 1 - distance_to_diag / (num_crops - 1)
        weight_l = distance_to_diag / (num_crops - 1)
        interpolated_value = weight_l.unsqueeze(0).unsqueeze(0) * y0_l.unsqueeze(-1).unsqueeze(-1) + weight_g.unsqueeze(0).unsqueeze(0) * y0_g.unsqueeze(-1).unsqueeze(-1)
        diag_indices = torch.arange(num_crops, device=self.device)
        # Vectorized assignment (no Python loops)
        interpolated_value[:, :, diag_indices, diag_indices] = y0_g.unsqueeze(-1)
        interpolated_value[:, :, num_crops-1, 0] = y0_l
        interpolated_value[:, :, 0, num_crops-1] = y0_l
        return interpolated_value

    def training_step(self, batch, batch_idx):
        self.model.train()
        self.aux_model.eval()
        
        x_batch, y_batch = batch
        y_batch, _ = cast_label_to_one_hot_and_prototype(y_batch, self.params)
        y_batch = y_batch.to(self.device)
        x_batch = x_batch.to(self.device)
        
        with torch.no_grad():
            y0_aux, y0_aux_global, y0_aux_local, patches, attns, attn_map = self.aux_model(x_batch)
        
        bz, nc, H, W = attn_map.size()
        bz, num_crops = attns.size()
        
        y_map = y_batch.unsqueeze(1).expand(-1, num_crops*num_crops, -1).reshape(bz*num_crops*num_crops, nc)
        noise = torch.randn_like(y_map).to(self.device)
        timesteps = torch.randint(0, self.DiffSampler.scheduler.config.num_train_timesteps, (bz*num_crops*num_crops,), device=self.device).long()

        noisy_y = self.DiffSampler.scheduler.add_noise(y_map, timesteps=timesteps, noise=noise)
        noisy_y = noisy_y.view(bz, num_crops*num_crops, -1).permute(0, 2, 1).reshape(bz, nc, num_crops, num_crops)
        
        y0_cond = self.guided_prob_map(y0_aux_global, y0_aux_local, bz, nc, num_crops)
        y_fusion = torch.cat([y0_cond, noisy_y], dim=1)

        attns = attns.unsqueeze(-1)
        attns = (attns * attns.transpose(1, 2)).unsqueeze(1)
        noise_pred = self.model(x_batch, y_fusion, timesteps, patches, attns)

        noise = noise.view(bz, num_crops*num_crops, -1).permute(0, 2, 1).reshape(bz, nc, num_crops, num_crops)
        # Paper Eq. 9: plain MSE loss — no focal weighting
        loss = F.mse_loss(noise_pred, noise)

        self.log("train_loss", loss, prog_bar=True)
        return {"loss": loss}

    def on_validation_epoch_end(self):
        if len(self.gts) == 0 or len(self.preds) == 0:
            return
        gt = torch.cat(self.gts)
        pred = torch.cat(self.preds)
        ACC, BACC, Prec, Rec, F1, AUC_ovo, kappa = compute_isic_metrics(gt, pred)

        self.log('accuracy', ACC)
        self.log('f1', F1)
        self.log('Precision', Prec)        
        self.log('Recall', Rec)
        self.log('AUC', AUC_ovo)
        self.log('kappa', kappa)   
        
        self.gts = []
        self.preds = []
        print("Val: Accuracy {0}, F1 score {1}, Precision {2}, Recall {3}, AUROC {4}, Cohen Kappa {5}".format(ACC, F1, Prec, Rec, AUC_ovo, kappa))

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        self.aux_model.eval()

        x_batch, y_batch = batch
        y_batch, _ = cast_label_to_one_hot_and_prototype(y_batch, self.params)
        y_batch = y_batch.to(self.device)
        x_batch = x_batch.to(self.device)
        
        with torch.no_grad():
            y0_aux, y0_aux_global, y0_aux_local, patches, attns, attn_map = self.aux_model(x_batch)

        bz, nc, H, W = attn_map.size()
        bz, num_crops = attns.size()

        y0_cond = self.guided_prob_map(y0_aux_global, y0_aux_local, bz, nc, num_crops)
        # Paper Eq.10: start from N(M, I) — Gaussian centered at dense guidance map
        yT = y0_cond + torch.randn_like(y0_cond)
        
        attns = attns.unsqueeze(-1)
        attns = (attns * attns.transpose(1, 2)).unsqueeze(1)
        
        y_pred = self.DiffSampler.sample_high_res(x_batch, yT, conditions=[y0_cond, patches, attns])
        y_pred = y_pred.reshape(bz, nc, num_crops*num_crops)
        y_pred = y_pred.mean(2)
        
        self.preds.append(y_pred)
        self.gts.append(y_batch)
    
    def train_dataloader(self):
        data_object, train_dataset, test_dataset = get_dataset(self.params)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.params.training.batch_size,
            shuffle=True,
            num_workers=self.params.data.num_workers,
            pin_memory=True,
        )
        return train_loader
    
    def val_dataloader(self):
        data_object, train_dataset, test_dataset = get_dataset(self.params)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.params.testing.batch_size,
            shuffle=False,
            num_workers=self.params.data.num_workers,
            pin_memory=True,
        )
        return test_loader  


def main():
    args = parse_args()

    # Load config
    with open(args.config, 'r') as f:
        params = yaml.safe_load(f)
    config = EasyDict(params)

    # Set seed
    seed = args.seed if args.seed else config.data.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    # Setup paths
    dataset_name = config.data.dataset.lower()
    exp_name = args.exp_name if args.exp_name else dataset_name
    output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Add DCG checkpoint path to config
    if args.dcg_ckpt:
        config.dcg_ckpt = args.dcg_ckpt
    else:
        config.dcg_ckpt = f'pretraining/ckpt/{dataset_name}_aux_model.pth'

    # Logger
    logger = TensorBoardLogger(name=exp_name, save_dir=output_dir)

    # Load DDIM options
    config_path = r'option/diff_DDIM.yaml'
    with open(config_path, 'r') as f:
        diff_params = yaml.safe_load(f)
    diff_opt = EasyDict(diff_params)

    # Initialize model
    model = DiffMICv2System(config, diff_opt)

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='f1',
        filename=f'{dataset_name}-epoch{{epoch:02d}}-accuracy-{{accuracy:.4f}}-f1-{{f1:.4f}}',
        auto_insert_metric_name=False,   
        every_n_epochs=1,
        save_top_k=1,
        mode="max",
        save_last=True
    )
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    
    trainer = pl.Trainer(
        check_val_every_n_epoch=5,
        max_epochs=config.training.n_epochs,
        accelerator='gpu',
        devices=[args.device],
        precision=32,
        logger=logger,
        strategy="auto",
        enable_progress_bar=True,
        log_every_n_steps=5,
        callbacks=[checkpoint_callback, lr_monitor_callback]
    ) 

    # Train
    trainer.fit(model, ckpt_path=args.resume)
    

if __name__ == '__main__':
    main()
