import os
import sys
import argparse
import yaml
import numpy as np
from easydict import EasyDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from pretraining.dcg import DCG as AuxCls
from model import ConditionalModel
from utils import get_dataset, cast_label_to_one_hot_and_prototype, compute_isic_metrics, set_random_seed
import pipeline


def parse_args():
    parser = argparse.ArgumentParser(description='DiffMIC-v2 Testing/Inference')
    parser.add_argument('--config', type=str, default='configs/aptos.yml', help='Path to config file')
    parser.add_argument('--device', type=int, default=0, help='GPU device id')
    parser.add_argument('--seed', type=int, default=2000, help='Random seed')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to diffusion model checkpoint')
    parser.add_argument('--dcg_ckpt', type=str, default=None, help='Path to DCG checkpoint')
    args = parser.parse_args()
    return args


def guided_prob_map(y0_g, y0_l, bz, nc, np, device):
    distance_to_diag = torch.tensor([[abs(i-j) for j in range(np)] for i in range(np)]).to(device)
    weight_g = 1 - distance_to_diag / (np - 1)
    weight_l = distance_to_diag / (np - 1)
    interpolated_value = weight_l.unsqueeze(0).unsqueeze(0) * y0_l.unsqueeze(-1).unsqueeze(-1) + weight_g.unsqueeze(0).unsqueeze(0) * y0_g.unsqueeze(-1).unsqueeze(-1)
    diag_indices = torch.arange(np)
    map = interpolated_value.clone()
    for i in range(bz):
        for j in range(nc):
            map[i, j, diag_indices, diag_indices] = y0_g[i, j]
            map[i, j, np-1, 0] = y0_l[i, j]
            map[i, j, 0, np-1] = y0_l[i, j]
    return map


def main():
    args = parse_args()

    # Load config
    with open(args.config, 'r') as f:
        params = yaml.safe_load(f)
    config = EasyDict(params)

    # Setup device
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set seed
    set_random_seed(args.seed)

    # Load dataset
    _, train_dataset, test_dataset = get_dataset(config)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.testing.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )

    # Initialize models
    model = ConditionalModel(config, guidance=config.diffusion.include_guidance).to(device)
    aux_model = AuxCls(config).to(device)

    # Load DCG checkpoint
    dcg_ckpt = args.dcg_ckpt if args.dcg_ckpt else f'pretraining/ckpt/{config.data.dataset.lower()}_aux_model.pth'
    if os.path.exists(dcg_ckpt):
        checkpoint = torch.load(dcg_ckpt, map_location=device)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint[0] if isinstance(checkpoint, list) else checkpoint
        aux_model.load_state_dict(state_dict, strict=False)
        print(f"Loaded DCG checkpoint from {dcg_ckpt}")
    else:
        print(f"Warning: DCG checkpoint not found at {dcg_ckpt}")

    # Load diffusion checkpoint
    if os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location=device)
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            # PyTorch Lightning checkpoint
            model.load_state_dict(ckpt['state_dict'], strict=False)
        elif isinstance(ckpt, list):
            model.load_state_dict(ckpt[0], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
        print(f"Loaded diffusion checkpoint from {args.ckpt}")
    else:
        print(f"Error: Diffusion checkpoint not found at {args.ckpt}")
        sys.exit(1)

    model.eval()
    aux_model.eval()

    # Setup DDIM sampler
    diff_config_path = 'option/diff_DDIM.yaml'
    with open(diff_config_path, 'r') as f:
        diff_params = yaml.safe_load(f)
    diff_opt = EasyDict(diff_params)
    
    sampler = pipeline.SR3Sampler(
        model=model,
        scheduler=pipeline.create_SR3scheduler(diff_opt['scheduler'], 'train'),
    )
    sampler.scheduler.set_timesteps(diff_opt['scheduler']['num_test_timesteps'])
    sampler.scheduler.diff_chns = config.data.num_classes

    # Inference loop
    all_preds = []
    all_gts = []

    pbar = tqdm(test_loader, desc='Testing')
    for x_batch, y_labels in pbar:
        x_batch = x_batch.to(device)
        y_labels = y_labels.to(device)
        y_batch, _ = cast_label_to_one_hot_and_prototype(y_labels, config)
        y_batch = y_batch.to(device)

        with torch.no_grad():
            y0_aux, y0_aux_global, y0_aux_local, patches, attns, attn_map = aux_model(x_batch)

            bz, nc, H, W = attn_map.size()
            bz, np = attns.size()

            y0_cond = guided_prob_map(y0_aux_global, y0_aux_local, bz, nc, np, device)
            yT = guided_prob_map(torch.rand_like(y0_aux_global), torch.rand_like(y0_aux_local), bz, nc, np, device)
            
            attns_mat = attns.unsqueeze(-1)
            attns_mat = (attns_mat * attns_mat.transpose(1, 2)).unsqueeze(1)
            
            y_pred = sampler.sample_high_res(x_batch, yT, conditions=[y0_cond, patches, attns_mat])
            y_pred = y_pred.reshape(bz, nc, np*np)
            y_pred = y_pred.mean(2)

            all_preds.append(y_pred)
            all_gts.append(y_batch)

    # Compute metrics
    all_preds = torch.cat(all_preds)
    all_gts = torch.cat(all_gts)
    
    ACC, BACC, Prec, Rec, F1, AUC_ovo, kappa = compute_isic_metrics(all_gts, all_preds)

    print("\n" + "="*60)
    print(f"Test Results for {config.data.dataset}")
    print("="*60)
    print(f"Accuracy:        {ACC:.4f}")
    print(f"Balanced Acc:    {BACC:.4f}")
    print(f"Precision:       {Prec:.4f}")
    print(f"Recall:          {Rec:.4f}")
    print(f"F1 Score:        {F1:.4f}")
    print(f"AUC (ovo):       {AUC_ovo:.4f}")
    print(f"Quadratic Kappa: {kappa:.4f}")
    print("="*60)


if __name__ == '__main__':
    main()
