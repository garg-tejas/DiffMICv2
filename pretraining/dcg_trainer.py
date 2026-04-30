import os
import sys
import argparse
import yaml
import pickle
import random
import numpy as np
from easydict import EasyDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from pretraining.dcg import DCG
from utils import get_dataset, cast_label_to_one_hot_and_prototype, compute_isic_metrics, set_random_seed


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss for imbalanced classification.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, num_classes, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = torch.ones(num_classes, dtype=torch.float32)
    
    def to(self, device):
        self.alpha = self.alpha.to(device)
        return self
    
    def forward(self, inputs, targets):
        """
        inputs: [N, C] raw logits
        targets: [N, C] one-hot labels
        """
        # Compute log probabilities
        log_probs = nn.functional.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - probs) ** self.gamma
        
        # Apply alpha weights per class
        alpha = self.alpha.unsqueeze(0).expand_as(targets)
        
        # Focal loss element-wise
        loss = -alpha * focal_weight * log_probs * targets
        
        if self.reduction == 'mean':
            return loss.sum() / targets.sum()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss.sum(dim=1)


def parse_args():
    parser = argparse.ArgumentParser(description='DiffMIC-v2 DCG Pretraining')
    parser.add_argument('--config', type=str, default='configs/aptos.yml', help='Path to config file')
    parser.add_argument('--device', type=int, default=0, help='GPU device id')
    parser.add_argument('--seed', type=int, default=2000, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='pretraining/ckpt', help='Output directory for checkpoint')
    parser.add_argument('--exp_name', type=str, default='aptos_aux_model', help='Experiment name')
    parser.add_argument('--loss_type', type=str, default=None, choices=['ce', 'focal'], help='Loss type override (ce or focal). If not set, uses config.dcg_loss.loss_type')
    args = parser.parse_args()
    return args


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

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f'{args.exp_name}.pth')

    # Load dataset
    _, train_dataset, test_dataset = get_dataset(config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.testing.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )

    # Initialize DCG model
    model = DCG(config).to(device)
    print(f"DCG model initialized. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.aux_optim.lr,
        betas=(config.aux_optim.beta1, 0.999),
        weight_decay=config.aux_optim.weight_decay,
        eps=config.aux_optim.eps,
    )

    # Loss function
    loss_config = config.get('dcg_loss', {})
    loss_type = args.loss_type if args.loss_type is not None else loss_config.get('loss_type', 'ce')
    
    if loss_type == 'focal':
        gamma = loss_config.get('focal_gamma', 2.0)
        alpha = loss_config.get('class_weights', None)
        criterion = FocalLoss(
            num_classes=config.data.num_classes,
            gamma=gamma,
            alpha=alpha
        ).to(device)
        print(f"Using Focal Loss (gamma={gamma}, alpha={alpha})")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using CrossEntropy Loss")

    # Training loop
    n_epochs = config.diffusion.aux_cls.n_pretrain_epochs
    best_acc = 0.0

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs}')
        for x_batch, y_labels in pbar:
            x_batch = x_batch.to(device)
            y_labels = y_labels.to(device)

            # Convert to one-hot for loss computation
            y_one_hot, _ = cast_label_to_one_hot_and_prototype(y_labels, config)
            y_one_hot = y_one_hot.to(device)

            # Forward pass
            y_fusion, y_global, y_local, patches, patch_attns, saliency_map = model(x_batch)

            # Loss on fusion, global, and local predictions (as per paper)
            loss = criterion(y_fusion, y_one_hot) + criterion(y_global, y_one_hot) + criterion(y_local, y_one_hot)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.aux_optim.grad_clip)
            optimizer.step()

            # Metrics
            train_loss += loss.item() * x_batch.size(0)
            pred = y_fusion.argmax(dim=1)
            true = y_labels
            train_correct += (pred == true).sum().item()
            train_total += x_batch.size(0)

            pbar.set_postfix({'loss': loss.item(), 'acc': train_correct / train_total})

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        all_preds = []
        all_gts = []

        with torch.no_grad():
            for x_batch, y_labels in test_loader:
                x_batch = x_batch.to(device)
                y_labels = y_labels.to(device)
                y_one_hot, _ = cast_label_to_one_hot_and_prototype(y_labels, config)
                y_one_hot = y_one_hot.to(device)

                y_fusion, y_global, y_local, patches, patch_attns, saliency_map = model(x_batch)

                pred = y_fusion.argmax(dim=1)
                val_correct += (pred == y_labels).sum().item()
                val_total += x_batch.size(0)

                all_preds.append(y_fusion)
                all_gts.append(y_one_hot)

        val_acc = val_correct / val_total

        # Compute detailed metrics
        all_preds = torch.cat(all_preds)
        all_gts = torch.cat(all_gts)
        ACC, BACC, Prec, Rec, F1, AUC_ovo, kappa = compute_isic_metrics(all_gts, all_preds)

        print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | F1: {F1:.4f} | Kappa: {kappa:.4f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config,
            }
            torch.save(checkpoint, save_path)
            print(f"  -> Saved best model to {save_path} (val_acc={val_acc:.4f})")

    print(f"\nDCG pretraining complete. Best val accuracy: {best_acc:.4f}")
    print(f"Checkpoint saved to: {save_path}")


if __name__ == '__main__':
    main()
