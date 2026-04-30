# DiffMIC-v2 APTOS-2019 Implementation Notes

## Summary of Changes

This document summarizes all changes made to the DiffMIC-v2 codebase to support APTOS-2019 training and evaluation with the exact paper architecture and hyperparameters.

---

## Files Created

### 1. `configs/aptos.yml`
Complete configuration for APTOS-2019 with paper hyperparameters:
- **Classes**: 5 (DR grades 0-4)
- **ROI patches**: 6 (K=6), 32×32 each
- **Image size**: 224×224
- **DCG pretraining**: 100 epochs, batch_size=64, Adam lr=2e-4
- **Diffusion training**: 1000 epochs, batch_size=64, Adam lr=1e-3 → cosine decay to 1e-5
- **Diffusion steps**: T=1000 training, 10 DDIM inference steps
- **Feature dim**: 6144 (H=6144, optimal per paper Table V)
- **Backbone**: EfficientSAM-vits (global), ResNet18 (local)

### 2. `pretraining/dcg_trainer.py`
Standalone DCG pretraining script:
- Loads config via CLI `--config`
- Trains DCG with CE loss on fusion + global + local predictions
- Validates after each epoch and saves best model
- Outputs: `pretraining/ckpt/aptos_aux_model.pth`

### 3. `test.py`
Standalone inference/evaluation script:
- Loads trained diffusion checkpoint and DCG checkpoint
- Runs DDIM sampling with 10 steps
- Computes Accuracy, F1, Precision, Recall, AUC, Quadratic Kappa

### 4. `run_aptos.bat`
Windows batch script to run the full pipeline sequentially.

---

## Files Modified

### 5. `diffuser_trainer.py` (Major Refactor)
**Before**: Hardcoded to placental dataset, GPU 6, no CLI args.
**After**: Fully dataset-agnostic with CLI arguments:
- `--config`: Config file path
- `--device`: GPU device ID
- `--seed`: Random seed
- `--resume`: Resume from checkpoint
- `--dcg_ckpt`: Path to pretrained DCG
- `--exp_name`: Experiment name
- `--output_dir`: Logging directory

Key fixes:
- Logger name derived from dataset config
- DCG checkpoint path configurable
- Removed all hardcoded placental references
- Uses `.to(self.device)` instead of `.cuda()` for flexibility

### 6. `model.py`
**Fixes**:
- Removed orphaned `DenoiseUNet` class (dead code not matching paper)
- Made `cond_weight` configurable via `config.model.num_k` instead of hardcoded `7`
- Added missing `densenet121` import

### 7. `pipeline.py`
**Fix**: `create_SR3scheduler()` now always uses `num_train_timesteps` (1000) for the underlying schedule, regardless of phase. This is critical for DDIM inference: the scheduler must know the full training schedule to correctly compute alphas for the 10-step inference subset.

### 8. `pretraining/tools.py`
**Fix**: `generate_mask_uplft()` GPU device handling:
```python
# Before (BUG):
device = gpu_number  # int
mask_x = mask_x.cuda().to(device)  # crashes

# After (FIX):
device = torch.device("cuda:{}".format(gpu_number))
mask_x = mask_x.to(device)
```

### 9. `pretraining/dcg.py`
**Fixes**:
- Made `gpu_number` configurable from config instead of hardcoded `6`
- `_retrieve_crop()` now uses `x_original_pytorch.device` instead of hardcoded `cuda:6`

### 10. `utils.py`
**Fix**: `compute_isic_metrics()` AUC computation:
```python
# Before:
AUC_ovo = metrics.roc_auc_score(gt_np, pred_np)  # fails for multi-class

# After:
AUC_ovo = metrics.roc_auc_score(gt_np, pred_np, average='macro', multi_class='ovo')
```

---

## Dataset Setup

### 11. `dataset/aptos/`
Created directory and copied pickles from DiffMIC v1:
- `aptos_train.pkl` (2564 samples)
- `aptos_test.pkl` (1098 samples)

**You need to place the actual images** at:
```
dataset/aptos/train/*.png
```

The pickle files contain paths like `./dataset/aptos/train/0981195eb9fb.png`. Both train and test pickles reference images from the `train/` folder.

---

## How to Run

### Prerequisites
1. Install dependencies from `requirements.txt`
2. Clone EfficientSAM repo at repo root:
   ```bash
   git clone https://github.com/yformer/EfficientSAM.git
   ```
3. Place APTOS-2019 images in `dataset/aptos/train/`

### Stage 1: DCG Pretraining
```bash
python pretraining/dcg_trainer.py --config configs/aptos.yml --device 0
```
Output: `pretraining/ckpt/aptos_aux_model.pth`

### Stage 2: Diffusion Training
```bash
python diffuser_trainer.py --config configs/aptos.yml --device 0 --dcg_ckpt pretraining/ckpt/aptos_aux_model.pth
```
Output: Checkpoints and logs in `logs/aptos/`

### Stage 3: Testing
```bash
python test.py --config configs/aptos.yml --device 0 --ckpt logs/aptos/version_X/checkpoints/last.ckpt --dcg_ckpt pretraining/ckpt/aptos_aux_model.pth
```

### Full Pipeline (Windows)
```bash
run_aptos.bat
```

---

## Expected Results (from Paper)

For APTOS-2019, DiffMIC-v2 paper reports:
- **Accuracy**: 0.871
- **F1-score**: 0.721

---

## Known Limitations / Notes

1. **ResNet18 pretrained weights**: The diffusion model's local encoder (`ResNetEncoder` in `model.py`) does NOT use `pretrained=True` by default, matching the original v2 codebase. If you want to experiment with pretrained weights, modify line 131 in `model.py`.

2. **Batch size**: Set to 64 per paper. With 24GB VRAM (L4), this should fit. If you encounter OOM, reduce to 32 in `configs/aptos.yml`.

3. **DCG backbone**: Uses `resnet18(pretrained=True)` for the global network, matching the paper's specification for the first three datasets.

4. **Feature dimension**: Set to 6144 (H=6144) which the paper identifies as optimal in Table V.

5. **Training time**: DCG pretraining ~30-60 min. Diffusion training ~several hours to days depending on GPU.
