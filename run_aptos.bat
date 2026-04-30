@echo off
REM DiffMIC-v2 APTOS Training Pipeline
REM Stage 1: DCG Pretraining
REM Stage 2: Diffusion Model Training
REM Stage 3: Testing (optional)

set CONFIG=configs/aptos.yml
set DEVICE=0
set SEED=2000

echo ========================================
echo Stage 1: DCG Pretraining
echo ========================================
python pretraining/dcg_trainer.py --config %CONFIG% --device %DEVICE% --seed %SEED%

echo.
echo ========================================
echo Stage 2: Diffusion Model Training
echo ========================================
python diffuser_trainer.py --config %CONFIG% --device %DEVICE% --seed %SEED% --dcg_ckpt pretraining/ckpt/aptos_aux_model.pth

echo.
echo ========================================
echo Training Complete!
echo To test, run:
echo python test.py --config %CONFIG% --device %DEVICE% --ckpt ^<path_to_checkpoint^> --dcg_ckpt pretraining/ckpt/aptos_aux_model.pth
echo ========================================
