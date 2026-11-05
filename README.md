# Generative Models of Images

Assignment 2 for Generative AI & LLM (Fall 2026), IML Lab, ITU.

Implementation of a Denoising Diffusion Probabilistic Model (DDPM) with a cosine noise schedule, trained on the AFHQ cat subset to generate 32x32 cat faces, together with written solutions to the theory questions on CNNs, Transformers, GANs, VAEs and diffusion models.

## Repository structure

| File | Description |
|------|-------------|
| `diffusion.py` | DDPM implementation: cosine schedule, forward process, training loss, reverse process, sampling |
| `unet.py` | U-Net denoiser that predicts the noise added at each diffusion step |
| `trainer.py` | Training loop, periodic sampling, FID evaluation, forward/backward diffusion visualization |
| `utils.py` | Experiment setup helpers (model, diffusion, trainer construction) |
| `main.py` | CLI entry point for training and visualization |
| `test_diffusion.py` | Unit tests for the diffusion and U-Net implementations |
| `run_in_colab.ipynb` / `run_in_kaggle.ipynb` | Cloud training notebooks |
| `report/` | Theory solutions and Report.pdf |
| `plots/` | Training loss curves, FID curve, sample grids, diffusion process visualizations |
| `requirements.txt` | Python dependencies |

## Setup

```
pip install -r requirements.txt
```

## Dataset

The AFHQ (Animal Faces-HQ) cat subset. Place the training images under `data/train/cat/` and the validation images under `data/val/cat/`. The dataloader resizes and center-crops images to 32x32 and normalizes them to [-1, 1]. The `data/` directory is excluded from version control.

## Usage

Training:

```
python main.py --train_steps 1000 --save_and_sample_every 100 --data_path ./data/train/ --data_class cat
```

Training with FID evaluation every 100 steps:

```
python main.py --train_steps 1000 --save_and_sample_every 100 --fid --data_path ./data/train/ --data_class cat
```

Full run used for the final results:

```
python main.py --train_steps 10000 --save_and_sample_every 1000 --data_path ./data/train/ --data_class cat
```

Forward and backward diffusion visualization with a trained checkpoint:

```
python main.py --visualize --load_path ./results/model.pt --data_path ./data/train/ --data_class cat
```

Training progress is logged to Weights & Biases (project `DDPM_AFHQ`), including per-step loss, periodic sample grids and, when enabled, the FID score.

## Experiments and results

To be filled as experiments complete.

## Model checkpoints

To be added: Google Drive link to the trained checkpoints.

## AI tools disclosure

To be added per the assignment policy.
