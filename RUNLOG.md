# Experiment Run Log

All runs use the AFHQ cat subset (5153 train / 500 val images), image size 32, T=50 diffusion steps, U-Net dim 16 with mults (1, 2, 4, 8), batch size 32, Adam lr 1e-3, gradient accumulation 2. Metrics are logged to Weights & Biases (project `DDPM_AFHQ`) and to `results/metrics.csv`, from which the plots in `plots/` are generated with `scripts/plot_metrics.py`.

| Experiment | Assignment item | Command | Status | Outputs |
|---|---|---|---|---|
| smoke | pipeline validation | `python main.py --train_steps 15 --save_and_sample_every 1000 --save_folder ./results_smoke --data_path ./data/train/ --data_class cat --dataloader_workers 4` | done | loss decreases, final sample grid and `model.pt` written |
| exp_1k_loss | Q6.1 | `python main.py --train_steps 1000 --save_and_sample_every 100 --save_folder ./results_61 --data_path ./data/train/ --data_class cat --dataloader_workers 0` | done | loss 0.85 to ~0.15 over 1000 steps, sample grids in `results_61/`, final grid copied to `plots/sample_1k.png`, curve in `plots/training_loss.png` |
| exp_1k_fid | Q6.2 | `python main.py --train_steps 1000 --save_and_sample_every 100 --fid --save_folder ./results_62 --data_path ./data/train/ --data_class cat --dataloader_workers 0` | queued | `plots/fid_curve.png` |
| exp_10k | Q6.3 | `python main.py --train_steps 10000 --save_and_sample_every 1000 --save_folder ./results_63 --data_path ./data/train/ --data_class cat --dataloader_workers 0` | running | final sample batch `plots/sample_10k.png`, checkpoint on Google Drive |
| vis_10k | Q6.4, Q6.5 | `python main.py --visualize --train_steps 10000 --load_path ./results_63/model.pt --save_folder ./results_vis --data_path ./data/train/ --data_class cat --dataloader_workers 0` | queued | `plots/forward_diffusion.png`, `plots/backward_diffusion.png` |
