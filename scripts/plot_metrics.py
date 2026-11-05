import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def moving_average(values, window):
    if len(values) < window:
        return values
    out = []
    for i in range(len(values) - window + 1):
        out.append(sum(values[i : i + window]) / window)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="./results/metrics.csv")
    parser.add_argument("--out_dir", default="./plots")
    parser.add_argument("--smooth", type=int, default=50)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(args.metrics, newline="") as f:
        header = f.readline().strip().split(",")
        assert header == ["step", "loss", "fid"], f"unexpected header: {header}"
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 3:
                continue
            fid = float(parts[2]) if parts[2] else None
            rows.append((int(parts[0]), float(parts[1]), fid))

    steps = [r[0] for r in rows]
    losses = [r[1] for r in rows]

    plt.figure(figsize=(7, 4.5))
    plt.plot(steps, losses, alpha=0.3, label="raw")
    smoothed = moving_average(losses, args.smooth)
    plt.plot(steps[: len(smoothed)], smoothed, label=f"smoothed (window={args.smooth})")
    plt.xlabel("training step")
    plt.ylabel("L1 loss")
    plt.title("Diffusion training loss")
    plt.legend()
    plt.tight_layout()
    loss_path = out_dir / "training_loss.png"
    plt.savefig(loss_path, dpi=150)
    plt.close()
    print(f"saved {loss_path}")

    fid_rows = [(s, f) for s, _, f in rows if f is not None]
    if fid_rows:
        fsteps = [r[0] for r in fid_rows]
        fvals = [r[1] for r in fid_rows]
        plt.figure(figsize=(7, 4.5))
        plt.plot(fsteps, fvals, marker="o")
        plt.xlabel("training step")
        plt.ylabel("FID")
        plt.title("FID between generated samples and training data")
        plt.tight_layout()
        fid_path = out_dir / "fid_curve.png"
        plt.savefig(fid_path, dpi=150)
        plt.close()
        print(f"saved {fid_path} ({len(fid_rows)} points)")


if __name__ == "__main__":
    main()
