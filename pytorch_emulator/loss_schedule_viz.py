import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def simulate_accuracy(epochs: int) -> np.ndarray:
    """Generate a synthetic classification accuracy curve that rapidly saturates."""
    epoch_idx = np.arange(epochs)

    # Start around 60% accuracy, saturate to 99.8% with a steep logistic ascent
    floor = 0.60
    ceiling = 0.998
    steepness = 0.25
    midpoint = epochs * 0.15

    logits = 1 / (1 + np.exp(-steepness * (epoch_idx - midpoint)))
    noise = np.random.default_rng(42).normal(scale=0.002, size=epochs)
    acc = floor + (ceiling - floor) * logits + noise
    return np.clip(acc, 0.0, 0.9995)


def compute_schedule(
    epochs: int,
    beta: float = 0.9,
    gamma: float = 3.0,
    epsilon_gate: float = 1e-4,
    acc_threshold: float = 0.995,
    temperature: float = 0.002,
    alpha_min: float = 1e-3,
    alpha_max: float = 0.999,
    latch_patience: int = 3,
):
    """Compute g_acc, g_epoch, and alpha_t across training epochs."""
    accuracies = simulate_accuracy(epochs)

    acc_ema = accuracies[0]
    ema_history = [acc_ema]
    g_acc_history = []
    g_epoch_history = []
    alpha_history = []

    consecutive_hits = 0
    latched = False

    for epoch in range(epochs):
        if epoch > 0:
            acc_ema = beta * acc_ema + (1.0 - beta) * accuracies[epoch]
        ema_history.append(acc_ema)

        g_acc = 1 / (1 + math.exp((acc_threshold - acc_ema) / temperature))
        g_epoch = max(0.0, 1.0 - epoch / (epochs - 1))
        g_epoch = g_epoch**gamma

        if latched:
            alpha_t = alpha_min
        else:
            raw_gate = np.clip(g_epoch * g_acc + epsilon_gate, 0.0, 1.0)
            alpha_t = alpha_min + (alpha_max - alpha_min) * raw_gate

            if acc_ema >= acc_threshold:
                consecutive_hits += 1
                if consecutive_hits >= latch_patience:
                    latched = True
                    alpha_t = alpha_min
            else:
                consecutive_hits = 0

        g_acc_history.append(g_acc)
        g_epoch_history.append(g_epoch)
        alpha_history.append(alpha_t)

    return {
        "epoch": np.arange(epochs),
        "accuracy": accuracies,
        "acc_ema": np.array(ema_history[:-1]),
        "g_acc": np.array(g_acc_history),
        "g_epoch": np.array(g_epoch_history),
        "alpha": np.array(alpha_history),
    }


def plot_schedule(data, output_path: Path):
    """Plot g_acc, g_epoch, and alpha_t on a shared figure."""
    epoch = data["epoch"]

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    axes[0].plot(epoch, data["accuracy"], label="Batch accuracy", alpha=0.4)
    axes[0].plot(epoch, data["acc_ema"], label="EMA accuracy", linewidth=2)
    axes[0].axhline(0.995, color="red", linestyle="--", label="Accuracy threshold")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0.6, 1.0)
    axes[0].legend(loc="lower right")
    axes[0].set_title("Synthetic Accuracy Trajectory")

    axes[1].plot(epoch, data["g_epoch"], label="g_epoch", color="tab:orange")
    axes[1].set_ylabel("g_epoch")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(loc="upper right")
    axes[1].set_title("Epoch Gate")

    axes[2].plot(epoch, data["g_acc"], label="g_acc", color="tab:green")
    axes[2].plot(epoch, data["alpha"], label="alpha_t", color="tab:purple")
    axes[2].set_ylabel("Value")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].legend(loc="upper right")
    axes[2].set_title("Accuracy Gate and Final Alpha")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    epochs = 120
    schedule = compute_schedule(epochs=epochs)

    out_dir = Path("plots") / "loss_schedule"
    plot_schedule(schedule, out_dir / "schedule.png")
    print(f"Saved visualization to {out_dir / 'schedule.png'}")


if __name__ == "__main__":
    main()

