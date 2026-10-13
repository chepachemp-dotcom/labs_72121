
import csv
import random
import statistics
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from neural_network import NeuralNetwork


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "lab02" / "results"

EPOCHS = 50
BATCH_SIZE = 32
WIDTHS = [16, 64]
LEARNING_RATES = [0.001, 0.01, 0.1]
SEED = 42


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)


def load_data():
    digits = load_digits()

    X = digits.data.astype(np.float64) / 16.0
    labels = digits.target
    y = np.eye(10, dtype=np.float64)[labels]

    X_train, X_test, y_train, y_test, labels_train, labels_test = (
        train_test_split(
            X,
            y,
            labels,
            test_size=0.30,
            stratify=labels,
            random_state=SEED,
        )
    )

    return X_train, X_test, y_train, y_test, labels_train, labels_test


def train_one_configuration(
    width,
    learning_rate,
    X_train,
    y_train,
    X_test,
    labels_test,
):
    model = NeuralNetwork(hidden_width=width, seed=SEED)
    rng = np.random.default_rng(SEED)

    epoch_losses = []

    for epoch in range(EPOCHS):
        # Shuffle training examples every epoch.
        order = rng.permutation(len(X_train))
        X_shuffled = X_train[order]
        y_shuffled = y_train[order]

        for start in range(0, len(X_train), BATCH_SIZE):
            end = start + BATCH_SIZE
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            model.forward(X_batch)
            gradients = model.backward(y_batch)
            model.update(gradients, learning_rate)

        # Measure loss over the complete training set.
        epoch_loss = model.loss(X_train, y_train)
        epoch_losses.append(epoch_loss)

        print(
            f"width={width:2d}, lr={learning_rate:g}, "
            f"epoch={epoch + 1:02d}/{EPOCHS}, "
            f"loss={epoch_loss:.6f}"
        )

    predictions = model.predict(X_test)
    accuracy = float(np.mean(predictions == labels_test))

    return model, epoch_losses, accuracy


def benchmark_model(model, X_train, y_train, repeats=100):
    """
    Benchmark forward and backward separately on a fixed batch.
    Each measurement follows one warm-up call.
    """
    X_batch = X_train[:BATCH_SIZE]
    y_batch = y_train[:BATCH_SIZE]

    # Warm up forward pass.
    model.forward(X_batch)

    forward_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.forward(X_batch)
        forward_times.append(time.perf_counter() - start)

    # Warm up backward pass, preceded by a forward pass.
    model.forward(X_batch)
    model.backward(y_batch)

    backward_times = []
    for _ in range(repeats):
        model.forward(X_batch)
        start = time.perf_counter()
        model.backward(y_batch)
        backward_times.append(time.perf_counter() - start)

    forward_ms = statistics.median(forward_times) * 1000
    backward_ms = statistics.median(backward_times) * 1000

    return forward_ms, backward_ms


def main():
    set_seeds()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Python: {sys.version.split()[0]}")
    print(f"NumPy: {np.__version__}")
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"Results directory: {RESULTS_DIR}")

    (
        X_train,
        X_test,
        y_train,
        y_test,
        labels_train,
        labels_test,
    ) = load_data()

    # Check the expected output shape and probability sums.
    check_model = NeuralNetwork(hidden_width=16, seed=SEED)
    probabilities = check_model.forward(X_train[:8])

    assert probabilities.shape == (8, 10)
    assert np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-6
    )

    initial_loss = check_model.loss(X_train[:32], y_train[:32])
    print(f"Initial loss for width 16: {initial_loss:.6f}")
    print(f"Expected reference: approximately {np.log(10):.6f}")

    all_results = []
    loss_curves = {}

    for width in WIDTHS:
        for learning_rate in LEARNING_RATES:
            print("\n" + "=" * 65)
            print(f"Training width={width}, learning_rate={learning_rate}")

            model, losses, accuracy = train_one_configuration(
                width,
                learning_rate,
                X_train,
                y_train,
                X_test,
                labels_test,
            )

            forward_ms, backward_ms = benchmark_model(
                model, X_train, y_train, repeats=100
            )

            loss_curves[(width, learning_rate)] = losses

            all_results.append({
                "width": width,
                "learning_rate": learning_rate,
                "final_loss": losses[-1],
                "test_accuracy": accuracy,
                "forward_ms": forward_ms,
                "backward_ms": backward_ms,
            })

            print(f"Test accuracy: {accuracy:.4f}")
            print(f"Forward median: {forward_ms:.6f} ms")
            print(f"Backward median: {backward_ms:.6f} ms")

    # Save experiment results.
    csv_path = RESULTS_DIR / "results.csv"
    fieldnames = [
        "width",
        "learning_rate",
        "final_loss",
        "test_accuracy",
        "forward_ms",
        "backward_ms",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    # Plot all six training-loss curves.
    plt.figure(figsize=(11, 6))

    for (width, learning_rate), losses in loss_curves.items():
        plt.plot(
            range(1, EPOCHS + 1),
            losses,
            label=f"W={width}, lr={learning_rate:g}",
        )

    plt.xlabel("Epoch")
    plt.ylabel("Training cross-entropy loss")
    plt.title("Lab 2: Training Loss for Six Configurations")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = RESULTS_DIR / "loss_curves.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print("\n" + "=" * 65)
    print("EXPERIMENT SUMMARY")
    for row in all_results:
        print(
            f"W={row['width']:2d}, "
            f"lr={row['learning_rate']:g}, "
            f"loss={row['final_loss']:.6f}, "
            f"accuracy={row['test_accuracy']:.4f}, "
            f"forward={row['forward_ms']:.4f} ms, "
            f"backward={row['backward_ms']:.4f} ms"
        )

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
