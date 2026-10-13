
import numpy as np

from neural_network import NeuralNetwork


def main():
    np.random.seed(42)

    rng = np.random.default_rng(42)
    X = rng.random((5, 64))
    labels = np.array([0, 2, 4, 6, 8])
    y = np.eye(10)[labels]

    model = NeuralNetwork(hidden_width=16, seed=42)

    # Calculate analytical gradients.
    model.forward(X)
    analytical = model.backward(y)

    # Select 10 distinct weight elements.
    weight_names = ["W1", "W2", "W3"]
    candidates = []

    for name in weight_names:
        parameter = model.parameters()[name]
        for index in np.ndindex(parameter.shape):
            candidates.append((name, index))

    chosen_indices = rng.choice(
        len(candidates), size=10, replace=False
    )

    epsilon = 1e-5
    passed = 0

    print("Gradient check")
    print(f"Epsilon: {epsilon}")
    print("-" * 72)

    for number, candidate_index in enumerate(chosen_indices, start=1):
        name, index = candidates[candidate_index]
        parameter = model.parameters()[name]
        original_value = parameter[index]

        # Loss at theta + epsilon.
        parameter[index] = original_value + epsilon
        loss_plus = model.loss(X, y)

        # Loss at theta - epsilon.
        parameter[index] = original_value - epsilon
        loss_minus = model.loss(X, y)

        # Restore the original parameter.
        parameter[index] = original_value

        numerical = (loss_plus - loss_minus) / (2.0 * epsilon)
        analytical_value = analytical[name][index]

        denominator = max(
            1e-12,
            abs(numerical) + abs(analytical_value),
        )
        relative_error = abs(
            numerical - analytical_value
        ) / denominator

        is_pass = relative_error < 1e-6
        passed += int(is_pass)

        print(
            f"{number:2d}. {name}{index}: "
            f"analytical={analytical_value:.10e}, "
            f"numerical={numerical:.10e}, "
            f"relative_error={relative_error:.3e}, "
            f"{'PASS' if is_pass else 'FAIL'}"
        )

    print("-" * 72)
    print(f"Passed: {passed}/10")
    print(f"Overall result: {'PASS' if passed >= 9 else 'FAIL'}")

    if passed < 9:
        raise SystemExit(
            "Gradient check failed. Inspect backpropagation."
        )


if __name__ == "__main__":
    main()
