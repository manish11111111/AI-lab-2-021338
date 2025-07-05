import numpy as np


def generate_dataset(n_features, n_samples=10):
    np.random.seed(42)
    X = np.random.rand(n_samples, n_features)  # Inputs in [0, 1]
    true_weights = np.random.uniform(-1, 1, size=n_features)  # True weights ∈ [-1, 1]
    bias = 5
    y = np.dot(X, true_weights) + bias
    return X, y, true_weights, bias


def train_perceptron(X, y, learning_rate=0.01, epochs=100):
    n_samples, n_features = X.shape
    weights = np.random.randn(n_features)
    bias = np.random.randn()

    for epoch in range(1, epochs + 1):
        total_error = 0
        for i in range(n_samples):
            x_i = X[i]
            y_pred = np.dot(weights, x_i) + bias
            error = y[i] - y_pred
            total_error += error ** 2

            # Update weights and bias
            weights += learning_rate * error * x_i
            bias += learning_rate * error

        mse = total_error / n_samples
        print(f"Epoch {epoch:3d} | MSE: {mse:.6f}")

    return weights, bias


def run_test(n_features):
    print(f"\n--- Training Perceptron with n = {n_features} Features ---")
    X, y, true_w, true_b = generate_dataset(n_features)
    print(f"True weights: {true_w}, True bias: {true_b}")
    weights, bias = train_perceptron(X, y)
    print(f"\nLearned Weights: {weights}")
    print(f"Learned Bias: {bias:.4f}")


# Run for n = 4 and n = 5
run_test(n_features=4)
run_test(n_features=5)
