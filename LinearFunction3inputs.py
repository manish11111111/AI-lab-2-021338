import numpy as np

# Seed for reproducibility
np.random.seed(42)


# Ground truth function
def true_function(x1, x2, x3):
    return 2 * x1 + 3 * x2 - x3 + 5


# Generate 10 random samples: x1, x2, x3 ∈ [0, 1]
X = np.random.rand(10, 3)
y_true = np.array([true_function(x[0], x[1], x[2]) for x in X])

# Print dataset
print("Dataset (x1, x2, x3, y):")
for i in range(10):
    print(f"{X[i][0]:.3f}\t{X[i][1]:.3f}\t{X[i][2]:.3f}\t{y_true[i]:.3f}")

# Initialize weights and bias randomly
w1, w2, w3 = np.random.randn(3)
b = np.random.randn()

# Learning rate and epochs
alpha = 0.01
epochs = 100

# Training loop
for epoch in range(1, epochs + 1):
    total_error = 0
    print(f"\nEpoch {epoch}: Weights => w1 = {w1:.4f}, w2 = {w2:.4f}, w3 = {w3:.4f}, bias = {b:.4f}")

    for i in range(len(X)):
        x1, x2, x3 = X[i]
        target = y_true[i]

        # Prediction (linear activation)
        y_pred = w1 * x1 + w2 * x2 + w3 * x3 + b
        error = target - y_pred
        total_error += error ** 2

        # Weight updates
        dw1 = alpha * error * x1
        dw2 = alpha * error * x2
        dw3 = alpha * error * x3
        db = alpha * error

        w1 += dw1
        w2 += dw2
        w3 += dw3
        b += db

        print(
            f"Sample {i}: x1={x1:.3f}, x2={x2:.3f}, x3={x3:.3f}, target={target:.3f}, pred={y_pred:.3f}, error={error:.3f}")
        print(f"         Updated weights => w1={w1:.4f}, w2={w2:.4f}, w3={w3:.4f}, bias={b:.4f}")

    mse = total_error / len(X)
    print(f"Epoch {epoch} MSE: {mse:.6f}")

    # Early stopping condition (optional)
    if mse < 1e-6:
        print("\nStopping early due to very low error.")
        break

# Final weights
print(f"\nFinal weights: w1 = {w1:.4f}, w2 = {w2:.4f}, w3 = {w3:.4f}, bias = {b:.4f}")
