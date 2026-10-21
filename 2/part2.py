
import numpy as np
import matplotlib.pyplot as plt

with open("data.txt", "r") as f:
    lines = f.readlines()[22:]

numbers = np.fromstring(" ".join(lines), sep=" ")
data = numbers.reshape(-1, 14)

X = data[:, :13]
y = data[:, 13]

def normalize_data(X_train, X_other):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1

    return (X_train - mean) / std, (X_other - mean) / std


def gaussian_basis(X, centers, sigma=1.0):
    # Squared Euclidean distance from every sample to every center
    distances = np.sum(
        (X[:, None, :] - centers[None, :, :]) ** 2,
        axis=2
    )
    return np.exp(-distances / (2 * sigma ** 2))


def gaussian_regression(X, y, sigma=1.0, seed=42):
    # First 100 samples for training; remaining samples for testing
    X_train_raw, y_train = X[:100], y[:100]
    X_test_raw, y_test = X[100:], y[100:]

    X_train, X_test = normalize_data(X_train_raw, X_test_raw)

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(X_train))

    basis_counts = np.arange(5, 100, 10)
    train_errors = []
    test_errors = []

    for K in basis_counts:
        # Randomly select K training samples as Gaussian centers
        centers = X_train[permutation[:K]]

        Phi_train = gaussian_basis(X_train, centers, sigma)
        Phi_test = gaussian_basis(X_test, centers, sigma)

        # No regularization
        w = np.linalg.lstsq(Phi_train, y_train, rcond=None)[0]

        train_pred = Phi_train @ w
        test_pred = Phi_test @ w

        train_rms = np.sqrt(np.mean((y_train - train_pred) ** 2))
        test_rms = np.sqrt(np.mean((y_test - test_pred) ** 2))

        train_errors.append(train_rms)
        test_errors.append(test_rms)

        print(
            f"K = {K:2d}: Training RMS = {train_rms:.4f}, "
            f"Testing RMS = {test_rms:.4f}"
        )

    plt.figure(figsize=(8, 5))
    plt.plot(basis_counts, train_errors, "o-", label="Training RMS")
    plt.plot(basis_counts, test_errors, "s-", label="Testing RMS")
    plt.xlabel("Number of Gaussian basis functions")
    plt.ylabel("RMS error")
    plt.title("Gaussian Basis Regression")
    plt.xticks(basis_counts)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fig/part2e_gaussian_errors.png",
                dpi=300, bbox_inches="tight")
    plt.close()

    return basis_counts, train_errors, test_errors

basis_counts, train_errors, test_errors = gaussian_regression(X, y)


def gaussian_regression_reg(X, y, sigma=1.0, seed=42):
    lambdas = np.array([0, 0.01, 0.1, 1, 10, 100, 1000])

    X = X[:100]
    y = y[:100]

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(y))
    folds = np.array_split(indices, 10)

    avg_errors = []

    for lam in lambdas:
        fold_errors = []

        for fold_idx in range(10):
            val_idx = folds[fold_idx]
            train_idx = np.concatenate([
                folds[j] for j in range(10) if j != fold_idx
            ])

            X_train_raw = X[train_idx]
            X_val_raw = X[val_idx]
            y_train = y[train_idx]
            y_val = y[val_idx]

            # Normalize using only this fold's training data
            X_train, X_val = normalize_data(
                X_train_raw, X_val_raw
            )

            # Use 90 training samples as Gaussian centers,
            # in randomized order
            center_order = rng.permutation(len(X_train))
            centers = X_train[center_order[:90]]

            Phi_train = gaussian_basis(X_train, centers, sigma)
            Phi_val = gaussian_basis(X_val, centers, sigma)

            # L2-regularized least squares
            if lam == 0:
                w = np.linalg.lstsq(
                    Phi_train, y_train, rcond=None
                )[0]
            else:
                A = Phi_train.T @ Phi_train
                b = Phi_train.T @ y_train
                w = np.linalg.solve(
                    A + lam * np.eye(A.shape[0]), b
                )

            pred = Phi_val @ w
            rms = np.sqrt(np.mean((y_val - pred) ** 2))
            fold_errors.append(rms)

        avg_error = np.mean(fold_errors)
        avg_errors.append(avg_error)

        print(
            f"Lambda = {lam:7g}, "
            f"Average validation RMS = {avg_error:.4f}"
        )

    avg_errors = np.array(avg_errors)

    # Plot positive lambdas on a logarithmic axis
    positive = lambdas > 0

    plt.figure(figsize=(8, 5))
    plt.semilogx(
        lambdas[positive],
        avg_errors[positive],
        "o-",
        label="10-fold CV error"
    )

    # Label each plotted point
    for lam, error in zip(lambdas[positive], avg_errors[positive]):
        plt.annotate(
            f"{error:.2f}",
            (lam, error),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center"
        )

    # Show the unregularized result as text, not a baseline line
    zero_error = avg_errors[lambdas == 0][0]
    plt.text(
        0.03, 0.97,
        f"Lambda = 0 error: {zero_error:.2f}",
        transform=plt.gca().transAxes,
        va="top"
    )

    plt.xlabel("Regularization constant (lambda)")
    plt.ylabel("Average validation RMS")
    plt.title("Gaussian Regression: 10-fold Cross-validation")
    plt.xticks(lambdas[positive], [str(v) for v in lambdas[positive]])
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig("fig/part2f_gaussian_regularization.png",
                dpi=300, bbox_inches="tight")
    plt.close()

    best_idx = np.argmin(avg_errors)
    best_lambda = lambdas[best_idx]

    print(f"\nBest lambda: {best_lambda}")
    print(f"Lowest average validation RMS: {avg_errors[best_idx]:.4f}")

    return best_lambda, lambdas, avg_errors


best_lambda, lambdas, cv_errors = gaussian_regression_reg(X, y)