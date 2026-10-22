import numpy as np
import matplotlib.pyplot as plt


with open("data.txt", "r") as f:
    lines = f.readlines()[22:]

numbers = np.fromstring(" ".join(lines), sep=" ")
data = numbers.reshape(-1, 14)

X = data[:, :13]
y = data[:, 13]

# Split: first 100 samples for training, remaining samples for testing
X_train = X[:100]
y_train = y[:100]

X_test = X[100:]
y_test = y[100:]


def make_basis(X, degree):
    Phi = [np.ones(X.shape[0])]

    for power in range(1, degree + 1):
        for j in range(X.shape[1]):
            Phi.append(X[:, j] ** power)

    return np.column_stack(Phi)

def polynomial_regression(X_train, y_train, X_test, degree):
    Phi_train = make_basis(X_train, degree)
    Phi_test = make_basis(X_test, degree)

    # Least-squares fitting, without regularization
    w = np.linalg.lstsq(Phi_train, y_train, rcond=None)[0]

    # Predictions
    y_train_pred = Phi_train @ w
    y_test_pred = Phi_test @ w

    # RMS errors
    train_rms = np.sqrt(np.mean((y_train - y_train_pred) ** 2))
    test_rms = np.sqrt(np.mean((y_test - y_test_pred) ** 2))

    return train_rms, test_rms


degrees = range(1, 8)
train_errors = []
test_errors = []

for degree in degrees:
    train_rms, test_rms = polynomial_regression(
        X_train, y_train, X_test, degree
    )
    print(degree)
    train_errors.append(train_rms)
    test_errors.append(test_rms)

    print(
        f"Degree {degree}: "
        f"Training RMS = {train_rms:.4f}, "
        f"Testing RMS = {test_rms:.4f}"
    )

# Plot
plt.figure(figsize=(8, 5))
plt.plot(list(degrees), train_errors, "o-", label="Training Error")
plt.plot(list(degrees), test_errors, "s-", label="Testing Error")

plt.xlabel("Polynomial Degree")
plt.ylabel("RMS Error")
plt.xticks(list(degrees))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("fig/part1a_rms_errors.png", dpi=300, bbox_inches="tight")
plt.close()

# Degree 1: linear regression
Phi_train = make_basis(X_train, 1)

w = np.linalg.lstsq(Phi_train, y_train, rcond=None)[0]

print("All weights, including bias:")
print(w)

print("Bias weight:", w[0])
print("Weight on 6th feature:", w[6])
print("Weight on 8th feature:", w[8])

def polynomial_regression_1d_vis(X_train, y_train, X_test, y_test,
                                 degree, feature_index=2):
    # Select one feature (RM is feature index 5)
    x_train = X_train[:, feature_index]
    x_test = X_test[:, feature_index]

    # Normalize using training-set mean and standard deviation
    x_mean = np.mean(x_train)
    x_std = np.std(x_train)

    x_train_norm = (x_train - x_mean) / x_std
    x_test_norm = (x_test - x_mean) / x_std

    # Create polynomial basis: 1, x, x^2, ..., x^degree
    Phi_train = np.column_stack(
        [x_train_norm ** p for p in range(degree + 1)]
    )
    Phi_test = np.column_stack(
        [x_test_norm ** p for p in range(degree + 1)]
    )

    # Fit weights using least squares
    w = np.linalg.lstsq(Phi_train, y_train, rcond=None)[0]

    # Generate smooth x values for plotting the polynomial
    x_plot = np.linspace(x_train.min(), x_train.max(), 300)
    x_plot_norm = (x_plot - x_mean) / x_std
    Phi_plot = np.column_stack(
        [x_plot_norm ** p for p in range(degree + 1)]
    )
    y_plot = Phi_plot @ w

    # Plot training data, fitted polynomial, and test data
    plt.figure(figsize=(8, 5))
    plt.scatter(x_train, y_train, label="Training data", alpha=0.7)
    plt.plot(x_plot, y_plot, label=f"Degree {degree} polynomial",
             linewidth=2)
    plt.scatter(x_test, y_test, label="Test data", alpha=0.7,
                marker="x")

    plt.xlabel("INDUS: Proportion of non-retail business acres per town")
    plt.ylabel("Median house value (MEDV)")
    plt.title(f"Polynomial Regression (Degree {degree})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"fig/polynomial_regression_degree_{degree}.png",
                dpi=300, bbox_inches="tight")
    plt.close()

    return w


w1 = polynomial_regression_1d_vis(
    X_train, y_train, X_test, y_test, degree=1
)

w3 = polynomial_regression_1d_vis(
    X_train, y_train, X_test, y_test, degree=3
)

w7 = polynomial_regression_1d_vis(
    X_train, y_train, X_test, y_test, degree=7
)



def polynomial_regression_reg(X, y, degree=8, feature_index=1,
                              lambdas=None, n_folds=10):
    if lambdas is None:
        lambdas = [0, 0.01, 0.1, 1, 10, 100, 1000]

    # Use only the first 100 data points
    X = X[:100, feature_index]
    y = y[:100]

    # Create 10 folds
    indices = np.arange(len(y))
    folds = np.array_split(indices, n_folds)

    avg_errors = []

    for lam in lambdas:
        fold_errors = []

        for i in range(n_folds):
            # Validation fold and remaining training data
            val_idx = folds[i]
            train_idx = np.concatenate(
                [folds[j] for j in range(n_folds) if j != i]
            )

            x_train = X[train_idx]
            x_val = X[val_idx]
            y_train = y[train_idx]
            y_val = y[val_idx]

            # Normalize using training-fold statistics only
            x_mean = np.mean(x_train)
            x_std = np.std(x_train)

            x_train = (x_train - x_mean) / x_std
            x_val = (x_val - x_mean) / x_std

            # Degree-8 polynomial basis
            Phi_train = np.column_stack(
                [x_train ** p for p in range(degree + 1)]
            )
            Phi_val = np.column_stack(
                [x_val ** p for p in range(degree + 1)]
            )

            # L2-regularized least squares
            # Do not penalize the bias weight
            if lam == 0:
                w = np.linalg.lstsq(
                    Phi_train, y_train, rcond=None
                )[0]
            else:
                penalty = np.eye(degree + 1)
                penalty[0, 0] = 0

                w = np.linalg.solve(
                    Phi_train.T @ Phi_train + lam * penalty,
                    Phi_train.T @ y_train
                )

            # Validation RMS error
            y_pred = Phi_val @ w
            rms = np.sqrt(np.mean((y_val - y_pred) ** 2))
            fold_errors.append(rms)

        avg_errors.append(np.mean(fold_errors))
        print(f"Lambda = {lam:7g}, Average validation RMS = "
              f"{avg_errors[-1]:.4f}")

      # Plot positive lambdas only
    lambdas = np.array(lambdas)
    avg_errors = np.array(avg_errors)
    positive = lambdas > 0

    plt.figure(figsize=(8, 5))
    plt.semilogx(
        lambdas[positive],
        avg_errors[positive],
        "o-",
        label="10-fold CV error"
    )

    # Annotate each point with its error value
    for lam, error in zip(lambdas[positive], avg_errors[positive]):
        plt.annotate(
            f"{error:.2f}",
            (lam, error),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center"
        )

    # Report unregularized error separately
    zero_idx = np.where(lambdas == 0)[0]
    if len(zero_idx) > 0:
        zero_error = avg_errors[zero_idx[0]]
        plt.text(
            0.03, 0.97,
            f"Lambda = 0 error: {zero_error:.2f}",
            transform=plt.gca().transAxes,
            va="top"
        )

    plt.xlabel("Regularization constant (lambda)")
    plt.ylabel("Average validation RMS")
    plt.title("10-fold Cross-validation: Degree-8 Polynomial")
    plt.xticks(lambdas[positive], [str(v) for v in lambdas[positive]])
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig("fig/part1d_regularization_cv.png",
                dpi=300, bbox_inches="tight")
    plt.close()

    best_idx = np.argmin(avg_errors)
    best_lambda = lambdas[best_idx]

    print(f"\nBest lambda: {best_lambda}")
    print(f"Lowest average validation RMS: {avg_errors[best_idx]:.4f}")

    return best_lambda, lambdas, avg_errors


best_lambda, lambdas, cv_errors = polynomial_regression_reg(
    X_train, y_train
)