import numpy as np
import matplotlib.pyplot as plt


def mse_line(x, y, w):
    y_hat = w[0] * x + w[1]
    return np.mean((y - y_hat) ** 2)


def mse_parabola(x, y, w):
    y_hat = w[0] * x**2 + w[1] * x + w[2]
    return np.mean((y - y_hat) ** 2)


x = np.array([0, 2, 1, 3])
y = np.array([0.5, 3.5, 1.5, 7.5])
tolerance = 0.001

n = len(x)

def line_gradient_a(a, b):
    residual = y - (a*x + b)
    return -(2/n) * np.sum(x * residual)


def line_gradient_b(a, b):
    residual = y - (a*x + b)
    return -(2/n) * np.sum(residual)


def line_hessian_a():
    return (2/n) * np.sum(x**2)


def line_hessian_b():
    return 2

a_line = 0.0
b_line = 0.0

for i in range(100):

    # Update a first
    ga = line_gradient_a(a_line, b_line)
    Ha = line_hessian_a()
    prev_a = a_line
    a_line = a_line - ga / Ha

    # Update b using NEW a
    gb = line_gradient_b(a_line, b_line)
    Hb = line_hessian_b()
    prev_b = b_line
    b_line = b_line - gb / Hb

    y_hat = a_line*x + b_line
    mse = np.mean((y - y_hat)**2)

    print(
        f"Line Iteration {i}: "
        f"a = {a_line:.6f}, "
        f"b = {b_line:.6f}, "
        f"MSE = {mse:.6f}"
    )
    if abs(prev_a - a_line) < tolerance and abs(prev_b - b_line) < tolerance:
        break

def parabola_gradient_a(a, b, c):
    residual = y - (a*x**2 + b*x + c)
    return -(2/n) * np.sum(x**2 * residual)


def parabola_gradient_b(a, b, c):
    residual = y - (a*x**2 + b*x + c)
    return -(2/n) * np.sum(x * residual)


def parabola_gradient_c(a, b, c):
    residual = y - (a*x**2 + b*x + c)
    return -(2/n) * np.sum(residual)


def parabola_hessian_a():
    return (2/n) * np.sum(x**4)


def parabola_hessian_b():
    return (2/n) * np.sum(x**2)


def parabola_hessian_c():
    return 2

a_para = 0.0
b_para = 0.0
c_para = 0.0

for i in range(100):

    # Update a first
    ga = parabola_gradient_a(a_para, b_para, c_para)
    Ha = parabola_hessian_a()
    prev_a = a_para
    a_para = a_para - ga / Ha

    # Update b using NEW a
    gb = parabola_gradient_b(a_para, b_para, c_para)
    Hb = parabola_hessian_b()
    prev_b = b_para
    b_para = b_para - gb / Hb

    # Update c using NEW a and b
    gc = parabola_gradient_c(a_para, b_para, c_para)
    Hc = parabola_hessian_c()
    prev_c = c_para
    c_para = c_para - gc / Hc

    y_hat = a_para*x**2 + b_para*x + c_para
    mse = np.mean((y - y_hat)**2)

    print(
        f"Parabola Iteration {i}: "
        f"a = {a_para:.6f}, "
        f"b = {b_para:.6f}, "
        f"c = {c_para:.6f}, "
        f"MSE = {mse:.6f}"
    )
    if abs(prev_a - a_para) < tolerance and abs(prev_b - b_para) < tolerance and abs(prev_c - c_para) < tolerance:
        break

x_plot = np.linspace(min(x), max(x), 200)

y_line = a_line*x_plot + b_line
y_para = a_para*x_plot**2 + b_para*x_plot + c_para

plt.scatter(x, y, label="Data")
plt.title("Newton's Method: Line and Parabola Fit")
plt.plot(x_plot, y_line, label=f"Line: y = {a_line:.2f}x + {b_line:.2f}")
plt.plot(x_plot, y_para, label=f"Parabola: y = {a_para:.2f}x² + {b_para:.2f}x + {c_para:.2f}")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.savefig("fig/part2")
plt.show()