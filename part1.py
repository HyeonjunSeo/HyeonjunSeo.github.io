import math
import numpy as np
import matplotlib.pyplot as plt

def find_distance_newton(x0, y0, f, df, ddf, initial_guess=0.0,
tolerance=1e-7, max_iter=100):
    x = initial_guess
    X = np.array([x])
    shortest_distance = abs(y0 - f(x))
    for i in range(max_iter):
        # D'(x)
        D_prime = 2 * (x - x0) + 2 * (f(x) - y0) * df(x)
        # D''(x)
        D_double_prime = 2 + 2 * (df(x)**2) + 2 * (f(x) - y0) * ddf(x)
        # Newton-Raphson update step
        next_x = x - D_prime / D_double_prime
        if abs(next_x - x) < tolerance:
            break
        x = next_x
        X = np.append(X, x)
        shortest_distance = ((x - x0)**2 + (f(x) - y0)**2)**0.5
    return shortest_distance, x, i, X

def golden_section_search(x0, y0, f, a, b, tolerance=1e-7):
    phi = (1 + math.sqrt(5)) / 2
    resphi = 2 - phi
    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)
    def dist_sq(x): return (x - x0)**2 + (f(x) - y0)**2
    f_x1 = dist_sq(x1)
    f_x2 = dist_sq(x2)
    while abs(b - a) > tolerance:
        if f_x1 < f_x2:
            b = x2
            x2 = x1
            f_x2 = f_x1
            x1 = a + resphi * (b - a)
            f_x1 = dist_sq(x1)
        else:
            a = x1
            x1 = x2
            f_x1 = f_x2
            x2 = b - resphi * (b - a)
            f_x2 = dist_sq(x2)
            best_x = (a + b) / 2
    return math.sqrt(dist_sq(best_x)), best_x

def f(x):
    return x**2 +5

def df(x):
    return 2*x

def ddf(x):
    return 2

def D_sq(x, x0, y0):
    return (x - x0)**2 + (f(x) - y0)**2

x0 = 6
y0 = 0
shortest_distance, x, i, X = find_distance_newton(x0,y0,f,df,ddf)
print(shortest_distance, x, i, X)

domain = np.arange(min(X), max(X)+0.001, 0.001)

print(X)
plt.figure()
plt.plot(domain, D_sq(domain, x0, y0))
plt.scatter(X, D_sq(X, x0, y0), color='red')
for i in range(len(X)):
    # plt.text(x_coordinate, y_coordinate, text_string)
    plt.text(X[i], D_sq(X, x0, y0)[i], i, fontsize=9)
plt.show()