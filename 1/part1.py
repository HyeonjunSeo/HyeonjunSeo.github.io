import math
import numpy as np
import matplotlib.pyplot as plt

def find_distance_newton(x0, y0, f, df, ddf, ft, initial_guess=0.0,
tolerance=1e-7, max_iter=100):
    x = initial_guess
    X = np.array([x])
    shortest_distance = abs(y0 - f(x, ft))
    for i in range(max_iter):
        # D'(x)
        D_prime = 2 * (x - x0) + 2 * (f(x, ft) - y0) * df(x, ft)
        # D''(x)
        D_double_prime = 2 + 2 * (df(x, ft)**2) + 2 * (f(x, ft) - y0) * ddf(x, ft)
        # Newton-Raphson update step
        next_x = x - D_prime / D_double_prime
        if abs(next_x - x) < tolerance:
            break
        x = next_x
        X = np.append(X, x)
        shortest_distance = ((x - x0)**2 + (f(x, ft) - y0)**2)**0.5
    return shortest_distance, x, i, X

def golden_section_search(x0, y0, f, a, b, ft, tolerance=1e-7):
    phi = (1 + math.sqrt(5)) / 2
    resphi = 2 - phi
    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)
    X = np.array([[a, b]])
    def dist_sq(x): return (x - x0)**2 + (f(x, ft) - y0)**2
    f_x1 = dist_sq(x1)
    f_x2 = dist_sq(x2)
    best_x = 99999999
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
        X = np.append(X, [[a, b]], axis = 0)
    return math.sqrt(dist_sq(best_x)), best_x, X

def f(x, ft):
    match ft:
        case 0:
            return x**2 +5
        case 1:
            return np.exp(x) + 2
        case 2:
            return np.sqrt(x + 11)
        case 3:
            return np.sin(x) + 5
        case 4:
            return np.tanh(x) + 5

def df(x, ft):
    match ft:
        case 0:
            return 2*x
        case 1:
            return np.exp(x)
        case 2:
            return 1 / (2 * np.sqrt(x + 11))
        case 3:
            return np.cos(x)
        case 4:
            return 1 / np.cosh(x)**2

def ddf(x, ft):
    match ft:
        case 0:
            return 2
        case 1:
            return np.exp(x)
        case 2:
            return -1 / (4 * (x + 11)**(3/2))
        case 3:
            return -np.sin(x)
        case 4:
            return -2 * np.tanh(x) / np.cosh(x)**2

def D_sq(x, x0, y0, f, ft):
    return (x - x0)**2 + (f(x, ft) - y0)**2

x0 = [0,-4,-8,2,6,0,0,0,1,2]
y0 = [0,0,0,0,0,-3,0,2,2,3]

a = -10
b = 10
f_type = [
    r"$f(x)=x^2+5$",
    r"$f(x)=e^x+2$",
    r"$f(x)=\sqrt{x+11}$",
    r"$f(x)=\sin(x)+5$",
    r"$f(x)=\tanh(x)+5$"
]
for ft in range(5):
    for i in range(len(x0)):
        shortest_distance, x, itr, X = find_distance_newton(x0[i],y0[i],f,df,ddf,ft)
        domain = np.arange(np.min(X)-0.3, np.max(X)+0.301, 0.001)
        plt.figure()
        plt.title(
            f"Newton-Raphson Method: Distance Minimization\n"
            f"Point ({x0[i]}, {y0[i]}) to {f_type[ft]}"
        )
        plt.xlabel("x pos in original func")
        plt.ylabel("Squared distance")
        plt.plot(domain, D_sq(domain, x0[i], y0[i], f,ft))
        plt.text(
            0.02, 0.95,
            r"$D^2(x)=(x-x_0)^2+(f(x)-y_0)^2$",
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top"
        )
        plt.scatter(X, D_sq(X, x0[i], y0[i], f,ft), color='red')
        for j in range(len(X)):
            # plt.text(x_coordinate, y_coordinate, text_string)
            plt.text(X[j], D_sq(X, x0[i], y0[i], f,ft)[j], j, fontsize=9)
        plt.savefig("fig/" +str(ft) + str(i) + 'fig1')
        plt.close()

        shortest_distance, x, X = golden_section_search(x0[i],y0[i],f,a,b,ft)
        domain = np.arange(a, b+0.001, 0.001)
        plt.figure()
        plt.title(
            f"Golden-Section Search: Distance Minimization\n"
            f"Point ({x0[i]}, {y0[i]}) to {f_type[ft]}"
        )
        plt.xlabel("x")
        plt.ylabel("Squared distance")
        plt.plot(domain, D_sq(domain, x0[i], y0[i], f, ft))
        plt.text(
            0.02, 0.95,
            r"$D^2(x)=(x-x_0)^2+(f(x)-y_0)^2$",
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top"
        )
        for j in range(len(X)):
            plt.axvspan(X[j][0], X[j][1], alpha = 0.3)
            # plt.text(x_coordinate, y_coordinate, text_string)
        plt.savefig("fig/" +str(ft) + str(i) + 'fig2')
        plt.close()