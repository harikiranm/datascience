from numpy.random import standard_normal, normal
from numpy import insert, arange, exp, cumprod, vstack, ones
from math import sqrt
import matplotlib.pyplot as plt


def standard_brownian_motion(start_p, sigma, dt, T):
    steps = int(T / dt)
    prices = standard_normal(size=steps) * sqrt(dt) * sigma
    prices = insert(prices, 0, start_p, axis=0).cumsum()
    return prices


def geometric_brownian_motion(start_p, mu, sigma, dt, T):
    n = int(T / dt)
    x = exp((mu - sigma ** 2 / 2) * dt + sigma * normal(0, sqrt(dt), size=n))
    x = insert(x, 0, 1, axis=0)
    prices = start_p * cumprod(x)
    return prices


if __name__ == '__main__':
    start_p = 100
    dt = 0.005
    T = 1

    sigma = 0.02
    mu = -0.02

    for _ in range(30):
        x = geometric_brownian_motion(start_p, mu, sigma, dt, T)
        plt.plot(x)
    plt.show()
