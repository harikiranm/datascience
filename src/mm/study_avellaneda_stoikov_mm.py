from itertools import product
from math import floor, ceil

import matplotlib.pyplot as plt
from numpy import arange, sqrt, insert, log, exp, zeros
from numpy.random import random, choice, standard_normal
from pandas import DataFrame, set_option
from src.mm.brownian_motion import geometric_brownian_motion, standard_brownian_motion

set_option('display.max_columns', 40)
set_option('display.width', 2000)
set_option('display.max_rows', 1000)


def asmm(*args):
    start_p, T, dt, mu, sigma, start_q, gamma, k, A, is_symmetric = args
    n = int(T / dt)
    t = arange(0, T + dt, dt)

    # prices = standard_normal(size=n) * sqrt(dt) * sigma
    # prices = choice([1, -1], size=n) * sqrt(dt) * sigma
    # prices = insert(prices, 0, start_p, axis=0).cumsum()
    # prices = standard_brownian_motion(start_p, sigma, dt, T)

    prices = geometric_brownian_motion(start_p, mu / start_p, sigma / start_p, dt, T)

    spread = gamma * (sigma ** 2) * (T - t) + (2 / gamma) * log(1 + (gamma / k))
    q = zeros(n + 1)
    q[0] = start_q
    cash = zeros(n + 1)
    res_price = zeros(n + 1)
    res_price[0] = prices[0]
    lambda_a = zeros(n + 1)
    lambda_b = zeros(n + 1)
    delta_a = zeros(n + 1)
    delta_b = zeros(n + 1)
    qte_a = zeros(n + 1)
    qte_b = zeros(n + 1)

    qte_a[0] = res_price[0] + spread[0] / 2
    qte_b[0] = res_price[0] - spread[0] / 2
    delta_a[0] = qte_a[0] - prices[0]
    delta_b[0] = prices[0] - qte_b[0]
    lambda_a[0] = A * exp(-k * delta_a[0])
    lambda_b[0] = A * exp(-k * delta_b[0])

    for i in range(1, n + 1):
        q[i] = q[i - 1]
        cash[i] = cash[i - 1]
        pa = random()
        if pa < lambda_a[i - 1] * dt:
            q[i] -= 1
            cash[i] += qte_a[i - 1]
        pb = random()
        if pb < lambda_b[i - 1] * dt:
            q[i] += 1
            cash[i] -= qte_b[i - 1]
        if is_symmetric:
            res_price[i] = prices[i]
        else:
            res_price[i] = prices[i] - q[i] * gamma * (sigma ** 2) * (T - t[i])

        qte_a[i] = res_price[i] + spread[i] / 2
        qte_b[i] = res_price[i] - spread[i] / 2
        delta_a[i] = qte_a[i] - prices[i]
        delta_b[i] = prices[i] - qte_b[i]
        lambda_a[i] = A * exp(-k * delta_a[i])
        lambda_b[i] = A * exp(-k * delta_b[i])
    pl = cash + q * prices
    return DataFrame(data={'time': t, 'prices': prices, 'spread': spread, 'q': q, 'cash': cash, 'ref_price': res_price,
                           'lambda_a': lambda_a, 'lambda_b': lambda_b, 'delta_a': delta_a, 'delta_b': delta_b,
                           'qte_a': qte_a, 'qte_b': qte_b, 'pl': pl})


def run_asmm(*args):
    df = asmm(*args)
    last = df.iloc[-1]
    return last.cash + last.prices * last.q, last.q, df.spread.mean()


def strategy_run(*args):
    res = []
    start_p, T, dt, mu, sigma, start_q, gamma, k, A = args
    for g, strt_type, run_no in product(gamma, ['Inventory', 'Symmetric'], range(1000)):
        df = asmm(start_p, T, dt, mu, sigma, start_q, g, k, A, strt_type == 'Symmetric')
        last = df.iloc[-1]
        res.append((run_no, g, strt_type, last.cash + last.prices * last.q, last.q, df.spread.mean()))
    df = DataFrame(data=res, columns=['run_no', 'gamma', 'strategy', 'profit', 'final_q', 'avg_spread'])
    return df


def plot_single_run(gamma, df):
    f = plt.figure(figsize=(18, 9))
    f.add_subplot(3, 1, 1)
    plt.title(f'gamma - {gamma}')
    plt.plot(df.time, df.prices, color='black', label='market price')
    plt.plot(df.time, df.ref_price, color='blue', linestyle='dashed', label='ref price')
    plt.plot(df.time, df.qte_a, color='red', linestyle='', marker='.', label='ask quote', markersize='4')
    plt.plot(df.time, df.qte_b, color='lime', linestyle='', marker='o', label='bid quote', markersize='2')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.grid(True)
    plt.legend()

    f.add_subplot(3, 1, 2)
    plt.plot(df.time, df.pl, color='black', label='P&L')
    plt.ylabel('PnL')
    plt.grid(True)
    plt.legend()

    f.add_subplot(3, 1, 3)
    plt.plot(df.time, df.q, color='black', label='position')
    plt.xlabel('Time')
    plt.ylabel('Inventory')
    plt.grid(True)
    plt.legend()
    plt.show(block=False)


def plot_pl_profiles(df):
    f = plt.figure(figsize=(18, 9))
    gammas = df.gamma.unique()
    n = len(gammas)
    for i in range(n):
        f.add_subplot(n, 1, i + 1)
        data = df[df.gamma == gammas[i]]
        dmin, dmax = floor(data.profit.min()), ceil(data.profit.max())
        plt.hist(data.loc[data.strategy == 'Inventory', 'profit'], bins=(dmax - dmin), range=(dmin, dmax),
                 edgecolor='black', linewidth=1, color='b', alpha=0.3, label='Inventory')
        plt.hist(data.loc[data.strategy == 'Symmetric', 'profit'], bins=(dmax - dmin), range=(dmin, dmax),
                 edgecolor='black', linewidth=1, color='w', alpha=0.2, label='Symmetric')

        plt.ylabel('count')
        plt.legend(title=f'gamma - {gamma[i]}')
    plt.xlabel('PnL')
    plt.show(block=False)


if __name__ == '__main__':
    start_p, T, dt, start_q, k, A = 100, 1, 0.005, 0, 1.5, 140
    sigma = 2
    mu = 1
    gamma = [0.1, 0.01, 0.5, 1]

    # df = asmm(start_p, T, dt, mu, sigma, start_q, gamma[0], k, A, False)
    # plot_single_run(gamma[0], df)

    # for i in range(len(gamma)):
    #     df = asmm(start_p, T, dt, sigma, start_q, gamma[i], k, A, False)
    #     plot_single_run(gamma, df)

    df = strategy_run(start_p, T, dt, mu, sigma, start_q, gamma, k, A)
    summary = df.groupby(['gamma', 'strategy']).agg(
        {'profit': ['mean', 'std'], 'final_q': ['mean', 'std'], 'avg_spread': ['mean']})
    print(summary)
    plot_pl_profiles(df)

    plt.show()
