from alphastats.descriptive.central import mean
from alphastats.utils import all_check


def variance(x) -> float:
    all_check(x, variance)
    mean_val = mean(x)
    n = len(x)
    _var = 0
    for i in x:
        _var += (i - mean_val) ** 2
    return (1 / n) * _var


def std(x) -> float:

    var = variance(x)
    std = var ** (1 / 2)
    return std


def count(x) -> float:

    all_check(x, count)

    return len(x)


def f_min(x) -> float:
    all_check(x, f_min)

    return min(x)


def f_max(x) -> float:
    all_check(x, f_max)

    return max(x)


def range(x) -> float:
    all_check(x, range)
    res = f_max(x) - f_min(x)
    return res


def quantile(data, p, method="linear"):

    all_check(data, quantile)

    x = sorted(data)
    n = len(x)

    if n == 0:
        raise ValueError("empty data")
    if not 0 <= p <= 1:
        raise ValueError("p must be in [0, 1]")

    if method == "nearest":
        idx = (int(p * n) - 1) + 1
        return x[max(0, min(idx, n - 1))]

    elif method == "linear":
        h = p * (n - 1)
    elif method == "hazen":
        h = p * n + 0.5 - 1
    elif method == "weibull":
        h = p * (n + 1) - 1
    elif method == "median_unbiased":
        h = p * (n + 1 / 3) + 1 / 3 - 1
    elif method == "normal_unbiased":
        h = p * (n + 0.25) + 0.375 - 1
    else:
        raise ValueError(f"unknown method: {method}")

    h = max(0, min(h, n - 1))

    lo = int(h)
    hi = int(h) + 1

    frac = h - lo

    if lo == hi or hi >= n:
        return x[lo]

    return x[lo] * (1 - frac) + x[hi] * frac


def quantiles(data, ps, method="linear"):
    return [quantile(data, p, method) for p in ps]


def iqr(data, method="linear"):
    return quantile(data, 0.75, method) - quantile(data, 0.25, method)


def percentile(data, k, method="linear"):
    return quantile(data, k / 100, method)


def ln(x):
    if x <= 0:
        raise ValueError("Math Domain Error")
    n = 1000
    return n * ((x ** (1 / n)) - 1)


pi = 22 / 7

E = 2.718281828459045
