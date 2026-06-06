from statitude.descriptive.central import mean
from statitude.utils import compare
import builtins


def covariance(x, y, type: str = "population") -> float:

    compare(x, y)

    mean_x = mean(x)
    mean_y = mean(y)
    n = len(x)

    if type.lower() == "population":
        _cov_p = 0
        for i, j in zip(x, y):
            _ = (i - mean_x) * (j - mean_y)
            _cov_p += _
        cov_p = (1 / n) * _cov_p
        return cov_p

    elif type.lower() == "sample":
        _cov_s = 0
        for i, j in zip(x, y):
            _ = (i - mean_x) * (j - mean_y)
            _cov_s += _
        cov_s = (1 / (n - 1)) * _cov_s
        return cov_s
    else:
        raise ValueError(f"unknown type: {type}")


# used in spearman corelation
def rank(data) -> list[float]:

    indexed = sorted(enumerate(data), key=lambda x: x[1])

    ranks = [0.0] * len(data)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) - 1 and indexed[j][1] == indexed[j + 1][1]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in builtins.range(i, j + 1):
            ranks[indexed[k][0]] = avg
        i = j + 1

    return ranks


def corelation(x, y, type: str = "pearson") -> float:

    compare(x, y)

    if type.lower() == "pearson":
        mean_x = mean(x)
        mean_y = mean(y)
        n = len(x)

        _nomin = 0
        for i, j in zip(x, y):
            _nomin += (i - mean_x) * (j - mean_y)

        _denom1 = sum((i - mean_x) ** 2 for i in x)
        _denom2 = sum((i - mean_y) ** 2 for i in y)

        denominator = (_denom1 * _denom2) ** (1 / 2)

        return _nomin / denominator

    if type.lower() == "spearman":
        rx = rank(x)
        ry = rank(y)

        d2 = sum((ri - rj) ** 2 for ri, rj in zip(rx, ry))
        n = len(x)

        nomi = 6 * d2
        denom = n * ((n**2) - 1)

        return 1 - (nomi / denom)

    raise ValueError(f"unknown type: {type}")
