from statitude.descriptive.central import mean
from statitude.utils import all_check


def Skewness(x) -> float:

    all_check(x, Skewness)

    mean_x = mean(x)
    n = len(x)

    _n = 0
    for i in x:
        _n_ = (i - mean_x) ** 3
        _n += _n_

    nominator = (1 / n) * _n

    _d = 0
    for i in x:
        _d_ = (i - mean_x) ** 2
        _d += _d_

    dominator = ((1 / n) * _d) ** (3 / 2)

    return nominator / dominator


def Kurtosis(x) -> float:

    all_check(x, Kurtosis)

    mean_x = mean(x)
    n = len(x)

    _n = 0
    for i in x:
        _n_ = (i - mean_x) ** 4
        _n += _n_
    nominator = (1 / n) * _n

    _d = 0
    for i in x:
        _d_ = (i - mean_x) ** 2
        _d += _d_
    dominator = ((1 / n) * _d) ** 2

    return (nominator / dominator) - 3
