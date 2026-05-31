from alphastats.descriptive import mean, std, pi, E, f_max, f_min
from alphastats.utils import all_check


def fit_distribution(data):
    pass


def cdf(x):

    mu = mean(x)
    sigma = std(x)

    return [0.5 * (1 + error_function((xi - mu) / (sigma * (2**0.5)))) for xi in x]


def pdf(x):

    all_check(x, pdf)

    mu = mean(x)
    sigma = std(x)

    return [
        (1 / (sigma * ((2 * pi) ** (1 / 2))))
        * (E ** ((-1 / 2) * (((xi - mu) / sigma) ** 2)))
        for xi in x
    ]


def error_function(z):

    # Constants required for the mathematical approximation

    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = 1 if z >= 0 else -1
    x = abs(z)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (E ** (-x * x))

    return sign * y


def ppf(x, target):
    all_check(x, ppf)

    x = sorted(x)

    mx = f_max(x)
    mn = f_min(x)

    calc_mean = mean(x)
    calc_std = std(x)

    while True:
        mid = (mn + mx) / 2

        current_prob = 0.5 * (
            1 + error_function((mid - calc_mean) / (calc_std * (2**0.5)))
        )

        if abs(target - current_prob) < 1e-6:
            return mid

        if target > current_prob:
            mn = mid
        else:
            mx = mid


def ks_test(x):

    all_check(x, ks_test)

    n = len(x)
    x_sorted = sorted(x)

    m_d = 0

    calc_mean = mean(x)
    calc_std = std(x)

    for index, i in enumerate(x_sorted):
        P_a = (index + i) / n
        P_e = 0.5 * (1 + error_function((i - calc_mean) / (calc_std * (2**0.5))))
        D_i = abs(P_a - P_e)
        if D_i > m_d:
            m_d = D_i

    return m_d


def anderson_darling():
    pass


def normality_test():
    pass
