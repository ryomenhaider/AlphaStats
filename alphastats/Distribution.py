from alphastats.descriptive import mean, std, pi, E


def fit_distribution(data):
    pass


def cdf():
    pass


def pdf(x):  # uh?
    return (1 / (std(x) * ((2 * pi) ** (1 / 2)))) * E ** (
        (-1 / 2) * ((x - mean(x)) / std(x))
    )


def ppf():
    pass


def ks_test():
    pass


def anderson_darling():
    pass


def normality_test():
    pass
