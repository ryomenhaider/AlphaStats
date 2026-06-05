from alphastats.utils import all_check


def mean(x) -> float:

    all_check(x, mean)

    n = len(x)
    x_sum = sum(x)
    return x_sum / n


def median(x) -> float:
    all_check(x, median)
    n = len(x)
    sx = sorted(x)
    if n % 2 == 1:
        return sx[n // 2]
    return (sx[n // 2 - 1] + sx[n // 2]) / 2


def mode(x) -> float:

    all_check(x, mode)

    count: dict = {}
    for i in x:
        if i in count:
            count[i] += 1
        else:
            count[i] = 1
    max_count = max(count.values())
    for key, value in count.items():
        if value == max_count:
            return key
    return 0.0
