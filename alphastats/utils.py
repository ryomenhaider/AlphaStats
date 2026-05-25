import pandas as pd
import numpy as np
from typing import Callable


def all_check(x, function: Callable):

    if not is_1d(x):
        print(f"{x} is not 1D")
        raise ValueError

    if not isinstance(x, list):
        x = _to_list(x)
        return function(x)

    is_empty(x)


def _to_list(n) -> list:

    if type(n) not in [np.ndarray, pd.Series, tuple, dict, set]:
        raise TypeError(
            f"This function is for numpy array and pd series you provided {type(n)}"
        )

    if isinstance(n, np.ndarray):
        return n.tolist()

    if isinstance(n, pd.Series):
        return n.tolist()

    if isinstance(n, dict):
        return list(n.values())

    if isinstance(n, (tuple, set)):
        return list(n)

    raise TypeError(f"unexpected type: {type(n)}")


def is_1d(x) -> bool:
    is_1d = isinstance(x, list) and not any(isinstance(item, list) for item in x)
    return is_1d


def compare(x: list, y: list) -> None:
    if not (is_1d(x) and is_1d(y)):
        raise TypeError(f"Either {x} or {y} or both are not 1D")
    if len(x) != len(y):
        raise RuntimeError("both list are not of equal length")


def is_empty(x) -> None:

    if len(x) == 0:
        raise ValueError(f"lenth of {x} should be greater then 0")
