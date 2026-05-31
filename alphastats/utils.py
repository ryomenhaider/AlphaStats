import numpy as np
from typing import Callable, Any, List


def all_check(x, function: Callable):

    if not is_1d(x):
        print(f"{x} is not 1D")
        raise ValueError

    if not isinstance(x, List):
        x = _to_list(x)
        return function(x)

    is_empty(x)


def _to_list(n: Any) -> List:

    if not isinstance(n, list):
        print(f"{n} is {type(n)}, not a list trying to convert it to list")

        try:
            if isinstance(n, np.ndarray):
                print(f"{n} is a np array trying to convert it to a list")
                n_l = n.tolist()

                if isinstance(n_l, list):
                    print("conversion successful")
                    return n_l
                else:
                    print("Conversion failed")
                    raise RuntimeError(
                        "NumPy array tolist() did not return a list."
                    )  # Fixed

            else:
                print(f"{n} has type {type(n)}, not list, trying to convert it to list")
                n_l = list(n)

                if isinstance(n_l, list):
                    print("conversion successful")
                    return n_l
                else:
                    print("Conversion failed")
                    raise RuntimeError(
                        "Standard list conversion did not return a list."
                    )  # Fixed

        except Exception as e:
            raise RuntimeError(f"Conversion failed, Error: {e}")
    else:
        return n


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
