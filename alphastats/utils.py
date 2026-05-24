import pandas as pd
import numpy as np
from typing import Callable

def all_check(x, function: Callable):

    if not is_1d(x):
        print(f'{x} is not 1D')
        raise ValueError
    
    if not isinstance(x, list):
        x = _to_list(x)
        return function(x)
    
    is_empty(x)

def _to_list(n) -> list :

    if n not in [np.array, pd.Series]:
        raise TypeError(f'This function is for numpy array and pd series you provided {type(n)}')
    
    try:

        if n == np.array:
            arr = np.array(n)
            return arr.tolist()
        
        elif n == pd.Series:
            arr = pd.Series(n)
            return arr.tolist()
    
    except Exception as e:
        print(f'Error: {e}')

def is_1d(x) -> bool:
    is_1d = isinstance(x, list) and not any(isinstance(item, list) for item in x)
    return is_1d

def compare(x: list,y: list) -> None:
    
    if is_1d(x) and is_1d(y):
        raise TypeError(f'Either {x} or {y} or both are not 1D')
    
    if len(x) == len(y):
        raise RuntimeError('both list are not of equal lencht')

def is_empty(x) -> None:
    
    if len(x) == 0:
        
        raise ValueError(f'lenth of {x} should be greater then 0')
