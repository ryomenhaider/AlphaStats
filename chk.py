import numpy as np
import pandas as pd
import sys
from pathlib import Path
import builtins

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from alphastats.descriptive import mean
from alphastats.utils import compare, all_check

def chk(n):
    try:
        if isinstance(n, dict):
                return list(n.items())
        elif isinstance(n, tuple):
                return list(n)
    except Exception as e:
          print(f'{e}')

n = (1,3,4,5,7)
x = {1,3,4,5,7}

# print(chk(n))
# print(chk(x))

def covariance(x, y, type: str = 'population') -> float:

    compare(x, y)
    
    mean_x = mean(x)
    mean_y = mean(y)
    n = len(x)

    if type.lower() == 'population':
        _cov_p = 0
        for i, j in zip(x,y):
            _ = (i - mean_x) * (j - mean_y)
            _cov_p += _
        cov_p = (1/n) * _cov_p
        return cov_p
    
    elif type.lower() == 'sample':
        _cov_s = 0
        for i, j in zip(x, y):
            _ = (i - mean_x) * (j - mean_y)
            _cov_s += _
        cov_s = (1/(n-1)) * _cov_s
        return cov_s

x = [1,2,3,4,5]
y = [6,7,8,9,10]

# print(covariance(x,y))

def rank(data):
    indexed = sorted(enumerate(data), key=lambda x: x[1])
    
    ranks = [0] * len(data)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) - 1 and indexed[j][1] == indexed[j+1][1]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in builtins.range(i, j+1):
            ranks[indexed[k][0]] = avg
        i = j + 1
    
    return ranks

# print(rank([44,56,23,45,10,43]))

def corelation(x, y, type: str = 'pearson'):
    
    compare(x, y)

    if type.lower() == 'pearson':
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

    if type.lower() == 'spearman':
        rx = rank(x)
        ry = rank(y)

        d2 = sum((ri - rj) ** 2 for ri, rj in zip(rx, ry))
        n = len(x)

        nomi = 6 * d2
        denom = n * ((n ** 2) - 1)

        return 1 - (nomi / denom)    

print(corelation(x, y))


def quantile(data, p, method='linear'):
    
    all_check(data, quantile)

    x = sorted(data)
    n = len(x)
    
    if n == 0:
        raise ValueError("empty data")
    if not 0 <= p <= 1:
        raise ValueError("p must be in [0, 1]")
    
    if method == 'nearest':      
        idx = (int(p * n) - 1) + 1
        return x[max(0, min(idx, n-1))]
    
    elif method == 'linear':     
        h = p * (n - 1)
    elif method == 'hazen':       
        h = p * n + 0.5 - 1      
    elif method == 'weibull':    
        h = p * (n + 1) - 1
    elif method == 'median_unbiased': 
        h = p * (n + 1/3) + 1/3 - 1
    elif method == 'normal_unbiased': 
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


print(quantile(x, 0.5))