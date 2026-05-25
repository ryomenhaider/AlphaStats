import builtins
from utils import _to_list, is_empty, is_1d, compare, all_check

def mean(x) -> float:

    all_check(x, mean)
    
    n = len(x)
    x_sum = sum(x)
    return x_sum / n

def median(x):
    all_check(x, median)
    n = len(x)
    sx = sorted(x)
    if n % 2 == 1:
        return sx[n // 2]
    return (sx[n // 2 - 1] + sx[n // 2]) / 2

def mode(x):

    all_check(x, mode)

    try:
        count = {}
        for i in x:
            if i in count:
                count[i] += 1
            else:
                count[i] = 1
        max_count = max(count.values())

        for key, value in count.items():
            
            if value == max_count:
                return key
            
    except Exception as e:
        print(f'Error: {e}')

def variance(x):
    all_check(x, variance)
    mean_val = mean(x)
    n = len(x)
    _var = 0
    for i in x:
        _var += (i - mean_val) ** 2
    return (1 / n) * _var

def std(x):

    var = variance(x)
    std = var ** (1/2)
    return std

def Skewness(x):

    all_check(x, Skewness)

    mean_x = mean(x)
    n = len(x)

    _n = 0
    for i in x:
        _n_ = (i - mean_x) ** 3
        _n += _n_
    
    nominator = (1/n) * _n

    _d = 0
    for i in x:
        _d_ = (i - mean_x) ** 2
        _d += _d_
    
    dominator = ((1/n) * _d) ** (3/2)

    return nominator / dominator

def Kurtosis(x):

    all_check(x, Kurtosis)

    mean_x = mean(x)
    n = len(x)

    _n = 0  
    for i in x:
        _n_ = (i - mean_x) ** 4
        _n += _n_
    nominator = (1/n) * _n

    _d = 0
    for i in x:
        _d_ = (i - mean_x) ** 2
        _d += _d_
    dominator = ((1/n) * _d) ** 2

    return (nominator / dominator) - 3

def count(x):
    
    all_check(x, count)

    return len(x)

def f_min(x):
    all_check(x, f_min)

    return min(x)

def f_max(x):
    all_check(x, f_max)

    return max(x)

def range(x):
    all_check(x, range)
    res = f_max(x) - f_min(x)
    return res

def covariance(x, y, type: str = 'population'):

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

# used in spearman corelation
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


def quantiles(data, ps, method='linear'):
    return [quantile(data, p, method) for p in ps]


def iqr(data, method='linear'):
    return quantile(data, 0.75, method) - quantile(data, 0.25, method)

def percentile(data, k, method='linear'):
    return quantile(data, k / 100, method)


def ln(x):
    if x <= 0: raise ValueError('Math Domain Error')
    n = 1000
    return n * ((x ** (1/n)) - 1)

pi = 22/7

E = 2.718281828459045