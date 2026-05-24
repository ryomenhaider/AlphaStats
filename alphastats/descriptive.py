from utils import _to_list, is_empty, is_1d

def mean(x) -> float:

    if not is_1d(x):
        print(f'{x} is not 1D')
        raise ValueError
    
    if not isinstance(x, list):
        x = _to_list(x)
        return mean(x)
    
    is_empty(x)
    n = len(x)
    x_sum = sum(x)
    return x_sum / n

def median(x):
    
    if not is_1d(x):
        print(f'{x} is not 1D')
        raise ValueError
    
    if not isinstance(x, list):
        x = _to_list(x)
        return median(x)
    
    is_empty(x)
    
    n = len(x)
    try:

        if x % 2:
            return ((x * (n/2)) + (x * (n/2 + 1)))/2 
        else:
            return (x * (n + 1))/2

    except Exception as e:
        print(f'Error: {e}')

def mode(x):

    if not is_1d(x):
        print(f'{x} is not 1D')
        raise ValueError
    
    if not isinstance(x, list):
        x = _to_list(x)
        return mode(x)

    is_empty(x)

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
    if not is_1d(x):
        print(f'{x} is not 1D')
        raise ValueError
    
    if not isinstance(x, list):
        x = _to_list(x)
        return variance(x)

    is_empty(x)

    mean = mean(x)
    n = len(x)
    
    _var = 0
    for i in x:
        n = (i - mean) ** 2
        _var += n

    var = (1/n) * _var

    return var

def std(x):

    var = variance(x)
    std = var ** (1/2)
    return std

def Skewness(x):

    if not is_1d(x):
        print(f'{x} is not 1D')
        raise ValueError
    
    if not isinstance(x, list):
        x = _to_list(x)
        return variance(x)

    is_empty(x)

    _d = 0