import numpy as np
import pandas as pd

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

print(chk(n))
print(chk(x))
