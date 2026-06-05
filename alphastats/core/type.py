from dataclasses import dataclass


@dataclass(frozen=True)
class descriptive:
    mean: float
    median: float
    mode: float
    variance: float
    std: float
    Skewness: float
    Kurtosis: float
    count: float
    f_min: float
    f_max: float
    range: float
    covariance: float
    rank: list[float]
    corelation: float
    quantile: list
    ln: float
