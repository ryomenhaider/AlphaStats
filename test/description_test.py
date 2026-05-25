import builtins
import numpy as np
import pytest
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from alphastats.descriptive import (  # noqa: E402
    mean,
    median,
    mode,
    variance,
    std,
    Skewness,
    Kurtosis,
    count,
    f_min,
    f_max,
    range as _range,
    covariance,
    corelation,
    rank,
    quantile,
    quantiles,
    iqr,
    percentile,
    ln,
)


class TestMean:
    def test_basic(self):
        assert mean([1, 2, 3, 4, 5]) == 3.0

    def test_floats(self):
        assert mean([1.5, 2.5, 3.5]) == 2.5

    def test_single_element(self):
        assert mean([42]) == 42.0

    def test_negatives(self):
        assert mean([-5, 0, 5]) == 0.0

    def test_all_same(self):
        assert mean([7, 7, 7]) == 7.0


class TestMedian:
    def test_odd_length(self):
        result = median([1, 3, 3, 6, 7, 8, 9])
        assert result == 6.0

    def test_even_length(self):
        result = median([1, 2, 3, 4, 5, 6])
        assert result == 3.5

    def test_single_element(self):
        assert median([42]) == 42.0

    def test_two_elements(self):
        assert median([10, 20]) == 15.0

    def test_negatives(self):
        assert median([-3, -1, 0, 2, 5]) == 0.0


class TestMode:
    def test_basic(self):
        assert mode([1, 2, 2, 3, 4, 4, 4, 5]) == 4

    def test_single_element(self):
        assert mode([42]) == 42

    def test_first_encountered_for_ties(self):
        assert mode([1, 1, 2, 2]) == 1

    def test_all_unique(self):
        assert mode([1, 2, 3, 4, 5]) == 1

    def test_strings(self):
        assert mode(["a", "b", "b", "c"]) == "b"


class TestVariance:
    def test_population_basic(self):
        result = variance([1, 2, 3, 4, 5])
        expected = np.var([1, 2, 3, 4, 5], ddof=0)
        assert result == pytest.approx(expected)

    def test_constant(self):
        assert variance([5, 5, 5, 5]) == 0.0

    def test_two_elements(self):
        result = variance([1, 3])
        expected = np.var([1, 3], ddof=0)
        assert result == pytest.approx(expected)

    def test_floats(self):
        result = variance([1.5, 2.5, 3.5])
        expected = np.var([1.5, 2.5, 3.5], ddof=0)
        assert result == pytest.approx(expected)


class TestStd:
    def test_basic(self):
        result = std([1, 2, 3, 4, 5])
        expected = np.std([1, 2, 3, 4, 5], ddof=0)
        assert result == pytest.approx(expected)

    def test_constant(self):
        assert std([5, 5, 5, 5]) == 0.0

    def test_single_element(self):
        assert std([42]) == 0.0


class TestSkewness:
    def test_symmetric(self):
        data = [1, 2, 3, 4, 5]
        result = Skewness(data)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_positive_skew(self):
        result = Skewness([1, 2, 2, 3, 5])
        assert result > 0

    def test_negative_skew(self):
        result = Skewness([1, 5, 5, 5, 5])
        assert result < 0


class TestKurtosis:
    def test_uniform(self):
        data = [1, 2, 3, 4, 5]
        result = Kurtosis(data)
        assert result < 0

    def test_normal_like(self):
        result = Kurtosis([-2, -1, 0, 1, 2])
        assert isinstance(result, float)


class TestCount:
    def test_basic(self):
        assert count([1, 2, 3]) == 3

    def test_single(self):
        assert count([42]) == 1

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            count([])


class TestFMin:
    def test_basic(self):
        assert f_min([3, 1, 4, 1, 5]) == 1

    def test_negative(self):
        assert f_min([-5, 0, 3]) == -5

    def test_single(self):
        assert f_min([42]) == 42


class TestFMax:
    def test_basic(self):
        assert f_max([3, 1, 4, 1, 5]) == 5

    def test_negative(self):
        assert f_max([-5, 0, 3]) == 3

    def test_single(self):
        assert f_max([42]) == 42


class TestRange:
    def test_basic(self):
        assert _range([3, 1, 4, 1, 5]) == 4

    def test_all_same(self):
        assert _range([5, 5, 5]) == 0

    def test_negative(self):
        assert _range([-10, 0, 10]) == 20


class TestCovariance:
    def test_population(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        result = covariance(x, y, "population")
        expected = np.cov(x, y, ddof=0)[0, 1]
        assert result == pytest.approx(expected)

    def test_sample(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        result = covariance(x, y, "sample")
        expected = np.cov(x, y, ddof=1)[0, 1]
        assert result == pytest.approx(expected)

    def test_default_population(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        result = covariance(x, y)
        expected = np.cov(x, y, ddof=0)[0, 1]
        assert result == pytest.approx(expected)

    def test_uncorrelated(self):
        x = [1, 2, 3, 4, 5]
        y = [5, 5, 5, 5, 5]
        result = covariance(x, y)
        assert result == pytest.approx(0.0)


class TestRank:
    def test_with_ties(self):
        result = rank([3, 1, 4, 1, 5])
        assert result == [3.0, 1.5, 4.0, 1.5, 5.0]

    def test_all_same(self):
        assert rank([5, 5, 5]) == [2.0, 2.0, 2.0]

    def test_single(self):
        assert rank([42]) == [1.0]

    def test_sorted(self):
        result = rank([1, 2, 3, 4])
        assert result == [1.0, 2.0, 3.0, 4.0]


class TestCorelation:
    def test_perfect_positive_pearson(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        result = corelation(x, y, "pearson")
        assert result == pytest.approx(1.0)

    def test_perfect_negative_pearson(self):
        x = [1, 2, 3, 4, 5]
        y = [10, 8, 6, 4, 2]
        result = corelation(x, y, "pearson")
        assert result == pytest.approx(-1.0)

    def test_constant_y_raises(self):
        x = [1, 2, 3, 4, 5]
        y = [5, 5, 5, 5, 5]
        with pytest.raises(ZeroDivisionError):
            corelation(x, y, "pearson")


class TestQuantile:
    def test_median(self):
        assert quantile([1, 2, 3, 4, 5], 0.5) == pytest.approx(3.0)

    def test_min(self):
        assert quantile([1, 2, 3, 4, 5], 0.0) == pytest.approx(1.0)

    def test_max(self):
        assert quantile([1, 2, 3, 4, 5], 1.0) == pytest.approx(5.0)

    def test_q1(self):
        result = quantile([1, 2, 3, 4, 5], 0.25)
        assert result == pytest.approx(2.0)

    def test_q3(self):
        result = quantile([1, 2, 3, 4, 5], 0.75)
        assert result == pytest.approx(4.0)

    def test_invalid_p_low(self):
        with pytest.raises(ValueError):
            quantile([1, 2, 3], -0.1)

    def test_invalid_p_high(self):
        with pytest.raises(ValueError):
            quantile([1, 2, 3], 1.5)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            quantile([], 0.5)

    def test_nearest_method(self):
        result = quantile([1, 2, 3, 4, 5], 0.5, method="nearest")
        assert result == 3

    def test_weibull_method(self):
        result = quantile([1, 2, 3, 4, 5], 0.5, method="weibull")
        assert result == pytest.approx(3.0)

    def test_hazen_method(self):
        result = quantile([1, 2, 3, 4, 5], 0.25, method="hazen")
        assert isinstance(result, float)

    def test_unknown_method(self):
        with pytest.raises(ValueError, match="unknown method"):
            quantile([1, 2, 3], 0.5, method="invalid")


class TestQuantiles:
    def test_basic(self):
        result = quantiles([1, 2, 3, 4, 5], [0.25, 0.5, 0.75])
        assert len(result) == 3
        assert result[0] == pytest.approx(2.0)
        assert result[1] == pytest.approx(3.0)
        assert result[2] == pytest.approx(4.0)

    def test_single(self):
        result = quantiles([1, 2, 3, 4, 5], [0.5])
        assert result == [pytest.approx(3.0)]


class TestIQR:
    def test_basic(self):
        result = iqr([1, 2, 3, 4, 5])
        assert result == pytest.approx(2.0)

    def test_constant(self):
        assert iqr([5, 5, 5, 5, 5]) == 0.0

    def test_larger(self):
        data = list(builtins.range(1, 101))
        result = iqr(data)
        assert result == pytest.approx(49.5)


class TestPercentile:
    def test_50th(self):
        result = percentile([1, 2, 3, 4, 5], 50)
        assert result == pytest.approx(3.0)

    def test_0th(self):
        assert percentile([1, 2, 3, 4, 5], 0) == pytest.approx(1.0)

    def test_100th(self):
        assert percentile([1, 2, 3, 4, 5], 100) == pytest.approx(5.0)

    def test_25th(self):
        result = percentile([1, 2, 3, 4, 5], 25)
        assert result == pytest.approx(2.0)


class TestLn:
    def test_ln_1(self):
        assert ln(1) == pytest.approx(0.0, abs=1e-4)

    def test_ln_e(self):
        assert ln(2.718281828459045) == pytest.approx(1.0, abs=6e-4)

    def test_ln_10(self):
        result = ln(10)
        assert result > 0
        assert result == pytest.approx(2.302585, abs=3e-3)

    def test_fractional(self):
        result = ln(0.5)
        assert result < 0

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="Math Domain Error"):
            ln(0)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="Math Domain Error"):
            ln(-1)
