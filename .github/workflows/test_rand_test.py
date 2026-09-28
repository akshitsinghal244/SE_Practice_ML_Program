"""Unit tests for the addition function."""

import pytest

from rand_test import add_numbers


@pytest.mark.parametrize(
    "x, y, expected",
    [
        (20, 15, 35),
        (10, 5, 15),
        (0, 0, 0),
        (-5, 10, 5),
        (-10, -20, -30),
        (100, 250, 350),
    ],
)
def test_add_numbers(x, y, expected):
    """Test addition with different x and y values."""
    assert add_numbers(x, y) == expected
