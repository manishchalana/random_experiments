import pytest

from main import convert_currency


def test_same_currency_is_unchanged():
    assert convert_currency(42.0, 'USD', 'USD') == 42.0


def test_eur_to_usd():
    assert convert_currency(100, 'EUR', 'USD') == 113.0


def test_gbp_to_jpy():
    assert convert_currency(50, 'GBP', 'JPY') == 9225.35


def test_unknown_currency():
    with pytest.raises(ValueError):
        convert_currency(100, 'USD', 'ABC')
