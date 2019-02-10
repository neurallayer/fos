# The the standard calculators that are being provided.
from fos.calc import AvgCalc, MomentumCalc, RecentCalc, MaxCalc, MinCalc


def test_avg():
    calc = AvgCalc()
    assert calc.result() is None

    calc.add(5.)
    assert calc.result() == 5.

    calc.add(15.)
    assert calc.result() == 10.

    calc.clear()
    assert calc.result() is None


def test_momentum():
    calc = MomentumCalc(beta=0.90, disable_clear=False)
    assert calc.result() is None

    calc.add(10.)
    assert calc.result() == 10.

    calc.add(20.)
    assert calc.result() == 11.

    calc.clear()
    assert calc.result() is None

    calc = MomentumCalc(beta=0.90, disable_clear=True)
    calc.add(10.)
    calc.clear()
    assert calc.result() == 10.


def test_recent():
    calc = RecentCalc()
    assert calc.result() is None

    calc.add(5.)
    assert calc.result() == 5.

    calc.add(15.)
    assert calc.result() == 15.

    calc.clear()
    assert calc.result() is None


def test_max():
    calc = MaxCalc()
    assert calc.result() is None

    calc.add(5.)
    assert calc.result() == 5.

    calc.add(15.)
    assert calc.result() == 15.

    calc.clear()
    assert calc.result() is None


def test_min():
    calc = MinCalc()
    assert calc.result() is None

    calc.add(5.)
    assert calc.result() == 5.

    calc.add(15.)
    assert calc.result() == 5.

    calc.clear()
    assert calc.result() is None

