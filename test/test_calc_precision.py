# The the standard calculators that are being provided.
from fos.calc import PrecisionCalculator, RecallCalculator, BetaCalculator


def add(calc):
    val = {
        "tp" : 2,
        "tn" : 4,
        "fp" : 6,
        "fn" : 2
    }
    calc.add(val)


def test_precision():
    calc = PrecisionCalculator()
    assert calc.result() is None

    add(calc)
    
def test_recall():
    calc = RecallCalculator()
    assert calc.result() is None

    add(calc) 
    
def test_recall():
    calc = BetaCalculator()
    assert calc.result() is None

    add(calc) 