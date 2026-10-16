import math

from cluedo.core.stats import holm, mcnemar_exact, wilson_ci


def test_wilson_ci_known_value():
    lo, hi = wilson_ci(50, 100)  # textbook: 0.4038 .. 0.5962
    assert abs(lo - 0.4038) < 1e-3 and abs(hi - 0.5962) < 1e-3


def test_wilson_ci_edges_stay_in_unit_interval():
    lo, hi = wilson_ci(0, 20)
    assert lo == 0.0 and 0 < hi < 0.2
    lo, hi = wilson_ci(20, 20)
    assert hi == 1.0 and 0.8 < lo < 1
    assert wilson_ci(0, 0) == (0.0, 1.0)


def test_mcnemar_exact_worked_examples():
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(0, 0) == 1.0
    # 9 vs 1 discordant: 2 * P(X<=1 | n=10, .5) = 2 * 11/1024
    assert math.isclose(mcnemar_exact(9, 1), 22 / 1024)
    assert mcnemar_exact(9, 1) == mcnemar_exact(1, 9)


def test_holm_worked_example():
    # sorted p: .01,.02,.03 with m=3 -> .03, .04, .04 (monotone), order restored
    assert holm([0.03, 0.01, 0.02]) == [0.04, 0.03, 0.04]
    assert holm([0.9, 0.001]) == [0.9, 0.002]
    assert holm([]) == []
