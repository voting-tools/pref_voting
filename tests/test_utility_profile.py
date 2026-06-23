import copy

import matplotlib
import pytest

from pref_voting.profiles_with_ties import ProfileWithTies
from pref_voting.utility_profiles import UtilityProfile

matplotlib.use("Agg")


@pytest.fixture
def uprof():
    # group of 3 voters with utilities {0:2, 1:5, 2:1} and 1 voter with {0:4, 1:1, 2:3}
    return UtilityProfile([{0: 2, 1: 5, 2: 1}, {0: 4, 1: 1, 2: 3}], ucounts=[3, 1])


def test_create(uprof):
    assert uprof.domain == [0, 1, 2]
    assert uprof.candidates == [0, 1, 2]
    assert uprof.num_cands == 3
    assert uprof.num_voters == 4


def test_util_sum(uprof):
    assert uprof.util_sum(0) == 10  # 2*3 + 4*1
    assert uprof.util_sum(1) == 16  # 5*3 + 1*1
    assert uprof.util_sum(2) == 6  # 1*3 + 3*1


def test_util_avg_is_voter_weighted(uprof):
    # Regression: util_avg multiplied each utility by its group count and then took
    # an UNWEIGHTED mean over groups, giving 5.0 instead of 2.5 for candidate 0.
    # The correct value is util_sum / (#voters rating the candidate).
    assert uprof.util_avg(0) == pytest.approx(2.5)
    assert uprof.util_avg(1) == pytest.approx(4.0)
    assert uprof.util_avg(2) == pytest.approx(1.5)
    # must agree with the (independently computed) avg_utility_function
    for c in [0, 1, 2]:
        assert uprof.util_avg(c) == pytest.approx(uprof.avg_utility_function()(c))


def test_util_max_min(uprof):
    assert uprof.util_max(0) == 4 and uprof.util_min(0) == 2
    assert uprof.util_max(1) == 5 and uprof.util_min(1) == 1
    assert uprof.util_max(2) == 3 and uprof.util_min(2) == 1


def test_unrated_candidate_returns_none():
    up = UtilityProfile([{0: 1.0}, {1: 2.0}], ucounts=[1, 1])
    assert up.has_utility(2) is False
    assert up.util_sum(2) is None
    assert up.util_avg(2) is None


def test_sum_and_avg_utility_function(uprof):
    sum_fn = uprof.sum_utility_function()
    avg_fn = uprof.avg_utility_function()
    for c in [0, 1, 2]:
        assert sum_fn(c) == uprof.util_sum(c)
        assert avg_fn(c) == pytest.approx(uprof.util_avg(c))


def test_to_ranking_profile(uprof):
    rp = uprof.to_ranking_profile()
    assert isinstance(rp, ProfileWithTies)
    assert rp.candidates == [0, 1, 2]
    # ranking TYPES (not count-expanded): higher utility -> better (lower rank)
    rmaps = [r.rmap for r in rp._rankings]
    assert {1: 1, 0: 2, 2: 3} in rmaps  # group1: 1 > 0 > 2
    assert {0: 1, 2: 2, 1: 3} in rmaps  # group2: 0 > 2 > 1


def test_to_approval_profile(uprof):
    # prob_to_cont_approving=1.0 -> deterministic: approve everyone strictly above
    # the voter's own average utility. group1 (avg 8/3) approves {1};
    # group2 (avg 8/3) approves {0, 2}.
    ap = uprof.to_approval_profile(prob_to_cont_approving=1.0, decay_rate=0.0)
    assert ap.approval_scores() == {0: 1, 1: 3, 2: 1}


def test_normalize_by_range(uprof):
    nr = uprof.normalize_by_range()
    assert isinstance(nr, UtilityProfile)
    assert nr.domain == [0, 1, 2]
    # group1 {0:2,1:5,2:1}: (u - 1) / (5 - 1)
    g1 = {c: float(nr._utilities[0](c)) for c in [0, 1, 2]}
    assert g1 == pytest.approx({0: 0.25, 1: 1.0, 2: 0.0})


def test_normalize_by_standard_score(uprof):
    ns = uprof.normalize_by_standard_score()

    assert isinstance(ns, UtilityProfile)
    # standard scores are mean-centered: each voter's values sum to ~0
    for u in ns._utilities:
        assert sum(float(u(c)) for c in [0, 1, 2]) == pytest.approx(0.0)


def test_as_dict_from_json_roundtrip(uprof):
    d = uprof.as_dict()
    up2 = UtilityProfile.from_json(copy.deepcopy(d))
    assert up2.as_dict() == d


def test_write_from_string_roundtrip(uprof):
    # Regression: from_string passes domain=range(num_alternatives), and __init__
    # does `_domain += [...]`, which raises TypeError on a range object. Fix in
    # __init__:  `_domain = list(domain) if domain is not None else []`.
    s = uprof.write()
    up2 = UtilityProfile.from_string(s)
    # util_sum is invariant under the count-expansion that write() performs
    for c in [0, 1, 2]:
        assert up2.util_sum(c) == uprof.util_sum(c)


def test_display_runs(uprof):
    # smoke test
    uprof.display()


def test_display_show_totals_does_not_crash(uprof):
    # Regression: display(show_totals=True) called self.cmap(x) on a dict cmap,
    # raising TypeError. It must index with self.cmap[x].
    uprof.display(show_totals=True)
