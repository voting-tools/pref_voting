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


# ── truncated ranking profiles ────────────────────────────────────────────────


def _ballots(prof):
    """(rmap dict, count) per ballot actually cast, in profile order."""
    return [(dict(r.rmap), c) for r, c in zip(prof.rankings, prof.rcounts)]


# radius: rank c iff u(best) - u(c) < radius (relative to the voter's own favorite)


def test_truncate_radius_keeps_within_radius_of_favorite():
    # u_best = -0.05; gaps below favorite: 1 -> 0.15, 3 -> 0.35
    up = UtilityProfile([{0: -0.05, 1: -0.20, 2: -0.05, 3: -0.40}])
    prof = up.to_truncated_ranking_profile("radius", radius=0.25)
    assert isinstance(prof, ProfileWithTies)
    ((rmap, count),) = _ballots(prof)
    assert rmap == {0: 1, 2: 1, 1: 2}  # 3 dropped (0.35 >= 0.25); 0, 2 tied favorites
    assert count == 1
    assert prof.candidates == [0, 1, 2, 3]  # full candidate set retained


def test_truncate_radius_is_relative_to_each_voters_own_best():
    # same utility spread on a shifted scale -> identical ballots, because the radius is
    # measured from each voter's own favorite (an absolute cutoff would differ).
    up = UtilityProfile(
        [
            {0: -0.05, 1: -0.20, 2: -0.40},  # favorite at -0.05
            {0: -10.05, 1: -10.20, 2: -10.40},
        ]
    )  # favorite at -10.05
    b = _ballots(up.to_truncated_ranking_profile("radius", radius=0.25))
    assert b[0][0] == b[1][0] == {0: 1, 1: 2}  # candidate 2 dropped for both


def test_truncate_radius_always_keeps_favorite_no_abstention():
    # even a tiny radius leaves a bullet vote for the favorite; nobody abstains
    up = UtilityProfile([{0: -0.30, 1: -0.31, 2: -0.40}])
    prof = up.to_truncated_ranking_profile("radius", radius=0.001)
    ((rmap, _),) = _ballots(prof)
    assert rmap == {0: 1}
    assert prof.num_voters == 1


def test_truncate_radius_no_truncation_matches_full_ranking():
    up = UtilityProfile(
        [{0: -0.1, 1: -0.5, 2: -0.2, 3: -0.2}, {0: -0.9, 1: -0.1, 2: -0.3, 3: -0.3}]
    )
    trunc = up.to_truncated_ranking_profile("radius", radius=1e9)
    full = up.to_ranking_profile()
    assert _ballots(trunc) == [
        (dict(r.rmap), c) for r, c in zip(full.rankings, full.rcounts)
    ]


def test_truncate_radius_counts_preserved():
    up = UtilityProfile([{0: -0.1, 1: -0.2}, {0: -0.9, 1: -0.1}], ucounts=[7, 3])
    prof = up.to_truncated_ranking_profile("radius", radius=1.0)
    assert [c for _, c in _ballots(prof)] == [7, 3]
    assert prof.num_voters == 10


# gap: rank best-to-worst, stop when two adjacent distinct levels are within min_gap


def test_truncate_gap_keeps_all_when_well_separated():
    # levels -0.05, -0.20, -0.40 -> gaps 0.15, 0.20 both >= 0.05
    up = UtilityProfile([{0: -0.05, 1: -0.20, 2: -0.40}])
    ((rmap, _),) = _ballots(up.to_truncated_ranking_profile("gap", min_gap=0.05))
    assert rmap == {0: 1, 1: 2, 2: 3}


def test_truncate_gap_drops_both_of_a_too_close_pair_and_below():
    # levels 10, 9, 8, 1, 0.9 ; gaps 1, 1, 7, 0.1 ; first gap < 0.5 is (1, 0.9)
    up = UtilityProfile([{0: 10, 1: 9, 2: 8, 3: 1, 4: 0.9}])
    ((rmap, _),) = _ballots(up.to_truncated_ranking_profile("gap", min_gap=0.5))
    assert rmap == {
        0: 1,
        1: 2,
        2: 3,
    }  # candidates 3 and 4 (the close pair) both dropped


def test_truncate_gap_ties_are_one_level():
    # 0 and 2 exactly tied (top level), then a big gap to 1
    up = UtilityProfile([{0: -0.10, 1: -0.50, 2: -0.10}])
    ((rmap, _),) = _ballots(up.to_truncated_ranking_profile("gap", min_gap=0.05))
    assert rmap == {0: 1, 2: 1, 1: 2}  # tie kept together; gap measured between levels


def test_truncate_gap_abstention_and_require_at_least_one():
    up = UtilityProfile([{0: -0.30, 1: -0.31, 2: -0.40}])  # first gap 0.01 < 0.10
    # abstain: voter ranks no one, so the profile is empty (turnout 0)
    abstain = up.to_truncated_ranking_profile(
        "gap", min_gap=0.10, require_at_least_one=False
    )
    assert _ballots(abstain) == []
    assert abstain.num_voters == 0
    # require_at_least_one: the same voter bullet-votes their favorite
    ((rmap, _),) = _ballots(
        up.to_truncated_ranking_profile("gap", min_gap=0.10, require_at_least_one=True)
    )
    assert rmap == {0: 1}


def test_truncate_drops_abstainers_keeps_others():
    # voter 0's top two are far apart (ranks both); voter 1's are within min_gap (abstains)
    up = UtilityProfile([{0: 1.0, 1: 0.0}, {0: 1.0, 1: 0.95}], ucounts=[4, 6])
    prof = up.to_truncated_ranking_profile(
        "gap", min_gap=0.10, require_at_least_one=False
    )
    assert _ballots(prof) == [({0: 1, 1: 2}, 4)]  # the 6 abstainers are dropped
    assert prof.num_voters == 4


# argument validation


def test_truncate_missing_parameter_raises():
    up = UtilityProfile([{0: -0.1, 1: -0.2}])
    with pytest.raises(ValueError):
        up.to_truncated_ranking_profile("radius")  # no radius
    with pytest.raises(ValueError):
        up.to_truncated_ranking_profile("gap")  # no min_gap


def test_truncate_unknown_method_raises():
    up = UtilityProfile([{0: -0.1, 1: -0.2}])
    with pytest.raises(ValueError):
        up.to_truncated_ranking_profile("nonsense", radius=0.5)


# per-voter Utility.to_truncated_ranking


def test_utility_to_truncated_ranking():
    from pref_voting.rankings import Ranking

    up = UtilityProfile([{0: -0.05, 1: -0.20, 2: -0.40}])
    util = up._utilities[0]
    r = util.to_truncated_ranking("radius", radius=0.25)
    assert isinstance(r, Ranking)
    assert dict(r.rmap) == {0: 1, 1: 2}  # candidate 2 truncated
    # top two within min_gap: abstain, or keep the favorite
    assert util.to_truncated_ranking(
        "gap", min_gap=0.20, require_at_least_one=False
    ).is_empty()
    assert dict(
        util.to_truncated_ranking("gap", min_gap=0.20, require_at_least_one=True).rmap
    ) == {0: 1}
