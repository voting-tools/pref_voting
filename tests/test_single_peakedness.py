import pytest

from pref_voting.profiles import Profile
from pref_voting.profiles_with_ties import ProfileWithTies
from pref_voting.rankings import Ranking
from pref_voting.single_peakedness import (
    is_single_peaked,
    num_mavericks,
    min_k_maverick_single_peaked,
)


# ---------------------------------------------------------------------------
# Linear rankings (lists)
# ---------------------------------------------------------------------------

def test_linear_sp_basic():
    assert is_single_peaked([1, 0, 2], [0, 1, 2]) is True
    assert is_single_peaked([0, 1, 2], [0, 1, 2]) is True
    assert is_single_peaked([2, 1, 0], [0, 1, 2]) is True


def test_linear_not_sp():
    # 0 > 2 > 1: bottom 1 is in the interior of the axis
    assert is_single_peaked([0, 2, 1], [0, 1, 2]) is False


def test_linear_two_candidate_always_sp():
    assert is_single_peaked([0, 1], [0, 1]) is True
    assert is_single_peaked([1, 0], [0, 1]) is True


def test_truncated_list_contiguous_is_sp():
    # Ranked candidates form a contiguous segment of the axis
    assert is_single_peaked([1, 0], [0, 1, 2]) is True
    assert is_single_peaked([1, 2], [0, 1, 2]) is True


def test_truncated_list_noncontiguous_not_sp():
    assert is_single_peaked([0, 2], [0, 1, 2]) is False


def test_truncated_list_treated_as_maverick():
    assert is_single_peaked([1, 0], [0, 1, 2], treat_truncated_as_maverick=True) is False


# ---------------------------------------------------------------------------
# Ranking objects: default 'maverick' handling
# ---------------------------------------------------------------------------

def test_ranking_linear_sp():
    r = Ranking({0: 1, 1: 2, 2: 3})
    assert is_single_peaked(r, [0, 1, 2], num_cands=3) is True


def test_ranking_with_tie_default_is_maverick():
    r = Ranking({0: 1, 1: 1, 2: 2})
    assert is_single_peaked(r, [0, 1, 2], num_cands=3) is False


# ---------------------------------------------------------------------------
# possibly_sp
# ---------------------------------------------------------------------------

def test_possibly_sp_tie_at_top():
    r = Ranking({0: 1, 1: 1, 2: 2})
    assert is_single_peaked(
        r, [0, 1, 2], num_cands=3, tied_ranking_handling='possibly_sp'
    ) is True


def test_possibly_sp_interior_class_not_extreme():
    # axis=[a,b,c,d,e], {c} > {a,e} > {b,d}: no SP linear extension exists
    # because the bottom class {b,d} sits in the interior of the axis.
    r = Ranking({'c': 1, 'a': 2, 'e': 2, 'b': 3, 'd': 3})
    assert is_single_peaked(
        r, ['a', 'b', 'c', 'd', 'e'], num_cands=5,
        tied_ranking_handling='possibly_sp',
    ) is False


def test_possibly_sp_truncated_contiguous():
    # Ranked candidates {1,2,3} form a contiguous segment on axis [0..4]
    r = Ranking({2: 1, 1: 1, 3: 2})
    assert is_single_peaked(
        r, [0, 1, 2, 3, 4], num_cands=5, tied_ranking_handling='possibly_sp'
    ) is True


# ---------------------------------------------------------------------------
# single_plateaued
# ---------------------------------------------------------------------------

def test_plateau_contiguous_strict_below():
    # {1,2} > 0 > 3 on axis [0,1,2,3]: plateau contiguous, strict on each side.
    r = Ranking({1: 1, 2: 1, 0: 2, 3: 3})
    assert is_single_peaked(
        r, [0, 1, 2, 3], num_cands=4, tied_ranking_handling='single_plateaued'
    ) is True


def test_plateau_noncontiguous_not_sp():
    # {0,2} are the top class but not contiguous on axis [0,1,2,3]
    r = Ranking({0: 1, 2: 1, 1: 2, 3: 3})
    assert is_single_peaked(
        r, [0, 1, 2, 3], num_cands=4, tied_ranking_handling='single_plateaued'
    ) is False


def test_plateau_cross_side_tie_allowed():
    # 1 > 0 ~ 2 on axis [0,1,2]: 0 and 2 are on opposite sides of the peak,
    # so the cross-side tie is fine.  This was incorrectly rejected before.
    r = Ranking({1: 1, 0: 2, 2: 2})
    assert is_single_peaked(
        r, [0, 1, 2], num_cands=3, tied_ranking_handling='single_plateaued'
    ) is True


def test_plateau_cross_side_tie_with_plateau():
    # {1,2} > {0,3} on axis [0,1,2,3]: plateau contiguous; the lower tie is
    # across opposite sides of the plateau.
    r = Ranking({1: 1, 2: 1, 0: 2, 3: 2})
    assert is_single_peaked(
        r, [0, 1, 2, 3], num_cands=4, tied_ranking_handling='single_plateaued'
    ) is True


def test_plateau_same_side_tie_below_not_allowed():
    # 2 > 1 ~ 0 on axis [0,1,2]: 0 and 1 are on the same side of peak 2.
    r = Ranking({2: 1, 1: 2, 0: 2})
    assert is_single_peaked(
        r, [0, 1, 2], num_cands=3, tied_ranking_handling='single_plateaued'
    ) is False


def test_plateau_monotonicity_must_strictly_worsen_away_from_plateau():
    # {2,3} > 0 > 1 on axis [0,1,2,3]: moving LEFT from the plateau we hit 1
    # (rank 2) then 0 (rank 1) — that's an improvement, not a worsening.
    # This was incorrectly accepted before.
    r = Ranking({2: 1, 3: 1, 0: 2, 1: 3})
    assert is_single_peaked(
        r, [0, 1, 2, 3], num_cands=4, tied_ranking_handling='single_plateaued'
    ) is False


# ---------------------------------------------------------------------------
# black_sp
# ---------------------------------------------------------------------------

def test_black_sp_unique_peak_strict():
    r = Ranking({1: 1, 0: 2, 2: 3})
    assert is_single_peaked(
        r, [0, 1, 2], num_cands=3, tied_ranking_handling='black_sp'
    ) is True


def test_black_sp_cross_side_tie_allowed():
    # 1 > 0 ~ 2: unique peak, 0 and 2 on opposite sides.
    r = Ranking({1: 1, 0: 2, 2: 2})
    assert is_single_peaked(
        r, [0, 1, 2], num_cands=3, tied_ranking_handling='black_sp'
    ) is True


def test_black_sp_same_side_tie_forbidden():
    # 2 > 1 ~ 0 > 3 on axis [0,1,2,3]: 0 and 1 are on the same side of peak 2,
    # so the tie is not permitted under Black SP.  Was incorrectly accepted.
    r = Ranking({2: 1, 1: 2, 0: 2, 3: 3})
    assert is_single_peaked(
        r, [0, 1, 2, 3], num_cands=4, tied_ranking_handling='black_sp'
    ) is False


def test_black_sp_no_unique_peak():
    r = Ranking({0: 1, 2: 1, 1: 2})
    assert is_single_peaked(
        r, [0, 1, 2], num_cands=3, tied_ranking_handling='black_sp'
    ) is False


# ---------------------------------------------------------------------------
# Profile-level: num_mavericks and min_k_maverick_single_peaked
# ---------------------------------------------------------------------------

def test_num_mavericks_linear_profile():
    prof = Profile([[0, 1, 2], [1, 0, 2], [1, 2, 0], [2, 1, 0], [0, 2, 1]])
    assert num_mavericks(prof, [0, 1, 2]) == 1


def test_min_k_linear_profile():
    prof = Profile([[0, 1, 2], [1, 0, 2], [1, 2, 0], [2, 1, 0], [0, 2, 1]])
    min_k, best_axis = min_k_maverick_single_peaked(prof)
    assert min_k == 1
    assert best_axis in ([0, 1, 2], [2, 1, 0])


def test_min_k_sp_profile_is_zero():
    prof = Profile([[0, 1, 2], [1, 0, 2], [2, 1, 0]])
    min_k, best_axis = min_k_maverick_single_peaked(prof)
    assert min_k == 0
    assert best_axis is not None


def test_min_k_single_candidate():
    prof = Profile([[0], [0]])
    assert min_k_maverick_single_peaked(prof) == (0, [0])


def test_min_k_returns_valid_axis_when_all_voters_maverick():
    # Every ballot is fully tied, default handling counts every voter as a
    # maverick for every axis.  best_axis must still be a valid axis, not None.
    prof = ProfileWithTies([{0: 1, 1: 1, 2: 1}] * 3)
    min_k, best_axis = min_k_maverick_single_peaked(prof)
    assert min_k == 3
    assert best_axis is not None
    assert sorted(best_axis) == [0, 1, 2]


def test_num_mavericks_with_possibly_sp():
    # 1 > 0 ~ 2 is possibly SP on [0,1,2] (break the tie either way).
    prof = ProfileWithTies([{1: 1, 0: 2, 2: 2}, {0: 1, 1: 2, 2: 3}])
    assert num_mavericks(
        prof, [0, 1, 2], tied_ranking_handling='possibly_sp'
    ) == 0


def test_invalid_tied_ranking_handling_raises():
    r = Ranking({0: 1, 1: 1, 2: 2})
    with pytest.raises(ValueError):
        is_single_peaked(
            r, [0, 1, 2], num_cands=3, tied_ranking_handling='bogus'
        )


def test_is_single_peaked_list_duplicate_candidate_raises():
    with pytest.raises(ValueError):
        is_single_peaked([0, 0, 1], [0, 1, 2])


def test_is_single_peaked_ranking_candidate_not_on_axis_raises():
    r = Ranking({0: 1, 3: 2})
    with pytest.raises(ValueError):
        is_single_peaked(r, [0, 1, 2], num_cands=3)


def test_is_single_peaked_num_cands_mismatch_raises():
    r = Ranking({0: 1, 1: 2, 2: 3})
    with pytest.raises(ValueError):
        is_single_peaked(r, [0, 1, 2], num_cands=4)


def test_num_mavericks_axis_must_match_profile_candidates():
    prof = Profile([[0, 1, 2]])
    with pytest.raises(
        ValueError, match="axis must contain exactly the profile candidates"
    ):
        num_mavericks(prof, [0, 1])


def test_num_mavericks_axis_duplicate_candidate_raises():
    prof = Profile([[0, 1, 2]])
    with pytest.raises(ValueError):
        num_mavericks(prof, [0, 1, 1])
