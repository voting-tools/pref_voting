import pickle

import matplotlib
import numpy as np
import pytest

from pref_voting.util_grade_profile import UtilGradeProfile
from pref_voting.utility_profiles import UtilityProfile
from pref_voting.grade_profiles import GradeProfile
from pref_voting.profiles_with_ties import ProfileWithTies
from pref_voting.pref_grade_profile import PrefGradeProfile
from pref_voting.mappings import Utility, Grade, _Mapping

matplotlib.use("Agg")


@pytest.fixture
def ug():
    # 3 voters: utilities {0:2,1:5,2:1}, grades {0:2,1:1,2:0}
    # 1 voter:  utilities {0:4,1:1,2:3}, grades {0:0,1:2,2:2}
    # numeric grades, no grade_order  -> use_grade_order is False, can_sum_grades True
    return UtilGradeProfile(
        [{0: 2, 1: 5, 2: 1}, {0: 4, 1: 1, 2: 3}],
        [{0: 2, 1: 1, 2: 0}, {0: 0, 1: 2, 2: 2}],
        [0, 1, 2],
        ucounts=[3, 1],
        candidates=[0, 1, 2],
    )


@pytest.fixture
def approval_ug():
    return UtilGradeProfile(
        [{0: 5, 1: 1}, {0: 1, 1: 5}],
        [{0: 1, 1: 0}, {0: 0, 1: 1}],
        [0, 1],
        ucounts=[2, 1],
    )


@pytest.fixture
def ug_ordered():
    # non-numeric grades with an explicit grade_order (best -> worst).
    # use_grade_order is True, can_sum_grades is False.
    # 2 voters: utilities {0:1,1:0}, grades {0:"good", 1:"bad"}
    # 1 voter:  utilities {0:0,1:1}, grades {0:"ok",   1:"good"}
    return UtilGradeProfile(
        [{0: 1, 1: 0}, {0: 0, 1: 1}],
        [{0: "good", 1: "bad"}, {0: "ok", 1: "good"}],
        ["good", "ok", "bad"],
        ucounts=[2, 1],
        candidates=[0, 1],
        grade_order=["good", "ok", "bad"],
    )


# ---------------------------------------------------------------------------
#  Construction
# ---------------------------------------------------------------------------

def test_create(ug):
    assert ug.candidates == [0, 1, 2]
    assert ug.num_cands == 3
    assert ug.num_voters == 4
    assert ug.ucounts == [3, 1]
    assert ug.can_sum_grades is True
    assert ug.use_grade_order is False
    assert ug.cmap == {0: "0", 1: "1", 2: "2"}
    assert ug.gmap == {0: "0", 1: "1", 2: "2"}


def test_create_discovers_candidates_when_not_given():
    p = UtilGradeProfile(
        [{0: 2, 1: 5}], [{0: 1, 1: 0}], [0, 1], ucounts=[2]
    )
    assert p.candidates == [0, 1]
    assert p.num_voters == 2


def test_create_from_utility_and_grade_objects_discovers_candidates():
    # candidates omitted -> discovered from the Utility / Grade objects themselves
    utils = [Utility({0: 2, 1: 5}), Utility({0: 4, 1: 1})]
    grades = [
        Grade({0: 1, 1: 0}, [0, 1]),
        Grade({0: 0, 1: 1}, [0, 1]),
    ]
    p = UtilGradeProfile(utils, grades, [0, 1], ucounts=[3, 1])
    assert p.candidates == [0, 1]
    assert p.num_voters == 4
    assert p.util_sum(0) == 10
    assert p.sum(0) == 3  # grade 1 (x3) + grade 0 (x1)


def test_create_default_ucounts():
    p = UtilGradeProfile([{0: 2}, {0: 4}], [{0: 1}, {0: 0}], [0, 1])
    assert p.ucounts == [1, 1]
    assert p.num_voters == 2


def test_create_length_mismatch_raises():
    with pytest.raises(AssertionError):
        UtilGradeProfile([{0: 1}], [{0: 1}, {0: 0}], [0, 1])
    with pytest.raises(AssertionError):
        UtilGradeProfile([{0: 1}], [{0: 1}], [0, 1], ucounts=[1, 1])


# ---------------------------------------------------------------------------
#  Utility side
# ---------------------------------------------------------------------------

def test_utilities_counts_property(ug):
    utils, counts = ug.utilities_counts
    assert counts == [3, 1]
    assert len(utils) == 2
    assert all(isinstance(u, Utility) for u in utils)


def test_utilities_property_is_voter_expanded(ug):
    us = ug.utilities
    assert len(us) == 4
    # first 3 are the first group, last 1 the second group
    assert [u(1) for u in us] == [5, 5, 5, 1]


def test_has_utility(ug):
    assert ug.has_utility(0) is True
    assert ug.has_utility(2) is True
    assert ug.has_utility(99) is False


def test_util_sum(ug):
    assert ug.util_sum(0) == 10  # 2*3 + 4*1
    assert ug.util_sum(1) == 16  # 5*3 + 1*1
    assert ug.util_sum(2) == 6   # 1*3 + 3*1
    assert ug.util_sum(99) is None


def test_util_avg_is_voter_weighted(ug):
    # Regression (bug shared with UtilityProfile.util_avg): util_avg multiplied each
    # utility by its group count and then took an unweighted mean over groups, giving
    # 5.0 instead of 2.5 for candidate 0. It must be util_sum / (#voters rating it).
    assert ug.util_avg(0) == pytest.approx(2.5)
    assert ug.util_avg(1) == pytest.approx(4.0)
    assert ug.util_avg(2) == pytest.approx(1.5)
    assert ug.util_avg(99) is None
    # must agree with avg_utility_function
    for c in [0, 1, 2]:
        assert ug.util_avg(c) == pytest.approx(ug.avg_utility_function()(c))


def test_util_max_min(ug):
    assert [ug.util_max(c) for c in [0, 1, 2]] == [4, 5, 3]
    assert [ug.util_min(c) for c in [0, 1, 2]] == [2, 1, 1]
    assert ug.util_max(99) is None
    assert ug.util_min(99) is None


def test_sum_utility_function(ug):
    suf = ug.sum_utility_function()
    assert isinstance(suf, Utility)
    assert [suf(c) for c in [0, 1, 2]] == [10, 16, 6]


def test_avg_utility_function(ug):
    auf = ug.avg_utility_function()
    assert isinstance(auf, Utility)
    assert auf(0) == pytest.approx(2.5)
    assert auf(1) == pytest.approx(4.0)
    assert auf(2) == pytest.approx(1.5)


def test_normalize_by_range_preserves_grades_and_bounds_utils(ug):
    n = ug.normalize_by_range()
    assert isinstance(n, UtilGradeProfile)
    assert n.candidates == ug.candidates
    assert n.grades == ug.grades
    assert n.ucounts == ug.ucounts
    # grade side untouched
    assert [n.sum(c) for c in [0, 1, 2]] == [ug.sum(c) for c in [0, 1, 2]]
    # every utility now lies in [0, 1], and each voter attains both endpoints
    for u in n._utilities:
        vals = [u(c) for c in n.candidates]
        assert min(vals) == pytest.approx(0.0)
        assert max(vals) == pytest.approx(1.0)


def test_normalize_by_standard_score_preserves_grades(ug):
    n = ug.normalize_by_standard_score()
    assert isinstance(n, UtilGradeProfile)
    assert n.candidates == ug.candidates
    assert [n.sum(c) for c in [0, 1, 2]] == [ug.sum(c) for c in [0, 1, 2]]
    # standard scores within each voter have mean ~0
    for u in n._utilities:
        assert np.mean([u(c) for c in n.candidates]) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
#  Grade side
# ---------------------------------------------------------------------------

def test_grades_counts_property(ug):
    grades, counts = ug.grades_counts
    assert counts == [3, 1]
    assert len(grades) == 2


def test_grade_functions_is_voter_expanded(ug):
    gs = ug.grade_functions
    assert len(gs) == 4
    assert [g(0) for g in gs] == [2, 2, 2, 0]


def test_has_grade(ug):
    assert ug.has_grade(0) is True
    assert ug.has_grade(99) is False


def test_grade_sum_avg(ug):
    assert [ug.sum(c) for c in [0, 1, 2]] == [6, 5, 2]
    assert ug.avg(0) == pytest.approx(1.5)
    assert ug.avg(1) == pytest.approx(1.25)
    assert ug.avg(2) == pytest.approx(0.5)
    assert ug.sum(99) is None
    assert ug.avg(99) is None


def test_grade_max_min_median(ug):
    assert [ug.max(c) for c in [0, 1, 2]] == [2, 2, 2]
    assert [ug.min(c) for c in [0, 1, 2]] == [0, 1, 0]
    assert [ug.median(c) for c in [0, 1, 2]] == [2, 1, 0]
    assert ug.max(99) is None
    assert ug.min(99) is None
    assert ug.median(99) is None


def test_median_variants_numeric(ug):
    # candidate 0 expands to grades [2,2,2,0] -> sorted [0,2,2,2], two middle = [2,2]
    assert ug.median(0, use_lower=True) == 2
    assert ug.median(0, use_lower=False, use_average=False) == [2, 2]
    assert ug.median(0, use_lower=False, use_average=True) == pytest.approx(2.0)
    # candidate 1 expands to [1,1,1,2] -> two middle = [1,1]
    assert ug.median(1, use_lower=False, use_average=True) == pytest.approx(1.0)


def test_grade_margin_and_proportion(ug):
    assert ug.grade_margin(0, 1) == 2
    assert ug.grade_margin(1, 0) == -2
    # all candidates graded -> extended margin equals the strict margin
    assert ug.grade_margin(0, 1, use_extended=True) == ug.grade_margin(0, 1)
    assert ug.proportion(0, 2) == pytest.approx(0.75)
    assert ug.proportion(0, 0) == pytest.approx(0.25)


def test_sum_and_avg_grade_functions(ug):
    sgf = ug.sum_grade_function()
    agf = ug.avg_grade_function()
    assert isinstance(sgf, _Mapping)
    assert isinstance(agf, _Mapping)
    assert [sgf(c) for c in [0, 1, 2]] == [6, 5, 2]
    assert agf(0) == pytest.approx(1.5)
    assert agf(1) == pytest.approx(1.25)
    assert agf(2) == pytest.approx(0.5)


def test_proportion_with_grade_family(ug):
    # candidate 0 grades: 2 (x3 voters), 0 (x1 voter)
    assert ug.proportion_with_grade(0, 2) == pytest.approx(0.75)
    assert ug.proportion_with_grade(0, 0) == pytest.approx(0.25)
    assert ug.proportion_with_grade(0, 1) == pytest.approx(0.0)

    assert ug.proportion_with_higher_grade(0, 0) == pytest.approx(0.75)
    assert ug.proportion_with_higher_grade(0, 2) == pytest.approx(0.0)

    assert ug.proportion_with_lower_grade(0, 2) == pytest.approx(0.25)
    assert ug.proportion_with_lower_grade(0, 0) == pytest.approx(0.0)


def test_proportion_with_grade_family_partitions(ug):
    # for any candidate/grade the three proportions partition all voters
    for c in ug.candidates:
        for grade in ug.grades:
            total = (
                ug.proportion_with_grade(c, grade)
                + ug.proportion_with_higher_grade(c, grade)
                + ug.proportion_with_lower_grade(c, grade)
            )
            assert total == pytest.approx(1.0)


def test_proportion_with_grade_validates_args(ug):
    with pytest.raises(AssertionError):
        ug.proportion_with_grade(99, 0)
    with pytest.raises(AssertionError):
        ug.proportion_with_grade(0, 99)


def test_approval_scores(approval_ug):
    assert approval_ug.approval_scores() == {0: 2, 1: 1}


def test_approval_scores_requires_binary_grades(ug):
    with pytest.raises(AssertionError):
        ug.approval_scores()


# ---------------------------------------------------------------------------
#  grade_order branch (non-numeric grades)
# ---------------------------------------------------------------------------

def test_ordered_create(ug_ordered):
    assert ug_ordered.use_grade_order is True
    assert ug_ordered.can_sum_grades is False
    assert ug_ordered.num_voters == 3


def test_ordered_max_min_use_grade_order(ug_ordered):
    # candidate 0 graded "good" (x2) and "ok" (x1); order good > ok > bad
    assert ug_ordered.max(0) == "good"
    assert ug_ordered.min(0) == "ok"
    # candidate 1 graded "bad" (x2) and "good" (x1)
    assert ug_ordered.max(1) == "good"
    assert ug_ordered.min(1) == "bad"


def test_ordered_median_use_grade_order(ug_ordered):
    # candidate 0 expands to [good, good, ok] -> median is good
    assert ug_ordered.median(0) == "good"


def test_ordered_proportion_family(ug_ordered):
    # candidate 1: "good" is better than "bad"
    assert ug_ordered.proportion_with_grade(1, "good") == pytest.approx(1 / 3)
    assert ug_ordered.proportion_with_higher_grade(1, "bad") == pytest.approx(1 / 3)
    assert ug_ordered.proportion_with_lower_grade(1, "good") == pytest.approx(2 / 3)


def test_non_summable_grades_raise(ug_ordered):
    with pytest.raises(AssertionError):
        ug_ordered.sum(0)
    with pytest.raises(AssertionError):
        ug_ordered.avg(0)
    with pytest.raises(AssertionError):
        ug_ordered.sum_grade_function()
    with pytest.raises(AssertionError):
        ug_ordered.avg_grade_function()
    with pytest.raises(AssertionError):
        ug_ordered.approval_scores()


# ---------------------------------------------------------------------------
#  Conversions
# ---------------------------------------------------------------------------

def test_to_utility_profile(ug):
    up = ug.to_utility_profile()
    assert isinstance(up, UtilityProfile)
    assert up.num_voters == 4


def test_to_grade_profile(ug):
    gp = ug.to_grade_profile()
    assert isinstance(gp, GradeProfile)
    assert [gp.sum(c) for c in [0, 1, 2]] == [6, 5, 2]


def test_to_ranking_profile(ug):
    rp = ug.to_ranking_profile()
    assert isinstance(rp, ProfileWithTies)
    rmaps = [r.rmap for r in rp._rankings]
    # ranks come from the UTILITIES (higher utility -> better)
    assert {1: 1, 0: 2, 2: 3} in rmaps  # group1: 1 > 0 > 2
    assert {0: 1, 2: 2, 1: 3} in rmaps  # group2: 0 > 2 > 1


def test_to_pref_grade_profile(ug):
    assert isinstance(ug.to_pref_grade_profile(), PrefGradeProfile)


# ---------------------------------------------------------------------------
#  Candidate manipulation
# ---------------------------------------------------------------------------

def test_remove_candidates(ug):
    p = ug.remove_candidates([2])
    assert isinstance(p, UtilGradeProfile)
    assert p.candidates == [0, 1]
    assert p.has_utility(2) is False
    assert p.has_grade(2) is False
    # remaining data preserved
    assert p.util_sum(0) == 10
    assert p.sum(0) == 6
    assert p.num_voters == 4
    assert p.cmap == {0: "0", 1: "1"}


# ---------------------------------------------------------------------------
#  Object protocol / serialization
# ---------------------------------------------------------------------------

def test_eq():
    # NOTE: UtilGradeProfile.__eq__ is positional (it zips the ballots in order),
    # unlike ProfileWithTies/PrefGradeProfile whose __eq__ is order-independent.
    a = UtilGradeProfile(
        [{0: 2, 1: 5}, {0: 4, 1: 1}], [{0: 2, 1: 1}, {0: 0, 1: 2}], [0, 1, 2],
        ucounts=[3, 1],
    )
    a_same = UtilGradeProfile(
        [{0: 2, 1: 5}, {0: 4, 1: 1}], [{0: 2, 1: 1}, {0: 0, 1: 2}], [0, 1, 2],
        ucounts=[3, 1],
    )
    different = UtilGradeProfile(
        [{0: 2, 1: 5}, {0: 4, 1: 1}], [{0: 2, 1: 1}, {0: 0, 1: 2}], [0, 1, 2],
        ucounts=[1, 1],
    )
    assert a == a_same
    assert a != different
    assert a != "not a profile"


def test_add(ug):
    combined = ug + ug
    assert isinstance(combined, UtilGradeProfile)
    assert combined.ucounts == [3, 1, 3, 1]
    assert combined.num_voters == 8
    assert combined.candidates == ug.candidates
    assert combined.util_sum(0) == 20
    assert combined.sum(0) == 12


def test_add_requires_matching_candidates_and_grades(ug):
    other = UtilGradeProfile([{0: 1}], [{0: 1}], [0, 1], candidates=[0])
    with pytest.raises(AssertionError):
        ug + other


def test_as_dict(ug):
    d = ug.as_dict()
    assert d["candidates"] == [0, 1, 2]
    assert d["grades"] == [0, 1, 2]
    assert d["ucounts"] == [3, 1]
    assert d["utilities"] == [{0: 2, 1: 5, 2: 1}, {0: 4, 1: 1, 2: 3}]
    assert d["grade_maps"] == [{0: 2, 1: 1, 2: 0}, {0: 0, 1: 2, 2: 2}]


def test_description_is_a_string(ug):
    desc = ug.description()
    assert isinstance(desc, str)
    assert "UtilGradeProfile" in desc
    assert "ucounts=[3, 1]" in desc


def test_pickle_roundtrip(ug):
    # UtilGradeProfile defines __getstate__/__setstate__, so it IS picklable.
    p2 = pickle.loads(pickle.dumps(ug))
    assert p2 == ug
    assert p2.util_sum(0) == ug.util_sum(0)
    assert p2.sum(0) == ug.sum(0)


# ---------------------------------------------------------------------------
#  Display / visualization (smoke tests)
# ---------------------------------------------------------------------------

def test_display_runs(ug):
    ug.display()
    ug.display(show_totals=True)
    ug.display(cmap={0: "A", 1: "B", 2: "C"})


def test_visualize_grades_runs(ug):
    import matplotlib.pyplot as plt

    ug.visualize_grades()
    plt.close("all")
