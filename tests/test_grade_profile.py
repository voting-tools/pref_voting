import matplotlib
import numpy as np
import pytest

from pref_voting.grade_profiles import GradeProfile
from pref_voting.profiles_with_ties import ProfileWithTies

matplotlib.use("Agg")


@pytest.fixture
def gp():
    # numeric grades 0..2; 3 voters grade {0:2,1:0,2:1}, 1 voter grades {0:0,1:2,2:2}
    return GradeProfile(
        [{0: 2, 1: 0, 2: 1}, {0: 0, 1: 2, 2: 2}],
        [0, 1, 2],
        gcounts=[3, 1],
        candidates=[0, 1, 2],
    )


@pytest.fixture
def approval_gp():
    return GradeProfile(
        [{0: 1, 1: 0, 2: 1}, {0: 0, 1: 1, 2: 1}],
        [0, 1],
        gcounts=[3, 1],
        candidates=[0, 1, 2],
    )


@pytest.fixture
def grade_order_gp():
    # non-numeric grades with an explicit order (A best, then B, then C)
    return GradeProfile(
        [{0: "A", 1: "C", 2: "B"}, {0: "C", 1: "A", 2: "A"}],
        ["A", "B", "C"],
        gcounts=[3, 1],
        candidates=[0, 1, 2],
        grade_order=["A", "B", "C"],
    )


def test_create(gp):
    assert gp.candidates == [0, 1, 2]
    assert gp.num_cands == 3
    assert gp.num_voters == 4
    assert gp.can_sum_grades is True


def test_has_grade(gp):
    assert gp.has_grade(0)
    gp2 = GradeProfile([{0: 1}], [0, 1], candidates=[0, 1])
    assert gp2.has_grade(0)
    assert not gp2.has_grade(1)


def test_sum(gp):
    assert gp.sum(0) == 6  # 2*3 + 0*1
    assert gp.sum(1) == 2  # 0*3 + 2*1
    assert gp.sum(2) == 5  # 1*3 + 2*1


def test_avg_is_voter_weighted(gp):
    # avg uses the count-expanded grade_functions, so it is correctly per-voter
    assert gp.avg(0) == pytest.approx(1.5)   # mean of [2,2,2,0]
    assert gp.avg(1) == pytest.approx(0.5)   # mean of [0,0,0,2]
    assert gp.avg(2) == pytest.approx(1.25)  # mean of [1,1,1,2]


def test_max_min(gp):
    assert [gp.max(c) for c in [0, 1, 2]] == [2, 2, 2]
    assert [gp.min(c) for c in [0, 1, 2]] == [0, 0, 1]


def test_median(gp):
    # default use_lower=True -> lower median
    assert [gp.median(c) for c in [0, 1, 2]] == [2, 0, 1]


def test_margin(gp):
    assert gp.margin(0, 2) == 2
    assert gp.margin(2, 0) == -2
    assert gp.margin(0, 1) == 2


def test_proportion(gp):
    assert gp.proportion(0, 2) == pytest.approx(0.75)   # 3 of 4 voters grade 0 with a 2
    assert gp.proportion(0, 0) == pytest.approx(0.25)
    assert gp.proportion_with_higher_grade(0, 0) == pytest.approx(0.75)
    assert gp.proportion_with_lower_grade(2, 2) == pytest.approx(0.75)


def test_sum_and_avg_grade_function(gp):
    sum_fn = gp.sum_grade_function()
    avg_fn = gp.avg_grade_function()
    for c in [0, 1, 2]:
        assert sum_fn(c) == gp.sum(c)
        assert avg_fn(c) == pytest.approx(gp.avg(c))


def test_to_ranking_profile(gp):
    rp = gp.to_ranking_profile()
    assert isinstance(rp, ProfileWithTies)
    assert rp.candidates == [0, 1, 2]
    rmaps = [r.rmap for r in rp._rankings]
    assert {0: 1, 2: 2, 1: 3} in rmaps   # group1: 0 > 2 > 1
    assert {1: 1, 2: 1, 0: 2} in rmaps   # group2: {1,2} tied > 0


def test_approval_scores(approval_gp):
    assert approval_gp.approval_scores() == {0: 3, 1: 1, 2: 4}


def test_approval_scores_requires_binary_grades(gp):
    # approval is only defined for grades {0, 1}
    with pytest.raises(AssertionError):
        gp.approval_scores()


def test_grade_order(grade_order_gp):
    go = grade_order_gp
    assert go.can_sum_grades is False
    assert go.use_grade_order is True
    assert [go.max(c) for c in [0, 1, 2]] == ["A", "A", "A"]
    assert [go.min(c) for c in [0, 1, 2]] == ["C", "C", "B"]
    assert [go.median(c) for c in [0, 1, 2]] == ["A", "C", "B"]
    # non-numeric grades cannot be summed/averaged
    with pytest.raises(AssertionError):
        go.sum(0)
    with pytest.raises(AssertionError):
        go.avg(0)


def test_display_runs(gp):
    gp.display()


def test_display_show_totals_runs(gp):
    gp.display(show_totals=True)


def test_write_from_string_roundtrip(gp):
    s = gp.write()
    gp2 = GradeProfile.from_string(s)
    assert gp2.candidates == gp.candidates
    assert gp2.num_voters == gp.num_voters
    for c in [0, 1, 2]:
        assert gp2.sum(c) == gp.sum(c)
        assert gp2.max(c) == gp.max(c)
        assert gp2.median(c) == gp.median(c)


def test_write_from_string_roundtrip_string_grades(grade_order_gp):
    # string grades + grade_order must survive the round-trip (the order is
    # serialized, and values are recovered with their original type, not float()).
    s = grade_order_gp.write()
    gp2 = GradeProfile.from_string(s)
    assert gp2.grades == grade_order_gp.grades
    for c in [0, 1, 2]:
        assert gp2.max(c) == grade_order_gp.max(c)
        assert gp2.min(c) == grade_order_gp.min(c)
        assert gp2.median(c) == grade_order_gp.median(c)
        assert gp2.margin(0, 1) == grade_order_gp.margin(0, 1)
