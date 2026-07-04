"""
Tests for pref_voting.interfaces.abcvoting_interface.
"""

import pytest

abcvoting = pytest.importorskip("abcvoting")

from abcvoting import abcrules
from abcvoting.preferences import Profile as AbcvotingProfile, Voter

from pref_voting.grade_profiles import GradeProfile
from pref_voting.utility_profiles import UtilityProfile
from pref_voting.interfaces.abcvoting_interface import (
    to_abcvoting_profile,
    abcvoting_to_grade_profile,
)


@pytest.fixture
def approval_profile():
    return GradeProfile(
        [
            {"x": 1, "y": 0, "z": 1},
            {"x": 0, "y": 1, "z": 0},
            {"x": 1, "y": 1, "z": 1},
        ],
        [0, 1],
        gcounts=[3, 2, 1],
        candidates=["x", "y", "z"],
    )


def test_to_abcvoting_profile(approval_profile):
    abc_profile = to_abcvoting_profile(approval_profile)
    assert abc_profile.num_cand == 3
    assert abc_profile.cand_names == ["x", "y", "z"]
    voters = sorted((tuple(sorted(v.approved)), v.weight) for v in abc_profile)
    assert voters == [((0, 1, 2), 1), ((0, 2), 3), ((1,), 2)]


def test_abcvoting_round_trip(approval_profile):
    gprof2 = abcvoting_to_grade_profile(to_abcvoting_profile(approval_profile))
    assert sorted(gprof2.grades) == [0, 1]
    assert [gprof2.cmap[c] for c in gprof2.candidates] == ["x", "y", "z"]
    # compare ballots by candidate name
    def named_ballots(gprof):
        return sorted(
            (
                tuple(sorted((gprof.cmap[c], g) for c, g in gfnc.as_dict().items())),
                count,
            )
            for gfnc, count in zip(*gprof.grades_counts)
        )

    assert named_ballots(approval_profile) == named_ballots(gprof2)


def test_abcvoting_to_grade_profile():
    abc_profile = AbcvotingProfile(3, cand_names=["a", "b", "c"])
    abc_profile.add_voter(Voter([0, 1], weight=2))
    abc_profile.add_voter(Voter([2]))
    gprof = abcvoting_to_grade_profile(abc_profile)
    assert gprof.candidates == [0, 1, 2]
    assert gprof.num_voters == 3
    ballots = sorted(
        (tuple(sorted(g.as_dict().items())), c) for g, c in zip(*gprof.grades_counts)
    )
    assert ballots == [
        (((0, 0), (1, 0), (2, 1)), 1),
        (((0, 1), (1, 1), (2, 0)), 2),
    ]
    assert gprof.approval_scores() == {0: 2, 1: 2, 2: 1}


def test_non_dichotomous_grades_raise():
    gprof = GradeProfile([{"x": 2, "y": 1}], [0, 1, 2], candidates=["x", "y"])
    with pytest.raises(ValueError, match="grades 0 and 1"):
        to_abcvoting_profile(gprof)


def test_fractional_voter_weights():
    abc_profile = AbcvotingProfile(2)
    abc_profile.add_voter(Voter([0], weight=0.5))
    abc_profile.add_voter(Voter([1], weight=1.5))
    with pytest.raises(ValueError, match="scale_weights"):
        abcvoting_to_grade_profile(abc_profile)
    gprof = abcvoting_to_grade_profile(abc_profile, scale_weights=True)
    assert sorted(gprof.gcounts) == [1, 3]


def test_abcvoting_rules_on_converted_profile(approval_profile):
    abc_profile = to_abcvoting_profile(approval_profile)
    for rule_id in ["av", "pav", "seqphragmen", "equal-shares"]:
        committees = abcrules.compute(
            rule_id, abc_profile, committeesize=2, resolute=True
        )
        assert len(committees) == 1
        assert len(committees[0]) == 2


def test_utility_profile_to_abcvoting():
    uprof = UtilityProfile(
        [{0: 0.9, 1: 0.2, 2: 0.7}, {0: 0.1, 1: 0.8, 2: 0.6}]
    )
    gprof = uprof.to_approval_profile(prob_to_cont_approving=1.0)
    abc_profile = to_abcvoting_profile(gprof)
    assert abc_profile.num_cand == 3
    assert len(abc_profile) == 2


def test_convenience_methods(approval_profile):
    abc_profile = approval_profile.to_abcvoting()
    assert abc_profile.num_cand == 3
    gprof = GradeProfile.from_abcvoting(abc_profile)
    assert isinstance(gprof, GradeProfile)
    assert sorted(gprof.grades) == [0, 1]
