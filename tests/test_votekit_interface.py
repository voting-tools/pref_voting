"""
Tests for pref_voting.interfaces.votekit_interface.
"""

import pytest

votekit = pytest.importorskip("votekit")

from votekit.ballot import Ballot
from votekit.pref_profile import PreferenceProfile

from pref_voting.profiles import Profile
from pref_voting.profiles_with_ties import ProfileWithTies
from pref_voting.grade_profiles import GradeProfile
from pref_voting.interfaces.votekit_interface import (
    to_votekit_profile,
    votekit_to_profile_with_ties,
    grade_profile_to_votekit,
    votekit_to_grade_profile,
)


@pytest.fixture
def linear_profile():
    return Profile(
        [[0, 1, 2], [2, 1, 0], [1, 0, 2]],
        rcounts=[2, 3, 1],
        cmap={0: "a", 1: "b", 2: "c"},
    )


@pytest.fixture
def profile_with_ties():
    return ProfileWithTies(
        [{"a": 1, "b": 1, "c": 2}, {"c": 1}, {"b": 1, "a": 2}],
        rcounts=[2, 1, 3],
        candidates=["a", "b", "c"],
    )


def test_profile_to_votekit(linear_profile):
    vk_profile = to_votekit_profile(linear_profile)
    assert sorted(vk_profile.candidates) == ["a", "b", "c"]
    assert vk_profile.total_ballot_wt == 6
    expected = {
        (("a",), ("b",), ("c",)): 2,
        (("c",), ("b",), ("a",)): 3,
        (("b",), ("a",), ("c",)): 1,
    }
    actual = {
        tuple(tuple(sorted(pos)) for pos in b.ranking): b.weight
        for b in vk_profile.ballots
    }
    assert actual == expected


def test_linear_profile_round_trip(linear_profile):
    vk_profile = to_votekit_profile(linear_profile)
    prof2 = votekit_to_profile_with_ties(vk_profile).to_linear_profile()
    assert prof2 is not None
    original = sorted(
        (tuple(r), c)
        for r, c in zip(linear_profile.ranking_types, linear_profile.counts)
    )
    # candidate i of linear_profile has name cmap[i]; prof2 candidates are the
    # indices of the sorted names, and names a < b < c match indices 0 < 1 < 2
    round_tripped = sorted(
        (tuple(r), c) for r, c in zip(prof2.ranking_types, prof2.counts)
    )
    assert original == round_tripped
    assert [prof2.cmap[c] for c in prof2.candidates] == ["a", "b", "c"]


def test_profile_with_ties_round_trip(profile_with_ties):
    vk_profile = to_votekit_profile(profile_with_ties)
    prof2 = votekit_to_profile_with_ties(vk_profile)
    assert prof2.candidates == profile_with_ties.candidates
    rmaps1, rcounts1 = profile_with_ties.rankings_as_dicts_counts
    rmaps2, rcounts2 = prof2.rankings_as_dicts_counts
    assert sorted(zip([sorted(r.items()) for r in rmaps1], rcounts1)) == sorted(
        zip([sorted(r.items()) for r in rmaps2], rcounts2)
    )


def test_rank_gaps_are_compressed():
    prof = ProfileWithTies([{"a": 1, "b": 5}], candidates=["a", "b"])
    vk_profile = to_votekit_profile(prof)
    prof2 = votekit_to_profile_with_ties(vk_profile)
    rmaps, _ = prof2.rankings_as_dicts_counts
    assert rmaps == [{"a": 1, "b": 2}]


def test_skipped_positions_preserved():
    vk_profile = PreferenceProfile(
        ballots=(Ballot(ranking=[{"a"}, frozenset(), {"b"}]),),
        candidates=("a", "b"),
        max_ranking_length=3,
    )
    prof = votekit_to_profile_with_ties(vk_profile)
    rmaps, _ = prof.rankings_as_dicts_counts
    assert rmaps == [{"a": 1, "b": 3}]


def test_integer_valued_float_weights():
    vk_profile = PreferenceProfile(
        ballots=(
            Ballot(ranking=[{"a"}, {"b"}], weight=2.0),
            Ballot(ranking=[{"b"}, {"a"}], weight=1.0),
        ),
        candidates=("a", "b"),
    )
    prof = votekit_to_profile_with_ties(vk_profile)
    assert sorted(prof.rcounts) == [1, 2]
    assert all(isinstance(c, int) for c in prof.rcounts)
    assert prof.num_voters == 3


def test_fractional_weights_raise_without_scaling():
    vk_profile = PreferenceProfile(
        ballots=(Ballot(ranking=[{"a"}, {"b"}], weight=0.5),),
        candidates=("a", "b"),
    )
    with pytest.raises(ValueError, match="scale_weights"):
        votekit_to_profile_with_ties(vk_profile)


def test_fractional_weights_scaled():
    vk_profile = PreferenceProfile(
        ballots=(
            Ballot(ranking=[{"a"}, {"b"}], weight=0.5),
            Ballot(ranking=[{"b"}, {"a"}], weight=1.5),
        ),
        candidates=("a", "b"),
    )
    prof = votekit_to_profile_with_ties(vk_profile, scale_weights=True)
    rmaps, rcounts = prof.rankings_as_dicts_counts
    assert dict(zip([tuple(sorted(r.items())) for r in rmaps], rcounts)) == {
        (("a", 1), ("b", 2)): 1,
        (("a", 2), ("b", 1)): 3,
    }


def test_grade_profile_to_votekit_scores():
    gprof = GradeProfile(
        [{"x": 3, "y": 1}, {"x": 2, "y": 2, "z": 0}],
        [0, 1, 2, 3],
        gcounts=[2, 1],
        candidates=["x", "y", "z"],
    )
    vk_profile = grade_profile_to_votekit(gprof)
    assert sorted(vk_profile.candidates) == ["x", "y", "z"]
    scores = sorted(
        (tuple(sorted(b.scores.items())), b.weight) for b in vk_profile.ballots
    )
    # note: votekit does not store scores of 0, so z is unscored on the second ballot
    assert scores == [
        ((("x", 2.0), ("y", 2.0)), 1),
        ((("x", 3.0), ("y", 1.0)), 2),
    ]


def test_grade_profile_round_trip():
    gprof = GradeProfile(
        [{"x": 3, "y": 1}, {"x": 2, "y": 2}],
        [1, 2, 3],
        gcounts=[2, 1],
        candidates=["x", "y", "z"],
    )
    gprof2 = votekit_to_grade_profile(
        grade_profile_to_votekit(gprof), grades=[1, 2, 3]
    )
    assert gprof2.candidates == ["x", "y", "z"]
    assert gprof2.grades == [1, 2, 3]
    grades1 = sorted(
        (tuple(sorted(g.as_dict().items())), c) for g, c in zip(*gprof.grades_counts)
    )
    grades2 = sorted(
        (tuple(sorted(g.as_dict().items())), c) for g, c in zip(*gprof2.grades_counts)
    )
    assert grades1 == grades2


def test_votekit_to_grade_profile_default_grades():
    vk_profile = PreferenceProfile(
        ballots=(
            Ballot(scores={"x": 4, "y": 2}),
            Ballot(scores={"x": 2, "z": 1}),
        ),
        candidates=("x", "y", "z"),
    )
    gprof = votekit_to_grade_profile(vk_profile)
    assert gprof.grades == [1, 2, 4]
    assert all(isinstance(g, int) for g in gprof.grades)


def test_non_numeric_grades_raise():
    gprof = GradeProfile(
        [{"x": "A", "y": "B"}],
        ["A", "B", "C"],
        candidates=["x", "y"],
        grade_order=["A", "B", "C"],
    )
    with pytest.raises(ValueError, match="numeric"):
        grade_profile_to_votekit(gprof)


def test_wrong_ballot_types_raise():
    score_profile = PreferenceProfile(
        ballots=(Ballot(scores={"x": 1}),), candidates=("x",)
    )
    with pytest.raises(ValueError, match="votekit_to_grade_profile"):
        votekit_to_profile_with_ties(score_profile)

    ranked_profile = PreferenceProfile(
        ballots=(Ballot(ranking=[{"x"}]),), candidates=("x",)
    )
    with pytest.raises(ValueError, match="votekit_to_profile_with_ties"):
        votekit_to_grade_profile(ranked_profile)


def test_to_votekit_profile_wrong_type_raises():
    gprof = GradeProfile([{"x": 1}], [0, 1], candidates=["x"])
    with pytest.raises(TypeError):
        to_votekit_profile(gprof)


def test_convenience_methods(linear_profile, profile_with_ties):
    assert to_votekit_profile(linear_profile).df.equals(
        linear_profile.to_votekit().df
    )
    assert to_votekit_profile(profile_with_ties).df.equals(
        profile_with_ties.to_votekit().df
    )
    vk_profile = to_votekit_profile(profile_with_ties)
    prof = ProfileWithTies.from_votekit(vk_profile)
    assert isinstance(prof, ProfileWithTies)

    gprof = GradeProfile([{"x": 1, "y": 0}], [0, 1], candidates=["x", "y"])
    vk_scores = gprof.to_votekit()
    assert isinstance(GradeProfile.from_votekit(vk_scores, grades=[0, 1]), GradeProfile)


def test_votekit_generator_to_pref_voting_methods():
    from votekit.ballot_generator import ic_profile_generator
    from pref_voting.scoring_methods import borda
    from pref_voting.margin_based_methods import split_cycle

    vk_profile = ic_profile_generator(
        candidates=["W", "X", "Y", "Z"], number_of_ballots=50
    )
    prof = votekit_to_profile_with_ties(vk_profile)
    assert prof.num_voters == 50
    for vm in [borda, split_cycle]:
        winners = vm(prof)
        assert len(winners) > 0
        assert all(w in prof.candidates for w in winners)


def test_pref_voting_generator_to_votekit_election():
    from pref_voting.generate_profiles import generate_profile
    from votekit.elections import Borda

    prof = generate_profile(4, 20)
    vk_profile = to_votekit_profile(prof)
    election = Borda(vk_profile, n_seats=1)
    elected = [c for cand_set in election.get_elected() for c in cand_set]
    assert len(elected) >= 1
    assert all(c in vk_profile.candidates for c in elected)
