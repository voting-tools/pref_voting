import pytest

from pref_voting.pref_grade_profile import PrefGradeProfile
from pref_voting.rankings import Ranking
from pref_voting.weighted_majority_graphs import MarginGraph, MajorityGraph, SupportGraph


@pytest.fixture
def paired_pref_grade_profile():
    return PrefGradeProfile(
        [
            {0: 1, 1: 2, 2: 3},
            {1: 1, 2: 1, 0: 2},
        ],
        [
            {0: 5, 1: 3, 2: 1},
            {0: 2, 1: 4, 2: 4},
        ],
        [1, 2, 3, 4, 5],
        rcounts=[2, 3],
    )


def test_pref_grade_profile_expands_rankings_and_grades_in_lockstep(
    paired_pref_grade_profile,
):
    paired_ballots = [
        (ranking.rmap, grade.as_dict())
        for ranking, grade in zip(
            paired_pref_grade_profile.rankings,
            paired_pref_grade_profile.grade_functions,
        )
    ]

    assert len(paired_pref_grade_profile.rankings) == 5
    assert len(paired_pref_grade_profile.grade_functions) == 5
    assert paired_ballots == [
        ({0: 1, 1: 2, 2: 3}, {0: 5, 1: 3, 2: 1}),
        ({0: 1, 1: 2, 2: 3}, {0: 5, 1: 3, 2: 1}),
        ({1: 1, 2: 1, 0: 2}, {0: 2, 1: 4, 2: 4}),
        ({1: 1, 2: 1, 0: 2}, {0: 2, 1: 4, 2: 4}),
        ({1: 1, 2: 1, 0: 2}, {0: 2, 1: 4, 2: 4}),
    ]


def test_pref_grade_profile_pairwise_margins_use_rankings_not_grades():
    pgprof = PrefGradeProfile(
        [{0: 1, 1: 2}, {1: 1, 0: 2}],
        [{0: 0, 1: 1}, {0: 0, 1: 1}],
        [0, 1],
        rcounts=[3, 1],
    )

    assert pgprof.support(0, 1) == 3
    assert pgprof.support(1, 0) == 1
    assert pgprof.margin(0, 1) == 2
    assert pgprof.margin(1, 0) == -2
    assert pgprof.majority_prefers(0, 1)
    assert not pgprof.majority_prefers(1, 0)

    # The grade ballots disagree with the ranking ballots here, so this stays separate.
    assert pgprof.grade_margin(0, 1) == -4


def test_pref_grade_profile_graphs_and_conversions_preserve_ranking_side():
    pgprof = PrefGradeProfile(
        [Ranking({0: 1, 1: 2}), {1: 1, 0: 2}],
        [{0: 2, 1: 0}, {0: 0, 1: 2}],
        [0, 1, 2],
        rcounts=[2, 1],
    )

    margin_graph = pgprof.margin_graph()
    support_graph = pgprof.support_graph()
    majority_graph = pgprof.majority_graph()
    ranking_profile = pgprof.to_ranking_profile()
    grade_profile = pgprof.to_grade_profile()

    assert isinstance(margin_graph, MarginGraph)
    assert isinstance(support_graph, SupportGraph)
    assert isinstance(majority_graph, MajorityGraph)

    assert margin_graph.margin(0, 1) == 1
    assert support_graph.support(0, 1) == 2
    assert majority_graph.majority_prefers(0, 1)

    assert ranking_profile.rcounts == [2, 1]
    assert ranking_profile.margin(0, 1) == pgprof.margin(0, 1)

    assert grade_profile.gcounts == [2, 1]
    assert grade_profile.margin(0, 1) == pgprof.grade_margin(0, 1)
