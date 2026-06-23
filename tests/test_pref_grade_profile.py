import pickle

import matplotlib
import pytest

from pref_voting.pref_grade_profile import PrefGradeProfile
from pref_voting.rankings import Ranking
from pref_voting.profiles_with_ties import ProfileWithTies
from pref_voting.grade_profiles import GradeProfile
from pref_voting.weighted_majority_graphs import MarginGraph, MajorityGraph, SupportGraph

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Existing tests (kept)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Added coverage
# ---------------------------------------------------------------------------

@pytest.fixture
def pgp():
    # 3 voters rank 0 > 1 > 2, 1 voter ranks 1 > 0 > 2 (ranking side).
    # grade side: 3 voters grade {0:2,1:1,2:0}, 1 voter grades {0:1,1:2,2:2}.
    return PrefGradeProfile(
        [{0: 1, 1: 2, 2: 3}, {1: 1, 0: 2, 2: 3}],
        [{0: 2, 1: 1, 2: 0}, {0: 1, 1: 2, 2: 2}],
        [0, 1, 2],
        rcounts=[3, 1],
    )


@pytest.fixture
def approval_pgp():
    return PrefGradeProfile(
        [{0: 1, 1: 2}, {1: 1, 0: 2}],
        [{0: 1, 1: 0}, {0: 0, 1: 1}],
        [0, 1],
        rcounts=[2, 1],
    )


def test_create(pgp):
    assert pgp.candidates == [0, 1, 2]
    assert pgp.num_cands == 3
    assert pgp.num_voters == 4


# --- ranking side ---

def test_support_margin(pgp):
    assert pgp.support(0, 1) == 3
    assert pgp.support(1, 0) == 1
    assert pgp.margin(0, 1) == 2
    assert pgp.margin(0, 2) == 4
    assert pgp.margin(1, 2) == 4
    assert pgp.majority_prefers(0, 1)
    assert not pgp.is_tied(0, 1)


def test_dominators_dominates(pgp):
    assert pgp.dominators(2) == [0, 1]
    assert pgp.dominates(0) == [1, 2]


def test_condorcet(pgp):
    assert pgp.condorcet_winner() == 0
    assert pgp.condorcet_loser() == 2
    assert pgp.weak_condorcet_winner() == [0]


def test_copeland_plurality_borda(pgp):
    assert pgp.copeland_scores() == {0: 2.0, 1: 0.0, 2: -2.0}
    assert pgp.plurality_scores() == {0: 3, 1: 1, 2: 0}
    assert pgp.borda_scores() == {0: 6, 1: 2, 2: -8}


def test_strict_maj_size(pgp):
    assert pgp.strict_maj_size() == 3


def test_count_methods(pgp):
    assert pgp.num_empty_rankings() == 0
    assert pgp.num_linear_orders() == 4
    assert pgp.num_rankings_with_ties() == 0
    assert pgp.num_truncated_linear_orders() == 0
    assert pgp.num_ranking_each_candidate() == {0: 4, 1: 4, 2: 4}


def test_cycles_and_uniquely_weighted(pgp):
    assert pgp.cycles() == []
    assert pgp.is_uniquely_weighted() is False


# --- grade side ---

def test_grade_sum_avg(pgp):
    assert [pgp.sum(c) for c in [0, 1, 2]] == [7, 5, 2]
    # avg is correctly voter-weighted (uses the count-expanded grade_functions)
    assert pgp.avg(0) == pytest.approx(1.75)
    assert pgp.avg(1) == pytest.approx(1.25)
    assert pgp.avg(2) == pytest.approx(0.5)


def test_grade_max_min_median(pgp):
    assert [pgp.max(c) for c in [0, 1, 2]] == [2, 2, 2]
    assert [pgp.min(c) for c in [0, 1, 2]] == [1, 1, 0]
    assert [pgp.median(c) for c in [0, 1, 2]] == [2, 1, 0]


def test_grade_margin_and_proportion(pgp):
    assert pgp.grade_margin(0, 1) == 2
    assert pgp.proportion(0, 2) == pytest.approx(0.75)
    assert pgp.proportion_with_higher_grade(0, 1) == pytest.approx(0.75)
    assert pgp.proportion_with_lower_grade(2, 1) == pytest.approx(0.75)


def test_approval_scores(approval_pgp):
    assert approval_pgp.approval_scores() == {0: 2, 1: 1}


def test_approval_scores_requires_binary_grades(pgp):
    with pytest.raises(AssertionError):
        pgp.approval_scores()


# --- conversions ---

def test_to_ranking_profile(pgp):
    rp = pgp.to_ranking_profile()
    assert isinstance(rp, ProfileWithTies)
    assert rp.rcounts == [3, 1]
    assert rp.margin(0, 1) == pgp.margin(0, 1)


def test_to_grade_profile(pgp):
    gp = pgp.to_grade_profile()
    assert isinstance(gp, GradeProfile)
    assert gp.gcounts == [3, 1]
    assert gp.margin(0, 1) == pgp.grade_margin(0, 1)


# --- object protocol / display ---

def test_eq_ignores_ballot_order(pgp):
    reordered = PrefGradeProfile(
        [{1: 1, 0: 2, 2: 3}, {0: 1, 1: 2, 2: 3}],
        [{0: 1, 1: 2, 2: 2}, {0: 2, 1: 1, 2: 0}],
        [0, 1, 2],
        rcounts=[1, 3],
    )
    assert pgp == reordered


def test_remove_candidates(pgp):
    new = pgp.remove_candidates([2])
    assert isinstance(new, PrefGradeProfile)
    assert new.candidates == [0, 1]


def test_display_runs(pgp):
    pgp.display()
    pgp.display(show_totals=True)


def test_description_is_str(pgp):
    assert isinstance(pgp.description(), str)


def test_pickle_roundtrip(pgp):
    # PrefGradeProfile must be picklable. It currently fails because
    # cand_to_cindex / cindex_to_cand are lambdas -- add __getstate__/__setstate__
    # (mirroring Profile / the ProfileWithTies fix) to make it work.
    p2 = pickle.loads(pickle.dumps(pgp))
    assert p2 == pgp
    assert p2.margin(0, 1) == pgp.margin(0, 1)
    assert p2.cand_to_cindex(1) == pgp.cand_to_cindex(1)
