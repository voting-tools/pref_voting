import pytest

from pref_voting.grade_profiles import GradeProfile
from pref_voting.grade_methods import (
    score_voting,
    approval,
    dis_and_approval,
    cumulative_voting,
    star,
    greatest_median,
    majority_judgement,
    tiebreaker_diff,
    tiebreaker_relative_shares,
    tiebreaker_normalized_difference,
    tiebreaker_majority_judgement,
)


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def approval_profile():
    # {0,1} grades; sums: 0 -> 2, 1 -> 2, 2 -> 1
    return GradeProfile(
        [{0: 1, 1: 0, 2: 0}, {0: 1, 1: 1, 2: 0}, {0: 0, 1: 1, 2: 1}], [0, 1]
    )

@pytest.fixture
def score_profile():
    # 0-5 grades; sums: 0 -> 10, 1 -> 12, 2 -> 6
    return GradeProfile(
        [{0: 5, 1: 4, 2: 0}, {0: 5, 1: 3, 2: 1}, {0: 0, 1: 5, 2: 5}],
        [0, 1, 2, 3, 4, 5],
    )

@pytest.fixture
def disapproval_profile():
    # {-1,0,1} grades; sums: 0 -> 1, 1 -> 0, 2 -> 0
    return GradeProfile(
        [{0: 1, 1: -1, 2: 0}, {0: 1, 1: 0, 2: -1}, {0: -1, 1: 1, 2: 1}], [-1, 0, 1]
    )


# ---------------------------------------------------------------------------
#  score_voting
# ---------------------------------------------------------------------------

def test_score_voting_sum(score_profile):
    assert score_voting(score_profile, evaluation_method="sum") == [1]

def test_score_voting_mean(score_profile):
    assert score_voting(score_profile, evaluation_method="mean") == [1]

def test_score_voting_median(score_profile):
    assert score_voting(score_profile, evaluation_method="median") == [0]

def test_score_voting_curr_cands(score_profile):
    # restrict to {0, 2}: sums 10 vs 6 -> 0 wins
    assert score_voting(score_profile, curr_cands=[0, 2]) == [0]

def test_score_voting_all_abstain_returns_all():
    # no candidate graded by anyone (and no ungraded_score default) -> all tied
    gp = GradeProfile([{}, {}], [0, 1], candidates=[0, 1])
    assert score_voting(gp) == [0, 1]


# ---------------------------------------------------------------------------
#  approval
# ---------------------------------------------------------------------------

def test_approval_winner(approval_profile):
    assert approval(approval_profile) == [0, 1]  # tie at sum 2

def test_approval_curr_cands(approval_profile):
    assert approval(approval_profile, curr_cands=[1, 2]) == [1]

def test_approval_treats_ungraded_as_zero():
    # voters submit only their approvals; candidate 2 is approved by nobody
    gp = GradeProfile([{0: 1}, {0: 1}, {1: 1}], [0, 1], candidates=[0, 1, 2])
    assert approval(gp) == [0]

def test_approval_grades_only_ones():
    # ballots use ONLY the grade 1 (approvals); the rest are ungraded -> count as 0
    gp = GradeProfile([{0: 1}, {1: 1}, {0: 1}], [1], candidates=[0, 1, 2])
    assert approval(gp) == [0]

def test_approval_all_abstain_is_all_tied():
    # nobody approves anyone -> all candidates tie at 0 (no crash)
    gp = GradeProfile([{}, {}], [0, 1], candidates=[0, 1])
    assert approval(gp) == [0, 1]

def test_approval_requires_binary_grades(score_profile):
    with pytest.raises(AssertionError):
        approval(score_profile)


# ---------------------------------------------------------------------------
#  dis_and_approval
# ---------------------------------------------------------------------------

def test_dis_and_approval_winner(disapproval_profile):
    assert dis_and_approval(disapproval_profile) == [0]

def test_dis_and_approval_ungraded_beats_all_negative():
    # candidate 0 is disapproved by everyone (-3); candidate 1 is graded by nobody (0).
    # the ungraded candidate (score 0) should beat the all-negative one.
    gp = GradeProfile([{0: -1}, {0: -1}, {0: -1}], [-1, 1], candidates=[0, 1])
    assert dis_and_approval(gp) == [1]

def test_dis_and_approval_grades_only_neg_one_and_one():
    # ballots use only -1 and 1 (no explicit 0); ungraded counts as neutral 0
    gp = GradeProfile([{0: 1, 1: -1}, {0: 1, 1: -1}, {0: -1, 1: 1}], [-1, 1],
                      candidates=[0, 1, 2])
    assert dis_and_approval(gp) == [0]

def test_dis_and_approval_requires_subset_of_neg_zero_one(score_profile):
    # grades 0..5 are not a subset of {-1, 0, 1} -> must raise
    with pytest.raises(AssertionError):
        dis_and_approval(score_profile)


# ---------------------------------------------------------------------------
#  cumulative_voting
#  (NOTE: the assert checks that the GRADE SET sums to max_total_grades, so it
#   only passes when max_total_grades == 1, i.e. grades == [0, 1].)
# ---------------------------------------------------------------------------

def test_cumulative_voting_max_one():
    gp = GradeProfile([{0: 1, 1: 0}, {0: 1, 1: 0}, {0: 0, 1: 1}], [0, 1])
    assert cumulative_voting(gp, max_total_grades=1) == [0]

def test_cumulative_voting_valid_ballots_sum_to_max():
    # Each ballot distributes max_total_grades (=5) points across the candidates.
    # candidate sums: 0 -> 8, 1 -> 4, 2 -> 3, so 0 should win.
    # BUG: the current assert checks that the grade SET [0..5] sums to 5 (it sums to
    # 15), so this valid cumulative-vote profile is wrongly rejected. The check should
    # instead require each voter's BALLOT to sum to max_total_grades.
    gp = GradeProfile(
        [{0: 3, 1: 2, 2: 0}, {0: 5, 1: 0, 2: 0}, {0: 0, 1: 2, 2: 3}],
        [0, 1, 2, 3, 4, 5],
    )
    assert cumulative_voting(gp, max_total_grades=5) == [0]

def test_cumulative_voting_rejects_ballots_not_summing_to_max(score_profile):
    # score_profile ballots sum to 9, 9, 10 -- not the required 5 -- so this must raise
    with pytest.raises(AssertionError):
        cumulative_voting(score_profile)  # default max_total_grades=5


# ---------------------------------------------------------------------------
#  STAR
# ---------------------------------------------------------------------------

def test_star_winner(score_profile):
    # top two by sum are 1 (12) and 0 (10); runoff is won by 0
    assert star(score_profile) == [0]

def test_star_single_candidate(score_profile):
    assert star(score_profile, curr_cands=[0]) == [0]

def test_star_requires_six_grades(approval_profile):
    with pytest.raises(AssertionError):
        star(approval_profile)

def test_star_tie_for_first_runoff_has_winner():
    # 0 and 1 tie for the top sum (10 each); the runoff between them is won by 0
    gp = GradeProfile([{0: 5, 1: 3}, {0: 5, 1: 3}, {0: 0, 1: 4}], [0, 1, 2, 3, 4, 5])
    assert [gp.sum(c) for c in [0, 1]] == [10, 10]
    assert star(gp) == [0]

def test_star_tie_for_first_runoff_tied():
    # 0 and 1 tie for top sum AND tie head-to-head -> both are winners
    gp = GradeProfile([{0: 5, 1: 3}, {0: 3, 1: 5}], [0, 1, 2, 3, 4, 5])
    assert star(gp) == [0, 1]


# ---------------------------------------------------------------------------
#  greatest_median / majority_judgement
# ---------------------------------------------------------------------------

def test_greatest_median_unique():
    # candidate 0 has a strictly greatest median (5 vs 1 vs 0)
    gp = GradeProfile(
        [{0: 5, 1: 1, 2: 0}, {0: 5, 1: 1, 2: 0}, {0: 5, 1: 2, 2: 0}],
        [0, 1, 2, 3, 4, 5],
    )
    assert greatest_median(gp) == [0]

def test_greatest_median_tiebreak(score_profile):
    # 0 and 1 both have median 4; the majority-judgement tiebreaker picks 0
    mj = GradeProfile(
        [{0: 5, 1: 4, 2: 0}, {0: 5, 1: 3, 2: 1}, {0: 0, 1: 5, 2: 5}, {0: 4, 1: 4, 2: 2}],
        [0, 1, 2, 3, 4, 5],
    )
    assert [mj.median(c) for c in [0, 1]] == [4, 4]
    assert greatest_median(mj) == [0]

def test_majority_judgement_matches_greatest_median():
    mj = GradeProfile(
        [{0: 5, 1: 4, 2: 0}, {0: 5, 1: 3, 2: 1}, {0: 0, 1: 5, 2: 5}, {0: 4, 1: 4, 2: 2}],
        [0, 1, 2, 3, 4, 5],
    )
    assert majority_judgement(mj) == greatest_median(mj)

def test_greatest_median_with_custom_tiebreaker():
    gp = GradeProfile([{0: 5, 1: 0}, {0: 4, 1: 0}], [0, 1, 2, 3, 4, 5], gcounts=[2, 3])
    assert greatest_median(gp, tb_func=tiebreaker_diff) == [0]


# ---------------------------------------------------------------------------
#  Tiebreaker functions (cand 0 graded [5,5,4,4,4] -> median 4)
#  proportion higher than 4 = 2/5; proportion lower than 4 = 0
# ---------------------------------------------------------------------------

@pytest.fixture
def tb_profile():
    return GradeProfile([{0: 5, 1: 0}, {0: 4, 1: 0}], [0, 1, 2, 3, 4, 5], gcounts=[2, 3])

def test_tiebreaker_diff(tb_profile):
    m = tb_profile.median(0)
    assert tiebreaker_diff(tb_profile, 0, m) == pytest.approx(0.4)

def test_tiebreaker_majority_judgement(tb_profile):
    m = tb_profile.median(0)
    # more proponents than opponents -> returns the proponent proportion
    assert tiebreaker_majority_judgement(tb_profile, 0, m) == pytest.approx(0.4)

def test_tiebreaker_majority_judgement_opponents_win():
    # cand 0 graded [5,5,0]: median 5, opponents (below) = 1/3, proponents = 0
    # -> returns -prop_opponents
    gp = GradeProfile([{0: 5, 1: 0}, {0: 0, 1: 0}], [0, 1, 2, 3, 4, 5], gcounts=[2, 1])
    m = gp.median(0)
    assert tiebreaker_majority_judgement(gp, 0, m) == pytest.approx(-1 / 3)

def test_tiebreaker_relative_shares(tb_profile):
    m = tb_profile.median(0)
    assert tiebreaker_relative_shares(tb_profile, 0, m) == pytest.approx(0.5)

def test_tiebreaker_normalized_difference(tb_profile):
    m = tb_profile.median(0)
    assert tiebreaker_normalized_difference(tb_profile, 0, m) == pytest.approx(1 / 3)
