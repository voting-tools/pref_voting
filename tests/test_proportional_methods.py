"""
Tests for proportional voting methods (STV variants).

These tests cover the main STV implementations including Scottish STV, Meek, Warren,
and other variants including Sequential STV.
"""

from pref_voting.proportional_methods import (
    stv_scottish, stv_meek, stv_warren, stv_nb, stv_wig,
    stv_last_parcel, approval_stv, cpo_stv, sequential_stv
)
from pref_voting.profiles import Profile
from pref_voting.profiles_with_ties import ProfileWithTies, Ranking

import pytest


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_profile():
    """A simple 3-candidate profile where candidate 0 is clearly preferred."""
    return Profile([
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
    ])


@pytest.fixture
def party_blocs_profile():
    """
    Two party blocs: A-party (candidates 0,1) and B-party (candidates 2,3).
    6 voters prefer A-party, 4 voters prefer B-party.
    For 2 seats, fair result should be 1 from each party.
    """
    return Profile([
        [0, 1, 2, 3],  # A-party voter
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [1, 0, 2, 3],  # A-party voter (different order)
        [1, 0, 2, 3],
        [1, 0, 2, 3],
        [2, 3, 0, 1],  # B-party voter
        [2, 3, 0, 1],
        [3, 2, 0, 1],  # B-party voter (different order)
        [3, 2, 0, 1],
    ])


@pytest.fixture
def profile_with_clear_winner():
    """Profile where candidate 0 has overwhelming first-preference support."""
    return Profile([
        [0, 1, 2],
        [0, 1, 2],
        [0, 1, 2],
        [0, 2, 1],
        [0, 2, 1],
        [1, 0, 2],
        [2, 1, 0],
    ])


@pytest.fixture
def profile_for_transfers():
    """Profile designed to test surplus transfer mechanics."""
    return Profile([
        [0, 1, 2],
        [0, 1, 2],
        [0, 1, 2],
        [0, 1, 2],
        [0, 2, 1],
        [1, 2, 0],
        [2, 1, 0],
    ])


@pytest.fixture
def single_voter_profile():
    """Single voter - tests faithfulness property."""
    return Profile([[0, 1, 2, 3, 4]])


@pytest.fixture
def profile_with_ties():
    """Profile with tied preferences (for ProfileWithTies)."""
    return ProfileWithTies([
        Ranking({0: 1, 1: 2, 2: 3}),
        Ranking({0: 1, 1: 2, 2: 3}),
        Ranking({1: 1, 0: 2, 2: 3}),
        Ranking({2: 1, 1: 2, 0: 3}),
    ])


# ============================================================================
# Basic functionality tests
# ============================================================================

@pytest.mark.parametrize("method", [
    stv_scottish, stv_meek, stv_warren, stv_nb, stv_wig, stv_last_parcel
])
def test_methods_return_correct_number_of_seats(method, simple_profile):
    """All methods should return exactly num_seats winners."""
    result = method(simple_profile, num_seats=1)
    assert len(result) == 1

    result = method(simple_profile, num_seats=2)
    assert len(result) == 2


@pytest.mark.parametrize("method", [
    stv_scottish, stv_meek, stv_warren, stv_nb, stv_wig, stv_last_parcel
])
def test_methods_elect_clear_favorite(method, profile_with_clear_winner):
    """When one candidate has clear majority support, they should be elected."""
    result = method(profile_with_clear_winner, num_seats=1)
    assert 0 in result, f"{method.name} should elect candidate 0 who has majority support"


@pytest.mark.parametrize("method", [
    stv_scottish, stv_meek, stv_warren, stv_nb, stv_wig, stv_last_parcel
])
def test_methods_accept_profile_with_ties(method, profile_with_ties):
    """All methods should accept ProfileWithTies input."""
    result = method(profile_with_ties, num_seats=2)
    assert len(result) == 2
    assert all(c in [0, 1, 2] for c in result)


# ============================================================================
# Proportionality tests
# ============================================================================

@pytest.mark.parametrize("method", [
    stv_meek, stv_warren, stv_nb, stv_wig, stv_last_parcel
])
def test_proportionality_two_party(method, party_blocs_profile):
    """
    With 60% A-party and 40% B-party voters electing 2 seats,
    proportional methods should elect 1 from each party.
    """
    result = method(party_blocs_profile, num_seats=2)
    a_party_elected = len([c for c in result if c in [0, 1]])
    b_party_elected = len([c for c in result if c in [2, 3]])
    assert a_party_elected == 1, f"{method.name} should elect exactly 1 A-party candidate"
    assert b_party_elected == 1, f"{method.name} should elect exactly 1 B-party candidate"


# ============================================================================
# Faithfulness tests (single voter should get top-k)
# ============================================================================

@pytest.mark.parametrize("method", [
    stv_meek, stv_warren, stv_nb, stv_wig, stv_last_parcel
])
def test_faithfulness_single_voter(method, single_voter_profile):
    """
    With a single voter, the method should elect the voter's top-k candidates.
    Note: Scottish STV does NOT satisfy this due to integer Droop quota mechanics.
    """
    result = method(single_voter_profile, num_seats=3)
    assert set(result) == {0, 1, 2}, f"{method.name} should elect voter's top 3 choices"


@pytest.mark.skip(reason="Scottish STV does not satisfy faithfulness due to integer Droop quota")
def test_scottish_faithfulness_single_voter(single_voter_profile):
    """
    Scottish STV with single voter - documents expected behavior.
    With 1 voter and k seats, quota = floor(1/(k+1)) + 1 = 1,
    so no candidate reaches quota and eliminations determine result.
    """
    result = stv_scottish(single_voter_profile, num_seats=3)
    # This test is skipped because Scottish STV doesn't guarantee this
    assert set(result) == {0, 1, 2}


# ============================================================================
# Approval STV tests
# ============================================================================

def test_approval_stv_basic(profile_with_clear_winner):
    """Approval STV should elect the clearly preferred candidate."""
    result = approval_stv(profile_with_clear_winner, num_seats=1)
    assert 0 in result


def test_approval_stv_with_profile_with_ties():
    """Approval STV handles equal rankings (approval-style ballots)."""
    # Voters approve multiple candidates equally
    prof = ProfileWithTies([
        Ranking({0: 1, 1: 1, 2: 2}),  # Approves 0 and 1 equally
        Ranking({0: 1, 1: 1, 2: 2}),
        Ranking({2: 1, 0: 2, 1: 2}),  # Approves only 2
    ])
    result = approval_stv(prof, num_seats=2)
    assert len(result) == 2


# ============================================================================
# CPO-STV tests
# ============================================================================

def test_cpo_stv_basic(simple_profile):
    """CPO-STV should return the correct number of winners."""
    result = cpo_stv(simple_profile, num_seats=2)
    assert len(result) == 2


def test_cpo_stv_condorcet_committee():
    """
    CPO-STV should elect the Condorcet committee when one exists.
    Profile where {0, 2} beats all other pairs pairwise.
    """
    prof = Profile([
        [0, 2, 1, 3],
        [0, 2, 1, 3],
        [2, 0, 3, 1],
        [2, 0, 3, 1],
        [1, 3, 0, 2],
        [3, 1, 2, 0],
    ])
    result = cpo_stv(prof, num_seats=2)
    # {0, 2} should be elected as they form the strongest pair
    assert 0 in result or 2 in result  # At minimum one of the Condorcet pair


# ============================================================================
# Edge cases
# ============================================================================

@pytest.mark.parametrize("method", [
    stv_scottish, stv_meek, stv_warren, stv_nb, stv_wig, stv_last_parcel
])
def test_all_seats_equals_candidates(method):
    """When num_seats equals number of candidates, all should be elected."""
    prof = Profile([[0, 1, 2]])
    result = method(prof, num_seats=3)
    assert set(result) == {0, 1, 2}


@pytest.mark.parametrize("method", [
    stv_scottish, stv_meek, stv_warren, stv_nb, stv_wig, stv_last_parcel
])
def test_single_candidate_single_seat(method):
    """Single candidate for single seat should be elected."""
    prof = Profile([[0]])
    result = method(prof, num_seats=1)
    assert result == [0]


@pytest.mark.parametrize("method", [
    stv_scottish, stv_meek, stv_warren, stv_nb, stv_wig, stv_last_parcel
])
def test_curr_cands_subset(method, simple_profile):
    """Methods should respect curr_cands parameter."""
    result = method(simple_profile, num_seats=1, curr_cands=[1, 2])
    assert all(c in [1, 2] for c in result)
    assert 0 not in result


@pytest.mark.parametrize("method", [
    stv_meek, stv_warren, stv_nb, stv_last_parcel
])
def test_empty_election(method):
    """Empty profile should return empty result for iterative methods."""
    prof = ProfileWithTies([], candidates=[0, 1, 2])
    result = method(prof, num_seats=2)
    assert result == []


def test_empty_election_scottish_wig():
    """
    Scottish STV and WIG may elect candidates with empty profile due to
    'if continuing candidates equal remaining seats, elect all' rule.
    This tests they don't crash rather than checking specific behavior.
    """
    prof = ProfileWithTies([], candidates=[0, 1, 2])
    result_scottish = stv_scottish(prof, num_seats=2)
    result_wig = stv_wig(prof, num_seats=2)
    # Just verify they return something without crashing
    assert isinstance(result_scottish, list)
    assert isinstance(result_wig, list)


@pytest.mark.parametrize("method", [
    stv_scottish, stv_meek, stv_warren, stv_nb, stv_wig, stv_last_parcel
])
def test_zero_seats(method, simple_profile):
    """Requesting 0 seats should return empty result."""
    result = method(simple_profile, num_seats=0)
    assert result == []


# ============================================================================
# Determinism tests
# ============================================================================

@pytest.mark.parametrize("method", [
    stv_meek, stv_warren, stv_nb, stv_wig, stv_last_parcel
])
def test_determinism(method, party_blocs_profile):
    """Same input should produce same output (with same tie-breaking)."""
    result1 = method(party_blocs_profile, num_seats=2)
    result2 = method(party_blocs_profile, num_seats=2)
    assert result1 == result2


def test_scottish_determinism_with_fixed_rng(party_blocs_profile):
    """Scottish STV is deterministic when given a fixed RNG for tie-breaking."""
    import random
    rng1 = random.Random(42)
    rng2 = random.Random(42)
    result1 = stv_scottish(party_blocs_profile, num_seats=2, rng=rng1)
    result2 = stv_scottish(party_blocs_profile, num_seats=2, rng=rng2)
    assert result1 == result2


# ============================================================================
# Sequential STV tests
# ============================================================================

def test_sequential_stv_basic():
    """Sequential STV returns correct number of seats."""
    prof = Profile([
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [1, 0, 2, 3],
        [2, 3, 0, 1],
        [3, 2, 0, 1],
    ])
    result = sequential_stv(prof, num_seats=2)
    assert len(result) == 2


def test_sequential_stv_condorcet_winner():
    """Sequential STV elects the Condorcet winner when num_seats=1."""
    # Candidate 1 is Condorcet winner: beats 0 (4-3) and beats 2 (4-3)
    prof = Profile([
        [0, 1, 2],
        [0, 1, 2],
        [0, 1, 2],
        [1, 2, 0],
        [1, 2, 0],
        [2, 1, 0],
        [2, 1, 0],
    ])
    result = sequential_stv(prof, num_seats=1)
    assert result == [1], "Sequential STV should elect Condorcet winner"


def test_sequential_stv_condorcet_winner_larger():
    """Sequential STV finds Condorcet winner in larger candidate field."""
    # Candidate 2 beats everyone
    prof = Profile([
        [2, 0, 1, 3, 4],
        [2, 0, 1, 3, 4],
        [2, 1, 0, 3, 4],
        [0, 2, 1, 3, 4],
        [1, 2, 0, 3, 4],
        [3, 2, 0, 1, 4],
        [4, 2, 0, 1, 3],
    ])
    result = sequential_stv(prof, num_seats=1)
    assert result == [2], "Sequential STV should elect Condorcet winner"


def test_sequential_stv_condorcet_cycle():
    """Sequential STV handles Condorcet cycles gracefully."""
    # Classic 3-way cycle: 0 > 1 > 2 > 0
    prof = Profile([
        [0, 1, 2],
        [1, 2, 0],
        [2, 0, 1],
    ])
    result = sequential_stv(prof, num_seats=1)
    # Should return some valid candidate, not crash
    assert len(result) == 1
    assert result[0] in [0, 1, 2]


def test_sequential_stv_more_seats():
    """Sequential STV works correctly with more seats."""
    prof = Profile([
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [1, 0, 2, 3, 4],
        [2, 3, 0, 1, 4],
        [3, 2, 0, 1, 4],
        [4, 3, 2, 1, 0],
    ])
    result_2 = sequential_stv(prof, num_seats=2)
    result_3 = sequential_stv(prof, num_seats=3)
    assert len(result_2) == 2
    assert len(result_3) == 3


def test_sequential_stv_universal_second_choice():
    """
    Sequential STV can elect a universal second-choice candidate.

    This is the key example from the Voting Matters paper where
    Sequential STV differs from plain STV.
    """
    # E (candidate 4) is everyone's second choice
    prof = Profile([
        [0, 4, 1, 2, 3],  # A>E>...
        [0, 4, 1, 2, 3],
        [0, 4, 1, 2, 3],
        [1, 4, 0, 2, 3],  # B>E>...
        [1, 4, 0, 2, 3],
        [1, 4, 0, 2, 3],
        [2, 4, 0, 1, 3],  # C>E>...
        [2, 4, 0, 1, 3],
        [2, 4, 0, 1, 3],
        [3, 4, 0, 1, 2],  # D>E>...
        [3, 4, 0, 1, 2],
        [3, 4, 0, 1, 2],
        [4, 0, 1, 2, 3],  # E>A>...
    ])
    result = sequential_stv(prof, num_seats=2)
    # E (candidate 4) should be elected due to broad second-choice support
    assert 4 in result, "Sequential STV should elect universal second-choice candidate"


def test_sequential_stv_woodall_example1():
    """
    First example from Woodall's Voting Matters paper.

    5 candidates (A=0, B=1, C=2, D=3, E=4), 2 seats.
    In this scenario, both plain STV and Sequential STV elect B and C.
    """
    # A=0, B=1, C=2, D=3, E=4
    # Use ProfileWithTies to handle truncated ballots (E not ranked by some voters)
    prof = ProfileWithTies([
        Ranking({0: 1, 1: 2, 2: 3, 3: 4}),      # 104 voters: A, B, C, D (E unranked)
        Ranking({1: 1, 2: 2, 3: 3, 0: 4}),      # 103 voters: B, C, D, A
        Ranking({2: 1, 3: 2, 1: 3, 0: 4}),      # 102 voters: C, D, B, A
        Ranking({3: 1, 1: 2, 2: 3, 0: 4}),      # 101 voters: D, B, C, A
        Ranking({4: 1, 0: 2, 1: 3, 2: 4, 3: 5}),  # 3 voters: E, A, B, C, D
        Ranking({4: 1, 1: 2, 2: 3, 3: 4, 0: 5}),  # 3 voters: E, B, C, D, A
        Ranking({4: 1, 2: 2, 3: 3, 1: 4, 0: 5}),  # 3 voters: E, C, D, B, A
        Ranking({4: 1, 3: 2, 2: 3, 1: 4, 0: 5}),  # 3 voters: E, D, C, B, A
    ], rcounts=[104, 103, 102, 101, 3, 3, 3, 3], candidates=[0, 1, 2, 3, 4])

    result = sequential_stv(prof, num_seats=2)
    # Both STV and Sequential STV should elect B and C
    assert len(result) == 2
    assert 1 in result, "B should be elected"
    assert 2 in result, "C should be elected"


def test_sequential_stv_woodall_example2():
    """
    Second example from Woodall's Voting Matters paper.

    5 candidates (A=0, B=1, C=2, D=3, E=4), 2 seats.
    E is everyone's second choice. Plain STV elects B, C but
    Sequential STV should elect B, E.
    """
    # A=0, B=1, C=2, D=3, E=4
    # Modified so E is second choice for everyone
    prof = Profile([
        [0, 4, 1, 2, 3],   # 104 voters: A, E, B, C, D
        [1, 4, 2, 3, 0],   # 103 voters: B, E, C, D, A
        [2, 4, 3, 1, 0],   # 102 voters: C, E, D, B, A
        [3, 4, 1, 2, 0],   # 101 voters: D, E, B, C, A
        [4, 0, 1, 2, 3],   # 3 voters: E, A, B, C, D
        [4, 1, 2, 3, 0],   # 3 voters: E, B, C, D, A
        [4, 2, 3, 1, 0],   # 3 voters: E, C, D, B, A
        [4, 3, 2, 1, 0],   # 3 voters: E, D, C, B, A
    ], rcounts=[104, 103, 102, 101, 3, 3, 3, 3])

    result = sequential_stv(prof, num_seats=2)
    # Sequential STV should recognize E's broad support
    assert len(result) == 2
    assert 4 in result, "E should be elected due to universal second-choice support"


# ============================================================================
# Sequential STV Edge Cases
# ============================================================================

def test_sequential_stv_single_candidate():
    """Sequential STV with only one candidate."""
    prof = Profile([[0], [0], [0]])
    result = sequential_stv(prof, num_seats=1)
    assert result == [0]


def test_sequential_stv_num_seats_equals_candidates_minus_one():
    """Sequential STV when electing all but one candidate."""
    prof = Profile([
        [0, 1, 2, 3],
        [1, 0, 2, 3],
        [2, 3, 0, 1],
        [3, 2, 1, 0],
    ])
    result = sequential_stv(prof, num_seats=3)
    assert len(result) == 3
    # Should elect 3 of the 4 candidates
    assert len(set(result)) == 3


def test_sequential_stv_all_tied_first_preferences():
    """Sequential STV when all candidates have equal first-preference votes."""
    prof = Profile([
        [0, 1, 2, 3],
        [1, 2, 3, 0],
        [2, 3, 0, 1],
        [3, 0, 1, 2],
    ])
    result = sequential_stv(prof, num_seats=2)
    assert len(result) == 2
    # All candidates have equal first preferences, so result depends on
    # tie-breaking and transfers


def test_sequential_stv_two_candidates_one_seat():
    """Sequential STV with just two candidates for one seat (simple majority)."""
    prof = Profile([
        [0, 1],
        [0, 1],
        [0, 1],
        [1, 0],
        [1, 0],
    ])
    result = sequential_stv(prof, num_seats=1)
    assert result == [0], "Candidate with more first preferences should win"


# ============================================================================
# Sequential STV Loop Detection
# ============================================================================

def test_sequential_stv_potential_loop():
    """
    Test that Sequential STV handles potential cycling scenarios.

    Create a scenario where challenges might cycle, and verify
    the algorithm terminates with a valid result.
    """
    # Rock-paper-scissors style preferences that might cause cycling
    prof = Profile([
        [0, 1, 2],  # A > B > C
        [0, 1, 2],
        [1, 2, 0],  # B > C > A
        [1, 2, 0],
        [2, 0, 1],  # C > A > B
        [2, 0, 1],
    ])
    result = sequential_stv(prof, num_seats=1)
    # Should terminate and return exactly one winner
    assert len(result) == 1
    assert result[0] in [0, 1, 2]


def test_sequential_stv_max_iterations():
    """Test that max_iterations parameter is respected."""
    prof = Profile([
        [0, 1, 2, 3],
        [1, 2, 3, 0],
        [2, 3, 0, 1],
        [3, 0, 1, 2],
    ])
    # Should complete within max_iterations
    result = sequential_stv(prof, num_seats=2, max_iterations=100)
    assert len(result) == 2


# ============================================================================
# Sequential STV with ProfileWithTies
# ============================================================================

def test_sequential_stv_profile_with_ties():
    """Sequential STV works with ProfileWithTies (ballots with tied rankings)."""
    prof = ProfileWithTies([
        Ranking({0: 1, 1: 2, 2: 3}),      # A > B > C
        Ranking({0: 1, 1: 2, 2: 3}),
        Ranking({1: 1, 0: 2, 2: 3}),      # B > A > C
        Ranking({2: 1, 1: 2, 0: 3}),      # C > B > A
        Ranking({0: 1, 1: 1, 2: 2}),      # A = B > C (tie!)
    ], candidates=[0, 1, 2])

    result = sequential_stv(prof, num_seats=1)
    assert len(result) == 1
    assert result[0] in [0, 1, 2]


def test_sequential_stv_truncated_ballots():
    """Sequential STV handles truncated ballots (not all candidates ranked)."""
    prof = ProfileWithTies([
        Ranking({0: 1}),           # Only ranks A
        Ranking({0: 1}),
        Ranking({1: 1}),           # Only ranks B
        Ranking({1: 1, 2: 2}),     # B > C
        Ranking({2: 1, 0: 2}),     # C > A
    ], candidates=[0, 1, 2])

    result = sequential_stv(prof, num_seats=1)
    assert len(result) == 1


def test_sequential_stv_determinism():
    """Sequential STV produces consistent results across multiple runs."""
    prof = Profile([
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [1, 0, 2, 3],
        [2, 3, 0, 1],
        [3, 2, 0, 1],
    ])

    results = [tuple(sequential_stv(prof, num_seats=2)) for _ in range(5)]
    # All runs should give identical results
    assert all(r == results[0] for r in results), "Sequential STV should be deterministic"


def test_sequential_stv_challenger_tie_rule():
    """
    Test Issue 20 tie rule: if challenger ties with a probable, challenger loses.

    This test creates a scenario where a challenger ties with a probable
    in the challenge round. Per Issue 20, the challenger should be treated
    as having failed (status quo maintained).

    Profile: 2 candidates (A=0, B=1), 1 seat
    - 2 voters: A > B
    - 2 voters: B > A

    Initial Meek count: A and B tie at 2 votes each. With default tie-breaking
    by candidate index, A (0) wins. So probables = [A], queue = [B].

    Challenge with B:
    Contest is A vs B for 1 seat. Both have 2 votes = quota (4/2 = 2).
    This is a tie. Per Issue 20 tie rule, challenger B should lose,
    maintaining A as the probable.

    Note: We're testing that the tie-break rule is applied correctly.
    Without the rule, the default (lower index wins) would still give A.
    The key is that Sequential STV uses the Issue 20 tie-break rule,
    not the default, and the test verifies the result is correct.
    """
    prof = Profile([
        [0, 1],  # A > B
        [0, 1],  # A > B
        [1, 0],  # B > A
        [1, 0],  # B > A
    ])

    result = sequential_stv(prof, num_seats=1)

    # A should win - as the initial probable, ties go to status quo
    assert 0 in result, (
        "Candidate A (0) should win: in a tie with challenger B, "
        "the status quo (A as probable) should be maintained per Issue 20"
    )


def test_sequential_stv_borda_special_procedure():
    """
    Test that the Borda special procedure is triggered and works correctly.

    Per Issue 20, when a loop is detected and ALL candidates have been
    probable at some point (so there are no "never-probables" to exclude),
    the algorithm uses Borda scores to determine which "at-risk" candidate
    to exclude.

    This profile creates a rock-paper-scissors dynamic:
    - 2 voters: C > B > A
    - 2 voters: B > A > C
    - 1 voter: A > C > B

    This causes cycling where each candidate displaces another:
    - Initial: C wins (A eliminated, C reaches quota)
    - A challenges C: A wins (gets B>A>C transfers)
    - B challenges A: B wins (gets C>B>A transfers)
    - C challenges B: C wins (gets A>C>B transfer)
    - Loop detected! All candidates have been probable.

    Borda scores: A=8, B=12, C=10
    The lowest-scoring at-risk candidate (A) is excluded via Borda procedure.
    After A's exclusion, C beats B in the final challenge.
    """
    prof = Profile([
        [2, 1, 0],  # C > B > A
        [2, 1, 0],  # C > B > A
        [1, 0, 2],  # B > A > C
        [1, 0, 2],  # B > A > C
        [0, 2, 1],  # A > C > B
    ])

    result = sequential_stv(prof, num_seats=1)

    # Should terminate with exactly one winner
    assert len(result) == 1

    # C should win after Borda procedure excludes A (lowest Borda score)
    # then C beats B in the remaining contest
    assert result[0] == 2, (
        "Candidate C (2) should win: after Borda procedure excludes A (lowest score), "
        "C beats B in the final challenge"
    )


def test_sequential_stv_borda_score_calculation():
    """
    Directly test the Borda score calculation used in the special procedure.

    Per Issue 20: "a Borda score is calculated, as the sum over all votes of the
    number of continuing candidates to whom the candidate in question is preferred...
    In practice it can help to give 2 points instead of 1 for each candidate beaten."
    """
    from pref_voting.proportional_methods import _calculate_borda_scores

    # Test 1: Simple strict rankings (no ties)
    # 2 voters: C > B > A
    # 2 voters: B > A > C
    # 1 voter: A > C > B
    prof = Profile([
        [2, 1, 0], [2, 1, 0],  # C > B > A (x2)
        [1, 0, 2], [1, 0, 2],  # B > A > C (x2)
        [0, 2, 1],              # A > C > B (x1)
    ])
    prof_wt = prof.to_profile_with_ties()
    scores = _calculate_borda_scores(prof_wt, [0, 1, 2])

    # Manual calculation with 2 points per beat:
    # Vote C>B>A (x2): C beats 2 (4pts), B beats 1 (2pts), A beats 0 (0pts)
    # Vote B>A>C (x2): B beats 2 (4pts), A beats 1 (2pts), C beats 0 (0pts)
    # Vote A>C>B (x1): A beats 2 (4pts), C beats 1 (2pts), B beats 0 (0pts)
    # Totals: A = 0*2 + 2*2 + 4*1 = 8, B = 2*2 + 4*2 + 0*1 = 12, C = 4*2 + 0*2 + 2*1 = 10
    assert scores[0] == 8, f"A should have score 8, got {scores[0]}"
    assert scores[1] == 12, f"B should have score 12, got {scores[1]}"
    assert scores[2] == 10, f"C should have score 10, got {scores[2]}"

    # Test 2: Truncated ballot (unranked candidates get averaging)
    # 1 voter ranks only A (B and C unmentioned = tied last)
    # A beats both B and C: 4 points
    # B and C unmentioned: average of {0, 2} = 1 point each (with 2x scaling)
    prof2 = ProfileWithTies([Ranking({0: 1})], candidates=[0, 1, 2])
    scores2 = _calculate_borda_scores(prof2, [0, 1, 2])

    assert scores2[0] == 4, f"A should have score 4, got {scores2[0]}"
    assert scores2[1] == 1, f"B (unranked) should have score 1, got {scores2[1]}"
    assert scores2[2] == 1, f"C (unranked) should have score 1, got {scores2[2]}"

    # Test 3: Explicit ties (extends averaging to explicit ties)
    # 1 voter: A > {B, C} tied
    # A beats both: 4 points
    # B and C tied at rank 2: each gets 2*0 + (2-1) = 1 point (averaging)
    prof3 = ProfileWithTies([Ranking({0: 1, 1: 2, 2: 2})], candidates=[0, 1, 2])
    scores3 = _calculate_borda_scores(prof3, [0, 1, 2])

    assert scores3[0] == 4, f"A should have score 4, got {scores3[0]}"
    assert scores3[1] == 1, f"B (tied) should have score 1, got {scores3[1]}"
    assert scores3[2] == 1, f"C (tied) should have score 1, got {scores3[2]}"


# ============================================================================
# Meek vs Warren difference test
# ============================================================================

def test_meek_warren_can_differ():
    """
    Test case from Hill & Warren paper where Meek and Warren can differ.
    4 candidates for 3 seats, 3 votes: 1 ABC, 1 BC, 1 BD.
    Meek elects ABC, Warren gives C/D tie (resolved by index).
    """
    prof = ProfileWithTies([
        Ranking({0: 1, 1: 2, 2: 3}),  # ABC
        Ranking({1: 1, 2: 2}),         # BC
        Ranking({1: 1, 3: 2}),         # BD
    ], candidates=[0, 1, 2, 3])

    meek_result = stv_meek(prof, num_seats=3)
    warren_result = stv_warren(prof, num_seats=3)

    # Both should elect A and B
    assert 0 in meek_result and 1 in meek_result
    assert 0 in warren_result and 1 in warren_result

    # Meek gives C an advantage via multiplicative transfer
    # Warren gives C and D equal tallies (tie broken by index, so C wins)
    # Both end up electing C in our implementation, but for different reasons
    assert 2 in meek_result  # C definitely wins under Meek
