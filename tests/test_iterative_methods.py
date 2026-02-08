from pref_voting.iterative_methods import *
from pref_voting.profiles import Profile
from pref_voting.profiles_with_ties import ProfileWithTies
import pytest

@pytest.mark.parametrize("voting_method, expected", [
    (instant_runoff, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1],
        'profile_single_voter': [0],
        'profile_single_voter_curr_cands': [0]
        }),
    (ranked_choice, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1],
        'profile_single_voter': [0],
        'profile_single_voter_curr_cands': [0]
        }),
    (hare, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1],
        'profile_single_voter': [0],
        'profile_single_voter_curr_cands': [0]
        }),
    (instant_runoff_tb, {
        'condorcet_cycle': [1], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1],
        'profile_single_voter': [0],
        'profile_single_voter_curr_cands': [0]
        }),
    (instant_runoff_put, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1],
        'profile_single_voter': [0],
        'profile_single_voter_curr_cands': [0]
        }),
    (bottom_two_runoff_instant_runoff, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1],
        'profile_single_voter': [0],
        'profile_single_voter_curr_cands': [0]
        }),
    (bottom_two_runoff_instant_runoff_put, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1],
        'profile_single_voter': [0],
        'profile_single_voter_curr_cands': [0]
        }),
    (benham, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1],
        'profile_single_voter': [0],
        'profile_single_voter_curr_cands': [0]
        }),
    (benham_put, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1],
        'profile_single_voter': [0],
        'profile_single_voter_curr_cands': [0]
        }),
    (benham_tb, {
        'condorcet_cycle': [1], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1],
        'profile_single_voter': [0],
        'profile_single_voter_curr_cands': [0]
        }),
    (plurality_with_runoff_put, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1],
        'profile_single_voter': [0],
        'profile_single_voter_curr_cands': [0]
        }),
    (coombs, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1],
        'profile_single_voter': [0],
        'profile_single_voter_curr_cands': [0]
        }),
    (coombs_tb, {
        'condorcet_cycle': [1], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1],
        'profile_single_voter': [0],
        'profile_single_voter_curr_cands': [0]
        }),
    (coombs_put, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1],
        'profile_single_voter': [0],
        'profile_single_voter_curr_cands': [0]
        }),
    (baldwin, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1],
        'profile_single_voter': [0],
        'profile_single_voter_curr_cands': [0]
        }),
    (baldwin_tb, {
        'condorcet_cycle': [1], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1],
        'profile_single_voter': [0],
        'profile_single_voter_curr_cands': [0]
        }),
    (baldwin_put, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1],
        'profile_single_voter': [0],
        'profile_single_voter_curr_cands': [0]
        }),
    (strict_nanson, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1],
        'profile_single_voter': [0],
        'profile_single_voter_curr_cands': [0]
        }),
    (weak_nanson, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1],
        'profile_single_voter': [0],
        'profile_single_voter_curr_cands': [0]
        }),
    (iterated_removal_cl, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1],
        'profile_single_voter': [0],
        'profile_single_voter_curr_cands': [0]
        }),
    (raynaud, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1],
        'profile_single_voter': [0],
        'profile_single_voter_curr_cands': [0]
        }),
    (tideman_alternative_smith, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1],
        'profile_single_voter': [0],
        'profile_single_voter_curr_cands': [0]
        }),
    (tideman_alternative_smith_put, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1],
        'profile_single_voter': [0],
        'profile_single_voter_curr_cands': [0]
        }),
    (tideman_alternative_gocha, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1],
        'profile_single_voter': [0],
        'profile_single_voter_curr_cands': [0]
        }),
    (tideman_alternative_gocha_put, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1],
        'profile_single_voter': [0],
        'profile_single_voter_curr_cands': [0]
        }),
    (woodall, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1],
        'profile_single_voter': [0],
        'profile_single_voter_curr_cands': [0]
        }),
    (knockout, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1],
        'profile_single_voter': [0],
        'profile_single_voter_curr_cands': [0]
        }),
])

def test_methods(
    voting_method, expected, 
    condorcet_cycle,
    profile_single_voter, 
    linear_profile_0):
    assert voting_method(condorcet_cycle) == expected['condorcet_cycle']
    assert voting_method(linear_profile_0) == expected['linear_profile_0']
    if 'linear_profile_0_curr_cands' in expected:
        assert voting_method(linear_profile_0, curr_cands=[1, 2]) == expected['linear_profile_0_curr_cands']
    assert voting_method(profile_single_voter) == expected['profile_single_voter']
    assert voting_method(profile_single_voter, curr_cands=[0, 1]) == expected['profile_single_voter_curr_cands']


def test_instant_runoff_for_truncated_linear_orders():
    prof = ProfileWithTies([
        {0:1, 1:2},
        {1:1, 0:3},
        {0:2, 1:1, 2:0}
    ],
    candidates=[0, 1, 2])   

    assert instant_runoff_for_truncated_linear_orders(prof) == [0, 1, 2]

def test_instant_runoff_recursive_tie_breaker_full_tie():
    """Recursive and basic IRV should agree when a tie-breaker resolves full ties."""
    prof = Profile([[0, 1, 2], [1, 2, 0], [2, 0, 1]], [1, 1, 1])
    tb = [0, 1, 2]
    assert instant_runoff(prof, algorithm="basic", tie_breaker=tb) == [1]
    assert instant_runoff(prof, algorithm="recursive", tie_breaker=tb) == [1]

def test_instant_runoff_with_explanation(condorcet_cycle, linear_profile_0):
    ws, exp = instant_runoff_with_explanation(condorcet_cycle)
    assert ws == [0, 1, 2]
    assert exp == [[0,1,2]]
    ws, exp = instant_runoff_with_explanation(linear_profile_0)
    assert ws == [0]
    assert exp == []
    prof = Profile([[0, 1, 2], [0, 1, 2], [1, 2, 0], [2, 1, 0]])
    ws, exp = instant_runoff_with_explanation(prof)
    assert ws == [0]
    assert exp == [[1, 2]]

def test_coombs_with_explanation(condorcet_cycle, linear_profile_0):
    ws, exp = coombs_with_explanation(condorcet_cycle)
    assert ws == [0, 1, 2]
    assert exp == [[0, 1, 2]]
    ws, exp = coombs_with_explanation(linear_profile_0)
    assert ws == [0]
    assert exp == []
    prof = Profile([[0, 1, 2], [0, 1, 2], [1, 2, 0], [2, 1, 0]])
    ws, exp = coombs_with_explanation(prof)
    assert ws == [1]
    assert exp == [[0,2]]


def test_plurality_with_runoff_put_with_explanation(condorcet_cycle, linear_profile_0):
    ws, exp = plurality_with_runoff_put_with_explanation(condorcet_cycle)
    assert ws == [0, 1, 2]
    assert exp == [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
    ws, exp = plurality_with_runoff_put_with_explanation(linear_profile_0)
    assert ws == [0]
    assert exp == [(0,2)]
    prof = Profile([[0, 1, 2], [0, 1, 2], [1, 2, 0], [2, 1, 0]])
    ws, exp = plurality_with_runoff_put_with_explanation(prof)
    assert ws == [0, 1, 2]
    assert exp == [(0, 1), (0, 2)]


def test_baldwin_with_explanation(condorcet_cycle, linear_profile_0):
    ws, exp = baldwin_with_explanation(condorcet_cycle)
    assert ws == [0, 1, 2]
    assert exp == [[[0, 1, 2], {0: 3, 1: 3, 2: 3}]]
    ws, exp = baldwin_with_explanation(linear_profile_0)
    assert ws == [0]
    assert exp == [[[2], {0: 4, 1: 3, 2: 2}], [[1], {0: 2, 1: 1}]]
    prof = Profile([[0, 1, 2], [0, 1, 2], [1, 2, 0], [2, 1, 0]])
    ws, exp = baldwin_with_explanation(prof)
    assert ws == [0, 1]
    assert exp == [[[2], {0: 4, 1: 5, 2: 3}], [[0, 1], {0: 2, 1: 2}]]

def test_strict_nanson_with_explanation(condorcet_cycle, linear_profile_0):
    ws, exp = strict_nanson_with_explanation(condorcet_cycle)
    assert ws == [0, 1, 2]
    assert exp == [{'avg_borda_score': 3.0, 'elim_cands': [], 'borda_scores': {0: 3, 1: 3, 2: 3}}, {'avg_borda_score': 3.0, 'elim_cands': [], 'borda_scores': {0: 3, 1: 3, 2: 3}}]
    ws, exp = strict_nanson_with_explanation(linear_profile_0)
    assert ws == [0]
    assert exp == [{'avg_borda_score': 3.0, 'elim_cands': [2], 'borda_scores': {0: 4, 1: 3, 2: 2}}, {'avg_borda_score': 1.5, 'elim_cands': [1], 'borda_scores': {0: 2, 1: 1}}]
    prof = Profile([[0, 1, 2], [0, 1, 2], [1, 2, 0], [2, 1, 0]])
    ws, exp = strict_nanson_with_explanation(prof)
    assert ws == [0, 1]
    assert exp == [{'avg_borda_score': 4.0, 'elim_cands': [2], 'borda_scores': {0: 4, 1: 5, 2: 3}}, {'avg_borda_score': 2.0, 'elim_cands': [], 'borda_scores': {0: 2, 1: 2}}]

def test_weak_nanson_with_explanation(condorcet_cycle, linear_profile_0):
    ws, exp = weak_nanson_with_explanation(condorcet_cycle)
    assert ws == [0, 1, 2]
    assert exp == [{'avg_borda_score': 3.0, 'elim_cands': [0, 1, 2], 'borda_scores': {0: 3, 1: 3, 2: 3}}]
    ws, exp = weak_nanson_with_explanation(linear_profile_0)
    assert ws == [0]
    assert exp == [{'avg_borda_score': 3.0, 'elim_cands': [1, 2], 'borda_scores': {0: 4, 1: 3, 2: 2}}]
    prof = Profile([[0, 1, 2], [0, 1, 2], [1, 2, 0], [2, 1, 0]])
    ws, exp = weak_nanson_with_explanation(prof)
    assert ws == [1]
    assert exp == [{'avg_borda_score': 4.0, 'elim_cands': [0, 2], 'borda_scores': {0: 4, 1: 5, 2: 3}}]

def test_iterated_removal_cl_with_explanation(condorcet_cycle, linear_profile_0):
    ws, exp = iterated_removal_cl_with_explanation(condorcet_cycle)
    assert ws == [0, 1, 2]
    assert exp == []
    ws, exp = iterated_removal_cl_with_explanation(linear_profile_0)
    assert ws == [0]
    assert exp ==  [2, 1]
    prof = Profile([[0, 1, 2], [0, 1, 2], [1, 2, 0], [2, 1, 0]])
    ws, exp = iterated_removal_cl_with_explanation(prof)
    assert ws == [0,1,2]
    assert exp == []

def test_plurality_veto():
    """Test the plurality_veto method."""
    # Test with a profile in which a candidate has zero initial plurality score
    prof = Profile([[0, 1, 2], [0, 1, 2], [2, 1, 0]], rcounts=[1, 1, 1])
    assert plurality_veto(prof) == [0]

    # Test with different voter orders producing different winners
    prof = Profile([[0, 1, 2], [0, 1, 2], [2, 1, 0], [2, 1, 0]], rcounts=[1, 1, 1, 1])
    # Default order [0,1,2,3] yields winner 0
    assert plurality_veto(prof) == [0]
    # Reverse order [3,2,1,0] yields winner 2
    voter_order = [3, 2, 1, 0]
    assert plurality_veto(prof, voter_order=voter_order) == [2]

    # Test with curr_cands parameter
    prof = Profile([[0, 1, 2], [1, 0, 2], [2, 0, 1]], rcounts=[1, 1, 1])
    assert plurality_veto(prof, curr_cands={0, 1}) == [0]


# =============================================================================
# Tests for Approval-IRV and Split-IRV (ProfileWithTies support)
# Based on Delemazure & Peters (2024) "Approval-Based Instant-Runoff Voting"
# =============================================================================

def test_approval_irv_paper_example():
    """Test Approval-IRV on Figure 3 from the paper.
    
    5 voters with weak orders over candidates {a=0, b=1, c=2, d=3}:
    - 2 voters: a ~ b > c > d  (a and b tied at top)
    - 1 voter: b > c > d > a
    - 1 voter: c > d > a > b
    - 1 voter: d > a > b > c
    
    Paper says: First eliminated is c (ranked on top on 1 ballot), 
    then d (ranked on top on 1 ballot), then b (ranked on top on 3 ballots).
    Winner is a.
    """
    prof = ProfileWithTies([
        {0: 1, 1: 1, 2: 2, 3: 3},  # a ~ b > c > d
        {0: 1, 1: 1, 2: 2, 3: 3},  # a ~ b > c > d
        {1: 1, 2: 2, 3: 3, 0: 4},  # b > c > d > a
        {2: 1, 3: 2, 0: 3, 1: 4},  # c > d > a > b
        {3: 1, 0: 2, 1: 3, 2: 4},  # d > a > b > c
    ], candidates=[0, 1, 2, 3])
    
    assert approval_irv(prof) == [0]


def test_approval_irv_tb():
    """Test Approval-IRV with tie-breaker."""
    prof = ProfileWithTies([
        {0: 1, 1: 1, 2: 2, 3: 3},
        {0: 1, 1: 1, 2: 2, 3: 3},
        {1: 1, 2: 2, 3: 3, 0: 4},
        {2: 1, 3: 2, 0: 3, 1: 4},
        {3: 1, 0: 2, 1: 3, 2: 4},
    ], candidates=[0, 1, 2, 3])
    
    # With tie_breaker [0,1,2,3], candidate 0 has lowest priority.
    # Round 1: c=2 and d=3 tied (score=1), eliminate 2 (lower index in TB)
    # Round 2: a=0 and d=3 tied (score=2), eliminate 0 (lower index in TB)
    # Round 3: b=1 (score=3) vs d=3 (score=2), eliminate d
    # Winner: b=1
    assert approval_irv_tb(prof, tie_breaker=[0, 1, 2, 3]) == [1]


def test_approval_irv_put():
    """Test Approval-IRV with parallel universe tie-breaking."""
    prof = ProfileWithTies([
        {0: 1, 1: 1, 2: 2, 3: 3},
        {0: 1, 1: 1, 2: 2, 3: 3},
        {1: 1, 2: 2, 3: 3, 0: 4},
        {2: 1, 3: 2, 0: 3, 1: 4},
        {3: 1, 0: 2, 1: 3, 2: 4},
    ], candidates=[0, 1, 2, 3])
    
    # PUT explores all elimination paths - both a and b can win depending on path
    assert approval_irv_put(prof) == [0, 1]


def test_approval_irv_all_tied():
    """Test Approval-IRV when all candidates are tied."""
    prof = ProfileWithTies([
        {0: 1, 1: 1, 2: 1},  # All tied at top
        {0: 1, 1: 1, 2: 1},
    ], candidates=[0, 1, 2])
    
    # All candidates should be returned as winners
    assert approval_irv(prof) == [0, 1, 2]


def test_approval_irv_tb_full_tie():
    """Test that TB eliminates one candidate when all are tied for lowest."""
    prof = ProfileWithTies([
        {0: 1, 1: 1, 2: 1},
    ], candidates=[0, 1, 2])
    
    # With TB [0, 1, 2], candidate 0 has lowest priority (eliminated first)
    # After eliminating 0, we have 1 and 2 tied, so 1 is eliminated
    # Winner is 2
    result = approval_irv_tb(prof, tie_breaker=[0, 1, 2])
    assert result == [2]


def test_approval_irv_put_full_tie_branches():
    """Test that PUT branches on all tied candidates."""
    prof = ProfileWithTies([
        {0: 1, 1: 1, 2: 1},
    ], candidates=[0, 1, 2])
    
    # PUT should explore all elimination paths and return all possible winners
    assert approval_irv_put(prof) == [0, 1, 2]


def test_approval_irv_with_explanation():
    """Test Approval-IRV with explanation."""
    prof = ProfileWithTies([
        {0: 1, 1: 1, 2: 2, 3: 3},
        {0: 1, 1: 1, 2: 2, 3: 3},
        {1: 1, 2: 2, 3: 3, 0: 4},
        {2: 1, 3: 2, 0: 3, 1: 4},
        {3: 1, 0: 2, 1: 3, 2: 4},
    ], candidates=[0, 1, 2, 3])
    
    ws, exp = approval_irv_with_explanation(prof)
    assert ws == [0]
    assert exp == [[2, 3], [1]]


def test_approval_irv_curr_cands():
    """Test Approval-IRV with restricted candidates."""
    prof = ProfileWithTies([
        {0: 1, 1: 1, 2: 2, 3: 3},
        {0: 1, 1: 1, 2: 2, 3: 3},
        {1: 1, 2: 2, 3: 3, 0: 4},
        {2: 1, 3: 2, 0: 3, 1: 4},
        {3: 1, 0: 2, 1: 3, 2: 4},
    ], candidates=[0, 1, 2, 3])
    
    # Restrict to candidates 0, 1, 2
    result = approval_irv(prof, curr_cands=[0, 1, 2])
    assert result == [0]


def test_split_irv_paper_example():
    """Test Split-IRV on Figure 3 from the paper.
    
    Split-IRV should elect b instead of a on this example.
    """
    prof = ProfileWithTies([
        {0: 1, 1: 1, 2: 2, 3: 3},
        {0: 1, 1: 1, 2: 2, 3: 3},
        {1: 1, 2: 2, 3: 3, 0: 4},
        {2: 1, 3: 2, 0: 3, 1: 4},
        {3: 1, 0: 2, 1: 3, 2: 4},
    ], candidates=[0, 1, 2, 3])
    
    assert split_irv(prof) == [1]


def test_split_irv_tb():
    """Test Split-IRV with tie-breaker."""
    prof = ProfileWithTies([
        {0: 1, 1: 1, 2: 2, 3: 3},
        {0: 1, 1: 1, 2: 2, 3: 3},
        {1: 1, 2: 2, 3: 3, 0: 4},
        {2: 1, 3: 2, 0: 3, 1: 4},
        {3: 1, 0: 2, 1: 3, 2: 4},
    ], candidates=[0, 1, 2, 3])
    
    assert split_irv_tb(prof, tie_breaker=[0, 1, 2, 3]) == [1]


def test_split_irv_put():
    """Test Split-IRV with parallel universe tie-breaking."""
    prof = ProfileWithTies([
        {0: 1, 1: 1, 2: 2, 3: 3},
        {0: 1, 1: 1, 2: 2, 3: 3},
        {1: 1, 2: 2, 3: 3, 0: 4},
        {2: 1, 3: 2, 0: 3, 1: 4},
        {3: 1, 0: 2, 1: 3, 2: 4},
    ], candidates=[0, 1, 2, 3])
    
    # PUT explores all elimination paths - both a and b can win depending on path
    assert split_irv_put(prof) == [0, 1]


def test_split_irv_all_tied():
    """Test Split-IRV when all candidates are tied."""
    prof = ProfileWithTies([
        {0: 1, 1: 1, 2: 1},
        {0: 1, 1: 1, 2: 1},
    ], candidates=[0, 1, 2])
    
    assert split_irv(prof) == [0, 1, 2]


def test_split_irv_float_tolerance():
    """Test that Split-IRV handles float comparison correctly."""
    # 3 voters each with 3 candidates tied at top
    # Each candidate gets 1/3 + 1/3 + 1/3 = 1.0 score
    prof = ProfileWithTies([
        {0: 1, 1: 1, 2: 1},
        {0: 1, 1: 1, 2: 1},
        {0: 1, 1: 1, 2: 1},
    ], candidates=[0, 1, 2])
    
    # All should be tied
    assert split_irv(prof) == [0, 1, 2]


def test_split_irv_tb_full_tie():
    """Test that TB eliminates one candidate when all are tied for lowest in Split-IRV."""
    prof = ProfileWithTies([
        {0: 1, 1: 1, 2: 1},
    ], candidates=[0, 1, 2])
    
    result = split_irv_tb(prof, tie_breaker=[0, 1, 2])
    assert result == [2]


def test_split_irv_put_full_tie_branches():
    """Test that PUT branches on all tied candidates in Split-IRV."""
    prof = ProfileWithTies([
        {0: 1, 1: 1, 2: 1},
    ], candidates=[0, 1, 2])
    
    assert split_irv_put(prof) == [0, 1, 2]


def test_split_irv_with_explanation():
    """Test Split-IRV with explanation."""
    prof = ProfileWithTies([
        {0: 1, 1: 1, 2: 2, 3: 3},
        {0: 1, 1: 1, 2: 2, 3: 3},
        {1: 1, 2: 2, 3: 3, 0: 4},
        {2: 1, 3: 2, 0: 3, 1: 4},
        {3: 1, 0: 2, 1: 3, 2: 4},
    ], candidates=[0, 1, 2, 3])
    
    ws, exp = split_irv_with_explanation(prof)
    assert ws == [1]
    assert exp == [[0, 2, 3]]


def test_instant_runoff_with_explanation_profile_with_ties():
    prof = ProfileWithTies([
        {0: 1, 1: 1, 2: 2, 3: 3},
        {0: 1, 1: 1, 2: 2, 3: 3},
        {1: 1, 2: 2, 3: 3, 0: 4},
        {2: 1, 3: 2, 0: 3, 1: 4},
        {3: 1, 0: 2, 1: 3, 2: 4},
    ], candidates=[0, 1, 2, 3])

    # Default (approval scoring)
    ws, exp = instant_runoff_with_explanation(prof)
    assert ws == [0]
    assert exp == [[2, 3], [1]]

    # Split scoring
    ws, exp = instant_runoff_with_explanation(prof, score_method="split")
    assert ws == [1]
    assert exp == [[0, 2, 3]]


def test_approval_vs_split_irv_different_winners():
    """Test that Approval-IRV and Split-IRV can produce different winners."""
    prof = ProfileWithTies([
        {0: 1, 1: 1, 2: 2, 3: 3},
        {0: 1, 1: 1, 2: 2, 3: 3},
        {1: 1, 2: 2, 3: 3, 0: 4},
        {2: 1, 3: 2, 0: 3, 1: 4},
        {3: 1, 0: 2, 1: 3, 2: 4},
    ], candidates=[0, 1, 2, 3])
    
    approval_winner = approval_irv(prof)
    split_winner = split_irv(prof)
    
    assert approval_winner == [0]
    assert split_winner == [1]
    assert approval_winner != split_winner


def test_approval_split_same_on_linear_orders():
    """Test that Approval-IRV and Split-IRV give same result on linear orders."""
    # Linear orders (no ties) - both methods should give same result
    prof = ProfileWithTies([
        {0: 1, 1: 2, 2: 3},
        {0: 1, 1: 2, 2: 3},
        {1: 1, 0: 2, 2: 3},
    ], candidates=[0, 1, 2])
    
    assert approval_irv(prof) == split_irv(prof)


def test_instant_runoff_with_profile_with_ties_default():
    """Test that instant_runoff uses approval scoring by default for ProfileWithTies."""
    prof = ProfileWithTies([
        {0: 1, 1: 1, 2: 2, 3: 3},
        {0: 1, 1: 1, 2: 2, 3: 3},
        {1: 1, 2: 2, 3: 3, 0: 4},
        {2: 1, 3: 2, 0: 3, 1: 4},
        {3: 1, 0: 2, 1: 3, 2: 4},
    ], candidates=[0, 1, 2, 3])
    
    # Default should be approval scoring
    assert instant_runoff(prof) == approval_irv(prof)


def test_instant_runoff_with_profile_with_ties_score_method():
    """Test instant_runoff with explicit score_method parameter."""
    prof = ProfileWithTies([
        {0: 1, 1: 1, 2: 2, 3: 3},
        {0: 1, 1: 1, 2: 2, 3: 3},
        {1: 1, 2: 2, 3: 3, 0: 4},
        {2: 1, 3: 2, 0: 3, 1: 4},
        {3: 1, 0: 2, 1: 3, 2: 4},
    ], candidates=[0, 1, 2, 3])
    
    assert instant_runoff(prof, score_method="approval") == [0]
    assert instant_runoff(prof, score_method="split") == [1]


def test_instant_runoff_tb_with_profile_with_ties():
    """Test instant_runoff_tb with ProfileWithTies."""
    prof = ProfileWithTies([
        {0: 1, 1: 1, 2: 2, 3: 3},
        {0: 1, 1: 1, 2: 2, 3: 3},
        {1: 1, 2: 2, 3: 3, 0: 4},
        {2: 1, 3: 2, 0: 3, 1: 4},
        {3: 1, 0: 2, 1: 3, 2: 4},
    ], candidates=[0, 1, 2, 3])
    
    # With TB, eliminates one at a time - result is b=1
    result = instant_runoff_tb(prof, tie_breaker=[0, 1, 2, 3])
    assert result == [1]


def test_instant_runoff_put_with_profile_with_ties():
    """Test instant_runoff_put with ProfileWithTies."""
    prof = ProfileWithTies([
        {0: 1, 1: 1, 2: 2, 3: 3},
        {0: 1, 1: 1, 2: 2, 3: 3},
        {1: 1, 2: 2, 3: 3, 0: 4},
        {2: 1, 3: 2, 0: 3, 1: 4},
        {3: 1, 0: 2, 1: 3, 2: 4},
    ], candidates=[0, 1, 2, 3])
    
    # PUT explores all elimination paths - both a and b can win
    result = instant_runoff_put(prof)
    assert result == [0, 1]


def test_tie_breaker_convention():
    """Test that tie_breaker[0] has lowest priority (eliminated first)."""
    prof = ProfileWithTies([
        {0: 1, 1: 1, 2: 1},
    ], candidates=[0, 1, 2])
    
    # With TB [0, 1, 2]: 0 eliminated first, then 1, winner is 2
    assert approval_irv_tb(prof, tie_breaker=[0, 1, 2]) == [2]
    
    # With TB [2, 1, 0]: 2 eliminated first, then 1, winner is 0
    assert approval_irv_tb(prof, tie_breaker=[2, 1, 0]) == [0]
    
    # With TB [1, 0, 2]: 1 eliminated first, then 0, winner is 2
    assert approval_irv_tb(prof, tie_breaker=[1, 0, 2]) == [2]
