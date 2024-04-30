from pref_voting.scoring_methods import *

import pytest


@pytest.mark.parametrize("voting_method, expected", [
    (plurality, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1],
        'profile_single_voter': [0],
        'profile_single_voter_curr_cands': [0]
        }),
    (borda, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1],
        'profile_single_voter': [0],
        'profile_single_voter_curr_cands': [0]
        }),
    (dowdall, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1],
        'profile_single_voter': [0],
        'profile_single_voter_curr_cands': [0]
        }),
    (anti_plurality, {
        'condorcet_cycle': [0, 1, 2],
        'linear_profile_0': [1], 
        'linear_profile_0_curr_cands': [1],
        'profile_single_voter': [0, 1, 2],
        'profile_single_voter_curr_cands': [0]
        }),
    (positive_negative_voting, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1],
        'profile_single_voter': [0],
        'profile_single_voter_curr_cands': [0]
        }),
])
def test_scoring_rules(
    voting_method, 
    expected, 
    condorcet_cycle, 
    linear_profile_0,
    profile_single_voter):
    assert voting_method(condorcet_cycle) == expected['condorcet_cycle']
    assert voting_method(linear_profile_0) == expected['linear_profile_0']
    if 'linear_profile_0_curr_cands' in expected:
        assert voting_method(linear_profile_0, curr_cands=[1, 2]) == expected['linear_profile_0_curr_cands']
    assert voting_method(profile_single_voter) == expected['profile_single_voter']
    assert voting_method(profile_single_voter, curr_cands = [0, 1]) == expected['profile_single_voter_curr_cands']

def test_scoring_rule(condorcet_cycle, linear_profile_0):
    assert scoring_rule(condorcet_cycle) == [0, 1, 2]
    assert scoring_rule(condorcet_cycle, 
                        score = lambda num_cands, rank: 1 if rank==2 else 0) == [0, 1, 2]
    assert scoring_rule(condorcet_cycle, curr_cands=[0, 1],
                        score = lambda num_cands, rank: 1 if rank==2 else 0) == [1]

    assert scoring_rule(linear_profile_0) == [0]
    assert scoring_rule(linear_profile_0, 
                        score = lambda num_cands, rank: 1 if rank==2 else 0) == [1]

def test_plurality_on_profiles_with_ties(profile_with_ties, profile_with_ties_linear_0):
    assert plurality(profile_with_ties_linear_0) == [0]
    with pytest.raises(AssertionError) as excinfo:
        plurality(profile_with_ties)
    assert "Cannot calculate plurality scores." in str(excinfo.value)

def test_borda_on_profiles_with_ties(profile_with_ties, profile_with_ties_linear_0):
    assert borda_for_profile_with_ties(profile_with_ties_linear_0) == [0]
    assert borda_for_profile_with_ties(profile_with_ties) == [0, 1]
    assert borda_for_profile_with_ties(profile_with_ties, borda_scores=symmetric_borda_scores) == [0, 1]
    assert borda_for_profile_with_ties(profile_with_ties, borda_scores=domination_borda_scores) == [0, 1, 2]
    assert borda_for_profile_with_ties(profile_with_ties, borda_scores=weak_domination_borda_scores) == [0, 1]
    assert borda_for_profile_with_ties(profile_with_ties, borda_scores=non_domination_borda_scores) == [0, 1]

    assert borda_for_profile_with_ties(profile_with_ties, curr_cands=[1, 2]) == [1]
    assert borda_for_profile_with_ties(profile_with_ties, borda_scores=symmetric_borda_scores, curr_cands=[1, 2]) == [1]
    assert borda_for_profile_with_ties(profile_with_ties, borda_scores=domination_borda_scores, curr_cands=[1, 2]) == [1]
    assert borda_for_profile_with_ties(profile_with_ties, borda_scores=weak_domination_borda_scores, curr_cands=[1, 2]) == [1]
    assert borda_for_profile_with_ties(profile_with_ties, borda_scores=non_domination_borda_scores, curr_cands=[1, 2]) == [1]

def test_plurality_ranking(condorcet_cycle, linear_profile_0, profile_with_ties, profile_with_ties_linear_0):
    assert plurality_ranking(condorcet_cycle) == Ranking({0:1, 1:1, 2:1})
    assert plurality_ranking(condorcet_cycle, local=False) == Ranking({0:1, 1:1, 2:1})
    assert plurality_ranking(condorcet_cycle, tie_breaking='alphabetic') == Ranking({0:1, 1:2, 2:3})
    assert plurality_ranking(linear_profile_0) == Ranking({0:1, 1:3, 2:2})
    assert plurality_ranking(profile_with_ties_linear_0) == Ranking({0:1, 1:3, 2:2})
    with pytest.raises(AssertionError) as excinfo:
        plurality_ranking(profile_with_ties)
    assert "Cannot calculate plurality scores." in str(excinfo.value)
 
def test_plurality_ranking_curr_cands(condorcet_cycle, linear_profile_0, profile_with_ties_linear_0):
    assert plurality_ranking(condorcet_cycle, curr_cands=[1, 2], local=False) == Ranking({1:1, 2:1})
    assert plurality_ranking(condorcet_cycle, curr_cands=[1, 2], local=True) == Ranking({1:1, 2:2})
    assert plurality_ranking(linear_profile_0, curr_cands=[1, 2]) == Ranking({1:1, 2:2})
    assert plurality_ranking(profile_with_ties_linear_0, curr_cands=[1, 2]) ==  Ranking({1:1, 2:2})

def test_borda_ranking(condorcet_cycle, linear_profile_0):
    assert borda_ranking(condorcet_cycle) == Ranking({0:1, 1:1, 2:1})
    assert borda_ranking(condorcet_cycle, curr_cands=[1, 2]) == Ranking({1:1, 2:2})
    assert borda_ranking(condorcet_cycle, curr_cands=[1, 2], local=False) == Ranking({1:1, 2:1})
    assert borda_ranking(condorcet_cycle, tie_breaking='alphabetic') == Ranking({0:1, 1:2, 2:3})
    assert borda_ranking(linear_profile_0) == Ranking({0:1, 1:2, 2:3})

def test_score_ranking(condorcet_cycle, linear_profile_0):
    assert score_ranking(condorcet_cycle) == Ranking({0:1, 1:1, 2:1})
    assert score_ranking(linear_profile_0, score = lambda num_cands, rank: 1 if rank==2 else 0) == Ranking({0:2, 1:1, 2:2})

def test_create_scoring_method():
    prof = Profile([[0, 1, 2]])
    scoring_method = create_scoring_method(lambda ncs, x: 1, "test")
    assert type(scoring_method) == VotingMethod
    assert scoring_method.name == "test"
    assert scoring_method(prof) == [0, 1, 2]