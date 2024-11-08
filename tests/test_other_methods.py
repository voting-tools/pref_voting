from pref_voting.other_methods import *
from pref_voting.profiles import Profile
import pytest

@pytest.mark.parametrize("voting_method, expected", [
    (absolute_majority, {
        'condorcet_cycle': [], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1]
    }),
    (pareto, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0, 1, 2], 
        'linear_profile_0_curr_cands': [1, 2]
    }),
    (kemeny_young, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1]
    }),
    (bucklin, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1]
    }),
    (simplified_bucklin, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1]
    }),
    (weighted_bucklin, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1]
    }),
    (superior_voting, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1]
    })
])
def test_other_methods(
    voting_method, 
    expected, 
    condorcet_cycle, 
    linear_profile_0):
    assert voting_method(condorcet_cycle) == expected['condorcet_cycle']
    assert voting_method(linear_profile_0) == expected['linear_profile_0']
    if 'linear_profile_0_curr_cands' in expected:
        assert voting_method(linear_profile_0, curr_cands=[1, 2]) == expected['linear_profile_0_curr_cands']


def test_pareto(): 
    prof = Profile([[0, 1, 2], [0, 2, 1], [2, 0, 1]])
    assert pareto(prof) == [0, 2]

    prof = ProfileWithTies([
        {0:1, 1:1},
        {0:1, 1:1, 2:2}
    ])
    curr_using_extended_strict_preference = prof.using_extended_strict_preference
    assert pareto(prof, use_extended_strict_preferences=False) == [0, 1, 2]
    assert prof.using_extended_strict_preference == curr_using_extended_strict_preference

    curr_using_extended_strict_preference = prof.using_extended_strict_preference
    assert pareto(prof, use_extended_strict_preferences=True) == [0, 1]
    assert prof.using_extended_strict_preference == curr_using_extended_strict_preference

def test_bracket(condorcet_cycle, linear_profile_0):
    assert bracket_voting(condorcet_cycle, seed=42) == [2]
    assert bracket_voting(linear_profile_0) == [0]
    assert bracket_voting(linear_profile_0, curr_cands=[1, 2]) == [1]


def test_bucklin_with_explanation(condorcet_cycle, linear_profile_0):
    ws, exp = bucklin_with_explanation(condorcet_cycle)
    assert ws == [0, 1, 2]
    assert exp == {0: 2, 1: 2, 2: 2}
    ws, exp = bucklin_with_explanation(linear_profile_0)
    assert ws == [0]
    assert exp ==  {0: 2, 1: 0, 2: 1}
    prof = Profile([[0, 1, 2], [0, 1, 2], [1, 2, 0], [2, 1, 0]])
    ws, exp = bucklin_with_explanation(prof)
    assert ws == [1]
    assert exp == {0: 2, 1: 4, 2: 2}


def test_simplified_bucklin_with_explanation(condorcet_cycle, linear_profile_0):
    ws, exp = simplified_bucklin_with_explanation(condorcet_cycle)
    assert ws == [0, 1, 2]
    assert exp == {0: 2, 1: 2, 2: 2}
    ws, exp = simplified_bucklin_with_explanation(linear_profile_0)
    assert ws == [0]
    assert exp ==  {0: 2, 1: 0, 2: 1}
    prof = Profile([[0, 1, 2], [0, 1, 2], [1, 2, 0], [2, 1, 0]])
    ws, exp = simplified_bucklin_with_explanation(prof)
    assert ws == [1]
    assert exp == {0: 2, 1: 4, 2: 2}


def test_kemeny_young_rankings(condorcet_cycle, linear_profile_0):
    assert kemeny_young_rankings(condorcet_cycle) == ([(0, 1, 2), (1, 2, 0), (2, 0, 1)],4)
    assert kemeny_young_rankings(linear_profile_0) == ([(0, 1, 2)], 3)