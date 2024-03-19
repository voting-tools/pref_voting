from pref_voting.iterative_methods import *
from pref_voting.profiles import Profile
from pref_voting.profiles_with_ties import ProfileWithTies
import pytest

@pytest.mark.parametrize("voting_method, expected", [
    (instant_runoff, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1]}),
    (ranked_choice, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1]}),
    (hare, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1]}),
    (instant_runoff_tb, {
        'condorcet_cycle': [1], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1]}),
    (instant_runoff_put, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1]}),
    (bottom_two_runoff_instant_runoff, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1]}),
    (bottom_two_runoff_instant_runoff_put, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1]}),
    (benham, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1]
    }),
    (benham_put, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1]
    }),
    (benham_tb, {
        'condorcet_cycle': [1], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1]
    }),
    (plurality_with_runoff_put, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1]
    }),
    (coombs, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1]
    }),
    (coombs_tb, {
        'condorcet_cycle': [1], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1]
    }),
    (coombs_put, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1]
    }),
    (baldwin, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1]
    }),
    (baldwin_tb, {
        'condorcet_cycle': [1], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1]
    }),
    (baldwin_put, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1]
    }),
    (strict_nanson, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1]
    }),
    (weak_nanson, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1]
    }),
    (iterated_removal_cl, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1]
    }),
    (raynaud, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1]
    }),
    (tideman_alternative_smith, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1]
    }),
    (tideman_alternative_smith_put, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1]
    }),
    (tideman_alternative_schwartz, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1]
    }),
    (tideman_alternative_schwartz_put, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1]
    }),
    (woodall, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1]
    }),
    (knockout, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_profile_0': [0], 
        'linear_profile_0_curr_cands': [1]
    })
])
def test_methods(
    voting_method, expected, 
    condorcet_cycle, 
    linear_profile_0):
    assert voting_method(condorcet_cycle) == expected['condorcet_cycle']
    assert voting_method(linear_profile_0) == expected['linear_profile_0']
    if 'linear_profile_0_curr_cands' in expected:
        assert voting_method(linear_profile_0, curr_cands=[1, 2]) == expected['linear_profile_0_curr_cands']


def test_instant_runoff_for_truncated_linear_orders():
    prof = ProfileWithTies([
        {0:1, 1:2},
        {1:1, 0:3},
        {0:2, 1:1, 2:0}
    ],
    candidates=[0, 1, 2])   

    assert instant_runoff_for_truncated_linear_orders(prof) == [0, 1, 2]

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
    assert exp == [[1,2]]


def test_coombs_with_explanation(condorcet_cycle, linear_profile_0):
    ws, exp = coombs_with_explanation(condorcet_cycle)
    assert ws == [0, 1, 2]
    assert exp == [[0,1,2]]
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

