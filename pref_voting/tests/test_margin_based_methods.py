from pref_voting.margin_based_methods import *
from pref_voting.profiles import Profile
from pref_voting.profiles_with_ties import ProfileWithTies
from pref_voting.weighted_majority_graphs import MarginGraph
import pytest

@pytest.fixture
def condorcet_cycle():
    return MarginGraph([0, 1, 2], [(0, 1, 1), (1, 2, 1), (2, 0, 1)])

@pytest.fixture
def cycle():
    return MarginGraph([0, 1, 2], [(0, 1, 1), (1, 2, 3), (2, 0, 5)])

@pytest.fixture
def linear_margin_graph_0():
    return MarginGraph([0, 1, 2], [(0, 1, 1), (1, 2, 11), (0, 2, 1)])

@pytest.fixture
def condorcet_cycle_prof():
    return Profile([[0, 1, 2], [1, 2, 0], [2, 0, 1]])

@pytest.fixture
def condorcet_cycle_prof_with_ties():
    return ProfileWithTies([{'a':0, 'b':1, 'c':2}, {'a':1, 'b':2, 'c':0}, {'a':2, 'b':0, 'c':1}])

@pytest.mark.parametrize("voting_method, expected", [
    (minimax, {
        'condorcet_cycle': [0, 1, 2], 
        'cycle': [1], 
        'linear_margin_graph_0': [0],
        'linear_margin_graph_0_curr_cands': [1],
        'condorcet_cycle_prof': [0, 1, 2], 
        'condorcet_cycle_prof_with_ties': ['a', 'b', 'c']
    }),
    (split_cycle, {
        'condorcet_cycle': [0, 1, 2], 
        'cycle': [1], 
        'linear_margin_graph_0': [0],
        'linear_margin_graph_0_curr_cands': [1],
        'condorcet_cycle_prof': [0, 1, 2], 
        'condorcet_cycle_prof_with_ties': ['a', 'b', 'c']
    }),
    (split_cycle_Floyd_Warshall, {
        'condorcet_cycle': [0, 1, 2], 
        'cycle': [1], 
        'linear_margin_graph_0': [0],
        'linear_margin_graph_0_curr_cands': [1],
        'condorcet_cycle_prof': [0, 1, 2],         'condorcet_cycle_prof_with_ties': ['a', 'b', 'c']
    }),
    (beat_path, {
        'condorcet_cycle': [0, 1, 2], 
        'cycle': [1], 
        'linear_margin_graph_0': [0],
        'linear_margin_graph_0_curr_cands': [1],
        'condorcet_cycle_prof': [0, 1, 2],         'condorcet_cycle_prof_with_ties': ['a', 'b', 'c']
    }),
    (beat_path_Floyd_Warshall, {
        'condorcet_cycle': [0, 1, 2], 
        'cycle': [1], 
        'linear_margin_graph_0': [0],
        'linear_margin_graph_0_curr_cands': [1],
        'condorcet_cycle_prof': [0, 1, 2],         'condorcet_cycle_prof_with_ties': ['a', 'b', 'c']
    }),
    (ranked_pairs, {
        'condorcet_cycle': [0, 1, 2], 
        'cycle': [1], 
        'linear_margin_graph_0': [0],
        'linear_margin_graph_0_curr_cands': [1],
        'condorcet_cycle_prof': [0, 1, 2],         'condorcet_cycle_prof_with_ties': ['a', 'b', 'c']
    }),
    (ranked_pairs_with_test, {
        'condorcet_cycle': [0, 1, 2], 
        'cycle': [1], 
        'linear_margin_graph_0': [0],
        'linear_margin_graph_0_curr_cands': [1],
        'condorcet_cycle_prof': [0, 1, 2],         'condorcet_cycle_prof_with_ties': ['a', 'b', 'c']
    }),
    # (ranked_pairs_zt, {
    #     'condorcet_cycle': [0, 1, 2], 
    #     'cycle': [1], 
    #     'linear_margin_graph_0': [0],
    #     'linear_margin_graph_0_curr_cands': [1],
    #     'condorcet_cycle_prof': [0, 1, 2],         'condorcet_cycle_prof_with_ties': ['a', 'b', 'c']
    # }),
    (ranked_pairs_from_stacks, {
        'condorcet_cycle': [0, 1, 2], 
        'cycle': [1], 
        'linear_margin_graph_0': [0],
        'linear_margin_graph_0_curr_cands': [1],
        'condorcet_cycle_prof': [0, 1, 2],         'condorcet_cycle_prof_with_ties': ['a', 'b', 'c']
    }),
    (ranked_pairs_tb, {
        'condorcet_cycle': [0], 
        'cycle': [1], 
        'linear_margin_graph_0': [0],
        'linear_margin_graph_0_curr_cands': [1],
        'condorcet_cycle_prof': [0],         'condorcet_cycle_prof_with_ties': ['a']
    }),
    (river, {
        'condorcet_cycle': [0, 1, 2], 
        'cycle': [1], 
        'linear_margin_graph_0': [0],
        'linear_margin_graph_0_curr_cands': [1],
        'condorcet_cycle_prof': [0, 1, 2],         'condorcet_cycle_prof_with_ties': ['a', 'b', 'c']
    }),
    (river_with_test, {
        'condorcet_cycle': [0, 1, 2], 
        'cycle': [1], 
        'linear_margin_graph_0': [0],
        'linear_margin_graph_0_curr_cands': [1],
        'condorcet_cycle_prof': [0, 1, 2],         'condorcet_cycle_prof_with_ties': ['a', 'b', 'c']
    }),
    (river_tb, {
        'condorcet_cycle': [0], 
        'cycle': [1], 
        'linear_margin_graph_0': [0],
        'linear_margin_graph_0_curr_cands': [1],
        'condorcet_cycle_prof': [0],         'condorcet_cycle_prof_with_ties': ['a']
    }),
    (simple_stable_voting, {
        'condorcet_cycle': [0, 1, 2], 
        'cycle': [1], 
        'linear_margin_graph_0': [0],
        'linear_margin_graph_0_curr_cands': [1],
        'condorcet_cycle_prof': [0, 1, 2],         'condorcet_cycle_prof_with_ties': ['a', 'b', 'c']
    }),
    (simple_stable_voting_faster, {
        'condorcet_cycle': [0, 1, 2], 
        'cycle': [1], 
        'linear_margin_graph_0': [0],
        'linear_margin_graph_0_curr_cands': [1],
        'condorcet_cycle_prof': [0, 1, 2],         'condorcet_cycle_prof_with_ties': ['a', 'b', 'c']
    }),
    (stable_voting, {
        'condorcet_cycle': [0, 1, 2], 
        'cycle': [1], 
        'linear_margin_graph_0': [0],
        'linear_margin_graph_0_curr_cands': [1],
        'condorcet_cycle_prof': [0, 1, 2],         'condorcet_cycle_prof_with_ties': ['a', 'b', 'c']
    }),
    (stable_voting_faster, {
        'condorcet_cycle': [0, 1, 2], 
        'cycle': [1], 
        'linear_margin_graph_0': [0],
        'linear_margin_graph_0_curr_cands': [1],
        'condorcet_cycle_prof': [0, 1, 2],         'condorcet_cycle_prof_with_ties': ['a', 'b', 'c']
    }),
    (essential, {
        'condorcet_cycle': [0, 1, 2], 
        'cycle': [0, 1, 2], 
        'linear_margin_graph_0': [0],
        'linear_margin_graph_0_curr_cands': [1],
        'condorcet_cycle_prof': [0, 1, 2],         'condorcet_cycle_prof_with_ties': ['a', 'b', 'c']
    }),
    # (weighted_covering, {
    #     'condorcet_cycle': [0, 1, 2], 
    #     'cycle': [0, 1, 2], 
    #     'linear_margin_graph_0': [0],
    #     'linear_margin_graph_0_curr_cands': [1],
    #     'condorcet_cycle_prof': [0, 1, 2],         'condorcet_cycle_prof_with_ties': ['a', 'b', 'c']
    # }),
    (loss_trimmer, {
        'condorcet_cycle': [0, 1, 2], 
        'cycle': [1], 
        'linear_margin_graph_0': [0],
        'linear_margin_graph_0_curr_cands': [1],
        'condorcet_cycle_prof': [0, 1, 2],         'condorcet_cycle_prof_with_ties': ['a', 'b', 'c']
    })
])
def test_margin_based_methods(
    voting_method, 
    expected, 
    condorcet_cycle, 
    cycle,
    linear_margin_graph_0,
    condorcet_cycle_prof,
    condorcet_cycle_prof_with_ties):
    assert voting_method(condorcet_cycle) == expected['condorcet_cycle']
    assert voting_method(cycle) == expected['cycle']
    assert voting_method(linear_margin_graph_0) == expected['linear_margin_graph_0']
    if 'linear_margin_graph_0_curr_cands' in expected:
        assert voting_method(linear_margin_graph_0, curr_cands=[1, 2]) == expected['linear_margin_graph_0_curr_cands']

    assert voting_method(condorcet_cycle_prof) == expected['condorcet_cycle_prof']
    assert voting_method(condorcet_cycle_prof_with_ties) == expected['condorcet_cycle_prof_with_ties']
