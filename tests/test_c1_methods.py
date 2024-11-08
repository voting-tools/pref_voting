from pref_voting.c1_methods import *
from pref_voting.profiles import Profile
from pref_voting.profiles_with_ties import ProfileWithTies
from pref_voting.weighted_majority_graphs import MajorityGraph, MarginGraph
import pytest

@pytest.fixture
def condorcet_cycle():
    return MajorityGraph([0, 1, 2], [(0, 1), (1, 2), (2, 0)])

@pytest.fixture
def linear_maj_graph_0():
    return MajorityGraph([0, 1, 2], [(0, 1), (1, 2), (0, 2)])

@pytest.fixture
def condorcet_cycle_prof():
    return Profile([[0, 1, 2], [1, 2, 0], [2, 0, 1]])

@pytest.fixture
def condorcet_cycle_prof_with_ties():
    return ProfileWithTies([{0:0, 1:1, 2:2}, {0:1, 1:2, 2:0}, {0:2, 1:0, 2:1}])

@pytest.fixture
def condorcet_cycle_margin():
    return MarginGraph([0, 1, 2], [(0, 1, 1), (1, 2, 5), (2, 0, 3)])

@pytest.fixture
def profile_single_voter():
    return Profile([[0, 1, 2]])

@pytest.mark.parametrize("voting_method, expected", [
(condorcet, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_maj_graph_0': [0], 
        'linear_maj_graph_0_curr_cands': [1],
        'condorcet_cycle_prof': [0, 1, 2], 
        'condorcet_cycle_margin': [0, 1, 2],
        'condorcet_cycle_prof_with_ties': [0, 1, 2],
        'profile_single_voter': [0]
    }),
    (copeland, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_maj_graph_0': [0], 
        'linear_maj_graph_0_curr_cands': [1],
        'condorcet_cycle_prof': [0, 1, 2], 
        'condorcet_cycle_margin': [0, 1, 2],
        'condorcet_cycle_prof_with_ties': [0, 1, 2],
        'profile_single_voter': [0]
    }),
    (uc_gill, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_maj_graph_0': [0], 
        'linear_maj_graph_0_curr_cands': [1],
        'condorcet_cycle_prof': [0, 1, 2], 
        'condorcet_cycle_margin': [0, 1, 2],
        'condorcet_cycle_prof_with_ties': [0, 1, 2],
        'profile_single_voter': [0]
    }),
    (llull, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_maj_graph_0': [0], 
        'linear_maj_graph_0_curr_cands': [1],
        'condorcet_cycle_prof': [0, 1, 2], 
        'condorcet_cycle_margin': [0, 1, 2],
        'condorcet_cycle_prof_with_ties': [0, 1, 2],
        'profile_single_voter': [0]
    }),
    (top_cycle, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_maj_graph_0': [0], 
        'linear_maj_graph_0_curr_cands': [1],
        'condorcet_cycle_prof': [0, 1, 2], 
        'condorcet_cycle_margin': [0, 1, 2],
        'condorcet_cycle_prof_with_ties': [0, 1, 2],
        'profile_single_voter': [0]
    }),
    (uc_fish, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_maj_graph_0': [0], 
        'linear_maj_graph_0_curr_cands': [1],
        'condorcet_cycle_prof': [0, 1, 2], 
        'condorcet_cycle_margin': [0, 1, 2],
        'condorcet_cycle_prof_with_ties': [0, 1, 2],
        'profile_single_voter': [0]
    }),
    (uc_bordes, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_maj_graph_0': [0], 
        'linear_maj_graph_0_curr_cands': [1],
        'condorcet_cycle_prof': [0, 1, 2], 
        'condorcet_cycle_margin': [0, 1, 2],
        'condorcet_cycle_prof_with_ties': [0, 1, 2],
        'profile_single_voter': [0]
    }),
    (uc_mckelvey, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_maj_graph_0': [0], 
        'linear_maj_graph_0_curr_cands': [1],
        'condorcet_cycle_prof': [0, 1, 2], 
        'condorcet_cycle_margin': [0, 1, 2],
        'condorcet_cycle_prof_with_ties': [0, 1, 2],
        'profile_single_voter': [0]
    }),
    (gocha, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_maj_graph_0': [0], 
        'linear_maj_graph_0_curr_cands': [1],
        'condorcet_cycle_prof': [0, 1, 2], 
        'condorcet_cycle_margin': [0, 1, 2],
        'condorcet_cycle_prof_with_ties': [0, 1, 2],
        'profile_single_voter': [0]
    }),
    (banks, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_maj_graph_0': [0], 
        'linear_maj_graph_0_curr_cands': [1],
        'condorcet_cycle_prof': [0, 1, 2], 
        'condorcet_cycle_margin': [0, 1, 2],
        'condorcet_cycle_prof_with_ties': [0, 1, 2],
        'profile_single_voter': [0]
    }),
    (slater, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_maj_graph_0': [0], 
        'linear_maj_graph_0_curr_cands': [1],
        'condorcet_cycle_prof': [0, 1, 2], 
        'condorcet_cycle_margin': [0, 1, 2],
        'condorcet_cycle_prof_with_ties': [0, 1, 2],
        'profile_single_voter': [0]
    }),
    (bipartisan, {
        'condorcet_cycle': [0, 1, 2], 
        'linear_maj_graph_0': [0], 
        'linear_maj_graph_0_curr_cands': [1],
        'condorcet_cycle_prof': [0, 1, 2], 
        'condorcet_cycle_margin': [0, 1, 2],
        'condorcet_cycle_prof_with_ties': [0, 1, 2],
        'profile_single_voter': [0]
    })
])
def test_c1_methods(
    voting_method, 
    expected, 
    condorcet_cycle, 
    linear_maj_graph_0,
    condorcet_cycle_prof,
    condorcet_cycle_margin,
    condorcet_cycle_prof_with_ties,
    profile_single_voter):
    assert voting_method(condorcet_cycle) == expected['condorcet_cycle']
    assert voting_method(linear_maj_graph_0) == expected['linear_maj_graph_0']
    if 'linear_maj_graph_0_curr_cands' in expected:
        assert voting_method(linear_maj_graph_0, curr_cands=[1, 2]) == expected['linear_maj_graph_0_curr_cands']

    assert voting_method(condorcet_cycle_prof) == expected['condorcet_cycle_prof']
    assert voting_method(condorcet_cycle_margin) == expected['condorcet_cycle_margin']
    assert voting_method(condorcet_cycle_prof_with_ties) == expected['condorcet_cycle_prof_with_ties']
    assert voting_method(profile_single_voter) == expected['profile_single_voter']
