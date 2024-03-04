from pref_voting.profiles import Profile
from pref_voting.profiles_with_ties import ProfileWithTies
from pref_voting.rankings import Ranking
from pref_voting.weighted_majority_graphs import MarginGraph, MajorityGraph, SupportGraph
from pref_voting.margin_based_methods import split_cycle_defeat
import numpy as np
import pytest
from collections import Counter
from preflibtools.instances import OrdinalInstance

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

@pytest.fixture
def test_profile_with_ties():
    return ProfileWithTies(
        [
            {0:1, 1:2},
            {1:1, 2:2, 0:3},
            {2:1, 0:1}
        ], 
        [2, 3, 1])

@pytest.fixture
def condorcet_cycle_profile_with_ties():
    return ProfileWithTies(
        [
            {0:1, 1:2, 2:3},
            {0:2, 1:3, 2:1},
            {0:3, 1:1, 2:2},
        ])

@pytest.fixture
def test_profile_with_ties2():
    return ProfileWithTies(
        [
            {0:1, 1:2},
            {0:2, 1:1},
            {0:1, 1:1}
        ])

def test_create_profile_with_ties_from_dicts():
    prof = ProfileWithTies([
        {0:1, 1:2}, 
        {2:1}, 
        {0:3, 1:1, 2:2}], 
        [2, 3, 1])
    assert prof.num_cands == 3
    assert prof.candidates == [0, 1, 2]
    assert prof.num_voters == 6
    assert prof.cindices == [0, 1, 2]

def test_create_profile_with_ties_from_rankings():
    prof = ProfileWithTies([
        Ranking({0:1, 1:2}), 
        {2:1}, 
        Ranking({0:3, 1:1, 2:2})], 
        [2, 3, 1])
    assert prof.num_cands == 3
    assert prof.candidates == [0, 1, 2]
    assert prof.num_voters == 6
    assert prof.cindices == [0, 1, 2]

def test_support(test_profile_with_ties):
    assert test_profile_with_ties.support(0, 1) == 2
    assert test_profile_with_ties.support(1, 0) == 3
    assert test_profile_with_ties.support(2, 0) == 3
    assert test_profile_with_ties.support(0, 2) == 0
    assert test_profile_with_ties.support(1, 2) == 3
    assert test_profile_with_ties.support(2, 1) == 0

def test_margin(test_profile_with_ties):
    assert test_profile_with_ties.margin(0, 1) == -1 
    assert test_profile_with_ties.margin(1, 0) == 1 
    assert test_profile_with_ties.margin(2, 0) == 3
    assert test_profile_with_ties.margin(0, 2) == -3
    assert test_profile_with_ties.margin(1, 2) == 3
    assert test_profile_with_ties.margin(2, 1) == -3

def test_use_extended_strict_reference(test_profile_with_ties):
    test_profile_with_ties.use_extended_strict_preference()
    assert test_profile_with_ties.support(0, 1) == 3
    assert test_profile_with_ties.support(1, 0) == 3
    assert test_profile_with_ties.support(2, 0) == 3
    assert test_profile_with_ties.support(0, 2) == 2
    assert test_profile_with_ties.support(1, 2) == 5
    assert test_profile_with_ties.support(2, 1) == 1

def test_use_strict_reference(test_profile_with_ties):
    test_profile_with_ties.use_extended_strict_preference()
    assert test_profile_with_ties.support(0, 1) == 3
    assert test_profile_with_ties.support(1, 0) == 3
    assert test_profile_with_ties.support(2, 0) == 3
    assert test_profile_with_ties.support(0, 2) == 2
    assert test_profile_with_ties.support(1, 2) == 5
    assert test_profile_with_ties.support(2, 1) == 1

    test_profile_with_ties.use_strict_preference()
    assert test_profile_with_ties.support(0, 1) == 2
    assert test_profile_with_ties.support(1, 0) == 3
    assert test_profile_with_ties.support(2, 0) == 3
    assert test_profile_with_ties.support(0, 2) == 0
    assert test_profile_with_ties.support(1, 2) == 3
    assert test_profile_with_ties.support(2, 1) == 0

def test_rankings(test_profile_with_ties): 
    rankings = test_profile_with_ties.rankings

    expected_values = [Ranking({0:1, 1:2}),Ranking({0:1, 1:2}), Ranking({1:1, 2:2, 0:3}),Ranking({1:1, 2:2, 0:3}),Ranking({1:1, 2:2, 0:3}),Ranking({2:1, 0:1})]

    assert len(rankings) == len(expected_values) and all([r in rankings for r in expected_values]) and all([r in expected_values for r in rankings])
    
def test_ranking_types(test_profile_with_ties): 
    rankings = test_profile_with_ties.ranking_types

    expected_values = [Ranking({0:1, 1:2}), Ranking({1:1, 2:2, 0:3}),Ranking({2:1, 0:1})]

    assert len(rankings) == len(expected_values) and all([r in rankings for r in expected_values]) and all([r in expected_values for r in rankings])
    
def test_ranking_counts(test_profile_with_ties): 
    rankings, counts = test_profile_with_ties.rankings_counts

    expected_values = [Ranking({0:1, 1:2}), Ranking({1:1, 2:2, 0:3}),Ranking({2:1, 0:1})]

    assert len(rankings) == len(expected_values) and all([r in rankings for r in expected_values]) and all([r in expected_values for r in rankings])
    assert sorted([2, 3, 1]) == sorted(counts)
    
def test_rankings_as_dicts_counts(test_profile_with_ties):
    rankings, counts = test_profile_with_ties.rankings_as_dicts_counts
    expected_rankings = [{0:1, 1:2}, {1:1, 2:2, 0:3}, {2:1, 0:1}]
    expected_counts = [2, 3, 1]
    assert rankings == expected_rankings
    assert sorted(expected_counts) == sorted(counts)

def test_is_tied(test_profile_with_ties, test_profile_with_ties2): 
    assert not test_profile_with_ties.is_tied(0, 1) 
    assert test_profile_with_ties2.is_tied(0, 1)
    assert test_profile_with_ties2.is_tied(1, 0)

def test_dominators(test_profile_with_ties, test_profile_with_ties2):

    assert test_profile_with_ties.dominators(0) == [1, 2]
    assert test_profile_with_ties.dominators(0, curr_cands=[0, 1]) == [1]
    assert test_profile_with_ties.dominators(1) == []
    assert test_profile_with_ties.dominators(2) == [1]
    assert test_profile_with_ties2.dominators(0) == []
    assert test_profile_with_ties2.dominators(1) == []

def test_dominates(test_profile_with_ties, test_profile_with_ties2):

    assert test_profile_with_ties.dominates(1) == [0, 2]
    assert test_profile_with_ties.dominates(1, curr_cands=[1, 2]) == [2]
    assert test_profile_with_ties.dominates(2) == [0]
    assert test_profile_with_ties2.dominates(0) == []
    assert test_profile_with_ties2.dominates(1) == []

def test_ratio(test_profile_with_ties, test_profile_with_ties2):
    assert test_profile_with_ties.ratio(0, 1) == 0.6666666666666666
    assert test_profile_with_ties.ratio(1, 0) == 1.5
    assert test_profile_with_ties.ratio(2, 0) == 9.0
    assert test_profile_with_ties.ratio(0, 2) == 0.1111111111111111
    assert test_profile_with_ties.ratio(1, 2) == 9.0
    assert test_profile_with_ties.ratio(2, 1) == 0.1111111111111111
    assert test_profile_with_ties2.ratio(0, 1) == 1.0
    assert test_profile_with_ties2.ratio(1, 0) == 1.0

def test_majority_prefers(test_profile_with_ties, test_profile_with_ties2):

    assert not test_profile_with_ties.majority_prefers(0, 1)
    assert test_profile_with_ties.majority_prefers(1, 0)
    assert test_profile_with_ties.majority_prefers(2, 0)
    assert not test_profile_with_ties.majority_prefers(0, 2)
    assert test_profile_with_ties.majority_prefers(1, 2)
    assert not test_profile_with_ties.majority_prefers(2, 1)
    assert not test_profile_with_ties2.majority_prefers(0, 1)
    assert not test_profile_with_ties2.majority_prefers(1, 0)

def test_strength_matrix(test_profile_with_ties): 
    strength_matrix, cands_to_cidx = test_profile_with_ties.strength_matrix()
    # check that two np arrays are equal
    np.testing.assert_array_equal(strength_matrix, np.array([[0, -1, -3], [1,  0,  3], [3, -3, 0]]))
    assert cands_to_cidx(0) == 0
    assert cands_to_cidx(1) == 1
    assert cands_to_cidx(2) == 2


    strength_matrix, cands_to_cidx = test_profile_with_ties.strength_matrix(curr_cands=[1,2])
    # check that two np arrays are equal
    np.testing.assert_array_equal(strength_matrix, np.array([[0, 3], [-3,0]]))
    assert cands_to_cidx(1) == 0
    assert cands_to_cidx(2) == 1

def test_condorcet_winner(test_profile_with_ties, condorcet_cycle_profile_with_ties):
    prof = ProfileWithTies([{0:1, 1:1}])
    assert test_profile_with_ties.condorcet_winner() == 1
    assert test_profile_with_ties.condorcet_winner(curr_cands=[0,2]) == 2
    assert condorcet_cycle_profile_with_ties.condorcet_winner() is None
    assert prof.condorcet_winner() is None

def test_condorcet_loser(test_profile_with_ties, condorcet_cycle_profile_with_ties):
    prof = ProfileWithTies([{0:1, 1:1}])
    assert test_profile_with_ties.condorcet_loser() == 0
    assert condorcet_cycle_profile_with_ties.condorcet_loser() is None
    assert condorcet_cycle_profile_with_ties.condorcet_loser(curr_cands=[0,2]) == 0
    assert prof.condorcet_loser() is None

def test_weak_condorcet_winner(test_profile_with_ties, condorcet_cycle_profile_with_ties):
    prof = ProfileWithTies([{0:1, 1:1}])
    assert test_profile_with_ties.weak_condorcet_winner() == [1]
    assert test_profile_with_ties.weak_condorcet_winner(curr_cands=[0,2]) == [2]
    assert condorcet_cycle_profile_with_ties.weak_condorcet_winner() is None
    assert condorcet_cycle_profile_with_ties.weak_condorcet_winner(curr_cands=[0, 1]) == [0]
    assert prof.weak_condorcet_winner() == [0, 1]

def test_copeland_scores(test_profile_with_ties, condorcet_cycle_profile_with_ties):
    prof = ProfileWithTies([{0:1, 1:1}])
    assert test_profile_with_ties.copeland_scores() == {0: -2.0, 1: 2.0, 2: 0.0}
    assert test_profile_with_ties.copeland_scores(curr_cands=[1,2]) == {1: 1.0, 2: 0.-1.0}
    assert condorcet_cycle_profile_with_ties.copeland_scores() == {0: 0.0, 1: 0.0, 2: 0.0}
    assert condorcet_cycle_profile_with_ties.copeland_scores(curr_cands=[0, 2]) == {0: -1.0, 2: 1.0}
    assert prof.copeland_scores() == {0: 0.0, 1: 0.0}

def test_strict_maj_size(test_profile_with_ties, condorcet_cycle_profile_with_ties):
    assert test_profile_with_ties.strict_maj_size() == 4
    assert condorcet_cycle_profile_with_ties.strict_maj_size() == 2

def test_plurality_scores(test_profile_with_ties, condorcet_cycle_profile_with_ties):
    assert test_profile_with_ties.plurality_scores() == {}
    assert condorcet_cycle_profile_with_ties.plurality_scores() == {0:1, 1:1, 2:1}
    assert condorcet_cycle_profile_with_ties.plurality_scores(curr_cands=[0, 2]) == {0:1, 2:2}

def test_plurality_scores_ignoring_overvotes(test_profile_with_ties, condorcet_cycle_profile_with_ties):
    assert test_profile_with_ties.plurality_scores_ignoring_overvotes() == {0:2, 1:3, 2:0}
    assert condorcet_cycle_profile_with_ties.plurality_scores_ignoring_overvotes() == {0:1, 1:1, 2:1}
    assert condorcet_cycle_profile_with_ties.plurality_scores_ignoring_overvotes(curr_cands=[0, 2]) == {0:1, 2:2}

def test_borda_scores(test_profile_with_ties, condorcet_cycle_profile_with_ties):
    assert test_profile_with_ties.borda_scores() == {0: -1, 1: 4, 2: -3}
    assert condorcet_cycle_profile_with_ties.borda_scores() == {0: 0, 1: 0, 2: 0}
    assert condorcet_cycle_profile_with_ties.borda_scores(curr_cands=[0, 2]) == {0: -1, 2: 1}

def test_remove_empty_rankings(): 
    prof = ProfileWithTies([{}, {}, {0:1, 1:2}])
    prof_without_ties = ProfileWithTies([{0:1, 1:2}])

    assert prof != prof_without_ties

    prof.remove_empty_rankings()
    assert prof == prof_without_ties

def test_truncate_overvotes(): 
    prof = ProfileWithTies([
        {0:1, 1:2}, 
        {1:1, 2:2, 0:1}, 
        {1:2, 2:1, 0:2}, 
        {2:1, 0:1}])
    
    prof2 = ProfileWithTies([
        {0:1, 1:2}, 
        {}, 
        {2:1}, 
        {}])
    
    prof_truncated, report = prof.truncate_overvotes()
    assert prof_truncated == prof2

def test_is_truncated_linear(condorcet_cycle_profile_with_ties):
    assert condorcet_cycle_profile_with_ties.is_truncated_linear()
    prof = ProfileWithTies([
        {0:1}, 
        {0:1, 1:2}, 
        {0:1, 1:2, 2:3}])
    assert prof.is_truncated_linear()
    prof = ProfileWithTies([
        {0:1}, 
        {0:1, 1:2}, 
        {0:1, 1:2, 2:3}], candidates=[0, 1, 2, 3])
    assert prof.is_truncated_linear()
    prof = ProfileWithTies([
        {},
        {0:1}, 
        {0:1, 1:2}, 
        {0:1, 1:2, 2:3}])
    assert prof.is_truncated_linear()
    prof = ProfileWithTies([
        {0:1}, 
        {0:1, 1:2}, 
        {0:1, 1:2, 2:2}])
    assert not prof.is_truncated_linear()

    prof = ProfileWithTies([
        {0:1}, 
        {0:1, 1:1}, 
        {0:1, 1:2, 2:3}])
    assert not prof.is_truncated_linear()


def test_add_unranked_candidates(): 
    prof = ProfileWithTies([
        {0:1, 1:2}, 
        {1:1, 2:2, 0:1}, 
        {2:1, 0:1}],
        candidates=[0, 1, 2, 3])
    prof2 = ProfileWithTies([
        {0:1, 1:2, 2:3, 3:3}, 
        {1:1, 2:2, 0:1, 3:3}, 
        {2:1, 0:1, 1:3, 3:3},],
        candidates=[0, 1, 2, 3])
    
    assert prof != prof2

    assert prof.add_unranked_candidates() == prof2

def to_linear_profile(condorcet_cycle_profile_with_ties):
    prof = condorcet_cycle_profile_with_ties.to_linear_profile()
    assert type(prof) == Profile
    assert prof == Profile([
        [0, 1, 2], 
        [1, 2, 0], 
        [2, 0, 1]])
    prof = ProfileWithTies([{0:1, 1:2}], candidates=[0, 1])
    prof2 = prof.to_linear_profile()
    assert type(prof2) == Profile
    assert prof2 == Profile([[0, 1]])
    prof = ProfileWithTies([{0:1, 1:2}], candidates=[0, 1, 2])
    prof2 = prof.to_linear_profile()
    assert prof2 is None

def test_margin_graph(test_profile_with_ties, condorcet_cycle_profile_with_ties):

    mg = test_profile_with_ties.margin_graph()
    assert isinstance(mg, MarginGraph)
    assert test_profile_with_ties.candidates == mg.candidates
    assert test_profile_with_ties.margin(0, 1) == mg.margin(0, 1)
    assert test_profile_with_ties.margin(0, 2) == mg.margin(0, 2)
    assert test_profile_with_ties.margin(1, 2) == mg.margin(1, 2)

    mg = condorcet_cycle_profile_with_ties.margin_graph()
    assert isinstance(mg, MarginGraph)
    assert condorcet_cycle_profile_with_ties.candidates == mg.candidates
    assert condorcet_cycle_profile_with_ties.margin(0, 1) == mg.margin(0, 1)
    assert condorcet_cycle_profile_with_ties.margin(0, 2) == mg.margin(0, 2)
    assert condorcet_cycle_profile_with_ties.margin(1, 2) == mg.margin(1, 2)

def test_support_graph(test_profile_with_ties):
    sg = test_profile_with_ties.support_graph()
    assert isinstance(sg, SupportGraph)
    assert test_profile_with_ties.candidates == sg.candidates
    assert test_profile_with_ties.margin(0, 1) == sg.margin(0, 1)
    assert test_profile_with_ties.margin(0, 2) == sg.margin(0, 2)
    assert test_profile_with_ties.margin(1, 2) == sg.margin(1, 2)

def test_majority_graph(test_profile_with_ties):
    mg = test_profile_with_ties.majority_graph()
    assert isinstance(mg, MajorityGraph)
    assert test_profile_with_ties.candidates == mg.candidates
    assert test_profile_with_ties.majority_prefers(0, 1) == mg.majority_prefers(0, 1)
    assert test_profile_with_ties.majority_prefers(0, 2) ==  mg.majority_prefers(0, 2)
    assert test_profile_with_ties.majority_prefers(1, 2) ==  mg.majority_prefers(1, 2)

def test_cycles(test_profile_with_ties, condorcet_cycle_profile_with_ties):
    assert test_profile_with_ties.cycles() == []
    assert condorcet_cycle_profile_with_ties.cycles() == [[0, 1, 2]]

def test_is_uniquely_weighted(condorcet_cycle_profile_with_ties): 

    assert not condorcet_cycle_profile_with_ties.is_uniquely_weighted()
    prof = ProfileWithTies([{0:1, 1:2}])
    assert prof.is_uniquely_weighted()
    prof = ProfileWithTies([{0:1, 1:2}], candidates=[0, 1, 2])
    assert not prof.is_uniquely_weighted()
    prof = ProfileWithTies([{0:1, 1:2}, {0:2, 1:1}])
    assert not prof.is_uniquely_weighted()

def test_remove_candidates(test_profile_with_ties):
    updated_prof = ProfileWithTies([
            {0:1},
            {2:2, 0:3},
            {2:1, 0:1}
        ], 
        [2, 3, 1])
    new_prof = test_profile_with_ties.remove_candidates([1])
    assert new_prof == updated_prof    
    assert new_prof.candidates == [0, 2]    

def test_report(capsys, test_profile_with_ties):
    test_profile_with_ties.report()
    captured = capsys.readouterr()
    assert "There are 3 candidates and 6 rankings:" in captured.out  
    assert "The number of empty rankings: 0" in captured.out
    assert "The number of rankings with ties: 1" in captured.out
    assert "The number of linear orders: 3" in captured.out
    assert "The number of truncated linear orders: 2" in captured.out

def test_display_rankings(test_profile_with_ties, capsys):

    test_profile_with_ties.display_rankings()
    captured = capsys.readouterr()
    assert "0 1 : 2" in captured.out
    assert "1 2 0 : 3" in captured.out
    assert "( 2  0 ) : 1" in captured.out

def test_anonymize(): 
    prof = ProfileWithTies([
        {0:1, 1:2},
        {0:1, 1:2},
        {0:1, 1:2},
        {0:1, 1:2},
    ])
    prof.rcounts == [1, 1, 1, 1]
    prof == prof.anonymize()
    prof.anonymize().rcounts == [4]

def test_description(condorcet_cycle_profile_with_ties): 
    print(condorcet_cycle_profile_with_ties.description())
    assert condorcet_cycle_profile_with_ties.description() == r'ProfileWithTies([{0: 1, 1: 2, 2: 3}, {0: 2, 1: 3, 2: 1}, {0: 3, 1: 1, 2: 2}], rcounts=[1, 1, 1], cmap={0: 0, 1: 1, 2: 2})'

def test_display(capsys, test_profile_with_ties):
    test_profile_with_ties.display()
    captured = capsys.readouterr()
    assert "| 2 | 3 |  1  |" in captured.out
    assert "| 0 | 1 | 2 0 |" in captured.out
    assert "| 1 | 2 |     |" in captured.out
    assert "|   | 0 |     |" in captured.out


def test_display_margin_graph(test_profile_with_ties):
    # just test that the function runs without error
    test_profile_with_ties.display_margin_graph()

def test_to_preflib_instance(test_profile_with_ties):
    inst = test_profile_with_ties.to_preflib_instance()
    assert isinstance(inst, OrdinalInstance)
    assert inst.num_voters == 6
    assert inst.num_alternatives == 3
    assert inst.full_profile() == [((0,), (1,)), ((0,), (1,)), ((1,), (2,), (0,)), ((1,), (2,), (0,)), ((1,), (2,), (0,)), ((2, 0),)]
    
def test_from_prelfib(test_profile_with_ties):
    inst = test_profile_with_ties.to_preflib_instance()
    prof = ProfileWithTies.from_preflib(inst)
    prof == test_profile_with_ties

    inst.write("./t.toi")
    prof = ProfileWithTies.from_preflib("./t.toi")
    prof == test_profile_with_ties

def test_write_read(test_profile_with_ties):
    test_profile_with_ties.write("./test", file_format='preflib')
    prof = ProfileWithTies.from_preflib("./test.toi")
    prof == test_profile_with_ties

    test_profile_with_ties.write("./test", file_format='csv')
    prof = ProfileWithTies.from_csv("./test.csv", is_truncated_linear=True)
    prof == test_profile_with_ties

def test_write_read2():
    prof = ProfileWithTies([{0:1, 1:1}], candidates=[0, 1, 2])
    prof.write("./test", file_format='preflib')
    prof2 = ProfileWithTies.from_preflib("./test.toi")
    prof == prof2

    prof = ProfileWithTies([{0:1, 1:1}], candidates=[0, 1, 2])
    prof.write("test", file_format='csv')
    prof = ProfileWithTies.from_csv("./test.csv", is_truncated_linear=False)
    prof == prof2

def test_eq():
    prof1 = ProfileWithTies([{0:1, 1:2}, {1:1, 2:2}, {0:1, 1:1}])
    prof2 = ProfileWithTies([{1:1, 2:2}, {0:1, 1:1}, {0:1, 1:2}])
    prof3 = ProfileWithTies([{1:1, 2:2}, {0:1, 1:2}, {0:1, 1:2}])
    assert prof1 == prof2
    assert not prof1 == prof3

def test_add():
    prof1 = ProfileWithTies([{0:1, 1:2}])    
    prof2 = ProfileWithTies([{0:1, 1:2}])
    prof3 = ProfileWithTies([{0:1, 1:2}], rcounts=[2])
    assert prof1 + prof2 == prof3
    prof1 = ProfileWithTies([{0:1, 1:2}])    
    prof2 = ProfileWithTies([{0:1, 2:2}])
    prof3 = ProfileWithTies([{0:1, 1:2}, {0:1, 2:2}])

