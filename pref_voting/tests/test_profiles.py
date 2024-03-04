from pref_voting.profiles import Profile
from pref_voting.profiles_with_ties import ProfileWithTies
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
def test_profile():
    return Profile([[0, 1, 2], [1, 2, 0], [2, 0, 1]], [2, 3, 1])

def test_create_profile():
    prof = Profile([[0, 1, 2], [1, 2, 0], [2, 0, 1]], [2, 3, 1])
    assert prof.num_cands == 3
    assert prof.candidates == [0, 1, 2]
    assert prof.num_voters == 6
    assert prof.cindices == [0, 1, 2]

def test_rankings_counts(test_profile):
    rankings, counts=test_profile.rankings_counts
    expected_rankings = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]])
    expected_rcounts = np.array([2, 3, 1])
    np.testing.assert_array_equal(rankings, expected_rankings)
    np.testing.assert_array_equal(counts, expected_rcounts)

def test_ranking_types1(test_profile):
    count_ranking_types1 = Counter(test_profile.ranking_types)
    count_ranking_types2 = Counter([(0, 1, 2), (1, 2, 0), (2, 0, 1)])
    assert count_ranking_types1 == count_ranking_types2

def test_ranking_types2():
    prof = Profile([[0, 1, 2], [1, 2, 0], [2, 0, 1], [2, 0, 1]], rcounts=[2, 3, 1, 2])
    count_ranking_types1 = Counter(prof.ranking_types)
    count_ranking_types2 = Counter([(0, 1, 2), (1, 2, 0), (2, 0, 1)])
    assert count_ranking_types1 == count_ranking_types2

def test_rankings1(test_profile):
    count_ranking_types1 = Counter(test_profile.rankings)
    count_ranking_types2 = Counter([(0, 1, 2), (0, 1, 2),  (1, 2, 0),(1, 2, 0),(1, 2, 0), (2, 0, 1)])
    assert count_ranking_types1 == count_ranking_types2

def test_rankings2():
    prof = Profile([[0, 1, 2], [1, 2, 0], [2, 0, 1], [2, 0, 1]], rcounts=[2, 3, 1, 2])
    count_ranking_types1 = Counter(prof.rankings)
    count_ranking_types2 = Counter([(0, 1, 2), (0, 1, 2),  (1, 2, 0),(1, 2, 0),(1, 2, 0), (2, 0, 1), (2, 0, 1), (2, 0, 1)])
    assert count_ranking_types1 == count_ranking_types2

def test_counts(test_profile):
    assert test_profile.counts == [2, 3, 1]

def test_support(test_profile):
    assert test_profile.support(0, 1) == 3
    assert test_profile.support(1, 0) == 3
    assert test_profile.support(2, 0) == 4
    assert test_profile.support(0, 2) == 2
    assert test_profile.support(1, 2) == 5
    assert test_profile.support(2, 1) == 1

def test_margin(test_profile):
    assert test_profile.margin(0, 1) == 0
    assert test_profile.margin(1, 0) == 0
    assert test_profile.margin(2, 0) == 2
    assert test_profile.margin(0, 2) == -2
    assert test_profile.margin(1, 2) == 4
    assert test_profile.margin(2, 1) == -4

def test_majority_prefers(test_profile):
    assert not test_profile.majority_prefers(0, 1)
    assert not test_profile.majority_prefers(1, 0) 
    assert test_profile.majority_prefers(2, 0) 
    assert not test_profile.majority_prefers(0, 2) 
    assert test_profile.majority_prefers(1, 2) 
    assert not test_profile.majority_prefers(2, 1) 

def test_is_tied(test_profile):
    assert test_profile.is_tied(0, 1)
    assert test_profile.is_tied(1, 0) 
    assert not test_profile.is_tied(2, 0) 
    assert not test_profile.is_tied(0, 2) 
    assert not test_profile.is_tied(1, 2) 
    assert not test_profile.is_tied(2, 1) 

def test_strict_maj_size(test_profile): 
    assert test_profile.strict_maj_size() == 4

def test_margin_graph(test_profile): 
    mg = test_profile.margin_graph()
    assert isinstance(mg, MarginGraph)
    assert test_profile.candidates == mg.candidates
    assert test_profile.margin(0, 1) == mg.margin(0, 1)
    assert test_profile.margin(0, 2) == mg.margin(0, 2)
    assert test_profile.margin(1, 2) == mg.margin(1, 2)

def test_support_graph(test_profile): 
    sg = test_profile.support_graph()
    assert isinstance(sg, SupportGraph)
    assert test_profile.candidates == sg.candidates
    assert test_profile.margin(0, 1) == sg.margin(0, 1)
    assert test_profile.margin(0, 2) == sg.margin(0, 2)
    assert test_profile.margin(1, 2) == sg.margin(1, 2)

    assert test_profile.support(0, 1) == sg.support(0, 1)
    assert test_profile.support(1, 0) == sg.support(1, 0)
    assert test_profile.support(0, 2) == sg.support(0, 2)
    assert test_profile.support(2, 0) == sg.support(2, 0)
    assert test_profile.support(1, 2) == sg.support(1, 2)
    assert test_profile.support(2, 1) == sg.support(2, 1)

def test_majority_graph(test_profile): 
    mg = test_profile.majority_graph()
    assert isinstance(mg, MajorityGraph)
    assert test_profile.candidates == mg.candidates
    assert test_profile.majority_prefers(0, 1) == mg.majority_prefers(0, 1)
    assert test_profile.majority_prefers(0, 2) == mg.majority_prefers(0, 2)
    assert test_profile.majority_prefers(1, 2) == mg.majority_prefers(1, 2)

def test_margin_matrix(test_profile): 
    mm = test_profile.margin_matrix
    assert mm[0][0] == 0
    assert mm[1][1] == 0
    assert mm[2][2] == 0
    assert mm[0][1] == -mm[1][0]
    assert mm[0][2] == -mm[2][0]
    assert mm[2][1] == -mm[1][2]
    assert mm[0][1] == 0
    assert mm[2][0] == 2
    assert mm[1][2] == 4
    
def test_is_uniquely_weighted(test_profile):
    assert not test_profile.is_uniquely_weighted()

def test_remove_candidates(test_profile):

    updated_prof = Profile([[0,  1], [1, 0], [1, 0]], [2, 3, 1])
    new_prof, orig_cnames = test_profile.remove_candidates([1])
    assert new_prof == updated_prof    
    assert orig_cnames == {0: 0, 1: 2}

def test_anonymize(): 
    prof = Profile([[0, 1], [0, 1], [0, 1]])
    anon_prof = prof.anonymize()
    assert anon_prof == prof
    assert list([list(r) for r in anon_prof._rankings]) == [[0, 1]]
    assert anon_prof._rcounts == [3]

def test_to_profile_with_ties(test_profile): 
    prof_w_ties = test_profile.to_profile_with_ties()
    assert type(prof_w_ties) == ProfileWithTies
    assert test_profile.candidates == prof_w_ties.candidates
    assert test_profile.margin(0, 1) == prof_w_ties.margin(0, 1)
    assert test_profile.margin(0, 2) == prof_w_ties.margin(0, 2)
    assert test_profile.margin(1, 2) == prof_w_ties.margin(1, 2)

def test_to_latex(condorcet_cycle): 
    
    assert condorcet_cycle.to_latex() == '\\begin{tabular}{ccc}\n$1$ & $1$ & $1$\\\\\\hline \n$0$ & $1$ & $2$\\\\ \n$1$ & $2$ & $0$\\\\ \n$2$ & $0$ & $1$\n\\end{tabular}'

    assert condorcet_cycle.to_latex(cmap={0:"a", 1:"b", 2:"c"}) == '\\begin{tabular}{ccc}\n$1$ & $1$ & $1$\\\\\\hline \n$a$ & $b$ & $c$\\\\ \n$b$ & $c$ & $a$\\\\ \n$c$ & $a$ & $b$\n\\end{tabular}'

def test_display_margin_matrix(capsys, condorcet_cycle):
    condorcet_cycle.display_margin_matrix()

    # Capture the output
    captured = capsys.readouterr()

    assert"""+----+----+----+
|  0 |  1 | -1 |
+----+----+----+
| -1 |  0 |  1 |
+----+----+----+
|  1 | -1 |  0 |
+----+----+----+
""" in captured.out


def test_display_margin_matrix(capsys, condorcet_cycle):
    condorcet_cycle.display_margin_matrix()

    # Capture the output
    captured = capsys.readouterr()

    assert"""+----+----+----+
|  0 |  1 | -1 |
+----+----+----+
| -1 |  0 |  1 |
+----+----+----+
|  1 | -1 |  0 |
+----+----+----+
""" in captured.out

def test_display_margin_graph(condorcet_cycle):
    # just test that the function runs
    condorcet_cycle.display_margin_graph()

def test_display_margin_graph_with_defeat(condorcet_cycle):
    # just test that the function runs
    condorcet_cycle.display_margin_graph_with_defeat(split_cycle_defeat(condorcet_cycle))

def test_description(condorcet_cycle):

    assert condorcet_cycle.description() == "Profile([[0, 1, 2], [1, 2, 0], [2, 0, 1]], rcounts=[1, 1, 1], cmap={0: '0', 1: '1', 2: '2'})"

def test_display(capsys, condorcet_cycle):
    # just test that the function runs
    condorcet_cycle.display()
    captured = capsys.readouterr()
    assert"""+---+---+---+
| 1 | 1 | 1 |
+---+---+---+
| 0 | 1 | 2 |
| 1 | 2 | 0 |
| 2 | 0 | 1 |
+---+---+---+
""" in captured.out
    
def test_to_preflib_instance(condorcet_cycle):

    preflib_instance = condorcet_cycle.to_preflib_instance()
    assert isinstance(preflib_instance, OrdinalInstance)
    assert preflib_instance.num_voters == 3    
    assert preflib_instance.num_alternatives == 3    
    assert preflib_instance.full_profile() == [((0,), (1,), (2,)), ((1,), (2,), (0,)), ((2,), (0,), (1,))]

def test_from_preflib(condorcet_cycle):
    preflib_instance = condorcet_cycle.to_preflib_instance()
    prof = Profile.from_preflib(preflib_instance)
    assert prof == condorcet_cycle

    preflib_instance.write("./condorcet_cycle.soc")
    prof = Profile.from_preflib("./condorcet_cycle.soc")
    assert prof == condorcet_cycle

def test_write(condorcet_cycle):
    condorcet_cycle.write("./condorcet_cycle.soc")
    prof = Profile.from_preflib("./condorcet_cycle.soc")
    assert prof == condorcet_cycle

    condorcet_cycle.write("./condorcet_cycle.csv", file_format="csv")

def read_write(condorcet_cycle):

    condorcet_cycle.write("./condorcet_cycle.csv", file_format="csv")
    prof = Profile.from_csv("./condorcet_cycle.csv")
    assert condorcet_cycle == prof    

def test_add(condorcet_cycle):
    r1 = Profile([[0, 1, 2]])
    r2 = Profile([[1, 2, 0]])
    r3 = Profile([[2, 0, 1]])
    prof = r1 + r2 + r3
    assert prof == condorcet_cycle

def test_eq(condorcet_cycle):   
    prof = Profile([[1, 2, 0], [0, 1, 2], [2, 0, 1]], [1, 1, 1])
    assert prof == condorcet_cycle