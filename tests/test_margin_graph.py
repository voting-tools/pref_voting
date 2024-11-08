# test_margin_graph.py

import pytest
import numpy as np
import networkx as nx
from pref_voting.weighted_majority_graphs import MarginGraph
from pref_voting.profiles import Profile

@pytest.fixture
def simple_margin_graph():
    return MarginGraph([0, 1, 2], [(0, 1, 1), (1, 2, 3), (2, 0, 5)])

@pytest.fixture
def simple_margin_graph2():
    return MarginGraph(['a', 'b', 'c'], [('a', 'b', 1), ('b', 'c', 3), ('c', 'a', 5)])

def test_margin_graph_init(simple_margin_graph, simple_margin_graph2):
    assert len(simple_margin_graph.candidates) == 3
    assert simple_margin_graph.candidates == [0, 1, 2]

    assert len(simple_margin_graph2.candidates) == 3
    assert simple_margin_graph2.candidates == ['a', 'b', 'c']

def test_margin(simple_margin_graph):
    assert simple_margin_graph.margin(0, 1) == 1
    assert simple_margin_graph.margin(1, 2) == 3
    assert simple_margin_graph.margin(2, 0) == 5

def test_strength_matrix(simple_margin_graph, simple_margin_graph2):

    s_matrix, cand_to_idx = simple_margin_graph.strength_matrix()
    assert np.array_equal(s_matrix, np.array([[0, 1, -5], [-1, 0, 3], [5, -3, 0]]))
    assert cand_to_idx(0) == 0
    assert cand_to_idx(1) == 1
    assert cand_to_idx(2) == 2

    s_matrix2, cand_to_idx2 = simple_margin_graph2.strength_matrix()
    assert np.array_equal(s_matrix, s_matrix2)
    assert cand_to_idx2('a') == 0
    assert cand_to_idx2('b') == 1
    assert cand_to_idx2('c') == 2

    s_matrix3, cand_to_idx2 = simple_margin_graph2.strength_matrix(curr_cands=['a', 'c'])
    assert np.array_equal(s_matrix3, [[0, -5], [5, 0]])
    assert cand_to_idx2('a') == 0
    assert cand_to_idx2('c') == 1

def test_edges_property(simple_margin_graph):
    expected_edges = [(0, 1, 1), (1, 2, 3), (2, 0, 5)]
    assert all(edge in simple_margin_graph.edges for edge in expected_edges)

def test_remove_candidates(simple_margin_graph):
    new_graph = simple_margin_graph.remove_candidates([1])
    assert 1 not in new_graph.candidates
    assert (0, 1, 1) not in new_graph.edges
    assert (1, 2, 3) not in new_graph.edges

def test_majority_prefers(simple_margin_graph):
    assert simple_margin_graph.majority_prefers(0, 1) is True
    assert simple_margin_graph.majority_prefers(1, 0) is False
    assert simple_margin_graph.majority_prefers(1, 2) is True
    assert simple_margin_graph.majority_prefers(2, 1) is False
    assert simple_margin_graph.majority_prefers(0, 2) is False
    assert simple_margin_graph.majority_prefers(2, 0) is True

def test_is_tied(simple_margin_graph):
    assert simple_margin_graph.is_tied(0, 1) is False
    assert simple_margin_graph.is_tied(1, 0) is False
    assert simple_margin_graph.is_tied(1, 2) is False
    assert simple_margin_graph.is_tied(0, 0) is True
    assert simple_margin_graph.is_tied(1, 1) is True
    assert simple_margin_graph.is_tied(2, 2) is True
    mg = MarginGraph([0, 1, 2], [(0, 2, 1)])
    assert mg.is_tied(1, 0) is True
    assert mg.is_tied(0, 1) is True
    assert mg.is_tied(2, 0) is False
    assert mg.is_tied(0, 2) is False

def test_is_uniquely_weighted(simple_margin_graph):
    mg = MarginGraph([0, 1, 2], [(0, 2, 1)])
    assert mg.is_uniquely_weighted() is False
    assert simple_margin_graph.is_uniquely_weighted() is True

# def test_add(simple_margin_graph):
#     additional_edges = MarginGraph([0, 1, 2], [(0, 1, 2), (1, 2, 1), (2, 0, 2)])
#     combined_graph = simple_margin_graph.add(additional_edges)
#     # Verify new margins
#     # Add assertions as necessary

def test_to_networkx(simple_margin_graph):
    nx_graph = simple_margin_graph.to_networkx()
    assert len(nx_graph.nodes) == 3
    assert len(nx_graph.edges) == 3

def test_minimal_profile(simple_margin_graph):
    min_prof = simple_margin_graph.minimal_profile()
    assert isinstance(min_prof, Profile)
    assert len(min_prof.candidates) == 3
    assert min_prof.margin(0, 1) == simple_margin_graph.margin(0, 1)
    assert min_prof.margin(1, 0) == simple_margin_graph.margin(1, 0)
    assert min_prof.margin(0, 2) == simple_margin_graph.margin(0, 2)
    assert min_prof.margin(2, 0) == simple_margin_graph.margin(2, 0)
    assert min_prof.margin(1, 2) == simple_margin_graph.margin(1, 2)
    assert min_prof.margin(2, 1) == simple_margin_graph.margin(2, 1)

def test_normalize_ordered_weights():
    mg = MarginGraph([0, 1, 2], [(0, 1, 5), (1, 2, 7), (2, 0, 11)])
    normalized_graph = mg.normalize_ordered_weights()
    assert isinstance(normalized_graph, MarginGraph)
    assert normalized_graph.candidates == mg.candidates
    assert normalized_graph.margin(0, 1) == 2
    assert normalized_graph.margin(1, 0) == -2
    assert normalized_graph.margin(1, 2) == 4
    assert normalized_graph.margin(2, 1) == -4
    assert normalized_graph.margin(0, 2) == -6
    assert normalized_graph.margin(2, 0) == 6

    mg = MarginGraph([0, 1, 2], [(0, 1, 20), (2, 1, 10)])
    normalized_graph = mg.normalize_ordered_weights()
    assert isinstance(normalized_graph, MarginGraph)
    assert normalized_graph.candidates == mg.candidates
    assert normalized_graph.margin(0, 1) == 4
    assert normalized_graph.margin(1, 0) == -4
    assert normalized_graph.margin(1, 2) == -2
    assert normalized_graph.margin(2, 1) == 2
    assert normalized_graph.margin(0, 2) == 0
    assert normalized_graph.margin(2, 0) == 0

def test_description(simple_margin_graph):
    description = simple_margin_graph.description()
    assert isinstance(description, str)
    assert description == "MarginGraph([0, 1, 2], [(0, 1, 1), (1, 2, 3), (2, 0, 5)], cmap={0: '0', 1: '1', 2: '2'})"

def test_display(simple_margin_graph, capsys):
    simple_margin_graph.display()
    simple_margin_graph.display(cmap={0: 'a', 1: 'b', 2: 'c'})

def test_display_cycles(simple_margin_graph, capsys):
    simple_margin_graph.display_cycles()
    simple_margin_graph.display_cycles(cmap={0: 'a', 1: 'b', 2: 'c'})

def test_display_with_defeat(simple_margin_graph, capsys):
    defeat = nx.DiGraph()
    defeat.add_edge(0, 1)
    simple_margin_graph.display_with_defeat(defeat)
    simple_margin_graph.display_with_defeat(defeat, cmap={0: 'a', 1: 'b', 2: 'c'})
    simple_margin_graph.display_with_defeat(defeat, cmap={0: 'a', 1: 'b', 2: 'c'}, show_undefeated=False)

def test_from_profile():
    prof = Profile([[0, 1, 2], [1, 0, 2], [2, 0, 1]])
    mg = MarginGraph.from_profile(prof)
    assert isinstance(mg, MarginGraph)
    assert mg.candidates == [0, 1, 2]
    assert mg.margin(0, 1) == prof.margin(0, 1)
    assert mg.margin(1, 0) == prof.margin(1, 0)
    assert mg.margin(0, 2) == prof.margin(0, 2)
    assert mg.margin(2, 0) == prof.margin(2, 0)
    assert mg.margin(1, 2) == prof.margin(1, 2)
    assert mg.margin(2, 1) == prof.margin(2, 1)

def test_to_latex(simple_margin_graph):
    latex = simple_margin_graph.to_latex()
    assert isinstance(latex, str)
    mg = MarginGraph([0, 1, 2, 3], [(0, 1, 2), (1, 2, 4), (2, 0, 6), (0, 3, 2), (1, 3, 4), (2, 3, 6)])
    latex = mg.to_latex(cmap={0: 'a', 1: 'b', 2: 'c', 3: 'd'})
    assert isinstance(latex, str)

def test_add():

    mg1 = MarginGraph([0, 1, 2], [(0, 1, 1), (1, 2, 3), (2, 0, 5)])
    mg2 = MarginGraph([0, 1, 2], [(1, 0, 1), (2, 1, 1), (2, 0, 3)])
    mg3 = mg1 + mg2
    assert isinstance(mg3, MarginGraph)
    assert mg3.candidates == [0, 1, 2]
    assert mg3.margin(0, 1) == 0
    assert mg3.margin(1, 0) == 0
    assert mg3.margin(1, 2) == 2
    assert mg3.margin(2, 1) == -2
    assert mg3.margin(0, 2) == -8
    assert mg3.margin(2, 0) == 8

def test_eq(): 
    mg1 = MarginGraph([0, 1, 2], [(0, 1, 1), (1, 2, 3), (2, 0, 5)])
    mg2 = MarginGraph([0, 1, 2], [(2, 0, 5), (0, 1, 1), (1, 2, 3)])    
    mg3 = MarginGraph([0, 1, 2], [(0, 1, 2), (1, 2, 4), (2, 0, 6)])

    assert mg1 == mg1
    assert mg1 == mg2
    assert mg1 != mg3