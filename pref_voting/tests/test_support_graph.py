import pytest
import numpy as np
from pref_voting.weighted_majority_graphs import SupportGraph 
from pref_voting.profiles import Profile
import matplotlib
matplotlib.use('Agg')

@pytest.fixture
def example_graph():
    return SupportGraph(
        [0, 1, 2], 
        [(0, 1, (4, 3)), (1, 2, (5, 2)), (2, 0, (6, 1))]
    )

def test_init():
    sg = SupportGraph(
        [0, 1, 2], 
        [(0, 1, (4, 3)), (1, 2, (5, 2)), (2, 0, (6, 1))]
    )
    assert sg.candidates == [0, 1, 2]
    assert sg.s_matrix == [[0, 4, 1], [3, 0, 5], [6, 2, 0]]

def test_edges(example_graph):
    expected_edges = [(0, 1, (4, 3)), (1, 2, (5, 2)), (2, 0, (6, 1))]
    assert sorted(example_graph.edges) == sorted(expected_edges)

def test_margin(example_graph):
    assert example_graph.margin(0, 1) == 1
    assert example_graph.margin(1, 2) == 3
    assert example_graph.margin(2, 0) == 5

def test_support(example_graph):
    assert example_graph.support(0, 1) == 4
    assert example_graph.support(1, 0) == 3
    assert example_graph.support(0, 2) == 1
    assert example_graph.support(2, 0) == 6
    assert example_graph.support(1, 2) == 5
    assert example_graph.support(2, 1) == 2

def test_majority_prefers(example_graph):
    assert example_graph.majority_prefers(0, 1) is True
    assert example_graph.majority_prefers(1, 0) is False

def test_is_tied(example_graph):
    assert example_graph.is_tied(0, 1) is False

def test_strength_matrix(example_graph):
    strength_matrix, _ = example_graph.strength_matrix()
    expected_matrix = np.array([[0, 4, 1], [3, 0, 5], [6, 2, 0]])
    np.testing.assert_array_equal(strength_matrix, expected_matrix)

def test_remove_candidates(example_graph):
    new_graph = example_graph.remove_candidates([1])
    assert new_graph.candidates == [0, 2]
    assert sorted(new_graph.edges) == sorted([(2, 0, (6, 1))])

# def test_display(example_graph):
#     example_graph.display()
#     example_graph.display(cmap={0: 'a', 1: 'b', 2: 'c'})

def test_from_profile():
    prof = Profile([[0, 1, 2], [1, 2, 0], [2, 0, 1]])
    sg = SupportGraph.from_profile(prof)
    assert sg.candidates == [0, 1, 2]
    assert sg.support(0, 1) == prof.support(0, 1)
    assert sg.support(1, 0) == prof.support(1, 0)
    assert sg.support(0, 2) == prof.support(0, 2)
    assert sg.support(2, 0) == prof.support(2, 0)
    assert sg.support(1, 2) == prof.support(1, 2)
    assert sg.support(2, 1) == prof.support(2, 1)
