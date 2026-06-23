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
    # equal support both ways -> tied (no majority edge), margin 0
    tied = SupportGraph([0, 1, 2], [(0, 1, (4, 4))])
    assert tied.is_tied(0, 1) is True
    assert tied.margin(0, 1) == 0

def test_strength_matrix(example_graph):
    strength_matrix, _ = example_graph.strength_matrix()
    expected_matrix = np.array([[0, 4, 1], [3, 0, 5], [6, 2, 0]])
    np.testing.assert_array_equal(strength_matrix, expected_matrix)

def test_strength_matrix_curr_cands(example_graph):
    # restricting to a subset returns the corresponding support submatrix
    s_matrix, cand_to_idx = example_graph.strength_matrix(curr_cands=[0, 2])
    np.testing.assert_array_equal(s_matrix, np.array([[0, 1], [6, 0]]))
    assert cand_to_idx(0) == 0
    assert cand_to_idx(2) == 1

def test_remove_candidates(example_graph):
    new_graph = example_graph.remove_candidates([1])
    assert new_graph.candidates == [0, 2]
    assert sorted(new_graph.edges) == sorted([(2, 0, (6, 1))])

def test_display(example_graph):
    example_graph.display()
    example_graph.display(cmap={0: 'a', 1: 'b', 2: 'c'})


# ---------------------------------------------------------------------------
#  cycles / has_cycle with curr_cands (Bug 2.1 regression, SupportGraph variant)
# ---------------------------------------------------------------------------

def test_cycles_and_has_cycle_with_curr_cands(example_graph):
    # the fixture is the 0 -> 1 -> 2 -> 0 cycle (margins 1, 3, 5)
    assert example_graph.cycles() == [[0, 1, 2]]
    assert len(example_graph.cycles(curr_cands=[0, 1, 2])) == 1
    assert example_graph.cycles(curr_cands=[0, 1]) == []
    assert example_graph.has_cycle(curr_cands=[0, 1, 2]) is True
    assert example_graph.has_cycle(curr_cands=[0, 1]) is False


# ---------------------------------------------------------------------------
#  Inherited Condorcet / Copeland query API exercised through a SupportGraph
#  (SupportGraph overrides margin/support/majority_prefers/is_tied, so these
#   must be verified directly on a support graph.)
# ---------------------------------------------------------------------------

@pytest.fixture
def condorcet_support_graph():
    # 0 beats 1 and 2; 1 beats 2  ->  CW = 0, CL = 2
    return SupportGraph([0, 1, 2], [(0, 1, (6, 1)), (0, 2, (6, 1)), (1, 2, (6, 1))])

def test_condorcet_winner_and_loser_none_in_cycle(example_graph):
    # the cycle fixture has no Condorcet winner / loser / weak winner
    assert example_graph.condorcet_winner() is None
    assert example_graph.condorcet_loser() is None
    assert example_graph.weak_condorcet_winner() is None

def test_condorcet_winner_and_loser(condorcet_support_graph):
    assert condorcet_support_graph.condorcet_winner() == 0
    assert condorcet_support_graph.condorcet_loser() == 2
    assert condorcet_support_graph.weak_condorcet_winner() == [0]

def test_copeland_scores(condorcet_support_graph):
    assert condorcet_support_graph.copeland_scores() == {0: 2.0, 1: 0.0, 2: -2.0}
    assert condorcet_support_graph.copeland_scores(curr_cands=[1, 2]) == {1: 1.0, 2: -1.0}

def test_dominators_and_dominates(condorcet_support_graph):
    assert sorted(condorcet_support_graph.dominators(2)) == [0, 1]
    assert sorted(condorcet_support_graph.dominates(0)) == [1, 2]

def test_is_tournament_property(condorcet_support_graph):
    assert condorcet_support_graph.is_tournament is True
    assert SupportGraph([0, 1, 2], [(0, 1, (4, 4))]).is_tournament is False

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
