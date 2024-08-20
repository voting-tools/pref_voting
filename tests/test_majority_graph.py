import pytest
from pref_voting.weighted_majority_graphs import MajorityGraph
from pref_voting.profiles import Profile
from networkx import DiGraph

@pytest.fixture
def condorcet_cycle():
    """Provides a MajorityGraph instance with a cycle for testing."""
    return MajorityGraph([0, 1, 2], [(0, 1), (1, 2), (2, 0)])

@pytest.fixture
def example_graph():
    """Provides a MajorityGraph instance without a cycle for testing."""
    return MajorityGraph([0, 1, 2], [(0, 1), (1, 2), (0, 2)])

@pytest.fixture
def example_graph2():
    """Provides another MajorityGraph instance for testing with a tie."""
    return MajorityGraph([0, 1, 2], [(0, 2), (1, 2)])

def test_constructor():
    """Test MajorityGraph constructor for correct graph initialization."""
    mg = MajorityGraph(['a', 'b', 'c'], [('a', 'b'), ('b', 'c'), ('c', 'a')])
    assert mg.candidates == ['a', 'b', 'c']
    assert mg.edges == [('a', 'b'), ('b', 'c'), ('c', 'a')]
    assert mg.num_cands == 3
    assert mg.cindices == [0, 1, 2]
    
    assert mg.maj_matrix == [[False, True, False], [False, False, True], [True, False, False]]
    
    assert mg.cand_to_cindex('a') == 0
    assert mg.cand_to_cindex('b') == 1
    assert mg.cand_to_cindex('c') == 2

    assert mg.cindex_to_cand(mg.cand_to_cindex('a')) == 'a'
    assert mg.cindex_to_cand(mg.cand_to_cindex('b')) == 'b'
    assert mg.cindex_to_cand(mg.cand_to_cindex('c')) == 'c'


def test_margin(condorcet_cycle):
    with pytest.raises(Exception) as excinfo:
        condorcet_cycle.margin(0, 1)
    assert "margin is not implemented for majority graphs" in str(excinfo.value)

def test_support(condorcet_cycle):
    """Test the support method."""
    with pytest.raises(Exception) as excinfo:
        condorcet_cycle.support(0, 1)
    assert "support is not implemented for majority graphs" in str(excinfo.value)

def test_ratio(condorcet_cycle):
    with pytest.raises(Exception) as excinfo:
        condorcet_cycle.ratio(0, 1)
    assert "ratio is not implemented for majority graphs" in str(excinfo.value)

def test_edges_property(condorcet_cycle, example_graph, example_graph2):
    """Test the edges property."""
    assert sorted(condorcet_cycle.edges) == sorted([(1, 2), (0, 1), (2,0)])
    assert sorted(example_graph.edges) == sorted([(1, 2), (0, 1), (0,2)])
    assert sorted(example_graph2.edges) == sorted([(0, 2), (1, 2)])

def test_is_tournament_property(condorcet_cycle, example_graph, example_graph2):
    """Test the is_tournament property."""
    assert condorcet_cycle.is_tournament
    assert example_graph.is_tournament
    assert not example_graph2.is_tournament

def test_majority_prefers(condorcet_cycle, example_graph, example_graph2):
    """Test the majority_prefers method."""
    assert condorcet_cycle.majority_prefers(0, 1)
    assert not condorcet_cycle.majority_prefers(1, 0)
    assert not condorcet_cycle.majority_prefers(0, 2)
    assert condorcet_cycle.majority_prefers(2, 0)
    assert not condorcet_cycle.majority_prefers(2, 1)
    assert condorcet_cycle.majority_prefers(1, 2)
    assert example_graph.majority_prefers(0, 1)
    assert not example_graph.majority_prefers(1, 0)
    assert not example_graph2.majority_prefers(0, 1)
    assert not example_graph2.majority_prefers(1, 0)

def test_majority_prefers2(condorcet_cycle):
    """Test the majority_prefers method comparing items that are not candidates."""
    # 3, 5, 7 are not candidates in condorcet_cycle
    assert not condorcet_cycle.majority_prefers(1, 3)
    assert not condorcet_cycle.majority_prefers(5, 7)

def test_is_tied(condorcet_cycle, example_graph, example_graph2):
    """Test the is_tied method."""
    assert not condorcet_cycle.is_tied(0, 1)
    assert not condorcet_cycle.is_tied(1, 0)
    assert not example_graph.is_tied(0, 1)
    assert not example_graph.is_tied(1, 0)
    assert example_graph2.is_tied(0, 1)
    assert example_graph2.is_tied(1, 0)

def test_copeland_scores(condorcet_cycle, example_graph, example_graph2):
    """Test the copeland_scores method."""
    assert condorcet_cycle.copeland_scores() == {0: 0.0, 1: 0.0, 2: 0.0}
    assert condorcet_cycle.copeland_scores(curr_cands=[0, 2]) == {0: -1.0, 2: 1.0}
    assert example_graph.copeland_scores() == {0: 2.0, 1: 0.0, 2: -2.0}
    assert example_graph2.copeland_scores() == {0: 1.0, 1: 1.0, 2: -2.0}
    assert example_graph2.copeland_scores(curr_cands=[0, 1]) == {0: 0.0, 1: 0.0}

def test_dominators(condorcet_cycle, example_graph, example_graph2):
    """Test the dominators method."""
    assert condorcet_cycle.dominators(0) == [2]
    assert condorcet_cycle.dominators(1) == [0]
    assert condorcet_cycle.dominators(2) == [1]
    
    assert example_graph.dominators(0) == []
    assert example_graph.dominators(1) == [0]
    assert example_graph.dominators(2) == [0, 1]

    assert example_graph2.dominators(0) == []
    assert example_graph2.dominators(1) == []
    assert example_graph2.dominators(2) == [0, 1]

def test_dominates(condorcet_cycle, example_graph, example_graph2):
    """Test the dominates method."""
    assert condorcet_cycle.dominates(0) == [1]
    assert condorcet_cycle.dominates(1) == [2]
    assert condorcet_cycle.dominates(2) == [0]
    
    assert example_graph.dominates(0) == [1, 2]
    assert example_graph.dominates(1) == [2]
    assert example_graph.dominates(2) == []

    assert example_graph2.dominates(0) == [2]
    assert example_graph2.dominates(1) == [2]
    assert example_graph2.dominates(2) == []

def test_condorcet_winner(condorcet_cycle, example_graph, example_graph2):
    """Test the condorcet_winner method."""
    assert condorcet_cycle.condorcet_winner() == None
    assert condorcet_cycle.condorcet_winner(curr_cands=[1,2]) == 1
    assert condorcet_cycle.condorcet_winner(curr_cands=[0]) == 0
    
    assert example_graph.condorcet_winner() == 0
    assert example_graph.condorcet_winner(curr_cands=[1,2]) == 1
    assert example_graph.condorcet_winner(curr_cands=[0]) == 0

    assert example_graph2.condorcet_winner() == None
    assert example_graph2.condorcet_winner(curr_cands=[1,2]) == 1
    assert example_graph2.condorcet_winner(curr_cands=[0,1]) == None
    assert example_graph2.condorcet_winner(curr_cands=[0]) == 0

def test_weak_condorcet_winner(condorcet_cycle, example_graph, example_graph2):
    """Test the weak_condorcet_winner method."""
    assert condorcet_cycle.weak_condorcet_winner() == None
    assert condorcet_cycle.weak_condorcet_winner(curr_cands=[1,2]) == [1]
    assert condorcet_cycle.weak_condorcet_winner(curr_cands=[0]) == [0]
    
    assert example_graph.weak_condorcet_winner() == [0]
    assert example_graph.weak_condorcet_winner(curr_cands=[1,2]) == [1]
    assert example_graph.weak_condorcet_winner(curr_cands=[0]) == [0]

    assert example_graph2.weak_condorcet_winner() == [0, 1]
    assert example_graph2.weak_condorcet_winner(curr_cands=[1,2]) == [1]
    assert example_graph2.weak_condorcet_winner(curr_cands=[0,1]) == [0, 1]
    assert example_graph2.weak_condorcet_winner(curr_cands=[0]) == [0]

def test_condorcet_loser(condorcet_cycle, example_graph, example_graph2):
    """Test the condorcet_loser method."""
    assert condorcet_cycle.condorcet_loser() == None
    assert condorcet_cycle.condorcet_loser(curr_cands=[1,2]) == 2
    assert condorcet_cycle.condorcet_loser(curr_cands=[0]) == 0
    
    assert example_graph.condorcet_loser() == 2
    assert example_graph.condorcet_loser(curr_cands=[1,2]) == 2
    assert example_graph.condorcet_loser(curr_cands=[0]) == 0

    assert example_graph2.condorcet_loser() == 2
    assert example_graph2.condorcet_loser(curr_cands=[1,2]) == 2
    assert example_graph2.condorcet_loser(curr_cands=[0,1]) == None
    assert example_graph2.condorcet_loser(curr_cands=[0]) == 0


def test_cycles(condorcet_cycle, example_graph, example_graph2):
    """Test the cycles method."""
    assert condorcet_cycle.cycles() == [[0, 1, 2]]
    assert condorcet_cycle.cycles(curr_cands=[1,2]) == []
    assert condorcet_cycle.cycles(curr_cands=[0]) == []
    
    assert example_graph.cycles() == []
    assert example_graph.cycles(curr_cands=[1,2]) == []
    assert example_graph.cycles(curr_cands=[0]) == []

    assert example_graph2.cycles() == []
    assert example_graph2.cycles(curr_cands=[1,2]) == []
    assert example_graph2.cycles(curr_cands=[0,1]) == []
    assert example_graph2.cycles(curr_cands=[0]) == []

def test_has_cycle(condorcet_cycle, example_graph, example_graph2):
    """Test the has_cycle method."""
    assert condorcet_cycle.has_cycle() 
    assert not condorcet_cycle.has_cycle(curr_cands=[1,2]) 
    assert not condorcet_cycle.has_cycle(curr_cands=[0]) 
    
    assert not example_graph.has_cycle() 
    assert not example_graph.has_cycle(curr_cands=[1,2]) 
    assert not example_graph.has_cycle(curr_cands=[0]) 

    assert not example_graph2.has_cycle() 
    assert not example_graph2.has_cycle(curr_cands=[1,2]) 
    assert not example_graph2.has_cycle(curr_cands=[0,1]) 
    assert not example_graph2.has_cycle(curr_cands=[0]) 

def test_remove_candidates(condorcet_cycle):
    """Test the remove_candidates method."""
    mg = condorcet_cycle.remove_candidates([1])
    sorted(mg.edges) == sorted([[2, 0]])
    mg = condorcet_cycle.remove_candidates([0])
    sorted(mg.edges) == sorted([(1, 2)])
    mg = condorcet_cycle.remove_candidates([0, 2])
    sorted(mg.edges) == sorted([])

def test_to_networkx(condorcet_cycle, example_graph, example_graph2):
    """Test the to_networkx method."""
    g = condorcet_cycle.to_networkx()
    assert type(g) == DiGraph
    assert sorted(g.edges) == sorted(condorcet_cycle.edges)
    g = example_graph.to_networkx()
    assert type(g) == DiGraph
    assert sorted(g.edges) == sorted(example_graph.edges)
    g = example_graph2.to_networkx()
    assert type(g) == DiGraph
    assert sorted(g.edges) == sorted(example_graph2.edges)

def test_description(condorcet_cycle, example_graph, example_graph2):
    """Test the description method."""
    assert condorcet_cycle.description() == "MajorityGraph([0, 1, 2], [(0, 1), (1, 2), (2, 0)], cmap={0: '0', 1: '1', 2: '2'})"
    assert example_graph.description() == "MajorityGraph([0, 1, 2], [(0, 1), (0, 2), (1, 2)], cmap={0: '0', 1: '1', 2: '2'})"
    assert example_graph2.description() == "MajorityGraph([0, 1, 2], [(0, 2), (1, 2)], cmap={0: '0', 1: '1', 2: '2'})"

def test_display(condorcet_cycle):
    """Test the display methods."""
    # just test that the function runs
    condorcet_cycle.display()
    condorcet_cycle.display(curr_cands=[1,2])

def test_display_cycles(condorcet_cycle, example_graph, example_graph2):
    """Test the display_cycles method."""
    condorcet_cycle.display_cycles()

def test_to_latex(condorcet_cycle, example_graph, example_graph2):
    """Test the to_latex method."""
    latex_code = condorcet_cycle.to_latex()
    print(latex_code)
    assert "\\begin{tikzpicture}" in latex_code
    assert "\\node[circle,draw,minimum width=0.25in] at (0,0) (a) {$0$};" in latex_code
    assert "\\end{tikzpicture}" in latex_code

def test_from_profile(condorcet_cycle):
    """Test the from_profile class method."""
    prof = Profile([[0, 1, 2], [1, 2, 0], [2, 0, 1]])
    mg = MajorityGraph.from_profile(prof)
    assert mg == condorcet_cycle   

def test_add(condorcet_cycle):
    mg1 = MajorityGraph([0, 1], [(0, 1)])
    mg2 = MajorityGraph([0, 2], [(2, 0)])
    mg3 = MajorityGraph([1, 2], [(1, 2)])

    mg = mg1 + mg2 + mg3
    assert mg.candidates == [0, 1, 2]
    assert mg == condorcet_cycle

    mg1 = MajorityGraph([0, 1], [(0, 1)])
    mg2 = MajorityGraph([0, 1, 2], [(1,0), (1, 2)])

    mg3 = MajorityGraph([0, 1, 2], [(1, 2)])
    mg = mg1 + mg2
    print(mg.edges)
    print(mg.candidates)
    assert mg.candidates == [0, 1, 2]
    assert mg == mg3

def test_eq(condorcet_cycle):
    mg = MajorityGraph([0, 1, 2], [(2, 0), (1, 2), (0, 1)])
    assert mg == condorcet_cycle