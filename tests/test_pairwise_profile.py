import pytest
import numpy as np

from pref_voting.pairwise_profiles import PairwiseProfile
from pref_voting.pairwise_profiles import PairwiseBallot
from pref_voting.weighted_majority_graphs import MarginGraph, MajorityGraph

@pytest.fixture
def sample_comparisons():
    comparisons1 = [({"A", "B"}, {"A"}), ({"A", "C"}, {"C"}), ({"B", "C"}, {"B"})]
    comparisons2 = [({"A", "B"}, {"B"}), ({"A", "C"}, {"A"}), ({"B", "C"}, {"C"})]
    return [PairwiseBallot(comparisons1), PairwiseBallot(comparisons2)]

@pytest.fixture
def sample_profile(sample_comparisons):
    return PairwiseProfile(sample_comparisons, rcounts=[3, 2])

def test_initialization(sample_profile):
    assert sample_profile.num_voters == 5
    assert len(sample_profile.candidates) == 3
    assert sample_profile.cand_to_cidx == {"A": 0, "B": 1, "C": 2}

def test_support(sample_profile):
    assert sample_profile.support("A", "B") == 3
    assert sample_profile.support("B", "A") == 2

def test_margin(sample_profile):
    assert sample_profile.margin("A", "B") == 1
    assert sample_profile.margin("B", "A") == -1

def test_majority_prefers(sample_profile):
    assert sample_profile.majority_prefers("A", "B")
    assert not sample_profile.majority_prefers("B", "A")

def test_is_tied(sample_profile,sample_comparisons):
    other_prof =  PairwiseProfile(sample_comparisons + [[({"B", "C"}, {"C"})]], rcounts=[3, 2, 1])

    assert not sample_profile.is_tied("A", "B")
    assert not sample_profile.is_tied("B", "C")
    assert other_prof.is_tied("B", "C")

def test_dominators(sample_profile):
    assert sample_profile.dominators("A") == ["C"]

def test_dominates(sample_profile):
    assert sample_profile.dominates("A") == ["B"]

def test_copeland_scores(sample_profile):
    scores = sample_profile.copeland_scores()
    assert scores["A"] == 0.0
    assert scores["B"] == 0.0
    assert scores["C"] == 0.0

def test_condorcet_winner(sample_profile):
    assert sample_profile.condorcet_winner() is None

def test_weak_condorcet_winner(sample_profile):
    assert sample_profile.weak_condorcet_winner() == []

def test_condorcet_loser(sample_profile):
    assert sample_profile.condorcet_loser() is None

def test_strict_maj_size(sample_profile):
    assert sample_profile.strict_maj_size() == 3

def test_margin_graph(sample_profile):
    margin_graph = sample_profile.margin_graph()
    assert isinstance(margin_graph, MarginGraph)
    assert all([margin_graph.margin(c1, c2) == sample_profile.margin(c1, c2) for c1 in sample_profile.candidates for c2 in sample_profile.candidates])

def test_majority_graph(sample_profile):
    majority_graph = sample_profile.majority_graph()
    assert isinstance(majority_graph, MajorityGraph)

def test_display(sample_profile, capsys):
    sample_profile.display()
    captured = capsys.readouterr()
    expected_output = "3: {A, B} -> {A}, {A, C} -> {C}, {B, C} -> {B}\n2: {A, B} -> {B}, {A, C} -> {A}, {B, C} -> {C}\n"
    assert captured.out == expected_output

def test_add(sample_comparisons):
    profile1 = PairwiseProfile([sample_comparisons[0]], rcounts=[3])
    profile2 = PairwiseProfile([sample_comparisons[1]], rcounts=[2])
    combined_profile = profile1 + profile2
    assert combined_profile.num_voters == 5
    assert combined_profile.support("A", "B") == 3
    assert combined_profile.support("B", "A") == 2
