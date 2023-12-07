from pref_voting.profiles import Profile
import numpy as np
import pytest
from collections import Counter

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
