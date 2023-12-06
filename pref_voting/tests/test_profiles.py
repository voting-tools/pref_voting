from pref_voting.profiles import Profile
import numpy as np
import pytest

@pytest.fixture
def test_profile():
    return Profile([[0, 1, 2], [1, 2, 0], [2, 0, 1]], [2, 3, 1])

def test_crete_profile():
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

def test_ranking_types(test_profile):
    print(test_profile.ranking_types)
    #assert test_profile.ranking_types == [(0, 1, 2), (1, 2, 0), (2, 0, 1)]

