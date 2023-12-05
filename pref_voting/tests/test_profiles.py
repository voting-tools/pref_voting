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
    assert test_profile.ranking_types == [(0, 1, 2), (1, 2, 0), (2, 0, 1)]

#     @property 
#     def ranking_types(self): 
#         """
#         Returns a list of all the type of rankings in the profile as a list of tuples.
#         """
#         return list(set([tuple(r) for r in self._rankings]))

#     @property
#     def rankings(self): 
#         """
#         Return a list of all individual rankings in the profile.  The type is a list of tuples of integers. 
#         """
        
#         return [tuple(r) for ridx,r in enumerate(self._rankings) for n in range(self._rcounts[ridx])]

#     @property
#     def counts(self): 
#         """
#         Returns a list of the counts of  the rankings in the profile.   The type is a list of integers. 
#         """
        
#         return list(self._rcounts)
        
#     def support(self, c1, c2):
#         """The number of voters that rank :math:`c1` above :math:`c2`
        
#         :param c1: the first candidate
#         :type c1: int
#         :param c2: the second candidate
#         :type c2: int
#         :rtype: int

#         """

#         return self._tally[c1][c2]
    
#     def margin(self, c1, c2):
#         """The number of voters that rank :math:`c1` above :math:`c2` minus the number of voters that rank :math:`c2` above :math:`c1`.
        
#         :param c1: the first candidate
#         :type c1: int
#         :param c2: the second candidate
#         :type c2: int
#         :rtype: int

#         """
#         return _margin(self._tally, c1, c2)
        
#     def majority_prefers(self, c1, c2): 
#         """Returns true if more voters rank :math:`c1` over :math:`c2` than :math:`c2` over :math:`c1`; otherwise false. 
        
#         :param c1: the first candidate
#         :type c1: int
#         :param c2: the second candidate
#         :type c2: int
#         :rtype: bool

#         """

#         return _margin(self._tally, c1, c2) > 0

#     def is_tied(self, c1, c2):
#         """Returns True if ``c1`` tied with ``c2``.  That is, the same number of voters rank ``c1`` over ``c2`` as ``c2`` over ``c1``. 
#         """ 

#         return _margin(self._tally, c1, c2) == 0
