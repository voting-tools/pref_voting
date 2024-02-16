from pref_voting.profiles import Profile
from pref_voting.weighted_majority_graphs import MarginGraph, MajorityGraph, SupportGraph
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
    
    # def is_uniquely_weighted(self): 
    #     """Returns True if the profile is uniquely weighted. 
        
    #     A profile is **uniquely weighted** when there are no 0 margins and all the margins between any two candidates are unique.     
    #     """
        
    #     return MarginGraph.from_profile(self).is_uniquely_weighted()
    
    # def remove_candidates(self, cands_to_ignore):
    #     """Remove all candidates from ``cands_to_ignore`` from the profile. 

    #     :param cands_to_ignore: list of candidates to remove from the profile
    #     :type cands_to_ignore: list[int]
    #     :returns: a profile with candidates from ``cands_to_ignore`` removed and a dictionary mapping the candidates from the new profile to the original candidate names. 

    #     .. warning:: Since the candidates in a Profile must be named :math:`0, 1, \ldots, n-1` (where :math:`n` is the number of candidates), you must use the candidate map returned to by the function to recover the original candidate names. 

    #     :Example: 

    #     .. exec_code::

    #         from pref_voting.profiles import Profile 
    #         prof = Profile([[0,1,2], [1,2,0], [2,0,1]])
    #         prof.display()
    #         new_prof, orig_cnames = prof.remove_candidates([1])
    #         new_prof.display() # displaying new candidates names
    #         new_prof.display(cmap=orig_cnames) # use the original candidate names
    #     """        
    #     updated_rankings = _find_updated_profile(self._rankings, np.array(cands_to_ignore), self.num_cands)
    #     new_names = {c:cidx  for cidx, c in enumerate(sorted(updated_rankings[0]))}
    #     orig_names = {v:k  for k,v in new_names.items()}
    #     return Profile([[new_names[c] for c in r] for r in updated_rankings], rcounts=self._rcounts, cmap=self.cmap), orig_names
    
    # def anonymize(self): 
    #     """
    #     Return a profile which is the anonymized version of this profile. 
    #     """

    #     rankings = list()
    #     rcounts = list()
    #     for r in self.rankings:
    #         found_it = False
    #         for _ridx, _r in enumerate(rankings): 
    #             if r == _r: 
    #                 rcounts[_ridx] += 1
    #                 found_it = True
    #                 break
    #         if not found_it: 
    #             rankings.append(r)
    #             rcounts.append(1)
    #     return Profile(rankings, rcounts=rcounts, cmap=self.cmap)
