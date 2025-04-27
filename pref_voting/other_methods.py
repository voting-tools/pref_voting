'''
    File: other_methods.py
    Author: Wes Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)
    Date: January 12, 2022
    Updated: April 21, 2025

'''
from pref_voting.voting_method import *
from pref_voting.scoring_methods import plurality
from pref_voting.profiles import _find_updated_profile, _num_rank
from pref_voting.profiles_with_ties import ProfileWithTies
from pref_voting.weighted_majority_graphs import MarginGraph
from itertools import combinations, permutations
from pref_voting.voting_method_properties import ElectionTypes
from pref_voting.rankings import Ranking
from pref_voting.social_welfare_function import swf
import numpy as np
from pref_voting.profiles_with_ties import _num_rank_profile_with_ties
import copy
from ortools.linear_solver import pywraplp

@vm(name = "Absolute Majority",
    skip_registration=True, # skip registration since aboslute majority may return an empty list
    input_types = [ElectionTypes.PROFILE])
def absolute_majority(profile, curr_cands = None):
    """The absolute majority winner is the candidate with a strict majority  of first place votes.  Returns an empty list if there is no candidate with a strict majority of first place votes. Otherwise returns the absolute majority winner in the ``profile`` restricted to ``curr_cands``.

    ..note:
        The term 'absolute majority' for this voting method comes from Charles Dodgson's famous pamplet of 1873, "A Discussion of the Various Methods of Procedure in Conducting Elections" (see I. McLean and A. Urken, *Classics of Social Choice*, 1995, p. 281, or A. D. Taylor, "Social Choice and the Mathematics of Manipulation," 2005, p. 11).

    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A sorted list of candidates

    .. important:: 
        Formally, this is *not* a voting method since the function might return an empty list (when there is no candidate with a strict majority of first place votes).  Also, if there is an absolute majority winner, then that winner is unique. 

    :Example:
        .. exec_code::

            from pref_voting.profiles import Profile
            from pref_voting.other_methods import absolute_majority
            
            prof1 = Profile([[0, 1, 2], [1, 0, 2], [2, 1, 0]], [3, 1, 2])
            prof1.display()
            absolute_majority.display(prof1)

            prof2 = Profile([[0, 1, 2], [1, 0, 2], [1, 2, 0]], [5, 1, 2])
            prof2.display()
            absolute_majority.display(prof2)

    """
    maj_size = profile.strict_maj_size()
    curr_cands = profile.candidates if curr_cands is None else curr_cands

    plurality_scores = profile.plurality_scores(curr_cands = curr_cands)
    abs_maj_winner = [c for c in curr_cands if plurality_scores[c] >= maj_size]

    return sorted(abs_maj_winner)

@vm(name = "Pareto",
    input_types = [ElectionTypes.PROFILE, ElectionTypes.PROFILE_WITH_TIES])
def pareto(profile, curr_cands = None, strong_Pareto = False, use_extended_strict_preferences = True):
    """Returns the set of candidates who are not Pareto dominated.

    For ProfilesWithTies, if strong_Pareto == True, then a dominates b if some voter strictly prefers a to b and no voter strictly prefers b to a.

    Args:
        prof (Profile, ProfileWithTies): An anonymous profile of linear (or strict weak) orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A sorted list of candidates

    """

    if type(profile) == ProfileWithTies:
        currently_using_extended_strict_preferences = profile.using_extended_strict_preference
        if use_extended_strict_preferences:
            profile.use_extended_strict_preference()
        
    Pareto_dominated = set()
    candidates = profile.candidates if curr_cands is None else curr_cands
    for a in candidates:
        for b in candidates:
            if not strong_Pareto and profile.support(a,b) == profile.num_voters:
                Pareto_dominated.add(b)

            if strong_Pareto and profile.support(a,b) > 0 and profile.support(b,a) == 0:
                Pareto_dominated.add(b)     

    if type(profile) == ProfileWithTies and use_extended_strict_preferences:
        if not currently_using_extended_strict_preferences:
            profile.use_strict_preference()

    return sorted(list(set(candidates) - Pareto_dominated))
    

## Kemeny-Young Method 
#
def kendalltau_dist(rank_a, rank_b):
    index_b = {c: i for i, c in enumerate(rank_b)}
    tau = 0
    for i, j in combinations(rank_a, 2):
        # by definition of itertools.combinations, index_a[i] < index_a[j]
        if index_b[i] > index_b[j]:
            tau += 1
    return tau

def kendalltau_dist_for_rankings_with_ties(
        candidates, 
        ranking1, 
        ranking2,
        penalty=0.5):

    tau = 0
    for c1, c2 in combinations(candidates, 2):
        # by definition of itertools.combinations, index_a[i] < index_a[j]
        
        if (ranking1.extended_strict_pref(c1, c2) and ranking2.extended_strict_pref(c2, c1)) or (ranking1.extended_strict_pref(c2, c1) and ranking2.extended_strict_pref(c1, c2)):
            tau += 1
        elif (ranking1.extended_strict_pref(c1, c2) and ranking2.extended_indiff(c1, c2)) or (ranking1.extended_strict_pref(c2, c1) and ranking2.extended_indiff(c1, c2)):
            tau += penalty
        elif (ranking1.extended_indiff(c1, c2) and ranking2.extended_strict_pref(c1, c2)) or (ranking1.extended_indiff(c1, c2) and ranking2.extended_strict_pref(c2, c1)) :
            tau += penalty

    return tau


def _kemeny_young_rankings(rankings, rcounts, candidates): 
    
    rankings_dist = dict()
    for ranking in permutations(candidates): 
        rankings_dist[tuple(ranking)] = sum(c * kendalltau_dist(tuple(r), ranking) 
        for r,c in zip(rankings, rcounts))
    min_dist = min(rankings_dist.values())

    lin_orders = [r for r in rankings_dist.keys() if rankings_dist[r] == min_dist]
    
    return lin_orders, min_dist

def kemeny_young_rankings(profile, curr_cands = None): 
    """
    A Kemeny-Young ranking is a ranking that minimizes the sum of the Kendall tau distances to the voters' rankings.  
    
    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        rankings: A list of Kemeny-Young rankings.
        
        dist: The minimum distance of the Kemeny-Young rankings.


    :Example:
        .. exec_code::

            from pref_voting.profiles import Profile
            from pref_voting.other_methods import kemeny_young, kemeny_young_rankings
            
            prof1 = Profile([[0, 1, 2], [1, 0, 2], [2, 1, 0]], [3, 1, 2])
            prof1.display()
            kyrs, d = kemeny_young_rankings(prof1)
            print(f"Minimal distance: {d}")
            for kyr in kyrs: 
                print(f"ranking: {kyr}")

            prof2 = Profile([[0, 1, 2], [1, 0, 2], [1, 2, 0]], [5, 1, 2])
            prof2.display()
            kyrs, d = kemeny_young_rankings(prof2)
            print(f"Minimal distance: {d}")
            for kyr in kyrs: 
                print(f"ranking: {kyr}")

    """
    candidates = profile.candidates if curr_cands is None else curr_cands
    
    rankings = profile._rankings if curr_cands is None else _find_updated_profile(profile._rankings, np.array([c for c in profile.candidates if c not in curr_cands]),  profile.num_cands)
    return _kemeny_young_rankings(list(rankings), list(profile._rcounts), candidates)


@vm(name = "Kemeny-Young",
    input_types = [ElectionTypes.PROFILE,ElectionTypes.PROFILE_WITH_TIES,  ElectionTypes.MARGIN_GRAPH])
def kemeny_young(edata, curr_cands = None, algorithm = "marginal"): 
    """A Kemeny-Young ranking is a ranking that maximizes the sum of the margins of pairs of candidates in the ranking. Equivalently, a Kemeny-Young ranking is a ranking that minimizes the sum of the Kendall tau distances to the voters' rankings. The Kemeny-Young winners are the candidates that are ranked first by some Kemeny-Young ranking.

    Args:
        edata (Profile, ProfileWithTies, MarginGraph): Any election data that has a `margin` method.
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``
        algorithm (str, optional): The algorithm to use.  Options are "marginal" and "Kendall tau". If "marginal" is used, then the Kemeny-Young rankings are computed by finding the sum of the margins of each pair of candidates in the ranking.  If "Kendall tau" is used, then the Kemeny-Young rankings are computed by summing the Kendall tau distances to the voters' rankings.  Default is "marginal".

    Returns: 
        A sorted list of candidates

    :Example:

        .. exec_code::

            from pref_voting.profiles import Profile
            from pref_voting.other_methods import kemeny_young, kemeny_young_rankings
            
            prof1 = Profile([[0, 1, 2], [1, 0, 2], [2, 1, 0]], [3, 1, 2])
            prof1.display()
            kyrs, d = kemeny_young_rankings(prof1)
            print(f"Minimal distance: {d}")
            for kyr in kyrs: 
                print(f"ranking: {kyr}")
            kemeny_young.display(prof1)

            prof2 = Profile([[0, 1, 2], [1, 0, 2], [1, 2, 0]], [5, 1, 2])
            prof2.display()
            kyrs, d = kemeny_young_rankings(prof2)
            print(f"Minimal distance: {d}")
            for kyr in kyrs: 
                print(f"ranking: {kyr}")
            kemeny_young.display(prof2)

    """
    assert algorithm in ["marginal", "Kendall tau"], "Algorithm must be either 'marginal' or 'Kendall tau'."

    candidates = edata.candidates if curr_cands is None else curr_cands

    if isinstance(edata, MarginGraph) or isinstance(edata,ProfileWithTies):
        algorithm = "marginal"

    if algorithm == "Kendall tau":
        rankings = edata._rankings if curr_cands is None else _find_updated_profile(edata._rankings, np.array([c for c in edata.candidates if c not in curr_cands]), edata.num_cands)
        ky_rankings, min_dist = _kemeny_young_rankings(list(rankings), list(edata._rcounts), candidates)

    if algorithm == "marginal":

        best_ranking_score = 0
        ky_rankings = []

        for r in permutations(candidates):

            score_of_r = 0 
            for i in r[:-1]:
                for j in r[r.index(i)+1:]:
                    score_of_r += edata.margin(i,j)

            if score_of_r > best_ranking_score:
                best_ranking_score = score_of_r
                ky_rankings = [r]
            if score_of_r == best_ranking_score:
                ky_rankings.append(r)
    
    return sorted(list(set([r[0] for r in ky_rankings])))

@vm("Preliminary Weighted Condorcet",
    input_types = [ElectionTypes.PROFILE])
def preliminary_weighted_condorcet(prof, curr_cands = None, show_orders = False, require_positive_plurality_score = False):
    """The preliminary version of the Weighted Condorcet Rule in Tideman's book, Collective Decisions and Voting (p. 223). The winners are the candidates ranked first by some linear order of the candidates with highest score, where the score of an order (c_1,...,c_n) is the sum over all i<j of the margin of c_i vs. c_j multiplied by the plurality scores of c_i and c_j. 
    
    The multiplication by plurality scores is what distinguishes this method from the Kemeny-Young method.
    
    Tideman (p. 224) defines a more complicated Weighted Condorcet rule that is intended to be used when some candidates receive zero first-place votes.
    
    Args:
        prof (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``
        show_orders (bool): If True, then print the set of best orders.
        require_positive_plurality_score (bool): If True, then require that all candidates have a positive plurality score.

    Returns:
        A sorted list of candidates
    """

    cands = curr_cands if curr_cands is not None else prof.candidates

    if require_positive_plurality_score:
        assert all([prof.plurality_scores(curr_cands=curr_cands)[c] > 0 for c in cands]), "All candidates must have a positive plurality score."

    best_order_score = 0
    best_orders = []

    for r in permutations(cands):

        score_of_r = 0 
        for i in r[:-1]:
            for j in r[r.index(i)+1:]:
                score_of_r += (prof.plurality_scores(curr_cands=curr_cands)[i] * prof.plurality_scores(curr_cands=curr_cands)[j] * prof.margin(i,j))

        if score_of_r > best_order_score:
            best_order_score = score_of_r
            best_orders = [r]
        if score_of_r == best_order_score:
            best_orders.append(r)

    if show_orders == True:
        print(f"Best orders: {set(best_orders)}")

    winners = [r[0] for r in best_orders]

    return list(set(winners))

### Bucklin

@vm(name = "Bucklin",
    input_types = [ElectionTypes.PROFILE])
def bucklin(profile, curr_cands = None): 
    """If a candidate has a strict majority of first-place votes, then that candidate is the winner. If no such candidate exists, then check the candidates that are ranked first or second.  If a candidate has a strict majority of first- or second-place voters, then that candidate is the winner. If no such winner is found move on to the 3rd, 4th, etc. place votes.  Return the candidates with the greatest overall score.  
    
    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A sorted list of candidates

    :Example: 

    .. exec_code:: 

        from pref_voting.profiles import Profile
        from pref_voting.other_methods import bucklin

        prof = Profile([[1, 0, 2], [0, 2, 1], [0, 1, 2]], [2, 1, 1])

        prof.display()
        bucklin.display(prof)

    """
    strict_maj_size = profile.strict_maj_size()
    
    candidates = profile.candidates if curr_cands is None else curr_cands
    
    num_cands = candidates
    
    rankings = profile._rankings if curr_cands is None else _find_updated_profile(profile._rankings, np.array([c for c in profile.candidates if c not in curr_cands]), profile.num_cands)

    rcounts = profile._rcounts
    
    num_cands = len(candidates)
    ranks = range(1, num_cands + 1)
    
    cand_to_num_voters_rank = dict()
    for r in ranks:
        cand_to_num_voters_rank[r] = {c: _num_rank(rankings, rcounts, c, r)
                                      for c in candidates}
        cand_scores = {c:sum([cand_to_num_voters_rank[_r][c] for _r in cand_to_num_voters_rank.keys()]) 
                       for c in candidates}
        if any([s >= strict_maj_size for s in cand_scores.values()]):
            break
    max_score = max(cand_scores.values())
    return sorted([c for c in candidates if cand_scores[c] >= max_score])


def bucklin_with_explanation(profile, curr_cands = None): 
    """Return the Bucklin winners and the score for each candidate. 
 
    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 

        A sorted list of candidates

        A dictionary assigning the score for each candidate. 

    :Example: 

    .. exec_code:: 

        from pref_voting.profiles import Profile
        from pref_voting.other_methods import bucklin_with_explanation

        prof = Profile([[1, 0, 2], [0, 2, 1], [0, 1, 2]], [2, 1, 1])

        prof.display()
        sb_ws, scores = bucklin_with_explanation(prof)

        print(f"The winners are {sb_ws}")
        print(f"The candidate scores are {scores}")

    """
    strict_maj_size = profile.strict_maj_size()
    
    candidates = profile.candidates if curr_cands is None else curr_cands
    
    num_cands = candidates
    
    rankings = profile._rankings if curr_cands is None else _find_updated_profile(profile._rankings, np.array([c for c in profile.candidates if c not in curr_cands]), profile.num_cands)

    rcounts = profile._rcounts
    
    num_cands = len(candidates)
    ranks = range(1, num_cands + 1)
    
    cand_to_num_voters_rank = dict()
    for r in ranks:
        cand_to_num_voters_rank[r] = {c: _num_rank(rankings, rcounts, c, r)
                                      for c in candidates}
        cand_scores = {c:sum([cand_to_num_voters_rank[_r][c] for _r in cand_to_num_voters_rank.keys()]) 
                       for c in candidates}
        if any([s >= strict_maj_size for s in cand_scores.values()]):
            break
    max_score = max(cand_scores.values())
    return sorted([c for c in candidates if cand_scores[c] >= max_score]), cand_scores

@vm(name="Bucklin for Truncated Linear Orders", 
    input_types=[ElectionTypes.PROFILE_WITH_TIES])
def bucklin_for_truncated_linear_orders(profile, curr_cands=None):
    """The Bucklin voting method adapted for truncated linear orders using ProfileWithTies.
    
    Args:
        profile: A ProfileWithTies object that represents an election profile.
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in curr_cands

    Returns: 
        A sorted list of candidates that win according to the Bucklin method.
        
    .. note::
        This is an adaptation of the Bucklin method for truncated linear orders. Empty ballots are removed before calculating
        the majority threshold, similar to how instant_runoff_for_truncated_linear_orders handles truncated profiles.
    """
    assert all([not r.has_overvote() for r in profile.rankings]), "Bucklin is only defined when all the ballots are truncated linear orders."

    # Remove empty rankings and get working copy of profile
    working_profile = copy.deepcopy(profile)
    working_profile.remove_empty_rankings()
    
    strict_maj_size = working_profile.strict_maj_size()
    candidates = working_profile.candidates if curr_cands is None else curr_cands
    
    if curr_cands is not None:
        working_profile = working_profile.remove_candidates([c for c in working_profile.candidates if c not in curr_cands])
    rcounts = working_profile.rcounts
    
    num_cands = len(candidates)
    ranks = range(1, num_cands + 1)
    
    # Get rankings and their counts
    rankings, rcounts = working_profile.rankings_counts
    
    # Track scores at each rank level
    cand_to_num_voters_rank = dict()
    for r in range(1, num_cands + 1):
        cand_to_num_voters_rank[r] = {c: _num_rank_profile_with_ties(rankings, rcounts, c, r)
                                     for c in candidates}
        # Calculate cumulative scores up to current rank
        cand_scores = {c: sum([cand_to_num_voters_rank[_r][c] for _r in range(1, r + 1)])
                      for c in candidates}
        
        # Check if any candidate has a majority
        if any([s >= strict_maj_size for s in cand_scores.values()]):
            max_score = max(cand_scores.values())
            return sorted([c for c in candidates if cand_scores[c] >= max_score])
    
    # If no candidate has majority, return those with highest cumulative score
    max_score = max(cand_scores.values())
    return sorted([c for c in candidates if cand_scores[c] >= max_score])


@vm(name = "Simplified Bucklin",
    input_types = [ElectionTypes.PROFILE])
def simplified_bucklin(profile, curr_cands = None): 
    """If a candidate has a strict majority of first-place votes, then that candidate is the winner. If no such candidate exists, then check the candidates that are ranked first or second.  If a candidate has a strict majority of first- or second-place voters, then that candidate is the winner. If no such winner is found move on to the 3rd, 4th, etc. place votes. 

    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A sorted list of candidates

    :Example: 

    .. exec_code:: 

        from pref_voting.profiles import Profile
        from pref_voting.other_methods import simplified_bucklin

        prof = Profile([[1, 0, 2], [0, 2, 1], [0, 1, 2]], [2, 1, 1])

        prof.display()
        simplified_bucklin.display(prof)

    """
    strict_maj_size = profile.strict_maj_size()
    
    candidates = profile.candidates if curr_cands is None else curr_cands
    
    num_cands = candidates
    
    rankings = profile._rankings if curr_cands is None else _find_updated_profile(profile._rankings, np.array([c for c in profile.candidates if c not in curr_cands]), profile.num_cands)

    rcounts = profile._rcounts
    
    num_cands = len(candidates)
    ranks = range(1, num_cands + 1)
    
    cand_to_num_voters_rank = dict()
    for r in ranks:
        cand_to_num_voters_rank[r] = {c: _num_rank(rankings, rcounts, c, r)
                                      for c in candidates}
        cand_scores = {c:sum([cand_to_num_voters_rank[_r][c] for _r in cand_to_num_voters_rank.keys()]) 
                       for c in candidates}
        if any([s >= strict_maj_size for s in cand_scores.values()]):
            break
            
    return sorted([c for c in candidates if cand_scores[c] >= strict_maj_size])

def simplified_bucklin_with_explanation(profile, curr_cands = None): 
    """Return the Simplified Bucklin winners and the score for each candidate. 
 
    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 

        A sorted list of candidates

        A dictionary assigning the score for each candidate. 

    :Example: 

    .. exec_code:: 

        from pref_voting.profiles import Profile
        from pref_voting.other_methods import simplified_bucklin_with_explanation

        prof = Profile([[1, 0, 2], [0, 2, 1], [0, 1, 2]], [2, 1, 1])

        prof.display()
        sb_ws, scores = simplified_bucklin_with_explanation(prof)

        print(f"The winners are {sb_ws}")
        print(f"The candidate scores are {scores}")

    """
    strict_maj_size = profile.strict_maj_size()
    
    candidates = profile.candidates if curr_cands is None else curr_cands
    
    num_cands = candidates
    
    rankings = profile._rankings if curr_cands is None else _find_updated_profile(profile._rankings, np.array([c for c in profile.candidates if c not in curr_cands]), profile.num_cands)

    rcounts = profile._rcounts
    
    num_cands = len(candidates)
    ranks = range(1, num_cands + 1)
    
    cand_to_num_voters_rank = dict()
    for r in ranks:
        cand_to_num_voters_rank[r] = {c: _num_rank(rankings, rcounts, c, r)
                                      for c in candidates}
        cand_scores = {c:sum([cand_to_num_voters_rank[_r][c] for _r in cand_to_num_voters_rank.keys()]) 
                       for c in candidates}
        if any([s >= strict_maj_size for s in cand_scores.values()]):
            break
            
    return sorted([c for c in candidates if cand_scores[c] >= strict_maj_size]), cand_scores

@vm(name = "Weighted Bucklin",
    input_types = [ElectionTypes.PROFILE])
def weighted_bucklin(profile, curr_cands = None, strict_threshold = False, score = lambda num_cands, rank: (num_cands - rank)/ (num_cands - 1) if num_cands > 1 else 1): 
    """The Weighted Bucklin procedure, studied by D. Marc Kilgour, Jean-Charles Grégoire, and Angèle Foley. The k-th Weighted Bucklin score of a candidate c is the sum for j \leq k of the product of score(num_cands,j) and the number of voters who rank c in j-th place. Compute higher-order Weighted Bucklin scores until reaching a k such that some candidate's k-th Weighted Bucklin score is at least half the number of voters (or the strict majority size if strict_threshold = True). Then return the candidates with maximal k-th Weighted Bucklin score. Bucklin is the special case where strict_threshold = True and score = lambda num_cands, rank: 1.
    
    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``
        strict_threshold: If True, makes the threshold for the Bucklin procedure the strict majority size; otherwise threshold is half the number of voters, following Kilgour et al.
        score (function): A function that accepts two parameters ``num_cands`` (the number of candidates) and ``rank`` (a rank of a candidate) used to calculate the score of a candidate. The default ``score`` function is the normalized version of the classic Borda score vector.

    Returns: 
        A sorted list of candidates

    :Example: 

    .. exec_code:: 

        from pref_voting.profiles import Profile
        from pref_voting.other_methods import weighted_bucklin

        prof = Profile([[1, 0, 2], [0, 2, 1], [0, 1, 2]], [2, 1, 1])

        prof.display()
        weighted_bucklin.display(prof)

    """
    if strict_threshold == True:
        threshold = profile.strict_maj_size()
    else:
        threshold = profile.num_voters / 2

    candidates = profile.candidates if curr_cands is None else curr_cands
    
    num_cands = candidates
    
    rankings = profile._rankings if curr_cands is None else _find_updated_profile(profile._rankings, np.array([c for c in profile.candidates if c not in curr_cands]), profile.num_cands)

    rcounts = profile._rcounts
    
    num_cands = len(candidates)
    ranks = range(1, num_cands + 1)
    
    cand_to_num_voters_rank = dict()
    for r in ranks:
        cand_to_num_voters_rank[r] = {c: _num_rank(rankings, rcounts, c, r)
                                      for c in candidates}
        cand_scores = {c:sum([score(len(candidates), _r) * cand_to_num_voters_rank[_r][c] for _r in cand_to_num_voters_rank.keys()]) 
                       for c in candidates}
        if any([s >= threshold for s in cand_scores.values()]):
            break
    max_score = max(cand_scores.values())

    return sorted([c for c in candidates if cand_scores[c] >= max_score])

def _dodgson_score(profile, cand):
    """
    Return the **Dodgson score** of ``cand`` in ``profile``.

    The Dodgson score of a candidate *c* is the minimum number of
    *adjacent* swaps, summed over *all* ballots, needed to make *c* the
    **Condorcet winner**.

    This is equivalent to the the minimum number of places that $c$ 
    must *move up* in ballots to become a Condorcet winner
    (see Lemma 4.0.5 of John C. McCabe-Dansted, 
    *Approximability and computational feasibility of Dodgson’s rule*. 
    Master’s thesis, University of Auckland, 2006.)

    We formulate this as a *mixed‑integer program* and solve it with
    OR‑Tools/SCIP:

        • Variables x[r, k] = number of voters with ranking r who move 
        *c* upward by *exactly* k positions (0 ≤ k ≤ pos₍r₎(c)).
        • Objective: minimise Σ k · x[r, k] (total swaps).
        • Constraints
            – Partition: for each distinct ranking r, the x[r,·] must
              sum to the observed multiplicity of r.
            – Condorcet: for every rival d ≠ c, after the moves
              *c* must beat d by a strict majority.

    Parameters
    ----------
    profile : pref_voting.profiles.Profile
    cand : int, the candidate whose Dodgson score we compute.

    Returns
    -------
    int, the Dodgson score of *cand*.
    """
    rankings, counts = profile.rankings_counts   # ranking types & their multiplicities
    majority = profile.strict_maj_size()         # ⌊number of voters / 2⌋ + 1

    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        raise EnvironmentError("This OR‑Tools build lacks a MIP solver.")

    # Variables x[r, k]
    #
    # For each ranking type and each possible upward
    # move k (0..pos_c) we create an *integer* variable representing
    # the number of voters with that ranking who move *c* upward by
    # exactly k positions.
    #
    # The partition constraints (one per ranking) ensure that we do not
    # duplicate or delete ballots but only *re‑order* the existing ones.

    x = {}  # maps (r_idx, k) -> IntVar
    for r_idx, (ranking, w_np) in enumerate(zip(rankings, counts)):
        w = int(w_np)                # NumPy scalar → Python int  (SCIP wants int bounds)
        order = list(ranking)
        pos_c = order.index(cand)    # 0‑based position of cand in this ballot

        # create variables for k = 0 … pos_c
        for k in range(pos_c + 1):
            x[(r_idx, k)] = solver.IntVar(0, w, f"x_{r_idx}_{k}")

        # partition constraint  Σ_k x[r,k] = multiplicity of ranking r
        solver.Add(solver.Sum(x[(r_idx, k)] for k in range(pos_c + 1)) == w)

    # Condorcet constraints:  make cand beat every rival d ≠ cand
    # For each rival d we compare:
    #   current_support   – voters already preferring cand over d
    #   contributed_flips – voters whose ballots we move enough for cand
    #                       to pass d (k ≥ distance in that ranking)
    # The sum must reach the strict majority threshold.
    for d in range(profile.num_cands):
        if d == cand:
            continue  # skip self‑comparison

        current_support = profile.support(cand, d)
        flip_terms = []

        # identify voters who can move cand up by k positions to go above d
        for r_idx, ranking in enumerate(rankings):
            order = list(ranking)
            pos_c = order.index(cand)
            pos_d = order.index(d)
            if pos_d < pos_c:                       # cand currently below d
                dist = pos_c - pos_d                # minimum upward steps to pass d
                for k in range(dist, pos_c + 1):    # any k ≥ dist suffices
                    flip_terms.append(x[(r_idx, k)])

        # enforce majority: support_after ≥ majority
        solver.Add(current_support + solver.Sum(flip_terms) >= majority)

    # Objective: minimise total adjacent swaps  
    # Each voter who moves cand up by k positions contributes exactly k swaps.
    solver.Minimize(
        solver.Sum(k * var for (r_idx, k), var in x.items() if k > 0)
    )

    if solver.Solve() != pywraplp.Solver.OPTIMAL:
        raise RuntimeError("SCIP failed to prove optimality.")

    return int(solver.Objective().Value())

@vm(name="Dodgson", 
    input_types = [ElectionTypes.PROFILE])
def dodgson(profile, curr_cands=None, global_score = True):
    """The Dodgson score of a candidate is the minimum number of adjacent swaps in the ballots needed to make them a Condorcet winner. The Dodgson method selects the candidate with the minimum Dodgson score.

    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the candidates in curr_cands with the best Dodgson score.
        global_score (bool, optional): If True, then the Dodgson score is computed using the entire profile, including candidates not in curr_cands. If False, then the Dodgson score is computed using the profile restricted to the candidates in curr_cands.

    Returns:
        A sorted list of candidates

    """
    if curr_cands is None:
        curr_cands = profile.candidates

    if not global_score:
        profile.remove_candidates([c for c in profile.candidates if c not in curr_cands])

    scores = {c: _dodgson_score(profile, c) for c in curr_cands}

    best   = min(scores.values())
    return sorted([c for c, s in scores.items() if s == best])

@vm(name = "Bracket Voting",
    input_types = [ElectionTypes.PROFILE])
def bracket_voting(profile, curr_cands = None, seed = None, tie_break = "random"):
    """The candidates with the top four plurality scores are seeded into a bracket: the candidate with the highest plurality score is seeded 1st, the candidate with the second highest plurality score is seeded 2nd, etc. The 1st seed faces the 4th seed in a head-to-head match decided by majority rule, and the 2nd seed faces the 3rd seed in a head-to-head match decided by majority rule. The winners of these two matches face each other in a final head-to-head match decided by majority rule. The winner of the final is the winner of the election.

    .. note::
        A version of bracket voting as proposed by Edward B. Foley. By default, this is a probabilistic method that always returns a unique winner, as ties are broken using a random tie breaking ordering of the candidates.

    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``
        seed (int, optional): The seed for the random tie breaking ordering of the candidates.   
        tie_break (str, optional): The method used to break ties in the head-to-head matches. If set to "random", then a random tie breaking ordering of the candidates is used. If set to "lexicographic", then the candidates are sorted lexicographically.
    Returns: 
        A sorted list of candidates

    """
    cands = curr_cands if curr_cands else profile.candidates

    if len(cands) == 2:
        return plurality(profile, curr_cands = curr_cands)
    
    if tie_break == "random":
        rng = np.random.default_rng(seed)
        tie_breaking_ordering = cands.copy()
        rng.shuffle(tie_breaking_ordering)

    elif tie_break == "lexicographic":
        tie_breaking_ordering = sorted(cands)

    plurality_scores = profile.plurality_scores(curr_cands = cands)
    descending_plurality_scores = sorted(plurality_scores.values(), reverse=True)
    
    # If there is a tie for max plurality score, the first seed is the candidate with max plurality score who appears first in the tie breaking ordering
    potential_first_seeds = [c for c in cands if plurality_scores[c] == descending_plurality_scores[0]]
    first_seed = min(potential_first_seeds, key = lambda c: tie_breaking_ordering.index(c)) 

    potential_second_seeds = [c for c in cands if plurality_scores[c] == descending_plurality_scores[1] and c != first_seed]
    second_seed = min(potential_second_seeds, key = lambda c: tie_breaking_ordering.index(c))

    potential_third_seeds = [c for c in cands if plurality_scores[c] == descending_plurality_scores[2] and c not in [first_seed, second_seed]]
    third_seed = min(potential_third_seeds, key = lambda c: tie_breaking_ordering.index(c))

    potential_fourth_seeds = [c for c in cands if plurality_scores[c] == descending_plurality_scores[3] and c not in [first_seed, second_seed, third_seed]] if len(cands) > 3 else []
    fourth_seed = min(potential_fourth_seeds, key = lambda c: tie_breaking_ordering.index(c)) if len(potential_fourth_seeds) > 0 else None

    # Ties in semi-final head-to-head matches are broken in favor of the higher-seeded candidate
    if len(cands) == 3:
        one_four_winner = first_seed
        one_four_winner_seed = 1
    else: 
        one_four_winner = first_seed if profile.margin(first_seed, fourth_seed) >= 0 else fourth_seed
        one_four_winner_seed = 1 if one_four_winner == first_seed else 4

    two_three_winner = second_seed if profile.margin(second_seed, third_seed) >= 0 else third_seed
    two_three_winner_seed = 2 if two_three_winner == second_seed else 3

    if profile.margin(one_four_winner, two_three_winner) > 0:
        winner = one_four_winner

    elif profile.margin(one_four_winner, two_three_winner) < 0:
        winner = two_three_winner

    # Ties in the final head-to-head match are broken in favor of the higher-seeded candidate
    else:
        winner = one_four_winner if one_four_winner_seed < two_three_winner_seed else two_three_winner

    return [winner]

@vm(name = "Superior Voting",
    input_types = [ElectionTypes.PROFILE])
def superior_voting(profile, curr_cands = None):
    """One candidate is superior to another if more ballots rank the first candidate above the second than vice versa. A candidate earns a point from a ballot if they are ranked first on that ballot or they are superior to the candidate ranked first on that ballot. The candidate with the most points wins.

    .. note::
        Devised by Wesley H. Holliday as a simple Condorcet-compliant method for political elections. Always elects a Condorcet winner if one exists and elects only the Condorcet winner provided the Condorcet winner receives at least one first-place vote. Edward B. Foley suggested the name 'Superior Voting' because the method is based on the idea that if A is superior to B, then A should get B's first-place votes added to their own.

    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A sorted list of candidates

    """
    curr_cands = profile.candidates if curr_cands is None else curr_cands

    # Calculate the points for each candidate
    points = {cand: profile.plurality_scores(curr_cands)[cand] for cand in curr_cands}
    for cand in curr_cands:
        for other_cand in curr_cands:
            if profile.margin(cand, other_cand) > 0:
                points[cand] += profile.plurality_scores(curr_cands)[other_cand]
    
    # Find the candidates with the most points
    max_score = max(points.values())
    winners = [cand for cand in curr_cands if points[cand] == max_score]

    return winners

def bt_mle(pmat, max_iter=100):
    """Lucas Maystre's implementation of MLE for the Bradley-Terry model (https://datascience.stackexchange.com/questions/18828/from-pairwise-comparisons-to-ranking-python). 
    
    Note we change the interpretation of p_{i,j} to be the probability that i is preferred to j, rather than vice versa as in the original implementation.
    """
    n = pmat.shape[0]
    wins = np.sum(pmat, axis=1)
    params = np.ones(n, dtype=float)
    for _ in range(max_iter):
        tiled = np.tile(params, (n, 1))
        combined = 1.0 / (tiled + tiled.T)
        np.fill_diagonal(combined, 0)
        nxt = wins / np.sum(combined, axis=0)
        nxt = nxt / np.mean(nxt)
        if np.linalg.norm(nxt - params, ord=np.inf) < 1e-6:
            return nxt
        params = nxt
    raise RuntimeError('did not converge')

@vm(name = "Bradley-Terry",
    input_types = [ElectionTypes.PROFILE])
def bradley_terry(prof, curr_cands = None, threshold = .00001):
    """The Bradley-Terry model is a probabilistic model for pairwise comparisons. In this model, the probability that a voter prefers candidate i to candidate j is given by p_{i,j} = v_i / (v_i + v_j), where v_i is the strength of candidate i. Given a profile, we take p_{i,j} to be the proportion of voters who prefer candidate i to candidate j. We then estimate the strength of each candidate using maximum likelihood estimation. The winning candidates are those whose estimated strength is within +/- threshold of the maximum strength.

    .. note::
        For profiles of linear ballots, this is equivalent to Borda (see Theorem 3.1 of https://arxiv.org/abs/2312.08358).

    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``
        threshold (float, optional): The threshold for determining the winners. The winners are those whose estimated strength is within +/- threshold of the maximum strength.

    Returns: 
        A sorted list of candidates
    """

    curr_cands = prof.candidates if curr_cands is None else curr_cands
    
    prop_matrix = np.zeros((len(curr_cands), len(curr_cands)))

    for i, c in enumerate(curr_cands):
        for j, d in enumerate(curr_cands):
            if i != j:
                prop_matrix[i][j] = prof.support(c,d) / prof.num_voters

    params = bt_mle(prop_matrix)

    max_value = np.max(params)
    winner_indices = np.where(np.abs(params - max_value) <= threshold)[0]
    winners = [curr_cands[i] for i in winner_indices]

    return sorted(winners)

@swf(name = "Bradley-Terry Ranking")
def bradley_terry_ranking(prof, curr_cands = None, threshold = .00001):
    """The Bradley-Terry model is a probabilistic model for pairwise comparisons. In this model, the probability that a voter prefers candidate i to candidate j is given by p_{i,j} = v_i / (v_i + v_j), where v_i is the strength of candidate i. Given a profile, we take p_{i,j} to be the proportion of voters who prefer candidate i to candidate j. We then estimate the strength of each candidate using maximum likelihood estimation. Finally, the candidates are ranked in decreasing order of their estimated strength (where candidates whose estimated strength is within +/- threshold of each other are considered tied).

    .. note::
        For profiles of linear ballots, this is equivalent to Borda (see Theorem 3.1 of https://arxiv.org/abs/2312.08358).

    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``
        threshold (float, optional): The threshold for equivalence classes of candidates. 

    Returns: 
        A Ranking object.
    """

    curr_cands = prof.candidates if curr_cands is None else curr_cands
    
    support_matrix = np.zeros((len(curr_cands), len(curr_cands)))

    for i, c in enumerate(curr_cands):
        for j, d in enumerate(curr_cands):
            if i != j:
                support_matrix[i][j] = prof.support(c,d) / prof.num_voters

    params = bt_mle(support_matrix)

    ranking_dict = dict()
    cands_assigned = list()
    curr_ranking = 1

    while len(cands_assigned) < len(curr_cands):

        max_value = np.max([params[curr_cands.index(c)] for c in curr_cands if c not in cands_assigned])

        for c in curr_cands:
            if c not in cands_assigned and np.abs(params[curr_cands.index(c)] - max_value) <= threshold:
                ranking_dict[c] = curr_ranking
                cands_assigned.append(c)

        curr_ranking += 1

    return Ranking(ranking_dict)