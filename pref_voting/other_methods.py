'''
    File: mg_methods.py
    Author: Wes Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)
    Date: January 12, 2022
    Updated: October 24, 2023
    
    Implementations of 
'''

from pref_voting.voting_method import *
from pref_voting.scoring_methods import plurality
from pref_voting.profiles import _find_updated_profile, _num_rank
from pref_voting.helper import get_mg
from itertools import combinations, permutations, chain
import networkx as nx


@vm(name = "Majority")
def majority(profile, curr_cands = None):
    """The majority winner is the candidate with a strict majority  of first place votes.  Returns an empty list if there is no candidate with a strict majority of first place votes. Returns the majority winner in the ``profile`` restricted to ``curr_cands``.

    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A sorted list of candidates

    .. important:: 
        Formally, this is *not* a voting method since the function might return an empty list (when there is no candidate with a strict majority of first place votes).  Also, if there is a majority winner, then that winner is unique. 

    :Example:
        .. exec_code::

            from pref_voting.profiles import Profile
            from pref_voting.other_methods import majority
            
            prof1 = Profile([[0, 1, 2], [1, 0, 2], [2, 1, 0]], [3, 1, 2])
            prof1.display()
            majority.display(prof1)

            prof2 = Profile([[0, 1, 2], [1, 0, 2], [1, 2, 0]], [5, 1, 2])
            prof2.display()
            majority.display(prof2)

    """
    maj_size = profile.strict_maj_size()
    curr_cands = profile.candidates if curr_cands is None else curr_cands

    plurality_scores = profile.plurality_scores(curr_cands = curr_cands)
    maj_winner = [c for c in curr_cands if plurality_scores[c] >= maj_size]

    return sorted(maj_winner)

@vm(name = "Pareto")
def pareto(prof, curr_cands = None, strong_Pareto = False, use_extended_strict_preferences = True):
    """Returns the set of candidates who are not Pareto dominated.

    For ProfilesWithTies, if strong_Pareto == True, then a dominates b if some voter strictly prefers a to b and no voter strictly prefers b to a.

    Args:
        prof (Profile, ProfileWithTies): An anonymous profile of linear (or strict weak) orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A sorted list of candidates

    """
    if use_extended_strict_preferences:
        prof.use_extended_strict_preference()
        
    Pareto_dominated = set()
    candidates = prof.candidates if curr_cands is None else curr_cands
    for a in candidates:
        for b in candidates:
            if not strong_Pareto and prof.support(a,b) == prof.num_voters:
                Pareto_dominated.add(b)

            if strong_Pareto and prof.support(a,b) > 0 and prof.support(b,a) == 0:
                Pareto_dominated.add(b)     

    return list(set(candidates) - Pareto_dominated)
    
## Banks
#

def seqs(iterable):
    s = list(iterable)
    return chain.from_iterable(permutations(s, r) for r in range(len(s)+1))

def is_transitive(G, p):
    for c1_idx, c1 in enumerate(p[:-1]):
        for c2 in p[c1_idx+1::]:            
            if not G.has_edge(c1,c2):
                return False
    return True

def is_subsequence(x, y):
    it = iter(y)
    return all(any(c == ch for c in it) for ch in x)

@vm(name = "Banks")
def banks(edata, curr_cands = None): 
    """ Say that a *chain* in majority graph is a subset of candidates that is linearly ordered by the majority relation. Then a candidate :math:`a` if :math:`a` is the maximum element with respect to the majority relation of some maximal chain in the majority graph.

    Args:
        edata (Profile, ProfileWithTies, MarginGraph): Any election data that has a `margin` method. 
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A sorted list of candidates


    :Example: 

    .. plot::  margin_graphs_examples/mg_ex_banks.py
        :context: reset  
        :include-source: True


    .. code-block:: 

        from pref_voting.other_methods import banks

        banks.display(prof)


    .. exec_code:: 
        :hide_code:

        from pref_voting.weighted_majority_graphs import MarginGraph
        from pref_voting.other_methods import banks
        
        mg = MarginGraph([0, 1, 2, 3], [(0, 2, 2), (0, 3, 6), (1, 0, 8), (2, 3, 4), (2, 1, 10), (3, 1, 12)])

        banks.display(mg)

    """

    mg = get_mg(edata, curr_cands = curr_cands)
    trans_paths = list()
    for s in seqs(mg.nodes):
        if nx.algorithms.simple_paths.is_simple_path(mg, s):
            if is_transitive(mg, s): 
                trans_paths.append(s)

    maximal_paths = list()
    #print("max paths")
    for s in trans_paths:
        is_max = True
        for other_s in trans_paths: 
            if s != other_s:
                if is_subsequence(s, other_s): 
                    is_max = False
                    break
        if is_max:
            maximal_paths.append(s)
    
    return sorted(list(set([p[0] for p in maximal_paths])))

def banks_with_explanation(edata, curr_cands = None): 
    """Return the Banks winners and the list of maximal chains in the majority graph. 

    Args:
        edata (Profile, ProfileWithTies, MarginGraph): Any election data that has a `margin` method. 
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A sorted list of candidates

        A list of lists of candidates each representing a maximal chain in the majority graph

    :Example: 

    .. plot::  margin_graphs_examples/mg_ex_banks.py
        :context: reset  
        :include-source: True


    .. code-block:: 

        from pref_voting.other_methods import banks_with_explanation

        bws, maximal_chains = banks_with_explanation(mg)

        print(f"Winning set: {bws}")
        for c in maximal_chains: 
            print(f"Maximal chain: {c}")


    .. exec_code:: 
        :hide_code:

        from pref_voting.weighted_majority_graphs import MarginGraph
        from pref_voting.other_methods import banks_with_explanation
        
        mg = MarginGraph([0, 1, 2, 3], [(0, 2, 2), (0, 3, 6), (1, 0, 8), (2, 3, 4), (2, 1, 10), (3, 1, 12)])

        bws, maximal_chains = banks_with_explanation(mg)

        print(f"Winning set: {bws}")
        for c in maximal_chains: 
            print(f"Maximal chain: {c}")

    """

    mg = get_mg(edata, curr_cands = curr_cands)
    trans_paths = list()
    for s in seqs(mg.nodes):
        if nx.algorithms.simple_paths.is_simple_path(mg, s):
            if is_transitive(mg, s): 
                trans_paths.append(s)

    maximal_paths = list()
    #print("max paths")
    for s in trans_paths:
        is_max = True
        for other_s in trans_paths: 
            if s != other_s:
                if is_subsequence(s, other_s): 
                    is_max = False
                    break
        if is_max:
            maximal_paths.append(s)
    
    return sorted(list(set([p[0] for p in maximal_paths]))), maximal_paths

## Slater
#

def distance_to_margin_graph(edata, rel, exp = 1, curr_cands = None): 
    """
    Calculate the distance of ``rel`` (a relation) to the majority graph of ``edata``. 
    """
    candidates = edata.candidates if curr_cands is None else curr_cands
    
    penalty = 0
    for a,b in combinations(candidates, 2): 
        if edata.majority_prefers(a, b) and (b,a) in rel: 
            penalty += (edata.margin(a, b) ** exp)
        elif edata.majority_prefers(b, a) and (a,b) in rel: 
            penalty += (edata.margin(b, a) ** exp)
        elif edata.majority_prefers(a, b) and (a,b) not in rel and (b,a) not in rel: 
            penalty += (edata.margin(a, b) ** exp) / 2 
        elif edata.majority_prefers(b, a) and (a,b) not in rel and (b,a) not in rel: 
            penalty += (edata.margin(b, a) ** exp)  / 2
    return penalty


def lin_order_to_rel(lin_order): 
    """Convert a linear order (a list of items) into a set of ordered pairs"""
    els = sorted(lin_order)
    rel = []
    for a,b in combinations(els, 2):
        if lin_order.index(a) < lin_order.index(b): 
            rel.append((a,b))
        elif lin_order.index(b) < lin_order.index(a): 
            rel.append((b,a))     
    return rel


def slater_rankings(edata, curr_cands = None): 
    """
    A Slater ranking is a linear order :math:`R` of the candidates that minimizes the number of edges in the majority graph we have to turn around before we obtain :math:`R`. 

    Args:
        edata (Profile, ProfileWithTies, MarginGraph): Any election data that has a `margin` method. 
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        rankings: A list of Slater rankings.

        dist: The minimum distance of the Slater rankings.

    :Example:

    .. exec_code::

        from pref_voting.weighted_majority_graphs import MarginGraph
        from pref_voting.other_methods import slater_rankings
        
        mg = MarginGraph([0, 1, 2, 3], [(0, 2, 2), (0, 3, 6), (1, 0, 8), (2, 3, 4), (2, 1, 10), (3, 1, 12)])

        srs, d = slater_rankings(mg)
        print(f"minimum distance: {d}")
        for sr in srs: 
            print(f"ranking: {sr}") 
    """
    candidates = edata.candidates if curr_cands is None else curr_cands
    min_dist = np.inf
    
    rankings = list()
    for lin_order in permutations(candidates): 
        lo_rel = lin_order_to_rel(lin_order)
        
        dist = distance_to_margin_graph(edata, lo_rel, exp = 0, curr_cands = curr_cands)
        if dist < min_dist: 
            min_dist = dist
            rankings = [lin_order]
        elif dist == min_dist: 
            rankings.append(lin_order)
    return rankings, min_dist

        
@vm(name = "Slater")
def slater(edata, curr_cands = None): 
    """A Slater ranking is a linear order :math:`R` of the candidates that minimizes the number of edges in the majority graph we have to turn around before we obtain :math:`R`.   A candidate is a Slater winner if the candidate is the top element of some Slater ranking.

    Args:
        edata (Profile, ProfileWithTies, MarginGraph): Any election data that has a `margin` method. 
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A sorted list of candidates


    :Example: 

    .. plot::  margin_graphs_examples/mg_ex_slater.py
        :context: reset  
        :include-source: True


    .. code-block:: 

        from pref_voting.other_methods import slater

        slater.display(prof)


    .. exec_code:: 
        :hide_code:

        from pref_voting.weighted_majority_graphs import MarginGraph
        from pref_voting.other_methods import slater
        
        mg = MarginGraph([0, 1, 2, 3], [(0, 2, 2), (0, 3, 6), (1, 0, 8), (2, 3, 4), (2, 1, 10), (3, 1, 12)])

        slater.display(mg)

    """    
    rankings, dist = slater_rankings(edata, curr_cands = curr_cands)
    
    return sorted(list(set([r[0] for r in rankings])))


## Kemeny-Young Method 
#
def kendalltau_dist(rank_a, rank_b):
    rank_a = tuple(rank_a)
    rank_b = tuple(rank_b)
    tau = 0
    candidates = sorted(rank_a)
    for i, j in combinations(candidates, 2):
        tau += (np.sign(rank_a.index(i) - rank_a.index(j)) == -np.sign(rank_b.index(i) - rank_b.index(j)))
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
        rankings: A list of Slater rankings.
        
        dist: The minimum distance of the Slater rankings.


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


@vm(name = "Kemeny-Young")
def kemeny_young(profile, curr_cands = None): 
    """A Kemeny-Young ranking is a ranking that minimizes the sum of the Kendall tau distances to the voters' rankings.  The Kemeny-Young winners are the candidates that are ranked first by some Kemeny-Young ranking.

    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

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

    candidates = profile.candidates if curr_cands is None else curr_cands
    
    rankings = profile._rankings if curr_cands is None else _find_updated_profile(profile._rankings, np.array([c for c in profile.candidates if c not in curr_cands]),  profile.num_cands)
    ky_rankings, min_dist = _kemeny_young_rankings(list(rankings), list(profile._rcounts), candidates)
    
    return sorted(list(set([r[0] for r in ky_rankings])))

### Bucklin

@vm(name = "Bucklin")
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



@vm(name = "Simplified Bucklin")
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


@vm(name = "Weighted Bucklin")
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


@vm(name = "Bracket Voting")
def bracket_voting(profile, curr_cands = None):
    """The candidates with the top four plurality scores are seeded into a bracket: the candidate with the highest plurality score is seeded 1st, the candidate with the second highest plurality score is seeded 2nd, etc. The 1st seed faces the 4th seed in a head-to-head match decided by majority rule, and the 2nd seed faces the 3rd seed in a head-to-head match decided by majority rule. The winners of these two matches face each other in a final head-to-head match decided by majority rule. The winner of the final is the winner of the election.

    .. note::
        A version of bracket voting as proposed by Edward B. Foley. This is a probabilistic method that always returns a unique winner. Ties are broken using a random tie breaking ordering of the candidates.

    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A sorted list of candidates

    """
    cands = curr_cands if curr_cands else profile.candidates

    if len(cands) == 2:
        return plurality(profile, curr_cands = curr_cands)
    
    # Generate a random tie breaking ordering of cands
    tie_breaking_ordering = cands.copy()
    random.shuffle(tie_breaking_ordering)

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

@vm(name = "Superior Voting")
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

other_vms = [
    banks,
    slater,
    kemeny_young, 
    majority, 
    bucklin,
    simplified_bucklin,
    weighted_bucklin,
    bracket_voting,
    superior_voting
]