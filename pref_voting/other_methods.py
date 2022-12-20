'''
    File: mg_methods.py
    Author: Eric Pacuit (epacuit@umd.edu)
    Date: January 12, 2022
    
    Implementations of 
'''

from pref_voting.voting_method import  *
from pref_voting.profiles import _find_updated_profile, _num_rank
from pref_voting.helper import get_mg
from itertools import combinations, permutations, chain
import networkx as nx


@vm(name = "Majority")
def majority(profile, curr_cands = None):
    """The majority winner is the candidate with a strict majority  of first place votes.  Returns an empty list if there is no candidate with a strict majority of first place votes. Returns the majority winner in the ``profile`` restricted to ``curr_cands``.

    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restrcited to the candidates in ``curr_cands``

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
    Calclulate the distance of ``rel`` (a relation) to the majority graph of ``edata``. 
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
    A Slater ranking is a linear order :math:`R` of the candidates that minimises the number of edges in the majority graph we have to turn around before we obtain :math:`R`. 

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
    """A Slater ranking is a linear order :math:`R` of the candidates that minimises the number of edges in the majority graph we have to turn around before we obtain :math:`R`.   A candidate is a Slater winner if the candidate is the top element of some Slater ranking.

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


## Kemmeny-Young Method 
#
def kendalltau_dist(rank_a, rank_b):
    rank_a = tuple(rank_a)
    rank_b = tuple(rank_b)
    tau = 0
    candidates = sorted(rank_a)
    for i, j in combinations(candidates, 2):
        tau += (np.sign(rank_a.index(i) - rank_a.index(j)) == -np.sign(rank_b.index(i) - rank_b.index(j)))
    return tau


def _kemmeny_young_rankings(rankings, rcounts, candidates): 
    
    rankings_dist = dict()
    for ranking in permutations(candidates): 
        rankings_dist[tuple(ranking)] = sum(c * kendalltau_dist(tuple(r), ranking) 
        for r,c in zip(rankings, rcounts))
    min_dist = min(rankings_dist.values())

    lin_orders = [r for r in rankings_dist.keys() if rankings_dist[r] == min_dist]
    
    return lin_orders, min_dist

def kemmeny_young_rankings(profile, curr_cands = None): 
    """
    A Kemmeny-Young ranking is a ranking that minimizes the sum of the Kendall tau distances to the voters' rankings.  
    
    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restrcited to the candidates in ``curr_cands``

    Returns: 
        rankings: A list of Slater rankings.
        
        dist: The minimum distance of the Slater rankings.


    :Example:
        .. exec_code::

            from pref_voting.profiles import Profile
            from pref_voting.other_methods import kemmeny_young, kemmeny_young_rankings
            
            prof1 = Profile([[0, 1, 2], [1, 0, 2], [2, 1, 0]], [3, 1, 2])
            prof1.display()
            kyrs, d = kemmeny_young_rankings(prof1)
            print(f"Minimal distance: {d}")
            for kyr in kyrs: 
                print(f"ranking: {kyr}")

            prof2 = Profile([[0, 1, 2], [1, 0, 2], [1, 2, 0]], [5, 1, 2])
            prof2.display()
            kyrs, d = kemmeny_young_rankings(prof2)
            print(f"Minimal distance: {d}")
            for kyr in kyrs: 
                print(f"ranking: {kyr}")

    """
    candidates = profile.candidates if curr_cands is None else curr_cands
    
    rankings = profile._rankings if curr_cands is None else _find_updated_profile(profile._rankings, np.array([c for c in profile.candidates if c not in curr_cands]),  profile.num_cands)
    return _kemmeny_young_rankings(list(rankings), list(profile._rcounts), candidates)


@vm(name = "Kemmeny-Young")
def kemmeny_young(profile, curr_cands = None): 
    """A Kemmeny-Young ranking is a ranking that minimizes the sum of the Kendall tau distances to the voters' rankings.  The Kemmeny-Young winners are the candidates that are ranked first by some Kemmeny-Young ranking.

    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restrcited to the candidates in ``curr_cands``

    Returns: 
        A sorted list of candidates

    :Example:

        .. exec_code::

            from pref_voting.profiles import Profile
            from pref_voting.other_methods import kemmeny_young, kemmeny_young_rankings
            
            prof1 = Profile([[0, 1, 2], [1, 0, 2], [2, 1, 0]], [3, 1, 2])
            prof1.display()
            kyrs, d = kemmeny_young_rankings(prof1)
            print(f"Minimal distance: {d}")
            for kyr in kyrs: 
                print(f"ranking: {kyr}")
            kemmeny_young.display(prof1)

            prof2 = Profile([[0, 1, 2], [1, 0, 2], [1, 2, 0]], [5, 1, 2])
            prof2.display()
            kyrs, d = kemmeny_young_rankings(prof2)
            print(f"Minimal distance: {d}")
            for kyr in kyrs: 
                print(f"ranking: {kyr}")
            kemmeny_young.display(prof2)

    """

    candidates = profile.candidates if curr_cands is None else curr_cands
    
    rankings = profile._rankings if curr_cands is None else _find_updated_profile(profile._rankings, np.array([c for c in profile.candidates if c not in curr_cands]),  profile.num_cands)
    ky_rankings, min_dist = _kemmeny_young_rankings(list(rankings), list(profile._rcounts), candidates)
    
    return sorted(list(set([r[0] for r in ky_rankings])))

### Bucklin

@vm(name = "Bucklin")
def bucklin(profile, curr_cands = None): 
    """If a candidate has a strict majority of first-place votes, then that candidate is the winner. If no such candidate exists, then check the candidates that are ranked first or second.  If a candidate has a strict majority of first- or second-place voters, then that candidate is the winner. If no such winner is found move on to the 3rd, 4th, etc. place votes.  Return the candidates with the greatest overall score.  
    
    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restrcited to the candidates in ``curr_cands``

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
        curr_cands (List[int], optional): If set, then find the winners for the profile restrcited to the candidates in ``curr_cands``

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
        curr_cands (List[int], optional): If set, then find the winners for the profile restrcited to the candidates in ``curr_cands``

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
        curr_cands (List[int], optional): If set, then find the winners for the profile restrcited to the candidates in ``curr_cands``

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



other_vms = [
    banks,
    slater,
    kemmeny_young, 
    majority, 
    bucklin,
    simplified_bucklin
]