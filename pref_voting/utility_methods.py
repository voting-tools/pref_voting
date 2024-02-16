'''
    File: utility_methods.py
    Author: Wes Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)
    Date: May 26, 2023
    
    Implementations of utility methods. 
'''
from pref_voting.voting_method import vm
from pref_voting.social_welfare_function import swf
from pref_voting.rankings import  Ranking
import numpy as np  

@swf(name="Sum Utilitarian")
def sum_utilitarian_ranking(uprof, curr_cands = None): 
    """Rank the alternatives according to the sum of the utilities. 
    Args:
        uprof (Profile): A Profile object.
        curr_cands (list): A list of candidates to restrict the ranking to. If ``None``, then the ranking is over the entire domain of the profile.
    Returns:
        Ranking: A ranking of the candidates in ``curr_cands`` according to sum of the utilities.
    """

    curr_cands = curr_cands if curr_cands is not None else uprof.domain

    sums = {c:uprof.util_sum(c) for c in curr_cands}
    sorted_sums = sorted(list(set(sums.values())), reverse=True)
    return Ranking({x: uidx+1 for uidx, u in enumerate(sorted_sums) 
                    for x in curr_cands if sums[x] == u})

@vm(name="Sum Utilitarian")
def sum_utilitarian(uprof, curr_cands=None): 
    """
    Return the winning set of candidates according to sum of the utilities.
    """
    return sorted(sum_utilitarian_ranking(uprof, curr_cands=curr_cands).first())


@swf(name="Relative Utilitarian")
def relative_utilitarian_ranking(uprof, curr_cands=None): 
    """
    Rank the alternatives according to sum of the normalized utilities. 

    Args:
        uprof (Profile): A Profile object.
        curr_cands (list): A list of candidates to restrict the ranking to. If ``None``, then the ranking is over the entire domain of the profile.
    Returns:
        Ranking: A ranking of the candidates in ``curr_cands`` according to sum of the normalized utilities.

    .. note::
        Before restricting to curr_cands, we normalize with respect to *all* alternatives in the domain. 
    """

    curr_cands = curr_cands if curr_cands is not None else uprof.domain

    rel_utils = [u.normalize_by_range() for u in uprof.utilities]
    sums = {c:np.sum([u(c) for u in rel_utils if u(c) is not None]) for c in curr_cands}
    sorted_sums = sorted(list(set(sums.values())), reverse=True)
    return Ranking({x: uidx+1 for uidx, u in enumerate(sorted_sums) 
                    for x in curr_cands if sums[x] == u})

@vm(name="Relative Utilitarian")
def relative_utilitarian(uprof, curr_cands=None): 
    """
    Return the winning set of candidates according to sum of the normalized utilities.
    """
    return sorted(relative_utilitarian_ranking(uprof, curr_cands=curr_cands).first())


@swf(name="Maximin")
def maximin_ranking(uprof, curr_cands=None): 
    """
    Rank the alternatives according to the minimum utility.

    Args:
        uprof (Profile): A Profile object.
        curr_cands (list): A list of candidates to restrict the ranking to. If ``None``, then the ranking is over the entire domain of the profile.
    Returns:    
        Ranking: A ranking of the candidates in ``curr_cands`` according to minimum utility.
    """
    
    curr_cands = curr_cands if curr_cands is not None else uprof.domain

    min_utils = {x:uprof.util_min(x) for x in curr_cands}
    sorted_min_utils = sorted(list(set(min_utils.values())), reverse=True)
    return Ranking({x: midx+1 
                    for midx, m in enumerate(sorted_min_utils) 
                    for x in min_utils.keys() if min_utils[x] == m})

@vm(name="Maximin")
def maximin(uprof, curr_cands=None):
    """
    Return the winning set of candidates according to minimum utility.
    """
    return sorted(maximin_ranking(uprof, curr_cands=curr_cands).first())


@swf(name="Lexicographic Maximin")
def lexicographic_maximin_ranking(uprof, curr_cands=None): 
    """
    Rank the alternatives according to the lexicographic maximin ranking. The lexicographic maximin ranking is the ranking that ranks alternatives according to the minimum utility, and then breaks ties by ranking alternatives according to the second minimum utility, and so on.

    Args:
        uprof (Profile): A Profile object.
        curr_cands (list): A list of candidates to restrict the ranking to. If ``None``, then the ranking is over the entire domain of the profile.
    Returns:
        Ranking: A ranking of the candidates in ``curr_cands`` according to lexicographic maximin ranking.
    """

    curr_cands = curr_cands if curr_cands is not None else uprof.domain

    utils = {x: tuple(sorted([u(x) for u in uprof.utilities if u(x) is not None])) for x in curr_cands}
    assert len(list(set([len(us) for us in utils.values()]))) == 1, "Not all the items have the same number of utilities."
    sorted_utils = sorted(list(set(utils.values())), reverse=True)
    return Ranking({x: idx+1 
                    for idx, us in enumerate(sorted_utils) 
                    for x in utils.keys() if utils[x] == us})

@vm(name="Lexicographic Maximin")
def lexicographic_maximin(uprof, curr_cands=None):
    """
    Return the winning set of candidates according to lexicographic maximin ranking.
    """
    return sorted(lexicographic_maximin_ranking(uprof, curr_cands=curr_cands).first())


@swf(name="Nash")
def nash_ranking(uprof, sq=None, curr_cands=None): 
    """
    Rank the alternatives according to the Nash product ranking. Given the status quo ``sq``, the Nash product ranking ranks alternatives according to the product of the utilities of the alternatives minus the utility of the status quo.
    
    Args:
        uprof (Profile): A Profile object.
        sq (Candidate): The status quo. If ``None``, then the status quo is the first candidate in the domain of the profile.
        curr_cands (list): A list of candidates to restrict the ranking to. If ``None``, then the ranking is over the entire domain of the profile.
    Returns:
        Ranking: A ranking of the candidates in ``curr_cands`` according to Nash product ranking.
    
    """
    
    assert sq is None or sq in uprof.domain, f"The status quo {sq} must be in the domain of the profile."
    
    curr_cands = curr_cands if curr_cands is not None else uprof.domain

    sq = curr_cands[0] if sq is None else sq

    items_to_rank = list(set([x for x in curr_cands if all([u(x) > u(sq) for u in uprof.utilities])] + [sq]))
    
    nash_utils = {x: np.prod([u(x) - u(sq) for u in uprof.utilities]) for x in items_to_rank}
    sorted_nash_utils = sorted(list(set(nash_utils.values())), reverse=True)
    
    return Ranking({x: nidx+1 
                    for nidx, n in enumerate(sorted_nash_utils) 
                    for x in nash_utils.keys() if nash_utils[x] == n})

@vm(name="Nash")
def nash(uprof, sq=None, curr_cands=None):
    """
    Return the winning set of candidates according to Nash product ranking.
    """
    return sorted(nash_ranking(uprof, sq=sq, curr_cands=curr_cands).first())

utilitarian_vms = [
    sum_utilitarian,
    relative_utilitarian,
    maximin,
    lexicographic_maximin,
    nash
]

utilitarian_swfs = [
    sum_utilitarian_ranking,
    relative_utilitarian_ranking,
    maximin_ranking,
    lexicographic_maximin_ranking,
    nash_ranking
]