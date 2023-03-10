'''
    File: iterative_methods.py
    Author: Eric Pacuit (epacuit@umd.edu)
    Date: January 6, 2022
    Update: July 23, 2022
    
    Implementations of iterative voting methods.
'''

from pref_voting.voting_method import  *
from pref_voting.voting_method import _num_rank_last, _num_rank_first
from pref_voting.profiles import  _borda_score, _find_updated_profile
from pref_voting.margin_based_methods import split_cycle_faster

import copy
from itertools import permutations, product
import numpy as np

@vm(name = "Instant Runoff")
def instant_runoff(profile, curr_cands = None):
    """
    If there is a majority winner then that candidate is the  winner.  If there is no majority winner, then remove all candidates that are ranked first by the fewest number of voters.  Continue removing candidates with the fewest number first-place votes until there is a candidate with a majority of first place votes.  
    
    .. important::
        If there is  more than one candidate with the fewest number of first-place votes, then *all* such candidates are removed from the profile. 
    

    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A sorted list of candidates

    .. seealso::

        Instant Runoff is also known as "Ranked Choice", "Hare", and "Alternative Vote".

        Related functions:  :func:`pref_voting.iterative_methods.instant_runoff_tb`, :func:`pref_voting.iterative_methods.instant_runoff_put`, :func:`pref_voting.iterative_methods.instant_runoff_with_explanation`
   
    :Example: 

    .. exec_code:: 

        from pref_voting.profiles import Profile
        from pref_voting.iterative_methods import instant_runoff, ranked_choice, alterantive_vote, hare
        
        prof = Profile([[2, 1, 0], [0, 2, 1], [1, 2, 0]], [1, 2, 2])

        prof.display()
        instant_runoff.display(prof)
        ranked_choice.display(prof)
        alterantive_vote.display(prof)
        hare.display(prof)

    """

    # need the total number of all candidates in a profile to check when all candidates have been removed   
    num_cands = profile.num_cands 
    
    candidates = profile.candidates if curr_cands is None else curr_cands
    cands_to_ignore = np.empty(0) if curr_cands is None else np.array([c for c in profile.candidates if c not in curr_cands])

    strict_maj_size = profile.strict_maj_size()
    
    rs, rcounts = profile.rankings_counts # get all the ranking data
    

    winners = [c for c in candidates 
               if _num_rank_first(rs, rcounts, cands_to_ignore, c) >= strict_maj_size]

    while len(winners) == 0:
        plurality_scores = {c: _num_rank_first(rs, rcounts, cands_to_ignore, c) for c in candidates 
                            if  not isin(cands_to_ignore,c)}  
        min_plurality_score = min(plurality_scores.values())
        lowest_first_place_votes = np.array([c for c in plurality_scores.keys() 
                                             if  plurality_scores[c] == min_plurality_score])

        # remove cands with lowest plurality score
        cands_to_ignore = np.concatenate((cands_to_ignore, lowest_first_place_votes), axis=None)
        if len(cands_to_ignore) == num_cands: # removed all of the candidates 
            winners = sorted(lowest_first_place_votes)
        else:
            winners = [c for c in candidates 
                       if not isin(cands_to_ignore,c) and _num_rank_first(rs, rcounts, cands_to_ignore, c) >= strict_maj_size]
     
    return sorted(winners)

# Create some aliases for instant runoff
instant_runoff.set_name("Hare")
hare = copy.deepcopy(instant_runoff)
instant_runoff.set_name("Ranked Choice")
ranked_choice = copy.deepcopy(instant_runoff)
instant_runoff.set_name("Alternative Vote")
alterantive_vote = copy.deepcopy(instant_runoff)

# reset the name Instant Runoff
instant_runoff.set_name("Instant Runoff")

@vm(name = "Instant Runoff TB")
def instant_runoff_tb(profile, curr_cands = None, tie_breaker = None):
    """Instant Runoff (``instant_runoff``) with tie breaking:  If there is  more than one candidate with the fewest number of first-place votes, then remove the candidate with lowest in the tie_breaker ranking from the profile.

    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``
        tie_breaker (List[int]): A list of the candidates in the profile to be used as a tiebreaker.

    Returns: 
        A sorted list of candidates
   
    :Example: 

    .. exec_code:: 

        from pref_voting.profiles import Profile
        from pref_voting.iterative_methods import instant_runoff, instant_runoff_tb
        
        prof = Profile([[1, 2, 0], [2, 1, 0], [0, 1, 2]], [1, 1, 1])

        prof.display()
        print("no tiebreaker")
        instant_runoff.display(prof)
        print("tie_breaker = [0, 1, 2]")
        instant_runoff_tb.display(prof) 
        print("tie_breaker = [1, 2, 0]")
        instant_runoff_tb.display(prof, tie_breaker=[1, 2, 0])

    """

    # the tie_breaker is any linear order (i.e., list) of the candidates
    tb = tie_breaker if tie_breaker is not None else list(range(profile.num_cands))
    
    # need the total number of all candidates in a profile to check when all candidates have been removed   
    num_cands = profile.num_cands 
    
    candidates = profile.candidates if curr_cands is None else curr_cands
    cands_to_ignore = np.empty(0) if curr_cands is None else np.array([c for c in profile.candidates if c not in curr_cands])

    strict_maj_size = profile.strict_maj_size()
    
    rs, rcounts = profile.rankings_counts # get all the ranking data
    
    winners = [c for c in candidates 
               if _num_rank_first(rs, rcounts, cands_to_ignore, c) >= strict_maj_size]

    while len(winners) == 0:
        plurality_scores = {c: _num_rank_first(rs, rcounts, cands_to_ignore, c) for c in candidates if not isin(cands_to_ignore,c)}  
        min_plurality_score = min(plurality_scores.values())
        lowest_first_place_votes = np.array([c for c in plurality_scores.keys() 
                                             if plurality_scores[c] == min_plurality_score])
        
        cand_to_remove = lowest_first_place_votes[0]
        for c in lowest_first_place_votes[1:]: 
            if tb.index(c) < tb.index(cand_to_remove):
                cand_to_remove = c

        # remove cands with lowest plurality winners
        cands_to_ignore = np.concatenate((cands_to_ignore, cand_to_remove), axis=None)
        if len(cands_to_ignore) == num_cands: #all the candidates where removed
            winners = sorted(lowest_first_place_votes)
        else:
            winners = [c for c in candidates 
                       if not isin(cands_to_ignore,c) and _num_rank_first(rs, rcounts, cands_to_ignore, c) >= strict_maj_size]
     
    return sorted(winners)


@vm(name = "Instant Runoff PUT")
def instant_runoff_put(profile, curr_cands = None):
    """Instant Runoff (:fun:`instant_runoff`) with parallel universe tie-breaking (PUT).  Apply the Instant Runoff method with a tie-breaker for each possible linear order over the candidates. 
    
    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A sorted list of candidates

    .. warning:: 
        This will take a long time on profiles with many candidates. 

    :Example: 

    .. exec_code:: 

        from pref_voting.profiles import Profile
        from pref_voting.iterative_methods import instant_runoff, instant_runoff_tb, instant_runoff_put
        
        prof = Profile([[1, 2, 0], [2, 1, 0], [0, 1, 2]], [1, 1, 1])

        prof.display()
        print("no tiebreaker")
        instant_runoff.display(prof)
        print("tie_breaker = [0, 1, 2]")
        instant_runoff_tb.display(prof, tie_breaker=[0, 1, 2]) 
        print("tie_breaker = [0, 2, 1]")
        instant_runoff_tb.display(prof, tie_breaker=[0, 2, 1])
        print("tie_breaker = [1, 0, 2]")
        instant_runoff_tb.display(prof, tie_breaker=[1, 0, 2])
        print("tie_breaker = [1, 2, 0]")
        instant_runoff_tb.display(prof, tie_breaker=[1, 2, 0])
        print("tie_breaker = [2, 0, 1]")
        instant_runoff_tb.display(prof, tie_breaker=[2, 0, 1])
        print("tie_breaker = [2, 1, 0]")
        instant_runoff_tb.display(prof, tie_breaker=[2, 1, 0])
        print()
        instant_runoff_put.display(prof)

    """
    
    candidates = profile.candidates if curr_cands is None else curr_cands
    cands_to_ignore = np.empty(0) if curr_cands is None else np.array([c for c in profile.candidates if c not in curr_cands])
    
    strict_maj_size = profile.strict_maj_size()
    
    rs, rcounts = profile.rankings_counts # get all the ranking data
        
    winners = [c for c in candidates 
               if _num_rank_first(rs, rcounts, cands_to_ignore, c) >= strict_maj_size]

    if len(winners) == 0:
        # run Instant Runoff with tie-breaker for each permulation of candidates
        for tb in permutations(candidates):
            winners += instant_runoff_tb(profile, curr_cands = curr_cands, tie_breaker = tb) 
    return sorted(list(set(winners)))


# Create some aliases for instant runoff
instant_runoff_put.set_name("Hare PUT")
hare_put = copy.deepcopy(instant_runoff_put)
instant_runoff_put.set_name("Ranked Choice PUT")
ranked_choice_put = copy.deepcopy(instant_runoff_put)

# reset the name Instant Runoff
instant_runoff_put.set_name("Instant Runoff PUT")


def instant_runoff_with_explanation(profile, curr_cands = None):
    """
    Instant Runoff with an explanation. In addition to the winner(s), return the order in which the candidates are eliminated as a list of lists.    

    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A sorted list of candidates

        A list describing the order in which candidates are eliminated

    :Example: 

    .. exec_code:: 

        from pref_voting.profiles import Profile
        from pref_voting.iterative_methods import instant_runoff, instant_runoff_with_explanation


        prof = Profile([[2, 1, 0], [0, 2, 1], [1, 2, 0]], [1, 2, 2])
        prof.display()
        instant_runoff.display(prof)
        ws, exp = instant_runoff_with_explanation(prof)
        print(f"winning set: {ws}")
        print(f"order of elimination: {exp}")

        prof = Profile([[1, 2, 0], [2, 1, 0], [0, 1, 2]], [1, 1, 1])
        prof.display()
        instant_runoff.display(prof)
        ws, exp = instant_runoff_with_explanation(prof)
        print(f"winning set: {ws}")
        print(f"order of elimination: {exp}")

        prof = Profile([[2, 0, 1, 3], [2, 0, 3, 1], [3, 0, 1, 2],  [3, 2, 1, 0], [0, 2, 1, 3]], [1, 1, 1, 1, 1])
        prof.display()
        instant_runoff.display(prof)
        ws, exp = instant_runoff_with_explanation(prof)
        print(f"winning set: {ws}")
        print(f"order of elimination: {exp}")

    """
    # need the total number of all candidates in a profile to check when all candidates have been removed   
    num_cands = profile.num_cands 
    
    candidates = profile.candidates if curr_cands is None else curr_cands
    cands_to_ignore = np.empty(0) if curr_cands is None else np.array([c for c in profile.candidates if c not in curr_cands])

    strict_maj_size = profile.strict_maj_size()
    
    rs, rcounts = profile.rankings_counts # get all the ranking data
    

    winners = [c for c in candidates 
               if _num_rank_first(rs, rcounts, cands_to_ignore, c) >= strict_maj_size]
    elims_list = list()

    while len(winners) == 0:
        plurality_scores = {c: _num_rank_first(rs, rcounts, cands_to_ignore, c) for c in candidates 
                            if  not isin(cands_to_ignore,c)}  
        min_plurality_score = min(plurality_scores.values())
        lowest_first_place_votes = np.array([c for c in plurality_scores.keys() 
                                             if  plurality_scores[c] == min_plurality_score])

        elims_list.append(list(lowest_first_place_votes))

        # remove cands with lowest plurality winners
        cands_to_ignore = np.concatenate((cands_to_ignore, lowest_first_place_votes), axis=None)
        if len(cands_to_ignore) == num_cands: # removed all of the candidates 
            winners = sorted(lowest_first_place_votes)
        else:
            winners = [c for c in candidates 
                       if not isin(cands_to_ignore,c) and _num_rank_first(rs, rcounts, cands_to_ignore, c) >= strict_maj_size]
     
    return sorted(winners), elims_list


@vm(name="Instant Runoff")
def instant_runoff_for_truncated_linear_orders(profile, curr_cands = None, threshold = None, hide_warnings = False): 
    """
    Intant Runoff for Truncated Linear Orders.  Iteratively remove the candidates with the fewest number 
    of first place votes, until there is a candidate with more than the treshold number of first-place votes. 
    If a threshold is not set, then it is stirclty more than half of the non-empty ballots. 
    
    Args:
        profile (ProfileWithTies): An anonymous profile with no ties in the ballots (note that ProfileWithTies allows for truncated linear orders).
        threshold (int, float, optional): The threshold needed to win the election.  If it is not set, then it is striclty more than half of the remaining ballots.
        hide_warnings (bool, optional): Show or hide the warnings when more than one candidate is eleminated in a round.

    Returns: 
        A sorted list of candidates
    
    .. note:: This is the simultaneous version of instant runoff, not the parallel-universe tiebreaking version. 
    It is intended to be run on profiles with large number of voters in which there is a very low probability 
    of a tie in the fewest number of first place votes.   A warning is displayed when more than one candidate is
    eliminated. 

    :Example: 

    .. exec_code:: 

        from pref_voting.profiles_with_ties import ProfileWithTies
        from pref_voting.iterative_methods import instant_runoff_for_truncated_linear_orders

        prof = ProfileWithTies([{0:1, 1:1},{0:1, 1:2, 2:3, 3:4}, {0:1, 1:3, 2:3}, {3:2}, {0:1}, {0:1}, {}, {}])
        prof.display()

        tprof, report = prof.truncate_overvotes()
        for r, new_r, count in report: 
            print(f"{r} --> {new_r}: {count}")
        tprof.display()
        instant_runoff_for_truncated_linear_orders.display(tprof)
    
    """
    
    assert all([not r.has_overvote() for r in profile.rankings]), "Instant Runoff is only defined when all the ballots are truncated linear orders."
    
    curr_cands = profile.candidates if curr_cands is None else curr_cands

    # we need to remove empty rankings during the algorithm, so make a copy of the profile
    prof2 = copy.deepcopy(profile) 
    
    _prof = prof2.remove_candidates([c for c in profile.candidates if c not in curr_cands])

    # remove the empty rankings
    _prof.remove_empty_rankings()
    
    threshold = threshold if threshold is not None else _prof.strict_maj_size()
    
    remaining_candidates = _prof.candidates
        
    pl_scores = _prof.plurality_scores()
    max_pl_score = max(pl_scores.values())
    
    while max_pl_score < threshold: 

        reduced_prof = _prof.remove_candidates([c for c in _prof.candidates if c not in remaining_candidates])
        
        # after removing the candidates, there might be some empty ballots.
        reduced_prof.remove_empty_rankings()

        pl_scores = reduced_prof.plurality_scores()
        min_pl_score = min(pl_scores.values())
            
        cands_to_remove = [c for c in pl_scores.keys() if pl_scores[c] == min_pl_score]

        if not hide_warnings and len(cands_to_remove) > 1: 
            print(f"Warning: multiple candidates removed in a round: {', '.join(map(str,cands_to_remove))}")
            
        if len(cands_to_remove) == len(reduced_prof.candidates): 
            # all remaining candidates have the same plurality score.
            break 
            
        # possibly update the threshold, so that it is a strict majority of the remaining ballots
        threshold = threshold if threshold is not None else reduced_prof.strict_maj_size()
        max_pl_score = max(pl_scores.values())

        remaining_candidates = [c for c in remaining_candidates if c not in cands_to_remove]


    reduced_prof = _prof.remove_candidates([c for c in _prof.candidates if c not in remaining_candidates])

    # after removing the candidates, there might be some empty ballots.
    reduced_prof.remove_empty_rankings()
        
    pl_scores = reduced_prof.plurality_scores()
    
    max_pl_score = max(pl_scores.values())
    
    return sorted([c for c in pl_scores.keys() if pl_scores[c] == max_pl_score])
    
@vm(name = "PluralityWRunoff")
def plurality_with_runoff(profile, curr_cands = None):
    """If there is a majority winner then that candidate is the plurality with runoff winner. If there is no majority winner, then hold a runoff with  the top two candidates: either two (or more candidates) with the most first place votes or the candidate with the most first place votes and the candidate with the 2nd highest first place votes are ranked first by the fewest number of voters.   A candidate is a Plurality with Runoff winner in the profile restricted to ``curr_cands`` if it is a winner in a runoff between two pairs of first- or second- ranked candidates. If the candidates are all tied for the most first place votes, then all candidates are winners.
        
    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A sorted list of candidates
        
    .. note:: 
        Plurality with Runoff is the same as Instant Runoff when there are 3 candidates, but give different answers with 4 or more candidates. 

    :Example: 

    .. exec_code:: 

        from pref_voting.profiles import Profile
        from pref_voting.iterative_methods import instant_runoff, plurality_with_runoff

        prof = Profile([[0, 1, 2, 3], [3, 1, 2, 0], [2, 0, 3, 1], [1, 2, 3, 0], [2, 3, 0, 1], [0, 3, 2, 1]], [2, 1, 2, 2, 1, 2])
        prof.display()
        instant_runoff.display(prof)
        plurality_with_runoff.display(prof)
    
    """    

    curr_cands = profile.candidates if curr_cands is None else curr_cands
    
    if len(curr_cands) == 1: 
        return list(curr_cands)
    
    plurality_scores = profile.plurality_scores(curr_cands = curr_cands)  

    max_plurality_score = max(plurality_scores.values())
    
    first = [c for c in curr_cands if plurality_scores[c] == max_plurality_score]
    second = list()
    if len(first) == 1:
        second_plurality_score = list(reversed(sorted(plurality_scores.values())))[1]
        second = [c for c in curr_cands if plurality_scores[c] == second_plurality_score]

    if len(second) > 0:
        all_runoff_pairs = product(first, second)
    else: 
        all_runoff_pairs = [(c1,c2) for c1,c2 in product(first, first) if c1 != c2]

    winners = list()
    for c1, c2 in all_runoff_pairs: 
        
        if profile.margin(c1,c2) > 0:
            winners.append(c1)
        elif profile.margin(c1,c2) < 0:
            winners.append(c2)
        elif profile.margin(c1,c2) == 0:
            winners.append(c1)
            winners.append(c2)
    
    return sorted(list(set(winners)))


@vm(name = "Coombs")
def coombs(profile, curr_cands = None):
    """If there is a majority winner then that candidate is the Coombs winner.     If there is no majority winner, then remove all candidates that are ranked last by the greatest number of voters.  Continue removing candidates with the most last-place votes until there is a candidate with a majority of first place votes.  
    
    .. important::
        If there is  more than one candidate with the largest number of last-place votes, then *all* such candidates are removed from the profile. 
    
    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A sorted list of candidates

    .. seealso::

        :func:`pref_voting.iterative_methods.coombs_with_tb`, :func:`pref_voting.iterative_methods.coomb_put`, :func:`pref_voting.iterative_methods.coombs_with_explanation`
   
    :Example: 

    .. exec_code:: 

        from pref_voting.profiles import Profile
        from pref_voting.iterative_methods import instant_runoff, coombs
        
        prof = Profile([[2, 1, 0], [0, 2, 1], [1, 2, 0]], [1, 2, 2])

        prof.display()
        coombs.display(prof)
        instant_runoff.display(prof)

    """   

    # need the total number of all candidates in a profile to check when all candidates have been removed   
    num_cands = profile.num_cands 
    
    candidates = profile.candidates if curr_cands is None else curr_cands
    cands_to_ignore = np.empty(0) if curr_cands is None else np.array([c for c in profile.candidates if c not in curr_cands])

    strict_maj_size = profile.strict_maj_size()
    
    rs, rcounts = profile.rankings_counts # get all the ranking data
    
    winners = [c for c in candidates 
               if _num_rank_first(rs, rcounts, cands_to_ignore, c) >= strict_maj_size]

    while len(winners) == 0:
        
        last_place_scores = {c: _num_rank_last(rs, rcounts, cands_to_ignore, c) for c in candidates 
                             if not isin(cands_to_ignore,c)}  
        max_last_place_score = max(last_place_scores.values())
        greatest_last_place_votes = np.array([c for c in last_place_scores.keys() 
                                              if  last_place_scores[c] == max_last_place_score])

        # remove candidates ranked last by the greatest number of voters
        cands_to_ignore = np.concatenate((cands_to_ignore, greatest_last_place_votes), axis=None)
        
        if len(cands_to_ignore) == num_cands: # removed all candidates 
            winners = list(greatest_last_place_votes)
        else:
            winners = [c for c in candidates 
                       if not isin(cands_to_ignore,c) and _num_rank_first(rs, rcounts, cands_to_ignore, c) >= strict_maj_size]

    return sorted(winners)


@vm(name = "Coombs TB")
def coombs_tb(profile, curr_cands = None, tie_breaker=None):
    """
    Coombs with a fixed tie-breaking rule: The tie-breaking rule is any linear order (i.e., list) of the candidates.  The default rule is to order the candidates as follows: 0,....,num_cands-1.

    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``
        tie_breaker (List[int]): A list of the candidates in the profile to be used as a tiebreaker.

    Returns: 

        A sorted list of candidates

    :Example: 

    .. exec_code:: 

        from pref_voting.profiles import Profile
        from pref_voting.iterative_methods import coombs, coombs_tb
        
        prof = Profile([[2, 0, 1], [0, 2, 1], [1, 0, 2], [2, 1, 0], [0, 1, 2]], [1, 1, 1, 2, 1])
        prof.display()
        print("no tiebreaker")
        coombs.display(prof)
        print("tie_breaker = [0, 1, 2]")
        coombs_tb.display(prof) 
        print("tie_breaker = [2, 1, 0]")
        coombs_tb.display(prof, tie_breaker=[2, 1, 0])

    """

    # the tie_breaker is any linear order (i.e., list) of the candidates
    tb = tie_breaker if tie_breaker is not None else list(range(profile.num_cands))

    num_cands = profile.num_cands   
    candidates = profile.candidates if curr_cands is None else curr_cands
    cands_to_ignore = np.empty(0) if curr_cands is None else np.array([c for c in profile.candidates if c not in curr_cands])

    strict_maj_size = profile.strict_maj_size()
    
    rs, rcounts = profile.rankings_counts # get all the ranking data

    winners = [c for c in candidates 
               if _num_rank_first(rs, rcounts, cands_to_ignore, c) >= strict_maj_size]

    while len(winners) == 0:
        
        last_place_scores = {c: _num_rank_last(rs, rcounts, cands_to_ignore, c) for c in candidates 
                             if not isin(cands_to_ignore,c)}  
        max_last_place_score = max(last_place_scores.values())
        greatest_last_place_votes = [c for c in last_place_scores.keys() if last_place_scores[c] == max_last_place_score]

        # select the candidate to remove using the tie-breaking rule (a linear order over the candidates)
        cand_to_remove = greatest_last_place_votes[0]
        for c in greatest_last_place_votes[1:]: 
            if tb.index(c) < tb.index(cand_to_remove):
                cand_to_remove = c
        
        # remove candidates ranked last by the greatest number of voters
        cands_to_ignore = np.concatenate((cands_to_ignore, np.array([cand_to_remove])), axis=None)
        
        if len(cands_to_ignore) == num_cands:
            winners = list(greatest_last_place_votes)
        else:
            winners = [c for c in candidates 
                       if not isin(cands_to_ignore,c) and _num_rank_first(rs, rcounts, cands_to_ignore, c) >= strict_maj_size]

    return sorted(winners)

@vm(name = "Coombs PUT")
def coombs_put(profile, curr_cands = None):
    """Coombs with parallel universe tie-breaking (PUT).  Apply the Coombs method with a tie-breaker for each possible linear order over the candidates. 
    
    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A sorted list of candidates

    .. warning:: 
        This will take a long time on profiles with many candidates. 

    :Example: 

    .. exec_code:: 

        from pref_voting.profiles import Profile
        from pref_voting.iterative_methods import coombs, coombs_tb, coombs_put
        
        prof = Profile([[2, 0, 1], [1, 0, 2], [0, 1, 2]], [2, 1, 1])

        prof.display()
        print("no tiebreaker")
        coombs.display(prof)
        print("tie_breaker = [0, 1, 2]")
        coombs_tb.display(prof, tie_breaker=[0, 1, 2]) 
        print("tie_breaker = [0, 2, 1]")
        coombs_tb.display(prof, tie_breaker=[0, 2, 1])
        print("tie_breaker = [1, 0, 2]")
        coombs_tb.display(prof, tie_breaker=[1, 0, 2])
        print("tie_breaker = [1, 2, 0]")
        coombs_tb.display(prof, tie_breaker=[1, 2, 0])
        print("tie_breaker = [2, 0, 1]")
        coombs_tb.display(prof, tie_breaker=[2, 0, 1])
        print("tie_breaker = [2, 1, 0]")
        coombs_tb.display(prof, tie_breaker=[2, 1, 0])
        print()
        coombs_put.display(prof)
    """

    candidates = profile.candidates if curr_cands is None else curr_cands

    strict_maj_size = profile.strict_maj_size()
    
    rs, rcounts = profile.rankings_counts # get all the ranking data
    
    winners = [c for c in candidates 
               if _num_rank_first(rs, rcounts, np.empty(0), c) >= strict_maj_size]

    if len(winners) == 0:
        # run Coombs with tie-breaker for each permulation of candidates
        for tb in permutations(candidates):
            winners += coombs_tb(profile, curr_cands = curr_cands, tie_breaker = tb) 

    return sorted(list(set(winners)))

def coombs_with_explanation(profile, curr_cands = None):
    """
    Coombs with an explanation. In addition to the winner(s), return the order in which the candidates are eliminated as a list of lists.    

    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A sorted list of candidates

        A list describing the order in which candidates are eliminated

    .. exec_code:: 

        from pref_voting.profiles import Profile
        from pref_voting.iterative_methods import coombs, coombs_with_explanation


        prof = Profile([[2, 1, 0], [0, 2, 1], [1, 2, 0]], [1, 2, 2])
        prof.display()
        coombs.display(prof)
        ws, exp = coombs_with_explanation(prof)
        print(f"winning set: {ws}")
        print(f"order of elimination: {exp}")

        prof = Profile([[1, 0, 3, 2], [2, 3, 1, 0], [2, 0, 3, 1], [1, 2, 3, 0]], [1, 1, 1, 1])
        prof.display()
        coombs.display(prof)
        ws, exp = coombs_with_explanation(prof)
        print(f"winning set: {ws}")
        print(f"order of elimination: {exp}")


    """
    num_cands = profile.num_cands 
    
    candidates = profile.candidates if curr_cands is None else curr_cands
    cands_to_ignore = np.empty(0) if curr_cands is None else np.array([c for c in profile.candidates if c not in curr_cands])

    strict_maj_size = profile.strict_maj_size()
    
    rs, rcounts = profile.rankings_counts # get all the ranking data
    
    winners = [c for c in candidates 
               if _num_rank_first(rs, rcounts, cands_to_ignore, c) >= strict_maj_size]

    elims_list = list()
    while len(winners) == 0:
        
        last_place_scores = {c: _num_rank_last(rs, rcounts, cands_to_ignore, c) for c in candidates 
                             if not isin(cands_to_ignore,c)}  
        max_last_place_score = max(last_place_scores.values())
        greatest_last_place_votes = np.array([c for c in last_place_scores.keys() 
                                              if  last_place_scores[c] == max_last_place_score])

        elims_list.append(list(greatest_last_place_votes))
        # remove candidates ranked last by the greatest number of voters
        cands_to_ignore = np.concatenate((cands_to_ignore, greatest_last_place_votes), axis=None)
        
        if len(cands_to_ignore) == num_cands:
            winners = list(greatest_last_place_votes)
        else:
            winners = [c for c in candidates 
                       if not isin(cands_to_ignore,c) and _num_rank_first(rs, rcounts, cands_to_ignore, c) >= strict_maj_size]

    return sorted(winners), elims_list

@vm(name = "Baldwin")
def baldwin(profile, curr_cands = None):
    """Iteratively remove all candidates with the lowest Borda score until a single candidate remains.  If, at any stage, all  candidates have the same Borda score,  then all (remaining) candidates are winners.

    .. note:: 
        Baldwin is a Condorcet consistent voting method means that if a Condorcet winner exists, then Baldwin will elect the Condorcet winner. 

    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A sorted list of candidates

    .. seealso::

        :func:`pref_voting.iterative_methods.baldwin_with_tb`, :func:`pref_voting.iterative_methods.baldwin`, :func:`pref_voting.iterative_methods.baldwin_with_explanation`
   
    :Example: 

    .. exec_code:: 

        from pref_voting.profiles import Profile
        from pref_voting.iterative_methods import baldwin
        
        prof = Profile([[1, 0, 2, 3], [3, 1, 0, 2], [2, 0, 3, 1]], [2, 1, 1])

        prof.display()
        baldwin.display(prof)
    """
    all_num_cands = profile.num_cands   
    candidates = profile.candidates if curr_cands is None else curr_cands
        
    rcounts = profile._rcounts # get all the ranking data
    rankings = profile._rankings if curr_cands is None else _find_updated_profile(profile._rankings, np.array([c for c in profile.candidates if c not in curr_cands]), all_num_cands)
    
    cands_to_ignore = np.empty(0) if curr_cands is None else np.array([c for c in profile.candidates if c not in curr_cands])

    borda_scores = {c: _borda_score(rankings, rcounts, len(candidates), c) for c in candidates}

    min_borda_score = min(list(borda_scores.values()))
    
    last_place_borda_scores = [c for c in candidates 
                               if c in borda_scores.keys() and borda_scores[c] == min_borda_score]
      
    cands_to_ignore = np.concatenate((cands_to_ignore, last_place_borda_scores), axis=None)
    
    winners = list()
    if cands_to_ignore.shape[0] ==  all_num_cands: # all candidates have lowest Borda score
        winners = sorted(last_place_borda_scores)
    else: # remove the candidates with lowest Borda score
        num_cands = len(candidates)
        updated_rankings = _find_updated_profile(rankings, cands_to_ignore, num_cands)
        
    while len(winners) == 0:
        borda_scores = {c: _borda_score(updated_rankings, rcounts, num_cands - cands_to_ignore.shape[0], c) for c in candidates if not isin(cands_to_ignore, c)}
                
        min_borda_score = min(borda_scores.values())
        last_place_borda_scores = [c for c in borda_scores.keys() if borda_scores[c] == min_borda_score]
        
        cands_to_ignore = np.concatenate((cands_to_ignore, last_place_borda_scores), axis=None)
                
        if cands_to_ignore.shape[0] == all_num_cands: # removed all remaining candidates
            winners = sorted(last_place_borda_scores)
        elif num_cands - cands_to_ignore.shape[0] ==  1: # only one candidate remains
            winners = sorted([c for c in candidates if c not in cands_to_ignore])
        else: 
            updated_rankings = _find_updated_profile(rankings, cands_to_ignore, num_cands)
    return sorted(winners)

@vm(name = "Baldwin TB")
def baldwin_tb(profile, curr_cands = None, tie_breaker=None):
    """
    Baldwin with a fixed tie-breaking rule: The tie-breaking rule is any linear order (i.e., list) of the candidates.  The default rule is to order the candidates as follows: 0,....,num_cands-1.

    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``
        tie_breaker (List[int]): A list of the candidates in the profile to be used as a tiebreaker.

    Returns: 

        A sorted list of candidates

    :Example: 

    .. exec_code:: 

        from pref_voting.profiles import Profile
        from pref_voting.iterative_methods import baldwin, baldwin_tb
        
        prof = Profile([[0, 2, 1, 3], [1, 3, 0, 2], [2, 3, 0, 1]], [1, 1, 1])
        prof.display()
        print("no tiebreaker")
        baldwin.display(prof)
        print("tie_breaker = [0, 1, 2, 3]")
        baldwin_tb.display(prof) 
        print("tie_breaker = [2, 1, 0, 3]")
        baldwin_tb.display(prof, tie_breaker=[2, 1, 0, 3])

    """
    
    # the tie_breaker is any linear order (i.e., list) of the candidates
    tb = tie_breaker if tie_breaker is not None else list(range(profile.num_cands))

    all_num_cands = profile.num_cands   
    candidates = profile.candidates if curr_cands is None else curr_cands
        
    rcounts = profile._rcounts # get all the ranking data
    rankings = profile._rankings if curr_cands is None else _find_updated_profile(profile._rankings, np.array([c for c in profile.candidates if c not in curr_cands]), all_num_cands)
    
    cands_to_ignore = np.empty(0) if curr_cands is None else np.array([c for c in profile.candidates if c not in curr_cands])

    borda_scores = {c: _borda_score(rankings, rcounts, len(candidates), c) for c in candidates}

    min_borda_score = min(list(borda_scores.values()))
    
    last_place_borda_scores = [c for c in candidates 
                               if c in borda_scores.keys() and borda_scores[c] == min_borda_score]
      
    cand_to_remove = last_place_borda_scores[0]
    for c in last_place_borda_scores[1:]: 
        if tb.index(c) < tb.index(cand_to_remove):
            cand_to_remove = c
    cands_to_ignore = np.concatenate((cands_to_ignore, np.array([cand_to_remove])), axis=None)
    
    winners = list()
    if cands_to_ignore.shape[0] ==  all_num_cands: # all candidates have lowest Borda score
        winners = sorted(last_place_borda_scores)
    else: # remove the candidates with lowest Borda score
        num_cands = len(candidates)
        updated_rankings = _find_updated_profile(rankings, cands_to_ignore, num_cands)
        
    while len(winners) == 0:
        borda_scores = {c: _borda_score(updated_rankings, rcounts, num_cands - cands_to_ignore.shape[0], c) 
                        for c in candidates if not isin(cands_to_ignore, c)}
                
        min_borda_score = min(borda_scores.values())
        last_place_borda_scores = [c for c in borda_scores.keys() if borda_scores[c] == min_borda_score]
        
        # select the candidate to remove using the tie-breaking rule (a linear order over the candidates)
        cand_to_remove = last_place_borda_scores[0]
        for c in last_place_borda_scores[1:]: 
            if tb.index(c) < tb.index(cand_to_remove):
                cand_to_remove = c
        
        cands_to_ignore = np.concatenate((cands_to_ignore, np.array([cand_to_remove])), axis=None)
                
        if cands_to_ignore.shape[0] == all_num_cands: # removed all remaining candidates
            winners = sorted(last_place_borda_scores)
        elif num_cands - cands_to_ignore.shape[0] ==  1: # only one candidate remains
            winners = sorted([c for c in candidates if c not in cands_to_ignore])
        else: 
            updated_rankings = _find_updated_profile(rankings, cands_to_ignore, num_cands)
    return sorted(winners)

@vm(name = "Baldwin PUT")
def baldwin_put(profile, curr_cands=None):
    """Baldwin with parallel universe tie-breaking (PUT).  Apply the Baldwin method with a tie-breaker for each possible linear order over the candidates. 
    
    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A sorted list of candidates

    .. warning:: 
        This will take a long time on profiles with many candidates. 

    :Example: 

    .. exec_code:: 

        from pref_voting.profiles import Profile
        from pref_voting.iterative_methods import baldwin, baldwin_tb, baldwin_put
        
        prof = Profile([[1, 2, 0], [0, 1, 2], [2, 0, 1]], [1, 3, 2])

        prof.display()
        print("no tiebreaker")
        baldwin.display(prof)
        print("tie_breaker = [0, 1, 2]")
        baldwin_tb.display(prof, tie_breaker=[0, 1, 2]) 
        print("tie_breaker = [0, 2, 1]")
        baldwin_tb.display(prof, tie_breaker=[0, 2, 1])
        print("tie_breaker = [1, 0, 2]")
        baldwin_tb.display(prof, tie_breaker=[1, 0, 2])
        print("tie_breaker = [1, 2, 0]")
        baldwin_tb.display(prof, tie_breaker=[1, 2, 0])
        print("tie_breaker = [2, 0, 1]")
        baldwin_tb.display(prof, tie_breaker=[2, 0, 1])
        print("tie_breaker = [2, 1, 0]")
        baldwin_tb.display(prof, tie_breaker=[2, 1, 0])
        print()
        baldwin_put.display(prof)
    """

    candidates = profile.candidates if curr_cands is None else curr_cands    
    cw = profile.condorcet_winner(curr_cands=curr_cands)
    
    winners = list() if cw is None else [cw]

    if len(winners) == 0:
        # run Coombs with tie-breaker for each permulation of candidates
        for tb in permutations(candidates):
            winners += baldwin_tb(profile, curr_cands = curr_cands, tie_breaker = tb) 

    return sorted(list(set(winners)))


def baldwin_with_explanation(profile, curr_cands = None):
    """Baldwin with an explanation. In addition to the winner(s), return the order in which the candidates are eliminated as a list of dictionaries specifying the Borda scores in the profile resctricted to the candidates that have not been eliminated.    

    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A sorted list of candidates

        A list describing for each round, the candidates that are eliminated and the Borda scores of the remaining candidates (in the profile restricted to candidates that have not been eliminated)

    .. exec_code:: 

        from pref_voting.profiles import Profile
        from pref_voting.iterative_methods import baldwin, baldwin_with_explanation

        prof = Profile([[2, 1, 0], [0, 2, 1], [1, 2, 0]], [1, 2, 2])
        prof.display()
        baldwin.display(prof)
        ws, exp = baldwin_with_explanation(prof)
        print(f"winning set: {ws}")
        print(f"order of elimination: {exp}")

        prof = Profile([[1, 0, 3, 2], [2, 3, 1, 0], [2, 0, 3, 1], [1, 2, 3, 0]], [1, 1, 1, 1])
        prof.display()
        baldwin.display(prof)
        ws, exp = baldwin_with_explanation(prof)
        print(f"winning set: {ws}")
        print(f"order of elimination: {exp}")


    """

    all_num_cands = profile.num_cands   
    candidates = profile.candidates if curr_cands is None else curr_cands
    elims_list = list()
        
    rcounts = profile._rcounts # get all the ranking data
    rankings = profile._rankings if curr_cands is None else _find_updated_profile(profile._rankings, np.array([c for c in profile.candidates if c not in curr_cands]), all_num_cands)
    
    cands_to_ignore = np.empty(0) if curr_cands is None else np.array([c for c in profile.candidates if c not in curr_cands])

    borda_scores = {c: _borda_score(rankings, rcounts, len(candidates), c) for c in candidates}

    min_borda_score = min(list(borda_scores.values()))
    
    last_place_borda_scores = [c for c in candidates 
                               if c in borda_scores.keys() and borda_scores[c] == min_borda_score]
    elims_list.append([last_place_borda_scores, borda_scores])
    cands_to_ignore = np.concatenate((cands_to_ignore, np.array(last_place_borda_scores)), axis=None)
    
    winners = list()
    if cands_to_ignore.shape[0] ==  all_num_cands: # all candidates have lowest Borda score
        winners = sorted(last_place_borda_scores)
    else: # remove the candidates with lowest Borda score
        num_cands = len(candidates)
        updated_rankings = _find_updated_profile(rankings, cands_to_ignore, num_cands)
        
    while len(winners) == 0:
        borda_scores = {c: _borda_score(updated_rankings, rcounts, num_cands - cands_to_ignore.shape[0], c) 
                        for c in candidates if not isin(cands_to_ignore, c)}
                
        min_borda_score = min(borda_scores.values())
        last_place_borda_scores = [c for c in borda_scores.keys() if borda_scores[c] == min_borda_score]
        elims_list.append([last_place_borda_scores, borda_scores])
        
        cands_to_ignore = np.concatenate((cands_to_ignore, np.array(last_place_borda_scores)), axis=None)
                
        if cands_to_ignore.shape[0] == all_num_cands: # removed all remaining candidates
            winners = sorted(last_place_borda_scores)
        elif num_cands - cands_to_ignore.shape[0] ==  1: # only one candidate remains
            winners = sorted([c for c in candidates if c not in cands_to_ignore])
        else: 
            updated_rankings = _find_updated_profile(rankings, cands_to_ignore, num_cands)
    return sorted(winners), elims_list


@vm(name = "Strict Nanson")
def strict_nanson(profile, curr_cands = None):
    """Iteratively remove all candidates with the  Borda score strictly below the average Borda score until one candidate remains.  If, at any stage, all  candidates have the same Borda score, then all (remaining) candidates are winners.

    .. note:: 
        
        Strict Nanson is a Condorcet consistent voting method means that if a Condorcet winner exists, then Strict Nanson will elect the Condorcet winner. 

    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A sorted list of candidates

    .. seealso::

        :func:`pref_voting.iterative_methods.strict_nanson_with_explanation`, :func:`pref_voting.iterative_methods.weak_nanson`
   
    :Example: 

    .. exec_code:: 

        from pref_voting.profiles import Profile
        from pref_voting.iterative_methods import strict_nanson
        
        prof = Profile([[2, 1, 0, 3], [0, 2, 1, 3], [1, 3, 0, 2], [0, 3, 2, 1]], [2, 1, 1, 1])

        prof.display()
        strict_nanson.display(prof)
    """
    
    all_num_cands = profile.num_cands   
    candidates = profile.candidates if curr_cands is None else curr_cands
        
    rcounts = profile._rcounts # get all the ranking data
    rankings = profile._rankings if curr_cands is None else _find_updated_profile(profile._rankings, np.array([c for c in profile.candidates if c not in curr_cands]), all_num_cands)
    cands_to_ignore = np.empty(0)

    borda_scores = {c: _borda_score(rankings, rcounts, len(candidates), c) for c in candidates}
    
    avg_borda_score = np.mean(list(borda_scores.values()))
    below_borda_avg_candidates = np.array([c for c in borda_scores.keys() if borda_scores[c] < avg_borda_score])
    
    cands_to_ignore = np.concatenate((cands_to_ignore, below_borda_avg_candidates), axis=None)
    winners = list()
    if cands_to_ignore.shape[0] == all_num_cands:  # all candidates have same Borda score
        winners = sorted(candidates)
    else: 
        num_cands = len(candidates)
        updated_rankings = _find_updated_profile(rankings, cands_to_ignore, num_cands)
    while len(winners) == 0: 
        
        borda_scores = {c: _borda_score(updated_rankings, rcounts, num_cands - cands_to_ignore.shape[0], c) 
                        for c in candidates if not isin(cands_to_ignore, c)}
        
        avg_borda_scores = np.mean(list(borda_scores.values()))
    
        below_borda_avg_candidates = np.array([c for c in borda_scores.keys() 
                                               if borda_scores[c] < avg_borda_scores])
        
        cands_to_ignore = np.concatenate((cands_to_ignore, below_borda_avg_candidates), axis=None)
                
        if (below_borda_avg_candidates.shape[0] == 0) or ((all_num_cands - cands_to_ignore.shape[0]) == 1):
            winners = sorted([c for c in candidates if c not in cands_to_ignore])
        else:
            updated_rankings = _find_updated_profile(rankings, cands_to_ignore, num_cands)
            
    return winners


def strict_nanson_with_explanation(profile, curr_cands = None):
    """Strict Nanson with an explanation. In addition to the winner(s), return the order in which the candidates are eliminated as a list of dictionaries specifying the Borda scores in the profile resctricted to the candidates that have not been eliminated and the average Borda score.    
    
    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A sorted list of candidates

        A list describing for each round, the candidates that are eliminated and the Borda scores of the remaining candidates (in the profile restricted to candidates that have not been eliminated)
    :Example: 

    .. exec_code:: 

        from pref_voting.profiles import Profile
        from pref_voting.iterative_methods import strict_nanson_with_explanation
        
        prof = Profile([[2, 1, 0, 3], [0, 2, 1, 3], [1, 3, 0, 2], [0, 3, 2, 1]], [2, 1, 1, 1])

        prof.display()
        print(strict_nanson_with_explanation(prof))
    """

    all_num_cands = profile.num_cands   
    candidates = profile.candidates if curr_cands is None else curr_cands
    
    rcounts = profile._rcounts # get all the ranking data
    rankings = profile._rankings if curr_cands is None else _find_updated_profile(profile._rankings, np.array([c for c in profile.candidates if c not in curr_cands]), all_num_cands)
    cands_to_ignore = np.empty(0)
    elim_list = list()
    
    borda_scores = {c: _borda_score(rankings, rcounts, len(candidates), c) for c in candidates}
    
    avg_borda_score = np.mean(list(borda_scores.values()))
    below_borda_avg_candidates = [c for c in borda_scores.keys() if borda_scores[c] < avg_borda_score]
    
    cands_to_ignore = np.concatenate((cands_to_ignore, np.array(below_borda_avg_candidates)), axis=None)
    winners = list()
    if cands_to_ignore.shape[0] == all_num_cands:  # all candidates have same Borda score
        elim_list.append({"avg_borda_score": avg_borda_score, 
                          "elim_cands": below_borda_avg_candidates,
                          "borda_scores": borda_scores})
        winners = sorted(candidates)
    else: 
        num_cands = len(candidates)
        elim_list.append({"avg_borda_score": avg_borda_score, 
                          "elim_cands": below_borda_avg_candidates,
                          "borda_scores": borda_scores})
        updated_rankings = _find_updated_profile(rankings, cands_to_ignore, num_cands)
    while len(winners) == 0: 
        
        borda_scores = {c: _borda_score(updated_rankings, rcounts, num_cands - cands_to_ignore.shape[0], c) 
                        for c in candidates if not isin(cands_to_ignore, c)}
        
        avg_borda_score = np.mean(list(borda_scores.values()))
    
        below_borda_avg_candidates = [c for c in borda_scores.keys() 
                                      if borda_scores[c] < avg_borda_score]
        
        cands_to_ignore = np.concatenate((cands_to_ignore, np.array(below_borda_avg_candidates)), axis=None)
        elim_list.append({"avg_borda_score": avg_borda_score, 
                          "elim_cands": below_borda_avg_candidates,
                          "borda_scores": borda_scores})
                
        if (len(below_borda_avg_candidates) == 0) or ((all_num_cands - cands_to_ignore.shape[0]) == 1):
            winners = sorted([c for c in candidates if c not in cands_to_ignore])
        else:
            updated_rankings = _find_updated_profile(rankings, cands_to_ignore, num_cands)
            
    return winners, elim_list


@vm(name = "Weak Nanson")
def weak_nanson(profile, curr_cands = None):
    """Iteratively remove all candidates with Borda score less than or equal the average Borda score until one candidate remains.  If, at any stage, all  candidates have the same Borda score, then all (remaining) candidates are winners.

    .. note:: 

        Weak Nanson is a Condorcet consistent voting method means that if a Condorcet winner exists, then Weak Nanson will elect the Condorcet winner. 

    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A sorted list of candidates

    .. seealso::

        :func:`pref_voting.iterative_methods.weak_nanson_with_explanation`, :func:`pref_voting.iterative_methods.strict_nanson`
   
    :Example: 

    .. exec_code:: 

        from pref_voting.profiles import Profile
        from pref_voting.iterative_methods import weak_nanson
        
        prof = Profile([[2, 1, 0, 3], [0, 2, 1, 3], [1, 3, 0, 2], [0, 3, 2, 1]], [2, 1, 1, 1])

        prof.display()
        weak_nanson.display(prof)

    """

    all_num_cands = profile.num_cands   
    candidates = profile.candidates if curr_cands is None else curr_cands
        
    rcounts = profile._rcounts # get all the ranking data
    rankings = profile._rankings if curr_cands is None else _find_updated_profile(profile._rankings, np.array([c for c in profile.candidates if c not in curr_cands]), all_num_cands)
    
    cands_to_ignore = np.empty(0) if curr_cands is None else np.array([c for c in profile.candidates if c not in curr_cands])

    borda_scores = {c: _borda_score(rankings, rcounts, len(candidates), c) for c in candidates}

    avg_borda_score = np.mean(list(borda_scores.values()))

    below_borda_avg_candidates = np.array([c for c in borda_scores.keys() if borda_scores[c] <= avg_borda_score])
    
    cands_to_ignore = np.concatenate((cands_to_ignore, below_borda_avg_candidates), axis=None)
    
    winners = list()
    if cands_to_ignore.shape[0] == all_num_cands:  # all candidates have same Borda score
        winners = sorted(candidates)
    elif all_num_cands - cands_to_ignore.shape[0]  == 1: # one candidate remains
        winners = [c for c in candidates if not isin(cands_to_ignore, c)]
    else: 
        num_cands = len(candidates)
        updated_rankings = _find_updated_profile(rankings, cands_to_ignore, num_cands)
        
    while len(winners) == 0: 
        
        borda_scores = {c: _borda_score(updated_rankings, rcounts, num_cands - cands_to_ignore.shape[0], c) 
                        for c in candidates if not isin(cands_to_ignore, c)}
        

        avg_borda_score = np.mean(list(borda_scores.values()))

        below_borda_avg_candidates = np.array([c for c in borda_scores.keys() 
                                               if borda_scores[c] <= avg_borda_score])
        
        cands_to_ignore = np.concatenate((cands_to_ignore, below_borda_avg_candidates), axis=None)
        
        if cands_to_ignore.shape[0] == all_num_cands:  # all remaining candidates have been removed
            winners = sorted(below_borda_avg_candidates)
        elif all_num_cands - cands_to_ignore.shape[0]  == 1: # one candidate remains
            winners = [c for c in candidates if not isin(cands_to_ignore, c)]
        else:
            updated_rankings = _find_updated_profile(rankings, cands_to_ignore, num_cands)
            
    return winners


def weak_nanson_with_explanation(profile, curr_cands = None):
    """
    Weak Nanson with an explanation. In addition to the winner(s), return the order in which the candidates are eliminated as a list of dictionaries specifying the Borda scores in the profile resctricted to the candidates that have not been eliminated and the average Borda score.    
    
    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A sorted list of candidates

        A list describing for each round, the candidates that are eliminated and the Borda scores of the remaining candidates (in the profile restricted to candidates that have not been eliminated)
    :Example: 

    .. exec_code:: 

        from pref_voting.profiles import Profile
        from pref_voting.iterative_methods import weak_nanson_with_explanation
        
        prof = Profile([[2, 1, 0, 3], [0, 2, 1, 3], [1, 3, 0, 2], [0, 3, 2, 1]], [2, 1, 1, 1])

        prof.display()
        print(weak_nanson_with_explanation(prof))
        
    """
    all_num_cands = profile.num_cands   
    candidates = profile.candidates if curr_cands is None else curr_cands
        
    rcounts = profile._rcounts # get all the ranking data
    rankings = profile._rankings if curr_cands is None else _find_updated_profile(profile._rankings, np.array([c for c in profile.candidates if c not in curr_cands]), all_num_cands)
    cands_to_ignore = np.empty(0) if curr_cands is None else np.array([c for c in profile.candidates if c not in curr_cands])
    elim_list = list()
    
    borda_scores = {c: _borda_score(rankings, rcounts, len(candidates), c) for c in candidates}
    
    avg_borda_score = np.mean(list(borda_scores.values()))
    below_borda_avg_candidates = [c for c in borda_scores.keys() 
                                  if borda_scores[c] <= avg_borda_score]
    
    cands_to_ignore = np.concatenate((cands_to_ignore, np.array(below_borda_avg_candidates)), axis=None)
    winners = list()
    if cands_to_ignore.shape[0] == all_num_cands:  # all candidates have same Borda score
        elim_list.append({"avg_borda_score": avg_borda_score, 
                          "elim_cands": below_borda_avg_candidates,
                          "borda_scores": borda_scores})
        winners = sorted(candidates)
    elif all_num_cands - cands_to_ignore.shape[0]  == 1: # one candidate remains
        elim_list.append({"avg_borda_score": avg_borda_score, 
                          "elim_cands": below_borda_avg_candidates,
                          "borda_scores": borda_scores})
        winners = [c for c in candidates if not isin(cands_to_ignore, c)]
    else: 
        num_cands = len(candidates)
        elim_list.append({"avg_borda_score": avg_borda_score, 
                          "elim_cands": below_borda_avg_candidates,
                          "borda_scores": borda_scores})
        updated_rankings = _find_updated_profile(rankings, cands_to_ignore, num_cands)
        

    while len(winners) == 0: 
        
        borda_scores = {c: _borda_score(updated_rankings, rcounts, num_cands - cands_to_ignore.shape[0], c) 
                        for c in candidates if not isin(cands_to_ignore, c)}
        
        avg_borda_score = np.mean(list(borda_scores.values()))
    
        below_borda_avg_candidates = [c for c in borda_scores.keys() 
                                      if borda_scores[c] <= avg_borda_score]
        
        cands_to_ignore = np.concatenate((cands_to_ignore, np.array(below_borda_avg_candidates)), axis=None)
        elim_list.append({"avg_borda_score": avg_borda_score, 
                          "elim_cands": below_borda_avg_candidates,
                          "borda_scores": borda_scores})
                
        if cands_to_ignore.shape[0] == all_num_cands:  # all remaining candidates have been removed
            winners = sorted(below_borda_avg_candidates)
        elif all_num_cands - cands_to_ignore.shape[0]  == 1: # one candidate remains
            winners = [c for c in candidates if not isin(cands_to_ignore, c)]
        else:
            updated_rankings = _find_updated_profile(rankings, cands_to_ignore, num_cands)
            
    return winners, elim_list


@vm(name = "Iterated Removal Condorcet Loser")
def iterated_removal_cl(edata, curr_cands = None):
    """
    Iteratively remove candidates that are Condorcet losers until there are no Condorcet losers.   A candidate :math:`c` is a **Condorcet loser** when every other candidate is majority preferred to :math:`c`. 

    Args:
        edata (Profile, ProfileWithTies, MajorityGraph, MarginGraph): Any election data that has a `condorcet_loser` method. 
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A sorted list of candidates

    .. seealso::

        :meth:`pref_voting.profiles.Profile.condorcet_loser`,  :meth:`pref_voting.profiles_with_ties.ProfileWithTies.condorcet_loser`, :meth:`pref_voting.weighted_majority_graphs.MajorityGraph.condorcet_loser`

    :Example: 

    .. exec_code:: 

        from pref_voting.profiles import Profile
        from pref_voting.profiles_with_ties import ProfileWithTies
        from pref_voting.iterative_methods import iterated_removal_cl
        
        prof = Profile([[2, 1, 3, 0], [2, 1, 0, 3], [3, 1, 2, 0], [1, 2, 3, 0]], [1, 1, 1, 1])

        prof.display()
        iterated_removal_cl.display(prof)
        iterated_removal_cl.display(prof.majority_graph())
        iterated_removal_cl.display(prof.margin_graph())

        prof2 = ProfileWithTies([{2:1, 1:1, 3:2, 0:3}, {2:1, 1:2, 0:3, 3:4}, {3:1, 1:2, 2:3, 0:4}, {1:1, 2:2, 3:3, 0:4}], [1, 1, 1, 1])

        prof2.display()
        iterated_removal_cl.display(prof2)

    """

    condorcet_loser = edata.condorcet_loser(curr_cands = curr_cands)  
    
    remaining_cands = edata.candidates if curr_cands is None else curr_cands
    
    while len(remaining_cands) > 1 and  condorcet_loser is not None:    
        remaining_cands = [c for c in remaining_cands if c not in [condorcet_loser]]
        condorcet_loser = edata.condorcet_loser(curr_cands = remaining_cands)
            
    return sorted(remaining_cands)


def iterated_removal_cl_with_explanation(edata, curr_cands = None):
    """
    Iterated Removal Condorcet Loser with an explanation. In addition to the winner(s), return the order of elimination, where each candidate in the list is a Condorcet loser in the profile (restricted to the remaining candidates). 

    Args:
        edata (Profile, ProfileWithTies, MajorityGraph, MarginGraph): Any election data that has a `condorcet_loser` method. 
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A sorted list of candidates

    :Example: 

    .. exec_code:: 

        from pref_voting.profiles import Profile
        from pref_voting.iterative_methods import iterated_removal_cl_with_explanation
        
        prof = Profile([[2, 1, 3, 0], [2, 1, 0, 3], [3, 1, 2, 0], [1, 2, 3, 0]], [1, 1, 1, 1])

        prof.display()
        ws, exp = iterated_removal_cl_with_explanation(prof)
        print(f"The winning set is {ws}")
        print(f"The order of elimination is {exp}")
    """

    elim_list = list()
    condorcet_loser = edata.condorcet_loser(curr_cands = curr_cands)  
    
    remaining_cands = edata.candidates if curr_cands is None else curr_cands
    
    while len(remaining_cands) > 1 and  condorcet_loser is not None: 
        elim_list.append(condorcet_loser)   
        remaining_cands = [c for c in remaining_cands if c not in [condorcet_loser]]
        condorcet_loser = edata.condorcet_loser(curr_cands = remaining_cands)
            
    return sorted(remaining_cands), elim_list

@vm(name="Iterated Split Cycle")
def iterated_split_cycle(edata, curr_cands = None, strength_function = None):
    """Iteratively remove candidates that are not Split Cycle winners until there is a unique winner or all remaining candidates are Split Cycle winners. 

    Args:
        edata (Profile, ProfileWithTies, MajorityGraph, MarginGraph): Any election data with a margin method. 
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    .. seealso: 
        :meth:pref_voting.margin_based_methods.split_cycle

    Returns: 
        A sorted list of candidates

    :Example: 

    .. exec_code:: 

        from pref_voting.profiles import Profile
        from pref_voting.iterative_methods import iterated_split_cycle
        from pref_voting.margin_based_methods import split_cycle_faster
        
        prof = Profile([[2, 0, 3, 1], [2, 3, 0, 1], [3, 1, 2, 0], [2, 1, 3, 0], [3, 0, 1, 2], [1, 2, 3, 0]], [1, 1, 1, 1, 1, 2])

        prof.display()
        split_cycle_faster.display(prof)
        iterated_split_cycle.display(prof)
    
    """    
    prev_sc_winners = edata.candidates if curr_cands is None else curr_cands
    sc_winners = split_cycle_faster(edata, curr_cands = curr_cands, strength_function = strength_function)
    
    while len(sc_winners) != 1 and sc_winners != prev_sc_winners: 
        prev_sc_winners = sc_winners
        sc_winners = split_cycle_faster(edata, curr_cands = sc_winners, strength_function = strength_function)
        
    return sorted(sc_winners)

@vm(name = "Benham")
def benham(profile, curr_cands = None):
    """
    As long as the profile has no Condorcet winner, eliminate the candidate with the lowest plurality score. Then elect the Condorcet winner of the restricted profile. 
    
    .. important::
        If there is  more than one candidate with the fewest number of first-place votes, then *all* such candidates are removed from the profile. 
    

    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A sorted list of candidates

    .. seealso::

        Related functions:  :func:`pref_voting.iterative_methods.benahm_put`

    """

    # need the total number of all candidates in a profile to check when all candidates have been removed   
    num_cands = profile.num_cands 
    
    candidates = profile.candidates if curr_cands is None else curr_cands
    cands_to_ignore = np.empty(0) if curr_cands is None else np.array([c for c in profile.candidates if c not in curr_cands])

    cw = profile.condorcet_winner(curr_cands = [c for c in profile.candidates if not isin(cands_to_ignore, c)])
    
    winners = [cw] if cw is not None else list()
    
    rs, rcounts = profile.rankings_counts # get all the ranking data
    
    while len(winners) == 0:
        plurality_scores = {c: _num_rank_first(rs, rcounts, cands_to_ignore, c) for c in candidates 
                            if  not isin(cands_to_ignore,c)}  
        min_plurality_score = min(plurality_scores.values())
        lowest_first_place_votes = np.array([c for c in plurality_scores.keys() 
                                             if  plurality_scores[c] == min_plurality_score])

        # remove cands with lowest plurality score
        cands_to_ignore = np.concatenate((cands_to_ignore, lowest_first_place_votes), axis=None)
        if len(cands_to_ignore) == num_cands: # removed all of the candidates 
            winners = sorted(lowest_first_place_votes)
        else:
            cw = profile.condorcet_winner([c for c in profile.candidates if not isin(cands_to_ignore, c)])
            if cw is not None: 
                winners = [cw]

    return sorted(winners)


@vm(name = "Benham TB")
def benham_tb(profile, curr_cands = None, tie_breaker = None):
    """Benham (``benham``) with tie breaking:  If there is  more than one candidate with the fewest number of first-place votes, then remove the candidate with lowest in the tie_breaker ranking from the profile.

    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``
        tie_breaker (List[int]): A list of the candidates in the profile to be used as a tiebreaker.

    Returns: 
        A sorted list of candidates
   
    """

    # the tie_breaker is any linear order (i.e., list) of the candidates
    tb = tie_breaker if tie_breaker is not None else list(range(profile.num_cands))
    
    # need the total number of all candidates in a profile to check when all candidates have been removed   
    num_cands = profile.num_cands 
    
    candidates = profile.candidates if curr_cands is None else curr_cands
    cands_to_ignore = np.empty(0) if curr_cands is None else np.array([c for c in profile.candidates if c not in curr_cands])

    rs, rcounts = profile.rankings_counts # get all the ranking data
    
    cw = profile.condorcet_winner(curr_cands = [c for c in profile.candidates if not isin(cands_to_ignore, c)])

    winners = [cw] if cw is not None else list()
    
    while len(winners) == 0:
        plurality_scores = {c: _num_rank_first(rs, rcounts, cands_to_ignore, c) for c in candidates if not isin(cands_to_ignore,c)}  
        min_plurality_score = min(plurality_scores.values())
        lowest_first_place_votes = np.array([c for c in plurality_scores.keys() 
                                             if plurality_scores[c] == min_plurality_score])
        
        cand_to_remove = lowest_first_place_votes[0]
        for c in lowest_first_place_votes[1:]: 
            if tb.index(c) < tb.index(cand_to_remove):
                cand_to_remove = c

        # remove cands with lowest plurality winners
        cands_to_ignore = np.concatenate((cands_to_ignore, cand_to_remove), axis=None)
        if len(cands_to_ignore) == num_cands: #all the candidates where removed
            winners = sorted(lowest_first_place_votes)
        else:
            cw = profile.condorcet_winner(curr_cands = [c for c in profile.candidates if not isin(cands_to_ignore, c)])
            if cw is not None: 
                winners = [cw]
    return sorted(winners)


@vm(name = "Benham PUT")
def benham_put(profile, curr_cands = None):
    """Benham (:fun:`benham`) with parallel universe tie-breaking (PUT).  Apply the Benham method with a tie-breaker for each possible linear order over the candidates. 
    
    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A sorted list of candidates

    .. warning:: 
        This will take a long time on profiles with many candidates. 


    """
    
    candidates = profile.candidates if curr_cands is None else curr_cands
    cands_to_ignore = np.empty(0) if curr_cands is None else np.array([c for c in profile.candidates if c not in curr_cands])
    
    cw = profile.condorcet_winner(curr_cands = [c for c in profile.candidates if not isin(cands_to_ignore, c)])
    
    winners = [cw] if cw is not None else list()
        
    if len(winners) == 0:
        # run Instant Runoff with tie-breaker for each permulation of candidates
        for tb in permutations(candidates):
            winners += instant_runoff_tb(profile, curr_cands = curr_cands, tie_breaker = tb) 
    return sorted(list(set(winners)))
    
iterated_vms = [
    instant_runoff,
    instant_runoff_tb,
    instant_runoff_put,
    hare,
    ranked_choice,
    plurality_with_runoff,
    coombs,
    coombs_tb,
    coombs_put,
    strict_nanson,
    weak_nanson,
    baldwin,
    baldwin_tb,
    baldwin_put,
    iterated_removal_cl,
    iterated_split_cycle,
    benham,
    benham_put,
    benham_tb
]

iterated_vms_with_explanation = [
    instant_runoff_with_explanation,
    coombs_with_explanation,
    baldwin_with_explanation,
    strict_nanson_with_explanation,
    weak_nanson_with_explanation,
    iterated_removal_cl_with_explanation,
]

