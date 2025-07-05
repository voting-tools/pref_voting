'''
    File: iterative_methods.py
    Author: Wes Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)
    Date: January 6, 2022
    Update: October 2, 2023
    
    Implementations of iterative voting methods.
'''
from pref_voting.voting_method import  *
from pref_voting.voting_method import _num_rank_last, _num_rank_first
from pref_voting.profiles import  _borda_score, _find_updated_profile
from pref_voting.margin_based_methods import split_cycle, minimax_scores
from pref_voting.c1_methods import top_cycle, gocha 
from pref_voting.rankings import Ranking
from pref_voting.social_welfare_function import swf
import copy
from itertools import permutations, product
import numpy as np
from pref_voting.voting_method_properties import ElectionTypes
from pref_voting.profiles import Profile
from pref_voting.profiles_with_ties import ProfileWithTies

def _instant_runoff_basic(profile,curr_cands = None):
    "The basic implementation of instant runoff"
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

def _instant_runoff_recursive(profile, curr_cands = None):
    "A recursive implementation of instant runoff"
    candidates = curr_cands if curr_cands is not None else profile.candidates
    cands_to_ignore = np.array([c for c in profile.candidates if c not in candidates])
    rs, rcounts = profile.rankings_counts # get all the ranking data
    plurality_scores = {c: _num_rank_first(rs, rcounts, cands_to_ignore, c) for c in candidates if not isin(cands_to_ignore,c)}  
    min_plurality_score = min(plurality_scores.values())
    lowest_first_place_votes = np.array([c for c in plurality_scores.keys() 
                                            if  plurality_scores[c] == min_plurality_score])

    if len(lowest_first_place_votes) == len(candidates):
        return sorted(lowest_first_place_votes)
    
    else:
        return _instant_runoff_recursive(profile, [c for c in candidates if c not in lowest_first_place_votes])


def _instant_runoff_for_truncated_linear_orders(profile, curr_cands = None, threshold = None, hide_warnings = True): 
    """
    Instant Runoff for Truncated Linear Orders.  Iteratively remove the candidates with the fewest number of first place votes, until there is a candidate with more than the threshold number of first-place votes. 
    If a threshold is not set, then it is strictly more than half of the non-empty ballots. 
    
    Args:
        profile (ProfileWithTies): An anonymous profile with no ties in the ballots (note that ProfileWithTies allows for truncated linear orders).
        threshold (int, float, optional): The threshold needed to win the election.  If it is not set, then it is strictly more than half of the remaining ballots.
        hide_warnings (bool, optional): Show or hide the warnings when more than one candidate is eliminated in a round.

    Returns: 
        A sorted list of candidates
    
    .. note:: This is the simultaneous version of instant runoff, not the parallel-universe tiebreaking version. It is intended to be run on profiles with large number of voters in which there is a very low probability of a tie in the fewest number of first place votes.  A warning is displayed when more than one candidate is eliminated. 

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

@vm(name = "Instant Runoff",
    input_types=[ElectionTypes.PROFILE])
def instant_runoff(profile, curr_cands = None, algorithm = "basic", **kwargs):
    """
    If there is a majority winner then that candidate is the winner. If there is no majority winner, then remove all candidates that are ranked first by the fewest number of voters. Continue removing candidates with the fewest number first-place votes until there is a candidate with a majority of first place votes.  
    
    .. important::
        If there is more than one candidate with the fewest number of first-place votes, then *all* such candidates are removed from the profile. 
    
    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``
        algorithm (str, optional): The algorithm to use.  Options are "basic" and "recursive".  The default is "basic".

    Returns: 
        A sorted list of candidates

    .. seealso::

        Instant Runoff is also known as "Ranked Choice", "Hare", and "Alternative Vote".

        Related functions:  :func:`pref_voting.iterative_methods.instant_runoff_tb`, :func:`pref_voting.iterative_methods.instant_runoff_put`, :func:`pref_voting.iterative_methods.instant_runoff_with_explanation`
   
    :Example: 

    .. exec_code:: 

        from pref_voting.profiles import Profile
        from pref_voting.iterative_methods import instant_runoff, ranked_choice, alternative_vote, hare
        
        prof = Profile([[2, 1, 0], [0, 2, 1], [1, 2, 0]], [1, 2, 2])

        prof.display()
        instant_runoff.display(prof)
        ranked_choice.display(prof)
        alternative_vote.display(prof)
        hare.display(prof)

    """
    if isinstance(profile, Profile): 
        if algorithm == "basic":
            return _instant_runoff_basic(profile, curr_cands = curr_cands)
        
        elif algorithm == "recursive":
            return _instant_runoff_recursive(profile, curr_cands = curr_cands)
        
        else:
            raise ValueError("Algorithm must be either 'basic' or 'recursive'.")
    elif isinstance(profile, ProfileWithTies): 
        return _instant_runoff_for_truncated_linear_orders(profile, curr_cands = curr_cands, **kwargs)
# Create some aliases for instant runoff
instant_runoff.set_name("Hare")
hare = copy.deepcopy(instant_runoff)
hare.skip_registration = True
instant_runoff.set_name("Ranked Choice")
ranked_choice = copy.deepcopy(instant_runoff)
ranked_choice.skip_registration = True
instant_runoff.set_name("Alternative Vote")
alternative_vote = copy.deepcopy(instant_runoff)
alternative_vote.skip_registration = True


# reset the name Instant Runoff
instant_runoff.set_name("Instant Runoff")

@swf(name = "Instant Runoff Ranking")
def instant_runoff_ranking(profile, curr_cands = None):
    """Returns the reverse of the elimination order in the instant runoff voting process.

    Args:
        profile (Profile): An anonymous Profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns:
        A Ranking of the candidates.
    """
    
    candidates = curr_cands if curr_cands is not None else profile.candidates
    cands_to_ignore = np.array([c for c in profile.candidates if c not in candidates])
    rs, rcounts = profile.rankings_counts # get all the ranking data
    plurality_scores = {c: _num_rank_first(rs, rcounts, cands_to_ignore, c) for c in candidates if not isin(cands_to_ignore,c)}  
    min_plurality_score = min(plurality_scores.values())
    lowest_first_place_votes = np.array([c for c in plurality_scores.keys() 
                                            if  plurality_scores[c] == min_plurality_score])
    
    if len(lowest_first_place_votes) == len(candidates):
        full_tie = Ranking({c:0 for c in candidates})
        return full_tie
    
    else:
        rec_ranking = instant_runoff_ranking(profile, [c for c in candidates if c not in lowest_first_place_votes])
        max_rank = max(rec_ranking.ranks)
        rec_ranking_dict = rec_ranking.rmap
        ranking = Ranking({c: rec_ranking_dict[c] if not isin(lowest_first_place_votes,c) else max_rank+1 for c in candidates})

        return ranking

@vm(name = "Instant Runoff TB",
    input_types=[ElectionTypes.PROFILE])
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

@vm(name = "Instant Runoff PUT",
    input_types=[ElectionTypes.PROFILE])
def instant_runoff_put(profile, curr_cands = None):
    """
    Instant Runoff (:func:`instant_runoff`) with parallel universe tie-breaking (PUT), defined recursively: if there is a candidate with a strict majority of first-place votes, that candidate is the IRV-PUT winner; otherwise a candidate x is an IRV-PUT winner if there is some candidate y with a minimal number of first-place votes such that after removing y from the profile, x is an IRV-PUT winner.
    
    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A sorted list of candidates

    .. warning:: 
        This will take a long time on profiles with many candidates having the same plurality scores

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

    plurality_scores = profile.plurality_scores(candidates)

    strict_maj_size = profile.strict_maj_size()
    majority_winner = [cand for cand, score in plurality_scores.items() if score >= strict_maj_size]

    if len(majority_winner) > 0:
        return majority_winner
    
    original_num_cands = len(candidates)
    
    # immediately eliminate candidates with plurality score 0
    # this is safe, because every elimination order will eliminate all these candidates first (in some order)
    candidates = [cand for cand in candidates if plurality_scores[cand] > 0]
    if len(candidates) < original_num_cands:
        # if we removed some candidates, we need to update the plurality scores
        plurality_scores = profile.plurality_scores(candidates)

    # plurality losers
    worst_score = min(plurality_scores.values())
    cands_to_remove = [cand for cand, value in plurality_scores.items() if value == worst_score]
    
    winners = []
    for cand_to_remove in cands_to_remove:
        new_winners = instant_runoff_put(profile, curr_cands = [c for c in candidates if not c == cand_to_remove])
        winners = winners + new_winners
    
    return sorted(set(winners))


# Create some aliases for instant runoff
instant_runoff_put.set_name("Hare PUT")
hare_put = copy.deepcopy(instant_runoff_put)
hare_put.skip_registration = True
instant_runoff_put.set_name("Ranked Choice PUT")
ranked_choice_put = copy.deepcopy(instant_runoff_put)
ranked_choice_put.skip_registration = True

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

@vm(name="Instant Runoff (Truncated Linear Orders)",
    input_types=[ElectionTypes.TRUNCATED_LINEAR_PROFILE])
def instant_runoff_for_truncated_linear_orders(profile, curr_cands = None, threshold = None, hide_warnings = True): 
    """
    Instant Runoff for Truncated Linear Orders.  Iteratively remove the candidates with the fewest number of first place votes, until there is a candidate with more than the threshold number of first-place votes. 
    If a threshold is not set, then it is strictly more than half of the non-empty ballots. 
    
    Args:
        profile (ProfileWithTies): An anonymous profile with no ties in the ballots (note that ProfileWithTies allows for truncated linear orders).
        threshold (int, float, optional): The threshold needed to win the election.  If it is not set, then it is strictly more than half of the remaining ballots.
        hide_warnings (bool, optional): Show or hide the warnings when more than one candidate is eliminated in a round.

    Returns: 
        A sorted list of candidates
    
    .. note:: This is the simultaneous version of instant runoff, not the parallel-universe tiebreaking version. It is intended to be run on profiles with large number of voters in which there is a very low probability of a tie in the fewest number of first place votes.  A warning is displayed when more than one candidate is eliminated. 

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

def top_n_instant_runoff_for_truncated_linear_orders(
    profile, 
    n,
    curr_cands = None, 
    threshold = None, 
    hide_warnings = True): 
    """
    Returns the top n candidates according to the Instant Runoff method: Iteratively remove candidates until there are at most n candidates left.   Note that since there may be multiple candidates with the lowest plurality score, it may not be possible to reduce to exactly n candidates, in which case the function will return None.
    """
    
    assert all([not r.has_overvote() for r in profile.rankings]), "Instant Runoff is only defined when all the ballots are truncated linear orders."
    
    curr_cands = profile.candidates if curr_cands is None else curr_cands

    if len(curr_cands) <= n:
        return sorted(curr_cands)

    # we need to remove empty rankings during the algorithm, so make a copy of the profile
    prof2 = copy.deepcopy(profile) 
    
    _prof = prof2.remove_candidates([c for c in profile.candidates if c not in curr_cands])

    # remove the empty rankings
    _prof.remove_empty_rankings()
    
    remaining_candidates = _prof.candidates
        
    pl_scores = _prof.plurality_scores()
    
    while len(remaining_candidates) > n: 
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
            remaining_candidates = reduced_prof.candidates
            break 

        remaining_candidates = [c for c in remaining_candidates if c not in cands_to_remove]

    if len(remaining_candidates) != n:
        if not hide_warnings:
            print(f"Warning: cannot reduce to exactly {n} candidates.")
        return None

    return sorted(remaining_candidates)


@vm(name="Bottom-Two-Runoff Instant Runoff",
    input_types=[ElectionTypes.PROFILE])
def bottom_two_runoff_instant_runoff(profile, curr_cands = None):
    """Find the two candidates with the lowest two plurality scores, remove the one who loses head-to-head to the other, and repeat until a single candidate remains. 
    
    If there is a tie for lowest or second lowest plurality score, consider all head-to-head matches between a candidate with lowest and a candidate with second lowest plurality score, and remove all the losers of the head-to-head matches, unless this would remove all candidates.

    .. note:: 
        BTR-IRV is a Condorcet consistent voting method, i.e., if a Condorcet winner exists, then BTR-IRV will elect the Condorcet winner. 

    .. seealso::

        Related functions:  :func:`pref_voting.iterative_methods.bottom_two_runoff_instant_runoff_put`

    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A sorted list of candidates
    """
    candidates = profile.candidates if curr_cands is None else curr_cands 

    if len(candidates) == 1:
        return candidates

    plurality_scores = profile.plurality_scores(candidates)
    worst_score = min(plurality_scores.values())
    cands_with_lowest_plurality_score = [cand for cand, value in plurality_scores.items() if value == worst_score]

    if len(cands_with_lowest_plurality_score) > 1:
        cands_with_second_lowest_plurality_score = cands_with_lowest_plurality_score
    else:
        second_lowest_plurality_score = sorted(plurality_scores.values())[1]
        cands_with_second_lowest_plurality_score = [cand for cand, value in plurality_scores.items() if value == second_lowest_plurality_score]
    
    cands_to_remove = []

    for c1 in cands_with_lowest_plurality_score:
        for c2 in cands_with_second_lowest_plurality_score:
            if c1 != c2:
                if profile.margin(c1,c2) <= 0:
                    cands_to_remove.append(c1)
                else:
                    cands_to_remove.append(c2)
    
    if len(set(cands_to_remove)) == len(candidates):
        return candidates
    else:
        return bottom_two_runoff_instant_runoff(profile, [cand for cand in candidates if cand not in set(cands_to_remove)])

@vm(name="Bottom-Two-Runoff Instant Runoff PUT",
    input_types=[ElectionTypes.PROFILE])
def bottom_two_runoff_instant_runoff_put(profile, curr_cands = None):
    """Find the two candidates with the lowest two plurality scores, remove the one who loses head-to-head to the other, and repeat until a single candidate remains. Parallel-universe tiebreaking is used to break ties for lowest or second lowest plurality scores. 

    .. note:: 
        BTR-IRV is a Condorcet consistent voting method, i.e., if a Condorcet winner exists, then BTR-IRV will elect the Condorcet winner. 

    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A sorted list of candidates
    """
    candidates = profile.candidates if curr_cands is None else curr_cands 

    if len(candidates) == 1:
        return candidates

    plurality_scores = profile.plurality_scores(candidates)
    worst_score = min(plurality_scores.values())
    cands_with_lowest_plurality_score = [cand for cand, value in plurality_scores.items() if value == worst_score]

    if len(cands_with_lowest_plurality_score) > 1:
        cands_with_second_lowest_plurality_score = cands_with_lowest_plurality_score
    else:
        second_lowest_plurality_score = sorted(plurality_scores.values())[1]
        cands_with_second_lowest_plurality_score = [cand for cand, value in plurality_scores.items() if value == second_lowest_plurality_score]
    
    winners = []

    for c1 in cands_with_lowest_plurality_score:
        for c2 in cands_with_second_lowest_plurality_score:
            if c1 != c2:
                if profile.margin(c1,c2) <= 0:
                    additional_winners = bottom_two_runoff_instant_runoff_put(profile, curr_cands = [c for c in candidates if not c == c1])
                else:
                    additional_winners = bottom_two_runoff_instant_runoff_put(profile, curr_cands = [c for c in candidates if not c == c2])
                
                winners = winners + additional_winners
    
    return sorted(set(winners))
    

@vm(name = "Plurality with Runoff PUT",
    input_types=[ElectionTypes.PROFILE])
def plurality_with_runoff_put(profile, curr_cands = None):
    """If there is a majority winner then that candidate is the Plurality with Runoff winner. Otherwise hold a runoff between the top two candidates: the candidate with the most first place votes and the candidate with the 2nd most first place votes (or perhaps tied for the most first place votes). In the case of multiple candidates tied for the most or 2nd most first place votes, use parallel-universe tiebreaking: a candidate is a Plurality with Runoff winner if it is a winner in some runoff as described. If the candidates are all tied for the most first place votes, then all candidates are winners.
        
    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A sorted list of candidates
        
    .. note:: 
        Plurality with Runoff is the same as Instant Runoff when there are 3 candidates, but they can give different answers with 4 or more candidates. 

    :Example: 

    .. exec_code:: 

        from pref_voting.profiles import Profile
        from pref_voting.iterative_methods import instant_runoff, plurality_with_runoff_put

        prof = Profile([[0, 1, 2, 3], [3, 1, 2, 0], [2, 0, 3, 1], [1, 2, 3, 0], [2, 3, 0, 1], [0, 3, 2, 1]], [2, 1, 2, 2, 1, 2])
        prof.display()
        instant_runoff.display(prof)
        plurality_with_runoff_put.display(prof)
    
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

def plurality_with_runoff_put_with_explanation(profile, curr_cands = None):
    """Plurality with Runoff with an explanation. In addition to the winner(s), return list of the pairs of candidate that move on to runoff round.    
  
    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A sorted list of candidates
        
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
        all_runoff_pairs = list(product(first, second))
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
    
    return sorted(list(set(winners))), list(all_runoff_pairs)

@vm(name = "Coombs",
    input_types=[ElectionTypes.PROFILE])
def coombs(profile, curr_cands = None):
    """If there is a majority winner then that candidate is the Coombs winner.   If there is no majority winner, then remove all candidates that are ranked last by the greatest number of voters.  Continue removing candidates with the most last-place votes until there is a candidate with a majority of first place votes.  
    
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

@vm(name = "Coombs TB",
    input_types=[ElectionTypes.PROFILE])
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

@vm(name = "Coombs PUT",
    input_types=[ElectionTypes.PROFILE])
def coombs_put(profile, curr_cands = None):
    """Coombs with parallel universe tie-breaking (PUT), defined recursively: if there is a candidate with a strict majority of first-place votes, that candidate is the Coombs-PUT winner; otherwise a candidate x is a Coombs-PUT winner if there is some candidate y with a maximal number of last-place votes such that after removing y from the profile, x is a Coombs-PUT winner.
    
    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A sorted list of candidates

    .. warning:: 
        This will take a long time on profiles with many candidates having the same number of last-place votes.

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
    cands_to_ignore = np.empty(0) if curr_cands is None else np.array([c for c in profile.candidates if c not in curr_cands])

    strict_maj_size = profile.strict_maj_size()
    majority_winner = [cand for cand, value in profile.plurality_scores(candidates).items() if value >= strict_maj_size]

    if len(majority_winner) > 0:
        return majority_winner
    
    rs, rcounts = profile.rankings_counts # get all the ranking data
    
    last_place_scores = {c: _num_rank_last(rs, rcounts, cands_to_ignore, c) for c in candidates}  
    max_last_place_score = max(last_place_scores.values())
    cands_to_remove = [c for c in last_place_scores.keys() if last_place_scores[c] == max_last_place_score]

    winners = []
    for cand_to_remove in cands_to_remove:
        new_winners = coombs_put(profile, curr_cands = [c for c in candidates if not c == cand_to_remove])
        winners = winners + new_winners
    
    return sorted(set(winners))

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

@vm(name = "Baldwin",
    input_types=[ElectionTypes.PROFILE])
def baldwin(profile, curr_cands = None):
    """Iteratively remove all candidates with the lowest Borda score until a single candidate remains.  If, at any stage, all  candidates have the same Borda score,  then all (remaining) candidates are winners.

    .. note:: 
        Baldwin is a Condorcet consistent voting method, i.e., if a Condorcet winner exists, then Baldwin will elect the Condorcet winner. 

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
    rankings = profile._rankings if curr_cands is None else _find_updated_profile(profile._rankings, np.array([c for c in profile.candidates if c not in candidates]), all_num_cands)
    num_cands = len(candidates)
    cands_to_ignore = np.empty(0) 
    borda_scores = {c: _borda_score(rankings, rcounts, len(candidates), c) for c in candidates}

    min_borda_score = min(list(borda_scores.values()))
    last_place_borda_scores = [c for c in candidates 
                               if c in borda_scores.keys() and borda_scores[c] == min_borda_score]
      
    cands_to_ignore = np.concatenate((cands_to_ignore, last_place_borda_scores), axis=None)
    
    winners = list()
    if cands_to_ignore.shape[0] ==  num_cands: # all candidates have lowest Borda score
        winners = sorted(last_place_borda_scores)
    else: # remove the candidates with lowest Borda score
        num_cands = len(candidates)
        updated_rankings = _find_updated_profile(rankings, cands_to_ignore, num_cands)
        
    while len(winners) == 0:
        borda_scores = {c: _borda_score(updated_rankings, rcounts, num_cands - cands_to_ignore.shape[0], c) for c in candidates if not isin(cands_to_ignore, c)}
                
        min_borda_score = min(borda_scores.values())
        last_place_borda_scores = [c for c in borda_scores.keys() if borda_scores[c] == min_borda_score]
        
        cands_to_ignore = np.concatenate((cands_to_ignore, last_place_borda_scores), axis=None)
                
        if cands_to_ignore.shape[0] == num_cands: # removed all remaining candidates
            winners = sorted(last_place_borda_scores)
        elif num_cands - cands_to_ignore.shape[0] ==  1: # only one candidate remains
            winners = sorted([c for c in candidates if c not in cands_to_ignore])
        else: 
            updated_rankings = _find_updated_profile(rankings, cands_to_ignore, num_cands)
    return sorted(winners)

@vm(name = "Baldwin TB",
    input_types=[ElectionTypes.PROFILE])
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

    if len(profile.candidates) <= 1:
        return sorted(profile.candidates)

    all_num_cands = profile.num_cands  
    candidates = profile.candidates if curr_cands is None else curr_cands
    rcounts = profile._rcounts # get all the ranking data
    rankings = profile._rankings if curr_cands is None else _find_updated_profile(profile._rankings, np.array([c for c in profile.candidates if c not in candidates]), all_num_cands)
    num_cands = len(candidates)
    cands_to_ignore = np.empty(0) 
    borda_scores = {c: _borda_score(rankings, rcounts, num_cands, c) for c in candidates}

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
                
        if cands_to_ignore.shape[0] == num_cands: # removed all remaining candidates
            winners = sorted(last_place_borda_scores)
        elif num_cands - cands_to_ignore.shape[0] ==  1: # only one candidate remains
            winners = sorted([c for c in candidates if c not in cands_to_ignore])
        else: 
            updated_rankings = _find_updated_profile(rankings, cands_to_ignore, num_cands)
    return sorted(winners)

@vm(name = "Baldwin PUT",
    input_types=[ElectionTypes.PROFILE])
def baldwin_put(profile, curr_cands=None):
    """Baldwin with parallel universe tie-breaking (PUT), defined recursively: if there is a single candidate in the profile, that candidate wins; otherwise a candidate x is a Baldwin-PUT winner if there is some candidate y with a minimal Borda score such that after removing y from the profile, x is a Baldwin-PUT winner.
    
    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A sorted list of candidates

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

    num_original_cands = len(profile.candidates)
    candidates = profile.candidates if curr_cands is None else curr_cands 
    
    if len(candidates) == 1:
        return candidates
    
    cands_to_ignore = np.empty(0) if curr_cands is None else np.array([c for c in profile.candidates if c not in curr_cands])
    
    rankings, rcounts = profile.rankings_counts # get all the ranking data
    updated_rankings = _find_updated_profile(rankings, cands_to_ignore, num_original_cands)

    borda_scores = {c: _borda_score(updated_rankings, rcounts, num_original_cands - cands_to_ignore.shape[0], c) for c in candidates if not isin(cands_to_ignore, c)}
    min_borda_score = min(list(borda_scores.values()))
    
    cands_to_remove = [c for c in candidates if c in borda_scores.keys() and borda_scores[c] == min_borda_score]

    winners = []
    for cand_to_remove in cands_to_remove:
        new_winners = baldwin_put(profile, curr_cands = [c for c in candidates if not c == cand_to_remove])
        winners = winners + new_winners
    
    return sorted(set(winners))


def baldwin_with_explanation(profile, curr_cands = None):
    """Baldwin with an explanation. In addition to the winner(s), return the order in which the candidates are eliminated as a list of dictionaries specifying the Borda scores in the profile restricted to the candidates that have not been eliminated.    

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

@vm(name = "Strict Nanson",
    input_types=[ElectionTypes.PROFILE])
def strict_nanson(profile, curr_cands = None):
    """Iteratively remove all candidates with the  Borda score strictly below the average Borda score until one candidate remains.  If, at any stage, all  candidates have the same Borda score, then all (remaining) candidates are winners.

    .. note:: 
        
        Strict Nanson is a Condorcet consistent voting method, i.e., if a Condorcet winner exists, then Strict Nanson will elect the Condorcet winner. 

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
    """Strict Nanson with an explanation. In addition to the winner(s), return the order in which the candidates are eliminated as a list of dictionaries specifying the Borda scores in the profile restricted to the candidates that have not been eliminated and the average Borda score.    
    
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

@vm(name = "Weak Nanson",
    input_types=[ElectionTypes.PROFILE])
def weak_nanson(profile, curr_cands = None):
    """Iteratively remove all candidates with Borda score less than or equal the average Borda score until one candidate remains.  If, at any stage, all  candidates have the same Borda score, then all (remaining) candidates are winners.

    .. note:: 

        Weak Nanson is a Condorcet consistent voting method, i.e.,  if a Condorcet winner exists, then Weak Nanson will elect the Condorcet winner. 

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
    Weak Nanson with an explanation. In addition to the winner(s), return the order in which the candidates are eliminated as a list of dictionaries specifying the Borda scores in the profile restricted to the candidates that have not been eliminated and the average Borda score.    
    
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


@vm(name = "Iterated Removal Condorcet Loser",
    input_types=[ElectionTypes.PROFILE, ElectionTypes.PROFILE_WITH_TIES, ElectionTypes.MAJORITY_GRAPH, ElectionTypes.MARGIN_GRAPH])
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

def _remove_worst_losers(edata,curr_cands,score_method):
    m_scores = minimax_scores(edata,curr_cands,score_method)
    worst_m_score = min([m_scores[c] for c in curr_cands])
    worst_losers = [c for c in curr_cands if m_scores[c] == worst_m_score]
    if len(worst_losers) == len(curr_cands):
        return curr_cands
    else:
        return [c for c in curr_cands if c not in worst_losers]

@vm(name = "Raynaud",
    input_types=[ElectionTypes.PROFILE, ElectionTypes.PROFILE_WITH_TIES, ElectionTypes.MARGIN_GRAPH])
def raynaud(edata, curr_cands=None, score_method = "margins"):
    """Iteratively remove the candidate(s) whose worst loss is biggest, unless all candidates have the same worst loss. See https://electowiki.org/wiki/Raynaud.
    
    Args:
        edata (Profile, ProfileWithTies, MarginGraph): Any election data that has a `margin` method. 
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``.
        score_method (str, optional): Options include "margins" (the default), "winning" assigns to each candidate :math:`c` the maximum support of a candidate majority preferred to :math:`c`,  and "pairwise_opposition" assigns to each candidate :math:`c` the maximum support of any candidate over :math:`c`.   These scores only lead to different results on non-linear profiles. 

    Returns: 
        A sorted list of candidates. 
    """
    candidates = edata.candidates if curr_cands is None else curr_cands
    new_cands = _remove_worst_losers(edata,candidates,score_method)
    while not new_cands == candidates:
        candidates = new_cands
        new_cands = _remove_worst_losers(edata,candidates,score_method)
    return sorted(candidates)

@vm(name = "Benham",
    input_types=[ElectionTypes.PROFILE])
def benham(profile, curr_cands = None):
    """
    As long as the profile has no Condorcet winner, eliminate the candidate with the lowest plurality score.
    
    .. important::
        If there is  more than one candidate with the fewest number of first-place votes, then *all* such candidates are removed from the profile. 
    

    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A sorted list of candidates

    .. seealso::

        Related functions:  :func:`pref_voting.iterative_methods.benham_put`

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

@vm(name = "Benham TB",
    input_types=[ElectionTypes.PROFILE])
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


@vm(name = "Benham PUT",
    input_types=[ElectionTypes.PROFILE])
def benham_put(profile, curr_cands = None):
    """Benham (:func:`benham`) with parallel universe tie-breaking (PUT), defined recursively: if there is a Condorcet winner, that candidate is the Benham-PUT winner; otherwise a candidate x is a Benham-PUT winner if there is some candidate y with minimal plurality score such that after removing y from the profile, x is a Benham-PUT winner.
    
    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A sorted list of candidates

    .. warning:: 
        This will take a long time on profiles with many candidates having the same plurality scores.

    """
    candidates = profile.candidates if curr_cands is None else curr_cands 

    cw = profile.condorcet_winner(candidates)
    if cw is not None:
        return [cw]
    
    plurality_scores = profile.plurality_scores(candidates)
    worst_score = min(plurality_scores.values())

    cands_to_remove = [cand for cand, value in plurality_scores.items() if value == worst_score]
    
    winners = []
    for cand_to_remove in cands_to_remove:
        new_winners = benham_put(profile, curr_cands = [c for c in candidates if not c == cand_to_remove])
        winners = winners + new_winners
    
    return sorted(set(winners))

def iterated(vm):
    """Iteratively restrict the set of candidates to the vm winners until reaching a fixpoint.

    Args:
        vm (VotingMethod): A voting method.

    Returns:
        A voting method that iterates vm.

    """

    def _vm(edata, curr_cands = None):

        candidates = edata.candidates if curr_cands is None else curr_cands

        vm_ws = vm(edata, curr_cands=candidates)

        while not vm_ws == candidates:
            candidates = vm_ws
            vm_ws = vm(edata, curr_cands=candidates)

        return vm_ws

    return VotingMethod(_vm, name=f"Iterated {vm.name}")

def tideman_alternative(vm):
    """Given a voting method vm, returns a voting method that restricts the profile to the set of vm winners, then eliminates all the candidate with the fewest first-place votes, and then repeats until there is only one vm winner. If at some stage all remaining candidates are tied for the fewest number of first-place votes, then all remaining candidates win.

    Args:
        vm (VotingMethod): A voting method.

    Returns:
        The Tideman Alternative PUT version of vm.

    """
    
    def _ta(profile, curr_cands = None):

        candidates = profile.candidates if curr_cands is None else curr_cands 
    
        vm_ws = vm(profile, curr_cands = candidates)

        plurality_scores = profile.plurality_scores(vm_ws)
        worst_score = min(plurality_scores.values())

        cands_to_remove = [cand for cand, value in plurality_scores.items() if value == worst_score]
        
        if len(cands_to_remove) == len(vm_ws):
            return vm_ws
            
        else:
            return _ta(profile, curr_cands = [c for c in candidates if not c in cands_to_remove])
        
    _ta.__name__ = f"tideman_alternative_{vm.__name__}"
    return VotingMethod(_ta, name=f"Tideman Alternative {vm.name}")

tideman_alternative_smith = tideman_alternative(top_cycle)
tideman_alternative_smith.load_properties()
tideman_alternative_smith.input_types = [ElectionTypes.PROFILE]

tideman_alternative_gocha = tideman_alternative(gocha)
tideman_alternative_gocha.load_properties()
tideman_alternative_gocha.input_types = [ElectionTypes.PROFILE]

def tideman_alternative_put(vm):
    """Given a voting method vm, returns a voting method that restricts the profile to the set of vm winners, then eliminates the candidate with the fewest first-place votes, and then repeats until there is only one vm winner. Parallel-universe tiebreaking is used when there are multiple candidates with the fewest first-place votes.

    Args:
        vm (VotingMethod): A voting method.

    Returns:
        The Tideman Alternative PUT version of vm.

    """
    
    def _ta(profile, curr_cands = None):

        candidates = profile.candidates if curr_cands is None else curr_cands 
    
        vm_ws = vm(profile, curr_cands = candidates)

        if len(vm_ws) == 1:
            return vm_ws
    
        else: 
            plurality_scores = profile.plurality_scores(vm_ws)
            worst_score = min(plurality_scores.values())
            cands_to_remove = [cand for cand, value in plurality_scores.items() if value == worst_score]
        
            winners = []
            for cand_to_remove in cands_to_remove:
                additional_winners = _ta(profile, curr_cands = [c for c in candidates if not c == cand_to_remove])
                winners = winners + additional_winners

        return sorted(set(winners))
    
    _ta.__name__ = f"tideman_alternative_{vm.__name__}_put"
    return VotingMethod(_ta, name=f"Tideman Alternative {vm.name} PUT")


tideman_alternative_smith_put = tideman_alternative_put(top_cycle)
tideman_alternative_smith_put.load_properties()
tideman_alternative_smith_put.input_types = [ElectionTypes.PROFILE]

tideman_alternative_gocha_put = tideman_alternative_put(gocha)
tideman_alternative_gocha_put.load_properties()
tideman_alternative_gocha_put.input_types = [ElectionTypes.PROFILE]


@vm(name = "Woodall",
    input_types=[ElectionTypes.PROFILE])
def woodall(profile, curr_cands = None):
    """
    If there is a single member of the Smith Set (i.e., a Condorcet winner) then that candidate is the winner.  If there the Smith Set contains more than one candidate, then remove all candidates that are ranked first by the fewest number of voters.  Continue removing candidates with the fewest number first-place votes until there is a single member of the originally Smith Set remaining.  
    
    .. important::
        If there is  more than one candidate with the fewest number of first-place votes, then *all* such candidates are removed from the profile. 
    

    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A sorted list of candidates

    .. seealso::

        Related functions:  :func:`pref_voting.iterative_methods.instant_runoff`

           
    """

    # need the total number of all candidates in a profile to check when all candidates have been removed   
    
    candidates = profile.candidates if curr_cands is None else curr_cands
    cands_to_ignore = np.empty(0) if curr_cands is None else np.array([c for c in profile.candidates if c not in curr_cands])

    s_set = top_cycle(profile, curr_cands=candidates)

    if len(s_set) == 1:
        return s_set
    
    rs, rcounts = profile.rankings_counts # get all the ranking data
    
    winners = []

    while len(winners) == 0:
        plurality_scores = {c: _num_rank_first(rs, rcounts, cands_to_ignore, c) 
                            for c in candidates if  not isin(cands_to_ignore,c)}  
        min_plurality_score = min(plurality_scores.values())
        lowest_first_place_votes = np.array([c for c in plurality_scores.keys() 
                                             if  plurality_scores[c] == min_plurality_score])

        remaining_cands_in_smith_set = [c for c in candidates if not isin(cands_to_ignore,c) and isin(np.array(s_set), c)]

        # remove cands with lowest plurality score
        new_cands_to_ignore = np.concatenate((cands_to_ignore, lowest_first_place_votes), axis=None)
        
        new_remaining_cands_in_smith_set = [c for c in candidates if not isin(new_cands_to_ignore,c) and isin(np.array(s_set), c)]

        if len(new_remaining_cands_in_smith_set) == 0: 
            winners = remaining_cands_in_smith_set
        
        if len(new_remaining_cands_in_smith_set) == 1:
            winners = new_remaining_cands_in_smith_set 
     
        cands_to_ignore = new_cands_to_ignore

    return sorted(winners)

@vm(name = "Knockout Voting",
    input_types=[ElectionTypes.PROFILE])
def knockout(profile, curr_cands=None):
    """Find the two candidates in curr_cands with the lowest and second lowest Borda scores among any candidates in curr_cands. Then remove from curr_cands whichever one loses to the other in a head-to-head majority comparison. Repeat this process, always using the original Borda score (i.e., the Borda scores calculated with respect to all candidates in the profile, not with respect to curr_cands as for Baldwin and Nanson) until only one candidate remains in curr_cands. Parallel universe tie-breaking (PUT) is used when there are ties in lowest or second lowest Borda scores.

    .. note::
        Proposed by Edward B. Foley (with unspecified handling of ties in Borda scores, so PUT is used here as an example).
    
    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A sorted list of candidates

    """
    candidates = profile.candidates if curr_cands is None else curr_cands 
    
    if len(candidates) == 1:
        return candidates

    # Key point: use global Borda score, calculated with respect to the full profile, not just the candidates in curr_cands
    borda_scores = profile.borda_scores()
    min_borda_score = min([borda_scores[c] for c in candidates])
    cands_with_lowest_borda_score = [c for c in candidates if c in borda_scores.keys() and borda_scores[c] == min_borda_score]
    
    winners = []

    # If multiple candidates tie for lowest Borda score, consider all head-to-head matchups of these candidates
    if len(cands_with_lowest_borda_score) > 1:
        for c1 in cands_with_lowest_borda_score:
            for c2 in cands_with_lowest_borda_score:
                if c1 != c2:
                    # If c1 has a non-negative margin over c2, then remove c2 from curr_cands and calculate the winning set
                    # Take the union over all such winning sets as the ultimate winning set
                    if profile.margin(c1, c2) >= 0:
                        new_winners = knockout(profile, curr_cands = [c for c in candidates if not c == c2])
                        winners = winners + new_winners
    
    # If there is a candidate with the uniquely lowest Borda score
    if len(cands_with_lowest_borda_score) == 1:
        cand_with_lowest_borda_score = cands_with_lowest_borda_score[0]

        # There may be multiple candidates with the second lowest Borda score
        second_lowest_borda_score = min([borda_scores[c] for c in candidates if c not in cands_with_lowest_borda_score])
        cands_with_second_lowest_borda_score = [c for c in candidates if c in borda_scores.keys() and borda_scores[c] == second_lowest_borda_score]

        # Consider all head-to-head matchups between the candidate with the lowest Borda score and the candidates with the second lowest Borda score
        for c2 in cands_with_second_lowest_borda_score:

            # If a candidate with second lowest Borda score has a non-negative margin over the candidate with the lowest Borda score, 
            # then remove the latter from curr_cands and calculate the winning set
            if profile.margin(c2, cand_with_lowest_borda_score) >= 0:
                new_winners = knockout(profile, curr_cands = [c for c in candidates if not c == cand_with_lowest_borda_score])
                winners = winners + new_winners

            # If the candidate with the lowest Borda score has a positive margin over a candidate with the second lowest Borda score, 
            # then remove the latter from curr_cands and calculate the winning set
            if profile.margin(cand_with_lowest_borda_score, c2) > 0:
                new_winners = knockout(profile, curr_cands = [c for c in candidates if not c == c2])
                winners = winners + new_winners

    return sorted(set(winners))

@vm(name="Plurality Veto",
    input_types=[ElectionTypes.PROFILE])
def plurality_veto(profile, curr_cands=None, voter_order=None):
    """Returns the winner using the Plurality Veto method of Kizilkaya and Kempe (https://arxiv.org/abs/2305.19632). 

    The method works as follows:
    1. Assign initial scores to candidates equal to their plurality scores
    2. Process voters one by one in the given order
    3. Each voter decrements the score of their bottom choice among non-eliminated candidates
    4. A candidate is eliminated when their score reaches zero
    5. The winner is the last remaining candidate

    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in curr_cands
        voter_order (List[int], optional): List of voters in the order to process them. If None, uses range(len(profile.rankings))

    Returns:
        A sorted list of candidates

    warning::
        If no voter order is specified, the method uses the default order of voter rankings in the profile. Note that anonymizing a profile changes the order of voter rankings.
    """
    candidates = profile.candidates if curr_cands is None else curr_cands

    # Initialize scores as plurality scores
    scores = profile.plurality_scores(curr_cands=candidates)

    # If no voter order specified, use default order
    if voter_order is None:
        voter_order = list(range(profile.num_voters))

    # Track non-eliminated candidates and last remaining
    active_candidates = set(candidates)
    last_remaining = None  # Track the last remaining candidate

    # Process each voter
    for voter in voter_order:
        # Get remaining candidates with positive scores
        remaining = {c for c in active_candidates if scores[c] > 0}
        if not remaining:
            # If all remaining candidates have 0 scores, return the last remaining
            return [last_remaining] if last_remaining is not None else sorted(active_candidates)

        # If only one candidate remains with positive score, they are the winner
        if len(remaining) == 1:
            return sorted(remaining)

        # Get voter's bottom choice among remaining candidates
        ranking = profile.rankings[voter]
        # Find the last ranked candidate among remaining ones
        bottom = next(c for c in reversed(ranking) if c in remaining)

        # Decrement score
        scores[bottom] -= 1
        if scores[bottom] == 0:
            active_candidates.remove(bottom)
            last_remaining = bottom

    # Return the last remaining candidate if there was one,
    # otherwise return candidates with highest remaining score
    if last_remaining is not None:
        return [last_remaining]
    else:
        max_score = max(scores.values())
        return sorted([c for c in candidates if scores[c] == max_score])

def plurality_veto_with_explanation(profile, curr_cands=None, voter_order=None):
    """Returns the winner using the Plurality Veto method, with a detailed explanation of the process.

    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to curr_cands
        voter_order (List[int], optional): List of voters in the order to process them. If None, uses range(len(profile.rankings))

    Returns:
        tuple: A tuple containing (winner list, explanation string)

    warning::
        If no voter order is specified, the method uses the default order of voter rankings in the profile. Note that anonymizing a profile changes the order of voter rankings.
    """
    curr_cands = profile.candidates if curr_cands is None else curr_cands
    scores = profile.plurality_scores(curr_cands=curr_cands)

    if voter_order is None:
        voter_order = list(range(profile.num_voters))

    explanation = [
        "Initial plurality scores: " + str(dict(scores)),
    ]

    # Note any candidates eliminated due to zero initial plurality scores
    zero_initial = [c for c in curr_cands if scores[c] == 0]
    if zero_initial:
        explanation.append(f"Candidates eliminated due to zero initial plurality score: {sorted(zero_initial)}")
    explanation.append("")  # Add blank line

    active_candidates = set(curr_cands)
    last_remaining = None

    # Add initially eliminated candidates
    for c in zero_initial:
        active_candidates.remove(c)
        last_remaining = c

    for step, voter in enumerate(voter_order):
        remaining = {c for c in active_candidates if scores[c] > 0}
        if not remaining:
            explanation.append("All remaining candidates have score 0")
            if last_remaining is not None:
                explanation.append(f"Winners are candidates [{last_remaining}] (highest remaining scores)")
                return [last_remaining], "\\n".join(explanation)
            else:
                winners = sorted(active_candidates)
                explanation.append(f"Winners are candidates {winners} (highest remaining scores)")
                return winners, "\\n".join(explanation)

        # If only one candidate remains with positive score, they are the winner
        if len(remaining) == 1:
            winners = sorted(remaining)
            explanation.append(f"Only one candidate remains with positive score")
            explanation.append(f"Winners: {winners} (highest remaining scores)")
            return winners, "\\n".join(explanation)

        ranking = profile.rankings[voter]
        # Filter ranking to show only active candidates
        active_ranking = [c for c in ranking if c in remaining]
        bottom = next(c for c in reversed(ranking) if c in remaining)

        explanation.append(f"Step {step + 1}:")
        explanation.append(f"Voter {voter} (active candidates in ranking: {active_ranking}) vetoes {bottom}")

        scores[bottom] -= 1
        explanation.append(f"Scores after veto: {dict({c: s for c, s in scores.items() if c in remaining})}")

        if scores[bottom] == 0:
            active_candidates.remove(bottom)
            last_remaining = bottom
            explanation.append(f"Candidate {bottom} eliminated!")
        explanation.append("")

    if last_remaining is not None:
        explanation.append(f"Winners: [{last_remaining}] (highest remaining scores)")
        return [last_remaining], "\\n".join(explanation)
    else:
        max_score = max(scores.values())
        winners = sorted([c for c in curr_cands if scores[c] == max_score])
        explanation.append(f"Winners: {winners} (highest remaining scores)")
        return winners, "\\n".join(explanation)
    
@vm(name="Consensus Builder",
    input_types=[ElectionTypes.PROFILE])
def consensus_builder(profile, curr_cands=None, consensus_building_ranking=None, beta=0.5):

    """Deterministic version of the Random Consensus Builder due to Charikar et al. (https://arxiv.org/abs/2306.17838).

    The method processes candidates in reverse order of the consensus building ranking. When processing
    candidate i, it eliminates any candidate j above i in the consensus building ranking if a large enough fraction of voters (>= beta) prefer i to j. The winner is the last candidate that gets processed.

    Args:
        profile (Profile): An anonymous profile of linear orders
        curr_cands (List[int], optional): Candidates to consider. Defaults to all candidates if not provided.
        consensus_building_ranking (List[int]): The ranking to use as the consensus builder. If not provided, uses the lexicographically first ranking of curr_cands.
        beta (float): Threshold for elimination (default 0.5). When processing candidate i, eliminates a candidate j above i in the consensus building ranking if the proportion of voters preferring i to j is >= beta

    Returns:
        list: List containing the winning candidate

    .. seealso::
        :meth:`pref_voting.probabilistic_methods.random_consensus_builder`
        :meth:`pref_voting.stochastic_methods.random_consensus_builder_st`
    """

    if curr_cands is None:
        curr_cands = profile.candidates

    if consensus_building_ranking is None:
        consensus_building_ranking = sorted(curr_cands)

    # all candidates in curr_cands must be in consensus_building_ranking
    assert len([c for c in curr_cands if c not in consensus_building_ranking]) == 0

    eliminated = set()
    last_processed = None

    for i in reversed(consensus_building_ranking):

        if i not in curr_cands or i in eliminated:
            continue

        for j in consensus_building_ranking:
            if j == i or j not in curr_cands or j in eliminated:
                continue

            if consensus_building_ranking.index(j) < consensus_building_ranking.index(i):
                support_ratio = profile.support(i, j) / profile.num_voters
                if support_ratio >= beta:
                    eliminated.add(j)

        last_processed = i

    return [last_processed]

iterated_vms_with_explanation = [
    instant_runoff_with_explanation,
    coombs_with_explanation,
    plurality_with_runoff_put_with_explanation,
    baldwin_with_explanation,
    strict_nanson_with_explanation,
    weak_nanson_with_explanation,
    iterated_removal_cl_with_explanation,
    plurality_veto_with_explanation
]