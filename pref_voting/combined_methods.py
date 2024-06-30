"""
    File: iterative_methods.py
    Author: Wes Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)
    Date: January 6, 2022
    Revised: November 13, 2023
    
    Implementations of voting methods that combine multiple methods
"""

from pref_voting.voting_method import *
from pref_voting.scoring_methods import plurality, borda
from pref_voting.iterative_methods import iterated_removal_cl, instant_runoff, instant_runoff_put, instant_runoff_for_truncated_linear_orders
from pref_voting.profiles import _find_updated_profile, _num_rank

from pref_voting.c1_methods import condorcet, smith_set, copeland, top_cycle
from pref_voting.margin_based_methods import minimax, minimax_scores
from pref_voting.profiles import Profile
from pref_voting.profiles_with_ties import ProfileWithTies
from pref_voting.voting_method_properties import VotingMethodProperties, ElectionTypes

@vm(name = "Daunou",
    input_types = [ElectionTypes.PROFILE])
def daunou(profile, curr_cands=None):
    """Implementation of Daunou's voting method as described in the paper: https://link.springer.com/article/10.1007/s00355-020-01276-w

    If there is a Condorcet winner, then that candidate is the winner.  Otherwise, iteratively remove all Condorcet losers then select the plurality winner from among the remaining candidates.

    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns:
        A sorted list of candidates

    :Example:

    .. exec_code::

        from pref_voting.profiles import Profile
        from pref_voting.combined_methods import daunou
        from pref_voting.scoring_methods import plurality

        prof = Profile([[1, 3, 2, 0], [0, 2, 3, 1], [1, 3, 0, 2], [3, 1, 0, 2]], [1, 1, 1, 1])

        prof.display()

        daunou.display(prof)
        plurality.display(prof)

    """

    candidates = profile.candidates if curr_cands is None else curr_cands
    cw = profile.condorcet_winner(curr_cands=candidates)
    if cw is not None:
        winners = [cw]
    else:
        cands_survive_it_rem_cl = iterated_removal_cl(profile, curr_cands=curr_cands)
        winners = plurality(profile, curr_cands=cands_survive_it_rem_cl)

    return sorted(winners)


@vm(name = "Blacks",
    input_types = [ElectionTypes.PROFILE])
def blacks(profile, curr_cands=None):
    """If a Condorcet winner exists return that winner. Otherwise, return the Borda winning set.

    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns:
        A sorted list of candidates

    :Example:

    .. exec_code::

        from pref_voting.profiles import Profile
        from pref_voting.combined_methods import blacks
        from pref_voting.scoring_methods import borda

        prof = Profile([[2, 0, 1], [0, 1, 2], [2, 1, 0], [1, 2, 0]], [1, 1, 1, 1])

        prof.display()

        blacks.display(prof)
        borda.display(prof)


    """

    cw = profile.condorcet_winner(curr_cands=curr_cands)

    if cw is not None:
        winners = [cw]
    else:
        winners = borda(profile, curr_cands=curr_cands)

    return winners

@vm(name = "Smith IRV",
    input_types = [ElectionTypes.PROFILE])
def smith_irv(profile, curr_cands=None):
    """After restricting to the Smith Set, return the Instant Runoff winner.

    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns:
        A sorted list of candidates

    :Example:

    .. exec_code::

        from pref_voting.profiles import Profile
        from pref_voting.combined_methods import smith_irv
        from pref_voting.iterative_methods import instant_runoff, instant_runoff_put

        prof = Profile([[0, 2, 1, 3], [1, 3, 0, 2], [2, 1, 3, 0], [2, 3, 0, 1]], [1, 1, 1, 1])

        prof.display()

        instant_runoff.display(prof)
        instant_runoff_put.display(prof)
        smith_irv.display(prof)

    """

    smith = smith_set(profile, curr_cands=curr_cands)

    return instant_runoff(profile, curr_cands=smith)

@vm(name = "Smith IRV PUT",
    input_types = [ElectionTypes.PROFILE])
def smith_irv_put(profile, curr_cands=None):
    """After restricting to the Smith Set, return the Instant Runoff winner.

    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns:
        A sorted list of candidates

    :Example:

    .. exec_code::

        from pref_voting.profiles import Profile
        from pref_voting.combined_methods import smith_irv_put
        from pref_voting.iterative_methods import instant_runoff, instant_runoff_put

        prof = Profile([[0, 2, 1, 3], [1, 3, 0, 2], [2, 1, 3, 0], [2, 3, 0, 1]], [1, 1, 1, 1])

        prof.display()

        instant_runoff.display(prof)
        instant_runoff_put.display(prof)
        smith_irv_put.display(prof)

    """

    smith = smith_set(profile, curr_cands=curr_cands)

    return instant_runoff_put(profile, curr_cands=smith)

@vm(name = "Condorcet IRV",
    input_types = [ElectionTypes.PROFILE])
def condorcet_irv(profile, curr_cands=None):
    """If a Condorcet winner exists, elect that candidate, otherwise return the instant runoff winners.

    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns:
        A sorted list of candidates

    :Example:

    .. exec_code::

        from pref_voting.profiles import Profile
        from pref_voting.combined_methods import condorcet_irv
        from pref_voting.iterative_methods import instant_runoff, instant_runoff_put

        prof = Profile([[0, 2, 1, 3], [1, 3, 0, 2], [2, 1, 3, 0], [2, 3, 0, 1]], [1, 1, 1, 1])

        prof.display()

        instant_runoff.display(prof)
        instant_runoff_put.display(prof)
        condorcet_irv.display(prof)

    """

    cw = profile.condorcet_winner(curr_cands=curr_cands)
    if cw is not None: 
        return [cw]
    else:
        return instant_runoff(profile, curr_cands=curr_cands)

@vm(name = "Condorcet IRV PUT",
    input_types = [ElectionTypes.PROFILE])
def condorcet_irv_put(profile, curr_cands=None):
    """If a Condorcet winner exists, elect that candidate, otherwise return the instant runoff put winners.

    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns:
        A sorted list of candidates

    :Example:

    .. exec_code::

        from pref_voting.profiles import Profile
        from pref_voting.combined_methods import condorcet_irv_put
        from pref_voting.iterative_methods import instant_runoff, instant_runoff_put

        prof = Profile([[0, 2, 1, 3], [1, 3, 0, 2], [2, 1, 3, 0], [2, 3, 0, 1]], [1, 1, 1, 1])

        prof.display()

        instant_runoff.display(prof)
        instant_runoff_put.display(prof)
        condorcet_irv_put.display(prof)

    """

    cw = profile.condorcet_winner(curr_cands=curr_cands)
    if cw is not None: 
        return [cw]
    else:
        return instant_runoff_put(profile, curr_cands=curr_cands)

def compose(vm1, vm2):
    """After restricting the profile to the set of vm1 winners, run vm2

    Args:
        vm1, vm2 (VotingMethod): The voting methods to be composed.

    Returns:
        A VotingMethod that composes vm1 and vm2.

    :Example:

    .. exec_code::

        from pref_voting.profiles import Profile
        from pref_voting.combined_methods import compose
        from pref_voting.scoring_methods import borda
        from pref_voting.c1_methods import copeland

        prof = Profile([[1, 3, 0, 2], [2, 1, 3, 0], [3, 0, 2, 1]], [1, 2, 1])

        prof.display()

        copeland_borda = compose(copeland, borda)

        copeland.display(prof)
        borda.display(prof)
        copeland_borda.display(prof)

    """

    def _vm(edata, curr_cands=None):

        vm1_ws = vm1(edata, curr_cands=curr_cands)

        return vm2(edata, curr_cands=vm1_ws)

    return VotingMethod(_vm, name=f"{vm1.name}-{vm2.name}")


def _compose(vm1, vm2):
    """
    Same as compose, but used to make it easier to document composed voting methods.
    """

    def _vm(edata, curr_cands=None):

        vm1_ws = vm1(edata, curr_cands=curr_cands)

        return vm2(edata, curr_cands=vm1_ws)

    return _vm

@vm(name = "Condorcet Plurality",
    input_types = [ElectionTypes.PROFILE])
def condorcet_plurality(profile, curr_cands = None):
    """Return the Condorcet winner if one exists, otherwise return the plurality winners.

    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns:
        A sorted list of candidates

    """

    return _compose(condorcet, plurality)(profile, curr_cands=curr_cands)


@vm(name="Smith-Minimax",
    input_types=[ElectionTypes.PROFILE, ElectionTypes.PROFILE_WITH_TIES, ElectionTypes.MARGIN_GRAPH])
def smith_minimax(edata, curr_cands = None):
    """Return the Minimax winner after restricting to the Smith set.

    Args:
        profile (Profile, ProfileWithTies, MarginGraph): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns:
        A sorted list of candidates

    """

    return _compose(top_cycle, minimax)(edata, curr_cands=curr_cands)

@vm(name="Copeland-Local-Borda",
    input_types=[ElectionTypes.PROFILE,  ElectionTypes.MARGIN_GRAPH])
def copeland_local_borda(edata, curr_cands = None):
    """Return the Borda winner after restricting to the Copeland winners.

    Args:
        profile (Profile, MarginGraph): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns:
        A sorted list of candidates

    """
    return _compose(copeland, borda)(edata, curr_cands=curr_cands)

def voting_method_with_scoring_tiebreaker(vm, score, name):

    def _vm(profile, curr_cands=None):

        vm_ws = vm(profile, curr_cands=curr_cands)

        if len(vm_ws) == 1: 
            return vm_ws
        
        # get (restricted) rankings
        _rankings, rcounts = profile.rankings_counts

        cands_to_ignore = np.array([c for c in profile.candidates if c not in curr_cands]) if curr_cands is not None else np.array([])

        rankings = _rankings if curr_cands is None else _find_updated_profile(np.array(_rankings), cands_to_ignore, len(profile.candidates))
        
        curr_cands = profile.candidates if curr_cands is None else curr_cands

        # find the candidate scores using the score function
        cand_scores = {c: sum(_num_rank(rankings, rcounts, c, level) * score(len(curr_cands), level) for level in range(1, len(curr_cands) + 1)) for c in curr_cands}
    
        max_ws_score = max([cand_scores[w] for w in vm_ws])

        return sorted([w for w in vm_ws if cand_scores[w] == max_ws_score])

    return _vm 

@vm(name="Copeland-Global-Borda",
    input_types=[ElectionTypes.PROFILE])
def copeland_global_borda(profile, curr_cands=None):
    """From the Copeland winners, return the candidate with the largest *global* Borda score.

    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns:
        A sorted list of candidates

    """

    return voting_method_with_scoring_tiebreaker(copeland, lambda num_cands, rank : num_cands - rank, "Copeland-Global-Borda")(profile, curr_cands=curr_cands)


@vm(name="Copeland-Global-Minimax",
    input_types=[ElectionTypes.PROFILE, ElectionTypes.PROFILE_WITH_TIES, ElectionTypes.MARGIN_GRAPH])
def copeland_global_minimax(edata, curr_cands=None):
    """From the Copeland winners, return the candidates with the best *global* Minimax score.

    Args:
        edata (Profile, ProfileWithTies, MarginGraph): Any edata with a Margin method.
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns:
        A sorted list of candidates

    """

    curr_cands = edata.candidates if curr_cands is None else curr_cands

    copeland_ws = copeland(edata, curr_cands=curr_cands)

    mm_scores = minimax_scores(edata, curr_cands=curr_cands)

    best_score = max([mm_scores[c] for c in copeland_ws])

    return sorted([c for c in copeland_ws if mm_scores[c] == best_score])

def faceoff(vm1, vm2):
    """If the vm1 and vm2 winners are the same, return that set of winners. Otherwise, for each choice of a vm1 winner A and vm2 winner B, add to the ultimate winners whichever of A or B is majority preferred to the other (or both if they are tied).

    Args:
        vm1, vm2 (VotingMethod): The voting methods to faceoff.

    Returns:
        A VotingMethod that runs the faceoff of vm1 and vm2.

    """

    def _vm(edata, curr_cands=None):

        curr_cands = edata.candidates if curr_cands is None else curr_cands

        vm1_winners = vm1(edata, curr_cands)
        vm2_winners = vm2(edata, curr_cands)

        if vm1_winners == vm2_winners:
            return vm1_winners
        
        else:
            winners = list()

            for a in vm1_winners:
                for b in vm2_winners:
                    if edata.margin(a,b) > 0:
                        winners.append(a)
                    elif edata.margin(b,a) > 0:
                        winners.append(b)
                    elif edata.margin(a,b) == 0:
                        winners.append(a)
                        winners.append(b) 

            return list(set(winners))

    return VotingMethod(_vm, name=f"{vm1.name}-{vm2.name} Faceoff")

def _faceoff(vm1, vm2):
    """
    Same as faceoff, but used to make it easier to document faceoff voting methods.
    """

    def _vm(edata, curr_cands=None):

        curr_cands = edata.candidates if curr_cands is None else curr_cands

        vm1_winners = vm1(edata, curr_cands)
        vm2_winners = vm2(edata, curr_cands)

        if vm1_winners == vm2_winners:
            return vm1_winners
        
        else:
            winners = list()

            for a in vm1_winners:
                for b in vm2_winners:
                    if edata.margin(a,b) > 0:
                        winners.append(a)
                    elif edata.margin(b,a) > 0:
                        winners.append(b)
                    elif edata.margin(a,b) == 0:
                        winners.append(a)
                        winners.append(b) 

            return list(set(winners))

    return _vm

@vm(name="Borda-Minimax Faceoff",
    input_types=[ElectionTypes.PROFILE])
def borda_minimax_faceoff(edata, curr_cands=None):
    """If the Borda and Minimax winners are the same, return that set of winners. Otherwise, for each choice of a Borda winner A and Minimax winner B, add to the ultimate winners whichever of A or B is majority preferred to the other (or both if they are tied).

    Args:
        profile (Profile, MarginGraph): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns:
        A sorted list of candidates

    ..note:
        Proposed by Edward B. Foley.

    """

    return _faceoff(borda, minimax)(edata, curr_cands=curr_cands)