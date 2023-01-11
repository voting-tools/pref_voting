"""
    File: iterative_methods.py
    Author: Eric Pacuit (epacuit@umd.edu)
    Date: January 6, 2022
    
    Implementations of voting methods that combine multiple methods
"""

from pref_voting.voting_method import *
from pref_voting.scoring_methods import plurality, borda
from pref_voting.iterative_methods import iterated_removal_cl, instant_runoff, instant_runoff_put, instant_runoff_for_truncated_linear_orders
from pref_voting.c1_methods import smith_set, copeland
from pref_voting.profiles import Profile
from pref_voting.profiles_with_ties import ProfileWithTies

@vm(name="Daunou")
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


@vm(name="Blacks")
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

vm(name="Condorcet IRV")
def condorcet_irv(prof, curr_cands = None): 
    """If there is a Condorcet winner, then the Condorcet winner is the winner.  Otherwise, return the Instant Runoff winners.

    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates, or a profile of truncated linear orders.
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns:
        A sorted list of candidates.

    """
    cw = prof.condorcet_winner(curr_cands = curr_cands)
    if cw is not None: 
        return [cw]
    
    if type(prof) == Profile: 
        return instant_runoff(prof, curr_cands = curr_cands)
    elif type(prof) == ProfileWithTies: 
        return instant_runoff_for_truncated_linear_orders(prof, curr_cands = curr_cands)

vm(name="Condorcet IRV PUT")
def condorcet_irv_put(prof, curr_cands = None): 
    """If there is a Condorcet winner, then the Condorcet winner is the winner.  Otherwise, return the Instant Runoff PUT winners.

    Args:
        profile (Profile): An anonymous profile of linear orders.
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns:
        A sorted list of candidates.

    """
    cw = prof.condorcet_winner(curr_cands = curr_cands)
    if cw is not None: 
        return [cw]
    
    return instant_runoff_put(prof, curr_cands = curr_cands)

@vm(name="Smith IRV")
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

@vm(name="Smith IRV PUT")
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


def compose(vm1, vm2):
    """After restricting the profile to the set of vm1 winners, run vm2

    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns:
        A sorted list of candidates

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

copeland_borda = compose(copeland, borda)

combined_vms = [
    daunou, 
    blacks, 
    condorcet_irv, 
    condorcet_irv_put, 
    smith_irv, 
    smith_irv_put, 
    copeland_borda
    ]
