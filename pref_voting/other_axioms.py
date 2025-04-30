"""
File: other_axioms.py
Date: April 29 2025
Authors: Wes Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)

Other axioms
---------------------
"""

from itertools import repeat
import numpy as np

from pref_voting.axiom import Axiom
from pref_voting.profiles import Profile
from pref_voting.profiles_with_ties import ProfileWithTies
from pref_voting.rankings import Ranking


def _reverse_ranking(ballot, all_cands):
    """
    Reverse a single ballot, treating *unranked* candidates as
    tied for last before the reversal.

    Parameters
    ----------
    ballot : tuple | Ranking
        • tuple  – a strict linear order
        • Ranking – weak order, possibly incomplete
    all_cands : list | set
        The full candidate set of the profile.
    """
    if isinstance(ballot, tuple):                 
        return tuple(reversed(ballot))              

    if isinstance(ballot, Ranking):
        full_rmap = ballot.rmap.copy()
        if full_rmap:
            last_rank = max(full_rmap.values()) + 1
        else:                                        
            last_rank = 1
        for c in all_cands:
            if c not in full_rmap:
                full_rmap[c] = last_rank

        max_rank = max(full_rmap.values())
        rev_rmap = {c: max_rank + 1 - k for c, k in full_rmap.items()}
        return Ranking(rev_rmap)

def _reverse_profile(P):
    """
    Build the reversed profile ``Pᵣ``.
    """
    all_cands = P.candidates
    rev_ballots = [_reverse_ranking(b, all_cands) for b in P.rankings]

    if isinstance(P, Profile):
        return Profile(rev_ballots)

    if isinstance(P, ProfileWithTies):
        P_r = ProfileWithTies(rev_ballots, candidates=all_cands)
        if P.using_extended_strict_preference:
            P_r.use_extended_strict_preference()
        return P_r
    
def _reverse_margin_graph(mg):
    """
    Return the *edge-reversed* margin graph.

    All positive-margin edges (u, v, w) become (v, u, w); weights are preserved.
    """
    rev_edges = [(v, u, w) for (u, v, w) in mg.edges]
    return MarginGraph(mg.candidates[:], rev_edges, cmap=mg.cmap)

def has_reversal_symmetry_violation(edata, vm, verbose=False):
    """
    Returns True iff ``vm`` violates reversal symmetry on *edata*.

    Reversal Symmetry states that if x is a **unique** winner in ``edata``, 
    then x should not be among the winners in the reversal of ``edata``.
    """

    if len(edata.candidates) <= 1:
        return False

    if isinstance(edata, MarginGraph):
        mg = edata
        winners = vm(mg)
        if len(winners) != 1:
            return False

        x = winners[0]
        mg_r = _reverse_margin_graph(mg)
        rev_winners = vm(mg_r)

        if x in rev_winners:
            if verbose:
                print(f"Reversal-symmetry violation for {vm.name} on a MarginGraph")
                print(f"Unique winner {x} also wins after edge reversal.")
                print("\nOriginal margin graph:")
                mg.display()
                print(mg.description())
                vm.display(mg)
                print('\nReversed margin graph:')
                mg_r.display()
                print(mg_r.description())
                vm.display(mg_r)

            return True
        return False

    winners = vm(edata)
    if isinstance(winners, np.ndarray):
        winners = winners.tolist()
    if len(winners) != 1:
        return False

    x = winners[0]
    P_r = _reverse_profile(edata)
    rev_winners = vm(P_r)
    if isinstance(rev_winners, np.ndarray):
        rev_winners = rev_winners.tolist()

    if x in rev_winners:
        if verbose:
            print(f"Reversal-symmetry violation for {vm.name}")
            print(f"Unique winner {x} also wins after reversal.")
            print("\nOriginal profile:")
            edata.display()
            print(edata.description())
            vm.display(edata)
            print("\nReversed profile:")
            P_r.display()
            print(P_r.description())
            vm.display(P_r)
        return True
    return False


def find_all_reversal_symmetry_violations(edata, vm, verbose=False):
    """
    Returns a one-item list [(unique_winner, winners_after_reversal)] describing the violation on *edata*, or [].
    """
    if not has_reversal_symmetry_violation(edata, vm, verbose):
        return []

    winners = vm(edata)
    winners = winners.tolist() if isinstance(winners, np.ndarray) else winners
    if isinstance(edata, MarginGraph):
        rev_winners = vm(_reverse_margin_graph(edata))
    else:
        rev_winners = vm(_reverse_profile(edata))

    rev_winners = rev_winners.tolist() if isinstance(rev_winners, np.ndarray) else rev_winners
    return [(winners, rev_winners)]


reversal_symmetry = Axiom(
    "Reversal Symmetry",
    has_violation=has_reversal_symmetry_violation,
    find_all_violations=find_all_reversal_symmetry_violations,
)