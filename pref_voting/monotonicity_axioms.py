"""
    File: monotonicity_axioms.py
    Author: Wes Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)
    Date: November 4, 2023
    
    Monotonicity axioms 
"""

from pref_voting.axiom import Axiom
from pref_voting.axiom_helpers import *
import numpy as np
from itertools import product
import copy

def one_rank_lift(ranking, c):
    """
    Return a ranking in which ``c`` is moved up one position in ``ranking``.
    """
    assert c != ranking[0], "can't lift a candidate already in first place"
    
    new_ranking = copy.deepcopy(ranking)
    c_idx = new_ranking.index(c)
    new_ranking[c_idx - 1], new_ranking[c_idx] = new_ranking[c_idx], new_ranking[c_idx-1]
    return new_ranking

def one_rank_drop(ranking, c):
    """
    Return a ranking in which ``c`` is moved down one position in ``ranking``.
    """
    assert c != ranking[-1], "can't drop a candidate already in last place"
    
    new_ranking = copy.deepcopy(ranking)
    c_idx = new_ranking.index(c)
    new_ranking[c_idx + 1], new_ranking[c_idx] = new_ranking[c_idx], new_ranking[c_idx+1]
    return new_ranking

def has_one_rank_monotonicity_violation(profile, vm, verbose = False, violation_type="Lift"): 
    """
    If violation_type = "Lift", returns True if there is some winning candidate A and some voter v such that lifting A up one position in v's ranking causes A to lose.

    If violation_type = "Drop", returns True if there is some losing candidate A and some voter v such that dropping A down one position in v's ranking causes A to win.
    
    Args:
        profile: a Profile object.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation. 
        violation_type: default is "Lift"

    Returns: 
        Result of the test (bool): Returns True if there is a violation and False otherwise. 

    ..note:
        If a voting method violates monotonicity, then it violates one-rank monotonicity, so this function is sufficient for testing whether a method violates monotonicity (though not for testing the frequency of monotonicity violations).

    """
    
    _rankings, _rcounts = profile.rankings_counts

    rankings = [list(r) for r in list(_rankings)]
    rcounts = list(_rcounts)
    old_rankings = copy.deepcopy(rankings)

    ws = vm(profile)

    if violation_type == "Lift":
        for w in ws: 
            for r_idx, r in enumerate(rankings): 
                if r[0] != w:
                    old_ranking = copy.deepcopy(r)
                    new_ranking = one_rank_lift(r, w)
                    new_rankings = old_rankings + [new_ranking]
                    new_rcounts  = copy.deepcopy(rcounts + [1])
                    new_rcounts[r_idx] -= 1
                    new_prof = Profile(new_rankings, new_rcounts)
                    new_ws = vm(new_prof)
                    if w not in new_ws: 
                        if verbose: 
                            print(f"One-rank monotonicity violation for {vm.name} by lifting {w}:")
                            profile.display()
                            print(f"{vm.name} winners: ", ws)
                            print("Original ranking: ", old_ranking)
                            print(f"New ranking: {new_ranking}")
                            new_prof.display()
                            print(f"{vm.name} winners in updated profile:", new_ws)
                        return True
                    
    elif violation_type == "Drop":
        for l in profile.candidates:
            if l not in ws:
                for r_idx, r in enumerate(rankings): 
                    if r[-1] != l:
                        old_ranking = copy.deepcopy(r)
                        new_ranking = one_rank_drop(r, l)
                        new_rankings = old_rankings + [new_ranking]
                        new_rcounts  = copy.deepcopy(rcounts + [1])
                        new_rcounts[r_idx] -= 1
                        new_prof = Profile(new_rankings, new_rcounts)
                        new_ws = vm(new_prof)
                        if l in new_ws: 
                            if verbose: 
                                print(f"One-rank monotonicity violation for {vm.name} by dropping {l}:")
                                profile.display()
                                print(f"{vm.name} winners: ", ws)
                                print("Original ranking: ", old_ranking)
                                print(f"New ranking: {new_ranking}")
                                new_prof.display()
                                print(f"{vm.name} winners in updated profile: ", new_ws)
                            return True

    return False

def find_all_one_rank_monotonicity_violations(profile, vm, verbose = False, violation_type="Lift"):
    """
    If violation_type = "Lift", returns all pairs (candidate, ranking) such that the candidate wins in the original profile but loses after lifting the candidate up one position in the ranking.

    If violation_type = "Drop", returns all pairs (candidate, ranking) such that the candidate loses in the original profile but wins after dropping the candidate down one position in the ranking.

    Args:
        profile: a Profile object.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation. 
        violation_type: default is "Lift"

    Returns:
        A list of pairs (candidate, ranking) witnessing violations of one-rank monotonicity.

    """

    _rankings, _rcounts = profile.rankings_counts

    rankings = [list(r) for r in list(_rankings)]
    rcounts = list(_rcounts)
    old_rankings = copy.deepcopy(rankings)

    ws = vm(profile)
    witnesses = list()

    if violation_type == "Lift":
        for w in ws: 
            for r_idx, r in enumerate(rankings): 
                if r[0] != w:
                    old_ranking = copy.deepcopy(r)
                    new_ranking = one_rank_lift(r, w)
                    new_rankings = old_rankings + [new_ranking]
                    new_rcounts  = copy.deepcopy(rcounts + [1])
                    new_rcounts[r_idx] -= 1
                    new_prof = Profile(new_rankings, new_rcounts)
                    new_ws = vm(new_prof)
                    if w not in new_ws: 
                        witnesses.append((w, old_ranking))
                        if verbose: 
                            print(f"One-rank monotonicity violation for {vm.name} by lifting {w}:")
                            profile.display()
                            print(f"{vm.name} winners: ", ws)
                            print("Original ranking ", old_ranking)
                            print(f"New ranking: {new_ranking}")
                            new_prof.display()
                            print(f"{vm.name} winners in updated profile: ", new_ws)

    elif violation_type == "Drop":
        for l in profile.candidates:
            if l not in ws:
                for r_idx, r in enumerate(rankings): 
                    if r[-1] != l:
                        old_ranking = copy.deepcopy(r)
                        new_ranking = one_rank_drop(r, l)
                        new_rankings = old_rankings + [new_ranking]
                        new_rcounts  = copy.deepcopy(rcounts + [1])
                        new_rcounts[r_idx] -= 1
                        new_prof = Profile(new_rankings, new_rcounts)
                        new_ws = vm(new_prof)
                        if l in new_ws: 
                            witnesses.append((l, old_ranking))
                            if verbose: 
                                print(f"One-rank monotonicity violation for {vm.name} by dropping {l}:")
                                profile.display()
                                print(f"{vm.name} winners: ", ws)
                                print("Original ranking: ", old_ranking)
                                print(f"New ranking: {new_ranking}")
                                new_prof.display()
                                print(f"{vm.name} winners in updated profile: ", new_ws)

    return witnesses
                    
one_rank_monotonicity = Axiom(
    "One-Rank Monotonicity",
    has_violation = has_one_rank_monotonicity_violation,
    find_all_violations = find_all_one_rank_monotonicity_violations,
)
                    
monotonicity_axioms = [
    one_rank_monotonicity
]