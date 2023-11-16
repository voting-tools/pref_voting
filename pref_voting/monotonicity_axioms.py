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

def has_one_rank_monotonicity_violation(profile, vm, verbose = False, violation_type="Lift", check_probabilities = False): 
    """
    If violation_type = "Lift", returns True if there is some winning candidate A and some voter v such that lifting A up one position in v's ranking causes A to lose.

    If violation_type = "Drop", returns True if there is some losing candidate A and some voter v such that dropping A down one position in v's ranking causes A to win.

    If checking_probabilities = True, returns True if there is some candidate whose probabilities of winning decreases after a lifting or increases after a dropping.
    
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
                            print(profile.description())
                            profile.display_margin_graph()
                            print(f"{vm.name} winners: ", ws)
                            print("Original ranking: ", old_ranking)
                            print(f"New ranking: {new_ranking}")
                            new_prof.display()
                            print(new_prof.description())
                            new_prof.display_margin_graph()
                            print(f"{vm.name} winners in updated profile:", new_ws)
                        return True
                    
                    if w in new_ws and check_probabilities == True and len(new_ws) > len(ws):
                        if verbose: 
                            print(f"One-rank probabilistic monotonicity violation for {vm.name} by lifting {w}:")
                            profile.display()
                            print(profile.description())
                            profile.display_margin_graph()
                            print(f"{vm.name} winners: ", ws)
                            print("Original ranking: ", old_ranking)
                            print(f"New ranking: {new_ranking}")
                            new_prof.display()
                            print(new_prof.description())
                            new_prof.display_margin_graph()
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
                                print(profile.description())
                                profile.display_margin_graph()
                                print(f"{vm.name} winners: ", ws)
                                print("Original ranking: ", old_ranking)
                                print(f"New ranking: {new_ranking}")
                                new_prof.display()
                                print(new_prof.description())
                                new_prof.display_margin_graph()
                                print(f"{vm.name} winners in updated profile: ", new_ws)
                            return True
                        
            if check_probabilities and l in ws:
                for r_idx, r in enumerate(rankings): 
                    if r[-1] != l:
                        old_ranking = copy.deepcopy(r)
                        new_ranking = one_rank_drop(r, l)
                        new_rankings = old_rankings + [new_ranking]
                        new_rcounts  = copy.deepcopy(rcounts + [1])
                        new_rcounts[r_idx] -= 1
                        new_prof = Profile(new_rankings, new_rcounts)
                        new_ws = vm(new_prof)
                        if l in new_ws and len(new_ws) < len(ws): 
                            if verbose: 
                                print(f"One-rank probabilistic monotonicity violation for {vm.name} by dropping {l}:")
                                profile.display()
                                print(profile.description())
                                profile.display_margin_graph()
                                print(f"{vm.name} winners: ", ws)
                                print("Original ranking: ", old_ranking)
                                print(f"New ranking: {new_ranking}")
                                new_prof.display()
                                print(new_prof.description())
                                new_prof.display_margin_graph()
                                print(f"{vm.name} winners in updated profile: ", new_ws)
                            return True

    return False

def find_all_one_rank_monotonicity_violations(profile, vm, verbose = False, violation_type="Lift", check_probabilities = False):
    """
    If violation_type = "Lift", returns all pairs (candidate, ranking) such that the candidate wins in the original profile but loses after lifting the candidate up one position in the ranking.

    If violation_type = "Drop", returns all pairs (candidate, ranking) such that the candidate loses in the original profile but wins after dropping the candidate down one position in the ranking.

    If checking_probabilities = True, returns all pairs (candidate, ranking) such that the candidate's probability of winning decreases after a lifting or increases after a dropping.

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
                        witnesses.append((w, old_ranking, "Lift"))
                        if verbose: 
                            print(f"One-rank monotonicity violation for {vm.name} by lifting {w}:")
                            profile.display()
                            print(profile.description())
                            profile.display_margin_graph()
                            print(f"{vm.name} winners: ", ws)
                            print("Original ranking ", old_ranking)
                            print(f"New ranking: {new_ranking}")
                            new_prof.display()
                            print(new_prof.description())
                            new_prof.display_margin_graph()
                            print(f"{vm.name} winners in updated profile: ", new_ws)

                    if w in new_ws and check_probabilities == True and len(new_ws) > len(ws):
                        witnesses.append((w, old_ranking, "Lift"))
                        if verbose: 
                            print(f"One-rank probabilistic monotonicity violation for {vm.name} by lifting {w}:")
                            profile.display()
                            print(profile.description())
                            profile.display_margin_graph()
                            print(f"{vm.name} winners: ", ws)
                            print("Original ranking: ", old_ranking)
                            print(f"New ranking: {new_ranking}")
                            new_prof.display()
                            print(new_prof.description())
                            new_prof.display_margin_graph()
                            print(f"{vm.name} winners in updated profile:", new_ws)

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
                            witnesses.append((l, old_ranking, "Drop"))
                            if verbose: 
                                print(f"One-rank monotonicity violation for {vm.name} by dropping {l}:")
                                profile.display()
                                print(profile.description())
                                profile.display_margin_graph()
                                print(f"{vm.name} winners: ", ws)
                                print("Original ranking: ", old_ranking)
                                print(f"New ranking: {new_ranking}")
                                new_prof.display()
                                print(new_prof.description())
                                new_prof.display_margin_graph()
                                print(f"{vm.name} winners in updated profile: ", new_ws)

            if check_probabilities and l in ws:
                for r_idx, r in enumerate(rankings): 
                    if r[-1] != l:
                        old_ranking = copy.deepcopy(r)
                        new_ranking = one_rank_drop(r, l)
                        new_rankings = old_rankings + [new_ranking]
                        new_rcounts  = copy.deepcopy(rcounts + [1])
                        new_rcounts[r_idx] -= 1
                        new_prof = Profile(new_rankings, new_rcounts)
                        new_ws = vm(new_prof)
                        if l in new_ws and len(new_ws) < len(ws): 
                            witnesses.append((l, old_ranking, "Drop"))
                            if verbose: 
                                print(f"One-rank probabilistic monotonicity violation for {vm.name} by dropping {l}:")
                                profile.display()
                                print(profile.description())
                                profile.display_margin_graph()
                                print(f"{vm.name} winners: ", ws)
                                print("Original ranking: ", old_ranking)
                                print(f"New ranking: {new_ranking}")
                                new_prof.display()
                                print(new_prof.description())
                                new_prof.display_margin_graph()
                                print(f"{vm.name} winners in updated profile: ", new_ws)

    return witnesses
                    
one_rank_monotonicity = Axiom(
    "One-Rank Monotonicity",
    has_violation = has_one_rank_monotonicity_violation,
    find_all_violations = find_all_one_rank_monotonicity_violations,
)

def lift_to_first(ranking, c):
    """
    Return a ranking in which ``c`` is moved to first position in ``ranking``.
    """
    assert c != ranking[0], "can't lift a candidate already in first place"
    
    new_ranking = copy.deepcopy(ranking)
    c_idx = new_ranking.index(c)
    new_ranking = [c] + new_ranking[:c_idx] + new_ranking[c_idx+1:]

    return new_ranking

def drop_to_last(ranking, c):
    """
    Return a ranking in which ``c`` is moved to last position in ``ranking``.
    """
    assert c != ranking[-1], "can't drop a candidate already in last place"
    
    new_ranking = copy.deepcopy(ranking)
    c_idx = new_ranking.index(c)
    new_ranking = new_ranking[:c_idx] + new_ranking[c_idx+1:] + [c]

    return new_ranking

def has_weak_positive_responsiveness_violation(profile, vm, verbose = False, violation_type="Lift"): 
    """
    If violation_type = "Lift", returns True if there is some winning candidate A and some voter v who ranks A last such that v moving A into first place does not make A the unique winner

    If violation_type = "Drop", returns True if there is some candidate A who is either a loser or a non-unique winner and some voter v who ranks A first such that v moving A into last place does not make A a loser.
    
    Args:
        profile: a Profile object.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation. 
        violation_type: default is "Lift"

    Returns: 
        Result of the test (bool): Returns True if there is a violation and False otherwise. 

    """
    
    _rankings, _rcounts = profile.rankings_counts

    rankings = [list(r) for r in list(_rankings)]
    rcounts = list(_rcounts)
    old_rankings = copy.deepcopy(rankings)

    ws = vm(profile)

    if violation_type == "Lift":
        for w in ws: 
            for r_idx, r in enumerate(rankings): 
                if r[-1] == w:
                    old_ranking = copy.deepcopy(r)
                    new_ranking = lift_to_first(r, w)
                    new_rankings = old_rankings + [new_ranking]
                    new_rcounts  = copy.deepcopy(rcounts + [1])
                    new_rcounts[r_idx] -= 1
                    new_prof = Profile(new_rankings, new_rcounts)
                    new_ws = vm(new_prof)
                    if len(new_ws) > 1 or (len(new_ws) == 1 and new_ws[0] != w):
                        if verbose: 
                            print(f"Weak positive responsiveness violation for {vm.name} by lifting {w}:")
                            profile.display()
                            print(profile.description())
                            profile.display_margin_graph()
                            print(f"{vm.name} winners: ", ws)
                            print("Original ranking: ", old_ranking)
                            print(f"New ranking: {new_ranking}")
                            new_prof.display()
                            print(new_prof.description())
                            new_prof.display_margin_graph()
                            print(f"{vm.name} winners in updated profile:", new_ws)
                        return True
                    
    elif violation_type == "Drop":
        for l in profile.candidates:
            if l not in ws or (l in ws and len(ws) > 1):
                for r_idx, r in enumerate(rankings): 
                    if r[0] == l:
                        old_ranking = copy.deepcopy(r)
                        new_ranking = drop_to_last(r, l)
                        new_rankings = old_rankings + [new_ranking]
                        new_rcounts  = copy.deepcopy(rcounts + [1])
                        new_rcounts[r_idx] -= 1
                        new_prof = Profile(new_rankings, new_rcounts)
                        new_ws = vm(new_prof)
                        if l in new_ws: 
                            if verbose: 
                                print(f"Weak positive responsiveness violation for {vm.name} by dropping {l}:")
                                profile.display()
                                print(profile.description())
                                profile.display_margin_graph()
                                print(f"{vm.name} winners: ", ws)
                                print("Original ranking: ", old_ranking)
                                print(f"New ranking: {new_ranking}")
                                new_prof.display()
                                print(new_prof.description())
                                new_prof.display_margin_graph()
                                print(f"{vm.name} winners in updated profile: ", new_ws)
                            return True

    return False

def find_all_weak_positive_responsiveness_violations(profile, vm, verbose = False, violation_type="Lift"):
    """
    If violation_type = "Lift", returns all pairs (candidate, ranking) such that the candidate is a unique winner in the original profile but is not a unique winner after the voter moves the candidate from last to first place in the ranking.

    If violation_type = "Drop", returns all pairs (candidate, ranking) such that the candidate is either a loser or a non-unique winner in the original profile but is a  winner after the voter moves the candidate from first to last place in the ranking.

    Args:
        profile: a Profile object.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation. 
        violation_type: default is "Lift"

    Returns:
        A list of pairs (candidate, ranking) witnessing violations of weak positive responsiveness.

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
                if r[-1] == w:
                    old_ranking = copy.deepcopy(r)
                    new_ranking = lift_to_first(r, w)
                    new_rankings = old_rankings + [new_ranking]
                    new_rcounts  = copy.deepcopy(rcounts + [1])
                    new_rcounts[r_idx] -= 1
                    new_prof = Profile(new_rankings, new_rcounts)
                    new_ws = vm(new_prof)
                    if len(new_ws) > 1 or (len(new_ws) == 1 and new_ws[0] != w):
                        witnesses.append((w, old_ranking, "Lift"))
                        if verbose: 
                            print(f"Weak positive responsiveness violation for {vm.name} by lifting {w}:")
                            profile.display()
                            print(profile.description())
                            profile.display_margin_graph()
                            print(f"{vm.name} winners: ", ws)
                            print("Original ranking ", old_ranking)
                            print(f"New ranking: {new_ranking}")
                            new_prof.display()
                            print(new_prof.description())
                            new_prof.display_margin_graph()
                            print(f"{vm.name} winners in updated profile: ", new_ws)

    elif violation_type == "Drop":
        for l in profile.candidates:
            if l not in ws or (l in ws and len(ws) > 1):
                for r_idx, r in enumerate(rankings): 
                    if r[0] == l:
                        old_ranking = copy.deepcopy(r)
                        new_ranking = drop_to_last(r, l)
                        new_rankings = old_rankings + [new_ranking]
                        new_rcounts  = copy.deepcopy(rcounts + [1])
                        new_rcounts[r_idx] -= 1
                        new_prof = Profile(new_rankings, new_rcounts)
                        new_ws = vm(new_prof)
                        if l in new_ws: 
                            witnesses.append((l, old_ranking, "Drop"))
                            if verbose: 
                                print(f"Weak positive responsiveness violation for {vm.name} by dropping {l}:")
                                profile.display()
                                print(profile.description())
                                profile.display_margin_graph()
                                print(f"{vm.name} winners: ", ws)
                                print("Original ranking: ", old_ranking)
                                print(f"New ranking: {new_ranking}")
                                new_prof.display()
                                print(new_prof.description())
                                new_prof.display_margin_graph()
                                print(f"{vm.name} winners in updated profile: ", new_ws)
    
    return witnesses

weak_positive_responsiveness = Axiom(
    "Weak Positive Responsiveness",
    has_violation = has_weak_positive_responsiveness_violation,
    find_all_violations = find_all_weak_positive_responsiveness_violations,
)
                   
monotonicity_axioms = [
    one_rank_monotonicity,
    weak_positive_responsiveness
]