"""
    File: monotonicity_axioms.py
    Author: Wes Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)
    Date: November 4, 2023
    
    Monotonicity axioms 
"""

from pref_voting.axiom import Axiom
from pref_voting.axiom_helpers import *
from pref_voting.rankings import Ranking
import numpy as np
from itertools import product
import copy

def ranks_above(ranking,c):
    """
    Returns the number of positions above candidate ``c`` in ``ranking``, taking into account ties in Ranking objects.
    """
    if isinstance(ranking, tuple):

        return ranking.index(c)

    if isinstance(ranking, Ranking):

        ranking.normalize_ranks()

        if any([ranking.rmap[d] == ranking.rmap[c] for d in ranking.cands if c!=d]):
            return 2 * (ranking.rmap[c] - 1) + 1
        else:
            return 2 * (ranking.rmap[c] - 1)
    
def ranks_below(ranking,c):
    """
    Returns the number of positions below candidate ``c`` in ``ranking``, taking into account ties in Ranking objects.
    """
    if isinstance(ranking, tuple):
            
        return len(ranking) - ranking.index(c) - 1
    
    if isinstance(ranking, Ranking):

        ranking.normalize_ranks()

        if any([ranking.rmap[d] == ranking.rmap[c] for d in ranking.cands if c!=d]):
            return 2 * (len(ranking.cands) - ranking.rmap[c]) + 1
        else:
            return 2 * (len(ranking.cands) - ranking.rmap[c])

def n_rank_lift(ranking, c, n):
    """
    Return a ranking in which ``c`` is moved up n positions in ``ranking``.
    """
    if isinstance(ranking, tuple):
        assert c not in ranking[:n], f"there are not enough ranks above {c} to lift {c} {n} ranks"
        _new_ranking = copy.deepcopy(ranking)
        c_idx = _new_ranking.index(c)
        new_ranking = _new_ranking[:c_idx-n] + (_new_ranking[c_idx],) + _new_ranking[c_idx-n:c_idx] + _new_ranking[c_idx+1:]
    
    if isinstance(ranking, Ranking):

        ranking.normalize_ranks()

        assert ranks_above(ranking,c) >= n, f"there are not enough ranks above {c} to lift {c} {n} ranks"

        new_ranking_dict = dict()

        if any([ranking.rmap[d] == ranking.rmap[c] for d in ranking.cands if c!=d]): # if c is tied with another candidate

            if n%2 == 0: 
                for d in ranking.cands:
                    if d == c:
                        new_ranking_dict[d] = ranking.rmap[d] - (n/2)
                    else:
                        new_ranking_dict[d] = ranking.rmap[d]

            else:
                for d in ranking.cands:
                    if d == c:
                        new_ranking_dict[d] = ranking.rmap[d] - (math.floor(n/2) + math.ceil(n/2))/2
                    else:
                        new_ranking_dict[d] = ranking.rmap[d]
                
            new_ranking = Ranking(new_ranking_dict)

        else:
            if n%2 == 0:
                for d in ranking.cands:
                    if d == c:
                        new_ranking_dict[d] = ranking.rmap[d] - ((n//2) + (n//2 + 1))/2
                    else:
                        new_ranking_dict[d] = ranking.rmap[d]
            else:
                for d in ranking.cands:
                    if d == c:
                        new_ranking_dict[d] = ranking.rmap[d] - math.ceil(n/2)
                    else:
                        new_ranking_dict[d] = ranking.rmap[d]

            new_ranking = Ranking(new_ranking_dict)

        new_ranking.normalize_ranks()

    return new_ranking

def n_rank_drop(ranking, c, n):
    """
    Return a ranking in which ``c`` is moved down n positions in ``ranking``.
    """
    if isinstance(ranking, tuple):
        # assert that there are n ranks below c
        assert len(ranking) - ranking.index(c) - 1 >= n, f"there are not enough ranks below {c} to drop {c} {n} ranks"
        _new_ranking = copy.deepcopy(ranking)
        c_idx = _new_ranking.index(c)
        new_ranking = _new_ranking[:c_idx] + _new_ranking[c_idx+1:c_idx+n+1] + (_new_ranking[c_idx],) + _new_ranking[c_idx+n+1:]
    
    if isinstance(ranking, Ranking):

        ranking.normalize_ranks()

        assert ranks_below(ranking,c) >= n, f"there are not enough ranks below {c} to drop {c} {n} ranks"

        new_ranking_dict = dict()

        if any([ranking.rmap[d] == ranking.rmap[c] for d in ranking.cands if c!=d]): # if c is tied with another candidate

            if n%2 == 0: 
                for d in ranking.cands:
                    if d == c:
                        new_ranking_dict[d] = ranking.rmap[d] + (n/2)
                    else:
                        new_ranking_dict[d] = ranking.rmap[d]

            else:
                for d in ranking.cands:
                    if d == c:
                        new_ranking_dict[d] = ranking.rmap[d] + (math.floor(n/2) + math.ceil(n/2))/2
                    else:
                        new_ranking_dict[d] = ranking.rmap[d]
                
            new_ranking = Ranking(new_ranking_dict)

        else:
            if n%2 == 0:
                for d in ranking.cands:
                    if d == c:
                        new_ranking_dict[d] = ranking.rmap[d] + ((n//2) + (n//2 + 1))/2
                    else:
                        new_ranking_dict[d] = ranking.rmap[d]
            else:
                for d in ranking.cands:
                    if d == c:
                        new_ranking_dict[d] = ranking.rmap[d] + math.ceil(n/2)
                    else:
                        new_ranking_dict[d] = ranking.rmap[d]

            new_ranking = Ranking(new_ranking_dict)

        new_ranking.normalize_ranks()
    
    return new_ranking

def has_monotonicity_violation(profile, vm, verbose = False, violation_type = "Lift", check_probabilities = False, one_rank_monotonicity = False): 
    """
    If violation_type = "Lift", returns True if there is some winning candidate A and some voter v such that lifting A up some number of positions in v's ranking causes A to lose.

    If violation_type = "Drop", returns True if there is some losing candidate A and some voter v such that dropping A down some number of positions in v's ranking causes A to win.

    If checking_probabilities = True, returns True if there is some candidate whose probability of winning decreases after a lifting or increases after a dropping.

    If one_rank_monotonicity = True, then the function will check lifts/drops of one rank only.
    
    Args:
        profile: a Profile object.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation. 
        violation_type: default is "Lift"

    Returns: 
        Result of the test (bool): Returns True if there is a violation and False otherwise. 

    .. note::
        If a voting method violates monotonicity, then it violates one-rank monotonicity, so setting one_rank_monotonicity = True is sufficient for testing whether a method violates monotonicity (though not for testing the frequency of monotonicity violations).

    """
    
    _rankings, _rcounts = profile.rankings_counts

    if isinstance(profile, Profile):
        rankings = [tuple(r) for r in list(_rankings)]

    if isinstance(profile, ProfileWithTies):
        rankings = _rankings

    rcounts = list(_rcounts)
    old_rankings = copy.deepcopy(rankings)

    ws = vm(profile)

    if violation_type == "Lift":
        for w in ws: 
            for r_idx, r in enumerate(rankings): 

                if isinstance(r, Ranking): # Make sure all candidates are ranked in r
                    r = Ranking({a: r.rmap[a] if a in r.cands else max(r.ranks)+1 for a in profile.candidates})

                if r[0] != w:
                    old_ranking = copy.deepcopy(r)

                    if one_rank_monotonicity:
                        ranks_above_w = 1
                    else:
                        ranks_above_w = ranks_above(r, w)

                    for n in range(1, ranks_above_w+1):

                        new_ranking = n_rank_lift(r, w, n)
                        new_rankings = old_rankings + [new_ranking]
                        new_rcounts  = copy.deepcopy(rcounts + [1])
                        new_rcounts[r_idx] -= 1

                        if isinstance(profile, Profile):
                            new_prof = Profile(new_rankings, new_rcounts)
                        if isinstance(profile, ProfileWithTies):
                            new_prof = ProfileWithTies(new_rankings, new_rcounts)
                            if profile.using_extended_strict_preference:
                                new_prof.use_extended_strict_preference()

                        new_ws = vm(new_prof)
                        
                        if w not in new_ws: 
                            if verbose: 
                                if n==1:
                                    print(f"Monotonicity violation for {vm.name} by lifting {w} one rank:")
                                else:
                                    print(f"Monotonicity violation for {vm.name} by lifting {w} by {n} ranks:")
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
                                if n==1:
                                    print(f"Probabilistic monotonicity violation for {vm.name} by lifting {w} one rank:")
                                else:
                                    print(f"Probabilistic monotonicity violation for {vm.name} by lifting {w} by {n} ranks:")
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

                    if isinstance(r, Ranking): # Make sure all candidates are ranked in r
                        r = Ranking({a: r.rmap[a] if a in r.cands else max(r.ranks)+1 for a in profile.candidates})

                    if r[-1] != l:
                        old_ranking = copy.deepcopy(r)

                        if one_rank_monotonicity:
                            ranks_below_l = 1
                        else:
                            ranks_below_l = ranks_below(r, l)

                        for n in range(1, ranks_below_l+1):
                            
                            new_ranking = n_rank_drop(r, l, n)
                            new_rankings = old_rankings + [new_ranking]
                            new_rcounts  = copy.deepcopy(rcounts + [1])
                            new_rcounts[r_idx] -= 1

                            if isinstance(profile, Profile):
                                new_prof = Profile(new_rankings, new_rcounts)
                            if isinstance(profile, ProfileWithTies):
                                new_prof = ProfileWithTies(new_rankings, new_rcounts)
                                if profile.using_extended_strict_preference:
                                    new_prof.use_extended_strict_preference()

                            new_ws = vm(new_prof)

                            if l in new_ws: 
                                if verbose: 
                                    if n==1:
                                        print(f"Monotonicity violation for {vm.name} by dropping {l} one rank:")
                                    else:
                                        print(f"Monotonicity violation for {vm.name} by dropping {l} by {n} ranks:")
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

                    if isinstance(r, Ranking): # Make sure all candidates are ranked in r
                        r = Ranking({a: r.rmap[a] if a in r.cands else max(r.ranks)+1 for a in profile.candidates})

                    if r[-1] != l:
                        old_ranking = copy.deepcopy(r)

                        if one_rank_monotonicity:
                            ranks_below_l = 1
                        else:
                            ranks_below_l = ranks_below(r, l)

                        new_ranking = one_rank_drop(r, l)
                        new_rankings = old_rankings + [new_ranking]
                        new_rcounts  = copy.deepcopy(rcounts + [1])
                        new_rcounts[r_idx] -= 1

                        if isinstance(profile, Profile):
                            new_prof = Profile(new_rankings, new_rcounts)
                        if isinstance(profile, ProfileWithTies):
                            new_prof = ProfileWithTies(new_rankings, new_rcounts)
                            if profile.using_extended_strict_preference:
                                new_prof.use_extended_strict_preference()

                        new_ws = vm(new_prof)
                        if l in new_ws and len(new_ws) < len(ws): 
                            if verbose: 
                                if n==1:
                                    print(f"Probabilistic monotonicity violation for {vm.name} by dropping {l} one rank:")
                                else:
                                    print(f"Probabilistic monotonicity violation for {vm.name} by dropping {l} by {n} ranks:")
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

def find_all_monotonicity_violations(profile, vm, verbose = False, violation_type = "Lift", check_probabilities = False, one_rank_monotonicity = False):
    """
    If violation_type = "Lift", returns all tuples (candidate, ranking, "Lift", n) such that the candidate wins in the original profile but loses after lifting the candidate up n positions in the ranking.

    If violation_type = "Drop", returns all tuples (candidate, ranking, "Drop", n) such that the candidate loses in the original profile but wins after dropping the candidate down n positions in the ranking.

    If checking_probabilities = True, returns all tuples (candidate, ranking, violation_type, n) such that the candidate's probability of winning decreases after a lifting or increases after a dropping.

    If one_rank_monotonicity = True, then the function will check lifts/drops of one rank only.

    Args:
        profile: a Profile object.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation. 
        violation_type: default is "Lift"

    Returns:
        A list of tuples (candidate, ranking, violation_type, positions lifted/dropped) witnessing violations of monotonicity.

    .. note::
        If a voting method violates monotonicity, then it violates one-rank monotonicity, so setting one_rank_monotonicity = True is sufficient for testing whether a method violates monotonicity (though not for testing the frequency of monotonicity violations).
    """

    _rankings, _rcounts = profile.rankings_counts

    if isinstance(profile, Profile):
        rankings = [tuple(r) for r in list(_rankings)]

    if isinstance(profile, ProfileWithTies):
        rankings = _rankings

    rcounts = list(_rcounts)
    old_rankings = copy.deepcopy(rankings)

    ws = vm(profile)
    witnesses = list()

    if violation_type == "Lift":
        for w in ws: 
            for r_idx, r in enumerate(rankings): 

                if isinstance(r, Ranking): # Make sure all candidates are ranked in r
                    r = Ranking({a: r.rmap[a] if a in r.cands else max(r.ranks)+1 for a in profile.candidates})
                    
                if r[0] != w:
                    old_ranking = copy.deepcopy(r)
                    
                    if one_rank_monotonicity:
                        ranks_above_w = 1
                    else:
                        ranks_above_w = ranks_above(r, w)

                    for n in range(1, ranks_above_w+1):
                        new_ranking = n_rank_lift(r, w, n)
                        new_rankings = old_rankings + [new_ranking]
                        new_rcounts  = copy.deepcopy(rcounts + [1])
                        new_rcounts[r_idx] -= 1

                        if isinstance(profile, Profile):
                            new_prof = Profile(new_rankings, new_rcounts)
                        if isinstance(profile, ProfileWithTies):
                            new_prof = ProfileWithTies(new_rankings, new_rcounts)
                            if profile.using_extended_strict_preference:
                                new_prof.use_extended_strict_preference()

                        new_ws = vm(new_prof)

                        if w not in new_ws: 
                            witnesses.append((w, old_ranking, "Lift", n))
                            if verbose: 
                                if n==1:
                                    print(f"Monotonicity violation for {vm.name} by lifting {w} one rank:")
                                else:
                                    print(f"Monotonicity violation for {vm.name} by lifting {w} by {n} ranks:")
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
                                print("")

                        if w in new_ws and check_probabilities == True and len(new_ws) > len(ws):
                            witnesses.append((w, old_ranking, "Lift", n))
                            if verbose: 
                                if n==1:
                                    print(f"Probabilistic monotonicity violation for {vm.name} by lifting {w} one rank:")
                                else:
                                    print(f"Probabilistic monotonicity violation for {vm.name} by lifting {w} by {n} ranks:")
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
                                print("")

    elif violation_type == "Drop":
        for l in profile.candidates:
            if l not in ws:
                for r_idx, r in enumerate(rankings): 

                    if isinstance(r, Ranking): # Make sure all candidates are ranked in r
                        r = Ranking({a: r.rmap[a] if a in r.cands else max(r.ranks)+1 for a in profile.candidates})

                    if r[-1] != l:
                        old_ranking = copy.deepcopy(r)
                        
                        if one_rank_monotonicity:
                            ranks_below_l = 1
                        else:
                            ranks_below_l = ranks_below(r, l)

                        for n in range(1, ranks_below_l+1):
                            new_ranking = n_rank_drop(r, l, n)
                            new_rankings = old_rankings + [new_ranking]
                            new_rcounts  = copy.deepcopy(rcounts + [1])
                            new_rcounts[r_idx] -= 1

                            if isinstance(profile, Profile):
                                new_prof = Profile(new_rankings, new_rcounts)
                            if isinstance(profile, ProfileWithTies):
                                new_prof = ProfileWithTies(new_rankings, new_rcounts)
                                if profile.using_extended_strict_preference:
                                    new_prof.use_extended_strict_preference()

                            new_ws = vm(new_prof)

                            if l in new_ws: 
                                witnesses.append((l, old_ranking, "Drop", n))
                                if verbose: 
                                    if n==1:
                                        print(f"Monotonicity violation for {vm.name} by dropping {l} one rank:")
                                    else:
                                        print(f"Monotonicity violation for {vm.name} by dropping {l} by {n} ranks:")
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
                                    print("")

            if check_probabilities and l in ws:
                for r_idx, r in enumerate(rankings): 

                    if isinstance(r, Ranking): # Make sure all candidates are ranked in r
                        r = Ranking({a: r.rmap[a] if a in r.cands else max(r.ranks)+1 for a in profile.candidates})

                    if r[-1] != l:
                        old_ranking = copy.deepcopy(r)
                        
                        if one_rank_monotonicity:
                            ranks_below_l = 1
                        else:
                            ranks_below_l = ranks_below(r, l)

                        for n in range(1, ranks_below_l+1):
                            new_ranking = n_rank_drop(r, l, n)
                            new_rankings = old_rankings + [new_ranking]
                            new_rcounts  = copy.deepcopy(rcounts + [1])
                            new_rcounts[r_idx] -= 1

                            if isinstance(profile, Profile):
                                new_prof = Profile(new_rankings, new_rcounts)
                            if isinstance(profile, ProfileWithTies):
                                new_prof = ProfileWithTies(new_rankings, new_rcounts)
                                if profile.using_extended_strict_preference:
                                    new_prof.use_extended_strict_preference()

                            new_ws = vm(new_prof)

                            if l in new_ws and len(new_ws) < len(ws): 
                                witnesses.append((l, old_ranking, "Drop", n))
                                if verbose: 
                                    if n==1:
                                        print(f"Probabilistic monotonicity violation for {vm.name} by dropping {l} one rank:")
                                    else:
                                        print(f"Probabilistic monotonicity violation for {vm.name} by dropping {l} by {n} ranks:")
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
                                    print("")

    return witnesses
                    
monotonicity = Axiom(
    "Monotonicity",
    has_violation = has_monotonicity_violation,
    find_all_violations = find_all_monotonicity_violations,
)

def lift_to_first(ranking, c):
    """
    Return a ranking in which ``c`` is moved to first position in ``ranking``.
    """

    if isinstance(ranking, tuple):
        assert c != ranking[0], "can't lift a candidate already in first place"
        new_ranking = copy.deepcopy(ranking)
        c_idx = new_ranking.index(c)
        new_ranking = (c,) + new_ranking[:c_idx] + new_ranking[c_idx+1:]

    if isinstance(ranking, Ranking):
        assert not ranking.first() == [c], "can't lift a candidate already uniquely in first place"
        new_ranking = Ranking({a: ranking.rmap[a] if a !=c else min(ranking.ranks) - 1 for a in ranking.cands})
        new_ranking.normalize_ranks()

    return new_ranking

def drop_to_last(ranking, c):
    """
    Return a ranking in which ``c`` is moved to last position in ``ranking``.
    """

    if isinstance(ranking, tuple):
        assert c != ranking[-1], "can't drop a candidate already in last place"
        new_ranking = copy.deepcopy(ranking)
        c_idx = new_ranking.index(c)
        new_ranking = new_ranking[:c_idx] + new_ranking[c_idx+1:] + (c,)

    if isinstance(ranking, Ranking):
        assert not ranking.last() == [c], "can't drop a candidate already uniquely in last place"
        new_ranking = Ranking({a: ranking.rmap[a] if a !=c else max(ranking.ranks) + 1 for a in ranking.cands})
        new_ranking.normalize_ranks()

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

    if isinstance(profile, Profile):
        rankings = [tuple(r) for r in list(_rankings)]
    
    if isinstance(profile, ProfileWithTies):
        rankings = _rankings

    rcounts = list(_rcounts)
    old_rankings = copy.deepcopy(rankings)

    ws = vm(profile)

    if violation_type == "Lift":
        for w in ws: 
            for r_idx, r in enumerate(rankings): 

                if isinstance(r, Ranking): # Make sure all candidates are ranked in r
                    r = Ranking({a: r.rmap[a] if a in r.cands else max(r.ranks)+1 for a in profile.candidates})

                if r[-1] == w and not r[0] == w:
                    old_ranking = copy.deepcopy(r)
                    new_ranking = lift_to_first(r, w)
                    new_rankings = old_rankings + [new_ranking]
                    new_rcounts  = copy.deepcopy(rcounts + [1])
                    new_rcounts[r_idx] -= 1

                    if isinstance(profile, Profile):
                        new_prof = Profile(new_rankings, new_rcounts)
                    
                    if isinstance(profile, ProfileWithTies):
                        new_prof = ProfileWithTies(new_rankings, new_rcounts)
                        if profile.using_extended_strict_preference:
                            new_prof.use_extended_strict_preference()

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

                    if isinstance(r, Ranking): # Make sure all candidates are ranked in r
                        r = Ranking({a: r.rmap[a] if a in r.cands else max(r.ranks)+1 for a in profile.candidates})

                    if r[0] == l and not r[-1] == l:
                        old_ranking = copy.deepcopy(r)
                        new_ranking = drop_to_last(r, l)
                        new_rankings = old_rankings + [new_ranking]
                        new_rcounts  = copy.deepcopy(rcounts + [1])
                        new_rcounts[r_idx] -= 1

                        if isinstance(profile, Profile):
                            new_prof = Profile(new_rankings, new_rcounts)
                        if isinstance(profile, ProfileWithTies):
                            new_prof = ProfileWithTies(new_rankings, new_rcounts)
                            if profile.using_extended_strict_preference:
                                new_prof.use_extended_strict_preference()

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

    if isinstance(profile, Profile):
        rankings = [tuple(r) for r in list(_rankings)]

    if isinstance(profile, ProfileWithTies):
        rankings = _rankings

    rcounts = list(_rcounts)
    old_rankings = copy.deepcopy(rankings)

    ws = vm(profile)
    witnesses = list()

    if violation_type == "Lift":
        for w in ws: 
            for r_idx, r in enumerate(rankings): 

                if isinstance(r, Ranking): # Make sure all candidates are ranked in r
                    r = Ranking({a: r.rmap[a] if a in r.cands else max(r.ranks)+1 for a in profile.candidates})

                if r[-1] == w and not r[0] == w:
                    old_ranking = copy.deepcopy(r)
                    new_ranking = lift_to_first(r, w)
                    new_rankings = old_rankings + [new_ranking]
                    new_rcounts  = copy.deepcopy(rcounts + [1])
                    new_rcounts[r_idx] -= 1

                    if isinstance(profile, Profile):
                        new_prof = Profile(new_rankings, new_rcounts)
                    if isinstance(profile, ProfileWithTies):
                        new_prof = ProfileWithTies(new_rankings, new_rcounts)
                        if profile.using_extended_strict_preference:
                            new_prof.use_extended_strict_preference()

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

                    if isinstance(r, Ranking): # Make sure all candidates are ranked in r
                        r = Ranking({a: r.rmap[a] if a in r.cands else max(r.ranks)+1 for a in profile.candidates})

                    if r[0] == l and not r[-1] == l:
                        old_ranking = copy.deepcopy(r)
                        new_ranking = drop_to_last(r, l)
                        new_rankings = old_rankings + [new_ranking]
                        new_rcounts  = copy.deepcopy(rcounts + [1])
                        new_rcounts[r_idx] -= 1

                        if isinstance(profile, Profile):
                            new_prof = Profile(new_rankings, new_rcounts)

                        if isinstance(profile, ProfileWithTies):
                            new_prof = ProfileWithTies(new_rankings, new_rcounts)
                            if profile.using_extended_strict_preference:
                                new_prof.use_extended_strict_preference()

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
    monotonicity,
    weak_positive_responsiveness
]