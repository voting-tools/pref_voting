"""
    File: variable_voter_axioms.py
    Author: Wes Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)
    Date: May 24, 2023
    
    Variable voter axioms 
"""

from pref_voting.axiom import Axiom
from pref_voting.axiom_helpers import *
import numpy as np
from itertools import product

def divide_electorate(prof):
    """Given a profile, yield all possible ways to divide the electorate into two nonempty electorates."""

    R, C = prof.rankings_counts

    ranges = [range(count+1) for count in C]

    # For each combination of divisions
    for division in product(*ranges):
        C1 = np.array(division)
        C2 = C - C1

        # We will filter out rankings where the count is zero
        nonzero_indices_C1 = np.nonzero(C1)[0]
        nonzero_indices_C2 = np.nonzero(C2)[0]

        # Only yield if both electorates have at least one voter
        if nonzero_indices_C1.size > 0 and nonzero_indices_C2.size > 0:

            rankings1 = R[nonzero_indices_C1].tolist()
            rankings2 = R[nonzero_indices_C2].tolist()
            counts1 = C1[nonzero_indices_C1].tolist()
            counts2 = C2[nonzero_indices_C2].tolist()

            if rankings1 <= rankings2: # This prevents yielding both prof1, prof2 and later on prof2, prof1, unless they are equal

                prof1 = Profile(rankings1, rcounts = counts1)
                prof2 = Profile(rankings2, rcounts = counts2)
            
                yield prof1, prof2


def has_reinforcement_violation_with_undergeneration(prof, vm, verbose=False):
    """Returns true if there is some binary partition of the electorate such that some candidate wins in both subprofiles but not in the full profile"""
    ws = vm(prof)

    for prof1, prof2 in divide_electorate(prof):
        winners_in_both = [c for c in vm(prof1) if c in vm(prof2)]
        if len(winners_in_both) > 0:
            undergenerated = [c for c in winners_in_both if c not in ws]
            if len(undergenerated) > 0:
                if verbose:
                    print(f"Candidate {undergenerated[0]} wins in subprofiles 1 and 2 but loses in the full profile:")
                    print("")
                    print("Subprofile 1")
                    prof1.display()
                    print(prof1.description())
                    vm.display(prof1)
                    print("")
                    print("Subprofile 2")
                    prof2.display()
                    print(prof2.description())
                    vm.display(prof2)
                    print("")
                    print("Full profile")
                    prof.display()
                    print(prof.description())
                    vm.display(prof)
                    print("")
                return True
            
    return False

def has_reinforcement_violation_with_overgeneration(prof, vm, verbose=False):
    """Returns true if there is some binary partition of the electorate such that some candidate wins in both subprofiles 
    but there is a winner in the full profile who is not among the winners in both subprofiles"""

    ws = vm(prof)

    for prof1, prof2 in divide_electorate(prof):
        winners_in_both = [c for c in vm(prof1) if c in vm(prof2)]
        if len(winners_in_both) > 0:
            overgenerated = [c for c in ws if c not in winners_in_both]
            if len(overgenerated) > 0:
                if verbose:
                    print(f"Candidate {overgenerated[0]} wins in the full profile but is not among the candidates who win in both subprofiles:")
                    print("")
                    print("Subprofile 1")
                    prof1.display()
                    print(prof1.description())
                    vm.display(prof1)
                    print("")
                    print("Subprofile 2")
                    prof2.display()
                    print(prof2.description())
                    vm.display(prof2)
                    print("")
                    print("Full profile")
                    prof.display()
                    print(prof.description())
                    vm.display(prof)
                    print("")
                return True
        
    return False


def has_reinforcement_violation(prof, vm, verbose=False):
    """
    Returns True if there is a binary partition of the electorate such that (i) at least one candidate wins in both subelections and either (ii) some candidate who wins in both subelections does not win in the full election or (iii) some candidate who wins in the full election does not win both subelections.
    
    Args:
        profile: a Profile object.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation. 

    Returns: 
        Result of the test (bool): Returns True if there is a violation and False otherwise. 

    """
    if has_reinforcement_violation_with_undergeneration(prof, vm, verbose):
        return True
    
    if has_reinforcement_violation_with_overgeneration(prof, vm, verbose):
        return True
    
    return False

def find_all_reinforcement_violations(prof, vm, verbose=False):
    """
    Returns all violations of reinforcement for a given profile and voting method.
    
    Args:
        profile: a Profile object.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation. 

    Returns: 
        Two list of triples (cand,prof1,prof2) where prof1 and prof2 partition the electorate. In the first list, (cand,prof1,prof2) indicates that cand wins in both prof1 and prof2 but loses in prof. In the second list, (cand,prof1,prof2) indicates that cand wins in prof but not in both prof1 and prof2 (and there are candidates who win in both prof1 and prof2).

    """
    ws = vm(prof)

    undergenerations = list()
    overgenerations = list()

    for prof1, prof2 in divide_electorate(prof):
        winners_in_both = [c for c in vm(prof1) if c in vm(prof2)]
        if len(winners_in_both) > 0:

            undergenerated = [c for c in winners_in_both if c not in ws]
            if len(undergenerated) > 0:
                for c in undergenerated:
                    undergenerations.append((c, prof1, prof2))
                if verbose:
                    print(f"Candidate {undergenerated[0]} wins in subprofiles 1 and 2 but loses in the full profile:")
                    print("")
                    print("Subprofile 1")
                    prof1.display()
                    print(prof1.description())
                    vm.display(prof1)
                    print("")
                    print("Subprofile 2")
                    prof2.display()
                    print(prof2.description())
                    vm.display(prof2)
                    print("")
                    print("Full profile")
                    prof.display()
                    print(prof.description())
                    vm.display(prof)
                    print("")
    
            overgenerated = [c for c in ws if c not in winners_in_both]
            if len(overgenerated) > 0:
                for c in overgenerated:
                    overgenerations.append((c, prof1, prof2))
                    if verbose:
                        print(f"Candidate {overgenerated[0]} wins in the full profile but is not among the candidates who win in both subprofiles:")
                        print("")
                        print("Subprofile 1")
                        prof1.display()
                        print(prof1.description())
                        vm.display(prof1)
                        print("")
                        print("Subprofile 2")
                        prof2.display()
                        print(prof2.description())
                        vm.display(prof2)
                        print("")
                        print("Full profile")
                        prof.display()
                        print(prof.description())
                        vm.display(prof)
                        print("")
        
    return undergenerations, overgenerations

reinforcement = Axiom(
    "Reinforcement",
    has_violation = has_reinforcement_violation,
    find_all_violations = find_all_reinforcement_violations, 
)

def has_positive_involvement_violation(prof, vm, verbose=False, violation_type="Removal"):
    """
    If violation_type = "Removal", returns True if removing some voter who ranked a losing candidate A in first place causes A to win, thereby violating positive involvement.
    
    Args:
        profile: a Profile object.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation.
        
    Returns:
        Result of the test (bool): Returns True if there is a violation and False otherwise."""

    winners = vm(prof)   
    losers = [c for c in prof.candidates if c not in winners]

    if violation_type == "Removal":
        for loser in losers:
            for r in prof._rankings: # for each type of ranking
                if r[0] == loser:
                    rankings = prof.rankings
                    rankings.remove(tuple(r)) # remove the first token of the type of ranking
                    prof2 = Profile(rankings)
                    if loser in vm(prof2):
                        if verbose:
                            print(f"{loser} loses in the full profile, but {loser} is a winner after removing a voter with the ranking {list(r)}:")
                            print("")
                            print("Full profile")
                            prof.display()
                            print(prof.description())
                            prof.display_margin_graph()
                            vm.display(prof)
                            print("")
                            print("Profile with voter removed")
                            anonprof2 = prof2.anonymize()
                            anonprof2.display()
                            print(anonprof2.description())
                            anonprof2.display_margin_graph()
                            vm.display(anonprof2)
                            print("")
                        return True
                    
def find_all_positive_involvement_violations(prof, vm, verbose=False, violation_type="Removal"):
    """
    If violation_type = "Removal", returns a list of pairs (loser,ranking) such that removing a voter with the given ranking causes the loser to win, thereby violating positive involvement.
    
    Args:
        profile: a Profile object.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation.
        
    Returns:
        A List of pairs (loser,ranking) witnessing violations of positive involvement."""

    winners = vm(prof)   
    losers = [c for c in prof.candidates if c not in winners]

    witnesses = list()

    if violation_type == "Removal":
        for loser in losers:
            for r in prof._rankings: # for each type of ranking
                if r[0] == loser:
                    rankings = prof.rankings
                    rankings.remove(tuple(r)) # remove the first token of the type of ranking
                    prof2 = Profile(rankings)
                    if loser in vm(prof2):
                        witnesses.append((loser, list(r)))
                        if verbose:
                            print(f"{loser} loses in the full profile, but {loser} is a winner after removing a voter with the ranking {list(r)}:")
                            print("")
                            print("Full profile")
                            prof.display()
                            print(prof.description())
                            prof.display_margin_graph()
                            vm.display(prof)
                            print("")
                            print("Profile with voter removed")
                            anonprof2 = prof2.anonymize()
                            anonprof2.display()
                            print(anonprof2.description())
                            anonprof2.display_margin_graph()
                            vm.display(anonprof2)
                            print("")
    return witnesses

positive_involvement = Axiom(
    "Positive Involvement",
    has_violation = has_positive_involvement_violation,
    find_all_violations = find_all_positive_involvement_violations, 
)

variable_voter_axioms = [
    reinforcement,
    positive_involvement
]