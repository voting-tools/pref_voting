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


def has_reinforcement_violation(prof, vm, verbose=False):
    """
    Returns True if there is a binary partition of the electorate such that some candidate wins in both subelections but loses in the full election.
    
    .. warning:: 
        This takes a long time with more than 10 voters. 

    Args:
        prof: a Profile object.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation. 

    Returns: 
        Result of the test (bool): Returns True if there is a violation and False otherwise. 

    """
    ws = vm(prof)

    for prof1, prof2 in divide_electorate(prof):
        ws_1 = vm(prof1)
        ws_2 = vm(prof2)
        witnesses = [w for w in ws_1 if w in ws_2 and not w in ws]
        if len(witnesses) > 0:
            if verbose:
                print(f"Candidate {witnesses[0]} wins in subprofiles 1 and 2 but loses in the full profile:")
                print("")
                print("Subprofile 1")
                prof1.display()
                print(prof1.description())
                print("")
                print("Subprofile 2")
                prof2.display()
                print(prof2.description())
                print("")
                print("Full profile")
                prof.display()
                print(prof.description())
                print("")
            return True
        
    return False

def find_all_reinforcement_violations(prof, vm, verbose=False):
    """
    Returns all violations of reinforcement for a given profile and voting method.

    .. warning:: 
        This takes a long time with more than 10 voters. 
    
    Args:
        prof: a Profile object.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation. 

    Returns: 
        A list of triples (cand,prof1,prof2) such that prof1 and prof2 partition the electorate of prof and cand wins in both prof1 and prof2 but loses in prof.

    """
    ws = vm(prof)

    violations = list()

    for prof1, prof2 in divide_electorate(prof):
        ws_1 = vm(prof1)
        ws_2 = vm(prof2)
        witnesses = [w for w in ws_1 if w in ws_2 and not w in ws]
        if len(witnesses) > 0:
            if verbose:
                print(f"Candidate {witnesses[0]} wins in subprofiles 1 and 2 but loses in the full profile:")
                print("")
                print("Subprofile 1")
                prof1.display()
                print(prof1.description())
                print("")
                print("Subprofile 2")
                prof2.display()
                print(prof2.description())
                print("")
                print("Full profile")
                prof.display()
                print(prof.description())
                print("")

            for w in witnesses:
                violations.append((w, prof1, prof2))
        
    return violations

reinforcement = Axiom(
    "Reinforcement",
    has_violation = has_reinforcement_violation,
    find_all_violations = find_all_reinforcement_violations, 
)

variable_voter_axioms = [
    reinforcement,
]