"""
    File: invariance_axioms.py
    Author: Wesley H. Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)
    Date: October 24, 2023
    
    Invariance axioms 
"""

from pref_voting.axiom import Axiom
from pref_voting.axiom_helpers import *
from itertools import permutations


def _homogeneity_violation(edata, vm, num_copies, violation_type, verbose=False):

    ws = vm(edata)

    if isinstance(edata, Profile):
        rankings, old_rcounts = edata.rankings_counts

        new_rcounts = [c * num_copies for c in old_rcounts]

        new_edata = Profile(list(map(list, rankings)), rcounts = new_rcounts, cmap = edata.cmap)
    else:
        old_rankings, old_rcounts = edata.rankings_counts

        new_rcounts = [c * num_copies for c in old_rcounts]

        new_edata = ProfileWithTies(rankings, rcounts = new_rcounts, candidates = edata.candidates, cmap = edata.cmap)
    
    new_ws = vm(new_edata)

    violation = False

    if violation_type == "Homogeneity":
        if new_ws != ws:
            violation = True
            return_value = list(set(new_ws) ^ set(ws))
    
    if violation_type == "Upward Homogeneity":
        if any([w for w in ws if w not in new_ws]):
            violation = True
            return_value = [w for w in ws if w not in new_ws]

    if violation_type == "Downward Homogeneity":
        if any([w for w in new_ws if w not in ws]):
            violation

    if violation:

        if verbose:
            print(f"{violation_type} Violation for {vm.name}")
            edata.display()
            print(edata.description())
            vm.display(edata)

            new_edata.display()
            print(new_edata.description())
            vm.display(new_edata)
 
        return True, return_value
    
    return False, list()

def has_homogeneity_violation(edata, vm, num_copies, verbose=False):
    """
    Returns True if replacing each ranking with num_copies of that ranking changes the set of winners.
    
    Args:
        edata (Profile, ProfileWithTies): the election data.
        vm (VotingMethod): A voting method to test.
        num_copies (int): The number of copies to multiply each ranking by.
        verbose (bool, default=False): If a violation is found, display the violation. 

    Returns: 
        Result of the test (bool): Returns True if there is a violation and False otherwise. 

    """
    return _homogeneity_violation(edata, vm, num_copies, "Homogeneity", verbose=verbose)[0]

def find_all_homogeneity_violations(edata, vm, num_copies, verbose=False):
    """
    Returns the symmetric difference of the winners before and after multiplying the number of copies of each ranking.
    
    Args:
        edata (Profile, ProfileWithTies): the election data.
        vm (VotingMethod): A voting method to test.
        num_copies (int): The number of copies to multiply each ranking by.
        verbose (bool, default=False): If a violation is found, display the violation. 

    Returns: 
        return_value: A list of the symmetric difference between the old and new winners.

    """
    return _homogeneity_violation(edata, vm, num_copies, "Homogeneity", verbose=verbose)[1]

homogeneity = Axiom(
    "Homogeneity",
    has_violation = has_homogeneity_violation,
    find_all_violations = find_all_homogeneity_violations, 
)

def has_upward_homogeneity_violation(edata, vm, num_copies, verbose=False):
    """
    Returns True if replacing each ranking with num_copies of that ranking causes some winner to lose.
    
    Args:
        edata (Profile, ProfileWithTies): the election data.
        vm (VotingMethod): A voting method to test.
        num_copies (int): The number of copies to multiply each ranking by.
        verbose (bool, default=False): If a violation is found, display the violation. 

    Returns: 
        Result of the test (bool): Returns True if there is a violation and False otherwise. 

    """
    return _homogeneity_violation(edata, vm, num_copies, "Upward Homogeneity", verbose=verbose)[0]

def find_all_upward_homogeneity_violations(edata, vm, num_copies, verbose=False):
    """
    Returns the set of winners who lose as a result of replacing each ranking with num_copies of that ranking.
    
    Args:
        edata (Profile, ProfileWithTies): the election data.
        vm (VotingMethod): A voting method to test.
        num_copies (int): The number of copies to multiply each ranking by.
        verbose (bool, default=False): If a violation is found, display the violation. 

    Returns: 
        return_value: the set of winners who lose as a result of replacing each ranking with num_copies of that ranking.

    """
    return _homogeneity_violation(edata, vm, num_copies, "Upward Homogeneity", verbose=verbose)[1]

upward_homogeneity = Axiom(
    "Upward Homogeneity",
    has_violation = has_upward_homogeneity_violation,
    find_all_violations = find_all_upward_homogeneity_violations, 
)

def has_downward_homogeneity_violation(edata, vm, num_copies, verbose=False):
    """
    Returns True if replacing each ranking with num_copies of that ranking causes some loser to win.
    
    Args:
        edata (Profile, ProfileWithTies): the election data.
        vm (VotingMethod): A voting method to test.
        num_copies (int): The number of copies to multiply each ranking by.
        verbose (bool, default=False): If a violation is found, display the violation. 

    Returns: 
        Result of the test (bool): Returns True if there is a violation and False otherwise. 

    """
    return _homogeneity_violation(edata, vm, num_copies, "Downward Homogeneity", verbose=verbose)[0]

def find_all_downward_homogeneity_violations(edata, vm, num_copies, verbose=False):
    """
    Returns the set of losers who win as a result of replacing each ranking with num_copies of that ranking.
    
    Args:
        edata (Profile, ProfileWithTies): the election data.
        vm (VotingMethod): A voting method to test.
        num_copies (int): The number of copies to multiply each ranking by.
        verbose (bool, default=False): If a violation is found, display the violation. 

    Returns: 
        return_value: the set of losers who win as a result of replacing each ranking with num_copies of that ranking.

    """
    return _homogeneity_violation(edata, vm, num_copies, "Downward Homogeneity", verbose=verbose)[1]

downward_homogeneity = Axiom(
    "Downward Homogeneity",
    has_violation = has_downward_homogeneity_violation,
    find_all_violations = find_all_downward_homogeneity_violations, 
)


def _has_block_violation(edata, vm, violation_type, verbose=False):

    ws = vm(edata)

    if isinstance(edata, Profile):
        old_rankings, old_rcounts = edata.rankings_counts

        new_rankings = list(permutations(edata.candidates))
        new_rcounts = [1] * len(new_rankings)

        new_edata = Profile(list(map(list, old_rankings)) + new_rankings, rcounts = list(old_rcounts) + new_rcounts, cmap = edata.cmap)
    else:
        old_rankings, old_rcounts = edata.rankings_counts

        new_rankings = [{c: perm.index(c)+1 for c in edata.candidates} for perm in permutations(edata.candidates)]
        new_rcounts = [1] * len(new_rankings)

        new_edata = ProfileWithTies(old_rankings + new_rankings, rcounts = old_rcounts + new_rcounts, candidates = edata.candidates, cmap = edata.cmap)
    
    new_ws = vm(new_edata)

    violation = False

    if violation_type == "Block Invariance":
        if new_ws != ws:
            violation = True
            return_value = list(set(new_ws) ^ set(ws))
    
    if violation_type == "Upward Block Preservation":
        if any([w for w in ws if w not in new_ws]):
            violation = True
            return_value = [w for w in ws if w not in new_ws]

    if violation_type == "Downward Block Preservation":
        if any([w for w in new_ws if w not in ws]):
            violation = True
            return_value = [w for w in new_ws if w not in ws]

    if violation:

        if verbose:
            print(f"{violation_type} Violation for {vm.name}")
            edata.display()
            print(edata.description())
            vm.display(edata)

            new_edata.display()
            print(new_edata.description())
            vm.display(new_edata)
 
        return True, return_value
    
    return False, list()

def has_block_invariance_violation(edata, vm, verbose=False):
    """
    Returns True if adding a block of all linear orders changes the set of winners.
    
    Args:
        edata (Profile, ProfileWithTies): the election data.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation. 

    Returns: 
        Result of the test (bool): Returns True if there is a violation and False otherwise. 

    """
    return _has_block_violation(edata, vm, "Block Invariance", verbose=verbose)[0]

def find_all_block_invariance_violations(edata, vm, verbose=False):
    """
    Returns the symmetric difference of the winners before and after adding a block of all linear orders.
    
    Args:
        edata (Profile, ProfileWithTies): the election data.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation. 

    Returns: 
        return_value: A list of the symmetric difference between the old and new winners.

    """
    return _has_block_violation(edata, vm, "Block Invariance", verbose=verbose)[1]

block_invariance = Axiom(
    "Block Invariance",
    has_violation = has_block_invariance_violation,
    find_all_violations = find_all_block_invariance_violations, 
)

def has_upward_block_preservation_violation(edata, vm, verbose=False):
    """
    Returns True if adding a block of all linear orders causes some winner to lose.
    
    Args:
        edata (Profile, ProfileWithTies): the election data.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation. 

    Returns: 
        Result of the test (bool): Returns True if there is a violation and False otherwise. 

    """
    return _has_block_violation(edata, vm, "Upward Block Preservation", verbose=verbose)[0]

def find_all_upward_block_preservation_violations(edata, vm, verbose=False):
    """
    Returns the set of winners who lose as a result of adding a block of all linear orders.
    
    Args:
        edata (Profile, ProfileWithTies): the election data.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation. 

    Returns: 
        return_value: the set of winners who lose as a result of adding a block of all linear orders.

    """
    return _has_block_violation(edata, vm, "Upward Block Preservation", verbose=verbose)[1]

upward_block_preservation = Axiom(
    "Upward Block Preservation",
    has_violation = has_upward_block_preservation_violation,
    find_all_violations = find_all_upward_block_preservation_violations, 
)

def has_downward_block_preservation_violation(edata, vm, verbose=False):
    """
    Returns True if adding a block of all linear orders causes some loser to win.
    
    Args:
        edata (Profile, ProfileWithTies): the election data.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation. 

    Returns: 
        Result of the test (bool): Returns True if there is a violation and False otherwise. 

    """
    return _has_block_violation(edata, vm, "Downward Block Preservation", verbose=verbose)[0]

def find_all_downward_block_preservation_violations(edata, vm, verbose=False):
    """
    Returns the set of losers who win as a result of adding a block of all linear orders.
    
    Args:
        edata (Profile, ProfileWithTies): the election data.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation. 

    Returns: 
        return_value: the set of losers who win as a result of adding a block of all linear orders.

    """
    return _has_block_violation(edata, vm, "Downward Block Preservation", verbose=verbose)[1]

downward_block_preservation = Axiom(
    "Downward Block Preservation",
    has_violation = has_downward_block_preservation_violation,
    find_all_violations = find_all_downward_block_preservation_violations, 
)

invariance_axioms = [
    block_invariance,
    upward_block_preservation,
    downward_block_preservation,
    homogeneity,
    upward_homogeneity,
    downward_homogeneity,
]