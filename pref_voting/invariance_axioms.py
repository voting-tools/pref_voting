"""
    File: invariance_axioms.py
    Author: Wesley H. Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)
    Date: January 11, 2024
    Updated: May 29, 2025
    
    Invariance axioms 
"""

from pref_voting.axiom import Axiom
from pref_voting.axiom_helpers import *
from itertools import permutations
from pref_voting.profiles import Profile


def _homogeneity_violation(edata, vm, num_copies, violation_type, verbose=False):

    ws = vm(edata)

    if isinstance(edata, Profile):
        rankings, old_rcounts = edata.rankings_counts

        new_rcounts = [c * num_copies for c in old_rcounts]

        new_edata = Profile(list(map(list, rankings)), rcounts = new_rcounts, cmap = edata.cmap)
    else:
        old_rankings, old_rcounts = edata.rankings_counts

        new_rcounts = [c * num_copies for c in old_rcounts]

        new_edata = ProfileWithTies(old_rankings, rcounts = new_rcounts, candidates = edata.candidates, cmap = edata.cmap)

        if edata.using_extended_strict_preference:
            new_edata.use_extended_strict_preference()
    
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
            print("")

            new_edata.display()
            print(new_edata.description())
            vm.display(new_edata)
 
        return True, return_value
    
    return False, list()

def has_homogeneity_violation(edata, vm, num_copies = 2, verbose=False):
    """
    Returns True if replacing each ranking with num_copies of that ranking changes the set of winners.
    
    Args:
        edata (Profile, ProfileWithTies): the election data.
        vm (VotingMethod): A voting method to test.
        num_copies (int, default=2): The number of copies to multiply each ranking by.
        verbose (bool, default=False): If a violation is found, display the violation. 

    Returns: 
        Result of the test (bool): Returns True if there is a violation and False otherwise. 

    """
    return _homogeneity_violation(edata, vm, num_copies, "Homogeneity", verbose=verbose)[0]

def find_all_homogeneity_violations(edata, vm, num_copies = 2, verbose=False):
    """
    Returns the symmetric difference of the winners before and after multiplying the number of copies of each ranking.
    
    Args:
        edata (Profile, ProfileWithTies): the election data.
        vm (VotingMethod): A voting method to test.
        num_copies (int, default=2): The number of copies to multiply each ranking by.
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

def has_upward_homogeneity_violation(edata, vm, num_copies = 2, verbose=False):
    """
    Returns True if replacing each ranking with num_copies of that ranking causes some winner to lose.
    
    Args:
        edata (Profile, ProfileWithTies): the election data.
        vm (VotingMethod): A voting method to test.
        num_copies (int, default=2): The number of copies to multiply each ranking by.
        verbose (bool, default=False): If a violation is found, display the violation. 

    Returns: 
        Result of the test (bool): Returns True if there is a violation and False otherwise. 

    """
    return _homogeneity_violation(edata, vm, num_copies, "Upward Homogeneity", verbose=verbose)[0]

def find_all_upward_homogeneity_violations(edata, vm, num_copies = 2, verbose=False):
    """
    Returns the set of winners who lose as a result of replacing each ranking with num_copies of that ranking.
    
    Args:
        edata (Profile, ProfileWithTies): the election data.
        vm (VotingMethod): A voting method to test.
        num_copies (int, default=2): The number of copies to multiply each ranking by.
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

def has_downward_homogeneity_violation(edata, vm, num_copies = 2, verbose=False):
    """
    Returns True if replacing each ranking with num_copies of that ranking causes some loser to win.
    
    Args:
        edata (Profile, ProfileWithTies): the election data.
        vm (VotingMethod): A voting method to test.
        num_copies (int, default=2): The number of copies to multiply each ranking by.
        verbose (bool, default=False): If a violation is found, display the violation. 

    Returns: 
        Result of the test (bool): Returns True if there is a violation and False otherwise. 

    """
    return _homogeneity_violation(edata, vm, num_copies, "Downward Homogeneity", verbose=verbose)[0]

def find_all_downward_homogeneity_violations(edata, vm, num_copies = 2, verbose=False):
    """
    Returns the set of losers who win as a result of replacing each ranking with num_copies of that ranking.
    
    Args:
        edata (Profile, ProfileWithTies): the election data.
        vm (VotingMethod): A voting method to test.
        num_copies (int, default=2): The number of copies to multiply each ranking by.
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

def has_preferential_equality_violation(prof, vm, verbose=False):
    """
    Check if a profile has a preferential equality violation for the voting method vm.

    See Definition 2.1 and Lemma 2.4 from the paper "Characterizations of voting rules based on majority margins" by Y. Ding,  W. Holliday, and E. Pacuit.

    """
    if isinstance(prof, ProfileWithTies):
       prof_constructor = ProfileWithTies
       prof = prof.add_unranked_candidates()
    
    else:
        prof_constructor = Profile
    
    for x, y in combinations(prof.candidates, 2):
        
        xy_rankings = [r for r in prof.rankings if get_rank(r, x) == get_rank(r, y) - 1 and not any(get_rank(r,z) == get_rank(r,x) for z in prof.candidates if z != x) and not any(get_rank(r,z) == get_rank(r,y) for z in prof.candidates if z != y)]

        other_rankings = [r for r in prof.rankings if r not in xy_rankings]

        if len(xy_rankings) != 0 and len(xy_rankings) % 2 == 0:
            for I, J in equal_size_partitions_with_duplicates(xy_rankings):
                new_rankings_I = [swap_candidates(r, x, y) for r in I] + list(J) + list(other_rankings)
                prof_I = prof_constructor(new_rankings_I)
                new_rankings_J = [swap_candidates(r, x, y) for r in J] + list(I) + list(other_rankings)
                prof_J = prof_constructor(new_rankings_J)

                if vm(prof_I) != vm(prof_J): 
                    if verbose:
                        print("The original profile")
                        prof.anonymize().display()
                        prof.display_margin_graph()
                        print(prof.description())
                        vm.display(prof)
                        if prof_constructor == ProfileWithTies:
                            print(f"\nThe profile after swapping {x} and {y} in the rankings {[r.rmap for r in I]}:")
                        else:
                            print(f"\nThe profile after swapping {x} and {y} in the rankings {[r for r in I]}:")
                        prof_I.anonymize().display()
                        prof_I.display_margin_graph()
                        print(prof_I.description())   
                        vm.display(prof_I)
                        if prof_constructor == ProfileWithTies:
                            print(f"\nThe profile after swapping {x} and {y} in the rankings {[r.rmap for r in J]}:")
                        else:
                            print(f"\nThe profile after swapping {x} and {y} in the rankings {[r for r in J]}:")
                        prof_J.anonymize().display()
                        prof_J.display_margin_graph()
                        print(prof_J.description())
                        vm.display(prof_J)
                    return True

        yx_rankings = [r for r in prof.rankings if get_rank(r, y) == get_rank(r, x) - 1 and not any(get_rank(r,z) == get_rank(r,x) for z in prof.candidates if z != x) and not any(get_rank(r,z) == get_rank(r,y) for z in prof.candidates if z != y)]

        other_rankings = [r for r in prof.rankings if r not in yx_rankings]

        if len(yx_rankings) != 0 and len(yx_rankings) % 2 == 0:
            for I, J in equal_size_partitions_with_duplicates(yx_rankings):
                new_rankings_I = [swap_candidates(r, y, x) for r in I] + list(J) + list(other_rankings)
                prof_I = prof_constructor(new_rankings_I)
                new_rankings_J = [swap_candidates(r, y, x) for r in J] + list(I) + list(other_rankings)
                prof_J = prof_constructor(new_rankings_J)

                if vm(prof_I) != vm(prof_J): 
                    if verbose:
                        print("The original profile")
                        prof.anonymize().display()
                        prof.display_margin_graph()
                        print(prof.description())
                        vm.display(prof)
                        if prof_constructor == ProfileWithTies:
                            print(f"\nThe profile after swapping {y} and {x} in the rankings {[r.rmap for r in I]}:")
                        else:
                            print(f"\nThe profile after swapping {y} and {x} in the rankings {[r for r in I]}:")
                        prof_I.anonymize().display()
                        prof_I.display_margin_graph()
                        print(prof_I.description())
                        vm.display(prof_I)
                        if prof_constructor == ProfileWithTies:
                            print(f"\nThe profile after swapping {y} and {x} in the rankings {[r.rmap for r in J]}:")
                        else:
                            print(f"\nThe profile after swapping {y} and {x} in the rankings {[r for r in J]}:")
                        prof_J.anonymize().display()
                        prof_J.display_margin_graph()
                        print(prof_J.description())
                        vm.display(prof_J)
                    return True
    return False

def find_all_preferential_equality_violations(prof, vm, verbose=False):
    """
    Return all the preferential equality violations for the voting method vm.  Returns a list of tuples of three profiles (prof, prof_I, prof_J) such that vm(prof_I) != vm(prof_J) and prof_I and prof_J are as defined Lemma 2.4 from the paper "Characterizations of voting rules based on majority margins" by Y. Ding,  W. Holliday, and E. Pacuit (see also Definition 2.1).

    """

    if isinstance(prof, ProfileWithTies):
       prof_constructor = ProfileWithTies
       prof = prof.add_unranked_candidates()
    
    else:
        prof_constructor = Profile
    
    violations = []
    for x, y in combinations(prof.candidates, 2):
        
        xy_rankings = [r for r in prof.rankings if get_rank(r, x) == get_rank(r, y) - 1 and not any(get_rank(r,z) == get_rank(r,x) for z in prof.candidates if z != x) and not any(get_rank(r,z) == get_rank(r,y) for z in prof.candidates if z != y)]

        other_rankings = [r for r in prof.rankings if r not in xy_rankings]

        if len(xy_rankings) != 0 and len(xy_rankings) % 2 == 0:
            for I, J in equal_size_partitions_with_duplicates(xy_rankings):
                new_rankings_I = [swap_candidates(r, x, y) for r in I] + list(J) + list(other_rankings)
                prof_I = prof_constructor(new_rankings_I)
                new_rankings_J = [swap_candidates(r, x, y) for r in J] + list(I) + list(other_rankings)   
                prof_J = prof_constructor(new_rankings_J)

                if vm(prof_I) != vm(prof_J): 
                    if verbose:
                        print("The original profile")
                        prof.anonymize().display()
                        prof.display_margin_graph()
                        print(prof.description())
                        vm.display(prof)
                        if prof_constructor == ProfileWithTies:
                            print(f"\nThe profile after swapping {x} and {y} in the rankings {[r.rmap for r in I]}:")
                        else:
                            print(f"\nThe profile after swapping {x} and {y} in the rankings {[r for r in I]}:")
                        prof_I.anonymize().display()
                        prof_I.display_margin_graph()
                        print(prof_I.description())
                        vm.display(prof_I)
                        if prof_constructor == ProfileWithTies:
                            print(f"\nThe profile after swapping {x} and {y} in the rankings {[r.rmap for r in J]}:")
                        else:
                            print(f"\nThe profile after swapping {x} and {y} in the rankings {[r for r in J]}:")
                        prof_J.anonymize().display()
                        prof_J.display_margin_graph()
                        print(prof_J.description())
                        vm.display(prof_J)
                    violations.append((prof, prof_I, prof_J))

        yx_rankings = [r for r in prof.rankings if get_rank(r, y) == get_rank(r, x) - 1 and not any(get_rank(r,z) == get_rank(r,x) for z in prof.candidates if z != x) and not any(get_rank(r,z) == get_rank(r,y) for z in prof.candidates if z != y)]

        other_rankings = [r for r in prof.rankings if r not in yx_rankings]

        if len(yx_rankings) != 0 and len(yx_rankings) % 2 == 0:
            for I, J in equal_size_partitions_with_duplicates(yx_rankings):
                new_rankings_I = [swap_candidates(r, y, x) for r in I] + list(J) + list(other_rankings)
                prof_I = prof_constructor(new_rankings_I)
                new_rankings_J = [swap_candidates(r, y, x) for r in J] + list(I) + list(other_rankings)
                prof_J = prof_constructor(new_rankings_J)

                if vm(prof_I) != vm(prof_J): 
                    if verbose:
                        print("The original profile")
                        prof.anonymize().display()
                        prof.display_margin_graph()
                        print(prof.description())
                        vm.display(prof)
                        if prof_constructor == ProfileWithTies:
                            print(f"\nThe profile after swapping {x} and {y} in the rankings {[r.rmap for r in I]}:")
                        else:
                            print(f"\nThe profile after swapping {x} and {y} in the rankings {[r for r in I]}:")
                        prof_I.anonymize().display()
                        prof_I.display_margin_graph()
                        print(prof_I.description())
                        vm.display(prof_I)
                        if prof_constructor == ProfileWithTies:
                            print(f"\nThe profile after swapping {x} and {y} in the rankings {[r.rmap for r in J]}:")
                        else:
                            print(f"\nThe profile after swapping {x} and {y} in the rankings {[r for r in J]}:")
                        prof_J.anonymize().display()
                        prof_J.display_margin_graph()
                        print(prof_J.description())
                        vm.display(prof_J)
                    violations.append((prof, prof_I, prof_J))
    return violations

preferential_equality = Axiom(
    "Preferential Equality",
    has_violation = has_preferential_equality_violation,
    find_all_violations = find_all_preferential_equality_violations, 
)


def has_tiebreaking_compensation_violation(prof, vm, verbose=False):
    """
    Return True if the profile prof has a tiebreaking compensation violation for the voting method vm.
    """
    if isinstance(prof, Profile):
        return False
    
    for cands in powerset(prof.candidates): 
        if len(cands) > 1: 

            rankings_with_tie = [r for r in prof.rankings if r.is_tied(cands)]

            checked_rankings = []
            for r1, r2 in combinations(rankings_with_tie, 2):
                if set([r1, r2]) in checked_rankings:
                    continue
                checked_rankings.append(set([r1, r2]))
                for lin_order, reverse_lin_order in linear_orders_with_reverse(cands): 

                    other_rankings = [r for r in prof.rankings if not r in rankings_with_tie]
                    
                    # rankings_with_tie without r1 and r2
                    other_rankings_with_tie = remove_first_occurrences(rankings_with_tie, r1, r2)

                    new_rankings = [r1.break_tie(lin_order),r2.break_tie(reverse_lin_order)] +  other_rankings_with_tie + other_rankings 

                    new_prof = ProfileWithTies(new_rankings, candidates=prof.candidates)
                    if vm(prof) != vm(new_prof): 
                        if verbose: 
                            prof.anonymize().display()
                            print(prof.description())
                            vm.display(prof)

                            print(f"\nAfter breaking the tie between the candidates {[prof.cmap[c] for c in cands]} in {r1} with {tuple([prof.cmap[c] for c in lin_order])} and {r2} with {tuple([prof.cmap[c] for c in reverse_lin_order])}: \n")
                            new_prof.anonymize().display()
                            print(new_prof.description())
                            vm.display(new_prof)
                        return True
    return False


def find_all_tiebreaking_compensation_violations(prof, vm, verbose=False):
    """
    Find all the violations of tiebreaking compensation for prof with respect to the voting method vm. Returns a list of tuples consisting of the rankings and the rankings with the ties broken. If there are no violations, return an empty list.
    """
    if isinstance(prof, Profile):
        return []

    violations = []
    for cands in powerset(prof.candidates): 
        if len(cands) > 1: 

            rankings_with_tie = [r for r in prof.rankings if r.is_tied(cands)]

            checked_rankings = []
            for r1, r2 in combinations(rankings_with_tie, 2):
                if set([r1, r2]) in checked_rankings:
                    continue
                checked_rankings.append(set([r1, r2]))
                for lin_order, reverse_lin_order in linear_orders_with_reverse(cands): 

                    other_rankings = [r for r in prof.rankings if not r in rankings_with_tie]
                    
                    # rankings_with_tie without r1 and r2
                    other_rankings_with_tie = remove_first_occurrences(rankings_with_tie, r1, r2)

                    new_rankings = [r1.break_tie(lin_order),r2.break_tie(reverse_lin_order)] +  other_rankings_with_tie + other_rankings 

                    new_prof = ProfileWithTies(new_rankings, candidates=prof.candidates)
                    if vm(prof) != vm(new_prof): 
                        if verbose: 
                            prof.anonymize().display()
                            print(prof.description())
                            vm.display(prof)

                            print(f"\nAfter breaking the tie between the candidates {[prof.cmap[c] for c in cands]} in {r1} with {tuple([prof.cmap[c] for c in lin_order])} and {r2} with {tuple([prof.cmap[c] for c in reverse_lin_order])}: \n")
                            new_prof.anonymize().display()
                            print(new_prof.description())
                            vm.display(new_prof)
                        violations.append(((r1, r1.break_tie(lin_order)), (r2, r2.break_tie(reverse_lin_order))))
    return violations

tiebreaking_compensation = Axiom(
    "Tiebreaking Compensation",
    has_violation = has_tiebreaking_compensation_violation,
    find_all_violations = find_all_tiebreaking_compensation_violations, 
)
invariance_axioms = [
    block_invariance,
    upward_block_preservation,
    downward_block_preservation,
    homogeneity,
    upward_homogeneity,
    downward_homogeneity,
    preferential_equality,
    tiebreaking_compensation,
]