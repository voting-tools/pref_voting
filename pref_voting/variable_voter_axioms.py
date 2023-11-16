"""
    File: variable_voter_axioms.py
    Author: Wes Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)
    Date: May 24, 2023
    
    Variable voter axioms 
"""

from pref_voting.axiom import Axiom
from pref_voting.axiom_helpers import *
import numpy as np
from itertools import product, combinations

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

def _submultisets_of_fixed_cardinality(elements, multiplicities, cardinality):

    # Yields all sub-multisets of the given multiset with fixed cardinality.
    # For a closed-form expression for the number of sub-multisets of fixed cardinality, see https://arxiv.org/abs/1511.06142

    multiplicity_dict = {element: multiplicity for element, multiplicity in zip(elements, multiplicities)}

    def valid_partitions(cardinality, remaining_elements):
        if cardinality == 0:
            yield ()
            return
        if not remaining_elements:
            return  
        first, *rest = remaining_elements
        max_count = min(cardinality, multiplicity_dict[first])
        for i in range(1, max_count + 1):
            for partition in valid_partitions(cardinality-i, rest):
                yield (i,) + partition

    for i in range(1, min(len(elements), cardinality) + 1):
        for subset in combinations(elements, i):
            for partition in valid_partitions(cardinality, subset):
                if len(partition) == len(subset):
                    yield (subset, partition)


def has_positive_involvement_violation(prof, vm, verbose=False, violation_type="Removal", coalition_size = 1, uniform_coalition = True, require_resoluteness = False, require_uniquely_weighted = False):
    """
    If violation_type = "Removal", returns True if removing some voter (or voters if coalition_size > 1) who ranked a losing candidate A in first place causes A to win, witnessing a violation of positive involvement.

    If uniform_coalition = True, then only coalitions of voters with the same ranking are considered.

    If require_resoluteness = True, then only profiles with a unique winner are considered.

    If require_uniquely_weighted = True, then only uniquely-weighted profiles are considered.
    
    Args:
        profile: a Profile object.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation.
        violation_type: default is "Removal"
        
    Returns:
        Result of the test (bool): Returns True if there is a violation and False otherwise."""

    winners = vm(prof)   
    losers = [c for c in prof.candidates if c not in winners]

    if require_resoluteness and len(winners) > 1:
        return False

    if require_uniquely_weighted and not prof.is_uniquely_weighted():
        return False

    if violation_type == "Removal":
        if uniform_coalition:
            for loser in losers:

                relevant_ranking_types = [tuple(r) for r in prof._rankings if r[0] == loser and prof.rankings.count(tuple(r)) >= coalition_size]

                for r in relevant_ranking_types:

                    rankings = prof.rankings

                    for i in range(coalition_size):
                        rankings.remove(tuple(r)) # remove coalition_size-many tokens of the type of ranking

                    prof2 = Profile(rankings)
                    winners2 = vm(prof2)              

                    if require_resoluteness and len(winners2) > 1:
                        continue

                    if require_uniquely_weighted and not prof2.is_uniquely_weighted():
                        continue
                    
                    if loser in winners2:

                        if verbose:
                            if coalition_size == 1:
                                print(f"{loser} loses in the full profile, but {loser} is a winner after removing voter with the ranking {list(r)}:")
                            else:
                                print(f"{loser} loses in the full profile, but {loser} is a winner after removing {coalition_size} voters with the ranking {list(r)}:")
                            print("")
                            print("Full profile:")
                            prof.display()
                            print(prof.description())
                            prof.display_margin_graph()
                            vm.display(prof)
                            print("")
                            if coalition_size == 1:
                                print(f"Profile with voter removed:")
                            else:
                                print(f"Profile with {coalition_size} voters removed:")
                            anonprof2 = prof2.anonymize()
                            anonprof2.display()
                            print(anonprof2.description())
                            anonprof2.display_margin_graph()
                            vm.display(anonprof2)
                            print("")
                        return True
        
        if not uniform_coalition:
            for loser in losers:

                relevant_ranking_types = [tuple(r) for r in prof._rankings if r[0] == loser]
                relevant_ranking_types_counts = [prof.rankings.count(r) for r in relevant_ranking_types]

                for coalition_rankings, coalition_rankings_counts in _submultisets_of_fixed_cardinality(relevant_ranking_types,relevant_ranking_types_counts,coalition_size):
                    
                    rankings = prof.rankings
                    
                    for r_idx, r in enumerate(coalition_rankings):
                        for i in range(coalition_rankings_counts[r_idx]):
                            rankings.remove(r)
                        
                    prof2 = Profile(rankings)
                    winners2 = vm(prof2)              

                    if require_resoluteness and len(winners2) > 1:
                        continue

                    if require_uniquely_weighted and not prof2.is_uniquely_weighted():
                        continue
                    
                    if loser in winners2:

                        if verbose:
                            print(f"{loser} loses in the full profile, but {loser} is a winner after removing a {coalition_size}-voter coalition with the rankings {coalition_rankings} and counts {coalition_rankings_counts}:")
                            print("")
                            print("Full profile:")
                            prof.display()
                            print(prof.description())
                            prof.display_margin_graph()
                            vm.display(prof)
                            print("")
                            print(f"Profile with coalition removed:")
                            anonprof2 = prof2.anonymize()
                            anonprof2.display()
                            print(anonprof2.description())
                            anonprof2.display_margin_graph()
                            vm.display(anonprof2)
                            print("")
                        return True
                    
def find_all_positive_involvement_violations(prof, vm, verbose=False, violation_type="Removal", coalition_size = 1, uniform_coalition = True, require_resoluteness = False, require_uniquely_weighted = False):
    """
    If violation_type = "Removal", returns a list of pairs (loser, rankings, counts) such that removing the indicated rankings with the indicated counts causes the loser to win, witnessing a violation of positive involvement.
    
    If uniform_coalition = True, then only coalitions of voters with the same ranking are considered.

    If require_resoluteness = True, then only profiles with a unique winner are considered.

    If require_uniquely_weighted = True, then only uniquely-weighted profiles are considered.

    Args:
        profile: a Profile object.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation.
        violation_type: default is "Removal"
        
    Returns:
        A List of triples (loser,rankings,counts) witnessing violations of positive involvement.
        
    .. warning::
        This function is slow when uniform_coalition = False and the numbers of voters and candidates are too large.
        """

    winners = vm(prof)   
    losers = [c for c in prof.candidates if c not in winners]

    witnesses = list()

    if require_resoluteness and len(winners) > 1:
        return witnesses
    
    if require_uniquely_weighted and not prof.is_uniquely_weighted():
        return witnesses

    if violation_type == "Removal":
        if uniform_coalition:
            for loser in losers:
                relevant_ranking_types = [tuple(r) for r in prof._rankings if r[0] == loser and prof.rankings.count(tuple(r)) >= coalition_size]

                for r in relevant_ranking_types: # for each type of ranking
                        
                    rankings = prof.rankings # copy the token rankings
                    
                    for i in range(coalition_size):
                        rankings.remove(tuple(r)) # remove coalition_size-many tokens of the type of ranking

                    prof2 = Profile(rankings)
                    winners2 = vm(prof2)

                    if require_resoluteness and len(winners2) > 1:
                        continue

                    if require_uniquely_weighted and not prof2.is_uniquely_weighted():
                        continue

                    if loser in winners2:
                        witnesses.append((loser, [list(r)], [coalition_size]))
                        if verbose:
                            if coalition_size == 1:
                                print(f"{loser} loses in the full profile, but {loser} is a winner after removing voter with the ranking {list(r)}:")
                            else:
                                print(f"{loser} loses in the full profile, but {loser} is a winner after removing {coalition_size} voters with the ranking {list(r)}:")
                            print("")
                            print("Full profile")
                            prof.display()
                            print(prof.description())
                            prof.display_margin_graph()
                            vm.display(prof)
                            print("")
                            if coalition_size == 1:
                                print(f"Profile with voter removed:")
                            else:
                                print(f"Profile with {coalition_size} voters removed:")
                            anonprof2 = prof2.anonymize()
                            anonprof2.display()
                            print(anonprof2.description())
                            anonprof2.display_margin_graph()
                            vm.display(anonprof2)
                            print("")

        if coalition_size > 1 and not uniform_coalition:
            for loser in losers:
                relevant_ranking_types = [tuple(r) for r in prof._rankings if r[0] == loser]
                relevant_ranking_types_counts = [prof.rankings.count(r) for r in relevant_ranking_types]

                for coalition_rankings, coalition_rankings_counts in _submultisets_of_fixed_cardinality(relevant_ranking_types,relevant_ranking_types_counts,coalition_size):
                
                    rankings = prof.rankings
                    
                    for r_idx, r in enumerate(coalition_rankings):
                        for i in range(coalition_rankings_counts[r_idx]):
                            rankings.remove(r)
                        
                    prof2 = Profile(rankings)
                    winners2 = vm(prof2)              

                    if require_resoluteness and len(winners2) > 1:
                        continue

                    if require_uniquely_weighted and not prof2.is_uniquely_weighted():
                        continue

                    if loser in winners2:
                        witnesses.append((loser, coalition_rankings, coalition_rankings_counts))
                        if verbose:
                            print(f"{loser} loses in the full profile, but {loser} is a winner after removing a {coalition_size}-voter coalition with the rankings {coalition_rankings} and counts {coalition_rankings_counts}:")
                            print("")
                            print("Full profile")
                            prof.display()
                            print(prof.description())
                            prof.display_margin_graph()
                            vm.display(prof)
                            print("")
                            print(f"Profile with coalition removed:")
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

def has_negative_involvement_violation(prof, vm, verbose=False, violation_type="Removal"):
    """
    If violation_type = "Removal", returns True if removing some voter who ranked a winning candidate A in last place causes A to lose, witnessing a violation of negative involvement 
    Args:
        profile: a Profile object.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation.
        violation_type: default is "Removal"

    Returns:
        Result of the test (bool): Returns True if there is a violation and False otherwise."""
    
    winners = vm(prof)   
    
    if violation_type == "Removal":
        for winner in winners:
            for r in prof._rankings: # for each type of ranking
                if r[-1] == winner:
                    rankings = prof.rankings
                    rankings.remove(tuple(r)) # remove the first token of the type of ranking
                    prof2 = Profile(rankings)
                    if winner not in vm(prof2):
                        if verbose:
                            print(f"{winner} wins in the full profile, but {winner} is a loser after removing a voter with the ranking {list(r)}:")
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
                    
def find_all_negative_involvement_violations(prof, vm, verbose=False, violation_type="Removal"):
    """
    If violation_type = "Removal", returns a list of pairs (winner,ranking) such that removing a voter with the given ranking causes the winner to lose, witnessing a violation of negative involvement.
    
    Args:
        profile: a Profile object.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation.
        violation_type: default is "Removal"
        
    Returns:
        A List of pairs (winner, ranking) witnessing violations of negative involvement."""
    
    winners = vm(prof)   
    
    witnesses = list()
    
    if violation_type == "Removal":
        for winner in winners:
            for r in prof._rankings: # for each type of ranking
                if r[-1] == winner:
                    rankings = prof.rankings
                    rankings.remove(tuple(r)) # remove the first token of the type of ranking
                    prof2 = Profile(rankings)
                    if winner not in vm(prof2):
                        witnesses.append((winner, list(r)))
                        if verbose:
                            print(f"{winner} wins in the full profile, but {winner} is a loser after removing a voter with the ranking {list(r)}:")
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

negative_involvement = Axiom(
    "Negative Involvement",
    has_violation = has_negative_involvement_violation,
    find_all_violations = find_all_negative_involvement_violations, 
)

def has_tolerant_positive_involvement_violation(prof, vm, verbose=False, violation_type="Removal"):
    """
    If violation_type = "Removal", returns True if it is possible to cause a loser A to win by removing some voter who ranked A above every candidate B such that A is not majority preferred to B, witnessing a violation of  tolerant positive involvement.

    ..note:
        A strengthening of positive involvement, introduced in https://arxiv.org/abs/2210.12503
    
    Args:
        profile: a Profile object.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation.
        violation_type: default is "Removal"
        
    Returns:
        Result of the test (bool): Returns True if there is a violation and False otherwise."""

    winners = vm(prof)   
    losers = [c for c in prof.candidates if c not in winners]

    if violation_type == "Removal":
        for loser in losers:
            for r in prof._rankings: # for each type of ranking
                
                rl = list(r)
                tolerant_ballot = True

                # check whether the loser is ranked above every candiddate c such that the loser is not majority preferred to c
                for c in prof.candidates:
                    if not prof.majority_prefers(loser, c):
                        if rl.index(c) < rl.index(loser):
                            tolerant_ballot = False
                            break

                if tolerant_ballot:

                    rankings = prof.rankings
                    rankings.remove(tuple(r)) # remove the first token of the type of ranking
                    prof2 = Profile(rankings)
                    if loser in vm(prof2):
                        if verbose:
                            print(f"{loser} loses in the full profile, but {loser} is a winner after removing a voter with the ranking {rl}:")
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
                    
def find_all_tolerant_positive_involvement_violations(prof, vm, verbose=False, violation_type="Removal"):
    """
    If violation_type = "Removal", returns a list of pairs (loser,ranking) such that removing a voter with the given ranking causes the loser to win, witnessing a violation of tolerant positive involvement.
    
    Args:
        profile: a Profile object.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation.
        violation_type: default is "Removal"
        
    Returns:
        A List of pairs (loser,ranking) witnessing violations of positive involvement."""

    winners = vm(prof)   
    losers = [c for c in prof.candidates if c not in winners]

    witnesses = list()

    if violation_type == "Removal":
        for loser in losers:
            for r in prof._rankings: # for each type of ranking

                rl = list(r)
                tolerant_ballot = True

                # check whether the loser is ranked above every candiddate c such that the loser is not majority preferred to c
                for c in prof.candidates:
                    if not prof.majority_prefers(loser, c):
                        if rl.index(c) < rl.index(loser):
                            tolerant_ballot = False
                            break

                if tolerant_ballot:

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
                    
tolerant_positive_involvement = Axiom(
    "Tolerant Positive Involvement",
    has_violation = has_tolerant_positive_involvement_violation,
    find_all_violations = find_all_tolerant_positive_involvement_violations, 
)

def has_bullet_vote_positive_involvement_violation(prof, vm, verbose=False, coalition_size = 1, uniform_coalition = True, require_resoluteness = False, require_uniquely_weighted = False):
    """
    Returns True if it is possible to cause a winner A to lose by adding coalition_size-many new voters who bullet vote for A.
    
    Args:
        profile: a Profile object.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation.
        violation_type: default is "Removal"
        
    Returns:
        Result of the test (bool): Returns True if there is a violation and False otherwise."""

    if require_uniquely_weighted == True and not  prof.is_uniquely_weighted():
        return False 

    ws = vm(prof)

    if require_resoluteness == True and len(ws) > 1:
        return False
    
    for w in ws:
        new_prof = ProfileWithTies([{c:c_indx+1 for c_indx, c in enumerate(r)} for r in prof.rankings] + [{w:1}] * coalition_size, candidates = prof.candidates)
        new_prof.use_extended_strict_preference()
        new_mg = new_prof.margin_graph()

        if require_uniquely_weighted == True and not new_mg.is_uniquely_weighted(): 
            continue
        
        new_ws = vm(new_prof)

        if require_resoluteness == True and len(new_ws) > 1:
            continue

        if w not in new_ws:
            if verbose:

                print(f"Violation of Bullet Vote Positive Involvement for {vm.name}")
                print("Original profile:")
                prof.display()
                print(prof.description())
                prof.display_margin_graph()
                vm.display(prof)

                print("New profile:")
                new_prof.display()
                print(new_prof.description())
                new_prof.display_margin_graph()
                vm.display(new_prof)
                print("")

            return True
        
    return False

def find_all_bullet_vote_positive_involvement_violations(prof, vm, verbose=False, coalition_size = 1, require_resoluteness = False, require_uniquely_weighted = False):
    """
    Returns a list of candidates who win in the given profile but lose after adding coalition_size-many new voters who bullet vote for them.

    Args:
        profile: a Profile object.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation.

    Returns:
        A List of candidates who win in the given profile but lose after adding coalition_size-many new voters who bullet vote for them.
    
    """

    if require_uniquely_weighted == True and not prof.is_uniquely_weighted():
        return False 

    ws = vm(prof)

    if require_resoluteness == True and len(ws) > 1:
        return False
    
    violations = list()

    for w in ws:
        new_prof = ProfileWithTies([{c:c_indx+1 for c_indx, c in enumerate(r)} for r in prof.rankings] + [{w:1}] * coalition_size, candidates = prof.candidates)
        new_prof.use_extended_strict_preference()
        new_mg = new_prof.margin_graph()

        if require_uniquely_weighted == True and not new_mg.is_uniquely_weighted(): 
            continue
        
        new_ws = vm(new_prof)

        if require_resoluteness == True and len(new_ws) > 1:
            continue

        if w not in new_ws:
            if verbose:
                
                print(f"Violation of Bullet Vote Positive Involvement for {vm.name}")
                print("Original profile:")
                prof.display()
                print(prof.description())
                prof.display_margin_graph()
                vm.display(prof)

                print("New profile:")
                new_prof.display()
                print(new_prof.description())
                new_prof.display_margin_graph()
                vm.display(new_prof)
                print("")

            violations.append(w)
        
    return violations

bullet_vote_positive_involvement = Axiom(
    "Bullet Vote Positive Involvement",
    has_violation = has_bullet_vote_positive_involvement_violation,
    find_all_violations = find_all_bullet_vote_positive_involvement_violations, 
)

variable_voter_axioms = [
    reinforcement,
    positive_involvement,
    negative_involvement,
    tolerant_positive_involvement,
    bullet_vote_positive_involvement
]