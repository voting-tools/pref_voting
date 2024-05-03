"""
    File: variable_voter_axioms.py
    Author: Wes Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)
    Date: May 24, 2023
    
    Variable voter axioms 
"""

from pref_voting.axiom import Axiom
from pref_voting.axiom_helpers import *
import numpy as np
from itertools import product, combinations, permutations
from pref_voting.helper import weak_orders
from pref_voting.rankings import Ranking

def divide_electorate(prof):
    """Given a Profile or ProfileWithTies object, yield all possible ways to divide the electorate into two nonempty electorates."""

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

                if isinstance(prof,Profile):
                    prof1 = Profile(rankings1, rcounts = counts1)
                    prof2 = Profile(rankings2, rcounts = counts2)
                
                if isinstance(prof,ProfileWithTies):
                    prof1 = ProfileWithTies(rankings1, rcounts = counts1)
                    prof2 = ProfileWithTies(rankings2, rcounts = counts2)

                    if prof.using_extended_strict_preference:
                        prof1.use_extended_strict_preference()
                        prof2.use_extended_strict_preference()
            
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
        prof: a Profile or ProfileWithTies object.
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
        prof: a Profile or ProfileWithTies object.
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

    def valid_partitions(cardinality, remaining_elements):
        if cardinality == 0:
            yield ()
            return
        if not remaining_elements:
            return  
        first, *rest = remaining_elements
        first_idx = elements.index(first)
        max_count = min(cardinality, multiplicities[first_idx])
        for i in range(1, max_count + 1):
            for partition in valid_partitions(cardinality-i, rest):
                yield (i,) + partition

    for i in range(1, min(len(elements), cardinality) + 1):
        for subset in combinations(elements, i):
            for partition in valid_partitions(cardinality, subset):
                if len(partition) == len(subset):
                    yield (subset, partition)


def has_positive_involvement_violation(prof, vm, verbose=False, violation_type="Removal", coalition_size = 1, uniform_coalition = True, require_resoluteness = False, require_uniquely_weighted = False, check_probabilities = False):
    """
    If violation_type = "Removal", returns True if removing some voter (or voters if coalition_size > 1) who ranked a losing candidate A in first place causes A to win, witnessing a violation of positive involvement.

    If uniform_coalition = True, then only coalitions of voters with the same ranking are considered.

    If require_resoluteness = True, then only profiles with a unique winner are considered.

    If require_uniquely_weighted = True, then only uniquely-weighted profiles are considered.

    If check_probabilities = True, the function also checks whether removing the voters who ranked A in first-place causes A's probability of winning to increase (in the case of a tie broken by even-chance tiebreaking).
    
    Args:
        prof: a Profile or ProfileWithTies object.
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

                relevant_ranking_types = [r for r in prof.ranking_types if r[0] == loser and prof.rankings.count(r) >= coalition_size]

                for r in relevant_ranking_types:

                    rankings = prof.rankings

                    for i in range(coalition_size):
                        rankings.remove(r) # remove coalition_size-many tokens of the type of ranking

                    if isinstance(prof,Profile):
                        prof2 = Profile(rankings)

                    if isinstance(prof,ProfileWithTies):
                        prof2 = ProfileWithTies(rankings, candidates = prof.candidates)
                        if prof.using_extended_strict_preference:
                            prof2.use_extended_strict_preference()

                    winners2 = vm(prof2)              

                    if require_resoluteness and len(winners2) > 1:
                        continue

                    if require_uniquely_weighted and not prof2.is_uniquely_weighted():
                        continue
                    
                    if loser in winners2:

                        if verbose:
                            prof = prof.anonymize()
                            if coalition_size == 1:
                                print(f"{loser} loses in the full profile, but {loser} is a winner after removing voter with the ranking {str(r)}:")
                            else:
                                print(f"{loser} loses in the full profile, but {loser} is a winner after removing {coalition_size} voters with the ranking {str(r)}:")
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
                            prof2 = prof2.anonymize()
                            prof2.display()
                            print(prof2.description())
                            prof2.display_margin_graph()
                            vm.display(prof2)
                            print("")
                        return True
                    
            if check_probabilities:
                for winner in winners:

                    relevant_ranking_types = [r for r in prof.ranking_types if r[0] == winner and prof.rankings.count(r) >= coalition_size]

                    for r in relevant_ranking_types:

                        rankings = prof.rankings

                        for i in range(coalition_size):
                            rankings.remove(r) # remove coalition_size-many tokens of the type of ranking

                        if isinstance(prof,Profile):
                            prof2 = Profile(rankings)

                        if isinstance(prof,ProfileWithTies):
                            prof2 = ProfileWithTies(rankings, candidates = prof.candidates)
                            if prof.using_extended_strict_preference:
                                prof2.use_extended_strict_preference()

                        winners2 = vm(prof2)              

                        if require_resoluteness and len(winners2) > 1:
                            continue

                        if require_uniquely_weighted and not prof2.is_uniquely_weighted():
                            continue
                        
                        if winner in winners2 and len(winners) > len(winners2):

                            if verbose:
                                prof = prof.anonymize()
                                if coalition_size == 1:
                                    print(f"{winner} has a higher probability of winning after removing voter with the ranking {str(r)}:")
                                else:
                                    print(f"{winner} has a higher probability of winning after removing removing {coalition_size} voters with the ranking {str(r)}:")
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
                                prof2 = prof2.anonymize()
                                prof2.display()
                                print(prof2.description())
                                prof2.display_margin_graph()
                                vm.display(prof2)
                                print("")
                            return True

        
        if not uniform_coalition:
            for loser in losers:

                relevant_ranking_types = [r for r in prof.ranking_types if r[0] == loser]
                relevant_ranking_types_counts = [prof.rankings.count(r) for r in relevant_ranking_types]

                for coalition_rankings, coalition_rankings_counts in _submultisets_of_fixed_cardinality(relevant_ranking_types,relevant_ranking_types_counts,coalition_size):
                    
                    rankings = prof.rankings
                    
                    for r_idx, r in enumerate(coalition_rankings):
                        for i in range(coalition_rankings_counts[r_idx]):
                            rankings.remove(r)
                        
                    if isinstance(prof,Profile):
                        prof2 = Profile(rankings)

                    if isinstance(prof,ProfileWithTies):
                        prof2 = ProfileWithTies(rankings, candidates = prof.candidates)
                        if prof.using_extended_strict_preference:
                            prof2.use_extended_strict_preference()

                    winners2 = vm(prof2)              

                    if require_resoluteness and len(winners2) > 1:
                        continue

                    if require_uniquely_weighted and not prof2.is_uniquely_weighted():
                        continue
                    
                    if loser in winners2:

                        if verbose:
                            prof = prof.anonymize()
                            print(f"{loser} loses in the full profile, but {loser} is a winner after removing a {coalition_size}-voter coalition with the rankings {[str(r) for r in coalition_rankings]} and counts {coalition_rankings_counts}:")
                            print("")
                            print("Full profile:")
                            prof.display()
                            print(prof.description())
                            prof.display_margin_graph()
                            vm.display(prof)
                            print("")
                            print(f"Profile with coalition removed:")
                            prof2 = prof2.anonymize()
                            prof2.display()
                            print(prof2.description())
                            prof2.display_margin_graph()
                            vm.display(prof2)
                            print("")
                        return True
                    
            if check_probabilities:
                for winner in winners:

                    relevant_ranking_types = [r for r in prof.ranking_types if r[0] == winner]
                    relevant_ranking_types_counts = [prof.rankings.count(r) for r in relevant_ranking_types]

                    for coalition_rankings, coalition_rankings_counts in _submultisets_of_fixed_cardinality(relevant_ranking_types,relevant_ranking_types_counts,coalition_size):
                        
                        rankings = prof.rankings
                        
                        for r_idx, r in enumerate(coalition_rankings):
                            for i in range(coalition_rankings_counts[r_idx]):
                                rankings.remove(r)
                            
                        if isinstance(prof,Profile):
                            prof2 = Profile(rankings)

                        if isinstance(prof,ProfileWithTies):
                            prof2 = ProfileWithTies(rankings)
                            if prof.using_extended_strict_preference:
                                prof2.use_extended_strict_preference()

                        winners2 = vm(prof2)              

                        if require_resoluteness and len(winners2) > 1:
                            continue

                        if require_uniquely_weighted and not prof2.is_uniquely_weighted():
                            continue
                        
                        if winner in winners2 and len(winners) > len(winners2):

                            if verbose:
                                prof = prof.anonymize()
                                print(f"{winner} has a higher probability of winning after removing a {coalition_size}-voter coalition with the rankings {[str(r) for r in coalition_rankings]} and counts {coalition_rankings_counts}:")
                                print("")
                                print("Full profile:")
                                prof.display()
                                print(prof.description())
                                prof.display_margin_graph()
                                vm.display(prof)
                                print("")
                                print(f"Profile with coalition removed:")
                                prof2 = prof2.anonymize()
                                prof2.display()
                                print(prof2.description())
                                prof2.display_margin_graph()
                                vm.display(prof2)
                                print("")
                            return True
    
    return False
                    
def find_all_positive_involvement_violations(prof, vm, verbose=False, violation_type="Removal", coalition_size = 1, uniform_coalition = True, require_resoluteness = False, require_uniquely_weighted = False, check_probabilities = False):
    """
    If violation_type = "Removal", returns a list of pairs (loser, rankings, counts) such that removing the indicated rankings with the indicated counts causes the loser to win, witnessing a violation of positive involvement.
    
    If uniform_coalition = True, then only coalitions of voters with the same ranking are considered.

    If require_resoluteness = True, then only profiles with a unique winner are considered.

    If require_uniquely_weighted = True, then only uniquely-weighted profiles are considered.

    If check_probabilities = True, the function also checks whether removing the voters who ranked A in first-place causes A's probability of winning to increase (in the case of a tie broken by even-chance tiebreaking).

    Args:
        prof: a Profile or ProfileWithTies object.
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
                relevant_ranking_types = [r for r in prof.ranking_types if r[0] == loser and prof.rankings.count(r) >= coalition_size]

                for r in relevant_ranking_types: # for each type of ranking
                        
                    rankings = prof.rankings # copy the token rankings
                    
                    for i in range(coalition_size):
                        rankings.remove(r) # remove coalition_size-many tokens of the type of ranking

                    if isinstance(prof,Profile):
                        prof2 = Profile(rankings)

                    if isinstance(prof,ProfileWithTies):
                        prof2 = ProfileWithTies(rankings, candidates = prof.candidates)
                        if prof.using_extended_strict_preference:
                            prof2.use_extended_strict_preference()

                    winners2 = vm(prof2)

                    if require_resoluteness and len(winners2) > 1:
                        continue

                    if require_uniquely_weighted and not prof2.is_uniquely_weighted():
                        continue

                    if loser in winners2:
                        witnesses.append((loser, [r], [coalition_size]))
                        if verbose:
                            prof = prof.anonymize()
                            if coalition_size == 1:
                                print(f"{loser} loses in the full profile, but {loser} is a winner after removing voter with the ranking {str(r)}:")
                            else:
                                print(f"{loser} loses in the full profile, but {loser} is a winner after removing {coalition_size} voters with the ranking {str(r)}:")
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
                            prof2 = prof2.anonymize()
                            prof2.display()
                            print(prof2.description())
                            prof2.display_margin_graph()
                            vm.display(prof2)
                            print("")
            
            if check_probabilities:
                for winner in winners:
                    relevant_ranking_types = [r for r in prof.ranking_types if r[0] == winner and prof.rankings.count(r) >= coalition_size]

                    for r in relevant_ranking_types:
                        
                        rankings = prof.rankings
                        
                        for i in range(coalition_size):
                            rankings.remove(r)

                        if isinstance(prof,Profile):
                            prof2 = Profile(rankings)

                        if isinstance(prof,ProfileWithTies):
                            prof2 = ProfileWithTies(rankings, candidates = prof.candidates)
                            if prof.using_extended_strict_preference:
                                prof2.use_extended_strict_preference()

                        winners2 = vm(prof2)

                        if require_resoluteness and len(winners2) > 1:
                            continue

                        if require_uniquely_weighted and not prof2.is_uniquely_weighted():
                            continue

                        if winner in winners2 and len(winners) > len(winners2):
                            witnesses.append((winner, [r], [coalition_size]))
                            if verbose:
                                prof = prof.anonymize()
                                if coalition_size == 1:
                                    print(f"{winner} has a higher probability of winning after removing voter with the ranking {str(r)}:")
                                else:
                                    print(f"{winner} has a higher probability of winning after removing {coalition_size} voters with the ranking {str(r)}:")
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
                                prof2 = prof2.anonymize()
                                prof2.display()
                                print(prof2.description())
                                prof2.display_margin_graph()
                                vm.display(prof2)
                                print("")


        if not uniform_coalition:
            for loser in losers:
                relevant_ranking_types = [r for r in prof.ranking_types if r[0] == loser]
                relevant_ranking_types_counts = [prof.rankings.count(r) for r in relevant_ranking_types]

                for coalition_rankings, coalition_rankings_counts in _submultisets_of_fixed_cardinality(relevant_ranking_types,relevant_ranking_types_counts,coalition_size):
                
                    rankings = prof.rankings
                    
                    for r_idx, r in enumerate(coalition_rankings):
                        for i in range(coalition_rankings_counts[r_idx]):
                            rankings.remove(r)
                        
                    if isinstance(prof,Profile):
                        prof2 = Profile(rankings)

                    if isinstance(prof,ProfileWithTies):
                        prof2 = ProfileWithTies(rankings, candidates = prof.candidates)
                        if prof.using_extended_strict_preference:
                            prof2.use_extended_strict_preference()

                    winners2 = vm(prof2)              

                    if require_resoluteness and len(winners2) > 1:
                        continue

                    if require_uniquely_weighted and not prof2.is_uniquely_weighted():
                        continue

                    if loser in winners2:
                        witnesses.append((loser, coalition_rankings, coalition_rankings_counts))
                        if verbose:
                            prof = prof.anonymize()
                            print(f"{loser} loses in the full profile, but {loser} is a winner after removing a {coalition_size}-voter coalition with the rankings {[str(r) for r in coalition_rankings]} and counts {coalition_rankings_counts}:")
                            print("")
                            print("Full profile")
                            prof.display()
                            print(prof.description())
                            prof.display_margin_graph()
                            vm.display(prof)
                            print("")
                            print(f"Profile with coalition removed:")
                            prof2 = prof2.anonymize()
                            prof2.display()
                            print(prof2.description())
                            prof2.display_margin_graph()
                            vm.display(prof2)
                            print("")

            if check_probabilities:
                for winner in winners:
                    relevant_ranking_types = [r for r in prof.ranking_types if r[0] == winner]
                    relevant_ranking_types_counts = [prof.rankings.count(r) for r in relevant_ranking_types]

                    for coalition_rankings, coalition_rankings_counts in _submultisets_of_fixed_cardinality(relevant_ranking_types,relevant_ranking_types_counts,coalition_size):
                        
                        rankings = prof.rankings
                        
                        for r_idx, r in enumerate(coalition_rankings):
                            for i in range(coalition_rankings_counts[r_idx]):
                                rankings.remove(r)
                            
                        if isinstance(prof,Profile):
                            prof2 = Profile(rankings)

                        if isinstance(prof,ProfileWithTies):
                            prof2 = ProfileWithTies(rankings, candidates = prof.candidates)
                            if prof.using_extended_strict_preference:
                                prof2.use_extended_strict_preference()

                        winners2 = vm(prof2)              

                        if require_resoluteness and len(winners2) > 1:
                            continue

                        if require_uniquely_weighted and not prof2.is_uniquely_weighted():
                            continue

                        if winner in winners2 and len(winners) > len(winners2):
                            witnesses.append((winner, coalition_rankings, coalition_rankings_counts))
                            if verbose:
                                prof = prof.anonymize()
                                print(f"{winner} has a higher probability of winning after removing a {coalition_size}-voter coalition with the rankings {[str(r) for r in coalition_rankings]} and counts {coalition_rankings_counts}:")
                                print("")
                                print("Full profile")
                                prof.display()
                                print(prof.description())
                                prof.display_margin_graph()
                                vm.display(prof)
                                print("")
                                print(f"Profile with coalition removed:")
                                prof2 = prof2.anonymize()
                                prof2.display()
                                print(prof2.description())
                                prof2.display_margin_graph()
                                vm.display(prof2)
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
        prof: a Profile or ProfileWithTies object.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation.
        violation_type: default is "Removal"

    Returns:
        Result of the test (bool): Returns True if there is a violation and False otherwise."""
    
    winners = vm(prof)   
    
    if violation_type == "Removal":
        for winner in winners:
            for r in prof.ranking_types: # for each type of ranking
                if r[-1] == winner:
                    rankings = prof.rankings
                    rankings.remove(r) # remove the first token of the type of ranking

                    if isinstance(prof,Profile):
                        prof2 = Profile(rankings)

                    if isinstance(prof,ProfileWithTies):
                        prof2 = ProfileWithTies(rankings, candidates = prof.candidates)
                        if prof.using_extended_strict_preference:
                            prof2.use_extended_strict_preference()

                    if winner not in vm(prof2):
                        if verbose:
                            prof = prof.anonymize()
                            print(f"{winner} wins in the full profile, but {winner} is a loser after removing a voter with the ranking {str(r)}:")
                            print("")
                            print("Full profile")
                            prof.display()
                            print(prof.description())
                            prof.display_margin_graph()
                            vm.display(prof)
                            print("")
                            print("Profile with voter removed")
                            prof2 = prof2.anonymize()
                            prof2.display()
                            print(prof2.description())
                            prof2.display_margin_graph()
                            vm.display(prof2)
                            print("")
                        return True
                    
def find_all_negative_involvement_violations(prof, vm, verbose=False, violation_type="Removal"):
    """
    If violation_type = "Removal", returns a list of pairs (winner,ranking) such that removing a voter with the given ranking causes the winner to lose, witnessing a violation of negative involvement.
    
    Args:
        prof: a Profile or ProfileWithTies object.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation.
        violation_type: default is "Removal"
        
    Returns:
        A List of pairs (winner, ranking) witnessing violations of negative involvement."""
    
    winners = vm(prof)   
    
    witnesses = list()
    
    if violation_type == "Removal":
        for winner in winners:
            for r in prof.ranking_types: # for each type of ranking
                if r[-1] == winner:
                    rankings = prof.rankings
                    rankings.remove(r) # remove the first token of the type of ranking
                    
                    if isinstance(prof,Profile):
                        prof2 = Profile(rankings)

                    if isinstance(prof,ProfileWithTies):
                        prof2 = ProfileWithTies(rankings, candidates = prof.candidates)
                        if prof.using_extended_strict_preference:
                            prof2.use_extended_strict_preference()

                    if winner not in vm(prof2):
                        witnesses.append((winner, r))
                        if verbose:
                            prof = prof.anonymize()
                            print(f"{winner} wins in the full profile, but {winner} is a loser after removing a voter with the ranking {str(r)}:")
                            print("")
                            print("Full profile")
                            prof.display()
                            print(prof.description())
                            prof.display_margin_graph()
                            vm.display(prof)
                            print("")
                            print("Profile with voter removed")
                            prof2 = prof2.anonymize()
                            prof2.display()
                            print(prof2.description())
                            prof2.display_margin_graph()
                            vm.display(prof2)
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
        prof: a Profile or ProfileWithTies object.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation.
        violation_type: default is "Removal"
        
    Returns:
        Result of the test (bool): Returns True if there is a violation and False otherwise."""

    winners = vm(prof)   
    losers = [c for c in prof.candidates if c not in winners]

    if violation_type == "Removal":
        for loser in losers:
            for r in prof.rankings_types: # for each type of ranking
                
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
                    rankings.remove(r) # remove the first token of the type of ranking
                    
                    prof2 = Profile(rankings)

                    if loser in vm(prof2):
                        if verbose:
                            prof = prof.anonymize()
                            print(f"{loser} loses in the full profile, but {loser} is a winner after removing a voter with the ranking {rl}:")
                            print("")
                            print("Full profile")
                            prof.display()
                            print(prof.description())
                            prof.display_margin_graph()
                            vm.display(prof)
                            print("")
                            print("Profile with voter removed")
                            prof2 = prof2.anonymize()
                            prof2.display()
                            print(prof2.description())
                            prof2.display_margin_graph()
                            vm.display(prof2)
                            print("")
                        return True
                    
def find_all_tolerant_positive_involvement_violations(prof, vm, verbose=False, violation_type="Removal"):
    """
    If violation_type = "Removal", returns a list of pairs (loser,ranking) such that removing a voter with the given ranking causes the loser to win, witnessing a violation of tolerant positive involvement.
    
    Args:
        prof: a Profile or ProfileWithTies object.
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
            for r in prof.ranking_types: # for each type of ranking

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
                    rankings.remove(r) # remove the first token of the type of ranking
                    
                    prof2 = Profile(rankings)

                    if loser in vm(prof2):
                        witnesses.append((loser, r))
                        if verbose:
                            prof = prof.anonymize()
                            print(f"{loser} loses in the full profile, but {loser} is a winner after removing a voter with the ranking {str(r)}:")
                            print("")
                            print("Full profile")
                            prof.display()
                            print(prof.description())
                            prof.display_margin_graph()
                            vm.display(prof)
                            print("")
                            print("Profile with voter removed")
                            prof2 = prof2.anonymize()
                            prof2.display()
                            print(prof2.description())
                            prof2.display_margin_graph()
                            vm.display(prof2)
                            print("")
    return witnesses
                    
tolerant_positive_involvement = Axiom(
    "Tolerant Positive Involvement",
    has_violation = has_tolerant_positive_involvement_violation,
    find_all_violations = find_all_tolerant_positive_involvement_violations, 
)
    

def has_bullet_vote_positive_involvement_violation(prof, vm, verbose=False, coalition_size = 1, require_resoluteness = False, require_uniquely_weighted = False, check_probabilities = False):
    """
    Returns True if it is possible to cause a winner A to lose by adding coalition_size-many new voters who bullet vote for A.

    If require_resoluteness = True, then only profiles with a unique winner are considered.

    If require_uniquely_weighted = True, then only uniquely-weighted profiles are considered.

    If check_probabilities = True, then the function also checks whether adding coalition_size-many new voters who bullet vote for A causes A's probability of winning to decrease (in the case of a tie broken by even-chance tiebreaking).
    
    Args:
        prof: a Profile or ProfileWithTies object.
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
                prof = prof.anonymize()
                new_prof = new_prof.anonymize()
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
        
        if check_probabilities and len(new_ws) > len(ws):

            if verbose:
                prof = prof.anonymize()
                new_prof = new_prof.anonymize()
                print(f"Violation of Probabilistic Bullet Vote Positive Involvement for {vm.name}")
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

def find_all_bullet_vote_positive_involvement_violations(prof, vm, verbose=False, coalition_size = 1, require_resoluteness = False, require_uniquely_weighted = False, check_probabilities = False):
    """
    Returns a list of candidates who win in the given profile but lose after adding coalition_size-many new voters who bullet vote for them.

    If require_resoluteness = True, then only profiles with a unique winner are considered.

    If require_uniquely_weighted = True, then only uniquely-weighted profiles are considered.

    If check_probabilities = True, then the function also checks whether adding coalition_size-many new voters who bullet vote for A causes A's probability of winning to decrease (in the case of a tie broken by even-chance tiebreaking).

    Args:
        prof: a Profile or ProfileWithTies object.
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
                prof = prof.anonymize()
                new_prof = new_prof.anonymize()
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

        if check_probabilities and len(new_ws) > len(ws):
                
                if verbose:
                    prof = prof.anonymize()
                    new_prof = new_prof.anonymize()
                    print(f"Violation of Probabilistic Bullet Vote Positive Involvement for {vm.name}")
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

def has_participation_violation(prof, vm, verbose = False, violation_type = "Removal", coalition_size = 1, uniform_coalition = True, set_preference = "single-winner"):
    """
    If violation_type = "Removal", returns True if removing some voter(s) from prof changes the vm winning set such that the (each) voter prefers the new winner(s) to the original winner(s), according to the set_preference relation.

    If violation_type = "Addition", returns True if adding some voter(s) from prof changes the vm winning set such that the (each) voter prefers the original winner(s) to the new winner(s), according to the set_preference relation.

    If coalition_size > 1, checks for a violation involving a coalition of voters acting together.

    If uniform_coalition = True, all voters in the coalition must have the same ranking.

    If set_preference = "single-winner", a voter prefers a set A of candidates to a set B of candidates if A and B are singletons and the voter ranks the candidate in A above the candidate in B.

    If set_preference = "weak-dominance", a voter prefers a set A to a set B if in their sincere ranking, all candidates in A are weakly above all candidates in B and some candidate in A is strictly above some candidate in B.

    If set_preference = "optimist", a voter prefers a set A to a set B if in their sincere ranking, their favorite from A is above their favorite from B.

    If set_preference = "pessimist", a voter prefers a set A to a set B if in their sincere ranking, their least favorite from A is above their least favorite from B.
    
    Args:
        prof: a Profile or ProfileWithTies object.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation.
        violation_type: default is "Removal"
        coalition_size: default is 1
        uniform_coalition: default is True
        set_preference: default is "single-winner". Other options are "weak-dominance", "optimist", and "pessimist".
        
    Returns:
        Result of the test (bool): Returns True if there is a violation and False otherwise.

    """
    
    winners = vm(prof)

    if isinstance(prof,ProfileWithTies):
        prof.use_extended_strict_preference()

    found_manipulator = False

    ranking_types = prof.ranking_types

    ws = vm(prof)

    if set_preference == "single-winner":
        if len(ws) > 1:
            return False
        
    if uniform_coalition:
        
        if violation_type == "Removal":

            relevant_ranking_types = [r for r in prof.ranking_types if prof.rankings.count(r) >= coalition_size]

            for r in relevant_ranking_types:
                if not found_manipulator:

                    ranking_tokens = [r for r in prof.rankings]

                    for i in range(coalition_size):
                        ranking_tokens.remove(r) # remove coalition_size-many tokens of the type of ranking

                    if isinstance(prof,Profile):

                        new_prof = Profile(ranking_tokens)
                        new_ws = vm(new_prof)

                        old_winner_to_compare = None
                        new_winner_to_compare = None

                        if set_preference == "single-winner" and len(new_ws) == 1:

                            old_winner_to_compare = ws[0]
                            new_winner_to_compare = new_ws[0] 
                        
                        elif set_preference == "weak-dominance":
                            r_as_ranking = Ranking({c: i for i, c in enumerate(r)})
                        
                        elif set_preference == "optimist":
                                
                            old_winner_to_compare = [cand for cand in r if cand in ws][0]
                            new_winner_to_compare = [cand for cand in r if cand in new_ws][0]

                        elif set_preference == "pessimist":
                            
                            old_winner_to_compare = [cand for cand in r if cand in ws][-1] 
                            new_winner_to_compare = [cand for cand in r if cand in new_ws][-1]

                        if old_winner_to_compare is not None and r.index(old_winner_to_compare) > r.index(new_winner_to_compare) or (set_preference == "weak-dominance" and r_as_ranking.weak_dom(new_ws,ws)):
                            
                            found_manipulator = True

                            if verbose:
                                prof = prof.anonymize()
                                new_prof = new_prof.anonymize()
                                print(f"Violation of Participation for {vm.name} under the {set_preference} set preference.")
                                if coalition_size == 1:
                                    print(f"A voter with the ranking {r} can benefit by abstaining.")
                                else:
                                    print(f"{coalition_size} voters with the ranking {r} can benefit by jointly abstaining.")
                                print("")
                                print("Original Profile:")
                                prof.display()
                                print(prof.description())
                                print("")
                                vm.display(prof)
                                prof.display_margin_graph()
                                print("")
                                if coalition_size == 1:
                                    print("Profile if the voter abstains:")
                                else:
                                    print("Profile if the voters abstain:")
                                new_prof.display()
                                print(new_prof.description())
                                print("")
                                vm.display(new_prof)
                                new_prof.display_margin_graph()

                    if isinstance(prof,ProfileWithTies):
                        r_dict = r.rmap

                        new_prof = ProfileWithTies(ranking_tokens, candidates = prof.candidates)
                        new_prof.use_extended_strict_preference()
                        new_ws = vm(new_prof)

                        ranked_old_winners = [c for c in ws if c in r_dict.keys()]
                        ranked_new_winners = [c for c in new_ws if c in r_dict.keys()]

                        rank_of_old_winner_to_compare = None
                        rank_of_new_winner_to_compare = None

                        if set_preference == "single-winner" and len(new_ws) == 1:

                            rank_of_old_winner_to_compare = r_dict[ws[0]] if ranked_old_winners else math.inf
                            rank_of_new_winner_to_compare = r_dict[new_ws[0]] if ranked_new_winners else math.inf
                        
                        elif set_preference == "optimist":

                            rank_of_old_winner_to_compare = min([r_dict[c] for c in ranked_old_winners]) if ranked_old_winners else math.inf
                            rank_of_new_winner_to_compare = min([r_dict[c] for c in ranked_new_winners]) if ranked_new_winners else math.inf

                        elif set_preference == "pessimist":

                            rank_of_old_winner_to_compare = max([r_dict[c] for c in ranked_old_winners]) if ranked_old_winners == ws else math.inf
                            rank_of_new_winner_to_compare = max([r_dict[c] for c in ranked_new_winners]) if ranked_new_winners == new_ws else math.inf

                        if rank_of_old_winner_to_compare is not None and rank_of_old_winner_to_compare > rank_of_new_winner_to_compare or (set_preference == "weak-dominance" and r.weak_dom(new_ws,ws,use_extended_preferences=True)):
                            
                            found_manipulator = True

                            if verbose:
                                prof = prof.anonymize()
                                new_prof = new_prof.anonymize()
                                print(f"Violation of Participation for {vm.name} under the {set_preference} set preference.")
                                if coalition_size == 1:
                                    print(f"A voter with the ranking {r} can benefit by abstaining.")
                                else:
                                    print(f"{coalition_size} voters with the ranking {r} can benefit by jointly abstaining.")
                                print("")
                                print("Original Profile:")
                                prof.display()
                                print(prof.description())
                                print("")
                                vm.display(prof)
                                prof.display_margin_graph()
                                print("")
                                if coalition_size == 1:
                                    print("Profile if the voter abstains:")
                                else:
                                    print("Profile if the voters abstain:")
                                new_prof.display()
                                print(new_prof.description())
                                print("")
                                vm.display(new_prof)
                                new_prof.display_margin_graph()

        if violation_type == "Addition":

            if isinstance(prof,Profile):

                for new_r in permutations(prof.candidates):
                    if not found_manipulator:

                        new_ranking_tokens = [r for r in prof.rankings]

                        for i in range(coalition_size):
                            new_ranking_tokens.append(new_r)

                        new_prof = Profile(new_ranking_tokens)
                        new_ws = vm(new_prof)

                        old_winner_to_compare = None
                        new_winner_to_compare = None

                        if set_preference == "single-winner" and len(new_ws) == 1:

                            old_winner_to_compare = ws[0]
                            new_winner_to_compare = new_ws[0] 
                        
                        elif set_preference == "weak-dominance":
                            new_r_as_ranking = Ranking({c: i for i, c in enumerate(new_r)})
                        
                        elif set_preference == "optimist":
                                
                            old_winner_to_compare = [cand for cand in new_r if cand in ws][0]
                            new_winner_to_compare = [cand for cand in new_r if cand in new_ws][0]

                        elif set_preference == "pessimist":
                            
                            old_winner_to_compare = [cand for cand in new_r if cand in ws][-1] 
                            new_winner_to_compare = [cand for cand in new_r if cand in new_ws][-1]

                        if old_winner_to_compare is not None and new_r.index(old_winner_to_compare) < new_r.index(new_winner_to_compare) or (set_preference == "weak-dominance" and new_r_as_ranking.weak_dom(ws,new_ws)):
                            
                            found_manipulator = True

                            if verbose:
                                prof = prof.anonymize()
                                new_prof = new_prof.anonymize()
                                print(f"Violation of Participation for {vm.name} under the {set_preference} set preference.")
                                if coalition_size == 1:
                                    print(f"A new voter who joins with the ranking {new_r} will wish they had abstained.")
                                else:
                                    print(f"{coalition_size} new voters who join with the ranking {new_r} will wish they had jointly abstained.")
                                print("")
                                print("Original Profile without voter(s):")
                                prof.display()
                                print(prof.description())
                                print("")
                                vm.display(prof)
                                prof.display_margin_graph()
                                print("")
                                print("New Profile with voter(s) added:")
                                new_prof.display()
                                print(new_prof.description())
                                print("")
                                vm.display(new_prof)
                                new_prof.display_margin_graph()

            if isinstance(prof,ProfileWithTies):

                for _new_r in weak_orders(prof.candidates):
                    new_r = Ranking(_new_r)
                    new_r_dict = new_r.rmap

                    if not found_manipulator:

                        new_ranking_tokens = [r for r in prof.rankings] 

                        for i in range(coalition_size):
                            new_ranking_tokens.append(new_r)

                        new_prof = ProfileWithTies(new_ranking_tokens, candidates = prof.candidates)
                        new_prof.use_extended_strict_preference()
                        new_ws = vm(new_prof)

                        ranked_old_winners = [c for c in ws if c in new_r_dict.keys()]
                        ranked_new_winners = [c for c in new_ws if c in new_r_dict.keys()]

                        rank_of_old_winner_to_compare = None
                        rank_of_new_winner_to_compare = None

                        if set_preference == "single-winner" and len(new_ws) == 1:

                            rank_of_old_winner_to_compare = new_r_dict[ws[0]] if ranked_old_winners else math.inf
                            rank_of_new_winner_to_compare = new_r_dict[new_ws[0]] if ranked_new_winners else math.inf
                        
                        elif set_preference == "optimist":

                            rank_of_old_winner_to_compare = min([new_r_dict[c] for c in ranked_old_winners]) if ranked_old_winners else math.inf
                            rank_of_new_winner_to_compare = min([new_r_dict[c] for c in ranked_new_winners]) if ranked_new_winners else math.inf

                        elif set_preference == "pessimist":

                            rank_of_old_winner_to_compare = max([new_r_dict[c] for c in ranked_old_winners]) if ranked_old_winners == ws else math.inf
                            rank_of_new_winner_to_compare = max([new_r_dict[c] for c in ranked_new_winners]) if ranked_new_winners == new_ws else math.inf

                        if rank_of_old_winner_to_compare is not None and rank_of_old_winner_to_compare < rank_of_new_winner_to_compare or (set_preference == "weak-dominance" and r.weak_dom(ws,new_ws,use_extended_preferences=True)):
                            
                            found_manipulator = True

                            if verbose:
                                prof = prof.anonymize()
                                new_prof = new_prof.anonymize()
                                print(f"Violation of Participation for {vm.name} under the {set_preference} set preference.")
                                if coalition_size == 1:
                                    print(f"A new voter who joins with the ranking {new_r} will wish they had abstained.")
                                else:
                                    print(f"{coalition_size} new voters who join with the ranking {new_r} will wish they had jointly abstained.")
                                print("")
                                print("Original Profile without voter(s):")
                                prof.display()
                                print(prof.description())
                                print("")
                                vm.display(prof)
                                prof.display_margin_graph()
                                print("")
                                print("New Profile with voter(s) added:")
                                new_prof.display()
                                print(new_prof.description())
                                print("")
                                vm.display(new_prof)
                                new_prof.display_margin_graph()
            
    return found_manipulator

def find_all_participation_violations(prof, vm, verbose = False, violation_type = "Removal", coalition_size = 1, uniform_coalition = True, set_preference = "single-winner"):
    """
    Returns a list of tuples (preferred_winners, dispreferred_winners, ranking) witnessing violations of participation.

    If violation_type = "Removal", returns a list of tuples (preferred_winners, dispreferred_winners, ranking) such that removing coalition_size-many voters with the given ranking changes the winning set from the preferred_winners to the dispreferred_winners, according to the set_preference relation.

    If violation_type = "Addition", returns a list of tuples (preferred_winners, dispreferred_winners, ranking) such that adding coalition_size-many voters with the given ranking changes the winning set from the preferred_winners to the dispreferred_winners, according to the set_preference relation.

    If coalition_size > 1, checks for a violation involving a coalition of voters acting together.

    If uniform_coalition = True, all voters in the coalition must have the same ranking.

    If set_preference = "single-winner", a voter prefers a set A of candidates to a set B of candidates if A and B are singletons and the voter ranks the candidate in A above the candidate in B.

    If set_preference = "weak-dominance", a voter prefers a set A to a set B if in their sincere ranking, all candidates in A are weakly above all candidates in B and some candidate in A is strictly above some candidate in B.

    If set_preference = "optimist", a voter prefers a set A to a set B if in their sincere ranking, their favorite from A is above their favorite from B.

    If set_preference = "pessimist", a voter prefers a set A to a set B if in their sincere ranking, their least favorite from A is above their least favorite from B.

    Args:
        prof: a Profile or ProfileWithTies object.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation.
        violation_type: default is "Removal"
        coalition_size: default is 1
        uniform_coalition: default is True
        set_preference: default is "single-winner". Other options are "weak-dominance", "optimist", and "pessimist".

    Returns:
        A List of tuples (preferred_winners, dispreferred_winners, ranking) witnessing violations of participation.
    """

    violations = list()

    winners = vm(prof)

    if isinstance(prof,ProfileWithTies):
        prof.use_extended_strict_preference()

    ranking_types = prof.ranking_types

    ws = vm(prof)

    if set_preference == "single-winner":
        if len(ws) > 1:
            return False
        
    if uniform_coalition:
        
        if violation_type == "Removal":

            relevant_ranking_types = [r for r in prof.ranking_types if prof.rankings.count(r) >= coalition_size]

            for r in relevant_ranking_types:
                ranking_tokens = [r for r in prof.rankings]

                for i in range(coalition_size):
                    ranking_tokens.remove(r) # remove coalition_size-many tokens of the type of ranking

                if isinstance(prof,Profile):

                    new_prof = Profile(ranking_tokens)
                    new_ws = vm(new_prof)

                    old_winner_to_compare = None
                    new_winner_to_compare = None

                    if set_preference == "single-winner" and len(new_ws) == 1:

                        old_winner_to_compare = ws[0]
                        new_winner_to_compare = new_ws[0] 
                    
                    elif set_preference == "weak-dominance":
                        r_as_ranking = Ranking({c: i for i, c in enumerate(r)})
                    
                    elif set_preference == "optimist":
                            
                        old_winner_to_compare = [cand for cand in r if cand in ws][0]
                        new_winner_to_compare = [cand for cand in r if cand in new_ws][0]

                    elif set_preference == "pessimist":
                        
                        old_winner_to_compare = [cand for cand in r if cand in ws][-1] 
                        new_winner_to_compare = [cand for cand in r if cand in new_ws][-1]

                    if old_winner_to_compare is not None and r.index(old_winner_to_compare) > r.index(new_winner_to_compare) or (set_preference == "weak-dominance" and r_as_ranking.weak_dom(new_ws,ws)):
                        
                        violations.append((ws, new_ws, r))

                        if verbose:
                            prof = prof.anonymize()
                            new_prof = new_prof.anonymize()
                            print(f"Violation of Participation for {vm.name} under the {set_preference} set preference.")
                            if coalition_size == 1:
                                print(f"A voter with the ranking {r} can benefit by abstaining.")
                            else:
                                print(f"{coalition_size} voters with the ranking {r} can benefit by jointly abstaining.")
                            print("")
                            print("Original Profile:")
                            prof.display()
                            print(prof.description())
                            print("")
                            vm.display(prof)
                            prof.display_margin_graph()
                            print("")
                            if coalition_size == 1:
                                print("Profile if the voter abstains:")

                            else:
                                print("Profile if the voters abstain:")
                            new_prof.display()
                            print(new_prof.description())
                            print("")
                            vm.display(new_prof)
                            new_prof.display_margin_graph()

                if isinstance(prof,ProfileWithTies):
                    r_dict = r.rmap

                    new_prof = ProfileWithTies(ranking_tokens, candidates = prof.candidates)
                    new_prof.use_extended_strict_preference()
                    new_ws = vm(new_prof)

                    ranked_old_winners = [c for c in ws if c in r_dict.keys()]
                    ranked_new_winners = [c for c in new_ws if c in r_dict.keys()]

                    rank_of_old_winner_to_compare = None
                    rank_of_new_winner_to_compare = None

                    if set_preference == "single-winner" and len(new_ws) == 1:

                        rank_of_old_winner_to_compare = r_dict[ws[0]] if ranked_old_winners else math.inf
                        rank_of_new_winner_to_compare = r_dict[new_ws[0]] if ranked_new_winners else math.inf
                    
                    elif set_preference == "optimist":

                        rank_of_old_winner_to_compare = min([r_dict[c] for c in ranked_old_winners]) if ranked_old_winners else math.inf
                        rank_of_new_winner_to_compare = min([r_dict[c] for c in ranked_new_winners]) if ranked_new_winners else math.inf

                    elif set_preference == "pessimist":

                        rank_of_old_winner_to_compare = max([r_dict[c] for c in ranked_old_winners]) if ranked_old_winners == ws else math.inf
                        rank_of_new_winner_to_compare = max([r_dict[c] for c in ranked_new_winners]) if ranked_new_winners == new_ws else math.inf

                    if rank_of_old_winner_to_compare is not None and rank_of_old_winner_to_compare > rank_of_new_winner_to_compare or (set_preference == "weak-dominance" and r.weak_dom(new_ws,ws,use_extended_preferences=True)):
                        
                        violations.append((ws, new_ws, r_dict))

                        if verbose:
                            prof = prof.anonymize()
                            new_prof = new_prof.anonymize()
                            print(f"Violation of Participation for {vm.name} under the {set_preference} set preference.")
                            if coalition_size == 1:
                                print(f"A voter with the ranking {r} can benefit by abstaining.")
                            else:
                                print(f"{coalition_size} voters with the ranking {r} can benefit by jointly abstaining.")
                            print("")
                            print("Original Profile:")
                            prof.display()
                            print(prof.description())
                            print("")
                            vm.display(prof)
                            prof.display_margin_graph()
                            print("")
                            if coalition_size == 1:
                                print("Profile if the voter abstains:")
                            else:
                                print("Profile if the voters abstain:")
                            new_prof.display()
                            print(new_prof.description())
                            print("")
                            vm.display(new_prof)
                            new_prof

        if violation_type == "Addition":

            if isinstance(prof,Profile):

                for new_r in permutations(prof.candidates):
                    new_ranking_tokens = [r for r in prof.rankings]

                    for i in range(coalition_size):
                        new_ranking_tokens.append(new_r)

                    new_prof = Profile(new_ranking_tokens)
                    new_ws = vm(new_prof)

                    old_winner_to_compare = None
                    new_winner_to_compare = None

                    if set_preference == "single-winner" and len(new_ws) == 1:

                        old_winner_to_compare = ws[0]
                        new_winner_to_compare = new_ws[0] 
                    
                    elif set_preference == "weak-dominance":
                        new_r_as_ranking = Ranking({c: i for i, c in enumerate(new_r)})
                    
                    elif set_preference == "optimist":
                            
                        old_winner_to_compare = [cand for cand in new_r if cand in ws][0]
                        new_winner_to_compare = [cand for cand in new_r if cand in new_ws][0]

                    elif set_preference == "pessimist":
                        
                        old_winner_to_compare = [cand for cand in new_r if cand in ws][-1] 
                        new_winner_to_compare = [cand for cand in new_r if cand in new_ws][-1]

                    if old_winner_to_compare is not None and new_r.index(old_winner_to_compare) < new_r.index(new_winner_to_compare) or (set_preference == "weak-dominance" and new_r_as_ranking.weak_dom(ws,new_ws)):
                        
                        violations.append((ws, new_ws, new_r))

                        if verbose:
                            prof = prof.anonymize()
                            new_prof = new_prof.anonymize()
                            print(f"Violation of Participation for {vm.name} under the {set_preference} set preference.")
                            if coalition_size == 1:
                                print(f"A new voter who joins with the ranking {new_r} will wish they had abstained.")
                            else:
                                print(f"{coalition_size} new voters who join with the ranking {new_r} will wish they had jointly abstained.")
                            print("")
                            print("Original Profile without voter(s):")
                            prof.display()
                            print(prof.description())
                            print("")
                            vm.display(prof)
                            prof.display_margin_graph()
                            print("")
                            print("New Profile with voter(s) added:")
                            new_prof.display()
                            print(new_prof.description())
                            print("")
                            vm.display(new_prof)
                            new_prof.display_margin_graph()

            if isinstance(prof,ProfileWithTies):
                
                    for _new_r in weak_orders(prof.candidates):
                        new_r = Ranking(_new_r)
                        new_r_dict = new_r.rmap
    
                        new_ranking_tokens = [r for r in prof.rankings] 
    
                        for i in range(coalition_size):
                            new_ranking_tokens.append(new_r)
    
                        new_prof = ProfileWithTies(new_ranking_tokens, candidates = prof.candidates)
                        new_prof.use_extended_strict_preference()
                        new_ws = vm(new_prof)
    
                        ranked_old_winners = [c for c in ws if c in new_r_dict.keys()]
                        ranked_new_winners = [c for c in new_ws if c in new_r_dict.keys()]
    
                        rank_of_old_winner_to_compare = None
                        rank_of_new_winner_to_compare = None
    
                        if set_preference == "single-winner" and len(new_ws) == 1:
    
                            rank_of_old_winner_to_compare = new_r_dict[ws[0]] if ranked_old_winners else math.inf
                            rank_of_new_winner_to_compare = new_r_dict[new_ws[0]] if ranked_new_winners else math.inf
                        
                        elif set_preference == "optimist":
    
                            rank_of_old_winner_to_compare = min([new_r_dict[c] for c in ranked_old_winners]) if ranked_old_winners else math.inf
                            rank_of_new_winner_to_compare = min([new_r_dict[c] for c in ranked_new_winners]) if ranked_new_winners else math.inf
    
                        elif set_preference == "pessimist":
    
                            rank_of_old_winner_to_compare = max([new_r_dict[c] for c in ranked_old_winners]) if ranked_old_winners == ws else math.inf
                            rank_of_new_winner_to_compare = max([new_r_dict[c] for c in ranked_new_winners]) if ranked_new_winners == new_ws else math.inf
    
                        if rank_of_old_winner_to_compare is not None and rank_of_old_winner_to_compare < rank_of_new_winner_to_compare or (set_preference == "weak-dominance" and r.weak_dom(ws,new_ws,use_extended_preferences=True)):
                            
                            violations.append((ws, new_ws, new_r_dict))
    
                            if verbose:
                                prof = prof.anonymize()
                                new_prof = new_prof.anonymize()
                                print(f"Violation of Participation for {vm.name} under the {set_preference} set preference.")
                                if coalition_size == 1:
                                    print(f"A new voter who joins with the ranking {new_r} will wish they had abstained.")
                                else:
                                    print(f"{coalition_size} new voters who join with the ranking {new_r} will wish they had jointly abstained.")
                                print("")
                                print(" Original Profile without voter(s):")
                                prof.display()
                                print(prof.description())
                                print("")
                                vm.display(prof)
                                prof.display_margin_graph()
                                print("")
                                print("New Profile with voter(s) added:")
                                new_prof.display()
                                print(new_prof.description())
                                print("")
                                vm.display(new_prof)
                                new_prof.display_margin_graph()

    return violations

participation = Axiom(
    "Participation",
    has_violation = has_participation_violation,
    find_all_violations = find_all_participation_violations, 
)

def has_single_voter_resolvability_violation(prof, vm, verbose=False):
    """
    If prof is a Profile, returns True if there are multiple vm winners in prof and for one such winner A, there is no linear ballot that can be added to prof to make A the unique winner.

    If prof is a ProfileWithTies, returns True if there are multiple vm winners in prof and for one such winner A, there is no Ranking (allowing ties) that can be added to prof to make A the unique winner. 

    Args:
        prof: a Profile or ProfileWithTies object.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation.

    Returns:
        Result of the test (bool): Returns True if there is a violation and False otherwise.
    """

    winners = vm(prof)

    if isinstance(prof,ProfileWithTies):
        prof.use_extended_strict_preference()

    if len(winners) > 1:
        for winner in winners:

            found_voter_to_add = False

            if isinstance(prof,Profile):
                for r in permutations(prof.candidates):
                    new_prof = Profile(prof.rankings + [r])
                    if vm(new_prof) == [winner]:
                        found_voter_to_add = True
                        break
                   
            if isinstance(prof,ProfileWithTies):
                for _r in weak_orders(prof.candidates):
                    r = Ranking(_r)
                    new_prof = ProfileWithTies(prof.rankings + [r], candidates = prof.candidates)
                    new_prof.use_extended_strict_preference()
                    if vm(new_prof) == [winner]:
                        found_voter_to_add = True
                        break
        
            if not found_voter_to_add:

                if verbose:
                    prof = prof.anonymize()
                    if isinstance(prof,Profile):
                        print(f"Violation of Single-Voter Resolvability for {vm.name}: cannot make {winner} the unique winner by adding a linear ballot.")
                    if isinstance(prof,ProfileWithTies):
                        print(f"Violation of Single-Voter Resolvability for {vm.name}: cannot make {winner} the unique winner by adding a Ranking.")
                    print("")
                    print("Profile:")
                    prof.display()
                    print(prof.description())
                    print("")
                    vm.display(prof)
                    prof.display_margin_graph()
                    print("")

                return True
            
        return False

def find_all_single_voter_resolvability_violations(prof, vm, verbose=False):
    """
    If prof is a Profile, returns a list of candidates who win in prof but who cannot be made the unique winner by adding a linear ballot.

    If prof is a ProfileWithTies, returns a list of candidates who win in prof but who cannot be made the unique winner by adding a Ranking (allowing ties).

    Args:
        prof: a Profile or ProfileWithTies object.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation.

    Returns:
        A List of candidates who win in the given profile but who cannot be made the unique winner by adding a ballot.
    """

    winners = vm(prof)

    if isinstance(prof,ProfileWithTies):
        prof.use_extended_strict_preference()

    violations = list()

    if len(winners) > 1:
        for winner in winners:

            found_voter_to_add = False

            if isinstance(prof,Profile):
                for r in permutations(prof.candidates):
                    new_prof = Profile(prof.rankings + [r])
                    if vm(new_prof) == [winner]:
                        found_voter_to_add = True
                        break
                   
            if isinstance(prof,ProfileWithTies):
                for _r in weak_orders(prof.candidates):
                    r = Ranking(_r)
                    new_prof = ProfileWithTies(prof.rankings + [r], candidates = prof.candidates)
                    new_prof.use_extended_strict_preference()
                    if vm(new_prof) == [winner]:
                        found_voter_to_add = True
                        break
        
            if not found_voter_to_add:

                if verbose:
                    prof = prof.anonymize()
                    if isinstance(prof,Profile):
                        print(f"Violation of Single-Voter Resolvability for {vm.name}: cannot make {winner} the unique winner by adding a linear ballot.")
                    if isinstance(prof,ProfileWithTies):
                        print(f"Violation of Single-Voter Resolvability for {vm.name}: cannot make {winner} the unique winner by adding a Ranking.")
                    print("")
                    print("Profile:")
                    prof.display()
                    print(prof.description())
                    print("")
                    vm.display(prof)
                    prof.display_margin_graph()
                    print("")

                violations.append(winner)
            
    return violations

single_voter_resolvability = Axiom(
    "Single-Voter Resolvability",
    has_violation = has_single_voter_resolvability_violation,
    find_all_violations = find_all_single_voter_resolvability_violations, 
)

variable_voter_axioms = [
    reinforcement,
    positive_involvement,
    negative_involvement,
    tolerant_positive_involvement,
    bullet_vote_positive_involvement,
    participation,
    single_voter_resolvability,
]