"""
    File: variable_voter_axioms.py
    Author: Wes Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)
    Date: March 16, 2024
    
    Variable voter axioms 
"""

from pref_voting.axiom import Axiom
from pref_voting.axiom_helpers import *
import numpy as np
from itertools import product, combinations, permutations
from pref_voting.helper import weak_orders
from pref_voting.rankings import Ranking
from pref_voting.generate_profiles import strict_weak_orders

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

            if isinstance(prof, Profile):
                rankings1 = R[nonzero_indices_C1].tolist()
                rankings2 = R[nonzero_indices_C2].tolist()
            else:  # ProfileWithTies
                rankings1 = [R[i] for i in nonzero_indices_C1]
                rankings2 = [R[i] for i in nonzero_indices_C2]
            
            counts1 = C1[nonzero_indices_C1].tolist()
            counts2 = C2[nonzero_indices_C2].tolist()

            # Convert rankings to comparable format for ordering check
            if isinstance(prof, Profile):
                comparable1 = rankings1
                comparable2 = rankings2
            else:  # ProfileWithTies - convert Ranking objects to tuples for comparison
                comparable1 = [tuple(r.rmap) for r in rankings1]
                comparable2 = [tuple(r.rmap) for r in rankings2]
            
            if comparable1 <= comparable2: # This prevents yielding both prof1, prof2 and later on prof2, prof1, unless they are equal

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

def has_negative_involvement_violation(prof, vm, verbose=False, violation_type="Removal", coalition_size=1, uniform_coalition=True, require_resoluteness=False, require_uniquely_weighted=False, check_probabilities=False):
    """
    If violation_type = "Removal", returns True if removing some voter(s) who ranked a winning candidate B in last place causes B to lose, witnessing a violation of negative involvement.
    
    If uniform_coalition = True, then only coalitions of voters with the same ranking are considered.

    If require_resoluteness = True, then only profiles with a unique winner are considered.

    If require_uniquely_weighted = True, then only uniquely-weighted profiles are considered.

    If check_probabilities = True, the function also checks whether removing the voters who ranked B in last-place causes B's probability of winning to decrease (in the case of a tie broken by even-chance tiebreaking).
    
    Args:
        prof: a Profile or ProfileWithTies object.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation.
        violation_type: default is "Removal"
        coalition_size: default is 1
        uniform_coalition: default is True
        require_resoluteness: default is False
        require_uniquely_weighted: default is False
        check_probabilities: default is False
        
    Returns:
        Result of the test (bool): Returns True if there is a violation and False otherwise."""
    
    winners = vm(prof)   
    
    if require_resoluteness and len(winners) > 1:
        return False

    if require_uniquely_weighted and not prof.is_uniquely_weighted():
        return False
    
    if violation_type == "Removal":
        if uniform_coalition:
            for winner in winners:
                relevant_ranking_types = [r for r in prof.ranking_types if r[-1] == winner and prof.rankings.count(r) >= coalition_size]

                for r in relevant_ranking_types:
                    rankings = prof.rankings.copy()

                    for i in range(coalition_size):
                        rankings.remove(r) # remove coalition_size-many tokens of the type of ranking

                    if isinstance(prof, Profile):
                        prof2 = Profile(rankings)

                    if isinstance(prof, ProfileWithTies):
                        prof2 = ProfileWithTies(rankings, candidates=prof.candidates)
                        if prof.using_extended_strict_preference:
                            prof2.use_extended_strict_preference()

                    winners2 = vm(prof2)

                    if require_resoluteness and len(winners2) > 1:
                        continue

                    if require_uniquely_weighted and not prof2.is_uniquely_weighted():
                        continue
                    
                    if winner not in winners2:
                        if verbose:
                            prof = prof.anonymize()
                            if coalition_size == 1:
                                print(f"{winner} wins in the full profile, but {winner} is a loser after removing a voter with the ranking {str(r)}:")
                            else:
                                print(f"{winner} wins in the full profile, but {winner} is a loser after removing {coalition_size} voters with the ranking {str(r)}:")
                            print("")
                            print("Full profile")
                            prof.display()
                            print(prof.description())
                            prof.display_margin_graph()
                            vm.display(prof)
                            print("")
                            if coalition_size == 1:
                                print("Profile with voter removed")
                            else:
                                print(f"Profile with {coalition_size} voters removed")
                            prof2 = prof2.anonymize()
                            prof2.display()
                            print(prof2.description())
                            prof2.display_margin_graph()
                            vm.display(prof2)
                            print("")
                        return True
                    
                    # Case: B's probability of winning decreases as winning set expands
                    if check_probabilities and winner in winners2 and len(winners) < len(winners2):
                        if verbose:
                            prof = prof.anonymize()
                            if coalition_size == 1:
                                print(f"{winner} becomes less likely to win after removing a voter with the ranking {str(r)}:")
                            else:
                                print(f"{winner} becomes less likely to win after removing {coalition_size} voters with the ranking {str(r)}:")
                            print("")
                            print("Full profile")
                            prof.display()
                            print(prof.description())
                            prof.display_margin_graph()
                            vm.display(prof)
                            print("")
                            if coalition_size == 1:
                                print("Profile with voter removed")
                            else:
                                print(f"Profile with {coalition_size} voters removed")
                            prof2 = prof2.anonymize()
                            prof2.display()
                            print(prof2.description())
                            prof2.display_margin_graph()
                            vm.display(prof2)
                            print("")
                        return True
        
        if not uniform_coalition:
            for winner in winners:
                relevant_ranking_types = [r for r in prof.ranking_types if r[-1] == winner]
                relevant_ranking_types_counts = [prof.rankings.count(r) for r in relevant_ranking_types]

                for coalition_rankings, coalition_rankings_counts in _submultisets_of_fixed_cardinality(relevant_ranking_types, relevant_ranking_types_counts, coalition_size):
                    
                    rankings = prof.rankings.copy()
                    
                    for r_idx, r in enumerate(coalition_rankings):
                        for i in range(coalition_rankings_counts[r_idx]):
                            rankings.remove(r)
                        
                    if isinstance(prof, Profile):
                        prof2 = Profile(rankings)

                    if isinstance(prof, ProfileWithTies):
                        prof2 = ProfileWithTies(rankings, candidates=prof.candidates)
                        if prof.using_extended_strict_preference:
                            prof2.use_extended_strict_preference()

                    winners2 = vm(prof2)              

                    if require_resoluteness and len(winners2) > 1:
                        continue

                    if require_uniquely_weighted and not prof2.is_uniquely_weighted():
                        continue
                    
                    if winner not in winners2:
                        if verbose:
                            prof = prof.anonymize()
                            print(f"{winner} wins in the full profile, but {winner} is a loser after removing a {coalition_size}-voter coalition with the rankings {[str(r) for r in coalition_rankings]} and counts {coalition_rankings_counts}:")
                            print("")
                            print("Full profile")
                            prof.display()
                            print(prof.description())
                            prof.display_margin_graph()
                            vm.display(prof)
                            print("")
                            print(f"Profile with coalition removed")
                            prof2 = prof2.anonymize()
                            prof2.display()
                            print(prof2.description())
                            prof2.display_margin_graph()
                            vm.display(prof2)
                            print("")
                        return True
                    
                    # Case: B's probability of winning decreases as winning set expands
                    if check_probabilities and winner in winners2 and len(winners) < len(winners2):
                        if verbose:
                            prof = prof.anonymize()
                            print(f"{winner} becomes less likely to win after removing a {coalition_size}-voter coalition with the rankings {[str(r) for r in coalition_rankings]} and counts {coalition_rankings_counts}:")
                            print("")
                            print("Full profile")
                            prof.display()
                            print(prof.description())
                            prof.display_margin_graph()
                            vm.display(prof)
                            print("")
                            print(f"Profile with coalition removed")
                            prof2 = prof2.anonymize()
                            prof2.display()
                            print(prof2.description())
                            prof2.display_margin_graph()
                            vm.display(prof2)
                            print("")
                        return True
    
    return False
                    
def find_all_negative_involvement_violations(prof, vm, verbose=False, violation_type="Removal", coalition_size=1, uniform_coalition=True, require_resoluteness=False, require_uniquely_weighted=False, check_probabilities=False):
    """
    If violation_type = "Removal", returns a list of tuples (winner, rankings, counts) such that removing the indicated rankings with the indicated counts causes the winner to lose, witnessing a violation of negative involvement.
    
    If uniform_coalition = True, then only coalitions of voters with the same ranking are considered.

    If require_resoluteness = True, then only profiles with a unique winner are considered.

    If require_uniquely_weighted = True, then only uniquely-weighted profiles are considered.

    If check_probabilities = True, the function also checks whether removing the voters who ranked B in last-place causes B's probability of winning to decrease (in the case of a tie broken by even-chance tiebreaking).
    
    Args:
        prof: a Profile or ProfileWithTies object.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation.
        violation_type: default is "Removal"
        coalition_size: default is 1
        uniform_coalition: default is True
        require_resoluteness: default is False
        require_uniquely_weighted: default is False
        check_probabilities: default is False
        
    Returns:
        A List of tuples (winner, rankings, counts) witnessing violations of negative involvement.
        
    .. warning::
        This function is slow when uniform_coalition = False and the numbers of voters and candidates are too large.
    """
    
    winners = vm(prof)   
    
    witnesses = list()

    if require_resoluteness and len(winners) > 1:
        return witnesses
    
    if require_uniquely_weighted and not prof.is_uniquely_weighted():
        return witnesses
    
    if violation_type == "Removal":
        if uniform_coalition:
            for winner in winners:
                relevant_ranking_types = [r for r in prof.ranking_types if r[-1] == winner and prof.rankings.count(r) >= coalition_size]

                for r in relevant_ranking_types:
                    rankings = prof.rankings.copy()

                    for i in range(coalition_size):
                        rankings.remove(r) # remove coalition_size-many tokens of the type of ranking

                    if isinstance(prof, Profile):
                        prof2 = Profile(rankings)

                    if isinstance(prof, ProfileWithTies):
                        prof2 = ProfileWithTies(rankings, candidates=prof.candidates)
                        if prof.using_extended_strict_preference:
                            prof2.use_extended_strict_preference()

                    winners2 = vm(prof2)

                    if require_resoluteness and len(winners2) > 1:
                        continue

                    if require_uniquely_weighted and not prof2.is_uniquely_weighted():
                        continue
                    
                    if winner not in winners2:
                        witnesses.append((winner, [r], [coalition_size]))
                        if verbose:
                            prof = prof.anonymize()
                            if coalition_size == 1:
                                print(f"{winner} wins in the full profile, but {winner} is a loser after removing a voter with the ranking {str(r)}:")
                            else:
                                print(f"{winner} wins in the full profile, but {winner} is a loser after removing {coalition_size} voters with the ranking {str(r)}:")
                            print("")
                            print("Full profile")
                            prof.display()
                            print(prof.description())
                            prof.display_margin_graph()
                            vm.display(prof)
                            print("")
                            if coalition_size == 1:
                                print("Profile with voter removed")
                            else:
                                print(f"Profile with {coalition_size} voters removed")
                            prof2 = prof2.anonymize()
                            prof2.display()
                            print(prof2.description())
                            prof2.display_margin_graph()
                            vm.display(prof2)
                            print("")
                    
                    # Case: B's probability of winning decreases as winning set expands
                    if check_probabilities and winner in winners2 and len(winners) < len(winners2):
                        witnesses.append((winner, [r], [coalition_size]))
                        if verbose:
                            prof = prof.anonymize()
                            if coalition_size == 1:
                                print(f"{winner} becomes less likely to win after removing a voter with the ranking {str(r)}:")
                            else:
                                print(f"{winner} becomes less likely to win after removing {coalition_size} voters with the ranking {str(r)}:")
                            print("")
                            print("Full profile")
                            prof.display()
                            print(prof.description())
                            prof.display_margin_graph()
                            vm.display(prof)
                            print("")
                            if coalition_size == 1:
                                print("Profile with voter removed")
                            else:
                                print(f"Profile with {coalition_size} voters removed")
                            prof2 = prof2.anonymize()
                            prof2.display()
                            print(prof2.description())
                            prof2.display_margin_graph()
                            vm.display(prof2)
                            print("")
        
        if not uniform_coalition:
            for winner in winners:
                relevant_ranking_types = [r for r in prof.ranking_types if r[-1] == winner]
                relevant_ranking_types_counts = [prof.rankings.count(r) for r in relevant_ranking_types]

                for coalition_rankings, coalition_rankings_counts in _submultisets_of_fixed_cardinality(relevant_ranking_types, relevant_ranking_types_counts, coalition_size):
                    
                    rankings = prof.rankings.copy()
                    
                    for r_idx, r in enumerate(coalition_rankings):
                        for i in range(coalition_rankings_counts[r_idx]):
                            rankings.remove(r)
                        
                    if isinstance(prof, Profile):
                        prof2 = Profile(rankings)

                    if isinstance(prof, ProfileWithTies):
                        prof2 = ProfileWithTies(rankings, candidates=prof.candidates)
                        if prof.using_extended_strict_preference:
                            prof2.use_extended_strict_preference()

                    winners2 = vm(prof2)              

                    if require_resoluteness and len(winners2) > 1:
                        continue

                    if require_uniquely_weighted and not prof2.is_uniquely_weighted():
                        continue
                    
                    if winner not in winners2:
                        witnesses.append((winner, coalition_rankings, coalition_rankings_counts))
                        if verbose:
                            prof = prof.anonymize()
                            print(f"{winner} wins in the full profile, but {winner} is a loser after removing a {coalition_size}-voter coalition with the rankings {[str(r) for r in coalition_rankings]} and counts {coalition_rankings_counts}:")
                            print("")
                            print("Full profile")
                            prof.display()
                            print(prof.description())
                            prof.display_margin_graph()
                            vm.display(prof)
                            print("")
                            print(f"Profile with coalition removed")
                            prof2 = prof2.anonymize()
                            prof2.display()
                            print(prof2.description())
                            prof2.display_margin_graph()
                            vm.display(prof2)
                            print("")
                    
                    # Case: B's probability of winning decreases as winning set expands
                    if check_probabilities and winner in winners2 and len(winners) < len(winners2):
                        witnesses.append((winner, coalition_rankings, coalition_rankings_counts))
                        if verbose:
                            prof = prof.anonymize()
                            print(f"{winner} becomes less likely to win after removing a {coalition_size}-voter coalition with the rankings {[str(r) for r in coalition_rankings]} and counts {coalition_rankings_counts}:")
                            print("")
                            print("Full profile")
                            prof.display()
                            print(prof.description())
                            prof.display_margin_graph()
                            vm.display(prof)
                            print("")
                            print(f"Profile with coalition removed")
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

def has_positive_negative_involvement_violation(prof, vm, verbose=False, violation_type="Removal", coalition_size=1, uniform_coalition=True, require_resoluteness=False, require_uniquely_weighted=False, check_probabilities=False):
    """
    If violation_type = "Removal", returns True if removing some voter(s) who ranked a losing candidate A in first place and a winning candidate B in last place causes A to win and B to lose, witnessing a violation of positive-negative involvement.
    
    If uniform_coalition = True, then only coalitions of voters with the same ranking are considered.

    If require_resoluteness = True, then only profiles with a unique winner are considered.

    If require_uniquely_weighted = True, then only uniquely-weighted profiles are considered.

    If check_probabilities = True, the function also checks whether removing the voters causes the probability of their favorite winning to increase and the probability of their least favorite winning to decrease.
    
    Args:
        prof: a Profile or ProfileWithTies object.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation.
        violation_type: default is "Removal"
        coalition_size: default is 1
        uniform_coalition: default is True
        require_resoluteness: default is False
        require_uniquely_weighted: default is False
        check_probabilities: default is False
        
    Returns:
        Result of the test (bool): Returns True if there is a violation and False otherwise.
    """
    
    winners = vm(prof)   
    non_winners = [c for c in prof.candidates if c not in winners]

    if require_resoluteness and len(winners) > 1:
        return False

    if require_uniquely_weighted and not prof.is_uniquely_weighted():
        return False
    
    if violation_type == "Removal":
        if uniform_coalition:
            # Check for standard positive-negative involvement violations and Case 1 probability violations
            for favorite in non_winners:
                for least_favorite in winners:
                    relevant_ranking_types = [r for r in prof.ranking_types if r[0] == favorite and r[-1] == least_favorite and prof.rankings.count(r) >= coalition_size]

                    for r in relevant_ranking_types:
                        rankings = prof.rankings

                        for i in range(coalition_size):
                            rankings.remove(r) # remove coalition_size-many tokens of the type of ranking

                        if isinstance(prof, Profile):
                            prof2 = Profile(rankings)

                        if isinstance(prof, ProfileWithTies):
                            prof2 = ProfileWithTies(rankings, candidates=prof.candidates)
                            if prof.using_extended_strict_preference:
                                prof2.use_extended_strict_preference()

                        winners2 = vm(prof2)

                        if require_resoluteness and len(winners2) > 1:
                            continue

                        if require_uniquely_weighted and not prof2.is_uniquely_weighted():
                            continue
                        
                        if favorite in winners2 and least_favorite not in winners2:
                            if verbose:
                                prof = prof.anonymize()
                                if coalition_size == 1:
                                    print(f"{favorite} loses and {least_favorite} wins in the full profile, but {favorite} is a winner and {least_favorite} is a loser after removing a voter with the ranking {str(r)}:")
                                else:
                                    print(f"{favorite} loses and {least_favorite} wins in the full profile, but {favorite} is a winner and {least_favorite} is a loser after removing {coalition_size} voters with the ranking {str(r)}:")
                                print("")
                                print("Full profile")
                                prof.display()
                                print(prof.description())
                                prof.display_margin_graph()
                                vm.display(prof)
                                print("")
                                if coalition_size == 1:
                                    print("Profile with voter removed")
                                else:
                                    print(f"Profile with {coalition_size} voters removed")
                                prof2 = prof2.anonymize()
                                prof2.display()
                                print(prof2.description())
                                prof2.display_margin_graph()
                                vm.display(prof2)
                                print("")
                            return True
                        
                        # Case 1: When removing a ballot, favorite goes from losing to winning, while least_favorite becomes less likely to win, since the winning set expands
                        # Viewed in terms of adding a ballot, favorite goes from winning to losing, while least_favorite becomes more likely to win, since the winning set shrinks
                        if check_probabilities and favorite in winners2 and least_favorite in winners2 and len(winners) < len(winners2):
                            if verbose:
                                prof = prof.anonymize()
                                if coalition_size == 1:
                                    print(f"{favorite} becomes more likely to win and {least_favorite} becomes less likely to win after removing a voter with the ranking {str(r)}:")
                                else:
                                    print(f"{favorite} becomes more likely to win and {least_favorite} becomes less likely to win after removing {coalition_size} voters with the ranking {str(r)}:")
                                print("")
                                print("Full profile")
                                prof.display()
                                print(prof.description())
                                prof.display_margin_graph()
                                vm.display(prof)
                                print("")
                                if coalition_size == 1:
                                    print("Profile with voter removed")
                                else:
                                    print(f"Profile with {coalition_size} voters removed")
                                prof2 = prof2.anonymize()
                                prof2.display()
                                print(prof2.description())
                                prof2.display_margin_graph()
                                vm.display(prof2)
                                print("")
                            return True
            
            # Check for Case 2 probability violations (where favorite is already a winner)
            if check_probabilities:
                for favorite in winners:
                    for least_favorite in winners:
                        if favorite == least_favorite:
                            continue
                        
                        relevant_ranking_types = [r for r in prof.ranking_types if r[0] == favorite and r[-1] == least_favorite and prof.rankings.count(r) >= coalition_size]

                    for r in relevant_ranking_types:
                        rankings = prof.rankings

                        for i in range(coalition_size):
                            rankings.remove(r) # remove coalition_size-many tokens of the type of ranking

                        if isinstance(prof, Profile):
                            prof2 = Profile(rankings)

                        if isinstance(prof, ProfileWithTies):
                            prof2 = ProfileWithTies(rankings, candidates=prof.candidates)
                            if prof.using_extended_strict_preference:
                                prof2.use_extended_strict_preference()

                        winners2 = vm(prof2)

                        if require_resoluteness and len(winners2) > 1:
                            continue

                        if require_uniquely_weighted and not prof2.is_uniquely_weighted():
                            continue
                        
                        # Case 2: When removing a ballot, least_favorite goes from winning to losing, while favorite remains a winner and becomes more likely to win, since the winning set shrinks
                        # Viewed in terms of adding a ballot, least_favorite goes from losing to winning, while favorite becomes less likely to win, since the winning set expands
                        if check_probabilities and favorite in winners and least_favorite in winners and least_favorite not in winners2 and len(winners) > len(winners2):
                            if verbose:
                                prof = prof.anonymize()
                                if coalition_size == 1:
                                    print(f"{least_favorite} becomes less likely to win and {favorite} remains a winner after removing a voter with the ranking {str(r)}:")
                                else:
                                    print(f"{least_favorite} becomes less likely to win and {favorite} remains a winner after removing {coalition_size} voters with the ranking {str(r)}:")
                                print("")
                                print("Full profile")
                                prof.display()
                                print(prof.description())
                                prof.display_margin_graph()
                                vm.display(prof)
                                print("")
                                if coalition_size == 1:
                                    print("Profile with voter removed")
                                else:
                                    print(f"Profile with {coalition_size} voters removed")
                                prof2 = prof2.anonymize()
                                prof2.display()
                                print(prof2.description())
                                prof2.display_margin_graph()
                                vm.display(prof2)
                                print("")
                            return True
        
        if not uniform_coalition:
            for favorite in non_winners:
                for least_favorite in winners:
                    relevant_ranking_types = [r for r in prof.ranking_types if r[0] == favorite and r[-1] == least_favorite]
                    relevant_ranking_types_counts = [prof.rankings.count(r) for r in relevant_ranking_types]

                    for coalition_rankings, coalition_rankings_counts in _submultisets_of_fixed_cardinality(relevant_ranking_types, relevant_ranking_types_counts, coalition_size):
                        
                        rankings = prof.rankings
                        
                        for r_idx, r in enumerate(coalition_rankings):
                            for i in range(coalition_rankings_counts[r_idx]):
                                rankings.remove(r)
                            
                        if isinstance(prof, Profile):
                            prof2 = Profile(rankings)

                        if isinstance(prof, ProfileWithTies):
                            prof2 = ProfileWithTies(rankings, candidates=prof.candidates)
                            if prof.using_extended_strict_preference:
                                prof2.use_extended_strict_preference()

                        winners2 = vm(prof2)              

                        if require_resoluteness and len(winners2) > 1:
                            continue

                        if require_uniquely_weighted and not prof2.is_uniquely_weighted():
                            continue
                        
                        if favorite in winners2 and least_favorite not in winners2:
                            if verbose:
                                prof = prof.anonymize()
                                print(f"{favorite} loses and {least_favorite} wins in the full profile, but {favorite} is a winner and {least_favorite} is a loser after removing a {coalition_size}-voter coalition with the rankings {[str(r) for r in coalition_rankings]} and counts {coalition_rankings_counts}:")
                                print("")
                                print("Full profile")
                                prof.display()
                                print(prof.description())
                                prof.display_margin_graph()
                                vm.display(prof)
                                print("")
                                print(f"Profile with coalition removed")
                                prof2 = prof2.anonymize()
                                prof2.display()
                                print(prof2.description())
                                prof2.display_margin_graph()
                                vm.display(prof2)
                                print("")
                            return True
                        
                        # Case 1: When removing a ballot, favorite goes from losing to winning, while least_favorite becomes less likely to win, since the winning set expands
                        # Viewed in terms of adding a ballot, favorite goes from winning to losing, while least_favorite becomes more likely to win, since the winning set shrinks
                        if check_probabilities and favorite in winners2 and least_favorite in winners2 and len(winners) < len(winners2):
                            if verbose:
                                prof = prof.anonymize()
                                print(f"{favorite} becomes more likely to win and {least_favorite} becomes less likely to win after removing a {coalition_size}-voter coalition with the rankings {[str(r) for r in coalition_rankings]} and counts {coalition_rankings_counts}:")
                                print("")
                                print("Full profile")
                                prof.display()
                                print(prof.description())
                                prof.display_margin_graph()
                                vm.display(prof)
                                print("")
                                print(f"Profile with coalition removed")
                                prof2 = prof2.anonymize()
                                prof2.display()
                                print(prof2.description())
                                prof2.display_margin_graph()
                                vm.display(prof2)
                                print("")
                            return True
                        
                        # Case 2: When removing a ballot, least_favorite goes from winning to losing, while favorite remains a winner and becomes more likely to win, since the winning set shrinks
                        # Viewed in terms of adding a ballot, least_favorite goes from losing to winning, while favorite becomes less likely to win, since the winning set expands
                        if check_probabilities and favorite in winners and least_favorite in winners and least_favorite not in winners2 and len(winners) > len(winners2):
                            if verbose:
                                prof = prof.anonymize()
                                print(f"{least_favorite} becomes less likely to win and {favorite} remains a winner after removing a {coalition_size}-voter coalition with the rankings {[str(r) for r in coalition_rankings]} and counts {coalition_rankings_counts}:")
                                print("")
                                print("Full profile")
                                prof.display()
                                print(prof.description())
                                prof.display_margin_graph()
                                vm.display(prof)
                                print("")
                                print(f"Profile with coalition removed")
                                prof2 = prof2.anonymize()
                                prof2.display()
                                print(prof2.description())
                                prof2.display_margin_graph()
                                vm.display(prof2)
                                print("")
                            return True
    
    return False
                    
def find_all_positive_negative_involvement_violations(prof, vm, verbose=False, violation_type="Removal", coalition_size=1, uniform_coalition=True, require_resoluteness=False, require_uniquely_weighted=False, check_probabilities=False):
    """
    If violation_type = "Removal", returns a list of tuples (favorite, least_favorite, rankings, counts) such that removing the indicated rankings with the indicated counts causes their favorite to win and their least_favorite to lose, witnessing a violation of positive-negative involvement.
    
    If uniform_coalition = True, then only coalitions of voters with the same ranking are considered.

    If require_resoluteness = True, then only profiles with a unique winner are considered.

    If require_uniquely_weighted = True, then only uniquely-weighted profiles are considered.

    If check_probabilities = True, the function also checks whether removing the voters causes the probability of their favorite winning to increase and the probability of their least favorite winning to decrease.
    
    Args:
        prof: a Profile or ProfileWithTies object.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation.
        violation_type: default is "Removal"
        coalition_size: default is 1
        uniform_coalition: default is True
        require_resoluteness: default is False
        require_uniquely_weighted: default is False
        check_probabilities: default is False
        
    Returns:
        A List of tuples (loser, winner, rankings, counts) witnessing violations of positive-negative involvement.
        
    .. warning::
        This function is slow when uniform_coalition = False and the numbers of voters and candidates are too large.
    """
    
    winners = vm(prof)   
    non_winners = [c for c in prof.candidates if c not in winners]
    
    witnesses = list()

    if require_resoluteness and len(winners) > 1:
        return witnesses
    
    if require_uniquely_weighted and not prof.is_uniquely_weighted():
        return witnesses
    
    if violation_type == "Removal":
        if uniform_coalition:
            # Check for standard positive-negative involvement violations and Case 1 probability violations
            for favorite in non_winners:
                for least_favorite in winners:
                    relevant_ranking_types = [r for r in prof.ranking_types if r[0] == favorite and r[-1] == least_favorite and prof.rankings.count(r) >= coalition_size]

                    for r in relevant_ranking_types:
                        rankings = prof.rankings

                        for i in range(coalition_size):
                            rankings.remove(r) # remove coalition_size-many tokens of the type of ranking

                        if isinstance(prof, Profile):
                            prof2 = Profile(rankings)

                        if isinstance(prof, ProfileWithTies):
                            prof2 = ProfileWithTies(rankings, candidates=prof.candidates)
                            if prof.using_extended_strict_preference:
                                prof2.use_extended_strict_preference()

                        winners2 = vm(prof2)

                        if require_resoluteness and len(winners2) > 1:
                            continue

                        if require_uniquely_weighted and not prof2.is_uniquely_weighted():
                            continue
                        
                        if favorite in winners2 and least_favorite not in winners2:
                            witnesses.append((favorite, least_favorite, [r], [coalition_size]))
                            if verbose:
                                prof = prof.anonymize()
                                if coalition_size == 1:
                                    print(f"{favorite} loses and {least_favorite} wins in the full profile, but {favorite} is a winner and {least_favorite} is a loser after removing a voter with the ranking {str(r)}:")
                                else:
                                    print(f"{favorite} loses and {least_favorite} wins in the full profile, but {favorite} is a winner and {least_favorite} is a loser after removing {coalition_size} voters with the ranking {str(r)}:")
                                print("")
                                print("Full profile")
                                prof.display()
                                print(prof.description())
                                prof.display_margin_graph()
                                vm.display(prof)
                                print("")
                                if coalition_size == 1:
                                    print("Profile with voter removed")
                                else:
                                    print(f"Profile with {coalition_size} voters removed")
                                prof2 = prof2.anonymize()
                                prof2.display()
                                print(prof2.description())
                                prof2.display_margin_graph()
                                vm.display(prof2)
                                print("")
                        
                        # Case 1: When removing a ballot, favorite goes from losing to winning, while least_favorite becomes less likely to win, since the winning set expands
                        # Viewed in terms of adding a ballot, favorite goes from winning to losing, while least_favorite becomes more likely to win, since the winning set shrinks
                        if check_probabilities and favorite in winners2 and least_favorite in winners2 and len(winners) < len(winners2):
                            witnesses.append((favorite, least_favorite, [r], [coalition_size]))
                            if verbose:
                                prof = prof.anonymize()
                                if coalition_size == 1:
                                    print(f"{favorite} becomes more likely to win and {least_favorite} becomes less likely to win after removing a voter with the ranking {str(r)}:")
                                else:
                                    print(f"{favorite} becomes more likely to win and {least_favorite} becomes less likely to win after removing {coalition_size} voters with the ranking {str(r)}:")
                                print("")
                                print("Full profile")
                                prof.display()
                                print(prof.description())
                                prof.display_margin_graph()
                                vm.display(prof)
                                print("")
                                if coalition_size == 1:
                                    print("Profile with voter removed")
                                else:
                                    print(f"Profile with {coalition_size} voters removed")
                                prof2 = prof2.anonymize()
                                prof2.display()
                                print(prof2.description())
                                prof2.display_margin_graph()
                                vm.display(prof2)
                                print("")
            
            # Check for Case 2 probability violations (where favorite is already a winner)
            if check_probabilities:
                for favorite in winners:
                    for least_favorite in winners:
                        if favorite == least_favorite:
                            continue
                        
                        relevant_ranking_types = [r for r in prof.ranking_types if r[0] == favorite and r[-1] == least_favorite and prof.rankings.count(r) >= coalition_size]

                    for r in relevant_ranking_types:
                        rankings = prof.rankings

                        for i in range(coalition_size):
                            rankings.remove(r) # remove coalition_size-many tokens of the type of ranking

                        if isinstance(prof, Profile):
                            prof2 = Profile(rankings)

                        if isinstance(prof, ProfileWithTies):
                            prof2 = ProfileWithTies(rankings, candidates=prof.candidates)
                            if prof.using_extended_strict_preference:
                                prof2.use_extended_strict_preference()

                        winners2 = vm(prof2)

                        if require_resoluteness and len(winners2) > 1:
                            continue

                        if require_uniquely_weighted and not prof2.is_uniquely_weighted():
                            continue
                        
                        # Case 2: When removing a ballot, least_favorite goes from winning to losing, while favorite remains a winner and becomes more likely to win, since the winning set shrinks
                        # Viewed in terms of adding a ballot, least_favorite goes from losing to winning, while favorite becomes less likely to win, since the winning set expands
                        if check_probabilities and favorite in winners and least_favorite in winners and least_favorite not in winners2 and len(winners) > len(winners2):
                            witnesses.append((favorite, least_favorite, [r], [coalition_size]))
                            if verbose:
                                prof = prof.anonymize()
                                if coalition_size == 1:
                                    print(f"{least_favorite} becomes less likely to win and {favorite} remains a winner after removing a voter with the ranking {str(r)}:")
                                else:
                                    print(f"{least_favorite} becomes less likely to win and {favorite} remains a winner after removing {coalition_size} voters with the ranking {str(r)}:")
                                print("")
                                print("Full profile")
                                prof.display()
                                print(prof.description())
                                prof.display_margin_graph()
                                vm.display(prof)
                                print("")
                                if coalition_size == 1:
                                    print("Profile with voter removed")
                                else:
                                    print(f"Profile with {coalition_size} voters removed")
                                prof2 = prof2.anonymize()
                                prof2.display()
                                print(prof2.description())
                                prof2.display_margin_graph()
                                vm.display(prof2)
                                print("")
        
        if not uniform_coalition:
            # Check for standard positive-negative involvement violations and Case 1 probability violations
            for favorite in non_winners:
                for least_favorite in winners:
                    relevant_ranking_types = [r for r in prof.ranking_types if r[0] == favorite and r[-1] == least_favorite]
                    relevant_ranking_types_counts = [prof.rankings.count(r) for r in relevant_ranking_types]

                    for coalition_rankings, coalition_rankings_counts in _submultisets_of_fixed_cardinality(relevant_ranking_types, relevant_ranking_types_counts, coalition_size):
                        
                        rankings = prof.rankings
                        
                        for r_idx, r in enumerate(coalition_rankings):
                            for i in range(coalition_rankings_counts[r_idx]):
                                rankings.remove(r)
                            
                        if isinstance(prof, Profile):
                            prof2 = Profile(rankings)

                        if isinstance(prof, ProfileWithTies):
                            prof2 = ProfileWithTies(rankings, candidates=prof.candidates)
                            if prof.using_extended_strict_preference:
                                prof2.use_extended_strict_preference()

                        winners2 = vm(prof2)              

                        if require_resoluteness and len(winners2) > 1:
                            continue

                        if require_uniquely_weighted and not prof2.is_uniquely_weighted():
                            continue
                        
                        if favorite in winners2 and least_favorite not in winners2:
                            witnesses.append((favorite, least_favorite, coalition_rankings, coalition_rankings_counts))
                            if verbose:
                                prof = prof.anonymize()
                                print(f"{favorite} loses and {least_favorite} wins in the full profile, but {favorite} is a winner and {least_favorite} is a loser after removing a {coalition_size}-voter coalition with the rankings {[str(r) for r in coalition_rankings]} and counts {coalition_rankings_counts}:")
                                print("")
                                print("Full profile")
                                prof.display()
                                print(prof.description())
                                prof.display_margin_graph()
                                vm.display(prof)
                                print("")
                                print(f"Profile with coalition removed")
                                prof2 = prof2.anonymize()
                                prof2.display()
                                print(prof2.description())
                                prof2.display_margin_graph()
                                vm.display(prof2)
                                print("")
                        
                        # Case 1: When removing a ballot, favorite goes from losing to winning, while least_favorite becomes less likely to win, since the winning set expands
                        # Viewed in terms of adding a ballot, favorite goes from winning to losing, while least_favorite becomes more likely to win, since the winning set shrinks
                        if check_probabilities and favorite in winners2 and least_favorite in winners2 and len(winners) < len(winners2):
                            witnesses.append((favorite, least_favorite, coalition_rankings, coalition_rankings_counts))
                            if verbose:
                                prof = prof.anonymize()
                                print(f"{favorite} becomes more likely to win and {least_favorite} becomes less likely to win after removing a {coalition_size}-voter coalition with the rankings {[str(r) for r in coalition_rankings]} and counts {coalition_rankings_counts}:")
                                print("")
                                print("Full profile")
                                prof.display()
                                print(prof.description())
                                prof.display_margin_graph()
                                vm.display(prof)
                                print("")
                                print(f"Profile with coalition removed")
                                prof2 = prof2.anonymize()
                                prof2.display()
                                print(prof2.description())
                                prof2.display_margin_graph()
                                vm.display(prof2)
                                print("")
            
            # Check for Case 2 probability violations (where favorite is already a winner)
            if check_probabilities:
                for favorite in winners:
                    for least_favorite in winners:
                        if favorite == least_favorite:
                            continue
                        
                        relevant_ranking_types = [r for r in prof.ranking_types if r[0] == favorite and r[-1] == least_favorite]
                        relevant_ranking_types_counts = [prof.rankings.count(r) for r in relevant_ranking_types]

                    for coalition_rankings, coalition_rankings_counts in _submultisets_of_fixed_cardinality(relevant_ranking_types, relevant_ranking_types_counts, coalition_size):
                        
                        rankings = prof.rankings
                        
                        for r_idx, r in enumerate(coalition_rankings):
                            for i in range(coalition_rankings_counts[r_idx]):
                                rankings.remove(r)
                            
                        if isinstance(prof, Profile):
                            prof2 = Profile(rankings)

                        if isinstance(prof, ProfileWithTies):
                            prof2 = ProfileWithTies(rankings, candidates=prof.candidates)
                            if prof.using_extended_strict_preference:
                                prof2.use_extended_strict_preference()

                        winners2 = vm(prof2)              

                        if require_resoluteness and len(winners2) > 1:
                            continue

                        if require_uniquely_weighted and not prof2.is_uniquely_weighted():
                            continue
                        
                        # Case 2: When removing a ballot, least_favorite goes from winning to losing, while favorite remains a winner and becomes more likely to win, since the winning set shrinks
                        # Viewed in terms of adding a ballot, least_favorite goes from losing to winning, while favorite becomes less likely to win, since the winning set expands
                        if check_probabilities and favorite in winners and least_favorite in winners and least_favorite not in winners2 and len(winners) > len(winners2):
                            witnesses.append((favorite, least_favorite, coalition_rankings, coalition_rankings_counts))
                            if verbose:
                                prof = prof.anonymize()
                                print(f"{least_favorite} becomes less likely to win and {favorite} remains a winner after removing a {coalition_size}-voter coalition with the rankings {[str(r) for r in coalition_rankings]} and counts {coalition_rankings_counts}:")
                                print("")
                                print("Full profile")
                                prof.display()
                                print(prof.description())
                                prof.display_margin_graph()
                                vm.display(prof)
                                print("")
                                print(f"Profile with coalition removed")
                                prof2 = prof2.anonymize()
                                prof2.display()
                                print(prof2.description())
                                prof2.display_margin_graph()
                                vm.display(prof2)
                                print("")
    
    return witnesses

positive_negative_involvement = Axiom(
    "Positive-Negative Involvement",
    has_violation=has_positive_negative_involvement_violation,
    find_all_violations=find_all_positive_negative_involvement_violations
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
    
    if isinstance(prof, ProfileWithTies):
        prof.use_extended_strict_preference()

    winners = vm(prof)   
    losers = [c for c in prof.candidates if c not in winners]

    if violation_type == "Removal":
        for loser in losers:
            for r in prof.ranking_types: # for each type of ranking

                rankings = prof.rankings
                rankings.remove(r) # remove the first token of the type of ranking
                
                if isinstance(prof, Profile):
                    prof2 = Profile(rankings)
                
                if isinstance(prof, ProfileWithTies):
                    prof2 = ProfileWithTies(rankings, candidates=prof.candidates)
                    if prof.using_extended_strict_preference:
                        prof2.use_extended_strict_preference()
                
                tolerant_ballot = True

                # check whether the loser is ranked above every candiddate c such that the loser is not majority preferred to c
                for c in prof.candidates:
                    if c != loser and not prof2.majority_prefers(loser, c):
                        # Handle different ranking types
                        if isinstance(r, tuple):
                            # Profile case: r is a tuple, use index comparison
                            if r.index(c) < r.index(loser):
                                tolerant_ballot = False
                                break
                        else:
                            # ProfileWithTies case: r is a Ranking object, use strict_pref method
                            if not r.strict_pref(loser, c):
                                tolerant_ballot = False
                                break

                if tolerant_ballot:
                    if loser in vm(prof2):
                        if verbose:
                            prof = prof.anonymize()
                            ranking_str = str(r) if hasattr(r, '__str__') else str(r)
                            print(f"{loser} loses in the full profile, but {loser} is a winner after removing a voter with the ranking {ranking_str}:")
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
                    
    return False

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
    
    if isinstance(prof, ProfileWithTies):
        prof.use_extended_strict_preference()

    winners = vm(prof)   
    losers = [c for c in prof.candidates if c not in winners]

    witnesses = list()

    if violation_type == "Removal":
        for loser in losers:
            for r in prof.ranking_types: # for each type of ranking

                rankings = prof.rankings
                rankings.remove(r) # remove the first token of the type of ranking
                
                if isinstance(prof, Profile):
                    prof2 = Profile(rankings)
                
                if isinstance(prof, ProfileWithTies):
                    prof2 = ProfileWithTies(rankings, candidates=prof.candidates)
                    if prof.using_extended_strict_preference:
                        prof2.use_extended_strict_preference()

                tolerant_ballot = True

                # check whether the loser is ranked above every candiddate c such that the loser is not majority preferred to c
                for c in prof.candidates:
                    if c != loser and not prof2.majority_prefers(loser, c):
                        # Handle different ranking types
                        if isinstance(r, tuple):
                            # Profile case: r is a tuple, use index comparison
                            if r.index(c) < r.index(loser):
                                tolerant_ballot = False
                                break
                        else:
                            # ProfileWithTies case: r is a Ranking object, use strict_pref method
                            if not r.strict_pref(loser, c):
                                tolerant_ballot = False
                                break

                if tolerant_ballot:
                    if loser in vm(prof2):
                        witnesses.append((loser, r))
                        if verbose:
                            prof = prof.anonymize()
                            ranking_str = str(r) if hasattr(r, '__str__') else str(r)
                            print(f"{loser} loses in the full profile, but {loser} is a winner after removing a voter with the ranking {ranking_str}:")
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
        if isinstance(prof, Profile):
            new_prof = ProfileWithTies([{c:c_indx+1 for c_indx, c in enumerate(r)} for r in prof.rankings] + [{w:1}] * coalition_size, candidates = prof.candidates)
            new_prof.use_extended_strict_preference()
            new_mg = new_prof.margin_graph()

        if isinstance(prof, ProfileWithTies):
            new_prof = ProfileWithTies(prof.rankings + [{w:1}] * coalition_size, candidates = prof.candidates)
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
        if isinstance(prof, Profile):
            new_prof = ProfileWithTies([{c:c_indx+1 for c_indx, c in enumerate(r)} for r in prof.rankings] + [{w:1}] * coalition_size, candidates = prof.candidates)
            new_prof.use_extended_strict_preference()
            new_mg = new_prof.margin_graph()
        if isinstance(prof, ProfileWithTies):
            new_prof = ProfileWithTies(prof.rankings + [{w:1}] * coalition_size, candidates = prof.candidates)
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

def has_semi_positive_involvement_violation(prof, vm, verbose=False):
    """
    Semi-Positive Involvement says that if A wins in an initial profile, and we add a voter with a truncated ballot
    ranking A first, then it cannot happen that all of the winners in the new profile are unranked by the truncated ballot.

    Rather than adding a ballot, this function removes a ballot from the profile and then adds a truncated version of the ballot.

    The function returns True if there is a ballot such that starting from an updated version of prof 
    with the ballot removed, adding a truncated version of the ballot causes the winner(s) 
    to shift from the top-ranked candidate (plus possibly other ranked candidates) to some unranked candidate(s).
    
    Args:
        prof: a Profile or ProfileWithTies object.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation.
        
    Returns:
        Result of the test (bool): Returns True if there is a violation and False otherwise.
    """
    
    if isinstance(prof, Profile):
        # For each ranking in the profile
        for r_idx, r in enumerate(prof.rankings):
            # Create a profile with this ballot removed
            removed_ballot_rankings = prof.rankings.copy()
            removed_ballot_rankings.pop(r_idx)
            removed_ballot_prof = Profile(removed_ballot_rankings)
            
            # Get winners when ballot is removed
            removed_ballot_winners = vm(removed_ballot_prof)
            
            # For each possible truncation of the ballot
            for truncation_level in range(1, len(prof.candidates)):
                # Create a truncated ballot with only the top truncation_level candidates
                ranked_candidates = r[:truncation_level]
                unranked_candidates = [c for c in prof.candidates if c not in ranked_candidates]
                
                # Get the top-ranked candidate
                top_ranked_candidate = r[0] if len(r) > 0 else None
                
                # Create a ranking object for the truncated ballot
                truncated_ballot = {c: i+1 for i, c in enumerate(ranked_candidates)}
                
                # Create a profile with the truncated ballot
                # Convert tuples to dictionaries for ProfileWithTies
                converted_rankings = []
                for ranking in removed_ballot_rankings:
                    converted_rankings.append({c: i+1 for i, c in enumerate(ranking)})
                
                # Add the truncated ballot to the converted rankings
                converted_rankings.append(truncated_ballot)
                truncated_ballot_prof = ProfileWithTies(converted_rankings, candidates=prof.candidates)
                truncated_ballot_prof.use_extended_strict_preference()
                
                # Get winners when truncated ballot is added
                truncated_ballot_winners = vm(truncated_ballot_prof)
                
                # Check if winners shifted from top-ranked+other-ranked to unranked candidates
                # The winners in the profile without the truncated ballot must include the top-ranked candidate
                winners_include_top_ranked = top_ranked_candidate in removed_ballot_winners
                # All winners in the profile with the truncated ballot must be among the unranked candidates
                winners_from_unranked = all(c in unranked_candidates for c in truncated_ballot_winners)
                
                if winners_include_top_ranked and winners_from_unranked:
                    if verbose:
                        print(f"Semi-Positive Involvement violation found:")
                        print("Original profile:")
                        prof.display()
                        print(prof.description())
                        vm.display(prof)
                        print("\nProfile with ballot removed:")
                        removed_ballot_prof.display()
                        print(removed_ballot_prof.description())
                        vm.display(removed_ballot_prof)
                        print("\nMargin graph of profile with ballot removed:")
                        removed_ballot_prof.display_margin_graph()
                        print(f"Winners: {removed_ballot_winners}")
                        print(f"\nTruncated ballot: {truncated_ballot}")
                        print(f"Top-ranked candidate: {top_ranked_candidate}")
                        print(f"Ranked candidates: {ranked_candidates}")
                        print(f"Unranked candidates: {unranked_candidates}")
                        print("\nProfile with truncated ballot:")
                        truncated_ballot_prof.display()
                        print(truncated_ballot_prof.description())
                        vm.display(truncated_ballot_prof)
                        print("\nMargin graph of profile with truncated ballot:")
                        truncated_ballot_prof.display_margin_graph()
                        print(f"Winners: {truncated_ballot_winners}")
                        print(f"\nWinners shifted from including top-ranked candidate {top_ranked_candidate} to unranked candidates {truncated_ballot_winners}")
                    return True
    
    elif isinstance(prof, ProfileWithTies):
        # For each ranking in the profile
        rankings, rcounts = prof.rankings_counts
        
        for r_idx, (r, count) in enumerate(zip(rankings, rcounts)):
            # Skip if this is not a single ballot
            if count != 1:
                continue
                
            # Create a profile with this ballot removed
            removed_ballot_rankings = rankings.copy()
            removed_ballot_rcounts = rcounts.copy()
            removed_ballot_rankings.pop(r_idx)
            removed_ballot_rcounts.pop(r_idx)
            removed_ballot_prof = ProfileWithTies(removed_ballot_rankings, rcounts=removed_ballot_rcounts, candidates=prof.candidates)
            if prof.using_extended_strict_preference:
                removed_ballot_prof.use_extended_strict_preference()
            
            # Get winners when ballot is removed
            removed_ballot_winners = vm(removed_ballot_prof)
            
            # For each possible truncation of the ballot
            ranked_candidates = r.cands
            for truncation_level in range(1, len(ranked_candidates)):
                # Get the candidates at each rank
                candidates_by_rank = [r.cands_at_rank(rank) for rank in r.ranks]
                
                # Take only the first truncation_level ranks
                truncated_ranked_candidates = [c for rank_candidates in candidates_by_rank[:truncation_level] for c in rank_candidates]
                unranked_candidates = [c for c in prof.candidates if c not in truncated_ranked_candidates]
                
                # Get the top-ranked candidates (those at rank 1)
                top_ranked_candidates = candidates_by_rank[0] if len(candidates_by_rank) > 0 else []
                
                # Create a ranking object for the truncated ballot
                truncated_ballot = {c: r.rmap[c] for c in truncated_ranked_candidates}
                
                # Create a profile with the truncated ballot
                truncated_ballot_rankings = removed_ballot_rankings + [Ranking(truncated_ballot)]
                truncated_ballot_rcounts = removed_ballot_rcounts + [1]
                truncated_ballot_prof = ProfileWithTies(truncated_ballot_rankings, rcounts=truncated_ballot_rcounts, candidates=prof.candidates)
                if prof.using_extended_strict_preference:
                    truncated_ballot_prof.use_extended_strict_preference()
                
                # Get winners when truncated ballot is added
                truncated_ballot_winners = vm(truncated_ballot_prof)
                
                # Check if winners shifted from top-ranked+other-ranked to unranked candidates
                # The winners in the profile without the truncated ballot must include at least one top-ranked candidate
                winners_include_top_ranked = any(c in top_ranked_candidates for c in removed_ballot_winners)
                # All winners in the profile with the truncated ballot must be among the unranked candidates
                winners_from_unranked = all(c in unranked_candidates for c in truncated_ballot_winners)
                
                if winners_include_top_ranked and winners_from_unranked:
                    if verbose:
                        print(f"Semi-Positive Involvement violation found:")
                        print("Original profile:")
                        prof.display()
                        print(prof.description())
                        vm.display(prof)
                        print("\nProfile with ballot removed:")
                        removed_ballot_prof.display()
                        print(removed_ballot_prof.description())
                        vm.display(removed_ballot_prof)
                        print("\nMargin graph of profile with ballot removed:")
                        removed_ballot_prof.display_margin_graph()
                        print(f"Winners: {removed_ballot_winners}")
                        print(f"\nTruncated ballot: {truncated_ballot}")
                        print(f"Top-ranked candidates: {top_ranked_candidates}")
                        print(f"Ranked candidates: {truncated_ranked_candidates}")
                        print(f"Unranked candidates: {unranked_candidates}")
                        print("\nProfile with truncated ballot:")
                        truncated_ballot_prof.display()
                        print(truncated_ballot_prof.description())
                        vm.display(truncated_ballot_prof)
                        print("\nMargin graph of profile with truncated ballot:")
                        truncated_ballot_prof.display_margin_graph()
                        print(f"Winners: {truncated_ballot_winners}")
                        print(f"\nWinners shifted from including top-ranked candidate(s) {[prof.cmap[c] for c in removed_ballot_winners if c in top_ranked_candidates]} to unranked candidates {[prof.cmap[c] for c in truncated_ballot_winners]}")
                    return True
    
    return False

def find_all_semi_positive_involvement_violations(prof, vm, verbose=False):
    """
    Returns all violations of semi-positive involvement for a given profile and voting method.
    
    Args:
        prof: a Profile or ProfileWithTies object.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation.
        
    Returns:
        A list of tuples (ballot, truncated_ballot, removed_ballot_winners, truncated_ballot_winners) where each tuple represents a violation.
    """
    
    violations = []
    
    if isinstance(prof, Profile):
        # For each ranking in the profile
        for r_idx, r in enumerate(prof.rankings):
            # Create a profile with this ballot removed
            removed_ballot_rankings = prof.rankings.copy()
            removed_ballot_rankings.pop(r_idx)
            removed_ballot_prof = Profile(removed_ballot_rankings)
            
            # Get winners when ballot is removed
            removed_ballot_winners = vm(removed_ballot_prof)
            
            # For each possible truncation of the ballot
            for truncation_level in range(1, len(prof.candidates)):
                # Create a truncated ballot with only the top truncation_level candidates
                ranked_candidates = r[:truncation_level]
                unranked_candidates = [c for c in prof.candidates if c not in ranked_candidates]
                
                # Get the top-ranked candidate
                top_ranked_candidate = r[0] if len(r) > 0 else None
                
                # Create a ranking object for the truncated ballot
                truncated_ballot = {c: i+1 for i, c in enumerate(ranked_candidates)}
                
                # Create a profile with the truncated ballot
                # Convert tuples to dictionaries for ProfileWithTies
                converted_rankings = []
                for ranking in removed_ballot_rankings:
                    converted_rankings.append({c: i+1 for i, c in enumerate(ranking)})
                
                # Add the truncated ballot to the converted rankings
                converted_rankings.append(truncated_ballot)
                truncated_ballot_prof = ProfileWithTies(converted_rankings, candidates=prof.candidates)
                truncated_ballot_prof.use_extended_strict_preference()
                
                # Get winners when truncated ballot is added
                truncated_ballot_winners = vm(truncated_ballot_prof)
                
                # Check if winners shifted from top-ranked+other-ranked to unranked candidates
                # The winners in the profile without the truncated ballot must include the top-ranked candidate
                winners_include_top_ranked = top_ranked_candidate in removed_ballot_winners
                # All winners in the profile with the truncated ballot must be among the unranked candidates
                winners_from_unranked = all(c in unranked_candidates for c in truncated_ballot_winners)
                
                if winners_include_top_ranked and winners_from_unranked:
                    violations.append((r, truncated_ballot, removed_ballot_winners, truncated_ballot_winners))
                    if verbose:
                        print(f"Semi-Positive Involvement violation found:")
                        print("")
                        print("Original profile:")
                        prof.display()
                        print(prof.description())
                        vm.display(prof)
                        print("\nProfile with ballot removed:")
                        removed_ballot_prof.display()
                        print(removed_ballot_prof.description())
                        vm.display(removed_ballot_prof)
                        print("\nMargin graph of profile with ballot removed:")
                        removed_ballot_prof.display_margin_graph()
                        print(f"Winners: {removed_ballot_winners}")
                        print(f"\nTruncated ballot: {truncated_ballot}")
                        print(f"Top-ranked candidate: {top_ranked_candidate}")
                        print(f"Ranked candidates: {ranked_candidates}")
                        print(f"Unranked candidates: {unranked_candidates}")
                        print("\nProfile with truncated ballot:")
                        truncated_ballot_prof.display()
                        print(truncated_ballot_prof.description())
                        vm.display(truncated_ballot_prof)
                        print("\nMargin graph of profile with truncated ballot:")
                        truncated_ballot_prof.display_margin_graph()
                        print(f"Winners: {truncated_ballot_winners}")
                        print(f"\nWinners shifted from including top-ranked candidate {top_ranked_candidate} to unranked candidates {truncated_ballot_winners}")
    
    elif isinstance(prof, ProfileWithTies):
        # For each ranking in the profile
        rankings, rcounts = prof.rankings_counts
        
        for r_idx, (r, count) in enumerate(zip(rankings, rcounts)):
            # Skip if this is not a single ballot
            if count != 1:
                continue
                
            # Create a profile with this ballot removed
            removed_ballot_rankings = rankings.copy()
            removed_ballot_rcounts = rcounts.copy()
            removed_ballot_rankings.pop(r_idx)
            removed_ballot_rcounts.pop(r_idx)
            removed_ballot_prof = ProfileWithTies(removed_ballot_rankings, rcounts=removed_ballot_rcounts, candidates=prof.candidates)
            if prof.using_extended_strict_preference:
                removed_ballot_prof.use_extended_strict_preference()
            
            # Get winners when ballot is removed
            removed_ballot_winners = vm(removed_ballot_prof)
            
            # For each possible truncation of the ballot
            ranked_candidates = r.cands
            for truncation_level in range(1, len(ranked_candidates)):
                # Get the candidates at each rank
                candidates_by_rank = [r.cands_at_rank(rank) for rank in r.ranks]
                
                # Take only the first truncation_level ranks
                truncated_ranked_candidates = [c for rank_candidates in candidates_by_rank[:truncation_level] for c in rank_candidates]
                unranked_candidates = [c for c in prof.candidates if c not in truncated_ranked_candidates]
                
                # Get the top-ranked candidates (those at rank 1)
                top_ranked_candidates = candidates_by_rank[0] if len(candidates_by_rank) > 0 else []
                
                # Create a ranking object for the truncated ballot
                truncated_ballot = {c: r.rmap[c] for c in truncated_ranked_candidates}
                
                # Create a profile with the truncated ballot
                truncated_ballot_rankings = removed_ballot_rankings + [Ranking(truncated_ballot)]
                truncated_ballot_rcounts = removed_ballot_rcounts + [1]
                truncated_ballot_prof = ProfileWithTies(truncated_ballot_rankings, rcounts=truncated_ballot_rcounts, candidates=prof.candidates)
                if prof.using_extended_strict_preference:
                    truncated_ballot_prof.use_extended_strict_preference()
                
                # Get winners when truncated ballot is added
                truncated_ballot_winners = vm(truncated_ballot_prof)
                
                # Check if winners shifted from top-ranked+other-ranked to unranked candidates
                # The winners in the profile without the truncated ballot must include at least one top-ranked candidate
                winners_include_top_ranked = any(c in top_ranked_candidates for c in removed_ballot_winners)
                # All winners in the profile with the truncated ballot must be among the unranked candidates
                winners_from_unranked = all(c in unranked_candidates for c in truncated_ballot_winners)
                
                if winners_include_top_ranked and winners_from_unranked:
                    violations.append((r, Ranking(truncated_ballot), removed_ballot_winners, truncated_ballot_winners))
                    if verbose:
                        print(f"Semi-Positive Involvement violation found:")
                        print("")
                        print("Original profile:")
                        prof.display()
                        print(prof.description())
                        vm.display(prof)
                        print("\nProfile with ballot removed:")
                        removed_ballot_prof.display()
                        print(removed_ballot_prof.description())
                        vm.display(removed_ballot_prof)
                        print("\nMargin graph of profile with ballot removed:")
                        removed_ballot_prof.display_margin_graph()
                        print(f"Winners: {removed_ballot_winners}")
                        print(f"\nTruncated ballot: {truncated_ballot}")
                        print(f"Top-ranked candidates: {top_ranked_candidates}")
                        print(f"Ranked candidates: {truncated_ranked_candidates}")
                        print(f"Unranked candidates: {unranked_candidates}")
                        print("\nProfile with truncated ballot:")
                        truncated_ballot_prof.display()
                        print(truncated_ballot_prof.description())
                        vm.display(truncated_ballot_prof)
                        print("\nMargin graph of profile with truncated ballot:")
                        truncated_ballot_prof.display_margin_graph()
                        print(f"Winners: {truncated_ballot_winners}")
                        print(f"\nWinners shifted from including top-ranked candidate(s) {[prof.cmap[c] for c in removed_ballot_winners if c in top_ranked_candidates]} to unranked candidates {[prof.cmap[c] for c in truncated_ballot_winners]}")
    
    return violations

semi_positive_involvement = Axiom(
    "Semi-Positive Involvement",
    has_violation = has_semi_positive_involvement_violation,
    find_all_violations = find_all_semi_positive_involvement_violations,
)

def has_truncated_involvement_violation(prof, vm, verbose=False):
    """
    Returns True if there is a ballot such that starting from an updated version of prof with the ballot removed, adding a truncated version of the ballot causes the winners to shift from ranked candidates to unranked candidates.
    
    Args:
        prof: a Profile or ProfileWithTies object.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation.
        
    Returns:
        Result of the test (bool): Returns True if there is a violation and False otherwise.
    """
    
    if isinstance(prof, Profile):
        # For each ranking in the profile
        for r_idx, r in enumerate(prof.rankings):
            # Create a profile with this ballot removed
            removed_ballot_rankings = prof.rankings.copy()
            removed_ballot_rankings.pop(r_idx)
            removed_ballot_prof = Profile(removed_ballot_rankings)
            
            # Get winners when ballot is removed
            removed_ballot_winners = vm(removed_ballot_prof)
            
            # For each possible truncation of the ballot
            for truncation_level in range(1, len(prof.candidates)):
                # Create a truncated ballot with only the top truncation_level candidates
                ranked_candidates = r[:truncation_level]
                unranked_candidates = [c for c in prof.candidates if c not in ranked_candidates]
                
                # Create a ranking object for the truncated ballot
                truncated_ballot = {c: i+1 for i, c in enumerate(ranked_candidates)}
                
                # Create a profile with the truncated ballot
                # Convert tuples to dictionaries for ProfileWithTies
                converted_rankings = []
                for ranking in removed_ballot_rankings:
                    converted_rankings.append({c: i+1 for i, c in enumerate(ranking)})
                
                # Add the truncated ballot to the converted rankings
                converted_rankings.append(truncated_ballot)
                truncated_ballot_prof = ProfileWithTies(converted_rankings, candidates=prof.candidates)
                truncated_ballot_prof.use_extended_strict_preference()
                
                # Get winners when truncated ballot is added
                truncated_ballot_winners = vm(truncated_ballot_prof)
                
                # Check if winners shifted from ranked to unranked candidates
                # All winners in the profile without the truncated ballot must be among the ranked candidates
                winners_from_ranked = all(c in ranked_candidates for c in removed_ballot_winners)
                # All winners in the profile with the truncated ballot must be among the unranked candidates
                winners_from_unranked = all(c in unranked_candidates for c in truncated_ballot_winners)
                
                if winners_from_ranked and winners_from_unranked:
                    if verbose:
                        print(f"Truncated Involvement violation found:")
                        print("")
                        print("Original profile:")
                        prof.display()
                        print(prof.description())
                        vm.display(prof)
                        print("\nProfile with ballot removed:")
                        removed_ballot_prof.display()
                        print(removed_ballot_prof.description())
                        vm.display(removed_ballot_prof)
                        print("\nMargin graph of profile with ballot removed:")
                        removed_ballot_prof.display_margin_graph()
                        print("\nProfile with truncated ballot:")
                        truncated_ballot_prof.display()
                        print(truncated_ballot_prof.description())
                        vm.display(truncated_ballot_prof)
                        print("\nMargin graph of profile with truncated ballot:")
                        truncated_ballot_prof.display_margin_graph()
                        print(f"\nTruncated ballot: {truncated_ballot}")
                        print(f"Ranked candidates: {ranked_candidates}")
                        print(f"Unranked candidates: {unranked_candidates}")
                        print(f"Winners: {truncated_ballot_winners}")
                        print(f"\nWinners shifted from ranked candidates {[prof.cmap[c] for c in removed_ballot_winners if c in ranked_candidates]} to unranked candidates {[prof.cmap[c] for c in truncated_ballot_winners]}")
                    return True
    
    elif isinstance(prof, ProfileWithTies):
        # For each ranking in the profile
        rankings, rcounts = prof.rankings_counts
        
        for r_idx, (r, count) in enumerate(zip(rankings, rcounts)):
            # Skip if this is not a single ballot
            if count != 1:
                continue
                
            # Create a profile with this ballot removed
            removed_ballot_rankings = rankings.copy()
            removed_ballot_rcounts = rcounts.copy()
            removed_ballot_rankings.pop(r_idx)
            removed_ballot_rcounts.pop(r_idx)
            removed_ballot_prof = ProfileWithTies(removed_ballot_rankings, rcounts=removed_ballot_rcounts, candidates=prof.candidates)
            if prof.using_extended_strict_preference:
                removed_ballot_prof.use_extended_strict_preference()
            
            # Get winners when ballot is removed
            removed_ballot_winners = vm(removed_ballot_prof)
            
            # For each possible truncation of the ballot
            ranked_candidates = r.cands
            for truncation_level in range(1, len(ranked_candidates)):
                # Get the candidates at each rank
                candidates_by_rank = [r.cands_at_rank(rank) for rank in r.ranks]
                
                # Take only the first truncation_level ranks
                truncated_ranked_candidates = [c for rank_candidates in candidates_by_rank[:truncation_level] for c in rank_candidates]
                unranked_candidates = [c for c in prof.candidates if c not in truncated_ranked_candidates]
                
                # Create a ranking object for the truncated ballot
                truncated_ballot = {c: r.rmap[c] for c in truncated_ranked_candidates}
                
                # Create a profile with the truncated ballot
                truncated_ballot_rankings = removed_ballot_rankings + [Ranking(truncated_ballot)]
                truncated_ballot_rcounts = removed_ballot_rcounts + [1]
                truncated_ballot_prof = ProfileWithTies(truncated_ballot_rankings, rcounts=truncated_ballot_rcounts, candidates=prof.candidates)
                if prof.using_extended_strict_preference:
                    truncated_ballot_prof.use_extended_strict_preference()
                
                # Get winners when truncated ballot is added
                truncated_ballot_winners = vm(truncated_ballot_prof)
                
                # Check if winners shifted from ranked to unranked candidates
                winners_from_ranked = any(c in truncated_ranked_candidates for c in removed_ballot_winners)
                winners_from_unranked = all(c in unranked_candidates for c in truncated_ballot_winners)
                
                if winners_from_ranked and winners_from_unranked:
                    if verbose:
                        print(f"Truncated Involvement violation found:")
                        print("")
                        print("Original profile:")
                        prof.display()
                        print(prof.description())
                        vm.display(prof)
                        print("\nProfile with ballot removed:")
                        removed_ballot_prof.display()
                        print(removed_ballot_prof.description())
                        vm.display(removed_ballot_prof)
                        print("\nMargin graph of profile with ballot removed:")
                        removed_ballot_prof.display_margin_graph()
                        print("\nProfile with truncated ballot:")
                        truncated_ballot_prof.display()
                        print(truncated_ballot_prof.description())
                        vm.display(truncated_ballot_prof)
                        print("\nMargin graph of profile with truncated ballot:")
                        truncated_ballot_prof.display_margin_graph()
                        print(f"\nTruncated ballot: {truncated_ballot}")
                        print(f"Ranked candidates: {truncated_ranked_candidates}")
                        print(f"Unranked candidates: {unranked_candidates}")
                        print(f"Winners: {truncated_ballot_winners}")
                        print(f"\nWinners shifted from ranked candidates {[prof.cmap[c] for c in removed_ballot_winners if c in truncated_ranked_candidates]} to unranked candidates {[prof.cmap[c] for c in truncated_ballot_winners]}")
                    return True
    
    return False

def find_all_truncated_involvement_violations(prof, vm, verbose=False):
    """
    Returns all violations of truncated involvement for a given profile and voting method.
    
    Args:
        prof: a Profile or ProfileWithTies object.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation.
        
    Returns:
        A list of tuples (ballot, truncated_ballot, removed_ballot_winners, truncated_ballot_winners) where each tuple represents a violation.
    """
    
    violations = []
    
    if isinstance(prof, Profile):
        # For each ranking in the profile
        for r_idx, r in enumerate(prof.rankings):
            # Create a profile with this ballot removed
            removed_ballot_rankings = prof.rankings.copy()
            removed_ballot_rankings.pop(r_idx)
            removed_ballot_prof = Profile(removed_ballot_rankings)
            
            # Get winners when ballot is removed
            removed_ballot_winners = vm(removed_ballot_prof)
            
            # For each possible truncation of the ballot
            for truncation_level in range(1, len(prof.candidates)):
                # Create a truncated ballot with only the top truncation_level candidates
                ranked_candidates = r[:truncation_level]
                unranked_candidates = [c for c in prof.candidates if c not in ranked_candidates]
                
                # Create a ranking object for the truncated ballot
                truncated_ballot = {c: i+1 for i, c in enumerate(ranked_candidates)}
                
                # Create a profile with the truncated ballot
                # Convert tuples to dictionaries for ProfileWithTies
                converted_rankings = []
                for ranking in removed_ballot_rankings:
                    converted_rankings.append({c: i+1 for i, c in enumerate(ranking)})
                
                # Add the truncated ballot to the converted rankings
                converted_rankings.append(truncated_ballot)
                truncated_ballot_prof = ProfileWithTies(converted_rankings, candidates=prof.candidates)
                truncated_ballot_prof.use_extended_strict_preference()
                
                # Get winners when truncated ballot is added
                truncated_ballot_winners = vm(truncated_ballot_prof)
                
                # Check if winners shifted from ranked to unranked candidates
                # All winners in the profile without the truncated ballot must be among the ranked candidates
                winners_from_ranked = all(c in ranked_candidates for c in removed_ballot_winners)
                # All winners in the profile with the truncated ballot must be among the unranked candidates
                winners_from_unranked = all(c in unranked_candidates for c in truncated_ballot_winners)
                
                if winners_from_ranked and winners_from_unranked:
                    violations.append((r, truncated_ballot, removed_ballot_winners, truncated_ballot_winners))
                    if verbose:
                        print(f"Truncated Involvement violation found:")
                        print("")
                        print("Original profile:")
                        prof.display()
                        print(prof.description())
                        vm.display(prof)
                        print("\nProfile with ballot removed:")
                        removed_ballot_prof.display()
                        print(removed_ballot_prof.description())
                        vm.display(removed_ballot_prof)
                        print("\nMargin graph of profile with ballot removed:")
                        removed_ballot_prof.display_margin_graph()
                        print("\nProfile with truncated ballot:")
                        truncated_ballot_prof.display()
                        print(truncated_ballot_prof.description())
                        vm.display(truncated_ballot_prof)
                        print("\nMargin graph of profile with truncated ballot:")
                        truncated_ballot_prof.display_margin_graph()
                        print(f"\nWinners shifted from ranked candidates {[prof.cmap[c] for c in removed_ballot_winners if c in ranked_candidates]} to unranked candidates {[prof.cmap[c] for c in truncated_ballot_winners]}")
    
    elif isinstance(prof, ProfileWithTies):
        # For each ranking in the profile
        rankings, rcounts = prof.rankings_counts
        
        for r_idx, (r, count) in enumerate(zip(rankings, rcounts)):
            # Skip if this is not a single ballot
            if count != 1:
                continue
                
            # Create a profile with this ballot removed
            removed_ballot_rankings = rankings.copy()
            removed_ballot_rcounts = rcounts.copy()
            removed_ballot_rankings.pop(r_idx)
            removed_ballot_rcounts.pop(r_idx)
            removed_ballot_prof = ProfileWithTies(removed_ballot_rankings, rcounts=removed_ballot_rcounts, candidates=prof.candidates)
            if prof.using_extended_strict_preference:
                removed_ballot_prof.use_extended_strict_preference()
            
            # Get winners when ballot is removed
            removed_ballot_winners = vm(removed_ballot_prof)
            
            # For each possible truncation of the ballot
            ranked_candidates = r.cands
            for truncation_level in range(1, len(ranked_candidates)):
                # Get the candidates at each rank
                candidates_by_rank = [r.cands_at_rank(rank) for rank in r.ranks]
                
                # Take only the first truncation_level ranks
                truncated_ranked_candidates = [c for rank_candidates in candidates_by_rank[:truncation_level] for c in rank_candidates]
                unranked_candidates = [c for c in prof.candidates if c not in truncated_ranked_candidates]
                
                # Create a ranking object for the truncated ballot
                truncated_ballot = {c: r.rmap[c] for c in truncated_ranked_candidates}
                
                # Create a profile with the truncated ballot
                truncated_ballot_rankings = removed_ballot_rankings + [Ranking(truncated_ballot)]
                truncated_ballot_rcounts = removed_ballot_rcounts + [1]
                truncated_ballot_prof = ProfileWithTies(truncated_ballot_rankings, rcounts=truncated_ballot_rcounts, candidates=prof.candidates)
                if prof.using_extended_strict_preference:
                    truncated_ballot_prof.use_extended_strict_preference()
                
                # Get winners when truncated ballot is added
                truncated_ballot_winners = vm(truncated_ballot_prof)
                
                # Check if winners shifted from ranked to unranked candidates
                winners_from_ranked = any(c in truncated_ranked_candidates for c in removed_ballot_winners)
                winners_from_unranked = all(c in unranked_candidates for c in truncated_ballot_winners)
                
                if winners_from_ranked and winners_from_unranked:
                    violations.append((r, Ranking(truncated_ballot), removed_ballot_winners, truncated_ballot_winners))
                    if verbose:
                        print(f"Truncated Involvement violation found:")
                        print("")
                        print("Original profile:")
                        prof.display()
                        print(prof.description())
                        vm.display(prof)
                        print("\nProfile with ballot removed:")
                        removed_ballot_prof.display()
                        print(removed_ballot_prof.description())
                        vm.display(removed_ballot_prof)
                        print("\nMargin graph of profile with ballot removed:")
                        removed_ballot_prof.display_margin_graph()
                        print("\nProfile with truncated ballot:")
                        truncated_ballot_prof.display()
                        print(truncated_ballot_prof.description())
                        vm.display(truncated_ballot_prof)
                        print("\nMargin graph of profile with truncated ballot:")
                        truncated_ballot_prof.display_margin_graph()
                        print(f"\nWinners shifted from ranked candidates {[prof.cmap[c] for c in removed_ballot_winners if c in truncated_ranked_candidates]} to unranked candidates {[prof.cmap[c] for c in truncated_ballot_winners]}")
    
    return violations

truncated_involvement = Axiom(
    "Truncated Involvement",
    has_violation = has_truncated_involvement_violation,
    find_all_violations = find_all_truncated_involvement_violations,
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

def has_neutral_reversal_violation(prof, vm, verbose=False):
    """Returns True if adding a reversal pair of voters (a voter with ranking L and a voter with the reverse ranking L^{-1}) changes the winners.
    
    Args:
        prof: a Profile or ProfileWithTies object
        vm: a voting method
        verbose: if True, display the violation when found
        
    Returns:
        True if there is a violation, False otherwise
    """
    
    winners = vm(prof)
    
    # Get all possible linear orders of the candidates
    all_rankings = list(permutations(prof.candidates))
    
    # For each linear order L, add L and its reverse L^-1 to create new profile
    for ranking in all_rankings:
        # Skip if we've already considered this pair (to avoid redundancy)
        reverse_ranking = tuple(reversed(ranking))
        if reverse_ranking < ranking:
            continue
            
        if isinstance(prof, Profile):
            # For Profile objects, we need to work with numpy arrays
            pair_arr = np.array([list(ranking), list(reverse_ranking)])
            combined_rankings = np.concatenate([prof._rankings, pair_arr], axis=0)
            combined_rcounts = np.concatenate([prof._rcounts, [1, 1]], axis=0)
            prof2 = Profile(combined_rankings, rcounts=combined_rcounts)
        else:
            # For ProfileWithTies, we work with Ranking objects
            ranking_obj = Ranking({c: i for i, c in enumerate(ranking)})
            reverse_ranking_obj = Ranking({c: len(ranking)-1-i for i, c in enumerate(ranking)})
            new_rankings = prof.rankings + [ranking_obj, reverse_ranking_obj]
            prof2 = ProfileWithTies(new_rankings, candidates=prof.candidates)
            if prof.using_extended_strict_preference:
                prof2.use_extended_strict_preference()
                
        # Check if winners changed
        winners2 = vm(prof2)
        if set(winners) != set(winners2):
            if verbose:
                print(f"Adding voters with rankings {ranking} and {reverse_ranking} changes the winners:")
                print("\nOriginal profile:")
                prof.display()
                vm.display(prof)
                print("\nProfile after adding reversal pair:")
                prof2.display()
                vm.display(prof2)
            return True
            
    return False

def find_all_neutral_reversal_violations(prof, vm, verbose=False):
    """Returns a list of reversal pairs (L, L^-1) that when added to the profile change the winners.
    
    Args:
        prof: a Profile or ProfileWithTies object
        vm: a voting method
        verbose: if True, display the violations when found
        
    Returns:
        A list of pairs (L, L^-1) where L is a linear order and L^-1 is its reverse
    """
    
    winners = vm(prof)
    violations = []
    
    # Get all possible linear orders of the candidates
    all_rankings = list(permutations(prof.candidates))
    
    # For each linear order L, add L and its reverse L^-1 to create new profile
    for ranking in all_rankings:
        # Create the reverse ranking
        reverse_ranking = tuple(reversed(ranking))
        
        # Skip if we've already considered this pair (to avoid redundancy)
        if reverse_ranking < ranking:
            continue
            
        # Add the ranking and its reverse to create new profile
        if isinstance(prof, Profile):
            # For Profile objects, we need to work with numpy arrays
            pair_arr = np.array([list(ranking), list(reverse_ranking)])
            combined_rankings = np.concatenate([prof._rankings, pair_arr], axis=0)
            combined_rcounts = np.concatenate([prof._rcounts, [1, 1]], axis=0)
            prof2 = Profile(combined_rankings, rcounts=combined_rcounts)
        else:
            # For ProfileWithTies, we work with Ranking objects
            ranking_obj = Ranking({c: i for i, c in enumerate(ranking)})
            reverse_ranking_obj = Ranking({c: len(ranking)-1-i for i, c in enumerate(ranking)})
            new_rankings = prof.rankings + [ranking_obj, reverse_ranking_obj]
            prof2 = ProfileWithTies(new_rankings, candidates=prof.candidates)
            if prof.using_extended_strict_preference:
                prof2.use_extended_strict_preference()
                
        # Check if winners changed
        winners2 = vm(prof2)
        if set(winners) != set(winners2):
            violations.append((ranking, reverse_ranking))
            if verbose:
                print(f"Adding voters with rankings {ranking} and {reverse_ranking} changes the winners:")
                print("\nOriginal profile:")
                prof.display()
                vm.display(prof)
                print("\nProfile after adding reversal pair:")
                prof2.display()
                vm.display(prof2)
                
    return violations

neutral_reversal = Axiom(
    "Neutral Reversal",
    has_violation = has_neutral_reversal_violation,
    find_all_violations = find_all_neutral_reversal_violations,
)


def has_neutral_indifference_violation(prof, vm, verbose=False): 
    """
    Return True if the profile prof has a neutral indifference violation for the voting method vm.  Otherwise, return False.  That is, return True if there is a tie ranking that can be added to the profile that changes winning set according to vm.  Otherwise, return False.
    """

    tie_ranking = Ranking({c:0 for c in prof.candidates})
    if isinstance(prof, ProfileWithTies):
        new_rankings = prof.rankings + [tie_ranking]
        new_prof = ProfileWithTies(new_rankings)
    elif isinstance(prof, Profile):
        new_rankings = [Ranking.from_linear_order(r) for r in prof.rankings] + [tie_ranking]
        new_prof = ProfileWithTies(new_rankings)
    if vm(prof) != vm(new_prof):
        if verbose:
            print("The original profile")
            prof.anonymize().display()
            print(prof.description())
            vm.display(prof)
            print("")
            print(f"The profile after adding a tie ranking:")
            new_prof.anonymize().display()
            print(new_prof.description())
            vm.display(new_prof)
        return True
    return False

def find_all_neutral_indifference_violations(prof, vm, verbose=False): 
    """
    Return a list containing the profile with an additional voter that ranks all candidates as a tie if this profile has a different winning set according to vm than the original profile.  Otherwise, return the empty list.
    """

    tie_ranking = Ranking({c:0 for c in prof.candidates})
    if isinstance(prof, ProfileWithTies):
        new_rankings = prof.rankings + [tie_ranking]
        new_prof = ProfileWithTies(new_rankings)
    elif isinstance(prof, Profile):
        new_rankings = [Ranking.from_linear_order(r) for r in prof.rankings] + [tie_ranking]
        new_prof = ProfileWithTies(new_rankings)
    if vm(prof) != vm(new_prof):
        if verbose:
            print("The original profile")
            prof.anonymize().display()
            print(prof.description())
            vm.display(prof)
            print("")
            print(f"The profile after adding a tie ranking:")
            new_prof.anonymize().display()
            print(new_prof.description())
            vm.display(new_prof)
        return [new_prof]
    return []

neutral_indifference = Axiom(
    "Neutral Indifference",
    has_violation = has_neutral_indifference_violation,
    find_all_violations = find_all_neutral_indifference_violations,
)

def has_nonlinear_neutral_reversal_violation(prof, vm, verbose=False): 
    """
    Return True if there is a violation of the nonlinear neutral reversal axiom for the voting method vm.  Otherwise, return False.  That is, return True if there is a strict weak order and its reverse that can be added to the profile that changes the winning set according to vm.  
    """
    for swo in strict_weak_orders(prof.candidates):

        ranking = Ranking.from_indiff_list(swo)
        ranking_reverse = ranking.reverse()

        if isinstance(prof, ProfileWithTies):
            new_rankings = prof.rankings + [ranking, ranking_reverse]
            new_prof = ProfileWithTies(new_rankings)

        elif isinstance(prof, Profile):
            new_rankings = [Ranking.from_linear_order(r) for r in prof.rankings] + [ranking, ranking_reverse]
            new_prof = ProfileWithTies(new_rankings)
        
        if vm(prof) != vm(new_prof):
            if verbose:
                print("The original profile")
                prof.anonymize().display()
                print(prof.description())
                vm.display(prof)
                print("")
                print(f"The profile after adding {ranking} and its reverse {ranking_reverse}:")
                new_prof.anonymize().display()
                print(new_prof.description())
                vm.display(new_prof)
            return True
    return False


def find_all_nonlinear_neutral_reversal_violations(prof, vm, verbose=False): 
    """
    Return the list of strict weak orders and their reverse such that adding them to prof results in a different winning set according to vm.  Otherwise, return the empty list.  
    """

    violation_swos = []
    for swo in strict_weak_orders(prof.candidates):

        ranking = Ranking.from_indiff_list(swo)
        ranking_reverse = ranking.reverse()

        if isinstance(prof, ProfileWithTies):
            new_rankings = prof.rankings + [ranking, ranking_reverse]
            new_prof = ProfileWithTies(new_rankings)

        elif isinstance(prof, Profile):
            new_rankings = [Ranking.from_linear_order(r) for r in prof.rankings] + [ranking, ranking_reverse]
            new_prof = ProfileWithTies(new_rankings)
        
        if vm(prof) != vm(new_prof):
            if verbose:
                print("The original profile")
                prof.anonymize().display()
                print(prof.description())
                vm.display(prof)
                print("")
                print(f"The profile after adding {ranking} and its reverse {ranking_reverse}:")
                new_prof.anonymize().display()
                print(new_prof.description())
                vm.display(new_prof)
            violation_swos.append((ranking, ranking_reverse))

    return violation_swos

nonlinear_neutral_reversal = Axiom(
    "Nonlinear Neutral Reversal",
    has_violation = has_nonlinear_neutral_reversal_violation,
    find_all_violations = find_all_nonlinear_neutral_reversal_violations,
)

variable_voter_axioms = [
    reinforcement,
    positive_involvement,
    negative_involvement,
    positive_negative_involvement,
    tolerant_positive_involvement,
    bullet_vote_positive_involvement,
    semi_positive_involvement,
    truncated_involvement,
    participation,
    single_voter_resolvability,
    neutral_reversal,
    neutral_indifference,
    nonlinear_neutral_reversal,
]