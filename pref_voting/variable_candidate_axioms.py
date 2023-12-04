"""
    File: variable_candidate_axioms.py
    Author: Wes Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)
    Date: November 18, 2023
    
    Variable candidate axioms 
"""

from pref_voting.axiom import Axiom
from pref_voting.axiom_helpers import *
from pref_voting.c1_methods import top_cycle
import numpy as np

def has_stability_for_winners_violation(edata, vm, verbose=False, strong_stability=False):
    """
    Returns True if there is some candidate A who wins without another candidate B in the election, A is majority preferred to B, but A loses when B is included in the election.

    If strong_stability is True, then A can be weakly majority preferred to B.

    Args:
        edata (Profile, ProfileWithTies, MarginGraph): Any election data that has a `margin` method. 
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation.

    Returns:
        bool: True if there is a violation, False otherwise.
    """
    winners = vm(edata)
    losers = [c for c in edata.candidates if c not in winners]

    for a in losers:
        for b in edata.candidates:
            if edata.margin(a,b) > 0:
                winners_in_reduced_prof = vm(edata, curr_cands = [x for x in edata.candidates if x != b])
                if a in winners_in_reduced_prof:
                    if verbose:
                        print(f"Stability for Winners violation for {vm.name}.")
                        print(f"{a} wins without {b} in the election and is majority preferred to {b} but loses when {b} is included:")
                        edata.display()
                        print(edata.description())
                        if isinstance(edata, Profile):
                            edata.display_margin_graph()
                        print(f"Winners in full election: {winners}")
                        print(f"Winners in election without {b}: {winners_in_reduced_prof}")
                    return True   

            if strong_stability and edata.margin(a,b) == 0:  
                winners_in_reduced_prof = vm(edata, curr_cands = [x for x in edata.candidates if x != b])
                if a in winners_in_reduced_prof:
                    if verbose:
                        print(f"Strong Stability for Winners violation for {vm.name}.")
                        print(f"{a} wins without {b} in the election and is weakly majority preferred to {b} but loses when {b} is included:")
                        edata.display()
                        print(edata.description())
                        if isinstance(edata, Profile):
                            edata.display_margin_graph()
                        print(f"Winners in full election: {winners}")
                        print(f"Winners in election without {b}: {winners_in_reduced_prof}")
                    return True    
    return False

def find_all_stability_for_winners_violations(edata, vm, verbose=False, strong_stability = False):
    """
    Returns all violations of Stability for Winners (some candidate A wins without another candidate B in the election, A is majority preferred to B, but A loses when B is included in the election) for the given election data and voting method.

    If strong_stability is True, then A can be weakly majority preferred to B.
    
    Args:
        edata (Profile, ProfileWithTies, MarginGraph): Any election data that has a `margin` method. 
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation. 

    Returns: 
        List of pairs (cand1,cand2) where cand1 wins without cand2 in the election, cand1 is majority preferred to cand2, but cand1 loses when cand2 is included in the election.

    """
    winners = vm(edata)
    losers = [c for c in edata.candidates if c not in winners]

    violations = list()

    for a in losers:
        for b in edata.candidates:
            if edata.margin(a,b) > 0:
                winners_in_reduced_prof = vm(edata, curr_cands = [x for x in edata.candidates if x != b])
                if a in winners_in_reduced_prof:
                    violations.append((a,b))
                    if verbose:
                        print(f"Stability for Winners violation for {vm.name}.")
                        print(f"{a} wins without {b} in the election and is majority preferred to {b} but loses when {b} is included:")
                        edata.display()
                        print(edata.description())
                        if isinstance(edata, Profile):
                            edata.display_margin_graph()
                        print(f"Winners in full election: {winners}")
                        print(f"Winners in election without {b}: {winners_in_reduced_prof}")

            if strong_stability and edata.margin(a,b) == 0:
                winners_in_reduced_prof = vm(edata, curr_cands = [x for x in edata.candidates if x != b])
                if a in winners_in_reduced_prof:
                    violations.append((a,b))
                    if verbose:
                        print(f"Strong Stability for Winners violation for {vm.name}.")
                        print(f"{a} wins without {b} in the election and is weakly majority preferred to {b} but loses when {b} is included:")
                        edata.display()
                        print(edata.description())
                        if isinstance(edata, Profile):
                            edata.display_margin_graph()
                        print(f"Winners in full election: {winners}")
                        print(f"Winners in election without {b}: {winners_in_reduced_prof}")

    return violations  

stability_for_winners = Axiom(
    "Stability for Winners",
    has_violation = has_stability_for_winners_violation,
    find_all_violations = find_all_stability_for_winners_violations, 
)

def has_immunity_to_spoilers_violation(edata, vm, verbose=False, strong_immunity = False):
    """
    Returns True if there is some candidate A who wins without another candidate B in the election, A is majority preferred to B, but both A and B lose when B is included in the election.

    If strong_immunity is True, then A can be weakly majority preferred to B.

    Args:
        edata (Profile, ProfileWithTies, MarginGraph): Any election data that has a `margin` method. 
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation.

    Returns:
        bool: True if there is a violation, False otherwise.
    """
    winners = vm(edata)
    losers = [c for c in edata.candidates if c not in winners]

    for a in losers:
        for b in losers:
            if edata.margin(a,b) > 0:
                winners_in_reduced_prof = vm(edata, curr_cands = [x for x in edata.candidates if x != b])
                if a in winners_in_reduced_prof:
                    if verbose:
                        print(f"Immunity to Spoilers violation for {vm.name}.")
                        print(f"{a} wins without {b} in the election and is majority preferred to {b} but both lose when {b} is included:")
                        edata.display()
                        print(edata.description())
                        if isinstance(edata, Profile):
                            edata.display_margin_graph()
                        print(f"Winners in full election: {winners}")
                        print(f"Winners in election without {b}: {winners_in_reduced_prof}")
                    return True   

            if strong_immunity and edata.margin(a,b) == 0:
                winners_in_reduced_prof = vm(edata, curr_cands = [x for x in edata.candidates if x != b])
                if a in winners_in_reduced_prof:
                    if verbose:
                        print(f"Strong Immunity to Spoilers violation for {vm.name}.")
                        print(f"{a} wins without {b} in the election and is weakly majority preferred to {b} but both lose when {b} is included:")
                        edata.display()
                        print(edata.description())
                        if isinstance(edata, Profile):
                            edata.display_margin_graph()
                        print(f"Winners in full election: {winners}")
                        print(f"Winners in election without {b}: {winners_in_reduced_prof}")
                    return True   
    return False

def find_all_immunity_to_spoilers_violations(edata, vm, verbose=False, strong_immunity = False):
    """
    Returns all violations of Immunity to Spoilers (some candidate A wins without another candidate B in the election, A is majority preferred to B, but both A and B lose when B is included in the election) for the given election data and voting method.

    If strong_immunity is True, then A can be weakly majority preferred to B.

    Args:
        edata (Profile, ProfileWithTies, MarginGraph): Any election data that has a `margin` method. 
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation.

    Returns:
        List of pairs (cand1,cand2) where cand1 wins without cand2 in the election, cand1 is majority preferred to cand2, but both cand1 and cand2 lose when cand2 is included in the election.
    
    """

    winners = vm(edata)
    losers = [c for c in edata.candidates if c not in winners]

    violations = list()

    for a in losers:
        for b in losers:
            if edata.margin(a,b) > 0:
                winners_in_reduced_prof = vm(edata, curr_cands = [x for x in edata.candidates if x != b])
                if a in winners_in_reduced_prof:
                    violations.append((a,b))
                    if verbose:
                        print(f"Immunity to Spoilers violation for {vm.name}.")
                        print(f"{a} wins without {b} in the election and is majority preferred to {b} but both lose when {b} is included:")
                        edata.display()
                        print(edata.description())
                        if isinstance(edata, Profile):
                            edata.display_margin_graph()
                        print(f"Winners in full election: {winners}")
                        print(f"Winners in election without {b}: {winners_in_reduced_prof}")

                if strong_immunity and edata.margin(a,b) == 0:
                    winners_in_reduced_prof = vm(edata, curr_cands = [x for x in edata.candidates if x != b])
                    if a in winners_in_reduced_prof:
                        violations.append((a,b))
                        if verbose:
                            print(f"Strong Immunity to Spoilers violation for {vm.name}.")
                            print(f"{a} wins without {b} in the election and is weakly majority preferred to {b} but both lose when {b} is included:")
                            edata.display()
                            print(edata.description())
                            if isinstance(edata, Profile):
                                edata.display_margin_graph()
                            print(f"Winners in full election: {winners}")
                            print(f"Winners in election without {b}: {winners_in_reduced_prof}")

    return violations

immunity_to_spoilers = Axiom(
    "Immunity to Spoilers",
    has_violation = has_immunity_to_spoilers_violation,
    find_all_violations = find_all_immunity_to_spoilers_violations,
)

def has_ISDA_violation(edata, vm, verbose=False):
    """
    Independence of Smith-Dominated Alternatives: returns True if there is a candidate A outside of the Smith set such that removing A changes the set of winners according to the voting method vm.

    Args:
        edata (Profile, ProfileWithTies, MarginGraph): Any election data that has a `margin` method. 
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation.

    Returns:
        bool: True if there is a violation, False otherwise.
    """

    winners = vm(edata)
    smith_set = top_cycle(edata)
    non_smith_set = [c for c in edata.candidates if c not in smith_set]

    for a in non_smith_set:
        winners_in_reduced_prof = vm(edata, curr_cands = [x for x in edata.candidates if x != a])
        if winners != winners_in_reduced_prof:
            if verbose:
                print(f"ISDA violation for {vm.name}.")
                print(f"{a} is outside of the Smith set and removing {a} changes the set of winners:")
                edata.display()
                print(edata.description())
                if isinstance(edata, Profile):
                    edata.display_margin_graph()
                print(f"Winners in full election: {winners}")
                print(f"Winners in election without {a}: {winners_in_reduced_prof}")
            return True
    return False

def find_all_ISDA_violations(edata, vm, verbose=False):
    """
    Returns all violations of ISDA (some candidate A outside of the Smith set such that removing A changes the set of winners according to the voting method vm) for the given election data and voting method.

    Args:
        edata (Profile, ProfileWithTies, MarginGraph): Any election data that has a `margin` method. 
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation.

    Returns:
        List of candidates A outside of the Smith set such that removing A changes the set of winners according to the voting method vm.
    """

    winners = vm(edata)
    smith_set = top_cycle(edata)
    non_smith_set = [c for c in edata.candidates if c not in smith_set]

    violations = list()

    for a in non_smith_set:
        winners_in_reduced_prof = vm(edata, curr_cands = [x for x in edata.candidates if x != a])
        if winners != winners_in_reduced_prof:
            violations.append(a)
            if verbose:
                print(f"ISDA violation for {vm.name}.")
                print(f"{a} is outside of the Smith set and removing {a} changes the set of winners:")
                edata.display()
                print(edata.description())
                if isinstance(edata, Profile):
                    edata.display_margin_graph()
                print(f"Winners in full election: {winners}")
                print(f"Winners in election without {a}: {winners_in_reduced_prof}")
    return violations

ISDA = Axiom(
    "Independence of Smith-Dominated Alternatives",
    has_violation = has_ISDA_violation,
    find_all_violations = find_all_ISDA_violations,
)

def has_IPDA_violation(prof, vm, verbose=False, strong_Pareto = False):
    """
    Independence of Pareto-Dominated Alternatives: returns True if there is a candidate A who is Pareto-dominated by another candidate B such that removing A changes the set of winners according to the voting method vm.

    If strong_Pareto is True, then a candidate A is dominated if there is a candidate B such that some voter prefers B to A and no voter prefers A to B.

    Args:
        prof (Profile, ProfileWithTies): the election data.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation.

    Returns:
        bool: True if there is a violation, False otherwise.
    """

    winners = vm(prof)
    pareto_dominated = list()

    for a in prof.candidates:
        for b in prof.candidates:
            if (strong_Pareto == False and prof.support(b,a) == prof.num_voters) or (strong_Pareto == True and prof.support(b,a) > 0 and prof.support(a,b) == 0): 
                pareto_dominated.append(a)
                break

    for a in pareto_dominated:
        winners_in_reduced_prof = vm(prof, curr_cands = [x for x in prof.candidates if x != a])
        if winners != winners_in_reduced_prof:
            if verbose:
                print(f"IPDA violation for {vm.name}.")
                print(f"{a} is Pareto-dominated by another candidate and removing {a} changes the set of winners:")
                prof.display()
                print(prof.description())
                prof.display_margin_graph()
                print(f"Winners in full election: {winners}")
                print(f"Winners in election without {a}: {winners_in_reduced_prof}")
            return True
    return False

def find_all_IPDA_violations(prof, vm, verbose=False, strong_Pareto = False):
    """
    Returns all violations of IPDA (some candidate A who is Pareto-dominated by another candidate B such that removing A changes the set of winners according to the voting method vm) for the given election data and voting method.

    If strong_Pareto is True, then a candidate A is dominated if there is a candidate B such that some voter prefers B to A and no voter prefers A to B.

    Args:
        prof (Profile, ProfileWithTies): the election data.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation.

    Returns:
        List of candidates A who are Pareto-dominated by another candidate such that removing A changes the set of winners according to the voting method vm.
    """

    winners = vm(prof)
    pareto_dominated = list()

    for a in prof.candidates:
        for b in prof.candidates:
            if (strong_Pareto == False and prof.support(b,a) == prof.num_voters) or (strong_Pareto == True and prof.support(b,a) > 0 and prof.support(a,b) == 0): 
                pareto_dominated.append(a)
                break

    violations = list()

    for a in pareto_dominated:
        winners_in_reduced_prof = vm(prof, curr_cands = [x for x in prof.candidates if x != a])
        if winners != winners_in_reduced_prof:
            violations.append(a)
            if verbose:
                print(f"IPDA violation for {vm.name}.")
                print(f"{a} is Pareto-dominated by another candidate and removing {a} changes the set of winners:")
                prof.display()
                print(prof.description())
                prof.display_margin_graph()
                print(f"Winners in full election: {winners}")
                print(f"Winners in election without {a}: {winners_in_reduced_prof}")
    return violations

IPDA = Axiom(
    "Independence of Pareto-Dominated Alternatives",
    has_violation = has_IPDA_violation,
    find_all_violations = find_all_IPDA_violations,
)

variable_candidate_axioms = [
    stability_for_winners,
    immunity_to_spoilers,
    ISDA,
    IPDA,
]