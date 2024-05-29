"""
    File: dominance_axioms.py
    Author: Wes Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)
    Date: April 27, 2023
    
    Dominance axioms 
"""

from pref_voting.axiom import Axiom
from pref_voting.c1_methods import smith_set, schwartz_set
from pref_voting.axiom_helpers import *


def has_pareto_dominance_violation(edata, vm, verbose=False, strong_Pareto = False):
    """
    Returns True if some winner according to vm is Pareto dominated (there is a candidate that is unanimously preferred to the winner).

    If strong_Pareto is True, then a candidate A is dominated if there is a candidate B such that some voter prefers B to A and no voter prefers A to B.
    
    Args:
        edata (Profile, ProfileWithTies): the election data.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation. 

    Returns: 
        Result of the test (bool): Returns True if there is a violation and False otherwise. 

    """

    ws = vm(edata)
    for w in ws: 
        for c in edata.candidates: 
            if (strong_Pareto == False and edata.support(c,w)==edata.num_voters) or (strong_Pareto == True and edata.support(c,w)> 0 and edata.support(w,c)==0):
                if verbose:  
                    print(f"Pareto violation by {vm}:")
                    edata.display()
                    print(edata.description())
                    vm.display(edata)
                    print(f"The winner {w} is Pareto dominated by {c}.")
                    print()
                return True
    return False

def find_all_pareto_dominance_violations(edata, vm, verbose=False, strong_Pareto = False):
    """
    Returns all Pareto-dominated winners.

    If strong_Pareto is True, then a candidate A is dominated if there is a candidate B such that some voter prefers B to A and no voter prefers A to B.
    
    Args:
        edata (Profile, ProfileWithTies): the election data.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation. 

    Returns: 
        pareto_dominated_winners: A list of tuples of candidates (w, c) where w is a winner according to vm and Pareto dominated by c (and the empty list if there are none). 

    """

    ws = vm(edata)
    pareto_dominated_winners = list()
    for w in ws: 
        for c in edata.candidates: 
            if (strong_Pareto == False and edata.support(c,w)==edata.num_voters) or (strong_Pareto == True and edata.support(c,w)> 0 and edata.support(w,c)==0):
                pareto_dominated_winners.append((w, c))
    
    if len(pareto_dominated_winners) > 0 and verbose: 
        print(f"Pareto violation by {vm}:")
        edata.display()
        print(edata.description())
        vm.display(edata)
        for w,c in pareto_dominated_winners:
            print(f"The winner {w} is Pareto dominated by {c}.")

    return pareto_dominated_winners

pareto_dominance = Axiom(
    "Pareto Dominance Criterion",
    has_violation = has_pareto_dominance_violation,
    find_all_violations = find_all_pareto_dominance_violations, 
)

def has_condorcet_winner_violation(edata, vm, only_resolute=False, verbose=False):
    """
    Returns True if there is a Condorcet winner in edata (a candidate that is majority preferred to every other candidate) that is not the unique winner according to vm.
    
    Args:
        edata (Profile, ProfileWithTies, MajorityGraph, or MarginGraph): the election data.
        vm (VotingMethod): A voting method to test.
        only_resolute (bool, default=False): If True, only consider profiles in which there is a unique winner according to vm
        verbose (bool, default=False): If a violation is found, display the violation. 

    Returns: 
        Result of the test (bool): Returns True if there is a violation and False otherwise. 

    """

    cw = edata.condorcet_winner()

    ws = vm(edata)
    
    if only_resolute and len(ws) != 1: 
        return False
    if cw is not None and ws != [cw]:
        if verbose:
            if isinstance(edata, (Profile, ProfileWithTies)):
                edata.display_margin_graph()
            else:
                edata.display()
            print(edata.description())
            print(f"The Condorcet winner {cw} is not the unique winner: ")
            vm.display(edata)
        return True
    return False

def find_condorcet_winner_violation(edata, vm, only_resolute=False, verbose=False):
    """
    Returns the Condorcet winner that is not the unique winner according to vm.
    
    Args:
        edata (Profile, ProfileWithTies, MajorityGraph, or MarginGraph): the election data.
        vm (VotingMethod): A voting method to test.
        only_resolute (bool, default=False): If True, only consider profiles in which there is a unique winner according to vm.
        verbose (bool, default=False): If a violation is found, display the violation. 

    Returns: 
        Result of the test (bool): Returns True if there is a violation and False otherwise. 

    """

    cw = edata.condorcet_winner()

    ws = vm(edata)

    if only_resolute and len(ws) != 1: 
        return list()

    if cw is not None and ws != [cw]:
        if verbose: 
            if isinstance(edata, (Profile, ProfileWithTies)):
                edata.display_margin_graph()
            else:
                edata.display()
            print(edata.description())
            print(f"The Condorcet winner {cw} is not the unique winner: ")
            vm.display(edata)
        return [cw] 
    return list()

condorcet_winner = Axiom(
    "Condorcet Winner",
    has_violation = has_condorcet_winner_violation,
    find_all_violations = find_condorcet_winner_violation, 
)

def has_condorcet_loser_violation(edata, vm, verbose=False):
    """
    Returns True if there is a winner according to vm that is a Condorcet loser (a candidate that loses head-to-head to every other candidate).  
    
    Args:
        edata (Profile, ProfileWithTies, MajorityGraph, or MarginGraph): the election data.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation. 

    Returns: 
        Result of the test (bool): Returns True if there is a violation and False otherwise. 

    """

    cl = edata.condorcet_loser()

    ws = vm(edata)

    if cl is not None and cl in ws:
        if verbose: 
            if type(edata) == Profile or type(edata) == ProfileWithTies: 
                edata.display_margin_graph()
            else: 
                edata.display()
            print(edata.description())
            print(f"The Condorcet loser {cl} is an element of the winning set: ")
            vm.display(edata)
        return True 
    return False

def find_condorcet_loser_violation(edata, vm, verbose=False):
    """
    Returns the Condorcet loser (a candidate that loses head-to-head to every other candidate) who is a winner according to vm.  
    
    Args:
        edata (Profile, ProfileWithTies, MajorityGraph, or MarginGraph): the election data.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation. 

    Returns: 
        Result of the test (bool): Returns True if there is a violation and False otherwise. 

    """

    cl = edata.condorcet_loser()

    ws = vm(edata)

    if cl is not None and cl in ws:
        if verbose: 
            if type(edata) == Profile or type(edata) == ProfileWithTies: 
                edata.display_margin_graph()
            else: 
                edata.display()
            print(edata.description())
            print(f"The Condorcet loser {cl} is an element of the winning set: ")
            vm.display(edata)
        return [cl] 
    return list()

condorcet_loser = Axiom(
    "Condorcet Loser",
    has_violation = has_condorcet_loser_violation,
    find_all_violations = find_condorcet_winner_violation, 
)

def has_smith_violation(edata, vm, verbose=False):
    """
    Returns True if there is a winner according to vm that is not in the Smith set (the smallest set of candidates such that every candidate in the set is majority preferred to every candidate outside the set).  
    
    Args:
        edata (Profile, ProfileWithTies, MajorityGraph, or MarginGraph): the election data.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation. 

    Returns: 
        Result of the test (bool): Returns True if there is a violation and False otherwise. 

    """

    s_set = smith_set(edata)
    ws = vm(edata)

    winners_not_in_smith = [w for w in ws if w not in s_set]
    if len(winners_not_in_smith) > 0: 
        if verbose:
            if type(edata) == Profile or type(edata) == ProfileWithTies: 
                edata.display_margin_graph()
            else: 
                edata.display()
            print(f"The winners that are not in the Smith set: {list_to_string(winners_not_in_smith, edata.cmap)}.")
            vm.display(edata)
        return True 
    return False

def find_all_smith_violations(edata, vm, verbose=False):
    """
    Returns the winners according to vm that are not in the Smith set (the smallest set of candidates such that every candidate in the set is majority preferred to every candidate outside the set).  
    
    Args:
        edata (Profile, ProfileWithTies, MajorityGraph, or MarginGraph): the election data.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation. 

    Returns: 
        Result of the test (bool): Returns True if there is a violation and False otherwise. 

    """

    s_set = smith_set(edata)
    ws = vm(edata)

    winners_not_in_smith = [w for w in ws if w not in s_set]
    if len(winners_not_in_smith) > 0: 
        if verbose:
            if type(edata) == Profile or type(edata) == ProfileWithTies: 
                edata.display_margin_graph()
            else: 
                edata.display()
            print(f"The winners that are not in the Smith set: {list_to_string(winners_not_in_smith, edata.cmap)}.")
            vm.display(edata)
        return winners_not_in_smith 
    return list()

smith = Axiom(
    "Smith",
    has_violation = has_smith_violation,
    find_all_violations = find_all_smith_violations, 
)

def has_schwartz_violation(edata, vm, verbose=False):
    """
    Returns True if there is a winner according to vm that is not in the Schwartz set (the set of all candidates x such that if y can reach x in the transitive closer of the majority relation, then x can reach y in the transitive closer of the majority relation.).  
    
    Args:
        edata (Profile, ProfileWithTies, MajorityGraph, or MarginGraph): the election data.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation. 

    Returns: 
        Result of the test (bool): Returns True if there is a violation and False otherwise. 

    """

    s_set = schwartz_set(edata)
    ws = vm(edata)

    winners_not_in_schwartz = [w not in s_set for w in ws]
    if len(winners_not_in_schwartz) > 0: 
        if verbose:
            if type(edata) == Profile or type(edata) == ProfileWithTies: 
                edata.display_margin_graph()
            else: 
                edata.display()
            print(f"The winners that are not in the Schwartz set: {list_to_string(winners_not_in_schwartz, edata.cmap)}.")
            vm.display(edata)
        return True 
    return False

def find_all_schwartz_violations(edata, vm, verbose=False):
    """
    Returns the winners according to vm that are not in the Schwartz set (the set of all candidates x such that if y can reach x in the transitive closer of the majority relation, then x can reach y in the transitive closer of the majority relation).  
    
    Args:
        edata (Profile, ProfileWithTies, MajorityGraph, or MarginGraph): the election data.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation. 

    Returns: 
        Result of the test (bool): Returns True if there is a violation and False otherwise. 

    """

    s_set = schwartz_set(edata)
    ws = vm(edata)

    winners_not_in_schwartz = [w not in s_set for w in ws]
    if len(winners_not_in_schwartz) > 0: 
        if verbose:
            if type(edata) == Profile or type(edata) == ProfileWithTies: 
                edata.display_margin_graph()
            else: 
                edata.display()
            print(f"The winners that are not in the Schwartz set: {list_to_string(winners_not_in_schwartz, edata.cmap)}.")
            vm.display(edata)
        return winners_not_in_schwartz 
    return list()

schwartz = Axiom(
    "Schwartz",
    has_violation = has_schwartz_violation,
    find_all_violations = find_all_schwartz_violations, 
)

dominance_axioms = [
    pareto_dominance, 
    condorcet_winner, 
    condorcet_loser,
    smith,
    schwartz
]