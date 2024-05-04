"""
    File: swf_axioms.py
    Author: Wesley H. Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)
    Date: April 29, 2024
    
    SWF axioms 
"""

from pref_voting.axiom import Axiom
from pref_voting.axiom_helpers import *
from itertools import permutations
from pref_voting.helper import weak_orders

def has_pareto_ranking_violation(prof,swf,verbose=False, strong_Pareto=False):

    """Returns True if all voters rank x above y in prof, but the SWF does not rank x above y. 
    
    If verbose is True, prints the profile and the violation.

    If strong_Pareto is True, then a violation occurs when all voters weakly prefer x to y, some voter strictly prefers x to y, but the SWF does not rank x above y.
    
    Args:
        prof: a Profile or ProfileWithTies
        swf (SocialWelfareFunction): An SWF to test.
        verbose (bool, default=False): If a violation is found, display the violation.

    Returns:
        Result of the test (bool): Returns True if there is a violation and False otherwise.
    
    """
    social_ranking = swf(prof)

    for x in prof.candidates: 
        for y in prof.candidates: 
            if not social_ranking.extended_strict_pref(x,y): 
                if (strong_Pareto == False and prof.support(x,y)==prof.num_voters) or (strong_Pareto == True and prof.support(x,y)> 0 and prof.support(y,x)==0):
                    if verbose:  
                        print(f"Pareto ranking violation by {swf}:")
                        prof.display()
                        print(prof.description())
                        swf.display(prof)
                        print(f"Candidate {x} Pareto dominates {y} but is not ranked above {y} by {swf}.")
                        print()
                    return True
    return False

def find_all_pareto_ranking_violations(prof,swf,verbose=False, strong_Pareto=False):

    """Returns a list of all pairs of candidates for which the SWF violates the Pareto ranking axiom. 
    
    If verbose is True, prints the profile and the violation.

    If strong_Pareto is True, then a violation occurs when all voters weakly prefer x to y, some voter strictly prefers x to y, but the SWF does not rank x above y.
    
    Args:
        prof: a Profile or ProfileWithTies
        swf (SocialWelfareFunction): An SWF to test.
        verbose (bool, default=False): If a violation is found, display the violation.

    Returns:
        List of violations (list): List of all pairs of candidates for which there is a violation.
    
    """
    social_ranking = swf(prof)

    violations = []

    for x in prof.candidates: 
        for y in prof.candidates: 
            if not social_ranking.extended_strict_pref(x,y): 
                if (strong_Pareto == False and prof.support(x,y)==prof.num_voters) or (strong_Pareto == True and prof.support(x,y)> 0 and prof.support(y,x)==0):
                    if verbose:  
                        print(f"Pareto ranking violation by {swf}:")
                        prof.display()
                        print(prof.description())
                        swf.display(prof)
                        print(f"Candidate {x} Pareto dominates {y} but is not ranked above {y} by {swf}.")
                        print()
                    violations.append((x,y))
                    
    return violations

pareto_ranking = Axiom(
    "Pareto Ranking",
    has_violation = has_pareto_ranking_violation,
    find_all_violations = find_all_pareto_ranking_violations, 
)

def has_core_support_violation(prof,swf,verbose=False):
    """Returns True if the swf violates the "core support criterion" of https://arxiv.org/abs/2308.08430 for the given profile, False otherwise. If verbose is True, prints the profile and the core support violation.

    Args:
        prof: a Profile.
        swf (SocialWelfareFunction): An SWF to test.
        verbose (bool, default=False): If a violation is found, display the violation.

    Returns:
        Result of the test (bool): Returns True if there is a violation and False otherwise.
    """

    social_ranking = swf(prof)

    for x in prof.candidates:
        for y in prof.candidates:
            if social_ranking.extended_strict_pref(x,y):

                maj_cand_for_y = [c for c in prof.candidates if social_ranking.extended_weak_pref(c,y)]

                if isinstance(prof,Profile):
                    core_support_for_x_vs_y = [r for r in prof.rankings if r.index(x) == 0 or r.index(x) < min([r.index(c) for c in maj_cand_for_y if c != x])]     
                    core_support_for_y_vs_y = [r for r in prof.rankings if r.index(y) == 0 or r.index(y) < min([r.index(c) for c in maj_cand_for_y if c != y])]
                    core_support = core_support_for_x_vs_y + core_support_for_y_vs_y
                    restricted_prof = Profile(core_support)

                if not restricted_prof.majority_prefers(x,y):
                    if verbose:
                        print(f"Core support violation by {swf} for {x} relative to {y}:")
                        print(prof.anonymize())
                        prof.display_margin_graph()
                        print("Social ranking:",social_ranking)
                        print(f"Major candidates relative to {y}:",maj_cand_for_y)
                        print(f"Profile restricted to voters in core support for {x} relative to {y} and for {y} relative to {y}:")
                        print(restricted_prof.anonymize())
                        restricted_prof.display_margin_graph()
                    return True
    return False

def find_all_core_support_violations(prof,swf,verbose=False):
    """Returns a list of all pairs of candidates for which the swf violates the "core support criterion" of https://arxiv.org/abs/2308.08430 for the given profile. If verbose is True, prints the profile and the core support violation.

    Args:
        prof: a Profile.
        swf (SocialWelfareFunction): An SWF to test.
        verbose (bool, default=False): If a violation is found, display the violation.

    Returns:
        List of violations (list): List of all pairs of candidates for which there is a violation.
    """

    social_ranking = swf(prof)

    violations = []

    for x in prof.candidates:
        for y in prof.candidates:
            if social_ranking.extended_strict_pref(x,y):

                maj_cand_for_y = [c for c in prof.candidates if social_ranking.extended_weak_pref(c,y)]

                if isinstance(prof,Profile):
                    core_support_for_x_vs_y = [r for r in prof.rankings if r.index(x) == 0 or r.index(x) < min([r.index(c) for c in maj_cand_for_y if c != x])]     
                    core_support_for_y_vs_y = [r for r in prof.rankings if r.index(y) == 0 or r.index(y) < min([r.index(c) for c in maj_cand_for_y if c != y])]
                    core_support = core_support_for_x_vs_y + core_support_for_y_vs_y
                    restricted_prof = Profile(core_support)

                if not restricted_prof.majority_prefers(x,y):
                    if verbose:
                        print(f"Core support violation by {swf} for {x} relative to {y}:")
                        print(prof.anonymize())
                        prof.display_margin_graph()
                        print("Social ranking:",social_ranking)
                        print(f"Major candidates relative to {y}:",maj_cand_for_y)
                        print(f"Profile restricted to voters in core support for {x} relative to {y} and for {y} relative to {y}:")
                        print(restricted_prof.anonymize())
                        restricted_prof.display_margin_graph()
                    violations.append((x,y))
    return violations

core_support = Axiom(
    "Core Support",
    has_violation = has_core_support_violation,
    find_all_violations = find_all_core_support_violations, 
)