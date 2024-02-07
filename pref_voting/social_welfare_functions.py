'''
    File: social_welfare_functions.py
    Author: Wes Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)
    Date: February 6, 2024
    
    Implementations of social welfare functions
'''

from pref_voting.social_welfare_function import *
from pref_voting.rankings import Ranking
from pref_voting.profiles import Profile
import random

def swf_from_vm(vm, tie_breaker = None):
    """
    Given a voting method, returns a social welfare function that uses the voting method to rank the candidates (winners are ranked first; then they are excluded from curr_cands and the new winners are ranked second; etc.).

    Args:
        vm (function): A voting method.
        tie_breaker (str): The tie-breaking method to use. Options are "alphabetic", "random", and None. Default is None.

    Returns:
        function: A social welfare function that uses the voting method to rank the candidates.
    """
    
    def f(prof, curr_cands = None):

        cands = prof.candidates if curr_cands == None else curr_cands

        ranked_cands = list()
        ranking_dict = dict()

        n=0

        while n < len(cands):

            if len(ranked_cands) == len(cands):
                break

            ws = vm(prof, curr_cands = [c for c in cands if c not in ranked_cands])
            ranked_cands = ranked_cands + ws

            if tie_breaker == None:
                for c in ws:
                    ranking_dict[c] = n
                n += 1

            if tie_breaker == "alphabetic":
                sorted_ws = sorted(ws)
                for c in sorted_ws:
                    ranking_dict[c] = n
                    n += 1

            if tie_breaker == "random":
                random.shuffle(ws)
                for c in ws:
                    ranking_dict[c] = n
                    n += 1            

        return Ranking(ranking_dict)
        
    return SWF(f, name = f"SWF from {vm.name}")