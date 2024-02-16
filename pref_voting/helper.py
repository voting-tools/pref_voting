
from pref_voting.profiles import Profile
from pref_voting.profiles_with_ties import ProfileWithTies
from pref_voting.weighted_majority_graphs import MajorityGraph
from pref_voting.rankings import Ranking
from pref_voting.social_welfare_function import *
from pref_voting.voting_method import *
import random

import networkx as nx

def get_mg(edata, curr_cands = None): 
    
    if curr_cands == None: 
        if type(edata) == Profile or type(edata) == ProfileWithTies: 
            mg = MajorityGraph.from_profile(edata).mg
        else:
            mg = edata.mg
    else: 
        if type(edata) == Profile or type(edata) == ProfileWithTies:  
            mg = nx.DiGraph()
            mg.add_nodes_from(curr_cands)
            mg.add_edges_from([(c1,c2) for c1 in curr_cands for c2 in curr_cands if edata.majority_prefers(c1, c2)])
        else:
            mg = edata.mg.copy()
            mg.remove_nodes_from([c for c in edata.candidates if c not in curr_cands])
    return mg


def get_weak_mg(edata, curr_cands = None): 
    
    if curr_cands == None: 
        if type(edata) == Profile or type(edata) == ProfileWithTies: 
            wmg = MajorityGraph.from_profile(edata).mg
        else:
            wmg = edata.mg
        wmg.add_edges_from([(c1, c2) for c1 in edata.candidates for c2 in edata.candidates if c1 != c2 and edata.is_tied(c1, c2)])
    else: 
        if type(edata) == Profile or type(edata) == ProfileWithTies:  
            wmg = nx.DiGraph()
            wmg.add_nodes_from(curr_cands)
            wmg.add_edges_from([(c1,c2) for c1 in curr_cands for c2 in curr_cands if c1 != c2 and (edata.majority_prefers(c1, c2) or edata.is_tied(c1, c2))])
        else:
            wmg = edata.mg.copy()
            wmg.remove_nodes_from([c for c in edata.candidates if c not in curr_cands])
            wmg.add_edges_from([(c1, c2) for c1 in curr_cands for c2 in curr_cands if c1 != c2 and edata.is_tied(c1, c2)])
    return wmg


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

            if tie_breaker is None:
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
        
    return SocialWelfareFunction(f, name = f"SWF from {vm.name}")


def vm_from_swf(swf):
    """
    Given a social welfare function, returns a voting method that selects all the candidates ranked first according to the swf.

    Args:
        swf (function): A social welfare function.

    Returns:
        function: A voting method that uses the swf to find the winning set.
    """
    
    def f(edata, curr_cands = None):
        return sorted(swf(edata, curr_cands = curr_cands).first())
        
    return VotingMethod(f, name = f"VM from {swf.name}")


def create_election(ranking_list, 
                    rcounts = None,
                    using_extended_strict_preference=None, 
                    candidates=None):
    """Creates an election from a list of rankings.
    
    Args:
        ranking_list (list): A list of rankings, which may be a list of tuples of candidates, a list of dictionaries or a list of Ranking objects.
        using_extended_strict_preference (bool, optional): Whether to use extended strict preference after creating a ProfileWithTies. Defaults to None.
        candidates (list, optional): A list of candidates.  Only used for creating a ProfileWithTies. Defaults to None (by default the candidates are all the candidates that are ranked by at least on voter).
    
    Returns:
        Profile or ProfileWithTies: The election profile.
    """

    if len(ranking_list) > 0 and (type(ranking_list[0]) == tuple or type(ranking_list[0]) == list):
        return Profile(ranking_list, rcounts=rcounts)
    elif len(ranking_list) > 0 and (type(ranking_list[0]) == dict or type(ranking_list[0]) == Ranking):
        if candidates is not None:
            prof = ProfileWithTies(ranking_list, candidates=candidates, rcounts=rcounts)
        else:
            prof = ProfileWithTies(ranking_list, rcounts=rcounts)       
        if using_extended_strict_preference:
            prof.use_extended_strict_preference()
        return prof
    else: # ranking_list is empty
        print("Warning: list of rankings is empty.")
        return Profile(ranking_list)