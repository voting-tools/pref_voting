
from pref_voting.profiles import Profile
from pref_voting.profiles_with_ties import ProfileWithTies
from pref_voting.weighted_majority_graphs import MajorityGraph
from pref_voting.rankings import Ranking
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