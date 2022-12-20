
from pref_voting.profiles import Profile
from pref_voting.profiles_with_ties import ProfileWithTies
from pref_voting.weighted_majority_graphs import MajorityGraph
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
