
from pref_voting.profiles import Profile
from pref_voting.profiles_with_ties import ProfileWithTies


def display_mg(edata): 
    if type(edata) == Profile or type(edata) == ProfileWithTies: 
        edata.display_margin_graph()
    else: 
        edata.display()

def list_to_string(cands, cmap): 
    return "{" + ', '.join([cmap[c] for c in cands]) + "}"

