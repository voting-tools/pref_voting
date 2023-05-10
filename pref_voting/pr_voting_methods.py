'''
    File: pr_voting_methods.py
    Author: Eric Pacuit (epacuit@umd.edu)
    Date: April 5, 2021
    
    Implementations of probabilistic voting methods.   
    Probabilistic voting methods returns a lottery over the candidates of a profile, 
    where a lottery is represented as a dictionary assigning elements from [0,1]
    to  the candidates that sums to 1.  
'''

import numpy as np
import os
import uuid
import random
from pref_voting.voting_methods_OLD import *

dir_path = os.path.dirname(os.path.realpath(__file__))

'''TODO: 
   * implement tie-breaking and PUT for Hare and other iterative methods
   * to optimize the iterative methods, I am currently using the private compiled methods
     _borda_score and _find_updated_profiles. We should think of a better way to deal with this issue. 
   * implement other voting methods: e.g., Dodgson
   * implement the linear programming version of Ranked Pairs: https://arxiv.org/pdf/1805.06992.pdf
   * implement a Voting Method class?
'''

######
# Helper functions
######

def same_lotteries(p1, p2): 
    '''returns true if p1 and p2 are the same lottery'''
    return all([c in p2.keys() and p1[c] == p2[c] for c in p1.keys()])


#####
# Methods with Even Chance Tiebreaking
#####

even_chance_vms = list()

def even_chance(vm): 
    '''Converts a voting method vm to an even chance probabilistic voting method. 
    These methods return the uniform probability over the set of winners according to vm
    and assign 0 probablity to all candidates not in the winning set. 
    
    vm is a voting method.  Returns a function named "vm_ec" with the name 
    attribute "vm.name EC"
    '''
    def pr_vm(profile): 
        ws = vm(profile)
        return {c: 1 / len(ws) if c in ws else 0.0 for c in profile.candidates}
    pr_vm.__name__ = vm.__name__ + '_ec'
    pr_vm.name = vm.name + ' EC'
    return pr_vm

# create even chance vms for all voting methods and
# add them to the global namespace 
for vm in all_vms: 
    vm_ec = even_chance(vm)
    globals()[vm_ec.__name__] = vm_ec
    even_chance_vms.append(vm_ec)

@vm_name("Split Cycle EC")
def split_cycle_faster_ec(profile):
    ws = split_cycle_faster(profile)
    return {c: 1 / len(ws) if c in ws else 0.0 for c in profile.candidates}


#####
# Methods with Random Vote Tiebreaking
#####

random_voter_vms = list()

def random_voter(vm): 
    '''Converts a voting method vm to an  probabilistic voting method. 
    These methods return the a lottery over the set of winners according to vm
    that is weighted according to the plurality scores in the profile restricted to 
    the winners according to vm (and assign 0 probablity to all candidates not in the winning set). 
    
    vm is a voting method.  Returns a function named "vm_rv" with the name 
    attribute "vm.name RV"
    '''
    def pr_vm(profile): 
        ws = vm(profile)
        winner_profile, cname = profile.remove_candidates([c for c in profile.candidates if c not in ws])
        _ws_pl_scores = winner_profile.plurality_scores()
        ws_pl_scores = {cname[c]: _ws_pl_scores[c] for c in _ws_pl_scores.keys()}
        total_pl_score = sum(ws_pl_scores.values())
        return {c: ws_pl_scores[c] / total_pl_score if c in ws_pl_scores.keys() else 0.0 
                for c in profile.candidates}
    pr_vm.__name__ = vm.__name__ + '_rv'
    pr_vm.name = vm.name + ' RV'
    return pr_vm

# create  vms with random voter tiebreaking for all voting methods and
# add them to the global namespace 
for vm in all_vms: 
    vm_rv = random_voter(vm)
    globals()[vm_rv.__name__] = vm_rv
    random_voter_vms.append(vm_rv)
    
@vm_name("Split Cycle RV")
def split_cycle_faster_rv(profile):
    ws = split_cycle_faster(profile)
    winner_profile, cname = profile.remove_candidates([c for c in profile.candidates if c not in ws])
    _ws_pl_scores = winner_profile.plurality_scores()
    ws_pl_scores = {cname[c]: _ws_pl_scores[c] for c in _ws_pl_scores.keys()}
    total_pl_score = sum(ws_pl_scores.values())
    return {c: ws_pl_scores[c] / total_pl_score if c in ws_pl_scores.keys() else 0.0 
            for c in profile.candidates}

    
######
# Probabilistic Voting Methods
#####

def _maximal_lottery(profile, margin_transformation = lambda x: x):
    '''Implementation of maximal lotteries.   See http://dss.in.tum.de/files/brandt-research/fishburn_slides.pdf
    Returns a list of all maximal lotteries (there may be more than 1)
    '''
    
    # lp solver to find the Nash equilibrium
    lrsnash = f'./voting/lrs/lrsnash'
    game_file = f'./voting/lrs/margin_gm{str(uuid.uuid4().hex)}'
    
    num_cands = profile.num_cands
    
    # write the margin game to a file
    margin_gm_file = open(f"{game_file}", "w")
    margin_gm_file.write(f"{num_cands} {num_cands}\n")
    margin_gm_file.write("\n")
    for c1 in profile.candidates:
        s = ''
        for c2 in profile.candidates: 
            s += f"{margin_transformation(profile.margin(c1,c2))} "
        margin_gm_file.write(s + "\n")
    margin_gm_file.write("\n")
    for c1 in profile.candidates:
        s = ''
        for c2 in profile.candidates: 
            s += f"{margin_transformation(profile.margin(c2,c1))} "
        margin_gm_file.write(s + "\n")
    margin_gm_file.close()

    # call lrsnash: http://cgm.cs.mcgill.ca/~avis/C/lrslib/USERGUIDE.html#nash
    stream = os.popen(f'{lrsnash} {game_file}')
    output = stream.read().split("\n")

    lotteries = list()
    for line in output: 
        if line.startswith("1"):
            lottery = dict()
            prs = line.split("  ")[1:num_cands + 1]
            for  c_idx,c in enumerate(profile.candidates):
                pr = prs[c_idx].split("/")
                prob_num = pr[0]
                prob_denom = pr[1] if len(pr) == 2 else 1.0
                lottery[c] = float(prob_num) / float(prob_denom)

            lotteries.append(lottery)
    os.remove(game_file) # delete the temporary game file
    
    return lotteries

@vm_name("C1 Maximal Lottery")
def c1_maximal_lottery(profile, return_one_lottery = False): 
    if return_one_lottery:
        return random.choice(_maximal_lottery(profile, margin_transformation = np.sign))
    else: 
        return _maximal_lottery(profile, margin_transformation = np.sign)

@vm_name("Bipartisan Set")
def bipartisan(profile): 
    """The Bipartisan Set is the support of a C1 maximal lottery
    """
    ml = c1_maximal_lottery(profile)

    return sorted(list(set([c  for l  in ml for c in l.keys() if l[c] > 0])))
@vm_name("C2 Maximal Lottery")
def c2_maximal_lottery(profile, return_one_lottery = False): 
    if return_one_lottery:
        return random.choice(_maximal_lottery(profile, margin_transformation = lambda x: x))
    else: 
        return _maximal_lottery(profile, margin_transformation = lambda x: x)

@vm_name("Essential Set")
def essential(profile): 
    """The Essential Set is the support of a C2 maximal lottery
    """
    ml = c2_maximal_lottery(profile)

    return sorted(list(set([c  for l  in ml for c in l.keys() if l[c] > 0])))

@vm_name("Proportional Borda")
def pr_borda(profile): 
    '''Returns lottery over the candidates that is proportional to the Borda scores'''
    borda_scores = profile.borda_scores()
    total_borda_scores = sum(list(borda_scores.values()))
    return {c: borda_scores[c] / total_borda_scores for c in profile.candidates}

@vm_name("Random Voter")
def random_dictator(profile): 
    '''Returns lottery over the candidates that is proportional to the Plurality scores'''    
    plurality_scores = profile.plurality_scores()
    total_plurality_scores = sum(list(plurality_scores.values()))
    return {c: plurality_scores[c] / total_plurality_scores for c in profile.candidates}

######
# Voting Methods using Random Tiehandler procedure
#####

@vm_name("Ranked Pairs RVT")
def ranked_pairs_rvt(profile):
    '''Ranked pairs (see the ranked_pairs docstring for an explanation) where a random voter is chosen to break any ties in the margins.  Since voters have strict preferences, this always returns a trivial lottery over the candidates.
    '''
    
    wmg = generate_weak_margin_graph(profile)
    cw = profile.condorcet_winner()
    _num_winner = {c: 0 for c in profile.candidates}
    for  tb_ranking, rcount in zip(profile._rankings, profile._rcounts):
        tb_ranking = tuple(tb_ranking)
        # Ranked Pairs is Condorcet consistent, so simply return the Condorcet winner if exists
        if cw is not None: 
            winners = [cw]
            _num_winner[cw] = sum(profile._rcounts)
        else:
            winners = list()            
            margins = sorted(list(set([e[2]['weight'] for e in wmg.edges(data=True)])), reverse=True)

            rp_defeat = nx.DiGraph() 
            for m in margins: 
                edges = [e for e in wmg.edges(data=True) if e[2]['weight'] == m]
                sorted_edges = sorted(edges, key = lambda e: (tb_ranking.index(e[0]), tb_ranking.index(e[1])), reverse=False)
                for e in sorted_edges: 
                    rp_defeat.add_edge(e[0], e[1], weight=e[2]['weight'])
                    if  has_cycle(rp_defeat):
                        rp_defeat.remove_edge(e[0], e[1])
            _num_winner[unbeaten_candidates(rp_defeat)[0]] += rcount
    return {c: float(_num_winner[c]) / sum(profile._rcounts) for c in profile.candidates}



all_pr_vms = even_chance_vms + random_voter_vms + [   
    c1_maximal_lottery,
    c2_maximal_lottery,
    pr_borda,
    random_dictator
]


    
    