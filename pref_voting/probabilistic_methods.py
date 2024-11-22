'''
    File: voting_methods.py
    Author: Wes Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)
    Date: Nove 21, 2024
    
    Implementations of probabilistic voting methods.
'''

from pref_voting.prob_voting_method import  *
from pref_voting.weighted_majority_graphs import  MajorityGraph, MarginGraph
from pred_voting.iterative_methods import consensus_builder
import random
import nashpy as nash

@pvm(name="Random Dictator")
def random_dictator(profile, curr_cands = None): 
    '''Returns lottery over the candidates that is proportional to the Plurality scores. 

    Args:
        profile (Profile): A Profile object.
        curr_cands (list): A list of candidates to restrict the ranking to. If ``None``, then the ranking is over the entire domain of the profile.

    Returns:
        dict: A dictionary mapping candidates to probabilities.
    ''' 
    
    plurality_scores = profile.plurality_scores(curr_cands = curr_cands)
    total_plurality_scores = sum(list(plurality_scores.values()))

    return {c: plurality_scores[c] / total_plurality_scores for c in plurality_scores.keys()}

@pvm(name="Proportional Borda")
def pr_borda(profile, curr_cands=None): 
    '''Returns lottery over the candidates that is proportional to the Borda scores.
    
    Args:   
        profile (Profile): A Profile object.
        curr_cands (list): A list of candidates to restrict the ranking to. If ``None``, then the ranking is over the entire domain of the profile.

    Returns:
        dict: A dictionary mapping candidates to probabilities.
    
    '''
    borda_scores = profile.borda_scores(curr_cands=curr_cands)
    total_borda_scores = sum(list(borda_scores.values()))

    return {c: borda_scores[c] / total_borda_scores for c in borda_scores.keys()}


def _maximal_lottery(edata, curr_cands = None, margin_transformation = lambda x: x):
    '''Implementation of maximal lotteries.   See http://dss.in.tum.de/files/brandt-research/fishburn_slides.pdf 
    
    Returns a randomly chosen maximal lottery.
    '''

    candidates = edata.candidates if curr_cands is None else curr_cands
    m_matrix, cand_to_cidx = edata.strength_matrix(curr_cands=candidates)

    A = np.array([[margin_transformation(m) for m in row] 
                  for row in m_matrix])

    # Create the game
    game = nash.Game(A)
    # Find the Nash Equilibrium with Vertex Enumeration
    equilibria = list(game.vertex_enumeration())
    if len(equilibria) == 0:
        return {c: 1/len(candidates) for c in candidates}
    else:
        eq = random.choice(equilibria)
        return {c: eq[0][cand_to_cidx(c)] for c in candidates}

@pvm(name="C1 Maximal Lottery")
def c1_maximal_lottery(edata, curr_cands=None): 

    '''Returns the C1 maximal lottery over the candidates.  See http://dss.in.tum.de/files/brandt-research/fishburn_slides.pdf.
    
    Args:   
        edata (Profile, MarginGraph): A Profile object.
        curr_cands (list): A list of candidates to restrict the ranking to. If ``None``, then the ranking is over the entire domain of the profile.
    
    Returns:
        dict: A dictionary mapping candidates to probabilities.
    '''

    if type(edata) == MajorityGraph:
        # if edata is a MajorityGraph, we need to add margins for the following code to work.  The margins do not matter when finding the c1 maximal lottery.

        candidates = edata.candidates if curr_cands is None else curr_cands 
          
        edata = MarginGraph(candidates, [(c1, c2, 1) for c1, c2 in edata.edges if (c1 in candidates and c2 in candidates)])

    return _maximal_lottery(edata, 
                            curr_cands=curr_cands, 
                            margin_transformation = np.sign)

@pvm(name="Maximal Lottery")
def maximal_lottery(edata, curr_cands=None): 
    '''Returns the maximal lottery over the candidates.  See http://dss.in.tum.de/files/brandt-research/fishburn_slides.pdf.
    
    Args:   
        edata (Profile, MarginGraph): A Profile object.
        curr_cands (list): A list of candidates to restrict the ranking to. If ``None``, then the ranking is over the entire domain of the profile.


    Returns:
        dict: A dictionary mapping candidates to probabilities.
    
    '''
    return _maximal_lottery(edata, 
                            curr_cands=curr_cands, 
                            margin_transformation = lambda x: x)

#@pvm(name="Random Consensus Builder")
def random_consensus_builder(profile, curr_cands=None, beta=0.5):
    """Random Consensus Builder (RCB) voting method due to Charikar et al. (https://arxiv.org/abs/2306.17838).

    For each ranking type in the profile, runs the deterministic Consensus Builder voting method using that ranking
    as the consensus building ranking. The probability of a candidate winning is proportional to the
    number of voters with rankings that would make that candidate win when used as the consensus
    building ranking.

    Args:
        profile (Profile): An anonymous profile of linear orders
        curr_cands (List[int], optional): Candidates to consider. Defaults to all candidates if not provided.
        beta (float): Threshold for elimination (default 0.5). When processing candidate i, eliminates a candidate j
                    above i in the consensus building ranking if the proportion of voters preferring i to j is >= beta

    Returns:
        dict: Maps each candidate to their probability of winning under the RCB method
"""
    if curr_cands is None:
        curr_cands = profile.candidates

    # Count how many times each ranking type produces each winner
    winner_counts = {c: 0 for c in curr_cands}

    # Process each unique ranking type
    for ranking_type in profile.ranking_types:
        # Count number of voters with this ranking type
        num_rankings_with_type = len([r for r in profile.rankings if r == ranking_type])
        winner = consensus_builder(profile, curr_cands=curr_cands,consensus_building_ranking=ranking_type, beta=beta)[0]
        winner_counts[winner] += num_rankings_with_type
        total_count += num_rankings_with_type

    # Convert counts to probabilities
    return {c: count/profile.num_voters for c, count in winner_counts.items()}


def create_probabilistic_method(vm):
    """
    Create a probabilistic voting method from a voting method.
    """
    
    from pref_voting.voting_method import VotingMethod
    if type(vm) != VotingMethod:
        raise TypeError("vm must be a VotingMethod object")
    
    def _pvm(profile, curr_cands=None, **kwargs):
        return vm.prob(profile, curr_cands=curr_cands, **kwargs)
    
    return ProbVotingMethod(_pvm, name=f'{vm.name} with Even Chance Tiebreaking')

def mixture(pvm1, pvm2, alpha):
    """
    Mixture of the two probabilistic voting methods pvm1 and pvm2 with mixing parameter alpha. With probability alpha, the output is the output of pvm1, and with probability 1-alpha, the output is the output of pvm2.
    """
    def _mixture(profile, curr_cands=None, **kwargs):
        return {c: alpha * pvm1(profile, curr_cands=curr_cands, **kwargs)[c] + (1-alpha) * pvm2(profile, curr_cands=curr_cands, **kwargs)[c] for c in profile.candidates}
    
    return ProbVotingMethod(_mixture, name=f'Mixture of {pvm1.name} and {pvm2.name} with alpha={alpha}')