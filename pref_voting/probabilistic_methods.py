'''
    File: voting_methods.py
    Author: Wes Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)
    Date: Nove 21, 2024
    
    Implementations of probabilistic voting methods.
'''

from pref_voting.prob_voting_method import  *
from pref_voting.weighted_majority_graphs import  MajorityGraph, MarginGraph
from scipy.optimize import linprog

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

@pvm(name="Random Dictator on the Beta-Uncovered Set")
def RaDiUS(profile, curr_cands = None, beta = 0.5):
    """
    Runs the Random Dictator method on the profile restricted to the beta-uncovered set, as proposed by Charikar et al. (https://arxiv.org/abs/2306.17838).

    Args:
        profile (Profile): An anonymous profile of linear orders
        curr_cands (List[int], optional): Candidates to consider. Defaults to all candidates if not provided. 

    Returns:
        dict: Maps each candidate to their probability of winning under the RaDiUS method.

    """
    from pref_voting.margin_based_methods import beta_uncovered_set

    curr_cands = profile.candidates if curr_cands is None else curr_cands

    rd_dist = random_dictator(profile, curr_cands = beta_uncovered_set(profile, curr_cands = curr_cands, beta = beta))
    
    rd_dist.update({c:0 for c in curr_cands if c not in rd_dist.keys()})

    return rd_dist

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

def clean_and_normalize(probs, threshold=1e-10):
    # Set  negative or small positive values to zero
    probs = np.where(probs < threshold, 0, probs)
    
    # Renormalize to ensure the probabilities sum to 1
    total = np.sum(probs)
    if total > 0:
        probs /= total
    return probs

def _maximal_lottery_enumeration(edata, curr_cands=None, margin_transformation=lambda x: x):
    '''
    Implementation of maximal lotteries. See http://dss.in.tum.de/files/brandt-research/fishburn_slides.pdf 
    
    Returns a randomly chosen maximal lottery.
    '''

    candidates = edata.candidates if curr_cands is None else curr_cands
    m_matrix, cand_to_cidx = edata.strength_matrix(curr_cands=candidates)

    A = np.array([[margin_transformation(m) for m in row] for row in m_matrix])

    # Create the game
    game = nash.Game(A)
    equilibria = []
    try:
        equilibria = list(game.vertex_enumeration())
        #print("Vertex Enumeration found equilibria.")
    except Exception as e:
        print(f"Vertex Enumeration failed: {e}")

    # Backup method 1: Support Enumeration
    if not equilibria:
        try:
            equilibria = list(game.support_enumeration())
            #print("Support Enumeration found equilibria.")
        except Exception as e:
            print(f"Support Enumeration failed: {e}")

    if len(equilibria) == 0:
        return {c: 1 / len(candidates) for c in candidates}
    else:
        # average the  equilibria component-wise to get a single equilibrium
        eq_probs = [np.mean([eq[0][idx] for eq in equilibria]) 
                    for idx in range(len(candidates))]

        eq_probs = clean_and_normalize(np.array(eq_probs))
        
        # Return the result as a dictionary
        return {c: eq_probs[cand_to_cidx(c)] for c in candidates}


def _maximal_lottery_lp(edata, curr_cands=None, margin_transformation=lambda x: x):
    '''
    Implementation of maximal lotteries using linear programming.
    '''
    candidates = edata.candidates if curr_cands is None else curr_cands
    m_matrix, cand_to_cidx = edata.strength_matrix(curr_cands=candidates)

    A = np.array([[margin_transformation(m) for m in row] for row in m_matrix])

    num_cands = len(candidates)
    c = np.zeros(num_cands + 1)
    c[-1] = -1  # Coefficient for v in the objective function (maximize v)

    # Inequalities: A^T p - v * 1 >= 0 => A^T p - v * 1 - s = 0 (s >= 0)
    # We need to convert this to the form: A_ub x <= b_ub

    A_ub = np.hstack([-A.T, np.ones((num_cands, 1))])
    b_ub = np.zeros(num_cands)

    # Equalities: sum p_i = 1
    A_eq = np.zeros((1, num_cands + 1))
    A_eq[0, :num_cands] = 1
    b_eq = np.array([1])

    bounds = [(0, None)] * num_cands + [(None, None)]  # p_i >= 0, v free

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if res.success:
        eq_probs = res.x[:num_cands]
        # Normalize to account for numerical errors
        eq_probs = np.maximum(eq_probs, 0)
        eq_probs /= np.sum(eq_probs)
        return {c: eq_probs[cand_to_cidx(c)] for c in candidates}
    else:
        # If LP fails, default to uniform distribution
        return {c: 1 / len(candidates) for c in candidates}


@pvm(name="C1 Maximal Lottery")
def c1_maximal_lottery(edata, curr_cands=None, algorithm='enumeration'): 

    '''Returns the C1 maximal lottery over the candidates.  See http://dss.in.tum.de/files/brandt-research/fishburn_slides.pdf.
    
    Args:   
        edata (Profile, MarginGraph): A Profile object.
        curr_cands (list): A list of candidates to restrict the ranking to. If ``None``, then the ranking is over the entire domain of the profile.
        algorithm (str): The algorithm to use. Either 'enumeration' or 'lp'. Defaults to 'enumeration'.
    
    Returns:
        dict: A dictionary mapping candidates to probabilities.

    .. note::
        The 'enumeration' algorithm averages over the extremal maximal lotteries.   The 'lp' is faster, but only returns a single maximal lottery (not necessarily the average)

    '''

    if type(edata) == MajorityGraph:
        # if edata is a MajorityGraph, we need to add margins for the following code to work.  The margins do not matter when finding the c1 maximal lottery.

        candidates = edata.candidates if curr_cands is None else curr_cands 
          
        edata = MarginGraph(candidates, [(c1, c2, 1) for c1, c2 in edata.edges if (c1 in candidates and c2 in candidates)])

    if algorithm == 'enumeration':
        return _maximal_lottery_enumeration(edata, curr_cands=curr_cands, margin_transformation = np.sign)
        
    elif algorithm == 'lp':
        return _maximal_lottery_lp(edata, curr_cands=curr_cands, margin_transformation = np.sign)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

@pvm(name="Maximal Lottery")
def maximal_lottery(edata, curr_cands=None, algorithm='lp'): 
    '''Returns the maximal lottery over the candidates.  See http://dss.in.tum.de/files/brandt-research/fishburn_slides.pdf.
    
    Args:   
        edata (Profile, MarginGraph): A Profile object.
        curr_cands (list): A list of candidates to restrict the ranking to. If ``None``, then the ranking is over the entire domain of the profile.
        algorithm (str): The algorithm to use. Either 'enumeration' or 'lp'. Defaults to 'enumeration'.

    Returns:
        dict: A dictionary mapping candidates to probabilities.
    
    .. note::
        The 'enumeration' algorithm averages over the extremal maximal lotteries.   The 'lp' is faster, but only returns a single maximal lottery (not necessarily the average)
    
    
    '''

    if algorithm == 'enumeration':
        return _maximal_lottery_enumeration(edata, curr_cands=curr_cands, margin_transformation = lambda x: x)
    
    elif algorithm == 'lp':
        return _maximal_lottery_lp(edata, curr_cands=curr_cands, margin_transformation = lambda x: x)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


@pvm(name="Random Consensus Builder")
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

    .. seealso::
        :meth:`pref_voting.iterative_methods.consensus_builder`
        :meth:`pref_voting.stochastic_methods.random_consensus_builder_st`
"""
    from pref_voting.iterative_methods import consensus_builder

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