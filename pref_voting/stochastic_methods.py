'''
    File: stochastic_methods.py
    Author: Wes Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)
    Date: November 22, 2024
    
    Implementations of voting methods that output winners stochastically (unlike probabilistic methods, which output a probability distribution in the form of a dictionary).
'''

from pref_voting.voting_method import *
from pref_voting.iterative_methods import consensus_builder
from pref_voting.probabilistic_methods import maximal_lottery, RaDiUS
import math

@vm(name="Random Consensus Builder (Stochastic)")
def random_consensus_builder_st(profile, curr_cands=None, beta=0.5):

    """Version of the Random Consensus Builder (RCB) voting method due to Charikar et al. (https://arxiv.org/abs/2306.17838) that actually chooses a winner stochastically rather than outputting a probability distribution.

    Args:

        profile (Profile): An anonymous profile of linear orders
        curr_cands (List[int], optional): Candidates to consider. Defaults to all candidates if not provided.
        beta (float): Threshold for elimination (default 0.5). When processing candidate i, eliminates a candidate j
                    above i in the consensus building ranking if the proportion of voters preferring i to j is >= beta

    Returns:
        A sorted list of candidates.

    .. seealso::
        :meth:`pref_voting.iterative_methods.consensus_builder`
        :meth:`pref_voting.probabilistic_methods.random_consensus_builder`

    """
    consensus_building_ranking = random.choice(profile.rankings)

    return consensus_builder(profile, curr_cands=curr_cands, consensus_building_ranking=consensus_building_ranking, beta=beta)

@vm(name="Maximal Lotteries mixed with Random Consensus Builder")
def MLRCB(profile, curr_cands=None, p = 1 / math.sqrt(2), B = math.sqrt(2) - 1/2):

    """With probability p, choose the winner from the Maximal Lotteries distribution. With probability 1-p, run the stochastic version of Random Consensus Builder with beta chosen uniformly from (1/2, B). Ths method comes from Theorem 4 of Charikar et al. (https://arxiv.org/abs/2306.17838).

    Args:

        profile (Profile): An anonymous profile of linear orders
        curr_cands (List[int], optional): Candidates to consider. Defaults to all candidates if not provided.
        p (float): Probability of choosing the winner from the Maximal Lotteries distribution
        B (float): Upper bound for elimination threshold in the Random Consensus Builder method

    Returns:
        A sorted list of candidates.
    """

    if random.random() < p:
        return [maximal_lottery.choose(profile, curr_cands=curr_cands)]

    else:
        beta = random.uniform(0.5, B)
        return random_consensus_builder_st(profile, curr_cands=curr_cands, beta=beta)
    
@vm(name="Maximal Lotteries mixed with RaDiUS")
def MLRaDiUS(profile, curr_cands=None):

    """For p, B, and the probability distribution over beta given in the proof of Theorem 5 of Charikar et al. (https://arxiv.org/abs/2306.17838), choose the winner from the Maximal Lotteries distribution with probability p; with probability 1-p, run the RaDiUS method with beta chosen according to the distribution over beta.

    Args:
        profile (Profile): An anonymous profile of linear orders
        curr_cands (List[int], optional): Candidates to consider. Defaults to all candidates if not provided.

    Returns:
        A sorted list of candidates.
    """
    # Parameters
    B = 0.876353 # given in the proof of Theorem 5 of Charikar et al. (https://arxiv.org/abs/2306.17838)

    # Calculate p as per proof
    ln3 = np.log(3)
    LB = np.log((1 + B) / (1 - B))
    I = 0.5 * (LB - ln3)  # Integral value
    p = 1 / (1 + I)

    def sample_beta(B):
        # Generate a single uniform random number
        u = np.random.uniform(0, 1)

        # Compute E(u)
        Eu = np.exp(u * (LB - ln3) + ln3)

        # Compute beta sample
        beta_sample = (Eu - 1) / (Eu + 1)

        return beta_sample
    
    if random.random() < p:
        return [maximal_lottery.choose(profile, curr_cands=curr_cands)]
    
    else:
        beta = sample_beta(B)
        return [RaDiUS.choose(profile, curr_cands=curr_cands, beta=beta)]