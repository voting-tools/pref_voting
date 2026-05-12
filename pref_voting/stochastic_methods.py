'''
    File: stochastic_methods.py
    Author: Wes Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)
    Date: November 22, 2024
    
    Implementations of voting methods that output winners stochastically (unlike probabilistic methods, which output a probability distribution in the form of a dictionary).
'''

from pref_voting.voting_method import *
from pref_voting.iterative_methods import consensus_builder
from pref_voting.probabilistic_methods import maximal_lottery, RaDiUS
from pref_voting.grade_profiles import GradeProfile
from networkx import topological_sort, is_directed_acyclic_graph
import math
import logging

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


logger = logging.getLogger("RGCR")

@vm(name="Randomized Grade Calibrated Ranking")
def RGCR(gprofile:GradeProfile, w=(lambda x: x/(1+x)), curr_cands=None):
    
    """
    An implementation of the cardinal ranking estimator proposed by Wang and Shah (2018) in https://arxiv.org/abs/1806.05085.
    by Avital Zar, 2026-04-21
    
    Args:
        gprofile: A profile of linear orders with associated cardinal scores (a GProfile).
        curr_cands: A list of candidates to consider. Defaults to all candidates if not provided.
        
    Returns:
        A sorted list of candidates.

    .. code block:: python
        # Example usage:
        from pref_voting.grade_profiles import GradeProfile
        from pref_voting.stochastic_methods import RGCR

        # Create a GProfile with 2 voters and 3 candidates
        gprofile = GradeProfile([{1: 4, 2: 8}, {2: 6, 3: 2}], range(0, 10), candidates=[1, 2, 3])
        
        # Get the ranking using RGCR
        ranking = RGCR(gprofile)
        print(ranking)
        # Output should be either [1, 0, 2] or [1, 2, 0], with higher probability for [1, 0, 2].
    """

    #TODO: check w
    
    
    candidates = curr_cands if curr_cands is not None else gprofile.candidates
    logger.info("Starting RGCR with candidates: %s", candidates)
    
    # This part isn't in the paper, the contrary - the paper says that ties broken is in order of the indices of the items.
    # However, such an arrangement creates a large bias in favor of the given order of candidates, which hurts the probability.
    # Naturally, I preferred to fix the algorithm rather than change all my probability calculations.
    gmap = [g.mapping for g in gprofile._grades]
    random.shuffle(gmap)
    Y = GradeProfile(gmap, gprofile.grades) # Create a copy of the evaluations to avoid modifying the original one.
    B = Y.to_ranking_profile() # The ordinaly ranking
    GB = B.majority_graph().to_networkx() # The graph g(B) which represent the ordinal ranking.
    GB.add_nodes_from(candidates) # Add any candidates that might not be in the majority graph (e.g. candidates that no one graded).
    if not is_directed_acyclic_graph(GB): # Then someone ranked a higher-ranked item lower, in contrast to the paper's assumption.
        logger.error("Cycle detected in majority graph - RGCR assumes a DAG.")
        raise   ValueError("As the algorithm assumes, there can't be cycles in voting order.")
    ordering = list(topological_sort(GB)) # Maybe the ties break could be more efficient
    ordering = [c for c in ordering if c in set(candidates)] # Remove candidates not in curr_cands
    logger.debug("Initial topological ordering: %s", ordering)

    def _our_can(tuple):
        # Helper random function which get two scores and return true if the first score probablistically beats the second.
        prob = (1+w(abs(tuple[0]-tuple[1])))/2 # The probanility that the higher-ranked item is really better.
        result = random.random() < prob # That is, if the first one is bigger then in probability prob we return true - the first beated the second.
        if tuple[0] < tuple[1]: # If the second one is bigger, then in probability 1-prob we return true because in probability 1-prob the first beats the second.
            result = not result

        logger.debug("our_can: scores %s, prob %.4f -> flip: %g", tuple, round(prob, 4), result)
        return result

    def _find_reviewer(item):
        # Helper function which finds a random voter who graded the given item.
        reviewer = None
        for voter in Y._grades:
            if voter.has_grade(item) and voter.val(item) is not None:
                reviewer = voter
                break
        return reviewer

    
    t = 0
    while(t < len(ordering)-1):
        t_th_item = ordering[t]
        t_plus_1_th_item = ordering[t+1]

        logger.debug("Checking pair: (%s, %s) at index %g", t_th_item, t_plus_1_th_item, t)

        t_reviewer = _find_reviewer(t_th_item)
        t_plus_1_reviewer = _find_reviewer(t_plus_1_th_item)
        # If the flipping of t and t+1 isn't a topological order, means there's no one who ranked t above t+1
        # Also if there's a reviewing for both items
        # Otherwise we continue
        if t_reviewer and t_plus_1_reviewer and not (B.majority_prefers(t_th_item, t_plus_1_th_item)): # The order is opposite (compared to the paper) because if there's no reviewer then the item doesn't exist at all in B.
                t_score = t_reviewer.val(t_th_item)
                t_plus_1_score = t_plus_1_reviewer.val(t_plus_1_th_item)

                logger.debug("Pair satisfies flip conditions. Scores: %s=%g, %s=%g", t_th_item, t_score, t_plus_1_th_item, t_plus_1_score)

                Y._grades.remove(t_reviewer)
                if(t_plus_1_reviewer in Y._grades): # In case we choose the same reviewer for both items.
                    Y._grades.remove(t_plus_1_reviewer)
                if(_our_can((t_plus_1_score, t_score))): # If the second item ranked higher, the we flip them.
                    logger.info("Flipping %s and %s", t_th_item, t_plus_1_th_item)
                    ordering[t], ordering[t+1] = ordering[t+1], ordering[t]
                t = t+2
        else:
            if not t_reviewer or not t_plus_1_reviewer:
                logger.debug("Skipping pair: missing reviewers.")
            else:
                logger.debug("Skipping pair: There's a reviewer prefers %s over %s", t_th_item, t_plus_1_th_item)
            t=t+1
    
    logger.info("Final RGCR ranking: %s", ordering)
    return ordering