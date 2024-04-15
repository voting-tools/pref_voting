'''
    File: voting_methods.py
    Author: Wes Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)
    Date: June 3, 2023
    
    Implementations of probabilistic voting methods.
'''

from pref_voting.prob_voting_method import  *
from pref_voting.weighted_majority_graphs import  MajorityGraph, MarginGraph
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

