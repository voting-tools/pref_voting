
'''
    File: generate_utility_profiles.py
    Author: Eric Pacuit (epacuit@umd.edu)
    Date: May 26, 2023
    
    Functions to generate utility profiles.
'''


from math import ceil
import numpy as np
from pref_voting.utility_profiles import UtilityProfile

# turn off future warnings.
# getting the following warning when calling tabulate to display a profile: 
# /Library/Frameworks/Python.framework/Versions/3.8/lib/python3.8/site-packages/tabulate.py:1027: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
#  if headers == "keys" and not rows:
# see https://stackoverflow.com/questions/40659212/futurewarning-elementwise-comparison-failed-returning-scalar-but-in-the-futur
#
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def generate_utility_profile_uniform(num_candidates, num_voters):
    """
    Generate a utility profile where each voter assigns a random number between 0 and 1 to each candidate.

    Args:   
        num_candidates (int): The number of candidates.
        num_voters (int): The number of voters.
    
    Returns:
        UtilityProfile: A utility profile.

    """

    utilities = [{c: np.random.uniform() for c in range(num_candidates)} 
                 for _ in range(num_voters)]
    
    return UtilityProfile(utilities)

def generate_utility_profile_normal(num_candidates, num_voters, std = 0.1):
    """
    Generate a utility profile where each voter assigns a random number drawn from a normal distribution with a randomly chosen mean (between 0 and 1) with standard deviation ``std`` to each candidate.  Normalize the 
    
    Args:
        num_candidates (int): The number of candidates.
        num_voters (int): The number of voters. 
        std (float): The standard deviation of the normal distribution. The default is 0.1.
    
    Returns:
        UtilityProfile: A utility profile.
    """

    utilities = list()
    
    mean_utilities = {c: np.random.uniform(0, 1) for c in range(num_candidates)}
    utilities = [{c: np.random.normal(mean_utilities[c], std) for c in range(num_candidates)} 
                 for _ in range(num_voters)]
    
    return UtilityProfile(utilities).normalize()


def voter_utility(v_pos, c_pos, beta):
    """Based on the Rabinowitz and Macdonald (1989) mixed model described in Section 3, pp. 745 - 747 of
    "Voting behavior under the directional spatial model of electoral competition" by S. Merrill III.

    beta = 1 is the proximity model
    beta = 0 is the directional model

    Args:
        v_pos (numpy array): The position(s) of the voter.
        c_pos (numpy array): The position(s) of the candidate.
        beta (float): The beta parameter of the mixed model.
    
    Returns:
        float: The utility of the candidate to the voter.
    """
    return 2 * np.dot(v_pos, c_pos) - beta * (
        np.linalg.norm(v_pos) ** 2 + np.linalg.norm(c_pos) ** 2
    )


def generate_spatial_utility_profile(num_cands, num_voters, params = None):
    """
    Create a spatial utility profile using the Rabinowitz and Macdonald (1989) mixed model described in Section 3, pp. 745 - 747 of "Voting behavior under the directional spatial model of electoral competition" by S. Merrill III. 

    .. note:  When, beta = 1, it is the proximity model (i.e., utilities are the negative of the squared Euclidean distance), and when beta = 0, it is the directional model.

    Args:
        num_cands (int): The number of candidates.
        num_voters (int): The number of voters.
        params (tuple): A tuple of the form (num_dim, beta) where num_dim is the number of dimensions and beta is the beta parameter of the mixed model. The default is (2, 1).

    Returns:
        UtilityProfile: A spatial utility profile.
    """
    params = params if params is not None else (2, 1)

    # the first component of the parameter is the number of dimensions, 
    # the second component is used to define the mixed model: 
    # beta = 1 is proximity model (i.e., squared Euclidean distance)
    num_dim, beta = params

    mean = [0] * num_dim  # mean is 0 for each dimension
    cov = np.diag([1] * num_dim)  # diagonal covariance

    # sample candidate/voter positions using a multivariate normal distribution
    cand_positions = np.random.multivariate_normal(np.array(mean), cov, num_cands)
    voter_positions = np.random.multivariate_normal(np.array(mean), cov, num_voters)

    utilities = [{c: voter_utility(v_pos, c_pos, beta) for c, c_pos in enumerate(cand_positions)} 
                 for _, v_pos in enumerate(voter_positions)]

    return UtilityProfile(utilities)


