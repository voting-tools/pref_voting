
'''
    File: generate_utility_profiles.py
    Author: Wes Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)
    Date: May 26, 2023
    
    Functions to generate utility profiles.
'''


from math import ceil
import numpy as np
from scipy.spatial import distance
from functools import partial

from pref_voting.utility_profiles import UtilityProfile
from pref_voting.utility_functions import *

# turn off future warnings.
# getting the following warning when calling tabulate to display a profile: 
# /Library/Frameworks/Python.framework/Versions/3.8/lib/python3.8/site-packages/tabulate.py:1027: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
#  if headers == "keys" and not rows:
# see https://stackoverflow.com/questions/40659212/futurewarning-elementwise-comparison-failed-returning-scalar-but-in-the-futur
#
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def generate_utility_profile_uniform(num_candidates, num_voters, num_profiles = 1):
    """
    Generate a utility profile where each voter assigns a random number between 0 and 1 to each candidate.

    Args:   
        num_candidates (int): The number of candidates.
        num_voters (int): The number of voters.
    
    Returns:
        UtilityProfile: A utility profile.

    """

    cand_utils = np.random.uniform(size=(num_profiles, num_voters, num_candidates))

    uprofs = [UtilityProfile([{c: cand_utils[pidx][v][c] 
                               for c in range(num_candidates)} 
                               for v in range(num_voters)]) 
                               for pidx in range(num_profiles)]
    
    return uprofs if num_profiles > 1 else uprofs[0]

def generate_utility_profile_normal(num_candidates, num_voters, std = 0.1, normalize = None, num_profiles = 1):
    """
    Generate a utility profile where each voter assigns a random number drawn from a normal distribution with a randomly chosen mean (between 0 and 1) with standard deviation ``std`` to each candidate.   
    
    Args:
        num_candidates (int): The number of candidates.
        num_voters (int): The number of voters. 
        std (float): The standard deviation of the normal distribution. The default is 0.1.
        normalize (str): The normalization method to use. The default is None.
    
    Returns:
        UtilityProfile: A utility profile.
    """
    
    mean_utilities = {c: np.random.uniform(0, 1) for c in range(num_candidates)}
    cand_utils = {c: np.random.normal(mean_utilities[c], std, size=(num_profiles, num_voters)) for c in range(num_candidates)}
    
    if normalize == "range": 
        uprofs = [UtilityProfile([{c: cand_utils[c][pidx][vidx] 
                                   for c in range(num_candidates)} 
                                   for vidx in range(num_voters)]).normalize_by_range() 
                                   for pidx in range(num_profiles)]
    elif normalize == "score":
        uprofs = [UtilityProfile([{c: cand_utils[c][pidx][vidx] 
                                   for c in range(num_candidates)} 
                                   for vidx in range(num_voters)]).normalize_by_standard_score() 
                                   for pidx in range(num_profiles)]
    else: # do not normalize
        uprofs = [UtilityProfile([{c: cand_utils[c][pidx][vidx] 
                                   for c in range(num_candidates)} 
                                   for vidx in range(num_voters)]) 
                                   for pidx in range(num_profiles)]


    return uprofs if num_profiles > 1 else uprofs[0]
utility_functions = {
    "RM": {
        "func": mixed_rm_utility,
        "param": 1
    },
    "Linear": {
        "func": linear_utility,
        "param": None
    },
    "Quadratic": 
    {   
        "func": quadratic_utility,
        "param": None
    },
    "Shepsle": {
        "func": shepsle_utility,
        "param": None
    },
    "City Block": { 
        "func": city_block_utility,
        "param": None
    },
    "Matthews": { 
        "func": matthews_utility,
        "param": None
    }

}
def generate_spatial_utility_profile(num_cands, 
                                     num_voters, 
                                     num_dims = 2, 
                                     utility_function = "Quadratic", 
                                     utility_function_param = None):
    
    """
    Create a spatial utility profile using specified utility functions. 


    Args:
        num_cands (int): The number of candidates.
        num_voters (int): The number of voters.
        num_dims (int): The number of dimensions. The default is 2.
        utility_function (str): The utility function to use. The default is "Linear".
        utility_function_param (float): The parameter of the utility function. The default is None.
        
    Returns:
        UtilityProfile: A spatial utility profile.
    """

    # the first component of the parameter is the number of dimensions, 
    # the second component is used to define the mixed model: 
    # beta = 1 is proximity model (i.e., squared Euclidean distance)

    mean = [0] * num_dims  # mean is 0 for each dimension
    cov = np.diag([1] * num_dims)  # diagonal covariance

    _utility_fnc = utility_functions[utility_function]["func"]

    if utility_functions[utility_function]["param"] is not None or utility_function_param is not None: 
        util_parm = utility_function_param  if utility_function_param is not None else utility_functions[utility_function]["param"]
        utility_fnc = partial(_utility_fnc, util_parm)

    else: 
        utility_fnc = _utility_fnc

    # sample candidate/voter positions using a multivariate normal distribution
    cand_positions = np.random.multivariate_normal(np.array(mean), cov, num_cands)
    voter_positions = np.random.multivariate_normal(np.array(mean), cov, num_voters)

    utilities = [{c: utility_fnc(v_pos, c_pos)  for c, c_pos in enumerate(cand_positions)} 
                 for _, v_pos in enumerate(voter_positions)]

    return UtilityProfile(utilities)


