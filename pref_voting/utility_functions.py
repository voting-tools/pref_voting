
import numpy as np
from scipy.spatial import distance
from numba import jit, float32 

@jit(nopython=True, fastmath=True)
def mixed_rm_utility(v_pos: float32[:], c_pos: float32[:], beta = 0.5):
    """Based on the Rabinowitz and Macdonald (1989) mixed model described on pages 43-44 of "A Unified Theory of Voting" by S. Merrill III and B. Grofman.

    beta = 1 is the proximity quadratic utility function
    beta = 0 is the RM directional utility function

    Args:
        v_pos (numpy array): The position(s) of the voter.
        c_pos (numpy array): The position(s) of the candidate.
        beta (float): The beta parameter of the mixed model.
    
    Returns:
        float: The utility of the candidate to the voter.
    """
    return 2 * (1-beta) * np.dot(v_pos, c_pos) - beta * np.linalg.norm(v_pos - c_pos) ** 2 

def rm_utility(v_pos: float32[:], c_pos: float32[:]): 
    """Based on the Rabinowitz and Macdonald (1989) pure directional model. See "A Unified Theory of Voting" by S. Merrill III and B. Grofman, pg. 31.  

    Args:
        v_pos (numpy array): The position(s) of the voter.
        c_pos (numpy array): The position(s) of the candidate.

    Returns:
        float: The utility of the candidate to the voter.
    """

    return np.dot(v_pos, c_pos)

def linear_utility(v_pos: float32[:], c_pos: float32[:]):
    """
    The utility of the candidate for the voter is negative of the Euclidean distance between the positions. 

    Args:   
        v_pos (numpy array): The position(s) of the voter.
        c_pos (numpy array): The position(s) of the candidate.  
    Returns:
        float: The utility of the candidate to the voter.
    """
    return -np.linalg.norm(v_pos - c_pos)

def quadratic_utility(v_pos: float32[:], c_pos: float32[:]):
    """ 
    The utility of the candidate for the voter is negative of the squared Euclidean distance between the positions. 

    Args:
        v_pos (numpy array): The position(s) of the voter.
        c_pos (numpy array): The position(s) of the candidate.
    Returns:
        float: The utility of the candidate to the voter.
    """
    return -np.linalg.norm(v_pos - c_pos)**2


def city_block_utility(v_pos: float32[:], c_pos: float32[:]):
    """
    The utility of the candidate for the voter is the negative of the city-block distance between the positions (also known as the Manhattan distance). 

    Args:
        v_pos (numpy array): The position(s) of the voter.
        c_pos (numpy array): The position(s) of the candidate.
    Returns:
        float: The utility of the candidate to the voter.
    """
    return -distance.cityblock(v_pos, c_pos)

@jit(nopython=True, fastmath=True)
def shepsle_utility(v_pos: float32[:], c_pos: float32[:], kappa: float32 = 1):
    """
    The Shepsle utility function from "The Strategy of Ambiguity: Uncertainty and Electoral Competition" by Kenneth A. Shepsle, American Political Science Review, 1972, vol. 66, issue 2, pp. 555-568.   For a justification of this utility function, see Appendix B from *Making Multicandidate Elections More Democratic* (https://doi.org/10.1515/9781400859504.114) by S. Merrill III. 

    Args:
        v_pos (numpy array): The position(s) of the voter.
        c_pos (numpy array): The position(s) of the candidate. 
        kappa (float): A parameter that determines the steepness of the utility function. 
    Returns:    
        float: The utility of the candidate to the voter.      
    """
    d = np.linalg.norm(v_pos - c_pos)
    return np.exp((kappa**2 * -d**2) / 2)


@jit(nopython=True, fastmath=True)
def matthews_utility(v_pos: float32[:], c_pos: float32[:]):
    """
    Based on the Matthews directional model.  See "A Unified Theory of Voting" by S. Merrill III and B. Grofman, pg. 26.
    
    Args:
        v_pos (numpy array): The position(s) of the voter.
        c_pos (numpy array): The position(s) of the candidate.  
    Returns:    
        float: The utility of the candidate to the voter.
        
    """
    return np.dot(v_pos, c_pos) / (np.linalg.norm(v_pos) * np.linalg.norm(c_pos))

