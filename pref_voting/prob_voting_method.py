'''
    File: prob_voting_method.py
    Author: Wes Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)
    Date: April 14, 2024
    
    The ProbabilisticVotingMethod class and helper functions for probabilistic voting methods.
'''

import functools
import numpy as np
import inspect

class ProbVotingMethod(object): 
    """
    A class to add functionality to probabilistic voting methods 

    Args:
        pvm (function): An implementation of a probabilistic voting method. The function should accept any type of profile, and a keyword parameter ``curr_cands`` to find the winner after restricting to ``curr_cands``. 
        name (string): The human-readable name of the social welfare function.

    Returns:
        A dictionary that represents the probability on the set of candidates.

    """
    def __init__(self, pvm, name = None): 
        
        self.pvm = pvm
        self.name = name
        self.algorithm = None

        functools.update_wrapper(self, pvm)   

    def __call__(self, edata, curr_cands = None, **kwargs):

        if (curr_cands is not None and len(curr_cands) == 0) or len(edata.candidates) == 0: 
            return {}
        return self.pvm(edata, curr_cands = curr_cands, **kwargs)
        
    def support(self, edata, curr_cands = None, **kwargs): 
        """
        Return the sorted list of the set of candidates that have non-zero probability. 
        """

        if (curr_cands is not None and len(curr_cands) == 0) or len(edata.candidates) == 0: 
            return []
        prob =  self.pvm(edata, curr_cands = curr_cands, **kwargs)
        return sorted([c for c,pr in prob.items() if pr > 0])
    
    def choose(self, edata, curr_cands = None, **kwargs): 
        """
        Return a randomly chosen element according to the probability. 
        """

        prob =  self.pvm(edata, curr_cands = curr_cands, **kwargs)
        
        # choose a candidate according to the probability distribution prob
        cands = list(prob.keys())
        probs = [prob[c] for c in cands]

        return np.random.choice(cands, p = probs)
    
    def display(self, edata, curr_cands = None, cmap = None, **kwargs): 
        """
        Display the winning set of candidates.
        """
 
        cmap = cmap if cmap is not None else edata.cmap

        prob = self.__call__(edata, curr_cands = curr_cands, **kwargs)

        if prob is None:  # some voting methods may return None if, for instance, it is taking long to compute the winner.
            print(f"{self.name} probability is not available")
        else: 
            w_str = f"{self.name} probability is " 
            print(w_str + "{" + ", ".join([f"{str(cmap[c])}: {round(pr,3)}" for c,pr in prob.items()]) + "}")
        

    def set_name(self, new_name):
        """Set the name of the social welfare function."""

        self.name = new_name

    def set_algorithm(self, algorithm):
        """
        Set the algorithm for the voting method if 'algorithm' is an accepted keyword parameter.

        Args:
            algorithm: The algorithm to set for the voting method.
        """
        params = inspect.signature(self.pvm).parameters
        if 'algorithm' in params and params['algorithm'].kind in [inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD]:
            self.algorithm = algorithm
        else:
            raise ValueError(f"The method {self.name} does not accept 'algorithm' as a parameter.")

    def __str__(self): 
        return f"{self.name}"

def pvm(name = None):
    """
    A decorator used when creating a social welfare function. 
    """
    def wrapper(f):
        return ProbVotingMethod(f, name=name)
    return wrapper
