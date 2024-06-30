'''
    File: social_welfare_function.py
    Author: Wes Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)
    Date: February 6, 2024
    
    The SWF class and helper functions for social welfare functions
'''

import functools

class SocialWelfareFunction(object): 
    """
    A class to add functionality to social welfare functions 

    Args:
        swf (function): An implementation of a voting method. The function should accept any type of profile, and a keyword parameter ``curr_cands`` to find the winner after restricting to ``curr_cands``. 
        name (string): The Human-readable name of the social welfare function.

    Returns:
        A ranking (Ranking) of the candidates.
    """
    def __init__(self, swf, name = None): 
        
        self.swf = swf
        self.name = name
        functools.update_wrapper(self, swf)   

    def __call__(self, edata, curr_cands = None, **kwargs):

        if (curr_cands is not None and len(curr_cands) == 0) or len(edata.candidates) == 0: 
            return []
        return self.swf(edata, curr_cands = curr_cands, **kwargs)
        
    def winners(self, edata, curr_cands = None, **kwargs):
        """Return a sorted list of the first place candidates."""

        return sorted(self.swf(edata, curr_cands = curr_cands, **kwargs).first())
    
    def display(self, edata, curr_cands = None, **kwargs):
        """Display the result of the social welfare function."""

        ranking = self.swf(edata, curr_cands = curr_cands, **kwargs)
        print(f"{self.name} ranking is {ranking}")

    def set_name(self, new_name):
        """Set the name of the social welfare function."""

        self.name = new_name

    def __str__(self): 
        return f"{self.name}"

def swf(name = None):
    """
    A decorator used when creating a social welfare function. 
    """
    def wrapper(f):
        return SocialWelfareFunction(f, name=name)
    return wrapper
