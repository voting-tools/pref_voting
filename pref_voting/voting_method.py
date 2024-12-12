'''
    File: voting_methods.py
    Author: Wes Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)
    Date: November 6, 2021
    Update: January 15, 2023
    
    The VotingMethod class and helper functions for voting methods
'''

import functools
import inspect
import numpy as np
from numba import jit # Remove until numba supports python 3.11
import random
import json
from pref_voting.voting_method_properties import VotingMethodProperties
from filelock import FileLock, Timeout
import importlib.resources

import glob
import os

class VotingMethod(object): 
    """
    A class to add functionality to voting methods. 

    Args:
        vm (function): An implementation of a voting method. The function should accept a Profile, ProfileWithTies, MajorityGraph, and/or MarginGraph, and a keyword parameter ``curr_cands`` to find the winner after restricting to ``curr_cands``. 
        name (string): The Human-readable name of the voting method.
        properties (VotingMethodProperties): The properties of the voting method.
        input_types (list): The types of input that the voting method can accept.

    """
    def __init__(self, 
                 vm, 
                 name=None, 
                 input_types=None, 
                 skip_registration=False,                properties_file=None): 
        
        self.vm = vm
        self.name = name

        # Determine the path to the properties file
        if properties_file is None:

            properties_file = importlib.resources.files('pref_voting') / 'data' / 'voting_methods_properties.json'


        # Get the properties of the voting method
        try:
            with open(properties_file, "r") as file:
                vm_props = json.load(file)
        except FileNotFoundError:
            vm_props = {}
        except Exception as e:
            print(f"An error occurred while opening the properties file: {e}")
            vm_props = {}

        if name in vm_props:
            properties = VotingMethodProperties(**vm_props[name])
        else:
            properties = VotingMethodProperties()

        self.properties = properties
        self.input_types = input_types
        self.skip_registration = skip_registration
        self.algorithm = None

        functools.update_wrapper(self, vm)   

    def __call__(self, edata, curr_cands = None, **kwargs):
        
        if (curr_cands is not None and len(curr_cands) == 0) or len(edata.candidates) == 0: 
            return []
        
        # Set the algorithm from self.algorithm if it's not already provided in kwargs
        if 'algorithm' not in kwargs and self.algorithm is not None:
            params = inspect.signature(self.vm).parameters
            if 'algorithm' in params and params['algorithm'].kind in [inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD]:
                kwargs['algorithm'] = self.algorithm

        return self.vm(edata, curr_cands=curr_cands, **kwargs)

    def set_algorithm(self, algorithm):
        """
        Set the algorithm for the voting method if 'algorithm' is an accepted keyword parameter.

        Args:
            algorithm: The algorithm to set for the voting method.
        """
        params = inspect.signature(self.vm).parameters
        if 'algorithm' in params and params['algorithm'].kind in [inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD]:
            self.algorithm = algorithm
        else:
            raise ValueError(f"The method {self.name} does not accept 'algorithm' as a parameter.")
    
    def choose(self, edata, curr_cands = None): 
        """
        Return a randomly chosen element from the winning set. 
        """

        ws = self.__call__(edata, curr_cands = curr_cands)
        return random.choice(ws)
    
    def prob(self, edata, curr_cands = None): 
        """
        Return a dictionary representing the even-chance tiebreaking for the voting method.
        """
        
        ws = self.__call__(edata, curr_cands = curr_cands)
        return {c: 1.0 / len(ws) if c in ws else 0.0 for c in edata.candidates}
    
    def display(self, edata, curr_cands = None, cmap = None, **kwargs): 
        """
        Display the winning set of candidates.
        """

        cmap = cmap if cmap is not None else edata.cmap

        ws = self.__call__(edata, curr_cands = curr_cands, **kwargs)

        if ws is None:  # some voting methods, such as ``ranked_pairs_with_test``, may return None if it is taking long to compute the winner.
            print(f"{self.name} winning set is not available")
        else: 
            w_str = f"{self.name} winner is " if len(ws) == 1 else f"{self.name} winners are "
            print(w_str + "{" + ", ".join([str(cmap[c]) for c in ws]) + "}")
        
    def set_name(self, new_name):
        """Set the name of the voting method."""

        self.name = new_name

    def add_property(self, prop, value):
        """Add a property to the voting method."""

        setattr(self.properties, prop, value)

    def remove_property(self, prop):
        """Remove a property from the voting method."""

        delattr(self.properties, prop)

    def load_properties(self, filename=None):
        """Load the properties of the voting method from a JSON file."""
        
        # Determine the path to the properties file
        if filename is None:
            filename = importlib.resources.files('pref_voting') / 'data' / 'voting_methods_properties.json'
        lock = FileLock(f"{filename}.lock")
        with lock:
            try:
                with open(filename, 'r') as file:
                    vm_props = json.load(file)
            except FileNotFoundError:
                vm_props = {}

            if self.name in vm_props:
                self.properties = VotingMethodProperties(**vm_props[self.name])
            else:
                self.properties = VotingMethodProperties()

    def has_property(self, prop):
        """Check if the voting method has a property."""

        return self.properties[prop]
    
    def get_properties(self): 
        """Return the properties of the voting method."""
        
        return {
            "satisfied": [prop 
                          for prop, val in self.properties.items() 
                          if val is True],
            "violated": [prop 
                         for prop, val in self.properties.items() 
                         if val is False],
            "na": [prop 
                   for prop, val in self.properties.items() 
                   if val is None]
                }

    def save_properties(self, filename=None, timeout=10):
        """Save the properties of the voting method to a JSON file."""

        # Determine the path to the properties file
        if filename is None:
            filename = importlib.resources.files('pref_voting') / 'data' / 'voting_methods_properties.json'


        lock = FileLock(f"{filename}.lock", timeout=timeout)
        try:
            with lock:
                try:
                    with open(filename, 'r') as file:
                        vm_props = json.load(file)
                except FileNotFoundError:
                    vm_props = {}

                vm_props[self.name] = self.properties.__dict__

                with open(filename, 'w') as file:
                    json.dump(vm_props, file, indent=4, sort_keys=True)
        except Timeout:
            print(f"Could not acquire the lock within {timeout} seconds.")

    def get_violation_witness(self, prop, minimal_resolute=False, minimal=False):
        """Return the election that witnesses a violation of prop."""

        from pref_voting.profiles import Profile

        elections = {
            "minimal resolute": None,
            "minimal": None,
            "any": None
        }
        if self.properties[prop]:
            print(f"{self.name} satisfies {prop}, no election returned.")
            return elections
        elif self.properties[prop] is None:
            print(f"{self.name} does not have a value for {prop}, no election returned.")
            return elections
        else:
            dir = importlib.resources.files('pref_voting') / 'data' / 'examples' / prop

            for f in glob.glob(f"{dir}*"):
                fname = os.path.basename(f)
                is_min = fname.startswith("minimal_")
                is_min_resolute = fname.startswith("minimal_resolute")
                found_it = False
                if is_min_resolute and fname.startswith(f"minimal_resolute_{self.name.replace(' ', '_')}"): 
                    print(f"Minimal resolute election for a violation of {prop} found.")
                    elections["minimal resolute"] = Profile.from_preflib(f)
                if is_min and not is_min_resolute and fname.startswith(f"minimal_{self.name.replace(' ', '_')}"):
                    print(f"Minimal election for a violation of {prop} found.")
                    elections["minimal"] = Profile.from_preflib(f)

                elif not is_min and not is_min_resolute and  fname.startswith(f"{self.name.replace(' ', '_')}"):
                    elections["any"] = Profile.from_preflib(f)
            if all([v is None for v in elections.values()]):
                print(f"No election found illustrating the violation of {prop}.")
            return elections

    def check_property(self, prop, include_counterexample=True): 
        """Check if the voting method satisfies a property."""
        from pref_voting.axioms import axioms_dict

        if not self.properties[prop]: 
            print(f"{self.name} does not satisfy {prop}")
            if include_counterexample:
                if prop in axioms_dict:
                    #prof = prof
                    axioms_dict[prop].counterexample(self)

        elif self.properties[prop] is None: 
            print(f"{self.name} does not have a value for {prop}")
        
        else:
            print(f"{self.name} satisfies {prop}")

    def __str__(self): 
        return f"{self.name}"

def vm(name=None, 
       input_types=None, 
       skip_registration=False):
    """
    A decorator used when creating a voting method. 
    """
    def wrapper(f):
        return VotingMethod(f, 
                            name=name,
                            input_types=input_types, 
                            skip_registration=skip_registration)
    return wrapper

@jit(nopython=True, fastmath=True)
def isin(arr, val):
    """compiled function testing if the value val is in the array arr
    """
    
    for i in range(arr.shape[0]):
        if (arr[i]==val):
            return True
    return False

@jit(nopython=True, fastmath=True)
def _num_rank_first(rankings, rcounts, cands_to_ignore, cand):
    """The number of voters that rank candidate cand first after ignoring the candidates in 
    cands_to_ignore
    
    Parameters
    ----------
    rankings:  2d numpy array
        list of linear orderings of the candidates  
    rcounts:  1d numpy array
        list of numbers of voters with the rankings  
    cands_to_ignore:   1d numpy array
        list of the candidates to ignore
    cand: int
        a candidate
    
    Key assumptions: 
        * the candidates are named 0...num_cands - 1, and c1 and c2 are 
          numbers between 0 and num_cands - 1
        * voters submit linear orders over the candidate        
    """
    
    num_voters = len(rankings)
    
    top_cands_indices = np.zeros(num_voters, dtype=np.int32)
    
    for vidx in range(num_voters): 
        for level in range(0, len(rankings[vidx])):
            if not isin(cands_to_ignore, rankings[vidx][level]):
                top_cands_indices[vidx] = level
                break                
    top_cands = np.array([rankings[vidx][top_cands_indices[vidx]] for vidx in range(num_voters)])
    is_cand = top_cands == cand # set to 0 each candidate not equal to cand
    return np.sum(is_cand * rcounts) 


@jit(nopython=True, fastmath=True)
def _num_rank_last(rankings, rcounts, cands_to_ignore, cand):
    """The number of voters that rank candidate cand last after ignoring the candidates in 
    cands_to_ignore
    
    Parameters
    ----------
    rankings:  2d numpy array
        list of linear orderings of the candidates  
    rcounts:  1d numpy array
        list of numbers of voters with the rankings  
    cands_to_ignore:   1d numpy array
        list of the candidates to ignore
    cand: int
        a candidate
    
    Key assumptions: 
        * the candidates are named 0...num_cands - 1, and c1 and c2 are 
          numbers between 0 and num_cands - 1
        * voters submit linear orders over the candidate        
    """
    
    num_voters = len(rankings)
    
    last_cands_indices = np.zeros(num_voters, dtype=np.int32)
    
    for vidx in range(num_voters): 
        for level in range(len(rankings[vidx]) - 1,-1,-1):
            if not isin(cands_to_ignore, rankings[vidx][level]):
                last_cands_indices[vidx] = level
                break                
    bottom_cands = np.array([rankings[vidx][last_cands_indices[vidx]] for vidx in range(num_voters)])
    is_cand = bottom_cands  == cand
    return np.sum(is_cand * rcounts) 
