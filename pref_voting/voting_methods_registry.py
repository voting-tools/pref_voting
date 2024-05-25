'''
    File: voting_methods_registry.py
    Author: Wes Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)
    Date: April 29, 2024
    
'''

from pref_voting.voting_method import VotingMethod
import inspect

class VotingMethodRegistry:
    def __init__(self):
        self.methods = {}

    def register(self, 
                 method, 
                 name, 
                 method_type, 
                 file_location, 
                 election_types, 
                 properties):
        self.methods[name] = {
            "method": method,
            "method_type": method_type,
            "file_location": file_location,
            "election_types": election_types,
            "properties": properties
        }

    def discover_methods(self, module):
        """Discovers and registers all VotingMethod instances in the given module."""

        fname_to_method_type = {
            "scoring_methods.py": "Scoring Rule",
            "iterative_methods.py": "Iterative Method",
            "c1_methods.py": "C1 Method",
            "margin_based_methods.py": "Margin Based Method",
            "other_methods.py": "Other Method",
            "combined_methods.py": "Combined Method",
        }
        for name, obj in inspect.getmembers(module, lambda member: isinstance(member, VotingMethod)):
            if hasattr(obj, 'name') and hasattr(obj, 'properties'):

                if getattr(obj, 'skip_registration', False):
                    continue

                self.register(obj, 
                              obj.name, 
                              fname_to_method_type[module.__file__.split("/")[-1]],
                              module.__file__.split("/")[-1], 
                              getattr(obj, 'input_types', []), 
                              getattr(obj, 'properties', {}))

    def list_methods(self):
        return list(self.methods.keys())

    def display_methods(self):
        for name, details in self.methods.items():
            print(f"{name} ({details['method_type']})")
            print(f"Satisfied properties: {details['properties'].satisfied() if details['properties'] is not None else 'N/A'}")
            print(f"Violated properties: {details['properties'].violated() if details['properties'] is not None else 'N/A'}")
            print()

    def get_methods(self, method_type): 
        return [self.methods[vm]['method'] for vm in self.methods if self.methods[vm]['method_type'] == method_type]

    def filter(self, 
               satisfies=None, 
               violates=None, 
               unknown=None, 
               election_types=None):
        
        if satisfies is None:
            satisfies = []
        if violates is None:
            violates = []
        if unknown is None:
            unknown = []
        if election_types is None:
            election_types = []

        found_methods = []
        for name, method_info in self.methods.items():
            properties = method_info['properties']
            method_election_types = method_info['election_types']

            # Check if the method satisfies the required properties
            if not all(getattr(properties, prop, None) is True for prop in satisfies):
                continue

            # Check if the method violates the required properties
            if not all(getattr(properties, prop, None) is False for prop in violates):
                continue

            # Check for unknown properties
            if not all(hasattr(properties, prop) and getattr(properties, prop, None) is None for prop in unknown):
                continue

            # Check if the method supports all required election types
            if not all(et in method_election_types for et in election_types):
                continue

            # If all conditions are met, add to results
            found_methods.append(method_info['method'])

        return found_methods

    def method_type(self, method_name):
        return self.methods[method_name]['method_type']
    def file_location(self, method_name):
        return self.methods[method_name]['file_location']
    
    def __len__(self):
        return len(self.methods)
    
    def __iter__(self):
        self._iter = iter(method_details['method'] for method_details in self.methods.values())
        return self

    def __next__(self):
        return next(self._iter)

voting_methods = VotingMethodRegistry()

import pref_voting.scoring_methods 
voting_methods.discover_methods(pref_voting.scoring_methods)
import pref_voting.iterative_methods 
voting_methods.discover_methods(pref_voting.iterative_methods)
import pref_voting.c1_methods 
voting_methods.discover_methods(pref_voting.c1_methods)
import pref_voting.margin_based_methods 
voting_methods.discover_methods(pref_voting.margin_based_methods)
import pref_voting.combined_methods 
voting_methods.discover_methods(pref_voting.combined_methods)
import pref_voting.other_methods 
voting_methods.discover_methods(pref_voting.other_methods)
