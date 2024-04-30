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

    def __iter__(self):
        self._iter = iter(method_details['method'] for method_details in self.methods.values())
        return self

    def __next__(self):
        return next(self._iter)
    
import pref_voting.scoring_methods 
voting_methods = VotingMethodRegistry()
voting_methods.discover_methods(pref_voting.scoring_methods)
