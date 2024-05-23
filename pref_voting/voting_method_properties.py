'''
    File: voting_methods.py
    Author: Wes Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)
    Date: November 6, 2021
    Update: January 15, 2023
    
    The VotingMethodProperties class to encapsulate properties of voting methods.
'''
from enum import Enum

class ElectionTypes(Enum): 
    PROFILE = "Profile"
    PROFILE_WITH_TIES = "ProfileWithTies"
    TRUNCATED_LINEAR_PROFILE = "ProfileWithTies"
    MAJORITY_GRAPH = "MajorityGraph"
    MARGIN_GRAPH = "MarginGraph"

class VotingMethodProperties:
    """
    Class to encapsulate properties of voting methods.

    Attributes:
        condorcet_consistent (bool): Indicates if the voting method always elects a Condorcet winner when one exists.
    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def satisfied(self):
        return [key for key, value in self.__dict__.items() if value==True]
    
    def violated(self):
        return [key for key, value in self.__dict__.items() if value==False]

    def __str__(self):
        properties = [f"{key}: {'Satisfied' if value==True else ('Violated' if value==False else 'N/A')}" for key, value in self.__dict__.items()]
        return "\n".join(properties)
