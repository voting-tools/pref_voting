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
    TRUNCATED_LINEAR_PROFILE = "TruncatedLinearProfile"
    MAJORITY_GRAPH = "MajorityGraph"
    MARGIN_GRAPH = "MarginGraph"

class VotingMethodProperties:
    """
    Class to represent the properties of a voting method.
    """
    def __init__(self, **kwargs):
        """
        Initialize the properties of the voting method.

        Args:
            **kwargs: Arbitrary keyword arguments representing properties and their boolean values.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    def items(self):
        """
        Return all properties as a list of items.

        Returns:
            List of all properties and their boolean values.
        """
        return self.__dict__.items()

    def satisfied(self):
        """
        List all properties that are satisfied (i.e., True).

        Returns:
            List of property names that are satisfied.
        """
        return [key for key, value in self.items() if value is True]
    
    def violated(self):
        """
        List all properties that are violated (i.e., False).

        Returns:
            List of property names that are violated.
        """
        return [key for key, value in self.items() if value is False]

    def not_available(self):
        """
        List all properties that are not available (i.e., None).

        Returns:
            List of property names that are None (currently not available).
        """
        return [key for key, value in self.items() if value is None]

    
    def __getitem__(self, prop):
        """
        Get the value of a property.

        Args:
            prop: The name of the property.

        Returns:
            The value of the property, or None if the property does not exist.
        """
        return self.__dict__.get(prop, None)

    def __str__(self) -> str:
        """
        Return a string representation of all properties.

        Returns:
            A string representation of all properties with their status (Satisfied, Violated, or N/A).
        """
        properties = [
            f"{key}: {'Satisfied' if value is True else 'Violated' if value is False else 'N/A'}"
            for key, value in self.__dict__.items()
        ]
        return "\n".join(properties)
