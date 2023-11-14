"""
    File: axiom.py
    Author: Wes Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)
    Date: April 27, 2023
    
    Define the Axiom class. 
"""

class Axiom(object): 
    """
    A class to represent voting method axioms. 

    Args:
        name (string): The human-readable name of the axiom.
        has_violation (function): function that returns a Boolean which is True when there is a violation of the axiom.
        find_all_violations (function): function that returns all instances of violations of the axiom.
        satisfying_vms (list): list of voting methods satisfying the axiom.
        violating_vms (list): list of voting methods violating the axiom.

    """
    def __init__(self, name, has_violation, find_all_violations):
        self.name = name
        self.has_violation = has_violation
        self.find_all_violations = find_all_violations
        self.satisfying_vms = list()
        self.violating_vms = list()
        
    def satisfies(vm): 
        return vm.name in self.satisfying_vms
        
    def violates(vm): 
        return vm.name in self.violating_vms
        
    def add_satisfying_vms(vms): 
        self.satisfying_vms += vms

    def add_violating_vms(vms): 
        self.violating_vms += vms
