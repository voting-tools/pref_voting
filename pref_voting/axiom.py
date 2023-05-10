"""
    File: axiom.py
    Author: Eric Pacuit (epacuit@umd.edu)
    Date: April 27, 2023
    
    Define the Axiom class. 
"""

class Axiom(object): 
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
