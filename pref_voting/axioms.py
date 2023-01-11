"""
    File: analysis.py
    Author: Eric Pacuit (epacuit@umd.edu)
    Date: January 9, 2023
    
    Functions to determined whether an axiom is violated by a voting method 
"""


def has_condorcet_loser_violation(edata, vm, verbose=False):
    """
    Returns True if there is a Condorcet loser in edata that is a winner according to vm.  

    """

    cl = edata.condorcet_loser()

    ws = vm(edata)

    if cl is not None and cl in ws:
        if verbose: 
            print(f"The Condorcet loser {cl} is an elment of the winning set: ")
            vm.display(edata)
        return True 
    
    return False
