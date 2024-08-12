'''
    File: create_methods.py
    Author: Wes Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)
    Date: August 8, 2024
    
'''

def compose(vm1, vm2):
    """After restricting the profile to the set of vm1 winners, run vm2

    Args:
        vm1, vm2 (VotingMethod): The voting methods to be composed.

    Returns:
        A VotingMethod that composes vm1 and vm2.

    :Example:

    .. exec_code::

        from pref_voting.profiles import Profile
        from pref_voting.combined_methods import compose
        from pref_voting.scoring_methods import borda
        from pref_voting.c1_methods import copeland

        prof = Profile([[1, 3, 0, 2], [2, 1, 3, 0], [3, 0, 2, 1]], [1, 2, 1])

        prof.display()

        copeland_borda = compose(copeland, borda)

        copeland.display(prof)
        borda.display(prof)
        copeland_borda.display(prof)

    """

    def _vm(edata, curr_cands=None):

        vm1_ws = vm1(edata, curr_cands=curr_cands)

        return vm2(edata, curr_cands=vm1_ws)

    return VotingMethod(_vm, name=f"{vm1.name}-{vm2.name}")


def _compose(vm1, vm2):
    """
    Same as compose, but used to make it easier to document composed voting methods.
    """

    def _vm(edata, curr_cands=None):

        vm1_ws = vm1(edata, curr_cands=curr_cands)

        return vm2(edata, curr_cands=vm1_ws)

    return _vm

def faceoff(vm1, vm2):
    """If the vm1 and vm2 winners are the same, return that set of winners. Otherwise, for each choice of a vm1 winner A and vm2 winner B, add to the ultimate winners whichever of A or B is majority preferred to the other (or both if they are tied).

    Args:
        vm1, vm2 (VotingMethod): The voting methods to faceoff.

    Returns:
        A VotingMethod that runs the faceoff of vm1 and vm2.

    """

    def _vm(edata, curr_cands=None):

        curr_cands = edata.candidates if curr_cands is None else curr_cands

        vm1_winners = vm1(edata, curr_cands)
        vm2_winners = vm2(edata, curr_cands)

        if vm1_winners == vm2_winners:
            return vm1_winners
        
        else:
            winners = list()

            for a in vm1_winners:
                for b in vm2_winners:
                    if edata.margin(a,b) > 0:
                        winners.append(a)
                    elif edata.margin(b,a) > 0:
                        winners.append(b)
                    elif edata.margin(a,b) == 0:
                        winners.append(a)
                        winners.append(b) 

            return list(set(winners))

    return VotingMethod(_vm, name=f"{vm1.name}-{vm2.name} Faceoff")

def _faceoff(vm1, vm2):
    """
    Same as faceoff, but used to make it easier to document faceoff voting methods.
    """

    def _vm(edata, curr_cands=None):

        curr_cands = edata.candidates if curr_cands is None else curr_cands

        vm1_winners = vm1(edata, curr_cands)
        vm2_winners = vm2(edata, curr_cands)

        if vm1_winners == vm2_winners:
            return vm1_winners
        
        else:
            winners = list()

            for a in vm1_winners:
                for b in vm2_winners:
                    if edata.margin(a,b) > 0:
                        winners.append(a)
                    elif edata.margin(b,a) > 0:
                        winners.append(b)
                    elif edata.margin(a,b) == 0:
                        winners.append(a)
                        winners.append(b) 

            return list(set(winners))

    return _vm