"""
    File: analysis.py
    Author: Eric Pacuit (epacuit@umd.edu)
    Date: August 9, 2022
    
    Functions to analyze voting methods
"""

from pref_voting.generate_profiles import generate_profile
from pref_voting.profiles import Profile
from tqdm.notebook import tqdm
from functools import partial
from multiprocess import Pool, cpu_count
import pandas as pd
import copy

def find_profiles_with_different_winners(
    vms,
    numbers_of_candidates=[3, 4, 5],
    numbers_of_voters=[5, 25, 50, 100],
    all_unique_winners=False,
    show_profiles=True,
    show_margin_graphs=True,
    show_winning_sets=True,
    show_rankings_counts=False,
    return_multiple_profiles=True,
    probmod="IC",
    num_trials=10000,
):
    """
    Given a list of voting methods, search for profiles with different winning sets.

    Args:
        vms (list(functions)): A list of voting methods,
        numbers_of_candidates (list(int), default = [3, 4, 5]): The numbers of candidates to check.
        numbers_of_voters (list(int), default = [5, 25, 50, 100]): The numbers of voters to check.
        all_unique_winners (bool, default = False): If True, only return profiles in which each voting method has a unique winner.
        show_profiles (bool, default=True): If True, show profiles with different winning sets for the voting methods when discovered.
        show_margin_graphs (bool, default=True): If True, show margin graphs of the profiles with different winning sets for the voting methods when discovered.
        show_winning_sets (bool, default=True): If True, show the different winning sets for the voting methods when discovered.
        show_rankings_counts (bool, default=True): If True, show the rankins and counts of the profiles with different winning sets for the voting methods.
        return_multiple_profiles (bool, default=True): If True, return all profiles that are found.
        probmod (str, default="IC"): The probability model to be passed to the ``generate_profile`` method
        num_trials (int, default=10000): The number of profiles to check for different winning sets.

    """
    profiles = list()
    for num_cands in numbers_of_candidates:
        for num_voters in numbers_of_voters:
            print(f"{num_cands} candidates, {num_voters} voters")
            for t in tqdm(range(num_trials), leave=False):
                prof = generate_profile(num_cands, num_voters, probmod=probmod)
                winning_sets = {vm.name: vm(prof) for vm in vms}
                wss = [tuple(ws) for ws in list(winning_sets.values())]
                if (
                    not all_unique_winners or all([len(ws) == 1 for ws in wss])
                ) and list(set(wss)) == list(wss):
                    if show_profiles:
                        prof.display()
                    if show_margin_graphs:
                        prof.display_margin_graph()
                    if show_winning_sets:
                        for vm in vms:
                            vm.display(prof)
                    if show_rankings_counts:
                        print(prof.rankings_counts)
                    if not return_multiple_profiles:
                        return prof
                    else:
                        profiles.append(prof)

    print(f"Found {len(profiles)} profiles with different winning sets")
    return profiles


def record_condorcet_efficiency_data(vms, num_cands, num_voters, probmod, t):

    prof = generate_profile(num_cands, num_voters, probmod=probmod)
    cw = prof.condorcet_winner()

    return {
        "has_cw": cw is not None,
        "cw_winner": {vm.name: cw is not None and [cw] == vm(prof) for vm in vms},
    }


def find_condorcet_efficiency(
    vms,
    numbers_of_candidates=[3, 4, 5],
    numbers_of_voters=[4, 10, 20, 50, 100, 500, 1000],
    probmods=["IC"],
    num_trials=10000,
    use_parallel=True,
    num_cpus=12,
):
    """
    Returns a Pandas DataFrame with the Condorcet efficiency of a list of voting methods.

    Args:
        vms (list(functions)): A list of voting methods,
        numbers_of_candidates (list(int), default = [3, 4, 5]): The numbers of candidates to check.
        numbers_of_voters (list(int), default = [5, 25, 50, 100]): The numbers of voters to check.
        probmod (str, default="IC"): The probability model to be passed to the ``generate_profile`` method
        num_trials (int, default=10000): The number of profiles to check for different winning sets.
        use_parallel (bool, default=True): If True, then use parallel processing.
        num_cpus (int, default=12): The number of (virtual) cpus to use if using parallel processing.

    """

    if use_parallel:
        pool = Pool(num_cpus)

    data_for_df = {
        "vm": list(),
        "num_cands": list(),
        "num_voters": list(),
        "probmod": list(),
        "percent_condorcet_winners": list(),
        "condorcet_efficiency": list(),
    }
    for probmod in probmods:
        for num_cands in numbers_of_candidates:
            for num_voters in numbers_of_voters:

                print(f"{num_cands} candidates, {num_voters}, {num_voters + 1} voters")
                get_data0 = partial(
                    record_condorcet_efficiency_data,
                    vms,
                    num_cands,
                    num_voters,
                    probmod,
                )

                get_data1 = partial(
                    record_condorcet_efficiency_data,
                    vms,
                    num_cands,
                    num_voters + 1,
                    probmod,
                )

                if use_parallel:
                    data0 = pool.map(get_data0, range(num_trials))
                    data1 = pool.map(get_data1, range(num_trials))
                else:
                    data0 = list(map(get_data0, range(num_trials)))
                    data1 = list(map(get_data1, range(num_trials)))

                data = data0 + data1
                num_cw = 0
                num_choose_cw = {vm.name: 0 for vm in vms}
                for d in data:
                    if d["has_cw"]:
                        num_cw += 1
                        for vm in vms:
                            num_choose_cw[vm.name] += int(d["cw_winner"][vm.name])

                num_profiles = 2 * num_trials
                for vm in vms:
                    data_for_df["vm"].append(vm.name)
                    data_for_df["num_cands"].append(num_cands)
                    data_for_df["num_voters"].append((num_voters, num_voters + 1))
                    data_for_df["probmod"].append(probmod)
                    data_for_df["percent_condorcet_winners"].append(
                        num_cw / num_profiles
                    )
                    data_for_df["condorcet_efficiency"].append(
                        num_choose_cw[vm.name] / num_cw
                    )

    return pd.DataFrame(data_for_df)

