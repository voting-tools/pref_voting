"""
    File: analysis.py
    Author: Eric Pacuit (epacuit@umd.edu)
    Date: August 9, 2022
    Updated: May 9, 2023

    Functions to analyze voting methods
"""

from pref_voting.generate_profiles import generate_profile
from functools import partial
from multiprocess import Pool, cpu_count
import pandas as pd
import numpy as np

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
        show_rankings_counts (bool, default=True): If True, show the rankings and counts of the profiles with different winning sets for the voting methods.
        return_multiple_profiles (bool, default=True): If True, return all profiles that are found.
        probmod (str, default="IC"): The probability model to be passed to the ``generate_profile`` method
        num_trials (int, default=10000): The number of profiles to check for different winning sets.

    """
    profiles = list()
    for num_cands in numbers_of_candidates:
        for num_voters in numbers_of_voters:
            print(f"{num_cands} candidates, {num_voters} voters")
            for t in range(num_trials):
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


def condorcet_efficiency_data(
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
        "num_trials": list(),
        "percent_condorcet_winners": list(),
        "condorcet_efficiency": list(),
    }
    for probmod in probmods:
        for num_cands in numbers_of_candidates:
            for num_voters in numbers_of_voters:

                print(f"{num_cands} candidates, {num_voters} voters")
                get_data = partial(
                    record_condorcet_efficiency_data,
                    vms,
                    num_cands,
                    num_voters,
                    probmod,
                )

                if use_parallel:
                    data = pool.map(get_data, range(num_trials))
                else:
                    data = list(map(get_data, range(num_trials)))
                num_cw = 0
                num_choose_cw = {vm.name: 0 for vm in vms}
                for d in data:
                    if d["has_cw"]:
                        num_cw += 1
                        for vm in vms:
                            num_choose_cw[vm.name] += int(d["cw_winner"][vm.name])

                for vm in vms:
                    data_for_df["vm"].append(vm.name)
                    data_for_df["num_cands"].append(num_cands)
                    data_for_df["num_voters"].append(num_voters)
                    data_for_df["probmod"].append(probmod)
                    data_for_df["num_trials"].append(num_trials)
                    data_for_df["percent_condorcet_winners"].append(
                        num_cw / num_trials
                    )
                    data_for_df["condorcet_efficiency"].append(
                        num_choose_cw[vm.name] / num_cw
                    )

    return pd.DataFrame(data_for_df)

# helper function for axiom_violations_data
def record_axiom_violation_data(
    axioms, 
    vms, 
    num_cands, 
    num_voters, 
    probmod, 
    verbose, 
    t
    ):

    prof = generate_profile(num_cands, num_voters, probmod=probmod)
    
    return {ax.name: {vm.name: ax.has_violation(prof, vm, verbose=verbose) for vm in vms} for ax in axioms}

def axiom_violations_data(
    axioms,
    vms,
    numbers_of_candidates=[3, 4, 5],
    numbers_of_voters=[4, 5, 10, 11,  20, 21, 50, 51,  100, 101,  500, 501,  1000, 1001],
    probmods=["IC"],
    num_trials=10000,
    verbose=False,
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
        "axiom": list(),
        "vm": list(),
        "num_cands": list(),
        "num_voters": list(),
        "probmod": list(),
        "num_trials": list(),
        "num_violations": list(),
    }
    for probmod in probmods:
        print(f"{probmod} probability model")
        for num_cands in numbers_of_candidates:
            for num_voters in numbers_of_voters:
                #print(f"{num_cands} candidates, {num_voters} voters")
                _verbose = verbose if not use_parallel else False
                get_data = partial(
                    record_axiom_violation_data,
                    axioms,
                    vms,
                    num_cands,
                    num_voters,
                    probmod,
                    _verbose
                )

                if use_parallel:
                    data = pool.map(get_data, range(num_trials))
                else:
                    data = list(map(get_data, range(num_trials)))

                for ax in axioms: 
                    for vm in vms: 
                        data_for_df["axiom"].append(ax.name)
                        data_for_df["vm"].append(vm.name)
                        data_for_df["num_cands"].append(num_cands)
                        data_for_df["num_voters"].append(num_voters)
                        data_for_df["probmod"].append(probmod)
                        data_for_df["num_trials"].append(num_trials)
                        data_for_df["num_violations"].append(sum([d[ax.name][vm.name] for d in data]))
    print("Done.")
    return pd.DataFrame(data_for_df)



def bootstrap_cia(
        generate_samples,
        process_samples, 
        precision, 
        averaging_axis = 0,
        confidence=0.95, 
        initial_trials=10000, 
        step_trials=1000, 
        bootstrap_samples=1000,
        verbose = False, 
        max_num_trials = None
        ):
    """
    Applies the percentile bootstrap confidence interval analysis using the functions generate_samples and process_samples.
    
    Args:
        generate_samples (function): A function that generates samples.
        process_samples (function): A function that processes the samples generated by generate_samples.
        precision (float): The desired precision of the confidence interval.
        averaging_axis (int, default=0): The axis along which to average the samples.
        confidence (float, default=0.95): The desired confidence level of the confidence interval.
        initial_trials (int, default=10000): The number of samples to initially generate.
        step_trials (int, default=1000): The number of samples to generate in each step.
        bootstrap_samples (int, default=1000): The number of bootstrap samples to select.
        verbose (bool, default=False): If True, then print progress information.
        max_num_trials (int, default=None): If not None, then the maximum number of trials to run.

    Returns:
        A tuple (means, half_widths, variance_bootstrap_means, variance_of_values, num_trials) where means is an array of the means of the bootstrap means, half_widths is an array of the half widths of the confidence intervals, variance_bootstrap_means is an array of the variances of the bootstrap means, and variance_of_values is an array of the variances of the values returned by process_samples, and num_trials is the total number of trials.

    """
    
    # Generate initial samples
    samples = generate_samples(num_samples = initial_trials)

    # Process the results
    values = process_samples(samples)

    # Generate bootstrap samples and calculate the confidence interval for values
    bootstrap_means = np.array([
        [np.mean(np.random.choice(s, size=len(s))) for s in values] 
        for _ in range(bootstrap_samples)
    ])
    
    means = np.mean(bootstrap_means, axis=0)

    lowers, uppers = np.percentile(bootstrap_means, 
                                     [(1-confidence)/2*100, 
                                      (1+confidence)/2*100], axis=0)
    half_widths = (uppers - lowers) / 2
    
    variance_bootstrap_means = np.var(bootstrap_means, axis=0)
    
    # Compute variance of each array in values
    variance_of_values = np.array([np.var(arr) for arr in values])

    # Continue running simulations until the confidence interval is narrow enough
    num_trials = initial_trials
    while np.any(half_widths > precision):
        if verbose:
            print("Number of trials:", num_trials)
            print(f"Remaining half widths > {precision}:", np.sum(half_widths > precision))

        new_samples = generate_samples(num_samples = step_trials)
        samples = np.concatenate((samples, new_samples), axis=averaging_axis)
        num_trials += step_trials

        values = process_samples(samples)

        bootstrap_means = np.array([
            [np.mean(np.random.choice(s, size=len(s))) for s in values] 
            for _ in range(bootstrap_samples)
        ])
        means = np.mean(bootstrap_means, axis=0)
        lowers, uppers = np.percentile(bootstrap_means,
                                       [(1-confidence)/2*100, 
                                        (1+confidence)/2*100], axis=0)
        half_widths = (uppers - lowers) / 2
        
        variance_bootstrap_means = np.var(bootstrap_means, axis=0)

        variance_of_values = np.array([np.var(arr) for arr in values])

        if max_num_trials is not None and num_trials > max_num_trials:
            break

    return means, half_widths, variance_bootstrap_means, variance_of_values, num_trials