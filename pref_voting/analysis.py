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


def record_condorcet_efficiency_data(vms, num_cands, num_voters, probmod, probmod_param, t):

    prof = generate_profile(num_cands, num_voters, probmod=probmod, probmod_param=probmod_param)
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
    probmod_params=None,
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

    probmod_params_list = [None]*len(probmods) if probmod_params is None else probmod_params

    assert len(probmod_params_list) == len(probmods), "probmod_params must be a list of the same length as probmods"

    if use_parallel:
        pool = Pool(num_cpus)

    data_for_df = {
        "vm": list(),
        "num_cands": list(),
        "num_voters": list(),
        "probmod": list(),
        "probmod_param":list(),
        "num_trials": list(),
        "percent_condorcet_winners": list(),
        "condorcet_efficiency": list(),
    }
    for probmod,probmod_param in zip(probmods, probmod_params_list):
        for num_cands in numbers_of_candidates:
            for num_voters in numbers_of_voters:

                print(f"{num_cands} candidates, {num_voters} voters")
                get_data = partial(
                    record_condorcet_efficiency_data,
                    vms,
                    num_cands,
                    num_voters,
                    probmod,
                    probmod_param
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
                    data_for_df["probmod_param"].append(probmod_param)
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


def estimated_variance_of_sampling_dist( 
    values_for_each_experiment,
    mean_for_each_experiment=None):
    # values_for_each_vm is a 2d numpy array

    mean_for_each_experiment = np.nanmean(values_for_each_experiment, axis=1) if mean_for_each_experiment is not None else mean_for_each_experiment

    num_val_for_each_exp = np.sum(~np.isnan(values_for_each_experiment), axis=1)
    
    row_means_reshaped = mean_for_each_experiment[:, np.newaxis]
    return np.where(
        num_val_for_each_exp * (num_val_for_each_exp - 1) != 0.0,
        (1 / (num_val_for_each_exp * (num_val_for_each_exp - 1))) * np.nansum(
            (values_for_each_experiment - row_means_reshaped) ** 2, 
            axis=1),
            np.nan
            )

def estimated_std_error(values_for_each_experiment, mean_for_each_experiment=None):
    # values_for_each_vm is a 2d numpy array
    return np.sqrt(estimated_variance_of_sampling_dist(values_for_each_experiment, mean_for_each_experiment=mean_for_each_experiment))

def means_with_estimated_standard_error(
        generate_samples, 
        max_std_error, 
        initial_trials=1000, 
        step_trials=1000,
        min_num_trials=10_000, 
        max_num_trials=None,
        verbose=False
        ):
    """
    For each list of numbers produced by generate_samples, returns the means, the estimated standard error (https://en.wikipedia.org/wiki/Standard_error) of the means, the variance of the samples, and the total number of trials.  

    Uses the estimated_variance_of_sampling_dist (as described in https://berkeley-stat243.github.io/stat243-fall-2023/units/unit9-sim.html) and estimated_std_error functions. 
    
    Args:
        generate_samples (function): A function that a 2d numpy array of samples.  It should take two arguments:  num_samples and step (only used if samples are drawn from a pre-computed source in order to ensure that we get new samples during the while loop below).
        max_std_error (float): The desired estimated standard error for the mean of each sample.
        initial_trials (int, default=1000): The number of samples to initially generate.
        step_trials (int, default=1000): The number of samples to generate in each step.
        min_num_trials (int, default=10000): The minimum number of trials to run.
        max_num_trials (int, default=None): If not None, then the maximum number of trials to run.
        verbose (bool, default=False): If True, then print progress information.

    Returns:
        A tuple (means, est_std_errors, variances, num_trials) where means is an array of the means of the samples, est_std_errors is an array of estimated standard errors of the samples,  variances is an array of the variances of the samples, and num_trials is the total number of trials.

    """
    
    # samples is a 2d numpy array
    step = 0
    samples = generate_samples(num_samples = initial_trials, step = step)
    
    means = np.nanmean(samples, axis=1)
    variances = np.nanvar(samples, axis=1)
    est_std_errors = estimated_std_error( 
        samples, 
        mean_for_each_experiment=means)
        
    if verbose:
        print("Initial number of trials:", initial_trials)
        print(f"Remaining estimated standard errors greater than {max_std_error}:", np.sum(est_std_errors > max_std_error))
        print(f"Estimated standard errors that are still greater than {max_std_error}:\n",est_std_errors[est_std_errors > max_std_error])

    num_trials = initial_trials
    
    while (np.isnan(est_std_errors).any() or np.any(est_std_errors > max_std_error) or (num_trials < min_num_trials)) and (max_num_trials is None or num_trials < max_num_trials):
        if verbose:
            print("Number of trials:", num_trials)
            print(f"Remaining estimated standard errors greater than {max_std_error}:", np.sum(est_std_errors > max_std_error))
            print(f"Estimated standard errors that are still greater than {max_std_error}:\n",est_std_errors[est_std_errors > max_std_error])

        step += 1
        new_samples = generate_samples(num_samples=step_trials, step=step)

        samples = np.concatenate((samples, new_samples), axis=1)

        num_trials += step_trials

        means = np.nanmean(samples, axis=1)
        variances = np.nanvar(samples, axis=1)
        est_std_errors = estimated_std_error(
            samples, 
            mean_for_each_experiment=means)

    return means, est_std_errors, variances, num_trials


#### Bootstrap confidence interval analysis (to be removed) ####

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
    
    num_gen_samples = 0
    # Generate initial samples
    samples = generate_samples(num_samples = initial_trials, num_gen_samples = num_gen_samples)

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
    num_gen_samples += 1
    while np.any(half_widths > precision):
        if verbose:
            print("Number of trials:", num_trials)
            print(f"Remaining half widths greater than {precision}:", np.sum(half_widths > precision))
            print(f"Half widths that are still greater than {precision}:\n",half_widths[half_widths > precision])

        new_samples = generate_samples(num_samples = step_trials, num_gen_samples = num_gen_samples)
        num_gen_samples += 1

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