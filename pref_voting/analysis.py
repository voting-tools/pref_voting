"""
    File: analysis.py
    Author: Wes Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)
    Date: August 9, 2022
    Updated: May 9, 2023

    Functions to analyze voting methods
"""

from pref_voting.generate_profiles import generate_profile
from functools import partial
from pathos.multiprocessing import ProcessingPool as Pool
from scipy.stats import binomtest

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



def record_condorcet_efficiency_data(vms, num_cands, num_voters, pm, t):

    prof = pm(num_cands, num_voters)
    cw = prof.condorcet_winner()

    return {
        "has_cw": cw is not None,
        "cw_winner": {vm.name: cw is not None and [cw] == vm(prof) for vm in vms},
    }

def condorcet_efficiency_data(
    vms,
    numbers_of_candidates=[3, 4, 5],
    numbers_of_voters=[4, 10, 20, 50, 100, 500, 1000],
    prob_models = {"IC": lambda nc, nv: generate_profile(nc, nv)},
    min_num_samples=1000,
    max_num_samples=100_000,
    max_error=0.01,
    use_parallel=True,
    num_cpus=12,
):
    """
    Returns a Pandas DataFrame with the Condorcet efficiency of a list of voting methods.

    Args:
        vms (list(functions)): A list of voting methods,
        numbers_of_candidates (list(int), default = [3, 4, 5]): The numbers of candidates to check.
        numbers_of_voters (list(int), default = [5, 25, 50, 100]): The numbers of voters to check.
        probmod (dict, default="IC"): A dictionary with keys as the names of the probability models and values as functions that generate profiles.  Each function should accept a number of candidates and a number of voters. 
        min_num_trials (int, default=1000): The minimum number of profiles to check.
        max_num_trials (int, default=100_000): The maximum number of profiles to check.
        max_error (float, default=0.01): The maximum error to allow in the 95% confidence interval.
        use_parallel (bool, default=True): If True, then use parallel processing.
        num_cpus (int, default=12): The number of (virtual) cpus to use if using parallel processing.

    """

    if use_parallel:
        pool = Pool(num_cpus)

    data_for_df = {
         "num_candidates": [],
         "num_voters": [],
         "prob_model": [],
         "voting_method": [],
         "condorcet_efficiency": [],
         "error": [],
         "num_samples": [],
         "percent_condorcet_winner": [],
         "percent_condorcet_winner_error": [],
         "min_num_samples": list(),
         "max_num_samples": list(),
         "max_error": list(),
    }
    for pm_name, pm in prob_models.items():
        for num_cands in numbers_of_candidates:
            for num_voters in numbers_of_voters:

                print(f"{pm_name}: {num_cands} candidates, {num_voters} voters")
                
                get_data = partial(
                    record_condorcet_efficiency_data,
                    vms,
                    num_cands,
                    num_voters,
                    pm
                )

                num_samples = 0
                error_ranges = []
                elect_condorcet_winner = {
                    vm.name: [] for vm in vms
                }
                has_condorcet_winner = []
                while num_samples < min_num_samples or (any([(err[1] - err[0]) > max_error for err in error_ranges]) and num_samples < max_num_samples):
                    
                    if use_parallel:
                        data = pool.map(get_data, range(min_num_samples))
                    else:
                        data = list(map(get_data, range(min_num_samples)))

                    for d in data:
                        has_condorcet_winner.append(d["has_cw"])
                        if d["has_cw"]:
                            for vm in vms:
                                elect_condorcet_winner[vm.name].append(d["cw_winner"][vm.name])

                    error_ranges = [binomial_confidence_interval(elect_condorcet_winner[vm.name])  if len(elect_condorcet_winner[vm.name]) > 0 else (0, np.inf) for vm in vms]

                    num_samples += min_num_samples

                for vm in vms:
                    data_for_df["num_candidates"].append(num_cands)
                    data_for_df["num_voters"].append(num_voters)
                    data_for_df["prob_model"].append(pm_name)
                    data_for_df["voting_method"].append(vm.name)
                    data_for_df["condorcet_efficiency"].append(np.mean(elect_condorcet_winner[vm.name]))
                    err_interval = binomial_confidence_interval(elect_condorcet_winner[vm.name])
                    data_for_df["error"].append(err_interval[1] - err_interval[0])
                    data_for_df["num_samples"].append(num_samples)
                    data_for_df["percent_condorcet_winner"].append(np.mean(has_condorcet_winner))
                    err_interval = binomial_confidence_interval(has_condorcet_winner)
                    data_for_df["percent_condorcet_winner_error"].append(err_interval[1] - err_interval[0])
                    data_for_df["min_num_samples"].append(min_num_samples)
                    data_for_df["max_num_samples"].append(max_num_samples)
                    data_for_df["max_error"].append(max_error)

    return pd.DataFrame(data_for_df)



def record_num_winners_data(vms, num_cands, num_voters, probmod, probmod_param, t):

    prof = generate_profile(num_cands, num_voters, probmod=probmod, probmod_param=probmod_param)

    return {
        "num_winners": {vm.name: len(vm(prof)) for vm in vms},
    }

def resoluteness_data(
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
    Returns a Pandas DataFrame with resoluteness data for a list of voting methods.

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
        "freq_multiple_winners": list(),
        "avg_num_winners": list(),
        "avg_percent_winners": list(),
    }
    for probmod,probmod_param in zip(probmods, probmod_params_list):
        for num_cands in numbers_of_candidates:
            for num_voters in numbers_of_voters:

                print(f"{num_cands} candidates, {num_voters} voters")
                get_data = partial(
                    record_num_winners_data,
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
                    
                num_winners = {vm.name: 0 for vm in vms}
                multiple_winners = {vm.name: 0 for vm in vms}

                for d in data:
                    for vm in vms:
                        num_winners[vm.name] += int(d["num_winners"][vm.name])
                        if d["num_winners"][vm.name] > 1:
                            multiple_winners[vm.name] += 1

                for vm in vms:
                    data_for_df["vm"].append(vm.name)
                    data_for_df["num_cands"].append(num_cands)
                    data_for_df["num_voters"].append(num_voters)
                    data_for_df["probmod"].append(probmod)
                    data_for_df["probmod_param"].append(probmod_param)
                    data_for_df["num_trials"].append(num_trials)
                    data_for_df["freq_multiple_winners"].append(multiple_winners[vm.name] / num_trials)
                    data_for_df["avg_num_winners"].append(
                        num_winners[vm.name] / num_trials
                    )
                    data_for_df["avg_percent_winners"].append(
                        (num_winners[vm.name] / (num_cands * num_trials))
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
    Returns a Pandas DataFrame with axiom violation data for a list of voting methods.

    Args:
        vms (list(functions)): A list of voting methods,
        numbers_of_candidates (list(int), default = [3, 4, 5]): The numbers of candidates to check.
        numbers_of_voters (list(int), default = [5, 25, 50, 100]): The numbers of voters to check.
        probmod (str, default="IC"): The probability model to be passed to the ``generate_profile`` method
        num_trials (int, default=10000): The number of profiles to check for axiom violations.
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

    mean_for_each_experiment = np.nanmean(values_for_each_experiment, axis=1) if mean_for_each_experiment is None else mean_for_each_experiment

    num_val_for_each_exp = np.sum(~np.isnan(values_for_each_experiment), axis=1)
    
    row_means_reshaped = mean_for_each_experiment[:, np.newaxis]
    return np.where(
        num_val_for_each_exp * (num_val_for_each_exp - 1) != 0.0,
        (1 / (num_val_for_each_exp * (num_val_for_each_exp - 1))) * np.nansum(
            (values_for_each_experiment - row_means_reshaped) ** 2, 
            axis=1),
            np.nan
            )


def binomial_confidence_interval(xs, confidence_level=0.95):
    """
    Calculate the exact confidence interval for a binomial proportion.

    This function computes the confidence interval for the true proportion of successes in a binary dataset using the exact binomial test. It is particularly useful for small sample sizes or when the normal approximation is not appropriate.

    Parameters
    ----------
    xs : array-like
        A sequence of binary observations (0 for failure, 1 for success).
    confidence_level : float, optional
        The confidence level for the interval, between 0 and 1. Default is 0.95.

    Returns
    -------
    tuple of float
        A tuple containing the lower and upper bounds of the confidence interval.

    Examples
    --------
    >>> xs = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0]
    >>> binomial_confidence_interval(xs)
    (0.4662563841506048, 0.9337436158493953)

    Notes
    -----
    - Uses the `binomtest` function from `scipy.stats` with the 'exact' method.
    - Suitable for datasets where the normal approximation may not hold.

    References
    ----------
    .. [1] "Binomial Test", SciPy v1.7.1 Manual, 
       https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html
    """
    binom_ci = binomtest(int(np.sum(xs)), len(xs)).proportion_ci(
        confidence_level=confidence_level, method='exact'
    )
    return (binom_ci.low, binom_ci.high)

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