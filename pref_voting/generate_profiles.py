"""
    File: gen_profiles.py
    Author: Wes Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)
    Date: December 7, 2020
    Updated: May 25, 2025
    
    Functions to generate profiles

"""

from itertools import combinations
from pref_voting.profiles import Profile
from pref_voting.generate_spatial_profiles import generate_spatial_profile
from pref_voting.generate_utility_profiles import linear_utility
import numpy as np 
import math
import random
from scipy.stats import gamma
from itertools import permutations
from pref_voting.helper import weak_compositions, weak_orders

from pref_voting.profiles_with_ties import ProfileWithTies
from ortools.linear_solver import pywraplp
from prefsampling.ordinal import impartial, impartial_anonymous, urn, plackett_luce, didi, stratification, single_peaked_conitzer, single_peaked_walsh, single_peaked_circle, single_crossing, euclidean, mallows

from prefsampling.core.euclidean import EuclideanSpace
from collections import Counter

# ############
# wrapper functions to interface with preflib tools for generating profiles
# ############


# Given the number m of candidates and a phi in [0,1], 
# compute the expected number of swaps in a vote sampled 
# from the Mallows model
def find_expected_number_of_swaps(num_candidates, phi):
    res = phi * num_candidates / (1 - phi)
    for j in range(1, num_candidates + 1):
        res = res + (j * (phi**j)) / ((phi**j) - 1)
    return res


# Given the number m of candidates and a absolute number of 
# expected swaps exp_abs, this function returns a value of 
# phi such that in a vote sampled from Mallows model with 
# this parameter the expected number of swaps is exp_abs
def phi_from_relphi(num_candidates, relphi=None, seed=None):

    rng = np.random.default_rng(seed)
    if relphi is None:
        relphi = rng.uniform(0.001, 0.999)
    if relphi == 1:
        return 1
    exp_abs = relphi * (num_candidates * (num_candidates - 1)) / 4
    low = 0
    high = 1
    while low <= high:
        mid = (high + low) / 2
        cur = find_expected_number_of_swaps(num_candidates, mid)
        if abs(cur - exp_abs) < 1e-5:
            return mid
        # If x is greater, ignore left half
        if cur < exp_abs:
            low = mid

        # If x is smaller, ignore right half
        elif cur > exp_abs:
            high = mid

    # If we reach here, then the element was not present
    return -1

# Return a list of phis from the relphi value
def phis_from_relphi(num_candidates, num, relphi=None, seed=None):

    rng = np.random.default_rng(seed)
    if relphi is None:
        relphis = rng.uniform(0.001, 0.999, size=num)
    else: 
        relphis = [relphi] * num
    
    return [phi_from_relphi(num_candidates, relphi=relphis[n]) for n in range(num)]


def get_rankings(num_candidates, num_voters, **kwargs): 
    """
    Get the rankings for a given number of candidates and voters using
    the [prefsampling library](https://comsoc-community.github.io/prefsampling/index.html). 

    Args:
        num_candidates (int): The number of candidates.
        num_voters (int): The number of voters.
        kwargs (dict): Any parameters for the probability model.
    
    Returns:
        list: A list of rankings.
    """

    if 'probmodel' in kwargs:
        probmodel = kwargs['probmodel']
    elif 'probmod' in kwargs: # for backwards compatibility
        probmodel = kwargs['probmod']
    else: 
        probmodel = "impartial"

    if 'seed' in kwargs:
        seed = kwargs['seed']
    else: 
        seed = None

    if probmodel == "IC" or probmodel == 'impartial': 
        
        rankings = impartial(num_voters, 
                             num_candidates, 
                             seed=seed) 
    
    elif probmodel == "IAC" or probmodel == 'impartial_anonymous': 
        
        rankings = impartial_anonymous(num_voters, 
                                       num_candidates, 
                                       seed=seed)
    elif probmodel == "MALLOWS" or probmodel == 'mallows':

        impartial_central_vote = True
        if 'phi' in kwargs: 
            phi = kwargs['phi']
        else:
            phi = 1.0
            
        if 'normalise_phi' in kwargs: 
            normalise_phi = kwargs['normalise_phi']
        else:
            normalise_phi = False

        if 'central_vote' in kwargs: 
            central_vote = kwargs['central_vote']
            impartial_central_vote = False
        else:
            central_vote = None

        rankings = mallows(num_voters,
                           num_candidates, 
                           phi,
                           normalise_phi=normalise_phi,
                           central_vote=central_vote,
                           impartial_central_vote=impartial_central_vote,
                           seed=seed)


    elif probmodel == "MALLOWS-0.8":

        phi = 0.8
        impartial_central_vote = True           
        if 'normalise_phi' in kwargs: 
            normalise_phi = kwargs['normalise_phi']
        else:
            normalise_phi = False

        if 'central_vote' in kwargs: 
            central_vote = kwargs['central_vote']
            impartial_central_vote = False
        else:
            central_vote = None

        rankings = mallows(num_voters,
                           num_candidates, 
                           phi,
                           normalise_phi=normalise_phi,
                           central_vote=central_vote,
                           impartial_central_vote=impartial_central_vote,
                           seed=seed)
        
    elif probmodel == "MALLOWS-0.2":

        phi = 0.2
        impartial_central_vote = True           
        if 'normalise_phi' in kwargs: 
            normalise_phi = kwargs['normalise_phi']
        else:
            normalise_phi = False

        if 'central_vote' in kwargs: 
            central_vote = kwargs['central_vote']
            impartial_central_vote = False
        else:
            central_vote = None

        rankings = mallows(num_voters,
                           num_candidates, 
                           phi,
                           normalise_phi=normalise_phi,
                           central_vote=central_vote,
                           impartial_central_vote=impartial_central_vote,
                           seed=seed)

    elif probmodel == "MALLOWS-R":

        rng = np.random.default_rng(seed)
        phi = rng.uniform(0.001, 0.999)
        impartial_central_vote = True
            
        if 'normalise_phi' in kwargs: 
            normalise_phi = kwargs['normalise_phi']
        else:
            normalise_phi = False

        if 'central_vote' in kwargs: 
            central_vote = kwargs['central_vote']
            impartial_central_vote = False
        else:
            central_vote = None

        rankings = mallows(num_voters,
                           num_candidates, 
                           phi,
                           normalise_phi=normalise_phi,
                           central_vote=central_vote,
                           impartial_central_vote=impartial_central_vote,
                           seed=seed)

    elif probmodel == "MALLOWS-RELPHI":

        impartial_central_vote = True
        if 'relphi' in kwargs: 
            relphi = kwargs['relphi']
        else:
            relphi = None
            
        if 'normalise_phi' in kwargs: 
            normalise_phi = kwargs['normalise_phi']
        else:
            normalise_phi = False

        if 'central_vote' in kwargs: 
            central_vote = kwargs['central_vote']
            impartial_central_vote = False
        else:
            central_vote = None

        phi = phi_from_relphi(num_candidates, relphi=relphi, seed=seed)

        rankings = mallows(num_voters,
                           num_candidates, 
                           phi,
                           normalise_phi=normalise_phi,
                           central_vote=central_vote,
                           impartial_central_vote=impartial_central_vote,
                           seed=seed)

    elif probmodel == "MALLOWS-RELPHI-0.375":
        
        relphi = 0.375
        impartial_central_vote = True
        if 'normalise_phi' in kwargs: 
            normalise_phi = kwargs['normalise_phi']
        else:
            normalise_phi = False

        if 'central_vote' in kwargs: 
            central_vote = kwargs['central_vote']
            impartial_central_vote = False
        else:
            central_vote = None

        phi = phi_from_relphi(num_candidates, relphi=relphi, seed=seed)

        rankings = mallows(num_voters,
                           num_candidates, 
                           phi,
                           normalise_phi=normalise_phi,
                           central_vote=central_vote,
                           impartial_central_vote=impartial_central_vote,
                           seed=seed)


    elif probmodel == "MALLOWS-RELPHI-R":
            
        impartial_central_vote = True
        if 'normalise_phi' in kwargs: 
            normalise_phi = kwargs['normalise_phi']
        else:
            normalise_phi = False

        if 'central_vote' in kwargs: 
            central_vote = kwargs['central_vote']
            impartial_central_vote = False
        else:
            central_vote = None

        phi = phi_from_relphi(num_candidates, relphi=None, seed=seed)

        rankings = mallows(num_voters,
                           num_candidates, 
                           phi,
                           normalise_phi=normalise_phi,
                           central_vote=central_vote,
                           impartial_central_vote=impartial_central_vote,
                           seed=seed)


    elif probmodel == "URN" or probmodel == 'urn': 

        if 'alpha' in kwargs: 
            alpha = kwargs['alpha']
        else:
            alpha = 0.0
            
        rankings = urn(num_voters,
                       num_candidates, 
                       alpha,
                       seed=seed)

    elif probmodel == "URN-10":
        
        alpha = 10
        rankings = urn(num_voters,
                       num_candidates, 
                       alpha,
                       seed=seed)
    
    elif probmodel == "URN-0.3":
        
        alpha = round(math.factorial(num_candidates) * 0.3)
        rankings = urn(num_voters,
                       num_candidates, 
                       alpha,
                       seed=seed)
        
    elif probmodel == "URN-R":
        
        rng = np.random.default_rng(seed)
        alpha = round(math.factorial(num_candidates) * gamma.rvs(0.8, random_state=rng))
        rankings = urn(num_voters,
                       num_candidates,
                       alpha,
                       seed=seed)
        
    elif probmodel == "plackett_luce":
        
        if 'alphas' not in kwargs:
            raise ValueError("Error: alphas parameter missing.  A value must be specified for each candidate indicating their relative quality.")
            #RaiseValueError()
        else:
            alphas = kwargs['alphas']

        rankings = plackett_luce(num_voters,
                                       num_candidates, 
                                       alphas,
                                       seed=seed)
        
    elif probmodel == "didi":
        
        if 'alphas' not in kwargs:
            raise ValueError("Error: alphas parameter missing.  A value must be specified for each candidate indicating each candidate's quality.")
        else:
            alphas = kwargs['alphas']

        rankings = didi(num_voters,
                        num_candidates, 
                        alphas,
                        seed=seed)
        
    elif probmodel == "stratification":
        
        if 'weight' not in kwargs:
            raise ValueError("Error: weight parameter missing.  The weight parameter specifies the size of the upper class of candidates.")
        else:
            weight = kwargs['weight']

        rankings = stratification(num_voters,
                                  num_candidates, 
                                  weight,
                                  seed=seed) 
    
    elif probmodel == "single_peaked_conitzer":
        
        rankings = single_peaked_conitzer(num_voters,
                                          num_candidates, 
                                          seed=seed) 
    
    elif probmodel == "SinglePeaked" or probmodel == "single_peaked_walsh":
        
        rankings = single_peaked_walsh(num_voters,
                                       num_candidates, 
                                       seed=seed) 

    elif probmodel == "single_peaked_circle":
        
        rankings = single_peaked_circle(num_voters,
                                        num_candidates, 
                                        seed=seed)       

    elif probmodel == "single_crossing":
        
        rankings = single_crossing(num_voters,
                                   num_candidates, 
                                   seed=seed) 
        
    elif probmodel == "euclidean":
        
        euclidean_spaces = {
            "gaussian_ball": EuclideanSpace.GAUSSIAN_BALL,
            "gaussian_cube": EuclideanSpace.GAUSSIAN_CUBE,
            "unbounded_gaussian": EuclideanSpace.UNBOUNDED_GAUSSIAN,
            "uniform_ball": EuclideanSpace.UNIFORM_BALL,
            "uniform_cube": EuclideanSpace.UNIFORM_CUBE,
            "uniform_sphere": EuclideanSpace.UNIFORM_SPHERE,
        }

        if 'space' in kwargs:
            space = kwargs['space']
        else:
            space = "uniform_ball"

        if 'dimension' in kwargs:
            dimension = kwargs['dimension']
        else:
            dimension = 2

        rankings = euclidean(num_voters,
                             num_candidates, 
                             voters_positions=euclidean_spaces[space],
                             candidates_positions=euclidean_spaces[space],
                             num_dimensions=dimension, 
                             seed=seed) 
    
    else: 
        raise ValueError("Error: The probability model is not recognized.")
        
    return rankings

def generate_profile(num_candidates, 
                     num_voters, 
                     anonymize=False,
                     num_profiles=1,
                     **kwargs): 
    """
    Generate profiles using the prefsampling library.

    Args:
        num_candidates (int): The number of candidates.
        num_voters (int): The number of voters.
        anonymize (bool): If True, anonymize the profiles.
        num_profiles (int): The number of profiles to generate.
        kwargs (dict): Any parameters for the probability model.

    Returns:
        list: A list of profiles or a single profile if num_profiles is 1.  
    """
            
    profs = [Profile(get_rankings(num_candidates,
                                  num_voters, 
                                  **kwargs))  
                                  for _ in range(num_profiles)]
    
    if anonymize: 
        profs = [prof.anonymize() for prof in profs]
        
    return profs[0] if num_profiles == 1 else profs

def generate_profile_with_groups(
        num_candidates, 
        num_voters, 
        probmodels, 
        weights=None,
        seed=None, 
        num_profiles=1, 
        anonymize=False):
    
    """
    Generate profiles with groups of voters generated from different probability models.
    The probability of selecting a probability model is proportional its weight in the list weight.

    Args:
        num_candidates (int): The number of candidates.
        num_voters (int): The number of voters.
        probmodels (list): A list of dictionaries specifying a probability model.
        weights (list): A list of weights for each probability model.
        seed (int): The random seed.
        num_profiles (int): The number of profiles to generate.
        anonymize (bool): If True, anonymize the profiles.
    """
    if weights is None:
        weights = [1] * len(probmodels)
    
    assert len(weights)==len(probmodels), "The number of weights must be equal to the number of probmodels"

    probs = [w / sum(weights) for w in weights]
    
    rng = np.random.default_rng(seed)

    profs = list()
    for _ in range(num_profiles):
        selected_probmodels = rng.choice(probmodels, num_voters, p=probs)

        selected_probmodels_num = Counter([tuple((k,v) if type(v) != list else (k, tuple(v)) for k,v in pm.items()) for pm in selected_probmodels])

        rankings = list()
        for pm_data, nv in selected_probmodels_num.items():
            rankings = rankings + list(get_rankings(num_candidates, nv, **dict(pm_data)))

        prof = Profile(rankings)
        if anonymize: 
            prof = prof.anonymize()
        profs.append(prof)

    return profs[0] if num_profiles == 1 else profs

####
# Enumerating profiles
####

def enumerate_anon_profile(num_cands, num_voters):
    """A generator that enumerates all anonymous profiles with num_cands candidates and num_voters voters.

    Args:
        num_cands (int): Number of candidates.
        num_voters (int): Number of voters.

    Yields:
        Profile: An anonymous profile.
    """
    
    ballot_types = list(permutations(range(num_cands)))
    num_ballot_types = len(ballot_types)

    for comp in weak_compositions(num_voters, num_ballot_types):
        instantiated_ballot_types = [ballot_types[idx] for idx, i in enumerate(comp) if i != 0]
        nonzerocomp = [i for i in comp if i != 0]
        yield Profile(instantiated_ballot_types, rcounts = nonzerocomp)

def canonical_ballot_multiset(profile: Profile) -> tuple:
    """
    Lexicographically minimal multiset of (ranking, count) pairs across
    all candidate permutations.
    """
    m = profile.num_cands
    rankings, counts = profile.rankings_counts
    counts = counts.astype(int)

    best = None
    for perm in permutations(range(m)):
        relabel = dict(zip(range(m), perm))
        canon   = tuple(sorted(
            (tuple(relabel[c] for c in r.tolist()), int(k))
            for r, k in zip(rankings, counts)
        ))
        if best is None or canon < best:
            best = canon
    return best


def enumerate_anon_neutral_profile(num_cands: int, num_voters: int):
    """
    A generator that yields one representative per neutrality-orbit of anonymous profiles.
    """
    seen = set()
    for prof in enumerate_anon_profile(num_cands, num_voters):
        key = canonical_ballot_multiset(prof)
        if key not in seen:
            seen.add(key)
            yield prof


def enumerate_anon_profile_with_ties(num_cands, num_voters):
    """A generator that enumerates all anonymous profiles--allowing ties in ballots--with num_cands candidates and num_voters voters

    Args:
        num_cands (int): Number of candidates.
        num_voters (int): Number of voters.

    Yields:
        ProfileWithTies: An anonymous profile.
    """
    
    ballot_types = list(weak_orders(range(num_cands)))
    num_ballot_types = len(ballot_types)

    for comp in weak_compositions(num_voters, num_ballot_types):
        instantiated_ballot_types = [ballot_types[idx] for idx, i in enumerate(comp) if i != 0]
        nonzerocomp = [i for i in comp if i != 0]
        yield ProfileWithTies(instantiated_ballot_types, rcounts = nonzerocomp)

def _weakorder_to_levels(order):
    """Convert a weak order dict -> tuple of rank-levels."""
    if not order:
        return tuple()
    max_rank = max(order.values())
    return tuple(
        tuple(sorted(c for c, r in order.items() if r == lev))
        for lev in range(max_rank + 1)
    )

def _canonical_multiset_with_ties(profile):
    """Canonical key for a *ProfileWithTies* under candidate permutations."""
    m = profile.num_cands
    ballots, counts = profile.rankings_counts
    ballots = list(ballots)
    counts  = [int(k) for k in counts]

    best = None
    for perm in permutations(range(m)):
        relabel = dict(zip(range(m), perm))

        canon = tuple(sorted(
            (
                _weakorder_to_levels(
                    {relabel[c]: r for c, r in (
                        ballot if isinstance(ballot, dict) else ballot.rmap
                    ).items()}
                ),
                k,
            )
            for ballot, k in zip(ballots, counts)
        ))
        if best is None or canon < best:
            best = canon
    return best

def enumerate_anon_neutral_profile_with_ties(num_cands, num_voters):
    """A generator that yields one representative per neutrality-orbit of anonymous profiles allowing ties.
    """
    seen = set()
    for prof in enumerate_anon_profile_with_ties(num_cands, num_voters):
        key = _canonical_multiset_with_ties(prof)
        if key not in seen:
            seen.add(key)
            yield prof


####
# Generating ProfilesWithTies
####


def strict_weak_orders(A):
    if not A:  # i.e., A is empty
        yield []
        return
    for k in range(1, len(A) + 1):
        for B in combinations(A, k):  # i.e., all nonempty subsets B
            for order in strict_weak_orders(set(A) - set(B)):
                yield [B] + order


def generate_truncated_profile(
        num_cands, 
        num_voters, 
        max_num_ranked=3,
        probmod="IC"):
    
    """Generate a :class:`ProfileWithTies` with ``num_cands`` candidates and ``num_voters``.  
    The ballots will be truncated linear orders of the candidates.  Returns a :class:`ProfileWithTies` that uses extended strict preference (so all ranked candidates are strictly preferred to any candidate that is not ranked).

    Args:
        num_cands (int): The number of candidates to include in the profile. 
        num_voters (int): The number of voters to include in the profile.
        max_num_ranked (int, default=3): The maximum level to truncate the linear ranking. 
        probmod (str): optional (default "IC")

    Returns: 
        ProfileWithTies 

    :Example:

        .. exec_code::

            from pref_voting.generate_profiles import generate_truncated_profile

            prof = generate_truncated_profile(6, 7)
            prof.display()

            prof = generate_truncated_profile(6, 7, max_num_ranked=6)
            prof.display()

    :Possible Values of probmod:
    
    - "IC" (Impartial Culture): each randomly generated linear order of all candidates is truncated at a level from 1 to max_num_ranked, where the probability of truncating at level t is the number of truncated linear orders of length t divided by the number of truncated linear orders of length from 1 to max_num_ranked. Then a voter is equally likely to get any of the truncated linear orders of length from 1 to max_num_ranked.
    - "RT" (Random Truncation): each randomly generated linear order of all candidates is truncated at a level that is randomly chosen from 1 to max_num_ranked.
        
    """
    
    if max_num_ranked > num_cands:
        max_num_ranked = num_cands

    if probmod == "IC":
        num_rankings_of_length = dict()

        for n in range(1, max_num_ranked + 1):
            num_rankings_of_length[n] = 1
            for i in range(num_cands,num_cands-n, -1):
                num_rankings_of_length[n] *= i

        num_all_rankings = sum([num_rankings_of_length[n] for n in range(1, max_num_ranked + 1)])
        probabilities = [num_rankings_of_length[n] / num_all_rankings for n in range(1, max_num_ranked + 1)]

    lprof = generate_profile(num_cands, num_voters)
    
    rmaps = list()
    for r in lprof.rankings:

        if probmod == "RT":
            truncate_at = random.choice(range(1, max_num_ranked + 1))

        if probmod == "IC":
            truncate_at = random.choices(range(1, max_num_ranked + 1), weights=probabilities, k=1)[0]

        truncated_r = r[0:truncate_at]

        rmap = {c: _r + 1 for _r, c in enumerate(truncated_r)}

        rmaps.append(rmap)

    prof = ProfileWithTies(
        rmaps,
        cmap=lprof.cmap,
        candidates=lprof.candidates
    )
    prof.use_extended_strict_preference()
    return prof

####
# Generating Profile from ordinal margin graph
####

def minimal_profile_from_edge_order(cands, edge_order):
    """Given a list of candidates and a list of edges (positive margin edges only) in order of descending strength, find a minimal profile whose ordinal margin graph has that edge order.

    Args: 
        cands (list): list of candidates
        edge_order (list): list of edges in order of descending strength

    Returns:
        Profile: a profile whose ordinal margin graph has the given edge order
    """

    solver = pywraplp.Solver.CreateSolver("SAT")

    num_cands = len(cands)
    rankings = list(permutations(range(num_cands)))

    ranking_to_var = dict()
    infinity = solver.infinity()
    for ridx, r in enumerate(rankings): 
        _v = solver.IntVar(0.0, infinity, f"x{ridx}")
        ranking_to_var[r] = _v

    nv = solver.IntVar(0.0, infinity, "nv")
    equations = list()
    for c1 in cands: 
        for c2 in cands: 
            if c1 != c2: 
                if (c1,c2) in edge_order:
                    rankings_c1_over_c2 = [ranking_to_var[r] for r in rankings if r.index(c1) < r.index(c2)]
                    rankings_c2_over_c1 = [ranking_to_var[r] for r in rankings if r.index(c2) < r.index(c1)]
                    equations.append(sum(rankings_c1_over_c2) - sum(rankings_c2_over_c1) >= 1)

                for c3 in cands:
                    for c4 in cands:
                        if c3 != c4:
                            if (c1,c2) in edge_order and (c3,c4) in edge_order and edge_order.index((c1,c2)) < edge_order.index((c3,c4)):
                                rankings_c3_over_c4 = [ranking_to_var[r] for r in rankings if r.index(c3) < r.index(c4)]
                                rankings_c4_over_c3 = [ranking_to_var[r] for r in rankings if r.index(c4) < r.index(c3)]
                                equations.append(sum(rankings_c1_over_c2) - sum(rankings_c2_over_c1) >= sum(rankings_c3_over_c4) - sum(rankings_c4_over_c3) + 1)
                    
    equations.append(nv == sum(list(ranking_to_var.values())))

    for eq in equations: 
        solver.Add(eq)

    solver.Minimize(nv)

    status = solver.Solve()

    if status == pywraplp.Solver.INFEASIBLE:
        print("Error: Did not find a solution.")
        return None

    if status != pywraplp.Solver.OPTIMAL: 
        print("Warning: Did not find an optimal solution.")

    _ranks = list()
    _rcounts = list()

    for r,v in ranking_to_var.items(): 

        if v.solution_value() > 0: 
            _ranks.append(r)
            _rcounts.append(int(v.solution_value()))
            if not v.solution_value().is_integer(): 
                print("ERROR: Found non integer, ", v.solution_value())
                return None

    return Profile(_ranks, rcounts = _rcounts)