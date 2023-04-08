"""
    File: gen_profiles.py
    Author: Eric Pacuit (epacuit@umd.edu)
    Date: December 7, 2020
    Updated: July 14, 2022
    
    Functions to generate profile

"""

from itertools import combinations
from pref_voting.profiles import Profile
import numpy as np  # for the SPATIAL model
import math
import random
from scipy.stats import gamma

from pref_voting.profiles_with_ties import ProfileWithTies

# ############
# wrapper functions to interface with preflib tools for generating profiles
# ############

# ## URN model ###

# Generate votes based on the URN Model.
# we need num_cands and num_voters  with replace replacements.
# This function is a small modification of the same function used
# in preflib.org to generate profiles
def gen_urn(num_cands, num_voters, replace):

    voteMap = {}
    ReplaceVotes = {}

    ICsize = math.factorial(num_cands)
    ReplaceSize = 0

    for x in range(num_voters):
        flip = random.randint(1, ICsize + ReplaceSize)
        if flip <= ICsize:
            # generate an IC vote and make a suitable number of replacements...
            tvote = tuple(np.random.permutation(num_cands))  # gen_ic_vote(alts)
            voteMap[tvote] = voteMap.get(tvote, 0) + 1
            ReplaceVotes[tvote] = ReplaceVotes.get(tvote, 0) + replace
            ReplaceSize += replace
        else:
            # iterate over replacement hash and select proper vote.
            flip = flip - ICsize
            for vote in ReplaceVotes.keys():
                flip = flip - ReplaceVotes[vote]
                if flip <= 0:
                    voteMap[vote] = voteMap.get(vote, 0) + 1
                    ReplaceVotes[vote] = ReplaceVotes.get(vote, 0) + replace
                    ReplaceSize += replace
                    break
            else:
                print("We have a problem... replace fell through....")
                exit()
    return voteMap


def create_rankings_urn(num_cands, num_voters, replace):
    """create a list of rankings using the urn model"""
    vote_map = gen_urn(num_cands, num_voters, replace)
    return [vc[0] for vc in vote_map.items()], [vc[1] for vc in vote_map.items()]


# ### Mallows Model ####

# For Phi and a given number of candidates, compute the
# insertion probability vectors.
def compute_mallows_insertvec_dist(ncand, phi):
    # Compute the Various Mallows Probability Distros
    vec_dist = {}
    for i in range(1, ncand + 1):
        # Start with an empty distro of length i
        dist = [0] * i
        # compute the denom = phi^0 + phi^1 + ... phi^(i-1)
        denom = sum([pow(phi, k) for k in range(i)])
        # Fill each element of the distro with phi^i-j / denom
        for j in range(1, i + 1):
            dist[j - 1] = pow(phi, i - j) / denom
        # print(str(dist) + "total: " + str(sum(dist)))
        vec_dist[i] = dist
    return vec_dist


# Return a value drawn from a particular distribution.
def draw(values, distro):
    # Return a value randomly from a given discrete distribution.
    # This is a bit hacked together -- only need that the distribution
    # sums to 1.0 within 5 digits of rounding.
    if round(sum(distro), 5) != 1.0:
        print("Input Distro is not a Distro...")
        print(str(distro) + "  Sum: " + str(sum(distro)))
        exit()
    if len(distro) != len(values):
        print("Values and Distro have different length")

    cv = 0
    draw = random.random() - distro[cv]
    while draw > 0.0:
        cv += 1
        draw -= distro[cv]
    return values[cv]


# Generate a Mallows model with the various mixing parameters passed in
# nvoters is the number of votes we need
# candmap is a candidate map
# mix is an array such that sum(mix) == 1 and describes the distro over the models
# phis is an array len(phis) = len(mix) = len(refs) that is the phi for the particular model
# refs is an array of dicts that describe the reference ranking for the set.
def gen_mallows(num_cands, num_voters, mix, phis, refs):

    if len(mix) != len(phis) or len(phis) != len(refs):
        print("Mix != Phis != Refs")
        exit()

    # Precompute the distros for each Phi and Ref.
    # Turn each ref into an order for ease of use...
    m_insert_dists = []
    for i in range(len(mix)):
        m_insert_dists.append(compute_mallows_insertvec_dist(num_cands, phis[i]))
    # Now, generate votes...
    votemap = {}
    for cvoter in range(num_voters):
        cmodel = draw(list(range(len(mix))), mix)
        # print("cmodel is ", cmodel)
        # Generate a vote for the selected model
        insvec = [0] * num_cands
        for i in range(1, len(insvec) + 1):
            # options are 1...max
            insvec[i - 1] = draw(list(range(1, i + 1)), m_insert_dists[cmodel][i])
        vote = []
        for i in range(len(refs[cmodel])):
            # print("building vote insvec[i] - 1", insvec[i]-1)
            vote.insert(insvec[i] - 1, refs[cmodel][i])
        tvote = tuple(vote)

        votemap[tuple(vote)] = votemap.get(tuple(vote), 0) + 1
    return votemap


def create_rankings_mallows(num_cands, num_voters, phi, ref=None):

    ref = tuple(np.random.permutation(num_cands))

    vote_map = gen_mallows(num_cands, num_voters, [1.0], [phi], [ref])

    return [vc[0] for vc in vote_map.items()], [vc[1] for vc in vote_map.items()]


def create_rankings_mallows_two_rankings(num_cands, num_voters, phi, ref=None):
    """create a profile using a Mallows model with dispersion param phi
    ref is two linear orders that are reverses of each other

    wrapper function to call the preflib function gen_mallows with 2 reference rankings

    """

    ref = np.random.permutation(range(num_cands))
    ref2 = ref[::-1]

    vote_map = gen_mallows(num_cands, num_voters, [0.5, 0.5], [phi, phi], [ref, ref2])

    return [vc[0] for vc in vote_map.items()], [vc[1] for vc in vote_map.items()]


# #####
# SinglePeaked
# #####

# Return a Tuple for a IC-Single Peaked... with alternatives in range 1....range.
def gen_icsp_single_vote(alts):
    a = 0
    b = len(alts) - 1
    temp = []
    while a != b:
        if random.randint(0, 1) == 1:
            temp.append(alts[a])
            a += 1
        else:
            temp.append(alts[b])
            b -= 1
    temp.append(alts[a])
    return tuple(temp[::-1])  # reverse


def gen_single_peaked_impartial_culture_strict(nvotes, alts):
    voteset = {}
    for i in range(nvotes):
        tvote = gen_icsp_single_vote(alts)
        voteset[tvote] = voteset.get(tvote, 0) + 1
    return voteset


def create_rankings_single_peaked(num_cands, num_voters, param):
    """create a single-peaked list of rankings

    wrapper function to call the preflib function gen_single_peaked_impartial_culture_strict
    """

    vote_map = gen_single_peaked_impartial_culture_strict(
        num_voters, list(range(num_cands))
    )
    return [vc[0] for vc in vote_map.items()], [vc[1] for vc in vote_map.items()]


# ##########
# generate profile using the spatial model
# #########
# # TODO: Needs updated


def voter_utility(v_pos, c_pos, beta):
    """Based on the Rabinowitz and Macdonald (1989) mixed model
    described in Section 3, pp. 745 - 747 of
    "Voting behavior under the directional spatial model of electoral competition" by S. Merrill III

    beta = 1 is the proximity model
    beta = 0 is the directional model
    """
    return 2 * np.dot(v_pos, c_pos) - beta * (
        np.linalg.norm(v_pos) ** 2 + np.linalg.norm(c_pos) ** 2
    )


def create_prof_spatial_model(num_voters, cmap, params):
    num_dim = params[
        0
    ]  # the first component of the parameter is the number of dimensions
    beta = params[
        1
    ]  # used to define the mixed model: beta = 1 is proximity model (i.e., Euclidean distance)
    num_cands = len(cmap.keys())
    mean = [0] * num_dim  # mean is 0 for each dimension
    cov = np.diag([1] * num_dim)  # diagonal covariance

    # sample candidate/voter positions using a multivariate normal distribution
    cand_positions = np.random.multivariate_normal(np.array(mean), cov, num_cands)
    voter_positions = np.random.multivariate_normal(np.array(mean), cov, num_voters)

    # generate the rankings and counts for each ranking
    ranking_counts = dict()
    for v, v_pos in enumerate(voter_positions):
        v_utils = {
            voter_utility(v_pos, c_pos, beta): c
            for c, c_pos in enumerate(cand_positions)
        }
        ranking = tuple([v_utils[_u] for _u in sorted(v_utils.keys(), reverse=True)])
        if ranking in ranking_counts.keys():
            ranking_counts[ranking] += 1
        else:
            ranking_counts.update({ranking: 1})

    # list of tuples where first component is a ranking and the second is the count
    prof_counts = ranking_counts.items()

    return [rc[0] for rc in prof_counts], [rc[1] for rc in prof_counts]


# Given the number m of candidates and a phi\in [0,1] function computes the expected number of swaps in a vote sampled from Mallows model
def calculateExpectedNumberSwaps(num_candidates, phi):
    res = phi * num_candidates / (1 - phi)
    for j in range(1, num_candidates + 1):
        res = res + (j * (phi**j)) / ((phi**j) - 1)
    return res


# Given the number m of candidates and a absolute number of expected swaps exp_abs, this function returns a value of phi such that in a vote sampled from Mallows model with this parameter the expected number of swaps is exp_abs
def phi_from_relphi(num_candidates, relphi=None):
    if relphi is None:
        relphi = np.random.uniform(0.001, 0.999)
    if relphi == 1:
        return 1
    exp_abs = relphi * (num_candidates * (num_candidates - 1)) / 4
    low = 0
    high = 1
    while low <= high:
        mid = (high + low) / 2
        cur = calculateExpectedNumberSwaps(num_candidates, mid)
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


# #########
# functions to generate profiles
# #########

prob_models = {
    "IC": {
        "func": create_rankings_urn,
        "param": 0,
    },  # IC model is the Urn model with alpha=0
    "IAC": {"func": create_rankings_urn, "param": 1},  # IAC model is urn with alpha=1
    "MALLOWS-0.8": {"func": create_rankings_mallows, "param": 0.8},
    "MALLOWS-0.2": {"func": create_rankings_mallows, "param": 0.2},
    "MALLOWS-R": {
        "func": create_rankings_mallows,
        "param": lambda nc: np.random.uniform(0.001, 0.999),
    },
    "MALLOWS-RELPHI-0.4": {
        "func": create_rankings_mallows,
        "param": lambda nc: phi_from_relphi(nc, 0.4),
    },
    "MALLOWS-RELPHI-0.375": {
        "func": create_rankings_mallows,
        "param": lambda nc: phi_from_relphi(nc, 0.375),
    },
    "MALLOWS-RELPHI-0": {
        "func": create_rankings_mallows,
        "param": lambda nc: phi_from_relphi(nc, 0),
    },
    "MALLOWS-RELPHI-1": {
        "func": create_rankings_mallows,
        "param": lambda nc: phi_from_relphi(nc, 1),
    },
    "MALLOWS-RELPHI-R": {
        "func": create_rankings_mallows,
        "param": lambda nc: phi_from_relphi(nc),
    },
    "MALLOWS-RELPHI-R2": {
        "func": create_rankings_mallows,
        "param": lambda nc: phi_from_relphi(nc, np.random.uniform(0.001, 0.5)),
    },
    "MALLOWS_2REF-0.8": {"func": create_rankings_mallows_two_rankings, "param": 0.8},
    "MALLOWS_2REF-RELPHI-R": {
        "func": create_rankings_mallows_two_rankings,
        "param": lambda nc: phi_from_relphi(nc),
    },
    "MALLOWS_2REF-RELPHI-R2": {
        "func": create_rankings_mallows_two_rankings,
        "param": lambda nc: phi_from_relphi(nc, np.random.uniform(0.001, 0.5)),
    },
    "URN-10": {"func": create_rankings_urn, "param": 10},
    "URN-0.1": {
        "func": create_rankings_urn,
        "param": lambda nc: round(math.factorial(nc) * 0.1),
    },
    "URN-0.3": {
        "func": create_rankings_urn,
        "param": lambda nc: round(math.factorial(nc) * 0.3),
    },
    "URN-R": {
        "func": create_rankings_urn,
        "param": lambda nc: round(math.factorial(nc) * gamma.rvs(0.8)),
    },
    "SinglePeaked": {"func": create_rankings_single_peaked, "param": None},
}


def get_replacement(num_cands, param):
    return int(num_cands * param)


def generate_profile(num_cands, num_voters, probmod="IC", probmod_param=None):
    """Generate a :class:`Profile` with ``num_cands`` candidates and ``num_voters`` voters using the  probabilistic model ``probmod`` (with parameter ``probmod_param``).

    :param num_cands: the number of candidates in the profile
    :type num_cands: int
    :param num_voters: the number of voters in the profile
    :type num_voters: int
    :param probmod: the probability model used to generate the :class:`Profile`
    :type probmod: str, optional (default "IC")
    :param probmod_param: a parameter to the probability model
    :type probmod_param: number or function, optional
    :returns: A profile of strict linear orders
    :rtype: Profile


    :Example:

    .. exec_code::

        from pref_voting.generate_profiles import generate_profile
        prof = generate_profile(4, 10) # default is probmod is IC
        prof.display()
        prof = generate_profile(4, 10, probmod="IAC")
        prof.display()
        prof = generate_profile(4, 10, probmod="URN-0.3")
        prof.display()
        prof = generate_profile(4, 10, probmod="MALLOWS-R")
        prof.display()
        prof = generate_profile(4, 10, probmod="MALLOWS-RELPHI-0.375")
        prof.display()
        prof = generate_profile(4, 10, probmod="SinglePeaked")
        prof.display()

    :Possible Values of probmod:

    - "IC" (Impartial Culture);
    - "IAC" (Impartial Anonymous Culture);
    - "URN-10" (URN model with :math:`\\alpha=10`), "URN-0.1"  (URN model with :math:`\\alpha=0.1*num\_cands!`), "URN-0.3" (URN model with :math:`\\alpha=0.3*num\_cands!`), "URN-R" (URN model with randomly chosen :math:`\\alpha`);
    - "MALLOWS-0.8" (Mallows model with :math:`\\phi=0.8`), "MALLOWS-0.2" (Mallows model with :math:`\\phi=0.2`), "MALLOWS-R" (Mallows model with :math:`\\phi` randomly chosen between 0 and 1);
    - "MALLOWS-RELPHI-0.4" (Mallows model with :math:`\\phi` defined from ``num_cands`` and the relphi value of 0.4), "MALLOWS-RELPHI-0.375" (Mallows model with :math:`\\phi` defined from ``num_cands`` and the relphi value of 0.375), "MALLOWS-RELPHI-0" (Mallows model with :math:`\\phi` defined from ``num_cands`` and the relphi value of 0),  "MALLOWS-RELPHI-1" (Mallows model with :math:`\\phi` defined from ``num_cands`` and the relphi value of 1), (Mallows model with :math:`\\phi` defined from ``num_cands`` and the relphi value randomly chosen based on the number of candidates), "MALLOWS-RELPHI-R2" (Mallows model with :math:`\\phi` defined from ``num_cands`` and the relphi value randomly chosen), "MALLOWS_2REF-0.8" (Mallows model with 2 reference rankings and :math:`\\phi = 0.8`),
    - "MALLOWS_2REF-RELPHI-R": (Mallows model with 2 reference rankings and :math:`\\phi` defined from ``num_cands`` and a randomly chosen relphi value based on the number of candidates), "MALLOWS_2REF-RELPHI-R2"(Mallows model with 2 reference rankings and :math:`\\phi` defined from ``num_cands`` and a randomly chosen relphi value); and
    - "SinglePeaked" (Single Peaked)

    In addition, you can customize the probability model used to generate a profile as follows:

    - ``probmod`` is "URN" and ``probmod_param`` is either a number or a function :math:`f` and the parameter is defined by applying :math:`f` to the number of candidates.

    - ``probmod`` is "MALLOWS" and ``probmod_param`` is either a number or a function :math:`f` and the parameter is defined by applying :math:`f` to the number of candidates.

    - ``probmod`` is "MALLOWS_2REF" and ``probmod_param`` is either a number or a function :math:`f` and the parameter is defined by applying :math:`f` to the number of candidates.

    :Example:

    .. exec_code::

        import math
        from pref_voting.generate_profiles import generate_profile
        prof = generate_profile(4, 10, probmod="URN", probmod_param=5)
        prof.display()
        prof = generate_profile(4, 10, probmod="MALLOWS", probmod_param=0.5)
        prof.display()
        prof = generate_profile(4, 10, probmod="MALLOWS_2REF", probmod_param=0.5)
        prof.display()
        prof = generate_profile(4, 10, probmod="URN", probmod_param=lambda nc: math.factorial(nc) * 0.5)
    """

    if probmod in prob_models.keys():

        create_rankings = prob_models[probmod]["func"]
        _probmod_param = prob_models[probmod]["param"]

    elif probmod == "URN":

        create_rankings = create_rankings_urn
        _probmod_param = probmod_param if probmod_param is not None else 0

    elif probmod == "MALLOWS":

        create_rankings = create_rankings_mallows
        _probmod_param = probmod_param if probmod_param is not None else 1

    elif probmod == "MALLOWS_2REF":

        create_rankings = create_rankings_mallows_two_rankings
        _probmod_param = probmod_param if probmod_param is not None else 1

    else:
        print(f"{probmod}: Probability model not implemented, no profile generated.")
        return None

    probmod_param = (
        _probmod_param(num_cands) if callable(_probmod_param) else _probmod_param
    )

    rankings, rcounts = create_rankings(num_cands, num_voters, probmod_param)

    return Profile(rankings, rcounts=rcounts)


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


def generate_truncated_profile(num_cands, num_voters, max_num_ranked=3,probmod="IC"):
    """Generate a :class:`ProfileWithTies` with ``num_cands`` candidates and ``num_voters``.  
    `The ballots will be truncated linear orders of the candidates.

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

    return ProfileWithTies(
        rmaps,
        cmap=lprof.cmap,
        candidates=lprof.candidates
    )
