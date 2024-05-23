'''
    File: scoring_rules.py
    Author: Wes Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)
    Date: January 6, 2022
    
    Implementations of scoring rules. 
'''
from pref_voting.voting_method import  *
from pref_voting.social_welfare_function import  *
from pref_voting.profiles import Profile
from pref_voting.rankings import Ranking, break_ties_alphabetically
from pref_voting.voting_method import _num_rank_last 
from pref_voting.profiles import _find_updated_profile, _num_rank
from pref_voting.weighted_majority_graphs import MarginGraph
from pref_voting.voting_method_properties import  ElectionTypes

@vm(name = "Plurality", 
    input_types=[ElectionTypes.PROFILE, 
                 ElectionTypes.TRUNCATED_LINEAR_PROFILE])
def plurality(profile, curr_cands = None):
    """The **Plurality score** of a candidate :math:`c` is the number of voters that rank :math:`c` in first place. The Plurality winners are the candidates with the largest Plurality score in the ``profile`` restricted to ``curr_cands``.

    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A sorted list of candidates

    .. seealso::

        The method :meth:`pref_voting.profiles.Profile.plurality_scores` returns a dictionary assigning the Plurality scores of each candidate. 
        
    :Example:

    .. exec_code::

        from pref_voting.profiles import Profile
        from pref_voting.scoring_methods import plurality
        
        prof1 = Profile([[0, 1, 2], [1, 0, 2], [2, 1, 0]], [3, 1, 2])
        prof1.display()
        print(plurality(prof1)) # [2]
        plurality.display(prof1)

        prof2 = Profile([[0, 1, 2], [1, 0, 2], [1, 2, 0]], [3, 1, 2])
        prof2.display()
        print(plurality(prof2)) # [0, 1]
        plurality.display(prof2)

    """
                                        
    curr_cands = profile.candidates if curr_cands is None else curr_cands

    # get the Plurality scores for all the candidates in curr_cands
    plurality_scores = profile.plurality_scores(curr_cands = curr_cands)
    
    assert plurality_scores != {}, "Cannot calculate plurality scores."

    max_plurality_score = max(plurality_scores.values())

    return sorted([c for c in curr_cands if plurality_scores[c] == max_plurality_score])


@swf(name="Plurality ranking")
def plurality_ranking(profile, curr_cands=None, local=True, tie_breaking=None):
    """The SWF that ranks the candidates in curr_cands according to their plurality scores. If local is True, then the plurality scores are computed with respect to the profile restricted to curr_cands. Otherwise, the plurality scores are computed with respect to the entire profile.

    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): The candidates to rank. If None, then all candidates in profile are ranked
        local (bool, optional): If True, then the plurality scores are computed with respect to the profile restricted to curr_cands. Otherwise, the plurality scores are computed with respect to the entire profile.
        tie_breaking (str, optional): The tie-breaking method to use. If None, then no tie-breaking is used. If "alphabetic", then the tie-breaking is done alphabetically.

    Returns:
        A Ranking object
    """

    cands = profile.candidates if curr_cands is None else curr_cands

    if local:
        plurality_scores_dict = profile.plurality_scores(curr_cands = cands)

    else:
        plurality_scores_dict = profile.plurality_scores()
        plurality_scores_dict = {k: v for k, v in plurality_scores_dict.items() if k in cands}

    assert plurality_scores_dict != {}, "Cannot calculate plurality scores."

    for cand in cands:
        plurality_scores_dict[cand] = -plurality_scores_dict[cand]

    p_ranking = Ranking(plurality_scores_dict)
    p_ranking.normalize_ranks()

    if tie_breaking == "alphabetic":
        p_ranking = break_ties_alphabetically(p_ranking)

    return p_ranking

@vm(name = "Borda",
    input_types=[ElectionTypes.PROFILE, ElectionTypes.MARGIN_GRAPH])
def borda(edata, curr_cands = None, algorithm = "positional"):
    """The **Borda score** of a candidate is calculated as follows: If there are :math:`m` candidates, then the Borda score of candidate :math:`c` is :math:`\sum_{r=1}^{m} (m - r) * Rank(c,r)` where :math:`Rank(c,r)` is the number of voters that rank candidate :math:`c` in position :math:`r`. The Borda winners are the candidates with the largest Borda score in the ``profile`` restricted to ``curr_cands``. 
    Args:
        edata (Profile, MarginGraph): An anonymous profile of linear orders or a MarginGraph.
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``
        algorithm (String): if "positional", then the Borda score of a candidate is calculated from each voter's ranking as described above. If "marginal", then the Borda score of a candidate is calculated as the sum of the margins of the candidate vs. all other candidates. The positional scores and marginal scores are affinely equivalent.

    Returns: 
        A sorted list of candidates

    .. note:
        If edata is a MarginGraph, then the "marginal" algorithm is used.

    .. seealso::

        The method :meth:`pref_voting.profiles.Profile.borda_scores` returns a dictionary assigning the Borda score to each candidate. 
        
    :Example:

    .. exec_code::

        from pref_voting.profiles import Profile
        from pref_voting.scoring_methods import borda
        
        prof1 = Profile([[0, 1, 2], [1, 0, 2], [2, 1, 0]], [3, 1, 2])
        prof1.display()
        print(borda(prof1)) # [0,1]
        borda.display(prof1)

        prof2 = Profile([[0, 1, 2], [1, 0, 2], [1, 2, 0]], [3, 1, 2])
        prof2.display()
        print(borda(prof2)) # [1]
        borda.display(prof2)

    """

    curr_cands = edata.candidates if curr_cands is None else curr_cands

    if isinstance(edata,MarginGraph):
        algorithm = "marginal"

    if algorithm == "positional":
        # get the Borda scores for all the candidates in curr_cands
        borda_scores = edata.borda_scores(curr_cands = curr_cands)

    if algorithm == "marginal":
        borda_scores = {c: sum([edata.margin(c,d) for d in curr_cands]) for c in curr_cands}
    
    max_borda_score = max(borda_scores.values())
    
    return sorted([c for c in curr_cands if borda_scores[c] == max_borda_score])

@swf(name="Borda ranking")
def borda_ranking(profile, curr_cands=None, local=True, tie_breaking=None):
    """The SWF that ranks the candidates in curr_cands according to their Borda scores. If local is True, then the Borda scores are computed with respect to the profile restricted to curr_cands. Otherwise, the Borda scores are computed with respect to the entire profile.

    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): The candidates to rank. If None, then all candidates in profile are ranked
        local (bool, optional): If True, then the Borda scores are computed with respect to the profile restricted to curr_cands. Otherwise, the Borda scores are computed with respect to the entire profile.
        tie_breaking (str, optional): The tie-breaking method to use. If None, then no tie-breaking is used. If "alphabetic", then the tie-breaking is done alphabetically.

    Returns:
        A Ranking object
    """

    cands = profile.candidates if curr_cands is None else curr_cands

    if local:
        borda_scores_dict = profile.borda_scores(curr_cands = cands)

    else:
        borda_scores_dict = profile.borda_scores()
        borda_scores_dict = {k: v for k, v in borda_scores_dict.items() if k in cands}

    for cand in cands:
        borda_scores_dict[cand] = -borda_scores_dict[cand]

    b_ranking = Ranking(borda_scores_dict)
    b_ranking.normalize_ranks()

    if tie_breaking == "alphabetic":
        b_ranking = break_ties_alphabetically(b_ranking)

    return b_ranking

@vm(name = "Anti-Plurality",
    input_types=[ElectionTypes.PROFILE])
def anti_plurality(profile, curr_cands = None):
    """The **Anti-Plurality score** of a candidate $c$ is the number of voters that rank $c$ in last place.  The Anti-Plurality winners are the candidates with the smallest Anti-Plurality score in the ``profile`` restricted to ``curr_cands``. 

    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A sorted list of candidates

    :Example:

    .. exec_code::

        from pref_voting.profiles import Profile
        from pref_voting.scoring_methods import anti_plurality
        
        prof1 = Profile([[2, 1, 0], [2, 0, 1], [0, 1, 2]], [3, 1, 2])
        prof1.display()
        print(anti_plurality(prof1)) # [1]
        anti_plurality.display(prof1)

        prof2 = Profile([[2, 1, 0], [2, 0, 1], [0, 2, 1]], [3, 1, 2])
        prof2.display()
        print(anti_plurality(prof2)) # [2]
        anti_plurality.display(prof2)

    """
    
    # get ranking data
    rankings, rcounts = profile.rankings_counts

    curr_cands = profile.candidates if curr_cands is None else curr_cands
    cands_to_ignore = np.array([c for c in profile.candidates if c not in curr_cands])
    
    last_place_scores = {c: _num_rank_last(rankings, rcounts, cands_to_ignore, c) for c in curr_cands}
    min_last_place_score = min(list(last_place_scores.values()))
    
    return sorted([c for c in curr_cands if last_place_scores[c] == min_last_place_score])

@swf(name="Anti-Plurality ranking")
def anti_plurality_ranking(profile, curr_cands=None, local=True, tie_breaking=None):
    """The SWF that ranks the candidates in curr_cands according to their Anti-Plurality scores. If local is True, then the Anti-Plurality scores are computed with respect to the profile restricted to curr_cands. Otherwise, the Anti-Plurality scores are computed with respect to the entire profile.

    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): The candidates to rank. If None, then all candidates in profile are ranked
        local (bool, optional): If True, then the Anti-Plurality scores are computed with respect to the profile restricted to curr_cands. Otherwise, the Anti-Plurality scores are computed with respect to the entire profile.
        tie_breaking (str, optional): The tie-breaking method to use. If None, then no tie-breaking is used. If "alphabetic", then the tie-breaking is done alphabetically.

    Returns:
        A Ranking object
    """

    cands = profile.candidates if curr_cands is None else curr_cands

    rankings, rcounts = profile.rankings_counts

    if local:
        cands_to_ignore = np.array([c for c in profile.candidates if c not in cands])

    else:
        cands_to_ignore = np.array([])
    
    anti_plurality_scores_dict = {c: _num_rank_last(rankings, rcounts, cands_to_ignore, c) for c in cands}

    ap_ranking = Ranking(anti_plurality_scores_dict)
    ap_ranking.normalize_ranks()

    if tie_breaking == "alphabetic":
        ap_ranking = break_ties_alphabetically(ap_ranking)

    return ap_ranking


@vm(name = "Scoring Rule",
    skip_registration=True,)
def scoring_rule(profile, curr_cands = None, score = lambda num_cands, rank : 1 if rank == 1 else 0):
    """A general scoring rule.  Each voter assign a score to each candidate using the ``score`` function based on their submitted ranking (restricted to candidates in ``curr_cands``).   Returns that candidates with the greatest overall score in the profile restricted to ``curr_cands``. 

    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``
        score (function): A function that accepts two parameters ``num_cands`` (the number of candidates) and ``rank`` (a rank of a candidate) used to calculate the score of a candidate.   The default ``score`` function assigns 1 to a candidate ranked in first place, otherwise it assigns 0 to the candidate. 

    Returns: 
        A sorted list of candidates

    .. important:: 
        The signature of the ``score`` function is:: 

            def score(num_cands, rank):
                # return an int or float 

    :Example:

    .. exec_code::

        from pref_voting.profiles import Profile
        from pref_voting.scoring_methods import scoring_rule, plurality, borda, anti_plurality
        
        prof = Profile([[0, 1, 2], [1, 0, 2], [2, 1, 0]], [3, 1, 2])
        prof.display()
        scoring_rule.display(prof) # Uses default scoring function, same a Plurality        
        plurality.display(prof)

        scoring_rule.display(prof, score=lambda num_cands, rank: num_cands - rank) # same a Borda
        borda.display(prof)

        scoring_rule.display(prof, score=lambda num_cands, rank: -1 if rank == num_cands else 0) # same as Anti-Plurality
        anti_plurality.display(prof)

    """
    
    # get ranking data
    _rankings, rcounts = profile.rankings_counts

    # get (restricted) rankings
    cands_to_ignore = np.array([c for c in profile.candidates if c not in curr_cands]) if curr_cands is not None else np.array([])
    rankings = _rankings if curr_cands is None else _find_updated_profile(np.array(_rankings), cands_to_ignore, len(profile.candidates))
    candidates = profile.candidates if curr_cands is None else curr_cands

    # find the candidate scores using the score function
    cand_scores = {c: sum(_num_rank(rankings, rcounts, c, level) * score(len(candidates), level) for level in range(1, len(candidates) + 1)) for c in candidates}
    
    # find maximum score
    max_score = max(cand_scores.values())

    return sorted([c for c in candidates if cand_scores[c] == max_score])

def create_scoring_method(score, name):
    """Create a scoring method using a given score function and name."""

    def _vm(profile, curr_cands = None):
        return scoring_rule(profile, curr_cands = curr_cands, score = score)

    return VotingMethod(_vm, name = name)

@vm(name = "Dowdall",
    input_types=[ElectionTypes.PROFILE])
def dowdall(profile, curr_cands = None):
    """The first-ranked candidate gets 1 point, the second-ranked candidate gets 1/2 point, the third-ranked candidate gets 1/3 point, and so on.  The Dowdall winners are the candidates with the greatest overall score in the profile restricted to ``curr_cands``.

    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``.

    Returns:
        A sorted list of candidates

    .. note::
        This system is used in Nauru. See, e.g., Jon Fraenkel & Bernard Grofman (2014), "The Borda Count and its real-world alternatives: Comparing scoring rules in Nauru and Slovenia," Australian Journal of Political Science, 49:2, 186-205, DOI: 10.1080/10361146.2014.900530.
    
    """

    return scoring_rule(profile, curr_cands = curr_cands, score = lambda num_cands, rank: 1 / rank)

@vm(name="Positive-Negative Voting",
    input_types=[ElectionTypes.PROFILE])
def positive_negative_voting(profile, curr_cands = None):
    """The **Positive-Negative Voting** method is a scoring rule where each voter assigns a score of 1 to their top-ranked candidate and a score of -1 to their bottom-ranked candidate.  See https://onlinelibrary.wiley.com/doi/10.1111/ecin.12929 for more information.

    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns:
        A sorted list of candidates

    """

    return scoring_rule(profile, 
                        curr_cands = curr_cands, 
                        score = lambda num_cands, rank : 1 if rank == 1 else (-1 if rank == num_cands else 0))


@swf(name = "Score Ranking")
def score_ranking(profile, curr_cands = None, score = lambda num_cands, rank : 1 if rank == 1 else 0):
    """A general swf that ranks the candidates according to the given score function.  Each voter assign a score to each candidate using the ``score`` function based on their submitted ranking (restricted to candidates in ``curr_cands``).   Returns the ranking of the candidates according to the scores. 

    Args:
        profile (Profile): An anonymous profile of linear orders on a set of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``
        score (function): A function that accepts two parameters ``num_cands`` (the number of candidates) and ``rank`` (a rank of a candidate) used to calculate the score of a candidate.   The default ``score`` function assigns 1 to a candidate ranked in first place, otherwise it assigns 0 to the candidate. 

    Returns: 
        A ranking of the candidates

    .. important:: 
        The signature of the ``score`` function is:: 

            def score(num_cands, rank):
                # return an int or float 
    """
    
    # get ranking data
    _rankings, rcounts = profile.rankings_counts

    # get (restricted) rankings
    cands_to_ignore = np.array([c for c in profile.candidates if c not in curr_cands]) if curr_cands is not None else np.array([])
    rankings = _rankings if curr_cands is None else _find_updated_profile(np.array(_rankings), cands_to_ignore, len(profile.candidates))
    candidates = profile.candidates if curr_cands is None else curr_cands

    # find the candidate scores using the score function
    cand_scores = {c: -1 * sum(_num_rank(rankings, rcounts, c, level) * score(len(candidates), level) for level in range(1, len(candidates) + 1)) for c in candidates}
    
    ranking = Ranking(cand_scores)
    ranking.normalize_ranks()
    return ranking

## Borda for ProfilesWithTies

def symmetric_borda_scores(profile): 
    """
    The symmetric Borda score of a candidate c for a ranking r is the number of candidates ranked strictly below c according to r
    minus the number of candidates ranked strictly above c according to r. 
    
    See http://www.illc.uva.nl/~ulle/pubs/files/TerzopoulouEndrissJME2021.pdf for a discussion. 
    """
    
    return  {cand: sum([len([_cand for _cand in profile.candidates if r.extended_strict_pref(cand, _cand)]) * c 
                    for r,c in zip(*profile.rankings_counts)]) - sum([len([_cand for _cand in profile.candidates if r.extended_strict_pref(_cand, cand)]) * c 
                    for r,c in zip(*profile.rankings_counts)]) for cand in profile.candidates}

def domination_borda_scores(profile): 
    """
    The domination Borda score of a candidate c for a ranking r is the number of candidates ranked strictly below c according to r. 
    
    See http://www.illc.uva.nl/~ulle/pubs/files/TerzopoulouEndrissJME2021.pdf for a discussion. 

    """
    
    return  {cand: sum([len([_cand for _cand in profile.candidates if r.extended_strict_pref(cand, _cand)]) * c 
                    for r,c in zip(*profile.rankings_counts)]) for cand in profile.candidates}


def weak_domination_borda_scores(profile): 
    """
    The weak domination Borda score of a candidate c for a ranking r is the number of candidates ranked weakly below c according to r. 
    
    See http://www.illc.uva.nl/~ulle/pubs/files/TerzopoulouEndrissJME2021.pdf for a discussion. 

    """
    
    return  {cand: sum([len([_cand for _cand in profile.candidates if r.extended_weak_pref(cand, _cand)]) * c 
                    for r,c in zip(*profile.rankings_counts)]) for cand in profile.candidates}

def non_domination_borda_scores(profile): 
    """
    The non-domination Borda score of a candidate c for a ranking r is -1 times the number of candidates ranked strictly above c according to r. 
    
    See http://www.illc.uva.nl/~ulle/pubs/files/TerzopoulouEndrissJME2021.pdf for a discussion. 

    """
    
    return  {cand: -sum([len([_cand for _cand in profile.candidates if r.extended_strict_pref(_cand, cand)]) * c 
                    for r,c in zip(*profile.rankings_counts)]) for cand in profile.candidates}


@vm(name="Borda (for Truncated Profiles)",
    input_types=[ElectionTypes.TRUNCATED_LINEAR_PROFILE])
def borda_for_profile_with_ties(
    profile, 
    curr_cands=None, 
    borda_scores=symmetric_borda_scores): 
    """
    Borda score for truncated linear orders using different ways of defining the Borda score for truncated linear
    orders.  
    """
    # profile must be a ProfileWithTies object
    if isinstance(profile, Profile): 
        return borda(profile, curr_cands = curr_cands)

    curr_cands = curr_cands if curr_cands is not None else profile.candidates 
    
    restricted_prof = profile.remove_candidates([c for c in profile.candidates if c not in curr_cands])
    
    b_scores = borda_scores(restricted_prof)
    
    max_borda_score = max(b_scores.values())
    
    return sorted([c for c in restricted_prof.candidates if b_scores[c] == max_borda_score])


scoring_swfs = [
    plurality_ranking,
    borda_ranking,
    anti_plurality_ranking,
    score_ranking   
]

