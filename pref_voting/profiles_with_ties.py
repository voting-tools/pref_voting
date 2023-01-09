"""
    File: profiles_with_ties.py
    Author: Eric Pacuit (epacuit@umd.edu)
    Date: January 5, 2022
    Updated: July 13, 2022
    Updated: December 19, 2022

    Functions to reason about profiles of (truncated) strict weak orders.
"""

from math import ceil
import copy
import numpy as np
from tabulate import tabulate
from pref_voting.weighted_majority_graphs import (
    MajorityGraph,
    MarginGraph,
    SupportGraph,
)


class Ranking(object):
    """A ranking of a set of candidates.

    A ranking is a map from candidates to ranks (integers).  There is no assumption that all candidates in an election are ranked.

    :param rmap: Dictionary in which the keys are the candidates and the values are the ranks.
    :type rmap: dict[int or str: int]
    :param cmap: Dictionary mapping candidates (keys of the ``rmap``) to candidate names (strings).  If not provied, each candidate  is mapped to itself.
    :type cmap: dict[int: str], optional

    :Example:

    The following code creates three rankings:

    1. ``rank1`` is the ranking where 0 is ranked first, 2 is ranked in second-place, and 1 is ranked last.
    2. ``rank2`` is the ranking where 0 and 1 are tied for first place, and 2 is ranked last.
    3. ``rank3`` is the ranking where 0 is ranked first, and 2 is ranked in last place.

    .. code-block:: python

            rank1 = Ranking({0:1, 1:3, 2:2})
            rank2 = Ranking({0:1, 1:1, 2:2})
            rank3 = Ranking({0:1, 2:3})

    .. important::
        The numerical value of the ranks do not mean anything.  They are only used to make ordinal comparisons.  For instance, each of the following represents the same ranking:
        0 is ranked  first, 2 is ranked second, and 1 is ranked in last place.

        .. code-block:: python

            rank1 = Ranking({0:1, 1:3, 2:2})
            rank2 = Ranking({0:1, 1:10, 2:3})
            rank3 = Ranking({0:10, 1:100, 2:30})

    """

    def __init__(self, rmap, cmap=None):
        """constructer method"""

        self.rmap = rmap
        self.cmap = cmap if cmap is not None else {c: str(c) for c in rmap.keys()}

    @property
    def ranks(self):
        """Returns a sorted list of the ranks."""
        return sorted(set(self.rmap.values()))

    @property
    def cands(self):
        """Returns a sorted list of the candidates that are ranked."""
        return sorted(list(self.rmap.keys()))

    def cands_at_rank(self, r):
        """Returns a list of the candidates that are assigned the rank ``r``."""
        return [c for c in self.rmap.keys() if self.rmap[c] == r]

    def is_ranked(self, c):
        """Returns True if the candidate ``c`` is ranked."""

        return c in self.rmap.keys()

    def strict_pref(self, c1, c2):
        """Returns True if ``c1`` is strictly preferred to ``c2``.

        The return value is True when both ``c1`` and ``c2`` are ranked and the rank of ``c1`` is strictly smaller than the rank of ``c2``.
        """

        return (self.is_ranked(c1) and self.is_ranked(c2)) and self.rmap[
            c1
        ] < self.rmap[c2]

    def extended_strict_pref(self, c1, c2):
        """Returns True when either ``c1`` is ranked and ``c2`` is not ranked or the rank of ``c1`` is strictly smaller than the rank of ``c2``."""

        return (self.is_ranked(c1) and not self.is_ranked(c2)) or (
            (self.is_ranked(c1) and self.is_ranked(c2))
            and self.rmap[c1] < self.rmap[c2]
        )

    def indiff(self, c1, c2):
        """Returns True if ``c1`` and ``c2`` are tied.

        The return value is True when  both ``c1`` and  ``c2`` are  ranked and the rank of ``c1`` equals the rank of ``c2``.

        """

        return (
            self.is_ranked(c1) and self.is_ranked(c2) and self.rmap[c1] == self.rmap[c2]
        )

    def extended_indiff(self, c1, c2):
        """Returns True  when either both ``c1`` and  ``c2`` are not ranked or the rank of ``c1`` equals the rank of ``c2``."""

        return (not self.is_ranked(c1) and not self.is_ranked(c2)) or (
            self.is_ranked(c1) and self.is_ranked(c2) and self.rmap[c1] == self.rmap[c2]
        )

    def weak_pref(self, c1, c2):
        """Returns True if ``c1`` is weakly preferred to ``c2``.

        The return value is True if either ``c1`` is tied with ``c2`` or ``c1`` is strictly preferred to ``c2``.
        """

        return self.strict_pref(c1, c2) or self.indiff(c1, c2)

    def extended_weak_pref(self, c1, c2):
        """Returns True when either ``c1`` and ``c2`` are in the relation of extended indifference or ``c1`` is extended strictly preferred to ``c2``."""

        return self.extended_strict_pref(c1, c2) or self.extended_indiff(c1, c2)

    def remove_cand(self, a):
        """Returns a Ranking with the candidate ``a`` removed."""

        new_rmap = {c: self.rmap[c] for c in self.rmap.keys() if c != a}
        new_cmap = {c: self.cmap[c] for c in self.cmap.keys() if c != a}
        return Ranking(new_rmap, cmap=new_cmap)

    def first(self, cs=None):
        """Returns the list of candidates from ``cs`` that have the highest ranking.   If ``cs`` is None, then use all the ranked candidates."""

        _ranks = list(self.rmap.values()) if cs is None else [self.rmap[c] for c in cs]
        _cands = list(self.rmap.keys()) if cs is None else cs
        min_rank = min(_ranks)
        return sorted([c for c in _cands if self.rmap[c] == min_rank])

    def last(self, cs=None):
        """Returns the list of candidates from ``cs`` that have the worst ranking.   If ``cs`` is None, then use all the ranked candidates."""

        _ranks = list(self.rmap.values()) if cs is None else [self.rmap[c] for c in cs]
        _cands = list(self.rmap.keys()) if cs is None else cs
        max_rank = max(_ranks)
        return sorted([c for c in _cands if self.rmap[c] == max_rank])

    def is_empty(self): 
        """Return True when the ranking is empty."""
        return len(self.rmap.keys()) == 0
        
    def has_tie(self): 
        """Return True when the ranking has a tie."""
        return len(list(set(self.rmap.values()))) != len(list(self.rmap.values()))

    def is_linear(self, num_cands):
        """Return True when the ranking is a linear order of ``num_cands`` candidates. 
        """

        return not self.has_tie() and len(self.rmap.keys()) == num_cands

    def is_truncated_linear(self, num_cands): 
        """Return True when the ranking is a truncated linear order, so it is linear but ranks fewer than ``num_cands`` candidates. 
        """
        return  not self.has_tie() and len(self.rmap.keys()) < num_cands
    
    def has_skipped_rank(self): 
        """Returns True when a rank is skipped."""
        return len(self.ranks) != 0 and len(list(set(self.rmap.values()))) != max(list(self.rmap.values()))

    def has_overvote(self): 
        """
        Return True if the voter submitted an overvote (a ranking with a tie). 
        """
        return self.has_tie()
    

    def truncate_overvote(self):
        """
        Truncate the ranking at an overvote.  
        """ 
        
        new_rmap = dict()

        for r in self.ranks:
            cands_at_rank = self.cands_at_rank(r)
            if len(cands_at_rank) == 1:
                new_rmap[cands_at_rank[0]] = r
            elif len(cands_at_rank) > 1: 
                break

        self.rmap = new_rmap

    def normalize_ranks(self):
        """Change the ranks so that they start with 1, and the next rank is the next integer after the previous rank.

        :Example:

        .. exec_code:: python

            from pref_voting.profiles_with_ties import Ranking
            r = Ranking({0:1, 1:3, 2:2})
            print(r.rmap)
            r.normalize_ranks()
            print("After normalizing: ", r.rmap)

            r = Ranking({0:1, 1:10, 2:3})
            print(r.rmap)
            r.normalize_ranks()
            print("After normalizing: ", r.rmap)

            r = Ranking({0:-100, 1:123, 2:0})
            print(r.rmap)
            r.normalize_ranks()
            print("After normalizing: ", r.rmap)

            r = Ranking({0:10, 1:10, 2:100})
            print(r.rmap)
            r.normalize_ranks()
            print("After normalizing: ", r.rmap)

        """
        self.rmap = {c: self.ranks.index(r) + 1 for c, r in self.rmap.items()}

    ## set preferences
    def AAdom(self, c1s, c2s, use_extended_preferences=False):
        """
        Returns True if every candidate in ``c1s`` is weakly preferred to every candidate in ``c2s``. If ``use_extended_preferences`` is True, then use the extended weak preference.
        """

        weak_pref = (
            self.extended_weak_pref if use_extended_preferences else self.weak_pref
        )

        return all([all([weak_pref(c1, c2) for c2 in c2s]) for c1 in c1s])

    def strong_dom(self, c1s, c2s, use_extended_preferences=False):
        """
        Returns True if ``AAdom(c1s, c2s)`` and there is some candidate in ``c1s`` that is strictly preferred to every candidate in ``c2s``. If ``use_extended_preferences`` is True, then use the extended  preferences.
        """

        strict_pref = (
            self.extended_strict_pref if use_extended_preferences else self.strict_pref
        )

        return self.AAdom(
            c1s, c2s, use_extended_preferences=use_extended_preferences
        ) and any([all([strict_pref(c1, c2) for c2 in c2s]) for c1 in c1s])

    def weak_dom(self, c1s, c2s, use_extended_preferences=False):
        """
        Returns True if ``AAdom(c1s, c2s)`` and there is some candidate in ``c1s`` that is strictly preferred to some candidate in ``c2s``. If ``use_extended_preferences`` is True, then use the extended  preferences.
        """

        strict_pref = (
            self.extended_strict_pref if use_extended_preferences else self.strict_pref
        )

        return self.AAdom(
            c1s, c2s, use_extended_preferences=use_extended_preferences
        ) and any([any([strict_pref(c1, c2) for c2 in c2s]) for c1 in c1s])

    def __str__(self):
        """
        Display the ranking as a string.
        """
        r_str = ""

        for r in self.ranks:
            cands_at_rank = self.cands_at_rank(r)
            if len(cands_at_rank) == 1:
                r_str += str(self.cmap[cands_at_rank[0]]) + " "
            else:
                r_str += "( " + " ".join(map(lambda c: str(self.cmap[c]) + " ", cands_at_rank)) + ")"
        return r_str


class ProfileWithTies(object):
    """An anonymous profile of (truncated) strict weak orders of :math:`n` candidates. 

    :param rankings: List of rankings in the profile, where a ranking is either a :class:`Ranking` object or a dictionary.
    :type rankings: list[dict[int or str: int]] or list[Ranking]
    :param rcounts: List of the number of voters associated with each ranking.  Should be the same length as rankings.   If not provided, it is assumed that 1 voters submitted each element of ``rankings``.
    :type rcounts: list[int], optional
    :param candidates: List of candidates in the profile.  If not provied, this is the list that is ranked by at least on voter.
    :type candidates: list[int] or list[str], optional
    :param cmap: Dictionary mapping candidates (integers) to candidate names (strings).  If not provied, each candidate name is mapped to itself.
    :type cmap: dict[int: str], optional

    :Example:

    The following code creates a profile in which
    2 voters submitted the ranking 0 ranked first, 1 ranked second, and 2 ranked third; 3 voters submitted the ranking 1 and 2 are tied for first place and 0 is ranked second; and 1 voter submitted the ranking in which 2 is ranked first and 0 is ranked second:

    .. code-block:: python

            prof = ProfileWithTies([{0: 1, 1: 2, 2: 3}, {1:1, 2:1, 0:2}, {2:1, 0:2}], [2, 3, 1])
    """

    def __init__(self, rankings, rcounts=None, candidates=None, cmap=None):
        """constructor method"""

        assert rcounts is None or len(rankings) == len(
            rcounts
        ), "The number of rankings much be the same as the number of rcounts"

        self.candidates = (
            sorted(candidates)
            if candidates is not None
            else sorted(list(set([c for r in rankings for c in r.keys()])))
        )
        """The candidates in the profile. """

        self.num_cands = len(self.candidates)
        """The number of candidates in the profile."""

        self.ranks = list(range(1, self.num_cands + 1))

        self.cmap = cmap if cmap is not None else {c: c for c in self.candidates}
        """The candidate map is a dictionary associating a candidate with the name used when displaying a candidate."""
        
        self.rankings = [
            Ranking(r, cmap=self.cmap)
            if type(r) == dict
            else Ranking(r.rmap, cmap=self.cmap)
            for r in rankings
        ]
        """The list of rankings in the Profile (each ranking is a :class:`Ranking` object). 
        """

        self.rcounts = [1] * len(rankings) if rcounts is None else list(rcounts)

        self.num_voters = np.sum(self.rcounts)
        """The number of voters in the profile. """

        self.using_extended_strict_preference = False
        """A flag indicating whether the profile is using extended strict preferences when calculating supports, margins, etc."""
        
        # memoize the supports
        self._supports = {
            c1: {
                c2: sum(
                    n
                    for r, n in zip(self.rankings, self.rcounts)
                    if r.strict_pref(c1, c2)
                )
                for c2 in self.candidates
            }
            for c1 in self.candidates
        }

    def use_extended_strict_preference(self):
        """Redefine the supports so that *extended strict preferences* are used. Using extended strict preference may change the margins between candidates."""

        self.using_extended_strict_preference = True
        self._supports = {
            c1: {
                c2: sum(
                    n
                    for r, n in zip(self.rankings, self.rcounts)
                    if r.extended_strict_pref(c1, c2)
                )
                for c2 in self.candidates
            }
            for c1 in self.candidates
        }

    def use_strict_preference(self):
        """Redefine the supports so that strict preferences are used. Using extended strict preference may change the margins between candidates."""

        self.using_extended_strict_preference = False
        self._supports = {
            c1: {
                c2: sum(
                    n
                    for r, n in zip(self.rankings, self.rcounts)
                    if r.strict_pref(c1, c2)
                )
                for c2 in self.candidates
            }
            for c1 in self.candidates
        }

    @property
    def rankings_counts(self):
        """Returns the rankings and the counts of each ranking."""

        return self.rankings, self.rcounts

    @property
    def rankings_as_dicts_counts(self):
        """Returns the rankings represented as dictionaries and the counts of each ranking."""

        return [r.rmap for r in self.rankings], self.rcounts

    def support(self, c1, c2, use_extended_preferences=False):
        """Returns the support of candidate ``c1`` over candidate ``c2``, where the support is the number of voters that rank ``c1`` strictly above ``c2``."""

        return self._supports[c1][c2]

    def margin(self, c1, c2):
        """Returns the margin of candidate ``c1`` over candidate ``c2``, where the maring is the number of voters that rank ``c1`` strictly above ``c2`` minus the number of voters that rank ``c2`` strictly above ``c1``."""

        return self._supports[c1][c2] - self._supports[c2][c1]

    def is_tied(self, c1, c2): 
        """Returns True if ``c1`` and ``c2`` are tied (i.e., the margin of ``c1`` over ``c2`` is 0)."""

        return self.margin(c1, c2) == 0

    def dominators(self, cand, curr_cands=None):
        """Returns the list of candidates that are majority preferred to ``cand`` in the profile restricted to the candidates in ``curr_cands``."""
        candidates = self.candidates if curr_cands is None else curr_cands

        return [c for c in candidates if self.majority_prefers(c, cand)]

    def dominates(self, cand, curr_cands=None):
        """Returns the list of candidates that ``cand`` is majority preferred to in the majority graph restricted to ``curr_cands``."""
        candidates = self.candidates if curr_cands is None else curr_cands

        return [c for c in candidates if self.majority_prefers(cand, c)]

    def ratio(self, c1, c2):
        """Returns the ratio of the support of ``c1`` over ``c2`` to the support ``c2`` over ``c1``."""

        if self.support(c1, c2) > 0 and self.support(c2, c1) > 0:
            return self.support(c1, c2) / self.support(c2, c1)
        elif self.support(c1, c2) > 0 and self.support(c2, c1) == 0:
            return float(self.num_voters + self.support(c1, c2))
        elif self.support(c1, c2) == 0 and self.support(c2, c1) > 0:
            return 1 / (self.num_voters + self.support(c2, c1))
        elif self.support(c1, c2) == 0 and self.support(c2, c1) == 0:
            return 1

    def majority_prefers(self, c1, c2):
        """Returns True if ``c1`` is majority preferred to ``c2``."""

        return self.margin(c1, c2) > 0

    def condorcet_winner(self, curr_cands=None):
        """Returns the Condorcet winner in the profile restricted to ``curr_cands`` if one exists, otherwise return None.

        The **Condorcet winner** is the candidate that is majority preferred to every other candidate.
        """
        curr_cands = curr_cands if curr_cands is not None else self.candidates

        cw = None
        for c in curr_cands:

            if all([self.majority_prefers(c, c1) for c1 in curr_cands if c1 != c]):
                cw = c
                break
        return cw

    def condorcet_loser(self, curr_cands=None):
        """Returns the Condorcet loser in the profile restricted to ``curr_cands`` if one exists, otherwise return None.

        A candidate :math:`c` is a  **Condorcet loser** if every other candidate  is majority preferred to :math:`c`.
        """

        curr_cands = curr_cands if curr_cands is not None else self.candidates

        cl = None
        for c1 in curr_cands:
            if all([self.majority_prefers(c2, c1) for c2 in curr_cands if c1 != c2]):
                cl = c1
                break  # if a Condorcet loser exists, then it is unique
        return cl

    def weak_condorcet_winner(self, curr_cands=None):
        """Returns a list of the weak Condorcet winners in the profile restricted to ``curr_cands`` (which may be empty).

        A candidate :math:`c` is a  **weak Condorcet winner** if there is no other candidate that is majority preferred to :math:`c`.

        .. note:: While the Condorcet winner is unique if it exists, there may be multiple weak Condorcet winners.
        """

        curr_cands = curr_cands if curr_cands is not None else self.candidates

        weak_cw = list()
        for c1 in curr_cands:
            if not any(
                [self.majority_prefers(c2, c1) for c2 in curr_cands if c1 != c2]
            ):
                weak_cw.append(c1)
        return sorted(weak_cw) if len(weak_cw) > 0 else None

    def copeland_scores(self, curr_cands = None, scores = (1,0,-1)):
        """The Copeland scores in the profile restricted to the candidates in ``curr_cands``. 

        The **Copeland score** for candidate :math:`c` is calculated as follows:  :math:`c` receives ``scores[0]`` points for every candidate that  :math:`c` is majority preferred to, ``scores[1]`` points for every candidate that is tied with :math:`c`, and ``scores[2]`` points for every candidate that is majority preferred to :math:`c`. The default ``scores`` is ``(1, 0, -1)``. 
        

        :param curr_cands: restrict attention to candidates in this list. Defaults to all candidates in the profile if not provided. 
        :type curr_cands: list[int], optional
        :param scores: the scores used to calculate the Copeland score of a candidate :math:`c`: ``scores[0]`` is for the candidates that :math:`c` is majority preferred to; ``scores[1]`` is the number of candidates tied with :math:`c`; and ``scores[2]`` is the number of candidate majority preferred to :math:`c`.  The default value is ``scores = (1, 0, -1)`` 
        :type scores: tuple[int], optional
        :returns: a dictionary associating each candidate in ``curr_cands`` with its Copeland score. 

        """        
    
        wscore, tscore, lscore = scores
        candidates = self.candidates if curr_cands is None else curr_cands
        c_scores = {c: 0.0 for c in candidates}
        for c1 in candidates:
            for c2 in candidates:
                if self.majority_prefers(c1, c2):
                    c_scores[c1] += wscore
                elif self.majority_prefers(c2, c1):
                    c_scores[c1] += lscore
                elif c1 != c2:
                    c_scores[c1] += tscore
        return c_scores


    def strict_maj_size(self):
        """Returns the strict majority of the number of voters."""

        return int(
            self.num_voters / 2 + 1
            if self.num_voters % 2 == 0
            else int(ceil(float(self.num_voters) / 2))
        )

    def plurality_scores(self): 
        """
        Return the Plurality Scores of the candidates given that each voter ranks a single candidate in first place.  
        """
        if any([len(r.first()) != 1 for r in self.rankings]): 
            print("Error: Cannot find the plurality scores unless all voters rank a unique candidate in first place.")
            return {}
        
        rankings, rcounts = self.rankings_counts
        
        return {cand: sum([c for r, c in zip(rankings, rcounts) if [cand] == r.first()]) 
                for cand in self.candidates}

    def plurality_scores_ignoring_overvotes(self): 
        """
        Return the Plurality scores ignoring empty rankings and overvotes.
        """
        rankings, rcounts = self.rankings_counts
        
        return {cand: sum([c for r, c in zip(rankings, rcounts) if len(r.cands) > 0 and [cand] == r.first()]) for cand in self.candidates}

    def remove_empty_rankings(self): 
        """
        Remove the empty rankings from the profile. 
        """
        new_rankings = list()
        new_rcounts = list()
                
        for r,c in zip(*(self.rankings_counts)):
            
            if len(r.cands) != 0: 
                new_rankings.append(r)
                new_rcounts.append(c)

        self.rankings = new_rankings
        self.rcounts = new_rcounts
        
        # update the number of voters
        self.num_voters = np.sum(self.rcounts)
        
        if self.using_extended_strict_preference: 
            self.use_extended_strict_preference()
        else: 
            self.use_strict_preference()


    def truncate_overvotes(self): 
        """Return a new profile in which all rankings with overvotes are truncated. """
        
        new_profile = copy.deepcopy(self)
        rankings, rcounts = new_profile.rankings_counts
        
        report = []
        for r,c in zip(rankings, rcounts): 
            old_ranking = copy.deepcopy(r)
            if r.has_overvote(): 
                r.truncate_overvote()
                report.append((old_ranking, r, c))
    
        if self.using_extended_strict_preference: 
            new_profile.use_extended_strict_preference()
        else: 
            new_profile.use_strict_preference()
            
        return new_profile, report

    def unique_rankings(self): 
        """Return to the list of unique rankings in the profile. 
        """
        
        return (list(set([str(r) for r in self.rankings])))
                
    def margin_graph(self):
        """Returns the margin graph of the profile.  See :class:`.MarginGraph`.

        :Example:

        .. exec_code:: python

                from pref_voting.profiles_with_ties import ProfileWithTies
                prof = ProfileWithTies([{0: 1, 1: 2, 2: 3}, {1:1, 2:1, 0:2}, {2:1, 0:2}], [2, 3, 1])

                mg = prof.margin_graph()
                print(mg.edges)
                print(mg.m_matrix)
        """

        return MarginGraph.from_profile(self)

    def support_graph(self):
        """Returns the support graph of the profile.  See :class:`.SupportGraph`.

        :Example:

        .. exec_code:: python

                from pref_voting.profiles_with_ties import ProfileWithTies
                prof = ProfileWithTies([{0: 1, 1: 2, 2: 3}, {1:1, 2:1, 0:2}, {2:1, 0:2}], [2, 3, 1])

                sg = prof.support_graph()
                print(sg.edges)
                print(sg.s_matrix)

        """

        return SupportGraph.from_profile(self)

    def majority_graph(self):
        """Returns the majority graph of the profile.  See :class:`.MarginGraph`.

        :Example:

        .. exec_code:: python

                from pref_voting.profiles_with_ties import ProfileWithTies
                prof = ProfileWithTies([{0: 1, 1: 2, 2: 3}, {1:1, 2:1, 0:2}, {2:1, 0:2}], [2, 3, 1])

                mg = prof.majority_graph()
                print(mg.edges)

        """

        return MajorityGraph.from_profile(self)

    def cycles(self):
        """Return a list of the cycles in the profile."""

        return self.margin_graph().cycles()

    def remove_candidates(self, cands_to_ignore):
        """Remove all candidates from ``cands_to_ignore`` from the profile.

        :param cands_to_ignore: list of candidates to remove from the profile
        :type cands_to_ignore: list[int]
        :returns: a profile with candidates from ``cands_to_ignore`` removed.

        :Example:

        .. exec_code::

            from pref_voting.profiles_with_ties import ProfileWithTies
            prof = ProfileWithTies([{0: 1, 1: 2, 2: 3}, {1:1, 2:1, 0:2}, {2:1, 0:2}], [2, 3, 1])
            prof.display()
            new_prof = prof.remove_candidates([1])
            new_prof.display()
            print(new_prof.ranks)
        """

        updated_rankings = [
            {c: r for c, r in rank.rmap.items() if c not in cands_to_ignore}
            for rank in self.rankings
        ]
        new_candidates = [c for c in self.candidates if c not in cands_to_ignore]
        restricted_prof = ProfileWithTies(
            updated_rankings,
            rcounts=self.rcounts,
            candidates=new_candidates,
            cmap=self.cmap,
        )
        
        if self.using_extended_strict_preference: 
            restricted_prof.use_extended_strict_preference()
            
        return restricted_prof

    def report(self): 
        """
        Display a report of the types of rankings in the profile. 
        """
        num_ties = 0
        num_empty_rankings = 0
        num_with_skipped_ranks = 0
        num_trucated_linear_orders = 0
        num_linear_orders = 0
        
        rankings, rcounts = self.rankings_counts
        
        for r, c in zip(rankings, rcounts): 

            if r.has_tie():
                num_ties += c    
            if r.is_empty(): 
                num_empty_rankings += c
            elif r.is_linear(len(self.candidates)): 
                num_linear_orders += c
            elif r.is_truncated_linear(len(self.candidates)): 
                num_trucated_linear_orders += c
            
            if r.has_skipped_rank(): 
                num_with_skipped_ranks += c
        print(f'''There are {len(self.candidates)} candidates and {str(sum(rcounts))} {'ranking: ' if sum(rcounts) == 1 else 'rankings: '} 
        The number of empty rankings: {num_empty_rankings}
        The number of rankings with ties: {num_ties}
        The number of linear orders: {num_linear_orders}
        The number of truncated linear orders: {num_trucated_linear_orders}
        
The number of rankings with skipped ranks: {num_with_skipped_ranks}
        
        ''')

    def display_rankings(self): 
        """
        Display a list of the rankings in the profile. 
        """
        rankings, rcounts = self.rankings_counts
        
        rs = dict()
        for r, c in zip(rankings, rcounts): 
            if str(r) in rs.keys(): 
                rs[str(r)] += c
            else: 
                rs[str(r)] = c
                
        for r,c in rs.items(): 
            print(f"{r} {c}")

    def display(self, cmap=None, style="pretty", curr_cands=None):
        """Display a profile (restricted to ``curr_cands``) as an ascii table (using tabulate).

        :param cmap: the candidate map (overrides the cmap associated with this profile)
        :type cmap: dict[int,str], optional
        :param style: the candidate map to use (overrides the cmap associated with this profile)
        :type style: str ---  "pretty" or "fancy_grid" (or any other style option for tabulate)
        :param curr_cands: list of candidates
        :type curr_cands: list[int], optional
        :rtype: None

        :Example:

        .. exec_code::

            from pref_voting.profiles_with_ties import ProfileWithTies
            prof = ProfileWithTies([{0: 1, 1: 2, 2: 3}, {1:1, 2:1, 0:2}, {2:1, 0:2}], [2, 3, 1])
            prof.display()
            prof.display(cmap={0:"a", 1:"b", 2:"c"})

        """

        _rankings = copy.deepcopy(self.rankings)
        _rankings = [r.normalize_ranks() or r for r in _rankings]
        curr_cands = curr_cands if curr_cands is not None else self.candidates
        cmap = cmap if cmap is not None else self.cmap

        print(
            tabulate(
                [
                    [
                        " ".join(
                            [
                                str(cmap[c])
                                for c in r.cands_at_rank(rank)
                                if c in curr_cands
                            ]
                        )
                        for r in _rankings
                    ]
                    for rank in self.ranks
                ],
                self.rcounts,
                tablefmt=style,
            )
        )

    def display_margin_graph(self, cmap=None, curr_cands=None):
        """
        Display the margin graph of the profile (restricted to ``curr_cands``) using the ``cmap``.  See :class:`.MarginGraph`.
        """

        cmap = cmap if cmap is not None else self.cmap
        MarginGraph.from_profile(self, cmap=cmap).display(curr_cands=curr_cands)

    def display_support_graph(self, cmap=None, curr_cands=None):
        """
        Display the support graph of the profile (restricted to ``curr_cands``) using the ``cmap``.  See :class:`.SupportGraph`.
        """

        cmap = cmap if cmap is not None else self.cmap
        SupportGraph.from_profile(self, cmap=cmap).display(curr_cands=curr_cands)
