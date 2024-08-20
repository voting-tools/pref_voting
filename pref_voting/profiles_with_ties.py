"""
    File: profiles_with_ties.py
    Author: Wes Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)
    Date: January 5, 2022

    A class that represents profiles of (truncated) strict weak orders.
"""

from math import ceil
import copy
import numpy as np
from tabulate import tabulate
from pref_voting.profiles import Profile
from pref_voting.rankings import Ranking
from pref_voting.scoring_methods import symmetric_borda_scores
from pref_voting.weighted_majority_graphs import (
    MajorityGraph,
    MarginGraph,
    SupportGraph,
)
import os
import pandas as pd

class ProfileWithTies(object):
    """An anonymous profile of (truncated) strict weak orders of :math:`n` candidates. 

    :param rankings: List of rankings in the profile, where a ranking is either a :class:`Ranking` object or a dictionary.
    :type rankings: list[dict[int or str: int]] or list[Ranking]
    :param rcounts: List of the number of voters associated with each ranking.  Should be the same length as rankings.  If not provided, it is assumed that 1 voters submitted each element of ``rankings``.
    :type rcounts: list[int], optional
    :param candidates: List of candidates in the profile.  If not provided, this is the list that is ranked by at least on voter.
    :type candidates: list[int] or list[str], optional
    :param cmap: Dictionary mapping candidates (integers) to candidate names (strings).  If not provided, each candidate name is mapped to itself.
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
        

        get_cands = lambda r: list(r.keys()) if type(r) == dict else r.cands
        self.candidates = (
            sorted(candidates)
            if candidates is not None
            else sorted(list(set([c for r in rankings for c in get_cands(r)])))
        )
        """The candidates in the profile. """

        self.num_cands = len(self.candidates)
        """The number of candidates in the profile."""

        self.cmap = cmap if cmap is not None else {c: c for c in self.candidates}
        """The candidate map is a dictionary associating a candidate with the name used when displaying a candidate."""

        self._rankings = [
            Ranking(r, cmap=self.cmap)
            if type(r) == dict
            else Ranking(r.rmap, cmap=self.cmap)
            for r in rankings
        ]
        """The list of rankings in the Profile (each ranking is a :class:`Ranking` object). 
        """

        self.ranks = list(range(1, self.num_cands + 1))
        """The ranks that are possible in the profile. """

        self.cindices = list(range(self.num_cands))
        self._cand_to_cindex = {c: i for i, c in enumerate(self.candidates)}
        self.cand_to_cindex = lambda c: self._cand_to_cindex[c]
        self._cindex_to_cand = {i: c for i, c in enumerate(self.candidates)}
        self.cindex_to_cand = lambda i: self._cindex_to_cand[i]
        """Maps candidates to their index in the list of candidates and vice versa. """
    
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
                    for r, n in zip(self._rankings, self.rcounts)
                    if r.strict_pref(c1, c2)
                )
                for c2 in self.candidates
            }
            for c1 in self.candidates
        }

    def use_extended_strict_preference(self):
        """
        Redefine the supports so that *extended strict preferences* are used. Using extended strict preference may change the margins between candidates.
        """

        self.using_extended_strict_preference = True
        self._supports = {
            c1: {
                c2: sum(
                    n
                    for r, n in zip(self._rankings, self.rcounts)
                    if r.extended_strict_pref(c1, c2)
                )
                for c2 in self.candidates
            }
            for c1 in self.candidates
        }

    def use_strict_preference(self):
        """
        Redefine the supports so that strict preferences are used. Using strict preference may change the margins between candidates.
        """

        self.using_extended_strict_preference = False
        self._supports = {
            c1: {
                c2: sum(
                    n
                    for r, n in zip(self._rankings, self.rcounts)
                    if r.strict_pref(c1, c2)
                )
                for c2 in self.candidates
            }
            for c1 in self.candidates
        }
    @property 
    def rankings(self): 
        """
        Return a list of all individual rankings in the profile. 
        """
        
        return [r for ridx,r in enumerate(self._rankings) 
                for _ in range(self.rcounts[ridx])]

    @property 
    def rankings_as_indifference_list(self): 
        """
        Return a list of all individual rankings as indifference lists in the profile. 
        """
        
        return [r.to_indiff_list() for ridx,r in enumerate(self._rankings) 
                for _ in range(self.rcounts[ridx])]

    @property 
    def ranking_types(self): 
        """
        Return a list of the types of rankings in the profile. 
        """
        
        unique_rankings = []
        for r in self._rankings: 
            if r not in unique_rankings: 
                unique_rankings.append(r)
        return unique_rankings
    
    @property
    def rankings_counts(self):
        """
        Returns the rankings and the counts of each ranking.
        """

        return self._rankings, self.rcounts

    @property
    def rankings_as_dicts_counts(self):
        """
        Returns the rankings represented as dictionaries and the counts of each ranking.
        """

        return [r.rmap for r in self._rankings], self.rcounts

    def support(self, c1, c2):
        """
        Returns the support of candidate ``c1`` over candidate ``c2``, where the support is the number of voters that rank ``c1`` strictly above ``c2``.
        """

        return self._supports[c1][c2]

    def margin(self, c1, c2):
        """
        Returns the margin of candidate ``c1`` over candidate ``c2``, where the margin is the number of voters that rank ``c1`` strictly above ``c2`` minus the number of voters that rank ``c2`` strictly above ``c1``.
        """

        return self._supports[c1][c2] - self._supports[c2][c1]

    @property
    def margin_matrix(self):
        """Returns the margin matrix of the profile, where the entry at row ``i`` and column ``j`` is the margin of candidate ``i`` over candidate ``j``."""

        return np.array(
            [[self.margin(self.cindex_to_cand(c1_idx), self.cindex_to_cand(c2_idx)) for c2_idx in self.cindices] for c1_idx in self.cindices]
        )
    
    def is_tied(self, c1, c2): 
        """Returns True if ``c1`` and ``c2`` are tied (i.e., the margin of ``c1`` over ``c2`` is 0)."""

        return self.margin(c1, c2) == 0

    def dominators(self, cand, curr_cands=None):
        """
        Returns the list of candidates that are majority preferred to ``cand`` in the profile restricted to the candidates in ``curr_cands``.
        """
        candidates = self.candidates if curr_cands is None else curr_cands

        return [c for c in candidates if self.majority_prefers(c, cand)]

    def dominates(self, cand, curr_cands=None):
        """
        Returns the list of candidates that ``cand`` is majority preferred to in the majority graph restricted to ``curr_cands``.
        """
        candidates = self.candidates if curr_cands is None else curr_cands

        return [c for c in candidates if self.majority_prefers(cand, c)]

    def ratio(self, c1, c2):
        """
        Returns the ratio of the support of ``c1`` over ``c2`` to the support ``c2`` over ``c1``.
        """

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

    def strength_matrix(self, curr_cands = None, strength_function = None): 
        """
        Return the strength matrix of the profile.  The strength matrix is a matrix where the entry in row :math:`i` and column :math:`j` is the number of voters that rank the candidate with index :math:`i` over the candidate with index :math:`j`.  If ``curr_cands`` is provided, then the strength matrix is restricted to the candidates in ``curr_cands``.  If ``strength_function`` is provided, then the strength matrix is computed using the strength function."""
        
        if curr_cands is not None: 
            cindices = [cidx for cidx, _ in enumerate(curr_cands)]
            cindex_to_cand = lambda cidx: curr_cands[cidx]
            cand_to_cindex = lambda c: cindices[curr_cands.index(c)]
            strength_function = self.margin if strength_function is None else strength_function
            strength_matrix = np.array([[strength_function(cindex_to_cand(a_idx), cindex_to_cand(b_idx)) for b_idx in cindices] for a_idx in cindices])
        else:  
            cindices = self.cindices
            cindex_to_cand = self.cindex_to_cand
            cand_to_cindex = self.cand_to_cindex
            strength_matrix = np.array(self.margin_matrix) if strength_function is None else np.array([[strength_function(cindex_to_cand(a_idx), cindex_to_cand(b_idx)) for b_idx in cindices] for a_idx in cindices])

        return strength_matrix, cand_to_cindex

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

    def plurality_scores(self, curr_cands=None):
        """
        Return the Plurality Scores of the candidates, assuming that each voter ranks a single candidate in first place.

        Parameters:
        - curr_cands: List of current candidates to consider. If None, use all candidates.

        Returns:
        - Dictionary with candidates as keys and their plurality scores as values.

        Raises:
        - ValueError: If any voter ranks multiple candidates in first place.
        """

        if curr_cands is None:
            curr_cands = self.candidates

        # Check if any voter ranks multiple candidates in first place
        if any(len(r.first(cs=curr_cands)) > 1 for r in self._rankings):
            raise ValueError("Cannot find the plurality scores unless all voters rank a unique candidate in first place.")

        rankings, rcounts = self.rankings_counts

        plurality_scores = {cand: 0 for cand in curr_cands}

        for ranking, count in zip(rankings, rcounts):
            first_place_candidates = ranking.first(cs=curr_cands)
            if len(first_place_candidates) == 1:
                cand = first_place_candidates[0]
                plurality_scores[cand] += count

        return plurality_scores

    def plurality_scores_ignoring_overvotes(self, curr_cands=None): 
        """
        Return the Plurality scores ignoring empty rankings and overvotes.
        """

        curr_cands = curr_cands if curr_cands is not None else self.candidates
        
        rankings, rcounts = self.rankings_counts
        
        return {cand: sum([c for r, c in zip(rankings, rcounts) if len(r.cands) > 0 and [cand] == r.first(cs=curr_cands)]) for cand in curr_cands}

    def borda_scores(self, 
                     curr_cands=None, 
                     borda_score_fnc=symmetric_borda_scores):
        
        curr_cands = self.candidates if curr_cands is None else curr_cands
        restricted_prof = self.remove_candidates([c for c in self.candidates if c not in curr_cands])
        return borda_score_fnc(restricted_prof)

    def tops_scores(
            self, 
            curr_cands=None, 
            score_type='approval'):
        """
        Return the tops scores of the candidates. 

        Parameters:
        - curr_cands: List of current candidates to consider. If None, use all candidates.
        - score_type: Type of tops score to compute. Options are 'approval' or 'split'.

        Returns:
        - Dictionary with candidates as keys and their tops scores as values.
        """

        if curr_cands is None:
            curr_cands = self.candidates

        rankings, rcounts = self.rankings_counts

        if score_type not in {'approval', 'split'}:
            raise ValueError("Invalid score_type specified. Use 'approval' or 'split'.")

        tops_scores = {cand: 0 for cand in curr_cands}

        if score_type == 'approval':
            for ranking, count in zip(rankings, rcounts):
                for cand in curr_cands:
                    if cand in ranking.first(cs=curr_cands):
                        tops_scores[cand] += count

        elif score_type == 'split':
            for ranking, count in zip(rankings, rcounts):
                for cand in curr_cands:
                    if cand in ranking.first(cs=curr_cands):
                        tops_scores[cand] += count * 1/len(ranking.first(cs=curr_cands))

        return tops_scores
        
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

        self._rankings = new_rankings
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

    def add_unranked_candidates(self): 
        """
        Return a profile in which for each voter, any unranked candidate is added to the bottom of their ranking. 
        """
        cands = self.candidates
        ranks = list()
        rcounts = list()

        for r in self._rankings: 
            min_rank = max(r.ranks) if len(r.ranks) > 0 else 1   
            new_r ={c:r for c, r in  r.rmap.items()}
            for c in cands: 
                if c not in new_r.keys(): 
                    new_r[c] = min_rank+1
            new_ranking = Ranking(new_r)

            found_it = False
            for _ridx, _r in enumerate(ranks):
                if new_ranking == _r: 
                    rcounts[_ridx] += 1
                    found_it = True
            if not found_it: 
                ranks.append(new_ranking)
                rcounts.append(1)

        return ProfileWithTies([r.rmap for r in ranks], rcounts=rcounts, cmap=self.cmap)

    @property
    def is_truncated_linear(self):
        """
        Return True if the profile only contains (truncated) linear orders.
        """
        return all([r.is_truncated_linear(len(self.candidates)) or r.is_linear(len(self.candidates)) for r in self._rankings])
    
    def to_linear_profile(self):
        """Return a linear profile from the profile with ties. If the profile is not a linear profile, then return None. 
        
        Note that the candidates in a Profile must be integers, so the candidates in the linear profile will be the indices of the candidates in the original profile.
        
        """
        rankings, rcounts = self.rankings_counts
        _new_rankings = [r.to_linear() for r in rankings]
        cand_to_cindx = {c:i for i,c in enumerate(sorted(self.candidates))}
        new_cmap = {cand_to_cindx[c]: self.cmap[c] for c in sorted(self.candidates)}
        if any([r is None or len(r) != len(self.candidates) for r in _new_rankings]): 
            print("Error: Cannot convert to linear profile.")
            return None
        new_rankings = [tuple([cand_to_cindx[c] for c in r]) for r in _new_rankings]
        return Profile(new_rankings, rcounts=rcounts, cmap=new_cmap)
                
    def margin_graph(self):
        """Returns the margin graph of the profile.  See :class:`.MarginGraph`.

        :Example:

        .. exec_code:: python

                from pref_voting.profiles_with_ties import ProfileWithTies
                prof = ProfileWithTies([{0: 1, 1: 2, 2: 3}, {1:1, 2:1, 0:2}, {2:1, 0:2}], [2, 3, 1])

                mg = prof.margin_graph()
                print(mg.edges)
                print(mg.margin_matrix)
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

    def is_uniquely_weighted(self): 
        """Returns True if the profile is uniquely weighted. 
        
        A profile is **uniquely weighted** when there are no 0 margins and all the margins between any two candidates are unique.     
        """
        
        return MarginGraph.from_profile(self).is_uniquely_weighted()

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
            for rank in self._rankings
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
            print(f"{r}: {c}")


    def anonymize(self): 
        """
        Return a profile which is the anonymized version of this profile. 
        """

        rankings = list()
        rcounts = list()
        for r in self.rankings:
            found_it = False
            for _ridx, _r in enumerate(rankings): 
                if r == _r: 
                    rcounts[_ridx] += 1
                    found_it = True
                    break
            if not found_it: 
                rankings.append(r)
                rcounts.append(1)
                
        prof = ProfileWithTies(rankings, rcounts=rcounts, cmap=self.cmap)

        if self.using_extended_strict_preference: 
            prof.use_extended_strict_preference()

        return prof

    def description(self): 
        """
        Return the Python code needed to create the profile.
        """
        return f"ProfileWithTies({[r.rmap for r in self._rankings]}, rcounts={[int(c) for c in self.rcounts]}, cmap={self.cmap})"

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

        _rankings = copy.deepcopy(self._rankings)
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

    def to_preflib_instance(self):
        """
        Returns an instance of the ``OrdinalInstance`` class from the ``preflibtools`` package. See ``pref_voting.io.writers.to_preflib_instance``.
        
        """
        from pref_voting.io.writers import to_preflib_instance

        return to_preflib_instance(self)

    @classmethod
    def from_preflib(
        cls, 
        instance_or_preflib_file, 
        include_cmap=False): 
        """
        Convert an preflib OrdinalInstance or file to a Profile.   See ``pref_voting.io.readers.from_preflib``.
        
        """
        from pref_voting.io.readers import preflib_to_profile

        return preflib_to_profile(
            instance_or_preflib_file, 
            include_cmap=include_cmap, 
            as_linear_profile=False)

    def write(
            self, 
            filename, 
            file_format="preflib", 
            csv_format="candidate_columns"):
        """
        Write a profile to a file.   See ``pref_voting.io.writers.write``.
        """
        from pref_voting.io.writers import  write
 
        return write(
            self, 
            filename, 
            file_format=file_format, 
            csv_format=csv_format)

    @classmethod
    def read(
        cls, 
        filename, 
        file_format="preflib",
        csv_format="candidate_columns",
        cand_type=None,
        items_to_skip=None): 
        """
        Read a profile from a file.  See ``pref_voting.io.readers.read``.
        
        """
        from pref_voting.io.readers import  read

        return read(
            filename, 
            file_format=file_format, 
            csv_format=csv_format,
            cand_type=cand_type,
            items_to_skip=items_to_skip,
            as_linear_profile=False,
            )


    def __eq__(self, other_prof): 
        """
        Returns true if two profiles are equal.  Two profiles are equal if they have the same rankings.  Note that we ignore the cmaps. 
        """

        rankings = self.rankings
        other_rankings = other_prof.rankings[:] # make a copy
        for r1 in rankings:
            for i, r2 in enumerate(other_rankings):
                if r1 == r2:   
                    # Remove the matched item to handle duplicates
                    del other_rankings[i]
                    break
            else:
                # If we didn't find a match for r1, the profiles are not identical
                return False
    
        return not other_rankings
    

    def __add__(self, other_prof): 
        """
        Returns the sum of two profiles.  The sum of two profiles is the profile that contains all the rankings from the first in addition to all the rankings from the second profile. 

        Note: the cmaps of the profiles are ignored. 
        """

        return ProfileWithTies(self._rankings + other_prof._rankings, rcounts=self.rcounts + other_prof.rcounts, candidates = sorted(list(set(self.candidates +other_prof.candidates))))
