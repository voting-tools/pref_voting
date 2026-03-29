'''
    File: pref_grade_profile.py
    Author: Wes Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)
    Date: March 28, 2026
    
    A class that represents profiles in which each voter submits both a 
    (truncated) strict weak order and an assignment of grades.
'''

import copy
import numpy as np
from math import ceil
from tabulate import tabulate
from pref_voting.rankings import Ranking
from pref_voting.mappings import Grade, _Mapping
from pref_voting.profiles import Profile
from pref_voting.profiles_with_ties import ProfileWithTies
from pref_voting.grade_profiles import GradeProfile
from pref_voting.scoring_methods import symmetric_borda_scores
from pref_voting.weighted_majority_graphs import (
    MajorityGraph,
    MarginGraph,
    SupportGraph,
)
import pandas as pd
import matplotlib.pyplot as plt


class PrefGradeProfile(object):
    """An anonymous profile in which each voter submits both a (truncated) strict weak order
    and an assignment of grades.

    :param rankings: List of rankings in the profile, where a ranking is either a :class:`Ranking` object or a dictionary.
    :type rankings: list[dict[int or str: int]] or list[Ranking]
    :param grade_maps: List of grades in the profile, where a grade is either a :class:`Grade` object or a dictionary.
    :type grade_maps: list[dict[int or str: int or str]] or list[Grade]
    :param grades: List of grades.
    :type grades: list[int or str]
    :param rcounts: List of the number of voters associated with each ranking/grade pair.  Should be the same length as rankings and grade_maps.  If not provided, it is assumed that 1 voter submitted each element.
    :type rcounts: list[int], optional
    :param candidates: List of candidates in the profile.  If not provided, this is the union of candidates appearing in the rankings and grade maps.
    :type candidates: list[int] or list[str], optional
    :param cmap: Dictionary mapping candidates to candidate names (strings).  If not provided, each candidate name is mapped to itself.
    :type cmap: dict[int or str: str], optional
    :param gmap: Dictionary mapping grades to grade names (strings).  If not provided, each grade is mapped to itself.
    :type gmap: dict[int or str: str], optional
    :param grade_order: A list of the grades representing the order of the grades. It is assumed the grades are listed from largest to smallest.  If not provided, the grades are assumed to be numbers and compared using the greater-than relation.
    :type grade_order: list[int or str], optional

    :Example:

    The following code creates a profile in which 2 voters submit the ranking
    0 first, 1 second, 2 third along with grades {0: 5, 1: 3, 2: 1};
    and 3 voters submit the ranking with 1 and 2 tied for first and 0 second
    along with grades {0: 2, 1: 4, 2: 4}:

    .. code-block:: python

        pgprof = PrefGradeProfile(
            [{0: 1, 1: 2, 2: 3}, {1: 1, 2: 1, 0: 2}],
            [{0: 5, 1: 3, 2: 1}, {0: 2, 1: 4, 2: 4}],
            [1, 2, 3, 4, 5],
            rcounts=[2, 3],
        )

        pgprof.display()

    """

    def __init__(
        self,
        rankings,
        grade_maps,
        grades,
        rcounts=None,
        candidates=None,
        cmap=None,
        gmap=None,
        grade_order=None,
    ):
        """Constructor method"""

        assert len(rankings) == len(
            grade_maps
        ), "The number of rankings must be the same as the number of grade maps"

        assert rcounts is None or len(rankings) == len(
            rcounts
        ), "The number of rankings must be the same as the number of rcounts"

        # Determine candidates from both rankings and grade_maps
        get_cands_ranking = lambda r: list(r.keys()) if type(r) == dict else r.cands
        get_cands_grade = lambda g: list(g.keys()) if type(g) == dict else g.graded_candidates

        if candidates is not None:
            self.candidates = sorted(candidates)
        else:
            ranking_cands = set([c for r in rankings for c in get_cands_ranking(r)])
            grade_cands = set([c for g in grade_maps for c in get_cands_grade(g)])
            self.candidates = sorted(list(ranking_cands | grade_cands))
        """The candidates in the profile. """

        self.num_cands = len(self.candidates)
        """The number of candidates in the profile."""

        self.cmap = cmap if cmap is not None else {c: str(c) for c in self.candidates}
        """The candidate map is a dictionary associating a candidate with the name used when displaying a candidate."""

        # --- Ranking data (from ProfileWithTies) ---

        self._rankings = [
            Ranking(r, cmap=self.cmap)
            if type(r) == dict
            else Ranking(r.rmap, cmap=self.cmap)
            for r in rankings
        ]
        """The list of rankings in the profile (each ranking is a :class:`Ranking` object)."""

        self.ranks = list(range(1, self.num_cands + 1))
        """The ranks that are possible in the profile. """

        self.cindices = list(range(self.num_cands))
        self._cand_to_cindex = {c: i for i, c in enumerate(self.candidates)}
        self.cand_to_cindex = lambda c: self._cand_to_cindex[c]
        self._cindex_to_cand = {i: c for i, c in enumerate(self.candidates)}
        self.cindex_to_cand = lambda i: self._cindex_to_cand[i]
        """Maps candidates to their index in the list of candidates and vice versa. """

        # --- Grade data (from GradeProfile) ---

        self.grades = grades
        """The grades in the profile. """

        self.can_sum_grades = all([isinstance(g, (float, int)) for g in self.grades])

        self.grade_order = grade_order if grade_order is not None else sorted(self.grades, reverse=True)
        """The order of the grades. If None, then order from largest to smallest"""

        self.use_grade_order = grade_order is not None

        self.compare_function = (
            (lambda v1, v2: (v1 > v2) - (v2 > v1))
            if grade_order is None
            else (lambda v1, v2: (grade_order.index(v1) < grade_order.index(v2)) - (grade_order.index(v2) < grade_order.index(v1)))
        )

        self.gmap = gmap if gmap is not None else {g: str(g) for g in self.grades}
        """The grade map is a dictionary associating a grade with the name used when displaying a grade."""

        self._grades = [
            Grade(
                g_map,
                self.grades,
                candidates=self.candidates,
                cmap=self.cmap,
                gmap=self.gmap,
                compare_function=self.compare_function,
            )
            if type(g_map) == dict
            else Grade(
                g_map.as_dict(),
                self.grades,
                candidates=self.candidates,
                cmap=self.cmap,
                gmap=self.gmap,
                compare_function=self.compare_function,
            )
            for g_map in grade_maps
        ]
        """The list of grades in the profile (each grade is a :class:`Grade` object)."""

        # --- Shared data ---

        self.rcounts = [1] * len(rankings) if rcounts is None else list(rcounts)

        self.num_voters = np.sum(self.rcounts)
        """The number of voters in the profile. """

        self.using_extended_strict_preference = False
        """A flag indicating whether the profile is using extended strict preferences when calculating supports, margins, etc."""

        # memoize the supports (based on rankings)
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

    # =========================================================================
    # Ranking-related methods (from ProfileWithTies)
    # =========================================================================

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

        return [
            r
            for ridx, r in enumerate(self._rankings)
            for _ in range(self.rcounts[ridx])
        ]

    @property
    def rankings_as_indifference_list(self):
        """
        Return a list of all individual rankings as indifference lists in the profile.
        """

        return [
            r.to_indiff_list()
            for ridx, r in enumerate(self._rankings)
            for _ in range(self.rcounts[ridx])
        ]

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
            [
                [
                    self.margin(
                        self.cindex_to_cand(c1_idx), self.cindex_to_cand(c2_idx)
                    )
                    for c2_idx in self.cindices
                ]
                for c1_idx in self.cindices
            ]
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

    def strength_matrix(self, curr_cands=None, strength_function=None):
        """
        Return the strength matrix of the profile.  The strength matrix is a matrix where the entry in row :math:`i` and column :math:`j` is the number of voters that rank the candidate with index :math:`i` over the candidate with index :math:`j`.  If ``curr_cands`` is provided, then the strength matrix is restricted to the candidates in ``curr_cands``.  If ``strength_function`` is provided, then the strength matrix is computed using the strength function.
        """

        if curr_cands is not None:
            cindices = [cidx for cidx, _ in enumerate(curr_cands)]
            cindex_to_cand = lambda cidx: curr_cands[cidx]
            cand_to_cindex = lambda c: cindices[curr_cands.index(c)]
            strength_function = (
                self.margin if strength_function is None else strength_function
            )
            strength_matrix = np.array(
                [
                    [
                        strength_function(
                            cindex_to_cand(a_idx), cindex_to_cand(b_idx)
                        )
                        for b_idx in cindices
                    ]
                    for a_idx in cindices
                ]
            )
        else:
            cindices = self.cindices
            cindex_to_cand = self.cindex_to_cand
            cand_to_cindex = self.cand_to_cindex
            strength_matrix = (
                np.array(self.margin_matrix)
                if strength_function is None
                else np.array(
                    [
                        [
                            strength_function(
                                cindex_to_cand(a_idx), cindex_to_cand(b_idx)
                            )
                            for b_idx in cindices
                        ]
                        for a_idx in cindices
                    ]
                )
            )

        return strength_matrix, cand_to_cindex

    def condorcet_winner(self, curr_cands=None):
        """Returns the Condorcet winner in the profile restricted to ``curr_cands`` if one exists, otherwise return None.

        The **Condorcet winner** is the candidate that is majority preferred to every other candidate.
        """
        curr_cands = curr_cands if curr_cands is not None else self.candidates

        cw = None
        for c in curr_cands:
            if all(
                [self.majority_prefers(c, c1) for c1 in curr_cands if c1 != c]
            ):
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
            if all(
                [self.majority_prefers(c2, c1) for c2 in curr_cands if c1 != c2]
            ):
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
                [
                    self.majority_prefers(c2, c1)
                    for c2 in curr_cands
                    if c1 != c2
                ]
            ):
                weak_cw.append(c1)
        return sorted(weak_cw) if len(weak_cw) > 0 else None

    def copeland_scores(self, curr_cands=None, scores=(1, 0, -1)):
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
        """

        if curr_cands is None:
            curr_cands = self.candidates

        if any(len(r.first(cs=curr_cands)) > 1 for r in self._rankings):
            raise ValueError(
                "Cannot find the plurality scores unless all voters rank a unique candidate in first place."
            )

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

        return {
            cand: sum(
                [
                    c
                    for r, c in zip(rankings, rcounts)
                    if len(r.cands) > 0 and [cand] == r.first(cs=curr_cands)
                ]
            )
            for cand in curr_cands
        }

    def borda_scores(self, curr_cands=None, borda_score_fnc=symmetric_borda_scores):

        curr_cands = self.candidates if curr_cands is None else curr_cands
        restricted_prof = self.to_ranking_profile().remove_candidates(
            [c for c in self.candidates if c not in curr_cands]
        )
        return borda_score_fnc(restricted_prof)

    def tops_scores(self, curr_cands=None, score_type="approval"):
        """
        Return the tops scores of the candidates.
        """

        if curr_cands is None:
            curr_cands = self.candidates

        rankings, rcounts = self.rankings_counts

        if score_type not in {"approval", "split"}:
            raise ValueError(
                "Invalid score_type specified. Use 'approval' or 'split'."
            )

        tops_scores = {cand: 0 for cand in curr_cands}

        if score_type == "approval":
            for ranking, count in zip(rankings, rcounts):
                for cand in curr_cands:
                    if cand in ranking.first(cs=curr_cands):
                        tops_scores[cand] += count

        elif score_type == "split":
            for ranking, count in zip(rankings, rcounts):
                for cand in curr_cands:
                    if cand in ranking.first(cs=curr_cands):
                        tops_scores[cand] += (
                            count * 1 / len(ranking.first(cs=curr_cands))
                        )

        return tops_scores

    def remove_empty_rankings(self):
        """
        Remove the empty rankings from the profile.
        """
        new_rankings = list()
        new_grades = list()
        new_rcounts = list()

        for r, g, c in zip(self._rankings, self._grades, self.rcounts):

            if len(r.cands) != 0:
                new_rankings.append(r)
                new_grades.append(g)
                new_rcounts.append(c)

        self._rankings = new_rankings
        self._grades = new_grades
        self.rcounts = new_rcounts

        # update the number of voters
        self.num_voters = np.sum(self.rcounts)

        if self.using_extended_strict_preference:
            self.use_extended_strict_preference()
        else:
            self.use_strict_preference()

    @property
    def is_truncated_linear(self):
        """
        Return True if the profile only contains (truncated) linear orders.
        """
        return all(
            [
                r.is_truncated_linear(len(self.candidates))
                or r.is_linear(len(self.candidates))
                for r in self._rankings
            ]
        )

    def num_bullet_votes(self):
        """
        Return the number of bullet votes in the profile.
        """

        return sum(
            [c for r, c in zip(*self.rankings_counts) if r.is_bullet_vote()]
        )

    def num_empty_rankings(self):
        """
        Return the number of empty rankings in the profile.
        """

        return sum(
            [c for r, c in zip(*self.rankings_counts) if r.is_empty()]
        )

    def num_linear_orders(self):
        """
        Return the number of linear orders in the profile.
        """

        return sum(
            [
                c
                for r, c in zip(*self.rankings_counts)
                if r.is_linear(len(self.candidates))
            ]
        )

    def num_truncated_linear_orders(self):
        """
        Return the number of truncated linear orders in the profile.
        """

        return sum(
            [
                c
                for r, c in zip(*self.rankings_counts)
                if r.is_truncated_linear(len(self.candidates))
            ]
        )

    def num_rankings_with_ties(self):
        """
        Return the number of rankings with ties in the profile.
        """

        return sum(
            [c for r, c in zip(*self.rankings_counts) if r.has_tie()]
        )

    def num_ranked_all_candidates(self):
        """
        Return the number of rankings that rank all candidates in the profile.
        """

        return sum(
            [
                c
                for r, c in zip(*self.rankings_counts)
                if all([r.is_ranked(cand) for cand in self.candidates])
            ]
        )

    def num_ranking_each_candidate(self):
        """Return a dictionary mapping each candidate to the number of voters that rank the candidate."""

        return {
            cand: sum(
                [
                    c
                    for r, c in zip(*self.rankings_counts)
                    if r.is_ranked(cand)
                ]
            )
            for cand in self.candidates
        }

    def margin_graph(self):
        """Returns the margin graph of the profile.  See :class:`.MarginGraph`."""

        return MarginGraph.from_profile(self)

    def support_graph(self):
        """Returns the support graph of the profile.  See :class:`.SupportGraph`."""

        return SupportGraph.from_profile(self)

    def majority_graph(self):
        """Returns the majority graph of the profile.  See :class:`.MajorityGraph`."""

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
        """

        updated_rankings = [
            {c: r for c, r in rank.rmap.items() if c not in cands_to_ignore}
            for rank in self._rankings
        ]
        updated_grade_maps = [
            {c: g.val(c) for c in g.graded_candidates if c not in cands_to_ignore}
            for g in self._grades
        ]
        new_candidates = [
            c for c in self.candidates if c not in cands_to_ignore
        ]

        restricted_prof = PrefGradeProfile(
            updated_rankings,
            updated_grade_maps,
            self.grades,
            rcounts=self.rcounts,
            candidates=new_candidates,
            cmap=self.cmap,
            gmap=self.gmap,
            grade_order=self.grade_order if self.use_grade_order else None,
        )

        if self.using_extended_strict_preference:
            restricted_prof.use_extended_strict_preference()

        return restricted_prof

    # =========================================================================
    # Grade-related methods (from GradeProfile)
    # =========================================================================

    @property
    def grades_counts(self):
        """Returns the grades and the counts of each grade."""

        return self._grades, self.rcounts

    @property
    def grade_functions(self):
        """Return all of the grade functions in the profile."""

        gs = list()
        for g, c in zip(self._grades, self.rcounts):
            gs += [g] * c
        return gs

    def has_grade(self, c):
        """Return True if ``c`` is assigned a grade by at least one voter."""

        return any([g.has_grade(c) for g in self._grades])

    def grade_margin(self, c1, c2, use_extended=False):
        """
        Return the grade-based margin of ``c1`` over ``c2``.
        """
        if use_extended:
            return np.sum(
                [
                    num
                    for g, num in zip(*self.grades_counts)
                    if g.extended_strict_pref(c1, c2)
                ]
            ) - np.sum(
                [
                    num
                    for g, num in zip(*self.grades_counts)
                    if g.extended_strict_pref(c2, c1)
                ]
            )
        else:
            return np.sum(
                [
                    num
                    for g, num in zip(*self.grades_counts)
                    if g.strict_pref(c1, c2)
                ]
            ) - np.sum(
                [
                    num
                    for g, num in zip(*self.grades_counts)
                    if g.strict_pref(c2, c1)
                ]
            )

    def proportion(self, cand, grade):
        """
        Return the proportion of voters that assign ``cand`` the grade ``grade``.

        Note that ``grade`` could be None, in which case the proportion of voters that do not assign ``cand`` a grade is returned.
        """
        return (
            np.sum(
                [
                    num
                    for g, num in zip(*self.grades_counts)
                    if g(cand) == grade
                ]
            )
            / self.num_voters
        )

    def sum(self, c):
        """Return the sum of the grades of ``c``.  If ``c`` is not assigned a grade by any voter, return None."""

        assert self.can_sum_grades, "The grades in the profile cannot be summed."

        return (
            np.sum(
                [
                    g(c) * num
                    for g, num in zip(*self.grades_counts)
                    if g.has_grade(c)
                ]
            )
            if self.has_grade(c)
            else None
        )

    def avg(self, c):
        """Return the average of the grades of ``c``.  If ``c`` is not assigned a grade by any voter, return None."""

        assert self.can_sum_grades, "The grades in the profile cannot be summed."

        return (
            np.mean([g(c) for g in self.grade_functions if g.has_grade(c)])
            if self.has_grade(c)
            else None
        )

    def max(self, c):
        """Return the maximum of the grade of ``c``.  If ``c`` is not assigned a grade by any voter, return None."""

        grades_for_c = (
            [-1 * self.grade_order.index(g(c)) for g in self._grades if g.has_grade(c)]
            if self.use_grade_order
            else [g(c) for g in self._grades if g.has_grade(c)]
        )

        return (
            (
                self.grade_order[-1 * max(grades_for_c)]
                if self.use_grade_order
                else max(grades_for_c)
            )
            if self.has_grade(c)
            else None
        )

    def min(self, c):
        """Return the minimum of the grades of ``c``.  If ``c`` is not assigned a grade by any voter, return None."""

        grades_for_c = (
            [-1 * self.grade_order.index(g(c)) for g in self._grades if g.has_grade(c)]
            if self.use_grade_order
            else [g(c) for g in self._grades if g.has_grade(c)]
        )

        return (
            (
                self.grade_order[-1 * min(grades_for_c)]
                if self.use_grade_order
                else min(grades_for_c)
            )
            if self.has_grade(c)
            else None
        )

    def median(self, c, use_lower=True, use_average=False):
        """Return the median of the grades of ``c``.  If ``c`` is not assigned a grade by any voter, return None."""

        grades_for_c = (
            [
                -1 * self.grade_order.index(g(c))
                for g in self.grade_functions
                if g.has_grade(c)
            ]
            if self.use_grade_order
            else [g(c) for g in self.grade_functions if g.has_grade(c)]
        )

        sorted_grades_for_c = sorted(grades_for_c)
        num_grades = len(sorted_grades_for_c)
        median_idx = num_grades // 2
        if num_grades % 2 == 0:
            median_grades = sorted_grades_for_c[median_idx - 1 : median_idx + 1]
        else:
            median_grades = [sorted_grades_for_c[median_idx]]

        if use_lower:
            return (
                (
                    self.grade_order[-1 * median_grades[0]]
                    if self.use_grade_order
                    else median_grades[0]
                )
                if self.has_grade(c)
                else None
            )
        elif use_average:
            return (
                (
                    np.average(
                        [self.grade_order[-1 * m] for m in median_grades]
                    )
                    if self.use_grade_order
                    else np.average(median_grades)
                )
                if self.has_grade(c)
                else None
            )
        else:
            return (
                (
                    [self.grade_order[-1 * m] for m in median_grades]
                    if self.use_grade_order
                    else median_grades
                )
                if self.has_grade(c)
                else None
            )

    def sum_grade_function(self):
        """Return the sum grade function of the profile."""

        assert self.can_sum_grades, "The grades in the profile cannot be summed."

        return _Mapping(
            {c: self.sum(c) for c in self.candidates if self.has_grade(c)},
            domain=self.candidates,
            item_map=self.cmap,
            compare_function=self.compare_function,
        )

    def avg_grade_function(self):
        """Return the average grade function of the profile."""

        assert self.can_sum_grades, "The grades in the profile cannot be summed."

        return _Mapping(
            {c: self.avg(c) for c in self.candidates if self.has_grade(c)},
            domain=self.candidates,
            item_map=self.cmap,
            compare_function=self.compare_function,
        )

    def proportion_with_grade(self, cand, grade):
        """
        Return the proportion of voters that assign a ``grade`` to ``cand``.
        """

        assert (
            cand in self.candidates
        ), f"{cand} is not a candidate in the profile."
        assert grade in self.grades, f"{grade} is not a grade in the profile."

        num_with_higher_grade = 0
        for g, num in zip(*self.grades_counts):
            if self.compare_function(g(cand), grade) == 0:
                num_with_higher_grade += num
        return num_with_higher_grade / self.num_voters

    def proportion_with_higher_grade(self, cand, grade):
        """
        Return the proportion of voters that assign a strictly higher grade to ``cand`` than ``grade``.
        """

        assert (
            cand in self.candidates
        ), f"{cand} is not a candidate in the profile."
        assert grade in self.grades, f"{grade} is not a grade in the profile."

        num_with_higher_grade = 0
        for g, num in zip(*self.grades_counts):
            if self.compare_function(g(cand), grade) == 1:
                num_with_higher_grade += num
        return num_with_higher_grade / self.num_voters

    def proportion_with_lower_grade(self, cand, grade):
        """
        Return the proportion of voters that assign a strictly lower grade to ``cand`` than ``grade``.
        """

        assert (
            cand in self.candidates
        ), f"{cand} is not a candidate in the profile."
        assert grade in self.grades, f"{grade} is not a grade in the profile."

        num_with_lower_grade = 0
        for g, num in zip(*self.grades_counts):
            if self.compare_function(g(cand), grade) == -1:
                num_with_lower_grade += num
        return num_with_lower_grade / self.num_voters

    def approval_scores(self):
        """
        Return a dictionary representing the approval scores of the candidates in the profile.
        """

        assert (
            self.can_sum_grades
        ), "The grades in the profile cannot be summed."
        assert sorted(self.grades) == [
            0,
            1,
        ], "The grades in the profile must be 0 and 1."

        return {c: self.sum(c) for c in self.candidates}

    # =========================================================================
    # Conversion methods
    # =========================================================================

    def to_ranking_profile(self):
        """Return a :class:`ProfileWithTies` corresponding to the ranking data in this profile."""

        return ProfileWithTies(
            self._rankings,
            rcounts=self.rcounts,
            candidates=self.candidates,
            cmap=self.cmap,
        )

    def to_grade_profile(self):
        """Return a :class:`GradeProfile` corresponding to the grade data in this profile."""

        return GradeProfile(
            [g.as_dict() for g in self._grades],
            self.grades,
            gcounts=self.rcounts,
            candidates=self.candidates,
            cmap=self.cmap,
            gmap=self.gmap,
            grade_order=self.grade_order if self.use_grade_order else None,
        )

    # =========================================================================
    # Display and report methods
    # =========================================================================

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
        print(
            f"""There are {len(self.candidates)} candidates and {str(sum(rcounts))} {'ranking: ' if sum(rcounts) == 1 else 'rankings: '} 
        The number of empty rankings: {num_empty_rankings}
        The number of rankings with ties: {num_ties}
        The number of linear orders: {num_linear_orders}
        The number of truncated linear orders: {num_trucated_linear_orders}
        
The number of rankings with skipped ranks: {num_with_skipped_ranks}
        
        """
        )

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

        for r, c in rs.items():
            print(f"{r}: {c}")

    def display(
        self,
        cmap=None,
        style="pretty",
        curr_cands=None,
        show_grades=True,
        show_totals=False,
    ):
        """Display the profile as an ascii table (using tabulate).

        The rankings are displayed as in :class:`ProfileWithTies`. If ``show_grades``
        is True, the grade assignment for each voter type is also displayed below
        the ranking table.

        :param cmap: the candidate map (overrides the cmap associated with this profile)
        :type cmap: dict[int,str], optional
        :param style: the table style for tabulate (default ``"pretty"``)
        :type style: str, optional
        :param curr_cands: list of candidates to display
        :type curr_cands: list[int], optional
        :param show_grades: whether to also display the grade assignments (default True)
        :type show_grades: bool, optional
        :param show_totals: whether to display grade totals (sum, median) when showing grades (default False)
        :type show_totals: bool, optional
        :rtype: None
        """

        _rankings = copy.deepcopy(self._rankings)
        _rankings = [r.normalize_ranks() or r for r in _rankings]
        curr_cands = curr_cands if curr_cands is not None else self.candidates
        cmap = cmap if cmap is not None else self.cmap

        # Display rankings table
        existing_ranks = (
            list(
                range(
                    min(min(r.ranks) for r in _rankings),
                    max(max(r.ranks) for r in _rankings) + 1,
                )
            )
            if len(_rankings) > 0
            else []
        )
        print("Rankings:")
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
                    for rank in existing_ranks
                ],
                self.rcounts,
                tablefmt=style,
            )
        )

        # Display grades table
        if show_grades:
            print("\nGrades:")
            if show_totals:
                sum_grade_fnc = self.sum_grade_function()
                headers = [""] + self.rcounts + ["Sum", "Median"]
                tbl = [
                    [cmap[c]]
                    + [
                        self.gmap[g(c)] if g.has_grade(c) else ""
                        for g in self._grades
                    ]
                    + [sum_grade_fnc(c), self.median(c)]
                    for c in curr_cands
                ]
            else:
                headers = [""] + self.rcounts
                tbl = [
                    [cmap[c]]
                    + [
                        self.gmap[g(c)] if g.has_grade(c) else ""
                        for g in self._grades
                    ]
                    for c in curr_cands
                ]
            print(tabulate(tbl, headers=headers))

    def display_margin_graph(self, cmap=None, curr_cands=None):
        """
        Display the margin graph of the profile (restricted to ``curr_cands``) using the ``cmap``.  See :class:`.MarginGraph`.
        """

        cmap = cmap if cmap is not None else self.cmap
        MarginGraph.from_profile(self, cmap=cmap).display(
            curr_cands=curr_cands
        )

    def display_support_graph(self, cmap=None, curr_cands=None):
        """
        Display the support graph of the profile (restricted to ``curr_cands``) using the ``cmap``.  See :class:`.SupportGraph`.
        """

        cmap = cmap if cmap is not None else self.cmap
        SupportGraph.from_profile(self, cmap=cmap).display(
            curr_cands=curr_cands
        )

    def visualize_grades(self):
        """Visualize the grade assignments as a stacked bar plot."""
        data_for_df = {"Candidate": [], "Grade": [], "Proportion": []}

        for c in self.candidates:
            for g in [None] + self.grades:
                data_for_df["Candidate"].append(self.cmap[c])
                data_for_df["Grade"].append(
                    self.gmap[g] if g is not None else "None"
                )
                data_for_df["Proportion"].append(self.proportion(c, g))
        df = pd.DataFrame(data_for_df)

        df_pivot = df.pivot(
            index="Candidate", columns="Grade", values="Proportion"
        )

        ax = df_pivot.plot(kind="barh", stacked=True, figsize=(10, 6), rot=0)
        ax.set_ylabel("Candidate")
        ax.set_xlabel("Proportion")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.1),
            ncol=len(self.grades) + 1,
            title="Grades",
        )

        plt.show()

    # =========================================================================
    # Description / code-generation methods
    # =========================================================================

    def description(self):
        """
        Return the Python code needed to create the profile.
        """
        return (
            f"PrefGradeProfile("
            f"{[r.rmap for r in self._rankings]}, "
            f"{[g.as_dict() for g in self._grades]}, "
            f"{self.grades}, "
            f"rcounts={[int(c) for c in self.rcounts]}, "
            f"cmap={self.cmap})"
        )

    # =========================================================================
    # Anonymize
    # =========================================================================

    def anonymize(self):
        """
        Return a profile which is the anonymized version of this profile.
        """

        anon_rankings = list()
        anon_grades = list()
        rcounts = list()
        for r, g in zip(self.rankings, [gf for gf in self.grade_functions]):
            found_it = False
            for _ridx, (_r, _g) in enumerate(
                zip(anon_rankings, anon_grades)
            ):
                if r == _r and g.as_dict() == _g.as_dict():
                    rcounts[_ridx] += 1
                    found_it = True
                    break
            if not found_it:
                anon_rankings.append(r)
                anon_grades.append(g)
                rcounts.append(1)

        prof = PrefGradeProfile(
            anon_rankings,
            [g.as_dict() for g in anon_grades],
            self.grades,
            rcounts=rcounts,
            cmap=self.cmap,
            gmap=self.gmap,
            grade_order=self.grade_order if self.use_grade_order else None,
        )

        if self.using_extended_strict_preference:
            prof.use_extended_strict_preference()

        return prof

    # =========================================================================
    # Dunder methods
    # =========================================================================

    def __eq__(self, other_prof):
        """
        Returns true if two profiles are equal.  Two profiles are equal if they have the same rankings and grade assignments.  Note that we ignore the cmaps.
        """

        rankings = self.rankings
        grades = self.grade_functions
        other_rankings = other_prof.rankings[:]
        other_grades = other_prof.grade_functions[:]
        for r1, g1 in zip(rankings, grades):
            for i, (r2, g2) in enumerate(
                zip(other_rankings, other_grades)
            ):
                if r1 == r2 and g1.as_dict() == g2.as_dict():
                    del other_rankings[i]
                    del other_grades[i]
                    break
            else:
                return False

        return not other_rankings

    def __add__(self, other_prof):
        """
        Returns the sum of two profiles.  The sum of two profiles is the profile that contains all the rankings and grade assignments from the first in addition to all the rankings and grade assignments from the second profile.

        Note: the cmaps of the profiles are ignored.
        """

        return PrefGradeProfile(
            self._rankings + other_prof._rankings,
            [g.as_dict() for g in self._grades]
            + [g.as_dict() for g in other_prof._grades],
            self.grades,
            rcounts=self.rcounts + other_prof.rcounts,
            candidates=sorted(
                list(set(self.candidates + other_prof.candidates))
            ),
        )
