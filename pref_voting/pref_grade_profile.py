"""
File: pref_grade_profile.py  (refactored)
Author: Wes Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)

A class that represents profiles in which each voter submits both a (truncated)
strict weak order and an assignment of grades.

Design
------
The ranking side of a ``PrefGradeProfile`` is exactly a :class:`ProfileWithTies`,
so this class **inherits** ``ProfileWithTies`` and gets the entire ranking-query API
for free (``support``, ``margin``, ``condorcet_winner``, ``copeland_scores``,
``plurality_scores``, ``borda_scores``, ``cycles``, the ``num_*`` family, the margin/
support/majority graphs, pickling, ...).

The grade side is exactly a :class:`GradeProfile`, so this class **composes** one
(``self._grade_profile``) and delegates the grade-query API to it (renaming ``margin``
to ``grade_margin`` to avoid colliding with the ranking margin).

Only the genuinely combined behavior is bespoke: the constructor, the conversions, the
lockstep mutations (``remove_candidates``, ``remove_empty_rankings``, ``anonymize``),
``display``, ``__eq__``, ``__add__``, and ``description``.
"""

import copy

import numpy as np
from tabulate import tabulate

from pref_voting.grade_profiles import GradeProfile
from pref_voting.profiles_with_ties import ProfileWithTies


class PrefGradeProfile(ProfileWithTies):
    """An anonymous profile in which each voter submits both a (truncated) strict weak
    order and an assignment of grades.  See the module docstring for the design.

    :param rankings: List of rankings (each a :class:`Ranking` or a dict).
    :param grade_maps: List of grade maps (each a :class:`Grade` or a dict).
    :param grades: List of grades.
    :param rcounts: Number of voters for each ranking/grade pair.
    :param candidates: List of candidates (defaults to the union appearing in the data).
    :param cmap: candidate-name map.
    :param gmap: grade-name map.
    :param grade_order: grades from largest to smallest (numeric ``>`` if None).
    """

    def __init__(self, rankings, grade_maps, grades, rcounts=None, candidates=None,
                 cmap=None, gmap=None, grade_order=None):
        assert len(rankings) == len(grade_maps), (
            "The number of rankings must be the same as the number of grade maps"
        )

        # Determine the candidate set from BOTH the rankings and the grade maps,
        # then let the ranking side (ProfileWithTies) and grade side (GradeProfile)
        # share that exact candidate set.
        if candidates is None:
            get_r = lambda r: list(r.keys()) if isinstance(r, dict) else r.cands
            get_g = lambda g: list(g.keys()) if isinstance(g, dict) else g.graded_candidates
            candidates = sorted(
                set(c for r in rankings for c in get_r(r))
                | set(c for g in grade_maps for c in get_g(g))
            )

        # --- ranking side: initialise the ProfileWithTies base ---
        super().__init__(rankings, rcounts=rcounts, candidates=candidates, cmap=cmap)

        # --- grade side: compose a GradeProfile over the same candidates/counts ---
        self._grade_profile = GradeProfile(
            grade_maps,
            grades,
            gcounts=self.rcounts,
            candidates=self.candidates,
            cmap=self.cmap,
            gmap=gmap,
            grade_order=grade_order,
        )

    # ------------------------------------------------------------------
    # Grade-side state exposed as read-only properties (delegated)
    # ------------------------------------------------------------------
    @property
    def grades(self):
        return self._grade_profile.grades

    @property
    def grade_order(self):
        return self._grade_profile.grade_order

    @property
    def use_grade_order(self):
        return self._grade_profile.use_grade_order

    @property
    def gmap(self):
        return self._grade_profile.gmap

    @property
    def can_sum_grades(self):
        return self._grade_profile.can_sum_grades

    @property
    def compare_function(self):
        return self._grade_profile.compare_function

    @property
    def _grades(self):
        return self._grade_profile._grades

    @property
    def grades_counts(self):
        return self._grade_profile.grades_counts

    @property
    def grade_functions(self):
        return self._grade_profile.grade_functions

    # ------------------------------------------------------------------
    # Grade-side query API (delegated to the composed GradeProfile)
    # ------------------------------------------------------------------
    def has_grade(self, c):
        return self._grade_profile.has_grade(c)

    def grade_margin(self, c1, c2, use_extended=False):
        return self._grade_profile.margin(c1, c2, use_extended=use_extended)

    def proportion(self, cand, grade):
        return self._grade_profile.proportion(cand, grade)

    def proportion_with_grade(self, cand, grade):
        return self._grade_profile.proportion_with_grade(cand, grade)

    def proportion_with_higher_grade(self, cand, grade):
        return self._grade_profile.proportion_with_higher_grade(cand, grade)

    def proportion_with_lower_grade(self, cand, grade):
        return self._grade_profile.proportion_with_lower_grade(cand, grade)

    def sum(self, c):
        return self._grade_profile.sum(c)

    def avg(self, c):
        return self._grade_profile.avg(c)

    def max(self, c):
        return self._grade_profile.max(c)

    def min(self, c):
        return self._grade_profile.min(c)

    def median(self, c, use_lower=True, use_average=False):
        return self._grade_profile.median(c, use_lower=use_lower, use_average=use_average)

    def sum_grade_function(self):
        return self._grade_profile.sum_grade_function()

    def avg_grade_function(self):
        return self._grade_profile.avg_grade_function()

    def approval_scores(self):
        return self._grade_profile.approval_scores()

    # ------------------------------------------------------------------
    # Conversions
    # ------------------------------------------------------------------
    def to_ranking_profile(self):
        """Return a :class:`ProfileWithTies` of just the ranking side."""
        return ProfileWithTies(
            self._rankings, rcounts=self.rcounts,
            candidates=self.candidates, cmap=self.cmap,
        )

    def to_grade_profile(self):
        """Return a :class:`GradeProfile` of just the grade side."""
        return GradeProfile(
            [g.as_dict() for g in self._grades],
            self.grades,
            gcounts=self.rcounts,
            candidates=self.candidates,
            cmap=self.cmap,
            gmap=self.gmap,
            grade_order=self.grade_order if self.use_grade_order else None,
        )

    # ------------------------------------------------------------------
    # Lockstep mutations (must keep rankings and grades aligned)
    # ------------------------------------------------------------------
    def remove_candidates(self, cands_to_ignore):
        """Remove ``cands_to_ignore`` from both the rankings and the grades."""
        updated_rankings = [
            {c: r for c, r in rank.rmap.items() if c not in cands_to_ignore}
            for rank in self._rankings
        ]
        updated_grade_maps = [
            {c: g.val(c) for c in g.graded_candidates if c not in cands_to_ignore}
            for g in self._grades
        ]
        new_candidates = [c for c in self.candidates if c not in cands_to_ignore]
        restricted = PrefGradeProfile(
            updated_rankings, updated_grade_maps, self.grades,
            rcounts=self.rcounts, candidates=new_candidates,
            cmap=self.cmap, gmap=self.gmap,
            grade_order=self.grade_order if self.use_grade_order else None,
        )
        if self.using_extended_strict_preference:
            restricted.use_extended_strict_preference()
        return restricted

    def remove_empty_rankings(self):
        """Remove ballots whose ranking is empty, keeping grades in lockstep."""
        new_rankings, new_grade_maps, new_rcounts = [], [], []
        for r, g, c in zip(self._rankings, self._grades, self.rcounts):
            if len(r.cands) != 0:
                new_rankings.append(r.rmap)
                new_grade_maps.append(g.as_dict())
                new_rcounts.append(c)
        # rebuild both sides in place
        ProfileWithTies.__init__(
            self, new_rankings, rcounts=new_rcounts,
            candidates=self.candidates, cmap=self.cmap,
        )
        self._grade_profile = GradeProfile(
            new_grade_maps, self.grades, gcounts=self.rcounts,
            candidates=self.candidates, cmap=self.cmap, gmap=self.gmap,
            grade_order=self.grade_order if self.use_grade_order else None,
        )

    def anonymize(self):
        """Group identical (ranking, grade) ballots together."""
        anon_rankings, anon_grades, rcounts = [], [], []
        for r, g in zip(self.rankings, self.grade_functions):
            for i, (_r, _g) in enumerate(zip(anon_rankings, anon_grades)):
                if r == _r and g.as_dict() == _g.as_dict():
                    rcounts[i] += 1
                    break
            else:
                anon_rankings.append(r)
                anon_grades.append(g)
                rcounts.append(1)
        prof = PrefGradeProfile(
            anon_rankings, [g.as_dict() for g in anon_grades], self.grades,
            rcounts=rcounts, cmap=self.cmap, gmap=self.gmap,
            grade_order=self.grade_order if self.use_grade_order else None,
        )
        if self.using_extended_strict_preference:
            prof.use_extended_strict_preference()
        return prof

    # ------------------------------------------------------------------
    # Display / description
    # ------------------------------------------------------------------
    def display(self, cmap=None, style="pretty", curr_cands=None,
                show_grades=True, show_totals=False):
        """Display the ranking table and (optionally) the grade table."""
        _rankings = [r.normalize_ranks() or r for r in copy.deepcopy(self._rankings)]
        curr_cands = curr_cands if curr_cands is not None else self.candidates
        cmap = cmap if cmap is not None else self.cmap

        _ranked = [r for r in _rankings if len(r.ranks) > 0]
        existing_ranks = list(range(
            min(min(r.ranks) for r in _ranked),
            max(max(r.ranks) for r in _ranked) + 1,
        )) if len(_ranked) > 0 else []

        print("Rankings:")
        print(tabulate(
            [[" ".join(str(cmap[c]) for c in r.cands_at_rank(rank) if c in curr_cands)
              for r in _rankings]
             for rank in existing_ranks],
            self.rcounts, tablefmt=style,
        ))

        if show_grades:
            print("\nGrades:")
            if show_totals:
                sum_fn = self.sum_grade_function()
                headers = [""] + self.rcounts + ["Sum", "Median"]
                tbl = [[cmap[c]]
                       + [self.gmap[g(c)] if g.has_grade(c) else "" for g in self._grades]
                       + [sum_fn(c), self.median(c)]
                       for c in curr_cands]
            else:
                headers = [""] + self.rcounts
                tbl = [[cmap[c]]
                       + [self.gmap[g(c)] if g.has_grade(c) else "" for g in self._grades]
                       for c in curr_cands]
            print(tabulate(tbl, headers=headers))

    def visualize_grades(self):
        self.to_grade_profile().visualize()

    def description(self):
        return (
            f"PrefGradeProfile("
            f"{[r.rmap for r in self._rankings]}, "
            f"{[g.as_dict() for g in self._grades]}, "
            f"{self.grades}, "
            f"rcounts={[int(c) for c in self.rcounts]}, "
            f"cmap={self.cmap})"
        )

    # ------------------------------------------------------------------
    # Equality / addition (consider BOTH rankings and grades)
    # ------------------------------------------------------------------
    def __eq__(self, other):
        rankings, grades = self.rankings, self.grade_functions
        other_rankings = other.rankings[:]
        other_grades = other.grade_functions[:]
        for r1, g1 in zip(rankings, grades):
            for i, (r2, g2) in enumerate(zip(other_rankings, other_grades)):
                if r1 == r2 and g1.as_dict() == g2.as_dict():
                    del other_rankings[i]
                    del other_grades[i]
                    break
            else:
                return False
        return not other_rankings

    __hash__ = None

    def __add__(self, other):
        return PrefGradeProfile(
            self._rankings + other._rankings,
            [g.as_dict() for g in self._grades] + [g.as_dict() for g in other._grades],
            self.grades,
            rcounts=self.rcounts + other.rcounts,
            candidates=sorted(set(self.candidates + other.candidates)),
        )
