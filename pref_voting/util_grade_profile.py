'''
    File: util_grade_profile.py
    Author: Wes Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)
    Date: April 8, 2026

    A profile that combines utilities and grades for each voter.
'''

import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
from tabulate import SEPARATING_LINE
from pref_voting.mappings import Utility, Grade, _Mapping
from pref_voting.profiles_with_ties import ProfileWithTies
from pref_voting.rankings import Ranking


class UtilGradeProfile(object):
    """An anonymous profile of utilities and grades.

    Each voter submits both a utility function over candidates and a grade
    assignment (e.g., approval/disapproval).  This class packages both into
    a single object, combining the functionality of
    :class:`UtilityProfile` and :class:`GradeProfile`.

    :param utilities: List of utility dicts or :class:`Utility` objects.
    :type utilities: list[dict] or list[Utility]
    :param grade_maps: List of grade dicts or :class:`Grade` objects.
    :type grade_maps: list[dict] or list[Grade]
    :param grades: List of possible grades.
    :type grades: list
    :param ucounts: List of the number of voters associated with each
        utility/grade pair.  If not provided, 1 per entry.
    :type ucounts: list[int], optional
    :param candidates: List of candidates.  If not provided, discovered
        from the utilities and grade maps.
    :type candidates: list, optional
    :param cmap: Dictionary mapping candidates to display names.
    :type cmap: dict, optional
    :param gmap: Dictionary mapping grades to display names.
    :type gmap: dict, optional
    :param grade_order: Grades listed from largest to smallest.  If not
        provided, grades are sorted in descending numeric order.
    :type grade_order: list, optional

    :Example:

    .. code-block:: python

        ugprof = UtilGradeProfile(
            [{"x": 5, "y": 3, "z": 1}, {"x": 2, "y": 4, "z": 4}],
            [{"x": 1, "y": 0, "z": 0}, {"x": 0, "y": 1, "z": 1}],
            [0, 1],
            candidates=["x", "y", "z"],
            gmap={0: "Not Approved", 1: "Approved"},
        )
    """

    def __init__(
        self,
        utilities,
        grade_maps,
        grades,
        ucounts=None,
        candidates=None,
        cmap=None,
        gmap=None,
        grade_order=None,
    ):
        """Constructor method"""

        assert len(utilities) == len(grade_maps), (
            "utilities and grade_maps must have the same length"
        )
        assert ucounts is None or len(utilities) == len(ucounts), (
            "ucounts must have the same length as utilities"
        )

        # ---- candidates / domain ----
        _candidates = list(candidates) if candidates is not None else []
        for u in utilities:
            if isinstance(u, dict):
                _candidates += [x for x in u.keys() if x not in _candidates]
            elif isinstance(u, Utility):
                _candidates += [x for x in u.domain if x not in _candidates]
        for g in grade_maps:
            if isinstance(g, dict):
                _candidates += [x for x in g.keys() if x not in _candidates]
            elif isinstance(g, Grade):
                _candidates += [
                    x for x in g.graded_candidates if x not in _candidates
                ]

        self.candidates = sorted(list(set(_candidates)))
        """Sorted list of candidates."""

        self.domain = self.candidates  # alias used by UtilityProfile methods

        self.num_cands = len(self.candidates)
        """Number of candidates."""

        self.cmap = (
            cmap if cmap is not None else {c: str(c) for c in self.candidates}
        )
        """Candidate display-name mapping."""

        # ---- utilities ----
        self._utilities = [
            Utility(
                u if isinstance(u, dict) else u.as_dict(),
                domain=self.candidates,
                cmap=self.cmap,
            )
            for u in utilities
        ]

        # ---- grades ----
        self.grades = grades
        """List of possible grades."""

        self.can_sum_grades = all(isinstance(g, (float, int)) for g in self.grades)

        self.grade_order = (
            grade_order
            if grade_order is not None
            else sorted(self.grades, reverse=True)
        )
        """Grade ordering (largest to smallest)."""

        self.use_grade_order = grade_order is not None

        self.compare_function = (
            (lambda v1, v2: (v1 > v2) - (v2 > v1))
            if grade_order is None
            else (
                lambda v1, v2: (
                    (grade_order.index(v1) < grade_order.index(v2))
                    - (grade_order.index(v2) < grade_order.index(v1))
                )
            )
        )

        self.gmap = gmap if gmap is not None else {g: str(g) for g in self.grades}
        """Grade display-name mapping."""

        self._grades = [
            Grade(
                g_map if isinstance(g_map, dict) else g_map.as_dict(),
                self.grades,
                candidates=self.candidates,
                cmap=self.cmap,
                gmap=self.gmap,
                compare_function=self.compare_function,
            )
            for g_map in grade_maps
        ]

        # ---- counts ----
        self.ucounts = [1] * len(utilities) if ucounts is None else list(ucounts)

        self.num_voters = int(np.sum(self.ucounts))
        """Total number of voters."""

    # ------------------------------------------------------------------
    #  Utility properties and methods  (from UtilityProfile)
    # ------------------------------------------------------------------

    @property
    def utilities_counts(self):
        """Return ``(utilities_list, ucounts)``."""
        return self._utilities, self.ucounts

    @property
    def utilities(self):
        """Return the expanded list of utility functions (one per voter)."""
        us = []
        for u, c in zip(self._utilities, self.ucounts):
            us += [u] * c
        return us

    def normalize_by_range(self):
        """Return a new profile with each utility normalized by range."""
        return UtilGradeProfile(
            [u.normalize_by_range() for u in self._utilities],
            [g.as_dict() for g in self._grades],
            self.grades,
            ucounts=self.ucounts,
            candidates=self.candidates,
            cmap=self.cmap,
            gmap=self.gmap,
            grade_order=self.grade_order if self.use_grade_order else None,
        )

    def normalize_by_standard_score(self):
        """Return a new profile with each utility normalized by standard score."""
        return UtilGradeProfile(
            [u.normalize_by_standard_score() for u in self._utilities],
            [g.as_dict() for g in self._grades],
            self.grades,
            ucounts=self.ucounts,
            candidates=self.candidates,
            cmap=self.cmap,
            gmap=self.gmap,
            grade_order=self.grade_order if self.use_grade_order else None,
        )

    def has_utility(self, x):
        """Return True if ``x`` is assigned a utility by at least one voter."""
        return any(u.has_utility(x) for u in self._utilities)

    def util_sum(self, x):
        """Return the sum of utilities of ``x``, or None if unrated."""
        if not self.has_utility(x):
            return None
        return np.sum(
            [u(x) * c for u, c in zip(*self.utilities_counts) if u.has_utility(x)]
        )

    def util_avg(self, x):
        """Return the average utility of ``x``, or None if unrated."""
        if not self.has_utility(x):
            return None
        return np.average(
            [u(x) * c for u, c in zip(*self.utilities_counts) if u.has_utility(x)]
        )

    def util_max(self, x):
        """Return the maximum utility of ``x``, or None if unrated."""
        if not self.has_utility(x):
            return None
        return max(u(x) for u in self._utilities if u.has_utility(x))

    def util_min(self, x):
        """Return the minimum utility of ``x``, or None if unrated."""
        if not self.has_utility(x):
            return None
        return min(u(x) for u in self._utilities if u.has_utility(x))

    def sum_utility_function(self):
        """Return the sum utility function."""
        return Utility(
            {x: self.util_sum(x) for x in self.candidates},
            domain=self.candidates,
        )

    def avg_utility_function(self):
        """Return the average utility function."""
        return Utility(
            {x: np.average([u(x) for u in self.utilities]) for x in self.candidates},
            domain=self.candidates,
        )

    # ------------------------------------------------------------------
    #  Grade properties and methods  (from GradeProfile)
    # ------------------------------------------------------------------

    @property
    def grades_counts(self):
        """Return ``(grades_list, ucounts)``."""
        return self._grades, self.ucounts

    @property
    def grade_functions(self):
        """Return the expanded list of grade functions (one per voter)."""
        gs = []
        for g, c in zip(self._grades, self.ucounts):
            gs += [g] * c
        return gs

    def has_grade(self, c):
        """Return True if ``c`` is assigned a grade by at least one voter."""
        return any(g.has_grade(c) for g in self._grades)

    def grade_margin(self, c1, c2, use_extended=False):
        """Return the grade-based margin of ``c1`` over ``c2``."""
        if use_extended:
            return np.sum(
                [n for g, n in zip(*self.grades_counts) if g.extended_strict_pref(c1, c2)]
            ) - np.sum(
                [n for g, n in zip(*self.grades_counts) if g.extended_strict_pref(c2, c1)]
            )
        else:
            return np.sum(
                [n for g, n in zip(*self.grades_counts) if g.strict_pref(c1, c2)]
            ) - np.sum(
                [n for g, n in zip(*self.grades_counts) if g.strict_pref(c2, c1)]
            )

    def proportion(self, cand, grade):
        """Return the proportion of voters assigning ``grade`` to ``cand``."""
        return (
            np.sum([n for g, n in zip(*self.grades_counts) if g(cand) == grade])
            / self.num_voters
        )

    def sum(self, c):
        """Return the sum of grades for ``c``, or None if ungraded."""
        assert self.can_sum_grades, "Grades cannot be summed."
        if not self.has_grade(c):
            return None
        return np.sum(
            [g(c) * n for g, n in zip(*self.grades_counts) if g.has_grade(c)]
        )

    def avg(self, c):
        """Return the average grade for ``c``, or None if ungraded."""
        assert self.can_sum_grades, "Grades cannot be summed."
        if not self.has_grade(c):
            return None
        return np.mean([g(c) for g in self.grade_functions if g.has_grade(c)])

    def max(self, c):
        """Return the maximum grade for ``c``, or None if ungraded."""
        if not self.has_grade(c):
            return None
        grades_for_c = (
            [-1 * self.grade_order.index(g(c)) for g in self._grades if g.has_grade(c)]
            if self.use_grade_order
            else [g(c) for g in self._grades if g.has_grade(c)]
        )
        return (
            self.grade_order[-1 * max(grades_for_c)]
            if self.use_grade_order
            else max(grades_for_c)
        )

    def min(self, c):
        """Return the minimum grade for ``c``, or None if ungraded."""
        if not self.has_grade(c):
            return None
        grades_for_c = (
            [-1 * self.grade_order.index(g(c)) for g in self._grades if g.has_grade(c)]
            if self.use_grade_order
            else [g(c) for g in self._grades if g.has_grade(c)]
        )
        return (
            self.grade_order[-1 * min(grades_for_c)]
            if self.use_grade_order
            else min(grades_for_c)
        )

    def median(self, c, use_lower=True, use_average=False):
        """Return the median grade for ``c``, or None if ungraded."""
        if not self.has_grade(c):
            return None
        grades_for_c = (
            [-1 * self.grade_order.index(g(c)) for g in self.grade_functions if g.has_grade(c)]
            if self.use_grade_order
            else [g(c) for g in self.grade_functions if g.has_grade(c)]
        )
        sorted_grades = sorted(grades_for_c)
        n = len(sorted_grades)
        mid = n // 2
        if n % 2 == 0:
            medians = sorted_grades[mid - 1 : mid + 1]
        else:
            medians = [sorted_grades[mid]]
        if use_lower:
            return (
                self.grade_order[-1 * medians[0]]
                if self.use_grade_order
                else medians[0]
            )
        elif use_average:
            return (
                np.average([self.grade_order[-1 * m] for m in medians])
                if self.use_grade_order
                else np.average(medians)
            )
        else:
            return (
                [self.grade_order[-1 * m] for m in medians]
                if self.use_grade_order
                else medians
            )

    def sum_grade_function(self):
        """Return the sum grade function."""
        assert self.can_sum_grades, "Grades cannot be summed."
        return _Mapping(
            {c: self.sum(c) for c in self.candidates if self.has_grade(c)},
            domain=self.candidates,
            item_map=self.cmap,
            compare_function=self.compare_function,
        )

    def avg_grade_function(self):
        """Return the average grade function."""
        assert self.can_sum_grades, "Grades cannot be summed."
        return _Mapping(
            {c: self.avg(c) for c in self.candidates if self.has_grade(c)},
            domain=self.candidates,
            item_map=self.cmap,
            compare_function=self.compare_function,
        )

    def proportion_with_grade(self, cand, grade):
        """Proportion of voters assigning exactly ``grade`` to ``cand``."""
        assert cand in self.candidates
        assert grade in self.grades
        total = 0
        for g, n in zip(*self.grades_counts):
            if self.compare_function(g(cand), grade) == 0:
                total += n
        return total / self.num_voters

    def proportion_with_higher_grade(self, cand, grade):
        """Proportion of voters assigning strictly higher grade to ``cand``."""
        assert cand in self.candidates
        assert grade in self.grades
        total = 0
        for g, n in zip(*self.grades_counts):
            if self.compare_function(g(cand), grade) == 1:
                total += n
        return total / self.num_voters

    def proportion_with_lower_grade(self, cand, grade):
        """Proportion of voters assigning strictly lower grade to ``cand``."""
        assert cand in self.candidates
        assert grade in self.grades
        total = 0
        for g, n in zip(*self.grades_counts):
            if self.compare_function(g(cand), grade) == -1:
                total += n
        return total / self.num_voters

    def approval_scores(self):
        """Return approval scores (requires grades [0, 1])."""
        assert self.can_sum_grades, "Grades cannot be summed."
        assert sorted(self.grades) == [0, 1], "Grades must be 0 and 1."
        return {c: self.sum(c) for c in self.candidates}

    # ------------------------------------------------------------------
    #  Conversion methods
    # ------------------------------------------------------------------

    def to_utility_profile(self):
        """Return a :class:`UtilityProfile` with just the utility data."""
        from pref_voting.utility_profiles import UtilityProfile

        return UtilityProfile(
            [u.as_dict() for u in self._utilities],
            ucounts=self.ucounts,
            domain=self.candidates,
            cmap=self.cmap,
        )

    def to_grade_profile(self):
        """Return a :class:`GradeProfile` with just the grade data."""
        from pref_voting.grade_profiles import GradeProfile

        return GradeProfile(
            [g.as_dict() for g in self._grades],
            self.grades,
            gcounts=self.ucounts,
            candidates=self.candidates,
            cmap=self.cmap,
            gmap=self.gmap,
            grade_order=self.grade_order if self.use_grade_order else None,
        )

    def to_ranking_profile(self):
        """Return a :class:`ProfileWithTies` with rankings derived from utilities."""
        return ProfileWithTies(
            [u.ranking() for u in self._utilities],
            rcounts=self.ucounts,
            candidates=self.candidates,
            cmap=self.cmap,
        )

    def to_pref_grade_profile(self):
        """Return a :class:`PrefGradeProfile` with rankings derived from utilities."""
        from pref_voting.pref_grade_profile import PrefGradeProfile

        return PrefGradeProfile(
            [u.ranking() for u in self._utilities],
            [g.as_dict() for g in self._grades],
            self.grades,
            rcounts=self.ucounts,
            candidates=self.candidates,
            cmap=self.cmap,
            gmap=self.gmap,
            grade_order=self.grade_order if self.use_grade_order else None,
        )

    # ------------------------------------------------------------------
    #  Candidate manipulation
    # ------------------------------------------------------------------

    def remove_candidates(self, cands_to_ignore):
        """Return a new profile with the specified candidates removed."""
        new_candidates = [c for c in self.candidates if c not in cands_to_ignore]
        new_utils = [
            {c: u(c) for c in new_candidates if u.has_utility(c)}
            for u in self._utilities
        ]
        new_grades = [
            {c: g(c) for c in new_candidates if g.has_grade(c)}
            for g in self._grades
        ]
        return UtilGradeProfile(
            new_utils,
            new_grades,
            self.grades,
            ucounts=self.ucounts,
            candidates=new_candidates,
            cmap={c: self.cmap[c] for c in new_candidates if c in self.cmap},
            gmap=self.gmap,
            grade_order=self.grade_order if self.use_grade_order else None,
        )

    # ------------------------------------------------------------------
    #  Display
    # ------------------------------------------------------------------

    def display(self, cmap=None, show_totals=False):
        """Display the profile as an ASCII table showing utilities and grades."""
        _cmap = cmap if cmap is not None else self.cmap

        if show_totals:
            headers = [""] + self.ucounts + [SEPARATING_LINE] + ["Sum U", "Sum G", "Median G"]
            tbl = [
                [_cmap[c]]
                + [
                    f"{u(c):.4g} [{self.gmap[g(c)]}]"
                    if u.has_utility(c)
                    else ""
                    for u, g in zip(self._utilities, self._grades)
                ]
                + [SEPARATING_LINE]
                + [self.util_sum(c), self.sum(c), self.median(c)]
                for c in self.candidates
            ]
        else:
            headers = [""] + self.ucounts
            tbl = [
                [_cmap[c]]
                + [
                    f"{u(c):.4g} [{self.gmap[g(c)]}]"
                    if u.has_utility(c)
                    else ""
                    for u, g in zip(self._utilities, self._grades)
                ]
                for c in self.candidates
            ]
        print(tabulate(tbl, headers=headers))

    def visualize_grades(self):
        """Visualize grade distributions as a stacked bar plot."""
        data_for_df = {"Candidate": [], "Grade": [], "Proportion": []}
        for c in self.candidates:
            for g in [None] + self.grades:
                data_for_df["Candidate"].append(self.cmap[c])
                data_for_df["Grade"].append(
                    self.gmap[g] if g is not None else "None"
                )
                data_for_df["Proportion"].append(self.proportion(c, g))
        df = pd.DataFrame(data_for_df)
        df_pivot = df.pivot(index="Candidate", columns="Grade", values="Proportion")
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

    # ------------------------------------------------------------------
    #  Serialization
    # ------------------------------------------------------------------

    def description(self):
        """Return Python code that recreates this profile."""
        return (
            f"UtilGradeProfile(\n"
            f"    {[u.as_dict() for u in self._utilities]},\n"
            f"    {[g.as_dict() for g in self._grades]},\n"
            f"    {self.grades},\n"
            f"    ucounts={self.ucounts},\n"
            f"    candidates={self.candidates},\n"
            f")"
        )

    def as_dict(self):
        """Return a dictionary representation of the profile."""
        return {
            "candidates": self.candidates,
            "utilities": [u.as_dict() for u in self._utilities],
            "grade_maps": [g.as_dict() for g in self._grades],
            "grades": self.grades,
            "ucounts": self.ucounts,
            "cmap": self.cmap,
            "gmap": self.gmap,
        }

    def __getstate__(self):
        return self.as_dict()

    def __setstate__(self, state):
        self.__init__(
            state["utilities"],
            state["grade_maps"],
            state["grades"],
            ucounts=state["ucounts"],
            candidates=state["candidates"],
            cmap=state.get("cmap"),
            gmap=state.get("gmap"),
        )

    # ------------------------------------------------------------------
    #  Dunder methods
    # ------------------------------------------------------------------

    def __eq__(self, other):
        if not isinstance(other, UtilGradeProfile):
            return False
        return (
            self.candidates == other.candidates
            and self.grades == other.grades
            and all(
                u1.as_dict() == u2.as_dict()
                for u1, u2 in zip(self._utilities, other._utilities)
            )
            and all(
                g1.as_dict() == g2.as_dict()
                for g1, g2 in zip(self._grades, other._grades)
            )
            and self.ucounts == other.ucounts
        )

    def __add__(self, other):
        assert self.candidates == other.candidates
        assert self.grades == other.grades
        return UtilGradeProfile(
            [u.as_dict() for u in self._utilities]
            + [u.as_dict() for u in other._utilities],
            [g.as_dict() for g in self._grades]
            + [g.as_dict() for g in other._grades],
            self.grades,
            ucounts=self.ucounts + other.ucounts,
            candidates=self.candidates,
            cmap=self.cmap,
            gmap=self.gmap,
            grade_order=self.grade_order if self.use_grade_order else None,
        )
