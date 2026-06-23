"""
File: util_grade_profile.py  (refactored)
Author: Wes Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)

A profile that combines utilities and grades for each voter.

Design
------
The utility side of a ``UtilGradeProfile`` is exactly a :class:`UtilityProfile`,
so this class **inherits** ``UtilityProfile`` and gets the entire utility-query API
for free (``util_sum``, ``util_avg``, ``util_max``/``util_min``,
``sum_utility_function``/``avg_utility_function``, ``utilities``/``utilities_counts``,
``has_utility``, ``num_cands``, the ``candidates`` property, ...).

The grade side is exactly a :class:`GradeProfile`, so this class **composes** one
(``self._grade_profile``) and delegates the grade-query API to it (renaming
``margin`` to ``grade_margin`` to avoid colliding with any ranking/utility notion).

Only the genuinely combined behavior is bespoke: the constructor (which must derive
the candidate set from *both* sides), the range/standard-score normalizations (which
must carry the grades along), the conversions, ``remove_candidates`` (a lockstep
mutation), ``display``, ``visualize_grades``, ``__eq__``, ``__add__``,
``as_dict``/``description``, and the ``__getstate__``/``__setstate__`` pair that keeps
the object picklable (the composed ``GradeProfile`` carries an unpicklable
``compare_function`` closure, so we pickle the data and rebuild instead).
"""

from tabulate import SEPARATING_LINE, tabulate

from pref_voting.grade_profiles import GradeProfile
from pref_voting.mappings import Grade, Utility
from pref_voting.profiles_with_ties import ProfileWithTies
from pref_voting.utility_profiles import UtilityProfile


class UtilGradeProfile(UtilityProfile):
    """An anonymous profile of utilities and grades.

    Each voter submits both a utility function over candidates and a grade
    assignment (e.g., approval/disapproval).  This class packages both into a
    single object by **inheriting** :class:`UtilityProfile` (the utility side) and
    **composing** a :class:`GradeProfile` (the grade side).  See the module docstring
    for the design.

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

        # ---- candidate set: union over BOTH the utilities and the grade maps ----
        _candidates = list(candidates) if candidates is not None else []
        for u in utilities:
            new = (
                u.keys()
                if isinstance(u, dict)
                else (u.domain if isinstance(u, Utility) else [])
            )
            _candidates += [x for x in new if x not in _candidates]
        for g in grade_maps:
            new = (
                g.keys()
                if isinstance(g, dict)
                else (g.graded_candidates if isinstance(g, Grade) else [])
            )
            _candidates += [x for x in new if x not in _candidates]
        all_candidates = sorted(set(_candidates))

        # ---- utility side: initialise the UtilityProfile base ----
        # (gives util_sum/avg/max/min, sum/avg_utility_function, utilities[_counts],
        #  has_utility, num_cands, the candidates property, ...)
        super().__init__(
            [u if isinstance(u, dict) else u.as_dict() for u in utilities],
            ucounts=ucounts,
            domain=all_candidates,
            cmap=cmap,
        )

        # ---- grade side: compose a GradeProfile over the same candidates/counts ----
        self._grade_profile = GradeProfile(
            [g if isinstance(g, dict) else g.as_dict() for g in grade_maps],
            grades,
            gcounts=self.ucounts,
            candidates=self.candidates,
            cmap=self.cmap,
            gmap=gmap,
            grade_order=grade_order,
        )

    # ------------------------------------------------------------------
    #  Grade-side state exposed as read-only properties (delegated)
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
    #  Grade-side query API (delegated to the composed GradeProfile)
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
        return self._grade_profile.median(
            c, use_lower=use_lower, use_average=use_average
        )

    def sum_grade_function(self):
        return self._grade_profile.sum_grade_function()

    def avg_grade_function(self):
        return self._grade_profile.avg_grade_function()

    def approval_scores(self):
        return self._grade_profile.approval_scores()

    # ------------------------------------------------------------------
    #  Normalizations (override: carry the grades along, return UtilGradeProfile)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    #  Conversions
    # ------------------------------------------------------------------
    def to_utility_profile(self):
        """Return a :class:`UtilityProfile` with just the utility data."""
        return UtilityProfile(
            [u.as_dict() for u in self._utilities],
            ucounts=self.ucounts,
            domain=self.candidates,
            cmap=self.cmap,
        )

    def to_grade_profile(self):
        """Return a :class:`GradeProfile` with just the grade data."""
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
    #  Candidate manipulation (lockstep: utilities and grades together)
    # ------------------------------------------------------------------
    def remove_candidates(self, cands_to_ignore):
        """Return a new profile with the specified candidates removed."""
        new_candidates = [c for c in self.candidates if c not in cands_to_ignore]
        new_utils = [
            {c: u(c) for c in new_candidates if u.has_utility(c)}
            for u in self._utilities
        ]
        new_grades = [
            {c: g(c) for c in new_candidates if g.has_grade(c)} for g in self._grades
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
            headers = (
                [""] + self.ucounts + [SEPARATING_LINE] + ["Sum U", "Sum G", "Median G"]
            )
            tbl = [
                [_cmap[c]]
                + [
                    f"{u(c):.4g} [{self.gmap[g(c)]}]" if u.has_utility(c) else ""
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
                    f"{u(c):.4g} [{self.gmap[g(c)]}]" if u.has_utility(c) else ""
                    for u, g in zip(self._utilities, self._grades)
                ]
                for c in self.candidates
            ]
        print(tabulate(tbl, headers=headers))

    def visualize_grades(self):
        """Visualize grade distributions as a stacked bar plot."""
        self.to_grade_profile().visualize()

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

    __hash__ = None

    def __add__(self, other):
        assert self.candidates == other.candidates
        assert self.grades == other.grades
        return UtilGradeProfile(
            [u.as_dict() for u in self._utilities]
            + [u.as_dict() for u in other._utilities],
            [g.as_dict() for g in self._grades] + [g.as_dict() for g in other._grades],
            self.grades,
            ucounts=self.ucounts + other.ucounts,
            candidates=self.candidates,
            cmap=self.cmap,
            gmap=self.gmap,
            grade_order=self.grade_order if self.use_grade_order else None,
        )
