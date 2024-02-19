
'''
    File: grade_profiles.py
    Author: Wes Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)
    Date: September 24, 2023
    
    Functions to reason about profiles of grades.
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate
from pref_voting.mappings import Grade, _Mapping
from pref_voting.profiles_with_ties import ProfileWithTies

class GradeProfile(object):
    """An anonymous profile of (truncated) grades.  

    :param grade_maps: List of grades in the profile, where a grade is either a :class:`Grade` object or a dictionary.
    :type grade_maps: list[dict[int or str: int or str]] or list[Grade]
    :param grades: List of grades.
    :type gcounts: list[int or str]
    :param gcounts: List of the number of voters associated with each grade.  Should be the same length as grade_maps.  If not provided, it is assumed that 1 voter submitted each element of ``grade_maps``.
    :type gcounts: list[int], optional
    :param candidates: List of candidates in the profile.  If not provided, it is the candidates that are assigned a grade by least on voter.
    :type candidates: list[int] or list[str], optional
    :param cmap: Dictionary mapping candidates to candidate names (strings).  If not provided, each candidate name is mapped to itself.
    :type cmap: dict[int or str: str], optional
    :param gmap: Dictionary mapping grades to grade names (strings).  If not provided, each grade is mapped to itself.
    :type gmap: dict[int or str: str], optional
    :param grade_order: A list of the grades representing the order of the grades. It is assumed the grades are listed from largest to smallest.  If not provided, the grades are assumed to be numbers and compared using the greater-than relation.
    :type gmap: list[int or str], optional

    :Example:

    The following code creates a profile in which
    2 voters submitted the ranking 0 ranked first, 1 ranked second, and 2 ranked third; 3 voters submitted the ranking 1 and 2 are tied for first place and 0 is ranked second; and 1 voter submitted the ranking in which 2 is ranked first and 0 is ranked second:

    .. code-block:: python

        gprof =  GradeProfile([{"x":1, "y":3, "z":1}, {"x":0, "y":-1, "z":3}, {"x":0, "y":-1}, {"x":0, "y":1, "z":2}], [-1, 0, 1, 2, 3], gcounts=[2, 3, 1, 1], candidates=["x", "y", "z"])

        gprof.display()

    """

    def __init__(
        self, 
        grade_maps, 
        grades, 
        gcounts=None, 
        candidates=None, 
        cmap=None,
        gmap=None,
        grade_order=None):
        """Constructor method"""

        assert gcounts is None or len(grade_maps) == len(
            gcounts
        ), "The number of grades much be the same as the number of gcounts"
        self.candidates = (
            sorted(candidates)
            if candidates is not None
            else sorted(list(set([x for g in grade_maps for x in g.keys()])))
        )
        """The domain of the profile. """
        self.cmap = cmap if cmap is not None else {c: str(c) for c in self.candidates}
        """The candidate map is a dictionary associating an alternative with the name used when displaying a alternative."""

        self.grades=grades
        """The grades in the profile. """

        self.can_sum_grades = all([isinstance(g, (float, int)) for g in self.grades])

        self.grade_order = grade_order if grade_order is not None else sorted(self.grades, reverse = True)
        """The order of the grades. If None, then order from largest to smallest"""

        self.use_grade_order = grade_order is not None

        self.compare_function = lambda v1, v2: (v1 > v2) - (v2 > v1) if grade_order is None else lambda v1, v2: (grade_order.index(v1) < grade_order.index(v2)) - (grade_order.index(v2) < grade_order.index(v1))

        self.gmap = gmap if gmap is not None else {g: str(g) for g in self.grades}
        """The candidate map is a dictionary associating an alternative with the name used when displaying a alternative."""

        self._grades = [
            Grade(g_map, self.grades, candidates=self.candidates, cmap=self.cmap, gmap=self.gmap, compare_function=self.compare_function)
            if type(g_map) == dict
            else Grade(g_map.as_dict(), self.grades, candidates=self.candidates, cmap=self.cmap, gmap=self.gmap, compare_function=self.compare_function)
            for g_map in grade_maps
        ]
        """The list of grades in the Profile (each utility is a :class:`Utility` object). 
        """
        self.gcounts = [1] * len(grade_maps) if gcounts is None else list(gcounts)

        self.num_voters = np.sum(self.gcounts)
        """The number of voters in the profile. """

    @property
    def grades_counts(self):
        """Returns the grade and the counts of each grade."""

        return self._grades, self.gcounts
    
    @property
    def grade_functions(self):
        """Return all of the grade functions in the profile."""
        
        gs = list()
        for g,c in zip(self._grades, self.gcounts): 
            gs += [g] * c
        return gs

    def has_grade(self, c):
        """Return True if ``c`` is assigned a grade by at least one voter."""

        return any([g.has_grade(c) for g in self._grades])

    def margin(self, c1, c2, use_extended=False): 
        """
        Return the margin of ``c1`` over ``c2``.  If ``c1`` is not assigned a grade by any voter, return None.
        """
        if use_extended: 
            return np.sum([num for g,num in zip(*self.grades_counts) if g.extended_strict_pref(c1, c2)]) - np.sum([num for g,num in zip(*self.grades_counts) if g.extended_strict_pref(c2, c1)])
        else: 
            return np.sum([num for g,num in zip(*self.grades_counts) if g.strict_pref(c1, c2)]) - np.sum([num for g,num in zip(*self.grades_counts) if g.strict_pref(c2, c1)])

    def proportion(self, cand, grade):
        """
        Return the proportion of voters that assign ``cand`` the grade ``grade``.

        Note that ``grade`` could be None, in which case the proportion of voters that do not assign ``cand`` a grade is returned.
        """
        return np.sum([num for g,num in zip(*self.grades_counts) if g(cand) == grade]) / self.num_voters

    def sum(self, c): 
        """Return the sum of the grades of ``c``.  If ``c`` is not assigned a grade by any voter, return None."""

        assert self.can_sum_grades, "The grades in the profile cannot be summed."

        return np.sum([g(c) * num for g,num in zip(*self.grades_counts) if g.has_grade(c)]) if self.has_grade(c) else None
    
    def avg(self, c): 
        """Return the average of the grades of ``c``.  If ``c`` is not assigned a grade by any voter, return None."""

        assert self.can_sum_grades, "The grades in the profile cannot be summed."

        return np.mean([g(c) for g in self.grade_functions if g.has_grade(c)]) if self.has_grade(c) else None
    
    def max(self, c): 
        """Return the maximum of the grade of ``c``.  If ``c`` is not assigned a grade by any voter, return None."""

        grades_for_c = [-1 * self.grade_order.index(g(c)) for g in self._grades if g.has_grade(c)] if self.use_grade_order else [g(c) for g in self._grades if g.has_grade(c)]

        return (self.grade_order[-1 * max(grades_for_c)] if self.use_grade_order else max(grades_for_c)) if self.has_grade(c) else None
    
    def min(self, c): 
        """Return the minimum of the grades of ``c``.  If ``c`` is not assigned a grade by any voter, return None."""

        grades_for_c = [-1 * self.grade_order.index(g(c)) for g in self._grades if g.has_grade(c)] if self.use_grade_order else [g(c) for g in self._grades if g.has_grade(c)]

        return (self.grade_order[-1 * min(grades_for_c)] if self.use_grade_order else min(grades_for_c)) if self.has_grade(c) else None
    
    def median(self, c, use_lower=True, use_average=False): 
        """Return the median of the grades of ``c``.  If ``c`` is not assigned a grade by any voter, return None."""

        grades_for_c = [-1 * self.grade_order.index(g(c)) for g in self.grade_functions if g.has_grade(c)] if self.use_grade_order else [g(c) for g in self.grade_functions if g.has_grade(c)]

        sorted_grades_for_c = sorted(grades_for_c)
        num_grades = len(sorted_grades_for_c)
        median_idx = num_grades // 2
        if num_grades % 2 == 0:
            median_grades = sorted_grades_for_c[median_idx - 1:median_idx + 1]
        else:
            median_grades = [sorted_grades_for_c[median_idx]]

        if use_lower:
            return (self.grade_order[-1 * median_grades[0]] if self.use_grade_order else median_grades[0]) if self.has_grade(c) else None
        elif use_average:
            return (np.average([self.grade_order[-1 * m] for m in median_grades]) if self.use_grade_order else np.average(median_grades)) if self.has_grade(c) else None
        else:
            return ([self.grade_order[-1 * m] for m in median_grades] if self.use_grade_order else median_grades) if self.has_grade(c) else None

    def sum_grade_function(self):
        """Return the sum grade function of the profile."""

        assert self.can_sum_grades, "The grades in the profile cannot be summed."

        return _Mapping(
            {
                c: self.sum(c)
                for c in self.candidates if self.has_grade(c)
            },
            domain=self.candidates, 
            item_map=self.cmap, 
            compare_function=self.compare_function
        )
    def avg_grade_function(self):
        """Return the average grade function of the profile."""

        assert self.can_sum_grades, "The grades in the profile cannot be summed."

        return _Mapping(
            {
                c: self.avg(c)
                for c in self.candidates if self.has_grade(c)
            },
            domain=self.candidates, 
            item_map=self.cmap, 
            compare_function=self.compare_function
        )


    def proportion_with_grade(self, cand, grade):
        """
        Return the proportion of voters that assign a ``grade`` to ``cand`` .
        """

        assert cand in self.candidates, f"{cand} is not a candidate in the profile."
        assert grade in self.grades, f"{grade} is not a grade in the profile."

        num_with_higher_grade = 0
        for g,num in zip(*self.grades_counts):
            if self.compare_function(g(cand),grade) == 0: 
                num_with_higher_grade += num
        return num_with_higher_grade / self.num_voters


    def proportion_with_higher_grade(self, cand, grade):
        """
        Return the proportion of voters that assign a strictly higher grade to ``cand`` than ``grade``.
        """
        
        assert cand in self.candidates, f"{cand} is not a candidate in the profile."
        assert grade in self.grades, f"{grade} is not a grade in the profile."

        num_with_higher_grade = 0
        for g,num in zip(*self.grades_counts):
            if self.compare_function(g(cand),grade) == 1: 
                num_with_higher_grade += num
        return num_with_higher_grade / self.num_voters


    def proportion_with_lower_grade(self, cand, grade):
        """
        Return the proportion of voters that assign a strictly lower grade to ``cand`` than ``grade``.
        """

        assert cand in self.candidates, f"{cand} is not a candidate in the profile."
        assert grade in self.grades, f"{grade} is not a grade in the profile."


        num_with_lower_grade = 0
        for g,num in zip(*self.grades_counts):
            if self.compare_function(g(cand),grade) == -1: 
                num_with_lower_grade += num
        return num_with_lower_grade / self.num_voters

    def approval_scores(self): 
        """
        Return a dictionary representing the approval scores of the candidates in the profile.
        """

        assert self.can_sum_grades, "The grades in the profile cannot be summed."
        assert sorted(self.grades) == [0,1], "The grades in the profile must be 0 and 1."

        return {c: self.sum(c) for c in self.candidates}
    
    def to_ranking_profile(self): 
        """Return a ranking profile (a :class:ProfileWithTies) corresponding to the profile."""

        return ProfileWithTies(
            [g.ranking() for g in self._grades],
            rcounts = self.gcounts,
            candidates = self.candidates, 
            cmap = self.cmap
        )
    
    def write(self):
        """Write the profile to a string."""

        gprof_str = f"{len(self.candidates)};{self.num_voters};{self.grades}"
        for g in self.grade_functions: 
            g_str = ''
            for c in g.graded_candidates: 
                g_str += f"{c}:{g(c)},"
            gprof_str += f";{g_str[0:-1]}"
        return str(gprof_str)

    @classmethod
    def from_string(cls, gprof_str): 
        """
        Returns a profile of utilities described by ``gprof_str``.

        ``gprof_str`` must be in the format produced by the :meth:`pref_voting.GradeProfile.write` function.
        """
        gprof_data = gprof_str.split(";")

        num_alternatives,num_voters,grades = int(gprof_data[0]),int(gprof_data[1]),gprof_data[2:]

        grade_maps = [{int(cg.split(":")[0]): float(cg.split(":")[1]) for cg in gs.split(",")} if gs != '' else {} for gs in grades]

        if len(grade_maps) != num_voters: 
            raise Exception("Number of voters does not match the number of utilities.")
        
        return cls(grade_maps, candidates=range(num_alternatives))

    def display(self, show_totals=False, average_median_ties=False):
        """Display a grade profile as an ascii table (using tabulate). If ``show_totals`` is true then the sum, min, and max of the grades are displayed.

        """
                        
        if show_totals:
            sum_grade_fnc = self.sum_grade_function()
            headers = [''] + self.gcounts + ["Sum", "Median"]
            tbl = [[self.cmap[c]] + [self.gmap[g(c)] if g.has_grade(c) else "" for g in self._grades] + [sum_grade_fnc(c), self.median(c)] for c in self.candidates]
 
        else: 
            headers = [''] + self.gcounts
            tbl = [[self.cmap[c]] + [self.gmap[g(c)] if g.has_grade(c) else "" for g in self._grades] for c in self.candidates]
        print(tabulate(tbl, headers=headers))

    def visualize(self):
        """Visualize a grade profile as a stacked bar plot."""
        data_for_df = {
            'Candidate': [],
            'Grade': [],
            'Proportion': []
        }

        for c in self.candidates: 
            for g in [None] + self.grades:
                data_for_df['Candidate'].append(self.cmap[c])
                data_for_df['Grade'].append(self.gmap[g] if g is not None else "None")
                data_for_df['Proportion'].append(self.proportion(c, g))
        df = pd.DataFrame(data_for_df)

        # Pivot the DataFrame to organize it for the stacked bar plot
        df_pivot = df.pivot(index='Candidate', columns='Grade', values='Proportion')

        df_pivot
        # # Create the stacked bar plot
        ax = df_pivot.plot(kind='barh', stacked=True, figsize=(10, 6), rot=0)
        ax.set_ylabel('Candidate')
        ax.set_xlabel('Proportion')
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        # Show the plot

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=len(self.grades) + 1, title="Grades")

        plt.show()
