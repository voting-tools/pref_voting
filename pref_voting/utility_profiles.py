'''
    File: utility_profiles.py
    Author: Wes Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)
    Date: May 26, 2023
    
    Functions to reason about profiles of utilities.
'''


from math import ceil
import numpy as np
import json
from scipy import stats
import networkx as nx
from tabulate import tabulate
from tabulate import  SEPARATING_LINE
from pref_voting.profiles_with_ties import ProfileWithTies
from pref_voting.rankings import Ranking
from pref_voting.mappings import Utility
from pref_voting.grade_profiles import GradeProfile

# turn off future warnings.
# getting the following warning when calling tabulate to display a profile: 
# /Library/Frameworks/Python.framework/Versions/3.8/lib/python3.8/site-packages/tabulate.py:1027: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
#  if headers == "keys" and not rows:
# see https://stackoverflow.com/questions/40659212/futurewarning-elementwise-comparison-failed-returning-scalar-but-in-the-futur
#
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class UtilityProfile(object):
    """An anonymous profile of (truncated) utilities.  

    :param utilities: List of utilities in the profile, where a utility is either a :class:`Utility` object or a dictionary.
    :type utilities: list[dict[int or str: float]] or list[Utility]
    :param ucounts: List of the number of voters associated with each utility.  Should be the same length as utilities.  If not provided, it is assumed that 1 voters submitted each element of ``utilities``.
    :type ucounts: list[int], optional
    :param domain: List of alternatives in the profile.  If not provided, it is the alternatives that are assigned a utility by least on voter.
    :type domain: list[int] or list[str], optional
    :param cmap: Dictionary mapping alternatives to alternative names (strings).  If not provided, each alternative name is mapped to itself.
    :type cmap: dict[int or str: str], optional

    :Example:

    The following code creates a profile in which
    2 voters submitted the ranking 0 ranked first, 1 ranked second, and 2 ranked third; 3 voters submitted the ranking 1 and 2 are tied for first place and 0 is ranked second; and 1 voter submitted the ranking in which 2 is ranked first and 0 is ranked second:

    .. code-block:: python

        uprof =  UtilityProfile([{"x":1, "y":3, "z":1}, {"x":0, "y":-1, "z":4}, {"x":0.5, "y":-1}, {"x":0, "y":1, "z":2}], ucounts=[2, 3, 1, 1], domain=["x", "y", "z"])

    """

    def __init__(self, utilities, ucounts=None, domain=None, cmap=None):
        """Constructor method"""

        assert ucounts is None or len(utilities) == len(
            ucounts
        ), "The number of utilities much be the same as the number of ucounts"

        _domain = domain if domain is not None else []
        for u in utilities:
            if isinstance(u, dict):
                _domain += [x for x in u.keys() if x not in _domain]
            elif isinstance(u, Utility):
                _domain += [x for x in u.domain if x not in _domain]

        self.domain = sorted(list(set(_domain)))
        """The domain of the profile. """

        self.cmap = cmap if cmap is not None else {c: str(c) for c in self.domain}
        """The candidate map is a dictionary associating an alternative with the name used when displaying a alternative."""

        self._utilities = [
            Utility(u, domain = self.domain, cmap=self.cmap)
            if type(u) == dict
            else Utility(u.as_dict(), domain=self.domain, cmap=self.cmap)
            for u in utilities
        ]
        """The list of utilities in the Profile (each utility is a :class:`Utility` object). 
        """

        self.ucounts = [1] * len(utilities) if ucounts is None else list(ucounts)

        self.num_voters = np.sum(self.ucounts)
        """The number of voters in the profile. """

    @property
    def candidates(self): 
        """Return the candidates in the profile."""
        return self.domain
    
    @property
    def utilities_counts(self):
        """Returns the utilities and the counts of each utility."""

        return self._utilities, self.ucounts
    
    @property
    def utilities(self):
        """Return all of the utilities in the profile."""
        
        us = list()
        for u,c in zip(self._utilities, self.ucounts): 
            us += [u] * c
        return us


    def normalize_by_range(self): 
        """Return a profile in which each utility is normalized by range."""
        
        return UtilityProfile([
            u.normalize_by_range() for u in self._utilities
        ], ucounts = self.ucounts, domain = self.domain, cmap=self.cmap)
    
    def normalize_by_standard_score(self):
        """Return a profile in which each utility is normalized by standard scores.
        """
        return UtilityProfile([
            u.normalize_by_standard_score() for u in self._utilities
        ], ucounts = self.ucounts, domain = self.domain, cmap=self.cmap)

    def has_utility(self, x):
        """Return True if ``x`` is assigned a utility by at least one voter."""

        return any([u.has_utility(x) for u in self._utilities])

    def util_sum(self, x): 
        """Return the sum of the utilities of ``x``.  If ``x`` is not assigned a utility by any voter, return None."""

        return np.sum([u(x) * c for u,c in zip(*self.utilities_counts) if u.has_utility(x)]) if self.has_utility(x) else None
    
    def util_avg(self, x): 
        """Return the sum of the utilities of ``x``.  If ``x`` is not assigned a utility by any voter, return None."""

        return np.average([u(x) * c for u,c in zip(*self.utilities_counts) if u.has_utility(x)]) if self.has_utility(x) else None
    
    def util_max(self, x): 
        """Return the maximum of the utilities of ``x``.  If ``x`` is not assigned a utility by any voter, return None."""

        return max([u(x)  for u in self._utilities if u.has_utility(x)]) if self.has_utility(x) else None
    
    def util_min(self, x): 
        """Return the minimum of the utilities of ``x``.  If ``x`` is not assigned a utility by any voter, return None."""

        return min([u(x)  for u in self._utilities if u.has_utility(x)]) if self.has_utility(x) else None

    def sum_utility_function(self):
        """Return the sum utility function of the profile."""

        return Utility(
            {
                x: self.util_sum(x)
                for x in self.domain
            },
            domain=self.domain,
        )
    def avg_utility_function(self):
        """Return the average utility function of the profile."""

        return Utility(
            {
                x: np.average([u(x) for u in self.utilities])
                for x in self.domain
            },
            domain=self.domain,
        )
    
    def to_ranking_profile(self): 
        """Return a ranking profile (a :class:ProfileWithTies) corresponding to the profile."""

        return ProfileWithTies(
            [u.ranking() for u in self._utilities],
            rcounts = self.ucounts,
            candidates = self.domain, 
            cmap = self.cmap
        )
    
    def to_approval_profile(self, prob_to_cont_approving=1.0, decay_rate=0.0):
        """
        Return a GradeProfile with each utility transformed to an approval ballot.

        See :meth:`pref_voting.Utility.to_approval_ballot` for more details.
        """

        return GradeProfile(
            [u.to_approval_ballot(
                prob_to_cont_approving=prob_to_cont_approving, 
                decay_rate=decay_rate) 
                for u in self._utilities],
            [0, 1],
            gcounts = self.ucounts,
            candidates = self.domain,
            cmap = self.cmap
        )
    
    def to_k_approval_profile(self, k, prob_to_cont_approving=1.0, decay_rate=0.0):
        """
        Return a GradeProfile with each utility transformed to a k-approval ballot.

        See :meth:`pref_voting.Utility.to_approval_ballot` for more details.
        """

        return GradeProfile(
            [u.to_k_approval_ballot(
                k,
                prob_to_cont_approving=prob_to_cont_approving, 
                decay_rate=decay_rate) 
                for u in self._utilities],
            [0, 1],
            gcounts = self.ucounts,
            candidates = self.domain,
            cmap = self.cmap
        )

        
    def write(self):
        """Write the profile to a string."""

        uprof_str = f"{len(self.domain)};{self.num_voters}"
        for u in self.utilities: 
            u_str = ''
            for c in u.domain: 
                if u.has_utility(c):
                    u_str += f"{c}:{u(c)},"
            uprof_str += f";{u_str[0:-1]}"
        return str(uprof_str)

    def as_dict(self): 
        """Return a the profile as a dictionary."""

        return {
            "domain": self.domain,
            "utilities": [u.as_dict() for u in self._utilities],
            "ucounts": self.ucounts,
            "cmap": self.cmap
        }
    
    @classmethod
    def from_json(cls, uprof_json): 
        """
        Returns a profile of utilities described by ``uprof_json``.

        ``uprof_json`` must be in the format produced by the :meth:`pref_voting.UtilityProfile.as_dict` function.
        """
        domain = uprof_json["domain"]
        util_maps = uprof_json["utilities"]
        ucounts = uprof_json["ucounts"]
        cmap = uprof_json["cmap"]

        # since json converts all keys to strings, we need to convert them back to integers if the domain is integers.
        integer_domain = all([type(x) == int for x in domain])
        if integer_domain:
            util_maps = [{int(c):v for c,v in u.items()} for u in util_maps]
            cmap = {int(c):v for c,v in cmap.items()}
        
        return cls(util_maps, domain=domain, ucounts=ucounts, cmap=cmap)

    @classmethod
    def from_string(cls, uprof_str): 
        """
        Returns a profile of utilities described by ``uprof_str``.

        ``uprof_str`` must be in the format produced by the :meth:`pref_voting.UtilityProfile.write` function.
        """
        uprof_data = uprof_str.split(";")

        num_alternatives,num_voters,utilities = int(uprof_data[0]),int(uprof_data[1]),uprof_data[2:]

        util_maps = [{int(cu.split(":")[0]): float(cu.split(":")[1]) for cu in utils.split(",")} if utils != '' else {} for utils in utilities]

        if len(util_maps) != num_voters: 
            raise Exception("Number of voters does not match the number of utilities.")
        
        return cls(util_maps, domain=range(num_alternatives))

    def display(self, vmap = None, show_totals=False):
        """Display a utility profile as an ascii table (using tabulate). If ``show_totals`` is true then the sum, min, and max of the utilities are displayed.

        """
        
        utilities = self.utilities
        
        vmap = vmap if vmap is not None else {vidx: str(vidx + 1) for vidx in range(len(utilities))}
        voters = range(len(utilities))
        
        if show_totals: 
            tbl ={"Voter" : [vmap[v] for v in voters] + [SEPARATING_LINE] + ["Sum", "Min", "Max"]}
            tbl.update({self.cmap(x): [utilities[v](x) for v in voters] + [SEPARATING_LINE] + [self.util_sum(x), self.util_min(x), self.util_max(x)] for x in self.domain})
        else: 
            tbl ={"Voter" : [vmap[v] for v in voters]}
            tbl.update({str(x): [utilities[v](x) for v in voters] for x in self.domain})
        print( tabulate(tbl, headers="keys"))

    def __getstate__(self):
        # Serialize only the essential data
        state = {
            'utilities': [u.as_dict() for u in self._utilities],
            'ucounts': self.ucounts,
            'domain': self.domain,
            'cmap': self.cmap
        }
        return state

    def __setstate__(self, state):
        # Restore essential data
        self.domain = state['domain']
        self.cmap = state['cmap']
        self.ucounts = state['ucounts']

        self._utilities = [Utility(u_dict, domain=self.domain, cmap=self.cmap) 
                           for u_dict in state['utilities']]

        self.num_voters = sum(self.ucounts)

def write_utility_profiles_to_json(uprofs, filename):
    """Write a list of utility profiles to a json file."""

    uprofs_json = [uprof.as_dict() for uprof in uprofs]
    with open(filename, "w") as f:
        json.dump(uprofs_json, f)

def read_utility_profiles_from_json(filename):
    """Read a list of utility profiles to a json file."""

    with open(filename, "r") as f:
        uprofs_json = json.load(f)
    return [UtilityProfile.from_json(uprof_json) for uprof_json in uprofs_json]