'''
    File: mappings.py
    Author: Wes Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)
    Date: September 23, 20203
    
    Classes that represent mappings of utilities/grades to candidates.
'''

import functools
import numpy as np
from pref_voting.rankings import Ranking

# turn off future warnings.
# getting the following warning when calling tabulate to display a profile: 
# /Library/Frameworks/Python.framework/Versions/3.8/lib/python3.8/site-packages/tabulate.py:1027: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
#  if headers == "keys" and not rows:
# see https://stackoverflow.com/questions/40659212/futurewarning-elementwise-comparison-failed-returning-scalar-but-in-the-futur
#
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# some helper functions

def val_map_function(v, val_map):
    return "None" if v is None else val_map[v]

def default_compare_function(v1, v2):
    if v1 > v2:
        return 1
    elif v2 > v1:
        return -1
    else:
        return 0

class _Mapping(object): 
    """
    A partial function on a set of items. 

    Attributes:
        mapping: a dictionary representing the mapping
        domain: the domain of the mapping
        codomain: the codomain of the mapping
        item_map: a dictionary mapping items to their names
        val_map: a dictionary mapping values to their names
        compare_function: a function used to compare values 
    """
    def __init__(
        self, 
        mapping, # a dictionary representing the mapping
        domain=None, # if domain is None, then it is assumed to be all keys (so mapping is a total function)
        codomain=None, # if codomain is None, then it is assumed to be any number
        compare_function=None, # function used to compare values
        item_map=None, # a dictionary mapping items to their names
        val_map=None, # a dictionary mapping values to their names
        ):
        
        assert domain is None or all([x in domain for x in mapping.keys()]), f"Not all keys in {mapping} are in the domain {domain}."
        
        assert domain is not None or all([isinstance(v, (int, float)) for v in mapping.values()]), f"Not all values in {mapping} are numbers."
        
        assert codomain is None or all([v in codomain for v in mapping.values()]), f"Not all values in {mapping} are in the codomain {codomain}."

        self.mapping = mapping
        self.domain = domain if domain is not None else sorted(list(mapping.keys())) # if domain is None, then it is assumed to be all keys (so mapping is a total function)
        self.codomain = codomain # if codomain is None, then it is assumed to be any number
        self.item_map = item_map if item_map is not None else {x:str(x) for x in self.domain} # a dictionary mapping items in the domain to their names

        val_map = val_map if val_map is not None else {v:str(v) for v in self.mapping.values()} # a dictionary mapping values to their names
        self.val_map = functools.partial(val_map_function, val_map = val_map) # a function mapping values to their names
        
        self.compare_function = compare_function if compare_function is not None else default_compare_function

    def val(self, x): 
        """
        The value assigned to x by the mapping. If x is in the domain but not defined by the mapping, then None is returned.
        """
        assert x in self.domain, f"{x} not in the domain {self.domain}"
        return self.mapping[x] if x in self.mapping.keys() else None

    def has_value(self, x):
        return x in self.mapping.keys()

    @property
    def defined_domain(self):
        return sorted(list(self.mapping.keys()))

    def inverse_image(self, v):
        """Return all the elements in the domain that are mapped to v."""
        return [x for x in self.domain if self.val(x) == v]
        
    def image(self, items=None): 
        """
        The image of the mapping.
        """
        items = self.defined_domain if items is None else items
        return list([self.val(x) for x in items if self.val(x) is not None])

    @property
    def range(self): 
        """
        The range of the mapping.
        """
        return sorted(list(set(self.mapping.values())))

    def average(self, **kwargs): 
        """
        Return the average utility of all elements in alternatives. If alternatives is None, then find the average utility of all elements that are assigned a utility.
        """

        items = kwargs.get("items", None) or kwargs.get("candidates", None) or kwargs.get("alternatives", None)
                
        assert items is None or all([isinstance(self.val(x), (int, float)) or self.val(x) is None for x in items]), "Not all values are numbers."

        img = self.image(items=items)
        return np.mean(img) if len(img) > 0 else None

    def median(self, **kwargs): 
        """
        Return the median utility of all elements in alternatives. If alternatives is None, then find the average utility of all elements that are assigned a utility.
        """

        items = kwargs.get("items", None) or kwargs.get("candidates", None) or kwargs.get("alternatives", None)

        assert items is None or all([isinstance(self.val(x), (int, float)) or self.val(x) is None for x in items]), "Not all values are numbers."

        img = self.image(items=items)
        return np.median(img) if len(img) > 0 else None

    def compare(self, x, y): 
        """
        Return 1 if the value of x is greater than the value of y, 0 if they are equal, and -1 if the value of x is less than the value of y.

        If either x or y is not defined, then None is returned.
        """
        assert x in self.domain, f"{x} not in the domain {self.domain}"
        assert y in self.domain, f"{y} not in the domain {self.domain}"

        return None if (not self.has_value(x) or not self.has_value(y)) else self.compare_function(self.val(x), self.val(y))

    def extended_compare(self, x, y): 
        """
        Return 1 if the value of x is greater than the value of y or x has a value and y does not have a value, 0 if they are equal or both do not have values, and -1 if the value of y is greater than the value of x or y has a value and x does not have a value.

        If either x or y is not defined, then None is returned.
        """
        assert x in self.domain, f"{x} not in the domain {self.domain}"
        assert y in self.domain, f"{y} not in the domain {self.domain}"

        if self.has_value(x) and not self.has_value(y): 
            return 1
        elif self.has_value(y) and not self.has_value(x):
            return -1
        elif not self.has_value(x) and not self.has_value(y):
            return 0
        else:
            return self.compare_function(self.val(x), self.val(y))

    def strict_pref(self, x, y):
        """Returns True if ``x`` is strictly preferred to ``y``.

        The return value is True when both ``x`` and ``y`` are assigned values and the value of ``x`` is strictly greater than the utility of ``y`` according to the compare function.
        """
        return self.compare(x, y) == 1

    def extended_strict_pref(self, x, y):
        """Returns True if ``x`` is strictly preferred to ``y`` using the extended compare function.

        The return value is True when the value of ``x`` is strictly greater than the value of ``y`` or ``x`` is assigned a value and ``y`` is not assigned a value.
        """
        return self.extended_compare(x, y) == 1

    def indiff(self, x, y):
        """Returns True if ``x`` is indifferent with ``y``.

        The return value is True when both ``x`` and ``y`` are assigned values and the value of ``x`` equals the value of ``y``.
        """
        return self.compare(x, y) == 0

    def extended_indiff(self, x, y):
        """Returns True if ``x`` is indifferent with ``y`` using the ``extended_compare`` function.

        The return value is True when the value of ``x`` equals the value of ``y`` or both ``x`` and ``y`` are not assigned values.
        """
        return self.extended_compare(x, y) == 0

    def weak_pref(self, x, y):
        """Returns True if ``x`` is weakly preferred to ``y``.

        The return value is True when both ``x`` and ``y`` are assigned utilities and the utility of ``x`` is at least as  greater than the utility of ``y``.
        """

        return self.strict_pref(x, y) or self.indiff(x, y)

    def extended_weak_pref(self, x, y):
        """Returns True if ``x`` is weakly preferred to ``y``.

        The return value is True when both ``x`` and ``y`` are assigned utilities and the utility of ``x`` is at least as  greater than the utility of ``y``.
        """

        return self.extended_strict_pref(x, y) or self.extended_indiff(x, y)

    def _indifference_classes(self, items, use_extended=False):
        """
        Return a list of the indifference classes of the items.
        """
        indiff_classes = list()
        processed_items = set()
        compare_fnc = self.extended_compare if use_extended else self.compare
        for x in items:
            if x not in processed_items:
                indiff = [y for y in items if compare_fnc(x, y) == 0]
                if len(indiff) > 0:
                    indiff_classes.append(indiff)
                for y in indiff:
                    processed_items.add(y)
        return indiff_classes

    def sorted_domain(self, extended=False):
        """
        Return a list of the indifference classes sorted according to the compare function (or extended compare function if extended is True).
        """
        indiff_classes = self._indifference_classes(self.domain, use_extended=True) if extended else self._indifference_classes(self.defined_domain)

        compare_fnc = self.extended_compare if extended else self.compare

        key_func = functools.cmp_to_key(lambda xs, ys : compare_fnc(xs[0], ys[0]))

        return sorted(indiff_classes, key=key_func, reverse=True)

    def as_dict(self):
        """
        Return the mapping as a dictionary.
        """
        return {c: self.val(c) for c in self.defined_domain}

    def display_str(self, func_name): 
        """
        Return a string representation of the mapping.
        """
        return f"{', '.join([f'{func_name}({self.item_map[x]}) = {self.val_map(self.val(x))}' for x in self.domain])}"
        
    def __call__(self, x): 
        return self.val(x)

    def __repr__(self): 
        return f"{self.mapping}"
    
    def __str__(self):
        return f'{", ".join([f"{self.item_map[x]}:{self.val_map(self.val(x))}" for x in self.domain])}'
    
########
# Utility Functions
########

class Utility(_Mapping):
    def __init__(self, utils, **kwargs):
        """Constructor method for the Utility class."""

        if "domain" in kwargs and "candidates" in kwargs:
            raise ValueError("You can only provide either 'domain' or 'candidates', not both.")
        if "domain" in kwargs:
            self.domain = kwargs["domain"]
        elif "candidates" in kwargs:
            self.domain = kwargs["candidates"]
        else:
            self.domain = list(utils.keys())

        self.cmap = {x:str(x) for x in self.domain} if "cmap" not in kwargs else kwargs["cmap"]

        assert self.domain is None or all([x in self.domain for x in utils.keys()]), f"The domain {self.domain} must contain all elements in the utility map {utils}"
        
        super().__init__(utils, domain=self.domain, item_map = self.cmap)

    @property
    def candidates(self):
        return self.domain

    def items_with_util(self, u):
        """Returns a list of the items that are assigned the utility ``u``."""
        return self.inverse_image(u)

    def has_utility(self, x): 
        """Return True if x has a utility."""
        return self.has_value(x)
    
    def remove_cand(self, x):
        """Returns a utility with the item ``x`` removed."""

        new_utils  = {y: self.val(y) for y in self.defined_domain if y != x}
        new_domain = [y for y in self.domain if y != x]
        new_cmap   = {y: self.cmap[y] for y in self.cmap.keys() if y != x}
        return Utility(new_utils, domain=new_domain, cmap=new_cmap)
    
    def to_approval_ballot(self, prob_to_cont_approving=1.0, decay_rate=0.0): 
        """
        Return an approval ballot representation of the mapping.  It is assumed that the voter approves of all candidates with a utility greater than the average of the utilities assigned to the candidates.

        The parameter ``prob_to_cont_approving`` is the probability that the voter continues to approve of candidates after the first candidate is approved.  The parameter ``decay_rate`` is the exponential decay rate constant in the exponential decay of the probability to continue approving.
        """
        avg_grade = self.average()

        main_approval_set = {x:self.val(x) for x in self.defined_domain if self.val(x) > avg_grade}

        sorted_approval_set = sorted(main_approval_set.items(), key=lambda a: a[1], reverse=True)

        # initialize approval set with the candidate with the highest utility
        approval_set = [sorted_approval_set[0][0]]

        t = 0
        for x, u in sorted_approval_set[1:]:
            if np.random.rand() < prob_to_cont_approving * np.exp(-decay_rate * t):
                approval_set.append(x)
                t += 1
            else:
                break
            
        return Grade(
            {
                x: 1 if x in approval_set else 0 for x in self.defined_domain
            },
            [0, 1],
            candidates=self.domain,
            cmap=self.cmap
        )

    def to_k_approval_ballot(self, k, prob_to_cont_approving=1.0, decay_rate=0.0): 
        """
        Return an k-approval ballot representation of the mapping.  It is assumed that the voter approves of the top k candidates with a utility greater than the average of the utilities assigned to the candidates.

        The parameter ``prob_to_cont_approving`` is the probability that the voter continues to approve of candidates after the first candidate is approved.  The parameter ``decay_rate`` is the exponential decay rate constant in the exponential decay of the probability to continue approving.
        """
        avg_grade = self.average()

        main_approval_set = {x:self.val(x) for x in self.defined_domain if self.val(x) > avg_grade}

        sorted_approval_set = sorted(main_approval_set.items(), key=lambda a: a[1], reverse=True)

        # initialize approval set with the candidate with the highest utility
        approval_set = [sorted_approval_set[0][0]]

        t = 0
        for x, u in sorted_approval_set[1:]:
            if np.random.rand() < prob_to_cont_approving * np.exp(-decay_rate * t):
                approval_set.append(x)
                t += 1
            else:
                break

            if len(approval_set) == k:
                break
            
        return Grade(
            {
                x: 1 if x in approval_set else 0 for x in self.defined_domain
            },
            [0, 1],
            candidates=self.domain,
            cmap=self.cmap
        )

    def ranking(self): 
        """Return the ranking generated by this utility function."""
        return Ranking(
            {x:idx + 1 for idx, indiff_class in enumerate(self.sorted_domain()) 
             for x in indiff_class})
    
    def extended_ranking(self): 
        """Return the ranking generated by this utility function."""

        return Ranking(
            {x:idx + 1 for idx, indiff_class in enumerate(self.sorted_domain(extended=True))
             for x in indiff_class})

    def has_tie(self, use_extended=False): 
        """Return True when there are at least two candidates that are assigned the same utility."""
        return any([len(cs) != 1 for cs in self.sorted_domain(extended=use_extended)])

    def is_linear(self, num_cands):
        """Return True when the assignment of utilities is a linear order of ``num_cands`` candidates. 
        """

        return self.ranking().is_linear(num_cands=num_cands)

    def represents_ranking(self, r, use_extended=False): 
        """Return True when the utility represents the ranking ``r``."""
        
        if use_extended: 
            for x in r.cands: 
                for y in r.cands: 
                    if r.extended_strict_pref(x, y) and not self.extended_strict_pref(x, y): 
                        return False
                    elif r.extended_indiff(x, y) and not self.extended_indiff(x, y): 
                        return False

        else:
            for x in r.cands: 
                if not self.has_utility(x):
                    return False
            for x in r.cands: 
                for y in r.cands: 
                    if r.strict_pref(x, y) and not self.strict_pref(x, y): 
                        return False
                    elif r.indiff(x, y) and not self.indiff(x, y): 
                        return False
        return True
    
    def transformation(self, func): 
        """
        Return a new utility function that is the transformation of this utility function by the function ``func``.        
        """
        return Utility({
            x: func(self.val(x)) for x in self.defined_domain
        }, 
        domain = self.domain, 
        cmap=self.cmap)
    
    def linear_transformation(self, a=1, b=0): 
        """Return a linear transformation of the utility function: ``a * u(x) + b``.
        """
        
        lin_func = lambda x: a * self.val(x) + b
        return self.transformation(lin_func)
    
    def normalize_by_range(self): 
        """Return a normalized utility function.  Applies the *Kaplan* normalization to the utility function: 
        The new utility of an element x of the domain is (u(x) - min({u(x) | x in the domain})) / (max({u(x) | x in the domain })).
        """

        max_util = max(self.range)
        min_util = min(self.range)
        
        if max_util == min_util: 
            return Utility(
                {x:0 for x in self.defined_domain}, 
                domain = self.domain, 
                cmap=self.cmap)
        else: 
            return Utility({
                x: (self.val(x) - min_util) / (max_util - min_util) for x in self.defined_domain
            }, 
            domain = self.domain, 
            cmap=self.cmap)
        
    def normalize_by_standard_score(self): 
        """Replace each utility value with its standard score.  The standard score of a value is the number of standard deviations it is above the mean.
        """

        utility_values = np.array(list(self.image()))
    
        mean_utility = np.mean(utility_values)
        std_dev_utility = np.std(utility_values)

        return Utility({
                x: (self.val(x) - mean_utility) / std_dev_utility for x in self.defined_domain
            }, 
            domain = self.domain, 
            cmap=self.cmap)
        
    def expectation(self, prob):
        """Return the expected utility given a probability distribution ``prob``."""

        assert all([x in self.domain for x in prob.keys()]), "The domain of the probability distribution must be a subset of the domain of the utility function."

        return sum([prob[x] * self.util(x) for x in self.domain if x in prob.keys() and self.has_utility(x)])

    @classmethod
    def from_linear_ranking(cls, ranking, seed=None):
        """
        Return a utility function that represents the linear ranking.

        Parameters:
        ranking (List[int]): A list representing the linear ranking.
        seed (Optional[int]): An optional seed for random number generation.

        Returns:
        Utility: An instance of the Utility class.
        """

        if not (isinstance(ranking, list) or isinstance(ranking, tuple)):
            raise ValueError("Ranking must be a list.")
        if not len(set(ranking)) == len(ranking):
            raise ValueError("Ranking must be a list of unique numbers.")

        num_cands = len(ranking)
        rng = np.random.default_rng(seed)
        
        utilities = sorted(rng.random(size=num_cands), reverse=True)
        u_dict = {c: u for c, u in zip(ranking, utilities)}

        return cls(u_dict)

    def __str__(self):
        return self.display_str("U")

######
# Grade Functions
######

class Grade(_Mapping):
    def __init__(
        self, 
        grade_map, 
        grades, 
        candidates=None, 
        cmap=None, 
        gmap=None, 
        compare_function=None):
        """Constructor method for a Grade function."""

        assert all([g in grades for g in grade_map.values()]), f"All the grades in the grade map {grade_map} must be in the grades {grades}"
        assert candidates is None or all([x in candidates for x in grade_map.keys()]), f"The candidates {candidates} must contain all elements in the grade map {grade_map}"

        self.candidates = sorted(candidates) if candidates is not None else sorted(list(grade_map.keys())) 

        self.cmap = {x:str(x) for x in self.candidates} if cmap is None else cmap

        self.grades = grades 

        self.gmap = {g:str(g) for g in self.grades} if gmap is None else gmap

        super().__init__(
            grade_map, 
            domain=self.candidates, 
            codomain=self.grades, 
            item_map=self.cmap, 
            val_map=self.gmap,
            compare_function=compare_function)

    @property
    def graded_candidates(self):
        """Returns a list of the items that are assigned a grade."""
        return self.defined_domain

    def candidates_with_grade(self, g):
        """Returns a list of the items that are assigned the grade ``g``."""
        return self.inverse_image(g)

    def has_grade(self, x): 
        """Return True if x has a grade."""
        return self.has_value(x)
    
    def remove_cand(self, x):
        """Returns a grade function with the item ``x`` removed."""

        new_grades  = {y: self.val(y) for y in self.defined_domain if y != x}
        new_candidates = [y for y in self.domain if y != x]
        new_cmap   = {y: self.cmap[y] for y in self.cmap.keys() if y != x}
        return Grade(new_grades, grades=self.grades, candidates=new_candidates, cmap=new_cmap, gmap=self.gmap, compare_function=self.compare_function)
    
    def ranking(self): 
        """Return the ranking generated by this grade function."""

        return Ranking(
            {x:idx + 1 for idx, indiff_class in enumerate(self.sorted_domain()) 
             for x in indiff_class})
    
    def extended_ranking(self): 
        """Return the ranking generated by this grade function."""

        return Ranking(
            {x:idx + 1 for idx, indiff_class in enumerate(self.sorted_domain(extended=True))
             for x in indiff_class})

    def has_tie(self, use_extended=False): 
        """Return True when the utility has a tie."""
        return any([len(cs) != 1 for cs in self.sorted_domain(extended=use_extended)])

    def is_linear(self, num_cands):
        """Return True when the assignment of grades is a linear order of ``num_cands`` candidates. 
        """

        return self.ranking().is_linear(num_cands=num_cands)
    
    def __str__(self):
        return self.display_str("grade")


