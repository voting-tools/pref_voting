"""
    File: profiles_with_ties.py
    Author: Wes Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)
    Date: January 5, 2022
    Updated: July 13, 2022
    Updated: December 19, 2022

    Functions to reason about rankings of candidates.
"""

import copy
from tabulate import tabulate

class Ranking(object):
    """A ranking of a set of candidates.

    A ranking is a map from candidates to ranks (integers).  There is no assumption that all candidates in an election are ranked.

    :param rmap: Dictionary in which the keys are the candidates and the values are the ranks.
    :type rmap: dict[int or str: int]
    :param cmap: Dictionary mapping candidates (keys of the ``rmap``) to candidate names (strings).  If not provided, each candidate  is mapped to itself.
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
        """Constructor method"""

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

        _ranks = list(self.rmap.values()) if cs is None else [self.rmap[c] for c in cs if c in self.rmap.keys()]
        _cands = list(self.rmap.keys()) if cs is None else cs
        min_rank = min(_ranks) if len(_ranks) > 0 else None
        return sorted([c for c in _cands if c in self.rmap.keys() and self.rmap[c] == min_rank])

    def last(self, cs=None):
        """Returns the list of candidates from ``cs`` that have the worst ranking.   If ``cs`` is None, then use all the ranked candidates."""

        _ranks = list(self.rmap.values()) if cs is None else [self.rmap[c] for c in cs if c in self.rmap.keys()]
        _cands = list(self.rmap.keys()) if cs is None else cs
        max_rank = max(_ranks) if len(_ranks) > 0 else None
        return sorted([c for c in _cands if c in self.rmap.keys() and self.rmap[c] == max_rank])

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

    def to_linear(self): 
        """
        If the ranking has no ties, return 
        a tuple representing the ranking; otherwise, return None.
        """
        if self.has_tie():
            return None
        else:   
            return tuple([c for c, r in sorted(self.rmap.items(), key=lambda x: x[1])])
        
    def is_truncated_linear(self, num_cands): 
        """Return True when the ranking is a truncated linear order, so it is linear but ranks fewer than ``num_cands`` candidates. 
        """
        return  not self.has_tie() and len(self.rmap.keys()) < num_cands
    
    def has_skipped_rank(self): 
        """Returns True when a rank is skipped."""

        return len(self.ranks) != 0 and self.ranks != list(range(1, len(self.ranks) + 1))

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

    def to_indiff_list(self): 
        """
        Returns the ranking as a tuple of indifference classes (represented as a tuple).
        """
        return tuple([tuple(self.cands_at_rank(r)) for r in self.ranks])
    
    def to_weak_order(self, candidates):
        """
        Returns the ranking as a weak order over the candidates in the list ``candidates``.
        """
        max_rank = max(self.ranks)
        new_ranks = self.rmap
        for c in candidates:
            if not self.is_ranked(c):
                new_ranks[c] = max_rank + 1

        new_cmap = {c: self.cmap[c] if c in self.cmap.keys() else f'{c}' for c in candidates}
        return Ranking(new_ranks, cmap=new_cmap)
    
    def display(self, cmap = None): 
        """
        Display the ranking vertically as a column of a table. 
        
        :Example:

        .. exec_code:: python

            from pref_voting.profiles_with_ties import Ranking
            r = Ranking({0:2, 1:1, 2:3})
            print(r)
            r.display()
            print()

            r = Ranking({0:1, 1:1, 2:3})
            print(r)
            r.display()

            print()
            r = Ranking({0:1,  2:3})
            print(r)
            r.display()

        """
        cmap = cmap if cmap is not None else self.cmap
        _r = copy.deepcopy(self)
        _r.normalize_ranks()
        print(
            tabulate([[" ".join([
                str(self.cmap[c])
                for c in _r.cands_at_rank(rank)])] 
                      for rank in _r.ranks],
                     tablefmt="pretty")
        )
     
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
                r_str += "( " + " ".join(map(lambda c: str(self.cmap[c]) + " ", cands_at_rank)) + ") "
        return r_str

    def __getitem__(self, r):
        """Returns the item at rank r + 1 if it is unique, otherwise return the list of items at rank r+1.  Raises an exception if there is no item at rank r+1."""

        normalized_ranks = {c: self.ranks.index(r) + 1 for c, r in self.rmap.items()}

        ranks = sorted(list(set(normalized_ranks.values())))

        assert r < len(ranks), "There is no item at rank " + str(r + 1)
        cands_at_rank = [c for c,crank in normalized_ranks.items() if crank == ranks[r]]

        return cands_at_rank[0] if len(cands_at_rank) == 1 else cands_at_rank
    
    def __eq__(self, other): 
        
        """
        Returns True if the rankings are the same.  
        
        :Example:

        .. exec_code:: python

            from pref_voting.profiles_with_ties import Ranking

            r = Ranking({1:2, 2:3})            
            r2 = Ranking({1:1, 2:2})
            r3 = Ranking({1:1})

            print(r == r2) # True
            print(r == r3) # False
        
        """
        
        self_ranks = self.ranks
        other_ranks = other.ranks
        
        if len(self_ranks) != len(other_ranks): 
            return False

        for self_rank, other_rank in zip(self_ranks, other_ranks): 
            if set(self.cands_at_rank(self_rank)) != set(other.cands_at_rank(other_rank)): 
                return False
        return True


def break_ties_alphabetically(ranking):
    """Break ties in the ranking alphabetically.

    Args:
        ranking (Ranking): A ranking object

    Returns:
        A ranking object
    """
    candidates = ranking.cands

    new_ranking_dict = {}

    n = 0
    level = 0

    while n < len(candidates):
        sorted_cands_at_rank = ranking.cands_at_rank(level)
        for c in sorted_cands_at_rank:
            new_ranking_dict[c] = n
            n += 1
        level += 1

    return Ranking(new_ranking_dict)