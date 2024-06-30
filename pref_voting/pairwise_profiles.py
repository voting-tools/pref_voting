'''
    File: pairwise_profiles.py
    Author: Wes Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)
    Date: June 3, 2024
    
    Functions to reason about profiles of pairwise comparisons.
'''


from math import ceil
import numpy as np
from numba import jit  
import networkx as nx
from tabulate import tabulate
import matplotlib.pyplot as plt
from pref_voting.weighted_majority_graphs import MajorityGraph, MarginGraph, SupportGraph
import os

# turn off future warnings.
# getting the following warning when calling tabulate to display a profile: 
# /Library/Frameworks/Python.framework/Versions/3.8/lib/python3.8/site-packages/tabulate.py:1027: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
#  if headers == "keys" and not rows:
# see https://stackoverflow.com/questions/40659212/futurewarning-elementwise-comparison-failed-returning-scalar-but-in-the-futur
#
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class PairwiseComparisons:
    
    def __init__(self, comparisons, candidates=None, cmap=None):
        """Constructor method for PairwiseComparisons.

        Args:
            comparisons (list): List of tuples, lists, or sets representing pairwise comparisons.
            candidates (list or set, optional): Initial set of candidates. Defaults to None.
            cmap (dict, optional): Mapping of candidates to their names. Defaults to None.
        """
        self._comparisons = []
        
        for comp in comparisons:
            if not isinstance(comp, (tuple, list)) or len(comp) != 2:
                raise ValueError("Each element of the list of comparisons should be a tuple or list of length 2.")
            
            if all(isinstance(comp[i], (int, str)) for i in [0, 1]):
                self._comparisons.append(({comp[0], comp[1]}, {comp[0]}))
            elif all(isinstance(comp[i], (set, list, tuple)) for i in [0, 1]):
                self._comparisons.append((set(comp[0]), set(comp[1])))
            else:
                raise ValueError("Each element of the list of comparisons should be a tuple of sets or lists of candidates.")

        if not self.is_coherent():
            raise ValueError("The pairwise comparisons are not coherent.")
        
        self.candidates = sorted(list(set(c for menu, _ in self._comparisons for c in menu)) if candidates is None else candidates)
        self.cmap = cmap if cmap is not None else {c: str(c) for c in self.candidates}

    def is_coherent(self):
        """Check if the pairwise comparisons are coherent.
        
        Returns:
            bool: True if the pairwise comparisons are coherent, False otherwise.
        """
        for menu, choice in self._comparisons:
            if not choice.issubset(menu):
                return False
        menus = [menu for menu, _ in self._comparisons]
        return len(menus) == len(set(frozenset(menu) for menu in menus))

    def weak_pref(self, c1, c2):
        """Return the revealed weak preference of a menu of choices.

        Args:
            c1 (str or int): First candidate.
            c2 (str or int): Second candidate.

        Returns:
            bool: True if there is a weak preference for c1 over c2, False otherwise.
        """
        return any(c1 in menu and c2 in menu and c1 in choice for menu, choice in self._comparisons)

    def strict_pref(self, c1, c2):
        """Return the revealed strict preference of a menu of choices.

        Args:
            c1 (str or int): First candidate.
            c2 (str or int): Second candidate.

        Returns:
            bool: True if there is a strict preference for c1 over c2, False otherwise.
        """
        return self.weak_pref(c1, c2) and not self.weak_pref(c2, c1)

    def indiff(self, c1, c2):
        """Return the revealed indifference of a menu of choices.

        Args:
            c1 (str or int): First candidate.
            c2 (str or int): Second candidate.

        Returns:
            bool: True if there is indifference between c1 and c2, False otherwise.
        """
        return self.weak_pref(c1, c2) and self.weak_pref(c2, c1)

    def has_comparison(self, c1, c2): 
        """Check if there is a comparison between two candidates.

        Args:
            c1 (str or int): First candidate.
            c2 (str or int): Second candidate.

        Returns:
            bool: True if there is a comparison between c1 and c2, False otherwise.
        """
        return any(c1 in menu and c2 in menu for menu, _ in self._comparisons)
    
    def get_comparison(self, c1, c2): 
        """Get the comparison between two candidates.

        Args:
            c1 (str or int): First candidate.
            c2 (str or int): Second candidate.

        Returns:
            tuple: The comparison between c1 and c2.
        """
        comp = [(menu, choice) for menu, choice in self._comparisons if c1 in menu and c2 in menu]
        return comp[0] if len(comp) == 1 else None
    
    def add_comparison(self, menu, choice):
        """Add a new comparison to the existing comparisons.

        Args:
            menu (set): A set of candidates representing the menu.
            choice (set): A set of candidates representing the choice set.

        Raises:
            ValueError: If the new comparison is not coherent with the existing comparisons.
        """
        new_comparison = (set(menu), set(choice))
        self._comparisons.append(new_comparison)
        if not self.is_coherent():
            self._comparisons.pop()
            raise ValueError("The new comparison is not coherent with the existing comparisons.")
        self.candidates = sorted(list(set(c for menu, _ in self._comparisons for c in menu)))

    def add_strict_preference(self, c1, c2):
        """Add a new comparison to the existing comparisons where c1 is strictly preferred to c2.

        Args:
            c1 (int, str): A candidate
            c2 (int, str): A candidate.

        Raises:
            ValueError: If the new comparison is not coherent with the existing comparisons.
        """
        self.add_comparison({c1, c2}, {c1})

    def display(self):
        """Display the pairwise comparisons in a readable format."""
        for menu, choice in self._comparisons:
            menu_str = ", ".join(sorted([self.cmap[c] for c in menu]))
            choice_str = ", ".join(sorted([self.cmap[c] for c in choice]))
            print(f"{{{menu_str}}} -> {{{choice_str}}}")


    def __str__(self):
        """Return the comparisons as a string."""

        str_comparisons = ''
        for menu, choice in self._comparisons:
            menu_str = ", ".join(sorted([self.cmap[c] for c in menu]))
            choice_str = ", ".join(sorted([self.cmap[c] for c in choice]))
            str_comparisons += f"{{{menu_str}}} -> {{{choice_str}}}, "
        return str_comparisons[:-2]
    
    
class PairwiseProfile:
    r"""An anonymous profile of pairwise comparisons.   

    Arguments: 
        pairwise_comparisons: List of comparisons or PairwiseComparisons instances.
    """

    def __init__(self, pairwise_comparisons, candidates=None, rcounts=None, cmap=None):
        """Constructor method for PairwiseProfile.

        Args:
            pairwise_comparisons (list): List of comparisons or PairwiseComparisons instances.
            candidates (list or set, optional): List of candidates. Defaults to None.
            rcounts (list, optional): List of counts for each comparison. Defaults to None.
            cmap (dict, optional): Mapping of candidates to their names. Defaults to None.
        """
        self._pairwise_comparisons = []
        
        for comps in pairwise_comparisons:
            if isinstance(comps, PairwiseComparisons):
                self._pairwise_comparisons.append(comps)
            else:
                self._pairwise_comparisons.append(PairwiseComparisons(comps, candidates=candidates))
        
        if candidates is None:
            candidates = {c for pc in self._pairwise_comparisons for c in pc.candidates}
        self.candidates = sorted(list(candidates))

        self.cand_to_cidx = {c: idx for idx, c in enumerate(self.candidates)}
        self.cidx_to_cand = {idx: c for c, idx in self.cand_to_cidx.items()}
        
        self._rcounts = rcounts if rcounts is not None else [1] * len(pairwise_comparisons)

        self._tally = np.array([[np.sum([count for pc, count in zip(self._pairwise_comparisons, self._rcounts) if pc.strict_pref(c1, c2)]) for c2 in self.candidates] for c1 in self.candidates])
        
        self.cmap = cmap if cmap is not None else {c: str(c) for c in self.candidates}
                
        self.num_voters = np.sum(self._rcounts)
        """The number of voters in the election."""

    @property
    def comparisons_counts(self):
        """Returns the submitted rankings and the list of counts."""
        return self._pairwise_comparisons, self._rcounts

    def support(self, c1, c2):
        """The number of voters that rank `c1` above `c2`.
        
        Args:
            c1 (str or int): The first candidate.
            c2 (str or int): The second candidate.
        
        Returns:
            int: Number of voters that rank `c1` above `c2`.
        """
        return self._tally[self.cand_to_cidx[c1]][self.cand_to_cidx[c2]]
    
    def margin(self, c1, c2):
        """The number of voters that rank `c1` above `c2` minus the number of voters that rank `c2` above `c1`.
        
        Args:
            c1 (str or int): The first candidate.
            c2 (str or int): The second candidate.
        
        Returns:
            int: Margin of votes.
        """
        idx1, idx2 = self.cand_to_cidx[c1], self.cand_to_cidx[c2]
        return self._tally[idx1][idx2] - self._tally[idx2][idx1]
        
    def majority_prefers(self, c1, c2): 
        """Returns true if more voters rank `c1` over `c2` than `c2` over `c1`.
        
        Args:
            c1 (str or int): The first candidate.
            c2 (str or int): The second candidate.
        
        Returns:
            bool: True if `c1` is majority preferred over `c2`, False otherwise.
        """
        return self.margin(c1, c2) > 0

    def is_tied(self, c1, c2):
        """Returns True if `c1` is tied with `c2`. 
        
        Args:
            c1 (str or int): The first candidate.
            c2 (str or int): The second candidate.
        
        Returns:
            bool: True if `c1` is tied with `c2`, False otherwise.
        """
        return self.margin(c1, c2) == 0

    def dominators(self, cand, curr_cands=None): 
        """Returns the list of candidates that are majority preferred to `cand` in the profile restricted to the candidates in `curr_cands`.
        
        Args:
            cand (str or int): The candidate.
            curr_cands (list, optional): List of candidates to consider. Defaults to None.
        
        Returns:
            list: List of candidates that are majority preferred to `cand`.
        """        
        candidates = self.candidates if curr_cands is None else curr_cands
        return [c for c in candidates if self.majority_prefers(c, cand)]

    def dominates(self, cand, curr_cands=None): 
        """Returns the list of candidates that `cand` is majority preferred to in the profile restricted to `curr_cands`.
        
        Args:
            cand (str or int): The candidate.
            curr_cands (list, optional): List of candidates to consider. Defaults to None.
        
        Returns:
            list: List of candidates that `cand` is majority preferred to.
        """
        candidates = self.candidates if curr_cands is None else curr_cands
        return [c for c in candidates if self.majority_prefers(cand, c)]
    
    def copeland_scores(self, curr_cands=None, scores=(1, 0, -1)):
        """The Copeland scores in the profile restricted to the candidates in `curr_cands`.
        
        Args:
            curr_cands (list, optional): List of candidates to consider. Defaults to None.
            scores (tuple, optional): Scores for win, tie, and loss. Defaults to (1, 0, -1).
        
        Returns:
            dict: Dictionary associating each candidate in `curr_cands` with its Copeland score.
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

    def condorcet_winner(self, curr_cands=None):
        """Returns the Condorcet winner in the profile restricted to `curr_cands` if one exists, otherwise return None.
        
        Args:
            curr_cands (list, optional): List of candidates to consider. Defaults to None.
        
        Returns:
            str or int: Condorcet winner if one exists, otherwise None.
        """
        curr_cands = curr_cands if curr_cands is not None else self.candidates
        for c1 in curr_cands: 
            if all(self.majority_prefers(c1, c2) for c2 in curr_cands if c1 != c2): 
                return c1
        return None

    def weak_condorcet_winner(self, curr_cands=None):
        """Returns a list of the weak Condorcet winners in the profile restricted to `curr_cands`.
        
        Args:
            curr_cands (list, optional): List of candidates to consider. Defaults to None.
        
        Returns:
            list: List of weak Condorcet winners.
        """
        curr_cands = curr_cands if curr_cands is not None else self.candidates
        return [c1 for c1 in curr_cands if not any(self.majority_prefers(c2, c1) for c2 in curr_cands if c1 != c2)]

    def condorcet_loser(self, curr_cands=None):
        """Returns the Condorcet loser in the profile restricted to `curr_cands` if one exists, otherwise return None.
        
        Args:
            curr_cands (list, optional): List of candidates to consider. Defaults to None.
        
        Returns:
            str or int: Condorcet loser if one exists, otherwise None.
        """
        curr_cands = curr_cands if curr_cands is not None else self.candidates
        for c1 in curr_cands: 
            if all(self.majority_prefers(c2, c1) for c2 in curr_cands if c1 != c2): 
                return c1
        return None
    
    def strict_maj_size(self):
        """Returns the strict majority of the number of voters.
        
        Returns:
            int: Size of the strict majority.
        """
        return int(self.num_voters / 2 + 1) if self.num_voters % 2 == 0 else int(ceil(float(self.num_voters) / 2))

    def margin_graph(self): 
        """Returns the margin graph of the profile.
        
        Returns:
            dict: Margin graph of the profile.
        """
        
        return MarginGraph(self.candidates, 
                           [(c1, c2, self.margin(c1, c2)) 
                            for c1 in self.candidates 
                            for c2 in self.candidates 
                            if self.majority_prefers(c1, c2)])

    def majority_graph(self): 
        """Returns the margin graph of the profile.
        
        Returns:
            dict: Margin graph of the profile.
        """
        
        return MajorityGraph(self.candidates, 
                             [(c1, c2) 
                              for c1 in self.candidates 
                              for c2 in self.candidates 
                              if self.majority_prefers(c1, c2)])

    def display(self, cmap=None, style="pretty", curr_cands=None):
        """Display the profile (restricted to `curr_cands`) as an ASCII table.
        
        Args:
            cmap (dict, optional): Mapping of candidates to their names. Defaults to None.
            style (str, optional): Style of the display. Defaults to "pretty".
            curr_cands (list, optional): List of candidates to consider. Defaults to None.
        """
        cmap = cmap if cmap is not None else self.cmap
        comparisons, counts = self.comparisons_counts
        for comp_idx, comps in enumerate(comparisons): 
            print(f'{counts[comp_idx]}: {comps}')

    def __add__(self, other_prof): 
        """Returns the sum of two profiles.
        
        Args:
            other_prof (PairwiseProfile): Another PairwiseProfile instance.
        
        Returns:
            PairwiseProfile: The combined profile.
        """
        assert self.candidates == other_prof.candidates, "The two profiles must have the same candidates"
        combined_comparisons = self._pairwise_comparisons + other_prof._pairwise_comparisons
        combined_rcounts = self._rcounts + other_prof._rcounts
        return PairwiseProfile(combined_comparisons, rcounts=combined_rcounts, candidates=self.candidates)
