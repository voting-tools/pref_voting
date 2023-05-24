'''
    File: profiles.py
    Author: Eric Pacuit (epacuit@umd.edu)
    Date: December 7, 2020
    Updated: January 5, 2022
    Updated: July 9, 2022
    
    Functions to reason about profiles of linear orders.
'''


from math import ceil
import numpy as np
#from numba import jit #  remove until numba supports python 3.11
import networkx as nx
from tabulate import tabulate
import matplotlib.pyplot as plt
from pref_voting.weighted_majority_graphs import MajorityGraph, MarginGraph, SupportGraph
from pref_voting.voting_method import _num_rank_first

# turn off future warnings.
# getting the following warning when calling tabulate to display a profile: 
# /Library/Frameworks/Python.framework/Versions/3.8/lib/python3.8/site-packages/tabulate.py:1027: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
#  if headers == "keys" and not rows:
# see https://stackoverflow.com/questions/40659212/futurewarning-elementwise-comparison-failed-returning-scalar-but-in-the-futur
#
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# #######
# Internal compiled functions to optimize reasoning with profiles
# #######

#@jit(fastmath=True)
def isin(arr, val):
    """optimized function testing if the value val is in the array arr
    """
    for i in range(arr.shape[0]):
        if (arr[i]==val):
            return True
    return False

#@jit(nopython=True)
def _support(ranks, rcounts, c1, c2):
    """The number of voters that rank candidate c1 over candidate c2
    
    Parameters
    ----------
    rankings:  2d numpy array
        list of linear orderings of the candidates  
    rcounts:   1d numpy array
        list of number of voters for each ranking
    c1: int
        a candidate
    c2: int
        a candidate. 
    
    Key assumptions: 
        * the candidates are named 0...num_cands - 1, and c1 and c2 are 
          numbers between 0 and num_cands - 1
        * voters submit linear orders over the candidate        
    """
    
    diffs = ranks[0:,c1] - ranks[0:,c2] # for each voter, the difference of the ranks of c1 and c2
    diffs[diffs > 0] = 0 # c1 is ranked below c2  
    diffs[diffs < 0] = 1 # c1 is ranked above c2 
    num_rank_c1_over_c2 = np.multiply(diffs, rcounts) # mutliply by the number of each ranking
    return np.sum(num_rank_c1_over_c2)

#@jit(fastmath=True,nopython=True)
def _margin(tally, c1, c2): 
    """The margin of c1 over c2: the number of voters that rank c1 over c2 minus 
    the number of voters that rank c2 over c1
    
    Parameters
    ----------
    tally:  2d numpy array
        the support for each pair of candidates  
    """
    return tally[c1][c2] - tally[c2][c1]


#@jit(nopython=True)
def _num_rank(rankings, rcounts, cand, level):
    """The number of voters that rank cand at level 

    Parameters
    ----------
    rankings:  2d numpy array
        list of linear orderings of the candidates  
    rcounts:   1d numpy array
        list of number of voters for each ranking
    """
    cands_at_level =  rankings[0:,level-1] # get all the candidates at level
    is_cand = cands_at_level == cand # set to 0 each candidate not equal to cand
    return np.sum(is_cand * rcounts) 

#@jit(nopython=True)
def _borda_score(rankings, rcounts, num_cands, cand):
    """The Borda score for cand 

    Parameters
    ----------
    rankings:  2d numpy array
        list of linear orderings of the candidates  
    rcounts:   1d numpy array
        list of number of voters for each ranking
    """
    
    bscores = np.arange(num_cands)[::-1]
    levels = np.arange(1,num_cands+1)
    num_ranks = np.array([_num_rank(rankings, rcounts, cand, level) for level in levels])
    return  np.sum(num_ranks * bscores)

#@jit(nopython=True)
def _find_updated_profile(rankings, cands_to_ignore, num_cands):
    """Optimized method to remove all candidates from cands_to_ignore
    from a list of rankings. 
    
    Parameters
    ----------
    rankings:  2d numpy array
        list of linear orderings of the candidates  
    cands_to_ignore:  1d numpy array
        list of candidates to ignore
    num_cands: int 
        the number of candidates in the original profile
    """
    updated_cand_num = num_cands - cands_to_ignore.shape[0]
    updated_prof_ranks = np.empty(shape=(rankings.shape[0],updated_cand_num), dtype=np.int32)
    
    for vidx in range(rankings.shape[0]):
        levels_idx = np.empty(num_cands - cands_to_ignore.shape[0], dtype=np.int32)
        _r = rankings[vidx]
        _r_level = 0
        for lidx in range(0, levels_idx.shape[0]): 
            for _r_idx in range(_r_level, len(_r)):
                if not isin(cands_to_ignore, _r[_r_idx]):
                    levels_idx[lidx]=_r_idx
                    _r_level = _r_idx + 1
                    break
        updated_prof_ranks[vidx] = np.array([_r[l] for l in levels_idx])
    return updated_prof_ranks

# #######
# Profiles
# #######

class Profile(object):
    """An anonymous profile of linear rankings of :math:`n` candidates.  It is assumed that the candidates are named :math:`0, 1, \ldots, n-1` and a ranking of the candidates is a list of candidate names.  For instance, the list ``[0, 2, 1]`` represents the ranking in which :math:`0` is ranked above :math:`2`, :math:`2` is ranked above :math:`1`, and :math:`0` is ranked above :math:`1`.   

    :param rankings: List of rankings in the profile, where a ranking is a list of candidates.
    :type rankings: list[list[int]]
    :param rcounts: List of the number of voters associated with each ranking.  Should be the same length as rankings.   If not provided, it is assumed that 1 voters submitted each element of ``rankings``.   
    :type rcounts: list[int], optional
    :param cmap: Dictionary mapping candidates (integers) to candidate names (strings).  If not provided, each candidate name is mapped to itself. 
    :type cmap: dict[int: str], optional

    :Example:
    
    The following code creates a profile in which 
    2 voters submitted the ranking ``[0, 1, 2]``, 3 voters submitted the ranking ``[1, 2, 0]``, and 1 voter submitted the ranking ``[2, 0, 1]``: 

    .. code-block:: python

            prof = Profile([[0, 1, 2], [1, 2, 0], [2, 0, 1]], [2, 3, 1])
    
    .. warning:: In profiles with :math:`n` candidates, the candidates must be named using the integers :math:`0, 1, 2, \ldots, n`. So, the following will produce an error: ``Profile([[0, 1, 3]])``.
    """

    def __init__(self, rankings, rcounts=None, cmap=None):
        """constructor method"""
        
        self.num_cands = len(rankings[0]) if len(rankings) > 0 else 0 
        """The number of candidates"""

        self.candidates = list(range(0, self.num_cands)) 
        
        # linear ordering of the candidates for each voter
        self._rankings = np.array(rankings)   
        
        # for number of each ranking
        self._rcounts = np.array([1]*len(rankings)) if rcounts is None else np.array(rcounts) 
        
        # for each voter, the rankings of each candidate
        self._ranks = np.array([[_r.index(c) + 1 for c in self.candidates] for  _r in rankings])
        
        # 2d array where the c,d entry is the support of c over d
        self._tally = np.array([[_support(self._ranks, self._rcounts, c1, c2) 
                                 for c2 in self.candidates] for c1 in self.candidates ])
        
        # mapping candidates to candidate names
        self.cmap = cmap if cmap is not None else {c:str(c) for c in self.candidates}
                
        # total number of voters
        self.num_voters = np.sum(self._rcounts)
        """The number of voters in the election."""

    @property
    def rankings_counts(self):
        """
        Returns the submitted rankings and the list of counts. 
        """

        return self._rankings, self._rcounts
    
    @property
    def rankings(self): 
        """
        Returns a list of all the rankings in the profile .   The type is a list of tuples of integers. 
        """
        
        return [tuple(r) for ridx,r in enumerate(self._rankings) for n in range(self._rcounts[ridx])]

    @property
    def counts(self): 
        """
        Returns a list of the counts of  the rankings in the profile.   The type is a list of integers. 
        """
        
        return list(self._rcounts)
        
    def support(self, c1, c2):
        """The number of voters that rank :math:`c1` above :math:`c2`
        
        :param c1: the first candidate
        :type c1: int
        :param c2: the second candidate
        :type c2: int
        :rtype: int

        """

        return self._tally[c1][c2]
    
    def margin(self, c1, c2):
        """The number of voters that rank :math:`c1` above :math:`c2` minus the number of voters that rank :math:`c2` above :math:`c1`.
        
        :param c1: the first candidate
        :type c1: int
        :param c2: the second candidate
        :type c2: int
        :rtype: int

        """
        return _margin(self._tally, c1, c2)
        
    def majority_prefers(self, c1, c2): 
        """Returns true if more voters rank :math:`c1` over :math:`c2` than :math:`c2` over :math:`c1`; otherwise false. 
        
        :param c1: the first candidate
        :type c1: int
        :param c2: the second candidate
        :type c2: int
        :rtype: bool

        """

        return _margin(self._tally, c1, c2) > 0

    def is_tied(self, c1, c2):
        """Returns True if ``c1`` tied with ``c2``.  That is, the same number of voters rank ``c1`` over ``c2`` as ``c2`` over ``c1``. 
        """ 

        return _margin(self._tally, c1, c2) == 0

    def cycles(self): 
        """Return a list of the cycles in the profile."""

        return self.margin_graph().cycles()

    def num_rank(self, c, level): 
        """The number of voters that rank candidate ``c`` at position ``level``

        :param c: the candidate
        :type c: int
        :param level: the position of the candidate in the rankings
        :type level: int

        """

        return _num_rank(self._rankings, self._rcounts, c, level=level)
        
    def plurality_scores(self, curr_cands = None):
        """The plurality scores in the profile restricted to the candidates in ``curr_cands``. 

        The **plurality score** for candidate :math:`c` is the number of voters that rank :math:`c` in first place. 

        :param curr_cands: restrict attention to candidates in this list. Defaults to all candidates in the profile if not provided. 
        :type curr_cands: list[int], optional
        :returns: a dictionary associating each candidate in ``curr_cands`` with its plurality score. 
        """        
        rankings, rcounts = self.rankings_counts

        curr_cands = self.candidates if curr_cands is None else curr_cands
        cands_to_ignore = np.array([c for c in self.candidates if c not in curr_cands])
        
        return {c: _num_rank_first(rankings, rcounts, cands_to_ignore, c) for c in curr_cands}

    def borda_scores(self, curr_cands = None):
        """The Borda scores in the profile restricted to the candidates in ``curr_cands``. 

        The **Borda score** for candidate :math:`c` is calculate as follows: the score assigned to :math:`c` by a ranking is the number of candidates ranked below :math:`c`.  The Borda score is the sum of the score assigned to :math:`c` by each ranking in the ballot. 
        
        number of voters that rank :math:`c` in first place. 

        :param curr_cands: restrict attention to candidates in this list. Defaults to all candidates in the profile if not provided. 
        :type curr_cands: list[int], optional
        :returns: a dictionary associating each candidate in ``curr_cands`` with its Borda score. 
        """        
        
        rankings = self._rankings if curr_cands is None else _find_updated_profile(self._rankings, np.array([c for c in self.candidates if c not in curr_cands]), len(self.candidates))
        curr_cands = self.candidates if curr_cands is None else curr_cands
        
        num_cands = len(curr_cands)
        return {c: _borda_score(rankings, self._rcounts, num_cands, c) for c in curr_cands}
    
    def dominators(self, cand, curr_cands = None): 
        """Returns the list of candidates that are majority preferred to ``cand`` in the profile restricted to the candidates in ``curr_cands``. 
        """        
        candidates = self.candidates if curr_cands is None else curr_cands
        
        return [c for c in candidates if self.majority_prefers(c, cand)]

    def dominates(self, cand, curr_cands = None): 
        """Returns the list of candidates that ``cand`` is majority preferred to in the profiles restricted to ``curr_cands``.
        """
        candidates = self.candidates if curr_cands is None else curr_cands
        
        return [c for c in candidates if self.majority_prefers(cand, c)]
    
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

    def condorcet_winner(self, curr_cands = None):
        """Returns the Condorcet winner in the profile restricted to ``curr_cands`` if one exists, otherwise return None.

        The **Condorcet winner** is the candidate that is majority preferred to every other candidate. 
        """
        
        curr_cands = curr_cands if curr_cands is not None else self.candidates

        cw = None
        for c1 in curr_cands: 
            if all([self.majority_prefers(c1,c2) for c2 in curr_cands if c1 != c2]): 
                cw = c1
                break # if a Condorcet winner exists, then it is unique
        return cw

    def weak_condorcet_winner(self, curr_cands = None):
        """Returns a list of the weak Condorcet winners in the profile restricted to ``curr_cands`` (which may be empty).

        A candidate :math:`c` is a  **weak Condorcet winner** if there is no other candidate that is majority preferred to :math:`c`.

        .. note:: While the Condorcet winner is unique if it exists, there may be multiple weak Condorcet winners.    
        """

        curr_cands = curr_cands if curr_cands is not None else self.candidates

        weak_cw = list()
        for c1 in curr_cands: 
            if not any([self.majority_prefers(c2,c1) for c2 in curr_cands if c1 != c2]): 
                weak_cw.append(c1)
        return sorted(weak_cw) if len(weak_cw) > 0 else None

    def condorcet_loser(self, curr_cands = None):
        """Returns the Condorcet loser in the profile restricted to ``curr_cands`` if one exists, otherwise return None.

        A candidate :math:`c` is a  **Condorcet loser** if every other candidate  is majority preferred to :math:`c`.
        """

        curr_cands = curr_cands if curr_cands is not None else self.candidates

        cl = None
        for c1 in curr_cands: 
            if all([self.majority_prefers(c2,c1) for c2 in curr_cands if c1 != c2]): 
                cl = c1
                break # if a Condorcet loser exists, then it is unique
        return cl
    
    def strict_maj_size(self):
        """Returns the strict majority of the number of voters.  
        """

        # return the size of  strictly more than 50% of the voters
        
        return int(self.num_voters/2 + 1 if self.num_voters % 2 == 0 else int(ceil(float(self.num_voters)/2)))

    def margin_graph(self): 
        """Returns the margin graph of the profile.  See :class:`.MarginGraph`.  
        """
    
        return MarginGraph.from_profile(self)

    def support_graph(self): 
        """Returns the margin graph of the profile.  See :class:`.SupportGraph`.  
        """
    
        return SupportGraph.from_profile(self)

    def majority_graph(self): 
        """Returns the majority graph of the profile.  See :class:`.MarginGraph`.  
        """
    
        return MajorityGraph.from_profile(self)

    @property
    def margin_matrix(self):
        """Returns the margin matrix of the profile: A matrix where the :math:`i, j` entry is the margin of candidate :math:`i` over candidate :math:`j`.    
        """

        return [[self.margin(c1,c2) for c2 in self.candidates] for c1 in self.candidates]
    
    def is_uniquely_weighted(self): 
        """Returns True if the profile is uniquely weighted. 
        
        A profile is **uniquely weighted** when there are no 0 margins and all the margins between any two candidates are unique.     
        """
        
        return MarginGraph.from_profile(self).is_uniquely_weighted()
    
    def remove_candidates(self, cands_to_ignore):
        """Remove all candidates from ``cands_to_ignore`` from the profile. 

        :param cands_to_ignore: list of candidates to remove from the profile
        :type cands_to_ignore: list[int]
        :returns: a profile with candidates from ``cands_to_ignore`` removed and a dictionary mapping the candidates from the new profile to the original candidate names. 

        .. warning:: Since the candidates in a Profile must be named :math:`0, 1, \ldots, n-1` (where :math:`n` is the number of candidates), you must use the candidate map returned to by the function to recover the original candidate names. 

        :Example: 

        .. exec_code::

            from pref_voting.profiles import Profile 
            prof = Profile([[0,1,2], [1,2,0], [2,0,1]])
            prof.display()
            new_prof, orig_cnames = prof.remove_candidates([1])
            new_prof.display() # displaying new candidates names
            new_prof.display(cmap=orig_cnames) # use the original candidate names
        """        
        updated_rankings = _find_updated_profile(self._rankings, np.array(cands_to_ignore), self.num_cands)
        new_names = {c:cidx  for cidx, c in enumerate(sorted(updated_rankings[0]))}
        orig_names = {v:k  for k,v in new_names.items()}
        return Profile([[new_names[c] for c in r] for r in updated_rankings], rcounts=self._rcounts, cmap=self.cmap), orig_names
    
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
        return Profile(rankings, rcounts=rcounts, cmap=self.cmap)
        
    def display(self, cmap=None, style="pretty", curr_cands=None):
        """Display a profile (restricted to ``curr_cands``) as an ascii table (using tabulate).

        :param cmap: the candidate map to use (overrides the cmap associated with this profile)
        :type cmap: dict[int,str], optional
        :param style: the candidate map to use (overrides the cmap associated with this profile)
        :type style: str ---  "pretty" or "fancy_grid" (or any other style option for tabulate)
        :param curr_cands: list of candidates
        :type curr_cands: list[int], optional
        :rtype: None

        :Example: 

        .. exec_code::

            from pref_voting.profiles import Profile 
            prof = Profile([[0,1,2], [1,2,0], [2,0,1]], [2, 3, 1])
            prof.display()
            prof.display(cmap={0:"a", 1:"b", 2:"c"})

        """
        cmap = cmap if cmap is not None else self.cmap
        
        rankings = self._rankings if curr_cands is None else _find_updated_profile(self._rankings, np.array([c for c in self.candidates if c not in curr_cands]), len(self.candidates))

        print(tabulate([[cmap[c] for c in cs] for cs in rankings.transpose()], self._rcounts, tablefmt=style))        

    def to_latex(self, cmap = None, curr_cands = None):
        """Returns a stringer describing the profile (restricted to ``curr_cands``) as a LaTeX table (use the provided ``cmap`` or the ``cmap`` associated with the profile).

        :Example: 

        .. exec_code::

            from pref_voting.profiles import Profile 
            prof = Profile([[0,1,2], [1,2,0], [2,0,1]], [2, 3, 1])
            print(prof.to_latex())
            print()
            print(prof.to_latex(cmap={0:"a", 1:"b", 2:"c"}))
        """
        
        cmap = cmap if cmap is not None else self.cmap
        rankings = self._rankings if curr_cands is None else _find_updated_profile(self._rankings, np.array([c for c in self.candidates if c not in curr_cands]), len(self.candidates))
        
        cs = 'c' * len(self._rcounts)
        
        latex_str = "\\begin{tabular}{" + str(cs) + "}\n"
        latex_str += " & ".join([f"${rc}$" for rc in self._rcounts]) + "\\\\\hline \n"
        latex_str +=  "\\\\ \n".join([" & ".join([f"${cmap[c]}$" for c in cs])  for cs in rankings.transpose()])
        latex_str += "\n\\end{tabular}"
        
        return latex_str

    def display_margin_matrix(self): 
        """Display the margin matrix using tabulate.
        """
        
        print(tabulate(self.margin_matrix, tablefmt="grid"))   

    def display_margin_graph(self, cmap=None, curr_cands = None):
        """ 
        Display the margin graph of the profile (restricted to ``curr_cands``) using the ``cmap``.  See :class:`.MarginGraph`. 
        """

        cmap = cmap if cmap is not None else self.cmap
        MarginGraph.from_profile(self, cmap=cmap).display(curr_cands = curr_cands)

    def display_margin_graph_with_defeat(self, defeat, curr_cands=None, show_undefeated=True, cmap=None):
        """ 
        Display the margin graph of the profile (restricted to ``curr_cands``) with the defeat edges highlighted using the ``cmap``.  See :class:`.MarginGraph`. 
        """

        MarginGraph.from_profile(self).display_with_defeat(defeat, curr_cands = curr_cands, show_undefeated = show_undefeated, cmap = cmap)

    def write(self): 
        """
        Returns a string encoding the profile for to store in a file.

        The format is based on the format used by PrefLib (https://www.preflib.org/data/format)
        
        the format is: 
        
        - The number of candidates
        - The number of submitted rankings
        - The list of the rankings, one on each line,  where a ranking is represented by ``n:a,b,c`` meaning the rcount is n for the ranking is [a, b, c]
        - List of the cmap, one on each line. 

        :Example: 

        .. exec_code::

            from pref_voting.profiles import Profile 
            prof = Profile([[0,1,2], [1,2,0], [2,0,1]], [2, 3, 1], cmap={0:"a", 1:"b", 2:"c"})
            prof.display()
            print('\\nThe string encoding: \\n')
            print(prof.write())

        """
        election_str = f"{self.num_cands}\n"
        election_str += f"{len(self._rankings)}\n"
        for rc,r in zip(self._rcounts, self._rankings): 
            r_str = ",".join([str(c) for c in r])
            election_str += f"{str(rc)}:{r_str}\n"
        for c,cname in self.cmap.items():
            election_str += f"{c}:{cname}\n"
        return election_str
    
    @classmethod
    def from_string(cls, e_str): 
        """
        Returns a profile described by ``e_str``.

        ``e_str`` must be in the format produced by the :meth:`pref_voting.Profile.write` function.
        """
        edata = e_str.split("\n")

        num_cands = int(edata[0])
        num_rankings = int(edata[1])

        rc_data = edata[2:2+num_rankings]

        cmap_data = edata[2+num_rankings:2+num_rankings + num_cands]
        rankings = list()
        rcounts = list()
        for rcd in rc_data:
            _rc, _r = rcd.split(":")
            rc = int(_rc)
            r = [int(c) for c in _r.split(",")]
            rcounts.append(rc)
            rankings.append(r)

        cmap = {int(c_cname.split(":")[0]): str(c_cname.split(":")[1]) for c_cname in cmap_data}

        return cls(rankings, num_cands, rcounts, cmap)
    
    def description(self): 
        """
        Returns a string describing the profile.
        """
        rs, cs = self.rankings_counts
        return f"Profile({[list(r) for r in rs]}, rcounts={[int(c) for c in cs]}, cmap={self.cmap})"
    
    def __str__(self): 
        # print the profile as a table
        
        return tabulate([[self.cmap[c] for c in cs] for cs in self._rankings.transpose()], self._rcounts, tablefmt="pretty")    

def simple_lift(ranking, c):
    """
    Return a ranking (i.e., a list of candidates) in which ``c`` is moved up one position in ``ranking``.
    """
    assert c != ranking[0], "can't lift a candidate already in first place"
    
    c_idx = ranking.index(c)
    ranking[c_idx - 1], ranking[c_idx] = ranking[c_idx],ranking[c_idx-1]
    return ranking

def simple_drop(ranking, c):
    """
    Return a ranking (i.e., a list of candidates) in which ``c`` is moved down one position in ``ranking``.
    """
    
    assert c != ranking[-1], "can't drop a candidate already in last place"
    
    c_idx = ranking.index(c)
    ranking[c_idx + 1], ranking[c_idx] = ranking[c_idx],ranking[c_idx + 1]
    return ranking
