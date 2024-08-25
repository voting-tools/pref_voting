
from pref_voting.profiles import Profile
from pref_voting.profiles_with_ties import ProfileWithTies
from pref_voting.weighted_majority_graphs import MajorityGraph
from pref_voting.rankings import Ranking
from pref_voting.social_welfare_function import *
from pref_voting.voting_method import *
from itertools import combinations
import random

import networkx as nx

def get_mg(edata, curr_cands = None): 
    
    if curr_cands == None: 
        if type(edata) == Profile or type(edata) == ProfileWithTies: 
            mg = MajorityGraph.from_profile(edata).mg
        else:
            mg = edata.mg
    else: 
        if type(edata) == Profile or type(edata) == ProfileWithTies:  
            mg = nx.DiGraph()
            mg.add_nodes_from(curr_cands)
            mg.add_edges_from([(c1,c2) for c1 in curr_cands for c2 in curr_cands if edata.majority_prefers(c1, c2)])
        else:
            mg = edata.mg.copy()
            mg.remove_nodes_from([c for c in edata.candidates if c not in curr_cands])
    return mg


def get_weak_mg(edata, curr_cands = None): 
    
    if curr_cands == None: 
        if type(edata) == Profile or type(edata) == ProfileWithTies: 
            wmg = MajorityGraph.from_profile(edata).mg
        else:
            wmg = edata.mg
        wmg.add_edges_from([(c1, c2) for c1 in edata.candidates for c2 in edata.candidates if c1 != c2 and edata.is_tied(c1, c2)])
    else: 
        if type(edata) == Profile or type(edata) == ProfileWithTies:  
            wmg = nx.DiGraph()
            wmg.add_nodes_from(curr_cands)
            wmg.add_edges_from([(c1,c2) for c1 in curr_cands for c2 in curr_cands if c1 != c2 and (edata.majority_prefers(c1, c2) or edata.is_tied(c1, c2))])
        else:
            wmg = edata.mg.copy()
            wmg.remove_nodes_from([c for c in edata.candidates if c not in curr_cands])
            wmg.add_edges_from([(c1, c2) for c1 in curr_cands for c2 in curr_cands if c1 != c2 and edata.is_tied(c1, c2)])
    return wmg


def swf_from_vm(vm, tie_breaker = None):
    """
    Given a voting method, returns a social welfare function that uses the voting method to rank the candidates (winners are ranked first; then they are excluded from curr_cands and the new winners are ranked second; etc.).

    Args:
        vm (function): A voting method.
        tie_breaker (str): The tie-breaking method to use. Options are "alphabetic", "random", and None. Default is None.

    Returns:
        function: A social welfare function that uses the voting method to rank the candidates.
    """
    
    def f(prof, curr_cands = None):

        cands = prof.candidates if curr_cands == None else curr_cands

        ranked_cands = list()
        ranking_dict = dict()

        n=0

        while n < len(cands):

            if len(ranked_cands) == len(cands):
                break

            ws = vm(prof, curr_cands = [c for c in cands if c not in ranked_cands])
            ranked_cands = ranked_cands + ws

            if tie_breaker is None:
                for c in ws:
                    ranking_dict[c] = n
                n += 1

            if tie_breaker == "alphabetic":
                sorted_ws = sorted(ws)
                for c in sorted_ws:
                    ranking_dict[c] = n
                    n += 1

            if tie_breaker == "random":
                random.shuffle(ws)
                for c in ws:
                    ranking_dict[c] = n
                    n += 1            

        return Ranking(ranking_dict)
        
    return SocialWelfareFunction(f, name = f"SWF from {vm.name}")


def vm_from_swf(swf):
    """
    Given a social welfare function, returns a voting method that selects all the candidates ranked first according to the swf.

    Args:
        swf (function): A social welfare function.

    Returns:
        function: A voting method that uses the swf to find the winning set.
    """
    
    def f(edata, curr_cands = None):
        return sorted(swf(edata, curr_cands = curr_cands).first())
        
    return VotingMethod(f, name = f"VM from {swf.name}")


def create_election(ranking_list, 
                    rcounts = None,
                    using_extended_strict_preference=None, 
                    candidates=None):
    """Creates an election from a list of rankings.
    
    Args:
        ranking_list (list): A list of rankings, which may be a list of tuples of candidates, a list of dictionaries or a list of Ranking objects.
        using_extended_strict_preference (bool, optional): Whether to use extended strict preference after creating a ProfileWithTies. Defaults to None.
        candidates (list, optional): A list of candidates.  Only used for creating a ProfileWithTies. Defaults to None (by default the candidates are all the candidates that are ranked by at least on voter).
    
    Returns:
        Profile or ProfileWithTies: The election profile.
    """

    if len(ranking_list) > 0 and (type(ranking_list[0]) == tuple or type(ranking_list[0]) == list):
        return Profile(ranking_list, rcounts=rcounts)
    elif len(ranking_list) > 0 and (type(ranking_list[0]) == dict or type(ranking_list[0]) == Ranking):
        if candidates is not None:
            prof = ProfileWithTies(ranking_list, candidates=candidates, rcounts=rcounts)
        else:
            prof = ProfileWithTies(ranking_list, rcounts=rcounts)       
        if using_extended_strict_preference:
            prof.use_extended_strict_preference()
        return prof
    else: # ranking_list is empty
        print("Warning: list of rankings is empty.")
        return Profile(ranking_list)
    

class SPO(object):
    """A strict partial order class due to Jobst Heitzig.
    
    The strict partial order P as a binary relation is encoded as a 2d numpy array.  The predecessors and successors of each object are precomputed.  The add method adds a new pair to the relation and computes the transitive closure.

    Args:
        n (int): The number of objects.
    
    """

    n = None
    """The number of objects"""
    objects = None
    """The list of objects"""
    P = None
    """The strict partial ordering P as a binary relation encoded as a 2d numpy array"""
    preds = None
    """The list of predecessors of each object"""
    succs = None
    """The list of successors of each object"""

    def __init__(self, n):
        self.n = n
        self.objects = list(range(n))
        self.P = np.zeros((n, n), dtype=bool)
        self.preds = [[] for _ in range(n)]
        self.succs = [[] for _ in range(n)]

    def add(self, a, b):
        """add a P b and all transitive consequences"""
        if not self.P[a][b]:
            self.P[a][b] = True
            self.preds[b].append(a)
            self.succs[a].append(b)
            for c in self.preds[a]:
                self._register(c, b)
                for d in self.succs[b]:
                    self._register(c, d)
            for d in self.succs[b]:
                self._register(a, d)

    def initial_elements(self):
        """return the initial elements of P (those without predecessors))"""
        return [i for i in self.objects if len(self.preds[i]) == 0]    

    def _register(self, a, b):
        """register that a P b, without forming the transitive closure"""
        if not self.P[a][b]:
            self.P[a][b] = True
            self.preds[b].append(a)
            self.succs[a].append(b)

    def to_numpy(self):
        """Return the partial order matrix P as a numpy array."""
        return self.P
    
    def to_networkx(self, cmap=None):
        """Convert the SPO to a networkx DiGraph.
        
        Args:
            cmap (dict): A dictionary mapping each number to a candidate name. If None, the identity map is used.
        
        Returns:
            nx.DiGraph: The resulting directed graph with nodes labeled according to cmap if provided.
        """
        G = nx.DiGraph()

        # Determine node labels based on cmap
        if cmap is not None:
            node_labels = {i: cmap[i] for i in self.objects}
        else:
            node_labels = {i: i for i in self.objects}

        # Add nodes with labels
        G.add_nodes_from(node_labels.values())

        # Add edges based on the partial order matrix
        for a in range(self.n):
            for b in range(self.n):
                if self.P[a][b]:
                    G.add_edge(node_labels[a], node_labels[b])

        return G
    
    def to_list(self, cmap=None):
        """If the SPO is a linear order, return a list representing the order.
        
        The list will contain candidate names based on the cmap if provided; otherwise, it will
        contain the numbers. 
        
        Returns None if the SPO is not a linear order.

        Args:
            cmap (dict): A dictionary mapping each number to a candidate name. If None, the identity map is used.
        """
        # Check if the SPO is a strict linear order
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if not (self.P[i][j] or self.P[j][i]):
                    return None  # i and j are not comparable

        # Create the linear order list by topologically sorting the nodes
        linear_order = []
        visited = [False] * self.n

        def visit(node):
            if not visited[node]:
                visited[node] = True
                for successor in self.succs[node]:
                    visit(successor)
                linear_order.append(node)

        for node in self.objects:
            visit(node)

        # Reverse to get the correct order
        linear_order = linear_order[::-1]

        # If cmap is provided, map the numbers to candidate names
        if cmap is not None:
            linear_order = [cmap[node] for node in linear_order]
        
        return linear_order

def weak_orders(A):
    """A generator for all weak orders on A"""
    if not A:
        yield {}
        return
    for k in range(1, len(A) + 1):
        for B in combinations(A, k):
            for order in weak_orders(set(A) - set(B)):
                new_order = {cand: rank + 1 for cand, rank in order.items()}
                yield {**new_order, **{cand: 0 for cand in B}}


def weak_compositions(n, k):
    """A generator for all weak compositions of n into k parts"""

    if k == 1:
        yield [n]
    else:
        for i in range(n + 1):
            for comp in weak_compositions(n - i, k - 1):
                yield [i] + comp

def compositions(n):
    """Generates all compositions of the integer n. Adapted from https://stackoverflow.com/questions/10244180/python-generating-integer-partitions."""

    a = [0 for i in range(n + 1)]
    k = 1
    a[0] = 0
    a[1] = n
    while k != 0:
        x = a[k - 1] + 1
        y = a[k] - 1
        k -= 1
        while 1 <= y:
            a[k] = x
            x = 1
            y -= x
            k += 1
        a[k] = x + y
        yield a[:k + 1]

def enumerate_compositions(int_list):
    """Given a list of integers, enumerate all the compositions of the integers."""

    first_int = int_list[0]

    if len(int_list) == 1:
        for composition in compositions(first_int):
            yield [composition]

    else:
        for composition in compositions(first_int):
            for comps in enumerate_compositions(int_list[1:]):
                yield [composition] + comps

def sublists(lst, length, x = None, partial_sublist = None): 
    """Generate all sublists of lst of a specified length."""
    
    x = length if x is None else x
    
    partial_sublist = list() if partial_sublist is None else partial_sublist
    
    if len(partial_sublist) == length: 
        yield partial_sublist
        
    for i,el in enumerate(lst):
        
        if i < x: 
            
            extended_partial_sublist = partial_sublist + [el]
            x += 1
            yield from sublists(lst[i+1::], length, x, extended_partial_sublist)

def convex_lexicographic_sublists(l):
    """Given a list l, return all convex sublists S such that S is already sorted lexicographically."""

    cl_sublists = []
    current_list = []

    for idx, p in enumerate(l):
        if current_list + [p] == sorted(current_list + [p]):
            current_list = current_list + [p]

            if idx == len(l)-1:
                cl_sublists.append(current_list)

        else:
            cl_sublists.append(current_list)
            current_list = [p]
            
            if idx == len(l) - 1:
                cl_sublists.append(current_list)

    return cl_sublists