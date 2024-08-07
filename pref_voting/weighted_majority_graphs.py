"""
    File: weighted_majority_graphs.py
    Author: Wes Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)
    Date: January 5, 2022
    Updated: July 12, 2022
    Updated: December 19, 2022
    
    Majority Graphs, Margin Graphs and Support Graphs
"""

import networkx as nx
from tabulate import tabulate
import matplotlib.pyplot as plt
import string
from itertools import combinations, permutations
from ortools.linear_solver import pywraplp

import numpy as np

class MajorityGraph(object):
    """A majority graph is an asymmetric directed graph.  The nodes are the candidates and an edge from candidate :math:`c` to :math:`d` means that :math:`c` is majority preferred to :math:`d`.

    :param candidates: List of the candidates.  To be used as nodes in the majority graph.
    :type candidates: list[int] or  list[str]
    :param edges: List of the pairs of candidates describing the edges in the majority graph.   If :math:`(c,d)` is in the list of edges, then there is an edge from :math:`c` to :math:`d`.
    :type edges: list
    :param cmap: Dictionary mapping candidates to candidate names (strings).  If not provided, each candidate name is mapped to itself.
    :type cmap: dict[int: str], optional

    :Example:

    The following code creates a majority graph in which 0 is majority preferred to 1, 1 is majority preferred to 2, and 2 is majority preferred to 0:

    .. code-block:: python

            mg = MajorityGraph([0, 1, 2], [(0,1), (1,2), (2,0)])

    .. warning:: Currently, there is no check that the edge relation is asymmetric.  It is assumed that the user provides an appropriate set of edges.
    """

    def __init__(self, candidates, edges, cmap=None):
        """constructer method"""

        mg = nx.DiGraph()
        mg.add_nodes_from(candidates)
        mg.add_edges_from(edges)
        self.mg = mg
        """A networkx DiGraph object representing the majority graph."""

        self.cmap = cmap if cmap is not None else {c: str(c) for c in candidates}
        
        self.candidates = list(candidates)
        """The list of candidates."""
        
        self.num_cands = len(self.candidates)
        """The number of candidates."""

        self.cindices = list(range(self.num_cands))
        self._cand_to_cindex = {c: i for i, c in enumerate(self.candidates)}
        self.cand_to_cindex = lambda c: self._cand_to_cindex[c]
        self._cindex_to_cand = {i: c for i, c in enumerate(self.candidates)}
        self.cindex_to_cand = lambda i: self._cindex_to_cand[i]
        """A dictionary mapping each candidate to its index in the list of candidates and vice versa."""

        self.maj_matrix = [[False for c2 in self.cindices] for c1 in self.cindices]
        """A matrix of Boolean values representing the majority graph."""

        for c1_idx in self.cindices:
            for c2_idx in self.cindices:
                if mg.has_edge(self.cindex_to_cand(c1_idx), self.cindex_to_cand(c2_idx)):
                    self.maj_matrix[c1_idx][c2_idx] = True
                    self.maj_matrix[c2_idx][c1_idx] = False
                elif mg.has_edge(self.cindex_to_cand(c2_idx), self.cindex_to_cand(c1_idx)):
                    self.maj_matrix[c2_idx][c1_idx] = True
                    self.maj_matrix[c1_idx][c2_idx] = False

    def margin(self, c1, c2):
        raise Exception("margin is not implemented for majority graphs.")

    def support(self, c1, c2):
        raise Exception("support is not implemented for majority graphs.")

    def ratio(self, c1, c2):
        raise Exception("ratio is not implemented for majority graphs.")

    @property
    def edges(self):
        """Returns a list of the edges in the majority graph."""

        return list(self.mg.edges)

    @property
    def is_tournament(self):
        """Returns True if the majority graph is a **tournament** (there is an edge between any two distinct nodes)."""

        return all([
            self.mg.has_edge(c1, c2) or self.mg.has_edge(c2, c1)
            for c1 in self.candidates
            for c2 in self.candidates
            if c1 != c2
        ])

    def majority_prefers(self, c1, c2):
        """Returns true if there is an edge from `c1` to `c2`."""
        return self.mg.has_edge(c1, c2)

    def is_tied(self, c1, c2):
        """Returns true if there is no edge from `c1` to `c2` or from `c2` to `c1`."""
        return not self.mg.has_edge(c1, c2) and not self.mg.has_edge(c2, c1)

    def copeland_scores(self, curr_cands=None, scores=(1, 0, -1)):
        """The Copeland scores in the profile restricted to the candidates in ``curr_cands``.

        The **Copeland score** for candidate :math:`c` is calculated as follows:  :math:`c` receives ``scores[0]`` points for every candidate that  :math:`c` is majority preferred to, ``scores[1]`` points for every candidate that is tied with :math:`c`, and ``scores[2]`` points for every candidate that is majority preferred to :math:`c`. The default ``scores`` is ``(1, 0, -1)``.

        :param curr_cands: restrict attention to candidates in this list. Defaults to all candidates in the profile if not provided.
        :type curr_cands: list[int], optional
        :param scores: the scores used to calculate the Copeland score of a candidate :math:`c`: ``scores[0]`` is for the candidates that :math:`c` is majority preferred to; ``scores[1]`` is for the candidates tied with :math:`c`; and ``scores[2]`` is for the candidates majority preferred to :math:`c`.  The default value is ``scores = (1, 0, -1)``
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

    def dominators(self, cand, curr_cands=None):
        """Returns the list of candidates that are majority preferred to ``cand`` in the majority graph restricted to ``curr_cands``."""
        candidates = self.candidates if curr_cands is None else curr_cands

        return [c for c in candidates if self.majority_prefers(c, cand)]

    def dominates(self, cand, curr_cands=None):
        """Returns the list of candidates that ``cand`` is majority preferred to in the majority graph restricted to ``curr_cands``."""
        candidates = self.candidates if curr_cands is None else curr_cands

        return [c for c in candidates if self.majority_prefers(cand, c)]

    def condorcet_winner(self, curr_cands=None):
        """Returns the Condorcet winner in the profile restricted to ``curr_cands`` if one exists, otherwise returns None.

        The **Condorcet winner** is the candidate that is majority preferred to every other candidate.
        """

        curr_cands = curr_cands if curr_cands is not None else self.candidates

        cw = None
        for c1 in curr_cands:
            if all([self.majority_prefers(c1, c2) for c2 in curr_cands if c1 != c2]):
                cw = c1
                break  # if a Condorcet winner exists, then it is unique
        return cw

    def weak_condorcet_winner(self, curr_cands=None):
        """Returns a list of the weak Condorcet winners in the profile restricted to ``curr_cands`` (which may be empty).

        A candidate :math:`c` is a  **weak Condorcet winner** if there is no other candidate that is majority preferred to :math:`c`.

        .. note:: While the Condorcet winner is unique if it exists, there may be multiple weak Condorcet winners.
        """

        curr_cands = curr_cands if curr_cands is not None else self.candidates

        weak_cw = list()
        for c1 in curr_cands:
            if not any(
                [self.majority_prefers(c2, c1) for c2 in curr_cands if c1 != c2]
            ):
                weak_cw.append(c1)
        return sorted(weak_cw) if len(weak_cw) > 0 else None

    def condorcet_loser(self, curr_cands=None):
        """Returns the Condorcet loser in the profile restricted to ``curr_cands`` if one exists, otherwise returns None.

        A candidate :math:`c` is a  **Condorcet loser** if every other candidate  is majority preferred to :math:`c`.
        """

        curr_cands = curr_cands if curr_cands is not None else self.candidates

        cl = None
        for c1 in curr_cands:
            if all([self.majority_prefers(c2, c1) for c2 in curr_cands if c1 != c2]):
                cl = c1
                break  # if a Condorcet loser exists, then it is unique
        return cl

    def cycles(self, curr_cands = None):
        """Returns True if the margin graph has a cycle.

        This uses the networkx method ``networkx.find_cycle`` to find the cycles in ``self.mg``.

        :Example:

        .. exec_code::

            from pref_voting.weighted_majority_graphs import MajorityGraph
            mg = MajorityGraph([0,1,2], [(0,1), (1,2), (0,2)])
            print(f"The cycles in the majority graph are {mg.cycles()}")
            mg = MajorityGraph([0,1,2], [(0,1), (1,2), (2,0)])
            print(f"The cycles in the majority graph are {mg.cycles()}")
            mg = MajorityGraph([0,1,2,3], [(0,1), (3,0), (1,2), (3,1), (2,0), (3,2)])
            print(f"The cycles in the majority graph are {mg.cycles()}")

        """

        if curr_cands is None:
            return list(nx.simple_cycles(self.mg))
        else:
            mg = nx.DiGraph()
            subgraph = mg.subgraph(curr_cands).copy()
            return list(nx.simple_cycles(subgraph))


    def has_cycle(self, curr_cands = None):
        """Returns True if there is a cycle in the majority graph."""
        if curr_cands is None:
            try:
                cycle = nx.find_cycle(self.mg)
            except:
                cycle = list()
        else:
            mg = nx.DiGraph()
            subgraph = mg.subgraph(curr_cands).copy()
            try:
                cycle = nx.find_cycle(subgraph)
            except:
                cycle = list()

        return len(cycle) != 0

    def remove_candidates(self, cands_to_ignore):
        """Remove all candidates from ``cands_to_ignore`` from the Majority Graph.

        :param cands_to_ignore: list of candidates to remove from the profile
        :type cands_to_ignore: list[int]
        :returns: a majority graph with candidates from ``cands_to_ignore`` removed and a dictionary mapping the candidates from the new profile to the original candidate names.

        :Example:

        .. exec_code::

            from pref_voting.weighted_majority_graphs import MajorityGraph
            mg = MajorityGraph([0, 1, 2], [(0, 1), (1, 2), (2, 0)])
            print(f"Candidates: {mg.candidates}")
            print(f"Edges: {mg.edges}")
            mg_new = mg.remove_candidates([1])
            print(f"Candidates: {mg_new.candidates}")
            print(f"Edges: {mg_new.edges}")
        """

        new_cands = [c for c in self.candidates if c not in cands_to_ignore]

        new_edges = [e for e in self.edges if e[0] in new_cands and e[1] in new_cands]

        new_cmap = {c: cname for c, cname in self.cmap.items() if c in new_cands}

        return MajorityGraph(new_cands, new_edges, cmap=new_cmap)

    def to_networkx(self): 
        """
        Return a networkx weighted DiGraph representing the margin graph. 
        """

        return self.mg

    def description(self): 
        """
        Returns a string describing the Majority Graph.
        """
        return f"MajorityGraph({self.candidates}, {self.edges}, cmap={self.cmap})"

    def display(self, cmap=None, curr_cands=None):
        """Display a majority graph (restricted to ``curr_cands``) using networkx.draw.

        :param cmap: the candidate map to use (overrides the cmap associated with this majority graph)
        :type cmap: dict[int,str], optional
        :param curr_cands: list of candidates
        :type curr_cands: list[int], optional
        :rtype: None

        :Example:

        .. code::

            from pref_voting.weighted_majority_graphs import MajorityGraph
            mg = MajorityGraph([0,1,2], [(0,1), (1,2), (2,0)])
            mg.display()

        .. image:: ./maj_graph_ex1.png
            :width: 400
            :alt: Alternative text

        """

        cmap = cmap if cmap is not None else self.cmap
        curr_cands = self.candidates if curr_cands is None else curr_cands

        mg = nx.DiGraph()
        mg.add_nodes_from([cmap[c] for c in curr_cands])
        mg.add_edges_from(
            [
                (cmap[c1], cmap[c2])
                for c1, c2 in self.mg.edges
                if c1 in curr_cands and c2 in curr_cands
            ]
        )

        pos = nx.circular_layout(mg)

        nx.draw(
            mg,
            pos,
            font_size=20,
            font_color="white",
            node_size=700,
            width=1.5,
            with_labels=True,
        )
        plt.show()

    def display_cycles(self, cmap=None):
        """
        Display the cycles in the margin graph.

        Args:
            cmap (dict, optional): The cmap used to map candidates to candidate names
        """

        cycles = self.cycles()
        
        print(f"There {'are' if len(cycles) != 1 else 'is'} {len(cycles)} {'cycle' if len(cycles) == 1 else 'cycles'}{':' if len(cycles) > 0 else '.'} \n")
        for cycle in cycles: 
            cmap = cmap if cmap is not None else self.cmap
            cmap_inverse = {cname: c for c, cname in cmap.items()}
            mg_with_cycle = nx.DiGraph()

            mg_with_cycle.add_nodes_from([cmap[c] for c in self.candidates])
            mg_with_cycle.add_edges_from([(cmap[e[0]], cmap[e[1]]) for e in self.edges])

            cands = self.candidates
            mg_edges = list(self.edges)

            cycle_edges = [(cmap[c], cmap[cycle[cidx + 1]]) for cidx,c in enumerate(cycle[0:-1])] + [(cmap[cycle[-1]], cmap[cycle[0]])]

            cands_in_cycle = [cmap[c1] for c1 in cands if c1 in cycle]

            node_colors = ["blue" if n in cands_in_cycle else "lightgray" for n in mg_with_cycle.nodes ]

            pos = nx.circular_layout(mg_with_cycle)
            nx.draw(mg_with_cycle, pos, width=1.5, edge_color="white")
            nx.draw_networkx_nodes(
                mg_with_cycle, pos, node_color=node_colors, node_size=700
            )
            nx.draw_networkx_labels(mg_with_cycle, pos, font_size=20, font_color="white")
            nx.draw_networkx_edges(
                mg_with_cycle,
                pos,
                edgelist=cycle_edges,
                width=10,
                alpha=1.0,
                edge_color="b",
                arrowsize=25,
                min_target_margin=15,
                node_size=700,
            )

            nx.draw_networkx_edges(
                mg_with_cycle,
                pos,
                edgelist=[(cmap[e[0]], cmap[e[1]]) for e in mg_edges if (cmap[e[0]], cmap[e[1]]) not in cycle_edges],
                width=1.5,
                 edge_color="lightgray",
                arrowsize=15,
                min_target_margin=15,
            )
            
            ax = plt.gca()
            ax.set_frame_on(False)
            plt.show()
            
    def to_latex(self, cmap=None, new_cand=None):
        """Outputs TikZ code for displaying the majority graph.

        :param cmap: the candidate map to use (overrides the cmap associated with this majority graph)
        :type cmap: dict[int,str], optional
        :param new_cand: the candidate that is displayed on the far right,  *only used for displaying 5 candidates*.
        :type new_cand: int
        :rtype: str

        .. warning:: This works best for 3, 4 or 5 candidates.   It will produce the code for more than 5 outputs, but the positioning of the nodes may need to be modified.

        :Example:

        .. exec_code::

            from pref_voting.weighted_majority_graphs import MajorityGraph
            mg = MajorityGraph([0,1,2], [(0,1), (1,2), (2,0)])
            print(mg.to_latex())
            print(mg.to_latex(cmap = {0:"a", 1:"b", 2:"c"}))

        """
        if len(self.candidates) == 3:
            return three_cand_tikz_str(self, cmap=cmap)
        elif len(self.candidates) == 4:
            return four_cand_tikz_str(self, cmap=cmap)
        elif len(self.candidates) == 5:
            return five_cand_tikz_str(self, cmap=cmap, new_cand=new_cand)
        else:
            pos = nx.circular_layout(self.mg)
            return to_tikz_str(self, pos, cmap=cmap)

    @classmethod
    def from_profile(cls, profile, cmap=None):
        """Generates a majority graph from a :class:`Profile`.

        :param profile: the profile
        :type profile: Profile
        :param cmap: the candidate map to use (overrides the cmap associated with this majority graph)
        :type cmap: dict[int,str], optional
        :rtype: str

        :Example:

        .. exec_code::

            from pref_voting.profiles import Profile
            from pref_voting.weighted_majority_graphs import MajorityGraph
            prof = Profile([[0,1,2], [1,2,0], [2,0,1]])
            mg = MajorityGraph.from_profile(prof)
            print(mg.edges)

            # it is better to use the Profile method
            mg = prof.majority_graph()
            print(mg.edges)

        """
        cmap = profile.cmap if cmap is None else cmap
        return cls(
            profile.candidates,
            [
                (c1, c2)
                for c1 in profile.candidates
                for c2 in profile.candidates
                if profile.majority_prefers(c1, c2)
            ],
            cmap=cmap,
        )

    def __add__(self, other_mg):
        """
        Add to majority graphs together.  The result is a majority graph with the union of the candidates and edges of the two majority graphs.  If there is an edge from :math:`c` to :math:`d` in both majority graphs, then the edge is included in the resulting majority graph.  If there is an edge from :math:`c` to :math:`d` in neither majority graph, then there is no edge from :math:`c` to :math:`d` in the resulting majority graph.
        """
        new_cands = list(set(self.candidates + other_mg.candidates))

        _new_edges = list(set(self.edges + other_mg.edges))

        new_edges = list()
        for c1, c2 in _new_edges:
            if (self.majority_prefers(c1, c2) and other_mg.majority_prefers(c2, c1)) or (self.majority_prefers(c2, c1) and other_mg.majority_prefers(c1, c2)):
                continue
            else: 
                new_edges.append((c1, c2))
        return MajorityGraph(new_cands, new_edges)
    
    def __eq__(self, other_mg): 
        """
        Check if two majority graphs are equal. 
        """
        return self.candidates == other_mg.candidates and sorted(self.edges) == sorted(other_mg.edges)

class MarginGraph(MajorityGraph):
    """A margin graph is a weighted asymmetric directed graph.  The nodes are the candidates and an edge from candidate :math:`c` to :math:`d` with weight :math:`w` means that :math:`c` is majority preferred to :math:`d` by a **margin** of :math:`w`.

    :param candidates: List of the candidates.  To be used as nodes in the majority graph.
    :type candidates: list[int] or  list[str]
    :param w_edges: List of the pairs of candidates describing the edges in the majority graph.   If :math:`(c,d,w)` is in the list of edges, then there is an edge from :math:`c` to :math:`d` with weight :math:`w`.
    :type w_edges: list
    :param cmap: Dictionary mapping candidates to candidate names (strings).  If not provided, each candidate name is mapped to itself.
    :type cmap: dict[int: str], optional

    :Example:

    The following code creates a margin graph in which 0 is majority preferred to 1 by a margin of 1, 1 is majority preferred to 2 by a margin of 3, and 2 is majority preferred to 0 by a margin of 5:

    .. code-block:: python

        mg = MarginGraph([0, 1, 2], [(0,1,1), (1,2,3), (2,0,5)])

    .. warning:: Currently, there is no check that the edge relation is asymmetric or that weights of edges are positive.  It is assumed that the user provides an appropriate set of edges with weights.
    """

    def __init__(self, candidates, w_edges, cmap=None):
        """constructor method"""

        super().__init__(candidates, [(e[0], e[1]) for e in w_edges], cmap=cmap)

        self.margin_matrix = [[0 for c2 in self.cindices] for c1 in self.cindices]
        """The margin matrix, where the :math:`(i, j)`-entry is the number of voters who rank candidate with index :math:`i` above the candidate with index :math:`j` minus the number of voters  who rank candidate with index :math:`j` above the candidate with index :math:`i`. """

        for c1, c2, margin in w_edges:
            self.margin_matrix[self.cand_to_cindex(c1)][self.cand_to_cindex(c2)] = margin
            self.margin_matrix[self.cand_to_cindex(c2)][self.cand_to_cindex(c1)] = -1 * margin

    def margin(self, c1, c2):
        """Returns the margin of ``c1`` over ``c2``."""
        return self.margin_matrix[self.cand_to_cindex(c1)][self.cand_to_cindex(c2)]

    def strength_matrix(self, curr_cands = None, strength_function = None): 
        """
        Return the strength matrix of the profile.  The strength matrix is a matrix where the entry in row :math:`i` and column :math:`j` is the number of voters that rank the candidate with index :math:`i` over the candidate with index :math:`j`.  If ``curr_cands`` is provided, then the strength matrix is restricted to the candidates in ``curr_cands``.  If ``strength_function`` is provided, then the strength matrix is computed using the strength function."""
        
        if curr_cands is not None: 
            cindices = [cidx for cidx, _ in enumerate(curr_cands)]
            cindex_to_cand = lambda cidx: curr_cands[cidx]
            cand_to_cindex = lambda c: cindices[curr_cands.index(c)]
            strength_function = self.margin if strength_function is None else strength_function
            strength_matrix = np.array([[strength_function(cindex_to_cand(a_idx), cindex_to_cand(b_idx)) for b_idx in cindices] for a_idx in cindices])
        else:  
            cindices = self.cindices
            cindex_to_cand = self.cindex_to_cand
            cand_to_cindex = self.cand_to_cindex
            strength_matrix = np.array(self.margin_matrix) if strength_function is None else np.array([[strength_function(cindex_to_cand(a_idx), cindex_to_cand(b_idx)) for b_idx in cindices] for a_idx in cindices])

        return strength_matrix, cand_to_cindex

    @property
    def edges(self):
        """Returns a list of the weighted edges in the margin graph."""

        return [(c1, c2, self.margin(c1, c2)) for c1, c2 in self.mg.edges]

    def remove_candidates(self, cands_to_ignore):
        """Remove all candidates from ``cands_to_ignore`` from the Majority Graph.

        :param cands_to_ignore: list of candidates to remove from the profile
        :type cands_to_ignore: list[int]
        :returns: a majority graph with candidates from ``cands_to_ignore`` removed and a dictionary mapping the candidates from the new profile to the original candidate names.

        :Example:

        .. exec_code::

            from pref_voting.weighted_majority_graphs import MarginGraph
            mg = MarginGraph([0, 1, 2], [(0, 1, 11), (1, 2, 13), (2, 0, 5)])
            print(f"Candidates: {mg.candidates}")
            print(f"Edges: {mg.edges}")
            mg_new = mg.remove_candidates([1])
            print(f"Candidates: {mg_new.candidates}")
            print(f"Edges: {mg_new.edges}")
        """

        new_cands = [c for c in self.candidates if c not in cands_to_ignore]

        new_edges = [e for e in self.edges if e[0] in new_cands and e[1] in new_cands]

        new_cmap = {c: cname for c, cname in self.cmap.items() if c in new_cands}

        return MarginGraph(new_cands, new_edges, cmap=new_cmap)

    def majority_prefers(self, c1, c2):
        """Returns True if the margin of ``c1`` over ``c2`` is positive."""
        return self.margin_matrix[self.cand_to_cindex(c1)][self.cand_to_cindex(c2)] > 0

    def is_tied(self, c1, c2):
        """Returns True if the margin ``c1`` over ``c2`` is zero."""
        return self.margin_matrix[self.cand_to_cindex(c1)][self.cand_to_cindex(c2)] == 0

    def is_uniquely_weighted(self):
        """Returns True if all the margins between distinct candidates are unique and there is no 0 margin between distinct candidates."""
        has_zero_margins = any(
            [
                self.margin(c1, c2) == 0
                for c1 in self.candidates
                for c2 in self.candidates
                if c1 != c2
            ]
        )
        return not has_zero_margins and len(
            list(set([self.margin(e[0], e[1]) for e in self.mg.edges]))
        ) == len(self.mg.edges)


    def to_networkx(self): 
        """
        Return a networkx weighted DiGraph representing the margin graph. 
        """

        g = nx.DiGraph()
        g.add_nodes_from(self.candidates)
        g.add_weighted_edges_from(self.edges)

        return g

    def debord_profile(self): 
        """
        Find a profile that generates the margin graph using the algorithm from Debord's (1987) proof.
        """
        
        from pref_voting.profiles import Profile
    
        candidates = self.candidates

        ranks = list()
        rcounts = list()

        if all([w % 2 == 0 for _,_,w in self.edges]):
            for c1, c2, w in self.edges:
                other_cands = [c for c in candidates if c != c1 and c != c2]

                lin_ord1 = sorted(other_cands)
                lin_ord2 = sorted(other_cands, reverse=True)

                ranks.append([c1, c2] + lin_ord1)
                rcounts.append(w//2)
                ranks.append(lin_ord2 + [c1, c2])
                rcounts.append(w//2)
            return Profile(ranks, rcounts, cmap=self.cmap)
        
        elif all([w % 2 == 1 for _,_,w in self.edges]): # all weights are odd
            single_prof = Profile([sorted(candidates)], [1])

            for c1, c2, w in self.edges:
                other_cands = [c for c in candidates if c != c1 and c != c2]
                lin_ord1 = sorted(other_cands)
                lin_ord2 = sorted(other_cands, reverse=True)
                if w-single_prof.margin(c1, c2) > 0:
                    ranks.append([c1, c2] + lin_ord1)
                    rcounts.append((w-single_prof.margin(c1, c2))//2)
                    ranks.append(lin_ord2 + [c1, c2])
                    rcounts.append((w-single_prof.margin(c1, c2))//2)
            ranks.append(sorted(candidates))
            rcounts.append(1)
            return Profile(ranks, rcounts, cmap=self.cmap)
        else: 
            print("Cannot find a Profile since the weights do not all have the same parity.")
            return None
        
    def minimal_profile(self): 
        """
        Use an integer linear program to find a minimal profile generating the margin graph. 
        """
        from pref_voting.profiles import Profile

        solver = pywraplp.Solver.CreateSolver("SAT")

        num_cands = len(self.candidates)
        rankings = list(permutations(range(num_cands)))
        
        ranking_to_var = dict()
        infinity = solver.infinity()
        for ridx, r in enumerate(rankings): 
            _v = solver.IntVar(0.0, infinity, f"x{ridx}")
            ranking_to_var[r] = _v

        nv = solver.IntVar(0.0, infinity, "nv")
        equations = list()
        for c1 in self.candidates: 
            for c2 in self.candidates: 
                if c1 != c2: 
                    margin = self.margin(c1, c2)
                    if margin >= 0: 
                        rankings_c1_over_c2 = [ranking_to_var[r] for r in rankings if r.index(c1) < r.index(c2)]
                        rankings_c2_over_c1 = [ranking_to_var[r] for r in rankings if r.index(c2) < r.index(c1)]
                        equations.append(sum(rankings_c1_over_c2) == margin + sum(rankings_c2_over_c1))
                        
        equations.append(nv == sum(list(ranking_to_var.values())))

        for eq in equations: 
            solver.Add(eq)

        solver.Minimize(nv)

        status = solver.Solve()

        if status == pywraplp.Solver.INFEASIBLE:
            print("Error: Did not find a solution.")
            return None
        
        if status != pywraplp.Solver.OPTIMAL: 
            print("Warning: Did not find an optimal solution.")

        _ranks = list()
        _rcounts = list()

        for r,v in ranking_to_var.items(): 

            if v.solution_value() > 0: 
                _ranks.append(r)
                _rcounts.append(int(v.solution_value()))
                if not v.solution_value().is_integer(): 
                    print("ERROR: Found non integer, ", v.solution_value())
                    return None
        return Profile(_ranks, rcounts = _rcounts)

    def normalize_ordered_weights(self):
        """
        Returns a MarginGraph with the same order of the edges, except that the weights are 2, 4, 6,...

        .. important::

            This function returns a margin graph that has the same ordering of the edges, but the edges may have different weights.  Qualitative margin graph invariant 
            voting methods will identify the same winning sets for both graphs. 

        """

        sorted_edges = sorted(self.edges, key=lambda e: e[2])
        sorted_margins = sorted(
            [
                self.margin(c1, c2)
                for c1 in self.candidates
                for c2 in self.candidates
                if self.margin(c1, c2) > 0
            ]
        )
        curr_margin = sorted_margins[0]
        new_margin = 2
        new_edges = list()
        for e in sorted_edges:
            if e[2] > curr_margin:
                curr_margin = e[2]
                new_margin += 2
                new_edges.append((e[0], e[1], new_margin))

            else:
                new_edges.append((e[0], e[1], new_margin))

        return MarginGraph(self.candidates, new_edges, cmap=self.cmap)

    def description(self): 
        """
        Returns a string describing the Margin Graph.
        """
        return f"MarginGraph({self.candidates}, {self.edges}, cmap={self.cmap})"

    def display(self, curr_cands=None, cmap=None):
        """Display a margin graph (restricted to ``curr_cands``) using networkx.draw.

        :param cmap: the candidate map to use (overrides the cmap associated with this majority graph)
        :type cmap: dict[int,str], optional
        :param curr_cands: list of candidates
        :type curr_cands: list[int], optional
        :rtype: None

        :Example:

        .. code::

            from pref_voting.weighted_majority_graphs import MarginGraph
            mg = MarginGraph([0,1,2], [(0,1,3), (1,2,1), (2,0,5)])
            mg.display()

        .. image:: ./margin_graph_ex1.png
            :width: 400
            :alt: Alternative text

        """

        cmap = cmap if cmap is not None else self.cmap
        curr_cands = self.candidates if curr_cands is None else curr_cands

        mg = nx.DiGraph()
        mg.add_nodes_from([cmap[c] for c in curr_cands])
        mg.add_weighted_edges_from(
            [
                (cmap[c1], cmap[c2], self.margin(c1, c2))
                for c1, c2 in self.mg.edges
                if c1 in curr_cands and c2 in curr_cands
            ]
        )

        pos = nx.circular_layout(mg)

        nx.draw(
            mg,
            pos,
            font_size=20,
            font_color="white",
            node_size=700,
            width=1.5,
            with_labels=True,
        )
        labels = nx.get_edge_attributes(mg, "weight")
        nx.draw_networkx_edge_labels(
            mg, pos, edge_labels=labels, font_size=14, label_pos=0.3
        )

        plt.show()

    def display_with_defeat(
        self, defeat, curr_cands=None, show_undefeated=True, cmap=None
    ):
        """
        Display the margin graph with any edges that are  ``defeat`` edges highlighted in blue.

        Args:
            defeat (networkx.DiGraph): The defeat relation represented as a networkx object.
            curr_cands (List[int], optional): If set, then use the defeat relation for the profile restricted to the candidates in ``curr_cands``
            show_undefeated (bool, optional): If true, color the undefeated candidates blue and the other candidates red.
            cmap (dict, optional): The cmap used to map candidates to candidate names


        """

        cmap = cmap if cmap is not None else self.cmap
        cmap_inverse = {cname: c for c, cname in cmap.items()}
        mg_with_defeat = nx.DiGraph()

        mg = nx.DiGraph()
        mg.add_nodes_from([cmap[c] for c in self.mg.nodes])
        mg.add_edges_from([(cmap[e[0]], cmap[e[1]]) for e in self.mg.edges])

        cands = self.candidates if curr_cands is None else curr_cands

        mg_edges = list(mg.edges())
        defeat_edges = list(defeat.edges())
        edges = mg_edges + [
            (cmap[e[0]], cmap[e[1]])
            for e in defeat_edges
            if (cmap[e[0]], cmap[e[1]]) not in mg_edges
        ]

        mg_with_defeat.add_nodes_from([cmap[c] for c in cands])
        mg_with_defeat.add_edges_from(edges)

        if show_undefeated:

            undefeated_cands = [
                cmap[c1]
                for c1 in cands
                if not any([defeat.has_edge(c2, c1) for c2 in cands])
            ]
            node_colors = [
                "blue" if n in undefeated_cands else "red" for n in mg_with_defeat.nodes
            ]

        else:
            node_colors = ["#1f78b4" for n in mg_with_defeat.nodes]

        pos = nx.circular_layout(mg_with_defeat)
        nx.draw(mg, pos, width=1.5, edge_color="white")
        nx.draw_networkx_nodes(
            mg_with_defeat, pos, node_color=node_colors, node_size=700
        )
        nx.draw_networkx_labels(mg_with_defeat, pos, font_size=20, font_color="white")
        nx.draw_networkx_edges(
            mg_with_defeat,
            pos,
            edgelist=[
                (cmap[e[0]], cmap[e[1]])
                for e in defeat_edges
                if (cmap[e[0]], cmap[e[1]]) in mg_edges
            ],
            width=10,
            alpha=0.5,
            edge_color="b",
            arrowsize=25,
            min_target_margin=15,
            node_size=700,
        )
        collection = nx.draw_networkx_edges(
            mg_with_defeat,
            pos,
            edgelist=[
                (cmap[e[0]], cmap[e[1]])
                for e in defeat_edges
                if (cmap[e[0]], cmap[e[1]]) not in mg_edges
            ],
            width=10,
            alpha=0.5,
            edge_color="b",
            arrowsize=25,
            min_target_margin=15,
        )
        if collection is not None:
            for patch in collection:
                patch.set_linestyle("dashed")

        nx.draw_networkx_edges(
            mg_with_defeat,
            pos,
            edgelist=mg_edges,
            width=1.5,
            arrowsize=15,
            min_target_margin=15,
        )
        labels = {
            e: self.margin(cmap_inverse[e[0]], cmap_inverse[e[1]]) for e in mg_edges
        }

        nx.draw_networkx_edge_labels(
            mg_with_defeat, pos, edge_labels=labels, font_size=14, label_pos=0.3
        )
        ax = plt.gca()
        ax.set_frame_on(False)
        plt.show()

    def display_cycles(self, cmap=None):
        """
        Display the cycles in the margin graph.

        Args:
            cmap (dict, optional): The cmap used to map candidates to candidate names.

        """
        
        cycles = self.cycles()
        
        print(f"There {'are' if len(cycles) != 1 else 'is'} {len(cycles)} {'cycle' if len(cycles) == 1 else 'cycles'}{':' if len(cycles) > 0 else '.'} \n")
        for cycle in cycles: 
            cmap = cmap if cmap is not None else self.cmap
            mg_with_cycle = nx.DiGraph()

            mg_with_cycle.add_nodes_from([cmap[c] for c in self.candidates])
            mg_with_cycle.add_edges_from([(cmap[e[0]], cmap[e[1]]) for e in self.edges])

            cands = self.candidates
            mg_edges = list(self.edges)

            cycle_edges = [(cmap[c], cmap[cycle[cidx + 1]]) for cidx,c in enumerate(cycle[0:-1])] + [(cmap[cycle[-1]], cmap[cycle[0]])]

            cands_in_cycle = [cmap[c1] for c1 in cands if c1 in cycle]

            node_colors = ["blue" if n in cands_in_cycle else "lightgray" for n in mg_with_cycle.nodes ]

            pos = nx.circular_layout(mg_with_cycle)
            nx.draw(mg_with_cycle, pos, width=1.5, edge_color="white")
            nx.draw_networkx_nodes(
                mg_with_cycle, pos, node_color=node_colors, node_size=700
            )
            nx.draw_networkx_labels(mg_with_cycle, pos, font_size=20, font_color="white")
            nx.draw_networkx_edges(
                mg_with_cycle,
                pos,
                edgelist=cycle_edges,
                width=10,
                alpha=1.0,
                edge_color="b",
                arrowsize=25,
                min_target_margin=15,
                node_size=700,
            )

            nx.draw_networkx_edges(
                mg_with_cycle,
                pos,
                edgelist=[(cmap[e[0]], cmap[e[1]]) for e in mg_edges if (cmap[e[0]], cmap[e[1]]) not in cycle_edges],
                width=1.5,
                 edge_color="lightgray",
                arrowsize=15,
                min_target_margin=15,
            )
            labels = {
                (cmap[e[0]], cmap[e[1]]): self.margin(e[0], e[1]) for e in mg_edges
            }

            nx.draw_networkx_edge_labels(
                mg_with_cycle, pos, edge_labels=labels, font_size=14, label_pos=0.3
            )
            ax = plt.gca()
            ax.set_frame_on(False)
            plt.show()

    @classmethod
    def from_profile(cls, profile, cmap=None):
        """Generates a majority graph from a :class:`Profile`.

        :param profile: the profile
        :type profile: Profile
        :param cmap: the candidate map to use (overrides the cmap associated with this majority graph)
        :type cmap: dict[int,str], optional
        :rtype: str

        :Example:

        .. exec_code::

            from pref_voting.profiles import Profile
            from pref_voting.weighted_majority_graphs import MarginGraph
            prof = Profile([[0,1,2], [1,2,0], [2,0,1]], [2, 1, 2])
            mg = MarginGraph.from_profile(prof)
            print(mg.edges)
            print(mg.margin_matrix)

            # it is better to use the Profile method
            mg = prof.margin_graph()
            print(mg.edges)
            print(mg.margin_matrix)

        """

        cmap = profile.cmap if cmap is None else cmap
        return cls(
            profile.candidates,
            [
                (c1, c2, profile.margin(c1, c2))
                for c1 in profile.candidates
                for c2 in profile.candidates
                if profile.majority_prefers(c1, c2)
            ],
            cmap=cmap,
        )

    def __add__(self, edata): 
        """
        Return a MarginGraph in which the new margin of candidate :math:`a` over :math:`b` is the sum of the 
        existing margin of :math:`a` over :math:`b` with with the margin :math:`a` over :math:`b` in ``edata``. 
        """
        candidates = self.candidates
        new_edges = list()
        for c1, c2 in combinations(candidates, 2): 
            
            new_margin = self.margin(c1, c2) + edata.margin(c1, c2)
            
            if new_margin > 0: 
                new_edges.append((c1, c2, new_margin))
            elif new_margin < 0: 
                
                new_edges.append((c2, c1, -1 * new_margin))
            
        return MarginGraph(candidates, new_edges, cmap=self.cmap)

    def __eq__(self, other_mg): 
        """
        Return True if the margin graphs are equal (the candidates, and all edges and weights are the same); otherwise, return False
        """

        return self.candidates == other_mg.candidates and sorted(self.edges) == sorted(other_mg.edges)
    
class SupportGraph(MajorityGraph):
    """A support graph is a weighted asymmetric directed graph.  The nodes are the candidates and an edge from candidate :math:`c` to :math:`d` with weight :math:`w` means that the **support** of  :math:`c` over :math:`d` is :math:`w`.

    :param candidates: List of the candidates.  To be used as nodes in the majority graph.
    :type candidates: list[int] or  list[str]
    :param w_edges: List representing the edges in the majority graph with supports. If :math:`(c,d,(n,m))` is in the list of edges, then there is an edge from :math:`c` to :math:`d`, the support for :math:`c` over :math:`d` is :math:`n`, and the support for :math:`d` over :math:`c` is :math:`m`. 
    :type w_edges: list
    :param cmap: Dictionary mapping candidates to candidate names (strings).  If not provided, each candidate name is mapped to itself.
    :type cmap: dict[int: str], optional

    :Example:

    The following code creates a support graph in which:

    - 0 is majority preferred to 1, the number of voters who rank 0 over 1 is 4, and the number of voters who rank 1 over 0 is 3;
    - 1 is majority preferred to 2, the number of voters who rank 1 over 2 is 5, and the number of voters who rank 2 over 1 is 2; and
    - 2 is majority preferred to 0, the number of voters who rank 2 over 0 is 6, and the number of voters who rank 0 over 2 is 1.

    .. code-block:: python

        sg = SupportGraph([0, 1, 2], [(0, 1, (4, 3)), (1, 2, (5, 2)), (2, 0, (6, 1))])

    .. warning:: Currently, there is no check to that the edge relation is asymmetric.  It is assumed that the user provides an appropriate set of edges with weights.
    """

    def __init__(self, candidates, w_edges, cmap=None):
        """constructor method"""

        super().__init__(
            candidates,
            [
                (e[0], e[1]) if e[2][0] > e[2][1] else (e[1], e[0])
                for e in w_edges
                if e[2][0] != e[2][1]
            ],
            cmap=cmap,
        )

        self.s_matrix = [[0 for c2 in self.cindices] for c1 in self.cindices]
        """The support matrix, where the   :math:`(i, j)`-entry is the number of voters who rank candidate with index :math:`i` above the candidate with index :math:`j`. """

        for c1, c2, support in w_edges:
            self.s_matrix[self.cand_to_cindex(c1)][self.cand_to_cindex(c2)] = support[0]
            self.s_matrix[self.cand_to_cindex(c2)][self.cand_to_cindex(c1)] = support[1]

    @property
    def edges(self):
        """Returns a list of the weighted edges in the margin graph."""

        return [
            (c1, c2, (self.support(c1, c2), self.support(c2, c1)))
            for c1, c2 in self.mg.edges
        ]

    def margin(self, c1, c2):
        """Returns the margin of ``c1`` over ``c2``."""

        return (
            self.s_matrix[self.cand_to_cindex(c1)][self.cand_to_cindex(c2)]
            - self.s_matrix[self.cand_to_cindex(c2)][self.cand_to_cindex(c1)]
        )

    def support(self, c1, c2):
        """Returns the support of ``c1`` over ``c2``."""

        return self.s_matrix[self.cand_to_cindex(c1)][self.cand_to_cindex(c2)]

    def majority_prefers(self, c1, c2):
        """Returns True if ``c1`` is majority preferred to ``c2``."""

        return (
            self.s_matrix[self.cand_to_cindex(c1)][self.cand_to_cindex(c2)]
            > self.s_matrix[self.cand_to_cindex(c2)][self.cand_to_cindex(c1)]
        )

    def is_tied(self, c1, c2):
        """Returns True if ``c1`` is tied with  ``c2``."""

        return (
            self.s_matrix[self.cand_to_cindex(c1)][self.cand_to_cindex(c2)]
            == self.s_matrix[self.cand_to_cindex(c2)][self.cand_to_cindex(c1)]
        )

    def strength_matrix(self, curr_cands = None, strength_function = None): 
        """
        Return the strength matrix of the profile.  The strength matrix is a matrix where the entry in row :math:`i` and column :math:`j` is the number of voters that rank the candidate with index :math:`i` over the candidate with index :math:`j`.  If ``curr_cands`` is provided, then the strength matrix is restricted to the candidates in ``curr_cands``.  If ``strength_function`` is provided, then the strength matrix is computed using the strength function."""
        
        if curr_cands is not None: 
            cindices = [cidx for cidx, _ in enumerate(curr_cands)]
            cindex_to_cand = lambda cidx: curr_cands[cidx]
            cand_to_cindex = lambda c: cindices[curr_cands.index(c)]
            strength_function = self.support if strength_function is None else strength_function
            strength_matrix = np.array([[strength_function(cindex_to_cand(a_idx), cindex_to_cand(b_idx)) for b_idx in cindices] for a_idx in cindices])
        else:  
            cindices = self.cindices
            cindex_to_cand = self.cindex_to_cand
            cand_to_cindex = self.cand_to_cindex
            strength_matrix = np.array(self.s_matrix) if strength_function is None else np.array([[strength_function(cindex_to_cand(a_idx), cindex_to_cand(b_idx)) for b_idx in cindices] for a_idx in cindices])

        return strength_matrix, cand_to_cindex

    def remove_candidates(self, cands_to_ignore):
        """Remove all candidates from ``cands_to_ignore`` from the Majority Graph.

        :param cands_to_ignore: list of candidates to remove from the profile
        :type cands_to_ignore: list[int]
        :returns: a majority graph with candidates from ``cands_to_ignore`` removed and a dictionary mapping the candidates from the new profile to the original candidate names.

        :Example:

        .. exec_code::

            from pref_voting.weighted_majority_graphs import SupportGraph
            sg = SupportGraph([0, 1, 2], [(0, 1, (11, 1)), (1, 2, (5, 13)), (2, 0, (5, 10))])
            print(f"Candidates: {sg.candidates}")
            print(f"Edges: {sg.edges}")
            sg_new = sg.remove_candidates([1])
            print(f"Candidates: {sg_new.candidates}")
            print(f"Edges: {sg_new.edges}")
        """

        new_cands = [c for c in self.candidates if c not in cands_to_ignore]

        new_edges = [e for e in self.edges if e[0] in new_cands and e[1] in new_cands]

        new_cmap = {c: cname for c, cname in self.cmap.items() if c in new_cands}

        return SupportGraph(new_cands, new_edges, cmap=new_cmap)


    def display(self, curr_cands=None, cmap=None):
        """Display a support graph (restricted to ``curr_cands``) using networkx.draw.

        :param cmap: the candidate map to use (overrides the cmap associated with this majority graph)
        :type cmap: dict[int,str], optional
        :param curr_cands: list of candidates
        :type curr_cands: list[int], optional
        :rtype: None

        """

        cmap = cmap if cmap is not None else self.cmap
        curr_cands = self.candidates if curr_cands is None else curr_cands

        mg = nx.DiGraph()
        mg.add_nodes_from([cmap[c] for c in curr_cands])
        mg.add_weighted_edges_from(
            [
                (cmap[c1], cmap[c2], self.support(c1, c2))
                for c1, c2 in self.mg.edges
                if c1 in curr_cands and c2 in curr_cands
            ]
        )

        pos = nx.circular_layout(mg)

        nx.draw(
            mg,
            pos,
            font_size=20,
            font_color="white",
            node_size=700,
            width=1.5,
            with_labels=True,
        )
        labels = nx.get_edge_attributes(mg, "weight")
        nx.draw_networkx_edge_labels(
            mg, pos, edge_labels=labels, font_size=14, label_pos=0.3
        )

        plt.show()

    @classmethod
    def from_profile(cls, profile, cmap=None):
        """Generates a support graph from a :class:`Profile`.

        :param profile: the profile
        :type profile: Profile
        :param cmap: the candidate map to use (overrides the cmap associated with this majority graph)
        :type cmap: dict[int,str], optional
        :rtype: str

        :Example:

        .. exec_code::

            from pref_voting.profiles import Profile
            from pref_voting.weighted_majority_graphs import SupportGraph
            prof = Profile([[0,1,2], [1,2,0], [2,0,1]], [2, 1, 2])
            sg = SupportGraph.from_profile(prof)
            print(sg.edges)
            print(sg.s_matrix)

            # it is better to use the Profile method
            sg = prof.support_graph()
            print(sg.edges)
            print(sg.s_matrix)

        """

        cmap = profile.cmap if cmap is None else cmap
        return cls(
            profile.candidates,
            [
                (c1, c2, (profile.support(c1, c2), profile.support(c2, c1)))
                for c1 in profile.candidates
                for c2 in profile.candidates
            ],
            cmap=cmap,
        )


###
# functions to display graphs in tikz
##
def three_cand_tikz_str(g, cmap=None):
    """Returns the TikZ code to display the graph ``g`` based on 3 candidates (may be a MajorityGraph, MarginGraph or a SupportGraph)."""

    a = g.candidates[0]
    b = g.candidates[1]
    c = g.candidates[2]

    if type(g) == MarginGraph:
        w = lambda c, d: f"node[fill=white] {{${g.margin(c,d)}$}}"
    elif type(g) == SupportGraph:
        w = lambda c, d: f"node[fill=white] {{${g.support(c,d)}$}}"
    else:
        w = lambda c, d: ""

    cmap = g.cmap if cmap is None else cmap

    nodes = f"""
\\begin{{tikzpicture}}
\\node[circle,draw,minimum width=0.25in] at (0,0) (a) {{${cmap[a]}$}}; 
\\node[circle,draw,minimum width=0.25in] at (3,0) (c) {{${cmap[c]}$}}; 
\\node[circle,draw,minimum width=0.25in] at (1.5,1.5) (b) {{${cmap[b]}$}};\n"""

    if g.majority_prefers(a, b):
        ab_edge = f"\\path[->,draw,thick] (a) to {w(a,b)} (b);\n"
    elif g.majority_prefers(b, a):
        ab_edge = f"\\path[->,draw,thick] (b) to {w(b,a)} (a);\n"
    else:
        ab_edge = ""

    if g.majority_prefers(b, c):
        bc_edge = f"\\path[->,draw,thick] (b) to {w(b,c)} (c);\n"
    elif g.majority_prefers(c, b):
        bc_edge = f"\\path[->,draw,thick] (c) to {w(c,b)} (b);\n"
    else:
        bc_edge = ""

    if g.majority_prefers(a, c):
        ac_edge = f"\\path[->,draw,thick] (a) to {w(a,c)} (c);\n"
    elif g.majority_prefers(c, a):
        ac_edge = f"\\path[->,draw,thick] (c) to {w(c,a)} (a);\n"
    else:
        ac_edge = ""

    return nodes + ab_edge + bc_edge + ac_edge + "\\end{tikzpicture}"


def four_cand_tikz_str(g, cmap=None):
    """Returns the TikZ code to display the graph ``g`` based on 4 candidates (may be a MajorityGraph, MarginGraph or a SupportGraph)."""

    a = g.candidates[0]
    b = g.candidates[1]
    c = g.candidates[2]
    d = g.candidates[3]

    if type(g) == MarginGraph:
        w = lambda c, d: f"node[fill=white] {{${g.margin(c,d)}$}}"
    elif type(g) == SupportGraph:
        w = lambda c, d: f"node[fill=white] {{${g.support(c,d)}$}}"
    else:
        w = lambda c, d: ""

    cmap = g.cmap if cmap is None else cmap

    nodes = f"""
\\begin{{tikzpicture}}
\\node[circle,draw,minimum width=0.25in] at (0,0)      (a) {{${cmap[a]}$}}; 
\\node[circle,draw,minimum width=0.25in] at (3,0)      (b) {{${cmap[b]}$}}; 
\\node[circle,draw,minimum width=0.25in] at (1.5,1.5)  (c) {{${cmap[c]}$}}; 
\\node[circle,draw,minimum width=0.25in] at (1.5,-1.5) (d) {{${cmap[d]}$}};\n"""

    if g.majority_prefers(a, b):
        ab_edge = f"\\path[->,draw,thick] (a) to[pos=.7] {w(a,b)} (b);\n"
    elif g.majority_prefers(b, a):
        ab_edge = f"\\path[->,draw,thick] (b) to[pos=.7] {w(b,a)} (a);\n"
    else:
        ab_edge = ""

    if g.majority_prefers(a, c):
        ac_edge = f"\\path[->,draw,thick] (a) to {w(a,c)} (c);\n"
    elif g.majority_prefers(c, a):
        ac_edge = f"\\path[->,draw,thick] (c) to {w(c,a)} (a);\n"
    else:
        ac_edge = ""

    if g.majority_prefers(a, d):
        ad_edge = f"\\path[->,draw,thick] (a) to {w(a,d)} (d);\n"
    elif g.majority_prefers(d, a):
        ad_edge = f"\\path[->,draw,thick] (d) to {w(d,a)} (a);\n"
    else:
        ad_edge = ""

    if g.majority_prefers(b, c):
        bc_edge = f"\\path[->,draw,thick] (b) to {w(b,c)} (c);\n"
    elif g.majority_prefers(c, b):
        bc_edge = f"\\path[->,draw,thick] (c) to {w(c,b)} (b);\n"
    else:
        bc_edge = ""

    if g.majority_prefers(b, d):
        bd_edge = f"\\path[->,draw,thick] (b) to {w(b,d)} (d);\n"
    elif g.majority_prefers(d, b):
        bd_edge = f"\\path[->,draw,thick] (d) to {w(d,b)} (b);\n"
    else:
        bd_edge = ""

    if g.majority_prefers(c, d):
        cd_edge = f"\\path[->,draw,thick] (c) to[pos=.7]  {w(c,d)} (d);\n"
    elif g.majority_prefers(d, c):
        cd_edge = f"\\path[->,draw,thick] (d) to[pos=.7]  {w(d,c)} (c);\n"
    else:
        cd_edge = ""

    return (
        nodes
        + ab_edge
        + ac_edge
        + ad_edge
        + bc_edge
        + bd_edge
        + cd_edge
        + "\\end{tikzpicture}"
    )


def five_cand_tikz_str(g, cmap=None, new_cand=None):

    candidates = list(g.candidates)

    if new_cand is not None:

        e = new_cand

        cands_minus = [_c for _c in candidates if _c != new_cand]

        a = cands_minus[0]
        b = cands_minus[1]
        c = cands_minus[2]
        d = cands_minus[3]
    else:
        a = candidates[0]
        b = candidates[1]
        c = candidates[2]
        d = candidates[3]
        e = candidates[4]

    # new_cand = candidates[4]

    node_id = {a: "a", b: "b", c: "c", d: "d", e: "e"}
    print(node_id)
    if type(g) == MarginGraph:
        w = lambda c, d: f"node[fill=white] {{${g.margin(c,d)}$}}"
    elif type(g) == SupportGraph:
        w = lambda c, d: f"node[fill=white] {{${g.support(c,d)}$}}"
    else:
        w = lambda c, d: ""

    cmap = g.cmap if cmap is None else cmap

    nodes = f"""
\\begin{{tikzpicture}}
\\node[circle,draw,minimum width=0.25in] at (2,1.5)  (a) {{${cmap[a]}$}}; 
\\node[circle,draw,minimum width=0.25in] at (0,1.5)  (b) {{${cmap[b]}$}}; 
\\node[circle,draw,minimum width=0.25in] at (0,-1.5) (c) {{${cmap[c]}$}}; 
\\node[circle,draw,minimum width=0.25in] at (2,-1.5) (d) {{${cmap[d]}$}}; 
\\node[circle,draw,minimum width=0.25in] at (3.5,0)  (e) {{${cmap[e]}$}};\n"""
    edges = [(a, b), (a, d), (a, e), (b, c), (c, d), (d, e)]
    edges_with_pos = [(a, c), (b, d), (b, e), (c, e)]

    edge_tikz_str = list()
    for c1, c2 in edges:
        if g.majority_prefers(c1, c2):
            edge_tikz_str.append(
                f"\\path[->,draw,thick] ({node_id[c1]}) to {w(c1,c2)} ({node_id[c2]});\n"
            )
        elif g.majority_prefers(c2, c1):
            edge_tikz_str.append(
                f"\\path[->,draw,thick] ({node_id[c2]}) to {w(c2,c1)} ({node_id[c1]});\n"
            )
        else:
            edge_tikz_str.append("")
    for c1, c2 in edges_with_pos:
        if g.majority_prefers(c1, c2):
            edge_tikz_str.append(
                f"\\path[->,draw,thick] ({node_id[c1]}) to[pos=.7] {w(c1,c2)} ({node_id[c2]});\n"
            )
        elif g.majority_prefers(c2, c1):
            edge_tikz_str.append(
                f"\\path[->,draw,thick] ({node_id[c2]}) to[pos=.7] {w(c2,c1)} ({node_id[c1]});\n"
            )
        else:
            edge_tikz_str.append("")

    return nodes + "".join(edge_tikz_str) + "\\end{tikzpicture}"


def to_tikz_str(g, pos, cmap=None):

    node_id = {c: string.ascii_lowercase[cidx] for cidx, c in enumerate(g.candidates)}

    if type(g) == MarginGraph:
        w = lambda c, d: f"node[fill=white] {{${g.margin(c,d)}$}}"
    elif type(g) == SupportGraph:
        w = lambda c, d: f"node[fill=white] {{${g.support(c,d)}$}}"
    else:
        w = lambda c, d: ""

    cmap = g.cmap if cmap is None else cmap

    node_tikz_str = list()

    for c in g.candidates:
        node_tikz_str.append(
            f"\\node[circle,draw,minimum width=0.25in] at ({float(2.4*list(pos[c])[0])},{float(2.6*list(pos[c])[1])})  ({node_id[c]}) {{${cmap[c]}$}};\n"
        )

    edges_tikz_str = list()
    for c1 in g.candidates:
        for c2 in g.candidates:
            if g.majority_prefers(c1, c2):
                edges_tikz_str.append(
                    f"\\path[->,draw,thick] ({node_id[c1]}) to[pos=.7] {w(c1,c2)} ({node_id[c2]});\n"
                )
            elif g.majority_prefers(c2, c1):
                edges_tikz_str.append(
                    f"\\path[->,draw,thick] ({node_id[c2]}) to[pos=.7] {w(c2,c1)} ({node_id[c1]});\n"
                )
            else:
                edges_tikz_str.append("")

    return (
        "\\begin{tikzpicture}\n"
        + "".join(node_tikz_str)
        + "".join(edges_tikz_str)
        + "\\end{tikzpicture}"
    )


def maximal_elements(g):
    """return the nodes in g with no incoming arrows."""
    return [n for n in g.nodes if g.in_degree(n) == 0]


def display_mg_with_sc(edata, curr_cands=None, cmap=None):
    """
    Display the margin graph with the Split Cycle defeat relation highlighted.
    """
    from pref_voting.margin_based_methods import split_cycle_defeat

    if type(edata) == MarginGraph:
        edata.display_with_defeat(
            split_cycle_defeat(edata, curr_cands=curr_cands, cmap=cmap)
        )
    else:
        edata.display_margin_graph_with_defeat(
            split_cycle_defeat(edata, curr_cands=curr_cands, cmap=cmap)
        )


def display_graph(g, curr_cands=None, cmap=None):
    """Helper function to display a weighted directed graph."""

    cmap = cmap if cmap is not None else {n: str(n) for n in g.nodes}

    candidates = g.nodes if curr_cands is None else curr_cands

    displayed_g = nx.DiGraph()
    displayed_g.add_nodes_from([cmap[c] for c in candidates])
    displayed_g.add_weighted_edges_from(
        [(cmap[e[0]], cmap[e[1]], e[2]["weight"]) for e in g.edges(data=True)]
    )

    pos = nx.circular_layout(displayed_g)
    nx.draw(
        displayed_g,
        pos,
        font_size=20,
        node_color="blue",
        font_color="white",
        node_size=700,
        with_labels=True,
    )
    labels = nx.get_edge_attributes(displayed_g, "weight")
    nx.draw_networkx_edge_labels(
        displayed_g, pos, edge_labels=labels, font_size=14, label_pos=0.3
    )
    plt.show()
