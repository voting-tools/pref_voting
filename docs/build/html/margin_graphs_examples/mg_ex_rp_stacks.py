
from pref_voting.weighted_majority_graphs import MarginGraph

mg = MarginGraph([0, 1, 2], [(0, 1, 2), (1, 2, 4), (2, 0, 2)])

mg.display()