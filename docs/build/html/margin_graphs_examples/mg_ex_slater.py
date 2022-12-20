
from pref_voting.weighted_majority_graphs import MarginGraph

mg = MarginGraph([0, 1, 2, 3], [(0, 2, 2), (0, 3, 6), (1, 0, 8), (2, 3, 4), (2, 1, 10), (3, 1, 12)])

mg.display()