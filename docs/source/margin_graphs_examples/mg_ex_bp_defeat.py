
from pref_voting.weighted_majority_graphs import MarginGraph
from pref_voting.margin_based_methods import beat_path_defeat

mg = MarginGraph([0, 1, 2, 3], [(0, 2, 3), (1, 0, 5), (2, 1, 5), (2, 3, 1), (3, 0, 3), (3, 1, 1)])

mg.display_with_defeat(beat_path_defeat(mg))