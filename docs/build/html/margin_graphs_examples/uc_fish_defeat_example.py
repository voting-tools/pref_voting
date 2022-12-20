from pref_voting.profiles import Profile
from pref_voting.c1_methods import uc_fish_defeat

prof = Profile(
    [
        [0, 3, 1, 2],
        [3, 2, 0, 1],
        [2, 0, 3, 1],
        [3, 2, 1, 0],
        [0, 2, 3, 1],
        [3, 1, 2, 0],
        [2, 3, 0, 1]
    ],
    [1, 1, 1, 1, 3, 2, 1])

uc_fish_def = uc_fish_defeat(prof)

prof.display_margin_graph_with_defeat(uc_fish_def, show_undefeated=True)

