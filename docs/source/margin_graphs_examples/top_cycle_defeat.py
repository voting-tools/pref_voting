
from pref_voting.profiles import Profile
from pref_voting.c1_methods import top_cycle_defeat

prof = Profile(
    [
        [0, 1, 2, 3], 
        [1, 2, 0, 3], 
        [2, 0, 1, 3]
    ], 
    [1, 1, 1])

tc_defeat = top_cycle_defeat(prof)

prof.display_margin_graph_with_defeat(tc_defeat, show_undefeated=True)