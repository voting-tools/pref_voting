from pref_voting.scoring_methods import *
from pref_voting.iterative_methods import *
from pref_voting.c1_methods import *
from pref_voting.margin_based_methods import *
from pref_voting.combined_methods import *
from pref_voting.other_methods import *

# List of all voting methods
voting_methods = scoring_vms + iterated_vms + c1_vms + mg_vms + combined_vms + other_vms 