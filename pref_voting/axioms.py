from pref_voting.dominance_axioms import *
from pref_voting.monotonicity_axioms import *
from pref_voting.invariance_axioms import *
from pref_voting.strategic_axioms import *
from pref_voting.variable_voter_axioms import *
from pref_voting.variable_candidate_axioms import *
from pref_voting.axiom import Axiom


axioms_dict = {name: obj for name, obj in globals().items() if isinstance(obj, Axiom)}
