"""
    File: strategic_axioms.py
    Author: Wesley H. Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)
    Date: April 18, 2024
    
    Strategic axioms 
"""

from pref_voting.axiom import Axiom
from pref_voting.axiom_helpers import *
from itertools import permutations
from pref_voting.helper import weak_orders

def has_strategy_proofness_violation(prof, vm, set_preference = "single-winner", verbose=False):
    """
    Returns True if there is a voter who can benefit by misrepresenting their preferences.

    If set_preference = "single-winner", a voter benefits only if they can change the unique winner in the original profile to a unique winner in the new profile such that in their original ranking, the new winner is above the old winner.

    If set_preference = "weak-dominance", a voter benefits only if in their original ranking, all new winners are weakly above all old winners and some new winner is strictly above some old winner.

    If set_preference = "optimist", a voter benefits only if in their original ranking, their favorite new winner is above their favorite old winner.

    If set_preference = "pessimist", a voter benefits only if in their original ranking, their least favorite new winner is above their least favorite old winner.

    Args:
        prof: a Profile or ProfileWithTies object.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation.

    Returns:
        Result of the test (bool): Returns True if there is a violation and False otherwise.

    .. note::

        The different set preference notions are drawn from Definition 2.1.1 (p. 42) of The Mathematics of Manipulation by Alan D. Taylor. 
    """

    winners = vm(prof)

    if isinstance(prof,ProfileWithTies):
        prof.use_extended_strict_preference()

    found_manipulator = False

    ranking_tokens = prof.rankings
    ranking_types = prof.ranking_types

    ws = vm(prof)

    if set_preference == "single-winner":
        if len(ws) > 1:
            return False


    for r in ranking_types:
        if not found_manipulator:

            ranking_tokens_minus_r = [r for r in ranking_tokens]
            ranking_tokens_minus_r.remove(r)

            if isinstance(prof,Profile):

                for new_r in permutations(prof.candidates):
                    if new_r != r and not found_manipulator:

                        new_ranking_tokens = ranking_tokens_minus_r + [new_r]
                        new_prof = Profile(new_ranking_tokens)
                        new_ws = vm(new_prof)

                        old_winner_to_compare = None
                        new_winner_to_compare = None

                        if set_preference == "single-winner" and len(new_ws) == 1:

                            old_winner_to_compare = ws[0]
                            new_winner_to_compare = new_ws[0] 
                        
                        elif set_preference == "weak-dominance":
                            r_as_ranking = Ranking({c: i for i, c in enumerate(r)})
                        
                        elif set_preference == "optimist":
                                
                            old_winner_to_compare = [cand for cand in r if cand in ws][0]
                            new_winner_to_compare = [cand for cand in r if cand in new_ws][0]

                        elif set_preference == "pessimist":
                            
                            old_winner_to_compare = [cand for cand in r if cand in ws][-1] 
                            new_winner_to_compare = [cand for cand in r if cand in new_ws][-1]

                        if old_winner_to_compare is not None and r.index(old_winner_to_compare) > r.index(new_winner_to_compare) or (set_preference == "weak-dominance" and r_as_ranking.weak_dom(new_ws,ws)):
                            
                            found_manipulator = True

                            if verbose:
                                print(f"Violation of Strategy-Proofness for {vm.name} under the {set_preference} set preference.")
                                print(f"A voter can benefit by changing their ranking from {r} to {new_r}.")
                                print("")
                                print("Original Profile:")
                                prof.display()
                                print(prof.description())
                                print("")
                                vm.display(prof)
                                prof.display_margin_graph()
                                print("")
                                print("New Profile:")
                                new_prof.display()
                                print(new_prof.description())
                                print("")
                                vm.display(new_prof)
                                new_prof.display_margin_graph()

            if isinstance(prof,ProfileWithTies):
                r_dict = r.rmap

                for _new_r in weak_orders(prof.candidates):
                    new_r = Ranking(_new_r)
                    if new_r != r and not found_manipulator:

                        new_ranking_tokens = ranking_tokens_minus_r + [new_r]
                        new_prof = ProfileWithTies(new_ranking_tokens, candidates = prof.candidates)
                        new_prof.use_extended_strict_preference()
                        new_ws = vm(new_prof)

                        ranked_old_winners = [c for c in ws if c in r_dict.keys()]
                        ranked_new_winners = [c for c in new_ws if c in r_dict.keys()]

                        rank_of_old_winner_to_compare = None
                        rank_of_new_winner_to_compare = None

                        if set_preference == "single-winner" and len(new_ws) == 1:

                            rank_of_old_winner_to_compare = r_dict[ws[0]] if ranked_old_winners else math.inf
                            rank_of_new_winner_to_compare = r_dict[new_ws[0]] if ranked_new_winners else math.inf
                        
                        elif set_preference == "optimist":

                            rank_of_old_winner_to_compare = min([r_dict[c] for c in ranked_old_winners]) if ranked_old_winners else math.inf
                            rank_of_new_winner_to_compare = min([r_dict[c] for c in ranked_new_winners]) if ranked_new_winners else math.inf

                        elif set_preference == "pessimist":

                            rank_of_old_winner_to_compare = max([r_dict[c] for c in ranked_old_winners]) if ranked_old_winners == ws else math.inf
                            rank_of_new_winner_to_compare = max([r_dict[c] for c in ranked_new_winners]) if ranked_new_winners == new_ws else math.inf

                        if rank_of_old_winner_to_compare is not None and rank_of_old_winner_to_compare > rank_of_new_winner_to_compare or (set_preference == "weak-dominance" and r.weak_dom(new_ws,ws,use_extended_preferences=True)):
                            
                            found_manipulator = True

                            if verbose:
                                print(f"Violation of Strategy-Proofness for {vm.name} under the {set_preference} set preference.")
                                print(f"A voter can benefit by changing their ranking from {r} to {new_r}.")
                                print("")
                                print("Original Profile:")
                                prof.display()
                                print(prof.description())
                                print("")
                                vm.display(prof)
                                prof.display_margin_graph()
                                print("")
                                print("New Profile:")
                                new_prof.display()
                                print(new_prof.description())
                                print("")
                                vm.display(new_prof)
                                new_prof.display_margin_graph()
        
    return found_manipulator

def find_all_strategy_proofness_violations(prof, vm, set_preference = "single-winner", verbose=False):
    """
    Returns a list of tuples (old_ranking, new_ranking) where old_ranking is the original ranking and new_ranking is the ranking that the voter can change to in order to benefit.

    If set_preference = "single-winner", a voter benefits only if they can change the unique winner in the original profile to a unique winner in the new profile such that in their original ranking, the new winner is above the old winner.

    If set_preference = "weak-dominance", a voter benefits only if in their original ranking, all new winners are weakly above all old winners and some new winner is strictly above some old winner.

    If set_preference = "optimist", a voter benefits only if in their original ranking, their favorite new winner is above their favorite old winner.

    If set_preference = "pessimist", a voter benefits only if in their original ranking, their least favorite new winner is above their least favorite old winner.

    Args:
        prof: a Profile or ProfileWithTies object.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation.

    Returns:
        A List of tuples (old_ranking, new_ranking) where old_ranking is the original ranking and new_ranking is the ranking that the voter can change to in order to benefit.
    """

    winners = vm(prof)

    if isinstance(prof,ProfileWithTies):
        prof.use_extended_strict_preference()

    violations = list()

    ranking_tokens = prof.rankings
    ranking_types = prof.ranking_types

    ws = vm(prof)

    if set_preference == "single-winner":
        if len(ws) > 1:
            return violations

    for r in ranking_types:
        ranking_tokens_minus_r = [r for r in ranking_tokens]
        ranking_tokens_minus_r.remove(r)

        if isinstance(prof,Profile):

            for new_r in permutations(prof.candidates):
                if new_r != r:

                    new_ranking_tokens = ranking_tokens_minus_r + [new_r]
                    new_prof = Profile(new_ranking_tokens)
                    new_ws = vm(new_prof)

                    old_winner_to_compare = None
                    new_winner_to_compare = None

                    if set_preference == "single-winner" and len(new_ws) == 1:

                        old_winner_to_compare = ws[0]
                        new_winner_to_compare = new_ws[0] 
                    
                    elif set_preference == "weak-dominance":
                        r_as_ranking = Ranking({c: i for i, c in enumerate(r)})
                    
                    elif set_preference == "optimist":
                            
                        old_winner_to_compare = [cand for cand in r if cand in ws][0]
                        new_winner_to_compare = [cand for cand in r if cand in new_ws][0]

                    elif set_preference == "pessimist":
                        
                        old_winner_to_compare = [cand for cand in r if cand in ws][-1] 
                        new_winner_to_compare = [cand for cand in r if cand in new_ws][-1]

                    if old_winner_to_compare is not None and r.index(old_winner_to_compare) > r.index(new_winner_to_compare) or (set_preference == "weak-dominance" and r_as_ranking.weak_dom(new_ws,ws)):
                        
                        violations.append((r,new_r))

                        if verbose:
                            print(f"Violation of Strategy-Proofness for {vm.name} under the {set_preference} set preference.")
                            print(f"A voter can benefit by changing their ranking from {r} to {new_r}.")
                            print("")
                            print("Original Profile:")
                            prof.display()
                            print(prof.description())
                            print("")
                            vm.display(prof)
                            prof.display_margin_graph()
                            print("")
                            print("New Profile:")
                            new_prof.display()
                            print(new_prof.description())
                            print("")
                            vm.display(new_prof)
                            new_prof.display_margin_graph()

        if isinstance(prof,ProfileWithTies):
            r_dict = r.rmap

            for _new_r in weak_orders(prof.candidates):
                new_r = Ranking(_new_r)
                if new_r != r:

                    new_ranking_tokens = ranking_tokens_minus_r + [new_r]
                    new_prof = ProfileWithTies(new_ranking_tokens, candidates = prof.candidates)
                    new_prof.use_extended_strict_preference()
                    new_ws = vm(new_prof)

                    ranked_old_winners = [c for c in ws if c in r_dict.keys()]
                    ranked_new_winners = [c for c in new_ws if c in r_dict.keys()]

                    rank_of_old_winner_to_compare = None
                    rank_of_new_winner_to_compare = None

                    if set_preference == "single-winner" and len(new_ws) == 1:

                        rank_of_old_winner_to_compare = r_dict[ws[0]] if ranked_old_winners else math.inf
                        rank_of_new_winner_to_compare = r_dict[new_ws[0]] if ranked_new_winners else math.inf

                    elif set_preference == "optimist":

                        rank_of_old_winner_to_compare = min([r_dict[c] for c in ranked_old_winners]) if ranked_old_winners else math.inf
                        rank_of_new_winner_to_compare = min([r_dict[c] for c in ranked_new_winners]) if ranked_new_winners else math.inf

                    elif set_preference == "pessimist":

                        rank_of_old_winner_to_compare = max([r_dict[c] for c in ranked_old_winners]) if ranked_old_winners == ws else math.inf
                        rank_of_new_winner_to_compare = max([r_dict[c] for c in ranked_new_winners]) if ranked_new_winners == new_ws else math.inf

                    if rank_of_old_winner_to_compare is not None and rank_of_old_winner_to_compare > rank_of_new_winner_to_compare or (set_preference == "weak-dominance" and r.weak_dom(new_ws,ws,use_extended_preferences=True)):

                        violations.append((r.rmap,new_r.rmap))

                        if verbose:
                            print(f"Violation of Strategy-Proofness for {vm.name} under the {set_preference} set preference.")
                            print(f"A voter can benefit by changing their ranking from {r} to {new_r}.")
                            print("")
                            print("Original Profile:")
                            prof.display()
                            print(prof.description())
                            print("")
                            vm.display(prof)
                            prof.display_margin_graph()
                            print("")
                            print("New Profile:")
                            new_prof.display()
                            print(new_prof.description())
                            print("")
                            vm.display(new_prof)
                            new_prof.display_margin_graph()

    return violations

strategy_proofness = Axiom(
    "Strategy Proofness",
    has_violation = has_strategy_proofness_violation,
    find_all_violations = find_all_strategy_proofness_violations, 
)