"""
    File: strategic_axioms.py
    Author: Wesley H. Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)
    Date: March 16, 2025
    
    Strategic axioms 
"""

import numpy as np
import math
from pref_voting.axiom import Axiom
from pref_voting.axiom_helpers import *
from itertools import permutations, combinations
from pref_voting.helper import weak_orders
from pref_voting.profiles import Profile
from pref_voting.profiles_with_ties import ProfileWithTies
from pref_voting.rankings import Ranking

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

def truncate_ballot_from_bottom(ranking, num_to_keep):
    """
    Truncate a ballot by keeping only the top num_to_keep candidates.
    
    Args:
        ranking: A Ranking object or tuple representing a ballot
        num_to_keep: Number of top candidates to keep
        
    Returns:
        A truncated Ranking object
    """
    if isinstance(ranking, tuple):
        # For tuple rankings, keep only the first num_to_keep candidates
        truncated = ranking[:num_to_keep]
        return Ranking({c: i+1 for i, c in enumerate(truncated)})
    else:
        # For Ranking objects, sort candidates by rank and keep only the top num_to_keep
        sorted_candidates = sorted(ranking.rmap.keys(), key=lambda c: ranking.rmap[c])
        truncated_rmap = {c: ranking.rmap[c] for c in sorted_candidates[:num_to_keep]}
        return Ranking(truncated_rmap)

def has_later_no_harm_violation(prof, vm, verbose=False, coalition_size=1, uniform_coalition=True, require_resoluteness=False):
    """
    Returns True if there is a ballot (or collection of ballots) such that by truncating it from the bottom, 
    a candidate who is ranked by the truncated ballot goes from losing to winning.

    Viewed in reverse, this means that adding previously unranked candidates to the bottom of a ballot harmed a higher ranked candidate.
    
    Args:
        prof: a Profile or ProfileWithTies object.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation.
        coalition_size (int, default=1): Size of the coalition of voters who truncate their ballots.
        uniform_coalition (bool, default=True): If True, all voters in the coalition have the same ballot.
        require_resoluteness (bool, default=False): If True, only profiles with a unique winner before and after truncation are considered.
        
    Returns:
        Result of the test (bool): Returns True if there is a violation and False otherwise.
    """
    # Get the current winners
    winners = vm(prof)
    
    # Convert numpy array winners to list if needed
    if isinstance(winners, np.ndarray):
        winners = winners.tolist()
    
    # If require_resoluteness is True, skip profiles with multiple winners before truncation
    if require_resoluteness and len(winners) > 1:
        return False
    
    # For individual voter case or uniform coalition
    if uniform_coalition:
        # Check each ranking type in the profile
        for r in prof.ranking_types:
            # Skip if there aren't enough voters with this ranking for the coalition
            if isinstance(r, tuple):
                if prof.rankings.count(r) < coalition_size:
                    continue
                ranked_candidates = list(r)
                r_as_ranking = Ranking({c: i+1 for i, c in enumerate(r)})
            else:
                if sum(1 for ballot in prof.rankings if isinstance(ballot, Ranking) and ballot.cands == r.cands) < coalition_size:
                    continue
                ranked_candidates = list(r.cands)
                r_as_ranking = r
            
            # Try truncating at different positions (keep only first i candidates)
            for i in range(1, len(ranked_candidates)):
                # Create truncated ballot keeping only the first i candidates
                # This is proper truncation - keeping the top i candidates in their original order
                truncated_r = truncate_ballot_from_bottom(r_as_ranking, i)
                
                # Skip if truncation didn't change anything or if truncated ballot is empty
                # Ensure at least one candidate remains ranked
                if len(truncated_r.cands) == 0 or truncated_r.rmap == r_as_ranking.rmap:
                    continue
                
                # Get the truncated ballot as a ranking map
                truncated_rmap = truncated_r.rmap
                
                # Create a new profile with the truncated ballot(s)
                modified_rankings = []
                
                # Add all ballots except those we'll truncate
                for ballot in prof.rankings:
                    # Skip the ballots we'll truncate
                    if isinstance(ballot, tuple) and ballot == r:
                        continue
                    elif isinstance(ballot, Ranking) and isinstance(r, Ranking) and ballot.cands == r.cands:
                        continue
                    
                    # Add the ballot in the correct format
                    if isinstance(ballot, tuple):
                        modified_rankings.append({c: i+1 for i, c in enumerate(ballot)})
                    else:
                        modified_rankings.append(ballot.rmap)
                
                # Make sure we've skipped the right number of ballots
                if isinstance(r, tuple):
                    original_count = prof.rankings.count(r)
                else:
                    original_count = sum(1 for ballot in prof.rankings if isinstance(ballot, Ranking) and ballot.cands == r.cands)
                
                # Add back any ballots we shouldn't have truncated
                for _ in range(original_count - coalition_size):
                    modified_rankings.append(r_as_ranking.rmap)
                
                # Add the truncated ballots
                for _ in range(coalition_size):
                    modified_rankings.append(truncated_rmap)
                
                # Create the new profile with ties
                new_prof = ProfileWithTies(modified_rankings, candidates=prof.candidates)
                if isinstance(prof, ProfileWithTies) and prof.using_extended_strict_preference:
                    new_prof.use_extended_strict_preference()
                
                # Get the new winners
                new_winners = vm(new_prof)
                
                # Convert numpy array winners to list if needed
                if isinstance(new_winners, np.ndarray):
                    new_winners = new_winners.tolist()
                
                # If require_resoluteness is True, skip profiles with multiple winners after truncation
                if require_resoluteness and len(new_winners) > 1:
                    continue
                
                # Check for Later No Harm violation: a candidate ranked in the truncated ballot
                # goes from losing to winning
                truncated_candidates = truncated_r.cands
                
                # Find candidates that are ranked in the truncated ballot and went from losing to winning
                new_winners_in_truncated = [c for c in new_winners if c in truncated_candidates and c not in winners]
                
                if new_winners_in_truncated:
                    if verbose:
                        print(f"Later No Harm violation: Candidate(s) {new_winners_in_truncated} went from losing to winning due to bottom truncation.")
                        print("")
                        print(f"Original winners: {winners}")
                        print(f"New winners: {new_winners}")
                        print(f"Original ballot: {r}")
                        print(f"Truncated ballot: {truncated_r}")
                        print("")
                        print("Original profile:")
                        prof.display()
                        prof.display_margin_graph()
                        vm.display(prof)
                        print("")
                        print("Modified profile:")
                        new_prof.display()
                        new_prof.display_margin_graph()
                        vm.display(new_prof)
                    return True
    
    # For non-uniform coalition
    if not uniform_coalition and coalition_size > 1:
        # Get all possible combinations of ranking types
        ranking_combinations = list(combinations(prof.ranking_types, coalition_size))
        
        for ranking_combo in ranking_combinations:
            # Check if we have enough of each ranking type
            valid_combo = True
            for r in ranking_combo:
                if isinstance(r, tuple):
                    if prof.rankings.count(r) < ranking_combo.count(r):
                        valid_combo = False
                        break
                else:
                    if sum(1 for ballot in prof.rankings if isinstance(ballot, Ranking) and ballot.cands == r.cands) < ranking_combo.count(r):
                        valid_combo = False
                        break
            
            if not valid_combo:
                continue
            
            # For each ranking in the combination, try truncating it
            for i, r in enumerate(ranking_combo):
                if isinstance(r, tuple):
                    ranked_candidates = list(r)
                    r_as_ranking = Ranking({c: i+1 for i, c in enumerate(r)})
                else:
                    ranked_candidates = list(r.cands)
                    r_as_ranking = r
                
                # Try truncating at different positions
                for j in range(1, len(ranked_candidates)):
                    # Create truncated ballot keeping only the first j candidates
                    # This is proper truncation - keeping the top j candidates in their original order
                    truncated_r = truncate_ballot_from_bottom(r_as_ranking, j)
                    
                    # Skip if truncation didn't change anything or if truncated ballot is empty
                    if len(truncated_r.cands) == 0 or truncated_r.rmap == r_as_ranking.rmap:
                        continue
                    
                    # Get the truncated ballot as a ranking map
                    truncated_rmap = truncated_r.rmap
                    
                    # Create a new profile with the truncated ballot(s)
                    modified_rankings = []
                    
                    # Add all ballots except those in the coalition
                    for ballot in prof.rankings:
                        skip = False
                        for coalition_r in ranking_combo:
                            if (isinstance(ballot, tuple) and ballot == coalition_r) or (isinstance(ballot, Ranking) and isinstance(coalition_r, Ranking) and ballot.cands == coalition_r.cands):
                                skip = True
                                break
                        
                        if skip:
                            continue
                        
                        # Add the ballot in the correct format
                        if isinstance(ballot, tuple):
                            modified_rankings.append({c: i+1 for i, c in enumerate(ballot)})
                        else:
                            modified_rankings.append(ballot.rmap)
                    
                    # Add back the coalition ballots, with the i-th one truncated
                    for k, coalition_r in enumerate(ranking_combo):
                        if k == i:
                            modified_rankings.append(truncated_rmap)
                        else:
                            if isinstance(coalition_r, tuple):
                                modified_rankings.append({c: i+1 for i, c in enumerate(coalition_r)})
                            else:
                                modified_rankings.append(coalition_r.rmap)
                    
                    # Create the new profile with ties
                    new_prof = ProfileWithTies(modified_rankings, candidates=prof.candidates)
                    if isinstance(prof, ProfileWithTies) and prof.using_extended_strict_preference:
                        new_prof.use_extended_strict_preference()
                    
                    # Get the new winners
                    new_winners = vm(new_prof)
                    
                    # Convert numpy array winners to list if needed
                    if isinstance(new_winners, np.ndarray):
                        new_winners = new_winners.tolist()
                    
                    # If require_resoluteness is True, skip profiles with multiple winners after truncation
                    if require_resoluteness and len(new_winners) > 1:
                        continue
                    
                    # Check for Later No Harm violation: a candidate ranked in the truncated ballot
                    # goes from losing to winning
                    truncated_candidates = truncated_r.cands
                    
                    # Find candidates that are ranked in the truncated ballot and went from losing to winning
                    new_winners_in_truncated = [c for c in new_winners if c in truncated_candidates and c not in winners]
                    
                    if new_winners_in_truncated:
                        if verbose:
                            print(f"Later No Harm violation: Candidate(s) {new_winners_in_truncated} went from losing to winning due to bottom truncation.")
                            print("")
                            print(f"Original winners: {winners}")
                            print(f"New winners: {new_winners}")
                            print(f"Original ballot: {r}")
                            print(f"Truncated ballot: {truncated_r}")
                            print("")
                            print("Original profile:")
                            prof.display()
                            prof.display_margin_graph()
                            vm.display(prof)
                            print("")
                            print("Modified profile:")
                            new_prof.display()
                            new_prof.display_margin_graph()
                            vm.display(new_prof)
                        return True
    
    return False

def find_all_later_no_harm_violations(prof, vm, verbose=False, coalition_size=1, uniform_coalition=True, require_resoluteness=False):
    """
    Returns a list of tuples (original_ballot, truncated_ballot, original_winners, new_winners, new_winners_in_truncated) such that bottom-truncating the original_ballot to the truncated_ballot causes a candidate ranked in the truncated ballot to go from losing to winning.
    
    Args:
        prof: a Profile or ProfileWithTies object.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation.
        coalition_size (int, default=1): Size of the coalition of voters who truncate their ballots.
        uniform_coalition (bool, default=True): If True, all voters in the coalition have the same ballot.
        require_resoluteness (bool, default=False): If True, only profiles with a unique winner before and after truncation are considered.
        
    Returns:
        A list of tuples (original_ballot, truncated_ballot, original_winners, new_winners, new_winners_in_truncated) 
        witnessing violations of Later No Harm. The new_winners_in_truncated element contains the candidates that 
        specifically caused the violation by going from losing to winning while being ranked in the truncated ballot.
    """
    violations = []
    
    # Get the current winners
    winners = vm(prof)
    
    # Convert numpy array winners to list if needed
    if isinstance(winners, np.ndarray):
        winners = winners.tolist()
    
    # If require_resoluteness is True, skip profiles with multiple winners before truncation
    if require_resoluteness and len(winners) > 1:
        return violations
    
    # For individual voter case or uniform coalition
    if uniform_coalition:
        # Check each ranking type in the profile
        for r in prof.ranking_types:
            # Skip if there aren't enough voters with this ranking for the coalition
            if isinstance(r, tuple):
                if prof.rankings.count(r) < coalition_size:
                    continue
                ranked_candidates = list(r)
                r_as_ranking = Ranking({c: i+1 for i, c in enumerate(r)})
            else:
                if sum(1 for ballot in prof.rankings if isinstance(ballot, Ranking) and ballot.cands == r.cands) < coalition_size:
                    continue
                ranked_candidates = list(r.cands)
                r_as_ranking = r
            
            # Try truncating at different positions (keep only first i candidates)
            for i in range(1, len(ranked_candidates)):
                # Create truncated ballot keeping only the first i candidates
                # This is proper truncation - keeping the top i candidates in their original order
                truncated_r = truncate_ballot_from_bottom(r_as_ranking, i)
                
                # Skip if truncation didn't change anything or if truncated ballot is empty
                # Ensure at least one candidate remains ranked
                if len(truncated_r.cands) == 0 or truncated_r.rmap == r_as_ranking.rmap:
                    continue
                
                # Get the truncated ballot as a ranking map
                truncated_rmap = truncated_r.rmap
                
                # Create a new profile with the truncated ballot(s)
                modified_rankings = []
                
                # Add all ballots except those we'll truncate
                for ballot in prof.rankings:
                    # Skip the ballots we'll truncate
                    if isinstance(ballot, tuple) and ballot == r:
                        continue
                    elif isinstance(ballot, Ranking) and isinstance(r, Ranking) and ballot.cands == r.cands:
                        continue
                    
                    # Add the ballot in the correct format
                    if isinstance(ballot, tuple):
                        modified_rankings.append({c: i+1 for i, c in enumerate(ballot)})
                    else:
                        modified_rankings.append(ballot.rmap)
                
                # Make sure we've skipped the right number of ballots
                if isinstance(r, tuple):
                    original_count = prof.rankings.count(r)
                else:
                    original_count = sum(1 for ballot in prof.rankings if isinstance(ballot, Ranking) and ballot.cands == r.cands)
                
                # Add back any ballots we shouldn't have truncated
                for _ in range(original_count - coalition_size):
                    modified_rankings.append(r_as_ranking.rmap)
                
                # Add the truncated ballots
                for _ in range(coalition_size):
                    modified_rankings.append(truncated_rmap)
                
                # Create the new profile with ties
                new_prof = ProfileWithTies(modified_rankings, candidates=prof.candidates)
                if isinstance(prof, ProfileWithTies) and prof.using_extended_strict_preference:
                    new_prof.use_extended_strict_preference()
                
                # Get the new winners
                new_winners = vm(new_prof)
                
                # Convert numpy array winners to list if needed
                if isinstance(new_winners, np.ndarray):
                    new_winners = new_winners.tolist()
                
                # If require_resoluteness is True, skip profiles with multiple winners after truncation
                if require_resoluteness and len(new_winners) > 1:
                    continue
                
                # Check for Later No Harm violation: a candidate ranked in the truncated ballot
                # goes from losing to winning
                truncated_candidates = truncated_r.cands
                
                # Find candidates that are ranked in the truncated ballot and went from losing to winning
                new_winners_in_truncated = [c for c in new_winners if c in truncated_candidates and c not in winners]
                
                if new_winners_in_truncated:
                    violations.append((r, truncated_r, winners, new_winners, new_winners_in_truncated))
                    if verbose:
                        print(f"Later No Harm violation: Candidate(s) {new_winners_in_truncated} went from losing to winning due to bottom truncation.")
                        print("")
                        print(f"Original winners: {winners}")
                        print(f"New winners: {new_winners}")
                        print(f"Original ballot: {r}")
                        print(f"Truncated ballot: {truncated_r}")
                        print("")
                        print("Original profile:")
                        prof.display()
                        prof.display_margin_graph()
                        vm.display(prof)
                        print("")
                        print("Modified profile:")
                        new_prof.display()
                        new_prof.display_margin_graph()
                        vm.display(new_prof)
    
    # For non-uniform coalition
    if not uniform_coalition and coalition_size > 1:
        # Get all possible combinations of ranking types
        ranking_combinations = list(combinations(prof.ranking_types, coalition_size))
        
        for ranking_combo in ranking_combinations:
            # Check if we have enough of each ranking type
            valid_combo = True
            for r in ranking_combo:
                if isinstance(r, tuple):
                    if prof.rankings.count(r) < ranking_combo.count(r):
                        valid_combo = False
                        break
                else:
                    if sum(1 for ballot in prof.rankings if isinstance(ballot, Ranking) and ballot.cands == r.cands) < ranking_combo.count(r):
                        valid_combo = False
                        break
            
            if not valid_combo:
                continue
            
            # For each ranking in the combination, try truncating it
            for i, r in enumerate(ranking_combo):
                if isinstance(r, tuple):
                    ranked_candidates = list(r)
                    r_as_ranking = Ranking({c: i+1 for i, c in enumerate(r)})
                else:
                    ranked_candidates = list(r.cands)
                    r_as_ranking = r
                
                # Try truncating at different positions
                for j in range(1, len(ranked_candidates)):
                    # Create truncated ballot keeping only the first j candidates
                    # This is proper truncation - keeping the top j candidates in their original order
                    truncated_r = truncate_ballot_from_bottom(r_as_ranking, j)
                    
                    # Skip if truncation didn't change anything or if truncated ballot is empty
                    if len(truncated_r.cands) == 0 or truncated_r.rmap == r_as_ranking.rmap:
                        continue
                    
                    # Get the truncated ballot as a ranking map
                    truncated_rmap = truncated_r.rmap
                    
                    # Create a new profile with the truncated ballot(s)
                    modified_rankings = []
                    
                    # Add all ballots except those in the coalition
                    for ballot in prof.rankings:
                        skip = False
                        for coalition_r in ranking_combo:
                            if (isinstance(ballot, tuple) and ballot == coalition_r) or (isinstance(ballot, Ranking) and isinstance(coalition_r, Ranking) and ballot.cands == coalition_r.cands):
                                skip = True
                                break
                        
                        if skip:
                            continue
                        
                        # Add the ballot in the correct format
                        if isinstance(ballot, tuple):
                            modified_rankings.append({c: i+1 for i, c in enumerate(ballot)})
                        else:
                            modified_rankings.append(ballot.rmap)
                    
                    # Add back the coalition ballots, with the i-th one truncated
                    for k, coalition_r in enumerate(ranking_combo):
                        if k == i:
                            modified_rankings.append(truncated_rmap)
                        else:
                            if isinstance(coalition_r, tuple):
                                modified_rankings.append({c: i+1 for i, c in enumerate(coalition_r)})
                            else:
                                modified_rankings.append(coalition_r.rmap)
                    
                    # Create the new profile with ties
                    new_prof = ProfileWithTies(modified_rankings, candidates=prof.candidates)
                    if isinstance(prof, ProfileWithTies) and prof.using_extended_strict_preference:
                        new_prof.use_extended_strict_preference()
                    
                    # Get the new winners
                    new_winners = vm(new_prof)
                    
                    # Convert numpy array winners to list if needed
                    if isinstance(new_winners, np.ndarray):
                        new_winners = new_winners.tolist()
                    
                    # If require_resoluteness is True, skip profiles with multiple winners after truncation
                    if require_resoluteness and len(new_winners) > 1:
                        continue
                    
                    # Check for Later No Harm violation: a candidate ranked in the truncated ballot
                    # goes from losing to winning
                    truncated_candidates = truncated_r.cands
                    
                    # Find candidates that are ranked in the truncated ballot and went from losing to winning
                    new_winners_in_truncated = [c for c in new_winners if c in truncated_candidates and c not in winners]
                    
                    if new_winners_in_truncated:
                        violations.append((r, truncated_r, winners, new_winners, new_winners_in_truncated))
                        if verbose:
                            print(f"Later No Harm violation: Candidate(s) {new_winners_in_truncated} went from losing to winning due to bottom truncation.")
                            print("")
                            print(f"Original winners: {winners}")
                            print(f"New winners: {new_winners}")
                            print(f"Original ballot: {r}")
                            print(f"Truncated ballot: {truncated_r}")
                            print("")
                            print("Original profile:")
                            prof.display()
                            prof.display_margin_graph()
                            vm.display(prof)
                            print("")
                            print("Modified profile:")
                            new_prof.display()
                            new_prof.display_margin_graph()
                            vm.display(new_prof)
    
    return violations

later_no_harm = Axiom(
    "Later No Harm",
    has_violation=has_later_no_harm_violation,
    find_all_violations=find_all_later_no_harm_violations
)

def truncate_ballot_from_top(ranking, num_to_keep):
    """
    Truncate a ballot by removing the top candidates and keeping only the bottom num_to_keep candidates.
    
    Args:
        ranking: A Ranking object or tuple representing a ballot
        num_to_keep: Number of bottom candidates to keep
        
    Returns:
        A truncated Ranking object
    """
    if isinstance(ranking, tuple):
        # For tuple rankings, keep only the last num_to_keep candidates
        truncated = ranking[-num_to_keep:]
        return Ranking({c: i+1 for i, c in enumerate(truncated)})
    else:
        # For Ranking objects, sort candidates by rank and keep only the bottom num_to_keep
        sorted_candidates = sorted(ranking.rmap.keys(), key=lambda c: ranking.rmap[c])
        truncated_rmap = {c: ranking.rmap[c] for c in sorted_candidates[-num_to_keep:]}
        return Ranking(truncated_rmap)

def has_earlier_no_help_violation(prof, vm, verbose=False, coalition_size=1, uniform_coalition=True, require_resoluteness=False):
    """
    Returns True if there is a ballot (or collection of ballots) such that by truncating it from the top, 
    a candidate who is ranked by the truncated ballot goes from winning to losing.

    Viewed in reverse, this means that adding previously unranked candidates to the top of a ballot helped lower ranked candidate to win. 

    Args:
        prof: a Profile or ProfileWithTies object.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation.
        coalition_size (int, default=1): Size of the coalition of voters who truncate their ballots.
        uniform_coalition (bool, default=True): If True, all voters in the coalition have the same ballot.
        require_resoluteness (bool, default=False): If True, only profiles with a unique winner before and after truncation are considered.
        
    Returns:
        Result of the test (bool): Returns True if there is a violation and False otherwise.
    """
    # Get the current winners
    winners = vm(prof)
    
    # Convert numpy array winners to list if needed
    if isinstance(winners, np.ndarray):
        winners = winners.tolist()
    
    # If require_resoluteness is True, skip profiles with multiple winners before truncation
    if require_resoluteness and len(winners) > 1:
        return False
    
    # For individual voter case or uniform coalition
    if uniform_coalition:
        # Check each ranking type in the profile
        for r in prof.ranking_types:
            # Skip if there aren't enough voters with this ranking for the coalition
            if isinstance(r, tuple):
                if prof.rankings.count(r) < coalition_size:
                    continue
                ranked_candidates = list(r)
                r_as_ranking = Ranking({c: i+1 for i, c in enumerate(r)})
            else:
                if sum(1 for ballot in prof.rankings if isinstance(ballot, Ranking) and ballot.cands == r.cands) < coalition_size:
                    continue
                ranked_candidates = list(r.cands)
                r_as_ranking = r
            
            # Try truncating at different positions (keep only last i candidates)
            for i in range(1, len(ranked_candidates)):
                # Create truncated ballot keeping only the last i candidates
                # This is proper truncation - keeping the bottom i candidates in their original order
                truncated_r = truncate_ballot_from_top(r_as_ranking, i)
                
                # Skip if truncation didn't change anything or if truncated ballot is empty
                # Ensure at least one candidate remains ranked
                if len(truncated_r.cands) == 0 or truncated_r.rmap == r_as_ranking.rmap:
                    continue
                
                # Get the truncated ballot as a ranking map
                truncated_rmap = truncated_r.rmap
                
                # Create a new profile with the truncated ballot(s)
                modified_rankings = []
                
                # Add all ballots except those we'll truncate
                for ballot in prof.rankings:
                    # Skip the ballots we'll truncate
                    if isinstance(ballot, tuple) and ballot == r:
                        continue
                    elif isinstance(ballot, Ranking) and isinstance(r, Ranking) and ballot.cands == r.cands:
                        continue
                    
                    # Add the ballot in the correct format
                    if isinstance(ballot, tuple):
                        modified_rankings.append({c: i+1 for i, c in enumerate(ballot)})
                    else:
                        modified_rankings.append(ballot.rmap)
                
                # Make sure we've skipped the right number of ballots
                if isinstance(r, tuple):
                    original_count = prof.rankings.count(r)
                else:
                    original_count = sum(1 for ballot in prof.rankings if isinstance(ballot, Ranking) and ballot.cands == r.cands)
                
                # Add back any ballots we shouldn't have truncated
                for _ in range(original_count - coalition_size):
                    modified_rankings.append(r_as_ranking.rmap)
                
                # Add the truncated ballots
                for _ in range(coalition_size):
                    modified_rankings.append(truncated_rmap)
                
                # Create the new profile with ties
                new_prof = ProfileWithTies(modified_rankings, candidates=prof.candidates)
                if isinstance(prof, ProfileWithTies) and prof.using_extended_strict_preference:
                    new_prof.use_extended_strict_preference()
                
                # Get the new winners
                new_winners = vm(new_prof)
                
                # Convert numpy array winners to list if needed
                if isinstance(new_winners, np.ndarray):
                    new_winners = new_winners.tolist()
                
                # If require_resoluteness is True, skip profiles with multiple winners after truncation
                if require_resoluteness and len(new_winners) > 1:
                    continue
                
                # Check for Earlier No Help violation: a candidate ranked in the truncated ballot
                # goes from winning to losing
                truncated_candidates = truncated_r.cands
                
                # Find candidates that are ranked in the truncated ballot and went from winning to losing
                winners_in_truncated = [c for c in winners if c in truncated_candidates]
                new_losers_in_truncated = [c for c in winners_in_truncated if c not in new_winners]
                
                if new_losers_in_truncated:
                    if verbose:
                        print(f"Earlier No Help violation: Candidate(s) {new_losers_in_truncated} went from winning to losing due to top truncation.")
                        print("")
                        print(f"Original winners: {winners}")
                        print(f"New winners: {new_winners}")
                        print(f"Original ballot: {r}")
                        print(f"Truncated ballot: {truncated_r}")
                        print("")
                        print("Original profile:")
                        prof.display()
                        prof.display_margin_graph()
                        vm.display(prof)
                        print("")
                        print("Modified profile:")
                        new_prof.display()
                        new_prof.display_margin_graph()
                        vm.display(new_prof)
                    return True
    
    # For non-uniform coalition
    if not uniform_coalition and coalition_size > 1:
        # Get all possible combinations of ranking types
        ranking_combinations = list(combinations(prof.ranking_types, coalition_size))
        
        for ranking_combo in ranking_combinations:
            # Check if we have enough of each ranking type
            valid_combo = True
            for r in ranking_combo:
                if isinstance(r, tuple):
                    if prof.rankings.count(r) < ranking_combo.count(r):
                        valid_combo = False
                        break
                else:
                    if sum(1 for ballot in prof.rankings if isinstance(ballot, Ranking) and ballot.cands == r.cands) < ranking_combo.count(r):
                        valid_combo = False
                        break
            
            if not valid_combo:
                continue
            
            # For each ranking in the combination, try truncating it
            for i, r in enumerate(ranking_combo):
                if isinstance(r, tuple):
                    ranked_candidates = list(r)
                    r_as_ranking = Ranking({c: i+1 for i, c in enumerate(r)})
                else:
                    ranked_candidates = list(r.cands)
                    r_as_ranking = r
                
                # Try truncating at different positions
                for j in range(1, len(ranked_candidates)):
                    # Create truncated ballot keeping only the last j candidates
                    # This is proper truncation - keeping the bottom j candidates in their original order
                    truncated_r = truncate_ballot_from_top(r_as_ranking, j)
                    
                    # Skip if truncation didn't change anything or if truncated ballot is empty
                    if len(truncated_r.cands) == 0 or truncated_r.rmap == r_as_ranking.rmap:
                        continue
                    
                    # Get the truncated ballot as a ranking map
                    truncated_rmap = truncated_r.rmap
                    
                    # Create a new profile with the truncated ballot(s)
                    modified_rankings = []
                    
                    # Add all ballots except those in the coalition
                    for ballot in prof.rankings:
                        skip = False
                        for coalition_r in ranking_combo:
                            if (isinstance(ballot, tuple) and ballot == coalition_r) or (isinstance(ballot, Ranking) and isinstance(coalition_r, Ranking) and ballot.cands == coalition_r.cands):
                                skip = True
                                break
                        
                        if skip:
                            continue
                        
                        # Add the ballot in the correct format
                        if isinstance(ballot, tuple):
                            modified_rankings.append({c: i+1 for i, c in enumerate(ballot)})
                        else:
                            modified_rankings.append(ballot.rmap)
                    
                    # Add back the coalition ballots, with the i-th one truncated
                    for k, coalition_r in enumerate(ranking_combo):
                        if k == i:
                            modified_rankings.append(truncated_rmap)
                        else:
                            if isinstance(coalition_r, tuple):
                                modified_rankings.append({c: i+1 for i, c in enumerate(coalition_r)})
                            else:
                                modified_rankings.append(coalition_r.rmap)
                    
                    # Create the new profile with ties
                    new_prof = ProfileWithTies(modified_rankings, candidates=prof.candidates)
                    if isinstance(prof, ProfileWithTies) and prof.using_extended_strict_preference:
                        new_prof.use_extended_strict_preference()
                    
                    # Get the new winners
                    new_winners = vm(new_prof)
                    
                    # Convert numpy array winners to list if needed
                    if isinstance(new_winners, np.ndarray):
                        new_winners = new_winners.tolist()
                    
                    # If require_resoluteness is True, skip profiles with multiple winners after truncation
                    if require_resoluteness and len(new_winners) > 1:
                        continue
                    
                    # Check for Earlier No Help violation: a candidate ranked in the truncated ballot
                    # goes from winning to losing
                    truncated_candidates = truncated_r.cands
                    
                    # Find candidates that are ranked in the truncated ballot and went from winning to losing
                    winners_in_truncated = [c for c in winners if c in truncated_candidates]
                    new_losers_in_truncated = [c for c in winners_in_truncated if c not in new_winners]
                    
                    if new_losers_in_truncated:
                        if verbose:
                            print(f"Earlier No Help violation: Candidate(s) {new_losers_in_truncated} went from winning to losing due to top truncation.")
                            print("")
                            print(f"Original winners: {winners}")
                            print(f"New winners: {new_winners}")
                            print(f"Original ballot: {r}")
                            print(f"Truncated ballot: {truncated_r}")
                            print("")
                            print("Original profile:")
                            prof.display()
                            prof.display_margin_graph()
                            vm.display(prof)
                            print("")
                            print("Modified profile:")
                            new_prof.display()
                            new_prof.display_margin_graph()
                            vm.display(new_prof)
                        return True
    
    return False

def find_all_earlier_no_help_violations(prof, vm, verbose=False, coalition_size=1, uniform_coalition=True, require_resoluteness=False):
    """
    Returns a list of tuples (original_ballot, truncated_ballot, original_winners, new_winners, new_losers_in_truncated) such that top-truncating the original_ballot to the truncated_ballot causes a candidate ranked in the truncated ballot to go from winning to losing.
    
    Args:
        prof: a Profile or ProfileWithTies object.
        vm (VotingMethod): A voting method to test.
        verbose (bool, default=False): If a violation is found, display the violation.
        coalition_size (int, default=1): Size of the coalition of voters who truncate their ballots.
        uniform_coalition (bool, default=True): If True, all voters in the coalition have the same ballot.
        require_resoluteness (bool, default=False): If True, only profiles with a unique winner before and after truncation are considered.
        
    Returns:
        A list of tuples (original_ballot, truncated_ballot, original_winners, new_winners, new_losers_in_truncated) 
        witnessing violations of Earlier No Help. The new_losers_in_truncated element contains the candidates that 
        specifically caused the violation by going from winning to losing while being ranked in the truncated ballot.
    """
    violations = []
    
    # Get the current winners
    winners = vm(prof)
    
    # Convert numpy array winners to list if needed
    if isinstance(winners, np.ndarray):
        winners = winners.tolist()
    
    # If require_resoluteness is True, skip profiles with multiple winners before truncation
    if require_resoluteness and len(winners) > 1:
        return violations
    
    # For individual voter case or uniform coalition
    if uniform_coalition:
        # Check each ranking type in the profile
        for r in prof.ranking_types:
            # Skip if there aren't enough voters with this ranking for the coalition
            if isinstance(r, tuple):
                if prof.rankings.count(r) < coalition_size:
                    continue
                ranked_candidates = list(r)
                r_as_ranking = Ranking({c: i+1 for i, c in enumerate(r)})
            else:
                if sum(1 for ballot in prof.rankings if isinstance(ballot, Ranking) and ballot.cands == r.cands) < coalition_size:
                    continue
                ranked_candidates = list(r.cands)
                r_as_ranking = r
            
            # Try truncating at different positions (keep only last i candidates)
            for i in range(1, len(ranked_candidates)):
                # Create truncated ballot keeping only the last i candidates
                # This is proper truncation - keeping the bottom i candidates in their original order
                truncated_r = truncate_ballot_from_top(r_as_ranking, i)
                
                # Skip if truncation didn't change anything or if truncated ballot is empty
                # Ensure at least one candidate remains ranked
                if len(truncated_r.cands) == 0 or truncated_r.rmap == r_as_ranking.rmap:
                    continue
                
                # Get the truncated ballot as a ranking map
                truncated_rmap = truncated_r.rmap
                
                # Create a new profile with the truncated ballot(s)
                modified_rankings = []
                
                # Add all ballots except those we'll truncate
                for ballot in prof.rankings:
                    # Skip the ballots we'll truncate
                    if isinstance(ballot, tuple) and ballot == r:
                        continue
                    elif isinstance(ballot, Ranking) and isinstance(r, Ranking) and ballot.cands == r.cands:
                        continue
                    
                    # Add the ballot in the correct format
                    if isinstance(ballot, tuple):
                        modified_rankings.append({c: i+1 for i, c in enumerate(ballot)})
                    else:
                        modified_rankings.append(ballot.rmap)
                
                # Make sure we've skipped the right number of ballots
                if isinstance(r, tuple):
                    original_count = prof.rankings.count(r)
                else:
                    original_count = sum(1 for ballot in prof.rankings if isinstance(ballot, Ranking) and ballot.cands == r.cands)
                
                # Add back any ballots we shouldn't have truncated
                for _ in range(original_count - coalition_size):
                    modified_rankings.append(r_as_ranking.rmap)
                
                # Add the truncated ballots
                for _ in range(coalition_size):
                    modified_rankings.append(truncated_rmap)
                
                # Create the new profile with ties
                new_prof = ProfileWithTies(modified_rankings, candidates=prof.candidates)
                if isinstance(prof, ProfileWithTies) and prof.using_extended_strict_preference:
                    new_prof.use_extended_strict_preference()
                
                # Get the new winners
                new_winners = vm(new_prof)
                
                # Convert numpy array winners to list if needed
                if isinstance(new_winners, np.ndarray):
                    new_winners = new_winners.tolist()
                
                # If require_resoluteness is True, skip profiles with multiple winners after truncation
                if require_resoluteness and len(new_winners) > 1:
                    continue
                
                # Check for Earlier No Help violation: a candidate ranked in the truncated ballot
                # goes from winning to losing
                truncated_candidates = truncated_r.cands
                
                # Find candidates that are ranked in the truncated ballot and went from winning to losing
                winners_in_truncated = [c for c in winners if c in truncated_candidates]
                new_losers_in_truncated = [c for c in winners_in_truncated if c not in new_winners]
                
                if new_losers_in_truncated:
                    violations.append((r, truncated_r, winners, new_winners, new_losers_in_truncated))
                    if verbose:
                        print(f"Earlier No Help violation: Candidate(s) {new_losers_in_truncated} went from winning to losing due to top truncation.")
                        print("")
                        print(f"Original winners: {winners}")
                        print(f"New winners: {new_winners}")
                        print(f"Original ballot: {r}")
                        print(f"Truncated ballot: {truncated_r}")
                        print("")
                        print("Original profile:")
                        prof.display()
                        prof.display_margin_graph()
                        vm.display(prof)
                        print("")
                        print("Modified profile:")
                        new_prof.display()
                        new_prof.display_margin_graph()
                        vm.display(new_prof)
    
    # For non-uniform coalition
    if not uniform_coalition and coalition_size > 1:
        # Get all possible combinations of ranking types
        ranking_combinations = list(combinations(prof.ranking_types, coalition_size))
        
        for ranking_combo in ranking_combinations:
            # Check if we have enough of each ranking type
            valid_combo = True
            for r in ranking_combo:
                if isinstance(r, tuple):
                    if prof.rankings.count(r) < ranking_combo.count(r):
                        valid_combo = False
                        break
                else:
                    if sum(1 for ballot in prof.rankings if isinstance(ballot, Ranking) and ballot.cands == r.cands) < ranking_combo.count(r):
                        valid_combo = False
                        break
            
            if not valid_combo:
                continue
            
            # For each ranking in the combination, try truncating it
            for i, r in enumerate(ranking_combo):
                if isinstance(r, tuple):
                    ranked_candidates = list(r)
                    r_as_ranking = Ranking({c: i+1 for i, c in enumerate(r)})
                else:
                    ranked_candidates = list(r.cands)
                    r_as_ranking = r
                
                # Try truncating at different positions
                for j in range(1, len(ranked_candidates)):
                    # Create truncated ballot keeping only the last j candidates
                    # This is proper truncation - keeping the bottom j candidates in their original order
                    truncated_r = truncate_ballot_from_top(r_as_ranking, j)
                    
                    # Skip if truncation didn't change anything or if truncated ballot is empty
                    if len(truncated_r.cands) == 0 or truncated_r.rmap == r_as_ranking.rmap:
                        continue
                    
                    # Get the truncated ballot as a ranking map
                    truncated_rmap = truncated_r.rmap
                    
                    # Create a new profile with the truncated ballot(s)
                    modified_rankings = []
                    
                    # Add all ballots except those in the coalition
                    for ballot in prof.rankings:
                        skip = False
                        for coalition_r in ranking_combo:
                            if (isinstance(ballot, tuple) and ballot == coalition_r) or (isinstance(ballot, Ranking) and isinstance(coalition_r, Ranking) and ballot.cands == coalition_r.cands):
                                skip = True
                                break
                        
                        if skip:
                            continue
                        
                        # Add the ballot in the correct format
                        if isinstance(ballot, tuple):
                            modified_rankings.append({c: i+1 for i, c in enumerate(ballot)})
                        else:
                            modified_rankings.append(ballot.rmap)
                    
                    # Add back the coalition ballots, with the i-th one truncated
                    for k, coalition_r in enumerate(ranking_combo):
                        if k == i:
                            modified_rankings.append(truncated_rmap)
                        else:
                            if isinstance(coalition_r, tuple):
                                modified_rankings.append({c: i+1 for i, c in enumerate(coalition_r)})
                            else:
                                modified_rankings.append(coalition_r.rmap)
                    
                    # Create the new profile with ties
                    new_prof = ProfileWithTies(modified_rankings, candidates=prof.candidates)
                    if isinstance(prof, ProfileWithTies) and prof.using_extended_strict_preference:
                        new_prof.use_extended_strict_preference()
                    
                    # Get the new winners
                    new_winners = vm(new_prof)
                    
                    # Convert numpy array winners to list if needed
                    if isinstance(new_winners, np.ndarray):
                        new_winners = new_winners.tolist()
                    
                    # If require_resoluteness is True, skip profiles with multiple winners after truncation
                    if require_resoluteness and len(new_winners) > 1:
                        continue
                    
                    # Check for Earlier No Help violation: a candidate ranked in the truncated ballot
                    # goes from winning to losing
                    truncated_candidates = truncated_r.cands
                    
                    # Find candidates that are ranked in the truncated ballot and went from winning to losing
                    winners_in_truncated = [c for c in winners if c in truncated_candidates]
                    new_losers_in_truncated = [c for c in winners_in_truncated if c not in new_winners]
                    
                    if new_losers_in_truncated:
                        violations.append((r, truncated_r, winners, new_winners, new_losers_in_truncated))
                        if verbose:
                            print(f"Earlier No Help violation: Candidate(s) {new_losers_in_truncated} went from winning to losing due to top truncation.")
                            print("")
                            print(f"Original winners: {winners}")
                            print(f"New winners: {new_winners}")
                            print(f"Original ballot: {r}")
                            print(f"Truncated ballot: {truncated_r}")
                            print("")
                            print("Original profile:")
                            prof.display()
                            prof.display_margin_graph()
                            vm.display(prof)
                            print("")
                            print("Modified profile:")
                            new_prof.display()
                            new_prof.display_margin_graph()
                            vm.display(new_prof)
    
    return violations

earlier_no_help = Axiom(
    "Earlier No Help",
    has_violation=has_earlier_no_help_violation,
    find_all_violations=find_all_earlier_no_help_violations
)