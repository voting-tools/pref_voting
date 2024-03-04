'''
    File: grade_methods.py
    Author: Wes Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)
    Date: September 24, 2023
    
    Implementations of grading methods (also called evaluative methods). 
'''
from pref_voting.voting_method import  *
from itertools import product

@vm(name="Score Voting")
def score_voting(gprofile, curr_cands=None, evaluation_method="sum"):
    """Return the candidates with the largest scores, where scores are evaluated using the ``evaluation_method``, where the default is summing the scores of the candidates.  If ``curr_cands`` is provided, then the score vote is restricted to the candidates in ``curr_cands``.
    """
    
    curr_cands = gprofile.candidates if curr_cands is None else curr_cands
    if evaluation_method == "sum":
        evaluation_method_func = gprofile.sum
    elif evaluation_method == "mean" or evaluation_method == "average": 
        evaluation_method_func = gprofile.avg
    elif evaluation_method == "median": # returns lower median
        evaluation_method_func = gprofile.median

    scores = {
        c: evaluation_method_func(c) 
        for c in curr_cands if gprofile.has_grade(c)
        }

    max_score = max(scores.values())

    return sorted([c for c in scores.keys() if scores[c] == max_score])

@vm(name="Approval")
def approval(gprofile, curr_cands=None):
    """Return the approval vote of the grade profile ``gprofile``.  If ``curr_cands`` is provided, then the approval vote is restricted to the candidates in ``curr_cands``.

    .. warning:: 
        Approval Vote only works on Grade Profiles that are based on 2 grades: 0 and 1.

    """
    assert sorted(gprofile.grades) == [0, 1], "The  grades in the profile must be {0, 1}."
    
    return score_voting(gprofile, curr_cands=curr_cands, evaluation_method="sum")

@vm(name="Dis&approval")
def dis_and_approval(gprofile, curr_cands=None):
    """Return the Dis&approval vote of the grade profile ``gprofile``.  If ``curr_cands`` is provided, then the dis&approval vote is restricted to the candidates in ``curr_cands``.  See https://link.springer.com/article/10.1007/s00355-013-0766-7 for more information.

    .. warning:: 
        Dis&approval only works on Grade Profiles that are based on 2 grades: -1 and 1.

    """
    assert sorted(gprofile.grades) == [-1, 0, 1], "The  grades in the profile must be {-1, 0, 1}."
    
    return score_voting(gprofile, curr_cands=curr_cands, evaluation_method="sum")

@vm(name="Cumulative Voting")
def cumulative_voting(gprofile, curr_cands=None, max_total_grades=5):
    """Return the cumulative vote winner of the grade profile ``gprofile``.   This is the candidates with the largest sum of the grades where each voter submits a ballot of scores that sum to ``max_total_grades``.   If ``curr_cands`` is provided, then the cumulative vote is restricted to the candidates in ``curr_cands``."""
    assert sorted(gprofile.grades) == list(range(max_total_grades + 1)) and np.sum(gprofile.grades) == max_total_grades , f"For cumulative voting, the sum the grades must be {max_total_grades}."
    
    return score_voting(gprofile, curr_cands=curr_cands, evaluation_method="sum")

@vm(name="STAR")
def star(gprofile, curr_cands=None):
    """ Identify the top two candidates according to the sum of the grades for each candidate. Then hold a runoff between the top two candidates where the candidate that is ranked above the other by the most voters is the winner.  The candidates that move to the runoff round are: the candidate(s) with the largest sum of the grades and the candidate(s) with the 2nd largest sum of the grades (or perhaps tied for the largest sum of the grades). In the case of multiple candidates tied for the largest or 2nd largest sum of the grades, use parallel-universe tiebreaking: a candidate is a Star Vote winner if it is a winner in some head-to-head runoff as described. If the candidates are all tied for the  largest sum of the grades, then all candidates are winners. 

    See https://starvoting.us for more information.
    
    If ``curr_cands`` is provided, then the winners is restricted to the candidates in ``curr_cands``.

    .. warning:: 
        Star Vote only works on Grade Profiles that are based on 6 grades: 0, 1, 2, 3, 4, and 5.
    """

    assert sorted(gprofile.grades) == [0, 1, 2, 3, 4, 5], "The  grades in the profile must be {0, 1, 2, 3, 4, 5}."

    curr_cands = gprofile.candidates if curr_cands is None else curr_cands

    if len(curr_cands) == 1: 
        return list(curr_cands)

    cand_to_scores = {
        c: gprofile.sum(c) 
        for c in curr_cands if gprofile.has_grade(c)
        }
    
    scores = sorted(list(set(cand_to_scores.values())), reverse=True)

    max_score = scores[0]
    first = [c for c in cand_to_scores.keys() if cand_to_scores[c] == max_score]

    second = list()
    if len(first) == 1:
        second_score = scores[1]
        second = [c for c in cand_to_scores.keys() if cand_to_scores[c] == second_score]

    if len(second) > 0:
        all_runoff_pairs = product(first, second)
    else: 
        all_runoff_pairs = [(c1,c2) for c1,c2 in product(first, first) if c1 != c2]

    winners = list()
    for c1, c2 in all_runoff_pairs: 
        
        if gprofile.margin(c1,c2) > 0:
            winners.append(c1)
        elif gprofile.margin(c1,c2) < 0:
            winners.append(c2)
        elif gprofile.margin(c1,c2) == 0:
            winners.append(c1)
            winners.append(c2)
    
    return sorted(list(set(winners)))


def tiebreaker_diff(gprofile, cand, median_grade):
    """
    Tiebreaker when the there are multiple candidates with the largest median grade.  
    The tiebreaker is the difference between the proportion of voters who grade the candidate higher than the median grade and the proportion of voters who grade the candidate lower than the median grade.
    """
    prop_proponents = gprofile.proportion_with_higher_grade(cand, median_grade)
    prop_opponents = gprofile.proportion_with_lower_grade(cand, median_grade)

    return prop_proponents - prop_opponents

def tiebreaker_relative_shares(gprofile, cand, median_grade):
    """
    Tiebreaker when the there are multiple candidates with the largest median grade. 
    Returns the *relative shares* of the proponents and opponents of the candidate. 
    """

    prop_proponents = gprofile.proportion_with_higher_grade(cand, median_grade)
    prop_opponents = gprofile.proportion_with_lower_grade(cand, median_grade)

    return  (prop_proponents - prop_opponents) / (2 * (prop_proponents + prop_opponents))

def tiebreaker_normalized_difference(gprofile, cand, median_grade):
    """
    Tiebreaker when the there are multiple candidates with the largest median grade. 
    Returns the *normalized difference* of the proponents and opponents of the candidate. 
    """

    prop_proponents = gprofile.proportion_with_higher_grade(cand, median_grade)
    prop_opponents = gprofile.proportion_with_lower_grade(cand, median_grade)

    return  (prop_proponents - prop_opponents) /(2 * (1 - prop_proponents -  prop_opponents))


def tiebreaker_majority_judgement(gprofile, cand, median_grade):
    """
    Tiebreaker when the there are multiple candidates with the largest median grade. 
    Returns the proportion of voters assigning a higher grade than the median to cand if it is greater than the proportion of voters assigning a lower grade than the median to cand, otherwise return -1 * the proportion of voters assigning a lower grade than the median to cand. 
    """

    prop_proponents = gprofile.proportion_with_higher_grade(cand, median_grade)
    prop_opponents = gprofile.proportion_with_lower_grade(cand, median_grade)

    if prop_proponents > prop_opponents:
        return prop_proponents
    elif prop_opponents >= prop_proponents:
        return -prop_opponents


def greatest_median(gprofile, curr_cands=None, tb_func = tiebreaker_majority_judgement):

    """
    Returns the candidate(s) with the greatest median grade.  If there is a tie, the tie is broken by the tiebreaker function.
    
    """
    median_winners = score_voting(gprofile, curr_cands=curr_cands, evaluation_method="median")

    if len(median_winners) == 1:
        return median_winners
    else:
        tb_scores = {c: tb_func(gprofile, c, gprofile.median(c)) for c in median_winners}
        return sorted([c for c in tb_scores if tb_scores[c] == max(tb_scores.values())])

@vm(name="Majority Judgement")
def majority_judgement(gprofile, curr_cands=None):
    """
    The Majority Judgement voting method as describe in Balinski and Laraki (https://mitpress.mit.edu/9780262545716/majority-judgment/).
    """
    return greatest_median(gprofile, curr_cands=curr_cands, tb_func = tiebreaker_majority_judgement)