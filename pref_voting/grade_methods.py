'''
    File: grade_methods.py
    Author: Eric Pacuit (epacuit@umd.edu)
    Date: September 24, 2023
    
    Implementations of grading methods (also called evaluative methods). 
'''
from pref_voting.voting_method import  *
from itertools import product

@vm(name="Score Vote")
def score_vote(gprofile, curr_cands=None):
    """Return the candidates with the largest sum of the scores.  If ``curr_cands`` is provided, then the score vote is restricted to the candidates in ``curr_cands``.
    """
    
    curr_cands = gprofile.candidates if curr_cands is None else curr_cands
    
    scores = {
        c: gprofile.sum(c) 
        for c in curr_cands if gprofile.has_grade(c)
        }

    max_score = max(scores.values())

    return sorted([c for c in scores.keys() if scores[c] == max_score])

@vm(name="Approval Vote")
def approval_vote(gprofile, curr_cands=None):
    """Return the approval vote of the grade profile ``gprofile``.  If ``curr_cands`` is provided, then the approval vote is restricted to the candidates in ``curr_cands``.

    .. warning:: 
        Approval Vote only works on Grade Profiles that are based on 2 grades: 0 and 1.

    """
    assert sorted(gprofile.grades) == [0, 1], "The  grades in the profile must be {0, 1}."
    
    return score_vote(gprofile, curr_cands=curr_cands)

@vm(name="STAR Vote")
def star_vote(gprofile, curr_cands=None):
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
