import pytest
from pref_voting.generate_profiles import generate_profile, generate_profile_with_groups, generate_truncated_profile, minimal_profile_from_edge_order
from pref_voting.profiles import Profile
from pref_voting.profiles_with_ties import ProfileWithTies

def test_generate_profile():
    prof = generate_profile(4, 3)
    assert type(prof) == Profile
    assert len(prof.candidates) == 4
    assert len(prof.rankings) == 3

def test_impartial_culture():
    prof = generate_profile(4, 3, probmodel="IC")
    assert type(prof) == Profile
    assert len(prof.candidates) == 4
    assert len(prof.rankings) == 3

    prof = generate_profile(4, 3, probmodel="impartial")
    assert type(prof) == Profile
    assert len(prof.candidates) == 4
    assert len(prof.rankings) == 3

def test_impartial_anonymous_culture():
    prof = generate_profile(4, 3, probmodel="IAC")
    assert type(prof) == Profile
    assert len(prof.candidates) == 4
    assert len(prof.rankings) == 3

    prof = generate_profile(4, 3, probmodel="impartial_anonymous")
    assert type(prof) == Profile
    assert len(prof.candidates) == 4
    assert len(prof.rankings) == 3

def test_mallows():
    prof = generate_profile(4, 3, probmodel="MALLOWS")
    assert type(prof) == Profile
    assert len(prof.candidates) == 4
    assert len(prof.rankings) == 3

    prof = generate_profile(4, 3, probmodel="mallows")
    assert type(prof) == Profile
    assert len(prof.candidates) == 4
    assert len(prof.rankings) == 3

    prof = generate_profile(4, 
                             3, 
                             probmodel="mallows", 
                             central_vote=[0, 1, 2, 3], 
                             phi=0.5)
    assert type(prof) == Profile
    assert len(prof.candidates) == 4
    assert len(prof.rankings) == 3
    
    prof = generate_profile(4, 
                             3, 
                             probmodel="mallows", 
                             normalise_phi=True,
                             central_vote=[0, 1, 2, 3], 
                             phi=0.5)
    assert type(prof) == Profile
    assert len(prof.candidates) == 4
    assert len(prof.rankings) == 3

    prof = generate_profile(4, 
                             3, 
                             probmodel="mallows", 
                             central_vote=[0, 1, 2, 3], 
                             phi=0.0)
    assert type(prof) == Profile
    assert len(prof.candidates) == 4
    assert len(prof.rankings) == 3
    assert len(prof.ranking_types) == 1

def test_mallows_fixed():
    prof = generate_profile(4, 
                             3, 
                             probmodel="MALLOWS-0.2")
    assert type(prof) == Profile
    assert len(prof.candidates) == 4
    assert len(prof.rankings) == 3
   
    prof = generate_profile(4, 
                             3, 
                             probmodel="MALLOWS-0.8")
    assert type(prof) == Profile
    assert len(prof.candidates) == 4
    assert len(prof.rankings) == 3
   
def test_mallows_random():
    prof = generate_profile(4, 
                             3, 
                             probmodel="MALLOWS-R")
    assert type(prof) == Profile
    assert len(prof.candidates) == 4
    assert len(prof.rankings) == 3

def test_relphi_mallows():

    prof = generate_profile(4, 3, probmodel="MALLOWS-RELPHI")
    assert type(prof) == Profile
    assert len(prof.candidates) == 4
    assert len(prof.rankings) == 3

    prof = generate_profile(4, 3,
                            probmodel="MALLOWS-RELPHI",
                            relphi=0.2)
    
    assert type(prof) == Profile
    assert len(prof.candidates) == 4
    assert len(prof.rankings) == 3

    prof = generate_profile(4, 3, 
                             probmodel="MALLOWS-RELPHI-0.375")
    assert type(prof) == Profile
    assert len(prof.candidates) == 4
    assert len(prof.rankings) == 3

    prof = generate_profile(4, 
                             3, 
                             probmodel="MALLOWS-RELPHI", 
                             normalise_phi=True,
                             centeral_vote=[0, 1, 2, 3], 
                             relphi=0.5)
    assert type(prof) == Profile
    assert len(prof.candidates) == 4
    assert len(prof.rankings) == 3

    prof = generate_profile(4, 
                            3,
                            probmodel="MALLOWS-RELPHI-R", 
                            )
    assert type(prof) == Profile
    assert len(prof.candidates) == 4
    assert len(prof.rankings) == 3


def test_urn():

    prof = generate_profile(4, 3, 
                             probmodel="URN")
    assert type(prof) == Profile
    assert len(prof.candidates) == 4
    assert len(prof.rankings) == 3

    prof = generate_profile(4, 3, 
                             probmodel="urn")
    assert type(prof) == Profile
    assert len(prof.candidates) == 4
    assert len(prof.rankings) == 3

    prof = generate_profile(4, 3, 
                             probmodel="urn",
                             alpha=0.5)
    assert type(prof) == Profile
    assert len(prof.candidates) == 4
    assert len(prof.rankings) == 3

def test_urn10():

    prof = generate_profile(4, 
                            3, 
                            probmodel="URN-10")
    assert type(prof) == Profile
    assert len(prof.candidates) == 4
    assert len(prof.rankings) == 3

def test_urn0_3():

    prof = generate_profile(4, 
                            3,
                            probmodel="URN-0.3")
    assert type(prof) == Profile
    assert len(prof.candidates) == 4
    assert len(prof.rankings) == 3

def test_urn_r():

    prof = generate_profile(4, 
                            3, 
                            probmodel="URN-R")
    assert type(prof) == Profile
    assert len(prof.candidates) == 4
    assert len(prof.rankings) == 3

def test_plackett_luce():

    prof = generate_profile(4, 
                            3, 
                            alphas=[1, 2, 5, 0.5],
                            probmodel="plackett_luce")
    assert type(prof) == Profile
    assert len(prof.candidates) == 4
    assert len(prof.rankings) == 3

def test_didi():

    prof = generate_profile(4, 3, 
                            alphas=[1, 2, 5, 0.5],
                            probmodel="didi")
    assert type(prof) == Profile
    assert len(prof.candidates) == 4
    assert len(prof.rankings) == 3

def test_stratification():

    prof = generate_profile(4, 3, 
                             alphas=[1, 2, 5, 0.5],
                             weight=0.75,
                             probmodel="stratification")
    assert type(prof) == Profile
    assert len(prof.candidates) == 4
    assert len(prof.rankings) == 3

def test_single_peaked_conitzer():

    prof = generate_profile(4, 3, 
                             probmodel="single_peaked_conitzer")
    assert type(prof) == Profile
    assert len(prof.candidates) == 4
    assert len(prof.rankings) == 3

def test_single_peaked():

    prof = generate_profile(4, 3, 
                             probmodel="SinglePeaked")
    assert type(prof) == Profile
    assert len(prof.candidates) == 4
    assert len(prof.rankings) == 3

    prof = generate_profile(4, 3, 
                             probmodel="single_peaked_walsh")
    assert type(prof) == Profile
    assert len(prof.candidates) == 4
    assert len(prof.rankings) == 3

def test_single_peaked_circle():

    prof = generate_profile(4, 3, 
                             probmodel="single_peaked_circle")
    assert type(prof) == Profile
    assert len(prof.candidates) == 4
    assert len(prof.rankings) == 3

def test_single_crossing():

    prof = generate_profile(4, 3, 
                             probmodel="single_crossing")
    assert type(prof) == Profile
    assert len(prof.candidates) == 4
    assert len(prof.rankings) == 3

def test_euclidean():

    prof = generate_profile(4, 3, 
                             probmodel="euclidean")
    assert type(prof) == Profile
    assert len(prof.candidates) == 4
    assert len(prof.rankings) == 3

    prof = generate_profile(4, 3, 
                             dim=5,
                             probmodel="euclidean")
    assert type(prof) == Profile
    assert len(prof.candidates) == 4
    assert len(prof.rankings) == 3

    prof = generate_profile(4, 3, 
                             dim=2,
                             space='uniform',
                             probmodel="euclidean")
    assert type(prof) == Profile
    assert len(prof.candidates) == 4
    assert len(prof.rankings) == 3

    prof = generate_profile(4, 3, 
                             dim=2,
                             space='ball',
                             probmodel="euclidean")
    assert type(prof) == Profile
    assert len(prof.candidates) == 4
    assert len(prof.rankings) == 3

    prof = generate_profile(4, 3, 
                             dim=2,
                             space='gaussian',
                             probmodel="euclidean")
    assert type(prof) == Profile
    assert len(prof.candidates) == 4
    assert len(prof.rankings) == 3


    prof = generate_profile(4, 3, 
                             dim=4,
                             space='sphere',
                             probmodel="euclidean")
    assert type(prof) == Profile
    assert len(prof.candidates) == 4
    assert len(prof.rankings) == 3


def test_generate_multiple_profiles():

    profs = generate_profile(4, 3, num_profiles=5)
    assert type(profs) == list
    assert len(profs) == 5
    assert all([len(prof.candidates) == 4 for prof in profs])
    assert all([len(prof.rankings) == 3 for prof in profs])

def test_generate_profile_with_groups():

    prof = generate_profile_with_groups(4, 3,
                                         [{"probmodel":"impartial"}])
    assert type(prof) == Profile
    assert len(prof.candidates) == 4
    assert len(prof.rankings) == 3

    prof = generate_profile_with_groups(4, 3,
                                         [{"probmodel":"mallows"}, 
                                          {"probmodel":"impartial"}])
    assert type(prof) == Profile
    assert len(prof.candidates) == 4
    assert len(prof.rankings) == 3

    prof = generate_profile_with_groups(4, 3,
                                         [{"probmodel":"mallows", 
                                           "central_vote":[0, 1, 2, 3],
                                           "phi":0.5}, 
                                          {"probmodel":"mallows",
                                            "central_vote":[3, 2, 1, 0], 
                                            "phi":0.5}],
                                          weights=[1, 2])
    assert type(prof) == Profile
    assert len(prof.candidates) == 4
    assert len(prof.rankings) == 3

def test_generate_profile_with_groups_multiple_profiles():

    profs = generate_profile(4, 3, 
                              [{
                                  "probmodel":"mallows", 
                                   "central_vote":[0, 1, 2, 3],
                                   "phi":0.5}, 
                                {
                                    "probmodel":"mallows",
                                    "central_vote":[3, 2, 1, 0],
                                    "phi":0.5}],
                              weights=[1, 2],
                              num_profiles=5)
    assert type(profs) == list
    assert len(profs) == 5
    assert all([len(prof.candidates) == 4 for prof in profs])
    assert all([len(prof.rankings) == 3 for prof in profs])

def test_generate_truncated_profile():

    prof = generate_truncated_profile(4, 5)
    assert type(prof) == ProfileWithTies
    assert len(prof.candidates) == 4
    assert len(prof.rankings) == 5


    prof = generate_truncated_profile(4, 5, probmod="RT")
    assert type(prof) == ProfileWithTies
    assert len(prof.candidates) == 4
    assert len(prof.rankings) == 5

def test_minimal_profile_from_edge_order():
    print()
    prof = minimal_profile_from_edge_order([0, 1, 2], [(0, 1), (1, 2), (0, 2)])
    assert type(prof) == Profile
    assert len(prof.candidates) == 3

    prof = minimal_profile_from_edge_order([0, 1, 2], [(0, 1), (1, 2), (2, 0)])
    assert type(prof) == Profile
    assert len(prof.candidates) == 3
