import pytest
import random
from pref_voting.probabilistic_voting_method import ProbabilisticVotingMethod, pvm 
from pref_voting.profiles import Profile


def simple_pvm_method(profile, curr_cands=None):
    if curr_cands is None:
        pr = {profile.candidates[0]: 1.0}
        pr.update({c: 0.0 for c in profile.candidates[1:]})
        return pr
    else:
        pr = {curr_cands[0]: 1.0}
        pr.update({c: 0.0 for c in curr_cands[1:]})
        return pr

@pytest.fixture
def dummy_profile():
    return Profile([[0, 1, 2]])

@pytest.fixture
def voting_method():
    return ProbabilisticVotingMethod(simple_pvm_method, "Simple VM")

def test_probabilistic_voting_method_initialization():
    pvm = ProbabilisticVotingMethod(simple_pvm_method, "Test VM")
    assert pvm.name == "Test VM"
    assert pvm.pvm == simple_pvm_method

def test_probabilistic_voting_method_call(voting_method, dummy_profile):
    prob = voting_method(dummy_profile)
    pr = {dummy_profile.candidates[0]: 1.0}
    pr.update({c: 0.0 for c in dummy_profile.candidates[1:]})
    assert prob == pr
    
def test_probabilistic_voting_method_choose(voting_method, dummy_profile):
    random.seed(0)  # Setting seed for reproducibility
    winner = voting_method.choose(dummy_profile)
    assert winner == dummy_profile.candidates[0]


def test_probabilistic_voting_method_support(voting_method, dummy_profile):
    support = voting_method.support(dummy_profile)
    assert support == [dummy_profile.candidates[0]]

def test_probabilistic_voting_method_display(voting_method, dummy_profile, capsys):
    voting_method.display(dummy_profile)
    captured = capsys.readouterr()

    assert "Simple VM probability is {0: 1.0, 1: 0.0, 2: 0.0}\n" in captured.out

def test_probabilistic_voting_method_set_name(voting_method):
    new_name = "Updated VM"
    voting_method.set_name(new_name)
    assert voting_method.name == new_name