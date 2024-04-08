import pytest
import random
from pref_voting.voting_method import VotingMethod, vm 
from pref_voting.profiles import Profile


def simple_vm_method(profile, curr_cands=None):
    if curr_cands is None:
        return sorted(profile.candidates)
    else:
        return sorted(curr_cands)

@pytest.fixture
def dummy_profile():
    return Profile([[0, 1, 2]])

@pytest.fixture
def voting_method():
    return VotingMethod(simple_vm_method, "Simple VM")

def test_voting_method_initialization():
    vm = VotingMethod(simple_vm_method, "Test VM")
    assert vm.name == "Test VM"
    assert vm.vm == simple_vm_method

def test_voting_method_call(voting_method, dummy_profile):
    ws = voting_method(dummy_profile)
    assert ws == sorted(dummy_profile.candidates)
    

def test_voting_method_choose(voting_method, dummy_profile):
    random.seed(0)  # Setting seed for reproducibility
    winner = voting_method.choose(dummy_profile)
    assert winner in dummy_profile.candidates

def test_voting_method_prob(voting_method, dummy_profile):
    probs = voting_method.prob(dummy_profile)

    expected_prob = 1.0 / len(dummy_profile.candidates)
    for c in dummy_profile.candidates:
        assert probs[c] == expected_prob

def test_voting_method_display(voting_method, dummy_profile, capsys):
    voting_method.display(dummy_profile)
    captured = capsys.readouterr()

    assert "Simple VM winners are {0, 1, 2}" in captured.out

def test_voting_method_set_name(voting_method):
    new_name = "Updated VM"
    voting_method.set_name(new_name)
    assert voting_method.name == new_name