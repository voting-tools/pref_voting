import pytest
import random
from pref_voting.num_evaluation_method import NumEvaluationMethod, nem 
from pref_voting.profiles import Profile
from pref_voting.mappings import Utility

def simple_nem_method(profile, curr_cands=None):
    if curr_cands is None:
        return Utility({c:1 for c in profile.candidates})
    else:
        return Utility({c:1 for c in profile.curr_cands})

@pytest.fixture
def dummy_profile():
    return Profile([[0, 1, 2]])

@pytest.fixture
def num_eval_method():
    return NumEvaluationMethod(simple_nem_method, "Simple NEM")

def test_num_evlauation_method_initialization():
    ev_method = NumEvaluationMethod(simple_nem_method, "Test NEM")
    assert ev_method.name == "Test NEM"
    assert ev_method.nem == simple_nem_method

def test_num_evlauation_method_call(num_eval_method, dummy_profile):
    ev = num_eval_method(dummy_profile)
    assert isinstance(ev, Utility)
    assert ev(0) == 1
    assert ev(1) == 1
    assert ev(2) == 1
    
def test_num_evlauation_method_display(num_eval_method, dummy_profile, capsys):
    num_eval_method.display(dummy_profile)
    captured = capsys.readouterr()

    assert "Simple NEM evaluation is U(0) = 1, U(1) = 1, U(2) = 1\n" in captured.out

def test_num_evlauation_method_set_name(num_eval_method):
    new_name = "Updated VM"
    num_eval_method.set_name(new_name)
    assert num_eval_method.name == new_name