import pytest
import random
from pref_voting.social_welfare_function import SocialWelfareFunction, swf 
from pref_voting.profiles import Profile
from pref_voting.rankings import Ranking

def simple_swf_method(profile, curr_cands=None):
    if curr_cands is None:
        return Ranking({c:cidx for cidx,c in enumerate(profile.candidates)})
    else:
        return Ranking({c:cidx for cidx,c in enumerate(curr_cands)})

@pytest.fixture
def dummy_profile():
    return Profile([[0, 1, 2]])

@pytest.fixture
def ranking_method():
    return SocialWelfareFunction(simple_swf_method, "Simple SWF")

def test_social_welfare_function_initialization():
    ranking_method = SocialWelfareFunction(simple_swf_method, "Test SWF")
    assert ranking_method.name == "Test SWF"
    assert ranking_method.swf == simple_swf_method

def test_social_welfare_function_call(ranking_method, dummy_profile):
    ranking = ranking_method(dummy_profile)
    r = Ranking({0: 0, 1: 1, 2: 2})
    assert ranking == r
    
def test_social_welfare_function_winners(ranking_method, dummy_profile):
    ws = ranking_method.winners(dummy_profile)
    assert ws == [0]
    
def test_social_welfare_function_display(ranking_method, dummy_profile, capsys):
    ranking_method.display(dummy_profile)
    captured = capsys.readouterr()

    assert "Simple SWF ranking is 0 1 2" in captured.out

def test_social_welfare_function_set_name(ranking_method):
    new_name = "Updated RANKING"
    ranking_method.set_name(new_name)
    assert ranking_method.name == new_name