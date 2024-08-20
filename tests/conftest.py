
import pytest
from pref_voting.profiles import Profile
from pref_voting.profiles_with_ties import ProfileWithTies

@pytest.fixture
def condorcet_cycle():
    return Profile([
        [0, 1, 2], 
        [1, 2, 0], 
        [2, 0, 1]])

@pytest.fixture
def linear_profile_0():
    return Profile([
        [0, 1, 2], 
        [2, 1, 0]], 
        rcounts=[2, 1])

@pytest.fixture
def profile_with_ties_linear_0():
    return ProfileWithTies([
        {0:1, 1:2, 2:3}, 
        {0:3, 1:2, 2:1}],
        rcounts=[2, 1])

@pytest.fixture
def profile_with_ties():
    return ProfileWithTies([
        {0:1, 1:1, 2:2}, 
        {0:2, 1:2, 2:1}],
        rcounts=[2, 1])

@pytest.fixture
def profile_single_voter():
    return Profile([[0, 1, 2, 3]])