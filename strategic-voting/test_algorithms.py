"""
Pytest unit tests for the Strategic-Voting algorithms.

The suite is split into two sections:
1. Algorithm 1 – single-voter manipulation.
2. Algorithm 2 – coalitional manipulation.

Tests that rely on functionality not yet implemented are marked
with @pytest.mark.xfail.
"""
from __future__ import annotations

import math
import random

import pytest
#from pref_voting.scoring_methods import borda
from strategic_voting_algorithms import (
    algorithm1_single_voter,
    algorithm2_coalitional,
    borda,
    make_x_approval,
)
from typing import List, Union
try:
    from pref_voting.profiles import Profile        # real class
except ImportError:
    Profile = None                                   # tests on a lean env

from pref_voting.helper import create_election
# ---------------------------------------------------------------------
# Shared constants – paper’s “4 honest + 1 manipulator” example
TEAM_PROFILE_EXAMPLE = [
    ["p", "c", "a", "b"],  # p>c>a>b
    ["p", "b", "a", "c"],  # p>b>a>c
    ["b", "p", "a", "c"],  # b>p>a>c
    ["b", "a", "c", "p"],  # b>a>c>p
]
OPPONENT_ORDER_EXAMPLE = ["b", "p", "a", "c"]  # b>p>a>c
# ---------------------------------------------------------------------
#                           Algorithm 1
# ---------------------------------------------------------------------


def test_alg1_borda_example():
    """Single-voter manipulation in the paper’s running example."""
    ok, vote = algorithm1_single_voter(
        borda, TEAM_PROFILE_EXAMPLE, OPPONENT_ORDER_EXAMPLE, preferred="p"
    )
    assert ok
    assert vote == ["a", "p", "c", "b"]


def test_alg1_empty_input():
    ok, vote = algorithm1_single_voter(borda, [], [], "p")
    assert not ok and vote is None


def test_alg1_threshold_guard():
    """pos(p) < ⌈m/2⌉ ⇒ manipulation impossible."""
    team_profile = [["b", "a", "p"]]        # m = 3
    opponent_order = ["b", "a", "p"]        # pos(p)=0 < ⌈3/2⌉=2
    ok, vote = algorithm1_single_voter(borda, team_profile, opponent_order, "p")
    assert not ok and vote is None


def test_alg1_simple_case():
    """Paper’s 5-candidate example (4 honest + 1 manipulator)."""
    ok, vote = algorithm1_single_voter(
        borda, TEAM_PROFILE_EXAMPLE, OPPONENT_ORDER_EXAMPLE, "p"
    )
    assert ok
    assert vote == ["a", "p", "c", "b"]


def test_alg1_random_threshold_guard():
    """Randomized guard: pos(p) < ⌈m/2⌉ must fail."""
    for _ in range(20):
        m = random.randint(3, 10)
        candidates = [chr(ord("a") + i) for i in range(m)]
        preferred = random.choice(candidates)

        pos = random.randint(math.ceil(m / 2), m - 1)
        others = [c for c in candidates if c != preferred]
        random.shuffle(others)
        opponent_order = others.copy()
        opponent_order.insert(pos, preferred)

        team_profile = [
            random.sample(candidates, k=m) for __ in range(random.randint(0, 5))
        ]
        ok, _ = algorithm1_single_voter(borda, team_profile, opponent_order, preferred)
        assert not ok


def test_alg1_random_output_shape():
    """If algorithm 1 succeeds, its ballot must be a permutation."""
    for _ in range(20):
        m = random.randint(3, 8)
        candidates = [chr(ord("a") + i) for i in range(m)]
        preferred = random.choice(candidates)
        opponent_order = random.sample(candidates, k=m)
        team_profile = [
            random.sample(candidates, k=m) for __ in range(random.randint(0, 4))
        ]

        ok, vote = algorithm1_single_voter(borda, team_profile, opponent_order, preferred)
        if ok:
            assert sorted(vote) == sorted(candidates)

@pytest.mark.skipif(Profile is None, reason="pref_voting not installed")
def test_alg1_accepts_pref_profile():
    # 4-candidate mapping: p=0, c=1, a=2, b=3
    ballots = [
        [0, 1, 2, 3],  # p c a b
        [0, 3, 2, 1],  # p b a c
        [3, 0, 2, 1],  # b p a c
        [3, 2, 1, 0],  # b a c p
    ]
    counts = [1, 1, 1, 1]

    prof = Profile(ballots, counts)

    opponent_order = [3, 0, 2, 1]   # b p a c  (same code)
    preferred      = 0              # p

    ok, _ = algorithm1_single_voter(borda, prof, opponent_order, preferred)
    assert ok


# ---------------------------------------------------------------------
#                           Algorithm 2
# ---------------------------------------------------------------------

def test_alg2_zero_manipulators():
    ok, prof = algorithm2_coalitional(borda, [], [], "p", k=0)
    assert not ok and prof is None


def test_alg2_threshold_guard():
    team_profile: list[list[str]] = []
    opponent_order = ["a", "b", "c", "p"]
    ok, votes = algorithm2_coalitional(borda, team_profile, opponent_order, "p", k=2)
    assert not ok and votes is None


def test_alg2_simple_case_1():
    """CC-MaNego example with k=2 manipulators."""
    team_profile = [
        ["p", "d", "a", "b", "c", "e"],
        ["a", "p", "b", "c", "d", "e"],
        ["b", "c", "a", "p", "d", "e"],
    ]
    opponent_order = ["a", "p", "b", "c", "d", "e"]
    ok, votes = algorithm2_coalitional(make_x_approval(2), team_profile, opponent_order, "p", k=2)
    assert ok
    assert votes == [["p", "e", "d", "c", "b", "a"], ["p", "e", "d", "c", "b", "a"]]


def test_alg2_simple_case_2():
    """X-approval (X=1) example with k=2 manipulators."""
    team_profile = [
        ["a", "p", "b", "c"],
        ["b", "a", "c", "p"],
    ]
    opponent_order = ["p", "c", "b", "a"]
    ok, votes = algorithm2_coalitional(make_x_approval(1), team_profile, opponent_order, "p", k=2)
    assert ok
    assert votes == [["p", "c", "b", "a"], ["p", "c", "b", "a"]]


def test_alg2_random_threshold_guard():
    for _ in range(20):
        m = random.randint(3, 10)
        candidates = [chr(ord("a") + i) for i in range(m)]
        preferred = random.choice(candidates)

        pos = random.randint(math.ceil(m / 2), m - 1)
        others = [c for c in candidates if c != preferred]
        random.shuffle(others)
        opponent_order = others.copy()
        opponent_order.insert(pos, preferred)

        k = random.randint(1, min(4, len(candidates)))
        team_profile = [
            random.sample(candidates, k=m) for __ in range(random.randint(0, 5))
        ]

        ok, votes = algorithm2_coalitional(
            borda, team_profile, opponent_order, preferred, k
        )
        assert not ok and votes is None


def test_alg2_random_output_shape():
    for _ in range(20):
        m = random.randint(3, 7)
        candidates = [chr(ord("A") + i) for i in range(m)]
        preferred = random.choice(candidates)
        opponent_order = random.sample(candidates, k=m)
        k = random.randint(1, 3)
        team_profile = [
            random.sample(candidates, k=m) for __ in range(random.randint(0, 4))
        ]

        ok, votes = algorithm2_coalitional(
            borda, team_profile, opponent_order, preferred, k
        )
        if ok:
            assert isinstance(votes, list) and len(votes) == k
            for v in votes:
                assert sorted(v) == sorted(candidates)


def test_alg2_all_top_preferred():
    m = random.randint(3, 8)
    others = list("abcdefgh")[: m - 1]

    team_profile = [
        ["p"] + random.sample(others, k=len(others))
        for _ in range(random.randint(1, 5))
    ]
    opponent_order = ["p"] + random.sample(others, k=len(others))
    k = random.randint(2, 3)
    ok, votes = algorithm2_coalitional(borda, team_profile, opponent_order, "p", k)
    assert ok
    assert votes is not None
