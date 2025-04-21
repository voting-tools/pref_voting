"""
Tests for the `dodgson` voting method implementation.

Profiles and published winners
------------------------------
 * Brandt (2008, https://doi.org/10.1002/malq.200810017)
     • Table 1 (homogeneity paradox) – original profile P and tripled profile 3 P
     • Table 2 (monotonicity paradox) – original and modified profiles
     • Table 3 (Condorcet‑loser winner)
     • Table 4 (clone‑independence failure)
 * Bartholdi, Tovey & Trick (1989, https://doi.org/10.1007/BF00303169) – three‑voter example, p. 159 
 * Caragiannis et al. (2011, https://doi.org/10.1016/j.artint.2012.04.004) – three‑voter example, p. 34
 * Fishburn (1982, https://doi.org/10.1016/0166-218X(82)90070-1) – modified monotonicity profile
 * Nurmi (2004, Theory & Decision 57) – Condorcet‑loser winner
 * Wikipedia “Dodgson’s method” page (symmetrical 6‑voter cycle)

Candidate names in the original papers (A, B …) are mapped to integers:
A→0, B→1, C→2, D→3, etc.
"""

import pytest
from pref_voting.profiles import Profile
from pref_voting.other_methods import dodgson

 
# Brandt (2008)
_rankings_t1 = [
    (3,2,0,1), (1,2,0,3), (2,0,1,3), (3,1,2,0),
    (0,1,2,3), (0,3,1,2), (3,0,1,2)
]
_counts_t1 = [2,2,2,2,2,1,1]

brandt_t1_P  = Profile(_rankings_t1, _counts_t1)               # winner 0  (A)
brandt_t1_3P = Profile(_rankings_t1, [3*c for c in _counts_t1])# winner 3  (D)
 
brandt_t2_orig = Profile(    # Table 2 original (A wins) 
    [(2,0,3,1), (1,3,2,0), (0,1,3,2), (0,2,1,3), (1,0,2,3)],
    [15,9,9,5,5])

brandt_t2_mod  = Profile(    # Table 2 with A raised (C wins) 
    [(2,0,3,1), (1,3,2,0), (0,1,3,2), (0,2,1,3), (0,1,2,3)],
    [15,9,9,5,5])

brandt_t3      = Profile(    # Table 3 – Condorcet‑loser (D wins)
    [(3,0,1,2), (1,2,0,3), (2,0,1,3), (3,2,0,1)],
    [10,8,7,4])

brandt_t4_orig = Profile(    # Table 4 before clone (A wins) 
    [(0,1,2), (1,2,0), (2,0,1)],
    [5,4,3])

brandt_t4_clone = Profile(   # Table 4 with clone C′ (B wins) 
    [(0,1,2,3), (1,2,3,0), (2,3,0,1), (3,0,1,2)],
    [5,4,3,0])

 
# Other published examples
bartholdi_tovey_trick = Profile(   # Bartholdi‑Tovey‑Trick 1989, p. 159
    [(0,3,1,2), (1,2,0,3), (2,0,3,1)],
    [1,1,1])

caragiannis_2011 = Profile(        # Caragiannis 2011, p. 34
    [(0,1,2), (1,0,2), (0,2,1)],
    [1,1,1])

fishburn_1982 = Profile(       # Fishburn 1982, p. 132 - A wins
    [(2,0,3,1), (1,3,2,0), (0,1,3,2), (0,2,1,3), (1,0,2,3)],
    [15,9,9,5,5])

fishburn_1982_mod = Profile(       # Fishburn 1982, p. 132 - C wins
    [(2,0,3,1), (1,3,2,0), (0,1,3,2), (0,2,1,3), (1,0,2,3), (0,1,2,3)],
    [15,9,9,5,3,2])

nurmi_2004 = Profile(              # Nurmi 2004, p. 10 – D wins
    [(3,0,1,2), (1,2,0,3), (2,0,1,3), (3,2,0,1)],
    [10,8,7,4])

wiki_cycle = Profile(              # Wikipedia (6‑voter ABC cycle)
    [(0,1,2), (1,2,0), (2,0,1)],
    [2,2,2])

ALL_CASES = {
    # Brandt verified
    "Brandt‑T1‑P"      : (brandt_t1_P,        {0}),
    "Brandt‑T1‑3P"     : (brandt_t1_3P,       {3}),
    "Brandt‑T2‑orig"   : (brandt_t2_orig,     {0}),
    "Brandt‑T2‑mod"    : (brandt_t2_mod,      {2}),
    "Brandt‑T3"        : (brandt_t3,          {3}),
    "Brandt‑T4‑orig"   : (brandt_t4_orig,     {0}),
    "Brandt‑T4‑clone"  : (brandt_t4_clone,    {1}),

    # Other literature
    "Bartholdi‑Tovey‑Trick" : (bartholdi_tovey_trick, {0,2}),
    "Caragiannis 2011"      : (caragiannis_2011,      {0}),
    "Fishburn 1982"         : (fishburn_1982,          {0}),
    "Fishburn 1982‑mod"     : (fishburn_1982_mod,     {2}),
    "Nurmi 2004"            : (nurmi_2004,           {3}),
    "Wikipedia cycle"       : (wiki_cycle,           {0,1,2}),
}

@pytest.mark.parametrize("label, prof_expected", ALL_CASES.items())
def test_dodgson(label, prof_expected):
    """Compare solver output to published Dodgson winners (sets)."""
    profile, expected = prof_expected
    assert set(dodgson(profile)) == expected, f"Mismatch on {label}"