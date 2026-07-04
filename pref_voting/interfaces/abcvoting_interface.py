"""
File: abcvoting_interface.py
Author: Eric Pacuit (epacuit@umd.edu) and Wes Holliday (wesholliday@berkeley.edu)

Functions to convert between pref_voting grade profiles and abcvoting profiles
(https://abcvoting.readthedocs.io).

abcvoting is an optional dependency of pref_voting: it is only imported when one
of the functions in this module is called.  Install it with
``uv pip install "pref_voting[abcvoting]"``.

abcvoting profiles consist of approval ballots, i.e., dichotomous ballots.  The
corresponding pref_voting object is a :class:`~pref_voting.grade_profiles.GradeProfile`
with grades 0 and 1 (as produced, e.g., by
:meth:`~pref_voting.utility_profiles.UtilityProfile.to_approval_profile`).
A candidate is approved by a voter if the voter assigns the candidate the grade 1.

abcvoting identifies candidates with the integers ``0, ..., num_cand - 1``; the
conversion functions preserve candidate names via the ``cmap`` of the grade
profile and the ``cand_names`` of the abcvoting profile.  abcvoting voters carry
weights: the ``gcounts`` of a grade profile become voter weights, and integer
voter weights become ``gcounts`` (fractional weights raise a ``ValueError``
unless ``scale_weights=True`` is passed).
"""

from pref_voting.grade_profiles import GradeProfile
from pref_voting.interfaces._utils import weights_to_counts

ABCVOTING_IMPORT_ERROR = (
    "abcvoting is required for this function. Install it with: "
    'uv pip install "pref_voting[abcvoting]" (or pip install "pref_voting[abcvoting]").'
)


def _abcvoting_classes():
    try:
        from abcvoting.preferences import Profile as AbcvotingProfile, Voter
    except ImportError as err:
        raise ImportError(ABCVOTING_IMPORT_ERROR) from err
    return AbcvotingProfile, Voter


def to_abcvoting_profile(gprofile):
    """Convert a :class:`GradeProfile` with grades 0 and 1 to an abcvoting
    ``Profile``.

    The candidates ``0, ..., num_cand - 1`` of the abcvoting profile are the
    indices of the (sorted) candidates of the grade profile, and the candidate
    names of the abcvoting profile are given by the grade profile's ``cmap``.
    Each ballot type contributes one abcvoting voter, whose approval set is the
    set of candidates assigned the grade 1 and whose weight is the ballot count.
    Use ``abcvoting.preferences.Profile.convert_to_unit_weights()`` on the
    result if unit weights are needed.

    :param gprofile: A grade profile with grades 0 and 1.
    :type gprofile: GradeProfile
    :returns: An abcvoting ``Profile``.
    """

    AbcvotingProfile, Voter = _abcvoting_classes()

    if sorted(gprofile.grades) != [0, 1]:
        raise ValueError(
            "Only grade profiles with grades 0 and 1 (approval ballots) can be "
            "converted to an abcvoting profile. See, e.g., "
            "UtilityProfile.to_approval_profile and "
            "UtilityProfile.to_k_approval_profile for ways to produce such a "
            "profile."
        )

    cand_to_index = {c: i for i, c in enumerate(gprofile.candidates)}

    abc_profile = AbcvotingProfile(
        gprofile.num_cands,
        cand_names=[str(gprofile.cmap[c]) for c in gprofile.candidates],
    )
    for grade_fnc, count in zip(*gprofile.grades_counts):
        approved = [
            cand_to_index[c] for c in grade_fnc.graded_candidates if grade_fnc(c) == 1
        ]
        abc_profile.add_voter(Voter(approved, weight=int(count), num_cand=gprofile.num_cands))

    return abc_profile


def abcvoting_to_grade_profile(abc_profile, scale_weights=False):
    """Convert an abcvoting ``Profile`` to a :class:`GradeProfile` with grades
    0 and 1.

    The candidates of the grade profile are the integers
    ``0, ..., num_cand - 1``, with the ``cand_names`` of the abcvoting profile
    as the ``cmap``.  Each voter contributes one ballot assigning the grade 1
    to each approved candidate and the grade 0 to every other candidate, with
    the voter's weight as the ballot count.

    :param abc_profile: An abcvoting ``Profile``.
    :param scale_weights: If True, fractional voter weights are scaled to integers by clearing denominators; otherwise fractional weights raise a ``ValueError``.
    :type scale_weights: bool, optional
    :returns: A :class:`GradeProfile` with grades 0 and 1.
    """

    candidates = list(range(abc_profile.num_cand))

    grade_maps = []
    weights = []
    for voter in abc_profile:
        grade_maps.append(
            {c: 1 if c in voter.approved else 0 for c in candidates}
        )
        weights.append(voter.weight)

    gcounts = weights_to_counts(weights, scale_weights=scale_weights, what="voter")

    return GradeProfile(
        grade_maps,
        [0, 1],
        gcounts=gcounts,
        candidates=candidates,
        cmap={c: str(name) for c, name in enumerate(abc_profile.cand_names)},
    )
