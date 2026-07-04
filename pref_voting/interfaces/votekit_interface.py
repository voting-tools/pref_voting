"""
File: votekit_interface.py
Author: Eric Pacuit (epacuit@umd.edu) and Wes Holliday (wesholliday@berkeley.edu)

Functions to convert between pref_voting profiles and VoteKit profiles
(https://votekit.readthedocs.io).

VoteKit is an optional dependency of pref_voting: it is only imported when one of the functions in this module is called.  Install it with
``uv pip install "pref_voting [votekit]"`` (VoteKit requires Python >= 3.11).

Ranked ballots: a :class:`~pref_voting.profiles.Profile` or
:class:`~pref_voting.profiles_with_ties.ProfileWithTies` corresponds to a VoteKit
``PreferenceProfile`` of ranked ballots (rankings are tuples of frozensets of
candidate names; ties and truncation are preserved).

Score ballots: a :class:`~pref_voting.grade_profiles.GradeProfile` with numeric
grades corresponds to a VoteKit ``PreferenceProfile`` of score ballots.

Weights: VoteKit ballots carry (possibly fractional) weights, while pref_voting
profiles use integer counts.  Integer-valued weights are converted directly;
fractional weights raise a ``ValueError`` unless ``scale_weights=True`` is
passed, in which case all weights are multiplied by a common denominator.
"""

from pref_voting.grade_profiles import GradeProfile
from pref_voting.interfaces._utils import weights_to_counts
from pref_voting.profiles import Profile
from pref_voting.profiles_with_ties import ProfileWithTies

VOTEKIT_IMPORT_ERROR = (
    "votekit is required for this function. Install it with: "
    'uv pip install "pref_voting[votekit]" (or pip install "pref_voting[votekit]"). '
    "Note that votekit requires Python >= 3.11."
)


def _votekit_classes():
    try:
        from votekit.ballot import Ballot
        from votekit.pref_profile import PreferenceProfile
    except ImportError as err:
        raise ImportError(VOTEKIT_IMPORT_ERROR) from err
    return Ballot, PreferenceProfile


def _candidate_names(profile):
    """Return the candidate names (via the profile's cmap) and the map from
    names back to candidates, checking that the names are unique."""

    names = {c: str(profile.cmap[c]) for c in profile.candidates}
    assert len(set(names.values())) == len(names), (
        "The candidate names in the cmap must be unique to convert to a VoteKit profile."
    )
    return names


def to_votekit_profile(profile):
    """Convert a :class:`Profile` or :class:`ProfileWithTies` to a VoteKit
    ``PreferenceProfile`` of ranked ballots.

    Candidates are named using the profile's ``cmap``.  For a
    :class:`ProfileWithTies`, tied candidates are placed in the same position of
    the VoteKit ranking, unranked candidates are omitted from the ballot, and
    gaps in the ranks are compressed (only the relative order of the ranks
    matters in a :class:`ProfileWithTies`).

    :param profile: The profile to convert.
    :type profile: Profile or ProfileWithTies
    :returns: A VoteKit ``PreferenceProfile`` of ranked ballots.
    """

    Ballot, PreferenceProfile = _votekit_classes()

    names = _candidate_names(profile)

    ballots = []
    if isinstance(profile, Profile):
        rankings, rcounts = profile.rankings_counts
        for ranking, count in zip(rankings, rcounts):
            vk_ranking = tuple(frozenset({names[int(c)]}) for c in ranking)
            ballots.append(Ballot(ranking=vk_ranking, weight=int(count)))
    elif isinstance(profile, ProfileWithTies):
        rmaps, rcounts = profile.rankings_as_dicts_counts
        for rmap, count in zip(rmaps, rcounts):
            ranks = sorted(set(rmap.values()))
            vk_ranking = tuple(
                frozenset({names[c] for c in rmap.keys() if rmap[c] == rank})
                for rank in ranks
            )
            ballots.append(Ballot(ranking=vk_ranking, weight=int(count)))
    else:
        raise TypeError(
            "to_votekit_profile expects a Profile or a ProfileWithTies; use "
            "grade_profile_to_votekit for a GradeProfile."
        )

    return PreferenceProfile(
        ballots=tuple(ballots),
        candidates=tuple(names[c] for c in profile.candidates),
        max_ranking_length=len(profile.candidates),
    )


def votekit_to_profile_with_ties(vk_profile, scale_weights=False):
    """Convert a VoteKit ``PreferenceProfile`` of ranked ballots to a
    :class:`ProfileWithTies`.

    The candidates of the resulting profile are the candidate names (strings)
    of the VoteKit profile.  Candidates tied in a position of a VoteKit ranking
    are assigned the same rank; candidates missing from a ballot are unranked;
    skipped positions (empty frozensets) are preserved as gaps in the ranks.

    :param vk_profile: A VoteKit ``PreferenceProfile`` of ranked ballots.
    :param scale_weights: If True, fractional ballot weights are scaled to integers by clearing denominators; otherwise fractional weights raise a ``ValueError``.
    :type scale_weights: bool, optional
    :returns: A :class:`ProfileWithTies`.
    """

    rmaps = []
    weights = []
    for ballot in vk_profile.ballots:
        if getattr(ballot, "ranking", None) is None:
            raise ValueError(
                "The VoteKit profile contains ballots without a ranking (e.g., "
                "score ballots); use votekit_to_grade_profile instead."
            )
        rmap = {}
        for pos, cands_at_pos in enumerate(ballot.ranking):
            for c in cands_at_pos:
                rmap[str(c)] = pos + 1
        rmaps.append(rmap)
        weights.append(ballot.weight)

    rcounts = weights_to_counts(weights, scale_weights=scale_weights)

    return ProfileWithTies(
        rmaps,
        rcounts=rcounts,
        candidates=[str(c) for c in vk_profile.candidates],
    )


def grade_profile_to_votekit(gprofile):
    """Convert a :class:`GradeProfile` with numeric grades to a VoteKit
    ``PreferenceProfile`` of score ballots.

    Candidates are named using the profile's ``cmap``.

    .. warning::

        VoteKit does not store scores of 0: a candidate assigned the grade 0
        becomes an unscored candidate in the VoteKit profile.  In particular,
        for approval ballots (grades 0 and 1), the explicit 0s are not preserved
        when converting to VoteKit and back.

    :param gprofile: A grade profile with numeric grades.
    :type gprofile: GradeProfile
    :returns: A VoteKit ``PreferenceProfile`` of score ballots.
    """

    Ballot, PreferenceProfile = _votekit_classes()

    if not gprofile.can_sum_grades:
        raise ValueError(
            "Only grade profiles with numeric grades can be converted to VoteKit "
            "score ballots."
        )

    names = _candidate_names(gprofile)

    ballots = []
    for grade_fnc, count in zip(*gprofile.grades_counts):
        scores = {names[c]: grade_fnc(c) for c in grade_fnc.graded_candidates}
        ballots.append(Ballot(scores=scores, weight=int(count)))

    return PreferenceProfile(
        ballots=tuple(ballots),
        candidates=tuple(names[c] for c in gprofile.candidates),
    )


def votekit_to_grade_profile(vk_profile, grades=None, scale_weights=False):
    """Convert a VoteKit ``PreferenceProfile`` of score ballots to a
    :class:`GradeProfile`.

    The candidates of the resulting profile are the candidate names (strings)
    of the VoteKit profile.  Since a VoteKit profile does not carry a grade
    scale, the grades of the resulting profile default to the set of score
    values that appear in the profile; pass ``grades`` to specify the intended
    scale instead.  Integer-valued scores (VoteKit stores scores as floats) are
    converted to ints.  Candidates without a score on a ballot are ungraded on
    that ballot.

    :param vk_profile: A VoteKit ``PreferenceProfile`` of score ballots.
    :param grades: The grades of the resulting profile.  If provided, every score in the profile must be one of these grades.
    :type grades: list, optional
    :param scale_weights: If True, fractional ballot weights are scaled to integers by clearing denominators; otherwise fractional weights raise a ``ValueError``.
    :type scale_weights: bool, optional
    :returns: A :class:`GradeProfile`.
    """

    def _as_grade(score):
        return int(score) if isinstance(score, float) and score.is_integer() else score

    grade_maps = []
    weights = []
    for ballot in vk_profile.ballots:
        if getattr(ballot, "scores", None) is None:
            raise ValueError(
                "The VoteKit profile contains ballots without scores (e.g., "
                "ranked ballots); use votekit_to_profile_with_ties instead."
            )
        grade_maps.append({str(c): _as_grade(s) for c, s in ballot.scores.items()})
        weights.append(ballot.weight)

    gcounts = weights_to_counts(weights, scale_weights=scale_weights)

    observed_grades = sorted(set([g for gmap in grade_maps for g in gmap.values()]))
    if grades is None:
        grades = observed_grades
    else:
        assert all(g in grades for g in observed_grades), (
            f"All the scores in the VoteKit profile {observed_grades} must be "
            f"included in the grades {grades}."
        )

    return GradeProfile(
        grade_maps,
        grades,
        gcounts=gcounts,
        candidates=[str(c) for c in vk_profile.candidates],
    )
