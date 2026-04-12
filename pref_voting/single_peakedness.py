"""
    File: single_peakedness.py
    Author: Wes Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)
    Date: April 11, 2026

    Functions to analyze single-peakedness of preference profiles.

    A preference profile is *single-peaked* with respect to an axis (linear order
    over the candidates) if every voter has a unique most-preferred candidate (peak)
    and the voter's preferences decrease monotonically in both directions from the
    peak along the axis.

    A profile is *k-maverick single-peaked* with respect to an axis if all but *k*
    voters are single-peaked with respect to that axis.  The minimum *k* over all
    axes measures how far the profile is from being single-peaked.

    Reference for k-maverick single-peakedness:
        Faliszewski, Hemaspaandra & Hemaspaandra (2014), "The complexity of
        manipulative attacks in nearly single-peaked electorates", *Artificial
        Intelligence* 207, 69-99.
        https://doi.org/10.1016/j.artint.2013.11.004

    Handling of ties in rankings
    ----------------------------
    Rankings with ties (weak orders) can be handled in four ways, controlled by
    the ``tied_ranking_handling`` parameter:

    - ``'maverick'`` (default): Rankings with ties are always counted as mavericks.
    - ``'possibly_sp'``: A weak order is possibly single-peaked with respect to the axis if there exists some way to
      break all ties that yields a linear order that is single-peaked with respect to the axis.  This is the most
      permissive notion.
      Reference: Lackner (AAAI 2014); Fitzsimmons & Lackner (JAIR 2020),
      "Incomplete Preferences in Single-Peaked Electorates", *Journal of
      Artificial Intelligence Research* 67, 797-833. https://doi.org/10.1613/jair.1.11577
    - ``'single_plateaued'``: A weak order is single-plateaued with respect to
      the axis if the top indifference class forms a contiguous interval on the
      axis (the "plateau") and preferences strictly worsen moving away from the
      plateau on each side.  Ties across opposite sides of the plateau are
      allowed (two candidates on opposite slopes may have the same rank), but
      same-side ties below the plateau are not.
      Reference: Berga & Moreno (2009), "Strategic requirements with indifference:
      single-peaked versus single-plateaued preferences", *Social Choice and
      Welfare* 32(2), 275-298.
    - ``'black_sp'``: A weak order is Black-single-peaked with respect to the
      axis if it has a unique peak and preferences strictly worsen moving away
      from the peak on each side.  Ties across opposite sides of the peak are
      allowed, but same-side ties below the peak are not.  Equivalently, this
      is single-plateauedness with a plateau of size 1.
      Reference: Black (1948), "On the Rationale of Group Decision-making",
      *Journal of Political Economy* 56(1), 23-34.  For the weak-order
      formulation used here, see also Fitzsimmons & Lackner (JAIR 2020,
      Section 6).

    The hierarchy among these notions (for individual voters) is:
        Black SP ⊊ Single-plateaued ⊊ Possibly SP.
    See Fitzsimmons & Lackner (JAIR 2020, Section 6) for a detailed comparison.

    Handling of truncated rankings
    ------------------------------
    Truncated linear orders (where some candidates are unranked) are handled by
    requiring that the ranked candidates form a contiguous segment of the axis and
    that the ranking is single-peaked on that sub-axis.  This corresponds to the
    treatment of "top orders" (weak orders where all unranked candidates are tied
    at the bottom) in Lackner (AAAI 2014) and Fitzsimmons & Lackner (JAIR 2020,
    Sections 2 and 5).

    For truncated weak orders (both truncated and with ties among ranked candidates),
    the contiguity check is applied first, then the appropriate SP notion is checked
    on the sub-axis of ranked candidates.
"""

from itertools import permutations
from pref_voting.profiles import Profile
from pref_voting.profiles_with_ties import ProfileWithTies
from pref_voting.rankings import Ranking


def is_single_peaked(ranking, axis, num_cands=None,
                     treat_truncated_as_maverick=False,
                     tied_ranking_handling='maverick'):
    """
    Check if a ranking is single-peaked with respect to the given axis.

    This function accepts either a list (a linear ranking, as used with
    :class:`Profile`) or a :class:`Ranking` object (as used with
    :class:`ProfileWithTies`), and automatically handles complete, truncated,
    and tied rankings.

    Args:
        ranking (list or Ranking): The voter's ranking.  If a list, it should
            contain candidates from most preferred to least preferred (a linear
            order, possibly truncated).  If a :class:`Ranking` object, ties and
            truncation are detected automatically.
        axis (list or tuple): Candidates in the left-to-right axis order.
        num_cands (int or None): Total number of candidates in the election.
            Required when ``ranking`` is a :class:`Ranking` object, to detect
            whether the ranking is truncated.  Ignored when ``ranking`` is a list
            (in which case ``len(axis)`` is used).
        treat_truncated_as_maverick (bool): If True, voters who don't rank all
            candidates are counted as mavericks.  If False (default), truncated
            rankings are checked for compatibility with single-peakedness
            (ranked candidates must form a contiguous segment of the axis and
            be single-peaked on that segment).
        tied_ranking_handling (str): How to handle rankings with ties (only
            relevant when ``ranking`` is a :class:`Ranking` object with ties).
            One of ``'maverick'`` (default), ``'possibly_sp'``,
            ``'single_plateaued'``, ``'black_sp'``.
            See module docstring for details.

    Returns:
        bool: True if the ranking is single-peaked-compatible with respect to
        the axis.

    Examples:

    With a list (linear ranking from a :class:`Profile`):

    .. code-block:: python

        >>> from pref_voting.single_peakedness import is_single_peaked

        >>> is_single_peaked([1, 0, 2], [0, 1, 2])
        True
        >>> is_single_peaked([0, 2, 1], [0, 1, 2])
        False

    With a :class:`Ranking` object (from a :class:`ProfileWithTies`):

    .. code-block:: python

        >>> from pref_voting.rankings import Ranking
        >>> from pref_voting.single_peakedness import is_single_peaked

        >>> # Linear ranking: 0 > 1 > 2
        >>> is_single_peaked(Ranking({0: 1, 1: 2, 2: 3}), [0, 1, 2], num_cands=3)
        True
        >>> # Weak order with tie: {0, 1} > 2
        >>> is_single_peaked(Ranking({0: 1, 1: 1, 2: 2}), [0, 1, 2], num_cands=3,
        ...                  tied_ranking_handling='possibly_sp')
        True
    """
    axis = _validate_axis(axis)
    axis_set = set(axis)

    if isinstance(ranking, Ranking):
        ranked_set = set(ranking.rmap.keys())
        if not ranked_set.issubset(axis_set):
            raise ValueError("ranking contains candidates not on the axis")
        if num_cands is None:
            num_cands = len(axis)
        elif num_cands != len(axis):
            raise ValueError("num_cands must equal len(axis)")
        return _is_ranking_sp(ranking, num_cands, axis,
                              treat_truncated_as_maverick, tied_ranking_handling)

    # List input: linear order (possibly truncated)
    ranking = list(ranking)
    if len(ranking) != len(set(ranking)):
        raise ValueError("ranking must not contain duplicates")
    if not set(ranking).issubset(axis_set):
        raise ValueError("ranking contains candidates not on the axis")
    if len(ranking) == len(axis):
        return _is_linear_sp(ranking, axis)
    if treat_truncated_as_maverick:
        return False
    return _is_truncated_linear_sp(ranking, axis)

def num_mavericks(profile, axis, treat_truncated_as_maverick=False,
                  tied_ranking_handling='maverick'):
    """
    Count the number of voters whose rankings are NOT single-peaked with
    respect to the given axis.

    Args:
        profile (Profile or ProfileWithTies): The preference profile.
        axis (list or tuple): Candidates in the left-to-right axis order.
        treat_truncated_as_maverick (bool): If True, voters who don't rank all
            candidates are counted as mavericks.  If False (default), truncated
            rankings are checked for compatibility with single-peakedness
            (ranked candidates must form a contiguous segment of the axis and
            be single-peaked on that segment).
        tied_ranking_handling (str): How to handle rankings with ties.
            One of ``'maverick'`` (default), ``'possibly_sp'``,
            ``'single_plateaued'``, ``'black_sp'``.
            See module docstring for details.

    Returns:
        int: The number of maverick voters.

    Example:

    .. code-block:: python

        from pref_voting.profiles import Profile
        from pref_voting.single_peakedness import num_mavericks

        prof = Profile([[0, 1, 2], [1, 0, 2], [1, 2, 0], [2, 1, 0], [0, 2, 1]])
        # 0 > 2 > 1 is not single-peaked on axis [0, 1, 2]
        num_mavericks(prof, [0, 1, 2])  # returns 1
    """
    axis = _validate_axis(axis)
    num_cands = len(profile.candidates)
    if set(axis) != set(profile.candidates):
        raise ValueError("axis must contain exactly the profile candidates")

    # Use anonymize() to deduplicate rankings
    if isinstance(profile, Profile):
        anon = profile.to_profile_with_ties().anonymize()
    else:
        anon = profile.anonymize()

    maverick_count = 0
    for ranking, count in zip(anon._rankings, anon.rcounts):
        if not _is_ranking_sp(ranking, num_cands, axis,
                              treat_truncated_as_maverick, tied_ranking_handling):
            maverick_count += int(count)
    return maverick_count

def min_k_maverick_single_peaked(profile, treat_truncated_as_maverick=False,
                                 tied_ranking_handling='maverick'):
    """
    Find the minimum *k* such that the profile is *k*-maverick single-peaked
    with respect to some ordering of the candidates.

    This function iterates over all possible axes (permutations of candidates)
    and returns the axis that minimizes the number of maverick voters.  An axis
    and its reverse are equivalent for single-peakedness, so only half of the
    permutations are checked.

    Suitable for small numbers of candidates (up to about 8).

    Args:
        profile (Profile or ProfileWithTies): The preference profile.
        treat_truncated_as_maverick (bool): If True, voters who don't rank all
            candidates are counted as mavericks.  If False (default), truncated
            rankings are checked for compatibility.
        tied_ranking_handling (str): How to handle rankings with ties.
            One of ``'maverick'`` (default), ``'possibly_sp'``,
            ``'single_plateaued'``, ``'black_sp'``.
            See module docstring for details.

    Returns:
        tuple: ``(min_k, best_axis)`` where ``min_k`` (int) is the minimum
        number of mavericks and ``best_axis`` (list) is an axis achieving it.

    Example:

    .. code-block:: python

        from pref_voting.profiles import Profile
        from pref_voting.single_peakedness import min_k_maverick_single_peaked

        prof = Profile([[0, 1, 2], [1, 0, 2], [1, 2, 0], [2, 1, 0], [0, 2, 1]])
        min_k, best_axis = min_k_maverick_single_peaked(prof)
        # min_k = 1, best_axis = [0, 1, 2]

    References:
        Faliszewski, Hemaspaandra & Hemaspaandra (2014), "The complexity of
        manipulative attacks in nearly single-peaked electorates", *Artificial
        Intelligence* 207, 69-99.
        https://doi.org/10.1016/j.artint.2013.11.004
    """
    candidates = profile.candidates
    m = len(candidates)
    if m <= 1:
        return 0, list(candidates)

    num_cands = m

    # Use anonymize() to deduplicate rankings
    if isinstance(profile, Profile):
        anon = profile.to_profile_with_ties().anonymize()
    else:
        anon = profile.anonymize()

    rankings_and_counts = list(zip(anon._rankings, anon.rcounts))

    best_k = profile.num_voters
    best_axis = list(candidates)

    # Symmetry: axis and its reverse are equivalent for single-peakedness.
    # We only consider permutations where candidates[0] is in the first half;
    # when m is odd and candidates[0] is at the center, break the tie by
    # requiring axis[0] < axis[-1].
    first = candidates[0]
    half = m // 2

    for perm in permutations(candidates):
        axis = list(perm)
        pos = axis.index(first)
        if pos > half or (pos == half and m % 2 == 0):
            continue
        if pos == half and axis[0] > axis[-1]:
            continue

        k = sum(int(c) for r, c in rankings_and_counts
                if not _is_ranking_sp(r, num_cands, axis,
                                      treat_truncated_as_maverick,
                                      tied_ranking_handling))

        if k < best_k:
            best_k = k
            best_axis = list(axis)
            if best_k == 0:
                break

    return best_k, best_axis


# =============================================================================
# Internal helpers
# =============================================================================

def _validate_axis(axis):
    """Validate that the axis is a duplicate-free list of candidates."""
    axis = list(axis)
    if len(axis) != len(set(axis)):
        raise ValueError("axis must not contain duplicates")
    return axis

def _is_linear_sp(ranking, axis):
    """Check if a complete linear ranking is single-peaked w.r.t. the axis.

    Uses the recursive characterization: the bottom-ranked candidate must be at
    one of the two extremes of the axis, and recursively for the rest.
    """
    if len(ranking) <= 2:
        return True
    bottom = ranking[-1]
    if bottom == axis[0]:
        return _is_linear_sp(ranking[:-1], axis[1:])
    elif bottom == axis[-1]:
        return _is_linear_sp(ranking[:-1], axis[:-1])
    else:
        return False


def _is_truncated_linear_sp(ranked_cands, axis):
    """Check if a truncated linear ranking is SP-compatible w.r.t. the axis.

    Requires that the ranked candidates form a contiguous segment of the axis
    and that the ranking is single-peaked on that sub-axis.
    """
    if len(ranked_cands) <= 1:
        return True

    ranked_set = set(ranked_cands)
    positions = [i for i, c in enumerate(axis) if c in ranked_set]

    if len(positions) != positions[-1] - positions[0] + 1:
        return False

    sub_axis = axis[positions[0]:positions[-1] + 1]
    return _is_linear_sp(ranked_cands, sub_axis)


def _is_ranking_sp(ranking, num_cands, axis, treat_truncated_as_maverick,
                   tied_ranking_handling):
    """Check if a pref_voting Ranking object is SP-compatible w.r.t. an axis."""
    is_complete = ranking.num_ranked_candidates() == num_cands

    if ranking.has_tie():
        if tied_ranking_handling == 'maverick':
            return False
        indiff_classes = [sorted(ranking.cands_at_rank(r))
                         for r in ranking.ranks]
        if is_complete:
            return _is_weak_order_sp(indiff_classes, axis,
                                     tied_ranking_handling)
        if treat_truncated_as_maverick:
            return False
        return _is_truncated_weak_order_sp(indiff_classes, axis,
                                           tied_ranking_handling)

    sorted_cands = sorted(ranking.rmap.keys(), key=lambda c: ranking.rmap[c])
    if is_complete:
        return _is_linear_sp(sorted_cands, axis)
    if treat_truncated_as_maverick:
        return False
    return _is_truncated_linear_sp(sorted_cands, axis)


def _is_weak_order_sp(indiff_classes, axis, method):
    """Dispatch to the appropriate weak-order SP check."""
    if method == 'possibly_sp':
        return _is_possibly_sp_weak_order(indiff_classes, axis)
    elif method == 'single_plateaued':
        return _is_single_plateaued_weak_order(indiff_classes, axis)
    elif method == 'black_sp':
        return _is_black_sp_weak_order(indiff_classes, axis)
    else:
        raise ValueError(
            f"Unknown tied_ranking_handling: {method!r}. "
            f"Expected 'possibly_sp', 'single_plateaued', or 'black_sp'."
        )


def _is_truncated_weak_order_sp(indiff_classes, axis, method):
    """Check if a truncated weak order is SP-compatible w.r.t. the axis.

    First checks contiguity, then applies the appropriate SP check on the
    sub-axis.
    """
    all_ranked = set()
    for ic in indiff_classes:
        all_ranked.update(ic)

    if len(all_ranked) <= 1:
        return True

    positions = [i for i, c in enumerate(axis) if c in all_ranked]

    if len(positions) != positions[-1] - positions[0] + 1:
        return False

    sub_axis = axis[positions[0]:positions[-1] + 1]
    return _is_weak_order_sp(indiff_classes, sub_axis, method)


def _is_possibly_sp_weak_order(indiff_classes, axis):
    """
    Check if a weak order is possibly single-peaked w.r.t. the given axis.

    A weak order (given as indifference classes from most to least preferred)
    is possibly single-peaked if there exists a linear extension (tie-breaking)
    that is single-peaked.  This is checked by a bottom-up peeling algorithm:
    starting from the least preferred class, each class's candidates must be
    removable from the left and/or right extremes of the remaining axis.

    References:
        Lackner (AAAI 2014), "Incomplete Preferences in Single-Peaked
        Electorates".

        Fitzsimmons & Lackner (JAIR 2020), "Incomplete Preferences in
        Single-Peaked Electorates", *Journal of Artificial Intelligence
        Research* 67, 797-833.
    """
    current_axis = list(axis)

    # Process classes from bottom (least preferred) to top (most preferred)
    for ic in reversed(indiff_classes):
        ic_set = set(ic)
        # Peel from left
        while current_axis and current_axis[0] in ic_set:
            ic_set.discard(current_axis[0])
            current_axis.pop(0)
        # Peel from right
        while current_axis and current_axis[-1] in ic_set:
            ic_set.discard(current_axis[-1])
            current_axis.pop()
        if ic_set:
            # Some candidates in this class are stuck in the interior
            return False
    return True


def _is_single_plateaued_weak_order(indiff_classes, axis):
    """
    Check if a weak order is single-plateaued w.r.t. the given axis.

    The top indifference class must form a contiguous interval on the axis
    (the "plateau"), and preferences must strictly worsen moving away from
    the plateau on each side.  Ties across opposite sides of the plateau are
    permitted; same-side ties below the plateau are not.

    References:
        Berga & Moreno (2009), "Strategic requirements with indifference:
        single-peaked versus single-plateaued preferences", *Social Choice and
        Welfare* 32(2), 275-298.

        Fitzsimmons & Lackner (JAIR 2020), "Incomplete Preferences in
        Single-Peaked Electorates", Section 6.
    """
    if not indiff_classes:
        return True

    top_set = set(indiff_classes[0])
    positions = [i for i, c in enumerate(axis) if c in top_set]

    if len(positions) != len(top_set):
        return False  # not all top candidates found on axis
    if positions[-1] - positions[0] + 1 != len(positions):
        return False  # not contiguous

    rank_of = {}
    for rank_idx, ic in enumerate(indiff_classes):
        for c in ic:
            rank_of[c] = rank_idx

    # Left of plateau: moving away from plateau, ranks must strictly worsen.
    prev_rank = 0
    for i in range(positions[0] - 1, -1, -1):
        r = rank_of[axis[i]]
        if r <= prev_rank:
            return False
        prev_rank = r

    # Right of plateau: moving away from plateau, ranks must strictly worsen.
    prev_rank = 0
    for i in range(positions[-1] + 1, len(axis)):
        r = rank_of[axis[i]]
        if r <= prev_rank:
            return False
        prev_rank = r

    return True


def _is_black_sp_weak_order(indiff_classes, axis):
    """
    Check if a weak order is Black single-peaked w.r.t. the given axis.

    Equivalent to single-plateauedness with a plateau of size 1: there must
    be a unique peak, preferences strictly worsen moving away from the peak
    on each side, and cross-side ties are permitted.

    References:
        Black (1948), "On the Rationale of Group Decision-making", *Journal of
        Political Economy* 56(1), 23-34.

        Fitzsimmons & Lackner (JAIR 2020), "Incomplete Preferences in
        Single-Peaked Electorates", Section 6.
    """
    if not indiff_classes:
        return True
    if len(indiff_classes[0]) != 1:
        return False
    return _is_single_plateaued_weak_order(indiff_classes, axis)
