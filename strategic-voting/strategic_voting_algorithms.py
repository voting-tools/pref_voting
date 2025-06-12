"""
Implementation of algorithms from:
    "Strategic Voting in the Context of Negotiating Teams",
    Leora Schmerler & Noam Hazon (2021) – https://arxiv.org/abs/2107.14097

Programmer: Elyasaf Kopel
Last revised: 18 May 2025

The module provides two functions:

    algorithm1_single_voter ─ C-MaNego (single manipulator)
    algorithm2_coalitional ─ CC-MaNego (coalition of k manipulators)

Both decide whether a preferred outcome `p` can be made the unique
sub-game perfect equilibrium (SPE) of a VAOV negotiation game.

Changes compared with the paper
-------------------------------
* `check_validation` encapsulate the common “sanity checks.”
* `_rc_result` models the Rational-Compromise (Bucklin-style) outcome
  and returns ``None`` whenever the first intersection is not singleton;
  this is enough because a manipulator must guarantee *uniqueness*.

"""
from __future__ import annotations
import logging, math
from typing import Callable, List, Optional, Sequence, Set, Tuple, Union

try:
    # Only present if the user has pref_voting installed
    from pref_voting.profiles import Profile as Profile
except ImportError:          # tests or minimal env
    class Profile:        # fake placeholder so isinstance() works
        pass

# ───────────────────────────── logging ───────────────────────────────────
# 4 levels we actually care about in this module:
#   INFO    – high-level algorithm steps
#   DEBUG   – detailed steps
#   WARNING – guards
#   ERROR   – unexpected errors
logger = logging.getLogger(__name__)

def _setup_logging(detailed: bool = False) -> None:
    """
    Attach a single console handler to this module’s logger.

    Parameters
    ----------
    detailed : bool, optional
        If True, use a verbose timestamped format
        ("%Y-%m-%d %H:%M:%S ..."); otherwise, use level-only format.

    Returns
    -------
    None
    """
    fmt = (
        "%(asctime)s  %(levelname)-8s  %(name)s:%(lineno)d  %(message)s"
        if detailed
        else "%(levelname)s: %(message)s"
    )
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt))
    handler.setLevel(logging.INFO)        # tweak as you wish
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.propagate = False

# ────────────────────────────────────────────────────────────────
def _explode_profile(profile: Union[Profile, Sequence[Sequence[str]]]) -> List[List[str]]:
    """
   Normalize a Profile or raw ballots into an explicit list of ballots.

    Parameters
    ----------
    profile : Profile or Sequence of Sequence of str
        - If a Profile, each ranking is replicated by its count.
        - If a raw sequence, it is assumed to already be a list of ballots.

    Returns
    -------
    List[List[str]]
        Expanded list of ballots.

    • If already a List[List[str]] → return it unchanged (cheap pointer copy).
    • If a Profile → replicates each unique ranking according to its count vector;
      if `profile.counts` is missing or None, assume 1 voter per ranking.
    """
    if isinstance(profile, Profile):
        # try to grab counts; default to [1, 1, …] if missing or None
        raw_counts = getattr(profile, 'rcounts', None)
        counts = raw_counts if raw_counts is not None else [1] * len(profile.rankings)

        return [
            ballot
            for ballot, n in zip(profile.rankings, counts)
            for _ in range(n)
        ]

    # if it's already a sequence of ballots
    return [list(b) for b in profile]

# ──────────────────── 0. built-in social-welfare rules ───────────────────
def borda(profile: List[List[str]]) -> List[str]:
    """
   Compute the Borda count ranking for a list of ballots.

    Parameters
    ----------
    profile : List of List of str
        Each inner list is a ballot ranking candidates.

    Returns
    -------
    List[str]
        Candidates sorted from highest to lowest Borda score.

    >>> borda([["C", "A", "B"], ["B", "C", "A"], ["C", "A", "B"]])
    ['C', 'A', 'B']
    """
    if not profile:
        logger.warning("Empty profile in Borda count")
        return []

    m = len(profile[0])
    scores = {c: 0 for c in profile[0]}

    for ballot in profile:
        for pos, cand in enumerate(ballot):
            scores[cand] += m - pos - 1

    ranking = sorted(scores, key=lambda c: (-scores[c], c))

    # ▲ log the full score vector in a stable, readable order
    logger.info(
        "[Borda] scores → %s",
        [(c, scores[c]) for c in ranking]
    )

    return ranking


def make_x_approval(x: int) -> Callable[[List[List[str]]], List[str]]:
    """
    Factory: x-approval rule (Plurality is x=1).
    Create an x-approval social welfare function.

    Parameters
    ----------
    x : int
        Number of top positions on each ballot that earn one point.

    Returns
    -------
    Callable[[List[List[str]]], List[str]]
        A function that maps ballots to a full ranking.

    Raises
    ------
    ValueError
        If x is not a positive integer.

    >>> x_approval = make_x_approval(2)
    >>> x_approval([["C", "A", "B"], ["B", "C", "A"], ["C", "A", "B"]])
    ['C', 'A', 'B']
    """
    if x <= 0:
        logger.error("x must be a positive integer")
        raise ValueError("x must be a positive integer")

    def rule(profile: List[List[str]]) -> List[str]:
        if not profile:
            logger.warning("Empty profile in x-approval count")
            return []
        scores = {c: 0 for c in profile[0]} # initialize scores for all candidates to 0
        for ballot in profile:
            for cand in ballot[:x]:
                scores[cand] += 1 # increment score for each of the top x candidates
        return sorted(scores, key=lambda c: (-scores[c], c))

    rule.__name__ = f"x_approval_{x}"
    return rule

# ───────────────────── 1. shared helper utilities ────────────────────────
def _pos(candidate: str, ranking: Sequence[str]) -> int:
    """
    Compute paper‐style position of `candidate` in `ranking`.

    "the number of outcomes that o (candidate) is preferred over them in pi (ranking)."

    Parameters
    ----------
    candidate
        label of the candidate to look up.
    ranking
        a full ranking list with most‐to‐least preferred.

    Returns
    -------
    int
        position value (higher ⇒ better).

    >>> _pos("A", ["C", "B", "A"])
    0
    """
    idx = ranking.index(candidate)
    result = len(ranking) - 1 - idx

    logger.debug(f"[_pos] candidate='{candidate}', index_in_ranking={idx}, returned_position={result}")
    return result

def _top_i(ranking: Sequence[str], i: int) -> List[str]:
    """
    Return the first i candidates from a ranking.

    Parameters
    ----------
    ranking : Sequence of str
        Full ranking, most-to-least preferred.
    i : int
        Number of top candidates to select.

    Returns
    -------
    List[str]
        The top i candidates.

    >>> _top_i(["C", "B", "A"], 2)
    ['C', 'B']
    """
    return list(ranking)[:i]

# ──────────────────── Rational-Compromise helper ────────────────────────
def _rc_result(pt: Sequence[str], po: Sequence[str]) -> Optional[str]:
    """
    Compute the Rational-Compromise winner between two orderings.
    VAOV-style intersection of top-i candidates.
    Parameters
    ----------
    pt : Sequence of str
        Social welfare function ranking (team).
    po : Sequence of str
        Opponent’s ranking.

    Returns
    -------
    Optional[str]
        The singleton intersection winner, or None if none found.

    Prints a concise trace of the top-j intersections as it searches for the
    first *singleton* intersection.  Returns the winner (string) or None.
    >>> _rc_result(["C", "A", "B"], ["B", "C", "A"])
    'C'
    """
    m = len(pt)
    for j in range(1, m + 1):
        inter = set(_top_i(pt, j)) & set(_top_i(po, j)) # intersection of top-j
        logger.debug(f"[RC] depth j={j}, pt_top={_top_i(pt, j)}, po_top={_top_i(po, j)}, intersection={sorted(inter)}")

        if inter:
            if len(inter) == 1:
                winner = next(iter(inter)) # the set has exactly one element, get it
                logger.info(f"[RC] singleton intersection at j={j} ⇒ returning '{winner}'")
                return winner
            else:
                logger.info(
                    "[RC] intersection at j=%d is not singleton (%d items: %s) ⇒ returning None",
                    j, len(inter), sorted(inter)
                )
                return None

    logger.debug("[RC] no intersection found at any depth ⇒ returning None")
    return None

def _compute_Hi(
        preferred: str,
        i: int,
        pt: Sequence[str],
        opponent_order: Sequence[str],
) -> List[str]:
    """
    Build a depth-i candidate set Hᵢ.

    Parameters
    ----------
    preferred : str
        The candidate we aim to promote.
    i : int
        Depth parameter (1 ≤ i ≤ m).
    pt : Sequence of str
        Honest team’s SWF ranking.
    opponent_order : Sequence of str
        Opponent’s full ranking.

    Returns
    -------
    List[str]
        A list of up to i candidates: starts with `preferred`, then the next
        best in `pt` not in opponent’s top-i.

    Hᵢ = {preferred} ∪ (i-1 best in pt not in Aᵢ(po)), keeping pt order.
    Aᵢ(po) = top-i candidates in po (opponent) order.
    >>> _compute_Hi("A", 2, ["C", "A", "B"], ["B", "C", "A"])
    ['A']
    """
    Ai_po: Set[str] = set(_top_i(opponent_order, i))
    logger.debug(f"[_compute_Hi] i={i}, opponent_top_i={sorted(Ai_po)}")

    H: List[str] = [preferred]
    logger.debug(f"[_compute_Hi] initially H = {H}")

    for c in pt:
        if len(H) == i: # already have i candidates
            break

        if c != preferred and c not in Ai_po: # not in opponent's top-i
            H.append(c)
            logger.debug(f"[_compute_Hi] adding '{c}' to H (size now {len(H)})")

    logger.debug(f"[_compute_Hi] final H_i = {H} for preferred='{preferred}' and i={i}")
    return H

def check_validation(opp: List[str], preferred: str, m: int) -> bool:
    """
    Guard: ensure opponent ranks `preferred` high enough for manipulation, if not, manipulation is impossible.

    Parameters
    ----------
    opp : List of str
        Opponent’s ranking.
    preferred : str
        Candidate to check.
    m : int
        Total number of candidates.

    Returns
    -------
    bool
        True if `preferred` is ranked at or above ceil(m/2); otherwise False.

    >>> check_validation(["A", "B", "C"], "A", 3)
    True
    >>> check_validation(["C", "B", "A"], "A", 2)
    False
    """
    if not opp:
        logger.debug("[check_validation] opponent profile is empty ⇒ returning False")
        return False

    # Compute the paper‐style “position” (higher ⇒ better for the opponent)
    pos_val = _pos(preferred, opp)
    threshold = math.ceil(m / 2)

    logger.debug(f"[check_validation] preferred='{preferred}', "
                 f"pos_val={pos_val}, threshold={threshold}, m={m}")

    if pos_val < threshold:
        logger.warning("Preferred candidate ranked too low by opponent – manipulation impossible.")
        return False

    return True

# ─────────────── Algorithm 1 – single-voter manipulation ────────────────
def algorithm1_single_voter(
    F: Callable[[List[List[str]]], List[str]],                   # social-welfare function
    team_profile : Union[List[List[str]], Profile],              # honest team ballots
    opponent_order: Union[List[str],List[int]],                  # opponent ranking
    preferred    : Union[str,int],                               # preferred candidate
) -> Tuple[bool, Optional[List[str]]]:
    """
    Single-voter manipulation: find a ballot that makes `preferred` the unique RC winner.

    Parameters
    ----------
    F: Callable[[List[List[str]]], List[str]]
        Social welfare function.
    team_profile : List of List of str or Profile
        Honest team ballots or Profile object.
    opponent_order : List of int or List of str
        Opponent’s ranking.
    preferred : str or int
        Candidate to manipulate for.

    Returns
    -------
    Tuple[bool, Optional[List[str]]]
        (success, manipulative_ballot) or (False, None).

    >>> algorithm1_single_voter(borda,[["b", "a", "p"]], ["b", "a", "p"], "p")
    (False, None)
    >>> algorithm1_single_voter(borda,[["p", "c", "a", "b"],["p", "b", "a", "c"],["b", "p", "a", "c"],["b", "a", "c", "p"],], ["b", "p", "a", "c"], "p")
    (True, ['a', 'p', 'c', 'b'])
    """
    if not opponent_order or not team_profile:
        logger.warning("[Alg-1] empty team profile or opponent ranking – exit")
        return False, None

    team_profile = _explode_profile(team_profile)

    logger.info("\n[Alg-1] =========================================================")
    logger.info(f"[Alg-1] opponent order  : {opponent_order}")
    logger.info(f"[Alg-1] preferred       : '{preferred}'")
    logger.info(f"[Alg-1] team profile ({len(team_profile)} voters): {team_profile}")

    m = len(opponent_order)
    if not check_validation(opponent_order, preferred, m):
        logger.warning("[Alg-1] guard failed – manipulation impossible")
        return False, None

    # SWF order of the honest team
    pt = F(team_profile)
    logger.info(f"[Alg-1] SWF order (pt)  : {pt}")

    # iterate i = 1 … ⌈m/2⌉
    for i in range(1, math.ceil(m / 2) + 1):
        logger.info(f"\n[Alg-1] ----- depth i = {i} -----")

        Hi = _compute_Hi(preferred, i, pt, opponent_order)
        logger.info(f"[Alg-1] H_i           = {Hi}   (size {len(Hi)} vs required {i})")
        if len(Hi) < i:
            logger.info("[Alg-1] › not enough candidates – skip depth")
            continue

        # build manipulator ballot: high block then low block (both reversed)
        hi_block = list(reversed([c for c in pt if c in Hi]))
        lo_block = list(reversed([c for c in pt if c not in Hi]))
        pa       = hi_block + lo_block

        logger.info(f"[Alg-1] hi_block      = {hi_block}")
        logger.info(f"[Alg-1] lo_block      = {lo_block}")
        logger.info(f"[Alg-1] ballot (pa)   = {pa}")

        # test if ‘preferred’ becomes the unique RC winner
        rc = _rc_result(F(team_profile + [pa]), opponent_order)
        if rc == preferred:
            logger.info("[Alg-1] ✅ success – manipulation ballot found")
            return True, pa

        logger.info("[Alg-1] ✘ depth failed – trying next i")

    logger.info("[Alg-1] ❌ no successful manipulation found")
    return False, None

# ───────── Algorithm 2 – coalition of k manipulators (CC-MaNego) ─────────
def algorithm2_coalitional(
    F: Callable[[List[List[str]]], List[str]],
    team_profile : Union[List[List[str]], Profile],
    opponent_order: Union[List[str],List[int]],
    preferred    : Union[str,int],
    k            : int,
) -> Tuple[bool, Optional[List[List[str]]]]:
    """
   Coalition manipulation of size k: find ballots that make `preferred` the unique RC winner.

    Parameters
    ----------
    F: Callable[[List[List[str]]], List[str]]
        Social welfare function.
    team_profile : List of List of str or Profile
        Honest team ballots or Profile object.
    opponent_order : List of int or List of str
        Opponent’s ranking.
    preferred : str or int
        Candidate to manipulate for.
    k : int
        Number of manipulators.

    Returns
    -------
    Tuple[bool, Optional[List[List[str]]]]
        (success, list_of_ballots) or (False, None).

    >>> algorithm2_coalitional(borda, [], ["a", "b", "c", "p"], "p", k=2)
    (False, None)
    >>> algorithm2_coalitional(borda, [["p", "d", "a", "b", "c", "e"],["a", "p", "b", "c", "d", "e"],["b", "c", "a", "p", "d", "e"],], ["a", "p", "b", "c", "d", "e"], "p", k=2)
    (True, [['p', 'e', 'd', 'c', 'b', 'a'], ['p', 'e', 'd', 'c', 'b', 'a']])
    """
    if k <= 0:
        logger.warning("[Alg-2] k=0 manipulators – impossible by definition")
        return False, None
    if not opponent_order:
        logger.warning("[Alg-2] empty opponent ranking – exit")
        return False, None

    team_profile = _explode_profile(team_profile)

    logger.info("\n[Alg-2] =========================================================")
    logger.info(f"[Alg-2] opponent order  : {opponent_order}")
    logger.info(f"[Alg-2] preferred       : '{preferred}'")
    logger.info(f"[Alg-2] team profile ({len(team_profile)} voters): {team_profile}")
    logger.info(f"[Alg-2] coalition size k = {k}")
    # ── 0  guards ─────────────────────────────────────────────────────────

    m = len(opponent_order)
    if not check_validation(opponent_order, preferred, m):
        logger.warning("[Alg-2] opponent ranks 'p' too low ⇒ manipulation impossible")
        return False, None

    # ── SWF order of the honest team ───────────────────────────────────
    pt = F(team_profile)
    logger.info("\n[Alg-2] =========================================================")
    logger.info(f"[Alg-2] SWF order before coalition: {pt}")

    # ── iterate depths i = 1 … ⌈m/2⌉ ──────────────────────────────────
    for i in range(1, math.ceil(m / 2) + 1):
        logger.info(f"\n[Alg-2] ===== depth i = {i} =====")

        Hi = _compute_Hi(preferred, i, pt, opponent_order)
        logger.info(f"[Alg-2] H_i = {Hi}   (size {len(Hi)} vs required {i})")
        if len(Hi) < i:
            logger.info("[Alg-2] › not enough candidates – skip depth")
            continue

        pm: List[List[str]] = []

        # ── construct ballots l = 1 … k with the *same* Hᵢ ────────────
        for l in range(1, k + 1):
            cur_pt = F(team_profile + pm)
            logger.info(f"[Alg-2] manipulator #{l} sees SWF: {cur_pt}")

            # top block (Hᵢ) – place the least-preferred first (rev order)
            hi_block = list(reversed([c for c in cur_pt if c in Hi]))

            # bottom block (O\Hᵢ) – place the most-preferred first
            lo_block = list(reversed([c for c in cur_pt if c not in Hi]))

            pa = hi_block + lo_block
            pm.append(pa)

            logger.info(f"[Alg-2] ballot   = {pa}")
            logger.info(f"[Alg-2] hi_block = {hi_block}")
            logger.info(f"[Alg-2] lo_block = {lo_block}")

        # ── 4 evaluate RC after the whole coalition is in ───────────────
        rc_outcome = _rc_result(F(team_profile + pm), opponent_order)
        logger.debug(f"[Alg-2] RC outcome with coalition = {rc_outcome}")

        if rc_outcome == preferred:
            logger.info("[Alg-2] ✅ success – coalition found")
            return True, pm

        logger.info("[Alg-2] ✘ depth failed – trying next i")

    logger.info("\n[Alg-2] ❌ no successful coalition manipulation found")
    return False, None


# ───────────────────────────── demo harness ──────────────────────────────
def main() -> None:
    """
    Demo harness for C-MaNego (single manipulator) and CC-MaNego (coalition).

    This function sets up logging, runs two example profiles
    through `algorithm1_single_voter` and `algorithm2_coalitional`,
    and prints the outcomes via the module logger.
    """
    _setup_logging(detailed=False)
    # ######################################
    # 1.  Algorithm 1 – single manipulator #
    # ######################################
    logger.info("\n===== DEMO 1-A: C-MaNego (single voter) =====")

    team_profile_1 = [
        ["p", "c", "a", "b"],
        ["p", "b", "a", "c"],
        ["b", "p", "a", "c"],
        ["b", "a", "c", "p"],
    ]
    opponent_order_1 = ["b", "p", "a", "c"]
    preferred_1 = "p"

    ok_1, ballot_1 = algorithm1_single_voter(
        borda,
        team_profile_1,
        opponent_order_1,
        preferred_1,
    )
    logger.info(f"Outcome               : {ok_1}")
    if ok_1:
        logger.info(f"Manipulative ballot   : {ballot_1}")
    else:
        logger.info("Manipulation impossible under this profile.")
    # ------------------------------------------------------------------
    # 1-B.  SAME EXAMPLE, Profile BUILT *AS IN* pref_voting DOCS (integers)
    # ------------------------------------------------------------------
    logger.info("\n===== DEMO 1-B: C-MaNego with canonical integer Profile =====")

    ballots = [
        (0, 1, 2, 3),  # 0>1>2>3   (p>c>a>b)
        (0, 3, 2, 1),  # 0>3>2>1   (p>b>a>c)
        (3, 0, 2, 1),  # 3>0>2>1   (b>p>a>c)
        (3, 2, 1, 0),  # 3>2>1>0   (b>a>c>p)
    ]
    counts = [1, 1, 1, 1]  # one voter per ranking

    profile = Profile(ballots, rcounts=counts)

    opponent_order = [3, 0, 2, 1]  # b p a c   (IDs)
    preferred = 0  # p

    ok, ballot = algorithm1_single_voter(
        borda,
        profile,
        opponent_order,
        preferred,
    )

    logger.info(f"Outcome               : {ok}")
    logger.info(f"Manipulative ballot   : {ballot}" if ok
                else "Manipulation impossible under this profile.")
    # ######################################################
    # 2.  Algorithm 2 – coalition of k manipulators (k = 2) #
    # ######################################################
    logger.info("\n===== DEMO 2: CC-MaNego (coalition, k = 2) =====")

    team_profile_2 = [
        ["p", "d", "a", "b", "c", "e"],
        ["a", "p", "b", "c", "d", "e"],
        ["b", "c", "a", "p", "d", "e"],
    ]
    opponent_order_2 = ["a", "p", "b", "c", "d", "e"]
    preferred_2 = "p"
    k = 2

    logger.info(f"Honest team profile   : {team_profile_2}")
    logger.info(f"Opponent ranking      : {opponent_order_2}")
    logger.info(f"Preferred candidate   : '{preferred_2}', coalition size k = {k}")

    ok_2, ballots_2 = algorithm2_coalitional(
        borda,
        team_profile_2,
        opponent_order_2,
        preferred_2,
        k,
    )
    logger.info(f"Outcome               : {ok_2}")
    if ok_2:
        for idx, b in enumerate(ballots_2, 1):
            logger.info(f"Manipulator #{idx} ballot : {b}")
    else:
        logger.info("Coalitional manipulation impossible under this profile.")


# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()
