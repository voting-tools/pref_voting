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
from typing import Callable, List, Optional, Sequence, Set, Tuple

# ───────────────────────────── logging ───────────────────────────────────
# 4 types of logging:
#   1. INFO    – high-level algorithm steps.
#   2. DEBUG   – detailed steps.
#   3. WARNING – guards.
#   4. ERROR   – unexpected errors.

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Configure the default formatting
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

# Create and configure a console handler
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)

# Define a more detailed formatter for console output
formatter = logging.Formatter(
    "%(asctime)s: %(levelname)s: %(name)s: Line %(lineno)s: %(message)s"
)
console.setFormatter(formatter)

# Attach the console handler to our module‐level logger
logger.addHandler(console)

# ──────────────────── 0. built-in social-welfare rules ───────────────────
def borda(profile: List[List[str]]) -> List[str]:
    """
    Return a complete Borda ranking for string ballots.
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
            scores[cand] += m - pos - 1            # m-1 … 0 points
    return sorted(scores, key=lambda c: (-scores[c], c))    # tie-break lexicographically


def make_x_approval(x: int) -> Callable[[List[List[str]]], List[str]]:
    """
    Factory: x-approval rule (Plurality is x=1).
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
        scores = {c: 0 for c in profile[0]}
        for ballot in profile:
            for cand in ballot[:x]:
                scores[cand] += 1
        return sorted(scores, key=lambda c: (-scores[c], c))

    rule.__name__ = f"x_approval_{x}"
    return rule

# ───────────────────── 1. shared helper utilities ────────────────────────
def _pos(candidate: str, ranking: Sequence[str]) -> int:
    """
    Paper position: higher ranked ⇒ larger value.
    >>> _pos("A", ["C", "B", "A"])
    0
    """
    idx = ranking.index(candidate)
    result = len(ranking) - 1 - idx

    logger.debug(f"[_pos] candidate='{candidate}', index_in_ranking={idx}, returned_position={result}")
    return result

def _top_i(ranking: Sequence[str], i: int) -> List[str]:
    """
    Return the top-i candidates in the given ranking.
    >>> _top_i(["C", "B", "A"], 2)
    ['C', 'B']
    """
    return list(ranking)[:i]

# ──────────────────── Rational-Compromise helper ────────────────────────
def _rc_result(pt: Sequence[str], po: Sequence[str]) -> Optional[str]:
    """
    Rational-Compromise outcome for two parties.

    Prints a concise trace of the top-j intersections as it searches for the
    first *singleton* intersection.  Returns the winner (string) or None.
    >>> _rc_result(["C", "A", "B"], ["B", "C", "A"])
    'C'
    """
    m = len(pt)
    for j in range(1, m + 1):
        inter = set(_top_i(pt, j)) & set(_top_i(po, j))
        logger.debug(f"[RC] depth j={j}, pt_top={_top_i(pt, j)}, po_top={_top_i(po, j)}, intersection={sorted(inter)}")

        if inter:
            if len(inter) == 1:
                winner = next(iter(inter))
                logger.info(f"[RC] singleton intersection at j={j} ⇒ returning '{winner}'")
                return winner
            else:
                logger.info(f"[RC] intersection at j={j} is not singleton ({len(inter)} items) ⇒ returning None")
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
        if len(H) == i:
            break

        if c != preferred and c not in Ai_po:
            H.append(c)
            logger.debug(f"[_compute_Hi] adding '{c}' to H (size now {len(H)})")

    logger.debug(f"[_compute_Hi] final H_i = {H} for preferred='{preferred}' and i={i}")
    return H

def check_validation(opp: List[str], preferred: str, m: int) -> bool:
    """
    Check if the opponent ranks the preferred candidate high enough.
    If not, manipulation is impossible.
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
    F: Callable[[List[List[str]]], List[str]],   # social-welfare function
    team_profile : List[List[str]],              # honest team ballots
    opponent_order: List[str],                   # opponent ranking
    preferred    : str,                          # preferred candidate
) -> Tuple[bool, Optional[List[str]]]:
    """
    Single-voter manipulation (C-MaNego) with logging for trace.
    >>> algorithm1_single_voter(borda,[["b", "a", "p"]], ["b", "a", "p"], "p")
    (False, None)
    """
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
    team_profile : List[List[str]],
    opponent_order: List[str],
    preferred    : str,
    k            : int,
) -> Tuple[bool, Optional[List[List[str]]]]:
    """
    Decide whether a coalition of size *k* can make `preferred` the unique
    Rational-Compromise (Bucklin) winner against `opponent_order`.
    >>> algorithm2_coalitional(borda, [], ["a", "b", "c", "p"], "p", k=2)
    (False, None)
    """

    # ── 0  guards ─────────────────────────────────────────────────────────
    if k <= 0:
        logger.warning("[Alg-2] 0 manipulators ⇒ impossible")
        return False, None

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

        # ── 4  evaluate RC after the whole coalition is in ───────────────
        rc_outcome = _rc_result(F(team_profile + pm), opponent_order)
        logger.debug(f"[Alg-2] RC outcome with coalition = {rc_outcome}")

        if rc_outcome == preferred:
            logger.info("[Alg-2] ✅ success – coalition found")
            return True, pm

        logger.info("[Alg-2] ✘ depth failed – trying next i")

    logger.info("\n[Alg-2] ❌ no successful coalition manipulation found")
    return False, None


if __name__== "__main__":
    print(algorithm1_single_voter(borda,[["b", "a", "p"]], ["b", "a", "p"], "p"))