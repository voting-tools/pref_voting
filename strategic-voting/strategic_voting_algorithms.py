from __future__ import annotations
import logging, math
from typing import Callable, List, Optional, Sequence, Set, Tuple

# ───────────────────────────── logging ───────────────────────────────────
_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

# ──────────────────── 0. built-in social-welfare rules ───────────────────
def borda(profile: List[List[str]]) -> List[str]:
    """
    Return a complete Borda ranking for string ballots.
    >>> borda([["C", "A", "B"], ["B", "C", "A"], ["C", "A", "B"]])
    ['C', 'A', 'B']
    """
    if not profile:
        return []
    m = len(profile[0])
    scores = {c: 0 for c in profile[0]}
    for ballot in profile:
        for pos, cand in enumerate(ballot):
            scores[cand] += m - pos - 1            # m-1 … 0 points
    return sorted(scores, key=lambda c: (-scores[c], c))    # tie-break lexicographically


def make_x_approval(x: int) -> Callable[[List[List[str]]], List[str]]:
    """Factory: x-approval rule (Plurality is x=1)."""
    if x <= 0:
        raise ValueError("x must be a positive integer")

    def rule(profile: List[List[str]]) -> List[str]:
        if not profile:
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
    """Paper position: higher ranked ⇒ larger value."""
    return len(ranking) - 1 - ranking.index(candidate)

def _top_i(ranking: Sequence[str], i: int) -> List[str]:
    return list(ranking)[:i]

# ──────────────────── Rational-Compromise helper ────────────────────────
def _rc_result(pt: Sequence[str], po: Sequence[str]) -> Optional[str]:
    """
    Rational-Compromise outcome for two parties.

    Prints a concise trace of the top-j intersections as it searches for the
    first *singleton* intersection.  Returns the winner (string) or None.
    """
    m = len(pt)
    for j in range(1, m + 1):
        inter = set(_top_i(pt, j)) & set(_top_i(po, j))
        _log.debug(f"[RC]  depth j={j:<2}  intersection = {sorted(inter)}")
        if inter:
            return next(iter(inter)) if len(inter) == 1 else None
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
    """
    Ai_po: Set[str] = set(_top_i(opponent_order, i)) # top-i in opponent order
    H: List[str] = [preferred] # preferred is always in Hᵢ

    for c in pt:
        if len(H) == i:
            break
        if c != preferred and c not in Ai_po:
            H.append(c)
    return H

def check_validation(opp: List[str], preferred: str, m: int) -> bool:
    if not opp:
        return False
    if _pos(preferred, opp) < math.ceil(m / 2):
        _log.warning("Preferred candidate ranked too low by opponent – manipulation impossible.")
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
    """
    _log.info("\n[Alg-1] =========================================================")
    _log.info(f"[Alg-1] opponent order  : {opponent_order}")
    _log.info(f"[Alg-1] preferred       : '{preferred}'")
    _log.info(f"[Alg-1] team profile ({len(team_profile)} voters): {team_profile}")

    m = len(opponent_order)
    if not check_validation(opponent_order, preferred, m):
        _log.warning("[Alg-1] guard failed – manipulation impossible")
        return False, None

    # SWF order of the honest team
    pt = F(team_profile)
    _log.info(f"[Alg-1] SWF order (pt)  : {pt}")

    # iterate i = 1 … ⌈m/2⌉
    for i in range(1, math.ceil(m / 2) + 1):
        _log.info(f"\n[Alg-1] ----- depth i = {i} -----")

        Hi = _compute_Hi(preferred, i, pt, opponent_order)
        _log.info(f"[Alg-1] H_i           = {Hi}   (size {len(Hi)} vs required {i})")
        if len(Hi) < i:
            _log.info("[Alg-1] › not enough candidates – skip depth")
            continue

        # build manipulator ballot: high block then low block (both reversed)
        hi_block = list(reversed([c for c in pt if c in Hi]))
        lo_block = list(reversed([c for c in pt if c not in Hi]))
        pa       = hi_block + lo_block

        _log.info(f"[Alg-1] hi_block      = {hi_block}")
        _log.info(f"[Alg-1] lo_block      = {lo_block}")
        _log.info(f"[Alg-1] ballot (pa)   = {pa}")

        # test if ‘preferred’ becomes the unique RC winner
        rc = _rc_result(F(team_profile + [pa]), opponent_order)
        if rc == preferred:
            _log.info("[Alg-1] ✅ success – manipulation ballot found")
            return True, pa

        _log.info("[Alg-1] ✘ depth failed – trying next i")

    _log.info("[Alg-1] ❌ no successful manipulation found")
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
    """

    # ── 0  guards ─────────────────────────────────────────────────────────
    if k <= 0:
        _log.warning("[Alg-2] 0 manipulators ⇒ impossible")
        return False, None

    m = len(opponent_order)
    if not check_validation(opponent_order, preferred, m):
        _log.warning("[Alg-2] opponent ranks 'p' too low ⇒ manipulation impossible")
        return False, None

    # ── SWF order of the honest team ───────────────────────────────────
    pt = F(team_profile)
    _log.info("\n[Alg-2] =========================================================")
    _log.info(f"[Alg-2] SWF order before coalition: {pt}")

    # ── iterate depths i = 1 … ⌈m/2⌉ ──────────────────────────────────
    for i in range(1, math.ceil(m / 2) + 1):
        _log.info(f"\n[Alg-2] ===== depth i = {i} =====")

        Hi = _compute_Hi(preferred, i, pt, opponent_order)
        _log.info(f"[Alg-2] H_i = {Hi}   (size {len(Hi)} vs required {i})")
        if len(Hi) < i:
            _log.info("[Alg-2] › not enough candidates – skip depth")
            continue

        pm: List[List[str]] = []

        # ── construct ballots l = 1 … k with the *same* Hᵢ ────────────
        for l in range(1, k + 1):
            cur_pt = F(team_profile + pm)
            _log.info(f"[Alg-2] manipulator #{l} sees SWF: {cur_pt}")

            # top block (Hᵢ) – place the least-preferred first (rev order)
            hi_block = list(reversed([c for c in cur_pt if c in Hi]))

            # bottom block (O\Hᵢ) – place the most-preferred first
            lo_block = list(reversed([c for c in cur_pt if c not in Hi]))

            pa = hi_block + lo_block
            pm.append(pa)

            _log.info(f"[Alg-2] ballot   = {pa}")
            _log.info(f"[Alg-2] hi_block = {hi_block}")
            _log.info(f"[Alg-2] lo_block = {lo_block}")

        # ── 4  evaluate RC after the whole coalition is in ───────────────
        rc_outcome = _rc_result(F(team_profile + pm), opponent_order)
        _log.debug(f"[Alg-2] RC outcome with coalition = {rc_outcome}")

        if rc_outcome == preferred:
            _log.info("[Alg-2] ✅ success – coalition found")
            return True, pm

        _log.info("[Alg-2] ✘ depth failed – trying next i")

    _log.info("\n[Alg-2] ❌ no successful coalition manipulation found")
    return False, None