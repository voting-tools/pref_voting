"""
File: _utils.py
Author: Eric Pacuit (epacuit@umd.edu) and Wes Holliday (wesholliday@berkeley.edu)

Helper functions shared by the interface modules.
"""

from fractions import Fraction
from math import lcm


def weights_to_counts(weights, scale_weights=False, what="ballot"):
    """Convert a list of ballot weights to a list of integer counts.

    Integer-valued weights (including floats such as ``2.0`` and integer-valued
    :class:`~fractions.Fraction` objects) are converted directly.  If any weight is
    not integer-valued, a ``ValueError`` is raised unless ``scale_weights`` is True,
    in which case all weights are multiplied by the least common multiple of their
    denominators.  This scaling is exact and does not change the relative weights
    of the ballots, but it does change the total number of voters.

    Non-integer float weights are interpreted via their decimal representation
    (e.g., ``0.1`` is treated as ``1/10``).

    :param weights: List of weights (ints, floats, or Fractions).
    :param scale_weights: If True, scale non-integer weights to integers by clearing denominators.
    :type scale_weights: bool, optional
    :param what: Name of the weighted objects, used in error messages.
    :type what: str, optional
    :returns: List of non-negative integers.
    """

    fracs = []
    for w in weights:
        if isinstance(w, int):
            f = Fraction(w)
        elif isinstance(w, Fraction):
            f = w
        elif isinstance(w, float):
            f = Fraction(int(w)) if w.is_integer() else Fraction(str(w))
        else:
            try:
                f = Fraction(w)
            except (TypeError, ValueError) as err:
                raise TypeError(
                    f"Cannot interpret {what} weight {w!r} as a number."
                ) from err
        if f < 0:
            raise ValueError(f"Negative {what} weight {w!r} is not allowed.")
        fracs.append(f)

    if all(f.denominator == 1 for f in fracs):
        return [f.numerator for f in fracs]

    if not scale_weights:
        raise ValueError(
            f"The profile contains non-integer {what} weights, but pref_voting "
            "profiles use integer counts. Pass scale_weights=True to multiply all "
            "weights by a common denominator (this changes the total number of "
            "voters but not the relative weights of the ballots)."
        )

    mult = lcm(*[f.denominator for f in fracs])
    return [int(f * mult) for f in fracs]
