"""Numerical representation of geometric objects in Newclid."""

ATOM = 1e-7
REL_TOL = 0.001


def close_enough(a: float, b: float) -> bool:
    diff = abs(a - b)
    if diff < 4 * ATOM:
        return True
    max_val = max(abs(a), abs(b))
    if max_val < 4 * ATOM:
        return True
    return diff / max_val < REL_TOL


def nearly_zero(a: float) -> bool:
    return abs(a) < 2 * ATOM


def sign(a: float) -> int:
    return 0 if nearly_zero(a) else (1 if a > 0 else -1)
