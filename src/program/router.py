"""
router.py — Argument-routing combinators for symbolic program evaluation.

Defines the argument-routing layer used by :class:`Program` to distribute
input arguments across the left and right subtrees of a program tree.  Each
router determines how a single incoming argument is split between two branches.

Routing combinators
-------------------
The four routing primitives correspond loosely to combinatory logic, but with
a custom semantics focused on splitting a list of arguments between two
subtrees (``left`` and ``right``) rather than function composition:

=======  ==============  ==============================================
Name     Function        Behaviour
=======  ==============  ==============================================
``B``    ``send_right``  Route argument to the **right** branch only.
``C``    ``send_left``   Route argument to the **left** branch only.
``S``    ``send_both``   Route argument to **both** branches (shared).
``K``    ``constant``    **Discard** argument (for constant sub-trees).
=======  ==============  ==============================================

The identity combinator ``I`` is defined as a :class:`Primitive` rather than
a :class:`Router` since it operates on the final argument list rather than
routing between two subtrees.

Composite routers
-----------------
A :class:`ComRouter` chains multiple single routers, one per argument, and is
used when a program takes more than one argument (e.g. ``note_count->note``
requires two routing decisions).
"""

import re
import itertools
import numpy as np
from typing import List
from pandas.core.common import flatten

from src.program.type import arrow, tint
from src.program.primitive import Primitive
from src.program.helpers import secure_list, copy_list


# ---------------------------------------------------------------------------
# Low-level routing functions
# ---------------------------------------------------------------------------


def send_right(arg_dict: dict, arg_list) -> dict:
    """Append ``arg_list`` to the right branch of ``arg_dict``.

    Args:
        arg_dict: Dict with keys ``"left"`` and ``"right"`` accumulating
            the argument lists for each subtree.
        arg_list: Argument(s) to route to the right subtree.

    Returns:
        Updated ``arg_dict`` with ``arg_list`` appended to ``arg_dict["right"]``.
    """
    arg_dict["right"] = arg_dict["right"] + secure_list(arg_list)
    return arg_dict


def send_left(arg_dict: dict, arg_list) -> dict:
    """Append ``arg_list`` to the left branch of ``arg_dict``.

    Args:
        arg_dict: Dict with keys ``"left"`` and ``"right"``.
        arg_list: Argument(s) to route to the left subtree.

    Returns:
        Updated ``arg_dict`` with ``arg_list`` appended to ``arg_dict["left"]``.
    """
    arg_dict["left"] = arg_dict["left"] + secure_list(arg_list)
    return arg_dict


def send_both(arg_dict: dict, arg_list) -> dict:
    """Append ``arg_list`` to **both** branches of ``arg_dict``.

    Used when the same argument must be consumed by both the left and right
    subtrees (e.g. the ``S`` combinator sharing an argument).

    Args:
        arg_dict: Dict with keys ``"left"`` and ``"right"``.
        arg_list: Argument(s) to broadcast to both subtrees.

    Returns:
        Updated ``arg_dict`` with ``arg_list`` appended to both branches.
    """
    arg_dict["left"] = list(flatten(arg_dict["left"] + [arg_list]))
    arg_dict["right"] = list(flatten(arg_dict["right"] + [arg_list]))
    return arg_dict


def constant(arg_dict: dict, _) -> dict:
    """Discard the incoming argument (constant / ``K`` combinator).

    Used for subtrees that take no inputs — the argument is ignored and
    ``arg_dict`` is returned unchanged.

    Args:
        arg_dict: Dict with keys ``"left"`` and ``"right"``.
        _: Ignored argument.

    Returns:
        ``arg_dict`` unchanged.
    """
    return arg_dict


def return_myself(arg_list):
    """Identity function: return the first element of ``arg_list``.

    Used by the ``I`` combinator primitive to pass a single argument through
    unchanged.

    Args:
        arg_list: A list or single value.

    Returns:
        ``arg_list[0]`` if ``arg_list`` is a list, otherwise ``arg_list``.
    """
    if isinstance(arg_list, list):
        return arg_list[0]
    else:
        return arg_list


# ---------------------------------------------------------------------------
# Router classes
# ---------------------------------------------------------------------------


class Router:
    """A single argument-routing primitive.

    Each router has a one-character name (``"B"``, ``"C"``, ``"S"``, ``"K"``)
    and a ``run`` function that updates an ``arg_dict`` by routing one argument
    to the left, right, or both subtrees.

    Attributes:
        ctype (str): Always ``"router"``.
        name (str): Single-character router symbol.
        n_arg (int): Number of characters in ``name`` (always 1 for a single router).
        run (callable): The routing function
            ``(arg_dict: dict, arg: Any) -> dict``.
    """

    ctype = "router"

    def __init__(self, name: str = None, func: callable = None):
        """Initialise a Router.

        Args:
            name: Single-character symbol (``"B"``, ``"C"``, ``"S"``, or ``"K"``).
            func: Routing function called as ``func(arg_dict, arg)``.
        """
        self.name = name
        self.n_arg = len(name) if name else 0
        self.run = func

    def __str__(self) -> str:
        return self.name if self.name else ""


class ComRouter:
    """A composite router built from a sequence of single :class:`Router` objects.

    Applied when a program has multiple arguments: each argument is routed
    by its corresponding single router.  For example, a program with type
    ``note_count->note`` requires two routing decisions — one for the note
    argument and one for the count argument.

    Attributes:
        ctype (str): Always ``"router"``.
        name (str): Concatenated names of all constituent routers (e.g. ``"BK"``).
        n_arg (int): Number of constituent routers (= number of arguments).
        routers (list): The ordered list of :class:`Router` objects.
    """

    ctype = "router"

    def __init__(self, router_list: List[Router]):
        """Initialise a ComRouter.

        Args:
            router_list: Ordered list of :class:`Router` objects, one per
                argument position.
        """
        self.name = "".join([router.name for router in router_list])
        self.n_arg = len(router_list)
        self.routers = router_list

    def run(self, arg_dict: dict, arg_list: list) -> dict:
        """Apply each constituent router to its corresponding argument.

        If ``arg_list`` is shorter than ``self.routers``, the missing
        positions are padded with ``None`` (those routing calls are no-ops).

        Args:
            arg_dict: Dict with keys ``"left"`` and ``"right"``.
            arg_list: List of arguments, one per router.

        Returns:
            Updated ``arg_dict`` after all routers have been applied.
        """
        # Pad arg_list with None if fewer args than routers
        if len(self.routers) != len(arg_list):
            num_none = len(self.routers) - len(arg_list)
            arg_list += [None] * num_none

        for i, router in enumerate(self.routers):
            if router.run:
                router.run(arg_dict, arg_list[i])

        return arg_dict

    def __str__(self) -> str:
        return self.name


# ---------------------------------------------------------------------------
# Singleton router instances
# ---------------------------------------------------------------------------

B = Router("B", send_right)  # route arg to right subtree only
C = Router("C", send_left)  # route arg to left subtree only
S = Router("S", send_both)  # broadcast arg to both subtrees
K = Router("K", constant)  # discard arg (constant sub-tree)

router_map = {"B": B, "C": C, "S": S, "K": K}

# Identity combinator: passes the single argument through unchanged
I = Primitive("I", arrow(tint, tint), ["note"], ["note"], return_myself)
