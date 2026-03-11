"""
melody_primitive.py — Domain-specific primitives and symbolic terms for melody.

Defines the type system, primitive functions, symbolic argument classes, and
program representation used for melodic program induction.

Note representation
-------------------
Melodies are sequences of discrete notes drawn from a 6-value pitch class set
{1, 2, 3, 4, 5, 6}.  A ``Note`` object encodes a fixed note sequence as a
symbolic token with a canonical string name of the form ``note_1_3_5`` for
the array ``[1, 3, 5]``.  A ``Count`` object encodes a positive integer with
name ``count_3`` for the value 3.

Primitives
----------
Seven melody-specific primitive functions are provided:

=============  ====================================================
``memorize``   Identity: returns the stored note sequence.
``repeat``     Tile a sequence ``count`` times.
``reverse``    Append a reversed copy of a sequence to itself.
``up``         Append a copy shifted up by ``count`` (mod 6).
``down``       Append a copy shifted down by ``count`` (mod 6).
``ranges``     Ascending scale: append ``count`` +1-shifted copies.
``concatenate`` Concatenate two note sequences.
=============  ====================================================

Global symbol table
-------------------
``global_melody_pms`` is a dict mapping string names to symbolic objects
(Notes, Counts, primitives, and routers).  It acts as the global namespace
for program evaluation.  :func:`create_or_get_pm_from_cache` provides an
LRU-cached lookup that dynamically creates new ``Note``/``Count`` objects on
demand and caches them.
"""

import sys

sys.path.append("..")

import ast
import numpy as np
from functools import lru_cache

from src.domain.melody.melody_utils import single_range
from src.program.router import *
from src.program.helpers import *
from src.program.primitive import *
from src.program.type import tarray, tint


# ---------------------------------------------------------------------------
# Type aliases (melody-domain names for the underlying base types)
# ---------------------------------------------------------------------------

tnote = tarray  # note sequences are represented as arrays
tcount = tint  # count arguments are integers
ttime = tint  # time indices are also integers


# ---------------------------------------------------------------------------
# Task-specific argument classes
# ---------------------------------------------------------------------------


class Note(TaskSpecificPrimitive):
    """Symbolic token representing a fixed note sequence.

    A ``Note`` stores a numpy array of pitch values in {1,…,6} and exposes a
    canonical string name of the form ``note_v1_v2_…_vk`` (e.g.
    ``note_1_3_5`` for ``[1, 3, 5]``).  The name is used as a key in the
    global symbol table and in program term strings.
    """

    def __init__(self, name: str, tp: type = tnote, value=None):
        TaskSpecificPrimitive.__init__(self, name, tp, value)
        self.ctype = "note"
        self.value = self.string_to_array(name)

    def __str__(self) -> str:
        return self.name

    @staticmethod
    def string_to_array(s: str) -> np.ndarray:
        """Parse a canonical note name string into a numpy array.

        The name format is ``note_v1_v2_…``, where each ``vi`` is an integer
        or ``"inf"`` for a wildcard/infinity placeholder.

        Args:
            s: String of the form ``"note_1_3_5"`` or ``"note_2_inf_4"``.

        Returns:
            Numpy array of pitch values (dtype float to accommodate ``np.inf``).
        """
        # Split the string on underscores and remove the leading 'note' part
        parts = s.split("_")[1:]

        # Convert each part to an integer or np.inf
        converted_parts = []
        for part in parts:
            if part == "inf":
                converted_parts.append(np.inf)
            else:
                converted_parts.append(int(part))

        return np.array(converted_parts)

    @staticmethod
    def array_to_string(arr: np.ndarray) -> str:
        """Serialise a note array back to its canonical string name.

        Inverse of :meth:`string_to_array`.  Infinity values are serialised as
        ``"inf"``; finite float values are cast to ``int`` first.

        Args:
            arr: Array of pitch values (possibly containing ``np.inf``).

        Returns:
            String of the form ``"note_v1_v2_…"``.
        """
        elements = []
        for elem in arr:
            if np.isinf(elem):
                elements.append("inf")
            else:
                elements.append(str(int(elem)))
        return "note_" + "_".join(elements)


class Count(TaskSpecificPrimitive):
    """Symbolic token representing a positive integer count argument.

    Name format: ``count_k`` for integer value ``k`` (e.g. ``count_3``).

    Attributes:
        ctype (str): Always ``"count"``.
        value (int): Integer value parsed from ``name``.
    """

    def __init__(self, name: str, tp: type = tcount, value=None):
        TaskSpecificPrimitive.__init__(self, name, tp, value)
        self.ctype = "count"
        self.value = int(name.split("_")[1])


# ---------------------------------------------------------------------------
# Melody primitive functions
# ---------------------------------------------------------------------------

# Each primitive is defined with:
#   (name, type_signature, arg_types, ret_types, implementation_lambda)
# The lambda receives a list of pre-evaluated argument values.

memorize = Primitive(
    "memorize",
    arrow(tnote, tnote),
    ["note"],
    ["note"],
    lambda x: np.array(x)[0],  # identity: return the stored note sequence
)
repeat = Primitive(
    "repeat",
    arrow(tcount, arrow(tnote, tnote)),
    ["note", "count"],
    ["note"],
    lambda x: np.tile(x[0], x[1]),  # tile x[0] (notes) x[1] (count) times
)
reverse = Primitive(
    "reverse",
    arrow(tnote, tnote),
    ["note"],
    ["note"],
    # Append a reversed copy: [x[0][:-1], reversed(x[0])] drops the shared boundary note
    lambda x: np.concatenate([x[0][:-1], np.array(list(reversed(x[0])))], -1),
)
up = Primitive(
    "up",
    arrow(tcount, arrow(tnote, tnote)),
    ["note", "count"],
    ["note"],
    # Append a copy shifted up by x[1] semitones (mod 6, values in {1,...,6})
    lambda x: (np.concatenate([x[0], x[0] + x[1]], -1) - 1) % 6 + 1,
)
down = Primitive(
    "down",
    arrow(tcount, arrow(tnote, tnote)),
    ["note", "count"],
    ["note"],
    # Append a copy shifted down by x[1] semitones (mod 6, values in {1,...,6})
    lambda x: (np.concatenate([x[0], x[0] - x[1]], -1) - 1) % 6 + 1,
)
ranges = Primitive(
    "ranges",
    arrow(tcount, arrow(tnote, tnote)),
    ["note", "count"],
    ["note"],
    # Generate ascending scale: append x[1] copies each shifted +1 from the last
    lambda x: single_range(x[0], x[1]),
)
concatenate = Primitive(
    "concatenate",
    arrow(tnote, arrow(tnote, tnote)),
    ["note", "note"],
    ["note"],
    lambda x: np.append(x[0], x[1]),  # concatenate two note sequences
)

melody_primitive_list = [
    memorize,
    up,
    down,
    reverse,
    repeat,
    concatenate,
    ranges,
]

melody_primitive_name_list = [primitive.name for primitive in melody_primitive_list]

# ---------------------------------------------------------------------------
# Placeholder objects for typed program enumeration
# ---------------------------------------------------------------------------

# TODO: define any_type
# TODO: define collection of note and notes
note = Placeholder("note")
pause = Placeholder("pause")
count = Placeholder("count")


# ---------------------------------------------------------------------------
# Global symbol table
# ---------------------------------------------------------------------------

# Maps string token names → symbolic objects (Notes, Counts, primitives, routers).
# Used as the evaluation namespace when running MelodyProgram instances.
global_melody_pms = {}

# Note arguments: note_1 … note_6 (single-note sequences)
for i in range(1, 7):
    global_melody_pms[f"note_{i}"] = Note(f"note_{i}")
# Count arguments: count_1 … count_6
for i in range(1, 7):
    global_melody_pms[f"count_{i}"] = Count(f"count_{i}")

# Argument-routing combinators (B, C, S, K) and identity (I)
global_melody_pms["K"] = K
global_melody_pms["S"] = S
global_melody_pms["C"] = C
global_melody_pms["B"] = B
global_melody_pms["I"] = I

# Melody primitives
for name, pm in zip(melody_primitive_name_list, melody_primitive_list):
    global_melody_pms[name] = pm


@lru_cache(maxsize=10000)
def create_or_get_pm_from_cache(name: str):
    """Look up or create a symbolic object by its string name.

    Checks the global symbol table first; if not found, creates a new
    ``Note``, ``Count``, or ``ComRouter`` object and caches it for reuse.

    The LRU cache (size 10 000) ensures that dynamically-created objects (e.g.
    multi-note sequences like ``note_1_3_5``) are not re-allocated on every
    call.

    Args:
        name: Token name string, e.g. ``"note_1_3"``, ``"count_2"``,
            ``"memorize"``, or a router sequence like ``"BK"``.

    Returns:
        The corresponding symbolic object (Note, Count, Primitive, Router,
        or ComRouter).
    """
    if name in global_melody_pms:
        return global_melody_pms[name]
    elif name.startswith("note"):
        note = Note(name=name)
        global_melody_pms[name] = note
        return note
    elif name.startswith("count"):
        count = Count(name=name)
        global_melody_pms[name] = count
        return count
    elif name in melody_primitive_name_list:
        global_melody_pms[name] = name
        return name
    else:
        # Interpret the name as a sequence of single-character router symbols
        routers = list(name)
        routers = [router_map[router] for router in routers]
        com_router = ComRouter(routers)
        global_melody_pms[name] = com_router
        return com_router


# ---------------------------------------------------------------------------
# Program class
# ---------------------------------------------------------------------------


class MelodyProgram(Program):
    """String-based program representation for the melody domain.

    Extends :class:`Program` to support programs stored as nested bracket
    strings (e.g. ``"[K,memorize,note_1_3]"``), which is the format used in
    the grammar production tables and compressor DataFrames.

    :meth:`run` parses the string representation, replaces each token with
    its corresponding symbolic object via :func:`create_or_get_pm_from_cache`,
    and delegates execution to :meth:`Program.run`.

    Attributes:
        ctype (str): Always ``"program"``.
        terms (str): The program as a bracket-notation string.
    """

    ctype = "program"

    def __init__(self, terms: str):
        Program.__init__(self, terms)

    @staticmethod
    def quote_elements_in_string(nested_string: str) -> str:
        """Wrap unquoted identifiers in a nested bracket string with double quotes.

        Transforms e.g. ``"[K,memorize,note_1]"`` into
        ``'["K","memorize","note_1"]'`` so that :func:`ast.literal_eval` can
        parse it as a nested list of strings.

        Args:
            nested_string: Program term string using bracket notation.

        Returns:
            Version of ``nested_string`` where all bare identifiers are
            double-quoted.
        """
        quoted_string = re.sub(
            r'"(?<![\'\""])([a-zA-Z_][a-zA-Z_0-9]*)(?![\'\""])"', r'"\1"', nested_string
        )
        return quoted_string

    @staticmethod
    def extract_elements_from_string(nested_string: str) -> list:
        """Parse a bracket-notation program string into a nested Python list.

        Args:
            nested_string: Program string, e.g. ``"[K,[B,memorize],note_1]"``.

        Returns:
            Nested list of string token names.
        """
        quoted_string = MelodyProgram.quote_elements_in_string(nested_string)
        nested_list = ast.literal_eval(quoted_string)
        return nested_list

    @staticmethod
    def replace_terms(nested_list: list) -> list:
        """Recursively replace string tokens with their symbolic objects.

        Traverses the nested list produced by :meth:`extract_elements_from_string`
        and replaces every string token with the corresponding object from the
        global symbol table via :func:`create_or_get_pm_from_cache`.

        Args:
            nested_list: Nested list of string token names (or already
                converted objects for recursive calls).

        Returns:
            Nested list with the same structure but string tokens replaced by
            symbolic objects (Note, Count, Primitive, Router, etc.).
        """
        if isinstance(nested_list, list):
            return [MelodyProgram.replace_terms(item) for item in nested_list]
        else:
            return create_or_get_pm_from_cache(nested_list)

    def run(self, arg_list=None) -> np.ndarray:
        """Parse the program string and execute it, returning the output melody.

        Steps:
        1. Parse ``self.terms`` (bracket string) into a nested list of strings.
        2. Replace every string token with its symbolic object.
        3. Delegate to :meth:`Program.run` with the resolved term tree.

        Args:
            arg_list: Optional list of additional arguments passed at runtime
                (forwarded to :meth:`Program.run`).

        Returns:
            Numpy array representing the output melody produced by the program.
        """
        nested_list = self.extract_elements_from_string(self.terms)
        term_objects = self.replace_terms(nested_list)
        return Program.run(self, terms=term_objects)
