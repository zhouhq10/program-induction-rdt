"""
primitive.py — Core symbolic computation classes.

Defines the type hierarchy for all symbolic terms used in program induction:

* :class:`Placeholder` — base class for all named symbolic terms; provides
  type-string utilities for the grammar's production tables.

* :class:`Primitive` — a named function with a fixed type signature and a
  Python implementation (``func`` lambda).

* :class:`PM` — a typed program placeholder used during frame enumeration to
  represent ``unfilled`` argument slots.

* :class:`Program` — a recursive evaluator that interprets nested lists of
  primitives, routers, and task-specific arguments.

* :class:`TaskSpecificPrimitive` — base class for task-specific constant
  arguments (e.g. :class:`Note`, :class:`Count` in the melody domain).

Program representation
----------------------
Programs are represented as nested Python lists::

    [router, left_subtree, right_subtree]   # internal node
    [primitive, arg1, arg2, ...]            # leaf with arguments
    [I]                                     # identity

:meth:`Program.run` evaluates these trees recursively by dispatching on the
type of ``terms[0]`` (list, int/ndarray, router, or primitive).
"""

import re
import numpy as np
from typing import List, Tuple
from pandas.core.common import flatten

from src.program.helpers import secure_list, copy_list, extract_names
from src.program.type import *


# ---------------------------------------------------------------------------
# Placeholder (base for all symbolic terms)
# ---------------------------------------------------------------------------


class Placeholder:
    """Base class for all named symbolic terms in the grammar.

    Provides a common ``name`` attribute and static utilities for converting
    between type lists and the underscore-separated type-string format used
    in grammar production tables (e.g. ``"note_count->note"``).

    Attributes:
        ctype (str): Always ``"placeholder"`` for this base class; overridden
            by subclasses to ``"primitive"``, ``"router"``, etc.
        name (str): The symbolic name of this term.
    """

    ctype = "placeholder"

    def __init__(self, name: str):
        self.name = name

    def __str__(self) -> str:
        return f"{self.name}"

    @property
    def type_string(self) -> str:
        return f"{self.arg_type}->{self.ret_type}"

    @staticmethod
    def typelist_to_string(type_list: List[str]) -> str:
        """Join a list of type names into an underscore-separated string.

        Args:
            type_list: List of type name strings, e.g. ``["note", "count"]``.

        Returns:
            Underscore-joined string, e.g. ``"note_count"``.  Returns ``""``
            for an empty list.
        """
        arg_strings = ""
        if len(type_list) == 0:
            return arg_strings
        else:
            arg_strings += f"{type_list[0]}"
            if len(type_list) == 1:
                return arg_strings
            else:
                for j in range(1, len(type_list)):
                    arg_strings += f"_{type_list[j]}"
                return arg_strings

    @staticmethod
    def string_to_typelist(type_string: str) -> Tuple[List[str], str]:
        """Parse a type-signature string into ``([input_types], output_type)``.

        Args:
            type_string: String of the form ``"note_count->note"`` or
                ``"note->note"``.

        Returns:
            A ``([input_types], output_type)`` tuple, e.g.
            ``(["note", "count"], "note")``.

        Raises:
            ValueError: If ``"->"`` is absent from ``type_string``.
        """
        if "->" not in type_string:
            raise ValueError("Invalid format, '->' missing from the string.")

        input_part, output_part = type_string.split("->")
        input_types = input_part.split("_")

        return [input_types, output_part]

    @staticmethod
    def complete_typelist_to_string(
        arg_type_list: List[str], ret_type_list: str
    ) -> str:
        """Construct a full type-signature string from argument and return types.

        Args:
            arg_type_list: List of input type names, e.g. ``["note", "count"]``.
            ret_type_list: Return type name, e.g. ``"note"``.

        Returns:
            Type-signature string, e.g. ``"note_count->note"``.
        """
        if not arg_type_list:
            return f"->{ret_type_list}"

        type_string = arg_type_list[0]
        for arg_type_list in arg_type_list[1:]:
            type_string += f"_{arg_type_list}"
        type_string += f"->{ret_type_list}"

        return type_string


# ---------------------------------------------------------------------------
# Primitive (named function with type signature)
# ---------------------------------------------------------------------------


class Primitive(Placeholder):
    """A named function with a fixed type signature and Python implementation.

    Primitives are the leaf computation nodes of a program tree.  They have a
    known arity (``n_arg``), a type signature, and an executable ``run``
    lambda.

    Attributes:
        ctype (str): Always ``"primitive"``.
        name (str): Unique identifier, e.g. ``"memorize"``, ``"repeat"``.
        tp: Full type object (e.g. ``arrow(tnote, tnote)``).
        arg_type (str): Underscore-joined input type string.
        ret_type (str): Return type string.
        n_arg (int): Number of arguments the primitive expects.
        run (callable): Implementation lambda; called as ``run(values_list)``.
    """

    ctype = "primitive"

    def __init__(
        self,
        name: str = None,
        tp: Type = None,
        arg_type_list: List[str] = None,
        ret_type_list: List[str] = None,
        func: callable = None,
    ):
        self.name = name
        self.tp = tp
        self.arg_type = Placeholder.typelist_to_string(arg_type_list)
        self.ret_type = Placeholder.typelist_to_string(ret_type_list)
        self.n_arg = len(list(flatten([arg_type_list])))
        self.run = func

    @property
    def isPrimitive(self) -> bool:
        return True

    def __eq__(self, o) -> bool:
        return isinstance(o, Primitive) and o.name == self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __str__(self) -> str:
        return f"{self.name} {self.arg_type} -> {self.ret_type}"

    def clone(self):
        return Primitive(self.name, self.tp, self.value)

    def evaluate(self, environment):
        return self.value

    def inferType(self, context, environment, freeVariables):
        # TODO: implement type inference
        return self.tp.instantiate(context)


# ---------------------------------------------------------------------------
# PM (typed program placeholder for frame enumeration)
# ---------------------------------------------------------------------------


class PM(Placeholder):
    """Program placeholder used during frame enumeration.

    Represents an unfilled argument slot in a program frame — a position that
    must be instantiated with a concrete program before evaluation.  The type
    signature determines which programs are valid instantiations.

    Attributes:
        arg_type (str): Input types of the placeholder program.
        ret_type (str): Return type of the placeholder program.

    Args:
        type_sig: Type signature string, e.g. ``"note_time->note"``.
        name: Symbolic name (default ``"pgm"``).
    """

    def __init__(self, type_sig: str, name: str = "pgm"):
        Placeholder.__init__(self, name)
        types = type_sig.split("->")
        self.arg_type = types[0]
        self.ret_type = types[1]

    def __str__(self) -> str:
        return f"{self.name} {self.arg_type} -> {self.ret_type}"


# ---------------------------------------------------------------------------
# Program (recursive evaluator)
# ---------------------------------------------------------------------------


class Program:
    """Recursive evaluator for nested-list program representations.

    A :class:`Program` wraps a nested list of symbolic objects (primitives,
    routers, and task-specific arguments) and evaluates it by dispatching on
    the type of the root element.

    Tree structure
    --------------
    Programs are represented as nested Python lists:

    * ``[router, left_subtree, right_subtree]`` — an internal node that
      routes arguments to two sub-programs.
    * ``[primitive, arg1, arg2, …]`` — a leaf that applies a primitive to
      its pre-bound arguments (plus any runtime ``arg_list``).
    * ``[I]`` — the identity primitive, applied to ``arg_list`` directly.
    * ``[[left], [right]]`` — two independent sub-programs whose outputs are
      concatenated.
    * ``[int_or_ndarray, …]`` — a literal value; returned as-is.

    Attributes:
        ctype (str): Always ``"program"``.
        terms: The nested-list program representation.
    """

    ctype = "program"

    def __init__(self, terms):
        self.terms = terms

    def run(self, arg_list=None, terms=None):
        """Recursively evaluate the program tree and return the result.

        Args:
            arg_list: Runtime arguments supplied by the calling context
                (i.e., arguments not yet bound in the term tree).
            terms: Override for ``self.terms`` (used in recursive calls).

        Returns:
            The output value produced by the program (typically a numpy array
            of note values).
        """
        terms = self.terms if terms is None else terms

        if isinstance(terms[0], list):
            # Two independent sub-trees: evaluate and concatenate their outputs
            left_terms, right_terms = terms
            left_ret = Program(secure_list(left_terms)).run()
            right_ret = Program(secure_list(right_terms)).run()
            return Program(secure_list(left_ret) + secure_list(right_ret)).run()

        elif isinstance(terms[0], int) or isinstance(terms[0], np.ndarray):
            # Literal value — return as-is
            return terms

        # Program: [router, left, right] or [I]
        elif terms[0].ctype == "router":
            if len(terms) == 1:  # Should be router I
                return terms[0].run(arg_list)
            else:
                assert len(terms) == 3
                router_term, left_terms, right_terms = terms
                left_tree = Program(secure_list(left_terms))
                right_tree = Program(secure_list(right_terms))

                # Distribute arg_list to left/right according to the router
                sorted_args = {"left": [], "right": []}
                sorted_args = router_term.run(
                    sorted_args, copy_list(secure_list(arg_list))
                )
                # Run each subtree with its assigned arguments
                left_ret = left_tree.run(sorted_args["left"])
                right_ret = right_tree.run(sorted_args["right"])
                return Program(secure_list(left_ret) + secure_list(right_ret)).run()

        # Program: [primitive, args(optional)]
        elif terms[0].ctype == "primitive":
            # TODO: include compound (learned) functions
            func_term = terms[0]
            args_term = terms[1:]

            # Extract values from task-specific arguments (Note, Count, etc.)
            values_term = [
                x.value if isinstance(x, TaskSpecificPrimitive) else x
                for x in args_term
            ]

            if arg_list is not None:
                values_term += secure_list(arg_list)

            if len(values_term) == func_term.n_arg:
                return func_term.run(values_term)  # All args collected — execute
            else:
                return [func_term] + values_term  # Partial application

        else:
            return terms


# ---------------------------------------------------------------------------
# TaskSpecificPrimitive (base for domain-specific constant arguments)
# ---------------------------------------------------------------------------


class TaskSpecificPrimitive:
    """Base class for task-specific constant argument objects.

    Subclassed by domain-specific argument types such as :class:`Note` and
    :class:`Count` in the melody domain.  Each instance has a ``name``
    (used as the key in the global symbol table) and a ``value`` (the Python
    object passed to primitive lambdas during evaluation).

    Attributes:
        name (str): Canonical string identifier (e.g. ``"note_1_3"``).
        ctype (str): Identifies the argument type (set by subclass, e.g.
            ``"note"`` or ``"count"``).
        tp: Type object.
        value: The concrete Python value used during evaluation.
    """

    def __init__(self, name: str, tp, value=None):
        self.name = name
        self.ctype = None
        self.tp = tp
        self.value = value

    def __str__(self) -> str:
        return self.name

    @property
    def type_string(self) -> str:
        return f"->{self.ctype}"


# class RandomNote(Note):
#     def __init__(self, length=None, copy_note=None):
#         # TODO: add inf
#         if length is not None and copy_note is not None:
#             prev_array = self.string_to_array(copy_note)
#             prev_length = len(prev_array)
#             random_name = "rnote"
#             for i in range(prev_length):
#                 if i == prev_length - 1:
#                     random_name += "_" + str(int(np.random.randint(1, 6)))
#                 else:
#                     # check whether it is inf
#                     if prev_array[i] == np.inf:
#                         random_name += "_" + "inf"
#                     else:
#                         random_name += "_" + str(int(prev_array[i]))

#         elif copy_note is not None:
#             copy_note_value = self.string_to_array(eval(copy_note).name)
#             length = len(copy_note_value)
#             random_name = "rnote_" + "_".join(
#                 [str(int(np.random.randint(1, 6))) for _ in range(length)]
#             )

#         elif length is not None:
#             random_name = "rnote_" + "_".join(
#                 [str(int(np.random.randint(1, 6))) for _ in range(length)]
#             )

#         Note.__init__(self, random_name)


# class RandomTimeIndex(TimeIndex):
#     def __init__(self):
#         random_name = "rtime_" + str(int(np.random.randint(1, 6)))
#         TimeIndex.__init__(self, random_name)


# class RandomCount(Count):
#     def __init__(self):
#         random_name = "rcount_" + str(int(np.random.randint(1, 6)))
#         Count.__init__(self, random_name)
