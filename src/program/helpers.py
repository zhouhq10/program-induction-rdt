"""
helpers.py — General-purpose utilities for symbolic program inference.

Provides miscellaneous helpers used across the grammar, compressor, and
domain modules:

* **Data structures** — safe list construction, copying, and flattening.
* **Serialisation** — pickle I/O, term-to-dict conversion, name extraction.
* **Sampling** — power-law sampler used for program-depth priors.
* **Numeric utilities** — normalisation, softmax, sigmoid with temperature.
* **String utilities** — type-string formatting, bracket extraction.
* **Domain helpers** — motor-noise approximation, task-type resolution.
"""

import os
import re
import pickle
import collections
import numpy as np
from copy import copy
from math import exp, fabs
from pandas.core.common import flatten
from pandas.core.algorithms import isin
from pandas.core.dtypes.missing import isneginf_scalar

from operator import sub, add, mul, truediv

MAX_NOTE = 128


def expit_temp(x: float, temperature: float = 1) -> float:
    """Sigmoid function with temperature scaling.

    Args:
        x: Input value.
        temperature: Softens (> 1) or sharpens (< 1) the sigmoid.

    Returns:
        ``1 / (1 + exp(-x / temperature))``.
    """
    return 1 / (1 + np.exp(-x / temperature))


def power_law_sampler(alpha, N):
    """
    Samples from a Power Law distribution for positive integers 1, 2, ..., N.

    Parameters:
    alpha : float
        The exponent controlling the skewness of the distribution.
    N : int
        The maximum integer value (positive integer support).
    num_samples : int
        The number of samples to generate.

    Returns:
    np.array : An array of sampled integers from 1 to N based on the Power Law distribution.
    """
    # Generate an array of integers from 1 to N
    integers = np.arange(1, N + 1).astype(float)
    probabilities = np.power(integers, -alpha)

    # Normalize the probabilities so that they sum to 1
    probabilities /= probabilities.sum()

    return np.random.choice(integers, p=probabilities)


# Stringify a list of term names for dataframe evaluation
def names_to_string(names_list):
    return str(names_list).replace("'", "").replace(" ", "")


# Return a list
def secure_list(input):
    output = input if isinstance(input, list) else [input]
    return output


# Return copy of a list (of objects)
def copy_list(lt):
    ret = []
    for _l in lt:
        ret.append(copy(_l))
    return ret


def flatten_arbitrary_list(x):
    """Recursively flatten an arbitrarily nested iterable into a flat list.

    Note: relies on ``flatten`` imported from ``pandas.core.common`` at the
    bottom of this module; call only after that import is in scope.

    Args:
        x: Arbitrarily nested iterable, or a scalar.

    Returns:
        A flat list of all leaf elements.
    """
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]


def extract_names(nested_list: list) -> list:
    """Recursively extract the ``.name`` attribute (or string) from a nested list.

    Traverses a nested list of symbolic objects and collects their names,
    preserving the nested structure.

    Args:
        nested_list: Nested list of symbolic objects (primitives, routers,
            task-specific arguments) or numpy arrays / integers.

    Returns:
        Nested list with the same structure but items replaced by name strings.
    """
    result = []
    for item in nested_list:
        if isinstance(item, list):
            # Recursively handle nested lists
            result.append(extract_names(item))
        else:
            # Access the name attribute of the item
            if isinstance(item, np.ndarray) or isinstance(item, int):
                result.append(str(item))
            else:
                result.append(item.name)
    return result


# Get human-readable translation of a list of objects
# Expensive operation (recursion), be careful
def print_name(mylist):
    retlist = []
    for x in mylist:
        if isinstance(x, list):
            named_list = print_name(x)
            retlist.append(named_list)
        elif isinstance(x, bool) or isinstance(x, int):
            retlist.append(str(x))
        else:
            retlist.append(x.name)
    return retlist


# Join a list of arg types into underscore-separated string
def args_to_string(arg_list):
    return "_".join(secure_list(arg_list))


# Normalize a list
def normalize(mylist):
    total = sum(mylist)
    return [l / total for l in mylist]


# Apply softmax on a list
def softmax(mylist, base):
    exp_list = [exp(l * base) for l in mylist]
    total = sum(exp_list)
    return [el / total for el in exp_list]


# Term a term object into dict
def term_to_dict(term):
    if isinstance(term, bool):
        terms = "True" if term == True else "False"
        arg_types = ""
        return_type = "bool"
        type = "base_term"
    elif isinstance(term, int):
        terms = str(term)
        arg_types = ""
        return_type = "num"
        type = "base_term"
    elif term.ctype == "primitive":
        terms = term.name
        arg_types = args_to_string(term.arg_type)
        return_type = term.return_type
        type = term.ctype
    else:  # base term
        terms = term.name
        arg_types = ""
        return_type = term.ctype
        type = "base_term"
    return dict(terms=terms, arg_types=arg_types, return_type=return_type, type=type)


# Approximate motor noise
# Input choice_list should be a one-hot coded list
def add_motor_noise(choice_list, softmax_base=0):
    center_index = choice_list.index(1)
    weights = [exp(-1 * fabs(x - center_index)) for x in range(len(choice_list))]
    if softmax_base == 0:
        return normalize(weights)
    else:
        return softmax(weights, softmax_base)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def extract_index_from_object_files(folder_path: str) -> list:
    """Extract task indices from saved result files in a folder.

    Scans ``folder_path`` for files matching the pattern
    ``task_{x}_prog.obj`` and returns the sorted list of integers ``x``.

    Args:
        folder_path: Path to the directory containing saved program files.

    Returns:
        Sorted list of integer task indices found in the folder.
    """
    # Initialize an empty list to store the extracted integers x
    x_values = []

    # Define a regular expression pattern to match the 'task_x_prog.obj' format
    pattern = r"task_(\d+)_prog\.obj"

    # List all files in the specified folder
    files = os.listdir(folder_path)

    # Iterate over each file and extract x if the file name matches the pattern
    for file_name in files:
        match = re.match(pattern, file_name)
        if match:
            # Extract the integer x from the matched pattern
            x = int(match.group(1))
            x_values.append(x)

    return sorted(x_values)


def extract_nested_brackets(string: str) -> list:
    """Extract all bracket-delimited substrings from a string.

    Finds every substring delimited by matching ``[`` … ``]`` pairs, including
    nested brackets.

    Args:
        string: Input string possibly containing bracket-notation substrings.

    Returns:
        List of extracted substrings including their enclosing brackets,
        ordered by closing bracket position.
    """
    stack = []
    extracted = []

    for i, char in enumerate(string):
        if char == "[":
            stack.append(i)  # Push the index of the opening bracket
        elif char == "]" and stack:
            start = stack.pop()  # Pop to get the matching opening bracket index
            extracted.append(string[start : i + 1])  # Extract the substring

    return extracted


def if_else(arg_list: list):
    """Conditional primitive: return ``ret_1`` if ``cond`` is truthy, else ``ret_2``.

    Args:
        arg_list: ``[cond, ret_1, ret_2]``.

    Returns:
        ``ret_1`` or ``ret_2`` depending on ``cond``.
    """
    cond, ret_1, ret_2 = arg_list
    return ret_1 if cond else ret_2


def get_types(arg_list: list) -> list:
    """Expand ``"any_*"`` wildcard type tokens into their concrete type lists.

    Converts a list of type strings (possibly containing ``"any_type"``,
    ``"any_num"``, ``"any_arr"``, ``"any_note"``, ``"any_pause"`` wildcards)
    into a list where each wildcard is replaced by its set of concrete types.

    Args:
        arg_list: List of type name strings, possibly with ``"any_*"`` entries.

    Returns:
        List where each ``"any_*"`` element is replaced by a list of concrete
        type strings, and non-wildcard elements are wrapped in single-element
        lists for consistency.

    Raises:
        ValueError: If an unrecognised ``"any_*"`` suffix is encountered.
    """
    new_arg_list = []
    all_types = ["note", "note_arr", "pause", "pause_arr", "mixed_arr", "count", "time"]
    for arg in arg_list:
        if "any" in arg:
            any_types = arg.split("_")[1]
            if any_types == "type":
                new_arg_list.append(all_types)
            elif any_types == "num":
                new_arg_list.append(["note", "count", "time"])
            elif any_types == "arr":
                new_arg_list.append(["note_arr", "pause_arr", "mixed_arr"])
            elif any_types == "note":
                new_arg_list.append(["note", "note_arr"])
            elif any_types == "pause":
                new_arg_list.append(["pause", "pause_arr"])
            else:
                raise ValueError("Invalid any type")
        else:
            new_arg_list.append(arg)

    return new_arg_list
