"""
grammar.py — Program library and grammar classes for melody program induction.

Implements three grammar curricula that define the prior over programs and
manage the program library throughout inference:

Class hierarchy
---------------
::

    Grammar                   — Fixed PCFG; no library updates.
    └── AdaGrammar            — Global Pitman-Yor library (count-based AG).
        └── HierAdaGrammar    — Global + per-task local PY libraries (HAG).

Grammar / production table
--------------------------
Each grammar maintains a ``production`` DataFrame with columns:

    ``term``, ``arg_type``, ``ret_type``, ``type_string``, ``ctype``,
    ``log_prob``, ``comp_lp``, ``adaptor_lp``, ``count``, ``is_init``,
    ``frame``, ``depth``

Row types (``ctype``):

* ``"base_term"`` — task-specific argument constants (Note, Count tokens).
* ``"primitive"`` — melody primitive functions (memorize, repeat, etc.).
* ``"program"``   — learned / cached compound programs.

Pitman-Yor process (AG / HAG)
------------------------------
Both AG and HAG use a Pitman-Yor (PY) process as the prior over programs.
The probability of a new program (opening a new table) for type ``j`` is::

    P(new) = (alpha + d * m_j) / (alpha + n_j)

where ``m_j`` = number of distinct programs (tables) and ``n_j`` = total
usage count (customers) for type ``j``.  The probability of re-using an
existing program is::

    P(reuse k) = (n_k - d) / (alpha + n_j)

HAG adds a second PY level: a local (per-task) library that conditions
on the global library as its base measure.
"""

import math
import time
import random
import numpy as np
import pandas as pd
from scipy.special import expit
from numpy import random as np_random
from collections import defaultdict, Counter
from itertools import product as itertools_product
from typing import List, Dict, Union, AnyStr, Tuple


from src.program.primitive import Placeholder, Primitive
from src.program.router import *
from src.domain.melody.melody_primitive import (
    melody_primitive_name_list,
    note,
    count,
    create_or_get_pm_from_cache,
)

from src.program.helpers import (
    args_to_string,
    names_to_string,
    term_to_dict,
    secure_list,
    normalize,
)

EPS = 1e-6


class Grammar(object):
    """Fixed probabilistic context-free grammar (PCFG) over melody programs.

    Maintains a ``production`` DataFrame that acts as both the program library
    and the generative prior.  Rows represent individual program tokens
    (base_term, primitive, or learned program); columns include ``term``,
    ``arg_type``, ``ret_type``, ``type_string``, ``ctype``, ``log_prob``,
    ``comp_lp``, ``adaptor_lp``, ``count``, ``is_init``, ``frame``, and
    ``depth``.

    The PCFG assigns a fixed uniform prior over primitives grouped by return
    type.  Library update methods (``update_post_lib``, ``update_local_lib``)
    are no-ops in this base class; they are overridden by :class:`AdaGrammar`
    and :class:`HierAdaGrammar` to implement Pitman-Yor library growth.
    """

    def __init__(self, production):
        self.production = production
        self.ERROR_TERM = {
            "term": "ERROR",
            "arg_type": "",
            "ret_type": "",
            "type_string": "",
            "ctype": "ERROR",
            "log_prob": 0.0,
        }
        self.typed_counts = {}
        self.type_markers = (
            self.production[self.production["ctype"] == "base_term"]
            .ret_type.unique()
            .tolist()
        )
        self.type_strings = [
            "note->note",
            "note_count->note",
            "note_note->note",
            "note_note_note->note",
            "note_note_count->note",
            "note_count_note->note",
        ]

    # ----- Initialize primitive prior -----
    # Different prior choices over primitives
    def prior_uniform_per_type(self, production=None) -> None:
        """Assign a uniform ``comp_lp`` to primitives and base terms by return type.

        For primitive rows the first line computes a per-return-type uniform
        (``1 / count_with_same_ret_type``) but is immediately overwritten by a
        global uniform across *all* primitives (``1 / len(df_pm)``).  For
        base-term rows a per-type-string uniform is applied.  The two halves
        are then concatenated and returned.

        Note: the per-return-type computation for primitives on line 1 is
        superseded by the global uniform on line 2 and has no effect.

        Args:
            production: Optional DataFrame to use instead of ``self.production``.

        Returns:
            pd.DataFrame: Concatenation of updated base-term and primitive rows
            with ``comp_lp`` filled in.
        """
        df = self.production if production is None else production
        df_base = df[df["ctype"] == "base_term"]
        df_pm = df[df["ctype"] == "primitive"]

        df_pm["comp_lp"] = df_pm.groupby("ret_type")["ret_type"].transform(
            lambda x: math.log(1 / len(x))
        )
        df_pm["comp_lp"] = math.log(1 / len(df_pm))

        df_base["comp_lp"] = df_base.groupby("type_string")["type_string"].transform(
            lambda x: math.log(1 / len(x))
        )
        # df_base["comp_lp"] = math.log(1 / len(df_base))

        return pd.concat([df_base, df_pm], ignore_index=True)

    def prior_uniform_per_type_string(self, production=None) -> None:
        """Calculates and merges computation log probabilities into the 'production' DataFrame.

        This method groups the 'production' DataFrame by 'arg_type', 'ret_type', and 'ctype', computes the logarithm of the
        inverse of the count for each group, and then merges these log probabilities back into the original 'production'
        DataFrame. The calculation provides a uniform prior based on the string representation types in a production dataset.

        Side effects:
            - Direct modification of the 'production' DataFrame by adding a 'comp_lp' column with computed values.

        Returns:
            None
        """
        df = self.production if production is None else production
        if "comp_lp" in df.columns:
            df = df.drop(columns=["comp_lp"])

        groups = ["arg_type", "ret_type", "ctype"]

        grouped = df.groupby(groups).sum().reset_index()
        grouped["comp_lp"] = grouped.apply(
            lambda row: math.log(1 / row["count"]), axis=1
        )
        df = pd.merge(
            df,
            grouped[groups + ["comp_lp"]],
            on=groups,
            how="left",
        )

        return df

    def prior_random(self) -> None:
        """Assigns a random computation log probability to each entry in the 'production' DataFrame.

        This method applies a uniformly distributed random log probability to the 'comp_lp' column of the
        'production' DataFrame. It leverages numpy's random functionality to generate random values for each row.

        Side effects:
            - Direct modification of the 'comp_lp' column in the 'production' DataFrame with new random values.

        Returns:
            None
        """
        self.production["comp_lp"] = np.log(np.random.rand(len(self.production)))

    def update_lp_adaptor(self) -> None:
        """
        Do not consider AG here (count-based).
        """
        pass
        # TODO: program should be considered count 1
        # self.production["adaptor_lp"] = 0

    # ----- Initialize primitive prior -----

    # ----- Construct all frames given primitives and depth -----
    @staticmethod
    def get_all_routers(
        arg_list: List[str], left_arg_list: List[str], free_index: int
    ) -> List[str]:
        """
        Get all routers for the current arguments and type constraints.

        Args:
            args (list): List of arguments to process.
            left_arg_types (list): List of argument types for the left term.
            free_index (int): Current index of the free variable.

        Returns:
            list: List of router strings or objects.
        """
        # Ensure that there are arguments
        assert len(arg_list) > 0, "No arguments for router!"

        candidates = ["B"]
        if free_index >= 0:
            candidates.append("K")
        if free_index > 0:  # and len(arg_list) <= len(left_arg_list):
            candidates.append("C")
            candidates.append("S")
        routers = []
        for r in list(itertools_product(candidates, repeat=len(arg_list))):
            routers.append("".join(r))
        return routers

    @staticmethod
    def combine_terms(
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        router: str = "",
        router_lp: float = 0.0,
    ) -> pd.DataFrame:
        """
        Combine terms from two dataframes with optional routing.

        Args:
            left_df (pd.DataFrame): DataFrame containing the left terms and their log probabilities.
            right_df (pd.DataFrame): DataFrame containing the right terms and their log probabilities.
            router (str): Optional router term to combine with the left and right terms.
            router_lp (float): Log probability associated with the router.

        Returns:
            pd.DataFrame: DataFrame containing combined terms and their log probabilities.
        """
        # Add prefixes to distinguish columns from left and right dataframes
        left_df = left_df.add_prefix("left_")
        right_df = right_df.add_prefix("right_")

        # Add a key column for merging
        left_df["key"] = 0
        right_df["key"] = 0

        # Merge left and right dataframes on the key column
        combined = left_df.merge(right_df, how="outer")
        if len(router) < 1:
            combined["term"] = (
                "[" + combined["left_term"] + "," + combined["right_term"] + "]"
            )
            combined["log_prob"] = (
                combined["left_log_prob"] + combined["right_log_prob"]
            )
        else:
            combined["term"] = (
                "["
                + router
                + ","
                + combined["left_term"]
                + ","
                + combined["right_term"]
                + "]"
            )
            combined["log_prob"] = (
                combined["left_log_prob"] + combined["right_log_prob"] + router_lp
            )
        return combined[["term", "log_prob"]]

    def match_type_string(
        self, type_string: str, include_base_term: bool = False
    ) -> pd.DataFrame:
        """
        Retrive programs in the cache by pairing the type signature.
        Args:
            type_string: in the format of "(arg_type1)_..._(arg_typeN)->return_type"
            include_base_term: whether to include base term in the search.
        """
        df = self.production

        # Filter out base term if needed
        exclusion = "" if include_base_term else 'ctype!="base_term"'
        filtered_df = df.query(f"{exclusion}&ctype!='primitive'")

        # Filter out programs that do not match the type signature
        matched_pms = filtered_df[
            filtered_df["type_string"].apply(lambda x: type_string == x)
        ]

        return matched_pms if not matched_pms.empty else None

    def match_ret_type(
        self, return_type: str, filter_prog: bool = True, filter_base: bool = True
    ) -> pd.DataFrame:
        """
        Retrive programs in the cache by pairing the return type.
        """
        filtered_df = self.production

        # Filter out programs
        if filter_prog:
            filter_program = 'ctype!="program"'
            filtered_df = filtered_df.query(f"{filter_program}")
        if filter_base:
            filter_base = 'ctype!="base_term"'
            filtered_df = filtered_df.query(f"{filter_base}")

        # Find programs that match the return type
        matched_pms = filtered_df[
            filtered_df["ret_type"].apply(lambda x: return_type == x)
        ]

        return matched_pms if not matched_pms.empty else None

    def expand_typed_bfs(
        self,
        left_term: str,
        left_arg_types: list,
        free_index: int,
        args: list,
        depth: int,
    ) -> pd.DataFrame:
        """
        Expand terms using breadth-first search (BFS) with type constraints.

        Args:
            left_term (str): The term to expand.
            left_arg_types (list): List of argument types for the left term.
            free_index (int): Current index of the free variable.
            args (list): List of arguments to process.
            depth (int): Current depth of the search.

        Returns:
            pd.DataFrame: DataFrame containing expanded terms and their log probabilities.
        """
        if free_index < 0:
            # Base case: if no free index, return the term with zero log probability
            return pd.DataFrame({"term": [left_term], "log_prob": [0]})
        else:
            if len(args) < 1:
                # If no arguments left, expand the term without args
                left = self.expand_typed_bfs(
                    left_term, left_arg_types, free_index - 1, [], depth
                )
                right = self.enumerate_typed_bfs(
                    [[], left_arg_types[free_index]], depth - 1
                )
                return self.combine_terms(left, right)
            else:
                # Get all routers for the current arguments and type constraints
                routers = self.get_all_routers(args, left_arg_types, free_index)
                terms_df = pd.DataFrame({"term": [], "log_prob": []})
                for rt in routers:
                    evalrt = ComRouter([router_map[char] for char in rt])
                    routed_args = evalrt.run({"left": [], "right": []}, args)
                    # Evaluate the router to get routed arguments
                    # routed_args = eval(rt).run({"left": [], "right": []}, args)

                    # Recursively expand left and right terms
                    left = self.expand_typed_bfs(
                        left_term,
                        left_arg_types,
                        free_index - 1,
                        routed_args["left"],
                        depth,
                    )
                    right = self.enumerate_typed_bfs(
                        [routed_args["right"], left_arg_types[free_index]], depth - 1
                    )

                    # Combine terms if both left and right are not empty
                    if not left.empty and not right.empty:
                        tmp_df = self.combine_terms(
                            left, right, rt, math.log(1 / len(routers))
                        )
                        terms_df = pd.concat([terms_df, tmp_df], ignore_index=True)

            return terms_df

    def enumerate_typed_bfs(self, type_signature: List, depth: int = 5) -> pd.DataFrame:
        """
        Typed breadth-first search for programs with the type signature.
        Args:
            type_signature: A list of input types and the output type.
                            Example: [["note", 'count'], "note"]
            depth: The maximum depth to search.
        """

        # Intialize the dataframe and type signature
        programs_df = pd.DataFrame({"term": [], "log_prob": [], "type_string": []})
        arg_type, ret_type = type_signature

        # This means all programs should have at least one argument
        if len(arg_type) < 1:
            empty_df = pd.DataFrame({"term": [ret_type], "log_prob": [0]})
            programs_df = pd.concat([programs_df, empty_df], ignore_index=True)
            return programs_df

        # Translate the type signature into a string
        type_string = Placeholder.complete_typelist_to_string(arg_type, ret_type)

        # Find direct matches and put a placeholder in the program dataframe if found
        if self.match_type_string(type_string) is not None:
            empty_df = pd.DataFrame(
                {
                    "term": [f'PM("{type_string}")'],
                    "log_prob": [0],
                    "type_string": type_string,
                }
            )
            programs_df = pd.concat([programs_df, empty_df], ignore_index=True)

        # Return direct matches if there is
        if depth < 1:
            return programs_df
        # Enumerate recursively
        else:
            left_trees = self.match_ret_type(ret_type)

            if left_trees is None:
                return programs_df

            for i in left_trees.index:
                left_term = left_trees.at[i, "term"]
                left_arg_type = left_trees.at[i, "arg_type"]
                left_arg_list = left_arg_type.split("_")

                # Free index is the index of the argument that is not filled
                # The -1 is because the right tree will always return one argument
                free_index = len(left_arg_list) - 1

                # get routers
                routers = self.get_all_routers(
                    arg_list=arg_type,
                    left_arg_list=left_arg_list,
                    free_index=free_index,
                )
                for rt in routers:
                    evalrt = ComRouter([router_map[char] for char in rt])
                    routed_args = evalrt.run({"left": [], "right": []}, arg_type)
                    # routed_args = eval(rt).run({"left": [], "right": []}, arg_type)
                    left = self.expand_typed_bfs(
                        left_term,
                        left_arg_list,
                        free_index - 1,
                        routed_args["left"],
                        depth,
                    )
                    right = self.enumerate_typed_bfs(
                        [routed_args["right"], left_arg_list[free_index]],
                        depth - 1,
                    )
                    if len(left) > 0 and len(right) > 0:
                        tmp_df = self.combine_terms(
                            left, right, rt, math.log(1 / len(routers))
                        )
                        tmp_df["type_string"] = [type_string] * len(tmp_df)
                        programs_df = pd.concat(
                            [programs_df, tmp_df], ignore_index=True
                        )
            return programs_df

    # ----- Construct all frames given primitives and depth -----

    # ----- Generate one frame given primitives and depth -----
    def expand_one_typed_bfs(
        self,
        left_term: str,
        left_arg_types: list,
        free_index: int,
        args: list,
        depth: int,
    ) -> pd.DataFrame:
        """
        Expand terms using breadth-first search (BFS) with type constraints.

        Args:
            left_term (str): The term to expand.
            left_arg_types (list): List of argument types for the left term.
            free_index (int): Current index of the free variable.
            args (list): List of arguments to process.
            depth (int): Current depth of the search.

        Returns:
            pd.DataFrame: DataFrame containing expanded terms and their log probabilities.
        """
        if free_index < 0:
            # Base case: if no free index, return the term with zero log probability
            return pd.DataFrame({"term": [left_term], "log_prob": [0]})
        else:
            if len(args) < 1:
                # If no arguments left, expand the term without args
                left = self.expand_one_typed_bfs(
                    left_term, left_arg_types, free_index - 1, [], depth
                )
                right = self.enumerate_one_typed_bfs(
                    [[], left_arg_types[free_index]], depth - 1
                )
                return self.combine_terms(left, right)
            else:
                # Get all routers for the current arguments and type constraints
                routers = self.get_all_routers(args, left_arg_types, free_index)
                terms_df = pd.DataFrame({"term": [], "log_prob": []})
                rt = random.choice(routers)
                evalrt = ComRouter([router_map[char] for char in rt])
                routed_args = evalrt.run({"left": [], "right": []}, args)

                # Recursively expand left and right terms
                left = self.expand_one_typed_bfs(
                    left_term,
                    left_arg_types,
                    free_index - 1,
                    routed_args["left"],
                    depth,
                )
                right = self.enumerate_one_typed_bfs(
                    [routed_args["right"], left_arg_types[free_index]], depth - 1
                )

                # Combine terms if both left and right are not empty
                if not left.empty and not right.empty:
                    tmp_df = self.combine_terms(
                        left, right, rt, math.log(1 / len(routers))
                    )
                    terms_df = pd.concat([terms_df, tmp_df], ignore_index=True)

            return terms_df

    def enumerate_one_typed_bfs(
        self, type_signature: List, depth: int = 5
    ) -> pd.DataFrame:
        """
        Typed breadth-first search for programs with the type signature.
        Args:
            type_signature: A list of input types and the output type.
                            Example: [["note", 'count'], "note"]
            depth: The maximum depth to search.
        """

        # Intialize the dataframe and type signature
        programs_df = pd.DataFrame({"term": [], "log_prob": [], "type_string": []})
        arg_type, ret_type = type_signature

        # This means all programs should have at least one argument
        if len(arg_type) < 1:
            empty_df = pd.DataFrame({"term": [ret_type], "log_prob": [0]})
            programs_df = pd.concat([programs_df, empty_df], ignore_index=True)
            return programs_df

        # Translate the type signature into a string
        type_string = Placeholder.complete_typelist_to_string(arg_type, ret_type)

        # Find direct matches and put a placeholder in the program dataframe if found
        if self.match_type_string(type_string) is not None:
            empty_df = pd.DataFrame(
                {
                    "term": [f'PM("{type_string}")'],
                    "log_prob": [0],
                    "type_string": type_string,
                }
            )
            programs_df = pd.concat([programs_df, empty_df], ignore_index=True)

        # Return direct matches if there is
        if depth < 1:
            return programs_df

        # Enumerate recursively
        left_trees = self.match_ret_type(ret_type)
        if left_trees is None:
            return programs_df

        left_tree = left_trees.sample(n=1)
        i = left_tree.index[0]
        left_term = left_trees.at[i, "term"]
        left_arg_type = left_trees.at[i, "arg_type"]
        left_arg_list = left_arg_type.split("_")

        # Free index is the index of the argument that is not filled
        # The -1 is because the right tree will always return one argument
        free_index = len(left_arg_list) - 1

        # get routers
        routers = self.get_all_routers(
            arg_list=arg_type,
            left_arg_list=left_arg_list,
            free_index=free_index,
        )
        rt = random.choice(routers)
        evalrt = ComRouter([router_map[char] for char in rt])
        routed_args = evalrt.run({"left": [], "right": []}, arg_type)
        left = self.expand_one_typed_bfs(
            left_term,
            left_arg_list,
            free_index - 1,
            routed_args["left"],
            depth,
        )
        right = self.enumerate_one_typed_bfs(
            [routed_args["right"], left_arg_list[free_index]],
            depth - 1,
        )
        if len(left) > 0 and len(right) > 0:
            tmp_df = self.combine_terms(left, right, rt, math.log(1 / len(routers)))
            tmp_df["type_string"] = [type_string] * len(tmp_df)
            programs_df = pd.concat([programs_df, tmp_df], ignore_index=True)
        return programs_df

    def generate_frame(
        self,
        type_string: str,
        depth: int,
    ) -> pd.DataFrame:
        """Sample a random program frame and unfold it into concrete programs.

        Calls :meth:`enumerate_one_typed_bfs` in a loop until a valid frame is
        produced (one whose token count is ≤ 7), then calls
        :meth:`unfold_frame` to substitute PM placeholders with actual
        library entries.

        Args:
            type_string: Type-signature string, e.g. ``"note_count->note"``.
            depth: Maximum BFS depth used when sampling the frame structure.

        Returns:
            pd.DataFrame: Rows of concrete programs with ``comp_lp``,
            ``log_prob``, ``type_string``, ``frame``, and ``depth`` columns.
        """
        # Translate the type signature into a string
        type_list = Placeholder.string_to_typelist(type_string)

        # Enumerate programs with the type signature
        nonerror = False
        while not nonerror:
            frame = self.enumerate_one_typed_bfs(type_list, depth)
            if not frame.empty:
                if (
                    frame["term"][0].count("note") + frame["term"][0].count("count")
                    <= 7
                ):
                    nonerror = True
                    frame_term = frame["term"][0]
                    frame_depth = depth

        # Unfold the frame to get the program
        prog = self.unfold_frame(frame["term"][0], type_string)
        prog["depth"] = frame_depth
        prog["frame"] = frame_term

        if len(prog) == 0:
            import ipdb

            ipdb.set_trace()
        return prog

    def generate_frame_new(
        self,
        type_string: str,
        depth: int,
    ) -> pd.DataFrame:
        """Sample a random program frame and unfold it (variant without the ipdb guard).

        Functionally identical to :meth:`generate_frame` but omits the
        ``len(prog) == 0`` debug breakpoint.

        TODO: needs to be combined with :meth:`generate_frame`.

        Args:
            type_string: Type-signature string, e.g. ``"note_count->note"``.
            depth: Maximum BFS depth used when sampling the frame structure.

        Returns:
            pd.DataFrame: Rows of concrete programs with ``comp_lp``,
            ``log_prob``, ``type_string``, ``frame``, and ``depth`` columns.
        """
        # Translate the type signature into a string
        type_list = Placeholder.string_to_typelist(type_string)

        # Enumerate programs with the type signature
        nonerror = False
        while not nonerror:
            frame = self.enumerate_one_typed_bfs(type_list, depth)
            if not frame.empty:
                if (
                    frame["term"][0].count("note") + frame["term"][0].count("count")
                    <= 7
                ):
                    nonerror = True
                    frame_term = frame["term"][0]
                    frame_depth = depth

        # Unfold the frame to get the program
        prog = self.unfold_frame(frame["term"][0], type_string)
        prog["depth"] = frame_depth
        prog["frame"] = frame_term

        return prog

    # ----- Generate one frame given primitives and depth -----

    # ----- Generate one program given primitives and depth -----
    @staticmethod
    def sample_router(arg_list, free_index):
        """Sample a random router string for the given argument list.

        Builds a router string of the same length as ``arg_list`` by choosing
        each combinator position uniformly from ``{B, C, S, K}``.  If
        ``free_index < 0`` (all argument slots already filled on the left side)
        the entire string defaults to all-``B`` (pass-right).

        Note: the ``len(arg_list) < 0`` guard at the top is unreachable (list
        length is never negative) and exists only as a historical stub.

        Args:
            arg_list: List of argument type strings to be routed.
            free_index: Index of the last unfilled argument slot on the left
                subtree; negative means the left side is fully saturated.

        Returns:
            str: Router string, e.g. ``"BC"`` for a two-argument program.
        """
        if len(arg_list) < 0:
            return None
        # assert len(arg_list) > 0, "No arguments for router!"
        if free_index < 0:
            return "B" * len(arg_list)
        else:
            return "".join([np.random.choice(["C", "B", "S", "K"]) for _ in arg_list])

    def sample_matched_program(self, return_type, filter_prog=False):
        """Uniformly sample one production row whose return type matches.

        Args:
            return_type: Target return type string (e.g. ``"note"``).
            filter_prog: If ``True``, exclude ``ctype=="program"`` rows
                (passed to :meth:`match_ret_type`).

        Returns:
            dict: A single production row as a dict, or ``self.ERROR_TERM``
            if no matching row exists.
        """
        matched = self.match_ret_type(return_type, filter_prog)
        if matched is None:
            return self.ERROR_TERM
        else:
            return matched.sample(n=1).iloc[0].to_dict()

    # Sample base terms (task specific)
    def sample_base(self, type, production=None):
        production = self.production if production is None else production
        bases = production.query(f'ret_type=="{type}"&ctype=="base_term"')
        if bases is None or bases.empty:
            return self.ERROR_TERM
        else:
            sampled = bases.sample(n=1).iloc[0].to_dict()
            return sampled

    def generate_program(self, type_signature, cur_step=0, max_step=5) -> Dict:
        """
        Args:
            type_signature: A list of input types and the output type (type_list)
        """
        # Check if the current step exceeds the maximum step
        if cur_step > max_step:
            return self.ERROR_TERM

        arg_t, ret_t = type_signature
        type_string = Placeholder.complete_typelist_to_string(arg_t, ret_t)
        matched = self.match_type_string(type_string)
        if matched is not None and cur_step > 0:
            # Sample from the matched programs
            matched_sample = (
                matched.sample(n=1, weights=np.exp(matched["log_prob"]) + EPS)
                .iloc[0]
                .to_dict()
            )
            prob_new_dish = np.exp(matched_sample["log_prob"])
            # Randomly sample a cached program -> could be used for a subtree in the new program
            if np_random.random() < prob_new_dish:
                return matched_sample

        cur_step += 1
        if len(arg_t) == 0:
            # return a base term
            base_term = self.sample_base(ret_t)
            return base_term
        else:
            # generate new program
            left = self.sample_matched_program(ret_t, filter_prog=True)
            left_args = left["arg_type"].split("_")
            free_index = len(left_args) - 2
            router = self.sample_router(arg_t, free_index)
            evalrouter = ComRouter([router_map[char] for char in router])
            routed_args = evalrouter.run({"left": [], "right": []}, arg_t)
            # routed_args = eval(router).run({"left": [], "right": []}, arg_t)

            # expand left side until no un-filled arguments
            left_pm = self.expand_program(
                left,
                routed_args["left"],
                free_index,
                cur_step,
                max_step,
            )
            right_pm = self.generate_program(
                [routed_args["right"], left_args[-1]],
                cur_step,
                max_step,
            )
            terms = [router, left_pm["term"], right_pm["term"]]
            program_dict = {
                "term": names_to_string(terms),
                "arg_type": args_to_string(type_signature[0]),
                "ret_type": type_signature[1],
                "type_string": type_string,
                "ctype": "program",
                "log_prob": left_pm["log_prob"] + right_pm["log_prob"],
            }
            return program_dict

    # Lefthand-side tree
    def expand_program(self, candidate, arg_list, free_index, cur_step, max_step):
        """Recursively fill the unfilled argument slots of a left-subtree candidate.

        Given a ``candidate`` primitive / program that still expects
        ``free_index + 1`` more arguments, routes ``arg_list`` through a
        sampled combinator and fills each slot by recursively calling
        :meth:`generate_program`.

        Args:
            candidate (dict): Production row for the left-subtree term.
            arg_list (list): Argument type strings available to be routed into
                the remaining slots of ``candidate``.
            free_index (int): Index of the rightmost unfilled slot in
                ``candidate``; negative means fully saturated (base case).
            cur_step (int): Current recursion depth (used to bound sampling).
            max_step (int): Maximum allowed recursion depth.

        Returns:
            dict: A production-row dict with ``term``, ``arg_type``,
            ``ret_type``, ``ctype``, and ``log_prob`` keys representing the
            fully or partially expanded left subtree.
        """
        if free_index < 0:
            return candidate  # checked
        else:
            left_args = candidate["arg_type"].split("_")
            if len(arg_list) < 1:
                left_pm = self.expand_program(
                    candidate,
                    arg_list,
                    free_index - 1,
                    cur_step,
                    max_step,
                )
                right_pm = self.generate_program(
                    [arg_list, left_args[free_index]], cur_step, max_step
                )
                terms = [left_pm["term"], right_pm["term"]]
            else:
                router = self.sample_router(arg_list, free_index - 1)
                evalrouter = ComRouter([router_map[char] for char in router])
                routed_args = evalrouter.run({"left": [], "right": []}, arg_list)
                # routed_args = eval(router).run({"left": [], "right": []}, arg_list)
                left_pm = self.expand_program(
                    candidate,
                    routed_args["left"],
                    free_index - 1,
                    cur_step,
                    max_step,
                )
                right_pm = self.generate_program(
                    [routed_args["right"], left_args[free_index]],
                    cur_step,
                    max_step,
                )
                terms = [router, left_pm["term"], right_pm["term"]]
            return {
                "term": names_to_string(terms),
                "arg_type": candidate["arg_type"],
                "ret_type": candidate["ret_type"],
                "ctype": "program",
                "log_prob": left_pm["log_prob"] + right_pm["log_prob"],
            }

    # ----- Generate one program given primitives and depth -----

    # ----- Fill frames and unfold programs -----
    @staticmethod
    def iter_compose_programs(terms_list, cp_list, lp_list):
        """
        Generate combinations of program terms, their composition log probabilities (cp_list),
        and their log probabilities (lp_list).
        """

        programs_list = list(itertools_product(*terms_list))
        programs_list_agg = [",".join(p) for p in programs_list]
        comp_lp_list = list(itertools_product(*cp_list))
        comp_lp_list_agg = [sum(x) for x in comp_lp_list]
        log_probs_list = list(itertools_product(*lp_list))
        log_probs_list_agg = [sum(x) for x in log_probs_list]
        return pd.DataFrame(
            {
                "term": programs_list_agg,
                "comp_lp": comp_lp_list_agg,
                "log_prob": log_probs_list_agg,
            }
        )

    def unfold_prog_with_lp(
        self,
        type_string: str,
        depth: int,
        num_sample: int = None,
    ) -> pd.DataFrame:
        """Sample a concrete program for ``type_string``, respecting the library prior.

        With probability ``exp(comp_lp_open_new_table)`` a brand-new program is
        generated via :meth:`generate_program`; otherwise an existing program is
        queried from the library via :meth:`query_existing_prog`.

        Args:
            type_string: Type-signature string, e.g. ``"note_count->note"``.
            depth: Maximum generation depth passed to :meth:`generate_program`.
            num_sample: Unused in the base class; kept for API consistency with
                subclass overrides.

        Returns:
            pd.DataFrame: Single-row DataFrame of the sampled program with at
            least ``term``, ``comp_lp``, and ``log_prob`` columns.
        """
        type_list = Placeholder.string_to_typelist(type_string)
        prob_new_table = np.exp(self.comp_lp_open_new_table(type_string))

        # Construct
        if np_random.random() < prob_new_table:
            nonerror = False
            while not nonerror:
                prog = self.generate_program(type_list, cur_step=0, max_step=depth)
                if not "ERROR" in prog["term"]:
                    nonerror = True
            prog = pd.DataFrame.from_dict(prog, orient="index").T
        # Query
        else:
            prog = self.query_existing_prog(type_string)

        return prog
        # NOTE:
        # Not sure whether we need to add additional length constraints here
        # because we have the depth sampling which corresponds to the length of frames (log_prob)
        # progs["comp_lp"] += cur_frame["log_prob"]
        # progs["log_prob"] += cur_frame["log_prob"]
        # progs["type_string"] = cur_frame["type_string"]

    def query_existing_prog(
        self, type_string: str = None, num_sample: int = 1
    ) -> pd.DataFrame:
        return pd.DataFrame({"term": [], "comp_lp": [], "log_prob": []})

    def unfold_frame(
        self, term: str, type_string: str, num_sample=5, production=None
    ) -> pd.DataFrame:
        """
        Build new programs given the term and type string.

        The probability of new program is composed by:
        - The composition log probability of the primitives in the program
        - The log probability of opening a new table
        """
        progs_list, comp_lps_list, log_probs_list = [], [], []
        term_list = term.split(",")

        production = self.production if production is None else production

        for t in term_list:
            tm = t.strip("[]")
            if tm in self.type_markers:
                unfolded = production.query(f'ret_type=="{tm}"&ctype=="base_term"')
            elif "PM" in tm:
                type_string = tm.split('"')[1]
                unfolded = production.query(
                    f'type_string=="{type_string}"&ctype=="program"'
                )
                if not unfolded.empty:
                    unfolded = unfolded.sample(
                        n=min(len(unfolded), num_sample)
                    )  # TODO: what criteria to sample
            elif any(x in tm for x in ["C", "B", "S", "K"]):
                unfolded = pd.DataFrame(
                    {"term": [tm], "comp_lp": [0], "log_prob": [0]}
                )  # Taken care of by the frame base lp
            elif tm in melody_primitive_name_list:
                unfolded = production.query(f'term=="{tm}" & ctype=="primitive"')[
                    ["term", "log_prob"]
                ]
                unfolded["comp_lp"] = unfolded["log_prob"]
            else:
                raise ValueError(f"Unknown term: {tm}")

            progs_list.append([t.replace(tm, u) for u in list(unfolded["term"])])
            comp_lps_list.append(list(unfolded["comp_lp"]))
            log_probs_list.append(list(unfolded["log_prob"]))

        # Compute the unfolded programs
        final_unfolded = self.iter_compose_programs(
            progs_list, comp_lps_list, log_probs_list
        )

        final_unfolded["type_string"] = type_string
        return final_unfolded

    # ----- Fill frames and unfold programs -----

    # ----- Helper functions -----
    def comp_lp_open_new_table(
        self, type_string: str, production: pd.DataFrame = None
    ) -> float:
        """Log probability of opening a new table (constructing a new program).

        For the base PCFG there is no library, so a new program is always
        constructed — this method returns ``0.0`` (log of probability 1).
        Overridden in :class:`AdaGrammar` to apply the Pitman-Yor formula.

        Args:
            type_string: Type-signature string for which a new program is
                being considered.
            production: Unused in PCFG; kept for API consistency.

        Returns:
            float: Log probability of opening a new table (``0.0`` here).
        """
        # Always generate new programs for PCFG
        return 0.0

    def _extract_pm_init_post(
        self, production: pd.DataFrame = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract the initial and post programs from the production dataframe.
        """
        prod = production if production is not None else self.production

        pm_init = prod.query("is_init == 1")
        pm_post = prod.query("is_init == 0")

        return pm_init, pm_post

    def _extract_pm_base_prog(
        self, production: pd.DataFrame = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract the base term and program term from the production dataframe.
        """
        prod = production if production is not None else self.production

        base = prod.query("ctype != 'program'")
        prog = prod.query("ctype == 'program'")

        return base, prog

    # ----- Helper functions -----

    # ----- Update library -----
    def update_overall_lp(self, production=None) -> bool:
        """
        Calculate the overall log probability of a program.
        If it is a primitive, the log probability is the log of the inverse of the count of the program given the type signature.
        If it is a program, the log probability is the sum of the adaptor_lp and the comp_lp.
        """
        df = self.production if production is None else production
        base_term, prog_term = self._extract_pm_base_prog(production=df)

        # Calculate the log probability of base term
        base_term["log_prob"] = base_term["comp_lp"]
        prog_term["log_prob"] = prog_term["adaptor_lp"]  # TODO prog_term["comp_lp"] +

        # Combine base term and composed term
        return pd.concat([base_term, prog_term], ignore_index=True)

    def initialize_local_lib(self) -> None:
        """
        Initialize the local library before the task starts (for several inference iterations on the same task).
        The base grammar, PCFG and AG, do not have local library.
        This is to be consistent with the HAG.
        """
        pass

    def update_local_lib(self, chosen_subprog: pd.DataFrame) -> None:
        """
        Update the local library based on the chosen subprogram
        In current PCFG, we do not have a library for caching the programs
        """
        pass

    def update_post_lib(self, new_progs, lib=None) -> None:
        """
        Update the gloabl library based on the compression degree and counts
        In current PCFG, we do not have a library for caching the programs
        """
        pass

    # ----- Update library -----


class AdaGrammar(Grammar):
    """Adaptor Grammar with a single global Pitman-Yor program library.

    Extends :class:`Grammar` with a count-based library that grows as
    compressed programs are added after each task.  Program prior probabilities
    are updated using the Pitman-Yor (PY) process::

        P(new)      = (alpha + d * m_j) / (alpha + n_j)
        P(reuse k)  = (n_k - d)         / (alpha + n_j)

    where ``m_j`` = number of distinct programs for type ``j``, ``n_j`` =
    total usage count, and ``n_k`` = usage count of program ``k``.

    The global library is stored in ``self.production``.  After compression
    :meth:`update_post_lib` adds newly-found programs, increments counts, and
    re-computes ``adaptor_lp`` and ``log_prob`` for all library entries.

    Attributes:
        lib_size (int): Maximum number of programs retained in the library
            (excess programs are pruned by probability-weighted sampling).
        global_alpha (float): Pitman-Yor concentration parameter α.
        global_d (float): Pitman-Yor discount parameter d ∈ [0, 1).
        local_alpha (float): Alias for ``global_alpha`` (for API consistency
            with :class:`HierAdaGrammar`).
        local_d (float): Alias for ``global_d``.
    """

    def __init__(
        self,
        production: pd.DataFrame,
        lib_size: int = 100,
        global_alpha: float = 1.0,
        global_d: float = 0.2,
    ):
        super().__init__(production)
        self.lib_size = lib_size

        # Parameters for the Pitman-Yor process
        self.global_alpha = global_alpha
        self.global_d = global_d
        self.local_alpha, self.local_d = self.global_alpha, self.global_d

    @staticmethod
    def _reformalize_new_prog(new_progs: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the new programs into the format of the production dataframe.
        """
        new_progs = new_progs[
            ["term", "type_string", "comp_lp", "log_prob", "frame", "depth"]
        ]
        new_progs["arg_type"] = new_progs["type_string"].apply(
            lambda x: x.split("->")[0]
        )
        new_progs["ret_type"] = new_progs["type_string"].apply(
            lambda x: x.split("->")[1]
        )
        new_progs["ctype"] = "program"
        new_progs["is_init"] = 0
        new_progs["count"] = 1
        new_progs["adaptor_lp"] = 0

        # Remove duplicates
        # TODO: option; this could also be done by changing the alpha and d values
        new_progs = new_progs.drop_duplicates(
            subset=["term", "type_string"], keep="first"
        )
        return new_progs

    def update_lp_adaptor(
        self, alpha: float, d: float, production: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Compute the adaptor log probability for the programs in the library.
        The adaptor log probability is given by:
            log((n_j - d) / (alpha + m_j)
        """
        prod = self.production if production is None else production

        # Adaptor grammar prior in log, see Percy Liang et al. 2010
        result = (
            prod.groupby("type_string")
            .agg(
                distinct_count_per_type=("term", "nunique"),
                total_count_per_type=("count", "sum"),
            )
            .reset_index()
        )
        prod = prod.merge(result, on="type_string", how="left")
        prod["adaptor_lp"] = np.log(
            (prod["count"] - d) / (alpha + prod["total_count_per_type"])
        )

        # Remove the term_count and total_count columns
        prod = prod.drop(columns=["distinct_count_per_type", "total_count_per_type"])

        return prod

    def _restrict_lib_size(
        self, production: pd.DataFrame = None, lib_size: int = None
    ) -> pd.DataFrame:
        """
        Restrict the library size to the given size.
        Current setting is to keep the top n programs based on the log probability.
        """
        df = self.production if production is None else production
        lib_size = self.lib_size if lib_size is None else lib_size

        lib_base, lib_prog = self._extract_pm_base_prog(production=df)

        lib_prog["exp_prob"] = np.exp(lib_prog["log_prob"])

        sampled_size = min(lib_size, len(lib_prog))
        lib_prog = lib_prog.sample(
            n=sampled_size, weights=lib_prog["exp_prob"]
        ).reset_index(drop=True)
        lib_prog = lib_prog.drop(columns=["exp_prob"])

        return pd.concat([lib_base, lib_prog]).reset_index(drop=True)

    def query_existing_prog(
        self, type_string: str, num_sample: int = 1
    ) -> pd.DataFrame:
        """
        Query the existing programs in the library.
        In non-hierarchical AG, we only consider the program in global library (which is self.production).

        The log_prob is the sum of the comp_lp and adaptor_lp, which is already updated given the count.
        """
        qs = '&ctype=="program"'
        unfolded = self.production.query(f'type_string=="{type_string}"{qs}')

        return unfolded.sample(
            n=min(num_sample, len(unfolded)), weights=np.exp(unfolded["log_prob"])
        )

    def comp_lib_table(self, type_string: str, production: pd.DataFrame = None) -> int:
        """
        Compute the number of tables in the library.
        -> The number of tables is the number of distinct programs in the library given one type.
        """
        prod = self.production if production is None else production

        # Number of distinct programs given type_string in the library
        queried_term = prod.query(f'type_string=="{type_string}"&ctype=="program"')
        distinct_terms = queried_term["term"].unique()

        return len(distinct_terms)

    def comp_lib_customer(
        self, type_string: str, production: pd.DataFrame = None
    ) -> int:
        """
        Compute the number of customers in the library.
        -> The number of customers is the sum number of counts for programs in the library given one type.
        """
        production = self.production if production is None else production

        queried_term = production.query(
            f'type_string=="{type_string}"&ctype=="program"'
        )
        num_customers = queried_term["count"].sum()

        return num_customers

    def comp_lp_open_new_table(
        self, type_string: str, production: pd.DataFrame = None
    ) -> float:
        """
        Compute the log probability of opening a new table.

        In the Pitman-Yor process, the probability of opening a new table is given by:
        log(alpha + d * m_j) - log(alpha + n_j).
        Opening a new table is the same as constructing a new program for a given type_string

        In non-hierarchical AG, self.production is the global library.
        -> To be consistent, we set the local parameters to be the same as the global parameters.

        In hierarchical AG, self.production is considered as the local library.
        """
        df = self.production if production is None else production

        # Number of distinct programs in the library
        m_j = self.comp_lib_table(type_string=type_string, production=df)
        n_j = self.comp_lib_customer(type_string=type_string, production=df)

        # NOTE: to be consistent with HierAdaGrammar, the local parameters are actually the global params in AdaGrammar
        lp_new_table = np.log(self.local_alpha + m_j * self.local_d) - np.log(
            self.local_alpha + n_j
        )

        return lp_new_table

    def generate_frame(
        self,
        type_string: str,
        depth: int,
    ) -> pd.DataFrame:
        """Sample a program frame using the global library prior (AG version).

        Decides whether to construct a new frame (probability
        ``exp(comp_lp_open_new_table)``) or to re-use an existing one from
        the global library.  When re-using, the existing frame structure is
        unfolded with fresh base-term assignments and its stored ``log_prob``
        is preserved.

        Args:
            type_string: Type-signature string, e.g. ``"note_count->note"``.
            depth: Maximum BFS depth for new frame enumeration.

        Returns:
            pd.DataFrame: Rows of concrete programs with ``comp_lp``,
            ``log_prob``, ``type_string``, ``frame``, and ``depth`` columns.
        """
        type_list = Placeholder.string_to_typelist(type_string)
        prob_new_table = np.exp(self.comp_lp_open_new_table(type_string))

        # Construct
        if np_random.random() < prob_new_table:
            nonerror = False
            while not nonerror:
                frame = self.enumerate_one_typed_bfs(type_list, depth)
                if not frame.empty:
                    frame_term = frame["term"][0]
                    frame_depth = depth
                    if (
                        frame["term"][0].count("note") + frame["term"][0].count("count")
                        <= 7
                    ):
                        nonerror = True
            prog = self.unfold_frame(frame["term"][0], type_string)

        # Query
        else:
            prog = self.query_existing_prog(type_string).reset_index(drop=True)
            frame_term = prog["frame"][0]
            frame_depth = prog["depth"][0]
            prog_prob = prog["log_prob"][0]

            # Unfold the frame
            prog = self.unfold_frame(frame_term, type_string)
            # Update the log probability
            prog["log_prob"] = prog_prob

        prog["depth"] = frame_depth
        prog["frame"] = frame_term

        return prog

    def update_post_lib(
        self, new_progs: pd.DataFrame, lib: pd.DataFrame = None
    ) -> None:
        """
        Update the global library by adding newly-found-useful programs.
        Here, the prior of cached programs will not be updated given the counts.

        Args:
            new_progs (pd.DataFrame): The new programs to be added to the library.
            lib (pd.DataFrame): The current library to be updated.
        """
        # Update the global library
        lib = self.production if lib is None else lib
        lib_base, lib_prog = self._extract_pm_base_prog(lib)

        # Reformalize the new programs
        new_progs = self._reformalize_new_prog(new_progs)

        # Combine the base and program terms to the production
        new_post = pd.concat([lib_prog, new_progs]).reset_index(drop=True)
        new_post = new_post.groupby(["term", "type_string"])
        # Merge new_progs with df posts
        new_post = pd.DataFrame(
            {
                "term": new_post["term"].first(),
                "arg_type": new_post["arg_type"].first(),
                "ret_type": new_post["ret_type"].first(),
                "type_string": new_post["type_string"].first(),
                "ctype": new_post["ctype"].first(),
                "count": new_post["count"].sum(),
                "is_init": new_post["is_init"].last(),
                "comp_lp": new_post["comp_lp"].last(),
                "adaptor_lp": new_post["adaptor_lp"].last(),
                "log_prob": new_post["log_prob"].last(),
                "frame": new_post["frame"].last(),
                "depth": new_post["depth"].last(),
            }
        ).reset_index(drop=True)

        # Update comp_lp given type strings
        new_post = self.prior_uniform_per_type_string(production=new_post)
        # Update adaptor_lp given counts
        new_post = self.update_lp_adaptor(
            production=new_post,
            alpha=self.global_alpha,
            d=self.global_d,
        )

        # Combine the base and program terms to the production
        self.production = pd.concat([lib_base, new_post]).reset_index(drop=True)

        # Update overall log probability
        self.production = self.update_overall_lp()

        # Restrict the library size
        self.production = self._restrict_lib_size()


class HierAdaGrammar(AdaGrammar):
    """Hierarchical Adaptor Grammar with global and per-task local PY libraries.

    Extends :class:`AdaGrammar` with a second Pitman-Yor level: each task
    has its own *local* library (``self.production``) that is initialised
    from the *global* library (``self.global_production``) and updated
    independently within a task.  After the task the global library is
    updated via :meth:`update_post_lib`.

    The two-level decision at frame-generation time is::

        1. Open a new local table?  P = exp(comp_lp_open_new_table)
           1.1 Open a new global dish?  P = exp(comp_lp_open_new_dish)
               1.1.1 Yes → generate a brand-new program (BFS enumeration).
               1.1.2 No  → query an existing program from global library.
           1.2 Re-use an existing local table → query local library.

    Attributes:
        global_production (pd.DataFrame): The shared global library, updated
            after every task via :meth:`update_post_lib`.
        local_lib_size (int): Maximum size of the per-task local library.
        global_alpha (float): Global PY concentration parameter α_g.
        global_d (float): Global PY discount parameter d_g.
        local_alpha (float): Local PY concentration parameter α_l.
        local_d (float): Local PY discount parameter d_l.
        local_pattern (bool | None): If truthy, skip frame unfolding after a
            local-library query and return the cached program directly.
        history_table (dict): Maps type-signature → cumulative number of
            distinct programs (tables) ever created in the global library,
            used for ``comp_lp_open_new_dish``.
    """

    def __init__(
        self,
        production,
        lib_size,
        local_lib_size=None,
        global_alpha=1.0,
        global_d=0.2,
        local_alpha=1.0,
        local_d=0.2,
        local_pattern=None,
    ):
        super().__init__(production, lib_size)
        self.local_lib_size = local_lib_size if local_lib_size is not None else lib_size

        # Parameters for the Pitman-Yor process
        self.global_alpha, self.global_d = global_alpha, global_d
        self.local_alpha, self.local_d = local_alpha, local_d
        self.local_pattern = local_pattern

        # Define the libraries
        # self.global_production is the global library
        # self.production is the local library
        self.global_production = production
        self.init_pm = production

        # TODO: this means for every restaurant, we have 3*6 tables already there
        # -> maybe we should not allow initial programs
        self.history_table = {}
        for type_string in self.type_strings:
            self.history_table[type_string] = len(
                self.production.query(
                    f'type_string=="{type_string}" & ctype=="program"'
                )
            )

    def _reset(self):
        self.production = self.init_pm
        self.global_production = self.init_pm
        self.history_table = {}
        for type_string in self.type_strings:
            self.history_table[type_string] = len(
                self.production.query(
                    f'type_string=="{type_string}" & ctype=="program"'
                )
            )

    # ----- Frame construction -----
    def generate_frame(
        self,
        type_string: str,
        depth: int,
    ) -> pd.DataFrame:
        """
        Unfolds the given term into their component programs from the program library.
        """
        type_list = Placeholder.string_to_typelist(type_string)

        # This is based on self.production which is local library
        prob_new_table = np.exp(self.comp_lp_open_new_table(type_string))

        if np_random.random() < prob_new_table:
            # 1.1 create new table
            # Q: new dish or query global library
            _, logprob_new_dish = self.comp_lp_open_new_dish(type_string)

            prob_new_dish = np.exp(logprob_new_dish)
            if np_random.random() < prob_new_dish:
                # import ipdb; ipdb.set_trace()
                # 1.1.1 generate new program
                nonerror = False
                while not nonerror:
                    frame = self.enumerate_one_typed_bfs(type_list, depth)
                    if not frame.empty:
                        frame_term = frame["term"][0]
                        frame_depth = depth
                        if (
                            frame["term"][0].count("note")
                            + frame["term"][0].count("count")
                            <= 7
                        ):
                            nonerror = True
                prog = self.unfold_frame(frame_term, type_string)
                prog["depth"] = frame_depth
                prog["frame"] = frame_term
                return prog

            else:
                # import ipdb; ipdb.set_trace()
                # 1.1.2 query global library
                prog = self.query_existing_prog(
                    type_string, production=self.global_production
                )

        else:
            # 1.2 query local library
            prog = self.query_existing_prog(type_string, production=self.production)
            if self.local_pattern:
                return prog

        # Unfold the frame
        frame_term = prog["frame"].item()
        frame_depth = prog["depth"].item()
        prog_prob = prog["log_prob"].item()

        prog = self.unfold_frame(frame_term, type_string)
        if prog.empty:
            prog = self.unfold_frame(
                frame_term, type_string, production=self.global_production
            )
        # Update the log probability
        prog["log_prob"] = prog_prob
        prog["depth"] = frame_depth
        prog["frame"] = frame_term

        return prog

    # ----- Frame construction -----

    # ----- Program construction -----
    def unfold_prog_with_lp(
        self,
        type_string,
        depth,
        num_sample: int = None,
    ) -> pd.DataFrame:
        """
        Unfolds the given term into their component programs from the program library.
        """
        type_list = Placeholder.string_to_typelist(type_string)
        # This is based on self.production which is local library
        prob_new_table = np.exp(self.comp_lp_open_new_table(type_string))

        if np_random.random() < prob_new_table:
            # 1.1 create new table
            # Q: new dish or query global library
            K_dish, logprob_new_dish = self.comp_lp_open_new_dish(type_string)
            prob_new_dish = np.exp(logprob_new_dish)

            if np_random.random() < prob_new_dish:
                # 1.1.1 generate new program
                nonerror = False
                while not nonerror:
                    prog = self.generate_program(type_list, cur_step=0, max_step=depth)
                    if not "ERROR" in prog["term"]:
                        nonerror = True
                prog = pd.DataFrame.from_dict(prog, orient="index").T
            else:
                # 1.1.2 query global library
                prog = self.query_existing_prog(
                    type_string, num_sample=1, production=self.global_production
                )

        else:
            # 1.2 query local library
            prog = self.query_existing_prog(
                type_string, num_sample=1, production=self.production
            )

        return prog

    def comp_lp_open_new_dish(self, type_string: str) -> pd.DataFrame:
        """Log probability of creating a brand-new program (new dish) in the global library.

        In the second-level PY process, a *dish* is a unique program in the
        global library.  The probability of creating a new dish (rather than
        re-using one already in the global library) is::

            P(new dish) = (alpha_g + K_dish * d_g) / (alpha_g + M_table)

        where ``K_dish`` = number of distinct programs currently in the global
        library for ``type_string`` and ``M_table`` = cumulative count of
        tables created globally for ``type_string`` (tracked in
        ``self.history_table``).

        Args:
            type_string: Type-signature string for which a new dish is
                being considered.

        Returns:
            tuple[int, float]: ``(K_dish, lp_new_dish)`` — the current
            number of distinct global programs and the log probability of
            opening a new dish.
        """
        # To choose a new dish in global library
        # TODO: this is dependent on the library size
        # K_dish represent the number of dishes served throughout the history
        K_dish = self.global_production.query(
            f'type_string=="{type_string}" & ctype=="program"'
        ).shape[0]
        M_table = self.history_table[type_string]

        lp_new_dish = np.log(self.global_alpha + K_dish * self.global_d) - np.log(
            self.global_alpha + M_table
        )

        return K_dish, lp_new_dish

    def query_existing_prog(
        self,
        type_string: str,
        production: pd.DataFrame,
        num_sample: int = 1,
    ) -> pd.DataFrame:
        """
        Query the existing programs in the library.
        In non-hierarchical AG, we only consider the program in global library (which is self.production).

        The log_prob is the sum of the comp_lp and adaptor_lp, which is already updated given the count.
        """
        qs = '&ctype=="program"'
        unfolded = production.query(f'type_string=="{type_string}"{qs}')

        return (
            unfolded.sample(
                n=min(num_sample, len(unfolded)), weights=np.exp(unfolded["log_prob"])
            )
            if len(unfolded) > 1
            else unfolded.reset_index(drop=True)
        )

    # ----- Program construction -----

    # ----- Library update -----
    def initialize_local_lib(self, sample: bool = False) -> None:
        """
        Initialize the local library

        Option:
        - Sample:     the local library is sampled from the global library
                      This means the inference of programs from local library will start from exitsing tables and dishes
        - Non-sample: the local library is initialized with the base terms
        """
        lib_base, lib_prog = self._extract_pm_base_prog(
            production=self.global_production
        )

        if sample:
            # TODO: is there a problem if local lib is sampled from glocal lib?
            lib_prog["exp_prob"] = np.exp(lib_prog["log_prob"])
            sampled_size = min(self.lib_size, len(lib_prog))
            lib_prog = lib_prog.sample(
                n=sampled_size, weights=lib_prog["exp_prob"]
            ).reset_index(drop=True)
            lib_prog = lib_prog.drop(columns=["exp_prob"])

            self.production = pd.concat([lib_base, lib_prog]).reset_index(drop=True)

        else:
            self.production = lib_base

    def update_local_lib(self, new_progs) -> None:
        """
        Update the local library
        """
        # Extract the initial and post programs or base/program terms
        lib_base, lib_prog = self._extract_pm_base_prog()

        new_term = new_progs["term"].item()
        if new_term in lib_prog["term"].tolist():
            lib_prog.loc[lib_prog["term"] == new_term, "count"] += 1
        else:
            new_progs = self._reformalize_new_prog(new_progs)
            lib_prog = pd.concat([lib_prog, new_progs]).reset_index(drop=True)

        # Update comp_lp given type strings
        lib_prog = self.prior_uniform_per_type_string(production=lib_prog)
        # Update adaptor_lp given counts
        lib_prog = self.update_lp_adaptor(
            production=lib_prog,
            alpha=self.local_alpha,
            d=self.local_d,
        )
        # Combine the base and program terms to the production
        local_production = pd.concat([lib_base, lib_prog]).reset_index(drop=True)
        # Update overall log probability
        local_production = self.update_overall_lp(local_production)

        # TODO: Restrict the library size; necessary?
        local_production = self._restrict_lib_size(
            local_production, self.local_lib_size
        )
        self.production = local_production

    def update_post_lib(self, new_progs: pd.DataFrame, lib: pd.DataFrame) -> None:
        """
        Update library based on the compression degree and counts
        """
        AdaGrammar.update_post_lib(self, new_progs, self.global_production)
        self.global_production = self.production

        for type_string in self.history_table.keys():
            self.history_table[type_string] += len(
                lib.query(f'type_string=="{type_string}" & ctype=="program"')
            )

    #  ----- Library update -----
