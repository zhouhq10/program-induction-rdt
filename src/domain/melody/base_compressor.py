"""
base_compressor.py — Base compressor class for melody program induction.

Defines :class:`Compressor`, the shared infrastructure for all DP-based
compressors.  Subclasses (see ``dp_compressor.py`` and
``greedy_dp_compressor.py``) extend this class with specific grammar
curricula (PCFG, AG, HAG) and search strategies.
"""

import sys

sys.path.append("..")

import re
import math
import random
import pickle
import numpy as np
import pandas as pd

import textdistance
from collections import deque
from typing import List, Dict, Optional, Tuple

from src.program.router import *
from src.domain.melody.melody_primitive import *
from src.program.helpers import power_law_sampler

EPS = 1e-6

# Log-probability constants for the memorise primitive:
#   p(memorise) = 0.25  (probability of the memorise frame type)
#   p(note_i)   = 1/7   (uniform prior over the 7-token note vocabulary)
#   p(value_i)  = 1/6   (uniform prior over the 6 note values per position)
log_prob_memorize = np.log(0.25) + np.log(1 / 7)
log_prob_note = np.log(1 / 6)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _d_levenshtein(gt: np.ndarray, pred: np.ndarray) -> int:
    """Compute the Levenshtein edit distance between two note sequences.

    Args:
        gt: Ground-truth note sequence.
        pred: Predicted (reconstructed) note sequence.

    Returns:
        Integer edit distance (number of insertions, deletions, or
        substitutions needed to transform ``pred`` into ``gt``).
    """
    return textdistance.levenshtein.distance(pred.tolist(), gt.tolist())


# ---------------------------------------------------------------------------
# Compressor base class
# ---------------------------------------------------------------------------


class Compressor:
    """Base class for grammar-based melody compressors.

    Holds all hyper-parameters, shared scoring utilities, and helper methods
    used by every compressor variant.  Not intended to be instantiated
    directly — use :class:`DP_PCFGCompressor`, :class:`DP_AGCompressor`, or
    :class:`DP_HAGCompressor` instead.
    """

    def __init__(self, program_lib: object, args: object) -> None:
        self.lib = program_lib

        self.beta = args.beta

        self.task_start_ind = args.task_start_ind
        self.task_num = args.task_num

        self.search_budget = args.search_budget
        self.frame_max_depth = args.frame_depth
        # Pre-compute depth sampling distribution (used throughout compression)
        self.depth_prob = self.power_law_dist(N=self.frame_max_depth)

    # ------------------------------------------------------------------
    # Utility: deque construction
    # ------------------------------------------------------------------

    def _construct_deque(self, max_length: int, element: object = 0) -> deque:
        """Create a fixed-length deque pre-filled with a sentinel value.

        Used to initialise the rolling DP state arrays (rates, distortions,
        program lists) with zero / empty base-case values before the first
        real DP step.

        Args:
            max_length: Maximum capacity of the deque (older entries are
                automatically discarded once the deque is full).
            element: Value to pre-fill every slot with.  Defaults to 0.

        Returns:
            A :class:`collections.deque` of length ``max_length`` with all
            slots set to ``element``.
        """
        tmp_list = [element] * max_length
        return deque(tmp_list, maxlen=max_length)

    # ------------------------------------------------------------------
    # Distortion and value computation
    # ------------------------------------------------------------------

    @staticmethod
    def _comp_subprog_distortion(recon: np.ndarray, subtask: np.ndarray) -> int:
        """Compute the Levenshtein distortion between a reconstruction and a subtask.

        Truncates both sequences to the shorter of the two lengths before
        comparison, so programs that output shorter sequences are not penalised
        for the unmatched suffix.

        Args:
            recon: Reconstructed note sequence produced by a candidate program.
            subtask: Ground-truth sub-melody segment.

        Returns:
            Levenshtein edit distance on the overlapping prefix.
        """
        recon_len = min(len(recon), len(subtask))
        return _d_levenshtein(subtask[:recon_len], recon[:recon_len])

    def _comp_subprog_value(
        self,
        cur_all_progs: pd.DataFrame,
        subtask: np.ndarray,
    ) -> pd.DataFrame:
        """Compute distortion and RD value for a set of candidate programs.

        Args:
            cur_all_progs: DataFrame of candidate programs; must already
                contain ``"recon"``, ``"recon_len"``, and ``"log_prob"``
                columns.
            subtask: Ground-truth sub-melody segment used to evaluate
                reconstruction accuracy.

        Returns:
            The same DataFrame with distortion and value columns added.
        """
        cur_all_progs["distortion"] = cur_all_progs["recon"].apply(
            self._comp_subprog_distortion, args=(subtask,)
        )
        cur_all_progs = self._comp_ll_value(cur_all_progs)
        return cur_all_progs

    def _comp_ll_value(
        self,
        prog: pd.DataFrame,
        beta: Optional[float] = None,
    ) -> pd.DataFrame:
        """Compute the log-likelihood and RD value of each program.

        Args:
            prog: DataFrame with ``"distortion"``, ``"recon_len"``, and
                ``"log_prob"`` columns.
            beta: Override for the instance ``self.beta``.  If None, uses
                ``self.beta``.

        Returns:
            The same DataFrame with ``"log_ll"`` and ``"value"`` columns
            added or updated.
        """
        beta = self.beta if beta is None else beta

        prog["log_ll"] = np.log(1 - prog["distortion"] / prog["recon_len"] + EPS)
        if beta > 1:
            prog["value"] = prog["log_ll"] / beta + prog["log_prob"] / prog["recon_len"]
        elif beta > 0:
            prog["value"] = prog["log_ll"] + beta * prog["log_prob"] / prog["recon_len"]
        else:
            prog["value"] = prog["log_ll"]
        return prog

    # ------------------------------------------------------------------
    # Memorise-frame construction
    # ------------------------------------------------------------------

    def _sample_mem_frame_prob(
        self, num_note: int, beta: Optional[float] = None
    ) -> np.ndarray:
        """Sample a probability distribution over memorise-prefix lengths.

        The distribution is proportional to the exponentiated β-scaled coding
        cost of memorising a sequence of each length.

        Args:
            num_note: Maximum prefix length to consider.
            beta: Override for ``self.beta``.

        Returns:
            Normalised probability array of shape ``(num_note,)``.
        """
        beta = self.beta if beta is None else beta

        if beta == 0.0:
            return [1 / num_note] * num_note

        rate = [
            beta * (log_prob_memorize + log_prob_note * (i + 1)) / (i + 1)
            for i in range(num_note)
        ]
        prob = np.exp(np.array(rate))
        return prob / prob.sum()

    def add_frame_for_memorization(self, subtask: np.ndarray) -> pd.DataFrame:
        """Build a stochastic memorise-program frame for a sub-melody.

        Samples a prefix length k from a β-scaled prior
        (see :meth:`_sample_mem_frame_prob`), then creates a memorise program
        ``[K, memorise, note_x_y_…]`` that stores exactly the first k notes
        of ``subtask`` verbatim.  The program has zero distortion by
        construction.

        Args:
            subtask: Sub-melody segment; at most
                ``self.reuse_num_note_subtask`` notes are considered as
                candidate prefix lengths.

        Returns:
            DataFrame of memorise-program candidates (one per sampled prefix),
            with ``"term"``, ``"log_prob"``, ``"distortion"``, ``"recon"``,
            ``"recon_len"``, and ``"value"`` columns populated.
        """
        reuse_num_note = min(self.reuse_num_note_subtask, len(subtask))
        sampled_mem_note = np.random.choice(
            np.arange(1, reuse_num_note + 1),
            p=self._sample_mem_frame_prob(reuse_num_note),
        )

        # Build candidate note objects for the sampled prefix length
        possible_notes = self.extract_subtask_note(
            subtask, fixed_num_note=sampled_mem_note
        )

        memorized_progs = []
        for i, possi_note in enumerate(possible_notes):
            log_prog_whole = log_prob_memorize + log_prob_note * sampled_mem_note
            memorized_progs.append(
                pd.DataFrame(
                    {
                        "term": f"[K,memorize,{possi_note.name}]",
                        "comp_lp": log_prog_whole,
                        "log_prob": log_prog_whole,
                        "type_string": "note->note",
                        "distortion": 0,
                        "recon": [np.array(possi_note.value)],
                        "recon_len": sampled_mem_note,
                        "frame": "[K,memorize,note]",
                        "depth": 1,
                    },
                    index=[i],
                )
            )

        memorized_progs = self._comp_ll_value(pd.concat(memorized_progs))
        return memorized_progs

    # ------------------------------------------------------------------
    # Note extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def extract_subtask_note(
        task: np.ndarray,
        reuse_num_note_subtask: int = 0,
        fixed_num_note: int = 0,
    ) -> List[object]:
        """Extract candidate note primitives from the beginning of a sub-melody.

        Args:
            task: Full sub-melody as a 1-D integer array.
            reuse_num_note_subtask: Maximum prefix length for the
                variable-length mode.
            fixed_num_note: Exact prefix length for the fixed-length mode.

        Returns:
            List of :class:`Note` (or cached note primitive) objects, one per
            extracted prefix.
        """
        if reuse_num_note_subtask:
            num_note = reuse_num_note_subtask
            len_task = len(task)
            iter_range = len_task if len_task < num_note else num_note
            start_index = 0
        else:
            iter_range = fixed_num_note
            start_index = iter_range - 1

        possible_notes = []
        for i in range(start_index, iter_range):
            note_value = task[: i + 1]
            note_name = Note.array_to_string(note_value)
            possi_note = create_or_get_pm_from_cache(note_name)
            possible_notes.append(possi_note)

        return possible_notes

    # ------------------------------------------------------------------
    # Frame sampling
    # ------------------------------------------------------------------

    @staticmethod
    def power_law_dist(N: int, alpha: float = 1) -> np.ndarray:
        """Compute a power-law probability distribution over depths 1 … N.

        Args:
            N: Maximum depth value (inclusive).
            alpha: Exponent controlling skewness.  Higher alpha
                concentrates more mass on depth 1.

        Returns:
            Normalised probability array of shape ``(N,)``.
        """
        integers = np.arange(1, N + 1).astype(float)
        probabilities = np.power(integers, -alpha)
        probabilities /= probabilities.sum()
        return probabilities

    def _power_law_sampler(self) -> int:
        """Draw a single program depth from the pre-computed power-law distribution.

        Returns:
            Integer depth in {1, …, ``frame_max_depth``}.
        """
        return np.random.choice(
            np.arange(1, self.frame_max_depth + 1), p=self.depth_prob
        )

    def sample_depth_subtask(self, num_sample: int) -> List[int]:
        """Draw ``num_sample`` program depths from the power-law distribution.

        Args:
            num_sample: Number of depth values to sample.

        Returns:
            List of integer depths.
        """
        return [self._power_law_sampler() for _ in range(num_sample)]

    def sample_type_string_subtask(self, num_sample: int) -> List[str]:
        """Draw ``num_sample`` type signatures uniformly from the grammar.

        Args:
            num_sample: Number of type strings to sample.

        Returns:
            List of type-signature strings.
        """
        type_strings = self.lib.type_strings
        return np.random.choice(type_strings, size=num_sample).tolist()

    def _sample_frame_prior(
        self,
        frame_left: pd.DataFrame,
        rescaled_prior: bool = False,
    ) -> pd.DataFrame:
        """Sample up to ``frame_sample`` frames from a pre-generated frame pool.

        If the pool is smaller than ``frame_sample``, the entire pool is
        returned.  Otherwise, weighted sampling (by the ``"prob"`` column)
        is performed.

        Args:
            frame_left: DataFrame of available frames with a ``"prob"``
                column used as sampling weights.
            rescaled_prior: Unused flag reserved for future rescaling
                of prior weights.

        Returns:
            DataFrame of sampled frames.
        """
        num_sample = self.frame_sample

        if len(frame_left) <= num_sample:
            sampled_frames = frame_left.copy()
        else:
            sampled_frames = frame_left.sample(
                n=num_sample, weights="prob"
            ).reset_index(drop=True)

        return sampled_frames

    def _sample_frame_switch(
        self,
        rescaled_prior: bool = False,
    ) -> pd.DataFrame:
        """Sample frames using a probabilistic switch between depth-1 and deeper frames.

        This prior mixture favours simpler programs while retaining coverage of more complex structures.

        Args:
            rescaled_prior: Passed through to :meth:`_sample_frame_prior`.

        Returns:
            DataFrame of sampled frames.
        """
        if np.random.random() < self.switch:
            frames_left = self.frames[0]
        else:
            frames_left = self.frames[1]

        return self._sample_frame_prior(frames_left, rescaled_prior)

    def sample_frame_subtask(self) -> pd.DataFrame:
        """Sample frames for the current sub-task using the configured strategy.

        Returns:
            DataFrame of sampled frames.
        """
        if self.sample_subprog == "uniform":
            sampled_frames = self._sample_frame_prior(frame_left=self.frames[-1])
        elif self.sample_subprog == "switch":
            sampled_frames = self._sample_frame_switch()
        return sampled_frames

    def sample_frames_and_argnotes(
        self, frames: pd.DataFrame, by: str = "term_value_likelihoods"
    ) -> pd.DataFrame:
        """Sample a frame and its argument notes weighted by likelihood.

        Args:
            frames: DataFrame of candidate frames with ``"term"`` and
                ``"term_value_likelihoods"`` columns.
            by: Column name to use as sampling weights.

        Returns:
            DataFrame of ``self.top_n_subtask`` sampled rows for the
            selected term.
        """
        frame_set = frames.groupby("term")
        frame_set = pd.DataFrame(
            {
                "term": frame_set["term"].first(),
                "term_value_likelihoods": frame_set["term_value_likelihoods"].mean(),
            }
        ).reset_index(drop=True)

        # Sample one program term proportional to its mean likelihood
        filtered_term = frame_set.sample(n=1, weights=by)["term"]
        filtered = frames.query(f'term=="{filtered_term.iloc[0]}"')

        # Sample argument-note instantiations of the selected term
        filtered = filtered.sample(n=self.top_n_subtask, weights=by)
        return filtered

    # ------------------------------------------------------------------
    # Library size analysis
    # ------------------------------------------------------------------

    def get_learned_lib_size(self, progs: pd.DataFrame) -> None:
        """Compute descriptive statistics about the learned program library.

        Calculates:

        * Number of unique program frames learned (``learned_f_lib_size``).
        * Mean token length of those frames (``learned_f_mean_length``).
        * Number of unique note arguments learned (``learned_note_lib_size``).
        * Mean note length of learned note arguments (``learned_note_mean_length``).
        * Number of frames reused across tasks (``shared_f_lib_size``).
        * Number of note arguments reused across tasks (``shared_note_lib_size``).
        * Number of full programs (frame + arg) reused (``shared_prog_lib_size``).

        .. note::
            This method currently has no return statement; all computed
            statistics are local variables.  TODO: return a summary dict.

        Args:
            progs: DataFrame of selected programs across all tasks, with
                ``"term"`` and ``"arg_notes"`` columns.
        """
        # Unique program frame structures
        learned_f_lib_size = len(progs["term"].unique())
        counts = progs["term"].apply(lambda x: x.count(",") + 1)
        learned_f_mean_length = counts.mean()

        # Learned note arguments (notes not in the initial primitive set)
        distinct_notes = progs["arg_notes"].unique().tolist()
        if None in distinct_notes:
            distinct_notes.remove(None)
        existing_args = self.init_term_lib["term"].tolist()
        learned_args = [arg for arg in distinct_notes if arg not in existing_args]
        learned_note_lib_size = len(learned_args)
        learned_note_mean_length = (
            sum([len(arg.split("_")) - 1 for arg in learned_args])
            / learned_note_lib_size
        )

        # Frames reused across multiple tasks
        term_counts = progs["term"].value_counts()
        repeated_term = term_counts[term_counts > 1]
        shared_f_lib_size = len(repeated_term)

        # Note arguments reused across multiple tasks
        note_counts = progs["arg_notes"].value_counts()
        repeated_notes = note_counts[note_counts > 1]
        shared_note_lib_size = len(repeated_notes)
        shared_learned_note_lib_size = len(
            [note for note in repeated_notes if note in learned_args]
        )

        # Full programs (frame + note argument) reused across tasks
        progs["term_arg"] = progs["term"] + " - " + progs["arg_notes"]
        term_arg_counts = progs["term_arg"].value_counts()
        repeated_term_args = term_arg_counts[term_arg_counts > 1]
        repeated_term_args = repeated_term_args.index.str.split(" - ")
        shared_prog_lib_size = len(repeated_term_args)

    # ------------------------------------------------------------------
    # Pattern matching helpers (used by RDT simulation)
    # ------------------------------------------------------------------

    @staticmethod
    def find_arg_param_patterns(text: str) -> List[str]:
        """Find all primitive argument tokens in a program string.

        Matches note sequences (``note_1_2_3``), count values
        (``count_2``), time indices (``time_3``), and infinite note
        placeholders (``note_inf``).

        Args:
            text: String representation of a program term.

        Returns:
            List of matched argument token strings.
        """
        pattern = r"note_(?:\d+)(?:_\d+)*|count_\d+|time_\d+|note_inf"
        return re.findall(pattern, text)

    @staticmethod
    def split_arg_notes(row: Optional[str]) -> List[str]:
        """Split a compound note string into individual single-note tokens.

        For example, ``"note_1_2_3"`` → ``["note_1", "note_2", "note_3"]``.

        Args:
            row: Compound note string (e.g. ``"note_1_2_3"``), or None.

        Returns:
            List of individual note token strings, or an empty list if
            ``row`` is None.
        """
        if row is None:
            return []
        parts = row.split("_")
        return ["note_" + part for part in parts[1:]]

    @staticmethod
    def join_arg_note_list(note_list: List[str]) -> Optional[str]:
        """Merge a list of individual note tokens into a compound note string.

        Inverse of :meth:`split_arg_notes`.  For example,
        ``["note_1", "note_2", "note_3"]`` → ``"note_1_2_3"``.

        Args:
            note_list: List of note token strings.

        Returns:
            Compound note string, or None if the list is empty.
        """
        if len(note_list) == 0:
            return None
        parts = [note.split("note_")[-1] for note in note_list]
        return "note_" + "_".join(parts)

    @staticmethod
    def find_arg_prog_patterns(term: str, patterns: List[str]) -> Optional[str]:
        """Find the first pattern from ``patterns`` that appears in ``term``.

        Args:
            term: Program term string to search.
            patterns: List of candidate sub-program strings.

        Returns:
            The first matching pattern string, or None if no match is found.
        """
        for pattern in patterns:
            if pattern in term:
                return pattern
        return None

    @staticmethod
    def extract_uppercase_parts(expression_str: str) -> List[str]:
        """Extract all-uppercase tokens (combinators) from a program string.

        Splits on brackets and commas, then keeps only tokens that are
        entirely uppercase and longer than one character (i.e. combinator
        names like ``"BB"``, ``"BK"``, etc.).

        Args:
            expression_str: Program string, e.g. ``"[BK,repeat,note_1_2]"``.

        Returns:
            List of combinator token strings found in the expression.
        """
        parts = re.split(r"[\[\],]", expression_str)
        uppercase_parts = [part for part in parts if part.isupper() and len(part) > 1]
        return uppercase_parts

    # ------------------------------------------------------------------
    # Rate-distortion simulation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_arg_length(progs: pd.DataFrame, init_pm: pd.DataFrame) -> int:
        """Count the total number of argument tokens across all selected programs.

        Computes the total description-length contribution from argument
        values (as opposed to program frame structures).

        Args:
            progs: DataFrame of selected programs with ``"term"``,
                ``"complete_types"``, and ``"arg_notes"`` columns.
            init_pm: Initial production table (before any library updates),
                used to determine which arguments are primitives vs. learned.

        Returns:
            Total integer argument-token count across all programs.
        """
        existing_args = init_pm["term"].tolist()
        init_pm_prog = init_pm.query("is_init==0")["term"].tolist()

        df_progs = progs.copy()
        arg_length = 0
        for i in range(len(df_progs)):
            cur_term = df_progs["term"].iloc[i]

            # If the term embeds a previously learned sub-program, count it as 1
            # TODO: think about whether reused sub-programs should cost more
            matched_pattern = Compressor.find_arg_prog_patterns(cur_term, init_pm_prog)
            if matched_pattern is not None:
                arg_length += 1
                cur_term = cur_term.replace(matched_pattern, "")

            # Count argument tokens by type
            arg_types = df_progs["complete_types"].iloc[i].split("->")[0].split("_")
            for arg_type in arg_types:
                if "note" in arg_type:
                    arg_notes = df_progs["arg_notes"].iloc[i]
                    if arg_notes is None:
                        # No stored argument note; cost = 1 placeholder token
                        arg_length += 1
                    elif arg_notes in existing_args:
                        # Argument is a primitive note; cost = 1 token
                        arg_length += 1
                    else:
                        # Argument is a learned (multi-note) sequence; cost = length
                        arg_length += len(Note.string_to_array(arg_notes))
                else:
                    arg_length += 1

        return arg_length

    @staticmethod
    def remove_args_params(
        progs: pd.DataFrame,
        forget_ratio: float = 0.0,
        forget_num: int = 0,
        forget_thres: int = 0,
        task: Optional[np.ndarray] = None,
    ) -> Tuple[int, float]:
        """Simulate memory degradation by randomly corrupting stored arguments.

        Models the effect of forgetting stored note arguments or primitive
        parameters by replacing a randomly sampled subset with new random
        values, then measuring the resulting reconstruction error.

        Note arguments are replaced with randomly sampled note sequences;
        count / time parameters are replaced with random count / time values.
        The replacement is effected by mutating the global namespace so that
        ``eval(program_string)`` can resolve the new names — this is
        necessary because programs are stored as eval-able strings.

        Args:
            progs: DataFrame of selected programs; must contain ``"term"``,
                ``"arg_notes"``, ``"recon_errors"``, and ``"recon_length"``
                columns.
            forget_ratio: Fraction of all stored arguments to corrupt
                (used when neither ``forget_num`` nor ``forget_thres`` is set).
            forget_num: Exact number of arguments to corrupt.
            forget_thres: Memory capacity threshold; items above this
                threshold are corrupted.
            task: Unused; reserved for future task-conditioned forgetting.

        Returns:
            ``(num_forgetted_args, error)`` where:
            - ``num_forgetted_args``: actual number of arguments corrupted.
            - ``error``: total reconstruction error (weighted Levenshtein
              distance) accumulated across all corrupted programs.
        """
        df_progs = progs.copy()
        # Enumerate all stored argument tokens per program
        df_progs["arg_param_list"] = df_progs["term"].apply(
            Compressor.find_arg_param_patterns
        )
        df_progs["arg_note_list"] = df_progs["arg_notes"].apply(
            Compressor.split_arg_notes
        )

        all_arg_notes = secure_list(df_progs["arg_note_list"].tolist())
        all_arg_params = secure_list(df_progs["arg_param_list"].tolist())
        num_arg_notes = len(all_arg_notes)
        num_arg_params = len(all_arg_params)

        # Determine how many arguments to corrupt
        if forget_thres:
            num_forgetted_args = num_arg_notes + num_arg_params - forget_thres
            if num_forgetted_args <= 0:
                return 0, (df_progs["recon_errors"] * df_progs["recon_length"]).sum()
        elif forget_num:
            num_forgetted_args = forget_num
        else:
            num_forgetted_args = math.ceil(
                (num_arg_notes + num_arg_params) * forget_ratio
            )

        # Build a flat index list: [0, i] = note argument i, [1, j] = param j
        indices_notes = [
            [i, j]
            for i, sublist in enumerate(all_arg_notes)
            for j in range(len(sublist))
        ]
        indices_params = [
            [i, j]
            for i, sublist in enumerate(all_arg_params)
            for j in range(len(sublist))
        ]
        combined_indices = [[0, i] for i in range(len(indices_notes))] + [
            [1, i] for i in range(len(indices_params))
        ]

        # Randomly sample which arguments to corrupt
        sampled_indices = random.sample(
            combined_indices, min(num_forgetted_args, len(combined_indices))
        )

        # Regex to match bare "note" tokens (not followed by word characters)
        # used to substitute note names into the program string
        pattern = r"note(?![\w\d_]+)"
        error = 0
        for index in sampled_indices:
            if index[0] == 0:
                # --- Corrupt a note argument ---
                tmp_ind = indices_notes[index[1]]
                prev_arg_notes = all_arg_notes[tmp_ind[0]]
                prev_note = Compressor.join_arg_note_list(prev_arg_notes)
                # Register the original note in global scope for eval()
                globals()[prev_note] = Note(prev_note)

                # Replace one note in the argument with a random alternative
                new_arg_notes = all_arg_notes[tmp_ind[0]].copy()
                new_arg_notes[tmp_ind[1]] = "note_{}".format(np.random.randint(1, 6))
                new_note = Compressor.join_arg_note_list(new_arg_notes)
                globals()[new_note] = Note(new_note)

                prev_frame = df_progs["term"][tmp_ind[0]]
                new_prog = re.sub(pattern, new_note, prev_frame)
                prev_prog = re.sub(pattern, prev_note, prev_frame)

            else:
                # --- Corrupt a primitive parameter (count / time / note param) ---
                tmp_ind = indices_params[index[1]]
                prev_arg_params = all_arg_params[tmp_ind[0]][tmp_ind[1]]
                prev_frame = df_progs["term"][tmp_ind[0]]

                if "note_" in prev_arg_params:
                    # Replace a note parameter with a random note of the same length
                    length = len(Note.string_to_array(prev_arg_params))
                    random_note = RandomNote(length=length, copy_note=prev_arg_params)
                    globals()[random_note.name] = random_note
                    new_prog = prev_frame.replace(
                        prev_arg_params + "]", random_note.name + "]"
                    )
                elif "count_" in prev_arg_params:
                    random_count = RandomCount()
                    globals()[random_count.name] = random_count
                    new_prog = prev_frame.replace(prev_arg_params, random_count.name)
                elif "time_" in prev_arg_params:
                    random_time = RandomTimeIndex()
                    globals()[random_time.name] = random_time
                    new_prog = prev_frame.replace(prev_arg_params, random_time.name)

                prev_note = df_progs["arg_notes"][tmp_ind[0]]
                if prev_note is not None:
                    globals()[prev_note] = Note(prev_note)
                    prev_prog = re.sub(pattern, prev_note, prev_frame)
                    new_prog = re.sub(pattern, prev_note, new_prog)
                else:
                    prev_prog = prev_frame

            # Measure the reconstruction error introduced by the corruption
            error_recon = df_progs["recon_errors"][tmp_ind[0]]
            prev_error = (
                df_progs["recon_errors"][tmp_ind[0]]
                * df_progs["recon_length"][tmp_ind[0]]
            )
            new_results = Program(eval(new_prog)).run()
            prev_results = Program(eval(prev_prog)).run()
            if len(new_results) != len(prev_results):
                error += (abs(len(new_results) - len(prev_results))) * (
                    1 - error_recon
                ) + prev_error
            else:
                error += (np.not_equal(new_results, prev_results).sum()) * (
                    1 - error_recon
                ) + prev_error

        return num_forgetted_args, error

    @staticmethod
    def simulate_rdt(
        progs: pd.DataFrame,
        init_pm: pd.DataFrame,
        sim: int = 10,
        sample_num: int = 10,
    ) -> List[List[float]]:
        """Generate an empirical rate-distortion curve by progressive forgetting.

        Starting from the full set of stored arguments (maximum rate, zero
        distortion), iteratively corrupts one additional argument at a time
        and measures the average reconstruction error.  Each corruption level
        is repeated ``sim`` times to estimate the expected distortion.

        The result is a list of ``[sample_num, rate, distortion]`` triplets
        that can be plotted to obtain the empirical RD curve.

        Args:
            progs: DataFrame of selected programs from a compression run.
            init_pm: Initial production table, used by
                :meth:`compute_arg_length` to determine which arguments are
                primitives.
            sim: Number of Monte Carlo repetitions per corruption level.
            sample_num: Identifier for this simulation run (stored in the
                first column of each output row; does not affect computation).

        Returns:
            List of ``[sample_num, rate, distortion]`` rows — one for the
            full-memory baseline plus one per (corruption level × repetition).
        """
        rdt_list = []
        task_num = progs["recon_length"].sum()
        arg_length = Compressor.compute_arg_length(progs, init_pm)
        # Baseline: no forgetting → rate = arg_length / task_num, distortion = 0
        rdt_list.append([sample_num, arg_length / task_num, 0])

        forget_nums = range(int(arg_length))
        for forget_num in forget_nums:
            for _ in range(sim):
                removed_length, error = Compressor.remove_args_params(
                    progs, forget_num=forget_num, init_pm=init_pm
                )
                rdt_list.append(
                    [
                        sample_num,
                        max((arg_length - removed_length), 0) / task_num,
                        min(error, task_num) / task_num,
                    ]
                )
        return rdt_list

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    @staticmethod
    def save_result_per_task(
        task_ind: int, task_results: Dict[str, object], save_path: str
    ) -> None:
        """Pickle per-task results to disk.

        Args:
            task_ind: Index of the task being saved (used in the filename).
            task_results: Dictionary mapping result names to Python objects
                (e.g. ``{"prog_trajs": df, "task_rates": [...]}``.
            save_path: Directory in which to write the files.
        """
        print(f"Saving task {task_ind} results", flush=True)
        for item_name, value in task_results.items():
            filehandler = open(f"{save_path}/task_{task_ind}_{item_name}.obj", "wb")
            pickle.dump(value, filehandler)
            filehandler.close()
